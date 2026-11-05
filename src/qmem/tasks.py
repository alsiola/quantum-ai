"""Synthetic sequence sources with exactly known optimal predictors.

Every task here is a finite hidden Markov model given by *labelled transition
matrices* ``L[y, i, j] = P(emit y, move to state j | state i)`` and an initial
state distribution ``pi``. One representation covers i.i.d. sources (one
state), order-``k`` Markov chains (state = last ``k`` symbols), the Even and
Golden-mean processes, and arbitrary HMMs, so the exact quantities below are
computed by the same forward algorithm for all of them:

* ``sample(n, length, seed)``: draw sequences (explicit ``torch.Generator``).
* ``predictive(seqs)``: the Bayes-optimal ``p(y_t | y_<t)`` for ``t = 1..T+1``.
* ``log_probs(seqs)``: the source's own ``log p(y_t | y_<t)``, shape ``(B, T)``.
  Its mean over a test set is the *oracle* test NLL: the per-sequence
  difference ``model - oracle`` is an unbiased estimate of the KL gap with
  much lower variance than comparing against the expected optimum.
* ``step_entropies(T)``: the exact ``H(Y_t | Y_<t)`` for ``t = 1..T``, by
  enumerating belief states (identical beliefs are merged, so this is
  polynomial for the unifilar sources used here).
* ``optimal_nll(T) = H(Y_1..Y_T) / T``: the best achievable expected
  per-symbol NLL on length-``T`` sequences started from ``pi``.
* ``entropy_rate()``: ``lim H(Y_t | Y_<t)``. Exact for unifilar presentations.

Markov sources start from their stationary context distribution (the context
before ``y_1`` is drawn but not observed), so all sources are stationary and
the optimum for early positions is the belief-averaged prediction.

Phase 3 tasks (delayed recall, parity, mixed timescales, random HMMs) are
also finite HMMs and plug in through ``HMMTask`` or a ``make_task`` kind.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .linalg import REAL_DTYPE

_ROUND = 11  # decimals used to merge identical belief states


def _entropy_nats(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) along the last axis; ``0 log 0 = 0``."""
    logp = torch.where(p > 0, torch.log(p.clamp(min=1e-300)), torch.zeros_like(p))
    return -(p * logp).sum(-1)


def stationary_distribution(transition: torch.Tensor) -> torch.Tensor:
    """Left eigenvector of a row-stochastic matrix for eigenvalue 1 (normalised)."""
    t = torch.as_tensor(transition, dtype=REAL_DTYPE)
    s = t.shape[0]
    # Solve pi (T - I) = 0 with sum(pi) = 1 by least squares.
    a = torch.cat([(t - torch.eye(s, dtype=REAL_DTYPE)).T, torch.ones(1, s, dtype=REAL_DTYPE)])
    b = torch.zeros(s + 1, dtype=REAL_DTYPE)
    b[-1] = 1
    pi = torch.linalg.lstsq(a, b[:, None]).solution[:, 0]
    pi = pi.clamp(min=0)
    return pi / pi.sum()


class HMMTask:
    """A stationary (or not) finite HMM over ``n_symbols`` outcomes.

    Args:
        labelled: ``(Y, S, S)`` with ``labelled[y, i, j] = P(y, j | i)``; rows of
            ``labelled.sum(0)`` must sum to 1.
        initial: ``(S,)`` start distribution; default the stationary one.
        name: short identifier used in results.
        params: the spec that built the task (recorded with results).
    """

    def __init__(
        self,
        labelled: torch.Tensor,
        initial: torch.Tensor | None = None,
        name: str = "hmm",
        params: dict[str, Any] | None = None,
    ):
        lab = torch.as_tensor(labelled, dtype=REAL_DTYPE)
        if lab.dim() != 3 or lab.shape[1] != lab.shape[2]:
            raise ValueError("labelled must have shape (n_symbols, n_states, n_states)")
        if (lab < 0).any():
            raise ValueError("labelled transition probabilities must be non-negative")
        rows = lab.sum(dim=(0, 2))
        if not torch.allclose(rows, torch.ones_like(rows), atol=1e-12):
            raise ValueError(f"each state's outgoing probabilities must sum to 1, got {rows}")
        self.labelled = lab
        self.transition = lab.sum(0)
        if initial is None:
            initial = stationary_distribution(self.transition)
        init = torch.as_tensor(initial, dtype=REAL_DTYPE)
        if init.shape != (lab.shape[1],) or abs(float(init.sum()) - 1) > 1e-12:
            raise ValueError("initial must be a probability vector over the states")
        self.initial = init
        self.name = name
        self.params = dict(params or {})

    @property
    def n_symbols(self) -> int:
        return self.labelled.shape[0]

    @property
    def n_states(self) -> int:
        return self.labelled.shape[1]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_symbols": self.n_symbols,
            "n_states": self.n_states,
            "params": self.params,
            "entropy_rate_nats": self.entropy_rate(),
            "unifilar": self.is_unifilar(),
        }

    # -- generation --------------------------------------------------------------
    def sample(self, n: int, length: int, seed: int) -> torch.Tensor:
        """``n`` sequences of ``length`` symbols, shape ``(n, length)``, from ``seed``."""
        gen = torch.Generator().manual_seed(seed)
        y_dim, s_dim = self.n_symbols, self.n_states
        # joint[i, y * S + j] = P(y, j | i)
        joint = self.labelled.permute(1, 0, 2).reshape(s_dim, y_dim * s_dim)
        state = torch.multinomial(self.initial.expand(n, s_dim), 1, generator=gen).squeeze(-1)
        out = torch.empty(n, length, dtype=torch.long)
        for t in range(length):
            k = torch.multinomial(joint[state], 1, generator=gen).squeeze(-1)
            out[:, t] = k // s_dim
            state = k % s_dim
        return out

    # -- exact forward algorithm ------------------------------------------------
    def _check(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = torch.as_tensor(sequences, dtype=torch.long)
        if seq.dim() == 1:
            seq = seq[None]
        if seq.numel() and (seq.min() < 0 or seq.max() >= self.n_symbols):
            raise ValueError(f"tokens must lie in [0, {self.n_symbols})")
        return seq

    def predictive(self, sequences: torch.Tensor) -> torch.Tensor:
        """Optimal ``p(. | y_<t)`` for ``t = 1..T+1``, shape ``(B, T+1, Y)``.

        If a prefix is impossible under the source, its later rows are NaN.
        """
        seq = self._check(sequences)
        belief = self.initial.expand(seq.shape[0], -1)
        emit = self.labelled.sum(-1).T  # (S, Y): P(y | i)
        out = [belief @ emit]
        for t in range(seq.shape[1]):
            unnorm = torch.einsum("bi,bij->bj", belief, self.labelled[seq[:, t]])
            belief = unnorm / unnorm.sum(-1, keepdim=True)
            out.append(belief @ emit)
        return torch.stack(out, dim=1)

    def log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """The source's ``log p(y_t | y_<t)``, shape ``(B, T)`` (``-inf`` if impossible)."""
        seq = self._check(sequences)
        pred = self.predictive(seq)[:, :-1]
        p = pred.gather(-1, seq[..., None]).squeeze(-1)
        # After an impossible symbol the belief is 0/0; the sequence stays impossible.
        return torch.log(p).nan_to_num(nan=-math.inf, neginf=-math.inf)

    def step_entropies(self, length: int, max_beliefs: int = 200_000) -> torch.Tensor:
        """Exact ``H(Y_t | Y_<t)`` in nats for ``t = 1..length``, shape ``(length,)``.

        Enumerates the reachable belief states with their probabilities,
        merging beliefs equal to ``1e-11``. Raises if more than ``max_beliefs``
        distinct beliefs are reachable (use a smaller ``length``). That is the
        case for long sequences from non-unifilar HMMs, whose beliefs need not
        repeat; there use the oracle NLL on the test set (``log_probs``), which
        is exact per sequence.
        """
        emit = self.labelled.sum(-1).T  # (S, Y)
        beliefs = self.initial[None].clone()  # (K, S)
        mass = torch.ones(1, dtype=REAL_DTYPE)
        out = []
        for _ in range(length):
            q = beliefs @ emit  # (K, Y)
            out.append(float((mass * _entropy_nats(q)).sum()))
            # successors: (K, Y, S)
            nxt = torch.einsum("ki,yij->kyj", beliefs, self.labelled)
            m = mass[:, None] * q
            keep = m > 0
            nxt, m = nxt[keep], m[keep]
            nxt = nxt / nxt.sum(-1, keepdim=True)
            merged: dict[tuple[float, ...], int] = {}
            rows, masses = [], []
            for b, w in zip(nxt.tolist(), m.tolist(), strict=True):
                key = tuple(round(x, _ROUND) for x in b)
                idx = merged.get(key)
                if idx is None:
                    merged[key] = len(rows)
                    rows.append(b)
                    masses.append(w)
                else:
                    masses[idx] += w
            if len(rows) > max_beliefs:
                raise RuntimeError(f"{len(rows)} belief states; reduce length")
            beliefs = torch.tensor(rows, dtype=REAL_DTYPE)
            mass = torch.tensor(masses, dtype=REAL_DTYPE)
        return torch.tensor(out, dtype=REAL_DTYPE)

    def optimal_nll(self, length: int) -> float:
        """``H(Y_1..Y_length) / length`` in nats: the best expected per-symbol NLL."""
        return float(self.step_entropies(length).mean())

    def is_unifilar(self) -> bool:
        """Whether (state, symbol) determines the next state."""
        return bool(((self.labelled > 0).sum(-1) <= 1).all())

    def entropy_rate(self, fallback_length: int = 200) -> float:
        """Entropy rate in nats per symbol.

        Exact for unifilar presentations (``sum_i pi_i H(Y | state i)`` with
        ``pi`` stationary); otherwise ``H(Y_T | Y_<T)`` at ``T = fallback_length``
        (or the longest ``T`` the belief enumeration allows), an upper bound that
        converges to the rate.
        """
        if self.is_unifilar():
            pi = stationary_distribution(self.transition)
            emit = self.labelled.sum(-1)  # (Y, S)
            return float((pi * _entropy_nats(emit.T)).sum())
        for t in range(fallback_length, 0, -1):
            try:
                return float(self.step_entropies(t, max_beliefs=20_000)[-1])
            except RuntimeError:
                continue
        raise RuntimeError("could not bound the entropy rate")


# -- constructors ----------------------------------------------------------------
def iid_task(probs: Sequence[float], name: str = "iid") -> HMMTask:
    p = torch.as_tensor(probs, dtype=REAL_DTYPE)
    lab = p[:, None, None].clone()
    return HMMTask(lab, name=name, params={"kind": "iid", "probs": list(map(float, probs))})


def markov_task(order: int, probs: Sequence[Sequence[float]], name: str | None = None) -> HMMTask:
    """Order-``order`` Markov chain from a conditional table.

    ``probs[c][y] = p(y_t = y | context c)``, with ``c`` the big-endian integer
    of ``(y_{t-order}, ..., y_{t-1})`` (oldest symbol most significant). The
    chain starts from its stationary context distribution.
    """
    table = torch.as_tensor(probs, dtype=REAL_DTYPE)
    y_dim = table.shape[1]
    s_dim = y_dim**order
    if order < 1 or table.shape[0] != s_dim:
        raise ValueError(f"order-{order} table over {y_dim} symbols needs {s_dim} rows")
    lab = torch.zeros(y_dim, s_dim, s_dim, dtype=REAL_DTYPE)
    for c in range(s_dim):
        for y in range(y_dim):
            lab[y, c, (c * y_dim + y) % s_dim] = table[c, y]
    params = {"kind": "markov", "order": order, "probs": table.tolist()}
    return HMMTask(lab, name=name or f"markov{order}", params=params)


def even_process(p: float = 0.5, name: str = "even") -> HMMTask:
    """The Even process: runs of 1s between 0s have even length.

    States A, B. From A: emit 0, stay in A (prob ``p``); emit 1, go to B
    (``1 - p``). From B: emit 1, go to A (prob 1). Infinite Markov order.
    """
    lab = torch.zeros(2, 2, 2, dtype=REAL_DTYPE)
    lab[0, 0, 0] = p
    lab[1, 0, 1] = 1 - p
    lab[1, 1, 0] = 1.0
    return HMMTask(lab, name=name, params={"kind": "even", "p": p})


def golden_mean_process(p: float = 0.5, name: str = "golden_mean") -> HMMTask:
    """The Golden-mean process: no two consecutive 0s (Markov order 1).

    States A, B. From A: emit 1, stay in A (prob ``p``); emit 0, go to B
    (``1 - p``). From B: emit 1, go to A (prob 1).
    """
    lab = torch.zeros(2, 2, 2, dtype=REAL_DTYPE)
    lab[1, 0, 0] = p
    lab[0, 0, 1] = 1 - p
    lab[1, 1, 0] = 1.0
    return HMMTask(lab, name=name, params={"kind": "golden_mean", "p": p})


def hmm_task(
    labelled: Sequence, initial: Sequence[float] | None = None, name: str = "hmm"
) -> HMMTask:
    lab = torch.as_tensor(labelled, dtype=REAL_DTYPE)
    init = None if initial is None else torch.as_tensor(initial, dtype=REAL_DTYPE)
    params = {"kind": "hmm", "labelled": lab.tolist(), "initial": initial}
    return HMMTask(lab, init, name=name, params=params)


TASK_KINDS: dict[str, Callable[..., HMMTask]] = {
    "iid": iid_task,
    "markov": markov_task,
    "even": even_process,
    "golden_mean": golden_mean_process,
    "hmm": hmm_task,
}


def register_task(kind: str, constructor: Callable[..., HMMTask]) -> None:
    """Add a task kind (Phase 3: delayed recall, parity, ...)."""
    if kind in TASK_KINDS:
        raise ValueError(f"task kind {kind!r} already registered")
    TASK_KINDS[kind] = constructor


def make_task(spec: dict[str, Any]) -> HMMTask:
    """Build a task from a config table: ``{"kind": ..., "name"?: ..., **params}``."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind not in TASK_KINDS:
        raise ValueError(f"unknown task kind {kind!r}; choose from {sorted(TASK_KINDS)}")
    task = TASK_KINDS[kind](**spec)
    task.params = {"kind": kind, **spec}
    return task


def bits(nats: float) -> float:
    return nats / math.log(2)
