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

from .instrument import Instrument
from .linalg import DTYPE, REAL_DTYPE, trace

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
        out = []
        for h in self._iter_step_entropies(max_beliefs):
            out.append(h)
            if len(out) == length:
                break
        return torch.tensor(out, dtype=REAL_DTYPE)

    def _iter_step_entropies(self, max_beliefs: int):
        """Yield ``H(Y_t | Y_<t)`` for ``t = 1, 2, ...``; raise once beliefs exceed the cap.

        The successor beliefs of step ``t`` are enumerated before ``H_t`` is
        yielded, so a yielded value always comes from a complete enumeration.
        """
        emit = self.labelled.sum(-1).T  # (S, Y)
        beliefs = self.initial[None].clone()  # (K, S)
        mass = torch.ones(1, dtype=REAL_DTYPE)
        while True:
            q = beliefs @ emit  # (K, Y)
            h = float((mass * _entropy_nats(q)).sum())
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
            yield h
            beliefs = torch.tensor(rows, dtype=REAL_DTYPE)
            mass = torch.tensor(masses, dtype=REAL_DTYPE)

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
        last = None
        try:
            for t, h in enumerate(self._iter_step_entropies(max_beliefs=20_000), start=1):
                last = h
                if t == fallback_length:
                    break
        except RuntimeError:
            pass
        if last is None:
            raise RuntimeError("could not bound the entropy rate")
        return float(last)


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


# -- Phase 3 benchmark sources (issue #8; see docs/benchmarks.md) -------------------
def _unifilar(
    n_symbols: int, n_states: int, edges: Sequence[tuple[int, int, int, float]]
) -> torch.Tensor:
    """Labelled matrices from edges ``(from, symbol, to, probability)``."""
    lab = torch.zeros(n_symbols, n_states, n_states, dtype=REAL_DTYPE)
    for i, y, j, p in edges:
        lab[y, i, j] += p
    return lab


def pad_alphabet(labelled: torch.Tensor, n_symbols: int | None) -> torch.Tensor:
    """Append never-emitted symbols so the alphabet has ``n_symbols`` letters.

    The model's alphabet is ``2**n_output``; a 3- or 6-symbol source is padded
    to 4 or 8 symbols that have probability 0 (the model must learn that too).
    """
    if n_symbols is None or n_symbols == labelled.shape[0]:
        return labelled
    if n_symbols < labelled.shape[0]:
        raise ValueError(f"cannot pad {labelled.shape[0]} symbols down to {n_symbols}")
    extra = torch.zeros(n_symbols - labelled.shape[0], *labelled.shape[1:], dtype=REAL_DTYPE)
    return torch.cat([labelled, extra])


def perturbed_coin(p: float = 0.3, name: str = "perturbed_coin") -> HMMTask:
    """Gu et al. 2012: a coin flipped with probability ``p`` each step, then observed.

    States = the coin face; from face ``j`` emit ``1 - j`` and move there with
    probability ``p``, else emit ``j`` and stay. Markov order 1; ``C_mu = 1``
    bit for ``p != 1/2``; ``C_q = H((1 + 2 sqrt(p(1-p)))/2)`` bits.
    """
    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1)")
    edges = [(0, 0, 0, 1 - p), (0, 1, 1, p), (1, 1, 1, 1 - p), (1, 0, 0, p)]
    return HMMTask(_unifilar(2, 2, edges), name=name, params={"kind": "perturbed_coin", "p": p})


def ghafari_process(
    p: float = 0.5, q: float = 0.5, n_symbols: int | None = 4, name: str = "ghafari"
) -> HMMTask:
    """Ghafari et al. 2019 / Yang et al. 2021 "asymmetric" 3-causal-state process.

    A perturbed coin with flip probabilities ``p`` (0 -> 1) and ``q`` (1 -> 0),
    post-processed so that every 0 that precedes a 1 is written 2. Causal
    states ``S_i`` = last output ``i`` (Ghafari Fig. 1b, Yang Fig. 3a):
    ``S0 -0|1-p-> S0``, ``S0 -2|p-> S2``, ``S1 -1|1-q-> S1``,
    ``S1 -0|q(1-p)-> S0``, ``S1 -2|pq-> S2``, ``S2 -1|1-> S1``.
    Three classical states; the quantum causal states fit in one qubit.
    Symbols 0, 1, 2 (padded to ``n_symbols``; default 4 for a 2-output-qubit model).
    """
    edges = [
        (0, 0, 0, 1 - p),
        (0, 2, 2, p),
        (1, 1, 1, 1 - q),
        (1, 0, 0, q * (1 - p)),
        (1, 2, 2, p * q),
        (2, 1, 1, 1.0),
    ]
    lab = pad_alphabet(_unifilar(3, 3, edges), n_symbols)
    params = {"kind": "ghafari", "p": p, "q": q, "n_symbols": n_symbols}
    return HMMTask(lab, name=name, params=params)


def renewal_task(
    hazards: Sequence[float], periodic: bool = False, name: str = "renewal"
) -> HMMTask:
    """Discrete renewal process: emit 1 (a tick) or 0; state = 0s since the last tick.

    ``hazards[n] = P(tick | n zeros since the last tick)``. With
    ``periodic=False`` the last hazard must be 1 (the counter is bounded);
    with ``periodic=True`` the counter is taken mod ``len(hazards)`` (valid when
    the hazard is periodic, as for PMD processes).
    """
    h = [float(x) for x in hazards]
    n = len(h)
    if not periodic and h[-1] != 1.0:
        raise ValueError("the last hazard must be 1 unless periodic=True")
    edges = []
    for k, hk in enumerate(h):
        if not 0 <= hk <= 1:
            raise ValueError(f"hazard {hk} is not a probability")
        if hk > 0:
            edges.append((k, 1, 0, hk))
        if hk < 1:
            edges.append((k, 0, (k + 1) % n, 1 - hk))
    lab = _unifilar(2, n, edges)
    params = {"kind": "renewal", "hazards": h, "periodic": periodic}
    return HMMTask(lab, name=name, params=params)


def pmd_survival(n: int, gamma: float, v: float, period: int) -> float:
    """Wu et al. 2023 Eq. 1: ``Phi(n) = Gamma^n (1 - V sin^2(n pi / N))``."""
    return gamma**n * (1 - v * math.sin(n * math.pi / period) ** 2)


def pmd_renewal(
    period: int = 4, gamma: float = 0.5, v: float = 0.4, name: str | None = None
) -> HMMTask:
    """Periodically modulated decay renewal process (Wu et al. 2023).

    ``Phi(n)`` is the probability that consecutive 1s are separated by at
    least ``n`` 0s; ``P(0 | n zeros since the last 1) = Phi(n+1) / Phi(n)``
    (Wu Methods Eq. 4). The hazard is ``period``-periodic, so the counter mod
    ``period`` is the causal state: ``period`` classical states, one qubit.
    """
    surv = [pmd_survival(k, gamma, v, period) for k in range(period + 1)]
    if any(surv[k + 1] >= surv[k] for k in range(period)):
        raise ValueError("Phi(n) must decrease strictly (Wu et al. footnote 2)")
    hazards = [1 - surv[k + 1] / surv[k] for k in range(period)]
    task = renewal_task(hazards, periodic=True, name=name or f"pmd_N{period}")
    task.params = {"kind": "pmd_renewal", "period": period, "gamma": gamma, "v": v}
    return task


def uniform_renewal(n: int = 3, name: str | None = None) -> HMMTask:
    """Yang et al. 2021 / Wang et al. 2026: the number of 0s between consecutive
    1s is uniform on ``0..N-1``. Hazard after ``k`` zeros: ``1 / (N - k)``; ``N`` states.
    """
    hazards = [1 / (n - k) for k in range(n)]
    task = renewal_task(hazards, name=name or f"uniform_renewal_N{n}")
    task.params = {"kind": "uniform_renewal", "n": n}
    return task


def upset_gambler(p: float = 0.7, q: float = 0.8, name: str = "upset_gambler") -> HMMTask:
    """Binder, Thompson & Gu 2018: two causal states A, B (infinite Markov order).

    ``P(0, B | A) = p``, ``P(1, A | A) = 1 - p``, ``P(0, A | B) = q``,
    ``P(1, A | B) = 1 - q``, with ``0 < p < q < 1``.
    """
    edges = [(0, 0, 1, p), (0, 1, 0, 1 - p), (1, 0, 0, q), (1, 1, 0, 1 - q)]
    params = {"kind": "upset_gambler", "p": p, "q": q}
    return HMMTask(_unifilar(2, 2, edges), name=name, params=params)


def rk_golden_mean(
    r: int = 4,
    k: int = 3,
    p: float = 0.5,
    loop_symbol: int = 0,
    name: str | None = None,
) -> HMMTask:
    """R-k golden mean process as drawn in Mahoney et al. 2016, Fig. 2 (R=4, k=3).

    State A emits ``loop_symbol`` and stays (probability ``p``) or emits the
    other symbol ``b`` (``1 - p``) and enters a deterministic chain that emits
    ``r`` copies of ``b`` in total, then ``k`` copies of ``loop_symbol``, and
    returns to A. ``r + k`` states. (Ho et al. 2020 Fig. 3a's "3-2" machine
    is ``r=2, k=3, loop_symbol=1``; see docs/benchmarks.md.)
    """
    if r < 1 or k < 1:
        raise ValueError("r and k must be >= 1")
    a, b = loop_symbol, 1 - loop_symbol
    n = r + k
    edges = [(0, a, 0, p), (0, b, 1 % n, 1 - p)]
    for s in range(1, n):
        sym = b if s < r else a
        edges.append((s, sym, (s + 1) % n, 1.0))
    params = {"kind": "rk_golden_mean", "r": r, "k": k, "p": p, "loop_symbol": loop_symbol}
    return HMMTask(_unifilar(2, n, edges), name=name or f"golden_mean_{r}_{k}", params=params)


def nemo_process(
    p: float = 0.666, q: float = 0.5, loop_symbol: int = 0, name: str = "nemo"
) -> HMMTask:
    """Nemo process (Mahoney et al. 2016 Fig. 3; Ho et al. 2020 Fig. 4a).

    ``A -a|p-> A``, ``A -b|1-p-> B``, ``B -b|1-> C``, ``C -a|1-q-> A``,
    ``C -b|q-> A`` with ``a = loop_symbol``. Mahoney: ``a = 0, q = 1/2``;
    Ho: ``a = 1`` (then ``C -0|1-q-> A``, ``C -1|q-> A``). Infinite Markov
    and cryptic order.
    """
    a, b = loop_symbol, 1 - loop_symbol
    edges = [(0, a, 0, p), (0, b, 1, 1 - p), (1, b, 2, 1.0), (2, b, 0, 1 - q), (2, a, 0, q)]
    params = {"kind": "nemo", "p": p, "q": q, "loop_symbol": loop_symbol}
    return HMMTask(_unifilar(2, 3, edges), name=name, params=params)


# Srinivasan, Gordon & Boots 2018, Sec. 5.4, Eq. 26: a hand-written 6-state,
# 6-symbol HMM, column-stochastic: A[i, j] = P(next = i | j), C[y, j] = P(y | j).
SRINIVASAN_HMM_A = [
    [0.8, 0.01, 0, 0.1, 0.3, 0],
    [0.02, 0.02, 0.1, 0.15, 0.05, 0],
    [0.08, 0.03, 0.1, 0.4, 0.05, 0.5],
    [0.05, 0.04, 0.5, 0.35, 0, 0.5],
    [0.03, 0.5, 0.03, 0, 0.6, 0],
    [0.02, 0.4, 0.27, 0, 0, 0],
]
SRINIVASAN_HMM_C = [
    [0.2, 0, 0.05, 0.95, 0.01, 0.05],
    [0.7, 0.1, 0.05, 0.01, 0.05, 0.05],
    [0.05, 0.8, 0.1, 0.02, 0.05, 0.04],
    [0.04, 0.04, 0.02, 0, 0.84, 0.11],
    [0.01, 0.03, 0.7, 0.01, 0.02, 0.2],
    [0, 0.03, 0.08, 0.01, 0.03, 0.55],
]


def srinivasan_hmm(n_symbols: int | None = 8, name: str = "srinivasan_hmm6") -> HMMTask:
    """The 6-state / 6-symbol HMM of Srinivasan et al. 2018 Eq. 26 (Moore form).

    In state ``j`` emit ``y ~ C[:, j]``, then move to ``i ~ A[:, j]``, so
    ``L[y, j, i] = C[y, j] A[i, j]``. Started from the stationary distribution
    (the paper gives none). Non-unifilar. Padded to ``n_symbols`` (default 8).
    """
    a = torch.tensor(SRINIVASAN_HMM_A, dtype=REAL_DTYPE)
    c = torch.tensor(SRINIVASAN_HMM_C, dtype=REAL_DTYPE)
    lab = torch.einsum("yj,ij->yji", c, a)
    params = {"kind": "srinivasan_hmm", "n_symbols": n_symbols}
    return HMMTask(pad_alphabet(lab, n_symbols), name=name, params=params)


# -- retention probes (issue #2, section 7) -----------------------------------------
def delayed_recall(tau: int = 4, eps: float = 0.0, name: str | None = None) -> HMMTask:
    """Delayed recall at delay ``tau``: trials of ``tau + 2`` binary symbols.

    Each trial is a uniform cue bit ``c``, ``tau`` i.i.d. uniform distractor
    bits, then the recall symbol, which equals ``c``. Trials repeat
    back to back; every sequence starts at a trial boundary. State =
    (phase, cue), index ``2 * phase + cue``. The optimal predictor is certain
    at the recall position (NLL 0, accuracy 1) and at chance elsewhere, so the
    per-symbol optimum is ``ln 2`` times the fraction of non-recall positions.
    Memory need: one bit held for ``tau + 1`` steps, plus the trial phase
    (``tau + 2`` values) that a time-homogeneous model must also track.

    ``eps`` (default 0): the recall symbol is wrong with probability ``eps``, so
    every emission probability lies in ``[eps, 1 - eps]``. The recall-position
    optimum is then ``H(eps)`` nats and accuracy ``1 - eps``.
    """
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if not 0 <= eps <= 0.5:
        raise ValueError("eps must lie in [0, 1/2]")
    period = tau + 2
    n = 2 * period
    edges = []
    for c in (0, 1):
        for y in (0, 1):  # cue: the state's own cue bit is irrelevant at phase 0
            edges.append((2 * 0 + c, y, 2 * 1 + y, 0.5))
        for ph in range(1, period - 1):
            for y in (0, 1):
                edges.append((2 * ph + c, y, 2 * (ph + 1) + c, 0.5))
        edges.append((2 * (period - 1) + c, c, 0, 1.0 - eps))
        if eps > 0:
            edges.append((2 * (period - 1) + c, 1 - c, 0, eps))
    init = torch.zeros(n, dtype=REAL_DTYPE)
    init[0] = 1.0
    params = {"kind": "delayed_recall", "tau": tau, "eps": eps}
    default = f"recall_tau{tau}" + (f"_eps{eps:g}" if eps else "")
    return HMMTask(_unifilar(2, n, edges), init, name=name or default, params=params)


def recall_positions(tau: int, length: int) -> torch.Tensor:
    """0-based positions of the recall symbols in a ``delayed_recall(tau)`` sequence."""
    return torch.arange(tau + 1, length, tau + 2)


# -- marked delayed recall (issue #10) ------------------------------------------------
DISTRACTOR_A, DISTRACTOR_B, CUE0, CUE1 = 0, 1, 2, 3


def marked_recall(
    tau: float = 4,
    eps: float = 0.0,
    timing: str = "geometric",
    name: str | None = None,
) -> HMMTask:
    """Delayed recall with the phase marked in a 4-symbol alphabet (issue #10).

    Symbols: ``0`` = distractor-a, ``1`` = distractor-b, ``2`` = cue-0, ``3`` = cue-1.
    On two output ancillas the first bit is the *type* (0 distractor, 1 cue-type)
    and the second the *value*. In architecture A an outcome bit 1 projects the
    memory qubit it reads, so the frequent symbols (distractors) are the ones
    whose type bit is 0 (docs/phase3e-retention.md).
    Cue-type symbols alternate in role: the 1st, 3rd, ... are *cues* (uniform
    values), the 2nd, 4th, ... are *recalls*, equal to the preceding cue with
    probability ``1 - eps``. Every sequence starts before a cue.

    ``timing = "geometric"`` (default): every symbol is cue-type with probability
    ``q = 1 / (tau + 1)`` and a distractor (``a`` or ``b``, uniform) otherwise,
    independently of everything else. The symbol *types* are i.i.d., so the
    distractors carry no information about the cue or the phase, and the number of
    distractors between a cue and its recall is geometric with mean ``tau``
    (``P(D = k) = q (1 - q)^k``). Causal states: ``S`` (next cue-type symbol is a
    fresh cue), ``W0``, ``W1`` (holding the cue). Three for every ``tau``: one cue
    bit plus one phase bit, no clock.

    ``timing = "fixed"``: a cue, exactly ``tau`` distractors, the recall, then the
    next cue at once (``tau`` must be an integer). The marking does not remove the
    clock: the source must count the distractors to know when the recall comes, so
    it has ``2 tau + 3`` causal states, as ``delayed_recall``.

    States: ``0`` = ``S``; geometric: ``1 + c`` = ``W_c``; fixed: ``1 + 2 k + c`` =
    cue ``c`` with ``k`` distractors seen (``k = 0..tau``).
    """
    if not 0 <= eps <= 0.5:
        raise ValueError("eps must lie in [0, 1/2]")
    cue = (CUE0, CUE1)
    edges: list[tuple[int, int, int, float]] = []
    if timing == "geometric":
        if tau < 0:
            raise ValueError("tau must be >= 0")
        q = 1.0 / (tau + 1.0)
        n_states = 3
        for s in range(3):
            for d in (DISTRACTOR_A, DISTRACTOR_B):
                if q < 1:
                    edges.append((s, d, s, (1 - q) / 2))
        for c in (0, 1):
            edges.append((0, cue[c], 1 + c, q / 2))
            edges.append((1 + c, cue[c], 0, q * (1 - eps)))
            if eps > 0:
                edges.append((1 + c, cue[1 - c], 0, q * eps))
    elif timing == "fixed":
        if tau < 0 or int(tau) != tau:
            raise ValueError("fixed timing needs an integer tau >= 0")
        tau = int(tau)
        n_states = 1 + 2 * (tau + 1)
        for c in (0, 1):
            edges.append((0, cue[c], 1 + c, 0.5))
            for k in range(tau):
                for d in (DISTRACTOR_A, DISTRACTOR_B):
                    edges.append((1 + 2 * k + c, d, 1 + 2 * (k + 1) + c, 0.5))
            last = 1 + 2 * tau + c
            edges.append((last, cue[c], 0, 1 - eps))
            if eps > 0:
                edges.append((last, cue[1 - c], 0, eps))
    else:
        raise ValueError("timing must be 'geometric' or 'fixed'")
    init = torch.zeros(n_states, dtype=REAL_DTYPE)
    init[0] = 1.0
    params = {"kind": "marked_recall", "tau": tau, "eps": eps, "timing": timing}
    default = f"marked_{timing[:3]}_tau{tau:g}" + (f"_eps{eps:g}" if eps else "")
    return HMMTask(_unifilar(4, n_states, edges), init, name=name or default, params=params)


def marked_recall_events(sequences: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cues, recalls and delays in ``marked_recall`` sequences (either timing).

    Returns boolean masks ``cue``, ``recall`` and ``distractor`` of shape ``(B, T)``,
    ``cue_value`` (the value of the trial's cue at every position from the cue up to
    and including its recall, ``-1`` elsewhere), ``lag`` (steps since the cue at the
    same positions: 1 = the step right after the cue, ``-1`` elsewhere) and
    ``delay`` (at recall positions: distractors since the cue, ``-1`` elsewhere).
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    if seq.dim() == 1:
        seq = seq[None]
    cue_type = (seq == CUE0) | (seq == CUE1)
    count = torch.cumsum(cue_type.long(), dim=1)  # cue-type symbols up to and incl. t
    cue = cue_type & (count % 2 == 1)
    recall = cue_type & (count % 2 == 0)
    value = (seq == CUE1).long()
    b, t = seq.shape
    cue_value = torch.full((b, t), -1, dtype=torch.long)
    lag = torch.full((b, t), -1, dtype=torch.long)
    delay = torch.full((b, t), -1, dtype=torch.long)
    cur_val = torch.full((b,), -1, dtype=torch.long)
    cur_pos = torch.full((b,), -1, dtype=torch.long)
    for i in range(t):
        is_cue, is_rec = cue[:, i], recall[:, i]
        waiting = cur_val >= 0
        cue_value[:, i] = torch.where(waiting, cur_val, cue_value[:, i])
        lag[:, i] = torch.where(waiting, i - cur_pos, lag[:, i])
        delay[:, i] = torch.where(is_rec, i - cur_pos - 1, delay[:, i])
        cur_val = torch.where(is_cue, value[:, i], torch.where(is_rec, -1, cur_val))
        cur_pos = torch.where(is_cue, torch.full_like(cur_pos, i), cur_pos)
    return {
        "cue": cue,
        "recall": recall,
        "distractor": ~cue_type,
        "cue_value": cue_value,
        "lag": lag,
        "delay": delay,
    }


def delayed_recall_events(tau: int, sequences: torch.Tensor) -> dict[str, torch.Tensor]:
    """``cue_value`` and ``lag`` for ``delayed_recall(tau)``, as ``marked_recall_events``.

    Positions ``k (tau + 2) + j`` with ``j = 1..tau+1`` carry the trial's cue and lag
    ``j`` (the recall is lag ``tau + 1``); the rest are ``-1``.
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    if seq.dim() == 1:
        seq = seq[None]
    b, t = seq.shape
    pos = torch.arange(t)
    phase = pos % (tau + 2)
    start = pos - phase
    lag = torch.where(phase >= 1, phase, torch.full_like(phase, -1)).expand(b, t)
    cue_value = torch.where(phase >= 1, seq[:, start], torch.full_like(seq, -1))
    return {"cue_value": cue_value, "lag": lag.clone()}


def parity_task(k: int = 3, noise: float = 0.0, name: str | None = None) -> HMMTask:
    """Noisy parity of the last ``k`` symbols: ``y_t = y_{t-k} xor ... xor y_{t-1} xor e_t``.

    ``e_t ~ Bernoulli(noise)`` (``noise`` is the probe's ``eps``; default 0, a
    deterministic parity after the first ``k`` symbols). An order-``k`` Markov
    chain whose next symbol depends on all ``k`` previous symbols jointly (no
    proper subset helps: without one of them the prediction is at chance), so
    the information is spread over the window. ``2^k`` causal states,
    ``C_mu = k`` bits. Starts from the uniform window distribution, which is
    stationary for every ``noise`` (the window map is a bijection).
    """
    if k < 1 or not 0 <= noise <= 0.5:
        raise ValueError("need k >= 1 and 0 <= noise <= 0.5")
    rows = []
    for c in range(2**k):
        par = bin(c).count("1") % 2
        rows.append([1 - noise, noise] if par == 0 else [noise, 1 - noise])
    chain = markov_task(k, rows)
    uniform = torch.full((2**k,), 2.0**-k, dtype=REAL_DTYPE)
    params = {"kind": "parity", "k": k, "noise": noise}
    return HMMTask(chain.labelled, uniform, name=name or f"parity_k{k}", params=params)


def mixed_timescales(
    switch_prob: float = 0.02,
    fast_stay: float = 0.9,
    marked: bool = True,
    eps: float = 0.0,
    name: str | None = None,
) -> HMMTask:
    """A slow regime bit ``r`` and a fast Markov bit ``b``.

    Each step the regime flips with probability ``switch_prob`` (slow
    timescale ``~1/switch_prob``); then the data bit follows the regime's fast
    Markov rule: in regime 0 it repeats the previous bit with probability
    ``fast_stay``, in regime 1 it alternates with probability ``fast_stay``.

    ``marked=True`` (default): the symbol is ``2 * flipped + b`` (4 symbols),
    so a switch is visible when it happens but the regime itself must be
    remembered for as long as it lasts (the parity of all switch marks so far).
    The source is unifilar, the regime starts at 0, and the optimum is exact.
    ``marked=False``: the symbol is ``b`` only; the regime is hidden, the
    source is non-unifilar and only the per-sequence oracle is exact.

    ``eps`` (default 0): the data bit is flipped relative to the regime's rule
    with probability ``eps`` (``fast_stay -> (1-eps) fast_stay + eps (1 - fast_stay)``),
    so its conditional probabilities lie in ``[eps, 1 - eps]``.

    State index ``2 * r + b``; ``b`` starts uniform.
    """
    if not 0 <= eps <= 0.5:
        raise ValueError("eps must lie in [0, 1/2]")
    s, f = switch_prob, (1 - eps) * fast_stay + eps * (1 - fast_stay)
    n_sym = 4 if marked else 2
    edges = []
    for r in (0, 1):
        for b in (0, 1):
            for flip, ps in ((0, 1 - s), (1, s)):
                r2 = r ^ flip
                for b2 in (0, 1):
                    same = b2 == b
                    pb = (f if same else 1 - f) if r2 == 0 else (1 - f if same else f)
                    y = 2 * flip + b2 if marked else b2
                    edges.append((2 * r + b, y, 2 * r2 + b2, ps * pb))
    init = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=REAL_DTYPE)
    params = {
        "kind": "mixed_timescales",
        "switch_prob": s,
        "fast_stay": fast_stay,
        "marked": marked,
        "eps": eps,
    }
    tag = ("" if marked else "_hidden") + (f"_eps{eps:g}" if eps else "")
    return HMMTask(
        _unifilar(n_sym, 4, edges),
        init,
        name=name or f"mixed_s{s:g}_f{fast_stay:g}{tag}",
        params=params,
    )


# -- sources generated by a quantum instrument (HQMMs) -----------------------------
class QuantumSourceTask:
    """A source generated by a quantum instrument: an HQMM with Kraus operators.

    Same interface as ``HMMTask`` (``sample``, ``predictive``, ``log_probs``,
    ``step_entropies``, ``optimal_nll``, ``entropy_rate``, ``describe``). The
    exact Bayes-optimal predictor is the instrument itself run forward on the
    observed symbols: ``p(y | rho) = Tr(M_y rho)``, then ``rho <- K_y rho K_y^+ / p``.

    Args:
        instrument: a ``qmem.Instrument`` with ``n_outcomes`` symbols.
        initial: ``(d, d)`` density matrix; default the channel's fixed point,
            so the source is stationary.
    """

    def __init__(
        self,
        instrument: Instrument,
        initial: torch.Tensor | None = None,
        name: str = "hqmm",
        params: dict[str, Any] | None = None,
    ):
        defect = float(instrument.completeness_defect())
        if defect > 1e-10:
            raise ValueError(f"instrument is not trace preserving (defect {defect:.2e})")
        self.instrument = instrument
        rho = channel_fixed_point(instrument) if initial is None else initial.to(DTYPE)
        self.initial = rho
        self.name = name
        self.params = dict(params or {})
        self._rate: float | None = None

    @property
    def n_symbols(self) -> int:
        return self.instrument.n_outcomes

    @property
    def dim(self) -> int:
        return self.instrument.dim

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_symbols": self.n_symbols,
            "quantum_dim": self.dim,
            "params": self.params,
            "entropy_rate_nats": self.entropy_rate(),
        }

    def is_unifilar(self) -> bool:
        return False

    def sample(self, n: int, length: int, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        rho = self.initial.expand(n, -1, -1)
        out = torch.empty(n, length, dtype=torch.long)
        for t in range(length):
            p = self.instrument.probabilities(rho).clamp(min=0)  # rounding: -1e-17
            y = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            unnorm = self.instrument.branch(rho, y)
            rho = unnorm / trace(unnorm).real[:, None, None]
            out[:, t] = y
        return out

    def _check(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = torch.as_tensor(sequences, dtype=torch.long)
        if seq.dim() == 1:
            seq = seq[None]
        if seq.numel() and (seq.min() < 0 or seq.max() >= self.n_symbols):
            raise ValueError(f"tokens must lie in [0, {self.n_symbols})")
        return seq

    def predictive(self, sequences: torch.Tensor) -> torch.Tensor:
        """Optimal ``p(. | y_<t)`` for ``t = 1..T+1``, shape ``(B, T+1, Y)``."""
        seq = self._check(sequences)
        rho = self.initial.expand(seq.shape[0], -1, -1)
        out = [self.instrument.probabilities(rho)]
        for t in range(seq.shape[1]):
            unnorm = self.instrument.branch(rho, seq[:, t])
            rho = unnorm / trace(unnorm).real[:, None, None]
            out.append(self.instrument.probabilities(rho))
        return torch.stack(out, dim=1)

    def log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = self._check(sequences)
        p = self.predictive(seq)[:, :-1].gather(-1, seq[..., None]).squeeze(-1)
        return torch.log(p).nan_to_num(nan=-math.inf, neginf=-math.inf)

    def _successors(
        self, beliefs: torch.Tensor, mass: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """One step of the belief enumeration: ``(next beliefs, their mass, H(Y | beliefs))``.

        Identical normalised conditional states (to ``1e-11``) are merged.
        """
        inst = self.instrument
        q = inst.probabilities(beliefs).clamp(min=0)  # (K, Y); rounding can give -1e-17
        h = float((mass * _entropy_nats(q)).sum())
        br = inst.branches(beliefs)  # (K, Y, d, d)
        m = mass[:, None] * q
        keep = m > 0
        br, m = br[keep], m[keep]
        br = br / trace(br).real[:, None, None]
        flat = torch.cat([br.real.flatten(1), br.imag.flatten(1)], dim=1)
        key = torch.round(flat * 10**_ROUND)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        first = torch.full((len(uniq),), len(inv), dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(inv)), reduce="amin")
        merged = torch.zeros(len(uniq), dtype=REAL_DTYPE).index_add_(0, inv, m)
        return br[first], merged, h

    def step_entropies(self, length: int, max_beliefs: int = 200_000) -> torch.Tensor:
        """Exact ``H(Y_t | Y_<t)`` by enumerating distinct conditional states."""
        beliefs, mass = self.initial[None].clone(), torch.ones(1, dtype=REAL_DTYPE)
        out = []
        for _ in range(length):
            beliefs, mass, h = self._successors(beliefs, mass)
            out.append(h)
            if len(mass) > max_beliefs:
                raise RuntimeError(f"{len(mass)} belief states; reduce length")
        return torch.tensor(out, dtype=REAL_DTYPE)

    def optimal_nll(self, length: int) -> float:
        return float(self.step_entropies(length).mean())

    def entropy_rate(self, max_length: int = 4000, tol: float = 1e-13) -> float:
        """``lim H(Y_t | Y_<t)``: step entropies until successive values agree to ``tol``.

        Raises if they have not converged by ``max_length`` (or the belief
        enumeration exceeds 200k states). The result is cached.
        """
        if self._rate is not None:
            return self._rate
        beliefs, mass = self.initial[None].clone(), torch.ones(1, dtype=REAL_DTYPE)
        prev = math.nan
        for _ in range(max_length):
            beliefs, mass, h = self._successors(beliefs, mass)
            if abs(h - prev) < tol:
                self._rate = h
                return h
            prev = h
            if len(mass) > 200_000:
                break
        raise RuntimeError("step entropies did not converge; no exact entropy rate")


def channel_fixed_point(instrument: Instrument) -> torch.Tensor:
    """Stationary state of the outcome-averaged channel (eigenvalue-1 eigenvector).

    Assumes the fixed point is unique (raises otherwise).
    """
    d = instrument.dim
    k = instrument.kraus.reshape(-1, d, d)
    # vec(K rho K^+) = (K ⊗ conj K) vec(rho) for row-major vec
    sup = torch.einsum("aij,akl->ikjl", k, k.conj()).reshape(d * d, d * d)
    ev, vecs = torch.linalg.eig(sup)
    one = (ev - 1).abs() < 1e-9
    if int(one.sum()) != 1:
        raise ValueError(f"channel has {int(one.sum())} fixed points; pass initial explicitly")
    rho = vecs[:, one.nonzero()[0, 0]].reshape(d, d)
    rho = rho / torch.diagonal(rho).sum()
    return ((rho + rho.conj().T) / 2).to(DTYPE)


def probability_clock(name: str = "probability_clock") -> QuantumSourceTask:
    """Srinivasan, Gordon & Boots 2018 Eq. 16: a 2-state, 2-symbol HQMM.

    ``K_1 = R(0.6) diag(0.6, 1)``, ``K_2 = diag(0.8, 0)`` (paper's symbols 1, 2
    are our 0, 1). Started from the channel's fixed point (the paper gives no
    initial state; it discards a burn-in of 1000 steps).
    """
    c, s = math.cos(0.6), math.sin(0.6)
    k = torch.zeros(2, 1, 2, 2, dtype=DTYPE)
    k[0, 0] = torch.tensor([[0.6 * c, -s], [0.6 * s, c]], dtype=DTYPE)
    k[1, 0] = torch.tensor([[0.8, 0.0], [0.0, 0.0]], dtype=DTYPE)
    return QuantumSourceTask(Instrument(k), name=name, params={"kind": "probability_clock"})


def stern_gerlach(n_symbols: int | None = 8, name: str = "stern_gerlach") -> QuantumSourceTask:
    """Srinivasan, Gordon & Boots 2018 Eqs. 23-25: a spin-1/2 measured along a
    uniformly random axis each step; 6 symbols (+z, -z, +x, -x, +y, -y = 0..5).

    Started from the fixed point ``I/2``. Padded to ``n_symbols`` (default 8)
    with zero Kraus operators.
    """
    a = 1 / math.sqrt(3)
    b = 1 / (2 * math.sqrt(3))
    j = 1j
    mats = [
        [[a, 0], [0, 0]],
        [[0, 0], [0, a]],
        [[b, b], [b, b]],
        [[b, -b], [-b, b]],
        [[b, -j * b], [j * b, b]],
        [[b, j * b], [-j * b, b]],
    ]
    total = len(mats) if n_symbols is None else n_symbols
    k = torch.zeros(total, 1, 2, 2, dtype=DTYPE)
    for y, m in enumerate(mats):
        k[y, 0] = torch.tensor(m, dtype=DTYPE)
    params = {"kind": "stern_gerlach", "n_symbols": n_symbols}
    return QuantumSourceTask(Instrument(k), name=name, params=params)


Source = HMMTask | QuantumSourceTask

TASK_KINDS: dict[str, Callable[..., Source]] = {
    "iid": iid_task,
    "markov": markov_task,
    "even": even_process,
    "golden_mean": golden_mean_process,
    "hmm": hmm_task,
    "perturbed_coin": perturbed_coin,
    "ghafari": ghafari_process,
    "renewal": renewal_task,
    "pmd_renewal": pmd_renewal,
    "uniform_renewal": uniform_renewal,
    "upset_gambler": upset_gambler,
    "rk_golden_mean": rk_golden_mean,
    "nemo": nemo_process,
    "srinivasan_hmm": srinivasan_hmm,
    "delayed_recall": delayed_recall,
    "marked_recall": marked_recall,
    "parity": parity_task,
    "mixed_timescales": mixed_timescales,
    "probability_clock": probability_clock,
    "stern_gerlach": stern_gerlach,
}


def register_task(kind: str, constructor: Callable[..., Source]) -> None:
    """Add a task kind (Phase 3: delayed recall, parity, ...)."""
    if kind in TASK_KINDS:
        raise ValueError(f"task kind {kind!r} already registered")
    TASK_KINDS[kind] = constructor


def make_task(spec: dict[str, Any]) -> Source:
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
