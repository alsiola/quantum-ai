"""The interface every sequence model shares, and a generic recurrent implementation.

Every baseline (and architecture A, ``qmem.QuantumMemoryModel``) exposes:

* ``step_log_probs(seqs)``: teacher-forced ``log p(y_t | y_<t)``, shape ``(B, T)``;
* ``log_prob(seqs)``: ``log p(y_1..y_T)``, shape ``(B,)``;
* ``predictive_probs(seqs)``: ``p(. | y_<t)`` for ``t = 1..T+1``, ``(B, T+1, Y)``;
* ``sample(n, length, seed)``: generation from the model's own distribution;
* ``n_parameters()``, ``diagnostics`` (a counter of zero / tiny probabilities);
* ``describe()``: family, parameter count, state dimension and compute cost.

That is all ``qmem.train.train`` and ``qmem.train.evaluate`` use, so every model
trains with the same trainer, data splits and metrics.

``RecurrentModel`` implements the interface from three pieces: an initial
state, ``probs(ctx, state)`` and ``update(ctx, state, y)``. ``ctx`` is computed
once per call from the parameters (``prepare``), e.g. normalised transition
matrices. As in architecture A, teacher forcing and generation are separate
code paths: ``step_log_probs`` conditions on given tokens; ``sample`` draws
tokens from ``probs`` with an explicit ``torch.Generator`` and updates on the
drawn token.

Probabilities are never clamped. A token with ``p = 0`` gets ``log p = -inf``
and is counted in ``diagnostics["zero_prob_steps"]``; ``0 < p < tiny_prob`` is
counted in ``diagnostics["tiny_prob_steps"]``.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import torch
from torch import nn

from ..linalg import REAL_DTYPE

logger = logging.getLogger("qmem")


def check_sequences(sequences: torch.Tensor, n_symbols: int) -> torch.Tensor:
    seq = torch.as_tensor(sequences, dtype=torch.long)
    if seq.dim() == 1:
        seq = seq[None]
    if seq.dim() != 2:
        raise ValueError("sequences must have shape (B, T)")
    if seq.numel() and (seq.min() < 0 or seq.max() >= n_symbols):
        raise ValueError(f"tokens must lie in [0, {n_symbols})")
    return seq


def guarded_log(p: torch.Tensor, diagnostics: Counter, tiny_prob: float) -> torch.Tensor:
    """``log p`` with zero and tiny probabilities counted (never clamped)."""
    n_zero = int((p <= 0).sum())
    n_tiny = int(((p > 0) & (p < tiny_prob)).sum())
    if n_zero:
        diagnostics["zero_prob_steps"] += n_zero
    if n_tiny:
        diagnostics["tiny_prob_steps"] += n_tiny
        logger.warning("qmem: %d step(s) with 0 < p(y) < %g", n_tiny, tiny_prob)
    return torch.log(p)


class SequenceModel(nn.Module):
    """Base class: bookkeeping shared by all baselines."""

    family: str = "base"

    def __init__(self, n_symbols: int, tiny_prob: float = 1e-10):
        super().__init__()
        self.n_symbols = n_symbols
        self.tiny_prob = tiny_prob
        self.diagnostics: Counter[str] = Counter()

    def n_parameters(self) -> int:
        """Trainable real parameters (complex parameters are stored as real pairs)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_dim(self) -> int:
        raise NotImplementedError

    def state_kind(self) -> str:
        raise NotImplementedError

    def flops_per_step(self) -> int:
        """Rough real multiply-adds per symbol in teacher forcing (analytic count)."""
        raise NotImplementedError

    def extra_info(self) -> dict[str, Any]:
        return {}

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_symbols": self.n_symbols,
            "n_parameters": self.n_parameters(),
            "state_dim": self.state_dim(),
            "state_kind": self.state_kind(),
            "flops_per_step": self.flops_per_step(),
            **self.extra_info(),
        }

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.step_log_probs(sequences).sum(-1)


class RecurrentModel(SequenceModel):
    """A model defined by ``init_state``, ``probs`` and ``update`` (see module doc)."""

    def prepare(self) -> Any:
        return None

    def init_state(self, ctx: Any, batch: int) -> Any:
        raise NotImplementedError

    def probs(self, ctx: Any, state: Any) -> torch.Tensor:
        """``p(. | state)``, real ``(B, Y)``, rows summing to 1."""
        raise NotImplementedError

    def update(self, ctx: Any, state: Any, y: torch.Tensor) -> Any:
        """State after observing tokens ``y`` of shape ``(B,)``."""
        raise NotImplementedError

    def observe(self, ctx: Any, state: Any, y: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """``(p(. | state), update(state, y))``; override to share work between the two."""
        return self.probs(ctx, state), self.update(ctx, state, y)

    # -- teacher forcing -------------------------------------------------------
    def step_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = check_sequences(sequences, self.n_symbols)
        ctx = self.prepare()
        state = self.init_state(ctx, seq.shape[0])
        out = []
        for t in range(seq.shape[1]):
            probs, state = self.observe(ctx, state, seq[:, t])
            p = probs.gather(-1, seq[:, t, None]).squeeze(-1)
            out.append(guarded_log(p, self.diagnostics, self.tiny_prob))
        if not out:
            return torch.zeros(seq.shape[0], 0, dtype=REAL_DTYPE)
        return torch.stack(out, dim=1)

    def predictive_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = check_sequences(sequences, self.n_symbols)
        ctx = self.prepare()
        state = self.init_state(ctx, seq.shape[0])
        out = [self.probs(ctx, state)]
        for t in range(seq.shape[1]):
            state = self.update(ctx, state, seq[:, t])
            out.append(self.probs(ctx, state))
        return torch.stack(out, dim=1)

    # -- generation ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, length: int, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        ctx = self.prepare()
        state = self.init_state(ctx, n)
        out = torch.empty(n, length, dtype=torch.long)
        for t in range(length):
            p = self.probs(ctx, state)
            y = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            out[:, t] = y
            state = self.update(ctx, state, y)
        return out
