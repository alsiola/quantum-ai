"""Order-``k`` Markov (n-gram) count model with add-alpha smoothing.

This is the Phase 2 count model (``qmem.analysis.markov_baseline_nll``), wrapped
in the common model interface. It reuses ``qmem.analysis._markov_counts``:
``counts[j][context, y]`` for context lengths ``j = 0..k``, pooled over positions.
Position ``t`` (0-based) predicts from the last ``min(t, k)`` symbols, so the
first ``k`` positions use the shorter tables. Every conditional is the smoothed
``(count + alpha) / (total + Y alpha)``, so it is an exactly normalised
autoregressive model and ``sample`` draws from the same conditionals.

``fit_ngram`` picks ``k`` and ``alpha`` on the validation split.
"""

from __future__ import annotations

from typing import Any

import torch

from ..analysis import _markov_counts
from ..linalg import REAL_DTYPE
from .base import RecurrentModel


class NGramModel(RecurrentModel):
    family = "ngram"

    def __init__(self, counts: list[torch.Tensor], order: int, alpha: float, n_symbols: int):
        super().__init__(n_symbols)
        if len(counts) != order + 1:
            raise ValueError("need count tables for context lengths 0..order")
        self.order = order
        self.alpha = alpha
        for j, c in enumerate(counts):
            self.register_buffer(f"counts{j}", torch.as_tensor(c, dtype=REAL_DTYPE))

    @classmethod
    def fit(cls, train: torch.Tensor, order: int, alpha: float, n_symbols: int) -> NGramModel:
        return cls(_markov_counts(train, order, n_symbols), order, alpha, n_symbols)

    def tables(self) -> list[torch.Tensor]:
        out = []
        for j in range(self.order + 1):
            c = getattr(self, f"counts{j}") + self.alpha
            out.append(c / c.sum(-1, keepdim=True))
        return out

    def prepare(self) -> list[torch.Tensor]:
        return self.tables()

    def init_state(self, ctx, batch: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(batch, dtype=torch.long), 0  # (context index, its length)

    def probs(self, ctx, state) -> torch.Tensor:
        idx, j = state
        return ctx[j][idx]

    def update(self, ctx, state, y: torch.Tensor):
        idx, j = state
        nj = min(j + 1, self.order)
        return (idx * self.n_symbols + y) % (self.n_symbols**nj), nj

    def n_parameters(self) -> int:
        """Free probabilities in the tables used (``sum_j Y^j (Y - 1)``)."""
        y = self.n_symbols
        return sum(y**j * (y - 1) for j in range(self.order + 1))

    def state_dim(self) -> int:
        return self.n_symbols**self.order

    def state_kind(self) -> str:
        return "last k symbols (Y^k contexts)"

    def flops_per_step(self) -> int:
        return 1  # a table lookup

    def extra_info(self) -> dict[str, Any]:
        return {"order": self.order, "alpha": self.alpha}


@torch.no_grad()
def fit_ngram(
    train: torch.Tensor,
    val: torch.Tensor,
    n_symbols: int,
    orders: list[int],
    alphas: list[float],
) -> tuple[NGramModel, list[dict[str, Any]]]:
    """Fit every ``(k, alpha)``; return the model with the best validation NLL and the grid."""
    counts = _markov_counts(train, max(orders), n_symbols)
    grid = []
    best = None
    for k in orders:
        for a in alphas:
            m = NGramModel(counts[: k + 1], k, a, n_symbols)
            v = float(-m.step_log_probs(val).mean())
            grid.append({"order": k, "alpha": a, "val_nll": v})
            if best is None or v < best[0]:
                best = (v, m)
    assert best is not None
    return best[1], grid
