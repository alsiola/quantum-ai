"""Tiny synthetic datasets (Phase 2+ will add the real memory tasks)."""

from __future__ import annotations

import torch


def markov_sequences(
    n: int,
    length: int,
    transition: torch.Tensor,
    seed: int,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``n`` first-order Markov chains, shape ``(n, length)``.

    ``transition[a, b] = p(y_t = b | y_{t-1} = a)``; ``initial`` defaults to uniform.
    """
    transition = torch.as_tensor(transition, dtype=torch.float64)
    k = transition.shape[0]
    initial = torch.full((k,), 1.0 / k) if initial is None else torch.as_tensor(initial)
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(n, length, dtype=torch.long)
    out[:, 0] = torch.multinomial(initial.expand(n, k).double(), 1, generator=gen).squeeze(-1)
    for t in range(1, length):
        out[:, t] = torch.multinomial(transition[out[:, t - 1]], 1, generator=gen).squeeze(-1)
    return out


def all_sequences(n_symbols: int, length: int) -> torch.Tensor:
    """Every sequence of ``length`` tokens over ``n_symbols``: ``(n_symbols**length, length)``.

    Row ``i`` is the base-``n_symbols`` expansion of ``i`` (most significant first).
    """
    idx = torch.arange(n_symbols**length)
    powers = n_symbols ** torch.arange(length - 1, -1, -1)
    return (idx[:, None] // powers) % n_symbols
