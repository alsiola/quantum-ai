"""Sequence data helpers: toy Markov chains, exhaustive enumeration, fixed splits.

The memory tasks themselves live in ``qmem.tasks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SequenceSource(Protocol):
    def sample(self, n: int, length: int, seed: int) -> torch.Tensor: ...


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


@dataclass(frozen=True)
class Splits:
    """Fixed train / validation / test sequences drawn from recorded seeds."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    seeds: dict[str, int]

    def describe(self) -> dict[str, object]:
        return {
            "seeds": dict(self.seeds),
            "shapes": {k: list(getattr(self, k).shape) for k in ("train", "val", "test")},
        }


def split_seeds(seed: int) -> dict[str, int]:
    """Seeds for the three splits, derived deterministically from one data seed."""
    return {"train": 1000 * seed + 1, "val": 1000 * seed + 2, "test": 1000 * seed + 3}


def make_splits(
    task: SequenceSource, n_train: int, n_val: int, n_test: int, length: int, seed: int
) -> Splits:
    """Draw the splits for ``task`` from ``seed`` (the same data for every model and phase)."""
    seeds = split_seeds(seed)
    return Splits(
        train=task.sample(n_train, length, seeds["train"]),
        val=task.sample(n_val, length, seeds["val"]),
        test=task.sample(n_test, length, seeds["test"]),
        seeds={"data_seed": seed, **seeds},
    )
