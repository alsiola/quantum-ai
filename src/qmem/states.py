"""Initial states and density-matrix diagnostics."""

from __future__ import annotations

import torch

from .linalg import DTYPE, REAL_DTYPE, ket_to_dm, trace

NAMED_STATES = ("zero", "plus", "mixed", "random_pure")


def initial_state(
    spec: str | torch.Tensor, n_qubits: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Build a ``(d, d)`` density matrix.

    ``spec`` is one of ``"zero"`` (``|0...0>``), ``"plus"`` (``|+>^n``),
    ``"mixed"`` (``I/d``), ``"random_pure"`` (Haar-random ket drawn from
    ``generator``), a ``(d,)`` ket, or a ``(d, d)`` density matrix.
    """
    d = 2**n_qubits
    if isinstance(spec, torch.Tensor):
        t = spec.to(DTYPE)
        if t.shape == (d,):
            return ket_to_dm(t / torch.linalg.vector_norm(t))
        if t.shape == (d, d):
            return t
        raise ValueError(f"state tensor has shape {tuple(t.shape)}, expected ({d},) or ({d},{d})")
    if spec == "zero":
        psi = torch.zeros(d, dtype=DTYPE)
        psi[0] = 1
        return ket_to_dm(psi)
    if spec == "plus":
        return ket_to_dm(torch.full((d,), d**-0.5, dtype=DTYPE))
    if spec == "mixed":
        return torch.eye(d, dtype=DTYPE) / d
    if spec == "random_pure":
        re = torch.randn(d, generator=generator, dtype=REAL_DTYPE)
        im = torch.randn(d, generator=generator, dtype=REAL_DTYPE)
        psi = torch.complex(re, im)
        return ket_to_dm(psi / torch.linalg.vector_norm(psi))
    raise ValueError(f"unknown initial state {spec!r}; choose from {NAMED_STATES}")


def random_density_matrix(
    n_qubits: int, generator: torch.Generator | None = None, rank: int | None = None
) -> torch.Tensor:
    """Random full-rank (or rank-``rank``) density matrix (Ginibre ensemble)."""
    d = 2**n_qubits
    r = rank or d
    g = torch.complex(
        torch.randn(d, r, generator=generator, dtype=REAL_DTYPE),
        torch.randn(d, r, generator=generator, dtype=REAL_DTYPE),
    )
    rho = g @ g.conj().T
    return rho / trace(rho)


def hermiticity_error(rho: torch.Tensor) -> torch.Tensor:
    return (rho - rho.conj().transpose(-1, -2)).abs().amax(dim=(-2, -1))


def trace_error(rho: torch.Tensor) -> torch.Tensor:
    return (trace(rho) - 1).abs()


def min_eigenvalue(rho: torch.Tensor) -> torch.Tensor:
    herm = (rho + rho.conj().transpose(-1, -2)) / 2
    return torch.linalg.eigvalsh(herm)[..., 0]


def is_density_matrix(rho: torch.Tensor, atol: float = 1e-10) -> bool:
    """Hermitian, unit trace and positive semidefinite to within ``atol`` (all batch items)."""
    return bool(
        (hermiticity_error(rho) <= atol).all()
        and (trace_error(rho) <= atol).all()
        and (min_eigenvalue(rho) >= -atol).all()
    )


def purity(rho: torch.Tensor) -> torch.Tensor:
    """``Tr(rho^2)``, real."""
    return torch.einsum("...ij,...ji->...", rho, rho).real


def von_neumann_entropy(rho: torch.Tensor) -> torch.Tensor:
    """``-Tr(rho log2 rho)`` in bits (eigenvalues below 1e-15 are dropped)."""
    ev = torch.linalg.eigvalsh((rho + rho.conj().transpose(-1, -2)) / 2).clamp(min=0)
    terms = torch.where(ev > 1e-15, -ev * torch.log2(ev.clamp(min=1e-300)), torch.zeros_like(ev))
    return terms.sum(-1)
