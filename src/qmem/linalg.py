"""Dtypes, qubit-ordering conventions and small linear-algebra helpers.

Conventions used across the package:

* Every complex tensor is ``torch.complex128`` (``DTYPE``); every real tensor is
  ``torch.float64`` (``REAL_DTYPE``).
* Qubits are numbered from 0 and ordered big-endian: qubit 0 is the most
  significant bit of a basis-state index, so ``|q0 q1 ... q_{n-1}>`` has index
  ``sum_k q_k 2^(n-1-k)``. Matrices follow ``torch.kron(A_q0, A_q1, ...)``.
* Density matrices may carry any number of leading batch dimensions:
  shape ``(..., d, d)``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch

DTYPE = torch.complex128
REAL_DTYPE = torch.float64


def dagger(x: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose over the last two dimensions."""
    return x.conj().transpose(-1, -2)


def kron(*ops: torch.Tensor) -> torch.Tensor:
    """Kronecker product of 2-D operators, left to right (qubit 0 first)."""
    return functools.reduce(torch.kron, ops)


def eye(dim: int) -> torch.Tensor:
    return torch.eye(dim, dtype=DTYPE)


def basis_ket(index: int, dim: int) -> torch.Tensor:
    ket = torch.zeros(dim, dtype=DTYPE)
    ket[index] = 1.0
    return ket


def ket_to_dm(psi: torch.Tensor) -> torch.Tensor:
    """``|psi><psi|`` for a (possibly batched) state vector of shape ``(..., d)``."""
    return psi.unsqueeze(-1) * psi.conj().unsqueeze(-2)


def trace(x: torch.Tensor) -> torch.Tensor:
    """Trace over the last two dimensions (complex)."""
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def embed(op: torch.Tensor, qubits: Sequence[int], n_qubits: int) -> torch.Tensor:
    """Lift a ``k``-qubit operator acting on ``qubits`` (in that order) to ``n_qubits``.

    ``op`` is a ``(2^k, 2^k)`` matrix whose own qubit ordering is ``qubits``;
    the result is the ``(2^n, 2^n)`` operator ``op ⊗ I`` with axes permuted so
    that ``qubits[i]`` is the system qubit the i-th factor of ``op`` acts on.
    Differentiable with respect to ``op``.
    """
    qubits = list(qubits)
    k = len(qubits)
    if len(set(qubits)) != k or not all(0 <= q < n_qubits for q in qubits):
        raise ValueError(f"invalid qubits {qubits} for a {n_qubits}-qubit system")
    if op.shape != (2**k, 2**k):
        raise ValueError(f"operator shape {tuple(op.shape)} does not match {k} qubits")
    rest = [q for q in range(n_qubits) if q not in qubits]
    full = torch.kron(op, eye(2 ** len(rest))) if rest else op
    order = qubits + rest  # axis i of `full` belongs to system qubit order[i]
    inverse = [order.index(q) for q in range(n_qubits)]
    perm = inverse + [n_qubits + i for i in inverse]
    full = full.reshape([2] * (2 * n_qubits)).permute(perm)
    return full.reshape(2**n_qubits, 2**n_qubits)
