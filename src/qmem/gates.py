"""Gate matrices. Rotations are differentiable functions of real angle tensors.

Rotation conventions (standard): ``R_P(theta) = exp(-i theta P / 2)`` for
``P in {X, Y, Z}``. A rotation called with an angle tensor of shape ``(...)``
returns matrices of shape ``(..., 2, 2)``.
"""

from __future__ import annotations

import torch

from .linalg import DTYPE


def _as_angle(theta: torch.Tensor | float) -> torch.Tensor:
    return torch.as_tensor(theta, dtype=torch.float64)


def rx(theta: torch.Tensor | float) -> torch.Tensor:
    t = _as_angle(theta) / 2
    c, s = torch.cos(t).to(DTYPE), torch.sin(t).to(DTYPE)
    row0 = torch.stack([c, -1j * s], dim=-1)
    row1 = torch.stack([-1j * s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def ry(theta: torch.Tensor | float) -> torch.Tensor:
    t = _as_angle(theta) / 2
    c, s = torch.cos(t).to(DTYPE), torch.sin(t).to(DTYPE)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def rz(theta: torch.Tensor | float) -> torch.Tensor:
    t = _as_angle(theta) / 2
    phase = torch.exp(-1j * t.to(DTYPE))
    zero = torch.zeros_like(phase)
    row0 = torch.stack([phase, zero], dim=-1)
    row1 = torch.stack([zero, phase.conj()], dim=-1)
    return torch.stack([row0, row1], dim=-2)


ROTATIONS = {"rx": rx, "ry": ry, "rz": rz}

I2 = torch.eye(2, dtype=DTYPE)
X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
Z = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE)
H = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / 2**0.5
P0 = torch.tensor([[1, 0], [0, 0]], dtype=DTYPE)
P1 = torch.tensor([[0, 0], [0, 1]], dtype=DTYPE)


def controlled(op: torch.Tensor) -> torch.Tensor:
    """Two-qubit ``|0><0| ⊗ I + |1><1| ⊗ op`` (control is the first qubit)."""
    return torch.kron(P0, I2) + torch.kron(P1, op)


CNOT = controlled(X)
CZ = controlled(Z)
ENTANGLERS = {"cnot": CNOT, "cz": CZ}
