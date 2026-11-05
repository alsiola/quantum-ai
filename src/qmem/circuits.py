"""Parameterised circuit blocks acting on the joint memory + ancilla system.

A circuit is an ordered list of blocks; each block is an ``nn.Module`` whose
``unitary()`` returns the full ``(2^n, 2^n)`` matrix it applies to the
``n``-qubit system. The circuit unitary is ``U = B_last ... B_2 B_1`` (the
first block in the list acts first).

Full matrices are fine here: the target systems have at most ~6 qubits
(``d <= 64``), so building and multiplying dense unitaries is cheap and keeps
every step easy to inspect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .gates import ENTANGLERS, ROTATIONS, controlled, ry
from .linalg import REAL_DTYPE, embed, eye, kron

PATTERNS = ("linear", "ring", "all", "none")


def entangling_pairs(qubits: Sequence[int], pattern: str) -> list[tuple[int, int]]:
    """Ordered (control, target) pairs for an entangling layer on ``qubits``."""
    q = list(qubits)
    if pattern == "none" or len(q) < 2:
        return []
    if pattern == "linear":
        return [(q[i], q[i + 1]) for i in range(len(q) - 1)]
    if pattern == "ring":
        pairs = [(q[i], q[i + 1]) for i in range(len(q) - 1)]
        if len(q) > 2:
            pairs.append((q[-1], q[0]))
        return pairs
    if pattern == "all":
        return [(q[i], q[j]) for i in range(len(q)) for j in range(i + 1, len(q))]
    raise ValueError(f"unknown entangling pattern {pattern!r}; choose from {PATTERNS}")


class Block(nn.Module):
    """Base class: a unitary on an ``n_qubits`` system acting on ``qubits``."""

    def __init__(self, qubits: Sequence[int], n_qubits: int):
        super().__init__()
        self.qubits = tuple(qubits)
        self.n_qubits = n_qubits
        if not all(0 <= q < n_qubits for q in self.qubits):
            raise ValueError(f"qubits {self.qubits} out of range for {n_qubits} qubits")

    def unitary(self) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class HardwareEfficientBlock(Block):
    """``depth`` layers of single-qubit rotations followed by fixed entanglers.

    Layer ``l``: on every qubit ``q`` in ``qubits`` apply the rotations named in
    ``rotations`` in order (``theta[l, i, r]`` is the angle of rotation ``r`` on
    the ``i``-th qubit of the block), then the entangler on each pair given by
    ``pattern``, in order.

    Parameter layout: ``theta`` has shape ``(depth, len(qubits), len(rotations))``.
    Initialisation: i.i.d. uniform on ``[-pi * init_scale, pi * init_scale]``
    drawn from ``generator`` (``init_scale=0`` gives the identity circuit up to
    entanglers).
    """

    def __init__(
        self,
        qubits: Sequence[int],
        n_qubits: int,
        depth: int,
        rotations: Sequence[str] = ("ry", "rz"),
        entangler: str = "cnot",
        pattern: str = "linear",
        init_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__(qubits, n_qubits)
        for r in rotations:
            if r not in ROTATIONS:
                raise ValueError(f"unknown rotation {r!r}; choose from {sorted(ROTATIONS)}")
        if entangler not in ENTANGLERS:
            raise ValueError(f"unknown entangler {entangler!r}; choose from {sorted(ENTANGLERS)}")
        self.depth = depth
        self.rotations = tuple(rotations)
        self.entangler = entangler
        self.pattern = pattern
        self.pairs = entangling_pairs(self.qubits, pattern)
        shape = (depth, len(self.qubits), len(self.rotations))
        u = torch.rand(shape, generator=generator, dtype=REAL_DTYPE)
        self.theta = nn.Parameter((2 * u - 1) * math.pi * init_scale)
        ent = eye(2**n_qubits)
        for c, t in self.pairs:
            ent = embed(ENTANGLERS[entangler], (c, t), n_qubits) @ ent
        self.register_buffer("entangling_layer", ent, persistent=False)

    def unitary(self) -> torch.Tensor:
        u = eye(2**self.n_qubits)
        for layer in range(self.depth):
            per_qubit = []
            for i in range(len(self.qubits)):
                op = torch.eye(2, dtype=u.dtype)
                for r, name in enumerate(self.rotations):
                    op = ROTATIONS[name](self.theta[layer, i, r]) @ op
                per_qubit.append(op)
            rot = embed(kron(*per_qubit), self.qubits, self.n_qubits)
            u = self.entangling_layer @ rot @ u
        return u


class ControlledRotationBlock(Block):
    """Measurement-strength coupling: controlled-``RY(2 g)`` from memory to ancilla.

    For each ``(control, target)`` pair, with the target ancilla starting in
    ``|0>``: if the control is ``|0>`` the ancilla stays ``|0>``; if it is
    ``|1>`` the ancilla becomes ``cos g |0> + sin g |1>``. Measuring the
    ancilla then implements the Kraus pair

        K_0 = |0><0| + cos g |1><1|,   K_1 = sin g |1><1|

    on the control qubit. ``g = 0`` gives no information and no backaction;
    ``g = pi/2`` is a projective (CNOT) measurement of the control in the Z basis.

    ``strength`` holds one ``g`` per pair. If ``trainable`` it is an
    ``nn.Parameter``, otherwise a buffer (so sweeps can set it exactly).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int]],
        n_qubits: int,
        strength: float | Sequence[float] = math.pi / 2,
        trainable: bool = False,
    ):
        pairs = [tuple(p) for p in pairs]
        super().__init__(sorted({q for p in pairs for q in p}), n_qubits)
        self.pairs = pairs
        g = torch.as_tensor(strength, dtype=REAL_DTYPE).expand(len(pairs)).clone()
        if trainable:
            self.strength = nn.Parameter(g)
        else:
            self.register_buffer("strength", g)

    def set_strength(self, strength: float | Sequence[float]) -> None:
        with torch.no_grad():
            self.strength.copy_(torch.as_tensor(strength, dtype=REAL_DTYPE))

    def unitary(self) -> torch.Tensor:
        u = eye(2**self.n_qubits)
        for k, (c, t) in enumerate(self.pairs):
            u = embed(controlled(ry(2 * self.strength[k])), (c, t), self.n_qubits) @ u
        return u


class FixedUnitaryBlock(Block):
    """A constant unitary on ``qubits`` (for tests and hand-built instruments)."""

    def __init__(self, op: torch.Tensor, qubits: Sequence[int], n_qubits: int):
        super().__init__(qubits, n_qubits)
        self.register_buffer("full", embed(op, qubits, n_qubits), persistent=False)

    def unitary(self) -> torch.Tensor:
        return self.full


class Circuit(nn.Module):
    """Ordered list of blocks; ``unitary() = B_last ... B_1``."""

    def __init__(self, blocks: Sequence[Block], n_qubits: int):
        super().__init__()
        for b in blocks:
            if b.n_qubits != n_qubits:
                raise ValueError("all blocks must act on the same system size")
        self.blocks = nn.ModuleList(blocks)
        self.n_qubits = n_qubits

    def unitary(self) -> torch.Tensor:
        u = eye(2**self.n_qubits)
        for b in self.blocks:
            u = b.unitary() @ u
        return u
