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

    ``strength`` holds one ``g`` per pair, or a single ``g`` used by every pair
    if ``shared`` (the global measurement strength). If ``trainable`` it is an
    ``nn.Parameter``, otherwise a buffer (so sweeps can set it exactly).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int]],
        n_qubits: int,
        strength: float | Sequence[float] = math.pi / 2,
        trainable: bool = False,
        shared: bool = False,
    ):
        pairs = [tuple(p) for p in pairs]
        super().__init__(sorted({q for p in pairs for q in p}), n_qubits)
        self.pairs = pairs
        self.shared = shared
        n_g = 1 if shared else len(pairs)
        g = torch.as_tensor(strength, dtype=REAL_DTYPE).expand(n_g).clone()
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
            g = self.strength[0 if self.shared else k]
            u = embed(controlled(ry(2 * g)), (c, t), self.n_qubits) @ u
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


class OutcomeConditionedBlock(Block):
    """Feed-forward: a trainable unitary ``W_y`` on the memory, chosen by the readout ``y``.

    The block is ``sum_y W_y ⊗ |y><y|_A ⊗ I_E``, a set of memory gates
    multi-controlled by the output ancillas. It is diagonal in the ancillas'
    computational basis, so it commutes with their measurement: applying it
    just before the readout is the same as measuring and then applying the
    classically controlled ``W_y`` (feed-forward). Placed after the coupling
    ``C(g)`` it turns the Kraus operators into ``K_{y,e} -> W_y K_{y,e}``, which
    leaves every effect ``E_y = sum_e K^dagger K`` unchanged
    (``W_y^dagger W_y = I``); see ``docs/architecture.md`` section 2.

    Each ``W_y`` is a ``HardwareEfficientBlock`` of ``depth`` layers on the
    memory qubits (their own parameters ``ws[y].theta``). ``memory`` and
    ``outputs`` must be contiguous and ordered ``memory, outputs`` (the default
    layout).
    """

    def __init__(
        self,
        memory: Sequence[int],
        outputs: Sequence[int],
        n_qubits: int,
        depth: int,
        rotations: Sequence[str] = ("ry", "rz"),
        entangler: str = "cnot",
        pattern: str = "linear",
        init_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        memory, outputs = list(memory), list(outputs)
        super().__init__(memory + outputs, n_qubits)
        if memory + outputs != list(range(memory[0], memory[0] + len(memory) + len(outputs))):
            raise ValueError("memory and output qubits must be contiguous, memory first")
        self.memory, self.outputs = tuple(memory), tuple(outputs)
        n_m, n_y = len(memory), len(outputs)
        self.ws = nn.ModuleList(
            HardwareEfficientBlock(
                list(range(n_m)), n_m, depth, rotations, entangler, pattern, init_scale, generator
            )
            for _ in range(2**n_y)
        )
        projectors = torch.zeros(2**n_y, 2**n_y, 2**n_y, dtype=eye(1).dtype)
        for y in range(2**n_y):
            projectors[y, y, y] = 1.0
        self.register_buffer("projectors", projectors, persistent=False)

    def outcome_unitaries(self) -> torch.Tensor:
        """``W_y`` for every outcome, shape ``(n_outcomes, d_memory, d_memory)``."""
        return torch.stack([w.unitary() for w in self.ws])

    def unitary(self) -> torch.Tensor:
        ws = self.outcome_unitaries()
        op = sum(torch.kron(ws[y], self.projectors[y]) for y in range(len(self.ws)))
        return embed(op, self.qubits, self.n_qubits)
