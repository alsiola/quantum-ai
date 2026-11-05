"""Measurement-conditioned quantum memory as an autoregressive sequence model.

Per step: the memory ``rho`` meets fresh ancillas in ``|0>``, the circuit
``U(theta)`` acts on memory + ancillas, the output ancillas are measured
(outcome = token ``y``), env ancillas are discarded, and the memory keeps the
outcome-conditioned state ``K_y rho K_y^dagger / p(y|rho)``.

Two code paths, deliberately separate:

* ``step_log_probs`` / ``log_prob`` / ``predictive_probs``: *teacher forcing*.
  The token is given (from data) and we condition the memory on it. This is
  classical conditioning of the model's distribution on an observed sequence;
  it is not a claim that a physical device would produce that token.
* ``sample``: *generation*. Tokens are drawn from the model's own outcome
  distribution and the memory is updated on the drawn outcome, which is what
  a physical run of the instrument does.

Probability safeguards (no silent clamping):

* By default nothing is clamped. If ``p(y) == 0`` the step log-probability is
  ``-inf`` (exactly the model's answer) and the memory row is left as it was
  so later steps stay finite; the sequence log-probability is still ``-inf``.
* Steps with ``0 < p(y) < tiny_prob`` are counted in ``diagnostics`` and logged
  once per call, because the normalisation loses relative precision there.
* ``prob_floor`` (off by default) replaces ``log p(y)`` with
  ``log max(p(y), prob_floor)``. This *changes the likelihood* (it is a bounded
  penalty, no longer a normalised model); every floored step is counted and
  logged. The memory update still uses the true ``p(y)``.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from .circuits import (
    Block,
    Circuit,
    ControlledRotationBlock,
    HardwareEfficientBlock,
    OutcomeConditionedBlock,
)
from .instrument import Instrument
from .linalg import trace
from .states import initial_state

logger = logging.getLogger("qmem")


@dataclass(frozen=True)
class Memory:
    """The persistent quantum memory carried between symbols: ``rho`` of shape ``(B, d, d)``."""

    rho: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.rho.shape[0]

    @property
    def dim(self) -> int:
        return self.rho.shape[-1]


class QuantumMemoryModel(nn.Module):
    """Autoregressive model whose memory update is the measurement backaction.

    Args:
        n_memory: memory qubits (persist across steps).
        n_output: output ancillas, read each step; alphabet size ``2**n_output``.
        n_env: extra ancillas traced out each step (``2**n_env`` Kraus per outcome).
        blocks: circuit blocks on the ``n_memory + n_output + n_env`` qubit
            system, ordered memory, output, env (see ``build_model`` for the
            default layout).
        initial: initial memory state spec (see ``qmem.states.initial_state``).
        prob_floor: optional floor on ``p(y)`` inside the log (alters the likelihood;
            logged).
        tiny_prob: threshold below which a step is counted and logged.
        generator: RNG for ``initial="random_pure"``.
    """

    def __init__(
        self,
        n_memory: int,
        n_output: int,
        blocks: list[Block],
        n_env: int = 0,
        initial: str | torch.Tensor = "zero",
        prob_floor: float | None = None,
        tiny_prob: float = 1e-10,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.n_memory, self.n_output, self.n_env = n_memory, n_output, n_env
        self.n_qubits = n_memory + n_output + n_env
        self.circuit = Circuit(blocks, self.n_qubits)
        self.register_buffer("rho0", initial_state(initial, n_memory, generator))
        self.prob_floor = prob_floor
        self.tiny_prob = tiny_prob
        self.diagnostics: Counter[str] = Counter()

    @property
    def n_symbols(self) -> int:
        return 2**self.n_output

    @property
    def memory_dim(self) -> int:
        return 2**self.n_memory

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- the instrument --------------------------------------------------------
    def unitary(self) -> torch.Tensor:
        return self.circuit.unitary()

    def instrument(self) -> Instrument:
        """The per-step instrument for the current parameters (differentiable)."""
        return Instrument.from_unitary(self.unitary(), self.n_memory, self.n_output, self.n_env)

    def initial_memory(self, batch_size: int) -> Memory:
        return Memory(self.rho0.expand(batch_size, -1, -1))

    # -- teacher forcing -------------------------------------------------------
    def observe(
        self, instrument: Instrument, memory: Memory, y: torch.Tensor
    ) -> tuple[Memory, torch.Tensor]:
        """Condition ``memory`` on observed tokens ``y`` (shape ``(B,)``).

        Returns the new memory and ``log p(y | memory)`` of shape ``(B,)``.
        """
        unnorm = instrument.branch(memory.rho, y)
        p = trace(unnorm).real
        possible = p > 0
        denom = torch.where(possible, p, torch.ones_like(p))[:, None, None]
        rho = torch.where(possible[:, None, None], unnorm / denom, memory.rho)
        return Memory(rho), torch.log(self._guard(p))

    def _guard(self, p: torch.Tensor) -> torch.Tensor:
        """Count/log tiny and zero probabilities; apply ``prob_floor`` if set.

        The returned value is used for the log-likelihood only. The memory is
        always normalised by the true ``p`` so it stays a density matrix.
        """
        n_zero = int((p <= 0).sum())
        n_tiny = int(((p > 0) & (p < self.tiny_prob)).sum())
        if n_zero:
            self.diagnostics["zero_prob_steps"] += n_zero
        if n_tiny:
            self.diagnostics["tiny_prob_steps"] += n_tiny
            logger.warning("qmem: %d step(s) with 0 < p(y) < %g", n_tiny, self.tiny_prob)
        if self.prob_floor is not None:
            n_floor = int((p < self.prob_floor).sum())
            if n_floor:
                self.diagnostics["floored_steps"] += n_floor
                logger.warning(
                    "qmem: prob_floor=%g applied to %d step(s); likelihood is altered",
                    self.prob_floor,
                    n_floor,
                )
            p = p.clamp(min=self.prob_floor)
        return p

    def _check_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = torch.as_tensor(sequences, dtype=torch.long)
        if seq.dim() == 1:
            seq = seq[None]
        if seq.dim() != 2:
            raise ValueError("sequences must have shape (B, T)")
        if seq.numel() and (seq.min() < 0 or seq.max() >= self.n_symbols):
            raise ValueError(f"tokens must lie in [0, {self.n_symbols})")
        return seq

    def step_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """Teacher-forced ``log p(y_t | y_<t)``, shape ``(B, T)``."""
        seq = self._check_sequences(sequences)
        inst = self.instrument()
        memory = self.initial_memory(seq.shape[0])
        out = []
        for t in range(seq.shape[1]):
            memory, logp = self.observe(inst, memory, seq[:, t])
            out.append(logp)
        if not out:
            return torch.zeros(seq.shape[0], 0, dtype=torch.float64)
        return torch.stack(out, dim=1)

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Teacher-forced sequence log-probability ``log p(y_1..y_T)``, shape ``(B,)``."""
        return self.step_log_probs(sequences).sum(-1)

    def predictive_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """Next-symbol distributions ``p(. | y_<t)`` for ``t = 1..T+1``: ``(B, T+1, Y)``."""
        seq = self._check_sequences(sequences)
        inst = self.instrument()
        memory = self.initial_memory(seq.shape[0])
        out = [inst.probabilities(memory.rho)]
        for t in range(seq.shape[1]):
            memory, _ = self.observe(inst, memory, seq[:, t])
            out.append(inst.probabilities(memory.rho))
        return torch.stack(out, dim=1)

    def final_memory(self, sequences: torch.Tensor) -> Memory:
        """Memory after teacher-forcing ``sequences``."""
        seq = self._check_sequences(sequences)
        inst = self.instrument()
        memory = self.initial_memory(seq.shape[0])
        for t in range(seq.shape[1]):
            memory, _ = self.observe(inst, memory, seq[:, t])
        return memory

    # -- unconditioned evolution (outcome discarded) ---------------------------
    def unconditioned_memory(self, n_steps: int) -> torch.Tensor:
        """``E^T(rho0)``: the memory averaged over all outcomes after ``n_steps``."""
        inst = self.instrument()
        rho = self.rho0
        for _ in range(n_steps):
            rho = inst.channel(rho)
        return rho

    # -- generation ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, length: int, seed: int) -> torch.Tensor:
        """Draw ``n`` sequences of ``length`` tokens from the model, shape ``(n, length)``.

        Each step computes ``p(.|rho)`` for every outcome, draws ``y`` with a
        ``torch.Generator`` seeded by ``seed``, and updates the memory on the
        drawn outcome. A drawn outcome always has ``p(y) > 0``.
        """
        gen = torch.Generator().manual_seed(seed)
        inst = self.instrument()
        rho = self.rho0.expand(n, -1, -1)
        out = torch.empty(n, length, dtype=torch.long)
        for t in range(length):
            p = inst.probabilities(rho).clamp(min=0)  # rounding can give -1e-17
            y = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            unnorm = inst.branch(rho, y)
            rho = unnorm / trace(unnorm).real[:, None, None]
            out[:, t] = y
        return out


@dataclass
class ModelConfig:
    """Default layout: memory HEA -> controlled-RY(2g) couplings -> [feed-forward] -> [joint HEA].

    * ``memory_depth`` hardware-efficient layers on the memory qubits only.
    * Coupling: output ancilla ``k`` is controlled by memory qubit
      ``k % n_memory`` with strength ``strength``; env ancilla ``j`` by memory
      qubit ``(n_output + j) % n_memory`` with ``env_strength``.
    * ``joint_depth`` hardware-efficient layers on every qubit, applied after
      the coupling. With ``joint_depth > 0`` the readout is a general circuit
      and ``strength`` no longer bounds how much is learnt per step.
    * ``feedforward_depth`` (default 0, off): an ``OutcomeConditionedBlock``
      right after the coupling, i.e. ``K_y = W_y D_y(g) V`` with a trainable
      memory unitary ``W_y`` (``feedforward_depth`` layers) per outcome. It
      leaves the effects, and so the information ``g`` allows, unchanged. The
      strength-faithful layout is ``feedforward_depth > 0, joint_depth = 0``.
    * ``strength_shared``: one ``g`` for every output coupling (the global
      measurement strength) instead of one per coupling.
    """

    n_memory: int = 2
    n_output: int = 1
    n_env: int = 0
    memory_depth: int = 2
    joint_depth: int = 0
    strength: float = math.pi / 4
    strength_trainable: bool = False
    strength_shared: bool = False
    feedforward_depth: int = 0
    env_strength: float = 0.0
    rotations: tuple[str, ...] = ("ry", "rz")
    entangler: str = "cnot"
    pattern: str = "linear"
    initial: str = "zero"
    init_scale: float = 1.0
    seed: int = 0
    prob_floor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rotations"] = list(self.rotations)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        d = dict(d)
        if "rotations" in d:
            d["rotations"] = tuple(d["rotations"])
        return cls(**d)


def build_model(config: ModelConfig) -> QuantumMemoryModel:
    """Build the default-layout model; parameters are drawn from ``config.seed``."""
    c = config
    gen = torch.Generator().manual_seed(c.seed)
    n = c.n_memory + c.n_output + c.n_env
    memory = list(range(c.n_memory))
    outputs = list(range(c.n_memory, c.n_memory + c.n_output))
    envs = list(range(c.n_memory + c.n_output, n))
    hea = {
        "rotations": c.rotations,
        "entangler": c.entangler,
        "pattern": c.pattern,
        "init_scale": c.init_scale,
        "generator": gen,
    }
    blocks: list[Block] = []
    if c.memory_depth > 0:
        blocks.append(HardwareEfficientBlock(memory, n, c.memory_depth, **hea))
    out_pairs = [(memory[k % c.n_memory], q) for k, q in enumerate(outputs)]
    blocks.append(
        ControlledRotationBlock(
            out_pairs, n, c.strength, c.strength_trainable, shared=c.strength_shared
        )
    )
    if envs:
        env_pairs = [(memory[(c.n_output + j) % c.n_memory], q) for j, q in enumerate(envs)]
        blocks.append(ControlledRotationBlock(env_pairs, n, c.env_strength, trainable=False))
    if c.feedforward_depth > 0:
        blocks.append(OutcomeConditionedBlock(memory, outputs, n, c.feedforward_depth, **hea))
    if c.joint_depth > 0:
        blocks.append(HardwareEfficientBlock(list(range(n)), n, c.joint_depth, **hea))
    return QuantumMemoryModel(
        c.n_memory,
        c.n_output,
        blocks,
        n_env=c.n_env,
        initial=c.initial,
        prob_floor=c.prob_floor,
        generator=gen,
    )
