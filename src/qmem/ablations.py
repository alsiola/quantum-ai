"""Phase 3b ablations of architecture A (issue #7).

Every variant is a ``QuantumMemoryModel`` subclass with the same interface as A
(``step_log_probs`` / ``log_prob`` / ``predictive_probs`` / ``sample`` /
``n_parameters`` / ``diagnostics``), so the trainer, splits and metrics are
shared. Architecture A itself is untouched: it is ``qmem.model.build_model``.

* **A** (``variant = "A"``): measurement-conditioned memory,
  ``rho -> Phi_y(rho) / p(y|rho)``. Built by ``build_model``.
* **B** (``"B"``, ``OutcomeAveragedModel``): a *control*, not a competitor.
  Same instrument as A, but the memory evolves under the unconditioned channel
  ``E = sum_y Phi_y`` whatever the token, so ``p(y_t | y_<t) = Tr(M_y E^{t-1}(rho_0))``
  ignores the history by construction. At A's parameters it is A's marginal
  over all histories.
* **C_k** (``"C"``, ``ResetMemoryModel``): A, with the memory reset to
  ``rho_0`` after every ``reset_every``-th token. ``reset_every = 1`` is
  memoryless (C); larger ``k`` keeps memory only within blocks of ``k`` tokens.
* **D** (``"D"``, ``SeparateUpdateReadoutModel``): separate readout and update.
  ``p(y | rho) = Tr(M_y rho)`` with the effects ``M_y`` of a readout instrument
  whose post-measurement state is *discarded*, then the memory is updated by a
  separately parameterised unitary controlled by the observed token,
  ``rho -> V_y rho V_y^dagger``. See ``docs/phase3b-ablations.md`` for why this
  is not a single-copy sequential process.
* **Hybrid** (``"hybrid"``, ``ClassicalFeedbackModel``): A plus a classical
  controller that rotates each output ancilla's measurement basis by angles
  linear in the last ``feedback_lags`` outcomes (after Stefanski et al.). The
  history enters through a classical side memory, so this is a hybrid baseline
  and never evidence for backaction.

All variants run the same per-step loop (``HookedMemoryModel``); each one
supplies three hooks: the per-call context (instrument and other operators
built once from the parameters), the readout ``p(.|state)`` and the state
update on a token. Teacher forcing (``step_log_probs``) and generation
(``sample``) stay separate methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from .circuits import (
    Block,
    ControlledRotationBlock,
    FixedUnitaryBlock,
    HardwareEfficientBlock,
    OutcomeConditionedBlock,
)
from .gates import ry, rz
from .instrument import kraus_from_unitary
from .linalg import REAL_DTYPE, eye, trace
from .model import Memory, ModelConfig, QuantumMemoryModel, build_model

VARIANTS = ("A", "B", "C", "D", "hybrid")


@dataclass
class State:
    """Per-step state of a variant: the quantum memory and any classical side data.

    ``rho`` is the persistent quantum memory ``(B, d, d)``. ``history`` is used
    only by the hybrid control (its classical outcome buffer); every other
    variant leaves it ``None``.
    """

    rho: torch.Tensor
    history: torch.Tensor | None = None


class HookedMemoryModel(QuantumMemoryModel):
    """``QuantumMemoryModel`` with the per-step loop written in terms of hooks.

    With the default hooks it computes exactly what A computes (tested).
    Subclasses override ``_context``, ``_begin``, ``_probs`` and ``_advance``.
    """

    # -- hooks -----------------------------------------------------------------
    def _context(self) -> Any:
        """Operators built once per call from the current parameters."""
        return self.instrument()

    def _begin(self, ctx: Any, batch_size: int) -> State:
        return State(self.rho0.expand(batch_size, -1, -1))

    def _probs(self, ctx: Any, state: State) -> torch.Tensor:
        """``p(. | state)``, shape ``(B, Y)``."""
        return ctx.probabilities(state.rho)

    def _advance(
        self, ctx: Any, state: State, y: torch.Tensor, t: int
    ) -> tuple[State, torch.Tensor]:
        """Update ``state`` on tokens ``y`` at step ``t`` (0-based); return it and ``p(y)``.

        Default: A's conditioned update. If ``p(y) == 0`` the row keeps its old
        memory so later steps stay finite (its log-probability is ``-inf``).
        """
        rho, p = _condition(ctx.kraus, state.rho, y)
        return State(rho), p

    # -- the shared loop -------------------------------------------------------
    def step_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = self._check_sequences(sequences)
        ctx = self._context()
        state = self._begin(ctx, seq.shape[0])
        out = []
        for t in range(seq.shape[1]):
            state, p = self._advance(ctx, state, seq[:, t], t)
            out.append(torch.log(self._guard(p)))
        if not out:
            return torch.zeros(seq.shape[0], 0, dtype=REAL_DTYPE)
        return torch.stack(out, dim=1)

    def predictive_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = self._check_sequences(sequences)
        ctx = self._context()
        state = self._begin(ctx, seq.shape[0])
        out = [self._probs(ctx, state)]
        for t in range(seq.shape[1]):
            state, _ = self._advance(ctx, state, seq[:, t], t)
            out.append(self._probs(ctx, state))
        return torch.stack(out, dim=1)

    def final_memory(self, sequences: torch.Tensor) -> Memory:
        seq = self._check_sequences(sequences)
        ctx = self._context()
        state = self._begin(ctx, seq.shape[0])
        for t in range(seq.shape[1]):
            state, _ = self._advance(ctx, state, seq[:, t], t)
        return Memory(state.rho)

    def memory_states(self, sequences: torch.Tensor) -> torch.Tensor:
        """Memory after each prefix ``y_<=t``, ``t = 0..T``: shape ``(B, T+1, d, d)``."""
        seq = self._check_sequences(sequences)
        ctx = self._context()
        state = self._begin(ctx, seq.shape[0])
        out = [state.rho]
        for t in range(seq.shape[1]):
            state, _ = self._advance(ctx, state, seq[:, t], t)
            out.append(state.rho)
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def sample(self, n: int, length: int, seed: int) -> torch.Tensor:
        """Generation: draw each token from the model's own ``p(.|state)``, then update on it."""
        gen = torch.Generator().manual_seed(seed)
        ctx = self._context()
        state = self._begin(ctx, n)
        out = torch.empty(n, length, dtype=torch.long)
        for t in range(length):
            p = self._probs(ctx, state).clamp(min=0)  # rounding can give -1e-17
            y = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            state, _ = self._advance(ctx, state, y, t)
            out[:, t] = y
        return out

    def validity_defect(self) -> float:
        """Largest deviation from a valid per-step operation (0 for a valid model)."""
        return float(self.instrument().completeness_defect().detach())


def _condition(
    kraus: torch.Tensor, rho: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outcome-conditioned update with the ``p(y) == 0`` safeguard.

    ``kraus`` is ``(Y, E, d, d)`` (shared) or ``(B, Y, E, d, d)`` (per row).
    Returns the normalised memory and the true ``p(y)``.
    """
    if kraus.dim() == 4:
        k = kraus[y]
    else:
        k = kraus[torch.arange(kraus.shape[0]), y]
    unnorm = torch.einsum("beij,bjk,belk->bil", k, rho, k.conj())
    p = trace(unnorm).real
    possible = p > 0
    denom = torch.where(possible, p, torch.ones_like(p))[:, None, None]
    return torch.where(possible[:, None, None], unnorm / denom, rho), p


class ConditionedModel(HookedMemoryModel):
    """A, written with the shared hooks (used to test the loop against ``build_model``'s A)."""


# -- B: outcome-averaged control ---------------------------------------------------
class OutcomeAveragedModel(HookedMemoryModel):
    """Control B: memory evolves under ``E(rho) = sum_y Phi_y(rho)``; the token is ignored.

    ``p(y_t | y_<t) = Tr(M_y E^{t-1}(rho_0))``: a context-independent (position-
    dependent) distribution, so B can do no better than the best i.i.d. model
    on a stationary source. It is a control for whether conditioning matters.
    """

    def _advance(self, ctx, state, y, t):
        p = ctx.probabilities(state.rho).gather(-1, y[:, None]).squeeze(-1)
        return State(ctx.channel(state.rho)), p


# -- C / C_k: reset memory ---------------------------------------------------------
class ResetMemoryModel(HookedMemoryModel):
    """C_k: A's conditioned update, then reset the memory to ``rho_0`` after every k-th token.

    ``reset_every = 1`` is the memoryless C; the prediction for token ``t`` then
    always comes from ``rho_0``. For ``k > 1`` the model has memory only within
    consecutive blocks of ``k`` tokens.
    """

    def __init__(self, *args: Any, reset_every: int = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if reset_every < 1:
            raise ValueError("reset_every must be >= 1")
        self.reset_every = reset_every

    def _advance(self, ctx, state, y, t):
        rho, p = _condition(ctx.kraus, state.rho, y)
        if (t + 1) % self.reset_every == 0:
            rho = self.rho0.expand_as(rho)
        return State(rho), p


# -- D: separate update and readout ------------------------------------------------
class SeparateUpdateReadoutModel(HookedMemoryModel):
    """D: readout POVM with discarded post-measurement state, then a token-controlled update.

    ``p(y | rho) = Tr(M_y rho)``, with ``M_y`` the effects of the readout
    instrument built from ``blocks`` (as in A). The memory is then updated by a
    separately parameterised map controlled by the observed token. The readout's
    own backaction is never applied, so prediction and memory update are
    separate operations.

    * ``update_env = 0``: ``rho -> V_y rho V_y^dagger`` with ``V_y`` the unitary of
      ``update_blocks[y]`` on the memory qubits (reversible: it cannot forget).
    * ``update_env = m > 0``: ``update_blocks[y]`` acts on the memory plus ``m``
      fresh ancillas in ``|0>`` that are then traced out. This gives a CPTP update
      ``rho -> sum_e L_{y,e} rho L_{y,e}^dagger`` that can dissipate (and so erase).
    """

    def __init__(self, *args: Any, update_blocks: list[Block], update_env: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if len(update_blocks) != self.n_symbols:
            raise ValueError("need one update block per token")
        for b in update_blocks:
            if b.n_qubits != self.n_memory + update_env:
                raise ValueError("update blocks act on the memory plus update_env ancillas")
        self.update_env = update_env
        self.update_blocks = nn.ModuleList(update_blocks)

    def update_unitaries(self) -> torch.Tensor:
        """Unitary of each ``update_blocks[y]``, ``(Y, D, D)``; ``D`` includes the ancillas."""
        return torch.stack([b.unitary() for b in self.update_blocks])

    def update_kraus(self) -> torch.Tensor:
        """Kraus operators of the token-controlled update channels, ``(Y, 2^update_env, d, d)``."""
        u = self.update_unitaries()
        return torch.stack(
            [kraus_from_unitary(u[y], self.n_memory, 0, self.update_env)[0] for y in range(len(u))]
        )

    def effects(self) -> torch.Tensor:
        return self.instrument().effects()

    def _context(self):
        return self.effects(), self.update_kraus()

    def _probs(self, ctx, state):
        effects, _ = ctx
        return torch.einsum("yij,...ji->...y", effects, state.rho).real

    def _advance(self, ctx, state, y, t):
        _, upd = ctx
        p = self._probs(ctx, state).gather(-1, y[:, None]).squeeze(-1)
        k = upd[y]
        return State(torch.einsum("beij,bjk,belk->bil", k, state.rho, k.conj())), p

    def validity_defect(self) -> float:
        effects, k = (x.detach() for x in self._context())
        d = self.memory_dim
        povm = (effects.sum(0) - eye(d)).abs().max()
        tp = (torch.einsum("yeji,yejk->yik", k.conj(), k) - eye(d)).abs().max()
        return float(max(povm, tp))


# -- Hybrid: classical feedback on the measurement basis -----------------------------
class ClassicalFeedbackModel(HookedMemoryModel):
    """A plus a classical controller that sets each output ancilla's measurement basis.

    Before the output ancillas are measured, ancilla ``q`` gets ``RY(a_q) RZ(b_q)``
    with ``(a, b) = W h + c``, where ``h`` holds the bits of the last
    ``feedback_lags`` tokens as ``+-1`` (most recent first; ``0`` before the
    sequence starts). So ``K_y(h) = sum_y' R(h)[y, y'] K_y'`` with ``K`` A's
    Kraus operators. ``W`` and ``c`` start at zero, so at initialisation the
    model *is* A. The buffer ``h`` is a classical hidden state.
    """

    def __init__(self, *args: Any, feedback_lags: int = 2, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.feedback_lags = feedback_lags
        n_in = feedback_lags * self.n_output
        self.feedback_weight = nn.Parameter(torch.zeros(self.n_output, 2, n_in, dtype=REAL_DTYPE))
        self.feedback_bias = nn.Parameter(torch.zeros(self.n_output, 2, dtype=REAL_DTYPE))
        bits = (
            torch.arange(self.n_symbols)[:, None] >> torch.arange(self.n_output - 1, -1, -1)
        ) & 1
        self.register_buffer("token_bits", (2 * bits - 1).to(REAL_DTYPE), persistent=False)

    def _begin(self, ctx, batch_size):
        h = torch.zeros(batch_size, self.feedback_lags * self.n_output, dtype=REAL_DTYPE)
        return State(self.rho0.expand(batch_size, -1, -1), h)

    def basis_rotation(self, history: torch.Tensor) -> torch.Tensor:
        """``R(h)`` on the output register, shape ``(B, Y, Y)``."""
        ang = torch.einsum("qai,bi->bqa", self.feedback_weight, history) + self.feedback_bias
        per_qubit = ry(ang[..., 0]) @ rz(ang[..., 1])  # (B, n_output, 2, 2)
        r = per_qubit[:, 0]
        for q in range(1, self.n_output):
            r = torch.einsum("bij,bkl->bikjl", r, per_qubit[:, q]).reshape(
                r.shape[0], 2 * r.shape[1], 2 * r.shape[2]
            )
        return r

    def context_kraus(self, inst, history: torch.Tensor) -> torch.Tensor:
        """Per-row Kraus operators ``(B, Y, E, d, d)`` for the controller's current input."""
        return torch.einsum("bxz,zeij->bxeij", self.basis_rotation(history), inst.kraus)

    def _probs(self, ctx, state):
        k = self.context_kraus(ctx, state.history)
        effects = torch.einsum("byeji,byejk->byik", k.conj(), k)
        return torch.einsum("byij,bji->by", effects, state.rho).real

    def _advance(self, ctx, state, y, t):
        rho, p = _condition(self.context_kraus(ctx, state.history), state.rho, y)
        h = torch.cat([self.token_bits[y], state.history], dim=-1)[:, : state.history.shape[-1]]
        return State(rho, h), p


# -- configuration and construction --------------------------------------------------
@dataclass
class VariantConfig:
    """The variant and its own settings (keys next to ``ModelConfig``'s in ``[model]``)."""

    variant: str = "A"
    reset_every: int = 1  # C
    update_depth: int = 1  # D: hardware-efficient layers per token unitary V_y
    update_env: int = 0  # D: ancillas traced out after each update (0: unitary update)
    feedback_lags: int = 2  # hybrid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_config(d: dict[str, Any]) -> tuple[VariantConfig, ModelConfig]:
    """Split a ``[model]`` dict into the variant settings and A's ``ModelConfig``."""
    vkeys = {f.name for f in fields(VariantConfig)}
    v = VariantConfig(**{k: x for k, x in d.items() if k in vkeys})
    if v.variant not in VARIANTS:
        raise ValueError(f"unknown variant {v.variant!r}; choose from {VARIANTS}")
    return v, ModelConfig.from_dict({k: x for k, x in d.items() if k not in vkeys})


def update_seed(seed: int) -> int:
    """Seed for D's update unitaries (the readout uses ``seed`` itself, like A)."""
    return seed + 1_000_003


def build_variant(d: dict[str, Any]) -> QuantumMemoryModel:
    """Build any variant from a ``[model]`` dict (``variant`` defaults to A).

    The circuit (and for D the readout) is ``build_model``'s default layout
    with the same seed, so A, B, C and the hybrid start from identical
    parameters for a given seed.
    """
    v, cfg = split_config(d)
    base = build_model(cfg)
    if v.variant == "A":
        return base
    args = (cfg.n_memory, cfg.n_output, list(base.circuit.blocks))
    kw = {"n_env": cfg.n_env, "initial": base.rho0, "prob_floor": cfg.prob_floor}
    if v.variant == "B":
        return OutcomeAveragedModel(*args, **kw)
    if v.variant == "C":
        return ResetMemoryModel(*args, reset_every=v.reset_every, **kw)
    if v.variant == "hybrid":
        return ClassicalFeedbackModel(*args, feedback_lags=v.feedback_lags, **kw)
    gen = torch.Generator().manual_seed(update_seed(cfg.seed))
    n_upd = cfg.n_memory + v.update_env
    update_blocks: list[Block] = [
        HardwareEfficientBlock(
            list(range(n_upd)),
            n_upd,
            v.update_depth,
            rotations=cfg.rotations,
            entangler=cfg.entangler,
            pattern=cfg.pattern,
            init_scale=cfg.init_scale,
            generator=gen,
        )
        for _ in range(2**cfg.n_output)
    ]
    return SeparateUpdateReadoutModel(
        *args, update_blocks=update_blocks, update_env=v.update_env, **kw
    )


def validity_defect(model: QuantumMemoryModel) -> float:
    """Completeness (and, for D, unitarity) defect of any variant."""
    if isinstance(model, HookedMemoryModel):
        return model.validity_defect()
    return float(model.instrument().completeness_defect().detach())


# -- resource counts ----------------------------------------------------------------
def block_gates(block: Block) -> list[tuple[str, tuple[int, ...]]]:
    """The gate sequence a block applies, as ``(name, qubits)`` (in time order)."""
    if isinstance(block, HardwareEfficientBlock):
        gates: list[tuple[str, tuple[int, ...]]] = []
        for _ in range(block.depth):
            for q in block.qubits:
                gates.extend((r, (q,)) for r in block.rotations)
            gates.extend((block.entangler, pair) for pair in block.pairs)
        return gates
    if isinstance(block, ControlledRotationBlock):
        return [("cry", tuple(p)) for p in block.pairs]
    if isinstance(block, OutcomeConditionedBlock):
        # Feed-forward: after the readout, the classically controlled W_y (one per token)
        # acts on the memory. Count one W_y's gates, mapped onto the memory qubits.
        mem = block.memory
        return [(f"ff_{n}", tuple(mem[q] for q in qs)) for n, qs in block_gates(block.ws[0])]
    if isinstance(block, FixedUnitaryBlock):
        return [("fixed", block.qubits)]
    raise TypeError(f"no gate list for {type(block).__name__}")


def gate_resources(gates: list[tuple[str, tuple[int, ...]]], qubits: list[int]) -> dict[str, int]:
    """Depth (ASAP schedule, every gate one time step) and gate counts."""
    level = dict.fromkeys(qubits, 0)
    for _, qs in gates:
        t = max(level.get(q, 0) for q in qs) + 1
        for q in qs:
            level[q] = t
    return {
        "depth": max(level.values(), default=0),
        "one_qubit_gates": sum(len(qs) == 1 for _, qs in gates),
        "two_qubit_gates": sum(len(qs) == 2 for _, qs in gates),
    }


def resource_counts(model: QuantumMemoryModel) -> dict[str, Any]:
    """Per-token resources of one step of ``model`` (see ``docs/phase3b-ablations.md``).

    ``cry`` (controlled-RY) counts as one two-qubit gate; it compiles to two
    CNOTs and two RY. Ancilla resets are the output and env ancillas returned
    to ``|0>`` each step.
    """
    n_anc = model.n_output + model.n_env
    gates = [g for b in model.circuit.blocks for g in block_gates(b)]
    out: dict[str, Any] = {
        "qubits": model.n_qubits,
        "memory_qubits": model.n_memory,
        "mid_circuit_measurements_per_token": model.n_output,
        "ancilla_resets_per_token": n_anc,
        "memory_resets_per_token": 0.0,
        "classically_controlled_gates_per_token": 0,
        "n_parameters": model.n_parameters(),
    }
    n_ff = sum(name.startswith("ff_") for name, _ in gates)
    out["classically_controlled_gates_per_token"] = n_ff
    if isinstance(model, ClassicalFeedbackModel):
        anc = list(range(model.n_memory, model.n_memory + model.n_output))
        gates = gates + [(r, (q,)) for q in anc for r in ("rz", "ry")]
        out["classically_controlled_gates_per_token"] += 2 * model.n_output
        out["classical_buffer_bits"] = model.feedback_lags * model.n_output
    out.update(gate_resources(gates, list(range(model.n_qubits))))
    if isinstance(model, ResetMemoryModel):
        out["memory_resets_per_token"] = model.n_memory / model.reset_every
    if isinstance(model, SeparateUpdateReadoutModel):
        upd = block_gates(model.update_blocks[0])
        r = gate_resources(upd, list(range(model.n_memory + model.update_env)))
        out["qubits"] = max(model.n_qubits, model.n_memory + model.update_env)
        out["ancilla_resets_per_token"] = n_anc + model.update_env
        out["update_unitary"] = r
        out["classically_controlled_gates_per_token"] = len(upd)
        out["readout"] = gate_resources(gates, list(range(model.n_qubits)))
        out["depth"] = out["readout"]["depth"] + r["depth"]
        out["one_qubit_gates"] += r["one_qubit_gates"]
        out["two_qubit_gates"] += r["two_qubit_gates"]
    return out
