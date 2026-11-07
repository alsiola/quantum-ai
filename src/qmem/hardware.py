"""Export trained models to Qiskit circuits, and simulate them exactly, with and without noise.

Needs the optional ``hardware`` dependency group (``uv sync --group hardware``); nothing else in
``qmem`` imports this module. Simulation only: nothing here talks to a quantum cloud service.

**The circuit** (``export_circuit``). One step of architecture A is the circuit ``U`` of
``model.circuit`` on memory, output ancillas and env ancillas (the project's order; Qiskit
qubit ``k`` is project qubit ``k``), followed by a mid-circuit measurement of the output
ancillas into that step's classical register ``y<t>``, and a reset of every ancilla before
the next step. Register value = token: output ancilla ``k`` (project bit ``n_output-1-k`` of
``y``, big-endian) is written to clbit ``n_output-1-k`` of ``y<t>`` (Qiskit registers are
little-endian), so the register's integer value is the token ``y``. Env ancillas are never
read; resetting them discards them, which is the partial trace.

Gates are those of the blocks, one to one: ``HardwareEfficientBlock`` gives its rotations
(``rx``/``ry``/``rz``, same convention ``exp(-i theta P/2)``) and CNOT/CZ layers,
``ControlledRotationBlock`` gives ``cry(2 g)``. The feed-forward ``OutcomeConditionedBlock``
``sum_y W_y ⊗ |y><y|`` can be exported two ways (``feedforward``):

* ``"dynamic"`` (default): measure first, then apply ``W_y`` classically conditioned on the
  register (``if_test``/``switch``). This is the hardware form: a dynamic circuit.
* ``"coherent"``: ``W_y`` multi-controlled by the output ancillas before the measurement
  (the deferred-measurement form; no classical control, more two-qubit gates).

The two are equal as instruments because the block is diagonal in the ancillas' basis
(``docs/architecture.md`` section 2.1).

**Exact simulation** (``branch_simulate``). An independent check of the exported circuit:
it walks the Qiskit instructions and keeps one unnormalised density matrix per classical
record, using Qiskit's own gate matrices, projecting on measurement and applying reset as
the reset channel. It never calls ``qmem``'s circuit or instrument code. Optional noise
(``StepNoise``) adds a channel after each gate, a classical confusion on each measurement
and a flip after each reset, and thermal relaxation on idling qubits during measurement and
reset (and the classical-control latency).

**Noisy instruments** (``noisy_instrument``). With noise, one step is still an instrument
on the memory: ``Phi_y(rho) = Tr_anc[ step_y(rho ⊗ rho_anc) ]``. It is built exactly by
running the branch simulator on the basis operators ``|i><j|`` of the memory, then turned
into Kraus operators (Choi decomposition) so that every ``qmem`` metric applies to it
(``InstrumentModel``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Qubit
from qiskit.circuit.controlflow import CASE_DEFAULT, IfElseOp, SwitchCaseOp
from qiskit.quantum_info import Operator

from .circuits import (
    Circuit,
    ControlledRotationBlock,
    HardwareEfficientBlock,
    OutcomeConditionedBlock,
)
from .instrument import Instrument
from .linalg import DTYPE
from .model import QuantumMemoryModel

FEEDFORWARD = ("dynamic", "coherent")


# -- export ---------------------------------------------------------------------------------
def _hea_gates(qc: QuantumCircuit, block: HardwareEfficientBlock, qubits: Sequence[Any]) -> None:
    """Append a hardware-efficient block; ``qubits[i]`` is the circuit qubit of block qubit i."""
    theta = block.theta.detach().to(torch.float64).numpy()
    local = {q: i for i, q in enumerate(block.qubits)}
    for layer in range(block.depth):
        for i in range(len(block.qubits)):
            for r, name in enumerate(block.rotations):
                getattr(qc, name)(float(theta[layer, i, r]), qubits[i])
        for c, t in block.pairs:
            if block.entangler == "cnot":
                qc.cx(qubits[local[c]], qubits[local[t]])
            elif block.entangler == "cz":
                qc.cz(qubits[local[c]], qubits[local[t]])
            else:  # pragma: no cover - ENTANGLERS has only these two
                raise ValueError(f"unknown entangler {block.entangler!r}")


def _w_circuit(block: OutcomeConditionedBlock, y: int) -> QuantumCircuit:
    """``W_y`` as a circuit on the memory qubits (local order)."""
    w = block.ws[y]
    qc = QuantumCircuit(len(block.memory), name=f"W{y}")
    _hea_gates(qc, w, list(qc.qubits))
    return qc


def _ctrl_state(y: int, n_output: int) -> int:
    """Qiskit ``ctrl_state`` for token ``y`` with controls ``[out_0, ..., out_{n-1}]``."""
    return sum(((y >> (n_output - 1 - k)) & 1) << k for k in range(n_output))


@dataclass
class Registers:
    mem: QuantumRegister
    out: QuantumRegister
    env: QuantumRegister | None
    ys: list[ClassicalRegister] = field(default_factory=list)

    @property
    def qubits(self) -> list[Qubit]:
        q = list(self.mem) + list(self.out)
        return q + (list(self.env) if self.env is not None else [])


def _check_model(model: QuantumMemoryModel) -> None:
    if type(model) is not QuantumMemoryModel:
        raise TypeError(f"only architecture A is exported, not {type(model).__name__}")
    for b in model.circuit.blocks:
        if not isinstance(
            b, HardwareEfficientBlock | ControlledRotationBlock | OutcomeConditionedBlock
        ):
            raise TypeError(f"no Qiskit export for block {type(b).__name__}")
    rho0 = model.rho0
    zero = torch.zeros_like(rho0)
    zero[0, 0] = 1
    if not torch.allclose(rho0, zero, atol=1e-12):
        raise ValueError("only the |0...0> initial memory is exported")


def _append_step(
    qc: QuantumCircuit,
    model: QuantumMemoryModel,
    regs: Registers,
    creg: ClassicalRegister,
    feedforward: str,
    measure: bool = True,
) -> None:
    """One step: the unitary blocks, the readout into ``creg`` and the conditioned ``W_y``."""
    q = regs.qubits
    n_out = model.n_output
    pending: OutcomeConditionedBlock | None = None
    for b in model.circuit.blocks:
        if isinstance(b, HardwareEfficientBlock):
            _hea_gates(qc, b, [q[i] for i in b.qubits])
        elif isinstance(b, ControlledRotationBlock):
            g = b.strength.detach().to(torch.float64).numpy()
            for k, (c, t) in enumerate(b.pairs):
                qc.cry(2 * float(g[0 if b.shared else k]), q[c], q[t])
        elif isinstance(b, OutcomeConditionedBlock):
            if feedforward == "dynamic" and b is not model.circuit.blocks[-1]:
                # W_y acts after the readout, so nothing may follow it in the step
                raise ValueError("dynamic feed-forward needs the conditioned block last")
            if feedforward == "dynamic":
                pending = b
            else:
                mem = [q[i] for i in b.memory]
                outs = [q[i] for i in b.outputs]
                for y in range(2**n_out):
                    gate = (
                        _w_circuit(b, y).to_gate().control(n_out, ctrl_state=_ctrl_state(y, n_out))
                    )
                    qc.append(gate, outs + mem)
    if not measure:
        return
    for k in range(n_out):
        qc.measure(regs.out[k], creg[n_out - 1 - k])
    if pending is not None:
        mem = [q[i] for i in pending.memory]
        if n_out == 1:
            with qc.if_test((creg, 1)) as else_:
                qc.compose(_w_circuit(pending, 1), mem, inplace=True)
            with else_:
                qc.compose(_w_circuit(pending, 0), mem, inplace=True)
        else:  # one if per outcome: if_else is supported more widely than switch_case
            for y in range(2**n_out):
                with qc.if_test((creg, y)):
                    qc.compose(_w_circuit(pending, y), mem, inplace=True)


def make_registers(model: QuantumMemoryModel) -> Registers:
    mem = QuantumRegister(model.n_memory, "mem")
    out = QuantumRegister(model.n_output, "out")
    env = QuantumRegister(model.n_env, "env") if model.n_env else None
    return Registers(mem, out, env)


def export_circuit(
    model: QuantumMemoryModel,
    length: int,
    feedforward: str = "dynamic",
    reset_first: bool = False,
) -> QuantumCircuit:
    """The model as a dynamic circuit for ``length`` tokens (registers ``y0 .. y<length-1>``).

    Ancillas are reset between steps (``(length - 1) * (n_output + n_env)`` resets; with
    ``reset_first`` also before the first step, which a step-by-step noise model uses).
    """
    _check_model(model)
    if feedforward not in FEEDFORWARD:
        raise ValueError(f"feedforward must be one of {FEEDFORWARD}")
    regs = make_registers(model)
    qregs = [r for r in (regs.mem, regs.out, regs.env) if r is not None]
    regs.ys = [ClassicalRegister(model.n_output, f"y{t}") for t in range(length)]
    qc = QuantumCircuit(*qregs, *regs.ys, name=f"qmem_T{length}")
    anc = list(regs.out) + (list(regs.env) if regs.env is not None else [])
    for t in range(length):
        if t > 0 or reset_first:
            for a in anc:
                qc.reset(a)
        _append_step(qc, model, regs, regs.ys[t], feedforward)
    return qc


def export_step(model: QuantumMemoryModel, feedforward: str = "dynamic") -> QuantumCircuit:
    """One step with the ancilla reset in front: the repeating unit of ``export_circuit``."""
    return export_circuit(model, 1, feedforward, reset_first=True)


def export_deferred(model: QuantumMemoryModel, length: int) -> QuantumCircuit:
    """Deferred-measurement form: fresh ancillas per step, no measurement, reset or control.

    Qubits: memory, then (output, env) ancillas of step 0, step 1, ... The joint
    distribution of all the output ancillas at the end equals the dynamic circuit's.
    """
    _check_model(model)
    n_m, n_o, n_e = model.n_memory, model.n_output, model.n_env
    n = n_m + length * (n_o + n_e)
    qc = QuantumCircuit(n, name=f"qmem_deferred_T{length}")
    for t in range(length):
        base = n_m + t * (n_o + n_e)
        sub = QuantumCircuit(n_m + n_o + n_e)
        regs = Registers(
            QuantumRegister(bits=sub.qubits[:n_m]),
            QuantumRegister(bits=sub.qubits[n_m : n_m + n_o]),
            QuantumRegister(bits=sub.qubits[n_m + n_o :]) if n_e else None,
        )
        _append_step(sub, model, regs, ClassicalRegister(n_o), "coherent", measure=False)
        qc.compose(sub, list(range(n_m)) + list(range(base, base + n_o + n_e)), inplace=True)
    return qc


def output_qubits_deferred(model: QuantumMemoryModel, length: int) -> list[list[int]]:
    n_m, n_o, n_e = model.n_memory, model.n_output, model.n_env
    return [[n_m + t * (n_o + n_e) + k for k in range(n_o)] for t in range(length)]


# -- counts ---------------------------------------------------------------------------------
def count_ops(qc: QuantumCircuit) -> dict[str, int]:
    """Gate counts, descending into control flow (every branch counted, as a worst case)."""
    counts: dict[str, int] = {}

    def walk(c: QuantumCircuit) -> None:
        for inst in c.data:
            op = inst.operation
            if isinstance(op, IfElseOp | SwitchCaseOp):
                counts[op.name] = counts.get(op.name, 0) + 1
                for blk in op.blocks:
                    walk(blk)
                continue
            counts[op.name] = counts.get(op.name, 0) + 1

    walk(qc)
    return counts


def control_groups(c: QuantumCircuit) -> list[Any]:
    """The circuit's instructions with each run of ``if_test`` on one register grouped.

    Consecutive ``if`` blocks without ``else`` that test the same classical register for
    different values are mutually exclusive (one per outcome, as ``export_circuit`` writes
    them for more than two outcomes): at most one body runs. Items are instructions or
    lists of such instructions.
    """
    out: list[Any] = []
    for inst in c.data:
        op = inst.operation
        if isinstance(op, IfElseOp) and len(op.blocks) == 1:
            target = op.condition[0]
            prev = out[-1] if out else None
            if isinstance(prev, list) and prev[0].operation.condition[0] is target:
                prev.append(inst)
                continue
            out.append([inst])
            continue
        out.append(inst)
    return out


def two_qubit_count(qc: QuantumCircuit, worst_branch: bool = True) -> int:
    """Two-qubit gates; for control flow the costlier branch (``worst_branch``) or all.

    Mutually exclusive ``if`` runs (``control_groups``) count their costliest body.
    """

    def walk(c: QuantumCircuit) -> int:
        n = 0
        for item in control_groups(c):
            if isinstance(item, list):
                per = [walk(i.operation.blocks[0]) for i in item]
                n += max(per) if worst_branch else sum(per)
                continue
            op = item.operation
            if isinstance(op, IfElseOp | SwitchCaseOp):
                per = [walk(b) for b in op.blocks]
                n += max(per) if worst_branch else sum(per)
            elif op.num_qubits == 2 and op.name not in ("barrier", "measure", "reset"):
                n += 1
        return n

    return walk(qc)


# -- noise ----------------------------------------------------------------------------------
def depolarizing_kraus(p: float, n_qubits: int) -> list[np.ndarray]:
    """``rho -> (1 - p) rho + p I/d`` on ``n_qubits`` (Pauli Kraus form)."""
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    d2 = 4**n_qubits
    out = []
    for idx in range(d2):
        op = np.eye(1, dtype=complex)
        for k in range(n_qubits):
            op = np.kron(op, paulis[(idx // 4 ** (n_qubits - 1 - k)) % 4])
        w = 1 - p + p / d2 if idx == 0 else p / d2
        if w > 0:
            out.append(math.sqrt(w) * op)
    return out


def relaxation_kraus(duration: float, t1: float, t2: float) -> list[np.ndarray]:
    """Amplitude damping then pure dephasing for ``duration`` (zero-temperature ``T1``, ``T2``).

    ``T2 <= 2 T1`` is enforced by capping ``T2``. Coherence decays as ``exp(-duration/T2)``,
    the excited population as ``exp(-duration/T1)``.
    """
    if duration <= 0:
        return [np.eye(2, dtype=complex)]
    t2 = min(t2, 2 * t1)
    gamma = 1 - math.exp(-duration / t1)
    # after amplitude damping coherence is sqrt(1 - gamma); dephase the rest
    lam_total = math.exp(-duration / t2)
    lam_ad = math.sqrt(1 - gamma)
    lam_phi = min(1.0, lam_total / lam_ad) if lam_ad > 0 else 0.0
    ad = [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex),
    ]
    q = (1 - lam_phi) / 2  # phase flip probability
    ph = [
        math.sqrt(1 - q) * np.eye(2, dtype=complex),
        math.sqrt(q) * np.diag([1, -1]).astype(complex),
    ]
    return [b @ a for a in ad for b in ph]


@dataclass
class StepNoise:
    """Noise for the exact simulator, per physical gate and per measurement/reset.

    * ``p1``, ``p2``: depolarising parameter after every 1- and 2-qubit gate (``rz`` is
      virtual on IBM hardware and gets no error), or per-qubit / per-pair dicts.
    * ``readout``: probability that a measured bit is recorded flipped (per qubit or one
      number); the true post-measurement state is the measured one.
    * ``reset``: probability that a reset leaves ``|1>`` instead of ``|0>``.
    * ``t1``, ``t2`` (seconds) and ``idle``: relaxation of every qubit that idles during a
      measurement or reset, for that operation's duration (``measure_time``,
      ``reset_time``), plus ``feedforward_time`` for the classical control of a dynamic
      circuit. ``None`` turns idle relaxation off.
    """

    p1: float | dict[int, float] = 0.0
    p2: float | dict[tuple[int, int], float] = 0.0
    readout: float | dict[int, float] = 0.0
    reset: float | dict[int, float] = 0.0
    t1: float | dict[int, float] | None = None
    t2: float | dict[int, float] | None = None
    measure_time: float = 0.0
    reset_time: float = 0.0
    feedforward_time: float = 0.0
    virtual_rz: bool = True

    def _get(self, v: Any, key: Any) -> Any:
        return v.get(key, 0.0) if isinstance(v, dict) else v

    def gate_kraus(self, name: str, qubits: tuple[int, ...]) -> list[np.ndarray] | None:
        if name in ("barrier", "delay", "id") or (self.virtual_rz and name == "rz"):
            return None
        if len(qubits) == 1:
            p = self._get(self.p1, qubits[0])
        elif len(qubits) == 2:
            p = self._get(self.p2, tuple(qubits))
            if isinstance(self.p2, dict) and not p:
                p = self.p2.get(tuple(reversed(qubits)), 0.0)
        else:
            p = self._get(self.p2, tuple(qubits)) if not isinstance(self.p2, dict) else 0.0
        return depolarizing_kraus(p, len(qubits)) if p > 0 else None

    def idle_kraus(self, qubit: int, duration: float) -> list[np.ndarray] | None:
        if self.t1 is None or duration <= 0:
            return None
        t1 = self._get(self.t1, qubit)
        t2 = self._get(self.t2, qubit) if self.t2 is not None else 2 * t1
        if not t1:
            return None
        return relaxation_kraus(duration, t1, t2)


def scale_noise(noise: StepNoise, factor: float) -> StepNoise:
    """Every error probability times ``factor``; ``T1``, ``T2`` divided by it (0: noiseless)."""

    def s(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: x * factor for k, x in v.items()}
        return v * factor

    def inv(v: Any) -> Any:
        if v is None or factor == 0:
            return None
        if isinstance(v, dict):
            return {k: x / factor for k, x in v.items()}
        return v / factor

    return StepNoise(
        p1=s(noise.p1),
        p2=s(noise.p2),
        readout=s(noise.readout),
        reset=s(noise.reset),
        t1=inv(noise.t1),
        t2=inv(noise.t2),
        measure_time=noise.measure_time,
        reset_time=noise.reset_time,
        feedforward_time=noise.feedforward_time,
        virtual_rz=noise.virtual_rz,
    )


# -- exact branch simulation ----------------------------------------------------------------
class _Sim:
    """Unnormalised density matrices per classical record over the qubits of a circuit.

    States are numpy tensors of shape ``(2,)*n + (2,)*n`` with axis ``k`` = qubit ``k``
    (circuit index; independent of Qiskit's little-endian vector convention).
    """

    def __init__(self, qc: QuantumCircuit, noise: StepNoise | None):
        self.qc = qc
        self.n = qc.num_qubits
        self.noise = noise
        self.qindex = {q: i for i, q in enumerate(qc.qubits)}
        self.cindex = {c: i for i, c in enumerate(qc.clbits)}

    # tensor helpers
    def apply(self, rho: np.ndarray, ops: list[np.ndarray], qubits: Sequence[int]) -> np.ndarray:
        """``sum_k K rho K^dagger`` with ``K`` acting on ``qubits`` (big-endian within ``K``)."""
        k = len(qubits)
        n = self.n
        out = np.zeros_like(rho)
        for op in ops:
            t = op.reshape((2,) * (2 * k))
            # left multiply on the ket axes
            r = np.tensordot(t, rho, axes=(list(range(k, 2 * k)), list(qubits)))
            r = np.moveaxis(r, list(range(k)), list(qubits))
            # right multiply by K^dagger on the bra axes
            r = np.tensordot(r, t.conj(), axes=([n + q for q in qubits], list(range(k, 2 * k))))
            r = np.moveaxis(r, list(range(2 * n - k, 2 * n)), [n + q for q in qubits])
            out = out + r
        return out

    def run(self, rho0: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        branches = {tuple([0] * self.qc.num_clbits): rho0}
        return self._walk(self.qc, list(range(self.n)), list(range(self.qc.num_clbits)), branches)

    def _idle(self, rho: np.ndarray, busy: Sequence[int], duration: float) -> np.ndarray:
        if self.noise is None:
            return rho
        for q in range(self.n):
            if q in busy:
                continue
            ks = self.noise.idle_kraus(q, duration)
            if ks is not None:
                rho = self.apply(rho, ks, [q])
        return rho

    def _walk(self, qc, qmap, cmap, branches):
        qidx = {q: i for i, q in enumerate(qc.qubits)}
        cidx = {c: i for i, c in enumerate(qc.clbits)}
        for inst in qc.data:
            op = inst.operation
            qs = [qmap[qidx[q]] for q in inst.qubits]
            cs = [cmap[cidx[c]] for c in inst.clbits]
            name = op.name
            if name == "barrier" or name == "delay":
                continue
            if name == "measure":
                q, c = qs[0], cs[0]
                eps = 0.0 if self.noise is None else self.noise._get(self.noise.readout, q)
                new: dict[tuple[int, ...], np.ndarray] = {}
                for rec, rho in branches.items():
                    for b in (0, 1):
                        proj = np.zeros((2, 2), dtype=complex)
                        proj[b, b] = 1
                        r = self.apply(rho, [proj], [q])
                        if self.noise is not None:
                            r = self._idle(r, [q], self.noise.measure_time)
                        for recorded, w in ((b, 1 - eps), (1 - b, eps)):
                            if w == 0:
                                continue
                            key = list(rec)
                            key[c] = recorded
                            key = tuple(key)
                            new[key] = new.get(key, 0) + w * r
                branches = new
                continue
            if name == "reset":
                q = qs[0]
                eps = 0.0 if self.noise is None else self.noise._get(self.noise.reset, q)
                k0 = np.array([[1, 0], [0, 0]], dtype=complex)
                k1 = np.array([[0, 1], [0, 0]], dtype=complex)
                ops = [k0, k1]
                if eps > 0:
                    x = np.array([[0, 1], [1, 0]], dtype=complex)
                    ops = [math.sqrt(1 - eps) * k for k in ops] + [
                        math.sqrt(eps) * x @ k for k in ops
                    ]
                out = {}
                for rec, rho in branches.items():
                    r = self.apply(rho, ops, [q])
                    if self.noise is not None:
                        r = self._idle(r, [q], self.noise.reset_time)
                    out[rec] = r
                branches = out
                continue
            if isinstance(op, IfElseOp | SwitchCaseOp):
                branches = self._control(op, inst, qs, cs, qc, cmap, cidx, branches)
                continue
            mat = Operator(op).data
            # Qiskit matrices are little-endian in their qargs: reverse to big-endian
            k = len(qs)
            mat_be = mat.reshape((2,) * (2 * k))
            perm = list(reversed(range(k))) + [k + i for i in reversed(range(k))]
            mat_be = mat_be.transpose(perm).reshape(2**k, 2**k)
            branches = {rec: self.apply(rho, [mat_be], qs) for rec, rho in branches.items()}
            if self.noise is not None:
                ks = self.noise.gate_kraus(name, tuple(qs))
                if ks is not None:
                    branches = {rec: self.apply(rho, ks, qs) for rec, rho in branches.items()}
        return branches

    def _value(self, target, rec, qc, cmap, cidx) -> int:
        if isinstance(target, Clbit):
            return rec[cmap[cidx[target]]]
        bits = [rec[cmap[cidx[b]]] for b in target]  # ClassicalRegister: bit j has weight 2^j
        return sum(b << j for j, b in enumerate(bits))

    def _control(self, op, inst, qs, cs, qc, cmap, cidx, branches):
        if self.noise is not None and self.noise.feedforward_time > 0:
            branches = {
                rec: self._idle(rho, [], self.noise.feedforward_time)
                for rec, rho in branches.items()
            }
        out: dict[tuple[int, ...], np.ndarray] = {}
        for rec, rho in branches.items():
            if isinstance(op, IfElseOp):
                target, value = op.condition
                hit = self._value(target, rec, qc, cmap, cidx) == value
                body = op.blocks[0] if hit else (op.blocks[1] if len(op.blocks) > 1 else None)
            else:
                v = self._value(op.target, rec, qc, cmap, cidx)
                body = None
                for values, blk in op.cases_specifier():
                    if any(x is CASE_DEFAULT or x == v for x in values):
                        body = blk
                        break
            if body is None:
                res = {rec: rho}
            else:
                sub_c = [cs[i] for i in range(len(body.clbits))]
                res = self._walk(body, qs, sub_c, {rec: rho})
            for k2, r in res.items():
                out[k2] = out.get(k2, 0) + r
        return out


def branch_simulate(
    qc: QuantumCircuit, rho0: np.ndarray | None = None, noise: StepNoise | None = None
) -> dict[tuple[int, ...], np.ndarray]:
    """Exact unnormalised state per classical record (clbit tuple in circuit order).

    ``rho0`` is a ``(2^n, 2^n)`` matrix in big-endian circuit-qubit order (default
    ``|0...0>``). Each returned state has the same layout; its trace is the record's
    probability.
    """
    n = qc.num_qubits
    if rho0 is None:
        rho0 = np.zeros((2**n, 2**n), dtype=complex)
        rho0[0, 0] = 1
    sim = _Sim(qc, noise)
    out = sim.run(np.asarray(rho0, dtype=complex).reshape((2,) * (2 * n)))
    return {k: v.reshape(2**n, 2**n) for k, v in out.items()}


def partial_trace_be(rho: np.ndarray, keep: Sequence[int], n: int) -> np.ndarray:
    """Reduce a big-endian ``n``-qubit matrix to ``keep`` (in that order)."""
    t = rho.reshape((2,) * (2 * n))
    drop = [q for q in range(n) if q not in keep]
    for q in sorted(drop, reverse=True):
        t = np.trace(t, axis1=q, axis2=q + t.ndim // 2)
    k = len(keep)
    order = sorted(keep)
    perm = [order.index(q) for q in keep]
    t = t.transpose(perm + [k + p for p in perm])
    return t.reshape(2**k, 2**k)


def sequence_branches(
    qc: QuantumCircuit, model: QuantumMemoryModel, noise: StepNoise | None = None
) -> dict[tuple[int, ...], tuple[float, np.ndarray]]:
    """``{tokens: (p(tokens), memory state given tokens)}`` of an ``export_circuit`` circuit."""
    n_o = model.n_output
    regs = [r for r in qc.cregs if r.name.startswith("y")]
    cpos = {c: i for i, c in enumerate(qc.clbits)}
    res = branch_simulate(qc, noise=noise)
    out: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    mem = list(range(model.n_memory))
    for rec, rho in res.items():
        toks = tuple(sum(rec[cpos[r[j]]] << j for j in range(n_o)) for r in regs)
        p = float(np.trace(rho).real)
        red = partial_trace_be(rho, mem, qc.num_qubits)
        prev = out.get(toks)
        if prev is not None:  # several records can map to one token string (not here)
            red = red + prev[0] * prev[1]
            p = p + prev[0]
        out[toks] = (p, red / p if p > 0 else red)
    return out


# -- noisy instruments ----------------------------------------------------------------------
def noisy_instrument(
    step: QuantumCircuit,
    n_memory: int,
    n_output: int,
    noise: StepNoise | None,
    memory_qubits: Sequence[int] | None = None,
    output_clbits: Sequence[int] | None = None,
    tol: float = 1e-12,
    memory_qubits_out: Sequence[int] | None = None,
) -> Instrument:
    """The instrument one step of ``step`` implements on the memory, with ``noise``.

    ``step`` starts with the ancilla reset (``export_step``, or a transpiled copy). The
    ancilla input is irrelevant (it is reset), so the memory input ``|i><j|`` is padded with
    ``|0><0|`` on every other qubit. ``memory_qubits`` are the step circuit's memory qubits
    in project order (default ``0 .. n_memory-1``) and ``output_clbits`` the clbits of the
    token, most significant first (default: the ``y0`` register, as exported).
    ``memory_qubits_out`` is where the memory ends up after the step (default: where it
    started), for transpiled circuits whose routing permutes qubits.
    """
    n = step.num_qubits
    mem = list(memory_qubits) if memory_qubits is not None else list(range(n_memory))
    mem_out = list(memory_qubits_out) if memory_qubits_out is not None else mem
    if output_clbits is None:
        reg = next(r for r in step.cregs if r.name == "y0")
        cpos = {c: i for i, c in enumerate(step.clbits)}
        output_clbits = [cpos[reg[n_output - 1 - k]] for k in range(n_output)]
    dm = 2**n_memory
    others = [q for q in range(n) if q not in mem]
    # superoperator per outcome: S[y][(a,b),(i,j)] = <a|Phi_y(|i><j|)|b>
    sup = np.zeros((2**n_output, dm, dm, dm, dm), dtype=complex)
    for i in range(dm):
        for j in range(dm):
            ein = np.zeros((dm, dm), dtype=complex)
            ein[i, j] = 1
            full = _embed_memory(ein, mem, others, n)
            for rec, rho in branch_simulate(step, full, noise).items():
                y = sum(rec[c] << (n_output - 1 - k) for k, c in enumerate(output_clbits))
                sup[y, :, :, i, j] += partial_trace_be(rho, mem_out, n)
    kraus = [_kraus_from_superop(sup[y], dm, tol) for y in range(2**n_output)]
    n_k = max(len(k) for k in kraus)
    out = np.zeros((2**n_output, n_k, dm, dm), dtype=complex)
    for y, ks in enumerate(kraus):
        for e, k in enumerate(ks):
            out[y, e] = k
    return Instrument(torch.as_tensor(out, dtype=DTYPE))


def _embed_memory(ein: np.ndarray, mem: list[int], others: list[int], n: int) -> np.ndarray:
    """``ein`` on ``mem`` ⊗ ``|0><0|`` elsewhere, big-endian over all ``n`` qubits."""
    zero = np.zeros((2 ** len(others), 2 ** len(others)), dtype=complex)
    zero[0, 0] = 1
    full = np.kron(ein, zero).reshape((2,) * (2 * n))
    order = mem + others  # axis i of `full` is circuit qubit order[i]
    inv = [order.index(q) for q in range(n)]
    return full.transpose(inv + [n + i for i in inv]).reshape(2**n, 2**n)


def _kraus_from_superop(s: np.ndarray, d: int, tol: float) -> list[np.ndarray]:
    """Kraus operators of the CP map ``s[a, b, i, j] = <a|Phi(|i><j|)|b>`` (Choi eigensystem)."""
    choi = s.transpose(2, 0, 3, 1).reshape(d * d, d * d)  # rows (i, a), cols (j, b)
    choi = (choi + choi.conj().T) / 2
    w, v = np.linalg.eigh(choi)
    if w.min() < -1e-9 * max(1.0, w.max()):
        raise ValueError(f"map is not completely positive (Choi eigenvalue {w.min():.2e})")
    ks = []
    for lam, vec in zip(w, v.T, strict=True):
        if lam > tol:
            ks.append(math.sqrt(lam) * vec.reshape(d, d).T)  # K[a, i] = vec[(i, a)]
    if not ks:
        ks.append(np.zeros((d, d), dtype=complex))
    return ks


class InstrumentModel(QuantumMemoryModel):
    """A ``QuantumMemoryModel`` whose per-step instrument is given (e.g. a noisy one).

    Everything ``qmem`` computes from ``instrument()`` (teacher forcing, generation,
    ``evaluate``, the retention diagnostics) then applies unchanged.
    """

    def __init__(self, instrument: Instrument, n_memory: int, n_output: int, rho0: Any = None):
        super().__init__(n_memory, n_output, [], n_env=0)
        self.circuit = Circuit([], n_memory + n_output)
        self._inst = instrument
        if rho0 is not None:
            self.rho0 = torch.as_tensor(rho0, dtype=DTYPE)

    def instrument(self) -> Instrument:
        return self._inst


# -- backends -------------------------------------------------------------------------------
def backend_noise(
    backend: Any, physical: Sequence[int], feedforward_time: float = 0.0
) -> tuple[StepNoise, dict[str, Any]]:
    """``StepNoise`` from a (fake) backend's calibration, on the step's ``physical`` qubits.

    Indices in the returned noise are the *compact* circuit indices ``0..len(physical)-1``
    (``physical[k]`` is compact qubit ``k``). Gate errors become depolarising parameters
    with the same average gate infidelity (``p = e d / (d - 1)``); the reported measurement
    error is a symmetric readout flip; a reset (measure + conditional X on IBM devices) is
    given the qubit's measurement error; idling qubits relax with the qubit's ``T1``, ``T2``
    for the measurement and reset durations.
    """
    t = backend.target
    phys = list(physical)
    comp = {p: k for k, p in enumerate(phys)}
    p1: dict[int, float] = {}
    p2: dict[tuple[int, int], float] = {}
    ro: dict[int, float] = {}
    rs: dict[int, float] = {}
    t1: dict[int, float] = {}
    t2: dict[int, float] = {}
    one_q = "sx"
    two_q = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
    for p in phys:
        p1[comp[p]] = _depol_param(t[one_q][(p,)].error or 0.0, 2)
        em = t["measure"][(p,)].error or 0.0
        ro[comp[p]] = em
        rs[comp[p]] = em
        qp = t.qubit_properties[p]
        t1[comp[p]] = qp.t1
        t2[comp[p]] = qp.t2
    for (a, b), props in t[two_q].items():
        if a in comp and b in comp and props is not None and props.error is not None:
            p2[(comp[a], comp[b])] = _depol_param(props.error, 4)
    m_time = max(t["measure"][(p,)].duration for p in phys)
    r_time = max(t["reset"][(p,)].duration for p in phys) if "reset" in t.operation_names else 0
    info = {
        "backend": backend.name,
        "physical_qubits": phys,
        "two_qubit_gate": two_q,
        "sx_error": {p: t[one_q][(p,)].error for p in phys},
        "two_qubit_error": {
            f"{phys[a]}-{phys[b]}": t[two_q][(phys[a], phys[b])].error for (a, b) in p2
        },
        "readout_error": {phys[k]: v for k, v in ro.items()},
        "t1_s": {phys[k]: v for k, v in t1.items()},
        "t2_s": {phys[k]: v for k, v in t2.items()},
        "measure_time_s": m_time,
        "reset_time_s": r_time,
        "two_qubit_time_s": max(
            (props.duration or 0.0)
            for (a, b), props in t[two_q].items()
            if a in comp and b in comp and props is not None
        )
        if len(phys) > 1
        else 0.0,
        "feedforward_time_s": feedforward_time,
    }
    noise = StepNoise(
        p1=p1,
        p2=p2,
        readout=ro,
        reset=rs,
        t1=t1,
        t2=t2,
        measure_time=m_time,
        reset_time=r_time,
        feedforward_time=feedforward_time,
    )
    return noise, info


def _depol_param(avg_gate_error: float, d: int) -> float:
    """Depolarising ``p`` with average gate infidelity ``e``: ``e = p (d - 1) / d``."""
    return avg_gate_error * d / (d - 1)


def compact(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[int]]:
    """Drop idle qubits of a transpiled circuit; return it and the kept physical indices."""
    used: set[int] = set()

    def walk(c: QuantumCircuit, qmap: list[int]) -> None:
        idx = {q: i for i, q in enumerate(c.qubits)}
        for inst in c.data:
            qs = [qmap[idx[q]] for q in inst.qubits]
            if isinstance(inst.operation, IfElseOp | SwitchCaseOp):
                for blk in inst.operation.blocks:
                    walk(blk, qs)
            elif inst.operation.name not in ("barrier", "delay"):
                used.update(qs)

    walk(qc, list(range(qc.num_qubits)))
    phys = sorted(used)
    keep = [qc.qubits[p] for p in phys]
    out = QuantumCircuit(QuantumRegister(len(phys), "q"), *qc.cregs)
    qpos = {q: i for i, q in enumerate(keep)}
    for inst in qc.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        if any(q not in qpos for q in inst.qubits):
            raise ValueError("instruction on a qubit outside the kept set")
        out.append(inst.operation, [out.qubits[qpos[q]] for q in inst.qubits], inst.clbits)
    return out, phys


def unitary_matrix_be(qc: QuantumCircuit) -> np.ndarray:
    """A measurement-free circuit's unitary in big-endian qubit order."""
    m = Operator(qc).data
    n = qc.num_qubits
    t = m.reshape((2,) * (2 * n))
    perm = list(reversed(range(n))) + [n + i for i in reversed(range(n))]
    return t.transpose(perm).reshape(2**n, 2**n)


__all__ = [
    "FEEDFORWARD",
    "InstrumentModel",
    "StepNoise",
    "backend_noise",
    "branch_simulate",
    "compact",
    "count_ops",
    "export_circuit",
    "export_deferred",
    "export_step",
    "noisy_instrument",
    "output_qubits_deferred",
    "partial_trace_be",
    "scale_noise",
    "sequence_branches",
    "two_qubit_count",
    "unitary_matrix_be",
]
