"""Phase 7 Stage 2A (issue #20): memory-qubit (q75) destruction / dephasing controls.

Each control modifies Phase 6's transpiled circuit (``results/summaries/phase6/``) only where
intended: one operation on the memory qubit q75 per step, placed after q75's leading
single-qubit gates and before the step's first CZ. In the schedule that operation runs while
q59 is being measured and reset, so it adds no duration.

=============  ======================================================  ====================
variant        operation on q75 in every step                          ideal witness
=============  ======================================================  ====================
``baseline``   none (Phase 6's circuit unchanged)                      Phase 6 prediction
``A1``         ``reset``                                               0 (i.i.d. outputs)
``A2``         ``measure`` into an unused clbit (Z dephasing)          0 (2-state HMM)
``A3``         a random Pauli, fixed per circuit instance              0 as M -> infinity
``A1s``        ``delay`` of the reset's duration                       = baseline
``A2s``        ``delay`` of the measurement's duration                 = baseline
``A2s_spec``   ``delay`` as A2s, plus ``measure`` of spectator q74     = baseline
``A3s``        ``delay`` of the ``x`` gate's duration                  = baseline
=============  ======================================================  ====================

Two parts:

* **numpy only** (importable without Qiskit): superoperators of the per-step maps on the
  joint (q75, q59) memory, composed from committed pieces (``StepPieces``); the predicted
  witness, block statistics and NLLs; random-Pauli instances; shot counts.
* **Qiskit** (the ``hardware`` group, imported lazily): building the control circuits from
  Phase 6's step and full circuits, and the instruments of their pieces through
  ``qmem.hardware.noisy_instrument``.

**The joint model.** Phase 6's step is ``reset q59 -> gates -> CZs -> measure q59``. Its
1-qubit instrument assumes the reset leaves q59 in a state that does not depend on what q59
held. To test alternatives in which it does, the step is factored as
``M_y = Post_y o Op o Pre o Reset`` on the memory (q75, q59), big-endian with q75 first:

* ``Reset``: q59's reset channel ⊗ q75's relaxation for the reset's duration;
* ``Pre``: q75's leading gates; ``Op``: the control's operation on q75 (q59 idles at |0>);
* ``Post_y``: q59's gates, the CZs, the final gates and the measurement of q59 with outcome y.

With the symmetric reset of the calibrated noise model this reproduces Phase 6's 1-qubit
instrument exactly (tested). Superoperators act on column-stacked ``vec(rho)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import witness as W

VARIANTS = ("baseline", "A1", "A2", "A3", "A1s", "A2s", "A2s_spec", "A3s")
PAULIS = ("I", "X", "Y", "Z")
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
_PAULI_M = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


# -- superoperators (numpy) ---------------------------------------------------------------------


def superop(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """``S`` with ``vec(sum K rho K^dag) = S vec(rho)`` (column-stacked ``vec``)."""
    return sum(np.kron(np.conj(k), k) for k in kraus)


def kraus_kron(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Kraus operators of ``A ⊗ B`` (``A`` on the most significant qubit)."""
    return [np.kron(x, y) for x in a for y in b]


def reset_kraus(e0: float, e1: float) -> list[np.ndarray]:
    """Reset that leaves |1> with probability ``e0`` from |0> and ``e1`` from |1>.

    ``e0 = e1 = eps`` is the calibrated model's reset (``qmem.hardware``: perfect reset, then a
    flip with probability ``eps``), which forgets the input. ``e0 != e1`` keeps one classical
    bit of the input: the residual memory of a measure-and-conditional-X reset.
    """
    return [
        math.sqrt(1 - e0) * np.array([[1, 0], [0, 0]], dtype=complex),
        math.sqrt(e0) * np.array([[0, 0], [1, 0]], dtype=complex),
        math.sqrt(1 - e1) * np.array([[0, 1], [0, 0]], dtype=complex),
        math.sqrt(e1) * np.array([[0, 0], [0, 1]], dtype=complex),
    ]


def dephase_kraus(p: float) -> list[np.ndarray]:
    """Z dephasing that multiplies coherences by ``1 - p`` (``p = 1``: a Z measurement)."""
    q = p / 2
    return [math.sqrt(1 - q) * _I2, math.sqrt(q) * _Z]


def depol_kraus(p: float) -> list[np.ndarray]:
    """``rho -> (1 - p) rho + p I/2`` on one qubit."""
    w = [1 - 3 * p / 4, p / 4, p / 4, p / 4]
    return [math.sqrt(x) * m for x, m in zip(w, (_I2, _X, _Y, _Z), strict=True) if x > 0]


def rz_unitary(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def on_q75(kraus: Sequence[np.ndarray]) -> np.ndarray:
    return superop(kraus_kron(kraus, [_I2]))


def on_q59(kraus: Sequence[np.ndarray]) -> np.ndarray:
    return superop(kraus_kron([_I2], kraus))


def ptrace_q59(m4: np.ndarray) -> np.ndarray:
    """Trace out q59 (the least significant qubit) of a 4x4 matrix."""
    return np.einsum("aibi->ab", m4.reshape(2, 2, 2, 2))


@dataclass
class StepPieces:
    """Committed pieces of one noise level (all on the (q75, q59) memory unless noted).

    ``pre`` (16, 16), ``post`` (2, 16, 16), the relaxation Kraus sets the calibrated model
    applies to idling qubits (``relax_q75_reset``: q75 during q59's reset;
    ``relax_q59_reset`` / ``relax_q59_measure``: q59 during q75's reset / measurement), the
    reset errors and ``p1_q75`` (depolarising after an ``x`` on q75).
    """

    pre: np.ndarray
    post: np.ndarray
    relax_q75_reset: np.ndarray
    relax_q59_reset: np.ndarray
    relax_q59_measure: np.ndarray
    eps_q59: float
    eps_q75: float
    p1_q75: float

    @classmethod
    def from_npz(cls, z: Any, noise: str) -> StepPieces:
        g = lambda k: z[f"{noise}__{k}"]  # noqa: E731
        return cls(
            g("pre"),
            g("post"),
            g("relax_q75_reset"),
            g("relax_q59_reset"),
            g("relax_q59_measure"),
            float(g("eps_q59")),
            float(g("eps_q75")),
            float(g("p1_q75")),
        )

    def to_dict(self, noise: str) -> dict[str, np.ndarray]:
        return {
            f"{noise}__pre": self.pre,
            f"{noise}__post": self.post,
            f"{noise}__relax_q75_reset": self.relax_q75_reset,
            f"{noise}__relax_q59_reset": self.relax_q59_reset,
            f"{noise}__relax_q59_measure": self.relax_q59_measure,
            f"{noise}__eps_q59": np.array(self.eps_q59),
            f"{noise}__eps_q75": np.array(self.eps_q75),
            f"{noise}__p1_q75": np.array(self.p1_q75),
        }


@dataclass
class Hypothesis:
    """Deviations from the calibrated model, as channels inserted into the step.

    * ``q59_reset``: ``(e0, e1)`` for q59's reset (``None``: the model's symmetric reset).
    * ``q75_reset``: the same for A1's reset of q75.
    * ``op_coherence``: coherence of q75 kept by A2's measurement (0: projective).
    * ``spec_dephasing``: Z dephasing of q75 by the spectator measurement in ``A2s_spec``.
    * ``q75_phase``: an extra Z rotation of q75 per step (a detuning), in rad.
    """

    name: str = "H_q75"
    q59_reset: tuple[float, float] | None = None
    q75_reset: tuple[float, float] | None = None
    op_coherence: float = 0.0
    spec_dephasing: float = 0.0
    q75_phase: float = 0.0


def op_superop(variant: str, pieces: StepPieces, hyp: Hypothesis, pauli: str = "I") -> np.ndarray:
    """The superoperator of the control's operation on q75 (with the idle of q59)."""
    if variant in ("baseline", "A1s", "A2s", "A3s"):
        return np.eye(16, dtype=complex)
    if variant == "A2s_spec":
        return on_q75(dephase_kraus(hyp.spec_dephasing))
    if variant == "A1":
        e = hyp.q75_reset or (pieces.eps_q75, pieces.eps_q75)
        return superop(kraus_kron(reset_kraus(*e), pieces.relax_q59_reset))
    if variant == "A2":
        return superop(kraus_kron(dephase_kraus(1 - hyp.op_coherence), pieces.relax_q59_measure))
    if variant == "A3":
        u = _PAULI_M[pauli]
        if pauli in ("X", "Y"):  # one physical x gate (Y = x . rz(pi)), depolarised after
            return on_q75([k @ u for k in depol_kraus(pieces.p1_q75)])
        return on_q75([u])  # I: a delay; Z: a virtual rz
    raise ValueError(f"unknown variant {variant!r}")


def reset_superop(pieces: StepPieces, hyp: Hypothesis) -> np.ndarray:
    e = hyp.q59_reset or (pieces.eps_q59, pieces.eps_q59)
    s = superop(kraus_kron(pieces.relax_q75_reset, reset_kraus(*e)))
    if hyp.q75_phase:
        s = on_q75([rz_unitary(hyp.q75_phase)]) @ s
    return s


def step_superops(
    variant: str, pieces: StepPieces, hyp: Hypothesis | None = None, pauli: str = "I"
) -> np.ndarray:
    """``S[y] = Post_y Op Pre Reset`` (2, 16, 16) for one step of ``variant``."""
    hyp = hyp or Hypothesis()
    op = op_superop(variant, pieces, hyp, pauli)
    pre = op @ pieces.pre @ reset_superop(pieces, hyp)
    return np.stack([pieces.post[y] @ pre for y in range(2)])


RHO0_JOINT = np.diag([1.0, 0, 0, 0]).astype(complex)  # q75 = q59 = |0>


def process(sup: np.ndarray, rho0: np.ndarray = RHO0_JOINT) -> W.LinearProcess:
    return W.LinearProcess.instrument(sup, rho0)


def mixture(procs: Sequence[W.LinearProcess], weights: Sequence[float]) -> W.LinearProcess:
    """The process that runs ``procs[i]`` with probability ``weights[i]`` (direct sum)."""
    dims = [len(p.init) for p in procs]
    d = sum(dims)
    ops = np.zeros((procs[0].ops.shape[0], d, d), dtype=complex)
    init = np.zeros(d, dtype=complex)
    final = np.zeros(d, dtype=complex)
    o = 0
    for p, w, n in zip(procs, weights, dims, strict=True):
        ops[:, o : o + n, o : o + n] = p.ops
        init[o : o + n] = w * p.init
        final[o : o + n] = p.final
        o += n
    return W.LinearProcess(init, ops, final)


# -- predicted statistics (numpy) ---------------------------------------------------------------


@dataclass
class Settings:
    length: int = 64
    burn: int = 16
    k_max: int = 7
    offsets: tuple[int, ...] = (0, 1, 2, 3)
    block: int = 4


def predict(proc: W.LinearProcess, s: Settings, ref: W.LinearProcess | None = None) -> dict:
    """Witness (common start set), g, P(1), aligned L-block probabilities and KL vs ``ref``."""
    g = proc.expected_windows(s.k_max, s.burn, s.length, common=True)
    out: dict[str, Any] = {"g": g.tolist()}
    for j in s.offsets:
        out[f"w{j}"] = float(W.witness(g, j))
    out["p1"] = float(proc.marginals(s.length).mean())
    starts = list(range(s.burn, s.length - s.block + 1, s.block))
    blocks = proc.block_probs(starts, s.block)
    out["blocks"] = blocks.tolist()
    if ref is not None:
        q = ref.block_probs(starts, s.block)
        out["block_kl_vs_ref"] = float(np.sum(blocks * np.log(blocks / q)) / s.block)
    return out


def paired_nll(
    proc: W.LinearProcess, ref: W.LinearProcess, n: int, length: int, seed: int
) -> tuple[float, float]:
    """Mean and SE (per symbol) of ``NLL_ref - NLL_proc`` on ``n`` shots drawn from ``proc``.

    Its expectation is the KL rate of ``proc`` from ``ref``: what a paired NLL of the control's
    data against Phase 6's calibrated model would show.
    """
    shots = proc.sample(n, length, seed)
    d = (ref.seq_nll(shots) - proc.seq_nll(shots)) / length
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))


def per_shot_sd(
    proc: W.LinearProcess, s: Settings, n: int, n_boot: int, seed: int
) -> dict[int, float]:
    """Per-shot SD of w(j) (bootstrap SE on ``n`` simulated shots, times sqrt(n))."""
    shots = proc.sample(n, s.length, seed)
    per = W.per_shot_windows(shots, s.k_max, s.burn, common=True)
    idx = W.bootstrap_indices(n, n_boot, seed + 1)
    r = W.witness_stats(per, list(s.offsets), idx)
    return {j: r.se[j] * math.sqrt(n) for j in s.offsets}


# -- random-Pauli instances (numpy) -------------------------------------------------------------


def instance_windows(
    bank: Sequence[W.LinearProcess], seqs: np.ndarray, s: Settings, chunk: int = 2000
) -> np.ndarray:
    """Exact expected ``g(k)`` (common start set) of each instance.

    ``bank[p]`` is the step process with Pauli ``p``; ``seqs`` (n_inst, length) picks the
    Pauli of every step of every instance. Returns (n_inst, k_max + 1).
    """
    ops = np.stack([b.ops for b in bank])  # (P, 2, D, D), row-vector convention
    chan = ops.sum(1)
    final = bank[0].final
    last = s.length - s.k_max - 2
    out = np.zeros((len(seqs), s.k_max + 1))
    for c0 in range(0, len(seqs), chunk):
        sq = seqs[c0 : c0 + chunk]
        st = np.tile(bank[0].init, (len(sq), 1))
        for t in range(s.burn):
            st = np.einsum("nd,nde->ne", st, chan[sq[:, t]])
        for t in range(s.burn, last + 1):
            cur = np.einsum("nd,nde->ne", st, ops[sq[:, t], 1])
            for k in range(s.k_max + 1):
                u = t + k + 1
                out[c0 : c0 + chunk, k] += np.einsum(
                    "nd,nde,e->n", cur, ops[sq[:, u], 1], final
                ).real
                cur = np.einsum("nd,nde->ne", cur, ops[sq[:, u], 0])
            st = np.einsum("nd,nde->ne", st, chan[sq[:, t]])
    return out / (last - s.burn + 1)


def finite_m_witness(
    g_inst: np.ndarray, m: int, draws: int, seed: int, j: int = 0
) -> tuple[float, float]:
    """Mean and SD of w(j) of the average of ``m`` instances drawn from the pool ``g_inst``.

    The realised circuit set is fixed for an experiment, so the shot-averaged witness
    converges to w of this ``m``-instance mixture, not to the M -> infinity value.
    """
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g_inst), size=(draws, m))
    w = W.witness(g_inst[pick].mean(1), j)
    return float(np.mean(w)), float(np.std(w, ddof=1))


# -- shots (numpy) ------------------------------------------------------------------------------


def shots_5sigma(delta: float, sd: float, z: float = 5.0) -> float:
    """Shots for ``|delta| = z * sd / sqrt(N)`` (``inf`` if ``delta = 0``)."""
    return math.inf if delta == 0 else (z * sd / delta) ** 2


def qpu_seconds(shots: float, duration_s: float, rep_delay_s: float, overhead_s: float) -> float:
    return overhead_s + shots * (rep_delay_s + duration_s)


# -- circuits (Qiskit; the `hardware` group) ----------------------------------------------------


def _is_first_cz_block(name: str) -> bool:
    return name == "cz"


def restructure(
    qc: Any,
    memory: Any,
    output: Any,
    make_op: Callable[[Any, int], None] | None,
    placement: str = "concurrent",
    drop_reset: bool = False,
    extra_regs: Sequence[Any] = (),
    keep: Callable[[Any], bool] | None = None,
) -> Any:
    """Copy ``qc`` with ``make_op(new_qc, t)`` inserted before the first CZ of every step.

    Within each step, the instructions before the first CZ are re-emitted as: resets, the
    memory qubit's gates, the operation (``placement="concurrent"``), the other gates, so the
    operation acts after q75's leading gates while q59 is still at its reset state. With
    ``placement="serial"`` it comes after q59's gates (q59 idles in a live state; a
    simulation sensitivity only). Qubit-wise order is preserved, so without an operation
    the circuit is unchanged as a sequence of operations on each qubit. Steps end at the
    measurement of ``output``. ``drop_reset`` removes the resets of ``output`` (for the
    joint-memory pieces); ``keep`` filters instructions.
    """
    new = qc.copy_empty_like()
    for r in extra_regs:
        new.add_register(r)
    buf: list[Any] = []
    t, pre = 0, True

    def emit(inst: Any) -> None:
        if keep is None or keep(inst):
            new.append(inst.operation, inst.qubits, inst.clbits)

    for inst in qc.data:
        name = inst.operation.name
        if pre and _is_first_cz_block(name):
            resets = [i for i in buf if i.operation.name == "reset"]
            mem = [i for i in buf if i.operation.name != "reset" and i.qubits == (memory,)]
            rest = [i for i in buf if i.operation.name != "reset" and i.qubits != (memory,)]
            for i in resets:
                if not (drop_reset and i.qubits == (output,)):
                    emit(i)
            for i in mem:
                emit(i)
            if placement == "serial":
                for i in rest:
                    emit(i)
            if make_op is not None:
                make_op(new, t)
            if placement != "serial":
                for i in rest:
                    emit(i)
            buf, pre = [], False
            emit(inst)
            continue
        if pre:
            buf.append(inst)
            continue
        emit(inst)
        if name == "measure" and inst.qubits == (output,):
            t, pre = t + 1, True
    if buf:  # a trailing segment without a CZ (not in Phase 6's circuits)
        for i in buf:
            emit(i)
    return new


@dataclass
class Durations:
    """Durations (s) used for the delays of the shams and in the duration table."""

    reset: float
    measure: float
    x: float
    dt: float

    def dt_units(self, seconds: float) -> int:
        return round(seconds / self.dt)


def op_builder(
    variant: str,
    memory: Any,
    dur: Durations,
    creg: Any = None,
    spectator: Any = None,
    spec_creg: Any = None,
    paulis: Sequence[str] | None = None,
) -> Callable[[Any, int], None] | None:
    """``make_op(qc, t)`` for ``restructure``: the control's operation in step ``t``."""
    if variant == "baseline":
        return None

    def op(qc: Any, t: int) -> None:
        if variant == "A1":
            qc.reset(memory)
        elif variant == "A2":
            qc.measure(memory, creg[t])
        elif variant == "A1s":
            qc.delay(dur.dt_units(dur.reset), memory, unit="dt")
        elif variant in ("A2s", "A2s_spec"):
            qc.delay(dur.dt_units(dur.measure), memory, unit="dt")
            if variant == "A2s_spec":
                qc.measure(spectator, spec_creg[t])
        elif variant == "A3s":
            qc.delay(dur.dt_units(dur.x), memory, unit="dt")
        elif variant == "A3":
            p = paulis[t]
            if p in ("Z", "Y"):
                qc.rz(math.pi, memory)
            if p in ("X", "Y"):
                qc.x(memory)
            else:
                qc.delay(dur.dt_units(dur.x), memory, unit="dt")
        else:
            raise ValueError(f"unknown variant {variant!r}")

    return op


def build_variant(
    qc: Any,
    variant: str,
    memory: int,
    output: int,
    dur: Durations,
    n_steps: int,
    spectator: int | None = None,
    paulis: Sequence[str] | None = None,
    placement: str = "concurrent",
    drop_reset: bool = False,
    keep: Callable[[Any], bool] | None = None,
) -> Any:
    """``qc`` (Phase 6's step or full circuit) with ``variant``'s operation on ``memory``."""
    from qiskit import ClassicalRegister

    extra, creg, scr = [], None, None
    if variant == "A2":
        creg = ClassicalRegister(n_steps, "m")
        extra.append(creg)
    if variant == "A2s_spec":
        scr = ClassicalRegister(n_steps, "s")
        extra.append(scr)
    mq, oq = qc.qubits[memory], qc.qubits[output]
    sq = qc.qubits[spectator] if spectator is not None else None
    make = op_builder(variant, mq, dur, creg, sq, scr, paulis)
    return restructure(qc, mq, oq, make, placement, drop_reset, extra, keep)


def pieces_circuits(step: Any, memory: int, output: int) -> tuple[Any, Any]:
    """``Pre`` (q75's leading gates) and ``Post`` (the rest, without the reset of q59)."""
    marker: dict[str, bool] = {"seen": False}
    full = restructure(step, step.qubits[memory], step.qubits[output], None, drop_reset=True)
    pre = full.copy_empty_like()
    post = full.copy_empty_like()
    for inst in full.data:
        if inst.operation.name == "cz":
            marker["seen"] = True
        target = post if (marker["seen"] or inst.qubits != (full.qubits[memory],)) else pre
        target.append(inst.operation, inst.qubits, inst.clbits)
    return pre, post


def instrument_superops(qc: Any, noise: Any, memory: Sequence[int], n_output: int) -> np.ndarray:
    """Superoperators (Y, d^2, d^2) of ``qc`` on ``memory`` via ``hardware.noisy_instrument``."""
    from . import hardware as hw

    clb = None if n_output else []
    inst = hw.noisy_instrument(qc, len(memory), n_output, noise, list(memory), output_clbits=clb)
    k = inst.kraus.numpy()
    return np.stack([superop(list(k[y])) for y in range(k.shape[0])])


def export_pieces(step: Any, noise: Any, memory: int, output: int) -> StepPieces:
    """The committed pieces of one noise level (``noise=None``: noiseless)."""
    from . import hardware as hw

    pre_c, post_c = pieces_circuits(step, memory, output)
    joint = [memory, output]  # q75 most significant
    pre = instrument_superops(pre_c, noise, joint, 0)[0]
    post = instrument_superops(post_c, noise, joint, 1)

    def relax(q: int, duration: float) -> np.ndarray:
        if noise is None:
            return np.array([_I2])
        ks = noise.idle_kraus(q, duration)
        return np.array(ks if ks is not None else [_I2])

    get = (lambda v, q: 0.0) if noise is None else (lambda v, q: float(noise._get(v, q)))
    rt = 0.0 if noise is None else noise.reset_time
    mt = 0.0 if noise is None else noise.measure_time
    p1 = 0.0 if noise is None else get(noise.p1, memory)
    del hw
    return StepPieces(
        pre=pre,
        post=post,
        relax_q75_reset=relax(memory, rt),
        relax_q59_reset=relax(output, rt),
        relax_q59_measure=relax(output, mt),
        eps_q59=0.0 if noise is None else get(noise.reset, output),
        eps_q75=0.0 if noise is None else get(noise.reset, memory),
        p1_q75=p1,
    )


__all__ = [
    "PAULIS",
    "RHO0_JOINT",
    "VARIANTS",
    "Durations",
    "Hypothesis",
    "Settings",
    "StepPieces",
    "build_variant",
    "dephase_kraus",
    "depol_kraus",
    "export_pieces",
    "finite_m_witness",
    "instance_windows",
    "instrument_superops",
    "mixture",
    "op_superop",
    "paired_nll",
    "per_shot_sd",
    "pieces_circuits",
    "predict",
    "process",
    "qpu_seconds",
    "reset_kraus",
    "reset_superop",
    "restructure",
    "shots_5sigma",
    "step_superops",
    "superop",
]
