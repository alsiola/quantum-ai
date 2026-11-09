"""Phase 8 Stage 2 (issue #32): coherence-specific tests on q75, local design only.

Numpy only (the circuits are in ``experiments/phase8_coherence.py``, ``hardware`` group).
Three parts:

**1. A fitted quantum model** (#30's review point 1). #27's calibrated model (#20's pieces at
the pilot's live calibration, the pilot's q75 decay under measurement) with a few physically
motivated free noise parameters (``NoiseParams``), all on the joint (q75, q59) memory:

* ``dephase``, ``damp``, ``detune``: extra q75 coherence loss, amplitude damping and Z rotation
  per step, applied while q75 idles through q59's reset (T2-, T1- and detuning-type);
* ``flip``: an extra symmetric flip of q59's recorded symbol (readout);
* ``reset_e0``, ``reset_e1``: q59's reset leaves ``|1>`` with these probabilities from
  ``|0>`` and ``|1>`` (``None``: the calibrated symmetric reset; #21's residual memory);
* ``meas_decay``: A2's ``|1> -> |0>`` decay of q75 per mid-circuit measurement (#26);
* ``spec_dephase``, ``spec_phase``: the q74 sham's effect on q75 (measurement-induced
  dephasing and a ZZ/Stark phase).

The arms of Stage 4c are its step superoperators ``S[y] = F(Post_y Slot Pre Reset)``. Every
default reproduces #27's analysis model exactly (tested against #30's likelihoods).

**2. Physical phase insertions and randomised dephasings.** IBM's ``rz`` is a frame change
that the compiler may cancel or commute. A *physical* Z rotation is a frame-neutral echo of
two ``x`` pulses, time order ``x, rz(a), x, rz(-a)``: its unitary is ``Rz(-a) X Rz(a) X =
Rz(-2a)``, so ``a = -phi / 2`` gives ``Rz(phi)``, and the frame after it is unchanged
(``echo_kraus``). ``e<k>`` is the echo with ``phi = k pi / 4``; ``e0`` (two ``x`` pulses) is
the matched-timing sham. A randomised dephasing draws its phase per shot and per step from
fresh ancillas prepared in ``|+>`` and measured; the bits ``b`` choose the echo by
feed-forward and are recorded. Averaged over ``b`` (independent of everything else), each step
applies the channel ``sum_b p(b) U_b rho U_b^dag``: a Z dephasing with coherence factor
``|sum_b p(b) exp(i phi_b)|`` (``randomised_channel``; tested against brute-force enumeration
of every bit sequence). The element ``ff<b1><b2>`` is the ancillas' measurement and the
feed-forward latency, with q75 idling (and, as an alternative, a ZZ phase on q75 conditional
on the bits).

**3. Arms, banks and statistics.** An arm is a ``qmem.classical.ArmSpec`` (programs of named
elements per step). ``QuantumModel.bank(arm)`` gives its step operators in the row-vector
convention of ``qmem.witness``, so the likelihood, sampling and statistics code is shared with
the classical program models of ``qmem.classical``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from . import ablate_hw as A
from . import classical as C
from . import drift as D
from . import witness as W

N_PHASES = 8  # e<k>: phi = k * 2 pi / N_PHASES
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
VEC0 = A.RHO0_JOINT.reshape(-1, order="F")
TRACE = np.eye(4, dtype=complex).reshape(-1, order="F")
# vec indices (column-stacked, q75 most significant) with q59 diagonal: every step ends with
# q59's measurement, so each step's image lies in this 8-dimensional subspace (exact)
DIAG59 = np.array([i + 4 * j for j in range(4) for i in range(4) if i % 2 == j % 2])


# -- one-qubit pieces -----------------------------------------------------------------------------


def rx(theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _I2 - 1j * math.sin(theta / 2) * _X


def compose(*kraus_sets: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Kraus operators of the channels applied in time order (first set first)."""
    out = [_I2]
    for ks in kraus_sets:
        out = [k @ o for k in ks for o in out]
    return out


def x_kraus(p1: float = 0.0, over: float = 0.0) -> list[np.ndarray]:
    """An ``x`` pulse (``Rx(pi (1 + over))``, ``over``: amplitude error), then depolarising
    ``p1`` (the calibrated model's gate error)."""
    return compose([rx(math.pi * (1 + over))], A.depol_kraus(p1) if p1 > 0 else [_I2])


def echo_phase(k: int, n: int = N_PHASES) -> float:
    return 2 * math.pi * k / n


def echo_angles(phi: float) -> tuple[float, float]:
    """The ``rz`` angles ``(a, -a)`` of the echo ``x, rz(a), x, rz(-a)`` giving ``Rz(phi)``."""
    a = -phi / 2
    return a, -a


def echo_kraus(phi: float, p1: float = 0.0, over: float = 0.0) -> list[np.ndarray]:
    """The physical phase insertion ``x, rz(a), x, rz(-a)`` (time order), ``a = -phi / 2``.
    Ideal unitary ``Rz(phi)`` up to a global phase; frame-neutral (the ``rz`` sum is 0)."""
    a, b = echo_angles(phi)
    return compose(x_kraus(p1, over), [A.rz_unitary(a)], x_kraus(p1, over), [A.rz_unitary(b)])


def echo_unitary(phi: float, over: float = 0.0) -> np.ndarray:
    (u,) = echo_kraus(phi, 0.0, over)
    return u


def kraus_channel(kraus: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    return sum(k @ rho @ k.conj().T for k in kraus)


def randomised_channel(
    phases: Sequence[float], probs: Sequence[float] | None = None
) -> list[np.ndarray]:
    """Kraus operators ``sqrt(p_b) Rz(phi_b)`` of the per-step random phase (the average over
    the feed-forward bits of one step)."""
    p = np.full(len(phases), 1 / len(phases)) if probs is None else np.asarray(probs)
    return [math.sqrt(q) * A.rz_unitary(ph) for q, ph in zip(p, phases, strict=True)]


def coherence_factor(phases: Sequence[float], probs: Sequence[float] | None = None) -> float:
    """``|sum_b p(b) exp(i phi_b)|``: what the randomised channel multiplies coherences by."""
    p = np.full(len(phases), 1 / len(phases)) if probs is None else np.asarray(probs)
    return float(abs(np.sum(p * np.exp(1j * np.asarray(phases)))))


def relax_kraus(duration: float, t1: float, t2: float) -> list[np.ndarray]:
    """Amplitude damping and pure dephasing of an idle qubit (zero temperature)."""
    if duration <= 0:
        return [_I2]
    g = 1 - math.exp(-duration / t1)
    tphi_rate = max(1 / t2 - 1 / (2 * t1), 0.0)
    lam = 1 - math.exp(-2 * duration * tphi_rate)  # coherence x sqrt(1 - g) x (1 - lam)
    return compose(A.amp_damp_kraus(g), A.dephase_kraus(lam))


# -- the quantum model ---------------------------------------------------------------------------


@dataclass
class NoiseParams:
    """Deviations from #27's calibrated model (defaults: none; see the module docstring)."""

    dephase: float = 0.0
    damp: float = 0.0
    detune: float = 0.0
    flip: float = 0.0
    reset_e0: float | None = None
    reset_e1: float | None = None
    meas_decay: float = 0.0
    meas_decay_first: float | None = None
    spec_dephase: float = 0.0
    spec_phase: float = 0.0
    # design alternatives (never fitted: no Stage 4c arm constrains them)
    over: float = 0.0  # x-pulse amplitude error
    ff_idle_s: float = 0.0  # q75 idle per feed-forward element (ancilla readout + latency)
    ff_phase: tuple[float, float] = (0.0, 0.0)  # q75 Z phase when ancilla bit 1 / bit 2 is 1
    stark: tuple[float, float] = (0.0, 0.0)  # q75 Z phase after q59 reads y = 0 / 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoiseParams:
        kw = dict(d)
        for k in ("ff_phase", "stark"):
            if k in kw:
                kw[k] = tuple(kw[k])
        return cls(**kw)


FITTED = (
    "dephase",
    "damp",
    "detune",
    "flip",
    "reset_e0",
    "reset_e1",
    "meas_decay",
    "spec_dephase",
    "spec_phase",
)
_UNIT = {"dephase", "damp", "flip", "meas_decay", "spec_dephase"}  # in [0, 1)
_HALF = {"reset_e0", "reset_e1"}  # in (0, 1/2)


def to_vector(p: NoiseParams, names: Sequence[str], pieces: A.StepPieces) -> np.ndarray:
    """Unconstrained coordinates (logit for bounded parameters) of ``names``."""
    out = []
    for k in names:
        v = getattr(p, k)
        if v is None:  # the calibrated symmetric reset
            v = pieces.eps_q59
        if k in _UNIT:
            v = min(max(v, 1e-6), 1 - 1e-6)
            out.append(math.log(v / (1 - v)))
        elif k in _HALF:
            v = min(max(2 * v, 1e-6), 1 - 1e-6)
            out.append(math.log(v / (1 - v)))
        else:
            out.append(float(v))
    return np.array(out)


def from_vector(x: np.ndarray, names: Sequence[str], base: NoiseParams) -> NoiseParams:
    kw = {}
    for k, v in zip(names, x, strict=True):
        s = 1 / (1 + math.exp(-float(np.clip(v, -50, 50))))
        kw[k] = s if k in _UNIT else (s / 2 if k in _HALF else float(v))
    return replace(base, **kw)


class QuantumModel:
    """The trained instrument's step on the (q75, q59) memory with ``NoiseParams``."""

    def __init__(
        self,
        pieces: A.StepPieces,
        readout_q75: float,
        params: NoiseParams | None = None,
        t1_q75: float = math.inf,
        t2_q75: float = math.inf,
    ):
        self.pieces = pieces
        self.readout = readout_q75
        self.p = params or NoiseParams()
        self.t1, self.t2 = t1_q75, t2_q75
        self._cache: dict[Any, np.ndarray] = {}
        p = self.p
        hyp = A.Hypothesis(
            q59_reset=None
            if p.reset_e0 is None and p.reset_e1 is None
            else (
                pieces.eps_q59 if p.reset_e0 is None else p.reset_e0,
                pieces.eps_q59 if p.reset_e1 is None else p.reset_e1,
            )
        )
        extra = compose(
            A.amp_damp_kraus(p.damp) if p.damp else [_I2],
            A.dephase_kraus(p.dephase) if p.dephase else [_I2],
            [A.rz_unitary(p.detune)],
        )
        self._lead = pieces.pre @ A.on_q75(extra) @ A.reset_superop(pieces, hyp)

    # elements ------------------------------------------------------------------------------
    def element(self, name: str, label: int | None = None) -> np.ndarray:
        """(16, 16) superoperator of one slot element (``meas`` with ``label``: q75's recorded
        bit; ``meas0`` the first step's measurement, with ``meas_decay_first``)."""
        key = (name, label)
        if key in self._cache:
            return self._cache[key]
        pc, p = self.pieces, self.p
        if name == "I":
            out = np.eye(16, dtype=complex)
        elif name.startswith("p:"):  # the echo at any phase (prediction curves)
            out = A.on_q75(echo_kraus(float(name[2:]), pc.p1_q75, p.over))
        elif name.startswith("e") and name[1:].isdigit():
            out = A.on_q75(echo_kraus(echo_phase(int(name[1:])), pc.p1_q75, p.over))
        elif name == "spec":
            out = A.on_q75(compose(A.dephase_kraus(p.spec_dephase), [A.rz_unitary(p.spec_phase)]))
        elif name == "reset":
            out = A.op_superop("A1", pc, A.Hypothesis())
        elif name in ("X", "Y", "Z"):
            out = A.op_superop("A3", pc, A.Hypothesis(), name)
        elif name in ("meas", "meas0"):
            decay = p.meas_decay
            if name == "meas0" and p.meas_decay_first is not None:
                decay = p.meas_decay_first
            phys = [A.superop(A.kraus_kron(k, pc.relax_q59_measure)) for k in A.meas_kraus(decay)]
            r = self.readout
            rec = [(1 - r) * phys[0] + r * phys[1], (1 - r) * phys[1] + r * phys[0]]
            out = rec[0] + rec[1] if label is None else rec[label]
        elif name.startswith("ff"):
            b = [int(c) for c in name[2:]] or [0]
            ph = sum(p.ff_phase[i] for i, bit in enumerate(b) if bit)
            out = A.on_q75(compose(relax_kraus(p.ff_idle_s, self.t1, self.t2), [A.rz_unitary(ph)]))
        elif name.startswith("prep"):  # prep0 / prep1 / prep+ / prep+i after a q75 reset
            u = {
                "prep0": _I2,
                "prep1": rx(math.pi),
                "prep+": np.array([[1, -1], [1, 1]], dtype=complex) / math.sqrt(2),  # Ry(pi/2)
                "prep+i": rx(-math.pi / 2),
            }[name]
            ks = [u] if name == "prep0" else compose([u], A.depol_kraus(pc.p1_q75))
            out = A.on_q75(ks) @ A.op_superop("A1", pc, A.Hypothesis())
        else:
            raise ValueError(f"unknown element {name!r}")
        self._cache[key] = out
        return out

    def is_labelled(self, name: str) -> bool:
        return name in ("meas", "meas0")

    def slot(self, program: Sequence[str], label: int | None = None) -> np.ndarray:
        """Superoperator of a program (time order: first element first)."""
        s = np.eye(16, dtype=complex)
        for e in program:
            s = self.element(e, label if self.is_labelled(e) else None) @ s
        return s

    def step(self, program: Sequence[str], label: int | None = None) -> np.ndarray:
        """``S[y]`` (2, 16, 16) of one step with the slot ``program`` (q59's recorded y)."""
        core = self.slot(program, label) @ self._lead
        p = self.p
        phys = []
        for y in (0, 1):
            s = self.pieces.post[y] @ core
            if p.stark[y]:
                s = A.on_q75([A.rz_unitary(p.stark[y])]) @ s
            phys.append(s)
        f = p.flip
        return np.stack([(1 - f) * phys[0] + f * phys[1], (1 - f) * phys[1] + f * phys[0]])

    @property
    def vec0(self) -> np.ndarray:
        return VEC0[DIAG59].astype(complex)

    @property
    def trace(self) -> np.ndarray:
        return TRACE[DIAG59]

    def bank(self, arm: C.ArmSpec, record: bool = True, full: bool = False) -> np.ndarray:
        """``(programs, symbols, 8, 8)`` in the row-vector convention (``state @ op``) on the
        q59-diagonal subspace (``full``: all 16 dimensions); symbols ``2 m + y`` for a
        recorded labelled element."""
        out = []
        for prog in arm.programs:
            lab = any(self.is_labelled(e) for e in prog)
            if lab and record:
                mats = [self.step(prog, m)[y] for m in (0, 1) for y in (0, 1)]
            else:
                mats = list(self.step(prog))
            mats = [m.T if full else m.T[np.ix_(DIAG59, DIAG59)] for m in mats]
            out.append(np.stack(mats))
        return np.stack(out)

    def nll(self, arm: C.ArmSpec, data: C.ProgramData, choice: np.ndarray | None = None):
        """Per-shot ``-log P`` of an arm's data (``choice`` overrides the data's)."""
        bank = self.bank(arm, data.m is not None)
        sym = data.y.astype(np.int64)
        if data.m is not None:
            sym = 2 * data.m.astype(np.int64) + sym
        ch = data.ch() if choice is None else choice
        return C.bank_nll(self.vec0, bank, self.trace, ch, sym)

    def averaged(self, arm: C.ArmSpec) -> W.LinearProcess:
        """The arm's y-process with the per-step random program averaged (exact for i.i.d.
        choices: the randomised *channel*)."""
        bank = self.bank(arm, record=False)
        if arm.schedule == "random":
            p = np.full(len(bank), 1 / len(bank)) if arm.probs is None else np.asarray(arm.probs)
            ops = np.tensordot(p, bank, 1)
        elif arm.schedule == "constant":
            ops = bank[0]
        else:
            raise ValueError("time-varying arm: use time_varying")
        return W.LinearProcess(self.vec0, ops, self.trace)

    def time_varying(self, arm: C.ArmSpec, length: int, choice=None) -> D.TimeVaryingProcess:
        bank = self.bank(arm, record=False)
        if choice is None:
            if arm.schedule == "random":
                return D.TimeVaryingProcess.constant(self.averaged(arm), length)
            choice = arm.choices(1, length)[0]
        return D.TimeVaryingProcess(self.vec0, bank[np.asarray(choice)], self.trace)


# -- classical starts from the quantum model ----------------------------------------------------


def bloch_maps(qm: QuantumModel, elements: Sequence[str], labelled: Sequence[str] = ()):
    """q75-only maps in Bloch coordinates (column convention, ``qmem.classical``'s
    ``effective_maps``): the slot-to-slot step ``pre @ post[y]`` of the baseline, the initial
    slot state, and each element (``labelled``: (2, 4, 4) by recorded bit)."""
    eff = C.effective_maps(qm.pieces, qm.readout)
    out = {"step": np.stack([eff["pre"] @ eff["post"][y] for y in (0, 1)])}
    out["init"] = eff["pre"] @ eff["r0"]
    pt59 = lambda m4: np.einsum("aibi->ab", m4.reshape(2, 2, 2, 2))  # noqa: E731
    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    k0 = np.diag([1.0, 0.0]).astype(complex)

    def _apply(sup: np.ndarray, rho: np.ndarray) -> np.ndarray:
        return pt59((sup @ vec(np.kron(rho, k0))).reshape(4, 4, order="F"))

    for e in elements:
        out[e] = C._bloch_map(lambda rho, e=e: _apply(qm.element(e), rho))
    for e in labelled:
        out[e] = np.stack(
            [C._bloch_map(lambda rho, e=e, m=m: _apply(qm.element(e, m), rho)) for m in (0, 1)]
        )
    return out


def polytope_start(
    maps: dict[str, np.ndarray],
    u: np.ndarray,
    elements: Sequence[str],
    labelled: Sequence[str] = (),
    eps: float = 0.01,
) -> C.ProgramModel:
    """A classical start carrying the quantum model's structure: hidden states the vertices
    ``u`` (4, d) (columns ``(1, x, y, z)``) of a polytope in q75's Bloch space, each map
    ``T`` taken to ``(pinv(u) T u)^T`` (exact when the polytope is invariant), clipped at 0,
    renormalised and smoothed by ``eps``. ``P = I``; ``Q[y]`` the slot-to-slot step."""
    d = u.shape[1]
    ui = np.linalg.pinv(u)
    row = lambda x: x / x.sum(-1, keepdims=True)  # noqa: E731

    def conv(t: np.ndarray) -> np.ndarray:
        return np.clip((ui @ t @ u).T, 0, None)

    post = np.stack([conv(t) for t in maps["step"]])
    post = post / post.sum(axis=(0, 2))[None, :, None]
    init = np.clip(ui @ maps["init"], 0, None)
    model = C.ProgramModel(
        init / init.sum(),
        np.eye(d),
        post,
        {e: row(conv(maps[e]) + 1e-12) for e in elements},
        {},
    )
    for e in labelled:
        lab = np.stack([conv(t) for t in maps[e]])
        model.labelled[e] = lab / lab.sum(axis=(0, 2))[None, :, None]
    return C.smooth_program(model, eps)


def octahedron(a: float = 1.0, c: float = 1.0) -> np.ndarray:
    """(4, 6): vertices ``(+-a, 0, 0), (0, +-a, 0), (0, 0, +-c)``."""
    v = [(a, 0, 0), (-a, 0, 0), (0, a, 0), (0, -a, 0), (0, 0, c), (0, 0, -c)]
    return np.array([[1.0, *x] for x in v]).T


# -- arms -----------------------------------------------------------------------------------------


def parse_arms(spec: dict[str, Any]) -> dict[str, C.ArmSpec]:
    """Arms from the config: ``{name: {programs = [[..], ..], schedule, probs}}``."""
    out = {}
    for name, s in spec.items():
        progs = tuple(tuple(p) for p in s.get("programs", [["I"]]))
        probs = tuple(s["probs"]) if "probs" in s else None
        out[name] = C.ArmSpec(name, progs, s.get("schedule", "constant"), probs, s.get("whole"))
    return out


def arm_elements(arms: dict[str, C.ArmSpec]) -> tuple[list[str], list[str]]:
    """The free (unlabelled) and labelled element names a classical model of ``arms`` needs."""
    names = sorted(set().union(*(a.elements() for a in arms.values())))
    lab = [n for n in names if n in ("meas", "meas0")]
    return [n for n in names if n not in lab], lab


def first_step_choice(n: int, length: int) -> np.ndarray:
    """Program 0 at step 0, program 1 after (A2's first measurement, a preparation)."""
    ch = np.ones((n, length), dtype=np.int64)
    ch[:, 0] = 0
    return ch


# -- statistics -----------------------------------------------------------------------------------


@dataclass
class StatSettings:
    length: int = 64
    burn: int = 16
    k_max: int = 7
    offsets: tuple[int, ...] = (0, 1, 2, 3)


def tv_stats(tv: D.TimeVaryingProcess, s: StatSettings) -> dict[str, Any]:
    """``g``, ``w(j)`` (common start set), ``P(1)`` after the burn-in and the lag-1 excess."""
    g = tv.expected_windows(s.k_max, s.burn, common=True)
    out: dict[str, Any] = {f"w{j}": float(W.witness(g, j)) for j in s.offsets}
    b = np.real(tv.block_probs(range(s.burn, s.length - 1), 2))
    out.update(
        {
            "g": g.tolist(),
            "p1": float(np.real(tv.marginals()[s.burn :]).mean()),
            "lag1": C.lag1_from_blocks(b),
        }
    )
    return out


def model_arm_stats(model, arm: C.ArmSpec, s: StatSettings) -> dict[str, Any]:
    """Exact statistics of an arm under a quantum or classical program model (random arms:
    the averaged channel; alternating arms: the period-2 process)."""
    if arm.schedule == "alternate":
        return tv_stats(model.time_varying(arm, s.length), s)
    proc = model.averaged(arm)
    return tv_stats(D.TimeVaryingProcess.constant(proc, s.length), s)


def data_stats(y: np.ndarray, s: StatSettings) -> dict[str, Any]:
    """Every statistic of ``tv_stats`` from shots, with whole-shot SEs."""
    from . import artefacts as AR

    per = W.per_shot_windows(y, s.k_max, s.burn, common=True)
    n = len(y)
    g = per.mean(0)
    out: dict[str, Any] = {"n": n, "g": g.tolist()}
    out["g_se"] = (per.std(0, ddof=1) / math.sqrt(n)).tolist()
    idx = W.bootstrap_indices(n, 200, 7)
    r = W.witness_stats(per, list(s.offsets), idx)
    for j in s.offsets:
        out[f"w{j}"] = float(W.witness(g, j))
        out[f"w{j}_se"] = float(r.se[j])
    p1s = y[:, s.burn :].mean(1)
    out["p1"], out["p1_se"] = float(p1s.mean()), float(p1s.std(ddof=1) / math.sqrt(n))
    out["lag1"] = float(AR.lag1_excess_data(y, s.burn))
    lag = np.array([AR.lag1_excess_data(y[i], s.burn) for i in np.array_split(np.arange(n), 20)])
    out["lag1_se"] = float(lag.std(ddof=1) / math.sqrt(len(lag)))
    return out


# -- Knee-style classical-disturbance controls ---------------------------------------------------


def block_distribution(tv: D.TimeVaryingProcess, size: int) -> np.ndarray:
    """``P(y_0 .. y_{size-1})`` (big-endian index) of a short circuit."""
    return np.real(tv.block_probs([0], size))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def knee_disturbances(
    model: QuantumModel, interventions: dict[str, Sequence[str]], sham: Sequence[str], size: int
) -> dict[str, dict[str, float]]:
    """Output disturbance ``D_s(I) = TVD(P_s^I, P_s^sham)`` of each intervention ``I`` for q75
    prepared (in step 0's slot, after a reset) in ``s = 0, 1, +, +i``, over the first ``size``
    outputs. A model whose hidden state after preparing a superposition is a mixture of those
    after preparing ``|0>`` and ``|1>`` obeys ``D_+ <= max(D_0, D_1)`` (convexity of the TVD)
    and ``P_+^sham = mix of P_0^sham, P_1^sham``; the quantum instrument need not."""
    out: dict[str, dict[str, float]] = {}
    probs = {}
    for s in ("0", "1", "+", "+i"):
        for name, prog in [("sham", sham), *interventions.items()]:
            arm = C.ArmSpec(f"{name}_{s}", ((f"prep{s}", *prog), ("I",)), "alternate")
            ch = first_step_choice(1, size)[0]
            tv = model.time_varying(arm, size, ch)
            probs[(name, s)] = block_distribution(tv, size)
    for name in interventions:
        out[name] = {s: tvd(probs[(name, s)], probs[("sham", s)]) for s in ("0", "1", "+", "+i")}
        out[name]["excess_plus"] = out[name]["+"] - max(out[name]["0"], out[name]["1"])
        out[name]["excess_plus_i"] = out[name]["+i"] - max(out[name]["0"], out[name]["1"])
    # the mixture test on the sham: the best mixture of the |0>, |1> preparations
    p0, p1, pp = probs[("sham", "0")], probs[("sham", "1")], probs[("sham", "+")]
    best = min(tvd(lam * p0 + (1 - lam) * p1, pp) for lam in np.linspace(0, 1, 1001))
    out["sham_mixture_gap"] = {"+": best}
    out["_probs"] = {f"{k[0]}|{k[1]}": v.tolist() for k, v in probs.items()}  # type: ignore[dict-item]
    return out


# -- benchmarks on q75 alone ----------------------------------------------------------------------


@dataclass
class Benchmark:
    """q75 alone: prepare ``state``, apply ``reps`` repetitions of an intervention separated by
    ``spacing_s`` of idle, read ``<X>, <Y>, <Z>`` (populations and Ramsey-type probes)."""

    t1: float
    t2: float
    p1: float
    spacing_s: float
    readout: float = 0.0


_PREP = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "+": np.array([1, 1], dtype=complex) / math.sqrt(2),
    "+i": np.array([1, 1j], dtype=complex) / math.sqrt(2),
}


def intervention_kraus(name: str, b: Benchmark, decay: float = 0.0) -> list[np.ndarray]:
    """One intervention on q75 (``eK`` echo, ``R2``/``R4`` randomised, ``A2`` measurement with
    decay and outcome discarded, ``A1`` reset, ``I`` nothing)."""
    if name == "I":
        return [_I2]
    if name.startswith("e"):
        return echo_kraus(echo_phase(int(name[1:])), b.p1)
    if name in ("R2", "R4", "RS"):
        phases = {
            "RS": [0.0],
            "R2": [0.0, math.pi],
            "R4": [0, math.pi / 2, math.pi, 3 * math.pi / 2],
        }
        pulses = compose(echo_kraus(0.0, b.p1), echo_kraus(0.0, b.p1))  # two echoes per step
        return compose(pulses, randomised_channel(phases[name]))
    if name == "A2":
        k0, k1 = A.meas_kraus(decay)
        return [*k0, *k1]
    if name == "A1":
        return A.reset_kraus(0.0, 0.0)
    raise ValueError(name)


def benchmark(
    name: str, state: str, reps: int, b: Benchmark, decay: float = 0.0
) -> dict[str, float]:
    rho = np.outer(_PREP[state], _PREP[state].conj())
    idle = relax_kraus(b.spacing_s, b.t1, b.t2)
    ks = intervention_kraus(name, b, decay)
    for _ in range(reps):
        rho = kraus_channel(ks, rho)
        rho = kraus_channel(idle, rho)
    ev = {k: float(np.real(np.trace(m @ rho))) for k, m in (("x", _X), ("y", _Y), ("z", _Z))}
    ev["p1"] = (1 - ev["z"]) / 2 * (1 - 2 * b.readout) + b.readout
    ev["coherence"] = math.hypot(ev["x"], ev["y"])
    return ev


# -- power -----------------------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def power_one_sided(delta: float, sd: float, n: float, z_alpha: float) -> float:
    """``P(mean of n per-shot differences > z_alpha SE)`` when the true mean is ``delta``."""
    if sd <= 0 or n <= 0:
        return math.nan
    return norm_cdf(delta * math.sqrt(n) / sd - z_alpha)


def shots_for_power(delta: float, sd: float, z_alpha: float, z_beta: float) -> float:
    """Shots so a one-sided test at ``z_alpha`` has power ``Phi(z_beta)`` (``inf``: delta <= 0)."""
    if delta <= 0:
        return math.inf
    return ((z_alpha + z_beta) * sd / delta) ** 2


@dataclass
class DesignCost:
    """Per-shot circuit time of each arm (s), from the baseline and the slot's additions."""

    baseline_s: float
    per_step_added_s: dict[str, float] = field(default_factory=dict)
    length: int = 64

    def duration(self, arm: str) -> float:
        return self.baseline_s + self.length * self.per_step_added_s.get(arm, 0.0)


__all__ = [
    "DIAG59",
    "FITTED",
    "N_PHASES",
    "TRACE",
    "VEC0",
    "Benchmark",
    "DesignCost",
    "NoiseParams",
    "QuantumModel",
    "StatSettings",
    "arm_elements",
    "benchmark",
    "bloch_maps",
    "coherence_factor",
    "compose",
    "data_stats",
    "echo_angles",
    "echo_kraus",
    "echo_phase",
    "echo_unitary",
    "first_step_choice",
    "from_vector",
    "intervention_kraus",
    "knee_disturbances",
    "kraus_channel",
    "model_arm_stats",
    "norm_cdf",
    "octahedron",
    "parse_arms",
    "polytope_start",
    "power_one_sided",
    "randomised_channel",
    "relax_kraus",
    "rx",
    "shots_for_power",
    "to_vector",
    "tv_stats",
    "tvd",
    "x_kraus",
]
