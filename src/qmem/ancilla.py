"""Output-ancilla (q59) reset infidelity, leakage and residual memory (Phase 7 Stage 2B, #21).

Numpy only (no Qiskit), so the main check runs it. The step is read from Phase 6's
``step.qasm`` text with ``qmem.spectator.parse_qasm`` and simulated exactly on a register of
*sites*, each a qubit or a qutrit (level ``|2>`` = leaked):

* the memory q75 (site 0, most significant), and
* one or more output ancillas (q59; for control B3 also q74).

Superoperators act on column-stacked ``vec(rho)`` (``qmem.witness.LinearProcess.instrument``).

**The calibrated model, rebuilt.** With the defaults of ``AncillaParams`` (symmetric reset
flip ``eps``, symmetric readout flip, no leakage) the step is Phase 6's calibrated model:
each gate is applied then depolarised (``rz`` is virtual), ``measure`` projects then records
a flipped bit with the readout error, ``reset`` leaves ``|1>`` with probability ``eps`` from
any input, and q75 relaxes (T1, T2) only while the ancilla is measured or reset. It equals
Stage 2A's pieces (``qmem.ablate_hw``) and #18's 1-qubit instrument (tested).

**How IBM implements ``reset``.** A measurement followed by an X conditioned on its outcome
(Qiskit's ``ResetAfterMeasureSimplification``; IBM's mid-circuit-measurement notes). Its
failure is therefore tied to readout error, and depends on the state it acts on:

* from ``|0>`` it leaves ``|1>`` if the reset's readout says 1 (``e0 ≈ P(1|0)``);
* from ``|1>`` it leaves ``|1>`` if the readout says 0 *without* the qubit having decayed
  (``e1 ≤ P(0|1)``; the decay part of ``P(0|1)`` ends in ``|0>``, which is a success);
* a leaked ``|2>`` is read mostly as 1 by the binary discriminator, and the conditional X
  (resonant with 0-1) leaves it in ``|2>``: reset does not remove leakage.

``e0 != e1`` keeps one classical bit of q59's previous state: the residual memory. This is
Stage 2A's ``ablate_hw.reset_kraus(e0, e1)``, reused here and extended to a qutrit.

**Error models** (``AncillaParams``):

(a) ``reset_e0``, ``reset_e1``: state-dependent reset failure (the previous outcome acts
    through q59's state, which the readout projected onto it).
(b) ``leak``: ``|1> -> |2>`` per readout of the ancilla (applied after the projection,
    recorded outcome unchanged); ``seep``: ``|2> -> |1>`` per step, applied before the
    reset (so a seeped ancilla is then reset normally); ``read2``: P(record 1 | ``|2>``);
    ``leak_cz_phase``: the phase a CZ gives ``|1>_q75 |2>_q59`` (and ``|2>_q75 |1>_q59``).
(c) ``flip01``, ``flip10``: non-QND readout, the post-measurement state flipped after the
    record (measurement-induced excitation / relaxation).
(d) ``read10`` = P(record 1 | ``|0>``), ``read01`` = P(record 0 | ``|1>``): assignment
    asymmetry of the step's readout. Only its effect on what q59's physical state does next
    belongs here (it enters through (a)); outcome-dependent assignment that does not act
    through the state is Stage 2D's (#23).

**Fact (tested).** Because q59 is always reset before reuse, anything that changes q59's
state between its readout and its reset (model (c), a T1 decay) enters the output process
only through the reset's state dependence: (c) is degenerate with (a).

**Memory.** Without leakage the persistent memory is q75 ⊗ (q59's 1-bit residue); with it,
q75 ⊗ (q59 in {0, 1, 2}). After every step q59 is diagonal, so the process is restricted
exactly to the q59-diagonal subspace (``compress``): dimension 4 × 3 = 12 for one qutrit
ancilla instead of 36.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import ablate_hw as A
from . import witness as W
from .spectator import Instr, parse_qasm

# Native durations (s) of the gates; measure and reset come from the calibration snapshot.
GATE_TIME = {"sx": 32e-9, "x": 32e-9, "cz": 68e-9, "rz": 0.0}
_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1.0, -1.0]).astype(complex),
]


# -- register algebra ---------------------------------------------------------------------------


def embed(op: np.ndarray, sites: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """``op`` on ``sites`` (big-endian within ``op``) as a matrix on the whole register."""
    n = len(dims)
    sub = [dims[s] for s in sites]
    rest = [s for s in range(n) if s not in sites]
    t = np.asarray(op, dtype=complex).reshape(sub + sub)
    eye = np.eye(int(np.prod([dims[s] for s in rest])) if rest else 1, dtype=complex)
    full = np.multiply.outer(t, eye.reshape([dims[s] for s in rest] * 2))
    k = len(sites)
    # axes now: out(sites), in(sites), out(rest), in(rest) -> out(all), in(all) in site order
    order_out = list(sites) + rest
    order_in = list(sites) + rest
    src_out = list(range(k)) + [2 * k + i for i in range(len(rest))]
    src_in = [k + i for i in range(k)] + [2 * k + len(rest) + i for i in range(len(rest))]
    perm = [0] * (2 * n)
    for pos, s in enumerate(order_out):
        perm[s] = src_out[pos]
    for pos, s in enumerate(order_in):
        perm[n + s] = src_in[pos]
    d = int(np.prod(dims))
    return full.transpose(perm).reshape(d, d)


def lift_qubit(op: np.ndarray, dim: int) -> np.ndarray:
    """A qubit operator on the {0, 1} block of a ``dim``-level site (zero on ``|2>``)."""
    out = np.zeros((dim, dim), dtype=complex)
    out[:2, :2] = op
    return out


def lift_unitary(u: np.ndarray, dim: int) -> np.ndarray:
    """A qubit unitary on {0, 1}, identity on ``|2>`` (an off-resonant leaked level)."""
    out = np.eye(dim, dtype=complex)
    out[:2, :2] = u
    return out


def superop(kraus: Sequence[np.ndarray]) -> np.ndarray:
    return A.superop(kraus)


def qubit_block_kraus(kraus: Sequence[np.ndarray], weights_rest: Sequence[float], dims, sites):
    """Kraus of a qubit channel on ``sites`` that acts as identity outside the qubit block.

    ``kraus[i]`` is a (2^k x 2^k) Kraus operator on the qubit subspace of ``sites``; on the
    complement (any site leaked) it is ``sqrt(weights_rest[i]) * I`` (weights sum to 1), so the
    lifted set is trace preserving.
    """
    sub = [dims[s] for s in sites]
    dsub = int(np.prod(sub))
    # indices of the qubit block within the sub-register
    idx = [i for i in range(dsub) if all(d < 2 for d in np.unravel_index(i, sub))]
    qb = np.zeros((dsub, len(idx)))
    for c, i in enumerate(idx):
        qb[i, c] = 1
    rest = np.eye(dsub) - qb @ qb.T
    out = []
    for k, w in zip(kraus, weights_rest, strict=True):
        m = qb @ k @ qb.T + math.sqrt(w) * rest
        out.append(embed(m, sites, dims))
    return out


def depol_kraus(p: float, n_qubits: int) -> tuple[list[np.ndarray], list[float]]:
    """``rho -> (1 - p) rho + p I/d`` (Pauli Kraus; ``qmem.hardware.depolarizing_kraus``)."""
    d2 = 4**n_qubits
    ks, ws = [], []
    for idx in range(d2):
        op = np.eye(1, dtype=complex)
        for k in range(n_qubits):
            op = np.kron(op, _PAULI[(idx // 4 ** (n_qubits - 1 - k)) % 4])
        w = 1 - p + p / d2 if idx == 0 else p / d2
        if w > 0:
            ks.append(math.sqrt(w) * op)
            ws.append(w)
    return ks, ws


def relaxation_kraus(duration: float, t1: float, t2: float) -> list[np.ndarray]:
    """Amplitude damping then pure dephasing (``qmem.hardware.relaxation_kraus``, numpy)."""
    if duration <= 0:
        return [np.eye(2, dtype=complex)]
    t2 = min(t2, 2 * t1)
    gamma = 1 - math.exp(-duration / t1)
    lam_total = math.exp(-duration / t2)
    lam_ad = math.sqrt(1 - gamma)
    lam_phi = min(1.0, lam_total / lam_ad) if lam_ad > 0 else 0.0
    ad = [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex),
    ]
    q = (1 - lam_phi) / 2
    ph = [math.sqrt(1 - q) * _PAULI[0], math.sqrt(q) * _PAULI[3]]
    return [b @ a for a in ad for b in ph]


# -- parameters -----------------------------------------------------------------------------


@dataclass
class SiteNoise:
    """Calibration of one physical qubit (snapshot or FakeKingston)."""

    p1: float = 0.0  # depolarising after sx / x
    readout: float = 0.0  # symmetric flip of the calibrated model
    reset: float = 0.0  # symmetric reset flip of the calibrated model
    t1: float | None = None
    t2: float | None = None


@dataclass
class AncillaParams:
    """Deviations of one output ancilla from the calibrated model (``None``: calibrated).

    See the module docstring for (a)–(d). Every default reproduces the calibrated model.
    """

    reset_e0: float | None = None
    reset_e1: float | None = None
    read10: float | None = None
    read01: float | None = None
    leak: float = 0.0
    seep: float = 0.0
    read2: float = 1.0
    leak_cz_phase: float = 0.0
    flip01: float = 0.0
    flip10: float = 0.0
    never_reset: bool = False  # the "eps = 1" limit: the reset does nothing


@dataclass
class StepConfig:
    """What the simulator needs besides the circuit."""

    noise: dict[int, SiteNoise]  # by physical qubit
    p2: dict[tuple[int, int], float]  # CZ depolarising per (unordered) physical pair
    measure_time: float
    reset_time: float
    memory: int = 75
    ancilla: dict[int, AncillaParams] = field(default_factory=dict)  # by physical qubit
    memory_levels: int = 2  # 3: q75 may leak (the A2 extension)
    memory_leak: float = 0.0  # |1> -> |2> of q75 per measurement of q75 (A2)
    memory_seep: float = 0.0  # |2> -> |1> of q75 per step
    memory_read2: float = 1.0
    memory_cz_phase: float = 0.0  # CZ phase of |2>_q75 |1>_ancilla
    relax_idle_ancillas: bool = False  # True: ancillas also relax while others are busy (B3)
    detuning: float = 0.0  # extra Z rotation of q75 per step (rad), a non-ancilla comparator

    def params(self, q: int) -> AncillaParams:
        return self.ancilla.get(q, AncillaParams())

    def p2_of(self, a: int, b: int) -> float:
        return self.p2.get((a, b), self.p2.get((b, a), 0.0))


def config_from_snapshot(noise: dict[str, Any], physical: Sequence[int] = (59, 75)) -> StepConfig:
    """``StepConfig`` from Phase 6's ``summary.json`` → ``prepare.calibration.noise``.

    The noise dict is keyed by *compact* index (0 = output q59, 1 = memory q75).
    """

    def d(key):
        return {int(k): v for k, v in noise[key]}

    p1, ro, rs, t1, t2 = d("p1"), d("readout"), d("reset"), d("t1"), d("t2")
    sites = {
        physical[k]: SiteNoise(p1[k], ro[k], rs[k], t1[k], t2[k]) for k in range(len(physical))
    }
    p2 = {}
    for (a, b), v in noise["p2"]:
        p2[(physical[a], physical[b])] = v
    return StepConfig(sites, p2, noise["measure_time"], noise["reset_time"], memory=physical[1])


def noiseless_config(physical: Sequence[int] = (59, 75)) -> StepConfig:
    return StepConfig({q: SiteNoise() for q in physical}, {}, 0.0, 0.0, memory=physical[1])


# -- the step simulator -----------------------------------------------------------------------


def _rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_XM = _PAULI[1]


@dataclass
class Register:
    """Physical qubits ``qubits`` (site order) with their level counts."""

    qubits: tuple[int, ...]
    dims: tuple[int, ...]

    @property
    def dim(self) -> int:
        return int(np.prod(self.dims))

    def site(self, q: int) -> int:
        return self.qubits.index(q)


class _Branches:
    """Superoperators per recorded outcome tuple (one entry per measurement so far)."""

    def __init__(self, d: int):
        self.maps: dict[tuple[int, ...], np.ndarray] = {(): np.eye(d * d, dtype=complex)}

    def apply(self, s: np.ndarray) -> None:
        self.maps = {k: s @ v for k, v in self.maps.items()}


def _projector(dim: int, b: int) -> np.ndarray:
    p = np.zeros((dim, dim), dtype=complex)
    p[b, b] = 1
    return p


def _relax_site(reg: Register, site: int, cfg_noise: SiteNoise, duration: float):
    if cfg_noise.t1 is None or duration <= 0:
        return None
    ks = relaxation_kraus(duration, cfg_noise.t1, cfg_noise.t2 or 2 * cfg_noise.t1)
    dims = reg.dims
    return superop(
        qubit_block_kraus(ks, [abs(np.trace(k.conj().T @ k)) / 2 for k in ks], dims, [site])
    )


def _idle_others(reg: Register, cfg: StepConfig, busy: int, duration: float, which: str):
    """Relaxation of the idling qubits (the calibrated model relaxes only the memory)."""
    s = np.eye(reg.dim**2, dtype=complex)
    for q in reg.qubits:
        if q == busy:
            continue
        if q != cfg.memory and which == "memory_only" and not cfg.relax_idle_ancillas:
            continue
        r = _relax_site(reg, reg.site(q), cfg.noise[q], duration)
        if r is not None:
            s = r @ s
    return s


def reset_channel(reg: Register, q: int, cfg: StepConfig) -> np.ndarray:
    """Seepage, then the measure-and-conditional-X reset of ancilla ``q`` (superoperator)."""
    site, dims = reg.site(q), reg.dims
    p = cfg.params(q)
    dq = dims[site]
    s = np.eye(reg.dim**2, dtype=complex)
    if dq == 3 and p.seep > 0:  # |2> -> |1> before the reset
        k_seep = [
            np.diag([1, 1, math.sqrt(1 - p.seep)]).astype(complex),
            math.sqrt(p.seep) * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex),
        ]
        s = superop([embed(k, [site], dims) for k in k_seep]) @ s
    if p.never_reset:
        return s
    eps = cfg.noise[q].reset
    e0 = eps if p.reset_e0 is None else p.reset_e0
    e1 = eps if p.reset_e1 is None else p.reset_e1
    ks = [lift_qubit(k, dq) for k in A.reset_kraus(e0, e1)]
    if dq == 3:
        ks.append(_projector(3, 2))  # the conditional X leaves |2> where it is
    return superop([embed(k, [site], dims) for k in ks]) @ s


def _measure(
    reg: Register,
    q: int,
    cfg: StepConfig,
    assign: np.ndarray,
    leak: float,
    flips: tuple[float, float],
) -> list[np.ndarray]:
    """Superoperators per recorded bit of a measurement of ``q``.

    ``assign[y, b]`` = P(record y | level b). After the projection on ``|b>``: non-QND flips
    (``flips = (0->1, 1->0)``) and leakage ``|1> -> |2>`` with probability ``leak``.
    """
    site, dims = reg.site(q), reg.dims
    dq = dims[site]
    out = [np.zeros((reg.dim**2,) * 2, dtype=complex) for _ in range(2)]
    f01, f10 = flips
    for b in range(dq):
        proj = superop([embed(_projector(dq, b), [site], dims)])
        # post-measurement transitions from |b>
        trans: list[tuple[int, float]] = [(b, 1.0)]
        if b == 0 and f01 > 0:
            trans = [(0, 1 - f01), (1, f01)]
        if b == 1:
            stay = 1 - f10 - (leak if dq == 3 else 0.0)
            trans = [(1, stay)]
            if f10 > 0:
                trans.append((0, f10))
            if dq == 3 and leak > 0:
                trans.append((2, leak))
        post = np.zeros_like(proj)
        for b2, w in trans:
            if w <= 0:
                continue
            k = np.zeros((dq, dq), dtype=complex)
            k[b2, b] = math.sqrt(w)
            post = post + superop([embed(k, [site], dims)])
        post = post @ proj
        for y in range(2):
            if assign[y, b] > 0:
                out[y] = out[y] + assign[y, b] * post
    return out


def _assign(cfg: StepConfig, q: int, dq: int, memory: bool = False) -> np.ndarray:
    a = np.zeros((2, dq))
    if memory:
        r = cfg.noise[q].readout
        a[:, 0], a[:, 1] = (1 - r, r), (r, 1 - r)
        if dq == 3:
            a[:, 2] = (1 - cfg.memory_read2, cfg.memory_read2)
        return a
    p = cfg.params(q)
    r = cfg.noise[q].readout
    r10 = r if p.read10 is None else p.read10
    r01 = r if p.read01 is None else p.read01
    a[:, 0] = (1 - r10, r10)
    a[:, 1] = (r01, 1 - r01)
    if dq == 3:
        a[:, 2] = (1 - p.read2, p.read2)
    return a


def _gate_superop(reg: Register, ins: Instr, cfg: StepConfig) -> np.ndarray:
    dims = reg.dims
    sites = [reg.site(q) for q in ins.qubits]
    if ins.name in ("rz", "sx", "x"):
        q = ins.qubits[0]
        u = {"rz": lambda: _rz(ins.param), "sx": lambda: _SX, "x": lambda: _XM}[ins.name]()
        s = superop([embed(lift_unitary(u, dims[sites[0]]), sites, dims)])
        if ins.name != "rz" and cfg.noise[q].p1 > 0:
            ks, ws = depol_kraus(cfg.noise[q].p1, 1)
            s = superop(qubit_block_kraus(ks, ws, dims, sites)) @ s
        return s
    if ins.name == "cz":
        a, b = sites
        da, db = dims[a], dims[b]
        diag = np.ones(da * db, dtype=complex)
        for i in range(da):
            for j in range(db):
                if i == 1 and j == 1:
                    diag[i * db + j] = -1
                elif (i, j) in ((1, 2), (2, 1)):
                    leaked = ins.qubits[1] if j == 2 else ins.qubits[0]
                    phi = (
                        cfg.memory_cz_phase
                        if leaked == cfg.memory
                        else cfg.params(leaked).leak_cz_phase
                    )
                    diag[i * db + j] = np.exp(1j * phi)
        s = superop([embed(np.diag(diag), sites, dims)])
        p2 = cfg.p2_of(*ins.qubits)
        if p2 > 0:
            ks, ws = depol_kraus(p2, 2)
            s = superop(qubit_block_kraus(ks, ws, dims, sites)) @ s
        return s
    raise ValueError(f"unsupported gate {ins.name!r}")


def step_superops(
    instrs: Sequence[Instr],
    reg: Register,
    cfg: StepConfig,
    record: Sequence[int] | None = None,
) -> np.ndarray:
    """Superoperators ``S[outcomes]`` of one step on the register.

    ``instrs`` use physical qubit labels. Every ``measure`` of an ancilla appends an outcome;
    ``record`` lists which measurements (by order) form the returned outcome index
    (default: all). The others are summed over (discarded). Returns ``(2**len(record), D², D²)``
    with the outcome index big-endian in ``record`` order. The memory's measurements (A2)
    use the memory's leakage parameters.
    """
    br = _Branches(reg.dim)
    if cfg.detuning:
        br.apply(superop([embed(lift_unitary(_rz(cfg.detuning), reg.dims[0]), [0], reg.dims)]))
    n_meas = 0
    for ins in instrs:
        if ins.name == "barrier":
            continue
        if ins.name == "delay":  # added idle time (B4): the memory relaxes for it
            if ins.param:
                br.apply(delay_superop(reg, cfg, ins.param))
            continue
        if ins.name == "reset":
            q = ins.qubits[0]
            s = _idle_others(reg, cfg, q, cfg.reset_time, "memory_only") @ reset_channel(
                reg, q, cfg
            )
            br.apply(s)
            continue
        if ins.name == "measure":
            q = ins.qubits[0]
            site = reg.site(q)
            dq = reg.dims[site]
            if q == cfg.memory:
                ms = _measure(
                    reg, q, cfg, _assign(cfg, q, dq, memory=True), cfg.memory_leak, (0.0, 0.0)
                )
            else:
                p = cfg.params(q)
                ms = _measure(reg, q, cfg, _assign(cfg, q, dq), p.leak, (p.flip01, p.flip10))
            idle = _idle_others(reg, cfg, q, cfg.measure_time, "memory_only")
            ms = [idle @ m for m in ms]
            new = {}
            for k, v in br.maps.items():
                for y in range(2):
                    new[(*k, y)] = ms[y] @ v
            br.maps = new
            n_meas += 1
            continue
        br.apply(_gate_superop(reg, ins, cfg))
    if cfg.memory_levels == 3 and cfg.memory_seep > 0:
        site = reg.site(cfg.memory)
        k_seep = [
            np.diag([1, 1, math.sqrt(1 - cfg.memory_seep)]).astype(complex),
            math.sqrt(cfg.memory_seep) * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex),
        ]
        br.apply(superop([embed(k, [site], reg.dims) for k in k_seep]))
    record = list(range(n_meas)) if record is None else list(record)
    out = np.zeros((2 ** len(record), reg.dim**2, reg.dim**2), dtype=complex)
    for k, v in br.maps.items():
        idx = 0
        for r in record:
            idx = 2 * idx + k[r]
        out[idx] += v
    return out


def delay_superop(reg: Register, cfg: StepConfig, duration: float) -> np.ndarray:
    """Relaxation of the memory (and nothing else) for an added idle ``duration`` (B4)."""
    r = _relax_site(reg, 0, cfg.noise[cfg.memory], duration)
    return np.eye(reg.dim**2, dtype=complex) if r is None else r


# -- circuits ---------------------------------------------------------------------------------


def physical_step(step_qasm: str, output: int = 59, memory: int = 75) -> list[Instr]:
    """``step.qasm`` (q[0] = output, q[1] = memory) on physical labels."""
    from .spectator import relabel

    return relabel(parse_qasm(step_qasm), {0: output, 1: memory})


def relabel_output(instrs: Sequence[Instr], old: int, new: int) -> list[Instr]:
    return [replace(i, qubits=tuple(new if q == old else q for q in i.qubits)) for i in instrs]


def variant_step(base: Sequence[Instr], variant: str, output: int = 59) -> list[Instr]:
    """The step of control ``variant`` (B1: ``reset; reset``; B2: ``reset; measure``)."""
    out: list[Instr] = []
    for i in base:
        out.append(i)
        if i.name == "reset" and i.qubits == (output,):
            if variant == "B1":
                out.append(Instr("reset", (output,)))
            elif variant == "B2":
                out.append(Instr("measure", (output,)))
    return out


# -- processes --------------------------------------------------------------------------------


def register_for(cfg: StepConfig, outputs: Sequence[int], output_levels: int = 3) -> Register:
    return Register((cfg.memory, *outputs), (cfg.memory_levels,) + (output_levels,) * len(outputs))


def diag_subspace(reg: Register) -> np.ndarray:
    """vec indices with every ancilla (sites 1..) diagonal (the invariant subspace)."""
    d = reg.dim
    keep = []
    for j in range(d):
        bj = np.unravel_index(j, reg.dims)
        for i in range(d):
            bi = np.unravel_index(i, reg.dims)
            if all(bi[s] == bj[s] for s in range(1, len(reg.dims))):
                keep.append(i + d * j)
    return np.array(keep)


def rho0(reg: Register) -> np.ndarray:
    r = np.zeros((reg.dim, reg.dim), dtype=complex)
    r[0, 0] = 1
    return r


def to_process(sup: np.ndarray, reg: Register, compress: bool = True) -> W.LinearProcess:
    """``LinearProcess`` of a homogeneous step, restricted to the ancilla-diagonal subspace."""
    p = W.LinearProcess.instrument(sup, rho0(reg))
    if not compress:
        return p
    k = diag_subspace(reg)
    return W.LinearProcess(p.init[k], p.ops[:, k][:, :, k], p.final[k])


def compress_ops(sup: np.ndarray, reg: Register) -> np.ndarray:
    """Row-convention ops (``state @ ops``) restricted to the diagonal subspace."""
    k = diag_subspace(reg)
    ops = np.transpose(sup, (0, 2, 1))
    return ops[:, k][:, :, k]


def init_final(reg: Register) -> tuple[np.ndarray, np.ndarray]:
    k = diag_subspace(reg)
    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    return vec(rho0(reg))[k], vec(np.eye(reg.dim, dtype=complex))[k]


def ancilla_process(
    instrs: Sequence[Instr], cfg: StepConfig, outputs: Sequence[int] = (59,), levels: int = 3
) -> W.LinearProcess:
    reg = register_for(cfg, outputs, levels)
    return to_process(step_superops(instrs, reg, cfg), reg)


# -- time-inhomogeneous / post-selected statistics --------------------------------------------


@dataclass
class SeqProcess:
    """``P(y_0..y_{L-1}, accept) = init @ ops[t][y_0] @ ... @ final`` with per-step ops.

    ``ops[t]`` is ``(2, D, D)`` in the row-vector convention; sub-normalised ops (a flag
    outcome kept only if accepted) give post-selected statistics. All statistics are
    conditioned on acceptance (divided by its probability).
    """

    init: np.ndarray
    ops: list[np.ndarray]
    final: np.ndarray

    @property
    def length(self) -> int:
        return len(self.ops)

    @classmethod
    def homogeneous(cls, p: W.LinearProcess, length: int) -> SeqProcess:
        return cls(p.init, [p.ops] * length, p.final)

    def _prefix_suffix(self):
        pre = [self.init]
        for o in self.ops:
            pre.append(pre[-1] @ o.sum(0))
        suf = [self.final]
        for o in reversed(self.ops):
            suf.append(o.sum(0) @ suf[-1])
        suf = suf[::-1]
        return pre, suf

    def accept_prob(self) -> float:
        _, suf = self._prefix_suffix()
        return float((self.init @ suf[0]).real)

    def marginals(self) -> np.ndarray:
        pre, suf = self._prefix_suffix()
        z = (self.init @ suf[0]).real
        return np.array(
            [(pre[t] @ self.ops[t][1] @ suf[t + 1]).real / z for t in range(self.length)]
        )

    def windows(self, k_max: int, burn: int) -> np.ndarray:
        """Expected ``per_shot_windows(..., common=True).mean(0)`` (given acceptance)."""
        pre, suf = self._prefix_suffix()
        z = (self.init @ suf[0]).real
        last = self.length - k_max - 2
        out = np.zeros(k_max + 1)
        for t in range(burn, last + 1):
            cur = pre[t] @ self.ops[t][1]
            for k in range(k_max + 1):
                u = t + k + 1
                out[k] += (cur @ self.ops[u][1] @ suf[u + 1]).real
                cur = cur @ self.ops[u][0]
        return out / (z * (last - burn + 1))

    def block_probs(self, size: int, burn: int) -> np.ndarray:
        pre, suf = self._prefix_suffix()
        z = (self.init @ suf[0]).real
        starts = list(range(burn, self.length - size + 1, size))
        out = np.zeros(2**size)
        for t0 in starts:
            st = pre[t0][None, :]
            for i in range(size):
                o = self.ops[t0 + i]
                st = np.stack([st @ o[0], st @ o[1]], axis=1).reshape(-1, st.shape[-1])
            out += (st @ suf[t0 + size]).real
        return out / (z * len(starts))


def predict_seq(
    sp: SeqProcess,
    k_max: int = 7,
    burn: int = 16,
    block: int = 4,
    offsets: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    g = sp.windows(k_max, burn)
    out: dict[str, Any] = {"g": g.tolist()}
    for j in offsets:
        out[f"w{j}"] = float(W.witness(g, j))
    out["p1"] = float(sp.marginals().mean())
    out["blocks"] = sp.block_probs(block, burn).tolist()
    out["accept"] = sp.accept_prob()
    return out


def kl_blocks(p: Sequence[float], q: Sequence[float], size: int) -> float:
    p, q = np.asarray(p), np.asarray(q)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])) / size)


# -- fitting ----------------------------------------------------------------------------------


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: float = 0.5,
    max_evals: int = 400,
    xtol: float = 1e-4,
    ftol: float = 1e-6,
) -> tuple[np.ndarray, float, int]:
    """Minimal Nelder–Mead (numpy; scipy is not a dependency). Returns (x, f(x), evals)."""
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if (
            max(np.max(np.abs(p - pts[0])) for p in pts[1:]) < xtol
            and abs(vals[-1] - vals[0]) < ftol
        ):
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1])
            fe = f(xe)
            evals += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen) if fr >= vals[-1] else cen + 0.5 * (xr - cen)
            fc = f(xc)
            evals += 1
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
                    evals += 1
    i = int(np.argmin(vals))
    return pts[i], vals[i], evals


@dataclass
class FitSpec:
    """A model family: which parameters are free (in logit space, all probabilities)."""

    name: str
    free: tuple[str, ...]
    fixed: dict[str, float] = field(default_factory=dict)


def apply_params(cfg: StepConfig, values: dict[str, float], output: int = 59) -> StepConfig:
    """``cfg`` with ancilla parameters (and ``detuning``) set from ``values``."""
    anc = dict(cfg.ancilla)
    ap = replace(anc.get(output, AncillaParams()))
    det = cfg.detuning
    for k, v in values.items():
        if k == "detuning":
            det = v
        else:
            setattr(ap, k, v)
    anc[output] = ap
    return replace(cfg, ancilla=anc, detuning=det)


def _decode(spec: FitSpec, x: np.ndarray) -> dict[str, float]:
    out = dict(spec.fixed)
    for k, v in zip(spec.free, x, strict=True):
        out[k] = float(v) if k in ("detuning", "leak_cz_phase") else _sigmoid(float(v))
    return out


def _encode(spec: FitSpec, init: dict[str, float]) -> np.ndarray:
    return np.array(
        [init[k] if k in ("detuning", "leak_cz_phase") else _logit(init[k]) for k in spec.free]
    )


def fit_model(
    spec: FitSpec,
    instrs: Sequence[Instr],
    cfg: StepConfig,
    train: np.ndarray,
    init: dict[str, float],
    max_evals: int = 400,
) -> dict[str, Any]:
    """Maximum-likelihood fit of ``spec``'s free parameters on ``train`` shots."""

    def model(values):
        c = apply_params(cfg, values)
        return ancilla_process(instrs, c)

    def nll(x):
        try:
            return float(model(_decode(spec, x)).seq_nll(train).sum() / train.size)
        except (FloatingPointError, ValueError):
            return math.inf

    if not spec.free:
        v = dict(spec.fixed)
        return {
            "values": v,
            "train_nll": float(model(v).seq_nll(train).sum() / train.size),
            "evals": 1,
            "se": {},
        }
    x0 = _encode(spec, init)
    x, fx, ev = nelder_mead(nll, x0, max_evals=max_evals)
    values = _decode(spec, x)
    se = wald_se(spec, x, nll, train.size)
    return {"values": values, "train_nll": fx, "evals": ev, "se": se}


def wald_se(spec: FitSpec, x: np.ndarray, nll: Callable, n_sym: int, h: float = 0.05) -> dict:
    """SEs from the numerical Hessian of the total NLL in the fit coordinates (delta method
    back to probabilities). Returns ``{param: (se, lo95, hi95)}``; ``nan`` if not positive."""
    n = len(x)
    f0 = nll(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            v = (nll(x + ei + ej) - nll(x + ei - ej) - nll(x - ei + ej) + nll(x - ei - ej)) / (
                4 * h * h
            )
            hess[i, j] = hess[j, i] = v * n_sym
    del f0
    out: dict[str, tuple[float, float, float]] = {}
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)
    for i, k in enumerate(spec.free):
        var = cov[i, i]
        s = math.sqrt(var) if var > 0 else math.nan
        if k in ("detuning", "leak_cz_phase"):
            out[k] = (s, x[i] - 1.96 * s, x[i] + 1.96 * s)
        else:
            p = _sigmoid(x[i])
            lo = _sigmoid(x[i] - 1.96 * s) if s == s else math.nan
            hi = _sigmoid(x[i] + 1.96 * s) if s == s else math.nan
            out[k] = (s * p * (1 - p), lo, hi)
    return out


# -- splits -----------------------------------------------------------------------------------


def clean_indices(n: int, episode: tuple[int, int]) -> np.ndarray:
    a, b = episode
    idx = np.arange(n)
    return idx[(idx < a) | (idx >= b)]


def split(idx: np.ndarray, kind: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``interleaved`` (even/odd positions) or ``random`` halves of ``idx``."""
    if kind == "interleaved":
        return idx[0::2], idx[1::2]
    if kind == "random":
        p = np.random.default_rng(seed).permutation(len(idx))
        h = len(idx) // 2
        return np.sort(idx[p[:h]]), np.sort(idx[p[h:]])
    raise ValueError(kind)


__all__ = [
    "AncillaParams",
    "FitSpec",
    "Register",
    "SeqProcess",
    "SiteNoise",
    "StepConfig",
    "ancilla_process",
    "apply_params",
    "clean_indices",
    "compress_ops",
    "config_from_snapshot",
    "delay_superop",
    "embed",
    "fit_model",
    "init_final",
    "kl_blocks",
    "nelder_mead",
    "noiseless_config",
    "physical_step",
    "predict_seq",
    "register_for",
    "relaxation_kraus",
    "reset_channel",
    "split",
    "step_superops",
    "to_process",
    "variant_step",
]
