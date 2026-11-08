"""Spectator, coupler and crosstalk models of the Phase 6 step (Phase 7 Stage 2C, #22).

Numpy only (no Qiskit), so the main check runs it. The step circuit is read from its QASM
text (``results/summaries/phase6/step.qasm``) by a small parser of the native gates
(``rz``, ``sx``, ``x``, ``cz``, ``reset``, ``measure``, ``delay``, ``barrier``).

**The simulator.** Exact density matrices of the core pair (output, memory) plus up to two
*spectators*, scheduled in time. With every coupling set to zero it reproduces the Phase 6
calibrated model (``qmem.hardware.StepNoise`` / ``noisy_instrument``) exactly:

* each gate is applied at its start time, then depolarised (``p1``/``p2``; ``rz`` is virtual);
* ``measure`` projects, then records the outcome with a symmetric flip ``readout``;
* ``reset`` leaves ``|1>`` with probability ``reset``;
* a core qubit relaxes (``T1``, ``T2``, zero temperature) only while the *other* core qubit
  is being measured or reset, as in Phase 6's model. The relaxation is applied in time
  slices, so pulses inside the window (dynamical decoupling) are placed correctly.

On top of that, between consecutive event times, every slice applies:

* static ZZ ``H = 2 pi sum_ab zeta_ab n_a n_b`` between any pair (``zz_hz``, in Hz; ``zeta``
  is the shift of one qubit's frequency when the other is in ``|1>``);
* static detunings ``2 pi delta_q n_q`` (``detuning_hz``);
* during a measurement or reset of the output: an AC Stark shift of the memory that may
  depend on the output's state (``stark_hz[y]``), and extra pure dephasing of the memory
  (``meas_dephasing_rate``, 1/s);
* spectator relaxation towards its equilibrium excited population ``p_eq`` (generalised
  amplitude damping with ``T1``; a two-level fluctuator is a spectator with a long ``T1``),
  plus spectator dephasing ``T2``;
* extra amplitude damping of the memory while a spectator is in ``|1>`` (``loss_rate``, 1/s):
  a TLS that, when switched on, shortens the memory's ``T1``.

**The instrument.** The output qubit is reset at the start of every step, so the persistent
memory is everything else: the memory qubit and the spectators. ``step_superops`` returns
``S[y]`` on column-stacked ``vec(rho)`` of that register, the convention of
``qmem.witness.LinearProcess.instrument``. Its dimension is ``2^(1 + n_spectators)``, so a
spectator can carry memory: it is part of the *effective memory of the hardware*, not memory
attributable to q75. Spectators start (and stay) diagonal, so in this model they act as
classical hidden variables that condition the memory's frequency or ``T1``.

Simulator qubit order: index 0 is the output, 1 the memory, 2.. the spectators (the order of
``step.qasm``'s ``q[0]``, ``q[1]``); matrices are big-endian (qubit 0 most significant).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import witness as W

# Phase 6 / ibm_kingston native durations (s); measure and reset come from the snapshot.
GATE_TIME = {"sx": 32e-9, "x": 32e-9, "cz": 68e-9, "rz": 0.0}


# -- circuits ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Instr:
    """One instruction on circuit labels (``q[i]`` -> ``i``, ``$N`` -> ``N``)."""

    name: str
    qubits: tuple[int, ...]
    param: float | None = None  # rz angle, or delay duration (s)
    in_window: bool = False  # may run during a measure/reset of another qubit (DD pulses)


_LINE = re.compile(
    r"^(?:(?P<lhs>\w+)\[\d+\]\s*=\s*)?(?P<name>\w+)(?:\[(?P<dur>[^\]]+)\])?"
    r"(?:\((?P<arg>[^)]*)\))?\s+(?P<qargs>[^;]+);$"
)
_QARG = re.compile(r"(?:\w+\[(\d+)\]|\$(\d+))")


def _duration(s: str) -> float:
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*(ns|us|µs|ms|s|dt)\s*", s)
    if not m:
        raise ValueError(f"cannot read duration {s!r}")
    scale = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}[m.group(2)]
    return float(m.group(1)) * scale


def parse_qasm(text: str) -> list[Instr]:
    """Instructions of a straight-line OpenQASM 3 circuit of native gates.

    Accepts the declarations, ``rz(theta)``, ``sx``, ``x``, ``cz``, ``reset``,
    ``b[0] = measure q``, ``delay[100ns]`` and ``barrier``. Control flow is refused.
    """
    out: list[Instr] = []
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "bit", "qubit", "creg", "qreg")):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"unsupported QASM line: {raw!r}")
        name = m.group("name")
        qs = tuple(int(a or b) for a, b in _QARG.findall(m.group("qargs")))
        if name == "rz":
            out.append(Instr("rz", qs, float(m.group("arg"))))
        elif name in ("sx", "x", "cz", "reset", "measure", "barrier"):
            out.append(Instr(name, qs))
        elif name == "delay":
            out.append(Instr("delay", qs, _duration(m.group("dur"))))
        else:
            raise ValueError(f"unsupported instruction {name!r} in {raw!r}")
    return out


def relabel(instrs: Sequence[Instr], mapping: dict[int, int]) -> list[Instr]:
    return [replace(i, qubits=tuple(mapping.get(q, q) for q in i.qubits)) for i in instrs]


def to_qasm(instrs: Sequence[Instr], n_bits: int = 1, physical: bool = True) -> str:
    """OpenQASM 3 text (``$N`` physical qubits; measurement ``t`` goes to ``y<t>``)."""
    q = (lambda i: f"${i}") if physical else (lambda i: f"q[{i}]")
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    lines += [f"bit[1] y{t};" for t in range(n_bits)]
    if not physical:
        n = 1 + max(x for i in instrs for x in i.qubits)
        lines.append(f"qubit[{n}] q;")
    t = 0
    for i in instrs:
        args = ", ".join(q(x) for x in i.qubits)
        if i.name == "rz":
            lines.append(f"rz({i.param!r}) {args};")
        elif i.name == "measure":
            lines.append(f"y{t}[0] = measure {args};")
            t += 1
        elif i.name == "delay":
            lines.append(f"delay[{round(i.param * 1e9)}ns] {args};")
        else:
            lines.append(f"{i.name} {args};")
    return "\n".join(lines) + "\n"


def add_step_dd(
    instrs: Sequence[Instr],
    memory: int,
    measure_time: float,
    reset_time: float,
    x_time: float = GATE_TIME["x"],
    grid: float = 0.0,
) -> list[Instr]:
    """Insert an XX echo on ``memory`` across the output's measure + reset window.

    The window ``M + R`` runs from the end-of-step measurement into the next step's reset.
    Pulses are centred at ``W/4`` and ``3W/4`` of it, so a static detuning (or a ZZ phase)
    accumulated in the window cancels. In the step program the first pulse goes with the
    measurement, the second with the reset at the start of the (next) step. Delays are
    rounded down to ``grid`` seconds if given (the backend's timing granularity).
    """
    win = measure_time + reset_time

    def rnd(x: float) -> float:
        return math.floor(x / grid + 1e-9) * grid if grid else x

    a = rnd(win / 4 - x_time / 2)  # measure part: delay a, x, delay b (total M)
    b = rnd(measure_time - a - x_time)
    c = rnd(3 * win / 4 - x_time / 2 - measure_time)  # reset part: delay c, x, delay d (total R)
    d = rnd(reset_time - c - x_time)
    if min(a, b, c, d) < 0:
        raise ValueError("window too short for the echo")
    meas_part = [
        Instr("delay", (memory,), a, True),
        Instr("x", (memory,), None, True),
        Instr("delay", (memory,), b, True),
    ]
    reset_part = [
        Instr("delay", (memory,), c, True),
        Instr("x", (memory,), None, True),
        Instr("delay", (memory,), d, True),
    ]
    out: list[Instr] = []
    for ins in instrs:
        out.append(ins)
        if ins.name == "measure":
            out += meas_part
        elif ins.name == "reset":
            out += reset_part
    return out


# -- parameters ---------------------------------------------------------------------------------


@dataclass
class CoreQubit:
    """Calibrated parameters of a core (output or memory) qubit, as Phase 6's ``StepNoise``."""

    t1: float
    t2: float
    p1: float = 0.0  # depolarising parameter after sx / x
    readout: float = 0.0
    reset: float = 0.0


@dataclass
class Spectator:
    """A neighbour (or a two-level fluctuator) that is never driven.

    ``p_init`` is its excited population at the start of a shot (default ``p_eq``), ``p_eq``
    the population it relaxes towards with ``t1``. ``None`` for ``t1`` freezes it.
    """

    t1: float | None
    p_eq: float = 0.0
    p_init: float | None = None
    t2: float | None = None
    loss_rate: float = 0.0  # extra amplitude-damping rate of the memory while this is |1>

    @property
    def p0(self) -> float:
        return self.p_eq if self.p_init is None else self.p_init


@dataclass
class StepModel:
    """Everything the step simulator needs. Qubits are simulator indices (0 out, 1 mem, 2..)."""

    output: CoreQubit
    memory: CoreQubit
    p2: float = 0.0
    measure_time: float = 0.0
    reset_time: float = 0.0
    spectators: list[Spectator] = field(default_factory=list)
    zz_hz: dict[tuple[int, int], float] = field(default_factory=dict)
    detuning_hz: dict[int, float] = field(default_factory=dict)
    stark_hz: tuple[float, float] = (0.0, 0.0)  # memory shift during the output's readout, y=0/1
    meas_dephasing_rate: float = 0.0
    tone_during_reset: bool = True  # IBM reset = measure + conditional X: the tone is on
    reset_memory: bool = False  # ideal reset of the memory at each step start (memoryless q75)
    gate_time: dict[str, float] = field(default_factory=lambda: dict(GATE_TIME))

    @property
    def n(self) -> int:
        return 2 + len(self.spectators)

    def with_(self, **kw: Any) -> StepModel:
        return replace(self, **kw)


def calibrated_model(noise: dict[str, Any]) -> StepModel:
    """``StepModel`` from Phase 6's ``prepare.calibration.noise`` (compact indices 0 = q59)."""

    def d(v: Any) -> dict[Any, float]:
        return {(tuple(a) if isinstance(a, list) else a): b for a, b in v}

    p1, ro, rs, t1, t2 = (d(noise[k]) for k in ("p1", "readout", "reset", "t1", "t2"))
    p2 = d(noise["p2"])
    cq = [CoreQubit(t1[q], t2[q], p1[q], ro[q], rs[q]) for q in (0, 1)]
    return StepModel(
        output=cq[0],
        memory=cq[1],
        p2=float(next(iter(p2.values()))),
        measure_time=float(noise["measure_time"]),
        reset_time=float(noise["reset_time"]),
    )


# -- the simulator ------------------------------------------------------------------------------


def _embed(op: np.ndarray, qs: Sequence[int], n: int) -> np.ndarray:
    """``op`` on qubits ``qs`` (big-endian within ``op``), identity elsewhere."""
    rest = [q for q in range(n) if q not in qs]
    full = np.kron(op, np.eye(2 ** len(rest))).reshape((2,) * (2 * n))
    order = list(qs) + rest
    inv = [order.index(q) for q in range(n)]
    return full.transpose(inv + [n + i for i in inv]).reshape(2**n, 2**n)


def _gate(name: str, param: float | None) -> np.ndarray:
    if name == "rz":
        th = float(param)
        return np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    raise ValueError(name)


_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1, -1]).astype(complex),
]


def _depol_kraus(p: float, k: int) -> list[np.ndarray]:
    out = []
    for idx in range(4**k):
        op = np.eye(1, dtype=complex)
        for j in range(k):
            op = np.kron(op, _PAULI[(idx // 4 ** (k - 1 - j)) % 4])
        w = 1 - p + p / 4**k if idx == 0 else p / 4**k
        out.append(math.sqrt(w) * op)
    return out


def _ad_kraus(gamma: float) -> list[np.ndarray]:
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def _gad_kraus(gamma: float, p: float) -> list[np.ndarray]:
    a, b = math.sqrt(1 - gamma), math.sqrt(gamma)
    return [
        math.sqrt(1 - p) * np.array([[1, 0], [0, a]], dtype=complex),
        math.sqrt(1 - p) * np.array([[0, b], [0, 0]], dtype=complex),
        math.sqrt(p) * np.array([[a, 0], [0, 1]], dtype=complex),
        math.sqrt(p) * np.array([[0, 0], [b, 0]], dtype=complex),
    ]


def _relax_factors(dt: float, t1: float, t2: float) -> tuple[float, float]:
    """``(gamma, lam_phi)`` of Phase 6's ``relaxation_kraus`` (T2 capped at 2 T1)."""
    t2 = min(t2, 2 * t1)
    gamma = 1 - math.exp(-dt / t1)
    lam_ad = math.sqrt(1 - gamma)
    lam_phi = min(1.0, math.exp(-dt / t2) / lam_ad) if lam_ad > 0 else 0.0
    return gamma, lam_phi


@dataclass
class _Event:
    t: float
    end: float
    order: int
    ins: Instr


def schedule(instrs: Sequence[Instr], model: StepModel) -> tuple[list[_Event], float]:
    """Start/end times of each instruction (ASAP), and the step duration.

    ``measure``/``reset`` on a core qubit start when every core qubit is free (as Phase 6's
    model, where the other core qubit idles during them). Gates on the other core qubit then
    wait for the window to end, unless marked ``in_window`` (decoupling pulses and delays).
    """
    clock = {0: 0.0, 1: 0.0}
    window_end = 0.0
    events: list[_Event] = []
    for k, ins in enumerate(instrs):
        qs = ins.qubits
        if ins.name == "barrier":
            t = max(max(clock[q] for q in qs), window_end)
            for q in qs:
                clock[q] = t
            continue
        if ins.name in ("measure", "reset"):
            t = max(max(clock.values()), window_end)
            dur = model.measure_time if ins.name == "measure" else model.reset_time
            events.append(_Event(t, t + dur, k, ins))
            for q in clock:
                clock[q] = t  # the other core qubit is free to run in-window pulses
            clock[qs[0]] = t + dur
            window_end = t + dur
            continue
        t = max(clock[q] for q in qs)
        if not ins.in_window:
            t = max(t, window_end)
        dur = ins.param if ins.name == "delay" else model.gate_time[ins.name]
        if ins.name != "delay":
            events.append(_Event(t, t + dur, k, ins))
        for q in qs:
            clock[q] = t + dur
    total = max(max(clock.values()), window_end)
    events.sort(key=lambda e: (e.t, e.order))
    return events, total


class _Sim:
    def __init__(self, model: StepModel):
        self.m = model
        self.n = model.n
        n = self.n
        self.d = 2**n
        self.bits = (np.arange(self.d)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
        self._cache: dict[Any, Any] = {}

    # primitives on a batch rho (B, d, d)
    def kraus(self, rho: np.ndarray, ks: list[np.ndarray]) -> np.ndarray:
        return sum(k @ rho @ k.conj().T for k in ks)

    def emb(self, key: Any, ops: list[np.ndarray], qs: Sequence[int]) -> list[np.ndarray]:
        if key not in self._cache:
            self._cache[key] = [_embed(o, qs, self.n) for o in ops]
        return self._cache[key]

    def dephase(self, rho: np.ndarray, q: int, lam: float) -> np.ndarray:
        if lam == 1.0:
            return rho
        diff = self.bits[:, None, q] != self.bits[None, :, q]
        return rho * np.where(diff, lam, 1.0)

    def phases(self, rho: np.ndarray, energy: np.ndarray, dt: float) -> np.ndarray:
        if not np.any(energy):
            return rho
        u = np.exp(-2j * np.pi * energy * dt)
        return rho * (u[:, None] * u.conj()[None, :])

    # one slice of continuous evolution
    def evolve(self, rho: np.ndarray, dt: float, window: str | None) -> np.ndarray:
        if dt <= 0:
            return rho
        m, b = self.m, self.bits
        energy = np.zeros(self.d)
        for (a, c), z in m.zz_hz.items():
            energy += z * b[:, a] * b[:, c]
        for q, dlt in m.detuning_hz.items():
            energy += dlt * b[:, q]
        tone = window == "measure" or (window == "reset" and m.tone_during_reset)
        if tone:
            s0, s1 = m.stark_hz
            energy += (s0 * (1 - b[:, 0]) + s1 * b[:, 0]) * b[:, 1]
        rho = self.phases(rho, energy, dt)
        if window is not None:  # calibrated idle relaxation of the memory (Phase 6's model)
            g, lam = _relax_factors(dt, m.memory.t1, m.memory.t2)
            rho = self.kraus(rho, self.emb(("ad", 1, g), _ad_kraus(g), [1]))
            rho = self.dephase(rho, 1, lam)
            if tone and m.meas_dephasing_rate:
                rho = self.dephase(rho, 1, math.exp(-m.meas_dephasing_rate * dt))
        for j, s in enumerate(m.spectators):
            q = 2 + j
            if s.loss_rate:
                g = 1 - math.exp(-s.loss_rate * dt)
                p0 = np.diag([1, 0]).astype(complex)
                p1 = np.diag([0, 1]).astype(complex)
                ks = [np.kron(np.eye(2), p0)] + [np.kron(a, p1) for a in _ad_kraus(g)]
                rho = self.kraus(rho, self.emb(("loss", q, g), ks, [1, q]))
            if s.t1 is not None:
                g = 1 - math.exp(-dt / s.t1)
                rho = self.kraus(rho, self.emb(("gad", q, g, s.p_eq), _gad_kraus(g, s.p_eq), [q]))
            if s.t2 is not None:
                rho = self.dephase(rho, q, math.exp(-dt / s.t2))
        return rho

    def gate(self, rho: np.ndarray, ins: Instr) -> np.ndarray:
        m = self.m
        u = _gate(ins.name, ins.param)
        rho = self.kraus(rho, self.emb(("u", ins.name, ins.param, ins.qubits), [u], ins.qubits))
        if ins.name == "rz":
            return rho
        if len(ins.qubits) == 1:
            cq = m.output if ins.qubits[0] == 0 else m.memory
            p = cq.p1
        else:
            p = m.p2
        if p > 0:
            k = len(ins.qubits)
            rho = self.kraus(rho, self.emb(("dep", p, ins.qubits), _depol_kraus(p, k), ins.qubits))
        return rho

    def run(self, rho: np.ndarray, instrs: Sequence[Instr]) -> dict[int, np.ndarray]:
        """Branches ``{recorded y: unnormalised rho}`` after one step (one measurement)."""
        m = self.m
        if any(q > 1 for i in instrs for q in i.qubits):
            raise ValueError("the step program acts on the core qubits 0 (output), 1 (memory)")
        events, total = schedule(instrs, m)
        windows = [(e.t, e.end, e.ins.name) for e in events if e.ins.name in ("measure", "reset")]
        if any(e.ins.name in ("measure", "reset") and e.ins.qubits != (0,) for e in events):
            raise ValueError("only the output (qubit 0) is measured or reset")
        times = sorted({0.0, total} | {e.t for e in events} | {w[1] for w in windows})
        if m.reset_memory:
            k0 = np.array([[1, 0], [0, 0]], dtype=complex)
            k1 = np.array([[0, 1], [0, 0]], dtype=complex)
            rho = self.kraus(rho, self.emb(("reset_mem",), [k0, k1], [1]))
        branches: dict[int | None, np.ndarray] = {None: rho}
        ei = 0
        for t0, t1 in zip(times, [*times[1:], None], strict=True):
            while ei < len(events) and events[ei].t <= t0 + 1e-15:
                branches = self.instant(branches, events[ei].ins)
                ei += 1
            if t1 is None:
                break
            win = next((w[2] for w in windows if w[0] <= t0 + 1e-15 and t1 <= w[1] + 1e-15), None)
            branches = {y: self.evolve(r, t1 - t0, win) for y, r in branches.items()}
        if None in branches:
            raise ValueError("the step has no measurement")
        return branches  # type: ignore[return-value]

    def instant(self, branches: dict, ins: Instr) -> dict:
        m = self.m
        if ins.name == "measure":
            if None not in branches:
                raise ValueError("one measurement per step")
            rho = branches[None]
            eps = m.output.readout
            out: dict[int, np.ndarray] = {}
            for b in (0, 1):
                proj = np.zeros((2, 2), dtype=complex)
                proj[b, b] = 1
                r = self.kraus(rho, self.emb(("proj", b), [proj], [0]))
                for rec, w in ((b, 1 - eps), (1 - b, eps)):
                    if w:
                        out[rec] = out.get(rec, 0) + w * r
            return out
        if ins.name == "reset":
            eps = m.output.reset
            k0 = np.array([[1, 0], [0, 0]], dtype=complex)
            k1 = np.array([[0, 1], [0, 0]], dtype=complex)
            x = _gate("x", None)
            ks = [math.sqrt(1 - eps) * k0, math.sqrt(1 - eps) * k1]
            if eps > 0:
                ks += [math.sqrt(eps) * x @ k0, math.sqrt(eps) * x @ k1]
            ks = self.emb(("reset", eps), ks, [0])
            return {y: self.kraus(r, ks) for y, r in branches.items()}
        return {y: self.gate(r, ins) for y, r in branches.items()}


def _partial_trace_first(rho: np.ndarray, n: int) -> np.ndarray:
    """Trace out simulator qubit 0 (the output) from a batch ``(B, 2^n, 2^n)``."""
    b = rho.shape[0]
    t = rho.reshape(b, 2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.einsum("biaic->bac", t)


def step_superops(instrs: Sequence[Instr], model: StepModel) -> np.ndarray:
    """``S[y]`` of one step on the persistent register (memory ⊗ spectators), column-stacked."""
    sim = _Sim(model)
    n = model.n
    dm = 2 ** (n - 1)
    # basis inputs |i><j| on the register, |0><0| on the output (it is reset first)
    rho = np.zeros((dm * dm, 2 * dm, 2 * dm), dtype=complex)
    for i in range(dm):
        for j in range(dm):
            rho[i + j * dm, i, j] = 1.0  # output |0> occupies the first block
    br = sim.run(rho, instrs)
    sup = np.zeros((2, dm * dm, dm * dm), dtype=complex)
    for y, r in br.items():
        red = _partial_trace_first(r, n)  # (B, dm, dm)
        sup[y] = red.transpose(0, 2, 1).reshape(dm * dm, dm * dm).T
    return sup


def initial_state(model: StepModel) -> np.ndarray:
    """Memory ``|0><0|`` ⊗ each spectator ``diag(1 - p_init, p_init)``."""
    rho = np.diag([1.0, 0.0]).astype(complex)
    for s in model.spectators:
        rho = np.kron(rho, np.diag([1 - s.p0, s.p0]).astype(complex))
    return rho


def process(instrs: Sequence[Instr], model: StepModel) -> W.LinearProcess:
    return W.LinearProcess.instrument(step_superops(instrs, model), initial_state(model))


def trace_out_spectators(sup: np.ndarray, rho_spec: np.ndarray) -> np.ndarray:
    """``Tr_spec S[y](rho ⊗ rho_spec)`` as a superoperator on the memory alone."""
    dm = round(math.sqrt(sup.shape[1]))
    ds = rho_spec.shape[0]
    d1 = dm // ds
    out = np.zeros((sup.shape[0], d1 * d1, d1 * d1), dtype=complex)
    for i in range(d1):
        for j in range(d1):
            e = np.zeros((d1, d1), dtype=complex)
            e[i, j] = 1
            vin = np.kron(e, rho_spec).reshape(-1, order="F")
            for y in range(sup.shape[0]):
                o = (sup[y] @ vin).reshape(dm, dm, order="F")
                red = np.einsum("asbs->ab", o.reshape(d1, ds, d1, ds))
                out[y][:, i + j * d1] = red.reshape(-1, order="F")
    return out


# -- statistics -----------------------------------------------------------------------------------


@dataclass
class Prediction:
    g: np.ndarray
    w: dict[int, float]
    p1: float

    def row(self, prefix: str = "") -> dict[str, float]:
        r = {f"{prefix}p1": self.p1}
        r |= {f"{prefix}g{k}": float(v) for k, v in enumerate(self.g)}
        r |= {f"{prefix}w{j}": v for j, v in self.w.items()}
        return r


def predict(
    proc: W.LinearProcess,
    k_max: int = 7,
    burn: int = 16,
    length: int = 64,
    offsets: Sequence[int] = (0, 1, 2, 3),
) -> Prediction:
    """Exact ``g`` (common start set), ``w(j)`` and the mean ``P(1)`` over all positions."""
    g = proc.expected_windows(k_max, burn, length, common=True)
    return Prediction(
        g, {j: W.witness(g, j) for j in offsets}, float(proc.marginals(length).mean())
    )


def per_shot_sd(proc: W.LinearProcess, j: int, n: int, seed: int, burn: int = 16) -> float:
    """Delta-method SD of ``w(j)`` for one shot (``SE(N) = sd / sqrt(N)``), from samples."""
    shots = proc.sample(n, 64, seed)
    per = W.per_shot_windows(shots, 7, burn, common=True)
    return W.delta_se(per, j) * math.sqrt(n)


def shots_for(delta: float, sd_a: float, sd_b: float, z: float = 5.0, inflate: float = 1.4) -> int:
    """Shots per arm so that ``|delta| / (inflate * SE_diff) >= z``."""
    if delta == 0:
        return 10**12
    return math.ceil((z * inflate) ** 2 * (sd_a**2 + sd_b**2) / delta**2)


def qpu_seconds(
    shots: int, circuit_s: float, jobs: int = 1, rep_delay: float = 250e-6, job_s: float = 2.0
) -> float:
    """Phase 6's cost model: ``2 s`` per job plus ``(rep_delay + duration)`` per shot."""
    return jobs * job_s + shots * (rep_delay + circuit_s)


__all__ = [
    "GATE_TIME",
    "CoreQubit",
    "Instr",
    "Prediction",
    "Spectator",
    "StepModel",
    "add_step_dd",
    "calibrated_model",
    "initial_state",
    "parse_qasm",
    "per_shot_sd",
    "predict",
    "process",
    "qpu_seconds",
    "relabel",
    "schedule",
    "shots_for",
    "step_superops",
    "to_qasm",
    "trace_out_spectators",
]
