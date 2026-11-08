"""Tests of the spectator / crosstalk step simulator (Phase 7 Stage 2C, #22)."""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path

import numpy as np
import pytest

from qmem import spectator as S

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "results" / "summaries" / "phase6"
AUDIT = ROOT / "results" / "summaries" / "phase7_audit"
SPEC = ROOT / "results" / "summaries" / "phase7_spectator"


@pytest.fixture(scope="module")
def cal() -> S.StepModel:
    s6 = json.loads((P6 / "summary.json").read_text())
    return S.calibrated_model(s6["prepare"]["calibration"]["noise"])


@pytest.fixture(scope="module")
def prog() -> list[S.Instr]:
    return S.parse_qasm((P6 / "step.qasm").read_text())


@pytest.fixture(scope="module")
def calibrated_sup() -> np.ndarray:
    return np.load(AUDIT / "instruments.npz")["calibrated"]


def idle_prog() -> list[S.Instr]:
    """No gates: the output is reset and measured, the memory only idles (4.39 us)."""
    return [S.Instr("reset", (0,)), S.Instr("measure", (0,))]


def apply(sup: np.ndarray, rho: np.ndarray) -> np.ndarray:
    d = rho.shape[0]
    return sum(s @ rho.reshape(-1, order="F") for s in sup).reshape(d, d, order="F")


# -- the zero-coupling limit ------------------------------------------------------------------


def test_schedule_reproduces_phase6_step_duration(cal, prog):
    s6 = json.loads((P6 / "summary.json").read_text())
    _, total = S.schedule(prog, cal)
    assert total == pytest.approx(s6["prepare"]["step_duration_s"], abs=1e-15)


def test_zero_coupling_recovers_the_calibrated_instrument(cal, prog, calibrated_sup):
    sup = S.step_superops(prog, cal)
    assert np.abs(sup - calibrated_sup).max() < 1e-13


def test_noiseless_recovers_the_noiseless_instrument(prog):
    z = np.load(AUDIT / "instruments.npz")
    ideal = S.StepModel(
        output=S.CoreQubit(math.inf, math.inf), memory=S.CoreQubit(math.inf, math.inf)
    )
    sup = S.step_superops(prog, ideal)
    assert np.abs(sup - z["noiseless"]).max() < 1e-13


@pytest.mark.parametrize(
    "t1, p, t2", [(323e-6, 0.05, 200e-6), (None, 0.5, None), (50e-6, 0.9, 30e-6)]
)
def test_decoupled_spectator_traces_out(cal, prog, calibrated_sup, t1, p, t2):
    m = cal.with_(spectators=[S.Spectator(t1=t1, p_eq=p, t2=t2)])
    sup = S.step_superops(prog, m)
    for pop in (0.0, p, 1.0):
        red = S.trace_out_spectators(sup, np.diag([1 - pop, pop]).astype(complex))
        assert np.abs(red - calibrated_sup).max() < 1e-13
    # the output process is unchanged too
    a = S.predict(S.process(prog, m))
    b = S.predict(S.process(prog, cal))
    assert np.allclose(a.g, b.g, atol=1e-14) and abs(a.w[0] - b.w[0]) < 1e-14


def test_two_decoupled_spectators_trace_out(cal, prog, calibrated_sup):
    m = cal.with_(spectators=[S.Spectator(323e-6, 0.05), S.Spectator(196e-6, 0.2)])
    sup = S.step_superops(prog, m)
    rs = np.kron(np.diag([0.95, 0.05]), np.diag([0.8, 0.2])).astype(complex)
    assert np.abs(S.trace_out_spectators(sup, rs) - calibrated_sup).max() < 1e-13


def test_matches_qmem_hardware_with_other_noise(prog):
    """Brute force against Phase 6's Qiskit branch simulator for a different noise setting."""
    hw = pytest.importorskip("qmem.hardware")
    from qiskit import qasm3

    m = S.StepModel(
        output=S.CoreQubit(100e-6, 60e-6, 3e-3, 0.02, 0.01),
        memory=S.CoreQubit(80e-6, 40e-6, 1e-3, 0.03, 0.04),
        p2=0.01,
        measure_time=1.5e-6,
        reset_time=1.0e-6,
    )
    noise = hw.StepNoise(
        p1={0: 3e-3, 1: 1e-3},
        p2={(0, 1): 0.01, (1, 0): 0.01},
        readout={0: 0.02, 1: 0.03},
        reset={0: 0.01, 1: 0.04},
        t1={0: 100e-6, 1: 80e-6},
        t2={0: 60e-6, 1: 40e-6},
        measure_time=1.5e-6,
        reset_time=1.0e-6,
    )
    qc = qasm3.loads((P6 / "step.qasm").read_text())
    inst = hw.noisy_instrument(qc, 1, 1, noise, [1], memory_qubits_out=[1])
    k = inst.kraus.numpy()
    ref = np.stack([sum(np.kron(a.conj(), a) for a in k[y]) for y in range(2)])
    assert np.abs(S.step_superops(prog, m) - ref).max() < 1e-12


# -- analytic cases ---------------------------------------------------------------------------


def test_zz_phase_and_idle_coherence_are_analytic(cal):
    """Memory |+>, a frozen spectator in |1>: rho01 picks up 2 pi zeta T and decays as e^-T/T2."""
    zeta = 37e3
    m = cal.with_(spectators=[S.Spectator(t1=None, p_eq=1.0)], zz_hz={(1, 2): zeta})
    sup = S.step_superops(idle_prog(), m)
    rho = np.kron(0.5 * np.ones((2, 2)), np.diag([0.0, 1.0])).astype(complex)
    red = np.einsum("asbs->ab", apply(sup, rho).reshape(2, 2, 2, 2))
    win = cal.measure_time + cal.reset_time
    expect = 0.5 * math.exp(-win / cal.memory.t2) * cmath.exp(2j * math.pi * zeta * win)
    assert abs(red[0, 1] - expect) < 1e-13


def test_frozen_excited_spectator_equals_a_detuning(cal, prog):
    zeta = 23e3
    m = cal.with_(spectators=[S.Spectator(t1=None, p_eq=1.0)], zz_hz={(1, 2): zeta})
    sup = S.step_superops(prog, m)
    red = S.trace_out_spectators(sup, np.diag([0.0, 1.0]).astype(complex))
    det = S.step_superops(prog, cal.with_(detuning_hz={1: zeta}))
    assert np.abs(red - det).max() < 1e-13


def test_spectator_relaxes_to_its_equilibrium(cal):
    t1, peq = 120e-6, 0.07
    m = cal.with_(spectators=[S.Spectator(t1=t1, p_eq=peq)])
    sup = S.step_superops(idle_prog(), m)
    rho = np.kron(np.diag([1.0, 0.0]), np.diag([0.0, 1.0])).astype(complex)
    _, total = S.schedule(idle_prog(), m)
    out = apply(sup, rho).reshape(2, 2, 2, 2)
    pop = np.einsum("asas->s", out).real[1]
    assert pop == pytest.approx(peq + (1 - peq) * math.exp(-total / t1), abs=1e-13)


def test_tls_loss_shortens_t1_only_when_on(cal):
    rate = 1 / 50e-6
    m = cal.with_(spectators=[S.Spectator(t1=None, p_eq=1.0, loss_rate=rate)])
    sup = S.step_superops(idle_prog(), m)
    win = cal.measure_time + cal.reset_time
    for spec, extra in ((0, 0.0), (1, rate)):
        rho = np.kron(np.diag([0.0, 1.0]), np.diag([1.0 - spec, spec])).astype(complex)
        out = apply(sup, rho).reshape(2, 2, 2, 2)
        p_exc = np.einsum("asas->a", out).real[1]
        assert p_exc == pytest.approx(math.exp(-win / cal.memory.t1 - extra * win), abs=1e-13)


def test_stark_shift_during_readout_and_reset(cal):
    s0, s1 = 11e3, -40e3  # the output is |0> here (reset, no gate), so only s0 acts
    clean = S.CoreQubit(cal.output.t1, cal.output.t2)  # no reset error: the output stays |0>
    m = cal.with_(stark_hz=(s0, s1), output=clean)
    sup = S.step_superops(idle_prog(), m)
    out = apply(sup, 0.5 * np.ones((2, 2), dtype=complex))
    win = cal.measure_time + cal.reset_time
    assert cmath.phase(out[0, 1]) == pytest.approx(2 * math.pi * s0 * win, abs=1e-12)
    m2 = cal.with_(stark_hz=(s0, s1), tone_during_reset=False, output=clean)
    out2 = apply(S.step_superops(idle_prog(), m2), 0.5 * np.ones((2, 2), dtype=complex))
    assert cmath.phase(out2[0, 1]) == pytest.approx(2 * math.pi * s0 * cal.measure_time, abs=1e-12)


def test_echo_cancels_a_static_detuning(cal):
    """XX centred at W/4 and 3W/4 of the idle window: the detuning phase cancels exactly."""
    m = cal.with_(detuning_hz={1: 70e3})
    m.output = S.CoreQubit(cal.output.t1, cal.output.t2)  # no readout error
    m.memory = S.CoreQubit(math.inf, math.inf)  # no relaxation, no gate error
    dd = S.add_step_dd(idle_prog(), 1, m.measure_time, m.reset_time)
    # within one step: + c, X, - (x + d) in the reset; - a, X, + (x + b) in the readout: 0
    rho = 0.5 * np.ones((2, 2), dtype=complex)
    echoed = apply(S.step_superops(dd, m), rho)
    plain = apply(S.step_superops(idle_prog(), m), rho)
    assert abs(cmath.phase(plain[0, 1])) > 0.5
    assert abs(echoed[0, 1] - 0.5) < 1e-12


def test_memoryless_with_one_spectator_has_no_witness(cal, prog):
    """With q75 reset every step, one spectator is a 2-state process: w = 0 exactly."""
    m = cal.with_(
        spectators=[S.Spectator(t1=300e-6, p_eq=0.3)],
        zz_hz={(1, 2): 80e3, (0, 2): 30e3},
        reset_memory=True,
    )
    pr = S.predict(S.process(prog, m))
    assert all(abs(v) < 1e-15 for v in pr.w.values())


def test_classical_spectator_mixture_matches_brute_force(cal, prog):
    """A frozen spectator in a mixture = the mixture of the two conditional processes."""
    p, zeta = 0.3, 40e3
    m = cal.with_(spectators=[S.Spectator(t1=None, p_eq=p)], zz_hz={(1, 2): zeta})
    mix = S.process(prog, m).expected_windows(7, 16, 64, common=True)
    p0 = S.process(prog, cal).expected_windows(7, 16, 64, common=True)
    p1 = S.process(prog, cal.with_(detuning_hz={1: zeta})).expected_windows(7, 16, 64, common=True)
    assert np.allclose(mix, (1 - p) * p0 + p * p1, atol=1e-15)


# -- circuits and costs -------------------------------------------------------------------------


def test_qasm_round_trip(prog):
    text = S.to_qasm(prog, physical=False)
    back = S.parse_qasm(text)
    assert [(i.name, i.qubits) for i in back] == [(i.name, i.qubits) for i in prog]
    assert all(
        a.param == pytest.approx(b.param) for a, b in zip(back, prog, strict=True) if a.param
    )


def test_dd_delays_fill_the_windows(cal):
    dd = S.add_step_dd(idle_prog(), 1, cal.measure_time, cal.reset_time, grid=4e-9)
    d = [i.param for i in dd if i.name == "delay"]
    x = S.GATE_TIME["x"]
    assert d[0] + x + d[1] <= cal.reset_time + 1e-15  # reset part (first in the step)
    assert d[2] + x + d[3] <= cal.measure_time + 1e-15
    assert all(abs(v / 4e-9 - round(v / 4e-9)) < 1e-6 for v in d)
    _, total = S.schedule(dd, cal)
    assert total == pytest.approx(cal.measure_time + cal.reset_time)


def test_cost_model_matches_phase6():
    # Phase 6 main job: 20,000 x 64 steps, 308.3 us circuits: about 14 s of QPU time.
    assert S.qpu_seconds(20000, 308.316e-6) == pytest.approx(13.17, abs=0.01)


def test_shots_for_scales_as_inverse_square():
    a = S.shots_for(1e-3, 0.015, 0.015)
    b = S.shots_for(2e-3, 0.015, 0.015)
    assert a == pytest.approx(4 * b, rel=1e-3)
    assert a == math.ceil((5 * 1.4) ** 2 * 2 * 0.015**2 / 1e-6)


def test_model_per_shot_sd_matches_device(cal, prog):
    proc = S.process(prog, cal)
    sd = S.per_shot_sd(proc, 0, 4000, seed=1)
    summary = json.loads((SPEC / "summary.json").read_text())
    dev = summary["device_segments"]["rest"]["sd_w0_per_shot"]
    assert sd == pytest.approx(dev, rel=0.15)


# -- the committed control QASM (hardware group) -------------------------------------------------


def test_control_qasm_is_native_and_on_the_coupling_map():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    b = FakeKingston()
    edges = {tuple(sorted(e)) for e in b.coupling_map.get_edges()}
    native = set(b.target.operation_names) | {"barrier"}
    files = sorted((SPEC / "qasm").glob("*.qasm"))
    assert len(files) >= 7
    for f in files:
        qc = qasm3.loads(f.read_text())
        assert sum(1 for i in qc.data if i.operation.name == "measure") == 64
        for inst in qc.data:
            assert inst.operation.name in native, (f.name, inst.operation.name)
            if inst.operation.name == "cz":
                pair = tuple(sorted(qc.find_bit(q).index for q in inst.qubits))
                assert pair in edges, (f.name, pair)
        # the parser used by the simulator reads them too
        S.parse_qasm(f.read_text())
