"""Phase 7 Stage 2A (issue #20): q75 destruction / dephasing controls (``qmem.ablate_hw``).

The numpy tests use the committed step pieces (``results/summaries/phase7_ablation/``); the
circuit tests need the optional ``hardware`` group and skip without it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import witness as W

REPO = Path(__file__).resolve().parents[1]
SUM = REPO / "results" / "summaries" / "phase7_ablation"
P6 = REPO / "results" / "summaries" / "phase6"
AUDIT = REPO / "results" / "summaries" / "phase7_audit"
S = A.Settings()
try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
needs_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="needs the hardware group")


@pytest.fixture(scope="module")
def pieces() -> dict[str, A.StepPieces]:
    z = np.load(SUM / "pieces.npz")
    return {n: A.StepPieces.from_npz(z, n) for n in ("noiseless", "calibrated", "calibrated_x3")}


def ws(proc: W.LinearProcess) -> list[float]:
    g = proc.expected_windows(S.k_max, S.burn, S.length, common=True)
    return [float(W.witness(g, j)) for j in S.offsets]


def audit_process(name: str = "calibrated") -> W.LinearProcess:
    z = np.load(AUDIT / "instruments.npz")
    return W.LinearProcess.instrument(z[name], z["rho0"])


def random_kraus(d: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """``n`` Kraus operators of a random CPTP map on dimension ``d``."""
    g = rng.normal(size=(n * d, d)) + 1j * rng.normal(size=(n * d, d))
    q, _ = np.linalg.qr(g)
    return [q[i * d : (i + 1) * d] for i in range(n)]


# -- the analytic zeros ------------------------------------------------------------------------


@pytest.mark.parametrize("noise", ["noiseless", "calibrated", "calibrated_x3"])
@pytest.mark.parametrize("variant", ["A1", "A2"])
def test_a1_a2_witness_is_exactly_zero(pieces, noise, variant):
    proc = A.process(A.step_superops(variant, pieces[noise]))
    assert max(abs(w) for w in ws(proc)) < 1e-12


def test_a1_outputs_are_iid(pieces):
    """A1 resets the memory every step: every output has the same probability, independently."""
    proc = A.process(A.step_superops("A1", pieces["calibrated"]))
    p = proc.marginals(S.length)
    assert np.ptp(p) < 1e-14
    b = proc.block_probs([S.burn], 3)
    iid = np.array(
        [np.prod([p[0] if (i >> k) & 1 else 1 - p[0] for k in range(3)]) for i in range(8)]
    )
    assert np.abs(b - iid).max() < 1e-14


def test_a3_infinite_m_is_zero(pieces):
    sup = np.mean([A.step_superops("A3", pieces["calibrated"], None, p) for p in A.PAULIS], 0)
    assert max(abs(w) for w in ws(A.process(sup))) < 1e-12


def test_dephased_memory_is_two_state_for_any_instrument():
    """The A2 argument: Z-dephasing a qubit memory each step leaves a 2-state HMM, so w = 0.

    Brute force on random instruments (rank-3 Kraus per outcome) and random pure starts; the
    same instruments without the dephasing have w != 0, and partial dephasing does too.
    """
    rng = np.random.default_rng(5)
    for _ in range(5):
        ks = random_kraus(2, 6, rng)
        sup = [A.superop(ks[:3]), A.superop(ks[3:])]
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        rho0 = np.outer(v, v.conj())
        for p, zero in ((1.0, True), (0.0, False), (0.7, False)):
            dz = A.superop(A.dephase_kraus(p))
            proc = W.LinearProcess.instrument(np.stack([dz @ s for s in sup]), rho0)
            g_fixed = proc.windows_at(proc.state_after(3), S.k_max)
            w = [abs(W.witness(g_fixed, j)) for j in S.offsets] + [abs(x) for x in ws(proc)]
            assert (max(w) < 1e-12) == zero


def test_residual_q59_reset_keeps_a1_at_zero_but_not_a2(pieces):
    """A residual q59 bit adds one classical bit: A1 stays 2-state (w = 0), A2 becomes 4-state."""
    h = A.Hypothesis("H1", q59_reset=(0.01, 0.08))
    a1 = ws(A.process(A.step_superops("A1", pieces["calibrated"], h)))
    a2 = ws(A.process(A.step_superops("A2", pieces["calibrated"], h)))
    assert max(abs(w) for w in a1) < 1e-12
    assert abs(a2[0]) > 1e-8


# -- channels ----------------------------------------------------------------------------------


def test_reset_channel_memory():
    rng = np.random.default_rng(1)
    ks = random_kraus(2, 1, rng)
    a = ks[0] @ np.diag([0.3, 0.7]) @ ks[0].conj().T
    b = np.diag([0.9, 0.1]).astype(complex)
    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    sym = A.superop(A.reset_kraus(0.02, 0.02))
    assert np.allclose(sym @ vec(a), sym @ vec(b), atol=1e-15)
    assert np.allclose(sym @ vec(a), vec(np.diag([0.98, 0.02])), atol=1e-15)
    asym = A.superop(A.reset_kraus(0.01, 0.05))
    assert np.allclose((asym @ vec(np.diag([0.0, 1.0]))), vec(np.diag([0.95, 0.05])))
    for ks2 in (A.reset_kraus(0.01, 0.05), A.dephase_kraus(0.3), A.depol_kraus(0.2)):
        assert np.allclose(sum(k.conj().T @ k for k in ks2), np.eye(2))


def test_mixture_is_weighted_sum(pieces):
    a = A.process(A.step_superops("baseline", pieces["calibrated"]))
    b = A.process(A.step_superops("baseline", pieces["calibrated_x3"]))
    m = A.mixture([a, b], [0.8, 0.2])
    for st in ([0], [16, 20]):
        assert np.allclose(
            m.block_probs(st, 4), 0.8 * a.block_probs(st, 4) + 0.2 * b.block_probs(st, 4)
        )


# -- baseline against Phase 6 ------------------------------------------------------------------


def test_joint_baseline_reproduces_phase6(pieces):
    s6 = json.loads((P6 / "summary.json").read_text())
    for noise in ("noiseless", "calibrated"):
        proc = A.process(A.step_superops("baseline", pieces[noise]))
        pred = s6["analysis"]["predictions"][noise]
        g = proc.windows_at(proc.state_after(S.burn), S.k_max)
        assert np.abs(g - np.array(pred["g"])).max() < 1e-12
        assert np.abs(proc.marginals(64) - np.array(pred["marginals"])).max() < 1e-12
    ref = audit_process()
    proc = A.process(A.step_superops("baseline", pieces["calibrated"]))
    gr = ref.expected_windows(S.k_max, S.burn, S.length, True)
    assert np.abs(proc.expected_windows(S.k_max, S.burn, S.length, True) - gr).max() < 1e-12


@pytest.mark.parametrize("variant", ["A1s", "A2s", "A3s", "A2s_spec"])
def test_shams_equal_baseline_by_construction(pieces, variant):
    for noise in ("noiseless", "calibrated"):
        assert np.array_equal(
            A.step_superops(variant, pieces[noise]), A.step_superops("baseline", pieces[noise])
        )


# -- random-Pauli instances --------------------------------------------------------------------


def test_instance_windows_brute_force(pieces):
    bank = [A.process(A.step_superops("A3", pieces["calibrated"], None, p)) for p in A.PAULIS]
    rng = np.random.default_rng(3)
    seqs = rng.integers(0, 4, size=(3, S.length))
    seqs[0] = 0  # all identity: the homogeneous baseline
    got = A.instance_windows(bank, seqs, S, chunk=2)
    base = A.process(A.step_superops("baseline", pieces["calibrated"]))
    assert np.abs(got[0] - base.expected_windows(S.k_max, S.burn, S.length, True)).max() < 1e-14
    # brute force: P(1 0^k 1 at t) summed over the realised step operators
    ops = np.stack([b.ops for b in bank])
    last = S.length - S.k_max - 2
    for i in (1, 2):
        g = np.zeros(S.k_max + 1)
        for t in range(S.burn, last + 1):
            st = bank[0].init
            for u in range(t):
                st = st @ ops[seqs[i, u]].sum(0)
            for k in range(S.k_max + 1):
                v = st @ ops[seqs[i, t], 1]
                for u in range(t + 1, t + k + 1):
                    v = v @ ops[seqs[i, u], 0]
                g[k] += (v @ ops[seqs[i, t + k + 1], 1] @ bank[0].final).real
        assert np.abs(got[i] - g / (last - S.burn + 1)).max() < 1e-14


def test_finite_m_shrinks(pieces):
    bank = [A.process(A.step_superops("A3", pieces["noiseless"], None, p)) for p in A.PAULIS]
    pool = A.instance_windows(bank, np.random.default_rng(0).integers(0, 4, (400, S.length)), S)
    sd = [A.finite_m_witness(pool, m, 300, 1)[1] for m in (4, 64)]
    assert sd[1] < sd[0] / 2
    assert abs(W.witness(pool.mean(0), 0)) < 3 * sd[1]


def test_shot_formulas():
    assert A.shots_5sigma(0.0, 1.0) == math.inf
    assert math.isclose(A.shots_5sigma(0.01, 0.02), 100.0)
    assert math.isclose(A.qpu_seconds(20000, 3.08316e-4, 250e-6, 2.0), 2 + 20000 * 5.58316e-4)


# -- circuits (hardware group) -----------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location(
        "p7a", REPO / "experiments" / "phase7_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.load_config(REPO / "experiments" / "configs" / "phase7_ablation.toml")
    return mod, cfg


@pytest.fixture(scope="module")
def hwsetup():
    if not HAS_QISKIT:
        pytest.skip("needs the hardware group")
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    mod, cfg = _script()
    s6 = mod.phase6_summary(cfg)
    backend = FakeKingston()
    return {
        "mod": mod,
        "cfg": cfg,
        "s6": s6,
        "backend": backend,
        "noise": mod.noise_levels(cfg, s6),
        "dur": mod.durations(cfg, s6, backend.target),
        "step": qasm3.loads((P6 / "step.qasm").read_text()),
        "full": qasm3.loads((P6 / "circuit.qasm").read_text()),
    }


@needs_qiskit
def test_baseline_step_is_phase6_instrument(hwsetup):
    z = np.load(AUDIT / "instruments.npz")
    for name, nz in (("calibrated", hwsetup["noise"]["calibrated"]), ("noiseless", None)):
        qc = A.build_variant(hwsetup["step"], "baseline", 1, 0, hwsetup["dur"], 1)
        sup = A.instrument_superops(qc, nz, [1], 1)
        assert np.abs(sup - z[name]).max() < 1e-12


@needs_qiskit
@pytest.mark.parametrize("noise", ["noiseless", "calibrated"])
@pytest.mark.parametrize(
    "variant,pauli",
    [("baseline", "I"), ("A1", "I"), ("A2", "I"), ("A3", "X"), ("A3", "Y"), ("A3", "Z")],
)
def test_composed_pieces_equal_circuit_simulation(hwsetup, pieces, noise, variant, pauli):
    """``Post Op Pre Reset`` from the committed pieces = the simulated control step."""
    nz = hwsetup["noise"][noise]
    qc = A.build_variant(hwsetup["step"], variant, 1, 0, hwsetup["dur"], 1, paulis=[pauli])
    qc_joint = A.build_variant(
        hwsetup["step"], variant, 1, 0, hwsetup["dur"], 1, paulis=[pauli], drop_reset=True
    )
    sim = A.instrument_superops(qc_joint, nz, [1, 0], 1)
    rs = A.reset_superop(pieces[noise], A.Hypothesis())
    comp = A.step_superops(variant, pieces[noise], None, pauli)
    assert np.abs(np.stack([sim[y] @ rs for y in range(2)]) - comp).max() < 1e-12
    # and the joint model has the same output process as the 1-qubit instrument
    one = W.LinearProcess.instrument(A.instrument_superops(qc, nz, [1], 1), np.diag([1.0, 0]))
    g1 = one.expected_windows(S.k_max, S.burn, S.length, True)
    gj = A.process(comp).expected_windows(S.k_max, S.burn, S.length, True)
    assert np.abs(g1 - gj).max() < 1e-12


@needs_qiskit
def test_a1_a2_zero_from_simulated_instruments(hwsetup):
    for v in ("A1", "A2"):
        qc = A.build_variant(hwsetup["step"], v, 1, 0, hwsetup["dur"], 1)
        for nz in (None, hwsetup["noise"]["calibrated"]):
            sup = A.instrument_superops(qc, nz, [1], 1)
            proc = W.LinearProcess.instrument(sup, np.diag([1.0, 0]))
            assert max(abs(w) for w in ws(proc)) < 1e-12


@needs_qiskit
def test_shams_equal_baseline_noiseless(hwsetup):
    from qiskit import QuantumCircuit

    step, dur = hwsetup["step"], hwsetup["dur"]
    base = A.instrument_superops(A.build_variant(step, "baseline", 1, 0, dur, 1), None, [1], 1)
    for v in ("A1s", "A2s", "A3s"):
        sup = A.instrument_superops(A.build_variant(step, v, 1, 0, dur, 1), None, [1], 1)
        assert np.abs(sup - base).max() < 1e-14
    wide = QuantumCircuit(3, name="step3")  # q[2]: the spectator
    for r in step.cregs:
        wide.add_register(r)
    wide.compose(step, [0, 1], step.clbits, inplace=True)
    qc = A.build_variant(wide, "A2s_spec", 1, 0, dur, 1, spectator=2)
    assert qc.count_ops()["measure"] == 2
    sup = A.instrument_superops(qc, None, [1], 1)
    assert np.abs(sup - base).max() < 1e-14


@needs_qiskit
def test_committed_pieces_reproduce(hwsetup, pieces):
    for name in ("noiseless", "calibrated"):
        p = A.export_pieces(hwsetup["step"], hwsetup["noise"][name], 1, 0)
        assert np.abs(p.pre - pieces[name].pre).max() < 1e-14
        assert np.abs(p.post - pieces[name].post).max() < 1e-14


@needs_qiskit
def test_circuits_and_durations_as_tabulated(hwsetup):
    """Rebuilt circuits: per-step added operations and durations match ``durations.csv``."""
    mod, cfg, backend = hwsetup["mod"], hwsetup["cfg"], hwsetup["backend"]
    full, dur, t = hwsetup["full"], hwsetup["dur"], backend.target
    rows = {r["circuit"]: r for r in csv.DictReader(open(SUM / "durations.csv"))}
    base = mod.isa(full, backend, cfg)
    assert mod.signature(base) == mod.signature(full)  # Phase 6's circuit, unchanged
    d0 = base.estimate_duration(t, unit="s")
    c0 = dict(base.count_ops())
    expect = {
        "A1": {"reset": 1.0},
        "A2": {"measure": 1.0},
        "A1s": {"delay": 1.0},
        "A2s": {"delay": 1.0},
        "A2s_spec": {"delay": 1.0, "measure": 1.0},
        "A3s": {"delay": 1.0},
    }
    paulis = mod.pauli_table(cfg)
    for v in ("baseline", *expect, "A3"):
        ps = [A.PAULIS[k] for k in paulis[0]] if v == "A3" else None
        tq = mod.isa(A.build_variant(full, v, 75, 59, dur, 64, 74, ps), backend, cfg)
        name = "A3_000" if v == "A3" else v
        d = tq.estimate_duration(t, unit="s")
        assert math.isclose(d * 1e6, float(rows[name]["duration_fake_us"]), abs_tol=1e-3)
        added = mod.per_step_counts(tq, c0, 64)
        for op, n in expect.get(v, {}).items():
            assert added[op] == n
        assert added.get("cz", 0) == 0 and added.get("sx", 0) == 0
        # every operation added to a step overlaps q59's measurement and reset: only the
        # first step (no readout to hide behind) can lengthen the circuit
        assert d - d0 <= max(dur.reset, t["reset"][(75,)].duration) + 1e-12
    for name in ("baseline", "A1", "A2"):
        from qiskit import qasm3

        tq = mod.isa(A.build_variant(full, name, 75, 59, dur, 64, 74), backend, cfg)
        assert qasm3.dumps(tq) == (SUM / "qasm" / f"{name}.qasm").read_text()
