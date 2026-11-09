"""Phase 8 Stage 2 (#32): qmem.coherence and the program models of qmem.classical."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import classical as C
from qmem import coherence as Q


def pieces(seed: int = 0) -> A.StepPieces:
    """Random but physical step pieces: CPTP ``pre`` and an instrument ``post`` on two qubits
    (q59 measured last, so every step ends with q59 diagonal)."""
    rng = np.random.default_rng(seed)

    def unitary(n: int) -> np.ndarray:
        z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        q, r = np.linalg.qr(z)
        return q * (np.diag(r) / np.abs(np.diag(r)))

    u_pre = np.kron(unitary(2), np.eye(2))  # q75's leading gates
    pre = A.superop([u_pre])
    u = unitary(4)
    deco = A.superop(A.kraus_kron(A.depol_kraus(0.02), A.depol_kraus(0.01)))
    ro = 0.02
    proj = [np.kron(np.eye(2), np.diag([1.0, 0])), np.kron(np.eye(2), np.diag([0, 1.0]))]
    meas = [A.superop([p]) for p in proj]
    rec = [(1 - ro) * meas[0] + ro * meas[1], (1 - ro) * meas[1] + ro * meas[0]]
    post = np.stack([rec[y] @ deco @ A.superop([u]) for y in (0, 1)])
    relax = np.array(A.amp_damp_kraus(0.01))
    return A.StepPieces(pre, post, relax, relax, relax, 0.01, 0.02, 0.001)


# -- the physical phase insertion -------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.3, math.pi / 2, math.pi, 4.0])
def test_echo_is_rz_up_to_phase_and_frame_neutral(phi):
    u = Q.echo_unitary(phi)
    target = A.rz_unitary(phi)
    ratio = u @ target.conj().T
    assert np.allclose(ratio, ratio[0, 0] * np.eye(2), atol=1e-12)
    a, b = Q.echo_angles(phi)
    assert a + b == 0  # the frame after the echo is unchanged


def test_echoes_compose_as_the_phase_group_with_pulse_noise():
    """Depolarising commutes with every unitary, so with gate noise E(a) E(b) = E(a + b) E(0)
    as channels: composition and inverse arms are exact in the quantum model."""
    p1 = 0.003
    a, b = 0.7, 2.1
    lhs = A.superop(Q.echo_kraus(b, p1)) @ A.superop(Q.echo_kraus(a, p1))
    rhs = A.superop(Q.echo_kraus(0.0, p1)) @ A.superop(Q.echo_kraus(a + b, p1))
    assert np.allclose(lhs, rhs, atol=1e-12)
    inv = A.superop(Q.echo_kraus(-a, p1)) @ A.superop(Q.echo_kraus(a, p1))
    sham = A.superop(Q.echo_kraus(0.0, p1)) @ A.superop(Q.echo_kraus(0.0, p1))
    assert np.allclose(inv, sham, atol=1e-12)


def test_over_rotation_breaks_the_group():
    e = 0.02
    lhs = A.superop(Q.echo_kraus(math.pi / 2, 0, e)) @ A.superop(Q.echo_kraus(math.pi / 2, 0, e))
    rhs = A.superop(Q.echo_kraus(0.0, 0, e)) @ A.superop(Q.echo_kraus(math.pi, 0, e))
    assert not np.allclose(lhs, rhs, atol=1e-6)


# -- the per-shot, per-step randomised channel --------------------------------------------------


@pytest.mark.parametrize("phases", [[0.0, math.pi], [0, math.pi / 2, math.pi, 3 * math.pi / 2]])
def test_randomised_channel_is_complete_dephasing(phases):
    rho = np.array([[0.3, 0.2 - 0.4j], [0.2 + 0.4j, 0.7]])
    out = Q.kraus_channel(Q.randomised_channel(phases), rho)
    assert np.allclose(out, np.diag(np.diag(rho)), atol=1e-12)
    assert np.allclose(out, Q.kraus_channel(A.dephase_kraus(1.0), rho), atol=1e-12)
    assert Q.coherence_factor(phases) < 1e-12


def test_biased_bits_leave_the_predicted_coherence():
    q = 0.52
    out = Q.kraus_channel(Q.randomised_channel([0, math.pi], [1 - q, q]), np.full((2, 2), 0.5))
    assert out[0, 1] == pytest.approx(0.5 * (1 - 2 * q))
    assert Q.coherence_factor([0, math.pi], [1 - q, q]) == pytest.approx(abs(1 - 2 * q))


def test_per_shot_per_step_random_phase_is_the_averaged_channel_by_enumeration():
    """Every bit sequence enumerated: the y-process of a feed-forward arm averaged over its
    i.i.d. per-step choices equals the stationary process of the averaged (dephasing) step."""
    qm = Q.QuantumModel(pieces(1), 0.01)
    arm = C.ArmSpec("R4", (("e0", "e0"), ("e0", "e2"), ("e4", "e0"), ("e4", "e2")), "random")
    T = 3
    bank = qm.bank(arm, record=False)
    avg = qm.averaged(arm)
    for ys in itertools.product((0, 1), repeat=T):
        tot = 0.0
        for ch in itertools.product(range(4), repeat=T):
            s = qm.vec0
            for t in range(T):
                s = s @ bank[ch[t], ys[t]]
            tot += 0.25**T * float((s @ qm.trace).real)
        s = avg.init
        for y in ys:
            s = s @ avg.ops[y]
        assert tot == pytest.approx(float((s @ avg.final).real), abs=1e-13)
    # and the averaged slot is complete Z dephasing of the deterministic sham
    sl = sum(qm.slot(p) for p in arm.programs) / 4
    deph = A.on_q75(A.dephase_kraus(1.0)) @ qm.slot(("e0", "e0"))
    assert np.allclose(sl, deph, atol=1e-12)


def test_pooled_fixed_patterns_are_not_the_channel():
    """A fixed per-circuit phase pattern pooled over instances is a mixture of processes: its
    two-step statistics differ from the per-step channel's (#28, #30 section 4)."""
    qm = Q.QuantumModel(pieces(2), 0.01)
    arm = C.ArmSpec("R2", (("e0",), ("e4",)), "random")
    bank = qm.bank(arm, record=False)
    avg = qm.averaged(arm)
    # two instances: all-0 and all-pi, pooled with weight 1/2 each
    pooled = {}
    for ys in itertools.product((0, 1), repeat=3):
        v = 0.0
        for c in (0, 1):
            s = qm.vec0
            for y in ys:
                s = s @ bank[c, y]
            v += 0.5 * float((s @ qm.trace).real)
        s = avg.init
        for y in ys:
            s = s @ avg.ops[y]
        pooled[ys] = (v, float((s @ avg.final).real))
    assert max(abs(a - b) for a, b in pooled.values()) > 1e-4


# -- the quantum model ----------------------------------------------------------------------------


def test_default_model_reproduces_the_calibrated_step():
    pc = pieces(3)
    qm = Q.QuantumModel(pc, 0.01)
    for arm, prog in (("baseline", ("I",)), ("A1", ("reset",))):
        assert np.allclose(qm.step(prog), A.step_superops(arm, pc), atol=1e-13)
    for p in "XYZ":
        assert np.allclose(qm.step((p,)), A.step_superops("A3", pc, pauli=p), atol=1e-13)
    rec = A.a2_record_superops(pc, 0.01, 0.1)
    qm2 = Q.QuantumModel(pc, 0.01, Q.NoiseParams(meas_decay=0.1))
    for m in (0, 1):
        assert np.allclose(qm2.step(("meas",), m), rec[m], atol=1e-13)


def _brute_prob(pc: A.StepPieces, p: Q.NoiseParams, prog, ys) -> float:
    """``P(y)`` by explicit density-matrix evolution: the extra channels applied as Kraus
    operators to rho (4 x 4), the pieces as superoperators, the flip as a mixture of the two
    physical outcomes; no subspace reduction, no row convention."""
    hyp = A.Hypothesis(q59_reset=(p.reset_e0, p.reset_e1) if p.reset_e0 is not None else None)
    reset = A.reset_superop(pc, hyp)
    extra = A.kraus_kron(
        Q.compose(A.amp_damp_kraus(p.damp), A.dephase_kraus(p.dephase), [A.rz_unitary(p.detune)]),
        [np.eye(2)],
    )
    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    unvec = lambda v: v.reshape(4, 4, order="F")  # noqa: E731

    def run(rho, ys_phys):
        for y in ys_phys:
            rho = unvec(reset @ vec(rho))
            rho = sum(k @ rho @ k.conj().T for k in extra)
            rho = unvec(pc.pre @ vec(rho))
            for e in prog:
                ks = A.kraus_kron(Q.echo_kraus(Q.echo_phase(int(e[1:])), pc.p1_q75), [np.eye(2)])
                rho = sum(k @ rho @ k.conj().T for k in ks)
            rho = unvec(pc.post[y] @ vec(rho))
        return float(np.trace(rho).real)

    rho0 = A.RHO0_JOINT
    tot = 0.0
    for ys_phys in itertools.product((0, 1), repeat=len(ys)):
        w = 1.0
        for a, b in zip(ys, ys_phys, strict=True):
            w *= (1 - p.flip) if a == b else p.flip
        tot += w * run(rho0, ys_phys)
    return tot


def test_fitted_quantum_likelihood_against_brute_force():
    pc = pieces(4)
    p = Q.NoiseParams(dephase=0.1, damp=0.05, detune=0.3, flip=0.03, reset_e0=0.02, reset_e1=0.08)
    qm = Q.QuantumModel(pc, 0.01, p)
    arm = C.ArmSpec("P", (("e2", "e3"),))
    ys = np.array(list(itertools.product((0, 1), repeat=4)))
    nll = qm.nll(arm, C.ProgramData(ys))
    brute = np.array([_brute_prob(pc, p, ("e2", "e3"), y) for y in ys])
    assert np.allclose(np.exp(-nll), brute, atol=1e-12)
    assert np.exp(-nll).sum() == pytest.approx(1.0, abs=1e-12)


def test_parameter_vector_round_trip():
    pc = pieces(5)
    p = Q.NoiseParams(dephase=0.2, detune=-0.4, reset_e0=0.01, reset_e1=0.3, spec_phase=1.1)
    x = Q.to_vector(p, Q.FITTED, pc)
    back = Q.from_vector(x, Q.FITTED, Q.NoiseParams())
    for k in Q.FITTED:
        a = getattr(p, k)
        assert getattr(back, k) == pytest.approx(pc.eps_q59 if a is None else a, rel=1e-9, abs=2e-6)


# -- the classical program models ----------------------------------------------------------------


def _arms():
    return {
        "P": C.ArmSpec("P", (("e1",),)),
        "COMP": C.ArmSpec("COMP", (("e1", "e2"),)),
        "ALT": C.ArmSpec("ALT", (("e1",), ("e2",)), "alternate"),
        "R": C.ArmSpec("R", (("ff0", "e0"), ("ff1", "e2")), "random"),
        "M": C.ArmSpec("M", (("meas",),)),
        "D": C.ArmSpec("D", whole="D"),
    }


def _model(d: int, seed: int) -> C.ProgramModel:
    rng = np.random.default_rng(seed)
    return C.random_program_model(d, ["e0", "e1", "e2", "ff0", "ff1"], rng, ["meas"], ["D"])


def test_program_model_likelihood_against_path_enumeration():
    m = _model(3, 0)
    rng = np.random.default_rng(1)
    T = 3
    for name, arm in _arms().items():
        y = rng.integers(0, 2, (6, T))
        ch = arm.choices(6, T, rng)
        mm = rng.integers(0, 2, (6, T)) if name == "M" else None
        nll = m.nll(arm, C.ProgramData(y, ch, mm))
        for i in range(6):
            bf = C.brute_force_program_prob(m, arm, y[i], ch[i], None if mm is None else mm[i])
            assert math.exp(-nll[i]) == pytest.approx(bf, rel=1e-11)


def test_program_model_equals_intervention_model_on_stage4c_arms():
    rng = np.random.default_rng(2)
    im = C.random_model(4, rng)
    pm = C.ProgramModel.from_intervention(im)
    data, pdata = {}, {}
    for arm in C.STAGE4C_ARMS:
        y = rng.integers(0, 2, (20, 6))
        a = C.ArmData(
            y,
            rng.integers(0, 2, (20, 6)) if arm == "A2" else None,
            rng.integers(0, 4, (20, 6)) if arm == "A3" else None,
        )
        data[arm], pdata[arm] = a, C.program_data_from_arm(arm, a)
        assert np.allclose(im.nll(arm, a.y, a.m, a.paulis), pm.nll(C.STAGE4C_ARMS[arm], pdata[arm]))
    new_i, _, ll_i = C.em_step(im, data, C.Structure(4))
    new_p, ll_p = C.program_em_step(pm, C.STAGE4C_ARMS, pdata)
    assert ll_i == pytest.approx(ll_p, rel=1e-12)
    assert np.allclose(new_i.pre, new_p.pre) and np.allclose(new_i.post, new_p.post)
    assert np.allclose(new_i.meas, new_p.labelled["meas"])
    assert np.allclose(new_i.pauli[2], new_p.maps["Y"])


def test_program_em_increases_the_likelihood():
    true = _model(3, 3)
    rng = np.random.default_rng(4)
    arms = _arms()
    data = {}
    for name, arm in arms.items():
        ch = arm.choices(150, 8, rng)
        bank = true.bank(arm, record=(name == "M"))
        sym = C.bank_sample(true.init, bank, np.ones(3), ch, rng)
        data[name] = C.ProgramData(sym % 2, ch, sym // 2) if name == "M" else C.ProgramData(sym, ch)
    m = _model(3, 5)
    prev = -math.inf
    for _ in range(6):
        m, ll = C.program_em_step(m, arms, data)
        assert ll >= prev - 1e-8
        prev = ll
    fit = C.fit_program_model(arms, data, _model(3, 6), max_iters=30)
    assert fit.history[-1] <= fit.history[0] + 1e-12


def test_program_model_statistics_match_sampling():
    m = _model(3, 7)
    arm = _arms()["R"]
    rng = np.random.default_rng(8)
    ch = arm.choices(40000, 10, rng)
    y = C.bank_sample(m.init, m.bank(arm), np.ones(3), ch, rng)
    p = m.averaged(arm)
    assert y[:, 5].mean() == pytest.approx(float(p.marginals(10)[5]), abs=0.01)


def test_divisible_phase_maps_in_a_finite_model_must_be_permutations_to_compose_exactly():
    """The Lemma (#30, section 1.2) on the new arms: if M(pi/2)^4 = I with a stochastic
    M(pi/2), it is a permutation, so a finite classical model can only mimic the phase group."""
    rng = np.random.default_rng(9)
    for _ in range(50):
        m = rng.dirichlet(np.ones(4), size=4)
        assert not np.allclose(np.linalg.matrix_power(m, 4), np.eye(4), atol=1e-3)
    perm = np.eye(4)[[1, 2, 3, 0]]
    assert np.allclose(np.linalg.matrix_power(perm, 4), np.eye(4))


# -- Knee-style controls ------------------------------------------------------------------------


def test_knee_inequality_holds_for_mixture_preparations():
    """A model whose superposition preparation gives a mixture of the |0>, |1> hidden states
    obeys D_+ <= max(D_0, D_1) for any intervention (convexity of the TVD)."""
    rng = np.random.default_rng(10)
    for _ in range(200):
        d = 4
        h0, h1 = rng.dirichlet(np.ones(d)), rng.dirichlet(np.ones(d))
        lam = rng.random()
        hp = lam * h0 + (1 - lam) * h1
        mi, ms = rng.dirichlet(np.ones(d), size=d), rng.dirichlet(np.ones(d), size=d)
        emit = rng.dirichlet(np.ones(5), size=d)  # any output distribution of the hidden state
        dist = {
            k: Q.tvd(h @ mi @ emit, h @ ms @ emit) for k, h in (("0", h0), ("1", h1), ("+", hp))
        }
        assert dist["+"] <= max(dist["0"], dist["1"]) + 1e-12


def test_quantum_instrument_violates_the_knee_inequality():
    qm = Q.QuantumModel(pieces(11), 0.01)
    d = Q.knee_disturbances(qm, {"e4": ["e4"]}, ["e0"], 3)
    assert d["e4"]["0"] < 1e-12 and d["e4"]["1"] < 1e-12
    assert d["e4"]["excess_plus"] > 0.01


def test_benchmark_populations_and_ramsey():
    b = Q.Benchmark(t1=300e-6, t2=150e-6, p1=1e-4, spacing_s=5e-6)
    for st in ("0", "1"):
        e = Q.benchmark("e4", st, 8, b)
        s = Q.benchmark("e0", st, 8, b)
        assert e["p1"] == pytest.approx(s["p1"], abs=1e-12)  # populations invariant
    r = Q.benchmark("R2", "+", 1, b)
    assert r["coherence"] < 1e-12
    e = Q.benchmark("e2", "+", 1, b)
    assert e["coherence"] > 0.9 and abs(e["y"]) > 0.9  # Rz(pi/2) |+> = |+i>


# -- transpilation (hardware group) ---------------------------------------------------------------


def test_physical_phase_insertions_survive_transpilation():
    pytest.importorskip("qiskit")
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import phase8_coherence_hw as HW
    from qiskit import QuantumCircuit, transpile

    backend = fake.FakeKingston()
    n = backend.num_qubits

    def circ(mode: str) -> QuantumCircuit:
        qc = QuantumCircuit(n, 1)
        qc.sx(HW.MEMORY)
        for phi in (math.pi / 2, -math.pi / 2, math.pi / 4):
            if mode == "virtual":
                qc.rz(phi, HW.MEMORY)
            else:
                HW._echo(qc, qc.qubits[HW.MEMORY], phi, fence=(mode == "physical"))
        qc.cz(HW.MEMORY, HW.OUTPUT)
        qc.measure(HW.OUTPUT, 0)
        return qc

    def xs(qc) -> int:
        return sum(1 for i in qc.data if i.operation.name == "x")

    for level in (0, 1, 2, 3):
        tq = transpile(
            circ("physical"), backend, optimization_level=level, initial_layout=list(range(n))
        )
        assert xs(tq) == 6, level  # every pulse of every echo survives
        rz = [float(i.operation.params[0]) for i in tq.data if i.operation.name == "rz"]
        assert len(rz) >= 6
    # the controls: without fences the optimiser merges the echoes, and the virtual scan's
    # inverse pair cancels (no pulse at all)
    unf = transpile(circ("unfenced"), backend, optimization_level=1, initial_layout=list(range(n)))
    assert xs(unf) < 6
    virt = transpile(circ("virtual"), backend, optimization_level=1, initial_layout=list(range(n)))
    assert xs(virt) == 0
