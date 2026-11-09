"""Phase 8 Stage 3 (#33): non-entanglement-breaking certification of q75's step channel.

Bounds on random EB channels (Bell F <= 1/2, P&M F_avg <= 2/3, MDI payoff <= 0 under any
measurement), the estimators against analytic cases (Werner state), the closed-form maximum
over local unitaries against brute force, the simulated channel at n = 0 against the ideal
Bell state, and the committed QASM against its hashes (and, with the hardware group, a rebuild).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import reference as R

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "results" / "summaries" / "phase8_reference"


def _random_channel_kraus(rng, k=3):
    g = rng.standard_normal((2 * k, 2)) + 1j * rng.standard_normal((2 * k, 2))
    q, _ = np.linalg.qr(g)  # an isometry 2 -> 2k: Kraus blocks
    return [q[2 * i : 2 * i + 2, :] for i in range(k)]


def _random_state(rng, d=4):
    h = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    r = h @ h.conj().T
    return r / np.trace(r).real


@pytest.fixture(scope="module")
def pieces():
    z = np.load(ROOT / "results/summaries/phase7_ablation/pieces.npz")
    live = A.StepPieces.from_npz(
        np.load(ROOT / "results/summaries/phase7_hw_4c/pieces_live.npz"), "live"
    )
    return {"noiseless": A.StepPieces.from_npz(z, "noiseless"), "live": live}


# -- the classical bounds ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_eb_bell_fidelity_at_most_half(seed):
    """Every EB map (trace-preserving or a trace-decreasing branch) leaves a separable Choi
    state: max over maximally entangled targets <= 1/2 (closed form and brute force), PPT."""
    rng = np.random.default_rng(seed)
    scale = 1.0 if seed % 2 == 0 else float(rng.uniform(0.2, 0.9))
    sup = R.superop(R.random_eb_kraus(rng, int(rng.integers(2, 6)), scale))
    j = R.choi(sup)
    if scale == 1.0:
        assert np.trace(j).real == pytest.approx(1.0)
    assert R.max_bell_fidelity(j) <= 0.5 + 1e-12
    best = max(R.bell_fidelity(j, R.random_unitary(rng)) for _ in range(300))
    assert best <= R.max_bell_fidelity(j) + 1e-12
    assert R.negativity(j) <= 1e-12


@pytest.mark.parametrize("seed", range(12))
def test_eb_prepare_and_measure_at_most_two_thirds(seed):
    """The six-axis P&M score of an EB channel is <= 2/3 for every target V (the pooled ratio
    too, for an EB branch), and stays <= 2/3 with balanced readout errors."""
    rng = np.random.default_rng(100 + seed)
    scale = 1.0 if seed % 2 == 0 else float(rng.uniform(0.2, 0.9))
    sup = R.superop(R.random_eb_kraus(rng, int(rng.integers(2, 6)), scale))
    v_opt = R.optimal_correction(R.choi(sup))
    for v in [v_opt, *(R.random_unitary(rng) for _ in range(50))]:
        assert R.axis_average_fidelity(sup, v) <= 2 / 3 + 1e-12
    spam = R.Spam(0.0, 0.0, 0.0, float(rng.uniform(0, 0.05)), float(rng.uniform(0, 0.05)))
    assert R.pm_score(R.pm_probabilities(sup, v_opt, spam)) <= 2 / 3 + 1e-12


@pytest.mark.parametrize("seed", range(8))
def test_average_fidelity_is_affine_in_the_entanglement_fidelity(seed):
    """For any channel, the six axis states (a 2-design) give F_avg = (2 F_e + 1)/3, with F_e
    the Choi state's fidelity with the same target (1 ⊗ V)|Phi+>."""
    rng = np.random.default_rng(200 + seed)
    sup = R.superop(_random_channel_kraus(rng))
    v = R.random_unitary(rng)
    fe = R.bell_fidelity(R.choi(sup), v)
    assert R.axis_average_fidelity(sup, v) == pytest.approx(R.f_avg_from_fe(fe), abs=1e-12)
    assert R.pm_score(R.pm_probabilities(sup, v, R.Spam())) == pytest.approx(
        R.f_avg_from_fe(fe), abs=1e-12
    )


@pytest.mark.parametrize("seed", range(8))
def test_mdi_payoff_is_fe_minus_half_and_eb_safe_under_any_measurement(seed):
    """Rosset et al.'s semiquantum game: with a perfect Bell measurement the payoff is
    tr(W J) = F_e - 1/2; for an EB channel it is <= 0 whatever POVM element replaces Phi+."""
    rng = np.random.default_rng(300 + seed)
    v = R.random_unitary(rng)
    om = R.mdi_weights(v)
    sup = R.superop(_random_channel_kraus(rng))
    p = R.mdi_probabilities(sup, R.Spam(), R.Spam(), 0.0)
    assert float((om * p).sum()) == pytest.approx(R.bell_fidelity(R.choi(sup), v) - 0.5, abs=1e-12)
    eb = R.superop(R.random_eb_kraus(rng, int(rng.integers(2, 5))))
    h = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    m = h @ h.conj().T
    m /= np.linalg.eigvalsh(m).max()  # a random POVM element 0 <= M <= 1
    pay = 0.0
    for x, kx in enumerate(R.TETRA_KETS):
        out = R.apply(eb, R.dm(kx))
        for y, ky in enumerate(R.TETRA_KETS):
            pay += om[x, y] * float(np.trace(np.kron(out, R.dm(ky)) @ m).real)
    assert pay <= 1e-12


def test_identity_and_dephasing_controls():
    """Positive control: the identity gives F_e = 1, P&M 1, MDI payoff 1/2. Negative control:
    full Z dephasing is EB and sits exactly on the bounds (1/2, 2/3)."""
    ident = np.eye(4, dtype=complex)
    assert R.bell_fidelity(R.choi(ident)) == pytest.approx(1.0)
    assert R.axis_average_fidelity(ident) == pytest.approx(1.0)
    deph = R.superop(A.dephase_kraus(1.0))
    assert R.max_bell_fidelity(R.choi(deph)) == pytest.approx(0.5)
    assert R.axis_average_fidelity(deph) == pytest.approx(2 / 3)


# -- estimators ----------------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.2, 1 / 3, 0.6, 1.0])
def test_bell_estimator_on_werner_state(p):
    """Werner state: correlators (p, -p, p), F = (1 + 3p)/4, entangled iff p > 1/3. The
    XX/YY/ZZ estimator recovers F from the parities, and its SE matches the spread of draws."""
    rho = R.werner(p)
    assert R.bell_fidelity(rho) == pytest.approx((1 + 3 * p) / 4)
    assert R.max_bell_fidelity(rho) == pytest.approx((1 + 3 * p) / 4)
    par = R.bell_parities(rho, R.I2, R.RefNoise())
    assert [2 * par[s] - 1 for s in ("xx", "yy", "zz")] == pytest.approx([p, -p, p])
    assert par["fidelity"] == pytest.approx((1 + 3 * p) / 4)
    assert (R.negativity(rho) > 1e-12) == (p > 1 / 3 + 1e-12)
    n = 4000
    pe = np.array([par[s] for s in ("xx", "yy", "zz")])
    f, _ = R.bell_estimate(pe * n, np.full(3, n))
    assert f == pytest.approx((1 + 3 * p) / 4)
    rng = np.random.default_rng(7)
    draws = [R.bell_estimate(rng.binomial(n, pe), np.full(3, n)) for _ in range(2000)]
    est = np.array([d[0] for d in draws])
    assert est.mean() == pytest.approx((1 + 3 * p) / 4, abs=4 * est.std() / math.sqrt(2000) + 1e-9)
    if 0 < p < 1:
        assert np.mean([d[1] for d in draws]) == pytest.approx(est.std(), rel=0.1)


def test_twirled_readout_shrinks_the_correlators():
    """Symmetric (twirled) readout errors e_a, e_b multiply each correlator by
    (1 - 2 e_a)(1 - 2 e_b): conservative for F > 1/4."""
    rn = R.RefNoise(ref_readout=0.03, mem_readout=0.01)
    par = R.bell_parities(R.werner(0.8), R.I2, rn)
    s = (1 - 0.06) * (1 - 0.02)
    assert 2 * par["zz"] - 1 == pytest.approx(0.8 * s)
    assert par["fidelity"] == pytest.approx((1 + 3 * 0.8 * s) / 4)


def test_pm_estimator_se_and_power():
    """The stratified P&M SE matches the Monte Carlo spread; normal and MC power agree."""
    rng = np.random.default_rng(3)
    probs = {"p_succ": rng.uniform(0.55, 0.9, (6, 2)), "p_branch": np.ones((6, 2))}
    n = 400
    ests = [
        R.pm_estimate(rng.binomial(n, probs["p_succ"].ravel()), np.full(12, n)) for _ in range(3000)
    ]
    e = np.array([x[0] for x in ests])
    assert np.mean([x[1] for x in ests]) == pytest.approx(e.std(), rel=0.08)
    thr = float(probs["p_succ"].mean()) - 0.012
    eff = float(probs["p_succ"].mean()) - thr
    pn = R.power_normal(eff, R.pm_sd1(probs), 12 * n, 0.05)
    pmc = R.mc_power_pm(probs, n, thr, 0.05, 3000, 5)
    assert pmc == pytest.approx(pn, abs=0.05)
    assert R.shots_for_power(eff, R.pm_sd1(probs), 0.05, pn) == pytest.approx(12 * n, rel=1e-6)


def test_ratio_estimator_for_a_branch():
    rng = np.random.default_rng(4)
    sup = R.superop(_random_channel_kraus(rng))
    # a branch: half of the channel's Kraus operators
    kr = _random_channel_kraus(rng)
    br = R.superop(kr[:2])
    probs = R.pm_probabilities(br, R.I2, R.Spam())
    exact = probs["p_succ"].sum() / probs["p_branch"].sum()
    assert exact == pytest.approx(R.axis_average_fidelity(br, R.I2))
    assert R.pm_score(R.pm_probabilities(sup, R.I2, R.Spam())) == pytest.approx(
        R.axis_average_fidelity(sup, R.I2)
    )


@pytest.mark.parametrize("seed", range(6))
def test_pauli_design_reproduces_tr_wj(seed):
    """tr(W J) = c + sum alpha m for a trace-preserving channel and any Hermitian W."""
    rng = np.random.default_rng(400 + seed)
    sup = R.superop(_random_channel_kraus(rng))
    h = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    w = h + h.conj().T
    c, alpha = R.pauli_design(w)
    m = R.pauli_expectations(sup, R.Spam())
    assert c + float((alpha * m).sum()) == pytest.approx(
        float(np.trace(w @ R.choi(sup)).real), abs=1e-10
    )


def test_ppt_witness_is_a_witness():
    rng = np.random.default_rng(9)
    rho = R.werner(0.5)
    w, lam = R.ppt_witness(rho)
    assert float(np.trace(w @ rho).real) == pytest.approx(lam) and lam < 0
    for _ in range(50):
        a, b = _random_state(rng, 2), _random_state(rng, 2)
        assert float(np.trace(w @ np.kron(a, b)).real) >= -1e-12


def test_leakage_thresholds_are_worst_case():
    """Leaked shots scored at the most favourable outcome raise the EB bounds by l/3 (P&M)
    and l/2 (Bell)."""
    leak = 0.01
    assert (1 - leak) * 2 / 3 + leak == pytest.approx(R.pm_threshold(leak))
    # Bell: EB correlators (1/3 each, F = 1/2) mixed with leaked shots of correlators (1, -1, 1)
    c = [(1 - leak) * x + leak * y for x, y in zip((1 / 3, -1 / 3, 1 / 3), (1, -1, 1), strict=True)]
    assert R.fidelity_from_correlators(*c) == pytest.approx(R.bell_threshold(leak))


# -- local unitaries -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_optimal_correction_attains_the_closed_form_maximum(seed):
    rng = np.random.default_rng(500 + seed)
    rho = _random_state(rng)
    v = R.optimal_correction(rho)
    assert np.allclose(v.conj().T @ v, np.eye(2))
    assert R.bell_fidelity(rho, v) == pytest.approx(R.max_bell_fidelity(rho), abs=1e-12)
    best = max(R.bell_fidelity(rho, R.random_unitary(rng)) for _ in range(2000))
    assert best <= R.max_bell_fidelity(rho) + 1e-12
    assert best >= R.max_bell_fidelity(rho) - 0.02


@pytest.mark.parametrize("seed", range(6))
def test_rotation_unitary_round_trip(seed):
    rng = np.random.default_rng(600 + seed)
    u = R.random_unitary(rng)
    r = R.adjoint_rotation(u)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(R.adjoint_rotation(R.rotation_to_unitary(r)), r, atol=1e-12)


# -- the simulated step channel ---------------------------------------------------------------


def test_simulated_channel_at_zero_steps_is_the_ideal_bell_state(pieces):
    for key in ("noiseless", "live"):
        s = R.step_ops(pieces[key])
        sup0 = R.n_step_map(s, 0)
        assert np.allclose(R.choi(sup0), R.BELL)
        assert np.allclose(R.bell_final_state(sup0, 0, R.RefNoise()), R.BELL)
        assert R.pm_score(R.pm_probabilities(sup0, R.I2, R.Spam())) == pytest.approx(1.0)


def test_step_channel_is_trace_preserving_and_branches_sum(pieces):
    s = R.step_ops(pieces["live"])
    for n in (1, 2, 3):
        un = R.n_step_map(s, n)
        assert np.trace(R.choi(un)).real == pytest.approx(1.0)
        tot = sum(R.n_step_map(s, n, list(b)) for b in np.ndindex(*(2,) * n))
        assert np.allclose(tot, un, atol=1e-12)


def test_trained_instrument_breaks_entanglement_from_three_steps(pieces):
    """The noiseless trained instrument's unconditional channel is non-EB for one and two
    steps (entangled Choi state) but PPT, hence EB, from three steps; the A2 step is EB."""
    s = R.step_ops(pieces["noiseless"])
    neg = [R.negativity(R.choi(R.n_step_map(s, n))) for n in (1, 2, 3, 4)]
    assert neg[0] > 0.1 and neg[1] > 0.01
    assert neg[2] <= 1e-12 and neg[3] <= 1e-12
    a2 = R.n_step_map(s, 1, first=R.step_ops(pieces["noiseless"], "A2"))
    assert R.negativity(R.choi(a2)) <= 1e-12
    assert R.max_bell_fidelity(R.choi(R.n_step_map(s, 1))) > 0.5
    assert R.max_bell_fidelity(R.choi(R.n_step_map(s, 2))) < 0.5


def test_extra_dephasing_is_a_pure_z_flip(pieces):
    s0 = R.step_ops(pieces["live"])
    s1 = R.step_ops(pieces["live"], extra_dephasing=1.0)
    # full dephasing before the step: the one-step map equals the step after a Z measurement
    d = R.superop(A.dephase_kraus(1.0))
    assert np.allclose(R.n_step_map(s1, 1), R.n_step_map(s0, 1) @ d, atol=1e-12)


# -- reference candidates -----------------------------------------------------------------------


def test_reference_rule_on_a_toy_graph():
    dev = {
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5]],
        "cz": {k: {"error": 0.001} for k in ("0-1", "1-2", "2-3", "3-4", "1-5")},
        "qubits": {str(q): {"readout": 0.01, "t1": 3e-4, "t2": 2e-4} for q in range(6)},
    }
    rows = {
        r["qubit"]: r
        for r in R.reference_candidates(
            dev, 2, 1, R.RefRule(min_dist_output=3, max_dist_memory=3, forbid_path=(1,))
        )
    }
    assert not rows[3]["admitted"] and "memory" in rows[3]["excluded_because"]
    assert not rows[0]["admitted"] and "output" in rows[0]["excluded_because"]
    assert rows[4]["admitted"] and rows[4]["path"] == "2-3-4" and rows[4]["swaps"] == 1
    assert rows[4]["cz_error_sum"] == pytest.approx(0.001 + 3 * 0.001)


def test_committed_reference_choice():
    """On the committed (FakeKingston) map: q74 is excluded (coupler to q75), q73 by its
    readout, and the rule picks q79."""
    rows = {int(r["qubit"]): r for r in _csv(SD / "references.csv")}
    assert rows[74]["admitted"] == "False"
    assert "readout" in rows[73]["excluded_because"]
    summ = json.loads((SD / "summary.json").read_text())
    assert summ["reference"]["pick"]["qubit"] == 79
    assert summ["plan"]["submitted"] is False


def _csv(path):
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


# -- QASM -----------------------------------------------------------------------------------------


def test_committed_qasm_matches_its_hashes():
    man = json.loads((SD / "qasm" / "manifest.json").read_text())
    assert man["max_noiseless_check"] < 1e-9
    assert len(man["circuits"]) == 112
    for name, c in man["circuits"].items():
        text = (SD / "qasm" / f"{name}.qasm").read_text()
        assert R.sha256_text(text) == c["sha256"], name
        assert c["noiseless_check"] < 1e-9


def test_rebuilt_qasm_matches_the_committed_files():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit import qasm3, transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    summ = json.loads((SD / "summary.json").read_text())
    man = json.loads((SD / "qasm" / "manifest.json").read_text())["circuits"]
    v = {int(n): np.array(m["re"]) + 1j * np.array(m["im"]) for n, m in summ["targets_V"].items()}
    step = qasm3.loads((ROOT / "results/summaries/phase6/step.qasm").read_text())
    pick = summ["reference"]["pick"]
    path = [int(q) for q in pick["path"].split("-")]
    be = FakeKingston()
    cases = {
        "pm_n1_px_b0": (R.pm_circuit(step, 1, 2, 0, v[1]), [59, 75]),
        "pm_neg_my_b1": (R.pm_circuit(step, 1, 5, 1, v[1], negative=True), [59, 75]),
        "bell_n1_yy_f1": (R.bell_circuit(step, 1, "yy", 1, v[1], pick["swaps"]), [59, *path]),
        "mdi_n1_x2_y3": (R.mdi_circuit(step, 1, 2, 3), [59, 75, 74]),
    }
    for name, (qc, lay) in cases.items():
        tq = transpile(qc, be, initial_layout=lay, optimization_level=0, seed_transpiler=0)
        assert R.sha256_text(qasm3.dumps(tq)) == man[name]["sha256"], name
