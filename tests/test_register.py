"""Phase 8 Stage 2b (#34): the phase-register model, frozen-map EM, the pipeline statistics."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from qmem import classical as C
from qmem import coherence as Q
from qmem import reanalysis as RA
from qmem import register as RG

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

ARMS = {
    "E0": C.ArmSpec("E0", (("e0",),)),
    "P3": C.ArmSpec("P3", (("e3",),)),
    "E00": C.ArmSpec("E00", (("e0", "e0"),)),
    "INV2": C.ArmSpec("INV2", (("e2", "e6"),)),
    "COMP22": C.ArmSpec("COMP22", (("e2", "e2"),)),
    "COMP40": C.ArmSpec("COMP40", (("e4", "e0"),)),
    "ALT2": C.ArmSpec("ALT2", (("e2",), ("e6",)), "alternate"),
    "R2": C.ArmSpec(
        "R2",
        (("ff00", "e0", "e0"), ("ff01", "e0", "e0"), ("ff10", "e4", "e0"), ("ff11", "e4", "e0")),
        "random",
    ),
    "A2": C.ArmSpec("A2", (("meas",),)),
}
ELEMENTS = [f"e{k}" for k in range(8)] + ["ff00", "ff01", "ff10", "ff11"]


def model(n_pop: int = 2, seed: int = 0) -> C.ProgramModel:
    return RG.random_register(n_pop, ELEMENTS, np.random.default_rng(seed), ["meas"])


def sample(m: C.ProgramModel, arms, n: int, length: int, seed: int) -> dict[str, C.ProgramData]:
    import phase8_coherence_null as NL

    rng = np.random.default_rng(seed)
    return {a: NL.simulate_program(m, arm, n, length, rng) for a, arm in arms.items()}


def test_echo_shifts_form_the_phase_group():
    for n_pop in (1, 2, 3):
        for a in range(8):
            for b in range(8):
                lhs = RG.shift(a, n_pop) @ RG.shift(b, n_pop)
                assert np.array_equal(lhs, RG.shift((a + b) % 8, n_pop))
        assert np.array_equal(RG.shift(8, n_pop), np.eye(8 * n_pop))


def test_register_composes_inverse_and_composition_arms_exactly():
    """The Lemma's escape: with permutation echoes the classical model makes INV2 = E00,
    COMP22 = COMP40 exactly, whatever the free maps."""
    m = model(2, 3)
    assert np.array_equal(m.bank(ARMS["INV2"]), m.bank(ARMS["E00"]))
    assert np.array_equal(m.bank(ARMS["COMP22"]), m.bank(ARMS["COMP40"]))
    # ALT2's slots are shift(2) and shift(6) = shift(2)^-1 (the steps between them are free)
    assert np.array_equal(m.slot(("e2",)) @ m.slot(("e6",)), np.eye(m.d))


def test_register_likelihood_against_path_enumeration():
    rng = np.random.default_rng(5)
    for name in ("P3", "INV2", "A2", "ALT2"):
        arm = ARMS[name]
        m = model(1 if name == "A2" else 2, 1)  # A2's dense meas map: the d = 8 register
        for _ in range(2):
            y = rng.integers(0, 2, 2)
            mm = rng.integers(0, 2, 2) if name == "A2" else None
            d = C.ProgramData(y[None], None, None if mm is None else mm[None])
            if arm.schedule == "alternate":
                d = C.ProgramData(y[None], np.array([[0, 1]]))
            brute = C.brute_force_program_prob(
                m, arm, list(y), None if d.choice is None else list(d.choice[0]), mm
            )
            assert math.isclose(math.exp(-m.nll(arm, d)[0]), brute, rel_tol=1e-10)
    # a random arm with recorded bits (one step: the ff maps are dense)
    small = model(1, 2)
    for ch in range(4):
        for y in (0, 1):
            d = C.ProgramData(np.array([[y]]), np.array([[ch]]))
            brute = C.brute_force_program_prob(small, ARMS["R2"], [y], [ch])
            assert math.isclose(math.exp(-small.nll(ARMS["R2"], d)[0]), brute, rel_tol=1e-10)


def test_prism_embeds_the_ideal_rotation_as_the_shift():
    """In the prism's coordinates the Bloch map of Rz(2 pi k / 8) is exactly shift(k): the
    register start reproduces the quantum echo."""
    for n_pop in (2, 3):
        u = RG.prism(n_pop, 1 / math.cos(math.pi / 8))
        for k in range(8):
            a = 2 * math.pi * k / 8
            t = np.eye(4)
            t[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
            # T u = u S^T for the column convention; qmem.coherence.polytope_start's
            # (pinv(u) T u)^T is S when the vertex set is invariant
            assert np.allclose(t @ u, u @ RG.shift(k, n_pop).T, atol=1e-12)


def test_frozen_em_keeps_the_permutations_and_increases_the_likelihood():
    truth = model(2, 7)
    data = sample(truth, ARMS, 60, 12, 8)
    start = model(2, 9)
    frozen = RG.echo_names(start)
    lls = []
    m = start
    for _ in range(4):
        new, ll = C.program_em_step(m, ARMS, data, frozen)
        lls.append(ll)
        m = new
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(lls))
    assert RG.is_register(m, 2)
    fit = C.fit_program_model(ARMS, data, start, 12, frozen=frozen)
    assert RG.is_register(fit.model, 2)
    assert -fit.nll >= lls[0]
    # frozen non-echo parts too (generator (b)'s construction)
    fit2 = C.fit_program_model(ARMS, data, start, 6, frozen=["init", "pre", "post", "meas"])
    for k in ("init", "pre", "post"):
        assert np.array_equal(getattr(fit2.model, k), getattr(start, k))
    assert np.array_equal(fit2.model.labelled["meas"], start.labelled["meas"])
    assert not np.array_equal(fit2.model.maps["e3"], start.maps["e3"])


def test_free_parameter_count():
    m = model(2)
    d = 16
    free = (d - 1) + d * (d - 1) + d * (2 * d - 1) + 4 * d * (d - 1) + d * (2 * d - 1)
    assert RG.n_free_params(m) == free
    assert m.n_params() == free + 8 * d * (d - 1)


def test_block_bootstrap_matches_the_shot_se_for_independent_shots():
    sp = Q.PlanSplit(400, 2, 8)
    rng = np.random.default_rng(0)
    keep = sp.test()
    diffs = [rng.normal(0.1, 1.0, keep.sum()) for _ in range(6)]
    b = sp.blocks(keep)
    j = sp.job()[keep]
    r = Q.block_bootstrap(diffs, [b] * 6, [j] * 6, 2000, 1)
    assert keep.sum() == 400 and len(np.unique(b)) == 100  # 4 test shots per 8-shot segment
    assert r["n"] == 2400
    assert abs(r["se_block"] / r["se_shot"] - 1) < 0.1
    # correlated within blocks: the block SE grows, the shot SE does not see it
    shared = np.repeat(rng.normal(0, 1, 100), 4)
    corr = [x + 2 * shared for x in diffs]
    rc = Q.block_bootstrap(corr, [b] * 6, [j] * 6, 2000, 1)
    assert rc["se_block"] > 1.5 * rc["se_shot"]


def test_holm():
    assert Q.holm({"d4": 0.01, "d6": 0.04}) == {"d4": True, "d6": True}
    assert Q.holm({"d4": 0.03, "d6": 0.04}) == {"d4": False, "d6": False}
    assert Q.holm({"d4": 0.2, "d6": 0.02}) == {"d4": False, "d6": True}


def test_clopper_pearson():
    import phase8_coherence_null as NL

    lo, hi = NL.clopper_pearson(0, 10)
    assert lo == 0 and abs(hi - 0.3084971) < 1e-5
    lo, hi = NL.clopper_pearson(1, 20)
    assert abs(lo - 0.0012650) < 1e-5 and abs(hi - 0.2487328) < 1e-5


def test_vanish_rule_power_reduces_to_the_single_job_formula():
    """One job: P(|X| + z SE < f |w_S|) with w_S known (sd_s = 0) is #29's vanish_power."""
    w_s, sd_r, n = -0.006, 0.05, 40000
    se = sd_r / math.sqrt(n)
    for w_r in (0.0, 0.0002):
        mc = Q.vanish_lojo_power(w_r, sd_r, w_s, 0.0, n, 1, 0.1, 1.96, 200000, 3)
        exact = RA.vanish_power(w_r, se, 0.1 * abs(w_s), 1.96)
        assert abs(mc - exact) < 0.005
    # leave-one-job-out can only make it harder
    two = Q.vanish_lojo_power(0.0, sd_r, w_s, 0.0, n, 2, 0.1, 1.96, 200000, 3)
    assert two <= Q.vanish_lojo_power(0.0, sd_r, w_s, 0.0, n, 1, 0.1, 1.96, 200000, 3) + 1e-9


@pytest.mark.parametrize("two_sided", [True, False])
def test_ratio_power_against_monte_carlo(two_sided):
    wa, sa, wb, sb, n, m, z = -0.0060, 0.06, -0.0065, 0.06, 60000, 0.2, 1.6448536269514722
    p = Q.ratio_power(wa, sa, wb, sb, n, m, z, two_sided)
    rng = np.random.default_rng(1)
    a = wa + sa / math.sqrt(n) * rng.standard_normal(40000)
    b = wb + sb / math.sqrt(n) * rng.standard_normal(40000)
    r = a / b
    se = np.abs(r) * np.sqrt((sa / a) ** 2 + (sb / b) ** 2) / math.sqrt(n)
    ok = r - z * se > 1 - m
    if two_sided:
        ok &= r + z * se < 1 + m
    assert abs(ok.mean() - p) < 0.03
    nn = Q.ratio_shots(wa, sa, wb, sb, m, z, 0.8, two_sided)
    assert abs(Q.ratio_power(wa, sa, wb, sb, nn, m, z, two_sided) - 0.8) < 1e-3


def test_controls_circuits():
    """#34's controls tier: the benchmark keeps every echo pulse and its repetitions; the Knee
    circuits prepare and intervene in step 0's slot only (hardware group)."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    import phase8_coherence_hw as HW
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    backend = FakeKingston()
    reps = 8
    for name, x_per_rep in (("e0", 2), ("e4", 2), ("R2", 4), ("A2", 0)):
        for st, extra in (("0", 0), ("1", 1)):
            qc = HW.benchmark_circuit(76, name, st, reps, 4.9e-6, (50, 21))
            tq = HW.isa(qc, backend, 0)
            ops = HW.q75_ops(tq)
            n_x = sum(1 for n, _ in ops if n == "x")
            # R2: both if_else branches are listed (2 x 2 pulses) plus the e0 after them
            want = reps * (x_per_rep + (2 if name == "R2" else 0)) + extra
            assert n_x == want, (name, st, n_x, want)
            assert sum(1 for n, _ in ops if n == "measure") == 1 + (reps if name == "A2" else 0)
    from qiskit import qasm3

    from qmem.records import REPO_ROOT

    full = qasm3.loads((REPO_ROOT / "results/summaries/phase6/circuit.qasm").read_text())
    for prog, n_x in ((["e4"], 2), (["meas"], 0), (["e0"], 2)):
        kc = HW.knee_circuit(full, "+", prog, 64)
        ops = HW.q75_ops(HW.isa(kc, backend, 0))
        base = HW.q75_ops(HW.isa(full, backend, 0))
        assert sum(1 for n, _ in ops if n == "reset") == sum(1 for n, _ in base if n == "reset") + 1
        assert sum(1 for n, _ in ops if n == "x") == sum(1 for n, _ in base if n == "x") + n_x
