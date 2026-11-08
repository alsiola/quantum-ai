"""Phase 7 Stage 3 (#24): schedule, guards, decision rule, models and circuits of the follow-up.

The Qiskit parts (QASM hashes, re-timing) skip without the ``hardware`` group. Nothing here
touches the network: the submission test replaces the live service by a function that fails.
"""

from __future__ import annotations

import csv
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from qmem import artefacts as AR
from qmem import followup as F
from qmem import witness as W
from qmem.records import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
import phase7_hw as H

CFG_PATH = REPO_ROOT / "experiments" / "configs" / "phase7_hw.toml"


@pytest.fixture(scope="module")
def cfg():
    return H.load_config(CFG_PATH)


# -- schedule ------------------------------------------------------------------------------------


def test_schedule_deterministic_and_complete(cfg):
    a = H.schedule(cfg)
    b = H.schedule(cfg)
    assert F.schedule_rows(a) == F.schedule_rows(b)
    arms, inst = H.plan_arms(cfg)
    s = cfg["schedule"]
    assert F.check_schedule(a, arms, s["tier1"], s["max_chunk"], inst) == []
    assert len(a) >= 4
    for job in a:  # every tier-1 arm in every job, at most 1,000 shots per pub
        assert set(s["tier1"]) <= {p.arm for p in job}
        assert max(p.shots for p in job) <= 1000
    every = {p.arm for job in a for p in job}
    assert every == set(arms) | set(inst)
    other = F.make_schedule(arms, s["n_jobs"], s["max_chunk"], s["seed"] + 1, inst)
    assert F.schedule_rows(other) != F.schedule_rows(a)


def test_schedule_split_and_checks():
    assert F.split_even(15000, 4) == [3750, 3750, 3750, 3750]
    assert F.split_even(10, 4) == [3, 3, 2, 2]
    jobs = F.make_schedule({"x": 2500, "y": 300}, 2, 1000, 0, {"z": (5, 64)})
    assert [p.shots for p in jobs[0] if p.arm == "x"] == [625, 625]
    assert sorted(p.instance for j in jobs for p in j if p.arm == "z") == list(range(5))
    probs = F.check_schedule(jobs, {"x": 2500, "y": 301}, ["x", "w"], 1000, {"z": (5, 64)})
    assert any("lacks tier-1 arm w" in p for p in probs)
    assert any("y: 300 shots scheduled" in p for p in probs)
    assert any("jobs < 4" in p for p in probs)


# -- budget and submission guards ----------------------------------------------------------------


def test_cost_model_matches_phase6():
    cm = F.CostModel()
    pubs = [F.Pub(0, 0, "a", 0, 20000)]
    c = cm.job(pubs, {"a": 3.08316e-4})
    assert c["estimate_s"] == pytest.approx(2 + 20000 * (250e-6 + 3.08316e-4))  # 13.17 s
    assert c["upper_s"] == pytest.approx(1.25 * c["estimate_s"] + 1.0)


def test_budget_guard_refuses():
    ok = dict(
        est_s=10.0,
        upper_s=13.5,
        max_qpu_seconds=20.0,
        remaining_s=584,
        plan="Open",
        spent_upper_s=0.0,
        hard_cap_s=60,
    )
    F.budget_check(**ok)  # passes
    for change, msg in (
        ({"est_s": 20.5}, "max_qpu_seconds"),
        ({"plan": "paygo"}, "never a paid plan"),
        ({"plan": None}, "never a paid plan"),
        ({"remaining_s": None}, "unknown"),
        ({"remaining_s": 13}, "remaining allowance"),
        ({"spent_upper_s": 47.0}, "hard cap"),
    ):
        with pytest.raises(F.Refused, match=msg):
            F.budget_check(**{**ok, **change})


def test_every_planned_job_within_guard(cfg):
    bt = H.budget_table(cfg)
    assert bt["every_job_within_max"] and bt["max_job_estimate_s"] <= 20.0
    assert bt["plan_upper_within_cap"]
    tot = next(r for r in bt["rows"] if r["item"] == "pilot + default plan")
    assert tot["upper_s"] <= cfg["budget"]["hard_cap_s"] < cfg["budget"]["allowance_after_16_s"]
    for r in bt["rows"]:
        if r["item"].startswith("tier 3"):
            assert (r["estimate_s"] - 2.0 * r["jobs"]) / r["jobs"] + 2.0 <= 20.0


def test_submission_refused():
    url = "https://github.com/alsiola/quantum-ai/issues/17#issuecomment-123"
    F.submission_check(url, True)
    for a, s in (("", True), (None, True), ("https://example.com/x", True), (url, False)):
        with pytest.raises(F.Refused):
            F.submission_check(a, s)


def test_script_refuses_before_any_network(cfg, monkeypatch, tmp_path):
    def no_network(_cfg):
        raise AssertionError("the live service must not be reached")

    monkeypatch.setattr(H, "live", no_network)
    args = SimpleNamespace(submit=True, from_run=tmp_path)
    assert cfg["ibm"]["approval_url"] == ""  # this issue never sets it
    with pytest.raises(F.Refused, match="approval_url"):
        H.stage4(cfg, args)
    # with an approval URL but no --submit there is no submission path at all: --dry-run
    with pytest.raises(F.Refused, match="--submit"):
        F.submission_check("https://github.com/a/b/issues/17#issuecomment-1", False)


def test_calibration_bounds():
    ref = {"q75": {"readout_error": 0.015, "t1": 3e-4, "t2": 1.4e-4}, "cz": {"cz_error": 1e-3}}
    b = F.CalibrationBounds()
    assert F.calibration_check(ref, ref, b) == []
    live = {
        "q75": {"readout_error": 0.031, "t1": 1.4e-4, "t2": 1.4e-4},
        "cz": {"cz_error": 2.5e-3},
        "q74": {"readout_error": 0.06, "t1": 4e-5, "t2": 1e-4},
    }
    v = F.calibration_check(live, ref, b)
    assert any(s.startswith("q75.readout_error") for s in v)
    assert any(s.startswith("q75.t1") for s in v)
    assert any(s.startswith("cz.cz_error") for s in v)
    assert any(s.startswith("q74.readout_error") for s in v)  # absolute ceiling, no reference
    assert any(s.startswith("q74.t1") for s in v)
    assert not any(s.startswith("q75.t2") for s in v)


# -- analysis pieces -----------------------------------------------------------------------------


def test_bootstrap_and_lag1_match_library():
    rng = np.random.default_rng(0)
    shots = (rng.random((300, 64)) < 0.55).astype(np.uint8)
    per = W.per_shot_windows(shots, 7, 16, common=True)
    idx = W.bootstrap_indices(len(per), 20, 3)
    assert np.allclose(F.boot_g(per, idx), W.bootstrap_g(per, idx), atol=1e-15)
    lc = F.lag1_counts(shots, 16).sum(0)
    assert F._lag1_from(lc)[0] == pytest.approx(AR.lag1_excess_data(shots, 16), abs=1e-15)
    segs = F.strata(np.repeat([2, 0, 1], 100))
    assert [len(s) for s in segs] == [100, 100, 100] and segs[0][0] == 0
    idx = F.stratified_indices(segs, 500, 5, 1)
    for k, s in enumerate(segs):  # resamples stay inside their stratum
        assert set(np.unique(idx[:, k * 100 : (k + 1) * 100])) <= set(s)


def test_holm():
    rej = F.holm({"a": 0.001, "b": 0.004, "c": 0.02, "d": 0.003}, 0.01)
    assert rej == {"a": True, "d": True, "b": True, "c": False}
    # step-down: once one fails (0.006 > 0.01 / 2) every larger p is retained
    assert F.holm({"a": 0.006, "b": 0.001, "c": 0.007}, 0.01) == {"b": True, "a": False, "c": False}
    assert F.holm({"a": 0.0024, "b": 0.0049}, 0.01) == {"a": True, "b": True}


def _arm(w0, se0, lag1=0.0, lag1_se=1e-3):
    return {"w0": w0, "se0": se0, "lag1": lag1, "lag1_se": lag1_se}


def test_decision_rule_exact_cases():
    r = F.DecisionRule()
    base = _arm(-0.006, 1e-4)
    cases = {
        "vanishes": _arm(0.0, 1e-4),  # |w| + 1.96 SE = 2e-4 < 6e-4
        "persists": _arm(-0.003, 1e-4),  # 30 SE, 50% of the baseline
        "ambiguous": _arm(-0.0008, 2e-4),  # 4 SE: neither bound
    }
    for want, a2 in cases.items():
        d = F.decide({"baseline": base, "A2": a2}, r)
        assert d.gate and d.primary == want and d.reported == want
    # edge: 5 SE but below 25% of the baseline is ambiguous, not persists
    assert F.primary_outcome(base, _arm(-0.0012, 1e-4), r) == "ambiguous"
    # the replication gate: z = -3, or the wrong sign
    for b in (_arm(-0.0003, 1e-4), _arm(0.006, 1e-4)):
        d = F.decide({"baseline": b, "A2": _arm(0.0, 1e-5)}, r)
        assert not d.gate and d.reported == "uninterpretable"
    # segment sensitivity: a flagged job that changes the outcome makes it ambiguous
    unf = {"baseline": base, "A2": _arm(-0.003, 1e-4)}
    d = F.decide({"baseline": base, "A2": _arm(0.0, 1e-4)}, r, unf, [2])
    assert d.primary_all_jobs == "vanishes" and d.primary == "ambiguous"


def test_secondary_family():
    r = F.DecisionRule()
    arms = {
        "baseline": _arm(-0.006, 1e-4),
        "A2": _arm(0.0, 1e-4),
        "A2s_spec": _arm(-0.003, 1e-4),  # 21 sigma below the baseline
        "A1": _arm(1e-5, 1e-4),
        "D1": _arm(0.0, 1e-6, lag1=0.026, lag1_se=1e-3),
    }
    d = F.decide(arms, r, ref_values={"D1_lag1_vs_prediction": -0.0009})
    s = d.secondary
    assert set(s) == {"A2s_spec_vs_baseline", "A1_vs_0", "D1_w0_vs_0", "D1_lag1_vs_prediction"}
    assert s["A2s_spec_vs_baseline"]["reject_holm"]
    assert s["A2s_spec_vs_baseline"]["ratio"] == pytest.approx(0.5)
    assert s["D1_lag1_vs_prediction"]["reject_holm"]
    assert s["D1_lag1_vs_prediction"]["difference"] == pytest.approx(0.0269)
    assert not s["A1_vs_0"]["reject_holm"] and not s["D1_w0_vs_0"]["reject_holm"]


# -- models --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exact(cfg):
    names = ("H_q75", "H_outside", "H_ancilla_fit", "H_spec_typical")
    return {h["name"]: H.exact_hypothesis(cfg, h) for h in cfg["hypotheses"] if h["name"] in names}


def test_h_q75_predictions_match_stage2(cfg, exact):
    """Phase 6 / #20 / #23 numbers: baseline w(0) -0.00690, A1 and A2 exactly 0, A3's realised
    64 instances -7.1e-5, D1 w = 0 with lag-1 excess (1 - 2p)(P(0|1) - P(1|0))."""
    e = exact["H_q75"]
    assert e["baseline"]["w0"] == pytest.approx(-0.006899, abs=2e-6)
    assert e["A2s_spec"]["w0"] == pytest.approx(e["baseline"]["w0"], abs=1e-15)
    assert abs(e["A1"]["w0"]) < 1e-12 and abs(e["A2"]["w0"]) < 1e-12
    assert e["A3"]["w0"] == pytest.approx(-7.1e-5, abs=0.2e-5)
    assert abs(e["D1"]["w0"]) < 1e-12
    q = H.snapshot_q59(cfg)
    d1 = H.d1_process(cfg)
    p = float(d1.marginals(64)[16:].mean())
    assert e["D1"]["lag1"] == pytest.approx((1 - 2 * p) * (q["e10"] - q["e01"]), abs=2e-4)


def test_ancilla_models_reduce_to_calibrated(cfg, exact):
    """#21's simulator with default parameters against #20's joint pieces: the baseline and
    every A3 Pauli step agree to 1e-12. A1 and A2 agree to 1e-4: the two simulators treat
    the idling during q75's own reset or measurement slightly differently (1e-5 in P(1)),
    and both give w = 0 exactly."""
    m = H.ancilla_models(cfg, {})
    a = H.ablate_models(cfg, H.A.Hypothesis())
    for arm, tol in (("baseline", 1e-12), ("A2", 1e-4), ("A1", 1e-4)):
        ga = m.procs[arm].expected_windows(7, 16, 64, True)
        gb = a.procs[arm].expected_windows(7, 16, 64, True)
        assert np.allclose(ga, gb, atol=tol), arm
        assert np.allclose(m.procs[arm].marginals(64), a.procs[arm].marginals(64), atol=tol)
    for arm in ("A2", "A1"):
        assert abs(W.witness(m.procs[arm].expected_windows(7, 16, 64, True), 0)) < 1e-12
    for p in range(4):
        assert np.allclose(m.bank[p].marginals(64), a.bank[p].marginals(64), atol=1e-12)


def test_reduced_spectator_matches_stage2c(cfg):
    """The reduced Stark model (a per-step q75 phase over q59's two readout windows) against
    #22's time-resolved simulator (``sweep_readout_crosstalk.csv``, uniform shift)."""
    rows = list(
        csv.DictReader(
            open(REPO_ROOT / cfg["experiment"]["spectator_dir"] / "sweep_readout_crosstalk.csv")
        )
    )
    base = H.ablate_models(cfg, H.A.Hypothesis()).procs["baseline"]
    w_ref = W.witness(base.expected_windows(7, 16, 64, True), 0)
    t_ro = 2.18e-6 + 2.212e-6
    for f in (-30e3, -10e3, 10e3, 30e3):
        r = next(
            x
            for x in rows
            if x["stark_hz"] and float(x["stark_hz"]) == f and x["kind"] == "uniform"
        )
        hyp = H.A.Hypothesis(q75_phase=-2 * math.pi * f * t_ro)
        p = H.ablate_models(cfg, hyp).procs["baseline"]
        ratio = W.witness(p.expected_windows(7, 16, 64, True), 0) / w_ref
        assert ratio == pytest.approx(float(r["w0_ratio"]), abs=0.01), f


def test_outside_hypothesis_keeps_witness(exact):
    e = exact["H_outside"]
    for arm in ("A2", "A2s_spec", "A1", "A3"):
        assert e[arm]["w0"] == pytest.approx(e["baseline"]["w0"], abs=1e-12)
    assert e["baseline"]["w0"] < -0.005


def test_episode_sampling_places_regime_b(cfg):
    """Shots in the episode's time window come from regime B, contiguous in time order."""
    h = next(x for x in cfg["hypotheses"] if x["name"] == "H_drift_episode")
    m = H.build_models(cfg, h)
    jobs = F.make_schedule({"baseline": 2000, "A2": 2000}, 4, 1000, 0)
    b, w, start = m.episode
    # mark regime B with an always-one process so its shots are recognisable
    ones = W.LinearProcess.hmm(np.array([[[0.0]], [[1.0]]]), np.ones(1))
    b.procs = {k: ones for k in b.procs}
    data = H.sample_plan(cfg, m, jobs, 0)
    total = 4000
    lo, hi = int(start * total), int((start + w) * total)
    n_b = sum(int((d["shots"].min(1) == 1).sum()) for d in data.values())
    assert n_b == hi - lo


# -- the full pipeline on synthetic data ---------------------------------------------------------


def test_pipeline_predicted_outcomes(cfg, exact):
    """At the planned sizes the full pipeline gives the predicted outcome on one dataset each
    of H_q75 (vanishes) and H_outside (persists); bootstrap and permutations reduced."""
    q = exact["H_q75"]
    marg = {a: np.asarray(q[a]["marginals"]) if "marginals" in q[a] else None for a in H.ARMS}
    refs = {"A3_vs_prediction": q["A3"]["w0"], "D1_lag1_vs_prediction": q["D1"]["lag1"]}
    s = H.analysis_settings(cfg, validation=True)
    s.n_boot, s.n_perm = 100, 20
    jobs = H.schedule(cfg)
    for name, want in (("H_q75", "vanishes"), ("H_outside", "persists")):
        h = next(x for x in cfg["hypotheses"] if x["name"] == name)
        data = H.sample_plan(cfg, H.build_models(cfg, h), jobs, 11)
        res = H.analyse_dataset(cfg, data, marg, s, refs, 5)
        d = res["decision"]
        assert d.gate and d.primary_all_jobs == want, name
        assert all(not g["integrity_fail"] for g in res["guards"].values())


# -- circuits (hardware group) -----------------------------------------------------------------


def test_qasm_hashes_match_stage2a(cfg):
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    backend = FakeKingston()
    built = H.build_circuits(cfg, backend, "snapshot")
    hc = H.check_hashes(cfg, built)
    assert hc["all_match"], {k: v for k, v in hc["rows"].items() if not v["match"]}
    assert len([k for k in hc["rows"] if k.startswith("A3_")]) == 64
    retimed = H.build_circuits(cfg, backend, "target")
    cmp = H.compare_retimed(built, retimed)
    assert all(v["differs_only_in_delays"] for v in cmp.values())
    assert [k for k, v in cmp.items() if not v["unchanged"]] == ["A2s_spec"]
    assert retimed["delays_s"]["measure"] == backend.target["measure"][(75,)].duration


def test_load_run_reassembles_time_order(cfg, tmp_path):
    """Stage 4's loader puts each arm's pubs back in (job, position) order with labels."""
    jobs = H.schedule(cfg)
    rng = np.random.default_rng(1)
    want: dict[str, list] = {}
    for j, job in enumerate(jobs):
        arrays = {}
        for p in job:
            y = (rng.random((p.shots, 64)) < 0.5).astype(np.uint8)
            arrays[f"{p.key}__y"] = np.packbits(y, axis=1)
            want.setdefault(p.arm, []).append(y)
            if p.arm == "A2":
                arrays[f"{p.key}__m"] = np.packbits(1 - y, axis=1)
        (tmp_path / f"job{j}").mkdir()
        np.savez_compressed(tmp_path / f"job{j}" / "shots.npz", length=64, **arrays)
    data = H.load_run(cfg, tmp_path)
    for arm, parts in want.items():
        assert np.array_equal(data[arm]["shots"], np.concatenate(parts))
        assert np.all(np.diff(data[arm]["jobs"]) >= 0)
    assert np.array_equal(data["A2"]["m"], 1 - data["A2"]["shots"])
