"""Phase 8 Stage 1 (#31): pair selection, the per-pair rule, TOST and the power engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import followup as F
from qmem import reanalysis as RA
from qmem import replication as REP
from qmem import witness as W

ROOT = Path(__file__).resolve().parents[1]
DEVICE = ROOT / "results" / "summaries" / "phase7_spectator" / "device.json"
CFG = ROOT / "experiments" / "configs" / "phase8_replication.toml"
SUM = ROOT / "results" / "summaries" / "phase8_replication"
HW4C = ROOT / "results" / "summaries" / "phase7_hw_4c"


def snapshot() -> dict:
    return json.loads(DEVICE.read_text())


def calibration_power(row: dict) -> dict[str, float]:
    """A deterministic stand-in for the pair model's power: calibration only."""
    t2 = row["memory_t2_us"]
    score = 1 - math.exp(-t2 / 150) * (1 + 10 * row["output_readout"])
    return {"power_joint": score, "snr": t2 / 1000}


# -- selection ----------------------------------------------------------------------------------


def test_selection_is_deterministic_and_ignores_witness_inputs():
    rules = REP.SelectionRules()
    s1 = REP.select(REP.Device.from_snapshot(snapshot()), rules, calibration_power)
    s2 = REP.select(REP.Device.from_snapshot(snapshot()), rules, calibration_power)
    pick = lambda s: [(r["memory"], r["output"], r["spectator"]) for r in s.chosen]  # noqa: E731
    assert pick(s1) == pick(s2) and len(pick(s1)) == rules.n_pairs

    # witness-like data planted everywhere in the snapshot cannot reach the selection
    snap = snapshot()
    snap["witness"] = {"67-68": -0.1}
    rng = np.random.default_rng(0)
    for q in snap["qubits"].values():
        q["w0"] = float(rng.normal())
        q["p1_observed"] = float(rng.random())
        q["pilot_w0"] = -1.0
    for c in snap["cz"].values():
        c["w0"] = float(rng.normal())
    dev = REP.Device.from_snapshot(snap)
    assert not any(hasattr(q, "w0") for q in dev.qubits.values())
    s3 = REP.select(dev, rules, calibration_power)
    assert pick(s3) == pick(s1)
    assert [r["reasons"] for r in s3.rows] == [r["reasons"] for r in s1.rows]


def test_screen_lists_every_directed_edge_with_reasons():
    dev = REP.Device.from_snapshot(snapshot())
    rows = REP.screen(dev, REP.SelectionRules())
    assert len(rows) == 2 * len(dev.edges)
    by = {(r["memory"], r["output"]): r for r in rows}
    for pair in REP.SelectionRules().named:
        assert by[pair]["screen_reasons"] == []
        assert by[pair]["named_candidate"]
    # the reference pair and its neighbours are excluded for their distance
    assert any("hops" in w for w in by[(75, 59)]["screen_reasons"])
    assert any("hops" in w for w in by[(74, 73)]["screen_reasons"])
    # q149's only other neighbour is q150: it is the spectator
    assert by[(149, 148)]["spectator"] == 150
    assert set(dev.neighbours()[67]) == {57, 66, 68}


def test_selection_spacing_and_power_threshold():
    dev = REP.Device.from_snapshot(snapshot())
    rules = REP.SelectionRules(n_pairs=3)
    sel = REP.select(dev, rules, calibration_power)
    foots = [[r["memory"], r["output"], r["spectator"]] for r in sel.chosen]
    for i, a in enumerate(foots):
        for b in foots[i + 1 :]:
            d = dev.distances(a)
            assert min(d[q] for q in b) >= rules.min_distance_pairs
    # a power function below the threshold everywhere: nothing qualifies, and it says so
    none = REP.select(dev, rules, lambda r: {"power_joint": 0.5, "snr": 1.0})
    assert none.chosen == [] and none.statement == "no pair qualifies"
    assert all(r["status"] == "excluded" for r in none.rows)


# -- models -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_pieces() -> A.StepPieces:
    return A.StepPieces.from_npz(np.load(HW4C / "pieces_live.npz"), "live")


def test_truth_model_reduces_to_the_calibrated_one_and_hits_its_shortfall(live_pieces):
    s = A.Settings()
    for arm in REP.ARMS:
        assert np.allclose(
            REP.truth_superops(arm, live_pieces, 0.0), A.step_superops(arm, live_pieces)
        )
    p = REP.shortfall_dephasing(live_pieces, 0.75, s)
    w_cal = A.predict(REP.arm_process("baseline", live_pieces), s)["w0"]
    w = A.predict(REP.arm_process("baseline", live_pieces, p), s)["w0"]
    assert w / w_cal == pytest.approx(0.75, abs=1e-8)
    # A2 still vanishes exactly, and the shams are the baseline, under the truth model
    assert abs(A.predict(REP.arm_process("A2", live_pieces, p), s)["w0"]) < 1e-15
    for sham in REP.SHAMS:
        assert A.predict(REP.arm_process(sham, live_pieces, p), s)["w0"] == pytest.approx(w)


def test_shot_influence_matches_the_bootstrap(live_pieces):
    proc = REP.arm_process("baseline", live_pieces)
    shots = proc.sample(6000, 64, 5)
    per = W.per_shot_windows(shots, 7, 16, common=True)
    lc = REP.lag1_parts(shots)
    inf = REP.shot_influence(per, lc)
    se_inf = inf.std(0, ddof=1) / math.sqrt(len(shots))
    idx = W.bootstrap_indices(len(shots), 400, 6)
    cnt = np.stack([np.bincount(i, minlength=len(shots)) for i in idx]).astype(float)
    boot = np.column_stack(
        [
            np.asarray(W.witness(cnt @ per / len(shots), 0)),
            cnt @ per / len(shots),
            REP.lag1_of(cnt @ lc),
        ]
    )
    se_boot = boot.std(0, ddof=1)
    assert np.allclose(se_inf, se_boot, rtol=0.15)
    # the statistic vector agrees with the estimators it stands for
    v = REP.stat_vector(per, lc)
    assert v[0] == pytest.approx(W.witness(per.mean(0), 0))
    c = F.lag1_counts(shots, 16).sum(0)
    assert v[-1] == pytest.approx(c[1] / (c[0] + c[1]) - c[3] / (c[2] + c[3]))


def test_arm_statistics_reproduce_followup(live_pieces):
    proc = REP.arm_process("baseline", live_pieces)
    shots = proc.sample(4000, 64, 9)
    jobs = np.repeat([0, 1, 2, 3], 1000)
    pubs = np.tile(np.repeat([0, 1], 500), 4)
    s = REP.AnalysisSettings(n_boot=200, seed=24)
    mine = REP.arm_statistics(shots, jobs, pubs, s, 24)
    ref = F.analyse_arm(shots, jobs, pubs, F.AnalysisSettings(n_boot=200), 24)
    assert mine["value"][0] == pytest.approx(ref["w0"], rel=1e-12)
    assert mine["se_shot"][0] == pytest.approx(ref["se_shot0"], rel=1e-10)
    assert mine["se_block"][0] == pytest.approx(ref["se_block0"], rel=1e-10)
    assert mine["value"][-1] == pytest.approx(ref["lag1"], rel=1e-12)
    assert mine["lojo_value"].shape == (4, len(REP.STATS))
    assert np.all(mine["se"] >= mine["se_primary"])


# -- the decision rule and TOST ------------------------------------------------------------------


def test_tost_against_the_analytic_case():
    rule = REP.Rule(margin=0.10, tost_alpha=0.05)
    z = RA.norm_ppf(0.95)
    # ratio exactly 1, equal relative SEs s: se_R = sqrt(2) s; equivalent iff z sqrt(2) s < 0.1
    s_edge = 0.1 / (z * math.sqrt(2))
    for s, want in ((0.99 * s_edge, True), (1.01 * s_edge, False)):
        r, se, eq, dif = REP.tost_ratio(1.0, s, 1.0, s, rule)
        assert r == 1.0 and se == pytest.approx(math.sqrt(2) * s)
        assert bool(eq) is want and not bool(dif)
    # the same test as reanalysis.tost on the ratio
    r, se, eq, _ = REP.tost_ratio(1.044, 0.03, 1.0, 0.01, rule)
    ref = RA.tost(float(r), float(se), 0.9, 1.1, 0.05)
    assert bool(eq) == ref["equivalent"]
    # a ratio outside the margin is never equivalent, however small its SE
    assert not bool(REP.tost_ratio(1.2, 1e-9, 1.0, 1e-9, rule)[2])


def test_tost_power_in_the_engine_matches_the_analytic_power():
    S = len(REP.STATS)
    cov = np.diag(np.full(S, 0.01**2))
    mu = np.ones(S)
    mom = {a: REP.Moments(mu.copy(), cov) for a in REP.ARMS}
    n = 1000
    design = REP.Design({"baseline": n, "A2": n, "A2s_spec": n, "A2s": n}, 4)
    rule = REP.Rule(lojo=False)
    pw = REP.power_mc(mom, design, rule, 40000, 1)
    se_ratio = math.sqrt(2) * 0.01 / math.sqrt(n)
    analytic = RA.tost_power(1.0, se_ratio, 0.9, 1.1, 0.05)
    assert pw["A2s_spec_equiv_w0"] == pytest.approx(analytic, abs=0.01)


def test_rule_reproduces_27s_call_from_its_summary():
    s27 = json.loads((HW4C / "summary.json").read_text())
    b, a2, sh = (s27["arms"][k] for k in ("baseline", "A2", "A2s_spec"))
    S = len(REP.STATS)

    def est(a):
        v, s = np.zeros(S), np.ones(S)
        v[0], s[0] = a["w0"], a["se0"]
        v[1:9], s[1:9] = a["g"], 1e-9
        v[-1], s[-1] = a["lag1"], a["lag1_se"]
        return {"value": v, "se": s}

    rule = REP.Rule(lojo=False, full_sham="A2s_spec", w0_sham=None)
    d = REP.decide({"baseline": est(b), "A2": est(a2), "A2s_spec": est(sh)}, rule)
    assert s27["decision"]["primary"] == "vanishes"
    assert str(d["a2_call"]) == "vanishes" and bool(d["h1"])
    assert float(d["a2_bound"]) == pytest.approx(0.0931, abs=5e-5)
    assert float(d["A2s_spec_ratio"][0]) == pytest.approx(1.044, abs=1e-3)
    # #29: TOST at +-10% on #27's w(0) ratio is not established (p = 0.054)
    assert not bool(d["A2s_spec_equiv_w0"])


def test_rule_on_27s_shots_vanishes_on_all_jobs_but_not_leaving_job1_out():
    data = RA.load_jobs([HW4C / f"shots_job{j}.npz" for j in range(4)])
    s = REP.AnalysisSettings()
    arms = {
        a: REP.arm_statistics(
            data[a]["shots"], data[a]["jobs"], data[a]["pubs"], s, s.seed + 11 * i
        )
        for i, a in enumerate(("baseline", "A2"))
    }
    est = {
        a: {
            "value": v["value"],
            "se": v["se_primary"],
            "lojo_value": v["lojo_value"],
            "lojo_se": v["lojo_se_primary"],
        }
        for a, v in arms.items()
    }
    rule = REP.Rule(full_sham=None, w0_sham=None)
    d = REP.decide(est, rule)
    assert bool(d["vanish_all_jobs"])
    assert float(d["a2_bound"]) == pytest.approx(0.0931, abs=0.003)
    # #29 finding 1: without job1 the bound is 0.101, so the LOJO rule does not call vanishes
    lb = (np.abs(est["A2"]["lojo_value"][:, 0]) + REP.Z975 * est["A2"]["lojo_se"][:, 0]) / np.abs(
        est["baseline"]["lojo_value"][:, 0]
    )
    assert arms["A2"]["jobs"] == [0, 1, 2, 3]
    assert lb[1] == pytest.approx(0.101, abs=0.004)
    assert not bool(d["vanish"])
    assert REP.call_label(d).startswith("ambiguous")


def test_power_engine_reproduces_29s_replicate_power():
    """#29: a same-design replicate (A2 10,000 shots at SE 1.41e-4, baseline 15,000 at 1.27e-4,
    w = -5.15e-3) vanishes with probability 0.910 at a true ratio of 0."""
    S = len(REP.STATS)
    mu_b = np.ones(S) * 0.1
    mu_b[0] = -5.153e-3
    cov_b = np.diag(np.full(S, 1e-6))
    cov_b[0, 0] = (1.2714e-4) ** 2 * 15000
    mu_a = mu_b.copy()
    mu_a[0] = 0.0
    cov_a = cov_b.copy()
    cov_a[0, 0] = (1.4066e-4) ** 2 * 10000
    mom = {"baseline": REP.Moments(mu_b, cov_b), "A2": REP.Moments(mu_a, cov_a)}
    design = REP.Design({"baseline": 15000, "A2": 10000}, 4)
    rule = REP.Rule(lojo=False, full_sham=None, w0_sham=None)
    pw = REP.power_mc(mom, design, rule, 200000, 2)
    assert pw["vanish"] == pytest.approx(0.910, abs=0.005)
    # the leave-one-job-out rule costs power at the same shots
    pw2 = REP.power_mc(mom, design, REP.Rule(full_sham=None, w0_sham=None), 200000, 2)
    assert pw2["vanish"] < pw["vanish"] - 0.05


def test_arm_job_interaction_flags_an_arm_specific_job_shift(live_pieces):
    proc = REP.arm_process("baseline", live_pieces)
    jobs = F.make_schedule({"baseline": 8000, "A2s": 8000}, 4, 1000, 1)
    s = REP.AnalysisSettings(n_boot=50)
    arms = {}
    for i, arm in enumerate(("baseline", "A2s")):
        pubs = [p for j in jobs for p in j if p.arm == arm]
        flip = {2: 0.08} if arm == "baseline" else None
        sim = REP.simulate_arm([proc], pubs, lambda p: 0, 100 + i, flip)
        arms[arm] = REP.arm_statistics(sim["shots"], sim["jobs"], sim["pubs"], s, 3 + i)
    t = next(x for x in REP.arm_job_interaction(arms) if x["stat"] == "p1")
    assert t["q_p"] < 1e-6
    assert int(np.argmax(np.abs(np.array(t["per_job"]) - t["fe"]))) == 2


def test_plan_cost_respects_the_job_limit():
    cm = F.CostModel()
    dur = {"a": 3.08e-4, "b": 3.1e-4}
    c = REP.plan_cost({"a": 100000, "b": 50000}, dur, cm, 12.0, 4)
    assert c["jobs"] >= 4
    assert cm.overhead_s + c["shot_s"] / c["jobs"] <= 12.0
    assert c["upper_s"] == pytest.approx((1 + cm.margin) * c["estimate_s"] + c["jobs"])


# -- committed outputs ---------------------------------------------------------------------------


def test_committed_qasm_hashes_reproduce():
    circ = json.loads((SUM / "circuits.json").read_text())
    n = 0
    for rec in circ["pairs"].values():
        for arm, r in rec["arms"].items():
            assert r["relabelled_equals_reference"], (rec["memory"], arm)
            f = SUM / r["file"]
            if f.exists():
                assert hashlib.sha256(f.read_bytes()).hexdigest() == r["sha256"]
                n += 1
    for r in circ.get("parallel", {}).get("arms", {}).values():
        assert hashlib.sha256((SUM / r["file"]).read_bytes()).hexdigest() == r["sha256"]
        assert r["pair_a_preserved"] and r["pair_b_preserved"] and r["only_pair_qubits"]
    assert n > 0


def test_rebuilt_circuits_match_the_committed_hashes():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "phase8_replication", ROOT / "experiments" / "phase8_replication.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "experiments"))
    spec.loader.exec_module(mod)
    with open(CFG, "rb") as f:
        cfg = tomllib.load(f)
    circ = json.loads((SUM / "circuits.json").read_text())
    first = circ["chosen"][0]
    cfg2 = copy.deepcopy(cfg)
    res = mod.circuit_check(cfg2, only=[first])
    assert res["all_match"] and res["all_equivalent"]
