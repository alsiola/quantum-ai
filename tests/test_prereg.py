"""Phase 8 Stage 4 (#36): the combined pre-registration's rules (``qmem.prereg``) and harness."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import coherence as Q
from qmem import followup as F
from qmem import prereg as P
from qmem import reference as R
from qmem import replication as REP

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results/summaries"
DEVICE = json.loads((SUM / "phase7_spectator/device.json").read_text())


@pytest.fixture(scope="module")
def live_pieces() -> A.StepPieces:
    with np.load(SUM / "phase7_hw_4c/pieces_live.npz") as z:
        return A.StepPieces.from_npz(z, "live")


# -- selection: FakeKingston reproduces the committed choices; perturbations move them -------------


def test_ancilla_rule_reproduces_q50_q21_and_moves_with_the_snapshot():
    rule = P.AncillaRule()
    assert P.select_ancillas(DEVICE, rule)["chosen"] == [50, 21]
    committed = json.loads((SUM / "phase8_coherence/circuits.json").read_text())["ancillas"]
    assert committed["chosen"] == [50, 21]
    worse = P.perturb(DEVICE, {50: {"readout": 0.05}})
    got = P.select_ancillas(worse, rule)["chosen"]
    assert got[0] == 21 and 50 not in got
    adj = P.adjacency(DEVICE)
    assert got[1] not in adj[got[0]]
    # the rule never picks a qubit within 2 hops of q59 or q75, whatever its readout
    near = P.perturb(DEVICE, {74: {"readout": 1e-6}, 55: {"readout": 1e-6}})
    assert not {74, 55} & set(P.select_ancillas(near, rule)["chosen"])
    assert DEVICE["qubits"]["50"]["readout"] != 0.05  # perturb copies


def _reference_score(dev):
    sys.path.insert(0, str(ROOT / "experiments"))
    import phase8_hw as H

    t1 = H.Tier1Ctx(H.PR.load_config(ROOT / "experiments/configs/phase8_reference.toml"))
    return t1, t1.ref_rule()


def test_reference_rule_reproduces_q79_and_moves_with_the_snapshot():
    t1, rule = _reference_score(DEVICE)
    assert P.select_reference(DEVICE, 75, 59, rule, t1.bell_score(DEVICE))["chosen"] == [79]
    better73 = P.perturb(DEVICE, {73: {"readout": 0.02}})
    assert P.select_reference(better73, 75, 59, rule, t1.bell_score(better73))["chosen"] == [73]
    no79 = P.perturb(DEVICE, {79: {"t2": 40e-6}})
    assert P.select_reference(no79, 75, 59, rule, t1.bell_score(no79))["chosen"] == []


def _committed_power():
    out = {}
    for r in csv.DictReader(open(SUM / "phase8_replication/candidates.csv")):
        if r["power_joint"]:
            out[(int(r["memory"]), int(r["output"]))] = float(r["power_joint"])
    return out


def test_pair_rule_reproduces_the_committed_pairs_and_drops_a_failing_one():
    pw = _committed_power()
    rules = REP.SelectionRules()

    def power(row):
        return {"power_joint": pw[(row["memory"], row["output"])], "snr": 1.0}

    sel = P.select_pairs(DEVICE, rules, power)
    assert sel["chosen"] == ["m67_o68", "m149_o148"]
    bad = P.perturb(DEVICE, {68: {"readout": 0.03}})
    sel2 = P.select_pairs(bad, rules, power)
    assert "m67_o68" not in sel2["chosen"] and len(sel2["chosen"]) == 2
    pol = P.ReselectionPolicy("tier3", "pairs", "drop", 0.8, "power")
    rows = {f"m{r['memory']}_o{r['output']}": r for r in sel2["rows"]}
    ok = {k: not rows[k]["screen_reasons"] for k in ("m67_o68", "m149_o148")}
    r = P.reselection(
        pol, ["m67_o68", "m149_o148"], sel2["chosen"], {"m67_o68": 0.84, "m149_o148": 0.84}, ok
    )
    assert r["use"] == ["m149_o148"] and not r["abort"]  # dropped, never replaced
    assert any("record only" in a for a in r["actions"])


def test_snapshot_hash_ignores_fields_the_selection_cannot_see():
    d = P.perturb(DEVICE, {50: {"witness_w0": -0.1}})
    assert P.snapshot_hash(d) == P.snapshot_hash(DEVICE)
    assert P.snapshot_hash(P.perturb(DEVICE, {50: {"readout": 0.01}})) != P.snapshot_hash(DEVICE)


def test_reselection_replace_policy():
    pol = P.ReselectionPolicy("tier2", "ancillas", "replace", 0.8, "power", ("RS",))
    same = P.reselection(pol, [50, 21], [50, 21], {})
    assert same["same"] and not same["abort"] and same["use"] == [50, 21]
    moved = P.reselection(pol, [50, 21], [21, 68], {(21, 68): 0.9})
    assert moved["use"] == [21, 68] and not moved["abort"]
    assert any("re-export" in a for a in moved["actions"]) and any(
        "re-hash" in a for a in moved["actions"]
    )
    weak = P.reselection(pol, [50, 21], [21, 68], {(21, 68): 0.7})
    assert weak["abort"] and weak["use"] == [50, 21]
    none = P.reselection(pol, [79], [], {(): 0.0})
    assert none["abort"]


# -- guards: approval, budget, ledger, resume, raw data ------------------------------------------


def test_approval_names_one_tier_and_is_frozen(tmp_path):
    url = "https://github.com/o/r/issues/28#issuecomment-123"
    with pytest.raises(F.Refused):
        P.Approval("tier1", None).check("tier1", True)
    with pytest.raises(F.Refused):
        P.Approval("tier1", url).check("tier2", True)
    with pytest.raises(F.Refused):
        P.Approval("tier1", url).check("tier1", False)  # no --submit flag
    P.Approval("tier1", url).check("tier1", True)
    P.freeze_approval(tmp_path, P.Approval("tier1", url))
    P.freeze_approval(tmp_path, P.Approval("tier1", url))  # unchanged: fine
    with pytest.raises(F.Refused):
        P.freeze_approval(tmp_path, P.Approval("tier1", url + "4"))


def _backend(tmp_path, charge):
    return P.LocalBackend(
        lambda st, pubs: {p.key: np.ones((p.shots, 2), np.uint8) for p in pubs},
        charge,
        tmp_path / "server",
    )


def test_resume_by_id_never_resubmits_and_raw_shots_are_kept(tmp_path):
    pubs = [F.Pub(0, 0, "a", 0, 3)]
    cost = {"estimate_s": 1.0, "upper_s": 2.25}
    be = _backend(tmp_path, lambda st, p: 1.0)
    lg = P.Ledger()
    P.run_stage("t", "s", pubs, be, tmp_path, cost, lg, lambda r: None)
    assert be.submissions == 1 and (tmp_path / "s" / "shots.npz").exists()
    first = (tmp_path / "s" / "shots.npz").read_bytes()
    be2 = _backend(tmp_path, lambda st, p: 1.0)  # a new process: the id is re-fetched

    def no_guard(rec):
        raise AssertionError("a resumed job must not pass through the submission guard")

    P.run_stage("t", "s", pubs, be2, tmp_path, cost, P.Ledger(), no_guard)
    assert be2.submissions == 0
    assert (tmp_path / "s" / "shots.npz").read_bytes() == first
    assert lg.counted("t") == pytest.approx(2.25)  # max(charged, upper)


def test_usage_guards(tmp_path):
    pubs = [F.Pub(0, 0, "a", 0, 1)]
    cost = {"estimate_s": 1.0, "upper_s": 2.0}
    with pytest.raises(F.UsageExceeded):
        P.run_stage(
            "t",
            "over",
            pubs,
            _backend(tmp_path, lambda s, p: 3.0),
            tmp_path,
            cost,
            P.Ledger(),
            lambda r: None,
        )
    lg = P.Ledger()
    P.run_stage(
        "t",
        "blind",
        pubs,
        _backend(tmp_path / "b", lambda s, p: None),
        tmp_path,
        cost,
        lg,
        lambda r: None,
    )
    with pytest.raises(F.Refused):  # an unreadable charge refuses the next job
        lg.counted("t")


def test_ledger_guard_per_job_limit_cap_allowance_and_reserve():
    tb = P.TierBudget("t", [{"estimate_s": 10, "upper_s": 13.5}], 30.0, 20.0)
    lg = P.Ledger()
    lg.guard(tb, {"estimate_s": 10, "upper_s": 13.5}, 100, "open", 60)
    with pytest.raises(F.Refused):  # per-job estimate
        lg.guard(tb, {"estimate_s": 21, "upper_s": 27.25}, 100, "open", 0)
    with pytest.raises(F.Refused):  # the reserve leaves too little
        lg.guard(tb, {"estimate_s": 10, "upper_s": 13.5}, 70, "open", 60)
    with pytest.raises(F.Refused):  # unknown allowance
        lg.guard(tb, {"estimate_s": 10, "upper_s": 13.5}, None, "open", 0)
    with pytest.raises(F.Refused):  # never a paid plan
        lg.guard(tb, {"estimate_s": 10, "upper_s": 13.5}, 100, "premium", 0)
    lg.add("t", F.job_charge("t:a", 20.0, {"x": 5.0}))
    with pytest.raises(F.Refused):  # the cap counts max(charged, upper) = 20
        lg.guard(tb, {"estimate_s": 10, "upper_s": 13.5}, 1000, "open", 0)


def test_budget_table_and_tier_problems():
    t1 = P.TierBudget("a", [{"estimate_s": 10, "upper_s": 13.5}], 30, 20)
    t2 = P.TierBudget("b", [{"estimate_s": 44, "upper_s": 56}], 225, 20)
    assert t2.problems() and not t1.problems()
    rows = P.budget_table([t1, t2], 100, 20)
    assert rows[0]["fits_upper"] and rows[1]["cumulative_upper_s"] == pytest.approx(69.5)
    assert rows[1]["fits_upper"] and not rows[1]["fits_cap"]
    assert not any(r["fits_upper"] for r in P.budget_table([t1, t2], None, 20))


def test_keep_first(tmp_path):
    f = tmp_path / "x.npz"
    assert P.keep_first(f) == f
    f.write_text("1")
    assert P.keep_first(f) != f


def test_order_warnings_and_chi2():
    assert P._chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert P._chi2_sf(16.918977604620448, 9) == pytest.approx(0.05, abs=1e-6)
    rng = np.random.default_rng(1)
    assert not P.order_warnings(rng.random(5000) < 0.3)["warn"]
    drift = np.concatenate([rng.random(2500) < 0.2, rng.random(2500) < 0.5])
    assert P.order_warnings(drift)["warn"]


def test_no_ibm_runtime_in_the_rules_or_the_harness():
    for path in (ROOT / "src/qmem/prereg.py", ROOT / "experiments/phase8_hw.py"):
        text = path.read_text()
        assert "import qiskit_ibm_runtime" not in text and "from qiskit_ibm_runtime" not in text
        assert "QiskitRuntimeService" not in text
    # #37: the live modes exist, in experiments/phase8_live.py (tests/test_phase8_live.py)


# -- Tier 1: the decision rule at known truths ----------------------------------------------------


def _tier1_counts(steps, neg, spam, seed=0):
    rng = np.random.default_rng(seed)
    v = {n: R.optimal_correction(R.choi(R.n_step_map(R.step_ops(_LIVE[0]), n))) for n in range(4)}
    shots = {"pm_n0": 500, "pm_n1": 1000, "pm_n2": 500, "pm_n3": 500, "pm_neg": 500}
    out = {}
    for arm, n in (("pm_n0", 0), ("pm_n1", 1), ("pm_n2", 2), ("pm_n3", 3), ("pm_neg", 1)):
        out[arm] = P.tier1_counts(
            steps,
            n,
            v[n],
            spam,
            shots[arm],
            rng,
            first=neg if arm == "pm_neg" else None,
            record=n in (1, 2),
        )
    return out


_LIVE: list[A.StepPieces] = []
SPAM = R.Spam(0.0098, 4e-4, 4e-4, 0.0137, 0.0039)


def test_tier1_rule_at_the_quantum_and_eb_truths(live_pieces):
    _LIVE[:] = [live_pieces]
    steps, neg = R.step_ops(live_pieces), R.step_ops(live_pieces, "A2")
    q = P.tier1_analyse(_tier1_counts(steps, neg, SPAM), P.Tier1Rule())
    assert (
        q["valid"] and q["tests"]["pm_n1"]["reject_eb"] and q["tests"]["branch00_n2"]["reject_eb"]
    )
    assert q["call"].startswith("claim 4") and not q["negative_control"]["falsely_passes"]
    assert q["model_check_n3"]["below_two_thirds"]
    eb = P.tier1_analyse(_tier1_counts(neg, neg, SPAM, 1), P.Tier1Rule())
    assert not any(t["reject_eb"] for t in eb["tests"].values())
    assert eb["call"].startswith("not supported")
    bad = R.Spam(0.2, 0.05, 0.05, 0.1, 0.1)
    s = P.tier1_analyse(_tier1_counts(steps, neg, bad, 2), P.Tier1Rule())
    assert s["call"].startswith("stopped")


def test_tier1_counts_sum_to_the_unconditional_channel(live_pieces):
    steps = R.step_ops(live_pieces)
    v = R.optimal_correction(R.choi(R.n_step_map(steps, 2)))
    pr = R.pm_probabilities(R.n_step_map(steps, 2), v, SPAM)["p_succ"].ravel()
    tot = sum(
        R.pm_probabilities(R.n_step_map(steps, 2, [a, b]), v, SPAM)["p_succ"].ravel()
        for a in (0, 1)
        for b in (0, 1)
    )
    assert np.allclose(tot, pr, atol=1e-12)


def test_random_eb_channels_never_pass_the_pm_threshold_in_expectation():
    rng = np.random.default_rng(5)
    for _ in range(50):
        sup = R.superop(R.random_eb_kraus(rng, 3))
        v = R.optimal_correction(R.choi(sup))
        assert R.pm_score(R.pm_probabilities(sup, v, R.Spam())) <= 2 / 3 + 1e-9


def test_stage2_models_and_the_stage4c_step(live_pieces):
    """Stage 2's fixed model is the Stage 4c step exactly; the fitted one keeps Tier 1's
    primaries above their EB bounds with 80% power at the planned shots."""
    with np.load(SUM / "phase7_hw_4c/pieces_live.npz") as z:
        ro = float(z["readout_q75"])
    fixed = Q.QuantumModel(live_pieces, ro, Q.NoiseParams())
    assert np.abs(fixed.step([]) - R.step_ops(live_pieces)).max() < 1e-14
    fit = json.loads((SUM / "phase8_coherence/review.json").read_text())["quantum_fitted"]["params"]
    qm = Q.QuantumModel(live_pieces, ro, Q.NoiseParams.from_dict(fit))
    steps = qm.step([])
    for n, br, shots in ((1, None, 12000), (2, [0, 0], 6000)):
        v = R.optimal_correction(R.choi(R.n_step_map(R.step_ops(live_pieces), n)))
        pm = R.pm_probabilities(R.n_step_map(steps, n, br), v, SPAM)
        eff = R.pm_score(pm) - R.pm_threshold(R.leakage_bound(n, 3, 1e-3))
        sd1 = R.pm_sd1(pm) if br is None else R.ratio_sd1(pm)
        assert eff > 0 and R.power_normal(eff, sd1, shots, 0.025) > 0.8


# -- Tier 2: legs and the primary contrast ---------------------------------------------------------


def test_first_harmonic_recovers_a_known_curve():
    phi = np.arange(8) * math.pi / 4
    y = 0.3 + 0.2 * np.cos(phi - 1.1)
    h = P.first_harmonic(phi, y)
    assert h["amplitude"] == pytest.approx(0.2) and h["phase"] == pytest.approx(1.1)
    h2 = P.first_harmonic(phi, y, np.full(8, 0.01))
    assert h2["amplitude_se"] > 0 and h2["phase_se"] > 0


def test_tost_and_primary_contrast():
    assert P.tost_diff(0.50, 0.001, 0.501, 0.001, 0.01, 0.05)
    assert not P.tost_diff(0.50, 0.01, 0.50, 0.01, 0.01, 0.05)
    pr = P.tier2_primary({4: {"mean": 0.2, "se_block": 0.01}, 6: {"mean": 0.03, "se_block": 0.004}})
    assert pr["rejected"] == [4, 6] or sorted(pr["rejected"]) == [4, 6]
    pr2 = P.tier2_primary(
        {4: {"mean": 0.2, "se_block": 0.01}, 6: {"mean": -0.04, "se_block": 0.004}}
    )
    assert pr2["rejected"] == [4]


def _exact_est(model, arms, s):
    """Exact statistics dressed as estimates with small SEs (a known truth for the legs)."""
    out = {}
    for a, arm in arms.items():
        st = Q.model_arm_stats(model, arm, s)
        val = np.array([st["w0"], *st["g"], st["lag1"]])
        se = np.full(len(REP.STATS), 1e-4)
        out[a] = {
            "value": val,
            "se": se,
            "lojo_value": np.stack([val, val]),
            "lojo_se": np.stack([se, se]) * 1.4,
            "p1": st["p1"],
            "p1_se": 1e-4,
        }
    return out


def test_tier2_legs_pass_at_the_quantum_prediction_and_fail_without_coherence(live_pieces):
    cfg = json.loads(
        json.dumps(
            __import__("tomllib").loads(
                (ROOT / "experiments/configs/phase8_coherence.toml").read_text()
            )
        )
    )
    arms = Q.parse_arms(cfg["arms"])
    with np.load(SUM / "phase7_hw_4c/pieces_live.npz") as z:
        ro = float(z["readout_q75"])
    qm = Q.QuantumModel(live_pieces, ro, Q.NoiseParams(ff_idle_s=1.064e-6), 3.2e-4, 1.4e-4)
    s = Q.StatSettings()
    pred = {a: Q.model_arm_stats(qm, arm, s) for a, arm in arms.items()}
    phases = {"E0": 0.0, **{f"P{k}": k * math.pi / 4 for k in range(1, 8)}}
    legs = P.tier2_legs(_exact_est(qm, arms, s), pred, phases, P.Tier2Rule())
    assert legs["leg_i_population"] and legs["leg_iii_dephasings"] and legs["leg_contrast"]
    assert legs["leg_ii_response"]
    # a model whose echoes do nothing to q75 (no phase response): leg (ii) must fail
    flat = Q.QuantumModel(live_pieces, ro, Q.NoiseParams(ff_idle_s=1.064e-6), 3.2e-4, 1.4e-4)
    flat._cache.update({(f"e{k}", None): flat.element("e0") for k in range(8)})
    legs2 = P.tier2_legs(_exact_est(flat, arms, s), pred, phases, P.Tier2Rule())
    assert not legs2["leg_ii_response"]
    gate = P.tier2_pilot_gate(
        {a: {"p1": pred[a]["p1"], "lag1": pred[a]["lag1"]} for a in arms}, pred, P.Tier2Rule()
    )
    assert gate["go"]


def test_tier2_decision_labels():
    legs = {
        "leg_i_population": True,
        "leg_ii_response": True,
        "leg_iii_dephasings": True,
        "leg_contrast": True,
    }
    assert P.tier2_decide({"rejected": [4, 6]}, legs, True).startswith("supported")
    assert P.tier2_decide({"rejected": [4]}, legs, True).startswith("partial")
    assert P.tier2_decide({"rejected": []}, legs, True).startswith("not supported")
    assert P.tier2_decide({"rejected": [4, 6]}, legs, False).startswith("stopped")
    assert P.tier2_decide({"rejected": [4, 6]}, {**legs, "leg_contrast": False}, True).startswith(
        "not supported"
    )


# -- the committed summaries ---------------------------------------------------------------------


def test_committed_prereg_summaries():
    sd = SUM / "phase8_prereg"
    s = json.loads((sd / "summary.json").read_text())
    assert all(s["selection"]["reproduces"].values())
    assert s["submitted"] is False
    rows = list(csv.DictReader(open(sd / "hashes.csv")))
    assert rows and all(r["status"] == "ok" for r in rows)
    tiers = {r["tier"]: r for r in csv.DictReader(open(sd / "tiers.csv"))}
    for t in ("tier1", "tier2", "tier3"):
        assert (
            float(tiers[t]["upper_s"]) <= float(tiers[t]["hard_cap_s"]) and not tiers[t]["problems"]
        )
    assert tiers["tier2 (as designed)"]["problems"]  # amendment A1's reason
    g = s["guards"]
    assert all(bool(v) for v in g.values())
