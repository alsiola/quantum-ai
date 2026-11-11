"""Phase 8 Stage 5 (#37) and 5b (#38): the live path's guards (``experiments/phase8_live.py``),
on a fake service and a fake sampler. No network, no Qiskit. #38 adds Tier 1's MDI sub-tier
(``tier1_mdi``): Tier 1's approval, a cap cumulative with P&M's committed ledger, the section
2.7 precondition, A4's digest and power gate, MDI decoding and analysis."""

from __future__ import annotations

import ast
import copy
import json
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qmem import followup as F
from qmem import prereg as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import phase8_live as L  # noqa: E402

CFG = tomllib.loads((ROOT / "experiments/configs/phase8_hw.toml").read_text())
URL = "https://github.com/alsiola/quantum-ai/issues/28#issuecomment-6782262170"


def cfg(**over) -> dict:
    c = copy.deepcopy(CFG)
    for k, v in over.items():
        sec, key = k.split("__")
        c[sec][key] = v
    return c


# -- a fake IBM service ----------------------------------------------------------------------------


class Bits:
    def __init__(self, a: np.ndarray):
        self.a = a

    def to_bool_array(self, order: str = "little") -> np.ndarray:
        assert order == "little"
        return self.a.astype(bool)


class FakeJob:
    def __init__(self, jid: str, pubs, charge: float | None, rng):
        self.jid, self.charge = jid, charge
        self.res = []
        for _circ, _par, shots in pubs:
            regs = [r if isinstance(r, tuple) else (r, 1) for r in _circ]
            data = SimpleNamespace(**{r: Bits(rng.integers(0, 2, (shots, w))) for r, w in regs})
            self.res.append(SimpleNamespace(data=data, metadata={}))

    def job_id(self) -> str:
        return self.jid

    def result(self):
        return self.res

    def usage(self):
        return self.charge

    def metrics(self):
        return {"usage": {"qpu_charge_time_seconds": self.charge}}

    def status(self):
        return "DONE"


class FakeService:
    def __init__(self, plan: str = "open", remaining: float | None = 538.0, charge=3.0):
        self.plan, self.remaining, self.charge = plan, remaining, charge
        self.jobs: dict[str, FakeJob] = {}
        self.consumed = 62.0
        self.rng = np.random.default_rng(0)

    def active_instance(self) -> str:
        return "crn:test"

    def instances(self):
        return [{"crn": "crn:test", "plan": self.plan}]

    def usage(self):
        u = {"usage_consumed_seconds": self.consumed, "instance_id": "secret"}
        if self.remaining is not None:
            u["usage_remaining_seconds"] = self.remaining
        return u

    def job(self, jid: str) -> FakeJob:
        return self.jobs[jid]

    def backend(self, name: str):
        return SimpleNamespace(name=name)

    def sampler(self, backend):
        svc = self

        class S:
            def run(self, pubs):
                jid = f"job{len(svc.jobs)}"
                svc.jobs[jid] = FakeJob(jid, pubs, svc.charge, svc.rng)
                svc.consumed += svc.charge or 0.0
                return svc.jobs[jid]

        return S()


def tier1_budget(c: dict) -> P.TierBudget:
    return L.tier_budget(c, "tier1", L.tier1_stages(c))


COST = {"estimate_s": 11.33, "upper_s": 15.16}


# -- 1. approval -----------------------------------------------------------------------------------


def test_approval_refuses_missing_other_tier_no_submit_and_a_changed_url(tmp_path):
    with pytest.raises(F.Refused):  # missing
        L.approval_guard(cfg(approval__tier1=""), "tier1", True, tmp_path)
    c = cfg()
    c["approval"] = {"tier2": URL}  # only another tier approved
    with pytest.raises(F.Refused):
        L.approval_guard(c, "tier1", True, tmp_path)
    with pytest.raises(F.Refused, match="approval is for"):
        P.Approval("tier2", URL).check("tier1", True)
    with pytest.raises(F.Refused, match="--submit"):
        L.approval_guard(cfg(), "tier1", False, tmp_path)
    assert L.approval_guard(cfg(), "tier1", True, tmp_path).url == URL
    other = URL.replace("6782262170", "1")
    with pytest.raises(F.Refused, match="changed within a run"):
        L.approval_guard(cfg(approval__tier1=other), "tier1", True, tmp_path)


def test_run_refuses_before_any_network_call_without_approval():
    def no_network(_cfg):
        raise AssertionError("contacted the service")

    args = SimpleNamespace(tier="tier1", submit=True, from_run=None, workers=1)
    with pytest.raises(F.Refused):
        L.run(cfg(approval__tier1=""), args, None, no_network)
    with pytest.raises(F.Refused, match="no live plan"):
        L.tier_entry("tier2")


# -- 2. the pre-registration's sections ------------------------------------------------------------


def test_prereg_guard_refuses_a_changed_section_and_an_unmerged_commit():
    def here(doc, commit):
        return (ROOT / doc).read_text()

    assert len(L.prereg_guard(cfg(), "tier1_mdi", here)) == 2
    # A4 appended to section 10, so P&M's recorded section 10 (#37, run and done) now refuses
    with pytest.raises(F.Refused, match="recorded hash"):
        L.prereg_guard(cfg(), "tier1", here)
    c = cfg()
    c["tier1"]["live"]["prereg"][1]["sha256"] = c["tier1"]["mdi_live"]["prereg"][1]["sha256"]
    assert len(L.prereg_guard(c, "tier1", here)) == 2
    c = cfg()
    c["tier1"]["mdi_live"]["prereg"][0]["sha256"] = "0" * 64
    with pytest.raises(F.Refused, match="recorded hash"):
        L.prereg_guard(c, "tier1_mdi", here)
    with pytest.raises(F.Refused, match="not on main"):
        L.prereg_guard(cfg(), "tier1_mdi", lambda d, c: None)
    c = cfg()
    c["tier1"]["mdi_live"]["prereg"][1]["amendments"] = ["**A9, 2026-11-11, pre-data.**"]
    with pytest.raises(F.Refused, match="A9"):
        L.prereg_guard(c, "tier1_mdi", here)
    c = cfg()
    c["tier1"]["mdi_live"]["prereg"] = []
    with pytest.raises(F.Refused):
        L.prereg_guard(c, "tier1_mdi", here)


# -- 3. circuits -----------------------------------------------------------------------------------


def test_circuit_check_digest_and_hash_mismatch(tmp_path):
    stages = L.tier1_stages(cfg())
    assert len(stages) == 1 and len(stages[0].pubs) == 60
    assert sum(p.shots for p in stages[0].pubs) == 36000
    assert stages[0].cost["estimate_s"] == pytest.approx(11.33, abs=0.01)
    assert stages[0].cost["upper_s"] == pytest.approx(15.16, abs=0.01)
    names = sorted(set(stages[0].names.values()))
    ok = L.circuit_check(cfg(), "tier1", names)
    assert ok["files"] == 60 and ok["digest"].startswith("4d17a450")
    assert not any(n.startswith("mdi") for n in names)  # MDI excluded
    c = cfg()
    c["tier1"]["live"]["qasm_digest"] = "0" * 64
    with pytest.raises(F.Refused, match="digest"):
        L.circuit_check(c, "tier1", names)
    qd = tmp_path / "qasm"
    shutil.copytree(ROOT / CFG["tier1"]["live"]["qasm_dir"], qd)
    f = qd / f"{names[0]}.qasm"
    f.write_text(f.read_text() + "// edited\n")
    c = cfg()
    c["tier1"]["live"]["qasm_dir"] = str(qd)
    with pytest.raises(F.Refused, match="hash mismatch"):
        L.circuit_check(c, "tier1", names)
    with pytest.raises(F.Refused, match="approved set"):
        L.circuit_check(cfg(), "tier1", names[:-1])


def test_circuit_names_follow_the_export():
    assert L.tier1_circuit_name("pm_n1", 0) == "pm_n1_pz_b0"
    assert L.tier1_circuit_name("pm_neg", 11) == "pm_neg_my_b1"
    assert L.name_digest([("b", "2"), ("a", "1")]) == L.sha_text("a 1\nb 2\n")


# -- 4. allowance, admission and the per-job guard -------------------------------------------------


def test_job_guard_refuses_paid_plan_unknown_allowance_over_cap_and_big_job():
    c = cfg()
    tb = tier1_budget(c)
    assert tb.hard_cap_s == 20.0 and tb.max_job_estimate_s == 20.0
    rec: dict = {}
    L.job_guard(c, "tier1", tb, P.Ledger(), FakeService(), COST)(rec)
    assert rec["plan"] == "open" and "instance_id" not in rec["usage_before"]
    for plan in ("pay-as-you-go", "premium", None):
        with pytest.raises(F.Refused, match="paid plan"):
            L.job_guard(c, "tier1", tb, P.Ledger(), FakeService(plan=plan), COST)({})
    with pytest.raises(F.Refused, match="unknown"):
        L.job_guard(cfg(allowance__remaining_s=None), "tier1", tb, P.Ledger(), FakeService(), COST)(
            {}
        )
    with pytest.raises(F.Refused, match="max_qpu_seconds"):
        L.job_guard(c, "tier1", tb, P.Ledger(), FakeService(), {"estimate_s": 20.5, "upper_s": 19})(
            {}
        )
    led = P.Ledger()  # a first job already counted 15.2 s: the second would pass 20 s
    led.add("tier1", F.job_charge("tier1:pm_primary", 15.16, {"job_usage": 3.0}))
    with pytest.raises(F.Refused, match="hard cap"):
        L.job_guard(c, "tier1", tb, led, FakeService(), COST)({})
    # the live reading caps the owner's: 70 s left, less the 60 s reserve, < 15.16 s
    with pytest.raises(F.Refused, match="remaining allowance"):
        L.job_guard(c, "tier1", tb, P.Ledger(), FakeService(remaining=70.0), COST)({})


def test_remaining_allowance_and_admission():
    c = cfg(allowance__prior_counted_s=0.0)
    led = P.Ledger()
    assert L.remaining_allowance(c, led, {}) == 538
    assert L.remaining_allowance(c, led, {"usage_remaining_seconds": 500}) == 500
    led.add("tier1", F.job_charge("x", 15.16, {"job_usage": 3.0}))
    assert L.remaining_allowance(c, led, None) == pytest.approx(538 - 15.16)
    # earlier Phase 8 jobs (the config's ledger entry) are taken off the owner's reading
    assert L.remaining_allowance(cfg(), P.Ledger(), {}) == pytest.approx(538 - 15.2)
    assert L.remaining_allowance(cfg(allowance__remaining_s=None), led, {}) is None
    tb = {"tier1": tier1_budget(c)}
    assert L.admission(c, "tier1", tb, 538)["fits_cap"]
    with pytest.raises(F.Refused, match="not admitted"):
        L.admission(c, "tier1", tb, 75)  # 20 s cap > 75 - 60
    with pytest.raises(F.Refused, match="not admitted"):
        L.admission(c, "tier1", tb, None)


# -- 5. jobs through qmem.prereg.run_stage on LiveBackend ------------------------------------------


def _backend(c, svc, out):
    regs = {"pm_n0": ["m"], "pm_n1": ["y0", "m"], "pm_n2": ["y0", "y1", "m"]}
    regs |= {"pm_n3": ["y0", "y1", "y2", "m"], "pm_neg": ["y0", "neg", "m"]}
    return L.LiveBackend(
        c, svc, svc.backend("ibm_kingston"), lambda p: regs[p.arm], L.tier1_decode, out, svc.sampler
    )


def test_live_backend_submits_once_resumes_by_id_and_decodes(tmp_path):
    c = cfg()
    st = L.tier1_stages(c)[0]
    svc = FakeService()
    lb = _backend(c, svc, tmp_path)
    lb.pubs = st.pubs
    tb = tier1_budget(c)
    led = P.Ledger()
    g = L.job_guard(c, "tier1", tb, led, svc, st.cost)
    arr = P.run_stage("tier1", st.stage, st.pubs, lb, tmp_path, st.cost, led, g)
    assert lb.submissions == 1 and len(svc.jobs) == 1
    rec = json.loads((tmp_path / st.stage / "job.json").read_text())
    assert rec["job_id"] == "job0" and rec["approval_url"] == URL and rec["plan"] == "open"
    ch = rec["charge"]
    assert ch["charged_s"] == 3.0 and ch["counted_s"] == pytest.approx(15.16, abs=0.01)
    side = json.loads((tmp_path / "live_jobs.json").read_text())["job0"]
    assert side["usage_readings"]["account_delta"] == pytest.approx(3.0)
    p = next(q for q in st.pubs if q.arm == "pm_n2")
    assert arr[p.key].shape == (p.shots, 3)
    q = next(q for q in st.pubs if q.arm == "pm_neg")
    assert arr[q.key].shape == (q.shots, 2) and arr[f"{q.key}__neg"].shape == (q.shots, 1)
    res = L.tier1_analyse(c, [st], {"pm_primary": arr})
    assert res["spam_gate"]["score"] < 0.9 and not res["valid"]  # random bits fail the gate
    assert res["call"].startswith("stopped")
    # a resume (new backend, same run dir) re-fetches by id and never resubmits
    lb2 = _backend(c, svc, tmp_path)
    lb2.pubs = st.pubs
    arr2 = P.run_stage("tier1", st.stage, st.pubs, lb2, tmp_path, st.cost, P.Ledger(), g)
    assert lb2.submissions == 0 and len(svc.jobs) == 1
    assert all(np.array_equal(arr[k], arr2[k]) for k in arr)


def test_a_charge_above_the_upper_bound_stops_the_run(tmp_path):
    c = cfg()
    st = L.tier1_stages(c)[0]
    svc = FakeService(charge=16.0)
    lb = _backend(c, svc, tmp_path)
    lb.pubs = st.pubs
    led = P.Ledger()
    g = L.job_guard(c, "tier1", tier1_budget(c), led, svc, st.cost)
    with pytest.raises(F.UsageExceeded):
        P.run_stage("tier1", st.stage, st.pubs, lb, tmp_path, st.cost, led, g)
    with pytest.raises(F.UsageExceeded):  # and refuses every later job
        g({})


def test_unreadable_usage_counts_the_upper_bound_and_refuses_the_next_job(tmp_path):
    c = cfg()
    st = L.tier1_stages(c)[0]
    svc = FakeService(charge=None)
    lb = _backend(c, svc, tmp_path)
    lb.pubs = st.pubs
    led = P.Ledger()
    g = L.job_guard(c, "tier1", tier1_budget(c), led, svc, st.cost)
    P.run_stage("tier1", st.stage, st.pubs, lb, tmp_path, st.cost, led, g)
    assert led.counted("tier1") == pytest.approx(st.cost["upper_s"])  # unknown: the upper bound


# -- imports: no IBM runtime outside the live path's functions -------------------------------------


def test_ibm_runtime_is_imported_inside_functions_only():
    for name in ("experiments/phase8_hw.py", "experiments/phase8_live.py", "src/qmem/prereg.py"):
        tree = ast.parse((ROOT / name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                mods.append(getattr(node, "module", "") or "")
                assert not any(m.startswith(("qiskit", "qiskit_ibm_runtime")) for m in mods), name
    text = (ROOT / "experiments/phase8_live.py").read_text()
    assert "os.environ" not in text  # credentials from .env only


# -- #38: Tier 1's MDI sub-tier --------------------------------------------------------------------

PM_COUNTED = 15.16217  # results/summaries/phase8_hw/tier1/ledger.json


def mdi_stage(c: dict | None = None) -> L.LiveStage:
    (st,) = L.tier1_mdi_stages(c or cfg())
    return st


def test_mdi_subtier_runs_on_tier1s_approval_and_config():
    assert L.base_tier("tier1_mdi") == "tier1" and L.base_tier("tier1") == "tier1"
    assert L.live_cfg(cfg(), "tier1_mdi") is not None
    assert L.live_cfg(cfg(), "tier1_mdi")["circuit_prefix"] == "mdi_"
    with pytest.raises(F.Refused):
        L.approval_guard(cfg(approval__tier1=""), "tier1_mdi", True, None)
    assert L.approval_guard(cfg(), "tier1_mdi", True, None).url == URL
    # an approval for another tier does not cover it
    c = cfg()
    c["approval"]["tier1"] = ""
    c["approval"]["tier2"] = URL
    with pytest.raises(F.Refused):
        L.approval_guard(c, "tier1_mdi", True, None)

    def no_network(_cfg):
        raise AssertionError("contacted the service")

    args = SimpleNamespace(tier="tier1_mdi", submit=True, from_run=None, workers=1)
    with pytest.raises(F.Refused):
        L.run(cfg(approval__tier1=""), args, None, no_network)


def test_mdi_prereg_refuses_until_a4_has_its_commit_on_main():
    c = cfg()
    if not c["tier1"]["mdi_live"]["prereg"][1]["commit"]:
        with pytest.raises(F.Refused, match="not on main"):
            L.prereg_guard(c, "tier1_mdi")
    c["tier1"]["mdi_live"]["prereg"][1]["amendments"] = ["**A5, 2026-11-12, pre-data.**"]
    with pytest.raises(F.Refused, match="A5"):
        L.prereg_guard(c, "tier1_mdi", lambda d, k: (ROOT / d).read_text())


def test_mdi_stage_is_33s_plan_with_the_reset_circuits_and_a4s_digest(tmp_path):
    st = mdi_stage()
    assert st.stage == "mdi_game" and len(st.pubs) == 32
    assert sum(p.shots for p in st.pubs) == 20000
    assert {p.arm: p.shots for p in st.pubs} == {"mdi_n0": 250, "mdi_n1": 1000}
    assert st.cost["upper_s"] < 30 - PM_COUNTED
    names = sorted(set(st.names.values()))
    assert len(names) == 32 and L.tier1_circuit_name("mdi_n1", 11) == "mdi_n1_x2_y3"
    ok = L.circuit_check(cfg(), "tier1_mdi", names)
    assert ok["files"] == 32 and ok["digest"].startswith("c36a119e")
    for n in names:  # A4: every MDI circuit resets q74 once, after the steps, before psi_y
        text = ok["texts"][n]
        assert text.count("reset $74;") == 1
        assert text.index("reset $74;") > text.rindex("barrier")
    c = cfg()
    c["tier1"]["mdi_live"]["qasm_digest"] = (
        "33c432509fd71f56d83da8a6afc8479e484d4af5fa5aa820af52f026edb91522"
    )
    with pytest.raises(F.Refused, match="digest"):  # #33's circuits, without the reset
        L.circuit_check(c, "tier1_mdi", names)
    with pytest.raises(F.Refused, match="approved set"):
        L.circuit_check(cfg(), "tier1_mdi", names[:-1])


def test_mdi_precondition_reads_pm_s_committed_decision(tmp_path):
    pre = L.mdi_precondition(cfg())
    assert pre["valid"] and pre["pm_n1_rejects_eb"] and pre["spam_gate"]
    d = json.loads((ROOT / CFG["tier1"]["mdi_live"]["precondition"]).read_text())
    for edit in ("gate", "neg", "reject"):
        e = copy.deepcopy(d)
        if edit == "gate":
            e["spam_gate"]["pass"] = False
        elif edit == "neg":
            e["negative_control"]["falsely_passes"] = True
        else:
            e["tests"]["pm_n1"]["reject_eb"] = False
        f = tmp_path / f"{edit}.json"
        f.write_text(json.dumps(e))
        c = cfg()
        c["tier1"]["mdi_live"]["precondition"] = str(f)
        with pytest.raises(F.Refused, match=r"2\.7"):
            L.tier1_mdi_stages(c)


def test_mdi_budget_is_cumulative_with_pm_against_tier1s_cap(tmp_path):
    c = cfg()
    st = mdi_stage(c)
    assert L.prior_counted(c, "tier1_mdi") == pytest.approx(PM_COUNTED)
    tb = L.tier_budget(c, "tier1_mdi", [st])
    assert tb.hard_cap_s == pytest.approx(30 - PM_COUNTED) and tb.max_job_estimate_s == 20.0
    assert not tb.problems()
    L.job_guard(c, "tier1_mdi", tb, P.Ledger(), FakeService(), st.cost)({})
    big = {"estimate_s": 12.0, "upper_s": 16.0}  # fits 20 s alone, not on top of P&M's 15.16 s
    with pytest.raises(F.Refused, match="hard cap"):
        L.job_guard(c, "tier1_mdi", tb, P.Ledger(), FakeService(), big)({})
    led = P.Ledger()  # a second MDI job after this one would pass the 30 s
    led.add("tier1_mdi", F.job_charge("tier1_mdi:mdi_game", st.cost["upper_s"], {"job_usage": 7}))
    with pytest.raises(F.Refused, match="hard cap"):
        L.job_guard(c, "tier1_mdi", tb, led, FakeService(), st.cost)({})
    with pytest.raises(F.Refused, match="paid plan"):
        L.job_guard(c, "tier1_mdi", tb, P.Ledger(), FakeService(plan="premium"), st.cost)({})
    with pytest.raises(F.Refused, match="max_qpu_seconds"):
        L.job_guard(
            c, "tier1_mdi", tb, P.Ledger(), FakeService(), {"estimate_s": 20.5, "upper_s": 1}
        )({})
    # a larger prior ledger leaves less: the cap follows the committed charges
    led_file = tmp_path / "ledger.json"
    rec = json.loads((ROOT / "results/summaries/phase8_hw/tier1/ledger.json").read_text())
    rec["charges"][0]["counted_s"] = 25.0
    led_file.write_text(json.dumps(rec))
    c["tier1"]["mdi_live"]["prior_ledgers"] = [str(led_file)]
    tb2 = L.tier_budget(c, "tier1_mdi", [st])
    assert tb2.hard_cap_s == pytest.approx(5.0) and tb2.problems()
    # admission uses Tier 1's whole cap (30 s) in the cumulative table
    budgets = {"tier1": tier1_budget(cfg())}
    assert L.admission(cfg(), "tier1_mdi", budgets, 538)["fits_cap"]


def test_mdi_power_gate_refuses_a_failing_check(monkeypatch):
    calls = {}

    def assess(c, dev):
        calls["dev"] = dev
        return {
            "rule": "r",
            "go": dev["go"],
            "cost_ok": True,
            "planned_cost": {},
            "tier1_cap_left_s": 14.8,
            "models": {"stage2 fitted (live)": {"n1": {"power_mc": 0.5}}},
        }

    monkeypatch.setitem(sys.modules, "phase8_mdi_power", SimpleNamespace(assess=assess))
    assert L.power_gate(cfg(), "tier1_mdi", {"go": True})["go"]
    with pytest.raises(F.Refused, match="power gate"):
        L.power_gate(cfg(), "tier1_mdi", {"go": False})
    with pytest.raises(F.Refused, match="no power gate"):
        L.power_gate(cfg(), "tier1", {"go": True})


def _mdi_backend(c, svc, out):
    regs = {"mdi_n0": [("m", 2)], "mdi_n1": ["y0", ("m", 2)]}
    return L.LiveBackend(
        c, svc, svc.backend("ibm_kingston"), lambda p: regs[p.arm], L.tier1_decode, out, svc.sampler
    )


def test_mdi_job_submits_once_resumes_by_id_decodes_and_analyses(tmp_path):
    c = cfg()
    st = mdi_stage(c)
    svc = FakeService(charge=7.0)
    lb = _mdi_backend(c, svc, tmp_path)
    lb.pubs = st.pubs
    tb = L.tier_budget(c, "tier1_mdi", [st])
    led = P.Ledger()
    g = L.job_guard(c, "tier1_mdi", tb, led, svc, st.cost)
    arr = P.run_stage("tier1_mdi", st.stage, st.pubs, lb, tmp_path, st.cost, led, g)
    assert lb.submissions == 1
    rec = json.loads((tmp_path / st.stage / "job.json").read_text())
    assert rec["approval_url"] == URL and rec["tier"] == "tier1_mdi"
    assert rec["charge"]["counted_s"] == pytest.approx(st.cost["upper_s"])  # max(7, upper)
    p1 = next(p for p in st.pubs if p.arm == "mdi_n1")
    p0 = next(p for p in st.pubs if p.arm == "mdi_n0")
    assert arr[p1.key].shape == (1000, 3) and arr[p0.key].shape == (250, 2)
    res = L.tier1_mdi_analyse(c, [st], {"mdi_game": arr})
    assert not res["n1"]["reject_eb"] and res["call"].startswith("MDI inconclusive")
    assert "does not undo the P&M result" in res["call"]
    assert res["alpha"] == 0.05 and res["precondition"]["valid"]
    lb2 = _mdi_backend(c, svc, tmp_path)
    lb2.pubs = st.pubs
    P.run_stage("tier1_mdi", st.stage, st.pubs, lb2, tmp_path, st.cost, P.Ledger(), g)
    assert lb2.submissions == 0 and len(svc.jobs) == 1


def test_mdi_analysis_rejects_eb_on_shots_from_a_good_channel():
    """Phi+ counts at the target unitary V_1's probabilities (payoff 1/2): the analysis
    rejects; at an EB channel's (full depolarising, payoff -1/4 at most 0): it does not."""
    from qmem import reference as R

    st = mdi_stage()
    rng = np.random.default_rng(38)
    s3 = json.loads((ROOT / "results/summaries/phase8_reference/summary.json").read_text())
    v1 = np.array(s3["targets_V"]["1"]["re"]) + 1j * np.array(s3["targets_V"]["1"]["im"])
    for sup, expect in ((R.superop([v1]), True), (R.depol_superop(1.0), False)):
        pq = R.mdi_probabilities(sup, R.Spam(), R.Spam(), 0.0)
        arr = {}
        for p in st.pubs:
            x, y = divmod(p.chunk, 4)
            n = int(p.arm[-1])
            hit = rng.random(p.shots) < pq[x, y]
            code = np.where(hit, 0, rng.integers(1, 4, p.shots))
            cols = [rng.integers(0, 2, p.shots)] * n + [code // 2, code % 2]
            arr[p.key] = np.stack(cols, 1).astype(np.uint8)
        res = L.tier1_mdi_analyse(cfg(), [st], {"mdi_game": arr})
        assert res["n1"]["reject_eb"] is expect
        assert res["call"].startswith("claim 4" if expect else "MDI inconclusive")
