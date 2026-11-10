"""Phase 8 Stage 4 (issue #36): the combined pre-registration's local dry-run harness.

**Local only.** There is no live mode and no submit path, and nothing here imports
``qiskit_ibm_runtime`` or contacts IBM. ``--dry-run-local`` validates every tier of
``docs/phase8-preregistration.md`` end to end on simulated data:

1. **Circuits.** Each tier's committed QASM is hashed against its stage summary (#33's
   ``manifest.json``, #32/#34's ``plan.json``, #31's ``circuits.json``).
2. **Live re-selection** on the FakeKingston snapshot: Stage 2's ancillas, Stage 3's
   reference, Stage 1's pairs (``qmem.prereg``), which must reproduce the committed choices,
   then perturbed snapshots through the predeclared consequences (re-export, re-hash, power
   re-check against a floor, abort).
3. **Tier 1 power under Stage 2's fitted model** (task 3): #33's P&M n = 1 and the record-00
   branch at n = 2, under the Stage 4c pieces, Stage 2's fixed model and its fitted 9-parameter
   model.
4. **Budget**: each tier's jobs under ``qmem.followup``'s cost model, the per-job guard, the
   per-tier hard caps, and the cumulative table against the remaining allowance (a
   parameter) less a reserve.
5. **Each tier on simulated data** at a quantum truth and a classical-alternative truth: shots
   simulated job by job through ``qmem.prereg.run_stage`` on ``LocalBackend`` (guards, resume
   by id, raw shots kept, usage counted), then the tier's pre-registered analysis and decision
   rule.

The live path is a separate, reviewable addition: a backend with ``submit``, ``result`` and
``usage`` (``qmem.prereg.Backend``), a calibration reader returning a ``device.json``-shaped
snapshot, and the CLI flags. It is added only after the owner approves one tier and a CLAUDE.md
exception on #28.

    uv run python experiments/phase8_hw.py --config experiments/configs/phase8_hw.toml \
        --dry-run-local
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import phase8_coherence as PC
import phase8_coherence_design as DS
import phase8_coherence_null as NULL
import phase8_reference as PR
import phase8_replication as PRp

from qmem import ablate_hw as A
from qmem import classical as C
from qmem import coherence as Q
from qmem import followup as F
from qmem import prereg as P
from qmem import reference as R
from qmem import replication as REP
from qmem.ancilla import nelder_mead
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# -- io --------------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def sdir(cfg: dict[str, Any]) -> Path:
    p = REPO_ROOT / cfg["experiment"]["summary_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(out: Path | None, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if out is not None:
        with open(out / "log.txt", "a") as f:
            f.write(line + "\n")


def r6(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else str(v)
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return r6(v.item())
    if isinstance(v, dict):
        return {str(k): r6(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r6(x) for x in v]
    if isinstance(v, np.ndarray):
        return r6(v.tolist())
    return v


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r6(v) for k, v in r.items()})


def sub_cfgs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ex = cfg["experiment"]
    return {
        "tier1": PR.load_config(REPO_ROOT / ex["reference_config"]),
        "tier2": PC.load_config(REPO_ROOT / ex["coherence_config"]),
        "tier3": PRp.load_config(REPO_ROOT / ex["replication_config"]),
    }


def cost_model(cfg: dict[str, Any]) -> F.CostModel:
    return F.CostModel(**cfg["cost"])


def device(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads((REPO_ROOT / cfg["experiment"]["device"]).read_text())


# -- 1. circuits and their hashes ------------------------------------------------------------------


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every circuit of every tier: the recorded hash, the committed file it matches (searched
    in the stage's ``qasm/`` and the places #32's baseline and A2 come from), its status."""
    sm = REPO_ROOT / "results/summaries"
    rows = []
    man = json.loads((sm / "phase8_reference/qasm/manifest.json").read_text())["circuits"]
    for name, c in man.items():
        f = sm / "phase8_reference/qasm" / f"{name}.qasm"
        rows.append(
            {
                "tier": "tier1",
                "circuit": name,
                "sha256": c["sha256"],
                "file": str(f.relative_to(REPO_ROOT)),
                "status": "ok" if f.exists() and sha_file(f) == c["sha256"] else "MISMATCH",
            }
        )
    plan = json.loads((sm / "phase8_coherence/plan.json").read_text())
    dirs = [
        sm / "phase8_coherence/qasm",
        sm / "phase8_prereg/qasm/tier2",
        sm / "phase7_ablation/qasm",
        sm / "phase6",
    ]
    by_hash: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.glob("*.qasm")):
            by_hash.setdefault(sha_file(f), f)
    for t in plan["tiers"]:
        if t["tier"] == "controls":
            for name, h in t["qasm_sha256"].items():
                f = sm / "phase8_coherence/qasm/controls" / f"{name}.qasm"
                ok = f.exists() and sha_file(f) == h
                rows.append(
                    {
                        "tier": "tier2:controls",
                        "circuit": name,
                        "sha256": h,
                        "file": str(f.relative_to(REPO_ROOT)),
                        "status": "ok" if ok else "MISMATCH",
                    }
                )
            continue
        for a in t["arms"]:
            h = plan["qasm_sha256"][a]
            f = by_hash.get(h)
            rows.append(
                {
                    "tier": f"tier2:{t['tier']}",
                    "circuit": a,
                    "sha256": h,
                    "file": "" if f is None else str(f.relative_to(REPO_ROOT)),
                    "status": "ok" if f is not None else "hash only",
                }
            )
    circ = json.loads((sm / "phase8_replication/circuits.json").read_text())
    for a, v in circ["parallel"]["arms"].items():
        f = sm / "phase8_replication" / v["file"]
        rows.append(
            {
                "tier": "tier3",
                "circuit": f"parallel_{a}",
                "sha256": v["sha256"],
                "file": str(f.relative_to(REPO_ROOT)),
                "status": "ok" if f.exists() and sha_file(f) == v["sha256"] else "MISMATCH",
            }
        )
    return rows


# -- 2. live re-selection --------------------------------------------------------------------------


class Tier1Ctx:
    """#33's inputs: the Stage 4c pieces, q75's SPAM, the targets V_n, the MDI partner's SPAM."""

    def __init__(self, rcfg: dict[str, Any]):
        self.cfg = rcfg
        self.inp = PR.inputs(rcfg)
        self.spam = PR.q75_spam(rcfg, self.inp["calib"])
        self.ms, self.ms_neg, self.p_def = PR.models(rcfg, self.inp)
        self.live = self.inp["live"]
        prim = self.ms["stage4c_live"]
        self.v = {n: R.optimal_correction(R.choi(R.n_step_map(prim, n))) for n in (0, 1, 2, 3)}
        lay = rcfg["layout"]
        dev = self.inp["device"]
        self.spam_y = PR.device_spam(dev, lay["bridge"], rcfg["spam"]["gates_per_prep"])
        self.bsm_depol = 4 / 3 * float(dev["cz"][rcfg["spam"]["bsm_cz_error_from"]]["error"])

    def committed_targets_ok(self) -> float:
        """Max |V_n - committed targets_V| (the QASM compiles the committed ones)."""
        s = json.loads(
            (REPO_ROOT / self.cfg["experiment"]["summary_dir"] / "summary.json").read_text()
        )
        worst = 0.0
        for n, v in self.v.items():
            t = s["targets_V"][str(n)]
            c = np.array(t["re"]) + 1j * np.array(t["im"])
            worst = max(worst, float(np.abs(c - v).max()))
        return worst

    def ref_rule(self) -> R.RefRule:
        rr = self.cfg["reference_rule"]
        return R.RefRule(
            rr["min_dist_memory"],
            rr["min_dist_output"],
            rr["max_dist_memory"],
            rr["max_readout"],
            rr["min_t2_us"] * 1e-6,
            rr["max_cz_error"],
            (self.cfg["layout"]["output"],),
        )

    def bell_score(self, dev: dict[str, Any], steps: np.ndarray | None = None):
        sup1 = R.n_step_map(self.ms["stage4c_live"] if steps is None else steps, 1)

        def score(c: dict[str, Any]) -> float:
            rn = PR.ref_noise(self.cfg, dev, c, self.spam)
            return R.bell_parities(R.bell_final_state(sup1, 1, rn), self.v[1], rn)["fidelity"]

        return score

    def bell_power(self, dev: dict[str, Any], cand: dict[str, Any], shots: int) -> float:
        """Bell-route power at n = 1 (deficit-matched model, the pre-stated floor's model)."""
        sup1 = R.n_step_map(self.ms["live_deficit_matched"], 1)
        rn = PR.ref_noise(self.cfg, dev, cand, self.spam)
        par = R.bell_parities(R.bell_final_state(sup1, 1, rn), self.v[1], rn)
        thr = R.bell_threshold(PR.leak(self.cfg, 1, cand["cz_count"]))
        return R.power_normal(par["fidelity"] - thr, R.bell_sd1(par), shots, 0.05)


def tier2_model_ctx(ccfg: dict[str, Any]) -> dict[str, Any]:
    """#27's calibrated model as Stage 2 uses it (``phase8_coherence.Ctx`` without the shots)."""
    import phase8_classical as P8

    ccl = P8.load_config(REPO_ROOT / ccfg["experiment"]["classical_config"])
    d = P8.rdir(ccl, "data_dir")
    with np.load(d / "pieces_live.npz") as z:
        pieces = A.StepPieces.from_npz(z, "live")
        readout = float(z["readout_q75"])
    rf = json.loads((d / "refit_pilot.json").read_text())
    s6 = json.loads((REPO_ROOT / ccfg["experiment"]["phase6_dir"] / "summary.json").read_text())
    noise = s6["prepare"]["calibration"]["noise"]
    t1 = {int(k): v for k, v in noise["t1"]}[1]
    t2 = {int(k): v for k, v in noise["t2"]}[1]
    ff = DS.ff_idle_s(ccfg)
    fixed = Q.NoiseParams(meas_decay=float(rf["decay"]), ff_idle_s=ff)
    fit = json.loads((PC.sdir(ccfg) / "review.json").read_text())["quantum_fitted"]["params"]
    fitted = replace(Q.NoiseParams.from_dict(fit), meas_decay_first=None, ff_idle_s=ff)

    def model(p: Q.NoiseParams) -> Q.QuantumModel:
        return Q.QuantumModel(pieces, readout, p, float(t1), float(t2))

    return {
        "pieces": pieces,
        "readout": readout,
        "decay": float(rf["decay"]),
        "t1": float(t1),
        "t2": float(t2),
        "fixed": fixed,
        "fitted": fitted,
        "model": model,
    }


def ancilla_power(cfg: dict[str, Any], ccfg: dict[str, Any], t2: dict[str, Any], bias: float):
    """Tier 2's floor quantities at an ancilla bias: R2/R4 vanish power (f = 0.25) and RS
    non-inferiority power (m = 0.20), fixed model, 8,000 shots per arm, two jobs."""
    arms = Q.parse_arms(ccfg["arms"])
    s = DS.settings(ccfg)
    qm = t2["model"](t2["fixed"])
    con = json.loads((PC.sdir(ccfg) / "contrast.json").read_text())["generators"]["fixed"]
    sd = con["w0_sd_per_shot"]
    w = {
        a: Q.model_arm_stats(qm, DS.biased(arms[a], bias), s)["w0"]
        for a in ("R2", "R4", "RS", "E00")
    }
    rs = cfg["reselection"]
    van = min(
        Q.vanish_lojo_power(
            w[a], sd[a], w["RS"], sd["RS"], 8000, 2, 0.25, 1.959963984540054, rs["n_mc"], rs["seed"]
        )
        for a in ("R2", "R4")
    )
    pres = Q.ratio_power(
        w["RS"], sd["RS"], w["E00"], sd["E00"], 8000, 0.20, 1.6448536269514722, False
    )
    return {"bias": bias, "w0": w, "vanish_power": van, "preserve_power": pres}


def tier3_inputs(rcfg: dict[str, Any]) -> dict[str, Any]:
    pieces = PRp.load_pieces(rcfg)
    rows = list(csv.DictReader(open(PRp.sdir(rcfg) / "inflation.csv")))
    infl = {}
    for a in ("baseline", "A2", "A2s_spec"):
        infl[a] = np.array([float(r["inflation"]) for r in rows if r["arm"] == a])
    infl["A2s"] = np.maximum.reduce([infl[a] for a in rcfg["inflation"]["delay_sham_from"]])
    return {"pieces": pieces, "inflation": infl}


def _pair_power(task: dict[str, Any]) -> dict[str, Any]:
    return PRp._power_task(task)


def pair_powers(rcfg, dev_obj: REP.Device, t3, keys: list[str] | None, workers: int) -> dict:
    """#31's selection power (P(H1 and H2) at the selection design) for every screened pair, from
    each pair's committed FakeKingston pieces (a live run rebuilds the pieces from the live
    snapshot first: hardware group)."""
    rules = PRp.selection_rules(rcfg)
    rows = REP.screen(dev_obj, rules)
    sd_ = rcfg["selection_design"]
    tasks = []
    for r in rows:
        if r["screen_reasons"]:
            continue
        k = PRp.key(r["memory"], r["output"])
        if keys is not None and k not in keys:
            continue
        tasks.append(
            {
                "key": k,
                "pieces": PRp.pieces_kw(t3["pieces"][k]),
                "shortfall": rcfg["model"]["shortfall"],
                "n": rcfg["model"]["moment_shots"],
                "seed": rcfg["model"]["moment_seed"],
                "shots": {a: sd_[a] for a in REP.ARMS},
                "n_jobs": sd_["n_jobs"],
                "inflation": t3["inflation"],
                "rule": PRp.rule_of(rcfg),
                "n_mc": sd_["n_mc"],
                "mc_seed": sd_["seed"],
            }
        )
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_pair_power, tasks))
    return {r["key"]: {k: v for k, v in r.items() if k not in ("key", "moments")} for r in res}


def selection_part(cfg, subs, t1: Tier1Ctx, t2, out: Path, workers: int) -> dict[str, Any]:
    """The three selection functions on the FakeKingston snapshot, and on perturbed ones."""
    dev = device(cfg)
    rs = cfg["reselection"]
    ccfg, rcfg = subs["tier2"], subs["tier3"]
    res: dict[str, Any] = {
        "snapshot": {
            "backend": dev["backend"],
            "date": dev["calibration_date"],
            "sha256": P.snapshot_hash(dev),
        }
    }
    committed = {
        "tier1_reference": json.loads(
            (REPO_ROOT / t1.cfg["experiment"]["summary_dir"] / "summary.json").read_text()
        )["reference"]["pick"]["qubit"],
        "tier2_ancillas": json.loads((PC.sdir(ccfg) / "circuits.json").read_text())["ancillas"][
            "chosen"
        ],
        "tier3_pairs": json.loads((PRp.sdir(rcfg) / "circuits.json").read_text())["chosen"],
    }
    res["committed"] = committed
    # Tier 2: the ancillas
    anc = P.select_ancillas(dev, P.AncillaRule())
    res["tier2_ancillas"] = {k: v for k, v in anc.items() if k != "top10"} | {
        "top10": anc["top10"][:5]
    }
    # Tier 1: the reference (Bell route only; not in Tier 1's plan)
    ref = P.select_reference(dev, 75, 59, t1.ref_rule(), t1.bell_score(dev))
    res["tier1_reference"] = {
        "chosen": ref["chosen"],
        "candidates": [
            {k: c[k] for k in ("qubit", "path", "admitted", "excluded_because", "score")}
            for c in ref["candidates"]
        ],
    }
    # Tier 3: the pairs (#31's algorithm, its power from each pair's pieces)
    t3 = tier3_inputs(rcfg)
    dev_obj = REP.Device.from_snapshot(dev)
    t0 = time.time()
    powers = pair_powers(rcfg, dev_obj, t3, None, workers)
    log(out, f"[selection] #31 power of {len(powers)} screened pairs in {time.time() - t0:.0f} s")
    pairs = P.select_pairs(
        dev, PRp.selection_rules(rcfg), lambda r: powers[PRp.key(r["memory"], r["output"])]
    )
    res["tier3_pairs"] = {k: v for k, v in pairs.items() if k != "rows"}
    res["reproduces"] = {
        "tier1_reference": ref["chosen"] == [committed["tier1_reference"]],
        "tier2_ancillas": anc["chosen"] == committed["tier2_ancillas"],
        "tier3_pairs": pairs["chosen"] == committed["tier3_pairs"],
    }
    # perturbed snapshots through the predeclared consequences
    pol = policies(cfg)
    scen = []

    def t2_case(name, d):
        a = P.select_ancillas(d, P.AncillaRule())
        e = max(float(d["qubits"][str(q)]["readout"]) for q in a["chosen"])
        pw = ancilla_power(cfg, ccfg, t2, e)
        ok = e <= rs["tier2_ancilla_readout_max"]
        q = min(pw["vanish_power"], pw["preserve_power"]) if ok else 0.0
        r = P.reselection(
            pol["tier2"], committed["tier2_ancillas"], a["chosen"], {tuple(a["chosen"]): q}
        )
        scen.append(
            {
                "scenario": name,
                "tier": "tier2",
                **_flat(r),
                "live_readout_max": e,
                "vanish_power": pw["vanish_power"],
                "preserve_power": pw["preserve_power"],
            }
        )

    def t1_case(name, d):
        s = P.select_reference(d, 75, 59, t1.ref_rule(), t1.bell_score(d))
        pw = t1.bell_power(d, s["pick"], 6000) if s["pick"] else 0.0
        r = P.reselection(
            pol["tier1"], [committed["tier1_reference"]], s["chosen"], {tuple(s["chosen"]): pw}
        )
        scen.append(
            {"scenario": name, "tier": "tier1 (Bell route only)", **_flat(r), "bell_power": pw}
        )

    def t3_case(name, d):
        dobj = REP.Device.from_snapshot(d)
        live_rows = {
            PRp.key(r["memory"], r["output"]): r
            for r in REP.screen(dobj, PRp.selection_rules(rcfg))
        }
        extra = [k for k, r in live_rows.items() if not r["screen_reasons"] and k not in powers]
        pw_all = dict(powers) | (pair_powers(rcfg, dobj, t3, extra, workers) if extra else {})
        s = P.select_pairs(
            d, PRp.selection_rules(rcfg), lambda r: pw_all[PRp.key(r["memory"], r["output"])]
        )
        ok = {k: not live_rows[k]["screen_reasons"] for k in committed["tier3_pairs"]}
        r = P.reselection(
            pol["tier3"],
            committed["tier3_pairs"],
            s["chosen"],
            {k: pw_all.get(k, {}).get("power_joint", 0.0) for k in committed["tier3_pairs"]},
            ok,
        )
        scen.append({"scenario": name, "tier": "tier3", **_flat(r)})

    t2_case("fakekingston", dev)
    t1_case("fakekingston", dev)
    t3_case("fakekingston", dev)
    t2_case("q50 readout 0.05 (worse)", P.perturb(dev, {50: {"readout": 0.05}}))
    t2_case(
        "q50 and q21 readout 0.025, q25 0.024",
        P.perturb(dev, {50: {"readout": 0.025}, 21: {"readout": 0.025}, 25: {"readout": 0.024}}),
    )
    t1_case("q73 readout 0.02 (now admitted)", P.perturb(dev, {73: {"readout": 0.02}}))
    t1_case("q79 T2 40 us (excluded)", P.perturb(dev, {79: {"t2": 40e-6}}))
    t3_case(
        "q68 readout 0.03 (pair m67_o68 fails the screen)", P.perturb(dev, {68: {"readout": 0.03}})
    )
    t3_case(
        "q68 and q148 readout 0.03 (both fail)",
        P.perturb(dev, {68: {"readout": 0.03}, 148: {"readout": 0.03}}),
    )
    res["scenarios"] = scen
    res["powers_fakekingston"] = {k: powers[k]["power_joint"] for k in committed["tier3_pairs"]}
    return res


def _flat(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "committed": " ".join(map(str, r["committed"])),
        "live": " ".join(map(str, r["live"])),
        "same": r["same"],
        "use": " ".join(map(str, r["use"])),
        "abort": r["abort"],
        "actions": " | ".join(r["actions"]),
    }


def policies(cfg: dict[str, Any]) -> dict[str, P.ReselectionPolicy]:
    rs = cfg["reselection"]
    return {
        "tier1": P.ReselectionPolicy(
            "tier1",
            "Bell-route reference (#33 section 6)",
            "replace",
            rs["tier1_bell_power_floor"],
            "Bell-route power at n = 1 (deficit-matched, 6,000 shots)",
            ("bell_n*", "q59ctrl_*"),
        ),
        "tier2": P.ReselectionPolicy(
            "tier2",
            "feed-forward ancillas (#32 section 2.2)",
            "replace",
            min(rs["tier2_vanish_power_floor"], rs["tier2_preserve_power_floor"]),
            "min(R2/R4 vanish power, RS preserve power) with live readout <= "
            f"{rs['tier2_ancilla_readout_max']}",
            ("RS", "R2", "R4"),
        ),
        "tier3": P.ReselectionPolicy(
            "tier3",
            "replication pairs (#31 sections 2, 12)",
            "drop",
            rs["tier3_power_floor"],
            "P(H1 and H2) at the selection design",
            ("parallel_*",),
        ),
    }


# -- 3. Tier 1 power under Stage 2's fitted model -------------------------------------------------


def tier1_models(t1: Tier1Ctx, t2: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (baseline step branches, the A2 step for the EB negative control)."""
    out = {
        "stage4c_pieces": (t1.ms["stage4c_live"], t1.ms_neg["stage4c_live"]),
        "deficit_matched": (t1.ms["live_deficit_matched"], t1.ms_neg["live_deficit_matched"]),
    }
    for name in ("fixed", "fitted"):
        qm = t2["model"](t2[name])
        out[f"stage2_{name}"] = (qm.step([]), qm.step(["meas"]))
    return out


def tier1_power(cfg: dict[str, Any], t1: Tier1Ctx, t2: dict[str, Any]) -> list[dict[str, Any]]:
    rc, tc = t1.cfg, cfg["tier1"]
    pm = rc["plan"]["pm_shots"]
    tp = cfg["tier1_power"]
    rows = []
    a_first = tc["alpha"] / 2  # Holm's first step over the two co-primaries (conservative)
    for name, (steps, neg) in tier1_models(t1, t2).items():
        for test, n, branch, per_circ in (
            ("pm_n1", 1, None, pm["pm_n1"]),
            ("branch00_n2", 2, [0, 0], pm["pm_n2"]),
            ("branch00_n2 at #33's 12,000", 2, [0, 0], pm["pm_n1"]),
        ):
            row = PR.evaluate(rc, steps, n, t1.v[n], t1.spam, None, None, 0.0, branch=branch)
            eff = row["pm_score"] - row["pm_threshold"]
            shots = 12 * per_circ
            r = {
                "model": name,
                "test": test,
                "n": n,
                "fe_fixed": row["fe_fixed"],
                "branch_prob": row["branch_prob"],
                "pm_score": row["pm_score"],
                "threshold": row["pm_threshold"],
                "effect": eff,
                "sd1": row["pm_sd1"],
                "planned_shots": shots,
                "expected_branch_shots": shots * row["branch_prob"] if branch else shots,
                "power_alpha_0.025": R.power_normal(eff, row["pm_sd1"], shots, a_first)
                if eff > 0
                else 0.0,
                "power_alpha_0.05": R.power_normal(eff, row["pm_sd1"], shots, tc["alpha"])
                if eff > 0
                else 0.0,
                "shots_80_alpha_0.025": R.shots_for_power(
                    eff, row["pm_sd1"], a_first, tc["power_target"]
                ),
                "power_mc_alpha_0.025": R.mc_power_pm(
                    row["_pm"],
                    per_circ,
                    row["pm_threshold"],
                    a_first,
                    tp["draws"],
                    tp["seed"],
                    ratio=branch is not None,
                ),
            }
            rows.append(r)
        n0 = PR.evaluate(rc, steps, 0, t1.v[0], t1.spam, None, None, 0.0)
        ng = PR.evaluate(rc, steps, 1, t1.v[1], t1.spam, None, None, 0.0, first=neg)
        rows.append(
            {
                "model": name,
                "test": "spam_gate_n0",
                "n": 0,
                "pm_score": n0["pm_score"],
                "threshold": cfg["tier1"]["spam_gate"],
            }
        )
        rows.append(
            {
                "model": name,
                "test": "negative_control_false_pass",
                "n": 1,
                "pm_score": ng["pm_score"],
                "threshold": ng["pm_threshold"],
                "power_mc_alpha_0.025": R.mc_power_pm(
                    ng["_pm"],
                    pm["pm_neg"],
                    ng["pm_threshold"],
                    a_first,
                    tp["draws"],
                    tp["seed"] + 1,
                ),
            }
        )
    return rows


# -- 4. budgets ------------------------------------------------------------------------------------


def tier_budgets(cfg: dict[str, Any], subs) -> dict[str, Any]:
    cm = cost_model(cfg)
    sm = REPO_ROOT / "results/summaries"
    s3 = json.loads((sm / "phase8_reference/summary.json").read_text())["plan"]
    t1 = P.TierBudget(
        "tier1",
        s3["pm_primary"]["costs"] + s3["mdi_game"]["costs"],
        cfg["tier1"]["hard_cap_s"],
        cfg["tier1"]["max_job_estimate_s"],
        ["pm_primary", "mdi_game"],
    )
    plan2 = json.loads((sm / "phase8_coherence/plan.json").read_text())
    c2 = cfg["tier2"]
    designed, amended, labels_d, labels_a = [], [], [], []
    for t in plan2["tiers"]:
        for j in range(t["jobs"]):
            designed.append(t["per_job"])
            labels_d.append(f"{t['tier']}{j}")
    ccfg = subs["tier2"]
    dur = {r["arm"]: r["circuit_s"] for r in DS.budget(ccfg, Q.parse_arms(ccfg["arms"]))["arms"]}
    sched = tier2_schedules(cfg, ccfg)
    for stage, pubs in sched["stages"]:
        d = dur if not stage.startswith("controls") else sched["controls_dur"]
        amended.append(cm.job(pubs, d))
        labels_a.append(stage)
    t2d = P.TierBudget(
        "tier2 (as designed)", designed, c2["hard_cap_s"], c2["max_job_estimate_s"], labels_d
    )
    t2a = P.TierBudget("tier2", amended, c2["hard_cap_s"], c2["max_job_estimate_s"], labels_a)
    s1 = json.loads((sm / "phase8_replication/summary.json").read_text())["plan"]
    t3 = P.TierBudget(
        "tier3",
        [s1["pilot_cost"], *s1["job_costs"]],
        cfg["tier3"]["hard_cap_s"],
        cfg["tier3"]["max_job_estimate_s"],
        ["pilot"] + [f"job{j}" for j in range(len(s1["job_costs"]))],
    )
    sub_caps = {}
    for part, cap in (
        ("pilot", c2["pilot_cap_s"]),
        ("main", c2["main_cap_s"]),
        ("controls", c2["controls_cap_s"]),
    ):
        u = sum(
            j["upper_s"] for j, lab in zip(amended, labels_a, strict=True) if lab.startswith(part)
        )
        sub_caps[part] = {"upper_s": u, "cap_s": cap, "within": u <= cap}
    return {
        "tier1": t1,
        "tier2": t2a,
        "tier2_designed": t2d,
        "tier3": t3,
        "tier2_parts": sub_caps,
        "tier2_dur": dur,
        "tier2_sched": sched,
    }


def tier2_schedules(cfg: dict[str, Any], ccfg: dict[str, Any]) -> dict[str, Any]:
    """Tier 2's jobs under amendment A1: the pilot (one job), the main tier's two time blocks
    as ``main_subjobs_per_block`` sub-jobs each (every sub-job holds every arm, pub order
    seeded), the controls tier as ``controls_subjobs`` jobs."""
    plan2 = json.loads((PC.sdir(ccfg) / "plan.json").read_text())
    c2 = cfg["tier2"]
    tiers = {t["tier"]: t for t in plan2["tiers"]}
    seed = cfg["tier2"]["dry_run"]["seed"]
    pilot = F.make_schedule(
        {a: tiers["pilot"]["shots_per_arm"] for a in tiers["pilot"]["arms"]}, 1, 1000, seed
    )
    k = c2["main_subjobs_per_block"]
    half = tiers["main"]["shots_per_arm"] // tiers["main"]["jobs"]  # each time block's share
    main = []
    for b in range(tiers["main"]["jobs"]):
        main += F.make_schedule(
            {a: half for a in tiers["main"]["arms"]}, k, 1000, seed + 1 + 10 * b
        )
    ctl = json.loads((PC.sdir(ccfg) / "controls.json").read_text())["rows"]
    cj = F.make_schedule(
        {r["circuit"]: r["shots"] for r in ctl}, c2["controls_subjobs"], 6000, seed + 2
    )
    stages = [("pilot", pilot[0])]
    stages += [(f"main_b{j // k}_s{j % k}", job) for j, job in enumerate(main)]
    stages += [(f"controls{j}", job) for j, job in enumerate(cj)]
    return {
        "stages": stages,
        "block_of": {f"main_b{j // k}_s{j % k}": j // k for j in range(2 * k)},
        "controls_dur": {r["circuit"]: r["duration_s"] for r in ctl},
    }


def budget_rows(cfg: dict[str, Any], tb: dict[str, Any]) -> dict[str, Any]:
    al = cfg["allowance"]
    order = [tb[t] for t in cfg["order"]["tiers"]]
    table = {}
    for a in al["scenarios_s"]:
        table[str(a)] = P.budget_table(order, a, al["reserve_s"], al["prior_counted_s"])
    table["unknown"] = P.budget_table(order, None, al["reserve_s"], al["prior_counted_s"])
    slot = cfg["tier2"]["slot_1rad"]
    with_slot = [
        tb["tier1"],
        P.TierBudget(
            "tier2 + 1 rad slot",
            [
                *tb["tier2"].jobs,
                {
                    "estimate_s": (slot["added_main_upper_s"] - 1) / 1.25,
                    "upper_s": slot["added_main_upper_s"],
                    "shots": 48000,
                    "shot_s": math.nan,
                },
            ],
            tb["tier2"].hard_cap_s + slot["added_main_upper_s"],
            math.inf,
        ),
        tb["tier3"],
    ]
    table["with_1rad_slot_at_remaining"] = P.budget_table(
        with_slot, al["remaining_s"], al["reserve_s"]
    )
    return table


# -- 5. the tiers on simulated data ----------------------------------------------------------------


def guard_for(cfg, ledger: P.Ledger, tbud: P.TierBudget):
    al = cfg["allowance"]

    def g(rec: dict[str, Any]) -> None:
        # the live path adds the calibration snapshot and its check here, before the budget
        rec["plan"] = "open (simulated)"
        remaining = None if al["remaining_s"] is None else al["remaining_s"] - ledger.counted()
        ledger.guard(tbud, rec, remaining, "open", al["reserve_s"], al["allowed_plans"])

    return g


def tier1_run(cfg, t1: Tier1Ctx, t2, truth: str, out: Path, tbud: P.TierBudget) -> dict[str, Any]:
    """Tier 1 on simulated shots at one truth: the P&M job, then (if it passes, or always here
    to exercise it) the MDI job; decode, analyse, decide."""
    models = tier1_models(t1, t2)
    if truth == "classical_eb":  # q75 measured in every step: an EB channel by construction
        steps = neg = t1.ms_neg["stage4c_live"]
    else:
        steps, neg = models[truth]
    plan = json.loads(
        (REPO_ROOT / t1.cfg["experiment"]["summary_dir"] / "summary.json").read_text()
    )["plan"]
    salt = {"stage4c_pieces": 1, "stage2_fitted": 2, "stage2_fixed": 3, "classical_eb": 4}[truth]
    rng = np.random.default_rng([cfg["tier2"]["dry_run"]["seed"], salt])
    n_of = {"pm_n0": 0, "pm_n1": 1, "pm_n2": 2, "pm_n3": 3, "pm_neg": 1}

    def sim(stage: str, pubs) -> dict[str, np.ndarray]:
        arrays = {}
        for p in pubs:
            if p.arm.startswith("pm"):
                n = n_of[p.arm]
                k, b = divmod(p.chunk, 2)
                recs = (
                    [()] if n == 0 else [tuple(int(c) for c in f"{r:0{n}b}") for r in range(2**n)]
                )
                probs = []
                for rec in recs:
                    sup = R.n_step_map(
                        steps, n, list(rec) if n else None, neg if p.arm == "pm_neg" else None
                    )
                    pr = R.pm_probabilities(sup, t1.v[n], t1.spam)
                    probs.append((pr["p_succ"][k, b], pr["p_branch"][k, b] - pr["p_succ"][k, b]))
                pv = np.clip(np.array(probs).ravel(), 0, None)
                cell = rng.choice(len(pv), size=p.shots, p=pv / pv.sum())
                r_idx, fail = cell // 2, cell % 2
                bits = np.zeros((p.shots, n + 1), dtype=np.uint8)
                for i, rec in enumerate(recs):
                    bits[r_idx == i, :n] = rec
                bits[:, n] = np.where(fail == 0, b, 1 - b)
                arrays[p.key] = bits
            else:  # MDI: Phi+ reads 00
                n = int(p.arm[-1])
                x, y = divmod(p.chunk, 4)
                sup = R.n_step_map(steps, n)
                pq = R.mdi_probabilities(sup, t1.spam, t1.spam_y, t1.bsm_depol)[x, y]
                hit = rng.random(p.shots) < pq
                other = rng.integers(1, 4, p.shots)
                code = np.where(hit, 0, other)
                arrays[p.key] = np.stack([code // 2, code % 2], 1).astype(np.uint8)
        return arrays

    ledger = P.Ledger()
    rd = out / f"tier1_{truth}"
    est = {o: plan[o]["costs"][0]["estimate_s"] for o in ("pm_primary", "mdi_game")}
    backend = P.LocalBackend(sim, lambda st, pubs: est[st], rd / "server")
    res: dict[str, Any] = {"truth": truth}
    counts: dict[str, dict[str, np.ndarray]] = {}
    warn = []
    for opt in ("pm_primary", "mdi_game"):
        pubs = [
            F.Pub(0, i, q["arm"], q["circuit"], q["shots"])
            for i, q in enumerate(plan[opt]["jobs"][0])
        ]
        cost = plan[opt]["costs"][0]
        arr = P.run_stage(
            "tier1", opt, pubs, backend, rd, cost, ledger, guard_for(cfg, ledger, tbud)
        )
        for p in pubs:
            w = P.order_warnings(arr[p.key][:, -1])
            if w["warn"]:
                warn.append(p.key)
        if opt == "pm_primary":
            for arm in n_of:
                n = n_of[arm]
                nr = 2**n if n else 1
                succ = np.zeros((nr, 12), dtype=np.int64)
                tot = np.zeros((nr, 12), dtype=np.int64)
                for p in pubs:
                    if p.arm != arm:
                        continue
                    bits = arr[p.key]
                    rec = np.zeros(len(bits), dtype=np.int64)
                    for t in range(n):
                        rec = 2 * rec + bits[:, t]
                    ok = bits[:, n] == (p.chunk % 2)
                    np.add.at(tot[:, p.chunk], rec, 1)
                    np.add.at(succ[:, p.chunk], rec[ok], 1)
                recs = [""] if n == 0 else [f"{r:0{n}b}" for r in range(nr)]
                counts[arm] = {"succ": succ, "tot": tot, "records": np.array(recs)}
            rule = P.Tier1Rule(
                cfg["tier1"]["alpha"],
                cfg["tier1"]["secondary_alpha"],
                cfg["tier1"]["spam_gate"],
                cfg["tier1"]["min_branch_shots"],
            )
            res["pm"] = P.tier1_analyse(counts, rule)
            res["mdi_would_run"] = bool(
                res["pm"]["tests"]["pm_n1"]["reject_eb"] and res["pm"]["valid"]
            )
        else:
            hits = np.zeros((4, 4))
            shots = 0
            for p in pubs:
                if p.arm != "mdi_n1":
                    continue
                x, y = divmod(p.chunk, 4)
                hits[x, y] = int(((arr[p.key][:, 0] == 0) & (arr[p.key][:, 1] == 0)).sum())
                shots = p.shots
            res["mdi"] = P.mdi_analyse(hits, shots, R.mdi_weights(t1.v[1]))
    res["order_warnings"] = warn
    res["counted_s"] = ledger.counted("tier1")
    res["within_cap"] = res["counted_s"] <= tbud.hard_cap_s
    return res


def tier2_truths(t2, ccfg) -> dict[str, Any]:
    """The quantum truth (Stage 2's fixed model, its pre-registered prediction) and the
    classical alternative: a d = 6 classical program model, the octahedron embedding of the
    same instrument (``qmem.coherence.polytope_start``)."""
    arms = Q.parse_arms(ccfg["arms"])
    qm = t2["model"](t2["fixed"])
    els, lab = Q.arm_elements(arms)
    cl = Q.polytope_start(Q.bloch_maps(qm, els, lab), Q.octahedron(0.9, 1.0), els, lab)
    return {"quantum_fixed": qm, "classical_d6": cl, "arms": arms}


def _simulate(model, arm, n, length, rng) -> C.ProgramData:
    if isinstance(model, Q.QuantumModel):
        return DS.simulate(model, arm, n, length, rng)
    return NULL.simulate_program(model, arm, n, length, rng)


def _pdata_arrays(key: str, d: C.ProgramData) -> dict[str, np.ndarray]:
    out = {f"{key}__y": d.y, f"{key}__choice": d.choice}
    if d.m is not None:
        out[f"{key}__m"] = d.m
    return out


def tier2_run(cfg, subs, t2, truth: str, out: Path, tb: dict[str, Any]) -> dict[str, Any]:
    ccfg = subs["tier2"]
    tr = tier2_truths(t2, ccfg)
    arms, model = tr["arms"], tr[truth]
    length = ccfg["analysis"]["length"]
    dr = cfg["tier2"]["dry_run"]
    rng = np.random.default_rng([dr["seed"], 1 + list(tr).index(truth)])
    sched = tb["tier2_sched"]
    qfix = tr["quantum_fixed"]
    st = DS.settings(ccfg)

    def sim(stage: str, pubs) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for p in pubs:
            if stage.startswith("controls"):
                arrays[p.key] = controls_shots(ccfg, t2, truth, p, rng)
            else:
                arrays |= _pdata_arrays(p.key, _simulate(model, arms[p.arm], p.shots, length, rng))
        return arrays

    ledger = P.Ledger()
    rd = out / f"tier2_{truth}"
    backend = P.LocalBackend(sim, lambda st_, pubs: tb["tier2_charge"](st_, pubs), rd / "server")
    data: dict[str, list] = {}
    ctl_arrays: dict[str, np.ndarray] = {}
    pilot_arrays = {}
    for stage, pubs in sched["stages"]:
        d = tb["tier2_dur"] if not stage.startswith("controls") else sched["controls_dur"]
        cost = cost_model(cfg).job(pubs, d)
        arr = P.run_stage(
            "tier2", stage, pubs, backend, rd, cost, ledger, guard_for(cfg, ledger, tb["tier2"])
        )
        if stage == "pilot":
            pa: dict[str, list] = {}
            for p in sorted(pubs, key=lambda q: q.position):
                pa.setdefault(p.arm, []).append(arr[f"{p.key}__y"])
            pilot_arrays = {a: np.concatenate(v) for a, v in pa.items()}
        elif stage.startswith("main"):
            b = sched["block_of"][stage]
            for p in sorted(pubs, key=lambda q: q.position):
                data.setdefault(p.arm, []).append(
                    (
                        b,
                        p.job,
                        p.position,
                        arr[f"{p.key}__y"],
                        arr[f"{p.key}__choice"],
                        arr.get(f"{p.key}__m"),
                    )
                )
        else:
            for p in pubs:
                prev = ctl_arrays.get(p.arm)
                ctl_arrays[p.arm] = (
                    arr[p.key] if prev is None else np.concatenate([prev, arr[p.key]])
                )
    res: dict[str, Any] = {"truth": truth, "counted_s": ledger.counted("tier2")}
    pred = {a: Q.model_arm_stats(qfix, arm, st) for a, arm in arms.items()}
    # pilot gate (point estimates; the fixed model's predictions)
    pst = {
        a: {"p1": float(y[:, st.burn :].mean()), "lag1": float(_lag1(y, st.burn))}
        for a, y in pilot_arrays.items()
    }
    res["pilot_gate"] = P.tier2_pilot_gate(pst, pred, P.Tier2Rule())
    # main: every arm in block-then-position order, analysis job = time block
    s = REP.AnalysisSettings(n_boot=dr["stat_boot"], seed=dr["seed"])
    est, full, warn = {}, {}, {}
    for a, chunks in data.items():
        chunks.sort(key=lambda c: (c[0], c[1], c[2]))
        y = np.concatenate([c[3] for c in chunks])
        jobs = np.concatenate([np.full(len(c[3]), c[0]) for c in chunks])
        pubs_ = np.concatenate([np.full(len(c[3]), c[1] * 1000 + c[2]) for c in chunks])
        ch = np.concatenate([c[4] for c in chunks])
        m = None if chunks[0][5] is None else np.concatenate([c[5] for c in chunks])
        full[a] = (C.ProgramData(y, ch, m), jobs)
        e = REP.arm_statistics(y, jobs, pubs_, s, s.seed + 13 * len(est))
        sh = y[:, st.burn :].mean(1)
        e["p1_se"] = max(float(sh.std(ddof=1) / math.sqrt(len(sh))), _chunk_re(sh, pubs_))
        est[a] = e
        g = F.guard(y, None, F.AnalysisSettings(), dr["seed"])
        warn[a] = g["decoding_warn"] + g["integrity_fail"]
    phases = {"E0": 0.0, **{f"P{k}": k * math.pi / 4 for k in range(1, 8)}}
    res["legs"] = P.tier2_legs(est, pred, phases, P.Tier2Rule())
    res["shot_warnings"] = {a: w for a, w in warn.items() if w}
    t0 = time.time()
    res["primary"] = tier2_primary_contrast(cfg, ccfg, t2, arms, full, dr, out)
    res["primary_seconds"] = time.time() - t0
    res["controls"] = controls_analysis(ccfg, ctl_arrays)
    res["call"] = P.tier2_decide(res["primary"], res["legs"], True)
    res["call_if_pilot_binding"] = P.tier2_decide(
        res["primary"], res["legs"], res["pilot_gate"]["go"]
    )
    res["within_cap"] = res["counted_s"] <= tb["tier2"].hard_cap_s
    return res


def _lag1(y: np.ndarray, burn: int) -> float:
    from qmem import artefacts as AR

    return float(AR.lag1_excess_data(y, burn))


def _chunk_re(x: np.ndarray, chunks: np.ndarray) -> float:
    from qmem import reanalysis as RA

    ests, ses = [], []
    for c in np.unique(chunks):
        v = x[chunks == c]
        if len(v) > 1:
            ests.append(v.mean())
            ses.append(v.std(ddof=1) / math.sqrt(len(v)))
    return float(RA.pool(ests, ses)["re_se"]) if len(ests) > 1 else 0.0


REFIT = DS.REFIT


def tier2_primary_contrast(cfg, ccfg, t2, arms, full, dr, out: Path) -> dict[str, Any]:
    """§4's primary contrast on the first ``primary_per_job`` shots of each time block: parity
    split, classical ProgramModel at d = 4 and 6 (one start each here), the quantum refit, the
    held-out deficit with the block bootstrap by job and 8-shot segment, Holm over D."""
    m = dr["primary_per_job"]
    split = Q.PlanSplit(m, 2, ccfg["null"]["segment"])
    train, test, jobs_t, blocks_t = {}, {}, {}, {}
    for a, (d, jobs) in full.items():
        idx = np.concatenate([np.nonzero(jobs == b)[0][:m] for b in (0, 1)])
        sub = d.subset(np.isin(np.arange(len(d.y)), idx))
        train[a], test[a] = sub.subset(split.train()), sub.subset(split.test())
        jobs_t[a], blocks_t[a] = split.job()[split.test()], split.blocks(split.test())
    qfix = t2["model"](t2["fixed"])
    els, lab = Q.arm_elements(arms)
    maps = Q.bloch_maps(qfix, els, lab)
    starts = {"klein": None, "octahedron": Q.octahedron(0.9, 1.0)}
    s30 = json.loads((REPO_ROOT / ccfg["experiment"]["classical_summary"]).read_text())["embedding"]
    starts["klein"] = C.klein_tetrahedron(s30["a"], s30["b"])
    fits = {}
    for d_, start in dr["starts"].items():
        st0 = Q.polytope_start(maps, starts[start], els, lab)
        fit = C.fit_program_model(arms, train, st0, dr["em_iters"], 1e-10)
        fits[int(d_)] = fit.model
        log(out, f"[tier2] d = {d_} ({start}): {fit.iters} EM steps")
    tr_nll = {
        d_: sum(float(mm.nll(arms[a], train[a]).sum()) for a in arms) for d_, mm in fits.items()
    }
    if tr_nll[6] > tr_nll[4]:  # a d = 6 fit is never credited with less than d = 4's best
        fits[6] = fits[4]

    def f(x):
        p = Q.from_vector(x, REFIT, t2["fixed"])
        mm = t2["model"](p)
        v = sum(float(mm.nll(arms[a], train[a]).sum()) for a in arms)
        return v if math.isfinite(v) else 1e30

    x0 = Q.to_vector(replace(t2["fixed"], over=0.0), REFIT, t2["pieces"])
    x, _, n_ev = nelder_mead(f, x0, step=0.4, max_evals=dr["refit_evals"], ftol=1e-4)
    qref = t2["model"](Q.from_vector(x, REFIT, t2["fixed"]))
    q_test = {a: qref.nll(arms[a], test[a]) for a in arms}
    deficits = {}
    for d_, mm in fits.items():
        diffs = [mm.nll(arms[a], test[a]) - q_test[a] for a in arms]
        bb = Q.block_bootstrap(
            diffs,
            [blocks_t[a] for a in arms],
            [jobs_t[a] for a in arms],
            dr["n_boot"],
            dr["seed"] + d_,
        )
        deficits[d_] = bb
    out_p = P.tier2_primary(deficits, cfg["tier1"]["alpha"])
    out_p["refit_evals"] = n_ev
    out_p["shots_per_arm"] = 2 * m
    return out_p


def controls_shots(ccfg, t2, truth: str, p: F.Pub, rng) -> np.ndarray:
    """Simulated shots of one controls circuit: the benchmark's q75 bit, the Knee circuits'
    first four outputs. The classical alternative's Knee preparations of |+>, |+i> are the even
    mixture of its |0>, |1> preparations (a preparation-noncontextual model)."""
    name = p.arm
    if name.startswith("bench_"):
        _, inter, state = name.split("_")
        qm = t2["model"](t2["fixed"])
        b = Q.Benchmark(
            t2["t1"], t2["t2"], qm.pieces.p1_q75, ccfg["benchmark"]["spacing_s"], t2["readout"]
        )
        pr = Q.benchmark(inter, state, ccfg["controls"]["benchmark_reps"], b, t2["decay"])["p1"]
        return (rng.random((p.shots, 1)) < pr).astype(np.uint8)
    probs = knee_probs(ccfg, t2, truth)
    _, inter, state = name.split("_", 2)
    pv = np.asarray(probs[(inter, state)])
    cell = rng.choice(len(pv), size=p.shots, p=pv / pv.sum())
    return np.stack([(cell >> (3 - i)) & 1 for i in range(4)], 1).astype(np.uint8)


_KNEE: dict[str, Any] = {}


def knee_probs(ccfg, t2, truth: str) -> dict[tuple[str, str], np.ndarray]:
    if truth in _KNEE:
        return _KNEE[truth]
    qm = t2["model"](t2["fixed"])
    kc = ccfg["controls"]["knee_interventions"]
    inter = {k: v for k, v in kc.items() if k != "sham"}
    d = Q.knee_disturbances(qm, inter, kc["sham"], ccfg["knee"]["size"])
    probs = {}
    for key, v in d["_probs"].items():
        name, s = key.split("|")
        probs[(name, s)] = np.asarray(v)
    if truth != "quantum_fixed":
        for name in [*inter, "sham"]:
            mix = 0.5 * (probs[(name, "0")] + probs[(name, "1")])
            probs[(name, "+")] = probs[(name, "+i")] = mix
    _KNEE[truth] = probs
    return probs


def controls_analysis(ccfg, arr: dict[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive checks: each benchmark circuit against e0 at the same preparation (TOST
    +-0.02), and the Knee inequality D_+ <= max(D_0, D_1) (excess over 5 bootstrap SDs)."""
    out: dict[str, Any] = {"benchmark": {}, "knee": {}}
    mar = ccfg["controls"]["benchmark_margin"]
    for name, x in arr.items():
        if not name.startswith("bench_") or name.startswith("bench_e0"):
            continue
        state = name.split("_")[2]
        ref = arr[f"bench_e0_{state}"]
        a, b = float(x.mean()), float(ref.mean())
        sa, sb = math.sqrt(a * (1 - a) / len(x)), math.sqrt(b * (1 - b) / len(ref))
        out["benchmark"][name] = {
            "p1": a,
            "p1_e0": b,
            "equivalent": P.tost_diff(a, sa, b, sb, mar, 0.05),
        }
    rng = np.random.default_rng(36)

    def dist(x):
        code = (x * np.array([8, 4, 2, 1])).sum(1)
        return np.bincount(code, minlength=16) / len(x)

    for inter in ("e4", "e2", "A2"):
        ds = {}
        for s in ("0", "1", "+"):
            ds[s] = (dist(arr[f"knee_{inter}_{s}"]), dist(arr[f"knee_sham_{s}"]))
        ex = Q.tvd(*ds["+"]) - max(Q.tvd(*ds["0"]), Q.tvd(*ds["1"]))
        boots = []
        for _ in range(200):
            v = {
                s: Q.tvd(rng.multinomial(1000, a) / 1000, rng.multinomial(1000, b) / 1000)
                for s, (a, b) in ds.items()
            }
            boots.append(v["+"] - max(v["0"], v["1"]))
        sd = float(np.std(boots, ddof=1))
        out["knee"][inter] = {"excess_plus": ex, "sd": sd, "violates": bool(ex > 5 * sd)}
    return out


def tier3_run(cfg, subs, truth: str, out: Path, tb: dict[str, Any]) -> dict[str, Any]:
    """Tier 3 (two parallel pairs, full arms) at one truth: the pilot and its gate (b), then
    the 7 main jobs; each pair's arms analysed and decided by #31's rule."""
    rcfg = subs["tier3"]
    s1 = json.loads((PRp.sdir(rcfg) / "summary.json").read_text())
    plan = s1["plan"]
    pairs = json.loads((PRp.sdir(rcfg) / "circuits.json").read_text())["chosen"]
    pieces = PRp.load_pieces(rcfg)
    sch = rcfg["schedule"]
    jobs = F.make_schedule(plan["arms"], plan["jobs"], sch["max_chunk"], sch["seed"])
    pilot = F.make_schedule(
        {f"par:{a}": n for a, n in rcfg["pilot"]["shots"].items()},
        1,
        sch["max_chunk"],
        rcfg["pilot"]["seed"],
    )[0]
    dr = cfg["tier3"]["dry_run"]
    procs = {}
    for k in pairs:
        pc = pieces[k]
        dep = REP.shortfall_dephasing(pc, rcfg["model"]["shortfall"], PRp.SETTINGS)
        pr = {a: REP.arm_process(a, pc, dep) for a in REP.ARMS}
        if truth == "classical_memory_elsewhere":  # A2 does not remove it: not on the memory
            pr["A2"] = pr["baseline"]
        procs[k] = pr
    pred_p1 = {
        k: A.predict(REP.arm_process("baseline", pieces[k]), PRp.SETTINGS)["p1"] for k in pairs
    }
    length = REP.LENGTH

    def sim(stage: str, pubs) -> dict[str, np.ndarray]:
        arrays = {}
        for p in pubs:
            arm = p.arm.split(":")[1]
            for i, k in enumerate(pairs):
                sd = (
                    dr["seed"]
                    + 100003 * p.job
                    + 1009 * p.position
                    + 17 * i
                    + (7 if stage == "pilot" else 0)
                )
                arrays[f"{p.key}__{k}"] = procs[k][arm].sample(p.shots, length, sd)
        return arrays

    ledger = P.Ledger()
    rd = out / f"tier3_{truth}"
    backend = P.LocalBackend(
        sim,
        lambda st, pubs: cost_model(cfg).job(pubs, plan["durations_s"])["estimate_s"],
        rd / "server",
    )
    g = guard_for(cfg, ledger, tb["tier3"])
    arr = P.run_stage(
        "tier3",
        "pilot",
        pilot,
        backend,
        rd,
        cost_model(cfg).job(pilot, plan["durations_s"]),
        ledger,
        g,
    )
    gate = {}
    for k in pairs:
        y = np.concatenate([arr[f"{p.key}__{k}"] for p in pilot if p.arm == "par:baseline"])
        p1 = float(y[:, PRp.SETTINGS.burn :].mean())
        gate[k] = {
            "p1": p1,
            "pred": pred_p1[k],
            "b_ok": abs(p1 - pred_p1[k]) <= rcfg["pilot"]["p1_stop"],
            "a_ok": bool(set(np.unique(y)) <= {0, 1}),
        }
    per_pair = {k: {a: [] for a in REP.ARMS} for k in pairs}
    for j, job in enumerate(jobs):
        arr = P.run_stage(
            "tier3",
            f"job{j}",
            job,
            backend,
            rd,
            cost_model(cfg).job(job, plan["durations_s"]),
            ledger,
            g,
        )
        for p in sorted(job, key=lambda q: q.position):
            arm = p.arm.split(":")[1]
            for k in pairs:
                per_pair[k][arm].append((p.job, p.position, arr[f"{p.key}__{k}"]))
    rule = PRp.rule_of(rcfg)
    s = REP.AnalysisSettings(n_boot=dr["n_boot"], seed=dr["seed"])
    res: dict[str, Any] = {"truth": truth, "pilot_gate": gate, "pairs": {}}
    for k in pairs:
        arms = {}
        for i, (a, chunks) in enumerate(per_pair[k].items()):
            y = np.concatenate([c[2] for c in chunks])
            jb = np.concatenate([np.full(len(c[2]), c[0]) for c in chunks])
            pb = np.concatenate([np.full(len(c[2]), c[1]) for c in chunks])
            arms[a] = REP.arm_statistics(y, jb, pb, s, s.seed + 11 * i)
        r = REP.analyse_pair(arms, rule, rcfg["rule"]["armjob_alpha"])
        d = r["decision"]
        res["pairs"][k] = {
            "label": r["label"],
            "flagged_jobs": r["flagged_jobs"],
            **{x: bool(d[x]) for x in ("h1", "vanish", "h2_sham", "h2")},
            "a2_bound": float(d["a2_bound"]),
        }
    res["counted_s"] = ledger.counted("tier3")
    res["within_cap"] = res["counted_s"] <= tb["tier3"].hard_cap_s
    return res


# -- integrity checks of the guards (also in tests/test_prereg.py) ---------------------------------


def guard_checks(cfg, tb, out: Path) -> dict[str, Any]:
    """The guards refuse what they must: no approval, a mismatched tier, a changed approval, a
    job over the per-job limit or the cap, a charge over its upper bound; resume never
    resubmits."""
    res = {}
    try:
        P.Approval("tier1", cfg["approval"]["tier1"] or None).check("tier1", True)
        res["no_approval_refused"] = False
    except F.Refused:
        res["no_approval_refused"] = True
    url = "https://github.com/o/r/issues/28#issuecomment-1"
    try:
        P.Approval("tier1", url).check("tier2", True)
        res["wrong_tier_refused"] = False
    except F.Refused:
        res["wrong_tier_refused"] = True
    d = out / "guard_checks"
    d.mkdir(exist_ok=True)
    P.freeze_approval(d, P.Approval("tier1", url))
    try:
        P.freeze_approval(d, P.Approval("tier1", url + "2"))
        res["changed_approval_refused"] = False
    except F.Refused:
        res["changed_approval_refused"] = True
    lg = P.Ledger()
    t2d = tb["tier2_designed"]
    try:
        lg.guard(t2d, t2d.jobs[1], 531, "open", 60)
        res["designed_tier2_main_job_refused"] = False
    except F.Refused as e:
        res["designed_tier2_main_job_refused"] = str(e)
    be = P.LocalBackend(lambda st, pubs: {"x": np.zeros(1)}, lambda st, pubs: 99.0, d / "server")
    pub = [F.Pub(0, 0, "a", 0, 1)]
    try:
        P.run_stage("t", "s", pub, be, d, {"estimate_s": 1.0, "upper_s": 5.0}, lg, lambda r: None)
        res["over_upper_charge_stops"] = False
    except F.UsageExceeded:
        res["over_upper_charge_stops"] = True
    n0 = be.submissions
    try:
        P.run_stage(
            "t", "s", pub, be, d, {"estimate_s": 1.0, "upper_s": 5.0}, P.Ledger(), lambda r: None
        )
    except F.UsageExceeded:
        pass
    res["resume_never_resubmits"] = be.submissions == n0
    return res


# -- the dry run -----------------------------------------------------------------------------------


def tier2_charge_fn(cfg, tb):
    def charge(stage, pubs):
        d = (
            tb["tier2_dur"]
            if not stage.startswith("controls")
            else tb["tier2_sched"]["controls_dur"]
        )
        return cost_model(cfg).job(pubs, d)["estimate_s"]

    return charge


def dry_run_local(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    t_start = time.time()
    out = Path(args.from_run) if args.from_run else new_run_dir(cfg["experiment"]["name"])
    meta = run_metadata(cfg, cfg["tier2"]["dry_run"]["seed"])
    save_json(out / "config.json", {"meta": meta})
    sd = sdir(cfg)
    workers = min(args.workers or cfg["experiment"]["workers"], 2)
    parts = (
        set(args.part.split(","))
        if args.part != "all"
        else {"hashes", "selection", "power", "budget", "tier1", "tier2", "tier3"}
    )
    subs = sub_cfgs(cfg)
    log(out, f"phase8_hw --dry-run-local: parts {sorted(parts)}; local only, nothing is submitted")
    t1 = Tier1Ctx(subs["tier1"])
    t2 = tier2_model_ctx(subs["tier2"])
    summary_path = sd / "summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["meta"] = meta
    tb = tier_budgets(cfg, subs)
    tb["tier2_charge"] = tier2_charge_fn(cfg, tb)
    if "hashes" in parts:
        rows = hash_rows(cfg)
        write_csv(sd / "hashes.csv", rows)
        summary["hashes"] = {
            s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}
        }
        log(out, f"[hashes] {summary['hashes']}")
        summary["targets_V_max_diff"] = t1.committed_targets_ok()
    if "selection" in parts:
        sel = selection_part(cfg, subs, t1, t2, out, workers)
        write_csv(sd / "reselection.csv", sel.pop("scenarios"))
        save_json(sd / "selection.json", r6(sel))
        summary["selection"] = {
            "reproduces": sel["reproduces"],
            "tier2_ancillas": sel["tier2_ancillas"]["chosen"],
            "tier1_reference": sel["tier1_reference"]["chosen"],
            "tier3_pairs": sel["tier3_pairs"]["chosen"],
            "tier3_power": sel["powers_fakekingston"],
        }
        log(out, f"[selection] {summary['selection']}")
    if "power" in parts:
        rows = tier1_power(cfg, t1, t2)
        write_csv(sd / "power_tier1.csv", rows)
        summary["tier1_power"] = [
            {
                k: r.get(k)
                for k in (
                    "model",
                    "test",
                    "pm_score",
                    "effect",
                    "planned_shots",
                    "power_alpha_0.025",
                    "power_mc_alpha_0.025",
                    "shots_80_alpha_0.025",
                )
            }
            for r in rows
            if r["test"] in ("pm_n1", "branch00_n2")
        ]
        log(
            out,
            "[power] "
            + "; ".join(
                f"{r['model']} {r['test']}: {r['power_alpha_0.025']:.3f}"
                for r in rows
                if "power_alpha_0.025" in r
            ),
        )
    if "budget" in parts:
        table = budget_rows(cfg, tb)
        write_csv(
            sd / "budget.csv",
            [{"allowance_scenario": k, **r} for k, rs in table.items() for r in rs],
        )
        tiers = []
        for t in ("tier1", "tier2_designed", "tier2", "tier3"):
            b = tb[t]
            tiers.append(
                {
                    "tier": b.tier,
                    "jobs": len(b.jobs),
                    "estimate_s": b.estimate_s,
                    "upper_s": b.upper_s,
                    "hard_cap_s": b.hard_cap_s,
                    "max_job_estimate_s": b.max_job_estimate_s,
                    "max_job_estimate_seen_s": max(j["estimate_s"] for j in b.jobs),
                    "problems": "; ".join(b.problems()),
                }
            )
        write_csv(sd / "tiers.csv", tiers)
        summary["tiers"] = tiers
        summary["tier2_parts"] = tb["tier2_parts"]
        summary["guards"] = guard_checks(cfg, tb, out)
        log(out, f"[budget] {[(t['tier'], round(t['upper_s'], 1), t['problems']) for t in tiers]}")
    dry = json.loads((sd / "dry_run.json").read_text()) if (sd / "dry_run.json").exists() else {}
    if "tier1" in parts:
        for truth in ("stage4c_pieces", "stage2_fitted", "classical_eb"):
            r = tier1_run(cfg, t1, t2, truth, out, tb["tier1"])
            dry[f"tier1_{truth}"] = r
            log(
                out,
                f"[tier1 {truth}] {r['pm']['call']}; MDI {r['mdi']['reject_eb']}; "
                f"counted {r['counted_s']:.1f} s",
            )
    if "tier2" in parts:
        for truth in ("quantum_fixed", "classical_d6"):
            t0 = time.time()
            r = tier2_run(cfg, subs, t2, truth, out, tb)
            dry[f"tier2_{truth}"] = r
            log(
                out,
                f"[tier2 {truth}] {r['call']} (pilot go {r['pilot_gate']['go']}); "
                f"{time.time() - t0:.0f} s",
            )
    if "tier3" in parts:
        for truth in ("quantum", "classical_memory_elsewhere"):
            t0 = time.time()
            r = tier3_run(cfg, subs, truth, out, tb)
            dry[f"tier3_{truth}"] = r
            log(
                out,
                f"[tier3 {truth}] "
                + "; ".join(f"{k}: {v['label']}" for k, v in r["pairs"].items())
                + f"; {time.time() - t0:.0f} s",
            )
    if dry:
        save_json(
            sd / "dry_run.json", r6({"meta": meta, **{k: v for k, v in dry.items() if k != "meta"}})
        )
    summary["elapsed_s"] = time.time() - t_start
    summary["run_dir"] = str(out.relative_to(REPO_ROOT))
    summary["submitted"] = False
    save_json(summary_path, r6(summary))
    log(out, f"done in {time.time() - t_start:.0f} s; summaries in {sd.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--dry-run-local",
        action="store_true",
        required=True,
        help="the only mode: local simulation, no network, nothing submitted",
    )
    ap.add_argument(
        "--part",
        default="all",
        help="comma list of hashes, selection, power, budget, tier1, tier2, tier3",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--from-run", default=None, help="write into an existing run directory")
    args = ap.parse_args()
    dry_run_local(load_config(Path(args.config)), args)


if __name__ == "__main__":
    main()
