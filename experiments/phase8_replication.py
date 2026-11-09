"""Phase 8 Stage 1 (issue #31): pair selection and the independent-pair replication design.

Design, simulation and an UNSUBMITTED plan only. No QPU, no IBM service, no network, and no
submit path. The device is FakeKingston's committed snapshot (``phase7_spectator/device.json``);
the reference pair's live calibration is #27's committed snapshot. See
``docs/phase8-replication-prereg.md``.

Modes:

* ``--export`` (once, ``hardware`` group, FakeKingston only): the step pieces of every pair that
  passes the calibration/topology screen (``pieces.npz``), the selection, and for the reference,
  the named candidates and the chosen pairs the ISA circuits of every arm (QASM, SHA-256, the
  relabelled-hash equivalence check, durations, routing), plus the two-pair parallel circuits.
* ``--dry-run-local`` (numpy; circuit checks too when Qiskit is importable): the selection on the
  snapshot, every arm's predictions (each pair's calibrated instrument, the shortfall truth,
  #30's classical model), the drift-conditional SE calibration on #27's data, the power of every
  option and its cost, the recommendation, the schedule and pilot, the full-pipeline
  validation, and #27's own data under this rule.
* ``--plan``: prints the unsubmitted execution plan of the recommended option.

    uv run --group hardware python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --export
    uv run python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --dry-run-local
    uv run python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --plan
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qmem import ablate_hw as A
from qmem import classical as C
from qmem import followup as F
from qmem import reanalysis as RA
from qmem import replication as REP
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

SETTINGS = A.Settings()


# -- config and io ------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def rdir(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["experiment"][key]


def sdir(cfg: dict[str, Any]) -> Path:
    d = rdir(cfg, "summary_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(out: Path | None, msg: str) -> None:
    print(msg, flush=True)
    if out is not None:
        with open(out / "log.txt", "a") as f:
            f.write(msg + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


def fmt(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v) if v and isinstance(v[0], str) else str(list(v))
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.str_):
        return str(x)
    return x


def key(m: int, o: int) -> str:
    return f"m{m}_o{o}"


def label(m: int, o: int) -> str:
    return f"q{m}/q{o}"


# -- selection inputs ------------------------------------------------------------------------------


def load_device(cfg: dict[str, Any]) -> REP.Device:
    snap = json.loads((rdir(cfg, "spectator_dir") / "device.json").read_text())
    return REP.Device.from_snapshot(snap)


def selection_rules(cfg: dict[str, Any]) -> REP.SelectionRules:
    s, r = cfg["selection"], cfg["reference"]
    return REP.SelectionRules(
        reference=(r["output"], r["memory"], r["spectator"]),
        min_distance_reference=s["min_distance_reference"],
        min_distance_pairs=s["min_distance_pairs"],
        n_pairs=s["n_pairs"],
        mem_t1_min=s["mem_t1_min"],
        mem_t2_min=s["mem_t2_min"],
        out_t1_min=s["out_t1_min"],
        out_t2_min=s["out_t2_min"],
        out_readout_max=s["out_readout_max"],
        mem_readout_max=s["mem_readout_max"],
        spec_readout_max=s["spec_readout_max"],
        cz_max=s["cz_max"],
        sx_max=s["sx_max"],
        power_min=s["power_min"],
        named=tuple(tuple(p) for p in s["named"]),
    )


def rule_of(cfg: dict[str, Any], arm_set: str = "full") -> REP.Rule:
    r = cfg["rule"]
    full, w0s = r["full_sham"], r["w0_sham"]
    if arm_set == "no_spectator_sham":
        w0s = None
    elif arm_set == "no_delay_sham":
        full, w0s = "A2s_spec", None
    return REP.Rule(
        gate_z=r["gate_z"],
        vanish_fraction=r["vanish_fraction"],
        vanish_z=r["vanish_z"],
        persist_z=r["persist_z"],
        persist_fraction=r["persist_fraction"],
        margin=r["margin"],
        tost_alpha=r["tost_alpha"],
        lojo=r["lojo"],
        full_sham=full,
        w0_sham=w0s,
        sham_stats=tuple(r["sham_stats"]),
        g_alpha=r["g_alpha"],
    )


def arms_of(arm_set: str) -> tuple[str, ...]:
    return {
        "full": ("baseline", "A2", "A2s_spec", "A2s"),
        "no_spectator_sham": ("baseline", "A2", "A2s"),
        "no_delay_sham": ("baseline", "A2", "A2s_spec"),
    }[arm_set]


# -- pieces -------------------------------------------------------------------------------------------


def load_pieces(cfg: dict[str, Any]) -> dict[str, A.StepPieces]:
    path = sdir(cfg) / "pieces.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: run once with --export (hardware group)")
    z = np.load(path)
    keys = sorted({k.split("__")[0] for k in z.files})
    return {k: A.StepPieces.from_npz(z, k) for k in keys}


def live_reference_pieces(cfg: dict[str, Any]) -> A.StepPieces:
    z = np.load(rdir(cfg, "hw4c_dir") / "pieces_live.npz")
    return A.StepPieces.from_npz(z, "live")


# -- per-pair moments and power (numpy, in worker processes) ----------------------------------------


def pair_moments(pc: A.StepPieces, shortfall: float, n: int, seed: int) -> dict[str, Any]:
    """Moments of every arm under the pair's truth model: the calibrated pieces with the memory
    dephasing that puts the baseline at ``shortfall`` of its calibrated w(0). Under it both
    shams are the baseline's process (exactly, in the model); each is an independent arm."""
    p = REP.shortfall_dephasing(pc, shortfall, SETTINGS) if shortfall < 1 else 0.0
    base = REP.moments(REP.arm_process("baseline", pc, p), n, seed)
    a2 = REP.moments(REP.arm_process("A2", pc, p), n, seed + 1)
    cal = A.predict(REP.arm_process("baseline", pc), SETTINGS)
    return {
        "dephase": p,
        "calibrated_w0": cal["w0"],
        "calibrated_p1": cal["p1"],
        "moments": {"baseline": base, "A2": a2, "A2s_spec": base, "A2s": base},
    }


def _power_task(task: dict[str, Any]) -> dict[str, Any]:
    pc = A.StepPieces(**task["pieces"])
    pm = pair_moments(pc, task["shortfall"], task["n"], task["seed"])
    design = REP.Design(task["shots"], task["n_jobs"], task["inflation"])
    pw = REP.power_mc(pm["moments"], design, task["rule"], task["n_mc"], task["mc_seed"])
    b = pm["moments"]["baseline"]
    iw = REP.STATS.index("w0")
    return {
        "key": task["key"],
        "power_joint": pw["h1_and_h2"],
        "power_h1": pw["h1"],
        "power_h2": pw["h2"],
        "power_vanish": pw["vanish"],
        "power_sham": pw["h2_sham"],
        "snr": abs(b.mean[iw]) / b.sd()[iw],
        "pred_w0_calibrated": pm["calibrated_w0"],
        "pred_w0_truth": float(b.mean[iw]),
        "pred_p1_calibrated": pm["calibrated_p1"],
        "truth_dephasing": pm["dephase"],
        "moments": {a: m.to_dict() for a, m in pm["moments"].items() if a in ("baseline", "A2")},
    }


def pieces_kw(pc: A.StepPieces) -> dict[str, Any]:
    return dict(pc.__dict__)


def run_pool(fn, tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, tasks))


# -- inflation: #27's drift-conditional SEs against the 59/75 truth model ------------------------------


def load_27(cfg: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    d = rdir(cfg, "hw4c_dir")
    return RA.load_jobs([d / f"shots_{j}.npz" for j in cfg["experiment"]["main_jobs"]])


def analysis_settings(cfg: dict[str, Any], n_boot: int | None = None) -> REP.AnalysisSettings:
    a = cfg["analysis"]
    return REP.AnalysisSettings(a["k_max"], a["burn"], a["block"], n_boot or a["n_boot"], a["seed"])


def analyse_27(cfg: dict[str, Any], n_boot: int | None = None) -> dict[str, dict[str, Any]]:
    data = load_27(cfg)
    s = analysis_settings(cfg, n_boot)
    out = {}
    for i, arm in enumerate(("baseline", "A2", "A2s_spec")):
        d = data[arm]
        out[arm] = REP.arm_statistics(d["shots"], d["jobs"], d["pubs"], s, s.seed + 11 * i)
    return out


def inflation(cfg: dict[str, Any], a27: dict[str, Any], live_mom: dict[str, Any]):
    """Per arm and statistic: #27's drift-conditional SE x sqrt(N) over the 59/75 truth model's
    per-shot SD, floored. The delay-only sham takes the larger of its sources'."""
    fl = cfg["inflation"]["floor"]
    out, rows = {}, []
    for arm in ("baseline", "A2", "A2s_spec"):
        a = a27[arm]
        sd = live_mom[arm if arm != "A2s_spec" else "baseline"].sd()
        ratio = a["se"] * math.sqrt(a["n"]) / sd
        out[arm] = np.maximum(fl, ratio)
        for k, st in enumerate(REP.STATS):
            rows.append(
                {
                    "arm": arm,
                    "stat": st,
                    "se_primary_27": a["se_primary"][k],
                    "se_re_chunks_27": a["se_re_chunks"][k],
                    "tau_chunks_27": a["tau_chunks"][k],
                    "se_used_27": a["se"][k],
                    "model_sd_per_shot": sd[k],
                    "ratio": ratio[k],
                    "inflation": out[arm][k],
                }
            )
    out["A2s"] = np.maximum.reduce([out[a] for a in cfg["inflation"]["delay_sham_from"]])
    return out, rows


def calibrate_inflation(cfg: dict[str, Any], out: Path | None):
    """The 59/75 truth model's moments, #27's analysed arms and the inflation they imply."""
    model = cfg["model"]
    live = live_reference_pieces(cfg)
    live_m = pair_moments(
        live, model["shortfall"], model["moment_shots_chosen"], model["moment_seed"]
    )
    t0 = time.time()
    a27 = analyse_27(cfg)
    infl, rows = inflation(cfg, a27, live_m["moments"])
    log(
        out,
        f"[#27] analysed in {time.time() - t0:.0f} s; inflation of w0: "
        + ", ".join(f"{a} {infl[a][0]:.3f}" for a in REP.ARMS),
    )
    return live_m, a27, infl, rows


# -- the classical alternative --------------------------------------------------------------------------


def classical_model(cfg: dict[str, Any]) -> C.InterventionModel:
    s = json.loads((rdir(cfg, "classical_dir") / "summary.json").read_text())
    return C.InterventionModel.from_dict(s["classes"][cfg["model"]["classical_class"]]["model"])


def classical_predictions(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """#30's fitted d = 6 model, transferred unchanged. The delay-only sham is ``M = I`` (the
    class's convention for a delay: A3's I and the baseline's idle)."""
    m = classical_model(cfg)
    out = {}
    for arm in ("baseline", "A2", "A2s_spec"):
        out[arm] = C.arm_stats(m, arm, REP.LENGTH, REP.BURN, REP.K_MAX, (0, 1, 2, 3))
    out["A2s"] = dict(out["baseline"])
    return out


def quantum_predictions(pc: A.StepPieces, dephase: float) -> dict[str, dict[str, Any]]:
    out = {}
    for arm in REP.ARMS:
        proc = REP.arm_process(arm, pc, dephase)
        p = A.predict(proc, SETTINGS)
        st = REP.exact_stats(proc)
        p.pop("blocks", None)
        p["lag1"] = float(st[-1])
        out[arm] = p
    return out


def prediction_rows(name: str, model: str, preds: dict[str, dict[str, Any]]) -> list[dict]:
    rows = []
    wb = preds["baseline"]["w0"]
    for arm, p in preds.items():
        rows.append(
            {
                "pair": name,
                "model": model,
                "arm": arm,
                "w0": p["w0"],
                "w1": p["w1"],
                "w2": p["w2"],
                "w3": p["w3"],
                "w0_over_baseline": p["w0"] / wb,
                "p1": p["p1"],
                "lag1": p["lag1"],
                **{f"g{k}": v for k, v in enumerate(p["g"])},
            }
        )
    return rows


# -- durations and cost -------------------------------------------------------------------------------


def cost_model(cfg: dict[str, Any]) -> F.CostModel:
    b = cfg["budget"]
    return F.CostModel(b["sub_job_overhead_s"], b["rep_delay_s"], b["margin"], b["billing_round_s"])


def load_circuits(cfg: dict[str, Any]) -> dict[str, Any] | None:
    p = sdir(cfg) / "circuits.json"
    return json.loads(p.read_text()) if p.exists() else None


def arm_durations(cfg: dict[str, Any], circ: dict[str, Any] | None, k: str | None) -> dict:
    """Circuit seconds per arm: Phase 6's live baseline estimate + the arm's FakeKingston
    duration minus the pair's FakeKingston baseline (#24's rule). ``k`` None: two-pair parallel."""
    base = cfg["budget"]["phase6_circuit_s"]
    if circ is None:
        ab = json.loads((rdir(cfg, "ablation_dir") / "circuits.json").read_text())
        rows = {r["circuit"]: r["duration_minus_baseline_s"] for r in ab["circuits"]}
        return {a: base + rows[a] for a in REP.ARMS}
    src = circ["parallel"]["duration_s"] if k is None else circ["pairs"][k]["duration_s"]
    return {a: base + src[a] - src["baseline"] for a in REP.ARMS}


# -- options ---------------------------------------------------------------------------------------------


def option_shots(arm_set: str, n_bs: int, n_a2: int) -> dict[str, int]:
    return {a: (n_a2 if a == "A2" else n_bs) for a in arms_of(arm_set)}


def evaluate_option(
    cfg, mode, arm_set, n_bs, n_a2, pairs, mom, infl, circ, cm, n_mc, seed, pilot_upper
) -> dict[str, Any]:
    b = cfg["budget"]
    shots = option_shots(arm_set, n_bs, n_a2)
    use = pairs[:1] if mode == "one" else pairs[:2]
    if mode != "one" and len(pairs) < 2:
        return {}
    if mode == "two_parallel":
        dur = arm_durations(cfg, circ, None) if circ and "parallel" in circ else None
        if dur is None:
            return {}
        c = REP.plan_cost(shots, dur, cm, b["max_job_estimate_s"], b["min_jobs"])
    else:
        per_shots = {}
        durs = {}
        for p in use:
            d = arm_durations(cfg, circ, p)
            for a, n in shots.items():
                per_shots[f"{p}:{a}"] = n
                durs[f"{p}:{a}"] = d[a]
        c = REP.plan_cost(per_shots, durs, cm, b["max_job_estimate_s"], b["min_jobs"])
    rule = rule_of(cfg, arm_set)
    row: dict[str, Any] = {
        "mode": mode,
        "arm_set": arm_set,
        "pairs": " + ".join(use),
        "shots_base_sham": n_bs,
        "shots_A2": n_a2,
        **{f"cost_{k}": v for k, v in c.items()},
        "upper_with_pilot_s": c["upper_s"] + pilot_upper.get(mode, 0.0),
    }
    joint = 1.0
    for p in use:
        design = REP.Design(shots, c["jobs"], {a: infl[a] for a in shots})
        pw = REP.power_mc(mom[p], design, rule, n_mc, seed)
        tag = p
        row[f"{tag}_h1"] = pw["h1"]
        row[f"{tag}_vanish"] = pw["vanish"]
        row[f"{tag}_sham"] = pw["h2_sham"]
        row[f"{tag}_h2"] = pw["h2"]
        row[f"{tag}_joint"] = pw["h1_and_h2"]
        joint *= pw["h1_and_h2"]
    row["min_pair_joint"] = min(row[f"{p}_joint"] for p in use)
    row["all_pairs_joint"] = joint
    return row


def pilot_cost(cfg, pairs, circ, cm) -> dict[str, float]:
    """The pilot's upper bound per mode (one job, every arm of every pair in the mode)."""
    out = {}
    ps = cfg["pilot"]["shots"]
    for mode in ("one", "two_sequential", "two_parallel"):
        use = pairs[:1] if mode == "one" else pairs[:2]
        if mode != "one" and len(pairs) < 2:
            continue
        if mode == "two_parallel":
            if not circ or "parallel" not in circ:
                continue
            dur = arm_durations(cfg, circ, None)
            sec = sum(cm.shot_seconds(n, dur[a]) for a, n in ps.items())
        else:
            sec = sum(
                cm.shot_seconds(n, arm_durations(cfg, circ, p)[a])
                for p in use
                for a, n in ps.items()
            )
        est = cm.overhead_s + sec
        out[mode] = (1 + cm.margin) * est + cm.billing_round_s
    return out


def recommend(rows: list[dict[str, Any]], cap: float, power_min: float) -> dict[str, Any]:
    """The predeclared choice: among options whose every pair reaches ``power_min`` for H1 and
    H2, prefer those within the cap (pilot included); then more pairs, the full arm set, parallel
    over sequential (less QPU time), and the lowest upper bound."""
    ok = [r for r in rows if r and r["min_pair_joint"] >= power_min]
    if not ok:
        return {
            "verdict": "fail",
            "reason": "no option reaches the power threshold",
            "option": None,
        }
    n_pairs = lambda r: 1 if r["mode"] == "one" else 2  # noqa: E731
    order = lambda r: (  # noqa: E731
        -n_pairs(r),
        r["arm_set"] != "full",
        r["upper_with_pilot_s"],
    )
    within = sorted([r for r in ok if r["upper_with_pilot_s"] <= cap], key=order)
    if within:
        return {
            "verdict": "pass",
            "option": within[0],
            "cheapest_powered": min(ok, key=lambda r: r["upper_with_pilot_s"]),
        }
    best = min(ok, key=lambda r: r["upper_with_pilot_s"])
    return {
        "verdict": "ambiguous",
        "reason": f"powered options exist but the cheapest costs {best['upper_with_pilot_s']:.1f} s > cap {cap}",
        "option": best,
    }


# -- schedule -------------------------------------------------------------------------------------------


def option_pubs(opt: dict[str, Any], pairs: list[str]) -> dict[str, int]:
    shots = option_shots(opt["arm_set"], opt["shots_base_sham"], opt["shots_A2"])
    if opt["mode"] == "two_parallel":
        return {f"par:{a}": n for a, n in shots.items()}
    use = pairs[:1] if opt["mode"] == "one" else pairs[:2]
    return {f"{p}:{a}": n for p in use for a, n in shots.items()}


def make_plan(cfg, opt, pairs, circ, cm) -> dict[str, Any]:
    arms = option_pubs(opt, pairs)
    s = cfg["schedule"]
    jobs = F.make_schedule(arms, opt["cost_jobs"], s["max_chunk"], s["seed"])
    if opt["mode"] == "two_parallel":
        d = arm_durations(cfg, circ, None)
        dur = {f"par:{a}": d[a] for a in REP.ARMS}
    else:
        dur = {f"{p}:{a}": arm_durations(cfg, circ, p)[a] for p in pairs[:2] for a in REP.ARMS}
    problems = F.check_schedule(
        jobs, arms, list(arms), s["max_chunk"], None, cfg["budget"]["min_jobs"]
    )
    pilot_arms = {}
    for a, n in cfg["pilot"]["shots"].items():
        if a not in arms_of(opt["arm_set"]):
            continue
        if opt["mode"] == "two_parallel":
            pilot_arms[f"par:{a}"] = n
        else:
            for p in pairs[: (1 if opt["mode"] == "one" else 2)]:
                pilot_arms[f"{p}:{a}"] = n
    pilot = F.make_schedule(pilot_arms, 1, s["max_chunk"], cfg["pilot"]["seed"])[0]
    jc = [cm.job(j, dur) for j in jobs]
    pc = cm.job(pilot, dur)
    return {
        "arms": arms,
        "jobs": jobs,
        "pilot": pilot,
        "durations": dur,
        "job_costs": jc,
        "pilot_cost": pc,
        "problems": problems,
        "estimate_s": pc["estimate_s"] + sum(c["estimate_s"] for c in jc),
        "upper_s": pc["upper_s"] + sum(c["upper_s"] for c in jc),
    }


# -- full-pipeline validation -----------------------------------------------------------------------------


def _validation_task(task: dict[str, Any]) -> dict[str, Any]:
    pc = A.StepPieces(**task["pieces"])
    p0, slope = task["dephase"], task["slope"]
    levels = task["levels"]
    procs = {
        a: [
            REP.arm_process(a, pc, max(0.0, min(1.0, p0 + task["drift_sd"][a] * x / slope)))
            for x in levels
        ]
        for a in task["arms"]
    }
    rng_seed = task["seed"]

    def level_of(arm_index: int):
        # arm-specific chunk heterogeneity (#29 finding 6): each (arm, job, chunk) draws its own
        def level(p: F.Pub) -> int:
            r = np.random.default_rng([rng_seed, arm_index, p.job, p.chunk, 7])
            return int(r.integers(len(levels)))

        return level

    s = REP.AnalysisSettings(n_boot=task["n_boot"], seed=task["seed"])
    arms = {}
    for i, arm in enumerate(task["arms"]):
        pubs = [p for job in task["jobs"] for p in job if p.arm == arm]
        flip = task["flip"] if arm == "baseline" else None
        sim = REP.simulate_arm(procs[arm], pubs, level_of(i), rng_seed + 101 * i, flip)
        arms[arm] = REP.arm_statistics(sim["shots"], sim["jobs"], sim["pubs"], s, s.seed + 11 * i)
    res = REP.analyse_pair(arms, task["rule"], task["alpha"])
    d = res["decision"]
    return {
        "pair": task["pair"],
        "scenario": task["scenario"],
        "rep": task["rep"],
        "label": res["label"],
        "flagged_jobs": res["flagged_jobs"],
        **{
            k: bool(d[k]) for k in ("h1", "vanish_all_jobs", "vanish", "h2_sham", "h2", "h1_and_h2")
        },
        "a2_bound": float(d["a2_bound"]),
        "a2_lojo_max_bound": float(d.get("a2_lojo_max_bound", math.nan)),
        **{f"{sh}_ratio_w0": float(d[f"{sh}_ratio"][0]) for sh in REP.SHAMS if f"{sh}_ratio" in d},
        **{
            f"{sh}_equiv_all": bool(d[f"{sh}_equiv_all"])
            for sh in REP.SHAMS
            if f"{sh}_equiv_all" in d
        },
        **{
            f"se_{a}_{st}": float(arms[a]["se"][REP.STATS.index(st)])
            for a in arms
            for st in ("w0", "lag1")
        },
        **{f"n_{a}": int(arms[a]["n"]) for a in arms},
        "base_se_w0": float(arms["baseline"]["se"][0]),
        "base_se_primary_w0": float(arms["baseline"]["se_primary"][0]),
        "base_w0": float(arms["baseline"]["value"][0]),
    }


def engine_at(cfg, opt, mom, rows) -> dict[str, float]:
    """The power engine with each arm's inflation set to what the simulated datasets realised
    (mean drift-conditional SE x sqrt(N) over the model's per-shot SD, for w(0) and lag-1):
    the like-for-like check of the engine against the full pipeline."""
    shots = option_shots(opt["arm_set"], opt["shots_base_sham"], opt["shots_A2"])
    infl = {}
    for a in shots:
        f = np.ones(len(REP.STATS))
        for st in ("w0", "lag1"):
            k = REP.STATS.index(st)
            f[k] = (
                np.mean([r[f"se_{a}_{st}"] * math.sqrt(r[f"n_{a}"]) for r in rows]) / mom[a].sd()[k]
            )
        infl[a] = f
    pw = REP.power_mc(
        mom,
        REP.Design(shots, opt["cost_jobs"], infl),
        rule_of(cfg, opt["arm_set"]),
        cfg["power"]["n_mc"],
        cfg["power"]["seed"],
    )
    out = {k: v for k, v in pw.items() if k in ("h1", "vanish", "h2_sham", "h2", "h1_and_h2")}
    out["inflation_w0"] = {a: float(f[0]) for a, f in infl.items()}
    return out


def drift_levels(n: int) -> list[float]:
    """``n`` equal-probability quantiles of a standard normal (mid-points), SD rescaled to 1."""
    q = np.array([RA.norm_ppf((i + 0.5) / n) for i in range(n)])
    return list(q / q.std())


# -- export (hardware group, FakeKingston only) --------------------------------------------------------------


def widen(qc: Any, n: int) -> Any:
    """``qc`` on at least ``n`` qubits (a remapped circuit's spectator may lie beyond its
    highest-numbered qubit); unchanged if it already has them."""
    if qc.num_qubits >= n:
        return qc
    from qiskit import QuantumCircuit

    out = QuantumCircuit(n)
    for r in qc.cregs:
        out.add_register(r)
    out.compose(qc, qubits=list(range(qc.num_qubits)), clbits=qc.clbits, inplace=True)
    return out


def wire_sequences(sig: list[tuple]) -> dict[int, list[tuple]]:
    """Per qubit, the ordered instructions touching it (a CZ's qubits as a set: it is
    symmetric). Equal sequences on every qubit = the same circuit up to the interleaving of
    instructions on disjoint qubits, which the transpiler orders by qubit index."""
    out: dict[int, list[tuple]] = {}
    for n, qs, p in sig:
        k = (n, tuple(sorted(qs)) if n == "cz" else qs, p)
        for q in qs:
            out.setdefault(q, []).append(k)
    return out


def dag_equal(text_a: str, text_b: str) -> bool:
    """Circuit equivalence of two QASM 3 programs as DAGs (Qiskit's DAG equality: the same
    operations on the same wires in the same order per wire)."""
    from qiskit import qasm3
    from qiskit.converters import circuit_to_dag

    a, b = qasm3.loads(text_a), qasm3.loads(text_b)
    n = max(a.num_qubits, b.num_qubits)
    return circuit_to_dag(widen(a, n)) == circuit_to_dag(widen(b, n))


def remap(text: str, mapping: dict[int, int]) -> str:
    return re.sub(
        r"\$(\d+)", lambda m_: f"${mapping.get(int(m_.group(1)), int(m_.group(1)))}", text
    )


def export(cfg: dict[str, Any], workers: int) -> None:
    import phase7_ablation as PA
    from qiskit import ClassicalRegister, QuantumCircuit, qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    from qmem import hardware as hw

    sd = sdir(cfg)
    out = new_run_dir(cfg["experiment"]["name"] + "_export")
    backend = FakeKingston()
    target = backend.target
    dev = load_device(cfg)
    rules = selection_rules(cfg)
    ref = cfg["reference"]
    rows = REP.screen(dev, rules)
    step = qasm3.loads((rdir(cfg, "phase6_dir") / "step.qasm").read_text())
    pairs = [(r["memory"], r["output"]) for r in rows if not r["screen_reasons"]]
    pairs += [tuple(p) for p in rules.named if tuple(p) not in pairs]
    pairs.append((ref["memory"], ref["output"]))
    npz: dict[str, np.ndarray] = {}
    noise_info = {}
    t0 = time.time()
    for m, o in pairs:
        noise, info = hw.backend_noise(backend, [o, m], cfg["model"]["feedforward_time"])
        pc = A.export_pieces(step, noise, 1, 0)
        npz.update(pc.to_dict(key(m, o)))
        noise_info[key(m, o)] = jsonable(info)
    np.savez_compressed(sd / "pieces.npz", **npz)
    log(out, f"[export] pieces of {len(pairs)} pairs in {time.time() - t0:.1f} s")

    # the selection (the same numpy code as the dry run) decides which circuits to build
    _, _, infl, _ = calibrate_inflation(cfg, out)
    sel, _ = run_selection(cfg, dev, load_pieces(cfg), infl, workers, out)
    chosen = [(r["memory"], r["output"], r["spectator"]) for r in sel.chosen]
    log(out, f"[export] selection: {sel.statement}: {chosen}")
    spec = {(r["memory"], r["output"]): r["spectator"] for r in sel.rows}
    build = [(ref["memory"], ref["output"], ref["spectator"])]
    build += [(m, o, spec[(m, o)]) for m, o in rules.named]
    build += [c for c in chosen if c not in build]

    acfg = copy.deepcopy(PA.load_config(REPO_ROOT / cfg["experiment"]["ablation_config"]))
    acfg["circuits"]["delays"] = cfg["circuits"]["delays"]
    s6 = PA.phase6_summary(acfg)
    dur = PA.durations(acfg, s6, target)
    length = cfg["circuits"]["length"]
    full_text = (rdir(cfg, "phase6_dir") / "circuit.qasm").read_text()
    ref_hash = json.loads((rdir(cfg, "ablation_dir") / "circuits.json").read_text())["qasm_sha256"]
    (sd / "qasm").mkdir(exist_ok=True)
    built: dict[tuple, dict[str, Any]] = {}
    pair_rec: dict[str, Any] = {}
    ref_sig: dict[str, list] = {}
    ref_text: dict[str, str] = {}
    for m, o, s in build:
        full = widen(
            qasm3.loads(remap(full_text, {ref["memory"]: m, ref["output"]: o})), max(m, o, s) + 1
        )
        rec: dict[str, Any] = {"memory": m, "output": o, "spectator": s, "arms": {}}
        built[(m, o)] = {}
        back = {m: ref["memory"], o: ref["output"], s: ref["spectator"]}
        for arm in cfg["circuits"]["arms"]:
            qc = A.build_variant(full, arm, m, o, dur, length, s if arm == "A2s_spec" else None)
            tq = PA.isa(qc, backend, acfg)
            text = qasm3.dumps(tq)
            h = hashlib.sha256(text.encode()).hexdigest()
            relabel_text = remap(text, back)
            relabel = hashlib.sha256(relabel_text.encode()).hexdigest()
            sig = PA.signature(tq)
            if (m, o) == (ref["memory"], ref["output"]):
                ref_sig[arm] = sig
            used = sorted({tq.find_bit(q).index for i in tq.data for q in i.qubits})
            czs = {
                tuple(tq.find_bit(q).index for q in i.qubits)
                for i in tq.data
                if i.operation.name == "cz"
            }
            if (m, o) == (ref["memory"], ref["output"]):
                ref_text[arm] = text
            rec["arms"][arm] = {
                "sha256": h,
                "relabelled_sha256": relabel,
                "relabelled_text_equals_reference": relabel == ref_hash.get(arm),
                "relabelled_equals_reference": dag_equal(relabel_text, ref_text[arm]),
                "reference_sha256": ref_hash.get(arm),
                "duration_fake_s": tq.estimate_duration(target, unit="s"),
                "count_ops": dict(tq.count_ops()),
                "qubits_used": used,
                "cz_edges": sorted(czs),
                "cz_native": all(e in target["cz"] for e in czs),
                "swaps": int(dict(tq.count_ops()).get("swap", 0)),
                "file": f"qasm/{key(m, o)}_{arm}.qasm",
            }
            built[(m, o)][arm] = tq
            if (m, o) != (ref["memory"], ref["output"]):
                (sd / "qasm" / f"{key(m, o)}_{arm}.qasm").write_text(text)
        bd = rec["arms"]["baseline"]["duration_fake_s"]
        rec["duration_s"] = {a: r["duration_fake_s"] for a, r in rec["arms"].items()}
        rec["duration_minus_baseline_s"] = {a: v - bd for a, v in rec["duration_s"].items()}
        qd = {q: dev.qubits[q] for q in (m, o, s)}
        rec["calibration_fakekingston"] = {
            f"q{q}": {
                "t1_us": v.t1 * 1e6,
                "t2_us": v.t2 * 1e6,
                "readout": v.readout,
                "sx": v.sx,
                "measure_time_s": v.measure_time,
                "reset_time_s": v.reset_time,
            }
            for q, v in qd.items()
        }
        rec["cz_error"] = dev.cz_error(m, o)
        rec["couplers"] = [list(e) for e in dev.couplers([m, o, s])]
        rec["noise"] = noise_info[key(m, o)]
        rec["routing"] = "trivial layout, level 0, no SWAPs (every CZ on a native coupler)"
        pair_rec[key(m, o)] = rec
    # differences from the reference, instruction by instruction (after relabelling)
    for rec in pair_rec.values():
        m, o, s = rec["memory"], rec["output"], rec["spectator"]
        back = {m: ref["memory"], o: ref["output"], s: ref["spectator"]}
        for arm, tq in built[(m, o)].items():
            sig = [(n, tuple(back.get(q, q) for q in qs), p) for n, qs, p in PA.signature(tq)]
            a_w, b_w = wire_sequences(sig), wire_sequences(ref_sig[arm])
            rec["arms"][arm]["per_qubit_sequence_differences_from_reference"] = sum(
                a_w.get(q) != b_w.get(q) for q in set(a_w) | set(b_w)
            )
            rec["arms"][arm]["text_order_differences_from_reference"] = sum(
                a != b for a, b in zip(sig, ref_sig[arm], strict=False)
            ) + abs(len(sig) - len(ref_sig[arm]))

    # the two chosen pairs side by side in one circuit per arm (parallel option)
    par: dict[str, Any] = {}
    if len(chosen) >= 2:
        (ma, oa, _), (mb, ob, _) = chosen[:2]
        par = {"pairs": [key(ma, oa), key(mb, ob)], "arms": {}, "duration_s": {}}
        for arm in cfg["circuits"]["arms"]:
            qa, qb = built[(ma, oa)][arm], built[(mb, ob)][arm]
            regs = list(qa.cregs) + [ClassicalRegister(r.size, "b_" + r.name) for r in qb.cregs]
            qc = QuantumCircuit(qa.qubits, *regs)
            qc.compose(
                qa,
                qubits=qa.qubits,
                clbits=[c for r in regs[: len(qa.cregs)] for c in r],
                inplace=True,
            )
            qc.compose(
                qb,
                qubits=qa.qubits,
                clbits=[c for r in regs[len(qa.cregs) :] for c in r],
                inplace=True,
            )
            tq = PA.isa(qc, backend, acfg)
            text = qasm3.dumps(tq)
            fa = {ma, oa} | ({chosen[0][2]} if arm == "A2s_spec" else set())
            fb = {mb, ob} | ({chosen[1][2]} if arm == "A2s_spec" else set())

            def part(c: Any, qs: set[int]) -> list[tuple]:
                return [x for x in PA.signature(c) if set(x[1]) <= qs]

            par["arms"][arm] = {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "pair_a_preserved": part(tq, fa) == part(qa, fa),
                "pair_b_preserved": part(tq, fb) == part(qb, fb),
                "only_pair_qubits": all(
                    set(x[1]) <= fa or set(x[1]) <= fb for x in PA.signature(tq)
                ),
                "file": f"qasm/parallel_{arm}.qasm",
            }
            par["duration_s"][arm] = tq.estimate_duration(target, unit="s")
            (sd / "qasm" / f"parallel_{arm}.qasm").write_text(text)
    save_json(
        sd / "circuits.json",
        {
            "meta": run_metadata(jsonable(cfg), None),
            "backend": backend.name,
            "delays_s": {"reset": dur.reset, "measure": dur.measure, "x": dur.x, "dt": dur.dt},
            "chosen": [key(m, o) for m, o, _ in chosen],
            "pairs": pair_rec,
            "parallel": par,
        },
    )
    ok = all(
        r["relabelled_equals_reference"] for p in pair_rec.values() for r in p["arms"].values()
    )
    log(out, f"[export] circuits of {len(build)} pairs; relabelled hashes equal #20's: {ok}")


# -- selection run -----------------------------------------------------------------------------------------


def run_selection(cfg, dev, pieces, infl, workers, out) -> tuple[REP.Selection, dict]:
    rules = selection_rules(cfg)
    rows = REP.screen(dev, rules)
    sd_ = cfg["selection_design"]
    shots = {a: sd_[a] for a in REP.ARMS}
    if infl is None:
        infl = {a: 1.0 for a in REP.ARMS}
    tasks = []
    for r in rows:
        if r["screen_reasons"]:
            continue
        k = key(r["memory"], r["output"])
        tasks.append(
            {
                "key": k,
                "pieces": pieces_kw(pieces[k]),
                "shortfall": cfg["model"]["shortfall"],
                "n": cfg["model"]["moment_shots"],
                "seed": cfg["model"]["moment_seed"],
                "shots": shots,
                "n_jobs": sd_["n_jobs"],
                "inflation": infl,
                "rule": rule_of(cfg),
                "n_mc": sd_["n_mc"],
                "mc_seed": sd_["seed"],
            }
        )
    t0 = time.time()
    res = {r["key"]: r for r in run_pool(_power_task, tasks, workers)}
    log(out, f"[selection] power of {len(tasks)} screened pairs in {time.time() - t0:.0f} s")

    def power(row: dict[str, Any]) -> dict[str, float]:
        r = res[key(row["memory"], row["output"])]
        return {k: v for k, v in r.items() if k not in ("key", "moments")}

    return REP.select(dev, rules, power), res


# -- the dry run ---------------------------------------------------------------------------------------------


def dry_run_local(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    t_start = time.time()
    sd = sdir(cfg)
    out = new_run_dir(cfg["experiment"]["name"])
    save_json(out / "config.json", {"meta": run_metadata(jsonable(cfg), cfg["power"]["seed"])})
    workers = args.workers or cfg["experiment"]["workers"]
    dev = load_device(cfg)
    pieces = load_pieces(cfg)
    rules = selection_rules(cfg)
    ref = cfg["reference"]
    ref_key = key(ref["memory"], ref["output"])
    model = cfg["model"]

    # 1. the drift-conditional SE calibration on #27 (needed by the selection's power)
    live = live_reference_pieces(cfg)
    live_m, a27, infl, infl_rows = calibrate_inflation(cfg, out)
    write_csv(sd / "inflation.csv", infl_rows)

    # 2. #27's data under this rule (no delay-only sham was run: its part of H2 is not evaluable)
    r27 = rule_of(cfg, "no_delay_sham")
    d27 = REP.analyse_pair(a27, r27, cfg["rule"]["armjob_alpha"])
    prim = {
        a: {
            "value": v["value"],
            "se": v["se_primary"],
            "lojo_value": v["lojo_value"],
            "lojo_se": v["lojo_se_primary"],
        }
        for a, v in a27.items()
    }
    dprim = REP.decide(prim, r27)
    s27 = json.loads((rdir(cfg, "hw4c_dir") / "summary.json").read_text())
    check27 = {
        "pre_registered_call_27": s27["decision"]["primary"],
        "a2_bound_27_summary": (
            abs(s27["arms"]["A2"]["w0"]) + 1.959963984540054 * s27["arms"]["A2"]["se0"]
        )
        / abs(s27["arms"]["baseline"]["w0"]),
        "this_rule_on_27_primary_se": {
            "a2_bound": float(dprim["a2_bound"]),
            "a2_call_all_jobs": "vanishes"
            if bool(dprim["vanish_all_jobs"])
            else str(dprim["a2_call"]),
            "a2_lojo_max_bound": float(dprim["a2_lojo_max_bound"]),
            "vanish_with_lojo": bool(dprim["vanish"]),
            "sham_w0_ratio": float(dprim["A2s_spec_ratio"][0]),
            "sham_w0_equivalent": bool(dprim["A2s_spec_equiv_w0"]),
            "sham_equivalent_per_stat": dict(
                zip(REP.STATS, map(bool, dprim["A2s_spec_equiv"]), strict=True)
            ),
            "label": REP.call_label(dprim),
        },
        "this_rule_on_27_drift_conditional_se": {
            "a2_bound": float(d27["decision"]["a2_bound"]),
            "a2_lojo_max_bound": float(d27["decision"]["a2_lojo_max_bound"]),
            "sham_w0_ratio": float(d27["decision"]["A2s_spec_ratio"][0]),
            "sham_w0_ratio_se": float(d27["decision"]["A2s_spec_ratio_se"][0]),
            "sham_equivalent_per_stat": dict(
                zip(REP.STATS, map(bool, d27["decision"]["A2s_spec_equiv"]), strict=True)
            ),
            "label": d27["label"],
            "flagged_jobs": d27["flagged_jobs"],
            "interaction": [{k: v for k, v in t.items()} for t in d27["interaction"]],
        },
    }
    log(
        out,
        f"[#27] this rule: {check27['this_rule_on_27_primary_se']['label']} "
        f"(bound {dprim['a2_bound']:.4f}, LOJO max {dprim['a2_lojo_max_bound']:.4f}); "
        f"drift-conditional: {d27['label']}",
    )

    # 3. selection on the snapshot
    sel, pres = run_selection(cfg, dev, pieces, infl, workers, out)
    cand_rows = []
    for r in sel.rows:
        cand_rows.append({k: v for k, v in r.items() if k not in ("screen_reasons",)})
    cand_rows.sort(key=lambda r: (r.get("rank") or 10**6, r["memory"], r["output"]))
    write_csv(sd / "candidates.csv", cand_rows)
    chosen = [key(r["memory"], r["output"]) for r in sel.chosen]
    log(
        out,
        f"[selection] {len(sel.rows)} directed pairs, {sum(1 for r in sel.rows if not r['screen_reasons'])} "
        f"pass the screen, {sel.n_qualified} qualify; chosen {chosen} ({sel.statement})",
    )

    # 4. predictions: each pair's calibrated instrument, the shortfall truth, #30's classical model
    named = [key(m, o) for m, o in rules.named]
    show = list(dict.fromkeys([*chosen, *named]))
    pred_rows = []
    preds = {}
    ref_fake = quantum_predictions(pieces[ref_key], 0.0)
    pred_rows += prediction_rows(
        label(ref["memory"], ref["output"]) + " (FakeKingston)", "calibrated", ref_fake
    )
    ref_live = quantum_predictions(live, 0.0)
    pred_rows += prediction_rows(
        label(ref["memory"], ref["output"]) + " (#27 live)", "calibrated", ref_live
    )
    pred_rows += prediction_rows(
        label(ref["memory"], ref["output"]) + " (#27 live)",
        f"truth (shortfall {model['shortfall']})",
        quantum_predictions(live, live_m["dephase"]),
    )
    for k in show:
        m, o = (int(x[1:]) for x in k.split("_"))
        pc = pieces[k]
        p = REP.shortfall_dephasing(pc, model["shortfall"], SETTINGS)
        preds[k] = {"calibrated": quantum_predictions(pc, 0.0), "truth": quantum_predictions(pc, p)}
        pred_rows += prediction_rows(label(m, o), "calibrated", preds[k]["calibrated"])
        pred_rows += prediction_rows(
            label(m, o), f"truth (shortfall {model['shortfall']})", preds[k]["truth"]
        )
    cl = classical_predictions(cfg)
    pred_rows += prediction_rows(
        "any pair (transferred)", f"classical {model['classical_class']} (#30)", cl
    )
    write_csv(sd / "predictions.csv", pred_rows)

    # 5. moments of the chosen pairs (large samples) and the options table
    circ = load_circuits(cfg)
    cm = cost_model(cfg)
    mom = {}
    for k in chosen:
        pm = pair_moments(
            pieces[k], model["shortfall"], model["moment_shots_chosen"], model["moment_seed"]
        )
        mom[k] = pm["moments"]
    mom_ref = live_m["moments"]
    pc_up = pilot_cost(cfg, chosen, circ, cm)
    rows = []
    o = cfg["options"]
    pw = cfg["power"]
    t0 = time.time()
    for mode, arm_set, n_bs, n_a2 in itertools.product(
        o["modes"], o["arm_sets"], o["base_sham_shots"], o["a2_shots"]
    ):
        r = evaluate_option(
            cfg,
            mode,
            arm_set,
            n_bs,
            n_a2,
            chosen,
            mom,
            infl,
            circ,
            cm,
            pw["n_mc"],
            pw["seed"],
            pc_up,
        )
        if r:
            rows.append(r)
    write_csv(sd / "options.csv", rows)
    log(out, f"[options] {len(rows)} options in {time.time() - t0:.0f} s")
    rec = recommend(rows, cfg["budget"]["cap_s"], cfg["selection"]["power_min"])
    opt = rec["option"]
    if opt:
        log(
            out,
            f"[recommend] {rec['verdict']}: {opt['mode']} / {opt['arm_set']} / {opt['shots_base_sham']} "
            f"(A2 {opt['shots_A2']}); upper {opt['upper_with_pilot_s']:.1f} s; min pair power {opt['min_pair_joint']:.3f}",
        )

    # 6. sensitivity of the recommended option
    sens = []
    if opt:
        use = chosen[:1] if opt["mode"] == "one" else chosen[:2]
        shots = option_shots(opt["arm_set"], opt["shots_base_sham"], opt["shots_A2"])
        rule = rule_of(cfg, opt["arm_set"])
        iw = REP.STATS.index("w0")
        for p in use:

            def run(tag, mom_p=None, infl_p=None, shift=None, p=p):
                design = REP.Design(shots, opt["cost_jobs"], infl_p or {a: infl[a] for a in shots})
                r = REP.power_mc(mom_p or mom[p], design, rule, pw["n_mc"], pw["seed"], shift)
                sens.append({"pair": p, "scenario": tag, **r})

            run("design (true A2 ratio 0, sham ratio 1, shortfall 0.75, drift-conditional SEs)")
            for tr in pw["true_ratio_sensitivity"]:
                b = mom[p]["baseline"].mean
                mu = mom[p]["A2"].mean.copy()
                mu[iw] = tr * b[iw]
                run(f"true A2 ratio {tr}", shift={"A2": mu})
            for sr in pw["sham_ratio_sensitivity"]:
                sh = {s: REP.scaled_shift(mom[p][s], ratio_w0=sr) for s in REP.SHAMS}
                run(f"sham w(0) ratio {sr}", shift=sh)
            run("no drift inflation", infl_p={a: 1.0 for a in shots})
            for sf in model["shortfall_sensitivity"]:
                pm = pair_moments(pieces[p], sf, model["moment_shots"], model["moment_seed"])
                run(f"shortfall {sf}", mom_p=pm["moments"])
        # the reference pair at the same design, for comparison
        design = REP.Design(shots, opt["cost_jobs"], {a: infl[a] for a in shots})
        sens.append(
            {
                "pair": ref_key + " (#27 live)",
                "scenario": "design",
                **REP.power_mc(mom_ref, design, rule, pw["n_mc"], pw["seed"]),
            }
        )
    write_csv(sd / "sensitivity.csv", sens)

    # 7. the plan: schedule, pilot, costs
    plan = make_plan(cfg, opt, chosen, circ, cm) if opt else None
    if plan:
        write_csv(sd / "schedule.csv", F.schedule_rows(plan["jobs"]))
        write_csv(sd / "pilot_schedule.csv", F.schedule_rows([plan["pilot"]]))

    # 8. the full-pipeline validation at the recommended option
    val_rows: list[dict[str, Any]] = []
    if opt and not args.skip_validation:
        v = cfg["validation"]
        use = chosen[:1] if opt["mode"] == "one" else chosen[:2]
        tasks = []
        for p in use:
            pc = pieces[p]
            p0 = REP.shortfall_dephasing(pc, model["shortfall"], SETTINGS)
            w_a = A.predict(REP.arm_process("baseline", pc, p0), SETTINGS)["w0"]
            w_b = A.predict(REP.arm_process("baseline", pc, p0 + 1e-3), SETTINGS)["w0"]
            slope = (w_b - w_a) / 1e-3 / abs(w_a)  # relative w(0) change per unit dephasing
            levels = drift_levels(v["drift_levels"])
            arms = list(arms_of(opt["arm_set"]))
            jobs_p = F.make_schedule(
                option_shots(opt["arm_set"], opt["shots_base_sham"], opt["shots_A2"]),
                opt["cost_jobs"],
                cfg["schedule"]["max_chunk"],
                cfg["schedule"]["seed"],
            )
            for scen in ("drift", "armjob"):
                flip = {v["armjob_job"]: v["armjob_flip"]} if scen == "armjob" else None
                for rep in range(args.reps or v["reps"]):
                    tasks.append(
                        {
                            "pair": p,
                            "scenario": scen,
                            "rep": rep,
                            "pieces": pieces_kw(pc),
                            "dephase": p0,
                            "slope": slope,
                            "drift_sd": {a: v["chunk_drift_sd"][a] for a in arms},
                            "levels": levels,
                            "arms": arms,
                            "jobs": jobs_p,
                            "flip": flip,
                            "n_boot": v["n_boot"],
                            "seed": v["seed"] + 1000 * rep + (0 if scen == "drift" else 500000),
                            "rule": rule_of(cfg, opt["arm_set"]),
                            "alpha": cfg["rule"]["armjob_alpha"],
                        }
                    )
        t0 = time.time()
        val_rows = run_pool(_validation_task, tasks, workers)
        log(out, f"[validation] {len(tasks)} simulated datasets in {time.time() - t0:.0f} s")
        write_csv(sd / "validation.csv", val_rows)
    val_summary = []
    for (p, scen), grp in itertools.groupby(
        sorted(val_rows, key=lambda r: (r["pair"], r["scenario"])),
        key=lambda r: (r["pair"], r["scenario"]),
    ):
        g = list(grp)
        n = len(g)
        val_summary.append(
            {
                "pair": p,
                "scenario": scen,
                "reps": n,
                **{
                    f"p_{k}": sum(r[k] for r in g) / n
                    for k in ("h1", "vanish_all_jobs", "vanish", "h2_sham", "h2", "h1_and_h2")
                },
                "p_flagged_any": sum(bool(r["flagged_jobs"]) for r in g) / n,
                "labels": {
                    lab: sum(r["label"] == lab for r in g)
                    for lab in sorted({r["label"] for r in g})
                },
                "median_a2_bound": float(np.median([r["a2_bound"] for r in g])),
                "mean_base_se_over_primary": float(
                    np.mean([r["base_se_w0"] / r["base_se_primary_w0"] for r in g])
                ),
                "engine_at_simulated_se": engine_at(cfg, opt, mom[p], g),
                "engine_at_design_se": {
                    k: v
                    for k, v in REP.power_mc(
                        mom[p],
                        REP.Design(
                            option_shots(opt["arm_set"], opt["shots_base_sham"], opt["shots_A2"]),
                            opt["cost_jobs"],
                            {a: infl[a] for a in arms_of(opt["arm_set"])},
                        ),
                        rule_of(cfg, opt["arm_set"]),
                        cfg["power"]["n_mc"],
                        cfg["power"]["seed"],
                    ).items()
                    if k in ("h1", "vanish", "h2_sham", "h2", "h1_and_h2")
                },
            }
        )

    # 9. circuits and hashes (hardware group), if available
    hw_check = circuit_check(cfg) if not args.skip_circuits else {"skipped": True}

    # 10. summary
    ref_row = {
        "w0_calibrated_fake": ref_fake["baseline"]["w0"],
        "w0_calibrated_live": ref_live["baseline"]["w0"],
    }
    summary = {
        "meta": run_metadata(jsonable(cfg), cfg["power"]["seed"]),
        "run_dir": str(out.relative_to(REPO_ROOT)),
        "snapshot": {
            "backend": dev.backend,
            "calibration_date": dev.date,
            "note": "dry run on FakeKingston's committed snapshot; the real selection runs on the live snapshot at Stage 4 only, after the owner approves",
        },
        "selection": {
            "n_candidates": len(sel.rows),
            "n_screened": sum(1 for r in sel.rows if not r["screen_reasons"]),
            "n_qualified": sel.n_qualified,
            "chosen": chosen,
            "statement": sel.statement,
            "named": {
                k: next(
                    (
                        {
                            "status": r["status"],
                            "reasons": r["reasons"],
                            "rank": r.get("rank"),
                            "power_joint": r.get("power_joint"),
                        }
                        for r in sel.rows
                        if key(r["memory"], r["output"]) == k
                    ),
                    None,
                )
                for k in named
            },
            "chosen_rows": [
                {k: v for k, v in r.items() if k != "screen_reasons"} for r in sel.chosen
            ],
        },
        "reference": ref_row,
        "inflation": {a: dict(zip(REP.STATS, map(float, v), strict=True)) for a, v in infl.items()},
        "rule_on_27": check27,
        "recommendation": rec,
        "sensitivity": sens,
        "plan": None
        if not plan
        else {
            "jobs": len(plan["jobs"]),
            "pubs": sum(len(j) for j in plan["jobs"]),
            "estimate_s": plan["estimate_s"],
            "upper_s": plan["upper_s"],
            "pilot_cost": plan["pilot_cost"],
            "job_costs": plan["job_costs"],
            "problems": plan["problems"],
            "arms": plan["arms"],
            "durations_s": plan["durations"],
        },
        "validation": val_summary,
        "circuits": hw_check,
        "classical": {
            "class": model["classical_class"],
            "predictions": {
                a: {k: v for k, v in p.items() if k in ("w0", "w1", "w2", "p1", "lag1")}
                for a, p in cl.items()
            },
            "note": "transferred unchanged from 59/75; the class has no parameter-free prediction for a new pair. Any invasive classical A2 gives w_A2 = 0 on any pair (#30, Theorem 3), so a replication cannot separate it from the quantum model",
        },
        "elapsed_s": time.time() - t_start,
    }
    save_json(sd / "summary.json", jsonable(summary))
    save_json(out / "summary.json", jsonable(summary))
    log(out, f"[done] {time.time() - t_start:.0f} s; summary in {sd / 'summary.json'}")


def circuit_check(cfg: dict[str, Any], only: list[str] | None = None) -> dict[str, Any]:
    """Rebuild the committed circuits and compare hashes (``hardware`` group; else skipped)."""
    try:
        import phase7_ablation as PA
        from qiskit import qasm3
        from qiskit_ibm_runtime.fake_provider import FakeKingston
    except ImportError:
        return {"skipped": "hardware group not installed"}
    circ = load_circuits(cfg)
    if circ is None:
        return {"skipped": "no circuits.json (run --export)"}
    backend = FakeKingston()
    acfg = copy.deepcopy(PA.load_config(REPO_ROOT / cfg["experiment"]["ablation_config"]))
    acfg["circuits"]["delays"] = cfg["circuits"]["delays"]
    dur = PA.durations(acfg, PA.phase6_summary(acfg), backend.target)
    full_text = (rdir(cfg, "phase6_dir") / "circuit.qasm").read_text()
    ref = cfg["reference"]
    rows = {}
    for k, rec in circ["pairs"].items():
        if only is not None and k not in only:
            continue
        m, o, s = rec["memory"], rec["output"], rec["spectator"]
        full = widen(
            qasm3.loads(remap(full_text, {ref["memory"]: m, ref["output"]: o})), max(m, o, s) + 1
        )
        for arm, r in rec["arms"].items():
            qc = A.build_variant(
                full, arm, m, o, dur, cfg["circuits"]["length"], s if arm == "A2s_spec" else None
            )
            text = qasm3.dumps(PA.isa(qc, backend, acfg))
            h = hashlib.sha256(text.encode()).hexdigest()
            f = sdir(cfg) / r["file"]
            file_ok = (not f.exists()) or hashlib.sha256(f.read_bytes()).hexdigest() == r["sha256"]
            rows[f"{k}:{arm}"] = {
                "rebuilt_equals_recorded": h == r["sha256"],
                "file_matches": file_ok,
                "relabelled_equals_reference": r["relabelled_equals_reference"],
            }
    return {
        "rows": rows,
        "all_match": all(v["rebuilt_equals_recorded"] and v["file_matches"] for v in rows.values()),
        "all_equivalent": all(v["relabelled_equals_reference"] for v in rows.values()),
    }


# -- --plan ---------------------------------------------------------------------------------------------


def print_plan(cfg: dict[str, Any]) -> None:
    s = json.loads((sdir(cfg) / "summary.json").read_text())
    rec = s["recommendation"]
    opt = rec["option"]
    if not opt:
        print(f"no recommended option ({rec['verdict']}: {rec.get('reason')})")
        return
    chosen = s["selection"]["chosen"]
    circ = load_circuits(cfg)
    plan = make_plan(cfg, opt, chosen, circ, cost_model(cfg))
    print("PHASE 8 STAGE 1 REPLICATION: UNSUBMITTED EXECUTION PLAN (no submit path exists here)")
    print(
        f"snapshot: {s['snapshot']['backend']} {s['snapshot']['calibration_date']} (dry run; "
        "the live selection runs at Stage 4 after the owner approves)"
    )
    print(
        f"pairs: {chosen}  mode: {opt['mode']}  arms: {opt['arm_set']}  "
        f"shots: baseline/shams {opt['shots_base_sham']}, A2 {opt['shots_A2']}"
    )
    print(
        "power per pair (P(H1 and H2)): "
        + ", ".join(
            f"{p} {opt.get(p + '_joint', float('nan')):.3f}"
            for p in chosen[: (1 if opt["mode"] == "one" else 2)]
        )
    )
    pc = plan["pilot_cost"]
    print(
        f"\npilot (1 job, stop the pair on any failed gate, no retry, no pair substitution): "
        f"{pc['shots']} shots, estimate {pc['estimate_s']:.2f} s, upper {pc['upper_s']:.2f} s"
    )
    for p in plan["pilot"]:
        print(f"   {p.key:40s} {p.shots:6d}")
    for j, (job, c) in enumerate(zip(plan["jobs"], plan["job_costs"], strict=True)):
        arms = {}
        for p in job:
            arms[p.arm] = arms.get(p.arm, 0) + p.shots
        print(
            f"job {j}: {len(job)} pubs, {c['shots']} shots, estimate {c['estimate_s']:.2f} s, upper "
            f"{c['upper_s']:.2f} s; " + ", ".join(f"{a} {n}" for a, n in arms.items())
        )
    print(
        f"\ntotal (pilot + main): estimate {plan['estimate_s']:.1f} s, upper bound {plan['upper_s']:.1f} s "
        f"(max(charged, upper) accounting); plan cap {cfg['budget']['cap_s']} s; "
        f"Open plan {cfg['budget']['open_plan_s']} s / 28 d shared with Stages 2-3"
    )
    print(
        "schedule: pub order in job j is default_rng([seed, j]) with seed "
        f"{cfg['schedule']['seed']}; per-pub execution spans must be recorded (prereg section 7)"
    )
    if plan["problems"]:
        print("SCHEDULE PROBLEMS:", plan["problems"])
    print(
        "NOT AUTHORISED: no figure here is an approval. Stage 4 needs the owner's explicit approval."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true")
    g.add_argument("--dry-run-local", action="store_true")
    g.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None, help="validation reps (default: config)")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--skip-circuits", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export:
        export(cfg, args.workers or cfg["experiment"]["workers"])
    elif args.plan:
        print_plan(cfg)
    else:
        dry_run_local(cfg, args)


if __name__ == "__main__":
    main()
