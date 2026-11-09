"""Phase 8 Stage 0A (issue #29): independent re-analysis of the Phase 7 Stage 4c result (#27).

Local only: no QPU, no IBM service, no network. From the committed shots in
``results/summaries/phase7_hw_4c/``:

1. provenance: SHA-256 of every Stage 4/4b/4c shot, QASM, circuit and calibration file, the
   IBM job ids and the commits (``--freeze`` writes ``provenance.json`` once; every run checks
   the files against it);
2. reproduction of every number in Stage 4c's ``summary.json`` by ``qmem.reanalysis`` (its own
   decoding, window count, determinant, resampling and decision arithmetic), compared with the
   pipeline's ``qmem.witness`` count on the same shots;
3. the fragility table of A2's vanish bound under the predeclared choices
   (``[fragility]`` in the config), the power of a replicate and the shots for a bound at 70%
   of the threshold;
4. SEs conditional on the job-scale P(1) drift, ``qmem.drift``'s segment tests, the balance
   of the interleaved schedule, and the drift bias needed to cross 10%;
5. baseline against the sham (A2s_spec) per job, with heterogeneity tests;
6. TOST equivalence of the sham at 5, 10 and 20%; the shots for Phase 8's margin;
7. the baseline's shortfall against H_q75.

    uv run python experiments/phase8_stage0.py --config experiments/configs/phase8_stage0.toml
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from qmem import drift as D
from qmem import reanalysis as R
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

ARMS = ("baseline", "A2", "A2s_spec", "A1", "A3", "D1")
STAT_KEYS = ("w0", "w1", "w2", "w3", "se_shot0", "se_block0", "se0", "se1", "se2", "se3", "z0")
STAT_KEYS += ("p1", "lag1", "lag1_se")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def log(out: Path, msg: str) -> None:
    print(msg, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(msg + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


# -- 1. provenance ----------------------------------------------------------------------------


def provenance_files(cfg: dict[str, Any]) -> list[str]:
    out = set()
    for g in cfg["provenance"]["globs"]:
        for p in glob.glob(str(REPO_ROOT / g)):
            if Path(p).is_file():
                out.add(str(Path(p).relative_to(REPO_ROOT)))
    return sorted(out)


def provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    files = provenance_files(cfg)
    rows = {}
    for rel in files:
        rows[rel] = {
            "sha256": R.sha256(REPO_ROOT / rel),
            "bytes": (REPO_ROOT / rel).stat().st_size,
            "added_in": git("log", "--diff-filter=A", "--format=%H", "--", rel).splitlines()[-1:]
            or [None],
        }
        rows[rel]["added_in"] = rows[rel]["added_in"][0]
    jobs = []
    for rel in files:
        if Path(rel).name.startswith("job_") and rel.endswith(".json"):
            j = json.loads((REPO_ROOT / rel).read_text())
            jobs.append(
                {
                    "file": rel,
                    "job_id": j.get("job_id"),
                    "stage": j.get("stage"),
                    "status": j.get("status"),
                    "shots": j.get("shots"),
                    "submitted_at": j.get("submitted_at"),
                    "approval_url": j.get("approval_url"),
                }
            )
    run_commits = {}
    for d in ("phase7_hw", "phase7_hw_4b", "phase7_hw_4c"):
        for name in ("summary.json", "config_meta.json", "live_dry_run.json"):
            p = REPO_ROOT / "results" / "summaries" / d / name
            if p.exists():
                meta = json.loads(p.read_text())
                meta = meta.get("meta", meta)
                if "git_commit" in meta:
                    run_commits[f"{d}/{name}"] = {
                        "git_commit": meta["git_commit"],
                        "git_dirty": meta.get("git_dirty"),
                    }
    return {"files": rows, "jobs": jobs, "run_commits": run_commits}


def check_provenance(cfg: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    now = provenance_files(cfg)
    changed = (
        [r for r in frozen["files"] if R.sha256(REPO_ROOT / r) != frozen["files"][r]["sha256"]]
        if all((REPO_ROOT / r).exists() for r in frozen["files"])
        else ["missing files"]
    )
    return {
        "n_files": len(frozen["files"]),
        "changed": changed,
        "missing": [r for r in frozen["files"] if not (REPO_ROOT / r).exists()],
        "new": [r for r in now if r not in frozen["files"]],
        "ok": not changed and set(now) == set(frozen["files"]),
    }


# -- 2. reproduction ------------------------------------------------------------------------


def load_main(cfg: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    d = REPO_ROOT / cfg["experiment"]["stage4c_dir"]
    return R.load_jobs([d / f"shots_{j}.npz" for j in cfg["experiment"]["main_jobs"]])


def arm_seed(cfg: dict[str, Any], data: dict[str, Any], arm: str) -> int:
    return cfg["analysis"]["seed"] + 11 * list(data).index(arm)


def all_arm_stats(cfg, data) -> dict[str, dict[str, Any]]:
    a = cfg["analysis"]
    return {
        arm: R.arm_stats(
            d["shots"],
            d["jobs"],
            a["k_max"],
            a["burn"],
            a["offsets"],
            a["block"],
            a["n_boot"],
            arm_seed(cfg, data, arm),
        )
        for arm, d in data.items()
    }


def decision(cfg, arms: dict[str, Any], refs: dict[str, float]) -> dict[str, Any]:
    """The pre-registered rule by this path: gate, primary, ratio CI, the secondary family."""
    dr = cfg["decision"]
    b, a2 = arms["baseline"], arms["A2"]
    gate = bool(b["w0"] < 0 and abs(b["w0"]) >= dr["gate_z"] * b["se0"])
    bound = R.vanish_bound(a2["w0"], a2["se0"], b["w0"])
    if bound < dr["vanish_fraction"]:
        primary = "vanishes"
    elif abs(a2["w0"]) >= dr["persist_z"] * a2["se0"] and abs(a2["w0"]) >= dr[
        "persist_fraction"
    ] * abs(b["w0"]):
        primary = "persists"
    else:
        primary = "ambiguous"
    lo = max(0.0, (abs(a2["w0"]) - R.Z975 * a2["se0"]) / abs(b["w0"]))
    tests = {
        "A2s_spec_vs_baseline": ("A2s_spec", "w0", "baseline", None),
        "A1_vs_0": ("A1", "w0", None, 0.0),
        "A3_vs_prediction": ("A3", "w0", None, refs["A3_vs_prediction"]),
        "D1_w0_vs_0": ("D1", "w0", None, 0.0),
        "D1_lag1_vs_prediction": ("D1", "lag1", None, refs["D1_lag1_vs_prediction"]),
    }
    sec, pv = {}, {}
    for name, (arm, stat, ref_arm, ref_v) in tests.items():
        sk = "se0" if stat == "w0" else "lag1_se"
        est, se = arms[arm][stat], arms[arm][sk]
        ref, rse = (arms[ref_arm][stat], arms[ref_arm][sk]) if ref_arm else (ref_v, 0.0)
        diff, sdiff = est - ref, math.hypot(se, rse)
        z = diff / sdiff
        row = {"estimate": est, "reference": ref, "difference": diff, "se": sdiff, "z": z}
        row["p"] = R.p_two_sided(z)
        row["ci95"] = [diff - R.Z975 * sdiff, diff + R.Z975 * sdiff]
        if ref_arm:
            row["ratio"], row["ratio_se"] = R.ratio_delta(est, se, ref, rse)
        sec[name], pv[name] = row, row["p"]
    rej = R.holm_reject(pv, dr["family_alpha"])
    for k in sec:
        sec[k]["reject_holm"] = rej[k]
    desc = {
        x: "vanishes" if R.vanish_bound(arms[x]["w0"], arms[x]["se0"], b["w0"]) < 0.1 else "other"
        for x in ("A1", "A3")
    }
    return {
        "gate": gate,
        "primary": primary,
        "ratio_a2": a2["w0"] / b["w0"],
        "ratio_ci95": [lo, bound],
        "bound": bound,
        "bound_fraction_of_threshold": bound / dr["vanish_fraction"],
        "secondary": sec,
        "descriptive": desc,
    }


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def reproduce(cfg, data, s4c) -> dict[str, Any]:
    a = cfg["analysis"]
    out: dict[str, Any] = {"arms": {}, "checks": []}
    # decoding and the window count against qmem.witness on the same shots
    d = REPO_ROOT / cfg["experiment"]["stage4c_dir"]
    dec_ok, cnt_ok = True, True
    for j in cfg["experiment"]["main_jobs"]:
        with np.load(d / f"shots_{j}.npz") as z:
            lw = int(z["length"])
            for k in z.files:
                if k.endswith("__y"):
                    dec_ok &= bool(np.array_equal(R.unpack(z[k], lw), W.unpack_shots(z[k], lw)))
    for dd in data.values():
        mine = R.window_freqs(dd["shots"], a["k_max"], a["burn"])
        theirs = W.per_shot_windows(dd["shots"], a["k_max"], a["burn"], common=True)
        cnt_ok &= bool(np.array_equal(mine, theirs))
    out["decoding_identical"] = dec_ok
    out["window_counts_identical"] = cnt_ok
    arms = all_arm_stats(cfg, data)
    worst = 0.0
    for arm, st in arms.items():
        ref = s4c["arms"][arm]
        rows = {}
        for k in STAT_KEYS:
            r = rel(st[k], ref[k])
            rows[k] = {"mine": st[k], "stage4c": ref[k], "rel_diff": r}
            worst = max(worst, r)
        for i, (pj, rj) in enumerate(zip(st["per_job"], ref["per_job"], strict=True)):
            for k in ("w0", "p1"):
                r = rel(pj[k], rj[k])
                rows[f"job{i}_{k}"] = {"mine": pj[k], "stage4c": rj[k], "rel_diff": r}
                worst = max(worst, r)
        for i in range(2):
            r = rel(st["ci0"][i], ref["ci0"][i])
            rows[f"ci0_{i}"] = {"mine": st["ci0"][i], "stage4c": ref["ci0"][i], "rel_diff": r}
            worst = max(worst, r)
        out["arms"][arm] = rows
    refs = {
        "A3_vs_prediction": s4c["h_q75"]["A3"]["w0"],
        "D1_lag1_vs_prediction": s4c["h_q75"]["D1"]["lag1"],
    }
    dec = decision(cfg, arms, refs)
    sd = s4c["decision"]
    dchecks = {
        "gate": dec["gate"] == sd["gate"],
        "primary": dec["primary"] == sd["primary"],
        "ratio_a2": rel(dec["ratio_a2"], sd["ratio_a2"]),
        "descriptive": dec["descriptive"]
        == {k: ("vanishes" if v == "vanishes" else "other") for k, v in sd["descriptive"].items()},
    }
    for name, row in dec["secondary"].items():
        ref = sd["secondary"][name]
        for k in ("estimate", "reference", "difference", "se", "z", "p"):
            dchecks[f"{name}.{k}"] = rel(row[k], ref[k])
        dchecks[f"{name}.reject_holm"] = row["reject_holm"] == ref["reject_holm"]
        if "ratio" in row:
            dchecks[f"{name}.ratio"] = rel(row["ratio"], ref["ratio"])
            dchecks[f"{name}.ratio_se"] = rel(row["ratio_se"], ref["ratio_se"])
    num = [v for v in dchecks.values() if not isinstance(v, bool)]
    worst = max([worst, *num])
    out["decision"] = dec
    out["decision_checks"] = dchecks
    out["worst_rel_diff"] = worst
    out["all_booleans_match"] = all(v for v in dchecks.values() if isinstance(v, bool))
    # printed precision (docs/phase7-hardware.md)
    printed = {
        "baseline w0": (f"{arms['baseline']['w0']:.2e}", "-5.15e-03"),
        "baseline se0": (f"{arms['baseline']['se0']:.2e}", "1.27e-04"),
        "baseline z0": (f"{arms['baseline']['z0']:.1f}", "-40.5"),
        "A2 w0": (f"{arms['A2']['w0']:.2e}", "-2.04e-04"),
        "A2 se0": (f"{arms['A2']['se0']:.2e}", "1.41e-04"),
        "ratio": (f"{100 * dec['ratio_a2']:.1f}%", "4.0%"),
        "bound": (f"{100 * dec['bound']:.1f}%", "9.3%"),
        "bound / threshold": (f"{100 * dec['bound_fraction_of_threshold']:.0f}%", "93%"),
        "sham ratio": (f"{dec['secondary']['A2s_spec_vs_baseline']['ratio']:.3f}", "1.044"),
        "sham ratio se": (f"{dec['secondary']['A2s_spec_vs_baseline']['ratio_se']:.3f}", "0.034"),
        "sham z": (f"{dec['secondary']['A2s_spec_vs_baseline']['z']:.2f}", "-1.32"),
    }
    out["printed"] = {
        k: {"mine": v[0], "reported": v[1], "match": v[0] == v[1]} for k, v in printed.items()
    }
    out["arm_stats"] = arms
    return out


# -- 3. fragility ---------------------------------------------------------------------------


def per_of(cfg, d) -> np.ndarray:
    return R.window_freqs(d["shots"], cfg["analysis"]["k_max"], cfg["analysis"]["burn"])


def w0_boot(per: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return R.wit(R.resample_means(per, idx), 0)


def se_block(per, strata, b, n_boot, seed, cap=True) -> tuple[float, np.ndarray]:
    ws = w0_boot(per, R.stratified_blocks(strata, b, n_boot, seed, cap))
    return R.sd(ws), ws


def subset(d: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[keep] for k, v in d.items()}


def row(label, group, reasonable, w_a2, se_a2, w_b, se_b, bound, note="") -> dict[str, Any]:
    return {
        "label": label,
        "group": group,
        "reasonable": reasonable,
        "w_A2": w_a2,
        "se_A2": se_a2,
        "w_base": w_b,
        "se_base": se_b,
        "bound": bound,
        "bound_over_threshold": bound / 0.10,
        "call": R.call(bound),
        "note": note,
    }


def fragility(cfg, data, arms, s4c) -> dict[str, Any]:
    fc, a = cfg["fragility"], cfg["analysis"]
    nb, seed = fc["n_boot"], fc["seed"]
    A2, B = data["A2"], data["baseline"]
    pa, pb = per_of(cfg, A2), per_of(cfg, B)
    sa, sb = arms["A2"], arms["baseline"]
    wa, wb = sa["w0"], sb["w0"]
    rows = []
    bnd = R.vanish_bound(wa, sa["se0"], wb)
    rows.append(
        row(
            "pre-registered (max(shot, block 500 by job))",
            "SE",
            True,
            wa,
            sa["se0"],
            wb,
            sb["se0"],
            bnd,
        )
    )
    rows.append(
        row(
            "shot bootstrap only",
            "SE",
            True,
            wa,
            sa["se_shot0"],
            wb,
            sb["se_shot0"],
            R.vanish_bound(wa, sa["se_shot0"], wb),
        )
    )
    se_delta = D.block_se(pa, 0, 1)
    rows.append(
        row(
            "delta method, i.i.d. shots (qmem.drift.block_se, b = 1)",
            "SE",
            True,
            wa,
            se_delta,
            wb,
            D.block_se(pb, 0, 1),
            R.vanish_bound(wa, se_delta, wb),
        )
    )
    strat = {
        "job": (R.label_strata(A2["jobs"]), R.label_strata(B["jobs"])),
        "none": ([np.arange(len(pa))], [np.arange(len(pb))]),
        "chunk": (
            R.label_strata(A2["jobs"] * 1000 + A2["pubs"]),
            R.label_strata(B["jobs"] * 1000 + B["pubs"]),
        ),
    }
    boots = {}
    for mode in ("job", "none", "chunk"):
        for b in fc["block_lengths"] if mode != "chunk" else [500]:
            sea, _ = se_block(pa, strat[mode][0], b, nb, seed + b)
            seb, _ = se_block(pb, strat[mode][1], b, nb, seed + 7 + b)
            eff = max(1, min(b, min(len(s) for s in strat[mode][0]) // 4))
            se_use = max(sa["se_shot0"], sea)
            rows.append(
                row(
                    f"block {b} (effective {eff}), stratified by {mode}",
                    "SE",
                    True,
                    wa,
                    se_use,
                    wb,
                    max(sb["se_shot0"], seb),
                    R.vanish_bound(wa, se_use, wb),
                    note=f"block SE alone {sea:.3e}",
                )
            )
            boots[(mode, b)] = (sea, seb)
    # ratio methods
    r, se_r = R.ratio_delta_abs(wa, sa["se0"], wb, sb["se0"])
    rows.append(
        row(
            "ratio: delta method with the baseline's SE",
            "ratio",
            True,
            wa,
            sa["se0"],
            wb,
            sb["se0"],
            r + R.Z975 * se_r,
        )
    )
    lo_f, hi_f = R.fieller(wa, sa["se0"], wb, sb["se0"])
    rows.append(
        row(
            "ratio: Fieller 95%",
            "ratio",
            True,
            wa,
            sa["se0"],
            wb,
            sb["se0"],
            hi_f,
            note=f"interval [{lo_f:.4f}, {hi_f:.4f}]",
        )
    )
    # bootstrap of the ratio: each arm resampled by the method that sets its primary SE
    ia = R.stratified_blocks(
        strat["job"][0], 1 if sa["se_shot0"] >= sa["se_block0"] else 500, nb, seed + 100
    )
    ib = R.stratified_blocks(
        strat["job"][1], 1 if sb["se_shot0"] >= sb["se_block0"] else 500, nb, seed + 101
    )
    rb = w0_boot(pa, ia) / w0_boot(pb, ib)
    rows.append(
        row(
            "ratio: percentile bootstrap (97.5%)",
            "ratio",
            True,
            wa,
            sa["se0"],
            wb,
            sb["se0"],
            float(np.quantile(rb, 0.975)),
            note=f"95% [{np.quantile(rb, 0.025):.4f}, {np.quantile(rb, 0.975):.4f}]",
        )
    )
    rows.append(
        row(
            "|ratio|: percentile bootstrap (95%, one-sided)",
            "ratio",
            True,
            wa,
            sa["se0"],
            wb,
            sb["se0"],
            float(np.quantile(np.abs(rb), 0.95)),
        )
    )
    # pooling
    pj = per_job(cfg, data, ("A2", "baseline"), seed + 200)
    pa_ = R.pool([x["w0"] for x in pj["A2"]], [x["se0"] for x in pj["A2"]])
    pb_ = R.pool([x["w0"] for x in pj["baseline"]], [x["se0"] for x in pj["baseline"]])
    rows.append(
        row(
            "fixed effect over jobs",
            "pooling",
            True,
            pa_["fe"],
            pa_["fe_se"],
            pb_["fe"],
            pb_["fe_se"],
            R.vanish_bound(pa_["fe"], pa_["fe_se"], pb_["fe"]),
        )
    )
    rows.append(
        row(
            "DerSimonian-Laird random effect over jobs",
            "pooling",
            True,
            pa_["re"],
            pa_["re_se"],
            pb_["re"],
            pb_["re_se"],
            R.vanish_bound(pa_["re"], pa_["re_se"], pb_["re"]),
            note=f"tau2 A2 {pa_['tau2']:.2e}, baseline {pb_['tau2']:.2e}",
        )
    )
    for drop in (1, 0, 2, 3):
        ka, kb = A2["jobs"] != drop, B["jobs"] != drop
        xa = R.arm_stats(
            A2["shots"][ka],
            A2["jobs"][ka],
            a["k_max"],
            a["burn"],
            [0],
            a["block"],
            a["n_boot"],
            arm_seed(cfg, data, "A2"),
        )
        xb = R.arm_stats(
            B["shots"][kb],
            B["jobs"][kb],
            a["k_max"],
            a["burn"],
            [0],
            a["block"],
            a["n_boot"],
            arm_seed(cfg, data, "baseline"),
        )
        rows.append(
            row(
                f"without job{drop}",
                "pooling",
                drop == 1,
                xa["w0"],
                xa["se0"],
                xb["w0"],
                xb["se0"],
                R.vanish_bound(xa["w0"], xa["se0"], xb["w0"]),
                note="the baseline's P(1) dip" if drop == 1 else "descriptive",
            )
        )
    for x, y in zip(pj["A2"], pj["baseline"], strict=True):
        rows.append(
            row(
                f"job{x['job']} alone",
                "pooling",
                False,
                x["w0"],
                x["se0"],
                y["w0"],
                y["se0"],
                R.vanish_bound(x["w0"], x["se0"], y["w0"]),
                note="descriptive: a quarter of the shots",
            )
        )
    # q75 models
    for label, sens in s4c["sensitivity"].items():
        same = math.isclose(
            sens["decision"]["ratio_a2"], s4c["decision"]["ratio_a2"], rel_tol=1e-12
        )
        rows.append(
            row(
                f"q75 {label}: observed baseline",
                "q75",
                True,
                wa,
                sa["se0"],
                wb,
                sb["se0"],
                bnd,
                note=f"decision data-only; ratio identical: {same}",
            )
        )
        pred = sens["partial_coherence"]["curve"][0]["w0_base"]
        rows.append(
            row(
                f"q75 {label}: H_q75 baseline prediction as denominator",
                "q75",
                True,
                wa,
                sa["se0"],
                pred,
                0.0,
                R.vanish_bound(wa, sa["se0"], pred),
                note=f"decay {sens['decay']:.4g}",
            )
        )
    reasonable = [x for x in rows if x["reasonable"]]
    worst = max(reasonable, key=lambda x: x["bound"])
    return {
        "rows": rows,
        "all_reasonable_vanish": all(x["call"] == "vanishes" for x in reasonable),
        "worst_reasonable": worst,
        "per_job": pj,
        "pool_A2": pa_,
        "pool_baseline": pb_,
        "ratio_boot_sd": R.sd(rb),
    }


def per_job(cfg, data, arms_: tuple[str, ...], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Per job: w(0) with the primary SE computed inside the job (max(shot, block 500))."""
    a = cfg["analysis"]
    out = {}
    for i, arm in enumerate(arms_):
        d = data[arm]
        rows = []
        for job in np.unique(d["jobs"]):
            k = d["jobs"] == job
            st = R.arm_stats(
                d["shots"][k],
                d["jobs"][k],
                a["k_max"],
                a["burn"],
                [0],
                a["block"],
                a["n_boot"],
                seed + 31 * i + int(job),
            )
            rows.append(
                {
                    "job": int(job),
                    "n": int(k.sum()),
                    "w0": st["w0"],
                    "se0": st["se0"],
                    "se_shot0": st["se_shot0"],
                    "se_block0": st["se_block0"],
                    "p1": st["p1"],
                    "lag1": st["lag1"],
                }
            )
        out[arm] = rows
    return out


# -- 3b. power and shots ------------------------------------------------------------------


def power(cfg, arms) -> dict[str, Any]:
    pc, dr = cfg["power"], cfg["decision"]
    b, a2 = arms["baseline"], arms["A2"]
    wb, seb, sea, n0 = abs(b["w0"]), b["se0"], a2["se0"], a2["n"]
    f = dr["vanish_fraction"]
    obs_ratio = abs(a2["w0"]) / wb
    rep = []
    for r in [*pc["ratios"], obs_ratio]:
        rep.append(
            {
                "ratio": r,
                "p_vanish_analytic": R.vanish_power(r * wb, sea, f * wb),
                "p_vanish_mc": R.vanish_power_mc(
                    r, b["w0"], seb, sea, f, dr["gate_z"], pc["n_mc"], pc["seed"]
                ),
            }
        )
    shots = []
    for frac in (1.0, pc["target_fraction"]):
        thr = frac * f * wb
        need = R.shots_for(lambda s, thr=thr: abs(a2["w0"]) + R.Z975 * s <= thr, n0, sea)
        shots.append(
            {
                "bound_at": frac,
                "criterion": "observed |w_A2| held fixed",
                "ratio": None,
                "probability": None,
                "shots_A2": need,
            }
        )
        for r in pc["ratios"]:
            exp = R.shots_for(
                lambda s, r=r, thr=thr: _expected_abs(r * wb, s) + R.Z975 * s <= thr, n0, sea
            )
            shots.append(
                {
                    "bound_at": frac,
                    "criterion": "expected bound",
                    "ratio": r,
                    "probability": None,
                    "shots_A2": exp,
                }
            )
            for p in pc["target_probability"]:
                need = R.shots_for(
                    lambda s, r=r, p=p, thr=thr: R.vanish_power(r * wb, s, thr) >= p, n0, sea
                )
                shots.append(
                    {
                        "bound_at": frac,
                        "criterion": "P(bound below) >= probability",
                        "ratio": r,
                        "probability": p,
                        "shots_A2": need,
                    }
                )
    return {
        "se_A2_per_10k": sea * math.sqrt(n0 / 1e4),
        "n0": n0,
        "replicate": rep,
        "shots": shots,
        "observed_ratio": obs_ratio,
    }


def _expected_abs(mu: float, s: float) -> float:
    """E|X| for X ~ N(mu, s^2) (folded normal mean)."""
    return s * math.sqrt(2 / math.pi) * math.exp(-(mu**2) / (2 * s * s)) + mu * (
        1 - 2 * R.norm_cdf(-mu / s)
    )


# -- 4. drift ---------------------------------------------------------------------------------


def drift(cfg, data, arms, pj) -> dict[str, Any]:
    dc, a = cfg["drift"], cfg["analysis"]
    out: dict[str, Any] = {"chunks": [], "tests": {}, "balance": {}, "se": {}, "bias": {}}
    pers = {arm: per_of(cfg, data[arm]) for arm in data if arm != "A3"}
    for arm, per in pers.items():
        d = data[arm]
        p_shot = d["shots"][:, a["burn"] :].mean(1)
        jobs_s = R.label_strata(d["jobs"])
        chunks = R.label_strata(d["jobs"] * 1000 + d["pubs"])
        for c in chunks:
            out["chunks"].append(
                {
                    "arm": arm,
                    "job": int(d["jobs"][c[0]]),
                    "pub": int(d["pubs"][c[0]]),
                    "n": len(c),
                    "p1": float(p_shot[c].mean()),
                    "p1_se": float(p_shot[c].std(ddof=1) / math.sqrt(len(c))),
                    "w0": R.wit(per[c].mean(0), 0),
                    "w0_se": D.block_se(per[c], 0, 1),
                }
            )
        cp = D.changepoint_test(p_shot, dc["cp_grid"], dc["n_perm"], dc["seed"])
        cpw = D.changepoint_test(D.influence(per, 0), dc["cp_grid"], dc["n_perm"], dc["seed"] + 1)
        hom = D.homogeneity_chi2(per, jobs_s)
        cj = D.block_witness_constancy(per, jobs_s, 0)
        cc = D.block_witness_constancy(per, chunks, 0)
        p1j = R.pool(
            [p_shot[s].mean() for s in jobs_s],
            [p_shot[s].std(ddof=1) / math.sqrt(len(s)) for s in jobs_s],
        )
        p1c = R.pool(
            [p_shot[s].mean() for s in chunks],
            [p_shot[s].std(ddof=1) / math.sqrt(len(s)) for s in chunks],
        )
        out["tests"][arm] = {
            "changepoint_p1": {k: cp[k] for k in ("stat", "start", "stop", "p")},
            "changepoint_w0_influence": {k: cpw[k] for k in ("stat", "start", "stop", "p")},
            "homogeneity_g_across_jobs_p": hom["p_joint"],
            "w0_constancy_across_jobs_p": cj["p"],
            "w0_constancy_across_chunks_p": cc["p"],
            "p1_q_across_jobs": {"q": p1j["q"], "p": p1j["q_p"], "i2": p1j["i2"]},
            "p1_q_across_chunks": {"q": p1c["q"], "p": p1c["q_p"], "i2": p1c["i2"]},
            "p1_chunk_pool_within_job": _within_job_q(p_shot, d["jobs"], d["pubs"]),
        }
    # balance of the interleaved schedule
    for arm in ("baseline", "A2", "A2s_spec", "D1", "A1"):
        d = data[arm]
        rows = []
        for job in np.unique(d["jobs"]):
            k = d["jobs"] == job
            pubs = sorted(set(d["pubs"][k].tolist()))
            rows.append(
                {
                    "job": int(job),
                    "shots": int(k.sum()),
                    "frac": float(k.mean()),
                    "pubs": pubs,
                    "mean_pub": float(np.mean(d["pubs"][k])),
                }
            )
        out["balance"][arm] = rows
    # the job1 dip: P(1) differences against the arms run in the same jobs
    dd = []
    for job in range(4):
        rowj = {"job": job}
        pb = [c for c in out["chunks"] if c["arm"] == "baseline" and c["job"] == job]
        for other in ("A2s_spec", "D1"):
            po = [c for c in out["chunks"] if c["arm"] == other and c["job"] == job]
            mb = np.mean([c["p1"] for c in pb])
            mo = np.mean([c["p1"] for c in po])
            se = math.sqrt(
                np.mean([c["p1_se"] ** 2 for c in pb]) / len(pb)
                + np.mean([c["p1_se"] ** 2 for c in po]) / len(po)
            )
            rowj[f"p1_base_minus_{other}"] = float(mb - mo)
            rowj[f"p1_base_minus_{other}_se"] = se
        dd.append(rowj)
    out["dip"] = dd
    for other in ("A2s_spec", "D1"):
        q = R.pool(
            [r[f"p1_base_minus_{other}"] for r in dd], [r[f"p1_base_minus_{other}_se"] for r in dd]
        )
        out[f"dip_q_{other}"] = q
    # drift-conditional SEs of w(0)
    fc = cfg["fragility"]
    for arm in ("baseline", "A2", "A2s_spec"):
        d, per, st = data[arm], pers[arm], arms[arm]
        chunks = R.label_strata(d["jobs"] * 1000 + d["pubs"])
        se_chunk, _ = se_block(per, chunks, 500, fc["n_boot"], dc["seed"] + 10)
        se_chunk_shot, _ = se_block(per, chunks, 1, fc["n_boot"], dc["seed"] + 11)
        se_un, _ = se_block(per, [np.arange(len(per))], 2000, fc["n_boot"], dc["seed"] + 12)
        pjv = pj.get(arm)
        pc = [c for c in out["chunks"] if c["arm"] == arm]
        prc = R.pool([c["w0"] for c in pc], [c["w0_se"] for c in pc])
        row = {
            "prereg": st["se0"],
            "shot": st["se_shot0"],
            "block500_by_job": st["se_block0"],
            "stratified_by_chunk_block500": max(se_chunk, se_chunk_shot),
            "unstratified_block2000": se_un,
            "re_over_chunks": prc["re_se"],
            "tau_chunks": math.sqrt(prc["tau2"]),
            "q_chunks_p": prc["q_p"],
        }
        if pjv:
            prj = R.pool([x["w0"] for x in pjv], [x["se0"] for x in pjv])
            row |= {
                "re_over_jobs": prj["re_se"],
                "tau_jobs": math.sqrt(prj["tau2"]),
                "q_jobs_p": prj["q_p"],
                "fe_jobs": prj["fe"],
                "re_jobs": prj["re"],
            }
        out["se"][arm] = row
    # bias: how far must the baseline's |w| move for the bound to reach 10%?
    wa, sea, wb = arms["A2"]["w0"], arms["A2"]["se0"], arms["baseline"]["w0"]
    flip = (abs(wa) + R.Z975 * sea) / cfg["decision"]["vanish_fraction"]
    jb = pj["baseline"]
    no1 = [x for x in jb if x["job"] != 1]
    mix = wb - R.pool([x["w0"] for x in jb], [x["se0"] for x in jb])["fe"]
    cases = {
        "observed": abs(wb),
        "pooling (mixture) effect removed: FE over jobs": abs(
            R.pool([x["w0"] for x in jb], [x["se0"] for x in jb])["fe"]
        ),
        "job1 replaced by the mean of the other jobs": abs(float(np.mean([x["w0"] for x in no1]))),
        "the weakest job's baseline": min(abs(x["w0"]) for x in jb),
        "observed less 1 between-job SD": abs(wb) - float(np.std([x["w0"] for x in jb], ddof=1)),
        "observed less 2 between-job SD": abs(wb)
        - 2 * float(np.std([x["w0"] for x in jb], ddof=1)),
        "observed less 2 primary SE": abs(wb) - 2 * arms["baseline"]["se0"],
    }
    out["bias"] = {
        "flip_baseline_abs_w0": flip,
        "flip_relative_change": flip / abs(wb) - 1,
        "flip_in_baseline_se": (abs(wb) - flip) / arms["baseline"]["se0"],
        "mixture_effect": mix,
        "cases": {
            k: {
                "abs_w_base": v,
                "bound": (abs(wa) + R.Z975 * sea) / v,
                "call": R.call((abs(wa) + R.Z975 * sea) / v),
            }
            for k, v in cases.items()
        },
    }
    return out


def _within_job_q(p_shot, jobs, pubs) -> dict[str, float]:
    """Cochran's Q of chunk P(1) within each job, summed over jobs (between-job removed)."""
    q, df = 0.0, 0
    for job in np.unique(jobs):
        k = jobs == job
        ch = R.label_strata(pubs[k])
        x = p_shot[k]
        r = R.pool([x[c].mean() for c in ch], [x[c].std(ddof=1) / math.sqrt(len(c)) for c in ch])
        q += r["q"]
        df += r["df"]
    return {"q": q, "df": df, "p": R.chi2_sf(q, df)}


# -- 5. baseline against the sham, per job ------------------------------------------------


def sham_jobs(cfg, data, pj) -> dict[str, Any]:
    a, dc = cfg["analysis"], cfg["drift"]
    rows = []
    stats: dict[str, dict[str, list]] = {}
    for job in range(4):
        r: dict[str, Any] = {"job": job}
        per_arm = {}
        for i, arm in enumerate(("baseline", "A2s_spec")):
            d = data[arm]
            k = d["jobs"] == job
            y = d["shots"][k]
            ps = y[:, a["burn"] :].mean(1)
            x = {"p1": (float(ps.mean()), float(ps.std(ddof=1) / math.sqrt(len(ps))))}
            st = next(q for q in pj[arm] if q["job"] == job)
            x["w0"] = (st["w0"], st["se0"])
            idx = R.label_strata(np.zeros(len(y)))
            ib = R.stratified_blocks(idx, 1, dc["n_boot"], dc["seed"] + 100 * job + i)
            for lag in dc["lags"]:
                ls = R.lag_stats(y, a["burn"], lag)
                est = float(R.corr_from(ls.sum(0))[0])
                x[f"acf{lag}"] = (est, R.sd(R.corr_from(R.resample_means(ls, ib))))
            per_arm[arm] = x
        for stat in per_arm["baseline"]:
            (eb, sb), (es, ss) = per_arm["baseline"][stat], per_arm["A2s_spec"][stat]
            dlt, sdl = es - eb, math.hypot(sb, ss)
            r[f"{stat}_base"], r[f"{stat}_sham"] = eb, es
            r[f"{stat}_diff"], r[f"{stat}_diff_se"], r[f"{stat}_z"] = dlt, sdl, dlt / sdl
            s = stats.setdefault(stat, {"d": [], "s": [], "b": [], "bs": [], "h": [], "hs": []})
            s["d"].append(dlt)
            s["s"].append(sdl)
            s["b"].append(eb)
            s["bs"].append(sb)
            s["h"].append(es)
            s["hs"].append(ss)
        rows.append(r)
    het = {}
    for stat, s in stats.items():
        pd_, pb_, ph_ = R.pool(s["d"], s["s"]), R.pool(s["b"], s["bs"]), R.pool(s["h"], s["hs"])
        het[stat] = {
            "diff_fe": pd_["fe"],
            "diff_fe_se": pd_["fe_se"],
            "diff_q": pd_["q"],
            "diff_q_p": pd_["q_p"],
            "diff_i2": pd_["i2"],
            "baseline_q_p": pb_["q_p"],
            "sham_q_p": ph_["q_p"],
        }
    return {"rows": rows, "heterogeneity": het}


def sham_pooled(arms) -> dict[str, Any]:
    """Sham minus baseline, pooled, at every witness offset, in P(1) and the lag-1 excess
    (primary SEs; not part of #27's family)."""
    b, s = arms["baseline"], arms["A2s_spec"]
    out = {}
    for j in (0, 1, 2, 3):
        d, se = s[f"w{j}"] - b[f"w{j}"], math.hypot(s[f"se{j}"], b[f"se{j}"])
        out[f"w{j}"] = {"diff": d, "se": se, "z": d / se, "ratio": s[f"w{j}"] / b[f"w{j}"]}
    d, se = s["lag1"] - b["lag1"], math.hypot(s["lag1_se"], b["lag1_se"])
    out["lag1"] = {"diff": d, "se": se, "z": d / se}
    return out


# -- 6. TOST ----------------------------------------------------------------------------------


def tost(cfg, data, arms, s4c, drift_out) -> dict[str, Any]:
    tc = cfg["tost"]
    b, s = arms["baseline"], arms["A2s_spec"]
    r, se_r = R.ratio_delta(s["w0"], s["se0"], b["w0"], b["se0"])
    zt = R.norm_ppf(1 - tc["alpha"])
    lo_f, hi_f = R.fieller(s["w0"], s["se0"], b["w0"], b["se0"], z=zt)
    pb, ps = per_of(cfg, data["baseline"]), per_of(cfg, data["A2s_spec"])
    sb_ = R.label_strata(data["baseline"]["jobs"])
    ss_ = R.label_strata(data["A2s_spec"]["jobs"])
    ib = R.stratified_blocks(
        sb_, 1 if b["se_shot0"] >= b["se_block0"] else 500, tc["n_boot"], tc["seed"]
    )
    is_ = R.stratified_blocks(
        ss_, 1 if s["se_shot0"] >= s["se_block0"] else 500, tc["n_boot"], tc["seed"] + 1
    )
    rb = w0_boot(ps, is_) / w0_boot(pb, ib)
    blo, bhi = float(np.quantile(rb, tc["alpha"])), float(np.quantile(rb, 1 - tc["alpha"]))
    rows = []
    for m in tc["margins"]:
        t = R.tost(r, se_r, 1 - m, 1 + m, tc["alpha"])
        rows.append(
            {
                "margin": m,
                "p_lower": t["p_lower"],
                "p_upper": t["p_upper"],
                "p": t["p"],
                "equivalent_delta": t["equivalent"],
                "equivalent_fieller": bool(1 - m < lo_f and hi_f < 1 + m),
                "equivalent_bootstrap": bool(1 - m < blo and bhi < 1 + m),
            }
        )
    smallest = {
        "delta": max(r + zt * se_r - 1, 1 - (r - zt * se_r)),
        "fieller": max(hi_f - 1, 1 - lo_f),
        "bootstrap": max(bhi - 1, 1 - blo),
    }
    true = s4c["h_q75"]["A2s_spec"]["w0"] / s4c["h_q75"]["baseline"]["w0"]
    plan = []
    n0 = b["n"]
    for m in tc["margins"]:
        for tr in [tc["true_ratio"], *tc["true_ratio_sensitivity"]]:
            for pw in tc["power"]:
                need = R.shots_for(
                    lambda sg, m=m, tr=tr, pw=pw: (
                        R.tost_power(tr, sg, 1 - m, 1 + m, tc["alpha"]) >= pw
                    ),
                    n0,
                    se_r,
                )
                plan.append(
                    {
                        "margin": m,
                        "true_ratio": tr,
                        "power": pw,
                        "shots_per_arm": need,
                        "recommended": m == tc["recommended_margin"],
                        "se": "primary",
                    }
                )
    cur = {
        str(m): R.tost_power(tc["true_ratio"], se_r, 1 - m, 1 + m, tc["alpha"])
        for m in tc["margins"]
    }
    # drift-conditional: each arm's SE from a random effect over its pub chunks
    se_b_dc = max(b["se0"], drift_out["se"]["baseline"]["re_over_chunks"])
    se_s_dc = max(s["se0"], drift_out["se"]["A2s_spec"]["re_over_chunks"])
    _, se_dc = R.ratio_delta(s["w0"], se_s_dc, b["w0"], se_b_dc)
    dc_rows = []
    for m in tc["margins"]:
        t = R.tost(r, se_dc, 1 - m, 1 + m, tc["alpha"])
        dc_rows.append({"margin": m, "p": t["p"], "equivalent_delta": t["equivalent"]})
    for tr in [tc["true_ratio"], *tc["true_ratio_sensitivity"]]:
        for pw in tc["power"]:
            m = tc["recommended_margin"]
            need = R.shots_for(
                lambda sg, m=m, tr=tr, pw=pw: R.tost_power(tr, sg, 1 - m, 1 + m, tc["alpha"]) >= pw,
                n0,
                se_dc,
            )
            plan.append(
                {
                    "margin": m,
                    "true_ratio": tr,
                    "power": pw,
                    "shots_per_arm": need,
                    "recommended": True,
                    "se": "drift-conditional (RE over chunks)",
                }
            )
    return {
        "ratio": r,
        "ratio_se": se_r,
        "ci90_delta": [r - zt * se_r, r + zt * se_r],
        "ci90_fieller": [lo_f, hi_f],
        "ci90_bootstrap": [blo, bhi],
        "rows": rows,
        "smallest_margin_established": smallest,
        "drift_conditional": {
            "se_base": se_b_dc,
            "se_sham": se_s_dc,
            "ratio_se": se_dc,
            "ci90": [r - zt * se_dc, r + zt * se_dc],
            "rows": dc_rows,
        },
        "h_q75_ratio": true,
        "plan": plan,
        "power_at_15000_true_1": cur,
        "recommended": {"margin": tc["recommended_margin"], "alpha": tc["recommended_alpha"]},
    }


# -- 7. the baseline's shortfall ---------------------------------------------------------------


def shortfall(cfg, data, arms, s4c, drift_out) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("baseline", "A2s_spec"):
        pred = s4c["h_q75"][arm]
        w, se = arms[arm]["w0"], arms[arm]["se0"]
        f = w / pred["w0"]
        out[arm] = {
            "w0": w,
            "w0_pred": pred["w0"],
            "fraction": f,
            "fraction_ci95": [f - R.Z975 * se / abs(pred["w0"]), f + R.Z975 * se / abs(pred["w0"])],
            "p1": arms[arm]["p1"],
            "p1_pred": pred["p1"],
            "p1_offset": arms[arm]["p1"] - pred["p1"],
            "lag1": arms[arm]["lag1"],
            "lag1_pred": pred["lag1"],
            "g_over_pred": [x / y for x, y in zip(arms[arm]["g"], pred["g"], strict=True)],
        }
    out["baseline_pred_by_model"] = {
        k: v["partial_coherence"]["curve"][0]["w0_base"] for k, v in s4c["sensitivity"].items()
    }
    ch = [c for c in drift_out["chunks"] if c["arm"] in ("baseline", "A2s_spec")]
    for arm in ("baseline", "A2s_spec", None):
        sel = [c for c in ch if arm is None or c["arm"] == arm]
        x = np.array([c["p1"] for c in sel])
        y = np.array([c["w0"] for c in sel])
        out[f"corr_p1_w0_chunks_{arm or 'both'}"] = float(np.corrcoef(x, y)[0, 1])
    gate = json.loads(
        (REPO_ROOT / cfg["experiment"]["stage4c_dir"] / "gate_pilot.json").read_text()
    )
    out["pilot_gate"] = gate
    out["A2_p_m1"] = s4c.get("A2_p_m1")
    return out


# -- main ---------------------------------------------------------------------------------------


def verdict(prov_ok, rep, frag) -> str:
    if not prov_ok or not rep["all_booleans_match"] or rep["worst_rel_diff"] > 1e-9:
        return "fail"
    return "pass" if frag["all_reasonable_vanish"] else "ambiguous"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--freeze", action="store_true", help="write provenance.json (once)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    sd.mkdir(parents=True, exist_ok=True)
    out = new_run_dir(cfg["experiment"]["name"])
    pf = sd / "provenance.json"
    if args.freeze:
        prov = provenance(cfg)
        if pf.exists():
            old = json.loads(pf.read_text())
            if {k: v["sha256"] for k, v in old["files"].items()} != {
                k: v["sha256"] for k, v in prov["files"].items()
            }:
                sys.exit(f"refusing to overwrite {pf}: the frozen hashes differ")
        save_json(pf, prov)
        log(out, f"froze {len(prov['files'])} files in {pf}")
    if not pf.exists():
        sys.exit(f"{pf} missing: run once with --freeze")
    pchk = check_provenance(cfg, json.loads(pf.read_text()))
    log(out, f"provenance: {pchk['n_files']} files, ok = {pchk['ok']}")
    s4c = json.loads((REPO_ROOT / cfg["experiment"]["stage4c_dir"] / "summary.json").read_text())
    data = load_main(cfg)
    rep = reproduce(cfg, data, s4c)
    log(
        out,
        f"reproduction: worst relative difference {rep['worst_rel_diff']:.2e}; "
        f"booleans match {rep['all_booleans_match']}",
    )
    arms = rep.pop("arm_stats")
    frag = fragility(cfg, data, arms, s4c)
    log(
        out,
        f"fragility: all reasonable vanish = {frag['all_reasonable_vanish']}; worst "
        f"{frag['worst_reasonable']['label']} at {frag['worst_reasonable']['bound']:.4f}",
    )
    pw = power(cfg, arms)
    dr = drift(cfg, data, arms, frag["per_job"])
    log(out, "drift done")
    sh = sham_jobs(
        cfg, data, per_job(cfg, data, ("baseline", "A2s_spec"), cfg["fragility"]["seed"] + 300)
    )
    ts = tost(cfg, data, arms, s4c, dr)
    sh["pooled"] = sham_pooled(arms)
    sf = shortfall(cfg, data, arms, s4c, dr)
    v = verdict(pchk["ok"], rep, frag)
    log(out, f"verdict: {v}")
    summary = {
        "meta": run_metadata(cfg, cfg["analysis"]["seed"]),
        "run_dir": str(out.relative_to(REPO_ROOT)),
        "verdict": v,
        "provenance_check": pchk,
        "reproduction": rep,
        "fragility": {k: val for k, val in frag.items()},
        "power": pw,
        "drift": dr,
        "sham_jobs": sh,
        "tost": ts,
        "shortfall": sf,
    }
    save_json(out / "summary.json", summary)
    save_json(sd / "summary.json", summary)
    write_csv(sd / "fragility.csv", frag["rows"])
    write_csv(sd / "drift_chunks.csv", dr["chunks"])
    write_csv(sd / "sham_jobs.csv", sh["rows"])
    write_csv(sd / "tost.csv", ts["rows"])
    write_csv(sd / "tost_plan.csv", ts["plan"])
    write_csv(sd / "power_shots.csv", pw["shots"])
    for name in ("fragility.csv", "drift_chunks.csv", "sham_jobs.csv", "tost.csv"):
        (out / name).write_bytes((sd / name).read_bytes())
    log(out, f"wrote {out} and {sd}")


if __name__ == "__main__":
    main()
