"""Phase 7 Stage 1 (#19): drift and nonstationary-mixture nulls for the Phase 6 witness.

Existing data and simulation only (no QPU, no IBM service calls). Reads the Phase 6 shots and
job records and the #18 instruments; writes raw outputs to ``results/phase7_drift/<ts>/`` and
curated ones to ``results/summaries/phase7_drift/``.

Stages (issue items): 1 shot order and an order-verification design; 2 segment statistics and
stationarity tests; 3 the episode's selection effect; 4 within- versus between-shot; 5 drift
nulls (fitted regimes, mean shifts, adversarial searches, simulation at 20,000 x 64);
6 physical reading of the episode; 7 uncertainty policy; then the gate.

    uv run python experiments/phase7_drift.py --config experiments/configs/phase7_drift.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from qmem import drift as D
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# -- io -------------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], digits: int | None = None) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        for r in rows:
            if digits:
                r = {
                    k: float(f"{v:.{digits}g}") if isinstance(v, float) else v for k, v in r.items()
                }
            wr.writerow(r)


def p6(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


def calibrated(cfg) -> W.LinearProcess:
    z = np.load(REPO_ROOT / cfg["experiment"]["audit_dir"] / "instruments.npz")
    return W.LinearProcess.instrument(z["calibrated"], z["rho0"])


def calibrated_superops(cfg) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(REPO_ROOT / cfg["experiment"]["audit_dir"] / "instruments.npz")
    return np.asarray(z["calibrated"]), np.asarray(z["rho0"])


class Data:
    """The main run, its per-shot windows (common start set) and predeclared segments."""

    def __init__(self, cfg: dict[str, Any]):
        w = cfg["witness"]
        self.arr = W.load_shots(str(p6(cfg) / "shots_main.npz"))
        self.n, self.length = self.arr.shape
        self.per = W.per_shot_windows(self.arr, w["k_max"], w["burn"], common=True)
        a, b = cfg["segments"]["episode"]
        self.episode = np.zeros(self.n, dtype=bool)
        self.episode[a:b] = True
        self.masks: dict[str, np.ndarray] = {
            "all": np.ones(self.n, bool),
            "episode": self.episode,
            "rest": ~self.episode,
        }
        sg = cfg["segments"]
        if "refined_episode" in sg:
            a, b = sg["refined_episode"]
            m = np.zeros(self.n, bool)
            m[a:b] = True
            self.masks["episode_refined"] = m
            a, b = sg["refined_exclusion"]
            m = np.ones(self.n, bool)
            m[a:b] = False
            self.masks["rest_refined"] = m
        for label, parts in (("half", 2), ("quarter", 4), ("decile", 10)):
            for i, idx in enumerate(np.array_split(np.arange(self.n), parts)):
                m = np.zeros(self.n, bool)
                m[idx] = True
                self.masks[f"{label}{i}"] = m


def primary_se(per: np.ndarray, j: int, b: int, n_boot: int, seed: int) -> dict[str, float]:
    """The predeclared primary SE: max(shot bootstrap, circular block bootstrap with b)."""
    s_shot = D.bootstrap_se(per, j, W.bootstrap_indices(len(per), n_boot, seed))
    bb = min(b, len(per))
    s_blk = D.bootstrap_se(per, j, D.block_bootstrap_indices(len(per), bb, n_boot, seed + 1))
    return {"shot": s_shot, "block": s_blk, "primary": max(s_shot, s_blk)}


def stratified_se(per: np.ndarray, masks: list[np.ndarray], j: int, b: int, n_boot: int, seed: int):
    idx = D.stratified_block_indices([np.nonzero(m)[0] for m in masks], b, n_boot, seed)
    s_blk = D.bootstrap_se(per, j, idx)
    idx1 = D.stratified_block_indices([np.nonzero(m)[0] for m in masks], 1, n_boot, seed + 1)
    return max(s_blk, D.bootstrap_se(per, j, idx1))


# -- 1. shot order --------------------------------------------------------------------------------


def order_stage(cfg) -> dict[str, Any]:
    jobs = {s: json.loads((p6(cfg) / f"job_{s}.json").read_text()) for s in ("main", "pilot")}
    res: dict[str, Any] = {"jobs": {}}
    for s, j in jobs.items():
        ns = j["metrics"]["circuits_execution_time_ns"]
        res["jobs"][s] = {
            "shots": j["shots"],
            "execution_spans": j["result_metadata"]["execution"]["execution_spans"],
            "circuits_execution_time_s": ns * 1e-9,
            "shot_period_s": ns * 1e-9 / j["shots"],
            "qpu_charge_s": j["metrics"]["usage"]["qpu_charge_time_seconds"],
        }
    period = res["jobs"]["main"]["shot_period_s"]
    a, b = cfg["segments"]["episode"]
    res["episode_duration_s_if_time_order"] = (b - a) * period
    res["episode_start_s_if_time_order"] = a * period
    res["runtime_source_findings"] = [
        "qiskit_ibm_runtime 0.51.0 json.py decodes each pub's BitArray from the server payload "
        "as is (RuntimeDecoder: BitArray(**obj_val)); the client never reorders shots.",
        "ExecutionSpans docstring: each span is a time window of data collection with a "
        "reference to which data were collected in it; it guarantees only that the referenced "
        "data occurred within the window, not that the window is tight.",
        "DoubleSliceSpan maps each pub to (shape, parameter slice, shot slice): spans resolve "
        "time only to the granularity of the spans the server reports; nothing in the client "
        "or the result format time-stamps individual shots or states their order.",
        "The main job reports one span covering all 20,000 shots (12 s window), so its data "
        "cannot verify that returned order is execution order: all segments are labelled "
        "'returned order'.",
    ]
    c = cfg["cost"]
    per_shot = c["rep_delay_s"] + c["circuit_s"]
    designs = []
    for name, jobs_n, shots, note in [
        ("one job (Phase 6 as run)", 1, 20000, "no time resolution inside the 12 s span"),
        (
            "one job, 10 pubs x 2,000 shots",
            1,
            20000,
            "free; verifies order only if the server reports one span per pub",
        ),
        (
            "4 jobs x 5,000 shots",
            4,
            20000,
            "guaranteed time labels at ~3 s resolution (job spans); +6 s over one job",
        ),
        (
            "standalone check: 4 jobs x 1,000 shots",
            4,
            4000,
            "only to test order; detects a 1,000-shot episode if it recurs",
        ),
    ]:
        designs.append(
            {
                "design": name,
                "jobs": jobs_n,
                "shots": shots,
                "qpu_s": jobs_n * c["job_overhead_s"] + shots * per_shot,
                "note": note,
            }
        )
    res["order_verification_designs"] = designs
    return res


# -- 2. segment statistics and stationarity -------------------------------------------------------


def block_stats(arr: np.ndarray, burn: int, size: int) -> np.ndarray:
    starts = list(range(burn, arr.shape[1] - size + 1, size))
    weights = 1 << np.arange(size - 1, -1, -1)
    f = np.zeros((len(arr), 2**size))
    for t in starts:
        np.add.at(f, (np.arange(len(arr)), arr[:, t : t + size].astype(np.int64) @ weights), 1)
    return f / len(starts)


def segment_stage(cfg, data: Data, cal: W.LinearProcess, out: Path) -> dict[str, Any]:
    w, sg, u = cfg["witness"], cfg["segments"], cfg["uncertainty"]
    rows, margs = [], []
    for name, m in data.masks.items():
        per, arr = data.per[m], data.arr[m]
        st = D.segment_stats(name, per, arr, w["offsets"], u["primary_b"], sg["n_boot"], sg["seed"])
        row: dict[str, Any] = {
            "segment": name,
            "first": int(np.nonzero(m)[0][0]),
            "last": int(np.nonzero(m)[0][-1]),
            "shots": st.n,
            "p1": st.p1,
        }
        row |= {f"g{k}": float(st.g[k]) for k in range(len(st.g))}
        for j in w["offsets"]:
            if name == "all":
                prim = stratified_se(
                    data.per,
                    [data.episode, ~data.episode],
                    j,
                    u["primary_b"],
                    sg["n_boot"],
                    sg["seed"] + 2,
                )
            else:
                prim = max(st.se_shot[j], st.se_block_boot[j])
            row |= {
                f"w{j}": st.w[j],
                f"se_shot_j{j}": st.se_shot[j],
                f"se_block_delta_j{j}": st.se_block[j],
                f"se_block_boot_j{j}": st.se_block_boot[j],
                f"se_primary_j{j}": prim,
                f"z_primary_j{j}": st.w[j] / prim,
                f"ess_j{j}": st.ess[j],
            }
        for size in sg["block_sizes"]:
            row[f"block_kl_L{size}"] = W.block_kl(arr, cal, size, w["burn"])
        rows.append(row)
        mg = arr.mean(0)
        margs.append({"segment": name, **{f"t{t}": float(mg[t]) for t in range(data.length)}})
    # rolling windows
    roll = []
    win, step = sg["rolling_window"], sg["rolling_step"]
    for s0 in range(0, data.n - win + 1, step):
        per = data.per[s0 : s0 + win]
        g = per.mean(0)
        r = {"start": s0, "centre": s0 + win / 2, "p1": float(data.arr[s0 : s0 + win].mean())}
        r |= {f"g{k}": float(g[k]) for k in range(len(g))}
        for j in w["offsets"]:
            r[f"w{j}"] = W.witness(g, j)
            r[f"se_j{j}"] = D.block_se(per, j, 1)
        roll.append(r)
    # per-1000-block witness over the whole run (blocks never straddle the episode edges)
    st_cfg = cfg["stationarity"]
    blocks_w = []
    for lab, m in (("rest", ~data.episode), ("episode", data.episode)):
        for blk in D.blocks_within(m, st_cfg["w_block"]):
            per = data.per[blk]
            r = {"segment": lab, "start": int(blk[0]), "stop": int(blk[-1] + 1)}
            for j in w["offsets"]:
                r[f"w{j}"] = W.witness(per.mean(0), j)
                r[f"se_j{j}"] = D.block_se(per, j, 1)
            blocks_w.append(r)
    blocks_w.sort(key=lambda r: r["start"])
    # pre-specified stationarity tests within the episode-free run and within the episode
    tests = {}
    for lab in cfg["segments"].get("test_segments", ["rest", "episode"]):
        m = data.masks[lab]
        per = data.per[m]
        # blocks in local coordinates that do not straddle the gap the episode leaves
        gap_mask = np.zeros(data.n, bool)
        gap_mask[m] = True
        pos = np.cumsum(m) - 1
        hb = [pos[b] for b in D.blocks_within(gap_mask, st_cfg["chi2_block"])]
        wb = [pos[b] for b in D.blocks_within(gap_mask, st_cfg["w_block"])]
        t: dict[str, Any] = {"homogeneity_g": D.homogeneity_chi2(per, hb)}
        t["constancy_w"] = {j: D.block_witness_constancy(per, wb, j) for j in w["offsets"]}
        grid = st_cfg["cp_grid"] if lab.startswith("rest") else st_cfg["cp_grid_episode"]
        t["changepoint"] = {}
        for i, s in enumerate(st_cfg["statistics"]):
            x = per_shot_statistic(s, per, data.arr[m])
            t["changepoint"][s] = D.changepoint_test(x, grid, st_cfg["n_perm"], st_cfg["seed"] + i)
        tests[lab] = t
        log(
            out,
            f"[stationarity] {lab}: homogeneity p_joint={t['homogeneity_g']['p_joint']:.3g}, "
            f"w0 constancy p={t['constancy_w'][0]['p']:.3g}, change-point p="
            + ", ".join(f"{k}:{v['p']:.3g}" for k, v in t["changepoint"].items()),
        )
    return {
        "segments": rows,
        "marginals": margs,
        "rolling": roll,
        "blocks_w": blocks_w,
        "tests": tests,
    }


def per_shot_statistic(name: str, per: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if name == "d13":
        return per[:, 1] - per[:, 3]
    if name == "p1":
        return arr.mean(1).astype(np.float64)
    if name == "w0_influence":
        return D.influence(per, 0)
    raise ValueError(name)


# -- 3. is the episode selected from the data? ----------------------------------------------------


def selection_stage(cfg, data: Data, out: Path) -> dict[str, Any]:
    sel, stc = cfg["selection"], cfg["stationarity"]
    a0, b0 = cfg["segments"]["episode"]
    res: dict[str, Any] = {"whole_run": {}, "boundaries": {}}
    for i, s in enumerate(stc["statistics"]):
        x = per_shot_statistic(s, data.per, data.arr)
        res["whole_run"][s] = D.changepoint_test(x, stc["cp_grid"], sel["n_perm"], sel["seed"] + i)
        _, a, b = D.epidemic_gain(x, sel["profile_step"])
        prof_a = D.boundary_profile(x, a, b, "start", sel["profile_step"])
        prof_b = D.boundary_profile(x, a, b, "stop", sel["profile_step"])
        ci = {}
        for lab, pr in (("start", prof_a), ("stop", prof_b)):
            ok = pr[:, 1] >= pr[:, 1].max() - 3.84
            ci[lab] = [int(pr[ok, 0].min()), int(pr[ok, 0].max())]
        res["boundaries"][s] = {
            "start": a,
            "stop": b,
            "ci95_start": ci["start"],
            "ci95_stop": ci["stop"],
        }
        if s == "d13":
            res["profile_start"] = prof_a.tolist()
            res["profile_stop"] = prof_b.tolist()
    # held out: boundaries from even shots, effect measured on odd shots
    d = per_shot_statistic("d13", data.per, data.arr)
    _, a2, b2 = D.epidemic_gain(d[0::2], sel["profile_step"] // 2 or 1)
    a_h, b_h = 2 * a2, 2 * b2
    odd = np.arange(1, data.n, 2)
    inside = (odd >= a_h) & (odd < b_h)
    x = d[odd]
    diff = x[inside].mean() - x[~inside].mean()
    se = math.sqrt(x[inside].var(ddof=1) / inside.sum() + x[~inside].var(ddof=1) / (~inside).sum())
    res["heldout_even_odd"] = {
        "start": int(a_h),
        "stop": int(b_h),
        "odd_shift_d13": float(diff),
        "odd_shift_z": float(diff / se),
    }
    # held out within the shot: find on positions 16-39, test on positions 40-63
    w = cfg["witness"]
    e = W.per_shot_windows(data.arr[:, 16:40], w["k_max"], 0, common=True)
    lt = W.per_shot_windows(data.arr[:, 40:64], w["k_max"], 0, common=True)
    _, a3, b3 = D.epidemic_gain(e[:, 1] - e[:, 3], sel["profile_step"])
    ins = np.zeros(data.n, bool)
    ins[a3:b3] = True
    y = lt[:, 1] - lt[:, 3]
    diff = y[ins].mean() - y[~ins].mean()
    se = math.sqrt(y[ins].var(ddof=1) / ins.sum() + y[~ins].var(ddof=1) / (~ins).sum())
    res["heldout_positions"] = {
        "start": int(a3),
        "stop": int(b3),
        "late_shift_d13": float(diff),
        "late_shift_z": float(diff / se),
    }
    # abrupt or gradual: ramp fits at each edge (centre profiled within +-100 shots)
    var = float(d.var())
    res["ramps"] = []
    for edge, c0 in (("start", a0), ("stop", b0)):
        rss = {}
        for wd in sel["ramp_widths"]:
            rss[wd] = min(
                D.ramp_fit(d, c, wd, sel["ramp_half"]) for c in range(c0 - 100, c0 + 101, 10)
            )
        best = min(rss.values())
        for wd, r in rss.items():
            res["ramps"].append({"edge": edge, "width": wd, "delta_lr": (r - best) / var})
    log(
        out,
        "[selection] whole-run p: "
        + ", ".join(
            f"{k}:{v['p']:.3g} ({v['start']}-{v['stop']})" for k, v in res["whole_run"].items()
        ),
    )
    return res


# -- 4. within-shot vs between-shot ---------------------------------------------------------------


def exact_windows_range(proc: W.LinearProcess, lo: int, hi: int, k_max: int) -> np.ndarray:
    """Exact mean of ``per_shot_windows(shots[:, lo:hi+1], k_max, 0, common=True)``."""
    gs = [proc.windows_at(proc.state_after(t), k_max) for t in range(lo, hi - k_max)]
    return np.mean(gs, axis=0)


def within_shot_stage(cfg, data: Data, cal: W.LinearProcess) -> dict[str, Any]:
    w, ws = cfg["witness"], cfg["within_shot"]
    (e0, e1), (l0, l1), (t0, t1) = ws["early"], ws["late"], ws["transient"]
    early = W.per_shot_windows(data.arr[:, e0 : e1 + 1], w["k_max"], 0, common=True)
    late = W.per_shot_windows(data.arr[:, l0 : l1 + 1], w["k_max"], 0, common=True)
    cal_e = exact_windows_range(cal, e0, e1, w["k_max"])
    cal_l = exact_windows_range(cal, l0, l1, w["k_max"])
    cal_m = cal.marginals(data.length)
    rows = []
    for name in ("all", "rest", "episode"):
        m = data.masks[name]
        n = int(m.sum())
        diff = late[m] - early[m]
        for k in range(w["k_max"] + 1):
            rows.append(
                {
                    "segment": name,
                    "k": k,
                    "g_early": float(early[m, k].mean()),
                    "g_late": float(late[m, k].mean()),
                    "late_minus_early": float(diff[:, k].mean()),
                    "z_paired": float(diff[:, k].mean() / (diff[:, k].std(ddof=1) / math.sqrt(n))),
                    "cal_early": float(cal_e[k]),
                    "cal_late": float(cal_l[k]),
                }
            )
    trans = []
    for name in ("all", "rest", "episode"):
        m = data.masks[name]
        n = int(m.sum())
        mg = data.arr[m].mean(0)
        sl = slice(t0, t1 + 1)
        c = cal_m[sl]
        chi_cal = float(np.sum((mg[sl] - c) ** 2 / (c * (1 - c) / n)))
        # shape only: transient relative to the segment's own post-burn level
        lvl = mg[w["burn"] :].mean() - cal_m[w["burn"] :].mean()
        chi_shape = float(np.sum((mg[sl] - c - lvl) ** 2 / (c * (1 - c) / n)))
        trans.append(
            {
                "segment": name,
                "positions": f"{t0}-{t1}",
                "chi2_vs_calibrated": chi_cal,
                "chi2_shape_vs_calibrated": chi_shape,
                "dof": t1 - t0 + 1,
                "p": D.chi2_sf(chi_cal, t1 - t0 + 1),
                "p_shape": D.chi2_sf(chi_shape, t1 - t0),
            }
        )
    witness_halves = {}
    for name in ("rest", "episode"):
        m = data.masks[name]
        witness_halves[name] = {
            "early": [W.witness(early[m].mean(0), j) for j in w["offsets"]],
            "late": [W.witness(late[m].mean(0), j) for j in w["offsets"]],
            "early_se": [D.block_se(early[m], j, 1) for j in w["offsets"]],
            "late_se": [D.block_se(late[m], j, 1) for j in w["offsets"]],
        }
    return {
        "g_halves": rows,
        "transient": trans,
        "witness_halves": witness_halves,
        "timescales": {
            "step_s": cfg["timing"]["step_s"],
            "shot_s": cfg["timing"]["shot_s"],
            "half_shot_s": cfg["timing"]["shot_s"] / 2,
        },
    }


# -- 5. drift nulls -------------------------------------------------------------------------------


def make_target(cfg, data: Data, seg: str, z_max: float | None = None) -> D.Target:
    w, nu, u = cfg["witness"], cfg["nulls"], cfg["uncertainty"]
    burn = w["burn"]
    m = data.masks[seg]
    parts = [data.episode, ~data.episode] if seg == "all" else [m]
    n = int(m.sum())

    def se_mean(x_all: np.ndarray) -> float:  # primary SE of a mean (stratified when pooled)
        tot = 0.0
        for pm in parts:
            x = x_all[pm]
            s = max(D.batch_means_se(x, u["primary_b"]), x.std(ddof=1) / math.sqrt(len(x)))
            tot += (len(x) / n) ** 2 * s**2
        return math.sqrt(tot)

    g_se = np.array([se_mean(data.per[:, k]) for k in range(w["k_max"] + 1)])
    arr = data.arr[m]
    p_shot = data.arr[:, burn:].mean(1).astype(np.float64)
    iid = p_shot[m].std(ddof=1) / math.sqrt(n)
    infl = max(1.0, se_mean(p_shot) / iid)
    marg = arr[:, burn:].mean(0)
    marg_se = np.sqrt(marg * (1 - marg) / n) * infl
    f = block_stats(arr, burn, 4)
    cov = np.cov(f, rowvar=False) / n * infl**2
    zm = nu["z_max"] if z_max is None else z_max
    scale = (zm / nu["z_max"]) ** 2
    q = nu["chi2_quantile"]
    return D.Target(
        g=data.per[m].mean(0),
        g_se=g_se,
        marg=marg,
        marg_se=marg_se,
        block=f.mean(0),
        block_cov_inv=np.linalg.pinv(cov, rcond=1e-10),
        z_max=zm,
        marg_chi2_max=D.chi2_ppf(q, data.length - burn) * scale,
        block_chi2_max=D.chi2_ppf(q, 15) * scale,
    )


def family_by_name(cfg, name: str, shot_period_s: float) -> D.Family:
    w, t = cfg["witness"], cfg["timing"]
    length, k_max, burn = 64, w["k_max"], w["burn"]
    if name.startswith("mixture_K"):
        k = int(name.removeprefix("mixture_K").split("_")[0])
        return D.mixture_family(k, length, k_max, burn, contiguous=name.endswith("contiguous"))
    if name.startswith("continuous"):
        return D.continuous_family(
            cfg["nulls"]["continuous_grid"],
            length,
            k_max,
            burn,
            contiguous=name.endswith("contiguous"),
        )
    if name.startswith("telegraph_tau"):
        tau = float(name.removeprefix("telegraph_tau"))
        return D.telegraph_family(tau, t["step_s"], shot_period_s, length, k_max, burn)
    if name == "within_shot_drift":
        return D.within_shot_drift_family(length, k_max, burn)
    if name.startswith("hmm_"):
        return D.hmm_family(
            int(name.removeprefix("hmm_").removesuffix("state")), length, k_max, burn
        )
    raise ValueError(name)


def drift_family_names(cfg) -> list[str]:
    nu = cfg["nulls"]
    names = [f"mixture_K{k}" for k in nu["mixtures"]] + [
        "continuous_contiguous",
        "within_shot_drift",
    ]
    names += [f"telegraph_tau{t:.0e}" for t in nu["taus_s"]]
    return names


def adversarial_task(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    cfg = task["cfg"]
    data = Data(cfg)
    tgt = make_target(cfg, data, task["target"], task.get("z_max"))
    fam = family_by_name(cfg, task["family"], task["shot_period_s"])
    nu = cfg["nulls"]
    w_scale = abs(W.witness(tgt.g, 0))
    t0 = time.time()
    r = D.adversarial_search(
        fam.stats,
        fam.n_params,
        tgt,
        cfg["witness"]["burn"],
        w_scale,
        nu["restarts"],
        tuple(nu["steps"]),
        task["seed"],
        nu["lr"],
    )
    fit = r["fit"]
    pen2 = D.penalty_t(
        {k: torch.as_tensor(r[k]) for k in ("z_g", "chi2_marg", "chi2_block")}, tgt
    ).numpy()
    rows = []
    for i in range(nu["restarts"]):
        rows.append(
            {
                "restart": i,
                "feasible": bool(r["feasible"][i]),
                **{f"w{j}": float(r[f"w{j}"][i]) for j in range(4)},
                "max_abs_z_g": float(r["max_abs_z_g"][i]),
                "chi2_marg": float(r["chi2_marg"][i]),
                "chi2_block": float(r["chi2_block"][i]),
                "penalty": float(pen2[i]),
                "fit_penalty": float(fit["penalty"][i]),
                "fit_max_abs_z_g": float(fit["max_abs_z_g"][i]),
                "fit_w0": float(fit["w0"][i]),
            }
        )
    feas = r["feasible"]
    best_feas = int(np.argmin(np.where(feas, r["w0"], np.inf))) if feas.any() else None
    i_cons = int(np.argmin(pen2))
    i_fit = int(np.argmin(fit["penalty"]))
    return {
        "target": task["target"],
        "family": task["family"],
        "z_max": tgt.z_max,
        "seconds": time.time() - t0,
        "rows": rows,
        "best_feasible": best_feas,
        "theta_best_feasible": None if best_feas is None else r["theta"][best_feas].tolist(),
        "most_consistent": i_cons,
        "theta_most_consistent": r["theta"][i_cons].tolist(),
        "best_fit": i_fit,
        "theta_best_fit": fit["theta"][i_fit].tolist(),
        "thresholds": {
            "z_max": tgt.z_max,
            "marg_chi2_max": tgt.marg_chi2_max,
            "block_chi2_max": tgt.block_chi2_max,
        },
    }


def diagnostics(cfg, shots: np.ndarray, n_perm: int, n_boot: int, seed: int) -> dict[str, float]:
    """The step-2 diagnostics of one (simulated or real) 20,000 x 64 run."""
    w, st, u = cfg["witness"], cfg["stationarity"], cfg["uncertainty"]
    per = W.per_shot_windows(shots, w["k_max"], w["burn"], common=True)
    g = per.mean(0)
    out: dict[str, float] = {"p1": float(shots.mean()), "g1": float(g[1]), "g3": float(g[3])}
    for j in w["offsets"]:
        out[f"w{j}"] = W.witness(g, j)
        se = primary_se(per, j, u["primary_b"], n_boot, seed + j)
        out[f"se_j{j}"] = se["primary"]
        out[f"z_j{j}"] = out[f"w{j}"] / se["primary"]
        out[f"block_over_shot_j{j}"] = se["block"] / se["shot"]
    allm = np.ones(len(shots), bool)
    out["homogeneity_p_joint"] = D.homogeneity_chi2(per, D.blocks_within(allm, st["chi2_block"]))[
        "p_joint"
    ]
    c = D.block_witness_constancy(per, D.blocks_within(allm, st["w_block"]), 0)
    out["w0_constancy_p"] = c["p"]
    out["w0_block_min"], out["w0_block_max"] = c["w_min"], c["w_max"]
    cp = D.changepoint_test(per[:, 1] - per[:, 3], st["cp_grid"], n_perm, seed + 99)
    out["changepoint_p_d13"] = cp["p"]
    out["changepoint_stat_d13"] = cp["stat"]
    return out


def sim_task(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    cfg = task["cfg"]
    nu = cfg["nulls"]
    rows = []
    for rep in task["reps"]:
        seed = task["seed"] + rep
        if task["kind"] == "family":
            fam = family_by_name(cfg, task["family"], task["shot_period_s"])
            shots = fam.sampler(np.array(task["theta"]), 20000, seed)
        else:  # fitted regimes
            comps = [(np.array(c["labelled"]), np.array(c["initial"])) for c in task["comps"]]
            if task["assignment"] == "contiguous":
                labels = np.zeros(20000, dtype=np.int64)
                a, b = cfg["segments"]["episode"]
                labels[a:b] = 1
            else:
                labels = np.random.default_rng(seed).choice(2, size=20000, p=task["weights"])
            shots = D.sample_mixture(comps, labels, 64, seed + 1)
        d = diagnostics(cfg, shots, nu["sim_perm"], nu["sim_boot"], seed)
        rows.append({"null": task["name"], "rep": rep, "seed": seed, **d})
    return {"rows": rows}


def fit_two_state(cfg, shots: np.ndarray) -> dict[str, Any]:
    nu = cfg["nulls"]
    best = None
    for lab, ini in (W.renewal_init(shots, 2), W.markov_init(shots, 1)):
        r = W.fit_em(lab, ini, shots, max_iters=nu["em_max_iters"], tol=nu["em_tol"])
        if best is None or r["train_nll"] < best["train_nll"]:
            best = r
    return {
        "labelled": np.asarray(best["labelled"]).tolist(),
        "initial": np.asarray(best["initial"]).tolist(),
        "train_nll": best["train_nll"],
    }


def fitted_regimes_stage(cfg, data: Data) -> dict[str, Any]:
    w, nu = cfg["witness"], cfg["nulls"]
    fits = {s: fit_two_state(cfg, data.arr[data.masks[s]]) for s in ("rest", "episode")}
    procs = {
        s: W.LinearProcess.hmm(np.array(f["labelled"]), np.array(f["initial"]))
        for s, f in fits.items()
    }
    gs = {s: p.expected_windows(w["k_max"], w["burn"], 64, common=True) for s, p in procs.items()}
    wt = data.episode.mean()
    mix = D.mixture([procs["rest"], procs["episode"]], [1 - wt, wt])
    gmix = mix.expected_windows(w["k_max"], w["burn"], 64, common=True)
    res: dict[str, Any] = {"fits": fits, "episode_weight": float(wt)}
    res["exact"] = {
        **{f"{s}_w{j}": W.witness(gs[s], j) for s in gs for j in w["offsets"]},
        **{f"mixture_w{j}": W.witness(gmix, j) for j in w["offsets"]},
        "mixture_p1_post_burn": float(mix.marginals(64)[w["burn"] :].mean()),
    }
    # mean shifts: the rest's fit plus a copy in which a fraction s of every 0-emission becomes
    # a 1-emission (L2[1] = L[1] + s L[0], L2[0] = (1 - s) L[0]): transitions are unchanged and
    # P(1) rises by s * P(0); s is set so that the shift in P(1) is delta.
    lab = np.array(fits["rest"]["labelled"])
    p0 = 1 - float(procs["rest"].marginals(64)[w["burn"] :].mean())
    shift = []
    for wgt in nu["shift_weights"]:
        for dl in nu["shift_deltas"]:
            sfr = dl / p0
            lab2 = np.stack([(1 - sfr) * lab[0], lab[1] + sfr * lab[0]])
            p2 = W.LinearProcess.hmm(lab2, np.array(fits["rest"]["initial"]))
            mx = D.mixture([procs["rest"], p2], [1 - wgt, wgt])
            g = mx.expected_windows(w["k_max"], w["burn"], 64, common=True)
            shift.append(
                {
                    "weight": wgt,
                    "delta_p1": dl,
                    "p1_shift": float(p2.marginals(64)[w["burn"] :].mean() - (1 - p0)),
                    "shifted_w0_alone": W.witness(
                        p2.expected_windows(w["k_max"], w["burn"], 64, common=True), 0
                    ),
                    **{f"w{j}": W.witness(g, j) for j in w["offsets"]},
                }
            )
    res["mean_shift"] = shift
    return res


# -- 6. physical reading of the episode -----------------------------------------------------------


def _rz_super(delta: float) -> np.ndarray:
    u = np.diag([np.exp(-0.5j * delta), np.exp(0.5j * delta)])
    return np.kron(u.conj(), u)


def _dephase_super(p: float) -> np.ndarray:
    return np.diag([1.0, 1.0 - p, 1.0 - p, 1.0]).astype(np.complex128)


def _relax_super(gamma: float) -> np.ndarray:
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - gamma)]])
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    return (np.kron(k0.conj(), k0) + np.kron(k1.conj(), k1)).astype(np.complex128)


def physical_superops(sup: np.ndarray, model: str, x: np.ndarray) -> np.ndarray:
    """The calibrated step superoperators (column-stacked ``vec``) with extra physics on the
    memory q75 after each step, or an extra assignment error on the q59 readout."""
    if model == "detuning":
        a = _rz_super(x[0])
        return np.stack([a @ s for s in sup])
    if model == "detuning_dephasing":
        a = _dephase_super(x[1]) @ _rz_super(x[0])
        return np.stack([a @ s for s in sup])
    if model == "dephasing_relaxation":
        a = _relax_super(x[1]) @ _dephase_super(x[0])
        return np.stack([a @ s for s in sup])
    if model == "readout":
        e01, e10 = x  # P(report 1 | would report 0), P(report 0 | would report 1)
        return np.stack([(1 - e01) * sup[0] + e10 * sup[1], e01 * sup[0] + (1 - e10) * sup[1]])
    raise ValueError(model)


def _bounds(model: str) -> list[tuple[float, float]]:
    return {
        "detuning": [(-math.pi, math.pi)],
        "detuning_dephasing": [(-math.pi, math.pi), (0.0, 0.4)],
        "dephasing_relaxation": [(0.0, 0.4), (0.0, 0.4)],
        "readout": [(0.0, 0.2), (0.0, 0.2)],
    }[model]


def nelder_mead(
    f, x0: np.ndarray, bounds, iters: int, scale: float = 0.05
) -> tuple[np.ndarray, float]:
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    clip = lambda x: np.clip(x, lo, hi)  # noqa: E731
    d = len(x0)
    simplex = [clip(np.asarray(x0, dtype=np.float64))]
    for i in range(d):
        e = np.zeros(d)
        e[i] = scale * (hi[i] - lo[i])
        simplex.append(clip(simplex[0] + e))
    vals = [f(x) for x in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(simplex[:-1], axis=0)
        xr = clip(c + (c - simplex[-1]))
        fr = f(xr)
        if fr < vals[0]:
            xe = clip(c + 2 * (c - simplex[-1]))
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = clip(c + 0.5 * (simplex[-1] - c))
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, d + 1):
                    simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    vals[i] = f(simplex[i])
    i = int(np.argmin(vals))
    return simplex[i], float(vals[i])


def physics_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    data = Data(cfg)
    sup, rho0 = calibrated_superops(cfg)
    w, ph = cfg["witness"], cfg["physics"]
    shots = data.arr[data.masks[task["segment"]]]
    model = task["model"]

    def nll(x):
        p = W.LinearProcess.instrument(physical_superops(sup, model, np.asarray(x)), rho0)
        return float(p.seq_nll(shots).sum() / shots.size)

    bnds = _bounds(model)
    grids = [
        np.linspace(lo, hi, ph["detuning_grid"] if lo < 0 else ph["prob_grid"]) for lo, hi in bnds
    ]
    best, bx = math.inf, None
    for x in np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(bnds), -1).T:
        v = nll(x)
        if v < best:
            best, bx = v, x
    x, v = nelder_mead(nll, bx, bnds, ph["nm_iters"])
    p = W.LinearProcess.instrument(physical_superops(sup, model, x), rho0)
    g = p.expected_windows(w["k_max"], w["burn"], 64, common=True)
    base = W.LinearProcess.instrument(sup, rho0)
    nll0 = float(base.seq_nll(shots).sum() / shots.size)
    return {
        "segment": task["segment"],
        "model": model,
        "params": x.tolist(),
        "nll": v,
        "nll_calibrated": nll0,
        "delta_nll_per_symbol": v - nll0,
        "lr_stat": 2 * (nll0 - v) * shots.size,
        "g": g.tolist(),
        "w": [W.witness(g, j) for j in w["offsets"]],
        "p1_post_burn": float(p.marginals(64)[w["burn"] :].mean()),
    }


# -- 7. uncertainty policy ------------------------------------------------------------------------


def uncertainty_stage(cfg, data: Data) -> list[dict[str, Any]]:
    u, w = cfg["uncertainty"], cfg["witness"]
    rows = []
    for name in ("all", "rest", "episode"):
        per = data.per[data.masks[name]]
        for b in u["block_sizes"]:
            idx = D.block_bootstrap_indices(len(per), min(b, len(per)), u["n_boot"], u["seed"] + b)
            r = {"segment": name, "b": b, "method": "block_bootstrap"}
            for j in w["offsets"]:
                r[f"se_j{j}"] = D.bootstrap_se(per, j, idx)
            rows.append(r)
            r = {"segment": name, "b": b, "method": "block_delta"}
            for j in w["offsets"]:
                r[f"se_j{j}"] = D.block_se(per, j, b)
            rows.append(r)
        if name == "all":
            for b in u["block_sizes"]:
                idx = D.stratified_block_indices(
                    [np.nonzero(data.episode)[0], np.nonzero(~data.episode)[0]],
                    b,
                    u["n_boot"],
                    u["seed"] + 7 * b,
                )
                r = {"segment": "all", "b": b, "method": "stratified_block_bootstrap"}
                for j in w["offsets"]:
                    r[f"se_j{j}"] = D.bootstrap_se(data.per, j, idx)
                rows.append(r)
    return rows


# -- pool -----------------------------------------------------------------------------------------


def task_key(t: dict[str, Any]) -> str:
    k = t["kind"]
    if k == "adversarial":
        return f"adv_{t['target']}_{t['family']}_z{t.get('z_max') or 'base'}"
    if k == "physics":
        return f"phys_{t['segment']}_{t['model']}"
    return f"sim_{t['name']}_{t['reps'][0]}"


def dispatch(task: dict[str, Any]) -> Any:
    if task["kind"] == "adversarial":
        return adversarial_task(task)
    if task["kind"] == "physics":
        return physics_task(task)
    return sim_task(task)


def run_pool(tasks: list[dict[str, Any]], workers: int, cache: Path, out: Path) -> list[Any]:
    cache.mkdir(parents=True, exist_ok=True)
    results: dict[int, Any] = {}
    todo = []
    for i, t in enumerate(tasks):
        f = cache / f"{task_key(t)}.json"
        if f.exists():
            results[i] = json.loads(f.read_text())["result"]
        else:
            todo.append(i)
    log(out, f"[pool] {len(tasks)} tasks, {len(todo)} to run on {workers} workers")

    def keep(i: int, r: Any) -> None:
        results[i] = r
        save_json(cache / f"{task_key(tasks[i])}.json", {"result": r})
        log(out, f"[pool] done {task_key(tasks[i])} ({len(results)}/{len(tasks)})")

    if workers <= 1:
        for i in todo:
            keep(i, dispatch(tasks[i]))
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(dispatch, tasks[i]): i for i in todo}
            for fu in as_completed(futs):
                keep(futs[fu], fu.result())
    return [results[i] for i in range(len(tasks))]


# -- gate -----------------------------------------------------------------------------------------


def gate(res: dict[str, Any], cfg) -> dict[str, Any]:
    seg = {r["segment"]: r for r in res["segments"]["segments"]}
    z_rest, z_ep = seg["rest"]["z_primary_j0"], seg["episode"]["z_primary_j0"]
    c1 = abs(z_rest) >= 5 and abs(z_ep) >= 5
    const_p = res["segments"]["tests"]["rest"]["constancy_w"][0]["p"]
    c2 = const_p >= 0.01
    w_rest = seg["rest"]["w0"]
    reach, contradict = [], []
    for r in res["adversarial"]:
        if r["target"] != "rest" or r["z_max"] != cfg["nulls"]["z_max"]:
            continue
        if r["family"].startswith("hmm_"):
            continue
        for row in r["rows"]:
            if not row["feasible"]:
                continue
            big = abs(row["w0"]) >= 0.5 * abs(w_rest) and np.sign(row["w0"]) == np.sign(w_rest)
            shape = all(
                abs(row[f"w{j}"] - seg["rest"][f"w{j}"]) <= 3 * seg["rest"][f"se_primary_j{j}"]
                for j in (1, 2, 3)
            )
            if big:
                reach.append((r["family"], row["restart"], row["w0"]))
            if big and shape:
                contradict.append((r["family"], row["restart"]))
    c3 = len(contradict) == 0 and len(reach) == 0
    verdict = "robust" if (c1 and c2 and c3) else ("contradicted" if contradict else "inconclusive")
    return {
        "z_primary_rest": z_rest,
        "z_primary_episode": z_ep,
        "criterion_1_nonzero": c1,
        "rest_w0_block_constancy_p": const_p,
        "criterion_2_constant": c2,
        "drift_nulls_reaching_half": reach,
        "drift_nulls_matching_shape": contradict,
        "criterion_3_no_drift_null": c3,
        "verdict": verdict,
    }


# -- plots ----------------------------------------------------------------------------------------


def plots(res: dict[str, Any], cfg, data: Data, cal: W.LinearProcess, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = cfg["witness"]
    a, b = cfg["segments"]["episode"]
    roll = res["segments"]["rolling"]
    x = np.array([r["centre"] for r in roll])
    cal_g = cal.expected_windows(w["k_max"], w["burn"], 64, common=True)
    seg = {r["segment"]: r for r in res["segments"]["segments"]}
    # 1. segment statistics
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for axi, key, lab in ((ax[0, 0], "w0", "w(0)"), (ax[0, 1], "w2", "w(2)")):
        j = int(key[1])
        y = np.array([r[key] for r in roll])
        s = np.array([r[f"se_j{j}"] for r in roll])
        axi.fill_between(x, y - 2 * s, y + 2 * s, color="#9ecae1", alpha=0.6, lw=0)
        axi.plot(x, y, color="#08519c", lw=1.2, label="rolling 1,000 shots (±2 i.i.d. SE)")
        dec = [seg[f"decile{i}"] for i in range(10)]
        axi.errorbar(
            [(d["first"] + d["last"]) / 2 for d in dec],
            [d[key] for d in dec],
            yerr=[2 * d[f"se_primary_j{j}"] for d in dec],
            fmt="o",
            color="#d94801",
            ms=4,
            label="deciles (±2 primary SE)",
        )
        axi.axhline(W.witness(cal_g, j), color="k", ls="--", lw=0.8, label="calibrated")
        axi.set_ylabel(lab)
    for axi, keys, lab in ((ax[1, 0], ("g1", "g3"), "g(k)"), (ax[1, 1], ("p1",), "P(1)")):
        for k in keys:
            axi.plot(
                x,
                [r[k] for r in roll],
                lw=1.2,
                label=k.replace("g", "g(") + ")" if k != "p1" else "P(1)",
            )
        axi.set_ylabel(lab)
        axi.set_xlabel("shot (returned order)")
    for axi in ax.ravel():
        axi.axvspan(a, b, color="#f2c14e", alpha=0.25)
        axi.legend(fontsize=7, loc="best")
    fig.suptitle("Segment statistics in returned order (episode shaded)")
    fig.tight_layout()
    fig.savefig(sd / "segments.png", dpi=120)
    plt.close(fig)
    # 2. SE against block size
    rows = res["uncertainty"]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for j in w["offsets"]:
        for name, col in (("all", "#cb181d"), ("rest", "#2171b5"), ("episode", "#238b45")):
            rr = [r for r in rows if r["segment"] == name and r["method"] == "block_bootstrap"]
            s1 = rr[0][f"se_j{j}"]
            ax[j].plot(
                [r["b"] for r in rr], [r[f"se_j{j}"] / s1 for r in rr], "o-", color=col, label=name
            )
        rr = [r for r in rows if r["method"] == "stratified_block_bootstrap"]
        s1 = next(
            r
            for r in rows
            if r["segment"] == "all" and r["b"] == 1 and r["method"] == "block_bootstrap"
        )[f"se_j{j}"]
        ax[j].plot(
            [r["b"] for r in rr],
            [r[f"se_j{j}"] / s1 for r in rr],
            "s--",
            color="#cb181d",
            mfc="white",
            label="all, stratified",
        )
        ax[j].axvline(cfg["uncertainty"]["primary_b"], color="grey", lw=0.8, ls=":")
        ax[j].set_xscale("symlog", linthresh=1)
        ax[j].set_title(f"w({j})")
        ax[j].set_xlabel("block size b (shots)")
    ax[0].set_ylabel("SE / whole-shot bootstrap SE")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sd / "se_vs_b.png", dpi=120)
    plt.close(fig)
    # 3. drift nulls: data consistency against |w(0)|
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = {
        "mixture": "#2171b5",
        "continuous": "#6a51a3",
        "within_shot": "#238b45",
        "telegraph": "#d94801",
        "hmm": "#636363",
    }
    for axi, tgt in zip(ax, ("rest", "episode"), strict=True):
        wt = seg[tgt]["w0"]
        seen = set()
        for r in res["adversarial"]:
            if r["target"] != tgt or r["z_max"] != cfg["nulls"]["z_max"]:
                continue
            grp = next(g for g in groups if r["family"].startswith(g))
            lab = "general 3-4 state HMM (control)" if grp == "hmm" else grp.replace("_", "-")
            zz = [row["max_abs_z_g"] for row in r["rows"]]
            ww = [row["w0"] / wt for row in r["rows"]]
            axi.scatter(
                zz,
                ww,
                s=8,
                color=groups[grp],
                alpha=0.5,
                label=None if lab in seen else lab,
                marker="x" if grp == "hmm" else "o",
            )
            seen.add(lab)
        axi.axvline(cfg["nulls"]["z_max"], color="k", lw=0.8, ls="--")
        axi.axhline(0.5, color="grey", lw=0.8, ls=":")
        axi.axhline(1.0, color="grey", lw=0.8)
        axi.set_xscale("log")
        axi.set_xlabel("max |z| of g(0..7) against the data")
        axi.set_ylabel(f"w(0) / device w(0) ({tgt})")
        axi.set_title(f"adversarial searches, target: {tgt}")
        axi.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sd / "nulls.png", dpi=120)
    plt.close(fig)
    # 4. the episode: g(k) and transient, data against calibrated and physical fits
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(w["k_max"] + 1)
    cols = {"rest": "#2171b5", "episode": "#cb181d"}
    for name in ("rest", "episode"):
        g = np.array([seg[name][f"g{i}"] for i in k])
        ax[0].plot(k, g, "o", color=cols[name], label=f"{name} (data)")
        best = min((p for p in res["physics"] if p["segment"] == name), key=lambda p: p["nll"])
        ax[0].plot(
            k, best["g"], "-", color=cols[name], lw=1, label=f"{name}: best fit ({best['model']})"
        )
    ax[0].plot(k, cal_g, "k--", lw=0.8, label="calibrated")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("g(k) = P(1 0^k 1)")
    ax[0].legend(fontsize=7)
    cm = cal.marginals(64)
    for mrow in res["segments"]["marginals"]:
        if mrow["segment"] in cols:
            ax[1].plot(
                range(64),
                [mrow[f"t{t}"] for t in range(64)],
                color=cols[mrow["segment"]],
                lw=1,
                label=mrow["segment"],
            )
    ax[1].plot(range(64), cm, "k--", lw=0.8, label="calibrated")
    ax[1].axvline(w["burn"] - 0.5, color="grey", lw=0.8, ls=":")
    ax[1].set_xlabel("position in shot")
    ax[1].set_ylabel("P(y_t = 1)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sd / "episode.png", dpi=120)
    plt.close(fig)


# -- main -----------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument(
        "--quick", action="store_true", help="smoke test (tiny searches, nothing curated)"
    )
    ap.add_argument("--from-run", type=Path, help="resume: reuse this run dir's task results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        nu = cfg["nulls"]
        nu["restarts"], nu["steps"], nu["sim_reps"], nu["sim_perm"], nu["sim_boot"] = (
            4,
            [20, 20],
            1,
            9,
            20,
        )
        nu["taus_s"], nu["mixtures"], nu["ladder_z"] = [3e-5], [1, 2], [10.0]
        nu["ladder_families"] = ["mixture_K2"]
        nu["positive_controls"] = [3]
        cfg["stationarity"]["n_perm"] = cfg["selection"]["n_perm"] = 19
        cfg["segments"]["n_boot"] = cfg["uncertainty"]["n_boot"] = 50
        cfg["physics"].update(detuning_grid=7, prob_grid=3, nm_iters=3)
    workers = args.workers or cfg["experiment"]["workers"]
    out = args.from_run.resolve() if args.from_run else new_run_dir(cfg["experiment"]["name"])
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    meta = run_metadata(clean, cfg["nulls"]["seed"])
    meta["quick"] = args.quick
    t0 = time.time()
    try:
        data = Data(cfg)
        cal = calibrated(cfg)
        res: dict[str, Any] = {"meta": meta}
        res["order"] = order_stage(cfg)
        period = res["order"]["jobs"]["main"]["shot_period_s"]
        log(
            out,
            f"[order] one span; shot period {period * 1e6:.1f} us; episode "
            f"{res['order']['episode_duration_s_if_time_order']:.2f} s if returned order is "
            "time order",
        )
        res["segments"] = segment_stage(cfg, data, cal, out)
        res["selection"] = selection_stage(cfg, data, out)
        res["within_shot"] = within_shot_stage(cfg, data, cal)
        res["uncertainty"] = uncertainty_stage(cfg, data)
        res["fitted_regimes"] = fitted_regimes_stage(cfg, data)
        log(
            out,
            "[fitted regimes] exact mixture w(0) = "
            f"{res['fitted_regimes']['exact']['mixture_w0']:.3g}",
        )
        save_json(out / "partial.json", res)
        nu = cfg["nulls"]
        tcfg = {k: v for k, v in cfg.items()}
        tasks: list[dict[str, Any]] = []
        names = drift_family_names(cfg) + [f"hmm_{d}state" for d in nu["positive_controls"]]
        for ti, tgt in enumerate(nu["targets"]):
            for fi, fam in enumerate(names):
                tasks.append(
                    {
                        "kind": "adversarial",
                        "cfg": tcfg,
                        "target": tgt,
                        "family": fam,
                        "seed": nu["seed"] + 1000 * ti + fi,
                        "shot_period_s": period,
                    }
                )
        for zi, z in enumerate(nu["ladder_z"]):
            for fi, fam in enumerate(nu["ladder_families"]):
                tasks.append(
                    {
                        "kind": "adversarial",
                        "cfg": tcfg,
                        "target": "rest",
                        "family": fam,
                        "z_max": z,
                        "seed": nu["seed"] + 50000 + 100 * zi + fi,
                        "shot_period_s": period,
                    }
                )
        for seg_name in ("rest", "episode"):
            for model in cfg["physics"]["models"]:
                tasks.append({"kind": "physics", "cfg": tcfg, "segment": seg_name, "model": model})
        results = run_pool(tasks, workers, out / "tasks", out)
        res["adversarial"] = [
            r for r in results if isinstance(r, dict) and "rows" in r and "family" in r
        ]
        res["physics"] = [r for r in results if isinstance(r, dict) and "model" in r]
        # simulations: fitted regimes and each drift family's most data-consistent member
        sims = []
        fr = res["fitted_regimes"]
        comps = [fr["fits"]["rest"], fr["fits"]["episode"]]
        wt = fr["episode_weight"]
        reps = list(range(nu["sim_reps"]))
        for asg in ("contiguous", "random"):
            sims.append(
                {
                    "kind": "sim_fitted",
                    "name": f"fitted_regimes_{asg}",
                    "comps": comps,
                    "weights": [1 - wt, wt],
                    "assignment": asg,
                    "reps": reps,
                    "seed": nu["sim_seed"],
                    "cfg": tcfg,
                }
            )
        for r in res["adversarial"]:
            if r["target"] != "all" or r["z_max"] != nu["z_max"] or r["family"].startswith("hmm_"):
                continue
            fams = [r["family"]] + (
                ["mixture_K2_contiguous"] if r["family"] == "mixture_K2" else []
            )
            for fam in fams:
                sims.append(
                    {
                        "kind": "family",
                        "name": f"{fam}_most_consistent",
                        "family": fam,
                        "theta": r["theta_most_consistent"],
                        "reps": reps,
                        "seed": nu["sim_seed"] + 10000 * (len(sims) + 1),
                        "shot_period_s": period,
                        "cfg": tcfg,
                    }
                )
        sim_res = run_pool(sims, workers, out / "tasks", out)
        res["simulations"] = [row for s in sim_res for row in s["rows"]]
        res["device_diagnostics"] = diagnostics(
            cfg, data.arr, cfg["stationarity"]["n_perm"], cfg["uncertainty"]["n_boot"], 7
        )
        res["gate"] = gate(res, cfg)
        log(
            out,
            f"[gate] {res['gate']['verdict']}: z_rest={res['gate']['z_primary_rest']:.1f}, "
            f"z_episode={res['gate']['z_primary_episode']:.1f}, constancy p="
            f"{res['gate']['rest_w0_block_constancy_p']:.3g}, "
            f"reach={len(res['gate']['drift_nulls_reaching_half'])}",
        )
        res["wall_seconds"] = time.time() - t0
        save_json(out / "drift.json", res)
        adv_rows = [
            {"target": r["target"], "family": r["family"], "z_max": r["z_max"], **row}
            for r in res["adversarial"]
            for row in r["rows"]
        ]
        write_csv(out / "adversarial_restarts.csv", adv_rows)
        write_csv(out / "simulations.csv", res["simulations"])
        if not args.quick:
            curate(res, cfg, data, cal, out, adv_rows)
        log(out, f"done in {time.time() - t0:.0f} s -> {out}")
    except BaseException as e:
        log(out, f"FAILED: {e!r}\n{traceback.format_exc()}")
        save_json(out / "failed.json", {"meta": meta, "error": repr(e)})
        raise


def adversarial_summary(res: dict[str, Any], cfg) -> list[dict[str, Any]]:
    seg = {r["segment"]: r for r in res["segments"]["segments"]}
    rows = []
    for r in res["adversarial"]:
        rr = r["rows"]
        feas = [x for x in rr if x["feasible"]]
        fit = min(rr, key=lambda x: x["fit_penalty"])
        cons = min(rr, key=lambda x: x["penalty"])
        most = min(rr, key=lambda x: x["w0"])
        dev = seg[r["target"]]
        row = {
            "target": r["target"],
            "family": r["family"],
            "z_max": r["z_max"],
            "restarts": len(rr),
            "feasible": len(feas),
            "best_fit_max_abs_z_g": fit["fit_max_abs_z_g"],
            "best_fit_w0": fit["fit_w0"],
            "most_consistent_max_abs_z_g": cons["max_abs_z_g"],
            "most_consistent_chi2_block": cons["chi2_block"],
            **{f"most_consistent_w{j}": cons[f"w{j}"] for j in range(4)},
            "largest_consistent_w0": min(x["w0"] for x in feas) if feas else None,
            "largest_any_w0": most["w0"],
            "largest_any_max_abs_z_g": most["max_abs_z_g"],
        }
        for lab in ("all", "rest", "episode"):
            ref = seg[lab]["w0"]
            row[f"ratio_consistent_over_{lab}"] = (
                row["largest_consistent_w0"] / ref if feas else None
            )
        row["ratio_any_over_target"] = most["w0"] / dev["w0"]
        rows.append(row)
    return rows


def curate(res, cfg, data, cal, out: Path, adv_rows) -> None:
    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    sd.mkdir(parents=True, exist_ok=True)
    res["meta"]["run_dir"] = str(out.relative_to(REPO_ROOT))
    res["adversarial_summary"] = adversarial_summary(res, cfg)
    sims = res["simulations"]
    sim_sum = []
    for name in dict.fromkeys(r["null"] for r in sims):
        rr = [r for r in sims if r["null"] == name]
        row = {"null": name, "reps": len(rr)}
        for key in (
            "w0",
            "w1",
            "w2",
            "w3",
            "z_j0",
            "p1",
            "g1",
            "g3",
            "block_over_shot_j0",
            "homogeneity_p_joint",
            "w0_constancy_p",
            "changepoint_p_d13",
        ):
            v = np.array([r[key] for r in rr])
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        sim_sum.append(row)
    res["simulation_summary"] = sim_sum
    small = {k: v for k, v in res.items() if k not in ("adversarial", "simulations")}
    small["adversarial_best"] = [
        {
            k: r[k]
            for k in (
                "target",
                "family",
                "z_max",
                "best_feasible",
                "most_consistent",
                "theta_most_consistent",
                "theta_best_fit",
                "thresholds",
                "seconds",
            )
        }
        for r in res["adversarial"]
    ]
    save_json(sd / "summary.json", small)
    write_csv(sd / "segments.csv", res["segments"]["segments"], 6)
    write_csv(sd / "segment_marginals.csv", res["segments"]["marginals"], 6)
    write_csv(sd / "rolling.csv", res["segments"]["rolling"], 6)
    write_csv(sd / "blocks_w.csv", res["segments"]["blocks_w"], 6)
    write_csv(sd / "within_shot.csv", res["within_shot"]["g_halves"], 6)
    write_csv(sd / "se_vs_b.csv", res["uncertainty"], 6)
    write_csv(sd / "nulls_adversarial.csv", res["adversarial_summary"], 6)
    write_csv(sd / "nulls_adversarial_restarts.csv", adv_rows, 5)
    write_csv(sd / "nulls_simulated.csv", sims, 6)
    write_csv(sd / "nulls_simulated_summary.csv", sim_sum, 6)
    write_csv(sd / "mean_shift.csv", res["fitted_regimes"]["mean_shift"], 6)
    write_csv(
        sd / "physics.csv",
        [
            {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in p.items()}
            for p in res["physics"]
        ],
    )
    plots(res, cfg, data, cal, sd)


if __name__ == "__main__":
    sys.exit(main())
