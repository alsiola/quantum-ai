"""Phase 7 Stage 0 (issue #18): independent audit of the Phase 6 Hankel witness and NLL claims.

Run: uv run python experiments/phase7_audit.py --config experiments/configs/phase7_audit.toml

No QPU of any kind. Reads the committed Phase 6 outputs in ``results/summaries/phase6/``
(read-only) and uses the independent estimator ``qmem.witness`` (numpy). The calibrated and
noiseless 1-qubit instruments are exported once from the committed step circuit and
calibration snapshot with ``--export-instruments`` (needs the ``hardware`` group) into
``results/summaries/phase7_audit/instruments.npz``; everything else runs without it.

Stages: reproduction (witness.csv, marginals, pilot gate, paired NLL), decoding, burn /
window / normalisation checks, uncertainty (bootstrap, jackknife, delta, batches, CIs), the
stationary 2-state null calibration, HMM refits under three shot splits, and a Stage 1 note
on mixtures of per-tenth 2-state fits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# -- helpers -----------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)


def p6(cfg: dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


def summary_dir(cfg: dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def phase6_summary(cfg) -> dict[str, Any]:
    return json.loads((p6(cfg) / "summary.json").read_text())


def shots(cfg, stage: str) -> np.ndarray:
    return W.load_shots(str(p6(cfg) / f"shots_{stage}.npz"))


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


# -- 0. export the instruments (hardware group) --------------------------------------------------


def export_instruments(cfg: dict[str, Any]) -> None:
    """Superoperators of the calibrated / no-idle / noiseless step, checked against Phase 6."""
    from qiskit import qasm3

    from qmem import hardware as hw

    s6 = phase6_summary(cfg)
    prep = s6["prepare"]
    cq = qasm3.loads((p6(cfg) / "step.qasm").read_text())

    def noise(nd: dict[str, Any], idle: bool = True) -> Any:
        kw = {}
        for k, v in nd.items():
            if isinstance(v, list):
                v = {(tuple(a) if isinstance(a, list) else a): b for a, b in v}
            kw[k] = v
        n = hw.StepNoise(**kw)
        if not idle:
            n.t1 = None
        return n

    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)  # the model's "zero" initial memory
    cal = prep["calibration"]["noise"]
    out, checks = {"rho0": rho0}, {}
    main = shots(cfg, "main")
    for name, nz in [
        ("calibrated", noise(cal)),
        ("calibrated_no_idle", noise(cal, idle=False)),
        ("noiseless", None),
    ]:
        inst = hw.noisy_instrument(
            cq, 1, 1, nz, prep["step_memory_in"], memory_qubits_out=prep["step_memory_out"]
        )
        k = inst.kraus.numpy()
        sup = np.stack([sum(np.kron(a.conj(), a) for a in k[y]) for y in range(k.shape[0])])
        out[name] = sup
        p = W.LinearProcess.instrument(sup, rho0)
        pred = s6["analysis"]["predictions"][name]
        checks[name] = {
            "max_abs_marginal_diff": float(np.abs(p.marginals(64) - pred["marginals"]).max()),
            "max_abs_g_diff": float(np.abs(p.windows_at(p.state_after(16), 7) - pred["g"]).max()),
            "device_main_nll": float(p.seq_nll(main).sum() / main.size),
            "phase6_device_main_nll": s6["analysis"]["device"]["main"]["nll"][name],
        }
    sd = summary_dir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    np.savez(sd / "instruments.npz", **out)
    meta = run_metadata({k: v for k, v in cfg.items() if not k.startswith("_")}, None)
    save_json(sd / "instruments_check.json", {"meta": meta, "checks": checks})
    print(json.dumps(checks, indent=2))


def instruments(cfg) -> dict[str, W.LinearProcess]:
    path = summary_dir(cfg) / "instruments.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: run once with --export-instruments (hardware group)")
    z = np.load(path)
    names = ("calibrated", "calibrated_no_idle", "noiseless")
    return {n: W.LinearProcess.instrument(z[n], z["rho0"]) for n in names}


def source_process() -> W.LinearProcess:
    from qmem.tasks import make_task

    t = make_task({"kind": "pmd_renewal", "period": 3, "gamma": 0.5, "v": 0.4})
    return W.LinearProcess.hmm(t.labelled.numpy(), t.initial.numpy())


def phase6_fits(s6) -> dict[int, W.LinearProcess]:
    return {
        f["states"]: W.LinearProcess.hmm(np.array(f["labelled"]), np.array(f["initial"]))
        for f in s6["analysis"]["hmm_fits"]
    }


# -- 1. reproduction -----------------------------------------------------------------------------


def torch_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Phase 6's bootstrap draws (torch.randint in a loop), to reproduce its SEs exactly."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    return np.stack([torch.randint(n, (n,), generator=gen).numpy() for _ in range(n_boot)])


def reproduce(cfg, s6, procs, out: Path) -> dict[str, Any]:
    w = cfg["witness"]
    rows6 = {r["data"]: r for r in csv.DictReader(open(p6(cfg) / "witness.csv"))}
    res: dict[str, Any] = {"witness": [], "marginals": {}, "g": {}}
    for stage in ("main", "pilot"):
        arr = shots(cfg, stage)
        per = W.per_shot_windows(arr, w["k_max"], w["burn"])
        idx = torch_indices(len(per), w["bootstrap"], w["seed"])
        r = W.witness_stats(per, w["offsets"], idx)
        ref = rows6[f"device_{stage}"]
        d6 = s6["analysis"]["device"][stage]
        for j in w["offsets"]:
            row = {"data": f"device_{stage}", "j": j}
            for key, mine, theirs in [
                ("w", r.w[j], float(ref[f"w_j{j}"])),
                ("se", r.se[j], float(ref[f"se_j{j}"])),
                ("z", r.z[j], float(ref[f"z_j{j}"])),
            ]:
                row[f"{key}_audit"] = mine
                row[f"{key}_phase6"] = theirs
                row[f"{key}_rel_diff"] = abs(mine - theirs) / abs(theirs)
            lo, hi = np.quantile(r.boots[j], [0.025, 0.975])
            row["ci95_rel_diff"] = float(
                max(
                    abs(lo - d6["witness"][f"ci95_j{j}"][0]),
                    abs(hi - d6["witness"][f"ci95_j{j}"][1]),
                )
                / abs(r.w[j])
            )
            res["witness"].append(row)
        res["g"][stage] = {
            "max_abs_diff": float(np.abs(r.g - d6["witness"]["g"]).max()),
            "g_se_max_rel_diff": float(
                np.max(
                    np.abs(per.std(0, ddof=1) / math.sqrt(len(per)) - d6["witness"]["g_se"])
                    / np.array(d6["witness"]["g_se"])
                )
            ),
        }
        res["marginals"][stage] = {
            "max_abs_diff": float(np.abs(arr.mean(0) - d6["marginals"]).max()),
            "mean_one_diff": float(arr.mean() - d6["mean_one"]),
        }
    worst = max(max(r["w_rel_diff"], r["se_rel_diff"], r["z_rel_diff"]) for r in res["witness"])
    res["worst_rel_diff"] = worst
    log(out, f"[repro] witness.csv device rows: worst relative difference {worst:.2e}")
    # exact predictions: Phase 6's fixed-state formula vs the estimator's exact expectation
    pred = []
    for name, p in procs.items():
        fixed = p.windows_at(p.state_after(w["burn"]), w["k_max"])
        expect = p.expected_windows(w["k_max"], w["burn"], 64)
        row = {"process": name}
        for j in w["offsets"]:
            row[f"w_fixed_state_j{j}"] = W.witness(fixed, j)
            row[f"w_expected_estimator_j{j}"] = W.witness(expect, j)
            if name in s6["analysis"]["predictions"]:
                row[f"phase6_exact_j{j}"] = s6["analysis"]["predictions"][name][f"w_j{j}"]
        pred.append(row)
    res["predictions"] = pred
    return res


def pilot_gate(cfg, s6, procs) -> dict[str, Any]:
    arr = shots(cfg, "pilot")
    n = len(arr)
    pm = procs["calibrated"].marginals(64)
    z_mean = (arr.mean() - pm.mean()) / math.sqrt(pm.mean() * (1 - pm.mean()) / (n * 64))
    chi2 = float(np.sum((arr.mean(0) - pm) ** 2 / (pm * (1 - pm) / n)))
    d6 = s6["analysis"]["device"]["pilot"]
    return {
        "mean_one": float(arr.mean()),
        "predicted_mean_one": float(pm.mean()),
        "z_mean": float(z_mean),
        "chi2_64dof": chi2,
        "phase6_chi2": d6["marginal_chi2_vs_calibrated"],
        "gate_ok": bool(abs(z_mean) < 6 and chi2 < 64 + 6 * math.sqrt(128)),
    }


def paired(test: np.ndarray, models: dict[str, W.LinearProcess], ref: str = "calibrated"):
    length = test.shape[1]
    per = {k: m.seq_nll(test) / length for k, m in models.items()}
    out = {}
    for k, v in per.items():
        d = v - per[ref]
        out[k] = {"nll": float(v.mean()), "minus_calibrated": mean_se(d)}
    return out


# -- 2. decoding ---------------------------------------------------------------------------------


def decoding(cfg, procs, out: Path) -> dict[str, Any]:
    qasm = (p6(cfg) / "circuit.qasm").read_text()
    decl = [int(t) for t in re.findall(r"^bit\[1\] y(\d+);", qasm, flags=re.M)]
    meas = re.findall(r"^y(\d+)\[0\] = measure \$(\d+);", qasm, flags=re.M)
    with np.load(p6(cfg) / "shots_main.npz") as z:
        packed, length = z["packed"], int(z["length"])
    pm = procs["calibrated"].marginals(length)
    n = len(packed)
    variants = {
        "as_decoded": W.unpack_shots(packed, length),
        "time_reversed": W.unpack_shots(packed, length)[:, ::-1],
        "little_endian_bytes": np.unpackbits(packed, axis=1, bitorder="little")[:, :length],
    }
    chi2 = {}
    for k, arr in variants.items():
        chi2[k] = float(np.sum((arr.mean(0) - pm) ** 2 / (pm * (1 - pm) / n)))
    res = {
        "registers_declared_in_order": decl == list(range(length)),
        "measured_in_step_order": [int(t) for t, _ in meas] == list(range(length)),
        "output_physical_qubits": sorted({int(q) for _, q in meas}),
        "n_measurements": len(meas),
        "marginal_chi2_vs_calibrated": chi2,
        "first_four_device": variants["as_decoded"].mean(0)[:4].tolist(),
        "first_four_calibrated": pm[:4].tolist(),
        "witness_j0_time_reversed": W.witness(
            W.per_shot_windows(variants["time_reversed"], 7, 16).mean(0), 0
        ),
    }
    log(out, f"[decode] chi2 vs calibrated marginals {chi2}")
    return res


# -- 3. windows, burn, normalisation, uncertainty ------------------------------------------------


def windows_and_uncertainty(cfg, procs, out: Path) -> dict[str, Any]:
    w, u = cfg["witness"], cfg["uncertainty"]
    offs = w["offsets"]
    arr = shots(cfg, "main")
    per = W.per_shot_windows(arr, w["k_max"], w["burn"])
    g = per.mean(0)
    res: dict[str, Any] = {"burn": [], "boot_stability": [], "se_methods": [], "ci": []}
    for b in u["burns"]:
        for common in (False, True):
            pb = W.per_shot_windows(arr, w["k_max"], b, common=common)
            idx = W.bootstrap_indices(len(pb), 500, seed=u["ci_seed"] + b)
            r = W.witness_stats(pb, offs, idx)
            row = {"burn": b, "starts": "common" if common else "per_k"}
            for j in offs:
                ex = procs["calibrated"].expected_windows(w["k_max"], b, 64, common=common)
                row |= {f"w_j{j}": r.w[j], f"z_j{j}": r.z[j], f"calibrated_j{j}": W.witness(ex, j)}
            res["burn"].append(row)
    for nb in u["boot_sizes"]:
        for s in u["boot_seeds"]:
            r = W.witness_stats(per, offs, W.bootstrap_indices(len(per), nb, seed=s))
            res["boot_stability"].append(
                {"n_boot": nb, "seed": s, **{f"se_j{j}": r.se[j] for j in offs}}
            )
    big = W.witness_stats(per, offs, W.bootstrap_indices(len(per), u["ci_bootstrap"], u["ci_seed"]))
    jack_g = W.jackknife_g(per)
    rng = np.random.default_rng(u["batch_seed"])
    shuffled = per[rng.permutation(len(per))]
    for j in offs:
        jack = W.witness(jack_g, j)
        se = {
            "bootstrap": big.se[j],
            "jackknife": W.jackknife_se(jack),
            "delta": W.delta_se(per, j),
            "batches_contiguous": W.batch_se(per, j, u["n_batches"]),
            "batches_random": W.batch_se(shuffled, j, u["n_batches"]),
        }
        res["se_methods"].append(
            {"j": j, **se, **{f"ratio_{k}": v / se["bootstrap"] for k, v in se.items()}}
        )
        lo, hi = np.quantile(big.boots[j], [0.025, 0.975])
        blo, bhi = W.bca_ci(big.w[j], big.boots[j], jack)
        am = W.argmax_entry(W.bootstrap_g(per, W.bootstrap_indices(len(per), 1000, 0)), j)
        b = big.boots[j]
        res["ci"].append(
            {
                "j": j,
                "w": big.w[j],
                "se": big.se[j],
                "z": big.z[j],
                "pct_lo": float(lo),
                "pct_hi": float(hi),
                "bca_lo": blo,
                "bca_hi": bhi,
                "boot_skew": float(np.mean((b - b.mean()) ** 3) / b.std() ** 3),
                "boot_bias": float(b.mean() - big.w[j]),
                "max_entry_index": int(W.argmax_entry(g, j)),
                "max_entry_switch_fraction": float(np.mean(am != W.argmax_entry(g, j))),
                "relative_gap_top_two": float(
                    np.diff(np.sort(np.abs(g[j : j + 5]))[-2:])[0] / np.abs(g[j : j + 5]).max()
                ),
            }
        )
    # deliberately wrong: resample windows (symbols) as if independent, g(0) only
    counts = W.window_counts(arr, w["k_max"], w["burn"])
    m = W.n_starts(64, w["k_max"], w["burn"])
    pk = counts.sum(0) / (len(arr) * m)
    res["g_se_shot_bootstrap"] = (
        W.bootstrap_g(per, W.bootstrap_indices(len(per), 1000, 0)).std(0, ddof=1).tolist()
    )
    res["g_se_if_windows_iid"] = np.sqrt(pk * (1 - pk) / (len(arr) * m)).tolist()
    ratios = [
        {k: round(v, 3) for k, v in r.items() if k.startswith("ratio")} for r in res["se_methods"]
    ]
    log(out, f"[uncert] SE ratios {ratios}")
    return res


# -- 4. null calibration ---------------------------------------------------------------------------


def family_hmm(fam: dict[str, Any], s6, rng) -> tuple[np.ndarray, np.ndarray]:
    if fam["kind"] == "random":
        lab, _ = W.random_hmm(2, rng)
        return lab, W.LinearProcess.hmm(lab, np.full(2, 0.5)).stationary().real
    if fam["kind"] == "phase6_hmm2":
        f = next(f for f in s6["analysis"]["hmm_fits"] if f["states"] == 2)
        lab = np.array(f["labelled"])
        if fam["start"] == "fitted":
            return lab, np.array(f["initial"])
        return lab, W.LinearProcess.hmm(lab, np.full(2, 0.5)).stationary().real
    lab, pi = W.two_state_hmm(
        tuple(fam["p_stay"]), tuple(fam["p_one"]), stationary=fam["start"] == "stationary"
    )
    return lab, pi


def null_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    cfg, fam, size, n, reps = task["cfg"], task["family"], task["size"], task["n"], task["reps"]
    w, nu = cfg["witness"], cfg["null"]
    s6 = task["s6"]
    rows = []
    for rep in reps:
        seed = task["seed0"] + rep
        rng = np.random.default_rng(seed)
        lab, pi = family_hmm(fam, s6, rng)
        arr = W.hmm_sample(lab, pi, n, nu["length"], seed=seed + 7_000_000)
        per = W.per_shot_windows(arr, w["k_max"], w["burn"])
        r = W.witness_stats(per, w["offsets"], W.bootstrap_indices(n, nu["bootstrap"], seed + 1))
        p = W.LinearProcess.hmm(lab, pi)
        ex = p.expected_windows(w["k_max"], w["burn"], nu["length"])
        row = {"family": fam["name"], "size": size, "rep": rep, "seed": seed}
        for j in w["offsets"]:
            lo, hi = np.quantile(r.boots[j], [0.025, 0.975])
            row |= {
                f"w_j{j}": r.w[j],
                f"se_j{j}": r.se[j],
                f"z_j{j}": r.z[j],
                f"expected_w_j{j}": W.witness(ex, j),
                f"covers0_j{j}": bool(lo <= 0 <= hi),
            }
        row["max_abs_z"] = r.max_abs_z
        row["p1"] = float(arr.mean())
        rows.append(row)
    return rows


def null_tasks(cfg, s6) -> list[dict[str, Any]]:
    nu = cfg["null"]
    tasks = []
    for fi, fam in enumerate(nu["families"]):
        for si, (size, n) in enumerate(nu["sizes"].items()):
            reps = list(range(nu["reps"][size]))
            seed0 = nu["seed"] + 10_000_000 * fi + 1_000_000 * si
            for c in range(0, len(reps), nu["chunk"]):
                tasks.append(
                    {
                        "kind": "null",
                        "cfg": cfg,
                        "s6": s6,
                        "family": fam,
                        "size": size,
                        "n": n,
                        "reps": reps[c : c + nu["chunk"]],
                        "seed0": seed0,
                    }
                )
    return tasks


def summarise_null(rows: list[dict[str, Any]], offsets: list[int], device_z: dict) -> list:
    out = []
    keys = sorted({(r["family"], r["size"]) for r in rows}, key=str)
    for fam, size in keys:
        rs = [r for r in rows if r["family"] == fam and r["size"] == size]
        n = len(rs)
        row: dict[str, Any] = {"family": fam, "size": size, "reps": n}
        for j in offsets:
            wv = np.array([r[f"w_j{j}"] for r in rs])
            se = np.array([r[f"se_j{j}"] for r in rs])
            z = np.array([r[f"z_j{j}"] for r in rs])
            z = z[np.isfinite(z)]
            ex = np.array([r[f"expected_w_j{j}"] for r in rs])
            sd_z = float(z.std(ddof=1))
            row |= {
                f"mean_z_j{j}": float(z.mean()),
                f"mean_z_mcse_j{j}": sd_z / math.sqrt(len(z)),
                f"sd_z_j{j}": sd_z,
                f"sd_z_mcse_j{j}": sd_z / math.sqrt(2 * (len(z) - 1)),
                f"mean_w_j{j}": float(wv.mean()),
                f"mean_expected_w_j{j}": float(ex.mean()),
                f"bias_over_sd_j{j}": float((wv - ex).mean() / wv.std(ddof=1)),
                f"sd_w_over_mean_se_j{j}": float(wv.std(ddof=1) / se.mean()),
                f"coverage95_j{j}": float(np.mean([r[f"covers0_j{j}"] for r in rs])),
                f"frac_abs_z_gt_1.96_j{j}": float(np.mean(np.abs(z) > 1.96)),
                f"max_abs_z_j{j}": float(np.abs(z).max()),
            }
        mz = np.array([r["max_abs_z"] for r in rs])
        row |= {
            "max4_abs_z_q95": float(np.quantile(mz, 0.95)),
            "max4_abs_z_q99": float(np.quantile(mz, 0.99)),
            "max4_abs_z_max": float(mz.max()),
            "p_max4_ge_device": (1 + int(np.sum(mz >= device_z[size]))) / (n + 1),
        }
        out.append(row)
    return out


# -- 4b. the shot-order episode ----------------------------------------------------------------


def find_episode(per: np.ndarray, grid: int) -> tuple[int, int]:
    """One contiguous segment ``[a, b)`` of shots whose mean of ``g(1) - g(3)`` per shot
    differs most from the rest (two-change-point least squares on a grid of ``grid`` shots)."""
    s = per[:, 1] - per[:, 3]
    n = len(s)
    cut = np.arange(0, n + 1, grid)
    cs = np.concatenate([[0.0], np.cumsum(s)])[cut]
    tot, best = cs[-1], (-np.inf, 0, 0)
    for i in range(len(cut)):
        for k in range(i + 1, len(cut)):
            m = cut[k] - cut[i]
            if m == n:
                continue
            inside = cs[k] - cs[i]
            gain = inside**2 / m + (tot - inside) ** 2 / (n - m) - tot**2 / n
            if gain > best[0]:
                best = (gain, int(cut[i]), int(cut[k]))
    return best[1], best[2]


def episode_analysis(cfg, procs, out: Path) -> dict[str, Any]:
    w, ep = cfg["witness"], cfg["episode"]
    arr = shots(cfg, "main")
    per = W.per_shot_windows(arr, w["k_max"], w["burn"])
    a, b = find_episode(per, ep["grid"])
    inside = np.zeros(len(arr), dtype=bool)
    inside[a:b] = True
    cal_g = procs["calibrated"].expected_windows(w["k_max"], w["burn"], arr.shape[1])
    res: dict[str, Any] = {"start": a, "stop": b, "segments": []}
    for name, mask in [("all", np.ones(len(arr), bool)), ("episode", inside), ("rest", ~inside)]:
        pm = per[mask]
        r = W.witness_stats(pm, w["offsets"], W.bootstrap_indices(len(pm), 1000, ep["seed"]))
        test = arr[mask]
        row = {
            "segment": name,
            "shots": int(mask.sum()),
            "p1": float(test.mean()),
            "g": r.g.tolist(),
            "nll_calibrated": float(procs["calibrated"].seq_nll(test).sum() / test.size),
        }
        for j in w["offsets"]:
            row |= {
                f"w_j{j}": r.w[j],
                f"se_j{j}": r.se[j],
                f"z_j{j}": r.z[j],
                f"over_calibrated_j{j}": r.w[j] / W.witness(cal_g, j),
            }
        res["segments"].append(row)
    # contiguous-batch SE without the episode
    res["batch_se_rest"] = []
    rest = per[~inside]
    idx = W.bootstrap_indices(len(rest), 1000, ep["seed"] + 1)
    rb = W.witness_stats(rest, w["offsets"], idx)
    for j in w["offsets"]:
        res["batch_se_rest"].append(
            {"j": j, "ratio_contiguous": W.batch_se(rest, j, 17) / rb.se[j]}
        )
    # rolling windows for the plot
    win, step = ep["rolling_window"], ep["rolling_step"]
    roll = []
    for s0 in range(0, len(arr) - win + 1, step):
        g = per[s0 : s0 + win].mean(0)
        roll.append(
            {
                "start": s0,
                "p1": float(arr[s0 : s0 + win].mean()),
                **{f"g{k}": float(g[k]) for k in range(len(g))},
                **{f"w_j{j}": W.witness(g, j) for j in w["offsets"]},
            }
        )
    res["rolling"] = roll
    log(
        out,
        f"[episode] shots [{a}, {b}): "
        + "; ".join(
            f"{r['segment']} w0={r['w_j0']:.4f} w1={r['w_j1']:.4f}" for r in res["segments"]
        ),
    )
    return res


# -- 5. HMM refits -------------------------------------------------------------------------------


def split_indices(
    sp: dict[str, Any], n: int, episode: tuple[int, int] = (0, 0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sp["kind"] == "returned":
        a, b = sp["train"], sp["train"] + sp["val"]
        return np.arange(a), np.arange(a, b), np.arange(n // 2, n)
    if sp["kind"] == "random":
        perm = np.random.default_rng(sp["seed"]).permutation(n)
        a, b = sp["train"], sp["train"] + sp["val"]
        return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[n // 2 :])
    if sp["kind"] == "returned_clean_train":  # Phase 6's split with the episode dropped
        a, b = episode
        first = np.array([i for i in range(n // 2) if not a <= i < b])
        n_val = len(first) // 10
        return first[:-n_val], first[-n_val:], np.arange(n // 2, n)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    return even[np.arange(len(even)) % 10 != 9], even[9::10], odd


def em_task(task: dict[str, Any]) -> dict[str, Any]:
    h = task["cfg"]["hmm"]
    arr = task["arr"]
    tr, va, te = (arr[i] for i in task["split_idx"])
    t0 = time.perf_counter()
    if task["patience"] is not None:
        r = W.fit_em(
            task["lab"],
            task["init"],
            tr,
            va,
            max_iters=task["max_iters"],
            patience=task["patience"],
        )
    else:
        r = W.fit_em(task["lab"], task["init"], tr, max_iters=task["max_iters"], tol=h["tol"])
    lab, init = r["labelled"], r["initial"]
    return {
        **{k: task[k] for k in ("split", "states", "init_kind", "restart", "procedure")},
        "iters": r["iters"],
        "seconds": time.perf_counter() - t0,
        "train_nll": float(W.hmm_nll(lab, init, tr).sum() / tr.size),
        "val_nll": float(W.hmm_nll(lab, init, va).sum() / va.size),
        "test_nll": float(W.hmm_nll(lab, init, te).sum() / te.size),
        "labelled": lab.tolist(),
        "initial": init.tolist(),
    }


def em_tasks_wave1(cfg, arr) -> list[dict[str, Any]]:
    h = cfg["hmm"]
    tasks = []
    for sp in h["splits"]:
        idx = split_indices(sp, len(arr), cfg["_episode"])
        tr = arr[idx[0]]
        base = {"cfg": cfg, "arr": arr, "split": sp["name"], "split_idx": idx}
        for d in h["states"]:
            inits = []
            rng = np.random.default_rng([h["seed"], d])
            for r in range(h["random_restarts"]):
                inits.append(("random", r, *W.random_hmm(d, rng)))
            inits.append(("renewal", 0, *W.renewal_init(tr, d)))
            if d & (d - 1) == 0:
                inits.append(
                    (f"markov{int(math.log2(d))}", 0, *W.markov_init(tr, int(math.log2(d))))
                )
            for kind, r, lab, ini in inits:
                tasks.append(
                    base
                    | {
                        "states": d,
                        "init_kind": kind,
                        "restart": r,
                        "lab": lab,
                        "init": ini,
                        "procedure": "converged",
                        "patience": None,
                        "max_iters": h["max_iters"],
                    }
                )
        if sp["kind"] == "returned":  # Phase 6's own initialisations and early stopping
            import torch

            from qmem.baselines.hmm import HMMModel

            for d in h["states"]:
                for r in range(h["phase6_restarts"]):
                    m = HMMModel(d, 2, seed=h["phase6_seed"] + 1000 * d + r)
                    with torch.no_grad():
                        lab, ini = m.labelled().numpy().copy(), m.initial().numpy().copy()
                    tasks.append(
                        base
                        | {
                            "states": d,
                            "init_kind": "phase6_torch",
                            "restart": r,
                            "lab": lab,
                            "init": ini,
                            "procedure": "phase6_early_stop",
                            "patience": h["phase6_patience"],
                            "max_iters": h["phase6_max_iters"],
                        }
                    )
    return tasks


def best_fits(rows, procedure="converged") -> dict[tuple[str, int], dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if r["procedure"] != procedure:
            continue
        k = (r["split"], r["states"])
        if k not in best or r["val_nll"] < best[k]["val_nll"]:
            best[k] = r
    return best


def em_tasks_wave2(cfg, arr, rows) -> list[dict[str, Any]]:
    """Split-state starts from the best smaller fit (per split)."""
    h = cfg["hmm"]
    best = best_fits(rows)
    tasks = []
    for sp in h["splits"]:
        idx = split_indices(sp, len(arr), cfg["_episode"])
        for d in h["states"][1:]:
            smaller = max(s for s in h["states"] if s < d)
            b = best[(sp["name"], smaller)]
            lab, ini = np.array(b["labelled"]), np.array(b["initial"])
            rng = np.random.default_rng([h["seed"], d, 99])
            while lab.shape[1] < d:
                lab, ini = W.split_init(lab, ini, rng)
            tasks.append(
                {
                    "cfg": cfg,
                    "arr": arr,
                    "split": sp["name"],
                    "split_idx": idx,
                    "states": d,
                    "init_kind": f"split_from_{smaller}",
                    "restart": 0,
                    "lab": lab,
                    "init": ini,
                    "procedure": "converged",
                    "patience": None,
                    "max_iters": h["max_iters"],
                }
            )
    return tasks


def nll_table(cfg, arr, procs, rows) -> list[dict[str, Any]]:
    """Held-out paired NLL differences vs the calibrated model, per split and fit."""
    out = []
    for sp in cfg["hmm"]["splits"]:
        _, _, te = split_indices(sp, len(arr), cfg["_episode"])
        test = arr[te]
        models = dict(procs)
        for proc in ("converged", "phase6_early_stop"):
            for (s, d), b in best_fits(rows, proc).items():
                if s == sp["name"]:
                    models[f"hmm{d}_{proc}"] = W.LinearProcess.hmm(
                        np.array(b["labelled"]), np.array(b["initial"])
                    )
        for k, v in paired(test, models).items():
            m, se = v["minus_calibrated"]
            out.append(
                {
                    "split": sp["name"],
                    "test_shots": len(test),
                    "model": k,
                    "test_nll": v["nll"],
                    "minus_calibrated": m,
                    "se": se,
                    "z": m / se if se > 0 else math.nan,
                }
            )
    return out


def mixture_note(cfg, arr, out: Path) -> dict[str, Any]:
    """Stage 1 pointer: w of an equal mixture of 2-state HMMs fitted to each tenth of the shots."""
    w, h = cfg["witness"], cfg["hmm"]
    gs, fits = [], []
    for i, part in enumerate(np.array_split(arr, h["mixture_tenths"])):
        best = None
        for lab, ini in (W.renewal_init(part, 2), W.markov_init(part, 1)):
            r = W.fit_em(lab, ini, part, max_iters=h["mixture_max_iters"], tol=h["tol"])
            if best is None or r["train_nll"] < best["train_nll"]:
                best = r
        p = W.LinearProcess.hmm(best["labelled"], best["initial"])
        g = p.expected_windows(w["k_max"], w["burn"], arr.shape[1])
        gs.append(g)
        fits.append({"tenth": i, "p1": float(part.mean()), "w_j0_component": W.witness(g, 0)})
    gm = np.mean(gs, axis=0)
    res = {"components": fits, **{f"w_mixture_j{j}": W.witness(gm, j) for j in w["offsets"]}}
    log(out, f"[stage1 note] mixture of per-tenth 2-state fits: w(0) = {res['w_mixture_j0']:.2e}")
    return res


# -- plots ----------------------------------------------------------------------------------------


def plots(null_rows, device, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = sorted({r["family"] for r in null_rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.linspace(-4.5, 4.5, 200)
    for ax, size in zip(axes, ("main", "pilot"), strict=True):
        z = np.array([r["z_j0"] for r in null_rows if r["size"] == size])
        z = z[np.isfinite(z)]
        ax.hist(np.clip(z, -4.5, 4.5), bins=60, density=True, color="#8aa1c1", label="null z(0)")
        ax.plot(
            x, np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi), color="#333", lw=1.2, label="N(0,1)"
        )
        ax.set_title(
            f"{size}: {len(z)} null sets, {len(fams)} families; device z(0) = {device[size]:.1f}"
        )
        ax.set_xlabel("z(0) under stationary 2-state HMMs")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(sd / "null_z.png", dpi=130)
    plt.close(fig)


def plot_rolling(ep, procs, cfg, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = cfg["witness"]
    cal = procs["calibrated"].expected_windows(w["k_max"], w["burn"], 64)
    roll = ep["rolling"]
    x = [r["start"] + cfg["episode"]["rolling_window"] / 2 for r in roll]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, lab in [
        (axes[0], "g1", "g(1) = P(1001)... P(101)"),
        (axes[1], "w_j0", "w(0)"),
        (axes[2], "w_j1", "w(1)"),
    ]:
        ax.plot(x, [r[key] for r in roll], color="#3b6ea5", lw=1.2, label="device (rolling)")
        ref = cal[1] if key == "g1" else W.witness(cal, int(key[-1]))
        ax.axhline(ref, color="#c0504d", lw=1, ls="--", label="calibrated model")
        if key != "g1":
            ax.axhline(0, color="#999", lw=0.8)
        ax.axvspan(ep["start"], ep["stop"], color="#f2c14e", alpha=0.25, label="episode")
        ax.set_xlabel("shot index (returned order)")
        ax.set_title(lab)
    axes[0].set_title("g(1) = P(1 0 1)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(sd / "episode.png", dpi=130)
    plt.close(fig)


# -- main ----------------------------------------------------------------------------------------


def run_pool(fn, tasks, workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def dispatch(task: dict[str, Any]) -> Any:
    return null_chunk(task) if task.get("kind") == "null" else em_task(task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export-instruments", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--quick", action="store_true", help="tiny null and EM (smoke test only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export_instruments:
        export_instruments(cfg)
        return
    if args.quick:
        cfg["null"]["reps"] = {"main": 4, "pilot": 8}
        cfg["null"]["chunk"] = 4
        cfg["hmm"]["random_restarts"] = 1
        cfg["hmm"]["phase6_restarts"] = 1
        cfg["hmm"]["max_iters"] = 5
        cfg["hmm"]["phase6_max_iters"] = 5
        cfg["hmm"]["mixture_max_iters"] = 5
        cfg["uncertainty"]["boot_sizes"] = [200]
        cfg["uncertainty"]["ci_bootstrap"] = 300
    workers = args.workers or cfg["experiment"]["workers"]
    out = new_run_dir(cfg["experiment"]["name"])
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    meta = run_metadata(clean, cfg["null"]["seed"])
    meta["quick"] = args.quick
    t0 = time.time()
    try:
        s6 = phase6_summary(cfg)
        inst = instruments(cfg)
        procs = {**inst, "source": source_process()} | {
            f"phase6_hmm{d}_fit": p for d, p in phase6_fits(s6).items()
        }
        res: dict[str, Any] = {"meta": meta}
        res["reproduction"] = reproduce(cfg, s6, procs, out)
        res["pilot_gate"] = pilot_gate(cfg, s6, procs)
        main_arr = shots(cfg, "main")
        test6 = main_arr[len(main_arr) // 2 :]
        repro_models = {k: procs[k] for k in ("calibrated", "noiseless", "source")} | {
            f"hmm{d}_fit": p for d, p in phase6_fits(s6).items()
        }
        res["paired_reproduction"] = {
            "audit": paired(test6, repro_models),
            "phase6": s6["paired"]["minus_calibrated"],
        }
        res["decoding"] = decoding(cfg, procs, out)
        res["windows"] = windows_and_uncertainty(cfg, procs, out)
        res["episode"] = episode_analysis(cfg, procs, out)
        cfg["_episode"] = (res["episode"]["start"], res["episode"]["stop"])
        save_json(out / "partial.json", res)
        # null calibration and EM refits share the pool
        tasks = null_tasks(cfg, s6) + em_tasks_wave1(cfg, main_arr)
        log(out, f"[pool] {len(tasks)} tasks on {workers} workers")
        results = run_pool(dispatch, tasks, workers)
        null_rows = [r for x in results if isinstance(x, list) for r in x]
        em_rows = [x for x in results if isinstance(x, dict)]
        em_rows += run_pool(dispatch, em_tasks_wave2(cfg, main_arr, em_rows), workers)
        log(out, f"[pool] done after {time.time() - t0:.0f} s")
        device_z = {}
        for stage in ("main", "pilot"):
            device_z[stage] = max(
                abs(r["z_audit"])
                for r in res["reproduction"]["witness"]
                if r["data"] == f"device_{stage}"
            )
        res["null"] = summarise_null(null_rows, cfg["witness"]["offsets"], device_z)
        nll = nll_table(cfg, main_arr, {"calibrated": procs["calibrated"]}, em_rows)
        res["nll"] = nll
        res["em_best"] = {
            f"{s}_{d}_{p}": {k: v for k, v in b.items() if k not in ("labelled", "initial")}
            for p in ("converged", "phase6_early_stop")
            for (s, d), b in best_fits(em_rows, p).items()
        }
        res["stage1_mixture_note"] = mixture_note(cfg, main_arr, out)
        res["wall_seconds"] = time.time() - t0
        save_json(out / "audit.json", res)
        write_csv(out / "null_replicates.csv", null_rows)
        write_csv(
            out / "em_fits.csv",
            [{k: v for k, v in r.items() if k not in ("labelled", "initial")} for r in em_rows],
        )
        save_json(out / "em_fits_full.json", {"meta": meta, "fits": em_rows})
        if not args.quick:
            sd = summary_dir(cfg)
            sd.mkdir(parents=True, exist_ok=True)
            res["meta"]["run_dir"] = str(out.relative_to(REPO_ROOT))
            save_json(sd / "summary.json", res)
            write_csv(sd / "null_summary.csv", res["null"])
            keep = ["family", "size", "rep", "seed", "p1", "max_abs_z"]
            keep += [f"{a}_j{j}" for j in cfg["witness"]["offsets"] for a in ("w", "se", "z")]
            write_csv(sd / "null_replicates.csv", [{k: r[k] for k in keep} for r in null_rows])
            write_csv(sd / "nll.csv", nll)
            write_csv(
                sd / "em_fits.csv",
                [{k: v for k, v in r.items() if k not in ("labelled", "initial")} for r in em_rows],
            )
            write_csv(sd / "se_methods.csv", res["windows"]["se_methods"])
            write_csv(sd / "ci.csv", res["windows"]["ci"])
            write_csv(sd / "burn.csv", res["windows"]["burn"])
            write_csv(sd / "boot_stability.csv", res["windows"]["boot_stability"])
            z0 = {
                r["data"].removeprefix("device_"): r["z_audit"]
                for r in res["reproduction"]["witness"]
                if r["j"] == 0
            }
            write_csv(sd / "episode_rolling.csv", res["episode"]["rolling"])
            write_csv(sd / "episode_segments.csv", res["episode"]["segments"])
            plots(null_rows, z0, sd)
            plot_rolling(res["episode"], procs, cfg, sd)
        log(out, f"done in {time.time() - t0:.0f} s -> {out}")
    except BaseException as e:
        log(out, f"FAILED: {e!r}\n{traceback.format_exc()}")
        save_json(out / "failed.json", {"meta": meta, "error": repr(e)})
        raise


if __name__ == "__main__":
    sys.exit(main())
