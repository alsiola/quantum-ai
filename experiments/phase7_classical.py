"""Phase 7 Stage 2D (issue #23): classical readout/control artefacts and memoryless controls.

Run::

    uv run python experiments/phase7_classical.py \\
        --config experiments/configs/phase7_classical.toml

No QPU and no IBM service of any kind. Reads the committed Phase 6 shots and calibration
snapshot (``results/summaries/phase6/``) and the Stage 0 instruments
(``results/summaries/phase7_audit/instruments.npz``), read-only. ``--export-qasm`` (needs the
``hardware`` group) builds the D1-D3 control circuits against ``FakeKingston`` only and
writes them, padded to Phase 6's cadence, as QASM; it submits nothing.

Stages:

1. control flow of the transpiled Phase 6 circuit (text audit of ``circuit.qasm``);
2. exact w(0..3) under every artefact model and strength, for an i.i.d. source, Phase 6's
   2-state fit and the calibrated PMD instrument; the strength needed to reach
   |w(0)| = 0.0059 or to shift the PMD witness by 10%; a worst case over random 2-state
   sources;
3. end to end on sampled 20,000 x 64 data with planted artefacts (witness pipeline and
   guard rails);
4. the device's own shots: guard rails, and a persistence fit on an interleaved split;
5. D1-D3: noiseless and noisy predictions, null distributions, shots and QPU seconds.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from qmem import artefacts as A
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# -- helpers -------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def r6(v: Any) -> Any:
    if isinstance(v, float) and math.isfinite(v):
        return float(f"{v:.6g}")
    return v


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows([{k: r6(r.get(k)) for k in keys} for r in rows])


def p6(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


def summary_dir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def wcfg(cfg) -> tuple[int, int, int]:
    w = cfg["witness"]
    return w["k_max"], w["burn"], w["length"]


def label(spec: dict[str, Any]) -> str:
    if "label" in spec:
        return spec["label"]
    parts = [spec["kind"]]
    if "mode" in spec:
        parts.append(str(spec["mode"]))
    if "strength" in spec:
        parts.append(f"{spec['strength']:g}")
    return "_".join(parts)


# -- sources -------------------------------------------------------------------------------------


def sources(cfg) -> dict[str, W.LinearProcess]:
    s6 = json.loads((p6(cfg) / "summary.json").read_text())
    f2 = next(f for f in s6["analysis"]["hmm_fits"] if f["states"] == 2)
    z = np.load(REPO_ROOT / cfg["experiment"]["audit_dir"] / "instruments.npz")
    return {
        "iid": A.iid_process(cfg["sources"]["iid_p1"]),
        "hmm2": W.LinearProcess.hmm(np.array(f2["labelled"]), np.array(f2["initial"])),
        "calibrated": W.LinearProcess.instrument(z["calibrated"], z["rho0"]),
    }


def sample_source(name: str, proc: W.LinearProcess, n: int, length: int, seed: int):
    if name == "iid":
        p = float(proc.ops[1, 0, 0])
        return (np.random.default_rng(seed).random((n, length)) < p).astype(np.uint8)
    if np.iscomplexobj(proc.ops):
        return proc.sample(n, length, seed)
    return W.hmm_sample(proc.ops, proc.init, n, length, seed)


def exact_w(g: np.ndarray, offsets) -> dict[str, float]:
    return {f"w{j}": float(W.witness(g, j)) for j in offsets}


# -- 1. control flow -----------------------------------------------------------------------------


def control_flow(cfg) -> dict[str, Any]:
    """What the transpiled Phase 6 circuit does classically (a text audit of the QASM)."""
    text = (p6(cfg) / "circuit.qasm").read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    body = [ln for ln in lines if not ln.startswith(("OPENQASM", "include", "bit[", "qubit"))]
    head = [re.match(r"[A-Za-z_]+", ln).group(0) for ln in body if re.match(r"[A-Za-z_]+", ln)]
    counts: dict[str, int] = {}
    for h in head:
        counts[h] = counts.get(h, 0) + 1
    meas = [ln for ln in body if "= measure" in ln]
    keywords = ("if", "else", "switch", "while", "for", "delay", "barrier", "box", "def")
    found = {k: sum(1 for ln in body if re.match(rf"{k}\b", ln)) for k in keywords}
    # classical assignments other than measure, and reads of a classical bit anywhere
    regs = sorted({m.group(1) for ln in meas if (m := re.match(r"(y\d+)\[0\] = measure", ln))})
    reads = [ln for ln in body if "= measure" not in ln and re.search(r"\by\d+\b", ln)]
    # order: y_t measured on $59, then (t < 63) immediately `reset $59` on the same qubit
    q59 = [ln for ln in body if "$59" in ln]
    pairs_ok, t_seen = True, []
    for i, ln in enumerate(q59):
        m = re.match(r"y(\d+)\[0\] = measure \$59;", ln)
        if m:
            t = int(m.group(1))
            t_seen.append(t)
            if t < 63 and (i + 1 >= len(q59) or q59[i + 1] != "reset $59;"):
                pairs_ok = False
    q75_meas_or_reset = [ln for ln in body if "$75" in ln and ("measure" in ln or "reset" in ln)]
    return {
        "statement_counts": counts,
        "control_flow_keywords": found,
        "no_control_flow": all(v == 0 for v in found.values()),
        "classical_bit_reads_outside_measure": len(reads),
        "registers": len(regs),
        "measured_in_order": t_seen == list(range(64)),
        "measure_then_reset_adjacent_on_59": pairs_ok,
        "q75_measured_or_reset": len(q75_meas_or_reset),
        "resets": counts.get("reset", 0),
        "first_step_has_reset": body[0].startswith("reset"),
    }


# -- 2. exact expectations -----------------------------------------------------------------------


def family_specs(cfg) -> list[tuple[str, str, dict[str, Any]]]:
    """(family, mode label, base spec) for every exact artefact family."""
    e = cfg["exact"]
    out = []
    for m in e["persistence"]["modes"]:
        out.append(("persistence", m, {"kind": "persistence", "mode": m}))
    for m in e["drift"]["modes"]:
        out.append(("drift", m, {"kind": "drift", "mode": m, "levels": e["drift"]["levels"]}))
    for m, par in e["burst"]["modes"]:
        out.append(
            (
                "burst",
                f"{m}({par:g})",
                {"kind": "burst", "mode": m, "param": par, "start": e["burst"]["start"]},
            )
        )
    return out


def exact_grid(cfg, procs, out: Path) -> list[dict[str, Any]]:
    k_max, burn, length = wcfg(cfg)
    n = cfg["exact"]["n_shots"]
    offs = cfg["witness"]["offsets"]
    rows = []
    for src, proc in procs.items():
        base = exact_w(proc.expected_windows(k_max, burn, length, common=True), offs)
        for fam, mode, spec in family_specs(cfg):
            for s in cfg["exact"][fam]["strengths"]:
                g = A.exact_windows(proc, {**spec, "strength": s}, n, k_max, burn, length)
                ww = exact_w(g, offs)
                rows.append(
                    {"source": src, "family": fam, "mode": mode, "strength": s}
                    | ww
                    | {f"dw{j}": ww[f"w{j}"] - base[f"w{j}"] for j in offs}
                )
        log(out, f"[exact] {src} done")
    return rows


def strengths_needed(cfg, procs, out: Path) -> list[dict[str, Any]]:
    k_max, burn, length = wcfg(cfg)
    n = cfg["exact"]["n_shots"]
    ref = cfg["reference"]
    target_abs = round(abs(ref["device_w0_episode_free"]), 4)  # 0.0059
    target_shift = ref["shift_fraction"] * abs(ref["device_w0_episode_free"])
    rng = cfg["exact"]["search"]
    pl = cfg["plausible"]
    rows = []
    for src, proc in procs.items():
        w_base = W.witness(proc.expected_windows(k_max, burn, length, common=True), 0)
        for fam, mode, spec in family_specs(cfg):

            def w0(s: float, spec=spec, proc=proc) -> float:
                g = A.exact_windows(proc, {**spec, "strength": s}, n, k_max, burn, length)
                return float(W.witness(g, 0))

            if src == "calibrated":
                fn, target, what = (lambda s, f=w0, b=w_base: abs(f(s) - b)), target_shift, "shift"
            else:
                fn, target, what = (lambda s, f=w0: abs(f(s))), target_abs, "abs"
            lo, hi = rng[fam]
            s_need, top = A.strength_for(fn, target, lo, hi, n_grid=rng["n_grid"])
            row = {
                "source": src,
                "family": fam,
                "mode": mode,
                "criterion": f"|w0 - w0_base| >= {target:.6f}"
                if what == "shift"
                else f"|w0| >= {target}",
                "w0_base": w_base,
                "strength_needed": s_need,
                "max_on_range": top,
                "range": f"[{lo}, {hi}]",
                "plausible_typical": pl[fam]["typical"],
                "plausible_generous": pl[fam]["generous"],
                "effect_at_typical": fn(pl[fam]["typical"]),
                "effect_at_generous": fn(pl[fam]["generous"]),
            }
            # < 1: the needed strength is beyond the generous plausible value
            row["generous_over_needed"] = pl[fam]["generous"] / s_need if s_need else None
            rows.append(row)
        log(out, f"[strengths] {src} done")
    return rows


def worst_case(cfg, out: Path, fitted_eps: float | None) -> dict[str, Any]:
    """Max |w(0)| over random 2-state sources under plausible-generous artefacts."""
    k_max, burn, length = wcfg(cfg)
    wc = cfg["worstcase"]
    pl = cfg["plausible"]
    n = cfg["exact"]["n_shots"]
    rng = np.random.default_rng(wc["seed"])
    arts = {
        "persistence_sticky_generous": {
            "kind": "persistence",
            "mode": "sticky",
            "strength": pl["persistence"]["generous"],
        },
        "drift_ramp_generous": {
            "kind": "drift",
            "mode": "ramp",
            "strength": pl["drift"]["generous"],
        },
        "drift_step_generous": {
            "kind": "drift",
            "mode": "step",
            "strength": pl["drift"]["generous"],
        },
        "burst_random_generous": {
            "kind": "burst",
            "mode": "random",
            "param": 0.5,
            "strength": pl["burst"]["generous"],
        },
    }
    if fitted_eps:
        arts["persistence_sticky_device_fit"] = {
            "kind": "persistence",
            "mode": "sticky",
            "strength": fitted_eps,
        }
    res = {k: {"all": 0.0, "device_like": 0.0, "argmax_p1_all": None} for k in arts}
    n_like = 0
    for _ in range(wc["n_random"]):
        lab, ini = W.random_hmm(2, rng)
        proc = W.LinearProcess.hmm(lab, ini)
        p1 = float(proc.stationary().real @ lab[1] @ np.ones(2))
        like = abs(p1 - wc["device_p1"]) <= wc["p1_window"]
        n_like += like
        for k, spec in arts.items():
            if spec["kind"] == "drift" and not like:
                continue  # drift is slow (41 levels); evaluate it on device-like sources
            v = abs(W.witness(A.exact_windows(proc, spec, n, k_max, burn, length), 0))
            if v > res[k]["all"] and spec["kind"] != "drift":
                res[k]["all"], res[k]["argmax_p1_all"] = v, p1
            if like:
                res[k]["device_like"] = max(res[k]["device_like"], v)
    for k in arts:
        if arts[k]["kind"] == "drift":
            res[k]["all"] = None
    log(out, f"[worstcase] {wc['n_random']} sources, {n_like} device-like")
    return {"n_random": wc["n_random"], "n_device_like": n_like, "max_abs_w0": res}


# -- 3. sampled end-to-end -----------------------------------------------------------------------


def sampled_tasks(cfg) -> list[dict[str, Any]]:
    s = cfg["sampled"]
    return [
        {"kind": "sampled", "source": src, "spec": spec, "seed": seed, "cfg": cfg}
        for seed in s["seeds"]
        for src in s["sources"]
        for spec in s["specs"]
    ]


def sampled_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    k_max, burn, length = wcfg(cfg)
    wc = cfg["witness"]
    n = cfg["sampled"]["n_shots"]
    procs = sources(cfg)
    proc = procs[task["source"]]
    spec = task["spec"]
    t0 = time.time()
    y = sample_source(task["source"], proc, n, length, task["seed"])
    r = A.plant(y, spec, task["seed"] + 1)
    summ = A.witness_summary(
        r, k_max, burn, wc["offsets"], wc["n_blocks"], wc["bootstrap"], wc["seed"]
    )
    clean = A.witness_summary(
        y, k_max, burn, wc["offsets"], wc["n_blocks"], wc["bootstrap"], wc["seed"]
    )
    gr = A.guard_rails(r, k_max, burn, wc["n_blocks"], wc["seed"], proc.marginals(length))
    ex_spec = spec
    if spec["kind"] == "chain":
        rest = [s for s in spec["steps"] if s["kind"] != "reorder"]
        ex_spec = rest[0] if len(rest) == 1 else {"kind": "unknown"}
    try:
        g_ex = A.exact_windows(proc, ex_spec, len(r), k_max, burn, length)
    except ValueError:
        g_ex = None
    row = {
        "source": task["source"],
        "artefact": label(spec),
        "seed": task["seed"],
        "n": summ["n"],
        "p1": float(r[:, burn:].mean()),
        "lag1_excess": A.lag1_excess_data(r, burn),
        "flags": ";".join(gr["flags"]),
        "duplicates": gr["duplicates"],
        "inflation_w0": gr["order_inflation"]["w0"],
        "inflation_p1": gr["order_inflation"]["p1"],
        "positional_chi2": gr["positional_chi2"]["chi2"],
        "max_abs_g_change": float(np.max(np.abs(np.array(summ["g"]) - np.array(clean["g"])))),
        "seconds": time.time() - t0,
    }
    for j in wc["offsets"]:
        row[f"w{j}"] = summ[f"w{j}"]
        row[f"w{j}_clean"] = clean[f"w{j}"]
        row[f"se_block{j}"] = summ[f"se_block{j}"]
        row[f"se_random{j}"] = summ[f"se_random{j}"]
        row[f"se_boot{j}"] = summ[f"se_boot{j}"]
        row[f"w{j}_exact"] = float(W.witness(g_ex, j)) if g_ex is not None else None
    if g_ex is not None:
        row["z0_vs_exact"] = (row["w0"] - row["w0_exact"]) / row["se_block0"]
    return row


# -- 4. the device's shots -----------------------------------------------------------------------


def device_analysis(cfg, procs, out: Path) -> dict[str, Any]:
    k_max, burn, length = wcfg(cfg)
    wc = cfg["witness"]
    y = W.load_shots(str(p6(cfg) / "shots_main.npz"))
    a, b = cfg["reference"]["episode"]
    clean = np.concatenate([y[:a], y[b:]])
    cal = procs["calibrated"]
    marg = cal.marginals(length)
    res: dict[str, Any] = {}
    for name, arr in (("all", y), ("episode_free", clean)):
        gr = A.guard_rails(arr, k_max, burn, wc["n_blocks"], wc["seed"], marg)
        res[f"guard_rails_{name}"] = gr
        res[f"lag1_excess_{name}"] = A.lag1_excess_data(arr, burn)
    res["lag1_excess_calibrated"] = A.lag1_excess(cal, burn, length)
    # persistence fit: even shots of the episode-free run fit, odd shots test
    ev, od = clean[0::2], clean[1::2]
    base_ev, base_od = cal.seq_nll(ev), cal.seq_nll(od)
    fits, curves = [], []
    for mode in cfg["device"]["fit_modes"]:
        grid = cfg["device"]["fit_grid"]
        tr = []
        for e in grid:
            p = A.with_persistence(cal, A.persistence_kernel(e, mode))
            tr.append(float(p.seq_nll(ev).sum() - base_ev.sum()))
            curves.append({"mode": mode, "eps": e, "train_dnll_total": tr[-1]})
        i = int(np.argmin(tr))
        eps = grid[i]
        if 0 < i < len(grid) - 1:  # parabola through the three points around the minimum
            x, f = np.array(grid[i - 1 : i + 2]), np.array(tr[i - 1 : i + 2])
            c = np.polyfit(x, f, 2)
            if c[0] > 0:
                eps = float(-c[1] / (2 * c[0]))
        p = A.with_persistence(cal, A.persistence_kernel(eps, mode))
        d = p.seq_nll(od) - base_od
        g = p.expected_windows(k_max, burn, length, common=True)
        fits.append(
            {
                "mode": mode,
                "eps_hat": eps,
                "test_dnll_per_symbol": float(d.sum() / od.size),
                "test_dnll_se": float(d.std(ddof=1) * math.sqrt(len(d)) / od.size),
                "lag1_excess_model": A.lag1_excess(p, burn, length),
                **exact_w(g, wc["offsets"]),
            }
        )
    res["persistence_fit"] = fits
    res["persistence_curves"] = curves
    res["calibrated_w"] = exact_w(
        cal.expected_windows(k_max, burn, length, common=True), wc["offsets"]
    )
    res["device_w_episode_free"] = exact_w(
        W.per_shot_windows(clean, k_max, burn, common=True).mean(0), wc["offsets"]
    )
    # approximate upper bound on a threshold drift from the block P(1) dispersion
    gr = res["guard_rails_episode_free"]
    blocks = np.array_split(np.arange(len(clean)), wc["n_blocks"])
    p_shot = clean[:, burn:].mean(1)
    se_b = np.array([p_shot[bb].std(ddof=1) / math.sqrt(len(bb)) for bb in blocks])
    chi2, dof = gr["dispersion_p1"]["chi2"], gr["dispersion_p1"]["dof"]
    lam_crit = max(chi2 - dof, 0.0) + 2 * math.sqrt(2 * dof)
    bounds = {}
    for mode in ("ramp", "step", "sine", "oneside"):

        def lam(amp: float, mode=mode) -> float:
            d = A.drift_profile(len(clean), amp, mode, cfg["exact"]["drift"]["levels"])
            vals = np.unique(d)
            p1v = {
                v: float(
                    A.with_readout(cal, A.threshold_matrix(float(v)))
                    .marginals(length)[burn:]
                    .mean()
                )
                for v in vals
            }
            pn = np.array([p1v[v] for v in d])
            m = np.array([pn[bb].mean() for bb in blocks])
            return float(np.sum((m - m.mean()) ** 2 / se_b**2))

        amp, _ = A.strength_for(lam, lam_crit, 0.0, 0.05, n_grid=26)
        bounds[mode] = amp
    res["drift_bound"] = {"lambda_crit": lam_crit, "amplitude_upper": bounds}
    log(out, "[device] done")
    return res


# -- 5. controls ---------------------------------------------------------------------------------


def calib(cfg) -> dict[str, Any]:
    job = json.loads((p6(cfg) / "job_main.json").read_text())["calibration_after"]

    def q(n: str) -> dict[str, float]:
        d = {p["name"]: p["value"] for p in job["qubits"][n]}
        return {"e01": d["prob_meas1_prep0"], "e10": d["prob_meas0_prep1"], "t1": d["T1"] * 1e-6}

    info = job["info"]
    hw = cfg["hardware"]
    n_steps = cfg["witness"]["length"]
    rot = 2 * 32e-9  # rx = 2 sx (rz virtual)
    pad = (
        hw["phase6_circuit_s"]
        - n_steps * info["measure_time_s"]
        - (n_steps - 1) * info["reset_time_s"]
        - n_steps * rot
    ) / n_steps
    return {
        "q59": q("59"),
        "q75": q("75"),
        "measure_s": info["measure_time_s"],
        "reset_s": info["reset_time_s"],
        "pad_s": pad,
        "step_s": hw["phase6_circuit_s"] / n_steps,
    }


def control_models(cfg, fitted_eps: float | None) -> dict[str, dict[str, Any]]:
    """name -> {"lab", "init", "control", "variant"} for D1-D3 (exact classical HMMs)."""
    c = calib(cfg)
    cc = cfg["controls"]
    idle = c["reset_s"] + c["pad_s"]  # D2/D3 idle in the post-measurement state
    out: dict[str, dict[str, Any]] = {}

    def add(name, ctrl, variant, lab_init, persistence=None):
        out[name] = {"control": ctrl, "variant": variant, "hmm": lab_init, "eps": persistence}

    q59, q75 = c["q59"], c["q75"]
    d59 = 1 - math.exp(-idle / q59["t1"])
    d75 = 1 - math.exp(-idle / q75["t1"])
    p1, fl = cc["d1"]["p1"], cc["d2"]["flip"]
    add("D1_noiseless", "D1", "noiseless", A.qubit_control(p1, True))
    add("D1_calibrated", "D1", "calibrated", A.qubit_control(p1, True, q59["e01"], q59["e10"]))
    add("D2_noiseless", "D2", "noiseless", A.qubit_control(fl, False))
    add(
        "D2_calibrated",
        "D2",
        "calibrated",
        A.qubit_control(fl, False, q59["e01"], q59["e10"], decay=d59),
    )
    seep = 1 - math.exp(-c["step_s"] / (cc["seep_t1_ratio"] * q59["t1"]))
    for leak in cc["leak"]:
        if leak == 0:
            continue
        for r2 in cc["read2"]:
            for ctrl, args in (
                ("D1", (p1, True, q59["e01"], q59["e10"])),
                ("D2", (fl, False, q59["e01"], q59["e10"])),
            ):
                dec = 0.0 if ctrl == "D1" else d59
                add(
                    f"{ctrl}_leak{leak:g}_read2_{r2:g}",
                    ctrl,
                    f"calibrated + leakage {leak:g}/step, |2> read 1 w.p. {r2:g}",
                    A.qubit_control(*args, decay=dec, leak=leak, seep=seep, read2=r2),
                )
    pl = cfg["plausible"]["persistence"]
    eps_list = [("typical", pl["typical"]), ("generous", pl["generous"])]
    if fitted_eps:
        eps_list.append(("device_fit", fitted_eps))
    for tag, e in eps_list:
        for ctrl in ("D1", "D2"):
            out[f"{ctrl}_persistence_{tag}"] = {
                **out[f"{ctrl}_calibrated"],
                "variant": f"calibrated + sticky persistence {e:g} ({tag})",
                "eps": e,
            }
    fa, fb = cc["d3"]["flip_a"], cc["d3"]["flip_b"]
    add(
        "D3_noiseless",
        "D3",
        "noiseless",
        A.product_and(A.qubit_control(fa, False), A.qubit_control(fb, False)),
    )
    add(
        "D3_calibrated",
        "D3",
        "calibrated",
        A.product_and(
            A.qubit_control(fa, False, q59["e01"], q59["e10"], decay=d59),
            A.qubit_control(fb, False, q75["e01"], q75["e10"], decay=d75),
        ),
    )
    return out


def model_proc(m: dict[str, Any]) -> W.LinearProcess:
    lab, ini = m["hmm"]
    p = W.LinearProcess.hmm(lab, ini)
    if m.get("eps"):
        p = A.with_persistence(p, A.persistence_kernel(m["eps"], "sticky"))
    return p


def control_predictions(cfg, models) -> list[dict[str, Any]]:
    k_max, burn, length = wcfg(cfg)
    rows = []
    for name, m in models.items():
        p = model_proc(m)
        g = p.expected_windows(k_max, burn, length, common=True)
        rows.append(
            {
                "model": name,
                "control": m["control"],
                "variant": m["variant"],
                "hidden_states": int(p.ops.shape[1]),
                "p1": float(p.marginals(length)[burn:].mean()),
                "lag1_excess": A.lag1_excess(p, burn, length),
                **exact_w(g, cfg["witness"]["offsets"]),
            }
        )
    return rows


def control_tasks(cfg, fitted_eps) -> list[dict[str, Any]]:
    cc = cfg["controls"]
    tasks = []
    chunk = 25
    for name in ("D1_calibrated", "D2_calibrated"):
        for n in cc["null_sizes"]:
            for r0 in range(0, cc["null_reps"], chunk):
                reps = list(range(r0, min(r0 + chunk, cc["null_reps"])))
                tasks.append({"kind": "control", "model": name, "n": n, "reps": reps, "cfg": cfg})
    n = max(cc["null_sizes"])
    tasks.append(
        {
            "kind": "control",
            "model": "D3_calibrated",
            "n": n,
            "reps": list(range(cc["d3_reps"])),
            "cfg": cfg,
            "fitted_eps": fitted_eps,
        }
    )
    return tasks


def control_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = task["cfg"]
    k_max, burn, length = wcfg(cfg)
    m = control_models(cfg, task.get("fitted_eps"))[task["model"]]
    lab, ini = m["hmm"]
    rows = []
    for rep in task["reps"]:
        seed = cfg["controls"]["null_seed"] + 7919 * rep + task["n"]
        y = W.hmm_sample(lab, ini, task["n"], length, seed)
        per = W.per_shot_windows(y, k_max, burn, common=True)
        g = per.mean(0)
        row = {"model": task["model"], "n": task["n"], "rep": rep, "seed": seed}
        row |= exact_w(g, cfg["witness"]["offsets"])
        row["lag1_excess"] = A.lag1_excess_data(y, burn)
        row["se_block0"] = W.batch_se(per, 0, cfg["witness"]["n_blocks"])
        rows.append(row)
    return rows


def summarise_controls(cfg, preds, reps) -> dict[str, Any]:
    ref = cfg["reference"]
    bound = ref["shift_fraction"] * abs(ref["device_w0_episode_free"])
    hw = cfg["hardware"]
    out: dict[str, Any] = {"bound_w0": bound, "sizes": []}
    for name in ("D1_calibrated", "D2_calibrated"):
        for n in cfg["controls"]["null_sizes"]:
            rs = [r for r in reps if r["model"] == name and r["n"] == n]
            if not rs:
                continue
            w0 = np.array([r["w0"] for r in rs])
            lag = np.array([r["lag1_excess"] for r in rs])
            out["sizes"].append(
                {
                    "model": name,
                    "n": n,
                    "reps": len(rs),
                    "mean_w0": float(w0.mean()),
                    "q95_abs_w0": float(np.quantile(np.abs(w0), 0.95)),
                    "sd_w0": float(w0.std(ddof=1)),
                    "mean_se_block0": float(np.mean([r["se_block0"] for r in rs])),
                    "mean_lag1": float(lag.mean()),
                    "sd_lag1": float(lag.std(ddof=1)),
                    "eps_detectable_5sigma": float(5 * lag.std(ddof=1)),
                    "qpu_s": A.qpu_seconds(
                        n, hw["phase6_circuit_s"], hw["rep_delay_s"], hw["sub_job_overhead_s"]
                    ),
                }
            )
    need = {}
    for name in ("D1_calibrated", "D2_calibrated"):
        ok = [s for s in out["sizes"] if s["model"] == name and s["q95_abs_w0"] < bound]
        need[name] = min((s["n"] for s in ok), default=None)
    out["shots_needed_w0_bound"] = need
    d3 = [r for r in reps if r["model"] == "D3_calibrated"]
    if d3:
        w0 = np.array([r["w0"] for r in d3])
        pred = next(p for p in preds if p["model"] == "D3_calibrated")
        sd = float(w0.std(ddof=1))
        out["d3"] = {
            "n": d3[0]["n"],
            "reps": len(d3),
            "pred_w0": pred["w0"],
            "mean_w0": float(w0.mean()),
            "sd_w0": sd,
            "z_at_n": float(pred["w0"] / sd) if sd > 0 else None,
            "shots_for_5sigma": float(d3[0]["n"] * (5 * sd / abs(pred["w0"])) ** 2),
        }
    out["qpu_s_per_job"] = {
        k: A.qpu_seconds(v, hw["phase6_circuit_s"], hw["rep_delay_s"], hw["sub_job_overhead_s"])
        for k, v in hw["shots"].items()
    }
    return out


# -- hardware export (FakeKingston only; nothing is submitted) ---------------------------------


def export_qasm(cfg) -> None:
    from qiskit import ClassicalRegister, QuantumCircuit, qasm3, transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    backend = FakeKingston()
    tgt = backend.target
    hw = cfg["hardware"]
    cc = cfg["controls"]
    n_steps = cfg["witness"]["length"]
    q_out, q_par = hw["physical_qubits"]["output"], hw["physical_qubits"]["partner"]
    p6c = qasm3.loads((p6(cfg) / "circuit.qasm").read_text())
    t_p6 = p6c.estimate_duration(tgt, unit="s")
    tc = tgt.timing_constraints()
    align = max(tc.granularity, tc.pulse_alignment)
    dt = backend.dt

    def build(kind: str, pad_dt: int) -> QuantumCircuit:
        two = kind == "d3"
        qc = QuantumCircuit(2 if two else 1)
        ya = [ClassicalRegister(1, f"y{t}") for t in range(n_steps)]
        for r in ya:
            qc.add_register(r)
        yb = []
        if two:
            yb = [ClassicalRegister(1, f"z{t}") for t in range(n_steps)]
            for r in yb:
                qc.add_register(r)
        reset_dt = round(tgt["reset"][(q_out,)].duration / dt)
        for t in range(n_steps):
            if kind == "d1":
                if t:
                    qc.reset(0)
                if pad_dt:
                    qc.delay(pad_dt, 0, unit="dt")
                qc.rx(A.rotation_angle(cc["d1"]["p1"]), 0)
                qc.measure(0, ya[t][0])
            else:  # no reset: idle for the reset's duration instead
                wait = pad_dt + (reset_dt if t else 0)
                qs = (0, 1) if two else (0,)
                for q in qs:
                    if wait:
                        qc.delay(wait, q, unit="dt")
                flips = (cc["d3"]["flip_a"], cc["d3"]["flip_b"]) if two else (cc["d2"]["flip"],)
                for q, f in zip(qs, flips, strict=True):
                    qc.rx(A.rotation_angle(f), q)
                qc.measure(0, ya[t][0])
                if two:
                    qc.measure(1, yb[t][0])
        return qc

    layouts = {"d1": [q_out], "d2": [q_out], "d3": [q_out, q_par]}
    info: dict[str, Any] = {
        "backend": backend.name,
        "dt_s": dt,
        "alignment_dt": align,
        "phase6_circuit_s_fake": t_p6,
        "phase6_circuit_s_live": hw["phase6_circuit_s"],
        "controls": {},
    }
    sd = summary_dir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    for kind, lay in layouts.items():
        opts = {"initial_layout": lay, "optimization_level": hw["optimization_level"]}
        opts["seed_transpiler"] = hw["seed_transpiler"]
        t0 = transpile(build(kind, 0), backend, **opts).estimate_duration(tgt, unit="s")
        pad = max(0, math.floor((t_p6 - t0) / n_steps / dt / align) * align)
        tq = transpile(build(kind, pad), backend, **opts)
        dur = tq.estimate_duration(tgt, unit="s")
        ops = {k: int(v) for k, v in tq.count_ops().items()}
        text = qasm3.dumps(tq)
        (sd / f"{kind}.qasm").write_text(text)
        info["controls"][kind] = {
            "physical_qubits": lay,
            "pad_dt_per_step": pad,
            "pad_s_per_step": pad * dt,
            "duration_s_fake": dur,
            "duration_unpadded_s_fake": t0,
            "duration_minus_phase6_s_fake": dur - t_p6,
            "ops": ops,
            "control_flow": any(
                k in ops for k in ("if_else", "switch_case", "while_loop", "for_loop")
            ),
            "qpu_s": A.qpu_seconds(
                hw["shots"][kind],
                hw["phase6_circuit_s"] + (dur - t_p6),
                hw["rep_delay_s"],
                hw["sub_job_overhead_s"],
            ),
        }
        print(kind, json.dumps(info["controls"][kind]))
    meta = run_metadata(
        {k: v for k, v in cfg.items() if not k.startswith("_")}, hw["seed_transpiler"]
    )
    save_json(sd / "controls_circuits.json", {"meta": meta, **info})


# -- pool ----------------------------------------------------------------------------------------


def task_key(t: dict[str, Any]) -> str:
    if t["kind"] == "sampled":
        return f"sampled_{t['source']}_{label(t['spec'])}_{t['seed']}"
    return f"control_{t['model']}_{t['n']}_{t['reps'][0]}"


def dispatch(task: dict[str, Any]) -> Any:
    return sampled_task(task) if task["kind"] == "sampled" else control_task(task)


def run_pool(tasks, workers: int, cache: Path, out: Path) -> list[Any]:
    """Run ``tasks``; every result is checkpointed in ``cache`` and reused on resume."""
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
        if len(results) % 10 == 0:
            log(out, f"[pool] {len(results)}/{len(tasks)}")

    if workers <= 1:
        for i in todo:
            keep(i, dispatch(tasks[i]))
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(dispatch, tasks[i]): i for i in todo}
            for fu in as_completed(futs):
                keep(futs[fu], fu.result())
    return [results[i] for i in range(len(tasks))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export-qasm", action="store_true", help="FakeKingston only (hardware group)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--quick", action="store_true", help="tiny sizes (smoke test only)")
    ap.add_argument("--from-run", type=Path, help="resume: reuse this run dir's task results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export_qasm:
        export_qasm(cfg)
        return
    if args.quick:
        cfg["sampled"]["n_shots"] = 2000
        cfg["sampled"]["specs"] = cfg["sampled"]["specs"][:3]
        cfg["controls"]["null_sizes"] = [500, 1000]
        cfg["controls"]["null_reps"] = 4
        cfg["controls"]["d3_reps"] = 3
        cfg["worstcase"]["n_random"] = 20
        cfg["exact"]["search"]["n_grid"] = 6
    workers = min(args.workers or cfg["experiment"]["workers"], 2)  # shared machine: <= 2
    out = args.from_run.resolve() if args.from_run else new_run_dir(cfg["experiment"]["name"])
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    meta = run_metadata(clean_cfg, cfg["witness"]["seed"])
    meta["quick"] = args.quick
    t0 = time.time()
    try:
        procs = sources(cfg)
        res: dict[str, Any] = {"meta": meta}
        res["control_flow"] = control_flow(cfg)
        log(out, f"[control flow] {res['control_flow']}")
        res["device"] = device_analysis(cfg, procs, out)
        fit = min(res["device"]["persistence_fit"], key=lambda f: f["test_dnll_per_symbol"])
        fitted_eps = (
            fit["eps_hat"]
            if fit["mode"] == "sticky"
            else next(
                f["eps_hat"] for f in res["device"]["persistence_fit"] if f["mode"] == "sticky"
            )
        )
        res["fitted_sticky_eps"] = fitted_eps
        save_json(out / "partial.json", res)
        exact_rows = exact_grid(cfg, procs, out)
        need_rows = strengths_needed(cfg, procs, out)
        res["worst_case"] = worst_case(cfg, out, fitted_eps)
        save_json(out / "partial.json", res)
        tasks = sampled_tasks(cfg) + control_tasks(cfg, fitted_eps)
        results = run_pool(tasks, workers, out / "tasks", out)
        sampled_rows = [r for r, t in zip(results, tasks, strict=True) if t["kind"] == "sampled"]
        ctrl_reps = [
            x for r, t in zip(results, tasks, strict=True) if t["kind"] == "control" for x in r
        ]
        models = control_models(cfg, fitted_eps)
        preds = control_predictions(cfg, models)
        res["controls"] = summarise_controls(cfg, preds, ctrl_reps)
        res["controls"]["calibration"] = calib(cfg)
        cc_path = summary_dir(cfg) / "controls_circuits.json"
        if cc_path.exists():
            res["controls"]["circuits"] = {
                k: v for k, v in json.loads(cc_path.read_text()).items() if k != "meta"
            }
        res["wall_seconds"] = time.time() - t0
        save_json(out / "classical.json", res)
        write_csv(out / "exact.csv", exact_rows)
        write_csv(out / "strengths.csv", need_rows)
        write_csv(out / "sampled.csv", sampled_rows)
        write_csv(out / "control_predictions.csv", preds)
        write_csv(out / "control_replicates.csv", ctrl_reps)
        if not args.quick:
            sd = summary_dir(cfg)
            sd.mkdir(parents=True, exist_ok=True)
            res["meta"]["run_dir"] = str(out.relative_to(REPO_ROOT))
            small = json.loads(json.dumps(res, default=str))
            for k in ("guard_rails_all", "guard_rails_episode_free"):
                for b in ("block_p1", "block_repeat", "block_g1_minus_g3"):
                    small["device"][k][b] = [r6(v) for v in small["device"][k][b]]
            save_json(sd / "summary.json", small)
            write_csv(sd / "exact.csv", exact_rows)
            write_csv(sd / "strengths.csv", need_rows)
            write_csv(sd / "sampled.csv", sampled_rows)
            write_csv(sd / "control_predictions.csv", preds)
            write_csv(sd / "control_null.csv", res["controls"]["sizes"])
            write_csv(sd / "device_persistence_fit.csv", res["device"]["persistence_fit"])
            write_csv(sd / "device_persistence_curves.csv", res["device"]["persistence_curves"])
            plot_exact(exact_rows, cfg, sd)
        log(out, f"done in {time.time() - t0:.0f} s -> {out}")
    except BaseException as e:
        log(out, f"FAILED: {e!r}\n{traceback.format_exc()}")
        save_json(out / "failed.json", {"meta": meta, "error": repr(e)})
        raise


def plot_exact(rows, cfg, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = cfg["reference"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    fams = [
        ("persistence", "sticky"),
        ("drift", "ramp"),
        ("drift", "step"),
        ("burst", "random(0.5)"),
    ]
    for ax, src in zip(axes, ("iid", "hmm2", "calibrated"), strict=True):
        for fam, mode in fams:
            rs = [
                r for r in rows if r["source"] == src and r["family"] == fam and r["mode"] == mode
            ]
            xs = [r["strength"] for r in rs if r["strength"] > 0]
            ys = [abs(r["w0"]) for r in rs if r["strength"] > 0]
            ax.loglog(xs, np.maximum(ys, 1e-20), marker="o", ms=3, label=f"{fam} {mode}")
        ax.axhline(abs(ref["device_w0_episode_free"]), color="k", ls="--", lw=1)
        if src == "calibrated":
            ax.axhline(0.9 * abs(ref["device_w0_episode_free"]), color="k", ls=":", lw=1)
        ax.set_title(
            {
                "iid": "i.i.d. P(1) = 0.58",
                "hmm2": "Phase 6 2-state fit",
                "calibrated": "calibrated PMD",
            }[src]
        )
        ax.set_xlabel("artefact strength")
        ax.set_ylim(1e-18, 3e-2)
    axes[0].set_ylabel("|w(0)| (exact)")
    axes[0].legend(fontsize=7)
    fig.savefig(sd / "exact_w0.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
