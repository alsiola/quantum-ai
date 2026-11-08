"""Phase 7 Stage 2B (issue #21): output-ancilla (q59) reset, leakage and residual memory.

Run: uv run python experiments/phase7_ancilla.py --config experiments/configs/phase7_ancilla.toml

No QPU and no IBM service of any kind. The default stage is numpy only (``qmem.ancilla``):

1. fits the ancilla error models of the config to the episode-free device shots (interleaved
   and random halves), with 2-, 3- and 4-state HMMs on the same halves;
2. reports held-out NLL, block KL (L = 4, 8, with a sampling null) and w(0..3) of each model;
3. scans each error model over its plausible range, and Stage 2A's A2 with q75 leakage;
4. predicts the controls B1–B4 under the null and the ancilla hypotheses, with shots and QPU
   seconds, and writes their QASM.

``--export`` (``hardware`` group) checks the control QASM against ``FakeKingston``'s target
and records durations. ``--plan`` prints the unsubmitted Stage 3 job. ``--from-run <dir>``
reuses the cached stages of an earlier run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from qmem import ancilla as AN
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# (control, its matched reference): B1/B2 against the delay-matched B4, B3 and B4 against
# Phase 6's circuit.
PAIRS = (("B1", "B4r"), ("B2", "B4m"), ("B2_post", "B4m"), ("B3", "baseline"), ("B4r", "baseline"))


# -- helpers -----------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def plain(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


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


def sdir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def p6dir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


def snapshot_noise(cfg) -> dict[str, Any]:
    s6 = json.loads((p6dir(cfg) / "summary.json").read_text())
    return s6["prepare"]["calibration"]["noise"]


def base_cfg(cfg) -> AN.StepConfig:
    return AN.config_from_snapshot(snapshot_noise(cfg))


def base_step(cfg) -> list:
    return AN.physical_step((p6dir(cfg) / "step.qasm").read_text())


def shots(cfg) -> np.ndarray:
    return W.load_shots(str(p6dir(cfg) / "shots_main.npz"))


def cached(run: Path, name: str, fn, *a):
    path = run / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = fn(*a)
    path.write_text(json.dumps(res, default=float))
    return res


def model_proc(cfg, values: dict[str, float]) -> W.LinearProcess:
    return AN.ancilla_process(base_step(cfg), AN.apply_params(base_cfg(cfg), values))


def stats(cfg, proc: W.LinearProcess) -> dict[str, Any]:
    w = cfg["witness"]
    g = proc.expected_windows(w["k_max"], w["burn"], 64, common=True)
    out = {f"w{j}": float(W.witness(g, j)) for j in w["offsets"]}
    out["p1"] = float(proc.marginals(64)[w["burn"] :].mean())
    out["g"] = g.tolist()
    return out


# -- 1. fits -----------------------------------------------------------------------------------


def split_sets(cfg) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arr = shots(cfg)
    ep = cfg["episode"]
    idx = AN.clean_indices(len(arr), (ep["start"], ep["stop"]))
    out = {}
    for kind in cfg["splits"]["kinds"]:
        tr, te = AN.split(idx, kind, cfg["splits"]["random_seed"])
        out[kind] = (arr[tr], arr[te])
    return out


def task_fit(task: dict[str, Any]) -> dict[str, Any]:
    cfg, spec_d, kind = task["cfg"], task["spec"], task["split"]
    train, test = split_sets(cfg)[kind]
    spec = AN.FitSpec(spec_d["name"], tuple(spec_d.get("free", [])), dict(spec_d.get("fixed", {})))
    t0 = time.time()
    fit = AN.fit_model(
        spec, base_step(cfg), base_cfg(cfg), train, cfg["fit"]["init"], cfg["fit"]["max_evals"]
    )
    proc = model_proc(cfg, fit["values"])
    ref = model_proc(cfg, {})
    nll_m = proc.seq_nll(test) / 64
    nll_r = ref.seq_nll(test) / 64
    d = nll_m - nll_r
    row = {
        "split": kind,
        "model": spec.name,
        "n_free": len(spec.free),
        "values": fit["values"],
        "se": {k: list(v) for k, v in fit["se"].items()},
        "train_nll": fit["train_nll"],
        "test_nll": float(nll_m.mean()),
        "test_minus_cal": float(d.mean()),
        "test_minus_cal_se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "per_shot_nll": nll_m.tolist(),
        "evals": fit["evals"],
        "seconds": time.time() - t0,
    }
    row.update(stats(cfg, proc))
    row.update(block_stats(cfg, proc, test, task["null_seed"]))
    row.update(twins(cfg, fit["values"], proc))
    return row


def block_stats(cfg, proc: W.LinearProcess, test: np.ndarray, seed: int) -> dict[str, Any]:
    w, nu = cfg["witness"], cfg["null"]
    out: dict[str, Any] = {}
    for L in w["block_sizes"]:
        out[f"kl{L}"] = W.block_kl(test, proc, L, w["burn"])
        nulls = [
            W.block_kl(proc.sample(len(test), 64, seed + r), proc, L, w["burn"])
            for r in range(nu["replicates"])
        ]
        out[f"kl{L}_null"] = float(np.mean(nulls))
        out[f"kl{L}_null_sd"] = float(np.std(nulls, ddof=1))
    return out


def twins(cfg, values: dict[str, float], proc: W.LinearProcess) -> dict[str, Any]:
    """w(0) with q59's memory removed: reset made state independent at the same average,
    and leakage switched off. The difference is the witness q59's memory adds (or removes)."""
    v = dict(values)
    out: dict[str, Any] = {}
    p1 = float(proc.marginals(64)[16:].mean())
    if "reset_e0" in v or "reset_e1" in v:
        eps = base_cfg(cfg).noise[59].reset
        e0, e1 = v.get("reset_e0", eps), v.get("reset_e1", eps)
        tw = dict(v, reset_e0=(1 - p1) * e0 + p1 * e1, reset_e1=(1 - p1) * e0 + p1 * e1)
        out["w0_twin_reset"] = stats(cfg, model_proc(cfg, tw))["w0"]
    if v.get("leak", 0) > 0:
        out["w0_no_leak"] = stats(cfg, model_proc(cfg, dict(v, leak=0.0)))["w0"]
    return out


def task_hmm(task: dict[str, Any]) -> dict[str, Any]:
    cfg, kind, d = task["cfg"], task["split"], task["d"]
    train, test = split_sets(cfg)[kind]
    h = cfg["hmm"]
    rng = np.random.default_rng(h["seed"] + d)
    starts = [W.renewal_init(train, d)]
    if d == 4:
        starts.append(W.markov_init(train, 2))
    if d == 2:
        starts.append(W.markov_init(train, 1))
    starts += [W.random_hmm(d, rng) for _ in range(h["random_starts"])]
    best = None
    for lab, ini in starts:
        f = W.fit_em(lab, ini, train, max_iters=h["max_iters"], tol=h["tol"])
        if best is None or f["train_nll"] < best["train_nll"]:
            best = f
    ref = model_proc(cfg, {})
    nll_m = W.hmm_nll(best["labelled"], best["initial"], test) / 64
    dd = nll_m - ref.seq_nll(test) / 64
    proc = W.LinearProcess.hmm(best["labelled"], best["initial"])
    row = {
        "split": kind,
        "model": f"hmm{d}",
        "n_free": d * (2 * d - 1) + d - 1,
        "train_nll": best["train_nll"],
        "test_nll": float(nll_m.mean()),
        "test_minus_cal": float(dd.mean()),
        "test_minus_cal_se": float(dd.std(ddof=1) / math.sqrt(len(dd))),
        "per_shot_nll": nll_m.tolist(),
    }
    g = proc.expected_windows(cfg["witness"]["k_max"], cfg["witness"]["burn"], 64, common=True)
    row.update({f"w{j}": float(W.witness(g, j)) for j in cfg["witness"]["offsets"]})
    row["p1"] = float(proc.marginals(64)[16:].mean())
    row.update(block_stats(cfg, proc, test, task["null_seed"]))
    return row


def run_fits(cfg, run: Path) -> list[dict[str, Any]]:
    tasks, htasks = [], []
    seed = cfg["null"]["seed"]
    for kind in cfg["splits"]["kinds"]:
        for spec in cfg["fits"]:
            tasks.append(dict(cfg=cfg, spec=spec, split=kind, null_seed=seed))
        for d in cfg["hmm"]["sizes"]:
            htasks.append(dict(cfg=cfg, split=kind, d=d, null_seed=seed))
    with ProcessPoolExecutor(cfg["experiment"]["workers"]) as ex:
        rows = list(ex.map(task_fit, tasks)) + list(ex.map(task_hmm, htasks))
    # paired differences against the best 3- and 4-state HMMs (same test shots)
    for kind in cfg["splits"]["kinds"]:
        hm = {r["model"]: np.array(r["per_shot_nll"]) for r in rows if r["split"] == kind}
        for r in rows:
            if r["split"] != kind:
                continue
            mine = np.array(r["per_shot_nll"])
            for h in ("hmm3", "hmm4"):
                dd = mine - hm[h]
                r[f"test_minus_{h}"] = float(dd.mean())
                r[f"test_minus_{h}_se"] = float(dd.std(ddof=1) / math.sqrt(len(dd)))
    for r in rows:
        r.pop("per_shot_nll", None)
    return rows


def device_reference(cfg) -> dict[str, Any]:
    """Device statistics of the episode-free shots (all, and each test half)."""
    w = cfg["witness"]
    arr = shots(cfg)
    ep = cfg["episode"]
    clean = arr[AN.clean_indices(len(arr), (ep["start"], ep["stop"]))]
    per = W.per_shot_windows(clean, w["k_max"], w["burn"], common=True)
    g = per.mean(0)
    out = {f"w{j}": float(W.witness(g, j)) for j in w["offsets"]}
    out["p1"] = float(clean[:, w["burn"] :].mean())
    out["n_shots"] = len(clean)
    out["g"] = g.tolist()
    return out


# -- 2. scans ----------------------------------------------------------------------------------


def run_scans(cfg) -> list[dict[str, Any]]:
    rows = []
    dev = cfg["episode"]["device_w0_clean"]
    cal = stats(cfg, model_proc(cfg, {}))
    for key, grid in cfg["scan"].items():
        for v in grid:
            vals = {key: float(v)}
            if key in ("leak_cz_phase", "read2"):
                vals["leak"] = 1e-3
                vals["seep"] = 0.038
            if key == "seep":
                vals["leak"] = 1e-3
            if key == "leak":  # leakage always seeps back (T1 of |2> ~ T1 / 2)
                vals["seep"] = 0.038
            s = stats(cfg, model_proc(cfg, vals))
            s.pop("g")
            rows.append(
                {
                    "param": key,
                    "value": v,
                    "others": json.dumps({k: x for k, x in vals.items() if k != key}),
                    **s,
                    "dw0_vs_cal": s["w0"] - cal["w0"],
                    "dw0_frac_device": (s["w0"] - cal["w0"]) / abs(dev),
                    "dp1_vs_cal": s["p1"] - cal["p1"],
                }
            )
    return rows


def run_a2_leak(cfg) -> list[dict[str, Any]]:
    """Stage 2A's A2 (q75 measured every step) with q75 leaking per measurement."""
    a = cfg["a2_leak"]
    step = AN.a2_step(base_step(cfg))
    rows = []
    for phi in a["cz_phase"]:
        for s in a["seep"]:
            for leak in a["leak"]:
                c = replace(
                    base_cfg(cfg),
                    memory_levels=3,
                    memory_leak=leak,
                    memory_seep=s,
                    memory_cz_phase=phi,
                )
                reg = AN.register_for(c, (59,), 2)
                n_meas = sum(1 for i in step if i.name == "measure")
                sup = AN.step_superops(step, reg, c, record=[n_meas - 1])
                proc = AN.to_process(sup, reg)
                st = stats(cfg, proc)
                st.pop("g")
                rows.append({"leak": leak, "seep": s, "cz_phase": phi, **st})
    return rows


# -- 3. controls -------------------------------------------------------------------------------


def hypotheses(cfg, fits: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """null (calibrated), the fitted ancilla models, and the largest plausible residue."""
    pl = cfg["plausible"]
    out = {"null": {}}
    for r in fits:
        if r["split"] == "interleaved" and r["model"] in ("ad_reset_readout", "abd_all"):
            out[f"fit_{r['model']}"] = {k: float(v) for k, v in r["values"].items()}
    out["max_reset"] = {"reset_e0": pl["reset_e0"][1], "reset_e1": pl["reset_e1"][0]}
    out["max_reset_rev"] = {"reset_e0": pl["reset_e0"][0], "reset_e1": pl["reset_e1"][1]}
    out["max_leak"] = {"leak": pl["leak"][1], "seep": pl["seep"][0], "read2": pl["read2"][1]}
    return out


def alt_noise(cfg) -> tuple[AN.SiteNoise, float]:
    """q74 (FakeKingston, stale) for B3: its SiteNoise and the 74-75 CZ depolarising."""
    dev = json.loads((REPO_ROOT / cfg["experiment"]["spectator_dir"] / "device.json").read_text())
    q = dev["qubits"]["74"]
    cz = dev["cz"]["74-75"]["error"]
    site = AN.SiteNoise(2 * q["sx"], q["readout"], q["readout"], q["t1"], q["t2"])
    return site, cz * 4 / 3


def control_seq(cfg, variant: str, values: dict[str, float], noiseless: bool = False):
    """``(init, ops list, final, extra)`` of a control under one hypothesis."""
    base = base_step(cfg)
    c = AN.noiseless_config() if noiseless else base_cfg(cfg)
    if noiseless:
        c = replace(c, measure_time=2.18e-6, reset_time=2.212e-6)
    alt = cfg["controls"]["alt_output"]
    if variant == "B3":
        site, p2 = alt_noise(cfg)
        noise = dict(c.noise)
        noise[alt] = AN.SiteNoise() if noiseless else site
        p2s = dict(c.p2)
        if not noiseless:
            p2s[(75, alt)] = p2
        anc = {59: AN.apply_params(c, values).params(59)}
        anc[alt] = replace(anc[59], read10=None, read01=None)  # same mechanism, own readout
        # ASAP on hardware runs each ancilla's reset during the other's step (FakeKingston:
        # 162 us against 321 us), so q75 no longer idles through a reset: reset_time = 0.
        c = replace(c, noise=noise, p2=p2s, ancilla=anc, relax_idle_ancillas=True, reset_time=0.0)
        reg = AN.register_for(c, (59, alt), 3)
        even = AN.compress_ops(AN.step_superops(base, reg, c), reg)
        odd = AN.compress_ops(AN.step_superops(AN.relabel_output(base, 59, alt), reg, c), reg)
        init, final = AN.init_final(reg)
        return init, [even if t % 2 == 0 else odd for t in range(64)], final, None
    c = AN.apply_params(c, values)
    step = AN.control_step(base, variant, c)
    reg = AN.register_for(c, (59,), 3)
    sup = AN.step_superops(step, reg, c)
    ops = AN.compress_ops(sup, reg)
    init, final = AN.init_final(reg)
    if variant in ("B2", "B2_post"):  # outcomes 2 f + y
        flags = AN.flag_stats(init, ops, final, 64, cfg["witness"]["burn"])
        y_ops = (
            np.stack([ops[0], ops[1]])
            if variant == "B2_post"
            else np.stack([ops[0] + ops[2], ops[1] + ops[3]])
        )
        return init, [y_ops] * 64, final, {"flags": flags, "four": ops}
    return init, [ops] * 64, final, None


def task_control(task: dict[str, Any]) -> dict[str, Any]:
    cfg, v, hname, vals = task["cfg"], task["variant"], task["hyp"], task["values"]
    noiseless = task.get("noiseless", False)
    init, ops, final, extra = control_seq(cfg, v, vals, noiseless)
    sp = AN.SeqProcess(init, ops, final)
    w = cfg["witness"]
    pr = AN.predict_seq(sp, w["k_max"], w["burn"], 4, w["offsets"])
    row = {"variant": v, "hypothesis": hname, "noise": "noiseless" if noiseless else "calibrated"}
    row.update({k: pr[k] for k in pr if k not in ("blocks",)})
    row["blocks4"] = pr["blocks"]
    if extra:
        row.update(extra["flags"])
    if task.get("sd"):
        cc = cfg["controls"]
        if extra is not None:  # B2: sample flags and outputs, post-select on all flags 0
            o = AN.sample_outcomes(
                init, [extra["four"]] * 64, final, cc["sim_shots"], cc["sd_seed"]
            )
            y = (o % 2).astype(np.uint8)
            if v == "B2_post":
                keep = (o // 2)[:, 1:].sum(1) == 0
                y = y[keep]
            sh = y
        else:
            sh = AN.sample_outcomes(init, ops, final, cc["sim_shots"], cc["sd_seed"]).astype(
                np.uint8
            )
        per = W.per_shot_windows(sh, w["k_max"], w["burn"], common=True)
        idx = W.bootstrap_indices(len(sh), cc["bootstrap"], cc["sd_seed"] + 1)
        r = W.witness_stats(per, [0], idx)
        row["sd_w0"] = r.se[0] * math.sqrt(len(sh))
        row["sim_w0"] = r.w[0]
    return row


def run_controls(cfg, hyps) -> list[dict[str, Any]]:
    tasks = []
    for v in cfg["controls"]["variants"]:
        tasks.append(dict(cfg=cfg, variant=v, hyp="null", values={}, noiseless=True))
        for h, vals in hyps.items():
            tasks.append(dict(cfg=cfg, variant=v, hyp=h, values=vals, sd=(h == "null")))
    with ProcessPoolExecutor(cfg["experiment"]["workers"]) as ex:
        return list(ex.map(task_control, tasks))


def durations(cfg) -> dict[str, float]:
    b = cfg["budget"]
    n = snapshot_noise(cfg)
    base = b["phase6_circuit_duration_s"]
    extra = {"B1": n["reset_time"], "B2": n["measure_time"], "B2_post": n["measure_time"]}
    extra.update({"B4r": n["reset_time"], "B4m": n["measure_time"], "baseline": 0.0})
    extra["B3"] = -n["reset_time"]  # resets overlap the other ancilla's step
    return {v: base + 63 * extra[v] for v in extra}


def shots_table(cfg, ctrl: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two-arm 5σ shots per arm for each (control, hypothesis), against the null's change."""
    infl = cfg["controls"]["se_inflation"]
    b = cfg["budget"]
    dur = durations(cfg)
    cal = {(r["variant"], r["hypothesis"]): r for r in ctrl if r["noise"] == "calibrated"}
    sd = {r["variant"]: r["sd_w0"] for r in ctrl if r.get("sd_w0") is not None}
    rows = []
    for v, ref in PAIRS:
        for h in {k[1] for k in cal if k[1] != "null"}:
            r, rr = cal[(v, h)], cal[(ref, h)]
            d_h = r["w0"] - rr["w0"]
            d_0 = cal[(v, "null")]["w0"] - cal[(ref, "null")]["w0"]
            delta = d_h - d_0
            s = infl * math.sqrt(sd[v] ** 2 + sd[ref] ** 2)
            n = (5 * s / delta) ** 2 if delta else math.inf
            n_shots = n / (r.get("accept") or 1.0) if v == "B2_post" else n
            q = b["sub_job_overhead_s"] + n_shots * (2 * b["rep_delay_s"] + dur[v] + dur[ref])
            rows.append(
                {
                    "variant": v,
                    "reference": ref,
                    "hypothesis": h,
                    "test": f"w0({v}) - w0({ref}) vs the null's",
                    "w0_change_hyp": d_h,
                    "w0_change_null": d_0,
                    "delta": delta,
                    "delta_frac_device": delta / abs(cfg["episode"]["device_w0_clean"]),
                    "shots_per_arm_5sigma": n_shots,
                    "qpu_s_both_arms": q,
                }
            )
    # the null itself: does the added idle (B4) change w0 from the baseline?
    for v in ("B4r", "B4m"):
        delta = cal[(v, "null")]["w0"] - cal[("baseline", "null")]["w0"]
        s = infl * math.sqrt(sd[v] ** 2 + sd["baseline"] ** 2)
        n = (5 * s / delta) ** 2
        rows.append(
            {
                "variant": v,
                "reference": "baseline",
                "hypothesis": "null",
                "test": "added idle changes w0 (null prediction)",
                "delta": delta,
                "delta_frac_device": delta / abs(cfg["episode"]["device_w0_clean"]),
                "shots_per_arm_5sigma": n,
                "qpu_s_both_arms": b["sub_job_overhead_s"]
                + n * (2 * b["rep_delay_s"] + dur[v] + dur["baseline"]),
            }
        )
    # B2's flag: rate and its dependence on the previous outcome (one arm)
    for (v, h), r in cal.items():
        if v != "B2" or h == "null":
            continue
        null = cal[("B2", "null")]
        n_flags = 63
        p = null["p_flag"]
        dp = r["p_flag"] - p
        n_rate = (5 / dp) ** 2 * p * (1 - p) / n_flags if dp else math.inf
        dh = r["p_flag_given_y1"] - r["p_flag_given_y0"]
        d0 = null["p_flag_given_y1"] - null["p_flag_given_y0"]
        py1 = r["p_y1"]
        var1 = r["p_flag"] * (1 - r["p_flag"]) * (1 / py1 + 1 / (1 - py1)) / n_flags
        n_hist = (5 / (dh - d0)) ** 2 * var1 if dh != d0 else math.inf
        for test, nn in (("flag rate vs null", n_rate), ("flag | y_prev contrast vs null", n_hist)):
            rows.append(
                {
                    "variant": "B2",
                    "hypothesis": h,
                    "test": test,
                    "delta": dp if "rate" in test else dh - d0,
                    "shots_per_arm_5sigma": nn,
                    "reference": "-",
                    "qpu_s_both_arms": b["sub_job_overhead_s"]
                    + nn * (b["rep_delay_s"] + dur["B2"]),
                }
            )
    return rows


def write_qasm(cfg) -> dict[str, Any]:
    text = (p6dir(cfg) / "circuit.qasm").read_text()
    n = snapshot_noise(cfg)
    out = sdir(cfg) / "qasm"
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for v in ("B1", "B2", "B3", "B4r", "B4m"):
        q = AN.control_qasm(
            text,
            v,
            alt=cfg["controls"]["alt_output"],
            reset_ns=round(n["reset_time"] * 1e9),
            measure_ns=round(n["measure_time"] * 1e9),
        )
        (out / f"circuit_{v}.qasm").write_text(q)
        info[v] = {"sha256": hashlib.sha256(q.encode()).hexdigest(), "lines": q.count("\n")}
    return info


# -- export (hardware group) -------------------------------------------------------------------


def export(cfg) -> None:
    """Check each control's QASM against FakeKingston's target; record durations."""
    import warnings

    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    write_qasm(cfg)
    backend = FakeKingston()
    t = backend.target
    res = {}
    base = qasm3.loads((p6dir(cfg) / "circuit.qasm").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d0 = base.estimate_duration(t, unit="s")
    for v in ("B1", "B2", "B3", "B4r", "B4m"):
        qc = qasm3.loads((sdir(cfg) / "qasm" / f"circuit_{v}.qasm").read_text())
        bad = []
        for inst in qc.data:
            qs = tuple(qc.find_bit(q).index for q in inst.qubits)
            name = inst.operation.name
            if name in ("barrier",):
                continue
            if not t.instruction_supported(name, qs):
                bad.append((name, qs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = qc.estimate_duration(t, unit="s")
        res[v] = {
            "unsupported": bad[:5],
            "n_unsupported": len(bad),
            "duration_fakekingston_s": d,
            "baseline_duration_fakekingston_s": d0,
            "ops": dict(qc.count_ops()),
            "num_clbits": qc.num_clbits,
        }
    res["coupling_74_75"] = t.instruction_supported("cz", (74, 75)) or t.instruction_supported(
        "cz", (75, 74)
    )
    res["meta"] = run_metadata(plain(cfg), cfg["experiment"]["seed"])
    save_json(sdir(cfg) / "circuits.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1, default=str))


# -- main --------------------------------------------------------------------------------------


def plan(cfg) -> None:
    hw = cfg["hardware"]
    b = cfg["budget"]
    dur = durations(cfg)
    total = b["sub_job_overhead_s"]
    print(f"backend {hw['backend']} (submit = {hw['submit']}; this script never submits)")
    for v, n in hw["shots_per_circuit"].items():
        s = n * (b["rep_delay_s"] + dur[v])
        total += s
        print(f"  {v:9s} {n:6d} shots  {dur[v] * 1e6:7.1f} us  {s:6.2f} QPU s")
    print(f"  total {total:.1f} QPU s (one job)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--from-run", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export:
        export(cfg)
        return
    if args.plan:
        plan(cfg)
        return
    run = args.from_run or new_run_dir(cfg["experiment"]["name"])
    log(run, f"run dir {run}")
    meta = run_metadata(plain(cfg), cfg["experiment"]["seed"])
    dev = cached(run, "device", device_reference, cfg)
    log(run, f"device (episode-free): {json.dumps({k: v for k, v in dev.items() if k != 'g'})}")
    fits = cached(run, "fits", run_fits, cfg, run)
    log(run, f"{len(fits)} fits")
    scans = cached(run, "scans", run_scans, cfg)
    a2 = cached(run, "a2_leak", run_a2_leak, cfg)
    log(run, "scans and A2 leakage done")
    hyps = hypotheses(cfg, fits)
    ctrl = cached(run, "controls", run_controls, cfg, hyps)
    log(run, f"{len(ctrl)} control predictions")
    shots_rows = shots_table(cfg, ctrl)
    qasm = write_qasm(cfg)

    out = sdir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    fit_rows = []
    for r in fits:
        row = {k: v for k, v in r.items() if k not in ("values", "se", "g")}
        row["values"] = json.dumps(r.get("values", {}))
        row["ci95"] = json.dumps({k: v[1:] for k, v in r.get("se", {}).items()})
        fit_rows.append(row)
    write_csv(out / "fits.csv", fit_rows)
    write_csv(out / "scans.csv", scans)
    write_csv(out / "a2_leak.csv", a2)
    write_csv(
        out / "controls.csv",
        [{k: v for k, v in r.items() if k not in ("g", "blocks4")} for r in ctrl],
    )
    write_csv(out / "shots.csv", shots_rows)
    # block KL of each control's H vs its null (L = 4)
    nulls = {
        r["variant"]: r for r in ctrl if r["hypothesis"] == "null" and r["noise"] != "noiseless"
    }
    for r in ctrl:
        r["kl4_vs_null"] = AN.kl_blocks(r["blocks4"], nulls[r["variant"]]["blocks4"], 4)
    summary = {
        "meta": meta,
        "device_episode_free": dev,
        "fits": fits,
        "hypotheses": hyps,
        "controls": [{k: v for k, v in r.items() if k != "blocks4"} for r in ctrl],
        "durations_s": durations(cfg),
        "qasm": qasm,
    }
    save_json(out / "summary.json", summary)
    save_json(run / "summary.json", summary)
    shutil.copy(run / "log.txt", out / "log.txt")
    log(run, f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
