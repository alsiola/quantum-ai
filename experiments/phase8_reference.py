"""Phase 8 Stage 3 (issue #33): feasibility of certifying q75's step channel as non-EB.

Run::

    uv run python experiments/phase8_reference.py --config experiments/configs/phase8_reference.toml
    uv run --group hardware python experiments/phase8_reference.py \\
        --config experiments/configs/phase8_reference.toml --export-qasm   # once

No QPU and no IBM service of any kind. Reads, read-only: Phase 6's trained step
(``results/summaries/phase6/step.qasm``), the Stage 4c pieces and calibration snapshot
(``results/summaries/phase7_hw_4c/``), Phase 6's noiseless and snapshot pieces
(``results/summaries/phase7_ablation/pieces.npz``) and FakeKingston's committed map and
calibration (``results/summaries/phase7_spectator/device.json``).

Stages:

1. the step channel's Choi state against n: fixed-target and maximal Bell fidelity,
   negativity (PPT), for every noise model;
2. the reference qubit: every candidate within 3 hops of q75, the predeclared rule, the pick;
3. the protocols (prepare-and-measure, Bell route via the reference, the MDI signalling game,
   a PPT witness at n = 2) with SPAM and leakage-as-loss thresholds;
4. q59-record branches (secondary, Holm);
5. power (normal and Monte Carlo) and shots for 80% / 90%;
6. budget with ``qmem.followup``'s cost model (max(charged, upper bound) accounting) and an
   unsubmitted, capped plan;
7. ``--export-qasm``: every circuit built against FakeKingston, checked against the numpy model
   (noiseless, unconditional), written as QASM with SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from qmem import ablate_hw as A
from qmem import followup as F
from qmem import reference as R
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json


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
    if isinstance(v, (float, np.floating)) and math.isfinite(v):
        return float(f"{float(v):.6g}")
    return v


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows([{k: r6(r.get(k)) for k in keys} for r in rows])


def cfg_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def mat(u: np.ndarray) -> dict[str, list]:
    return {"re": np.real(u).round(12).tolist(), "im": np.imag(u).round(12).tolist()}


# -- inputs ----------------------------------------------------------------------------------------


def inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    ex = cfg["experiment"]
    live = A.StepPieces.from_npz(np.load(REPO_ROOT / ex["pieces"]), ex["pieces_key"])
    z6 = np.load(REPO_ROOT / ex["phase6_pieces"])
    calib = json.loads((REPO_ROOT / ex["calibration"]).read_text())["snapshot"]
    device = json.loads((REPO_ROOT / ex["device"]).read_text())
    return {
        "live": live,
        "noiseless": A.StepPieces.from_npz(z6, "noiseless"),
        "phase6": A.StepPieces.from_npz(z6, "calibrated"),
        "calib": calib,
        "device": device,
    }


def q75_spam(cfg: dict[str, Any], calib: dict[str, Any]) -> R.Spam:
    q = {p["name"]: p["value"] for p in calib["qubits"][str(cfg["layout"]["memory"])]}
    sx = next(
        p["value"]
        for g in calib["gates"]
        if g["name"] == f"sx{cfg['layout']['memory']}"
        for p in g["parameters"]
        if p["name"] == "gate_error"
    )
    p_gate = 2 * sx * cfg["spam"]["gates_per_prep"]  # depolarising p = 2 r per gate
    return R.Spam(q["init_error"], p_gate, p_gate, q["prob_meas0_prep1"], q["prob_meas1_prep0"])


def device_spam(device: dict[str, Any], q: int, gates: int) -> R.Spam:
    c = device["qubits"][str(q)]
    p_gate = 2 * c["sx"] * gates
    return R.Spam(0.0, p_gate, p_gate, c["readout"], c["readout"])


def mdi_reset(
    cfg: dict[str, Any], calib: dict[str, Any], device: dict[str, Any]
) -> R.MdiReset | None:
    """Amendment A4 (#38): q74's reset before psi_y, or None (#33's circuits) if ``[mdi] reset``
    is off. q75 idles for q74's reset time (FakeKingston's map) at the Stage 4c snapshot's T1
    and T2; q74 is left excited with probability its readout error (the reset is a measurement
    and a conditional X)."""
    m = cfg.get("mdi", {})
    if not m.get("reset"):
        return None
    q = {p["name"]: p["value"] for p in calib["qubits"][str(cfg["layout"]["memory"])]}
    c = device["qubits"][str(cfg["layout"]["bridge"])]
    return R.MdiReset(c["reset_time"], q["T1"] * 1e-6, q["T2"] * 1e-6, c["readout"])


# -- models ----------------------------------------------------------------------------------------


def deficit_dephasing(cfg: dict[str, Any], live: A.StepPieces) -> tuple[float, float]:
    """Extra per-step q75 dephasing that scales the baseline w(0) to ``deficit_ratio`` of the
    live model's (bisection)."""
    w = cfg["witness"]
    s = A.Settings(length=w["length"], burn=w["burn"], k_max=w["k_max"])

    def w0(p: float) -> float:
        proc = W.LinearProcess.instrument(R.step_ops(live, extra_dephasing=p), A.RHO0_JOINT)
        return A.predict(proc, s)["w0"]

    ref = w0(0.0)
    target = cfg["sensitivity"]["deficit_ratio"]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if w0(mid) / ref > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, ref


def models(cfg: dict[str, Any], inp: dict[str, Any]) -> tuple[dict, dict, float]:
    """name -> step branches, and name -> the same model's A2 step (the EB negative control)."""
    live = inp["live"]
    sens = cfg["sensitivity"]
    p_def, _ = deficit_dephasing(cfg, live)
    spec: dict[str, tuple[A.StepPieces, A.Hypothesis | None, float]] = {
        "stage4c_live": (live, None, 0.0),
        "noiseless": (inp["noiseless"], None, 0.0),
        "phase6_snapshot": (inp["phase6"], None, 0.0),
        "live_q59_reset_fit": (live, A.Hypothesis(q59_reset=tuple(sens["q59_reset"])), 0.0),
    }
    for ph in sens["q75_phase"]:
        spec[f"live_phase_{ph:g}"] = (live, A.Hypothesis(q75_phase=ph), 0.0)
    for p in sens["extra_dephasing"]:
        if p:
            spec[f"live_dephase_{p:g}"] = (live, None, p)
    spec["live_deficit_matched"] = (live, None, p_def)
    steps = {k: R.step_ops(pc, "baseline", h, d) for k, (pc, h, d) in spec.items()}
    neg = {k: R.step_ops(pc, "A2", h, d) for k, (pc, h, d) in spec.items()}
    return steps, neg, p_def


# -- protocols -------------------------------------------------------------------------------------


def ref_noise(cfg, device, cand: dict[str, Any], spam_mem: R.Spam) -> R.RefNoise:
    p = cfg["protocol"]
    return R.RefNoise(
        prep_infidelity=cand["cz_error_sum"],
        ref_t1=cand["t1_s"],
        ref_t2=cand["t2_s"],
        step_s=p["step_s"],
        ref_readout=cand["readout"],
        mem_readout=(spam_mem.p01 + spam_mem.p10) / 2,
        extra_ref_idle_s=p["end_s"],
    )


def leak(cfg, n: int, extra_cz: int = 0) -> float:
    lk = cfg["leakage"]
    return R.leakage_bound(n, lk["cz_per_step"], lk["leak_per_cz"], extra_cz)


def evaluate(
    cfg,
    steps,
    n,
    v,
    spam,
    rnoise,
    spam_y,
    bsm_depol,
    branch=None,
    first=None,
    bell_cz=0,
    reset: R.MdiReset | None = None,
):
    """Every route's prediction for one model, n and (optional) branch."""
    sup = R.n_step_map(steps, n, branch, first)
    j = R.choi(sup)
    prob = float(np.trace(j).real)
    pm = R.pm_probabilities(sup, v, spam)
    row = {
        "branch_prob": prob,
        "fe_fixed": R.bell_fidelity(j, v),
        "fe_max": R.max_bell_fidelity(j),
        "negativity": R.negativity(j),
        "pm_score": R.pm_score(pm),
        "pm_sd1": R.pm_sd1(pm) if branch is None else R.ratio_sd1(pm),
        "pm_threshold": R.pm_threshold(leak(cfg, n)),
    }
    if rnoise is not None:
        par = R.bell_parities(R.bell_final_state(sup, n, rnoise), v, rnoise)
        row |= {
            "bell_F": par["fidelity"],
            "bell_sd1": R.bell_sd1(par),
            "bell_threshold": R.bell_threshold(leak(cfg, n, bell_cz)),
        }
        row["_bell_par"] = par
    if spam_y is not None and branch is None:
        om = R.mdi_weights(v)
        pq = R.mdi_probabilities(sup, spam, spam_y, bsm_depol, reset)
        row |= {
            "mdi_payoff": float((om * pq).sum()),
            "mdi_sd1": R.mdi_sd1(om, pq),
            "mdi_threshold": 0.0,
        }
    row["_pm"] = pm
    return row


# -- main analysis ---------------------------------------------------------------------------------


def analyse(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    sd.mkdir(parents=True, exist_ok=True)
    inp = inputs(cfg)
    lay, prot, an = cfg["layout"], cfg["protocol"], cfg["analysis"]
    spam = q75_spam(cfg, inp["calib"])
    ms, ms_neg, p_def = models(cfg, inp)
    primary = ms["stage4c_live"]
    all_n = sorted(set(prot["steps"]) | set(prot["descriptive_steps"]))
    log(out, f"q75 SPAM {spam}; deficit-matched dephasing p = {p_def:.4f}")

    # the predeclared targets: V_n from the primary (Stage 4c) model's unconditional Choi state
    vs = {n: R.optimal_correction(R.choi(R.n_step_map(primary, n))) for n in all_n}

    # -- reference candidates and the pick
    rr = cfg["reference_rule"]
    rule = R.RefRule(
        rr["min_dist_memory"],
        rr["min_dist_output"],
        rr["max_dist_memory"],
        rr["max_readout"],
        rr["min_t2_us"] * 1e-6,
        rr["max_cz_error"],
        (lay["output"],),
    )
    cands = R.reference_candidates(inp["device"], lay["memory"], lay["output"], rule)
    sup1 = R.n_step_map(primary, prot["primary_n"])
    for c in cands:
        rn = ref_noise(cfg, inp["device"], c, spam)
        par = R.bell_parities(
            R.bell_final_state(sup1, prot["primary_n"], rn), vs[prot["primary_n"]], rn
        )
        c["bell_F_n1"] = par["fidelity"]
    admitted = [c for c in cands if c["admitted"]]
    pick = (
        max(admitted, key=lambda c: (round(c["bell_F_n1"], 6), -c["swaps"], -c["qubit"]))
        if admitted
        else None
    )
    write_csv(sd / "references.csv", cands)
    log(
        out,
        f"reference candidates {len(cands)}, admitted {len(admitted)}; pick "
        f"{pick['qubit'] if pick else None}",
    )
    rnoise = ref_noise(cfg, inp["device"], pick, spam) if pick else None
    bcz = pick["cz_count"] if pick else 0
    bridge = next(c for c in cands if c["qubit"] == lay["bridge"])
    rnoise_bridge = ref_noise(cfg, inp["device"], bridge, spam)
    spam_y = device_spam(inp["device"], lay["bridge"], cfg["spam"]["gates_per_prep"])
    bsm_depol = 4 / 3 * float(inp["device"]["cz"][cfg["spam"]["bsm_cz_error_from"]]["error"])
    rs = mdi_reset(cfg, inp["calib"], inp["device"])
    log(out, f"MDI q74 reset (A4): {rs}")

    # -- 1. the channel against n, every model
    chan_rows, keep = [], {}
    for name, steps in ms.items():
        for n in all_n:
            row = evaluate(
                cfg, steps, n, vs[n], spam, rnoise, spam_y, bsm_depol, bell_cz=bcz, reset=rs
            )
            keep[(name, n)] = row
            chan_rows.append(
                {"model": name, "n": n, **{k: v for k, v in row.items() if not k.startswith("_")}}
            )
        # the EB negative control (q75 measured in step 1) and the bridge-as-reference variant
        neg = evaluate(
            cfg, steps, 1, vs[1], spam, rnoise, None, 0.0, first=ms_neg[name], bell_cz=bcz
        )
        keep[(name, "neg")] = neg
        chan_rows.append(
            {"model": name, "n": "neg_1", **{k: v for k, v in neg.items() if not k.startswith("_")}}
        )
    write_csv(sd / "channel.csv", chan_rows)

    # q74 directly as the reference (excluded by the rule): what isolation costs
    par74 = R.bell_parities(R.bell_final_state(sup1, 1, rnoise_bridge), vs[1], rnoise_bridge)

    # -- 2. branches (primary model; targets V_n of the unconditional channel, same circuits)
    br_rows = []
    for n in an["branch_steps"]:
        for rec in itertools.product((0, 1), repeat=n):
            row = evaluate(
                cfg, primary, n, vs[n], spam, rnoise, None, 0.0, branch=list(rec), bell_cz=bcz
            )
            br_rows.append(
                {
                    "n": n,
                    "record": "".join(map(str, rec)),
                    **{k: v for k, v in row.items() if not k.startswith("_")},
                    "_pm": row["_pm"],
                }
            )

    # -- 3. power and shots
    alpha, draws, seed = an["alpha"], an["mc_draws"], an["seed"]
    m_sec = len(prot["secondary_n"]) + sum(2**n for n in an["branch_steps"])
    alpha_sec = an["secondary_alpha"] / m_sec  # Holm's first step (conservative for power)
    plan = cfg["plan"]
    s_pm = plan["pm_shots"][f"pm_n{prot['primary_n']}"]
    s_bell = plan["bell_shots"][f"bell_n{prot['primary_n']}"]
    pow_rows = []
    for name in ms:
        for n in all_n:
            row = keep[(name, n)]
            a = alpha if n == prot["primary_n"] else alpha_sec
            for route, eff, sd1, shots in (
                (
                    "pm",
                    row["pm_score"] - row["pm_threshold"],
                    row["pm_sd1"],
                    12 * plan["pm_shots"].get(f"pm_n{n}", 0),
                ),
                (
                    "bell",
                    row.get("bell_F", math.nan) - row.get("bell_threshold", math.nan),
                    row.get("bell_sd1", math.nan),
                    6 * plan["bell_shots"].get(f"bell_n{n}", 0),
                ),
                (
                    "mdi",
                    row["mdi_payoff"],
                    row["mdi_sd1"],
                    16 * plan["mdi_shots"].get(f"mdi_n{n}", 0),
                ),
            ):
                pr = {
                    "model": name,
                    "n": n,
                    "route": route,
                    "alpha": a,
                    "effect": eff,
                    "sd1": sd1,
                    "planned_shots": shots,
                    "power_planned": R.power_normal(eff, sd1, shots, a) if eff > 0 else 0.0,
                }
                for pt in an["power_targets"]:
                    pr[f"shots_{int(pt * 100)}"] = R.shots_for_power(eff, sd1, a, pt)
                pow_rows.append(pr)
    # Monte Carlo checks of the primary claims at the planned shots
    mc = {}
    for name in ("stage4c_live", "live_deficit_matched"):
        row = keep[(name, prot["primary_n"])]
        mc[f"pm_n1_{name}"] = R.mc_power_pm(
            row["_pm"], s_pm, row["pm_threshold"], alpha, draws, seed
        )
        mc[f"bell_n1_{name}"] = R.mc_power_bell(
            row["_bell_par"], 2 * s_bell, row["bell_threshold"], alpha, draws, seed + 1
        )
        mc[f"pm_neg_false_pass_{name}"] = R.mc_power_pm(
            keep[(name, "neg")]["_pm"],
            s_pm,
            keep[(name, "neg")]["pm_threshold"],
            alpha,
            draws,
            seed + 2,
        )
    for b in br_rows:
        eff = b["pm_score"] - b["pm_threshold"]
        b["expected_branch_shots"] = 12 * s_pm * b["branch_prob"]
        b["power_planned_holm"] = (
            R.power_normal(eff, b["pm_sd1"], 12 * s_pm, alpha_sec) if eff > 0 else 0.0
        )
        b["shots_80_holm"] = R.shots_for_power(eff, b["pm_sd1"], alpha_sec, 0.8)
        b["testable"] = b["expected_branch_shots"] >= an["min_branch_shots"]
        if b["n"] == 1:
            b["power_mc_holm"] = R.mc_power_pm(
                b["_pm"], s_pm, b["pm_threshold"], alpha_sec, draws, seed + 3, ratio=True
            )
    zrows = []
    for n in plan["zero_branch_steps"]:
        sup = R.n_step_map(primary, n, [0] * n)
        j = R.choi(sup)
        zrows.append(
            {
                "n": n,
                "record": "0" * n,
                "prob": float(np.trace(j).real),
                "fe_fixed_unconditional_V": R.bell_fidelity(j, vs[n]),
                "fe_max": R.max_bell_fidelity(j),
                "negativity": R.negativity(j),
            }
        )
    write_csv(sd / "zero_branch.csv", zrows)
    write_csv(sd / "branches.csv", [{k: v for k, v in b.items() if k != "_pm"} for b in br_rows])
    write_csv(sd / "power.csv", pow_rows)
    log(out, "MC power: " + json.dumps({k: round(v, 4) for k, v in mc.items()}))

    # -- 4. PPT witness at n = 2 (the route left when F_e < 1/2 but J is entangled)
    n_ppt = plan["ppt_n"]
    w_ppt, lam = R.ppt_witness(R.choi(R.n_step_map(primary, n_ppt)))
    c0, alpha_w = R.pauli_design(w_ppt)
    ppt_rows = []
    for name in (
        "noiseless",
        "stage4c_live",
        "live_deficit_matched",
        "live_dephase_0.02",
        "live_dephase_0.05",
    ):
        sup = R.n_step_map(ms[name], n_ppt)
        mvals = R.pauli_expectations(sup, spam)
        val = c0 + float((alpha_w * mvals).sum())
        sd1 = R.pauli_sd1(alpha_w, mvals)
        shift = 2 * leak(cfg, n_ppt) * float(np.abs(alpha_w).sum())
        eff = -val - shift
        ppt_rows.append(
            {
                "model": name,
                "n": n_ppt,
                "witness_ideal_spam": float(np.trace(w_ppt @ R.choi(sup)).real),
                "witness_spam": val,
                "leak_shift": shift,
                "sd1": sd1,
                "shots_80": R.shots_for_power(eff, sd1, alpha_sec, 0.8),
                "shots_90": R.shots_for_power(eff, sd1, alpha_sec, 0.9),
            }
        )
    write_csv(sd / "ppt_witness.csv", ppt_rows)

    # -- 5. sensitivity table (n = 1, 2)
    sens_rows = []
    for name in ms:
        for n in (1, 2):
            r = keep[(name, n)]
            pw = next(
                p for p in pow_rows if p["model"] == name and p["n"] == n and p["route"] == "pm"
            )
            sens_rows.append(
                {
                    "model": name,
                    "n": n,
                    "fe_fixed": r["fe_fixed"],
                    "fe_max": r["fe_max"],
                    "negativity": r["negativity"],
                    "pm_score": r["pm_score"],
                    "pm_threshold": r["pm_threshold"],
                    "pm_power_planned": pw["power_planned"],
                    "pm_shots_80": pw["shots_80"],
                    "bell_F": r["bell_F"],
                    "mdi_payoff": r["mdi_payoff"],
                }
            )
    write_csv(sd / "sensitivity.csv", sens_rows)

    # tolerance to coherent SPAM error at n = 1 (primary and deficit-matched models)
    tol = {}
    r_sx = spam.prep_depol / (2 * cfg["spam"]["gates_per_prep"])
    worst_coh = math.sqrt(1.5 * r_sx) * cfg["spam"]["gates_per_prep"]  # all infidelity coherent
    for name in ("stage4c_live", "live_deficit_matched"):
        r = keep[(name, 1)]
        margin = r["pm_score"] - r["pm_threshold"]
        tol[name] = {
            "margin": margin,
            "tolerated_per_operation": margin / 2,
            "worst_case_coherent_per_operation": worst_coh,
        }

    # -- 6. q59-only control (Bell route): a bound on the reference -> q59 information
    nq = plan["q59_control_steps"]
    proc = W.LinearProcess.instrument(primary, A.RHO0_JOINT)
    marg = proc.marginals(nq)
    p1 = float(np.mean(marg))
    se_diff = math.sqrt(2 * p1 * (1 - p1) / (plan["q59_control_shots"] * nq)) * math.sqrt(2)
    q59_ctrl = {
        "steps": nq,
        "shots_per_state": plan["q59_control_shots"],
        "predicted_delta_p1": 0.0,
        "p1": p1,
        "detectable_delta_p1_95": 1.96 * se_diff,
        "note": "SE inflated by sqrt(2) for within-shot correlation of the record",
    }

    # -- 7. budget and the unsubmitted plan
    budget, plan_json = budget_plan(cfg, pick)
    write_csv(sd / "budget.csv", budget)

    # -- headline
    prim = keep[("stage4c_live", prot["primary_n"])]
    pess = keep[("live_deficit_matched", prot["primary_n"])]
    n_feasible = [
        n
        for n in all_n
        if n >= 1
        and keep[("stage4c_live", n)]["pm_score"] > keep[("stage4c_live", n)]["pm_threshold"]
    ]
    eb_from = next(
        (n for n in all_n if keep[("noiseless", n)]["negativity"] <= 1e-12 and n > 0), None
    )
    summary = {
        "meta": run_metadata(cfg_meta(cfg), an["seed"]),
        "inputs": {
            k: cfg["experiment"][k] for k in ("pieces", "phase6_pieces", "calibration", "device")
        },
        "q75_spam": vars(spam),
        "mdi_reset": None if rs is None else vars(rs),
        "deficit_matched_dephasing_p": p_def,
        "targets_V": {str(n): mat(vs[n]) for n in all_n},
        "reference": {
            "rule": vars(rule) | {"forbid_path": list(rule.forbid_path)},
            "candidates": len(cands),
            "admitted": [c["qubit"] for c in admitted],
            "pick": pick,
            "bell_F_n1_q74_direct": par74["fidelity"],
        },
        "primary": {
            "n": prot["primary_n"],
            "route": "pm",
            "alpha": alpha,
            "pm_score": prim["pm_score"],
            "pm_threshold": prim["pm_threshold"],
            "fe_fixed": prim["fe_fixed"],
            "fe_max": prim["fe_max"],
            "negativity": prim["negativity"],
            "shots": 12 * s_pm,
            "power_normal": next(
                p["power_planned"]
                for p in pow_rows
                if p["model"] == "stage4c_live" and p["n"] == 1 and p["route"] == "pm"
            ),
            "power_mc": mc["pm_n1_stage4c_live"],
            "deficit_matched": {
                "pm_score": pess["pm_score"],
                "power_mc": mc["pm_n1_live_deficit_matched"],
                "fe_fixed": pess["fe_fixed"],
            },
        },
        "unconditional_feasible_n_live": n_feasible,
        "noiseless_eb_from_n": eb_from,
        "mc_power": mc,
        "secondary_alpha_per_test": alpha_sec,
        "coherent_spam_tolerance": tol,
        "q59_control": q59_ctrl,
        "ppt_witness_n2": {"lambda_min_live": lam, "rows": ppt_rows},
        "plan": plan_json,
    }
    save_json(sd / "summary.json", summary)
    save_json(out / "summary.json", summary)
    log(
        out,
        f"primary: P&M n=1 score {prim['pm_score']:.4f} vs {prim['pm_threshold']:.4f}, "
        f"power {mc['pm_n1_stage4c_live']:.3f}; deficit-matched "
        f"{pess['pm_score']:.4f}, power {mc['pm_n1_live_deficit_matched']:.3f}",
    )
    return summary


def durations(cfg: dict[str, Any]) -> dict[str, float]:
    p = cfg["protocol"]
    d = {}
    for n in p["steps"]:
        d[f"pm_n{n}"] = n * p["step_s"] + p["end_s"] + 0.1e-6
        d[f"bell_n{n}"] = n * p["step_s"] + p["end_s"] + 0.5e-6
    d["pm_neg"] = p["negative_control_n"] * p["step_s"] + p["end_s"] + 0.1e-6
    extra = cfg.get("mdi", {}).get("reset_s", 0.0) if cfg.get("mdi", {}).get("reset") else 0.0
    for n in p["mdi_steps"]:
        d[f"mdi_n{n}"] = n * p["step_s"] + p["end_s"] + 0.3e-6 + extra
    d["q59_ctrl"] = cfg["plan"]["q59_control_steps"] * p["step_s"] + p["end_s"]
    # the longer of this estimate and FakeKingston's schedule of the exported circuits
    man = REPO_ROOT / cfg["experiment"]["summary_dir"] / "qasm" / "manifest.json"
    if man.exists():
        circ = json.loads(man.read_text())["circuits"]
        for arm in d:
            fake = [
                c["duration_s_fake"]
                for k, c in circ.items()
                if k.startswith(arm + "_") or k.startswith(arm.replace("_ctrl", "ctrl"))
            ]
            if fake:
                d[arm] = max(d[arm], max(fake))
    return d


def jobs_for(cfg, arms: dict[str, tuple[int, int]], dur: dict[str, float], seed: int):
    """Pubs (one per circuit, shuffled) packed into jobs under ``max_qpu_seconds``."""
    cm = F.CostModel(**cfg["cost"])
    rng = np.random.default_rng(seed)
    items = [(a, c, s) for a, (nc, s) in arms.items() for c in range(nc)]
    order = rng.permutation(len(items))
    jobs: list[list[F.Pub]] = [[]]
    for i in order:
        a, c, s = items[i]
        trial = jobs[-1] + [F.Pub(len(jobs) - 1, len(jobs[-1]), a, c, s)]
        if cm.job(trial, dur)["estimate_s"] > cfg["plan"]["max_qpu_seconds"] and jobs[-1]:
            jobs.append([F.Pub(len(jobs), 0, a, c, s)])
        else:
            jobs[-1] = trial
    return jobs, [cm.job(j, dur) for j in jobs]


def budget_plan(cfg: dict[str, Any], pick: dict[str, Any] | None):
    plan = cfg["plan"]
    dur = durations(cfg)
    options = {
        "pm_primary": {a: (12, s) for a, s in plan["pm_shots"].items()},
        "bell_route": {a: (6, s) for a, s in plan["bell_shots"].items()}
        | {"q59_ctrl": (2, plan["q59_control_shots"])},
        "mdi_game": {a: (16, s) for a, s in plan["mdi_shots"].items()},
    }
    rows, out = [], {}
    for k, (name, arms) in enumerate(options.items()):
        jobs, costs = jobs_for(cfg, arms, dur, cfg["analysis"]["seed"] + k)
        est = sum(c["estimate_s"] for c in costs)
        ub = sum(c["upper_s"] for c in costs)
        rows.append(
            {
                "option": name,
                "jobs": len(jobs),
                "circuits": sum(len(j) for j in jobs),
                "shots": sum(c["shots"] for c in costs),
                "estimate_s": est,
                "upper_s": ub,
                "counted_if_unread_s": ub,
                "fits_cap": ub <= plan["hard_cap_s"],
                "fits_remaining": ub <= plan["remaining_after_phase7_s"],
            }
        )
        out[name] = {
            "jobs": [
                [{"pub": q.key, "arm": q.arm, "circuit": q.chunk, "shots": q.shots} for q in j]
                for j in jobs
            ],
            "costs": costs,
        }
    # a dry run of the guards on the recommended option (nothing is submitted, ever, from here)
    spent = 0.0
    guard = []
    for c in out["pm_primary"]["costs"]:
        try:
            F.budget_check(
                c["estimate_s"],
                c["upper_s"],
                plan["max_qpu_seconds"],
                plan["remaining_after_phase7_s"] - spent,
                "open",
                spent,
                plan["hard_cap_s"],
            )
            guard.append("pass")
            spent += F.job_charge("plan", c["upper_s"], {}).counted_s
        except F.Refused as e:
            guard.append(f"refused: {e}")
    out["guard_dry_run_pm_primary"] = guard
    out["reference"] = pick["qubit"] if pick else None
    out["submitted"] = False
    out["approval"] = None
    return rows, out


# -- QASM export (hardware group; FakeKingston only) -----------------------------------------------


def export_qasm(cfg: dict[str, Any], out: Path) -> None:
    from qiskit import qasm3, transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    qd = sd / "qasm"
    qd.mkdir(parents=True, exist_ok=True)
    summ = json.loads((sd / "summary.json").read_text())
    vs = {int(n): np.array(m["re"]) + 1j * np.array(m["im"]) for n, m in summ["targets_V"].items()}
    pick = summ["reference"]["pick"]
    path = [int(q) for q in pick["path"].split("-")]  # 75, 74, ..., reference
    lay = cfg["layout"]
    step = qasm3.loads((REPO_ROOT / cfg["experiment"]["phase6_dir"] / "step.qasm").read_text())
    inp = inputs(cfg)
    s0 = R.step_ops(inp["noiseless"])
    s0_neg = R.step_ops(inp["noiseless"], "A2")
    be = FakeKingston()
    hw = cfg["hardware"]
    p = cfg["protocol"]
    manifest: dict[str, Any] = {}
    worst = 0.0

    def emit(name: str, qc, layout: list[int], expect: float, prob) -> None:
        nonlocal worst
        got = prob(qc)
        worst = max(worst, abs(got - expect))
        tq = transpile(
            qc,
            be,
            initial_layout=layout,
            optimization_level=hw["optimization_level"],
            seed_transpiler=hw["seed_transpiler"],
        )
        text = qasm3.dumps(tq)
        (qd / f"{name}.qasm").write_text(text)
        manifest[name] = {
            "sha256": R.sha256_text(text),
            "physical_qubits": layout,
            "duration_s_fake": tq.estimate_duration(be.target, unit="s"),
            "ops": {k: int(v) for k, v in tq.count_ops().items()},
            "noiseless_check": abs(got - expect),
        }

    def pm_prob(label):
        return lambda qc: float(R.noiseless_unconditional_rho(qc, [1])[label, label].real)

    for n in p["steps"]:
        v = vs[n]
        sup = R.n_step_map(s0, n)
        ps = R.pm_probabilities(sup, v, R.Spam())["p_succ"]
        for k, b in itertools.product(range(6), (0, 1)):
            emit(
                f"pm_n{n}_{R.AXIS_LABELS[k].replace('+', 'p').replace('-', 'm')}_b{b}",
                R.pm_circuit(step, n, k, b, v),
                [lay["output"], lay["memory"]],
                ps[k, b],
                pm_prob(b),
            )
    sup = R.n_step_map(s0, 1, first=s0_neg)
    ps = R.pm_probabilities(sup, vs[1], R.Spam())["p_succ"]
    for k, b in itertools.product(range(6), (0, 1)):
        emit(
            f"pm_neg_{R.AXIS_LABELS[k].replace('+', 'p').replace('-', 'm')}_b{b}",
            R.pm_circuit(step, 1, k, b, vs[1], negative=True),
            [lay["output"], lay["memory"]],
            ps[k, b],
            pm_prob(b),
        )
    four = [lay["output"], *path]
    nsw = pick["swaps"]
    ref_idx = 2 + nsw
    for n in cfg["plan"]["bell_steps"]:
        sup = R.n_step_map(s0, n)
        par = R.bell_parities(R.bell_final_state(sup, n, R.RefNoise()), vs[n], R.RefNoise())
        for s, fl in itertools.product(("xx", "yy", "zz"), (0, 1)):

            def even(qc):
                r = R.noiseless_unconditional_rho(qc, [1, ref_idx])
                return float(r[0, 0].real + r[3, 3].real)

            emit(
                f"bell_n{n}_{s}_f{fl}",
                R.bell_circuit(step, n, s, fl, vs[n], nsw),
                four,
                par[s],
                even,
            )
    nq = cfg["plan"]["q59_control_steps"]
    for b in (0, 1):
        emit(
            f"q59ctrl_r{b}",
            R.q59_control_circuit(step, nq, b, nsw),
            four,
            float(b),
            lambda qc: float(R.noiseless_unconditional_rho(qc, [ref_idx])[1, 1].real),
        )
    reset = bool(cfg.get("mdi", {}).get("reset"))  # A4 (#38): noiselessly, an exact |0>
    for n in p["mdi_steps"]:
        sup = R.n_step_map(s0, n)
        pq = R.mdi_probabilities(sup, R.Spam(), R.Spam(), 0.0, R.MdiReset() if reset else None)
        for x, y in itertools.product(range(4), range(4)):
            emit(
                f"mdi_n{n}_x{x}_y{y}",
                R.mdi_circuit(step, n, x, y, reset),
                [lay["output"], lay["memory"], lay["bridge"]],
                pq[x, y],
                lambda qc: float(R.noiseless_unconditional_rho(qc, [1, 2])[0, 0].real),
            )
    meta = run_metadata(cfg_meta(cfg), hw["seed_transpiler"])
    save_json(
        qd / "manifest.json",
        {"meta": meta, "backend": be.name, "max_noiseless_check": worst, "circuits": manifest},
    )
    log(out, f"exported {len(manifest)} circuits; worst noiseless check {worst:.2e}")
    if worst > 1e-9:
        raise RuntimeError(f"circuits disagree with the numpy model: {worst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export-qasm", action="store_true", help="FakeKingston only (hardware group)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = new_run_dir(cfg["experiment"]["name"])
    t0 = time.time()
    if args.export_qasm:
        export_qasm(cfg, out)
    else:
        analyse(cfg, out)
    log(out, f"done in {time.time() - t0:.1f} s; outputs in {out}")
    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    dest = sd / "qasm" / "export_log.txt" if args.export_qasm else sd / "log.txt"
    dest.write_text((out / "log.txt").read_text())


if __name__ == "__main__":
    main()
