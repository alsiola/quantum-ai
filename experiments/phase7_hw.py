"""Phase 7 Stage 3 (issue #24): the pre-registered hardware follow-up on ibm_kingston.

Three modes (``docs/phase7-preregistration.md``):

* ``--dry-run-local`` (this issue; no network of any kind): rebuild every circuit from the
  committed Phase 6 / Stage 2 files against ``FakeKingston`` and check the QASM hashes;
  exercise the live-delay regeneration and the calibration check on ``FakeKingston``'s
  target; the schedule, the budget, the exact prediction table and the end-to-end validation
  of the decision rule on synthetic data; optionally ``--aer-shots N`` runs the pilot job's
  circuits through a local Aer simulator via Phase 6's job path. Without the ``hardware``
  group the circuit steps are skipped and the numpy steps still run.
* ``--dry-run`` (Stage 4): read the live target, calibration, plan and allowance; build and
  check the circuits; predict the pilot gate from the live calibration; submit nothing.
* ``--submit`` (Stage 4): refuses unless ``[ibm] approval_url`` names the owner's approving
  comment. Pilot job, the pilot gate (stop on failure, never retry automatically), then the
  main jobs, each guarded by the budget and calibration checks. Every stage saves to the run
  directory; ``--from-run <dir>`` resumes, and a submitted job is re-fetched by its id.

Phase 6's job, resume-by-id, calibration-capture and saving code paths are imported from
``experiments/phase6.py``; #20's circuit builders from ``experiments/phase7_ablation.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import tomllib
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_ablation as PA

from qmem import ablate_hw as A
from qmem import ancilla as AN
from qmem import artefacts as AR
from qmem import drift as D
from qmem import followup as F
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

ARMS = ("baseline", "A2", "A2s_spec", "A1", "A3", "D1")
PMD_ARMS = ("baseline", "A2", "A2s_spec", "A1", "A3")


# -- config ------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def plain(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def rdir(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["experiment"][key]


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


def analysis_settings(cfg: dict[str, Any], validation: bool = False) -> F.AnalysisSettings:
    a = cfg["analysis"]
    s = F.AnalysisSettings(
        a["k_max"],
        a["burn"],
        tuple(a["offsets"]),
        a["block"],
        a["n_boot"],
        a["seed"],
        a["alpha_segment"],
        a["n_perm"],
        a["cp_grid"],
        a["homog_block"],
        a["const_block"],
    )
    if validation:
        s.n_boot = cfg["validation"]["n_boot"]
        s.n_perm = cfg["validation"]["n_perm"]
    return s


def rule(cfg: dict[str, Any]) -> F.DecisionRule:
    d = cfg["decision"]
    return F.DecisionRule(
        gate_z=d["gate_z"],
        vanish_fraction=d["vanish_fraction"],
        persist_z=d["persist_z"],
        persist_fraction=d["persist_fraction"],
        family_alpha=d["family_alpha"],
    )


def plan_arms(cfg: dict[str, Any], tiers=None) -> tuple[dict[str, int], dict[str, tuple]]:
    """(shots per plain arm, instanced arms) of the arms in ``tiers`` (default: the plan)."""
    tiers = tiers or cfg["schedule"]["include_tiers"]
    plain_arms, inst = {}, {}
    for name, a in cfg["arms"].items():
        if a["tier"] not in tiers:
            continue
        if "instances" in a:
            inst[name] = (a["instances"], a["shots_per_instance"])
        else:
            plain_arms[name] = a["shots"]
    return plain_arms, inst


def schedule(cfg: dict[str, Any]) -> list[list[F.Pub]]:
    s = cfg["schedule"]
    arms, inst = plan_arms(cfg)
    return F.make_schedule(arms, s["n_jobs"], s["max_chunk"], s["seed"], inst)


def pilot_schedule(cfg: dict[str, Any]) -> list[F.Pub]:
    p = cfg["pilot"]
    return F.make_schedule(p["arms"], 1, cfg["schedule"]["max_chunk"], p["seed"])[0]


# -- durations and budget (numpy; from the committed circuit records) ---------------------------


def durations(cfg: dict[str, Any]) -> dict[str, float]:
    """Circuit seconds per arm: Phase 6's live baseline estimate plus the arm's FakeKingston
    duration minus FakeKingston's baseline (#20, #21, #23 and #22 records)."""
    base = cfg["budget"]["phase6_circuit_s"]
    ab = json.loads((rdir(cfg, "ablation_dir") / "circuits.json").read_text())
    rows = {r["circuit"]: r for r in ab["circuits"]}
    out = {v: base + rows[v]["duration_minus_baseline_s"] for v in ("baseline", "A1", "A2")}
    out["A2s_spec"] = base + rows["A2s_spec"]["duration_minus_baseline_s"]
    out["A3"] = base + max(
        r["duration_minus_baseline_s"] for k, r in rows.items() if k.startswith("A3_")
    )
    cl = json.loads((rdir(cfg, "classical_dir") / "controls_circuits.json").read_text())
    for k in ("d1", "d3"):
        out[k.upper()] = base + cl["controls"][k]["duration_minus_phase6_s_fake"]
    an = json.loads((rdir(cfg, "ancilla_dir") / "circuits.json").read_text())
    b2 = an["B2"]
    out["B2"] = base + b2["duration_fakekingston_s"] - b2["baseline_duration_fakekingston_s"]
    with open(rdir(cfg, "spectator_dir") / "controls_cost.csv") as f:
        c3 = next(r for r in csv.DictReader(f) if r["control"] == "C3")
    out["C3"] = float(c3["circuit_us"]) * 1e-6
    return out


def cost_model(cfg: dict[str, Any]) -> F.CostModel:
    b = cfg["budget"]
    return F.CostModel(b["sub_job_overhead_s"], b["rep_delay_s"], b["margin"], b["billing_round_s"])


def budget_table(cfg: dict[str, Any]) -> dict[str, Any]:
    """Estimates and upper bounds of the pilot, each main job, each tier and the tier-3 arms."""
    cm, dur = cost_model(cfg), durations(cfg)
    b = cfg["budget"]
    jobs = schedule(cfg)
    rows = []
    pj = cm.job(pilot_schedule(cfg), dur)
    rows.append({"item": "pilot", "jobs": 1, **pj})
    for j, job in enumerate(jobs):
        rows.append({"item": f"main job {j}", "jobs": 1, **cm.job(job, dur)})
    tier_s = {}
    for t in (1, 2):
        arms, inst = plan_arms(cfg, [t])
        shots = sum(arms.values()) + sum(n * s for n, s in inst.values())
        sec = sum(cm.shot_seconds(n, dur[a]) for a, n in arms.items())
        sec += sum(cm.shot_seconds(n * s, dur[a]) for a, (n, s) in inst.items())
        tier_s[t] = sec
        rows.append(
            {
                "item": f"tier {t} (shot time only; job overheads below)",
                "jobs": 0,
                "shots": shots,
                "shot_s": sec,
                "estimate_s": sec,
                "upper_s": (1 + cm.margin) * sec,
            }
        )
    n_main = len(jobs)
    ov = n_main * cm.overhead_s
    main_est = sum(r["estimate_s"] for r in rows if r["item"].startswith("main job"))
    main_ub = sum(r["upper_s"] for r in rows if r["item"].startswith("main job"))
    rows.append(
        {
            "item": f"default plan (tiers 1+2, {n_main} jobs)",
            "jobs": n_main,
            "shots": sum(r["shots"] for r in rows if r["item"].startswith("main job")),
            "shot_s": main_est - ov,
            "estimate_s": main_est,
            "upper_s": main_ub,
        }
    )
    rows.append(
        {
            "item": "pilot + default plan",
            "jobs": n_main + 1,
            "shots": pj["shots"] + rows[-1]["shots"],
            "shot_s": pj["shot_s"] + rows[-1]["shot_s"],
            "estimate_s": pj["estimate_s"] + main_est,
            "upper_s": pj["upper_s"] + main_ub,
        }
    )
    for name, t3 in cfg["tier3"].items():
        sec = cm.shot_seconds(t3["shots"], dur[name])
        n_j = max(1, math.ceil((cm.overhead_s + sec) / b["max_qpu_seconds"]))
        while cm.overhead_s + sec / n_j > b["max_qpu_seconds"]:
            n_j += 1
        est = n_j * cm.overhead_s + sec
        rows.append(
            {
                "item": f"tier 3 {name} ({n_j} job{'s' if n_j > 1 else ''})",
                "jobs": n_j,
                "shots": t3["shots"],
                "shot_s": sec,
                "estimate_s": est,
                "upper_s": (1 + cm.margin) * est + n_j * cm.billing_round_s,
            }
        )
    worst_job = max(
        r["estimate_s"] for r in rows if r["item"] == "pilot" or r["item"].startswith("main job")
    )
    return {
        "rows": rows,
        "durations_s": dur,
        "tier_shot_s": tier_s,
        "max_job_estimate_s": worst_job,
        "max_qpu_seconds": b["max_qpu_seconds"],
        "every_job_within_max": worst_job <= b["max_qpu_seconds"],
        "hard_cap_s": b["hard_cap_s"],
        "plan_upper_within_cap": next(
            r["upper_s"] for r in rows if r["item"] == "pilot + default plan"
        )
        <= b["hard_cap_s"],
        "allowance_after_16_s": b["allowance_after_16_s"],
    }


# -- hypotheses: exact models of every arm (numpy) ----------------------------------------------


class Models:
    """One hypothesis: a ``LinearProcess`` per arm, A3 as a bank of four Pauli-step processes
    (index into ``A.PAULIS``), and optionally an episode regime B (contiguous in time)."""

    def __init__(self, procs: dict[str, W.LinearProcess], bank: list[W.LinearProcess]):
        self.procs, self.bank = procs, bank
        self.episode: tuple[Models, float, float] | None = None

    def mapped(self, fn) -> Models:
        return Models({k: fn(v) for k, v in self.procs.items()}, [fn(b) for b in self.bank])


def mix(models: list[Models], weights: list[float]) -> Models:
    """Per-shot random mixture (direct sum) of hypotheses with the same arms."""
    procs = {k: A.mixture([m.procs[k] for m in models], weights) for k in models[0].procs}
    bank = [A.mixture([m.bank[p] for m in models], weights) for p in range(4)]
    return Models(procs, bank)


def pieces(cfg: dict[str, Any]) -> A.StepPieces:
    z = np.load(rdir(cfg, "ablation_dir") / "pieces.npz")
    return A.StepPieces.from_npz(z, "calibrated")


def snapshot_q59(cfg: dict[str, Any]) -> dict[str, float]:
    s6 = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
    q = {r["name"]: r["value"] for r in s6["prepare"]["calibration"]["qubits"]["59"]}
    return {"e01": q["prob_meas1_prep0"], "e10": q["prob_meas0_prep1"]}


def d1_process(cfg: dict[str, Any], **kw: Any) -> W.LinearProcess:
    """#23's D1 model: reset, rx with P(1) = 0.58, measure, on q59 with snapshot errors."""
    cl = json.loads((rdir(cfg, "classical_dir") / "controls_circuits.json").read_text())
    p1 = cl["meta"]["config"]["controls"]["d1"]["p1"]
    q = snapshot_q59(cfg)
    args = {"e01": q["e01"], "e10": q["e10"], **kw}
    lab, init = AR.qubit_control(p1, True, **args)
    return W.LinearProcess.hmm(lab, init)


def ablate_models(cfg: dict[str, Any], hyp: A.Hypothesis) -> Models:
    p = pieces(cfg)
    procs = {v: A.process(A.step_superops(v, p, hyp)) for v in ("baseline", "A2", "A2s_spec", "A1")}
    bank = [A.process(A.step_superops("A3", p, hyp, ps)) for ps in A.PAULIS]
    procs["D1"] = d1_process(cfg)
    return Models(procs, bank)


def _insert_before_cz(base: list, ops: list) -> list:
    out, done = [], False
    for i in base:
        if i.name == "cz" and not done:
            out += ops
            done = True
        out.append(i)
    return out


def ancilla_models(cfg: dict[str, Any], params: dict[str, float]) -> Models:
    """#21's q59-qutrit simulator with ``params`` on q59; A2s_spec equals the baseline (its
    q74 readout sits in q75's idle window); D1 with the same reset/leak/readout numbers."""
    from qmem.spectator import Instr

    s6 = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
    c = AN.apply_params(AN.config_from_snapshot(s6["prepare"]["calibration"]["noise"]), params)
    base = AN.physical_step((rdir(cfg, "phase6_dir") / "step.qasm").read_text())
    reg = AN.register_for(c, (59,), 3)

    def proc(ops: list, record=None) -> W.LinearProcess:
        sup = AN.step_superops(_insert_before_cz(base, ops), reg, c, record=record)
        return AN.to_process(sup, reg)

    procs = {
        "baseline": proc([]),
        "A2": proc([Instr("measure", (75,))], record=[1]),
        "A1": proc([Instr("reset", (75,))]),
    }
    procs["A2s_spec"] = procs["baseline"]
    bank = []
    for ps in A.PAULIS:
        ops = []
        if ps in ("Z", "Y"):
            ops.append(Instr("rz", (75,), math.pi))
        if ps in ("X", "Y"):
            ops.append(Instr("x", (75,)))
        bank.append(proc(ops))
    q = snapshot_q59(cfg)
    procs["D1"] = d1_process(
        cfg,
        e01=params.get("read10", q["e01"]),
        e10=params.get("read01", q["e10"]),
        reset_e01=params.get("reset_e0"),
        reset_e10=params.get("reset_e1"),
        leak=params.get("leak", 0.0),
        seep=params.get("seep", 0.0),
        read2=params.get("read2", 1.0),
    )
    return Models(procs, bank)


def hmm_source(cfg: dict[str, Any], name: str) -> W.LinearProcess:
    """#19's committed parameter vectors (target: the episode-free rest)."""
    dsum = json.loads((rdir(cfg, "drift_dir") / "summary.json").read_text())
    lw = cfg["layout"]["length"]
    a = cfg["analysis"]
    fam = {"drift_mixture_K3": "mixture_K3", "hmm_3state": "hmm_3state"}[name]
    row = next(r for r in dsum["adversarial_best"] if r["family"] == fam and r["target"] == "rest")
    if fam == "mixture_K3":
        f = D.mixture_family(3, lw, a["k_max"], a["burn"])
        return f.process(np.asarray(row["theta_most_consistent"]))
    f = D.hmm_family(3, lw, a["k_max"], a["burn"])
    return f.process(np.asarray(row["theta_best_fit"]))


def build_models(cfg: dict[str, Any], h: dict[str, Any]) -> Models:
    """The exact models of every arm under hypothesis ``h`` (a ``[[hypotheses]]`` entry)."""
    kind = h["kind"]
    if kind == "ancilla":
        an = json.loads((rdir(cfg, "ancilla_dir") / "summary.json").read_text())
        return ancilla_models(cfg, an["hypotheses"][h["params"]])
    if kind == "hmm":
        p = hmm_source(cfg, h["source"])
        m = Models({a: p for a in PMD_ARMS if a != "A3"}, [p] * 4)
        m.procs["D1"] = d1_process(cfg)
        return m
    hyp = A.Hypothesis(name=h["name"], op_coherence=h.get("op_coherence", 0.0))
    if "spectator" in h:
        sp = h["spectator"]
        t_step = cfg["budget"]["phase6_circuit_s"] / cfg["layout"]["length"]
        n6 = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
        n6 = n6["prepare"]["calibration"]["noise"]
        t_ro = n6["measure_time"] + n6["reset_time"]  # q59 read out twice per step
        # #22's convention: a shift f (Hz) on |1> of q75 multiplies it by exp(-2 pi i f t), an
        # rz of -2 pi f t; with it this reduced model reproduces #22's Stark sweep exactly
        ph_s = -2 * math.pi * sp["stark_hz"] * t_ro
        ph_z = -2 * math.pi * sp["zz_hz"] * t_step
        cold = ablate_models(cfg, A.Hypothesis(q75_phase=ph_s, spec_dephasing=sp["spec_dephasing"]))
        hot = ablate_models(
            cfg, A.Hypothesis(q75_phase=ph_s + ph_z, spec_dephasing=sp["spec_dephasing"])
        )
        m = mix([cold, hot], [1 - sp["p_thermal"], sp["p_thermal"]])
        m.procs["D1"] = d1_process(cfg)
        return m
    m = ablate_models(cfg, hyp)
    if "persistence" in h:
        k = AR.persistence_kernel(h["persistence"], "sticky")
        m = m.mapped(lambda p: AR.with_persistence(p, k))
    if "episode" in h:
        e = h["episode"]
        rm = AR.readout_matrix(e["e01"], 0.0)
        b = ablate_models(cfg, A.Hypothesis(q75_phase=e["q75_phase"]))
        b = b.mapped(lambda p: AR.with_readout(p, rm))
        m.episode = (b, e["weight"], e["start"])
    return m


def pauli_table(cfg: dict[str, Any]) -> np.ndarray:
    return PA.pauli_table(PA.load_config(REPO_ROOT / cfg["experiment"]["ablation_config"]))


def a3_instance(models: Models, paulis: np.ndarray) -> D.TimeVaryingProcess:
    ops = np.stack([models.bank[p].ops for p in paulis])
    b = models.bank[0]
    return D.TimeVaryingProcess(b.init, ops, b.final)


def exact_arm(models: Models, arm: str, paulis: np.ndarray, cfg: dict[str, Any]) -> dict:
    a = cfg["analysis"]
    lw, burn, k_max = cfg["layout"]["length"], a["burn"], a["k_max"]
    if arm == "A3":
        gs, p1s, bl = [], [], np.zeros(4)
        for row in paulis:
            tv = a3_instance(models, row)
            gs.append(tv.expected_windows(k_max, burn, common=True))
            p1s.append(tv.marginals()[burn:].mean())
            bl += tv.block_probs(range(burn, lw - 1), 2)
        g = np.mean(gs, 0)
        p1 = float(np.mean(p1s))
        bl /= len(paulis)
        lag = float(bl[3] / (bl[2] + bl[3]) - bl[1] / (bl[0] + bl[1]))
        marg = None
    else:
        p = models.procs[arm]
        g = p.expected_windows(k_max, burn, lw, common=True)
        marg = p.marginals(lw)
        p1 = float(marg[burn:].mean())
        lag = AR.lag1_excess(p, burn, lw)
    out = {f"w{j}": float(W.witness(g, j)) for j in a["offsets"]}
    out.update({"p1": p1, "lag1": lag, "g": g.tolist()})
    if marg is not None:
        out["marginals"] = marg.tolist()
    return out


def exact_hypothesis(cfg: dict[str, Any], h: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Exact statistics of every arm; with an episode, at its weight as a per-shot mixture."""
    m = build_models(cfg, h)
    if m.episode is not None:
        b, w, _ = m.episode
        m = mix([m, b], [1 - w, w])
    paulis = pauli_table(cfg)
    return {arm: exact_arm(m, arm, paulis, cfg) for arm in ARMS}


# -- synthetic data in the planned schedule (numpy) ------------------------------------------------


def sample_plan(
    cfg: dict[str, Any], models: Models, jobs: list[list[F.Pub]], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    """Shots of every arm in nominal time order (job by job, pub by pub), with job and pub
    labels. With an episode, shots whose global time index falls in
    ``[start, start + weight) x total`` come from regime B."""
    lw = cfg["layout"]["length"]
    paulis = pauli_table(cfg)
    total = sum(p.shots for job in jobs for p in job)
    lo = hi = -1
    if models.episode is not None:
        _, w, start = models.episode
        lo, hi = int(start * total), int((start + w) * total)
    out: dict[str, dict[str, list]] = {}
    t = 0
    for job in jobs:
        for p in job:
            s = np.random.SeedSequence([seed, p.job, p.position]).generate_state(1)[0]
            n_b = max(0, min(t + p.shots, hi) - max(t, lo))
            first_b = max(t, lo) - t if n_b > 0 else 0
            parts = []
            for regime, n in ((models, p.shots - n_b), (None, n_b)):
                mdl = regime if regime is not None else (models.episode[0] if n else None)
                if n == 0 or mdl is None:
                    parts.append(np.zeros((0, lw), dtype=np.uint8))
                    continue
                if p.arm == "A3":
                    parts.append(
                        a3_instance(mdl, paulis[p.instance]).sample(n, int(s) + len(parts))
                    )
                else:
                    parts.append(mdl.procs[p.arm].sample(n, lw, int(s) + len(parts)))
            a_sh, b_sh = parts
            arr = np.concatenate([a_sh[:first_b], b_sh, a_sh[first_b:]])
            d = out.setdefault(p.arm, {"shots": [], "jobs": [], "pubs": []})
            d["shots"].append(arr)
            d["jobs"].append(np.full(p.shots, p.job))
            d["pubs"].append(np.full(p.shots, p.position))
            t += p.shots
    return {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in out.items()}


def analyse_dataset(
    cfg: dict[str, Any],
    data: dict[str, dict[str, np.ndarray]],
    marginals: dict[str, np.ndarray | None],
    s: F.AnalysisSettings,
    ref_values: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    """The full pre-registered pipeline on one dataset (every arm in nominal time order)."""
    arms, guards, segs = {}, {}, {}
    for i, (arm, d) in enumerate(data.items()):
        guards[arm] = F.guard(d["shots"], marginals.get(arm), s, seed + 7 * i)
        if arm == "A3":  # 64 different circuits: block dispersion across instances is by design
            guards[arm]["drift_flags_not_applicable"] = guards[arm]["drift_flags"]
            guards[arm]["drift_flags"] = []
        arms[arm] = F.analyse_arm(d["shots"], d["jobs"], d["pubs"], s, seed + 11 * i)
        segs[arm] = F.segment_policy(d["shots"], d["jobs"], s, seed + 13 * i)
    flagged = sorted(set(segs["baseline"]["flagged"]) | set(segs["A2"]["flagged"]))
    unflagged = None
    if flagged:
        unflagged = {}
        for arm in ("baseline", "A2"):
            d = data[arm]
            keep = ~np.isin(d["jobs"], flagged)
            if keep.sum() < 100:
                unflagged = None
                break
            unflagged[arm] = F.analyse_arm(
                d["shots"][keep], d["jobs"][keep], d["pubs"][keep], s, seed + 17
            )
    dec = F.decide(arms, rule(cfg), unflagged, flagged, ref_values=ref_values)
    return {"arms": arms, "guards": guards, "segments": segs, "decision": dec}


# -- validation (numpy; parallel over replicates) ------------------------------------------


def _replicate(task: dict[str, Any]) -> dict[str, Any]:
    cfg, h, rep = task["cfg"], task["hyp"], task["rep"]
    models = task.get("_models") or build_models(cfg, h)
    jobs = schedule(cfg)
    seed = int(
        np.random.SeedSequence([cfg["validation"]["seed"], task["h_index"], rep]).generate_state(1)[
            0
        ]
        % 2**31
    )
    data = sample_plan(cfg, models, jobs, seed)
    s = analysis_settings(cfg, validation=True)
    res = analyse_dataset(cfg, data, task["marginals"], s, task["ref_values"], seed)
    dec: F.Decision = res["decision"]
    out = {
        "hypothesis": h["name"],
        "rep": rep,
        "gate": dec.gate,
        "primary": dec.primary,
        "primary_all_jobs": dec.primary_all_jobs,
        "primary_unflagged": dec.primary_unflagged,
        "n_flagged_jobs": len(dec.flagged_jobs),
        "ratio_a2": dec.ratio_a2,
        "integrity_fail": sum(len(g["integrity_fail"]) for g in res["guards"].values()),
        "decoding_warn": sum(len(g["decoding_warn"]) for g in res["guards"].values()),
        "drift_flags": sum(len(g["drift_flags"]) for g in res["guards"].values()),
    }
    for arm, r in res["arms"].items():
        out[f"{arm}_w0"] = r["w0"]
        out[f"{arm}_se0"] = r["se0"]
        out[f"{arm}_se0_shot"] = r["se_shot0"]
        out[f"{arm}_lag1"] = r["lag1"]
    for k, r in dec.secondary.items():
        out[f"sec_{k}_z"] = r["z"]
        out[f"sec_{k}_reject"] = r["reject_holm"]
    return out


def _replicates(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Several replicates of one hypothesis in one worker (its models built once)."""
    models = build_models(tasks[0]["cfg"], tasks[0]["hyp"])
    return [_replicate({**t, "_models": models}) for t in tasks]


def validation(cfg, exact: dict[str, dict], out: Path, reps: int, workers: int) -> list[dict]:
    q75 = exact["H_q75"]
    marginals = {
        a: np.asarray(q75[a]["marginals"]) if "marginals" in q75[a] else None for a in ARMS
    }
    ref_values = {"A3_vs_prediction": q75["A3"]["w0"], "D1_lag1_vs_prediction": q75["D1"]["lag1"]}
    tasks = []
    for hi, h in enumerate(cfg["hypotheses"]):
        for rep in range(reps):
            tasks.append(
                dict(
                    cfg=cfg, hyp=h, h_index=hi, rep=rep, marginals=marginals, ref_values=ref_values
                )
            )
    per = 10
    batches = [tasks[i : i + per] for i in range(0, len(tasks), per)]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    if workers <= 1:
        for b in batches:
            rows += _replicates(b)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_replicates, batches)):
                rows += r
                if i % 10 == 0:
                    log(out, f"validation {len(rows)}/{len(tasks)} ({time.time() - t0:.0f} s)")
    return rows


def summarise_validation(cfg, exact, rows) -> list[dict[str, Any]]:
    """Per hypothesis: the predicted outcome (rule on exact values and the median SEs), the
    replicate frequencies of each outcome, the error rate, and the secondary rejection rates."""
    rl = rule(cfg)
    out = []
    for h in cfg["hypotheses"]:
        rs = [r for r in rows if r["hypothesis"] == h["name"]]
        if not rs:
            continue
        ex = exact[h["name"]]
        se_b = float(np.median([r["baseline_se0"] for r in rs]))
        se_a = float(np.median([r["A2_se0"] for r in rs]))
        pred = F.expected_outcome(ex["baseline"]["w0"], ex["A2"]["w0"], se_b, se_a, rl)
        pred_gate = ex["baseline"]["w0"] < 0 and abs(ex["baseline"]["w0"]) >= rl.gate_z * se_b
        n = len(rs)
        row: dict[str, Any] = {
            "hypothesis": h["name"],
            "reps": n,
            "exact_w0_baseline": ex["baseline"]["w0"],
            "exact_w0_A2": ex["A2"]["w0"],
            "median_se0_baseline": se_b,
            "median_se0_A2": se_a,
            "predicted_gate": pred_gate,
            "predicted_primary": pred,
            "gate_rate": sum(r["gate"] for r in rs) / n,
        }
        for o in ("vanishes", "persists", "ambiguous"):
            row[f"rate_{o}"] = sum(r["primary"] == o for r in rs) / n
        row["error_rate_primary"] = sum(r["primary"] != pred for r in rs) / n
        row["error_rate_gate"] = sum(r["gate"] != pred_gate for r in rs) / n
        row["segment_sensitive_rate"] = sum(r["primary"] != r["primary_all_jobs"] for r in rs) / n
        row["flagged_job_rate"] = float(np.mean([r["n_flagged_jobs"] > 0 for r in rs]))
        row["guard_drift_flag_rate"] = float(np.mean([r["drift_flags"] > 0 for r in rs]))
        row["guard_integrity_fail_rate"] = float(np.mean([r["integrity_fail"] > 0 for r in rs]))
        for k in [c[4:-7] for c in rs[0] if c.startswith("sec_") and c.endswith("_reject")]:
            row[f"reject_{k}"] = sum(bool(r[f"sec_{k}_reject"]) for r in rs) / n
        for arm in ARMS:
            ws = np.array([r[f"{arm}_w0"] for r in rs])
            ses = np.array([r[f"{arm}_se0"] for r in rs])
            row[f"{arm}_w0_mean"] = float(ws.mean())
            row[f"{arm}_w0_sd"] = float(ws.std(ddof=1))
            row[f"{arm}_se0_median"] = float(np.median(ses))
            row[f"{arm}_z_exact_vs_mean"] = float(
                (ws.mean() - exact[h["name"]][arm]["w0"]) / (ws.std(ddof=1) / math.sqrt(n))
            )
        out.append(row)
    return out


# -- outcome map ------------------------------------------------------------------------------

LADDER = {
    "vanishes": "Rung 4 supported (A2 with A2s_spec persisting): the rank > 2 statistics need "
    "q75's coherence. Not shown: quantum memory in a stronger sense; that no other path below "
    "the vanish bound exists; a physical memory dimension (rung 5).",
    "persists": "Rung 4 not supported: rank > 2 survives a classical q75, so the effective "
    "hardware memory has a carrier outside q75's coherence. Not shown: which carrier; that "
    "q75 is irrelevant (it may still carry a classical bit).",
    "ambiguous": "No rung moves: partial loss (incomplete dephasing, a weak second path) or "
    "too little power; report effect sizes and CIs only.",
    "uninterpretable": "Replication gate failed: the ablation arms are not interpreted; the "
    "baseline result itself is reported against Phase 6 (a replication question).",
}


def prediction_rows(cfg, exact, val_summary) -> list[dict[str, Any]]:
    rows = []
    vs = {r["hypothesis"]: r for r in val_summary}
    for h in cfg["hypotheses"]:
        ex = exact[h["name"]]
        v = vs.get(h["name"], {})
        base = ex["baseline"]["w0"]
        for arm in ARMS:
            e = ex[arm]
            rows.append(
                {
                    "hypothesis": h["name"],
                    "arm": arm,
                    **{f"w{j}": e[f"w{j}"] for j in cfg["analysis"]["offsets"]},
                    "p1": e["p1"],
                    "lag1": e["lag1"],
                    "w0_over_baseline": e["w0"] / base if base else math.nan,
                    "predicted_primary": v.get("predicted_primary"),
                }
            )
    return rows


# -- circuits (hardware group) ---------------------------------------------------------------------


def ablation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return PA.load_config(REPO_ROOT / cfg["experiment"]["ablation_config"])


def strip_delays(sig: list[tuple]) -> list[tuple]:
    return [(n, q, () if n == "delay" else p) for n, q, p in sig]


def build_circuits(cfg: dict[str, Any], backend: Any, delays: str = "snapshot") -> dict[str, Any]:
    """ISA circuits of the plan's arms on ``backend`` exactly as #20 built them (level 0,
    trivial layout, delays from the Phase 6 snapshot or ``"target"``); D1 from its committed
    QASM. Returns circuits by pub name (``A3_###`` per instance), their QASM sha256 and
    FakeKingston-style ``estimate_duration`` on ``backend.target``."""
    from qiskit import qasm3

    acfg = ablation_cfg(cfg)
    acfg = copy.deepcopy(acfg)
    acfg["circuits"]["delays"] = delays
    s6 = PA.phase6_summary(acfg)
    target = backend.target
    lay = acfg["layout"]
    length = acfg["circuits"]["length"]
    dur = PA.durations(acfg, s6, target)
    full = qasm3.loads((PA.p6dir(acfg) / "circuit.qasm").read_text())
    circuits, hashes, durs, sigs = {}, {}, {}, {}
    arms, inst = plan_arms(cfg)

    def put(name: str, tq: Any) -> None:
        circuits[name] = tq
        hashes[name] = hashlib.sha256(qasm3.dumps(tq).encode()).hexdigest()
        durs[name] = tq.estimate_duration(target, unit="s")
        sigs[name] = PA.signature(tq)

    for v in ("baseline", "A1", "A2", "A2s_spec"):
        if v in arms or v == "baseline":
            qc = A.build_variant(
                full, v, lay["memory"], lay["output"], dur, length, lay["spectator"]
            )
            put(v, PA.isa(qc, backend, acfg))
    if "A3" in inst:
        paulis = PA.pauli_table(acfg)
        for i in range(inst["A3"][0]):
            ps = [A.PAULIS[k] for k in paulis[i]]
            qc = A.build_variant(full, "A3", lay["memory"], lay["output"], dur, length, None, ps)
            put(f"A3_{i:03d}", PA.isa(qc, backend, acfg))
    for name, a in cfg["arms"].items():
        if a.get("source") == "qasm" and name in arms:
            text = (REPO_ROOT / a["file"]).read_text()
            tq = PA.isa(qasm3.loads(text), backend, acfg)
            if delays == "target" and "baseline" in circuits:
                tq = repad(tq, target, circuits["baseline"].estimate_duration(target, unit="s"))
            put(name, tq)
            hashes[f"{name}_committed_file"] = hashlib.sha256(text.encode()).hexdigest()
    return {
        "circuits": circuits,
        "sha256": hashes,
        "duration_s": durs,
        "signature": sigs,
        "delays_s": {"reset": dur.reset, "measure": dur.measure, "x": dur.x, "dt": dur.dt},
    }


def repad(tq: Any, target: Any, t_ref: float) -> Any:
    """#23's per-step pad of a memoryless control, recomputed on ``target`` so that the circuit
    lasts as long as the baseline ``t_ref`` (every ``delay`` in it is the pad; #23's rule:
    floor to the timing alignment)."""
    from qiskit.circuit import Delay

    delays = [i for i in tq.data if i.operation.name == "delay"]
    if not delays:
        return tq
    old = int(delays[0].operation.duration)
    n = len(delays)
    tc = target.timing_constraints()
    align = max(tc.granularity, tc.pulse_alignment)
    t0 = tq.estimate_duration(target, unit="s") - n * old * target.dt
    new = max(0, math.floor((t_ref - t0) / n / target.dt / align) * align)
    out = tq.copy_empty_like()
    for i in tq.data:
        op = Delay(new, unit="dt") if i.operation.name == "delay" else i.operation
        out.append(op, i.qubits, i.clbits)
    return out


def check_hashes(cfg: dict[str, Any], built: dict[str, Any]) -> dict[str, Any]:
    """#20's committed hashes vs the rebuilt circuits; D1 against its committed file."""
    ref = json.loads((rdir(cfg, "ablation_dir") / "circuits.json").read_text())["qasm_sha256"]
    from qiskit import qasm3

    rows = {}
    for name, h in built["sha256"].items():
        if name.endswith("_committed_file"):
            continue
        if name in ref:
            rows[name] = {"expected": ref[name], "got": h, "match": ref[name] == h}
        else:
            text = (REPO_ROOT / cfg["arms"][name]["file"]).read_text()
            same = qasm3.dumps(built["circuits"][name]) == text
            rows[name] = {"expected": "committed file", "got": h, "match": same}
    return {"rows": rows, "all_match": all(r["match"] for r in rows.values())}


def compare_retimed(snap: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Which circuits change when delays come from the target, and is it delays only?"""
    out = {}
    for name in snap["circuits"]:
        same = snap["sha256"][name] == live["sha256"][name]
        delay_only = strip_delays(snap["signature"][name]) == strip_delays(live["signature"][name])
        out[name] = {"unchanged": same, "differs_only_in_delays": delay_only}
    return out


def target_calibration(target: Any, qubits: list[int], pair: tuple[int, int]) -> dict[str, Any]:
    """Readout error, T1, T2 per qubit and the CZ error of ``pair``, from a target."""
    out: dict[str, Any] = {}
    for q in qubits:
        qp = target.qubit_properties[q]
        m = target["measure"].get((q,))
        out[f"q{q}"] = {
            "readout_error": None if m is None else m.error,
            "t1": qp.t1,
            "t2": qp.t2,
        }
    cz = target["cz"].get(pair) or target["cz"].get(pair[::-1])
    out[f"cz_{pair[0]}_{pair[1]}"] = {"cz_error": None if cz is None else cz.error}
    return out


def reference_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    """#16's snapshot (q59, q75 and their CZ); q74 has none, so only absolute bounds apply."""
    info = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
    info = info["prepare"]["calibration"]["info"]
    ref = {}
    for q in ("59", "75"):
        ref[f"q{q}"] = {
            "readout_error": info["readout_error"][q],
            "t1": info["t1_s"][q],
            "t2": info["t2_s"][q],
        }
    ref["cz_59_75"] = {"cz_error": info["two_qubit_error"]["59-75"]}
    return ref


def calibration_report(cfg: dict[str, Any], target: Any) -> dict[str, Any]:
    c = cfg["calibration"]
    live = target_calibration(
        target, c["qubits"], (cfg["layout"]["output"], cfg["layout"]["memory"])
    )
    b = F.CalibrationBounds(c["factor"], c["abs_readout_max"], c["abs_t1_min_s"], c["abs_t2_min_s"])
    ref = reference_calibration(cfg)
    return {"live": live, "reference": ref, "violations": F.calibration_check(live, ref, b)}


def live_pilot_predictions(cfg: dict[str, Any], noise_json: dict[str, Any]) -> dict[str, float]:
    """Pilot-gate predictions from a calibration's noise model (the live one in Stage 4):
    the baseline's P(1) after the burn-in and A2's P(m = 1), via #20's pieces."""
    from qiskit import qasm3

    acfg = ablation_cfg(cfg)
    lay = acfg["layout"]
    step = qasm3.loads((PA.p6dir(acfg) / "step.qasm").read_text())
    noise = PA.noise_from_json(noise_json)
    pc = A.export_pieces(step, noise, lay["step_memory"], lay["step_output"])
    base = A.process(A.step_superops("baseline", pc))
    acfg = copy.deepcopy(acfg)
    acfg["a2_record"]["readout_error"] = dict(noise_json["readout"])[lay["step_memory"]]
    rec = PA.a2_record(acfg, {"calibrated": pc})
    burn = cfg["analysis"]["burn"]
    return {
        "baseline_p1": float(base.marginals(cfg["layout"]["length"])[burn:].mean()),
        "A2_p_m1": rec["p_m1"],
    }


def decode(res_pub: Any, arm: str, length: int) -> dict[str, np.ndarray]:
    """``y`` (shots, length) from registers ``y0..``; A2's ``m`` and A2s_spec's ``s_0`` kept."""
    import phase6

    out = {"y": phase6.shots_from_result([res_pub], length)}
    data = res_pub.data
    for reg in ("m", "s_0"):
        if hasattr(data, reg):
            out[reg] = getattr(data, reg).to_bool_array(order="little").astype(np.uint8)
    return out


def execution_seconds(res: Any) -> float | None:
    """Total seconds of the job's execution spans (``None`` if the result has none)."""
    try:
        spans = res.metadata["execution"]["execution_spans"]
        return float(sum((sp.stop - sp.start).total_seconds() for sp in spans))
    except Exception:
        return None


def pubs_for(jobs_pubs: list[F.Pub], circuits: dict[str, Any]) -> list[tuple]:
    out = []
    for p in jobs_pubs:
        name = f"A3_{p.instance:03d}" if p.arm == "A3" else p.arm
        out.append((circuits[name], None, p.shots))
    return out


def aer_pilot(cfg: dict[str, Any], built: dict[str, Any], out: Path, shots: int) -> dict:
    """The pilot job's circuits through local Aer (FakeKingston noise) via Phase 6's job path."""
    import phase6
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mode = AerSimulator.from_backend(FakeKingston())
    pubs = [F.Pub(0, i, a, 0, shots) for i, a in enumerate(cfg["pilot"]["arms"])]
    d = out / "aer_pilot"
    d.mkdir(exist_ok=True)
    extra = {"shots": [p.shots for p in pubs], "arms": [p.arm for p in pubs], "aer": True}
    job, rec = phase6.submit_or_fetch(
        "aer_pilot", d, out, None, mode, pubs_for(pubs, built["circuits"]), cfg, extra, None, True
    )
    res = phase6.finish_job(job, rec, d, None, None, True)
    lw = cfg["layout"]["length"]
    s = analysis_settings(cfg)
    summ = {}
    for p, r in zip(pubs, res, strict=True):
        dec = decode(r, p.arm, lw)
        y = dec["y"]
        summ[p.arm] = {
            "shots": len(y),
            "p1_after_burn": float(y[:, s.burn :].mean()),
            "guard_flags": F.guard(y, None, s, 0)["flags"],
            "registers": sorted(dec),
        }
        if "m" in dec:
            summ[p.arm]["p_m1"] = float(dec["m"][:, s.burn :].mean())
        np.savez_compressed(
            d / f"{p.arm}.npz", **{k: np.packbits(v, axis=1) for k, v in dec.items()}
        )
    return summ


# -- the local dry run ---------------------------------------------------------------------------


def dry_run_local(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    out = (args.from_run or new_run_dir(cfg["experiment"]["name"] + "_dryrun")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(plain(cfg), cfg["schedule"]["seed"])
    save_json(out / "config_meta.json", meta)
    log(out, f"run dir {out} (local dry run: no network)")
    sd = rdir(cfg, "summary_dir")
    sd.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"meta": meta, "run_dir": str(out.relative_to(REPO_ROOT))}

    # schedule
    jobs = schedule(cfg)
    arms, inst = plan_arms(cfg)
    problems = F.check_schedule(
        jobs, arms, cfg["schedule"]["tier1"], cfg["schedule"]["max_chunk"], inst
    )
    rows = F.schedule_rows(jobs)
    dur = durations(cfg)
    t_job: dict[int, float] = {}
    for r in rows:
        r["expected_span_s"] = r["shots"] * (cfg["budget"]["rep_delay_s"] + dur[r["arm"]])
        r["expected_start_s"] = t_job.get(r["job"], 0.0)
        t_job[r["job"]] = r["expected_start_s"] + r["expected_span_s"]
    write_csv(sd / "schedule.csv", rows)
    write_csv(sd / "schedule_pilot.csv", F.schedule_rows([pilot_schedule(cfg)]))
    summary["schedule"] = {
        "problems": problems,
        "n_pubs": len(rows),
        "pubs_per_job": [len(j) for j in jobs],
        "shots_per_job": [sum(p.shots for p in j) for j in jobs],
        "expected_job_execution_s": t_job,
    }
    log(out, f"schedule: {summary['schedule']}")

    # budget
    bt = budget_table(cfg)
    write_csv(sd / "budget.csv", bt["rows"])
    summary["budget"] = {k: v for k, v in bt.items() if k != "rows"}
    log(out, f"budget: {json.dumps(summary['budget'], default=str)}")

    # circuits (hardware group)
    if args.skip_circuits:
        summary["circuits"] = "skipped (--skip-circuits)"
    else:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
        except ImportError:
            FakeKingston = None
        if FakeKingston is None:
            summary["circuits"] = "skipped: the hardware group is not installed"
            log(out, summary["circuits"])
        else:
            backend = FakeKingston()
            built = build_circuits(cfg, backend, "snapshot")
            hc = check_hashes(cfg, built)
            retimed = build_circuits(cfg, backend, "target")
            cmp = compare_retimed(built, retimed)
            cal = calibration_report(cfg, backend.target)
            s6 = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
            lp = live_pilot_predictions(cfg, s6["prepare"]["calibration"]["noise"])
            summary["circuits"] = {
                "hash_check": hc,
                "retimed_on_fakekingston_target": {
                    "delays_s": retimed["delays_s"],
                    "snapshot_delays_s": built["delays_s"],
                    "changed": sorted(k for k, v in cmp.items() if not v["unchanged"]),
                    "all_changes_delay_only": all(
                        v["differs_only_in_delays"] for v in cmp.values()
                    ),
                },
                "duration_fakekingston_s": {
                    k: v
                    for k, v in built["duration_s"].items()
                    if not k.startswith("A3_0") or k == "A3_000"
                },
                "calibration_check_on_fakekingston": cal,
                "pilot_predictions_from_snapshot_noise": lp,
            }
            log(
                out,
                f"hashes all match: {hc['all_match']}; retimed changes: "
                f"{summary['circuits']['retimed_on_fakekingston_target']['changed']}",
            )
            log(
                out,
                f"pilot predictions (snapshot noise): {lp}; "
                f"FakeKingston calibration violations: {cal['violations']}",
            )
            if args.aer_shots:
                t0 = time.time()
                summary["aer_pilot"] = aer_pilot(cfg, built, out, args.aer_shots)
                summary["aer_pilot"]["seconds"] = time.time() - t0
                log(out, f"aer pilot: {summary['aer_pilot']}")

    # exact predictions
    exact = {h["name"]: exact_hypothesis(cfg, h) for h in cfg["hypotheses"]}
    save_json(out / "exact.json", exact)
    log(out, "exact predictions done")

    # validation
    reps = args.reps if args.reps is not None else cfg["validation"]["reps"]
    cache = out / "validation.json"
    if cache.exists():
        vrows = json.loads(cache.read_text())
    else:
        vrows = validation(cfg, exact, out, reps, cfg["experiment"]["workers"])
        cache.write_text(json.dumps(vrows))
    vsum = summarise_validation(cfg, exact, vrows)
    write_csv(sd / "validation_replicates.csv", vrows)
    write_csv(sd / "validation.csv", vsum)
    preds = prediction_rows(cfg, exact, vsum)
    write_csv(sd / "predictions.csv", preds)
    summary["validation"] = vsum
    summary["ladder"] = LADDER
    save_json(out / "summary.json", summary)
    save_json(sd / "summary.json", summary)
    shutil.copy(out / "log.txt", sd / "log.txt")
    log(out, "done")


# -- Stage 4: live dry run and submission (never run in issue #24) -------------------------------


def live(cfg: dict[str, Any]) -> dict[str, Any]:
    """Service, backend, plan and usage (network: Stage 4 only)."""
    import phase6

    svc = phase6.service(cfg)
    backend = svc.backend(cfg["ibm"]["backend"])
    crn = svc.active_instance()
    plan = next((i.get("plan") for i in svc.instances() if i.get("crn") == crn), None)
    return {"svc": svc, "backend": backend, "plan": plan, "usage": phase6.usage(svc)}


def stage4(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """``--dry-run`` (no submission) or ``--submit`` (guarded) against the live backend."""
    import phase6

    if args.submit:
        F.submission_check(cfg["ibm"].get("approval_url"), True)  # before any network call
    name = cfg["experiment"]["name"] + ("" if args.submit else "_livedry")
    out = (args.from_run or new_run_dir(name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config_meta.json", run_metadata(plain(cfg), cfg["schedule"]["seed"]))
    try:
        ctx = live(cfg)
        backend = ctx["backend"]
        log(out, f"plan {ctx['plan']!r}; usage {ctx['usage']}")
        ff = 1e-6
        phys = [cfg["layout"]["output"], cfg["layout"]["memory"]]
        cal0 = phase6.calibration(backend, phys, ff)
        rep = calibration_report(cfg, backend.target)
        save_json(out / "calibration_before.json", {"snapshot": cal0, "check": rep})
        if rep["violations"]:
            raise SystemExit(f"calibration outside the bounds, ask the owner: {rep['violations']}")
        built = build_circuits(cfg, backend, "target")
        snap = build_circuits(cfg, backend, "snapshot")
        hc = check_hashes(cfg, snap)
        cmp = compare_retimed(snap, built)
        if not hc["all_match"] or not all(v["differs_only_in_delays"] for v in cmp.values()):
            raise SystemExit("circuit check failed: rebuilt ISA circuits differ from #20's")
        save_json(
            out / "circuits.json", {"hash_check": hc, "retimed": cmp, "sha256": built["sha256"]}
        )
        lp = live_pilot_predictions(cfg, cal0["noise"])
        save_json(out / "pilot_predictions.json", lp)
        cm = cost_model(cfg)
        dur = {k: v for k, v in durations(cfg).items()}
        for arm in ("baseline", "A1", "A2", "A2s_spec", "D1"):
            if arm in built["duration_s"]:
                dur[arm] = built["duration_s"][arm]
        dur["A3"] = max(v for k, v in built["duration_s"].items() if k.startswith("A3_"))
        if not args.submit:
            log(out, f"live dry run: {lp}; durations {dur}; nothing submitted")
            return
        run_jobs(cfg, out, ctx, built, cm, dur, lp)
    except BaseException as e:
        log(out, f"FAILED: {e!r}\n{traceback.format_exc()}")
        raise


def spent_upper(out: Path) -> float:
    tot = 0.0
    for p in out.glob("*/job.json"):
        rec = json.loads(p.read_text())
        if "job_id" in rec and not rec.get("dry_run"):
            tot += rec.get("upper_s", 0.0)
    return tot


def run_jobs(cfg, out: Path, ctx, built, cm: F.CostModel, dur, lp) -> None:
    """Pilot (with its gate), then the main jobs; each guarded; idempotent by job id."""
    import phase6

    svc, backend = ctx["svc"], ctx["backend"]
    lw = cfg["layout"]["length"]
    b = cfg["budget"]
    phys = [cfg["layout"]["output"], cfg["layout"]["memory"]]
    if (out / "pilot" / "gate_failed.json").exists():
        raise SystemExit("the pilot gate failed in this run: stop; never retried automatically")
    stages = [("pilot", pilot_schedule(cfg))] + [
        (f"job{j}", job) for j, job in enumerate(schedule(cfg))
    ]
    for stage, pubs in stages:
        d = out / stage
        d.mkdir(exist_ok=True)
        if (d / "shots.npz").exists():
            log(out, f"[{stage}] shots saved; skipping")
            continue
        cost = cm.job(pubs, dur)

        def guard(rec: dict[str, Any], cost=cost, stage=stage) -> None:
            u = phase6.usage(svc)
            rec["usage_before"] = u
            rec["plan"] = ctx["plan"]
            rec["calibration_before"] = phase6.calibration(svc.backend(backend.name), phys, 1e-6)
            rep = calibration_report(cfg, svc.backend(backend.name).target)
            rec["calibration_check"] = rep
            if rep["violations"]:
                raise SystemExit(f"[{stage}] calibration outside the bounds: {rep['violations']}")
            F.budget_check(
                cost["estimate_s"],
                cost["upper_s"],
                b["max_qpu_seconds"],
                u.get("usage_remaining_seconds"),
                ctx["plan"],
                spent_upper(out),
                b["hard_cap_s"],
                cfg["ibm"]["allowed_plans"],
            )

        extra = {**cost, "pubs": [p.key for p in pubs], "approval_url": cfg["ibm"]["approval_url"]}
        job, rec = phase6.submit_or_fetch(
            stage, d, out, svc, backend, pubs_for(pubs, built["circuits"]), cfg, extra, guard, False
        )
        after = lambda: phase6.calibration(svc.backend(backend.name), phys, 1e-6)  # noqa: E731
        res = phase6.finish_job(job, rec, d, svc, after, False)
        rec["execution_seconds"] = execution_seconds(res)
        rec["execution_spans"] = repr(res.metadata.get("execution", {}).get("execution_spans"))
        save_json(d / "job.json", rec)
        arrays = {}
        for p, r in zip(pubs, res, strict=True):
            for k, v in decode(r, p.arm, lw).items():
                arrays[f"{p.key}__{k}"] = np.packbits(v, axis=1)
        np.savez_compressed(d / "shots.npz", length=lw, **arrays)
        log(out, f"[{stage}] saved {len(pubs)} pubs; usage {rec.get('usage')}")
        if stage == "pilot":
            gate = pilot_gate(cfg, d, pubs, lp, rec, cost, dur)
            save_json(d / "gate.json", gate)
            if not gate["go"]:
                save_json(d / "gate_failed.json", gate)
                raise SystemExit(f"pilot gate failed: {gate}; stop and report (no retry)")
    log(out, "all jobs done; analyse with --analyse-run")


def pilot_gate(cfg, d: Path, pubs, lp, rec, cost, dur) -> dict[str, Any]:
    """The four pre-registered pilot conditions (a)-(d)."""
    p = cfg["pilot"]
    s = analysis_settings(cfg)
    z = np.load(d / "shots.npz")
    lw = int(z["length"])
    out: dict[str, Any] = {}
    by = {}
    for pub in pubs:
        y = W.unpack_shots(z[f"{pub.key}__y"], lw)
        by.setdefault(pub.arm, []).append(y)
        if f"{pub.key}__m" in z:
            by.setdefault("A2_m", []).append(W.unpack_shots(z[f"{pub.key}__m"], lw))
    y = {k: np.concatenate(v) for k, v in by.items()}
    flags = {a: F.guard(y[a], None, s, 0) for a in ("baseline", "A2")}
    out["a_decoding_and_guards"] = all(not f["integrity_fail"] for f in flags.values()) and all(
        y[a].shape == (p["arms"][a], lw) for a in ("baseline", "A2")
    )
    p1 = float(y["baseline"][:, s.burn :].mean())
    out["b_p1"] = {
        "got": p1,
        "pred": lp["baseline_p1"],
        "ok": abs(p1 - lp["baseline_p1"]) <= p["p1_tolerance"],
    }
    pm = float(y["A2_m"][:, s.burn :].mean())
    out["c_pm1"] = {
        "got": pm,
        "pred": lp["A2_p_m1"],
        "ok": abs(pm - lp["A2_p_m1"]) <= p["pm1_tolerance"],
    }
    ex = rec.get("execution_seconds")
    per_model = (cost["shot_s"]) / cost["shots"]
    per_got = None if ex is None else ex / cost["shots"]
    out["d_time"] = {
        "got_s_per_shot": per_got,
        "model_s_per_shot": per_model,
        "ok": per_got is not None and abs(per_got / per_model - 1) <= p["time_tolerance"],
    }
    out["go"] = bool(
        out["a_decoding_and_guards"]
        and out["b_p1"]["ok"]
        and out["c_pm1"]["ok"]
        and out["d_time"]["ok"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-local", action="store_true", help="no network (this issue)")
    mode.add_argument("--dry-run", action="store_true", help="Stage 4: live target, no submit")
    mode.add_argument("--submit", action="store_true", help="Stage 4: submit (needs approval_url)")
    ap.add_argument("--from-run", type=Path)
    ap.add_argument("--aer-shots", type=int, default=0, help="local Aer pilot, shots per arm")
    ap.add_argument("--skip-circuits", action="store_true")
    ap.add_argument("--reps", type=int, default=None, help="override validation replicates")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["experiment"]["workers"] > 2:
        sys.exit("at most 2 workers on the shared machine")
    if args.dry_run_local:
        dry_run_local(cfg, args)
    else:
        stage4(cfg, args)


if __name__ == "__main__":
    main()
