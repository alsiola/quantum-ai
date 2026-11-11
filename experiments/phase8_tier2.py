"""Phase 8 Stage 5d (issue #40): Tier 2 main's pre-registered analysis on its committed shots.

No QPU, no IBM service. Reads the six main sub-jobs' shots that ``phase8_hw.py --submit --tier
tier2_main`` committed to ``results/summaries/phase8_hw/tier2_main/`` and runs, unchanged:

* ``--part primary``: section 3.2's primary contrast, #34's pipeline exactly as its null
  calibration ran it (``phase8_coherence_null.pipeline_tasks``: parity split within each time
  block, d = 4 from the Klein and random starts, d = 6 adding the octahedron and split starts,
  1,500 EM steps, the quantum refit with #32's Nelder-Mead budget, the block bootstrap by
  time block and 8-shot segment with 1,000 resamples, Holm over D = 4, 6 at 0.05).
* ``--part secondary``: the register (Z_8 x 2, 3, 4) and free d = 8 fits, reported only
  (#34 section 4.1 item 2, #35 section 4.2 item 3), against the refit and the fixed model.
* ``--part decide``: the legs recomputed from the committed shots (they must equal the live
  run's), the primary's z values, and section 3.7's call (``P.tier2_decide``), with the
  headline naming the excluded class (#34 section 4.1 item 2).

Every fit is a resumable task (``--from-run <dir>``); at most 2 workers.

    uv run python experiments/phase8_tier2.py \
        --config experiments/configs/phase8_hw.toml --part primary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import phase8_coherence as PC
import phase8_coherence_null as NULL
import phase8_hw as H
import phase8_live as L

from qmem import coherence as Q
from qmem import prereg as P
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

#: the excluded class, as #34 section 4.1 item 2 (and #35 section 4.2 item 2) state it
EXCLUDED = (
    "element-wise classical hidden-state models with d <= 6 (not a classical phase register, "
    "which is excluded only conditional on the quantum model's specification, and not free "
    "d = 8)"
)


def main_dir(cfg: dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["ibm"]["live_summary_dir"] / "tier2_main"


def load_main(cfg: dict[str, Any]) -> tuple[list[L.LiveStage], dict[str, dict]]:
    """Main's stages (the pre-registered schedule) and their committed shots; each sub-job's
    pubs must be the ones its ``job.json`` records."""
    stages = L.tier2_stages(cfg, "tier2_main")
    d = main_dir(cfg)
    arrays = {}
    for st in stages:
        rec = json.loads((d / f"{st.stage}_job.json").read_text())
        if rec["pubs"] != [p.key for p in st.pubs]:
            raise ValueError(f"{st.stage}: committed pubs differ from the schedule")
        with np.load(d / f"{st.stage}_shots.npz") as z:
            arrays[st.stage] = {k: z[k] for k in z.files}
    return stages, arrays


def write_hw_data(cfg: dict[str, Any], ccfg: dict[str, Any], run: Path) -> dict[str, Any]:
    """``hw_data.npz``: per arm, the 8,000 shots in analysis order (block 0, then block 1; in
    each, sub-job then pub position), with the program choices and A2's record; the split
    sees each block as one job of 4,000 (A1)."""
    stages, arrays = load_main(cfg)
    arms = Q.parse_arms(ccfg["arms"])
    rows = L.tier2_main_rows(stages, arrays, arms)
    sp = NULL.split_of(ccfg)
    out: dict[str, np.ndarray] = {}
    for a, chunks in rows.items():
        chunks.sort(key=lambda c: (c[0], c[1], c[2]))
        for b in range(sp.jobs):
            n_b = sum(len(c[3]) for c in chunks if c[0] == b)
            if n_b != sp.per_job:
                raise ValueError(f"{a}: block {b} has {n_b} shots, not {sp.per_job}")
        out[f"{a}|y"] = np.concatenate([c[3] for c in chunks]).astype(np.uint8)
        out[f"{a}|choice"] = np.concatenate([c[4] for c in chunks]).astype(np.int64)
        if chunks[0][5] is not None:
            out[f"{a}|m"] = np.concatenate([c[5] for c in chunks]).astype(np.uint8)
    np.savez_compressed(run / "hw_data.npz", **out)
    meta = {"source_id": int(L.live_cfg(cfg, "tier2_main")["primary_source_id"])}
    save_json(run / "hw_meta.json", meta)
    return meta


def primary(cfg: dict[str, Any], ccfg: dict[str, Any], run: Path, done: dict) -> dict[str, Any]:
    """Section 3.2's decision against the refit (the registered evaluator), with the fixed
    model's deficit reported; in ``P.tier2_primary``'s shape for ``P.tier2_decide``."""
    r = NULL.pipeline_result(ccfg, run, done, NULL.HW, 0, ("refit", "fixed"))
    by_d = {}
    for D in (4, 6):
        x = r["refit"][f"d{D}"]
        by_d[str(D)] = {
            "fit": x["fit"],
            "mean": x["mean"],
            "se_block": x["se_block"],
            "z": x["z"],
            "p": x["p"],
            "reject": bool(x["reject_holm"]),
            "fixed_mean": r["fixed"][f"d{D}"]["mean"],
            "fixed_z": r["fixed"][f"d{D}"]["z"],
        }
    q = done[f"{NULL.HW}_r0_qref"]
    return {
        "by_d": by_d,
        "rejected": [D for D in (4, 6) if by_d[str(D)]["reject"]],
        "evaluator": "refit (section 3.2); fixed reported",
        "alpha": ccfg["null"]["alpha"],
        "pipeline": r,
        "refit_params": q["refit_params"],
        "refit_evals": q["evals"],
        "fits": {
            k: {f: v.get(f) for f in ("iters", "converged", "last_gain_per_symbol")}
            | {"train_nll_per_shot": v.get("train_nll_per_shot")}
            for k, v in done.items()
            if v.get("kind") == "fit"
        },
    }


def secondary(ccfg: dict[str, Any], run: Path, done: dict) -> list[dict[str, Any]]:
    """The register and free d = 8 classes against the refit and the fixed model (reported,
    never part of the call)."""
    p = f"{NULL.HW}_r0"
    arms = list(Q.parse_arms(ccfg["arms"]))
    q = NULL._load(run, f"{p}_qref")
    rc = ccfg["register"]
    fams = {
        "free d<=4": [f"{p}_d4_klein", f"{p}_d4_random"],
        "free d<=6": [f"{p}_d4_klein", f"{p}_d4_random", f"{p}_d6_octahedron", f"{p}_d6_split"],
    }
    fams["free d<=8"] = fams["free d<=6"] + [f"{p}_d8_cube", f"{p}_d8_split"]
    reg: list[str] = []
    for n_pop in rc["n_pop"]:
        reg = reg + [f"{p}_reg{n_pop}_{s}" for s in rc["starts"]]
        fams[f"register Z8 x {n_pop} (d = {8 * n_pop})"] = list(reg)
    rows = []
    for name, labs in fams.items():
        if not all(lab in done for lab in labs):
            continue
        b = NULL.best_fit(done, labs)
        row: dict[str, Any] = {
            "class": name,
            "best": b,
            "n_params": done[b]["n_params"],
            "iters": done[b]["iters"],
            "train_nll_per_shot": done[b]["train_nll_per_shot"],
        }
        for ev in ("refit", "fixed"):
            t = NULL.deficit(ccfg, run, b, q, ev, arms)
            row[f"{ev}_heldout"] = t["mean"]
            row[f"{ev}_se_block"] = t["se_block"]
            row[f"{ev}_z_block"] = t["z"]
        rows.append(row)
    return rows


def legs_from_committed(cfg: dict[str, Any], ccfg: dict[str, Any]) -> dict[str, Any]:
    stages, arrays = load_main(cfg)
    r = L.tier2_main_legs(cfg, L.tier2_main_rows(stages, arrays, Q.parse_arms(ccfg["arms"])))
    r.pop("full")
    return r


def decide(cfg: dict[str, Any], ccfg: dict[str, Any]) -> dict[str, Any]:
    sd = main_dir(cfg)
    prim = json.loads((sd / "primary.json").read_text())
    pilot = json.loads(
        (REPO_ROOT / cfg["ibm"]["live_summary_dir"] / "tier2/decision.json").read_text()
    )
    legs = legs_from_committed(cfg, ccfg)
    live = json.loads((sd / "decision.json").read_text())
    same = H.r6(legs["legs"]) == live["legs"]
    call = P.tier2_decide(prim, legs["legs"], bool(pilot["gate"]["go"]))
    supported = call.startswith("supported")
    head = (
        f"Claim 3 {call.split(':')[0]}: the excluded class is {EXCLUDED}"
        if supported or call.startswith("partial")
        else f"Claim 3 {call}. Nothing is excluded by this tier; the class it would have "
        f"excluded is {EXCLUDED}"
    )
    out = {
        "call": call,
        "headline": head,
        "excluded_class": EXCLUDED,
        "pilot_go": bool(pilot["gate"]["go"]),
        "primary": {k: prim["by_d"][k] for k in prim["by_d"]},
        "rejected": prim["rejected"],
        "legs": legs["legs"],
        "legs_match_live_run": bool(same),
        "statistics": legs["statistics"],
        "fixed_prediction": legs["fixed_prediction"],
    }
    sec = sd / "secondary.json"
    if sec.exists():
        out["secondary"] = json.loads(sec.read_text())["classes"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["primary", "secondary", "decide"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--from-run", default=None)
    args = ap.parse_args()
    cfg = H.load_config(Path(args.config))
    ccfg_path = str((REPO_ROOT / cfg["experiment"]["coherence_config"]).resolve())
    ccfg = PC.load_config(Path(ccfg_path))
    sd = main_dir(cfg)
    meta = run_metadata(cfg, int(L.live_cfg(cfg, "tier2_main")["primary_source_id"]))
    if args.part == "decide":
        res = decide(cfg, ccfg) | {"meta": meta}
        save_json(sd / "decision_final.json", H.r6(res))
        print(res["headline"])
        return
    run = Path(args.from_run) if args.from_run else new_run_dir("phase8_tier2")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "hw_data.npz").exists():
        write_hw_data(cfg, ccfg, run)
    workers = min(args.workers, 2)
    tasks = NULL.pipeline_tasks(NULL.HW, 0, stats=args.part == "secondary")
    if args.part == "secondary":
        tasks += NULL.register_tasks(ccfg, NULL.HW, 0)
    done = NULL.schedule(tasks, ccfg_path, workers, run)
    if args.part == "primary":
        res = primary(cfg, ccfg, run, done)
        res |= {"meta": meta, "run_dir": str(run.relative_to(REPO_ROOT))}
        save_json(sd / "primary.json", H.r6(res))
        for D, x in res["by_d"].items():
            print(
                f"d <= {D}: deficit {x['mean']:.5f} nats/shot, z {x['z']:.2f}, reject {x['reject']}"
            )
    else:
        res = {"classes": secondary(ccfg, run, done), "meta": meta}
        res["run_dir"] = str(run.relative_to(REPO_ROOT))
        save_json(sd / "secondary.json", H.r6(res))
        for r in res["classes"]:
            print(f"{r['class']}: vs refit {r['refit_heldout']:.5f} (z {r['refit_z_block']:.1f})")


if __name__ == "__main__":
    main()
