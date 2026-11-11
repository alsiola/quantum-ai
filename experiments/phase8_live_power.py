"""Phase 8 Stage 5 (#37): Tier 1's pre-data power check at a live calibration snapshot.

The coordinator's condition on #37 (issuecomment-6783342126). Before submitting, #33's P&M
prediction and power are re-computed, with q75's and q59's T1, T2 and readout taken from the
live snapshot that ``phase8_hw.py --dry-run`` saved. If the power of P&M at n = 1 is at least
0.8 at the planned 12,000 shots, the job may go. Otherwise wait for a new calibration, for up to
48 h. This decides only *when* to submit, before any data, so the analysis is unchanged.

The model is the Stage 4c noise model (``phase7_hw_4c/calibration_before.json``) with those six
numbers replaced. Its step pieces are rebuilt the way #27 built them (``phase7_hw.live_pieces``,
``hardware`` group), and the rebuild is checked against the committed pieces at the unchanged
calibration. The P&M circuits compile a fixed correction V (#33's, from the Stage 4c pieces), so
the prediction uses that V, not one re-optimised for the new model. A second variant also takes
the live CZ and sx errors.

    uv run --group hardware python experiments/phase8_live_power.py \
        --config experiments/configs/phase8_hw.toml \
        --snapshot results/summaries/phase8_hw/tier1_livedry_calibration.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_hw as H7
import phase8_hw as H
import phase8_reference as PR

from qmem import reference as R
from qmem.records import REPO_ROOT, run_metadata, save_json

STEP_INDEX = {59: 0, 75: 1}  # the step circuit's qubits: (q59, q75) = (0, 1)


def swap(noise: dict[str, Any], dev: dict[str, Any], gates: bool) -> dict[str, Any]:
    """The Stage 4c noise JSON with q59/q75's T1, T2 and readout (and reset, which #27 set to the
    readout) from ``dev``; ``gates``: the CZ and sx errors too (p = 2 r, as #27)."""
    n = copy.deepcopy(noise)
    for key, field in (("t1", "t1"), ("t2", "t2"), ("readout", "readout"), ("reset", "readout")):
        n[key] = [[STEP_INDEX[q], dev["qubits"][str(q)][field]] for q in (59, 75)]
    if gates:
        n["p1"] = [[STEP_INDEX[q], 2 * dev["qubits"][str(q)]["sx"]] for q in (59, 75)]
        cz = 4 / 3 * dev["cz"]["59-75"]["error"]  # depolarising p from the average error
        n["p2"] = [[[0, 1], cz], [[1, 0], cz]]
    return n


def spam_at(base: R.Spam, r: float) -> R.Spam:
    """q75's SPAM with its readout flips scaled to the live average readout error."""
    import dataclasses

    avg = (base.p01 + base.p10) / 2
    k = r / avg if avg > 0 else 1.0
    return dataclasses.replace(base, p01=base.p01 * k, p10=base.p10 * k)


def predict(cfg, t1: H.Tier1Ctx, steps, neg, spam: R.Spam) -> dict[str, Any]:
    rc, pm = t1.cfg, t1.cfg["plan"]["pm_shots"]
    a = cfg["tier1"]["alpha"] / 2  # Holm's first step (as power_tier1.csv)
    tp = cfg["tier1_power"]
    out: dict[str, Any] = {}
    for test, n, br, per in (
        ("pm_n1", 1, None, pm["pm_n1"]),
        ("branch00_n2", 2, [0, 0], pm["pm_n2"]),
    ):
        row = PR.evaluate(rc, steps, n, t1.v[n], spam, None, None, 0.0, branch=br)
        eff = row["pm_score"] - row["pm_threshold"]
        shots = 12 * per
        out[test] = {
            "score": row["pm_score"],
            "threshold": row["pm_threshold"],
            "effect": eff,
            "planned_shots": shots,
            "expected_branch_shots": shots * row["branch_prob"] if br else shots,
            "power_normal": R.power_normal(eff, row["pm_sd1"], shots, a) if eff > 0 else 0.0,
            "power_mc": R.mc_power_pm(
                row["_pm"],
                per,
                row["pm_threshold"],
                a,
                tp["draws"],
                tp["seed"],
                ratio=br is not None,
            ),
            "shots_80": R.shots_for_power(eff, row["pm_sd1"], a, 0.8),
        }
    n0 = PR.evaluate(rc, steps, 0, t1.v[0], spam, None, None, 0.0)
    out["spam_gate_n0"] = {"score": n0["pm_score"], "threshold": cfg["tier1"]["spam_gate"]}
    ng = PR.evaluate(rc, steps, 1, t1.v[1], spam, None, None, 0.0, first=neg)
    out["negative_control"] = {"score": ng["pm_score"], "threshold": ng["pm_threshold"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, help="a live snapshot saved by --dry-run")
    args = ap.parse_args()
    cfg = H.load_config(Path(args.config))
    dev = json.loads((REPO_ROOT / args.snapshot).read_text())
    t1 = H.Tier1Ctx(H.sub_cfgs(cfg)["tier1"])
    c7 = H7.load_config(REPO_ROOT / "experiments/configs/phase7_hw_4c.toml")
    noise4c = json.loads(
        (REPO_ROOT / "results/summaries/phase7_hw_4c/calibration_before.json").read_text()
    )["snapshot"]["noise"]
    # the rebuild at the unchanged calibration must reproduce the committed Stage 4c pieces
    pc0, _ = H7.live_pieces(c7, noise4c)
    s0, s_ref = R.step_ops(pc0, "baseline"), t1.ms["stage4c_live"]
    rebuild_diff = float(max(np.abs(a - b).max() for a, b in zip(s0, s_ref, strict=True)))
    res: dict[str, Any] = {
        "meta": run_metadata(cfg, cfg["tier1_power"]["seed"]),
        "snapshot": args.snapshot,
        "calibration_date": dev["calibration_date"],
        "rebuild_max_diff": rebuild_diff,
        "live": {
            q: {k: dev["qubits"][str(q)][k] for k in ("t1", "t2", "readout", "sx")}
            for q in (59, 75)
        },
        "stage4c": {
            q: {
                k: dict((int(i), v) for i, v in noise4c[k])[STEP_INDEX[q]]
                for k in ("t1", "t2", "readout")
            }
            for q in (59, 75)
        },
        "models": {},
    }
    r75 = dev["qubits"]["75"]["readout"]
    res["models"]["stage4c (committed)"] = predict(
        cfg, t1, s_ref, t1.ms_neg["stage4c_live"], t1.spam
    )
    for label, gates in (("live T1/T2/readout", False), ("live T1/T2/readout + CZ/sx", True)):
        pc, _ = H7.live_pieces(c7, swap(noise4c, dev, gates))
        res["models"][label] = predict(
            cfg, t1, R.step_ops(pc, "baseline"), R.step_ops(pc, "A2"), spam_at(t1.spam, r75)
        )
    prim = res["models"]["live T1/T2/readout"]["pm_n1"]
    res["go"] = bool(prim["power_mc"] >= 0.8 and prim["power_normal"] >= 0.8)
    res["rule"] = "submit if P&M n = 1 power >= 0.8 at 12,000 shots (#37, coordinator)"
    sd = REPO_ROOT / cfg["ibm"]["live_summary_dir"]
    save_json(sd / "tier1_live_power.json", H.r6(res))
    for k, m in res["models"].items():
        print(
            f"{k}: pm_n1 {m['pm_n1']['score']:.4f} (thr {m['pm_n1']['threshold']:.4f}) "
            f"power {m['pm_n1']['power_normal']:.3f}/{m['pm_n1']['power_mc']:.3f}; "
            f"b00 {m['branch00_n2']['score']:.4f} power {m['branch00_n2']['power_mc']:.3f}; "
            f"n0 {m['spam_gate_n0']['score']:.4f}; neg {m['negative_control']['score']:.4f}"
        )
    print(f"rebuild diff {rebuild_diff:.2e}; go {res['go']}")


if __name__ == "__main__":
    main()
