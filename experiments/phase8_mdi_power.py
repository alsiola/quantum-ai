"""Phase 8 Stage 5b (#38): the MDI sub-tier's pre-data power check at a live calibration.

#38 task 3, the way ``phase8_live_power.py`` did it for P&M (#37). The MDI game at n = 1 with
amendment A4's ``reset q74`` before psi_y: predicted payoff and power at the planned shots,
under three models of the step, each at the committed calibration and at the live one:

* **Stage 4c pieces** (#27's analysis model); live: rebuilt (``phase7_hw.live_pieces``) with
  q59's and q75's live T1, T2 and readout.
* **Stage 2 fitted model** (#32's nine noise parameters on the Stage 4c pieces); live: the same
  parameters on the rebuilt pieces, with q75's live readout, T1 and T2. **This is the model the
  go/no-go rule reads.**
* **deficit-matched** (#33: extra q75 dephasing p per step so that w(0) is 0.747 of the
  model's); live: the same p on the rebuilt pieces.

Outside the step, every live variant takes: q75's live readout (its assignment errors scaled
to it), q75's idle through q74's reset at q75's live T1/T2 and q74's live reset duration, q74's
live readout (its assignment errors, and the excitation the reset leaves), q74's live sx error
(its psi_y preparation), and the live 74-75 CZ error (the Bell measurement). The MDI circuits
compile #33's fixed V_1 into the weights (``R.mdi_weights(V_1)``), so the prediction uses it.

The rule (#38): power >= 0.80 under the fitted model at the planned shots: go. Otherwise the
shots may be raised, if A4 fixes them before any data, the job stays <= 20 s estimated and
Tier 1's cumulative counted seconds stay <= 30 s. ``--search`` reports every admissible n = 1
shot count per input pair (n = 0 held at its plan) with its cost and power.

    uv run --group hardware python experiments/phase8_mdi_power.py \\
        --config experiments/configs/phase8_hw.toml \\
        --snapshot results/summaries/phase8_hw/tier1_live_calibration.json --search
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_hw as H7
import phase8_hw as H
import phase8_live_power as LP

from qmem import coherence as Q
from qmem import followup as F
from qmem import prereg as P
from qmem import reference as R
from qmem.records import REPO_ROOT, run_metadata, save_json

MEM, OUT, BRIDGE = 75, 59, 74


def live_reset(dev: dict[str, Any]) -> R.MdiReset:
    """A4's reset at a live snapshot: q75 idles for q74's reset time at q75's T1/T2; q74 is
    left excited with probability its readout error."""
    q74, q75 = dev["qubits"][str(BRIDGE)], dev["qubits"][str(MEM)]
    return R.MdiReset(q74["reset_time"], q75["t1"], q75["t2"], q74["readout"])


def live_spam_y(rc: dict[str, Any], dev: dict[str, Any]) -> R.Spam:
    """q74's trusted-input noise at a live snapshot (``phase8_reference.device_spam``'s form)."""
    import phase8_reference as PR

    return PR.device_spam(dev, BRIDGE, rc["spam"]["gates_per_prep"])


def live_bsm(dev: dict[str, Any]) -> float:
    return 4 / 3 * float(dev["cz"][f"{BRIDGE}-{MEM}"]["error"])


def mc_power(p: np.ndarray, om: np.ndarray, per_pair: int, alpha: float, draws: int, seed: int):
    """Monte Carlo power of ``P.mdi_analyse`` (payoff - z SE > 0) at ``per_pair`` shots."""
    rng = np.random.default_rng(seed)
    hits = sum(
        P.mdi_analyse(rng.binomial(per_pair, p), per_pair, om, alpha)["reject_eb"]
        for _ in range(draws)
    )
    return hits / draws


def predict(cfg, ctx: dict[str, Any], per_pair: int) -> dict[str, Any]:
    """Payoff at n = 1 (and the n = 0 positive control), the per-shot SD, the normal and MC
    power at ``per_pair`` shots per input pair, and the shots for 80%."""
    a = cfg["tier1"]["mdi_alpha"]
    tp = cfg["tier1_power"]
    out: dict[str, Any] = {}
    for n in (0, 1):
        sup = R.n_step_map(ctx["steps"], n)
        om = R.mdi_weights(ctx["v"][n])
        p = R.mdi_probabilities(sup, ctx["spam_x"], ctx["spam_y"], ctx["bsm"], ctx["reset"])
        pay = R.mdi_payoff(om, p)
        sd1 = R.mdi_sd1(om, p)
        row = {"payoff": pay, "sd1": sd1, "fe_fixed": R.bell_fidelity(R.choi(sup), ctx["v"][n])}
        if n == 1:
            shots = 16 * per_pair
            row |= {
                "per_pair": per_pair,
                "shots": shots,
                "power_normal": R.power_normal(pay, sd1, shots, a) if pay > 0 else 0.0,
                "power_mc": mc_power(p, om, per_pair, a, tp["draws"], tp["seed"] + 38),
                "shots_80": R.shots_for_power(pay, sd1, a, 0.8),
            }
        out[f"n{n}"] = row
    return out


def models(cfg, dev: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """name -> the step branches and the SPAM of q75 / q74 / the Bell measurement / the reset.
    ``dev`` None: the committed calibration only."""
    import phase8_reference as PR

    t1 = H.Tier1Ctx(H.sub_cfgs(cfg)["tier1"])
    rc = t1.cfg
    t2 = H.tier2_model_ctx(H.sub_cfgs(cfg)["tier2"])
    committed_reset = PR.mdi_reset(rc, t1.inp["calib"], t1.inp["device"])
    base = {
        "rc": rc,
        "v": t1.v,
        "spam_x": t1.spam,
        "spam_y": t1.spam_y,
        "bsm": t1.bsm_depol,
        "reset": committed_reset,
    }
    fitted = t2["fitted"]
    out = {
        "stage4c (committed)": base | {"steps": t1.ms["stage4c_live"]},
        "stage2 fitted (committed)": base | {"steps": t2["model"](fitted).step([])},
        "deficit-matched (committed)": base | {"steps": t1.ms["live_deficit_matched"]},
    }
    if dev is None:
        return out
    c7 = H7.load_config(REPO_ROOT / "experiments/configs/phase7_hw_4c.toml")
    noise4c = json.loads(
        (REPO_ROOT / "results/summaries/phase7_hw_4c/calibration_before.json").read_text()
    )["snapshot"]["noise"]
    pc, r75 = H7.live_pieces(c7, LP.swap(noise4c, dev, False))
    q75 = dev["qubits"][str(MEM)]
    live = {
        "rc": rc,
        "v": t1.v,
        "spam_x": LP.spam_at(t1.spam, q75["readout"]),
        "spam_y": live_spam_y(rc, dev),
        "bsm": live_bsm(dev),
        "reset": live_reset(dev),
    }
    qm = Q.QuantumModel(pc, r75, fitted, q75["t1"], q75["t2"])
    out |= {
        "stage4c (live)": live | {"steps": R.step_ops(pc, "baseline")},
        "stage2 fitted (live)": live | {"steps": qm.step([])},
        "deficit-matched (live)": live | {"steps": R.step_ops(pc, extra_dephasing=t1.p_def)},
    }
    return out


def mdi_cost(cfg, per_pair_n0: int, per_pair_n1: int, dur: dict[str, float]) -> dict[str, float]:
    """The MDI job's cost (one job, 32 pubs) under the campaign's cost model."""
    pubs = [F.Pub(0, i, "mdi_n0", i, per_pair_n0) for i in range(16)] + [
        F.Pub(0, 16 + i, "mdi_n1", i, per_pair_n1) for i in range(16)
    ]
    return H.cost_model(cfg).job(pubs, dur)


def durations(dev: dict[str, Any] | None) -> dict[str, float]:
    """MDI circuit durations: the larger of the committed FakeKingston schedule (manifest) and
    the plan's analytic one; the live target's schedule, if known, is applied by the live path."""
    import phase8_reference as PR

    rc = PR.load_config(REPO_ROOT / "experiments/configs/phase8_reference.toml")
    d = PR.durations(rc)
    return {k: v for k, v in d.items() if k.startswith("mdi_")}


RULE_MODEL = "stage2 fitted (live)"


def assess(cfg: dict[str, Any], dev: dict[str, Any], search: bool = False) -> dict[str, Any]:
    """The pre-data check at snapshot ``dev``: every model's prediction at the planned shots,
    the planned cost against the per-job limit and Tier 1's cap left after P&M, and ``go``
    (#38's rule on ``RULE_MODEL``). ``search``: every admissible n = 1 shot count per pair."""
    import phase8_live as L

    rc = H.sub_cfgs(cfg)["tier1"]
    plan = rc["plan"]["mdi_shots"]
    per0, per1 = int(plan["mdi_n0"]), int(plan["mdi_n1"])
    ms = models(cfg, dev)
    dur = durations(dev)
    prior = L.prior_counted(cfg, "tier1_mdi")
    cap_left = cfg["tier1"]["hard_cap_s"] - prior
    target = cfg["tier1"]["power_target"]
    res: dict[str, Any] = {
        "calibration_date": dev["calibration_date"],
        "read_at": dev["read_at"],
        "live": {
            q: {k: dev["qubits"][str(q)][k] for k in ("t1", "t2", "readout", "sx", "reset_time")}
            for q in (OUT, BRIDGE, MEM)
        },
        "cz_74_75": dev["cz"][f"{BRIDGE}-{MEM}"],
        "alpha": cfg["tier1"]["mdi_alpha"],
        "planned_per_pair": {"mdi_n0": per0, "mdi_n1": per1},
        "tier1_prior_counted_s": prior,
        "tier1_cap_left_s": cap_left,
        "max_job_estimate_s": cfg["tier1"]["max_job_estimate_s"],
        "planned_cost": mdi_cost(cfg, per0, per1, dur),
        "models": {k: predict(cfg, m, per1) for k, m in ms.items()},
        "rule": f"go if the MDI n = 1 power (normal and MC) >= {target} under '{RULE_MODEL}' "
        "at the planned shots (#38)",
    }
    prim = res["models"][RULE_MODEL]["n1"]
    pc = res["planned_cost"]
    res["cost_ok"] = bool(
        pc["estimate_s"] <= cfg["tier1"]["max_job_estimate_s"] and pc["upper_s"] <= cap_left
    )
    res["go"] = bool(min(prim["power_mc"], prim["power_normal"]) >= target and res["cost_ok"])
    if search:
        rows = []
        for s in range(per1, 4001, 50):
            c = mdi_cost(cfg, per0, s, dur)
            if c["estimate_s"] > cfg["tier1"]["max_job_estimate_s"] or c["upper_s"] > cap_left:
                break
            row = {"per_pair_n1": s, "shots_n1": 16 * s, **c}
            for k in ("stage4c (live)", RULE_MODEL, "deficit-matched (live)"):
                m = ms[k]
                sup = R.n_step_map(m["steps"], 1)
                om = R.mdi_weights(m["v"][1])
                p = R.mdi_probabilities(sup, m["spam_x"], m["spam_y"], m["bsm"], m["reset"])
                pay, sd1 = R.mdi_payoff(om, p), R.mdi_sd1(om, p)
                a = cfg["tier1"]["mdi_alpha"]
                row[f"power {k}"] = R.power_normal(pay, sd1, 16 * s, a) if pay > 0 else 0.0
            rows.append(row)
        res["search"] = rows
        res["max_admissible_per_pair_n1"] = rows[-1]["per_pair_n1"] if rows else None
        first = next((r for r in rows if r[f"power {RULE_MODEL}"] >= target), None)
        res["min_per_pair_n1_for_target"] = None if first is None else first["per_pair_n1"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, help="a live snapshot saved by --dry-run")
    ap.add_argument("--search", action="store_true", help="every admissible n = 1 shot count")
    ap.add_argument("--out", default="tier1_mdi_power.json")
    args = ap.parse_args()
    cfg = H.load_config(Path(args.config))
    dev = json.loads((REPO_ROOT / args.snapshot).read_text())
    res = {
        "meta": run_metadata(cfg, cfg["tier1_power"]["seed"] + 38),
        "snapshot": args.snapshot,
        **assess(cfg, dev, args.search),
    }
    sd = REPO_ROOT / cfg["ibm"]["live_summary_dir"]
    save_json(sd / args.out, H.r6(res))
    for k, m in res["models"].items():
        n1 = m["n1"]
        print(
            f"{k}: n1 payoff {n1['payoff']:.4f} (F_e {n1['fe_fixed']:.4f}) sd1 {n1['sd1']:.3f} "
            f"power {n1['power_normal']:.3f}/{n1['power_mc']:.3f} at {n1['shots']}; "
            f"80% at {n1['shots_80']:.0f}; n0 {m['n0']['payoff']:.4f}"
        )
    pc = res["planned_cost"]
    print(
        f"planned cost: est {pc['estimate_s']:.2f} s, upper {pc['upper_s']:.2f} s; cap left "
        f"{res['tier1_cap_left_s']:.2f} s; go {res['go']}"
    )
    if args.search:
        print(
            f"max admissible n1 per pair {res['max_admissible_per_pair_n1']}; "
            f"min for target {res['min_per_pair_n1_for_target']}"
        )


if __name__ == "__main__":
    main()
