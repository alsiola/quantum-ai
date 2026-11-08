"""Phase 7 Stage 4b (issue #26), step 1: refit q75 under repeated mid-circuit measurement.

#25's pilot stopped at gate (c): A2's q75 record gave P(m = 1) = 0.419 after the burn-in
against 0.4925 predicted from the live calibration. This script fits small extensions of the
calibrated model (``qmem.refit``) to the pilot's joint A2 record (m_t, y_t) by maximum
likelihood, with a whole-shot bootstrap. It checks each fit against the record (step 0, the
steady state, A2's output and the conditional statistics), compares a leakage alternative,
and derives the amended gate (c) prediction and tolerance. No QPU; no IBM service.

    uv run --group hardware python experiments/phase7_refit.py \
        --config experiments/configs/phase7_refit.toml --export     # once, Qiskit
    uv run python experiments/phase7_refit.py --config experiments/configs/phase7_refit.toml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qmem import ablate_hw as A
from qmem import ancilla as AN
from qmem import refit as R
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.spectator import Instr

PIECES = "pieces_pilot25.npz"


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def rdir(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["experiment"][key]


def live_noise(cfg: dict[str, Any]) -> dict[str, Any]:
    cal = json.loads((rdir(cfg, "pilot_dir") / "calibration_before.json").read_text())
    return cal["snapshot"]["noise"]


def readout_q75(noise: dict[str, Any], cfg: dict[str, Any]) -> float:
    return float(dict(noise["readout"])[cfg["layout"]["step_memory"]])


def export(cfg: dict[str, Any]) -> None:
    """#20's pieces rebuilt under #25's live calibration (Qiskit; the hardware group)."""
    import phase7_ablation as PA
    from qiskit import qasm3

    lay = cfg["layout"]
    step = qasm3.loads((rdir(cfg, "phase6_dir") / "step.qasm").read_text())
    noise = live_noise(cfg)
    pc = A.export_pieces(step, PA.noise_from_json(noise), lay["step_memory"], lay["step_output"])
    sd = rdir(cfg, "summary_dir")
    sd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sd / PIECES, **pc.to_dict("pilot25"), readout_q75=np.array(readout_q75(noise, cfg))
    )
    # the gate's own prediction must come back: the calibrated record model at this noise
    pred = R.predicted_p_m1(pc, readout_q75(noise, cfg), 0.0, None, lay["length"], lay["burn"])
    print(f"exported {sd / PIECES}; calibrated P(m = 1) after burn {pred:.6f}")


def load_pilot(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    lay = cfg["layout"]
    z = np.load(rdir(cfg, "pilot_dir") / "shots_pilot.npz")
    lw = int(z["length"])
    k = lay["pilot_key_A2"]
    return {
        "m": W.unpack_shots(z[f"{k}__m"], lw),
        "y": W.unpack_shots(z[f"{k}__y"], lw),
        "baseline": W.unpack_shots(z[f"{lay['pilot_key_baseline']}__y"], lw),
    }


def pieces(cfg: dict[str, Any]) -> tuple[A.StepPieces, float]:
    z = np.load(rdir(cfg, "summary_dir") / PIECES)
    return A.StepPieces.from_npz(z, "pilot25"), float(z["readout_q75"])


def model_stats(mod: R.RecordModel, cfg: dict[str, Any]) -> dict[str, float]:
    lay = cfg["layout"]
    marg = mod.marginals(lay["length"])
    out = {"p_m1_step0": float(marg["m"][0]), "p_m1_step1": float(marg["m"][1])}
    out.update({k: float(v) for k, v in mod.pair_stats(lay["length"], lay["burn"]).items()})
    return out


# -- bootstrap (2 workers) ----------------------------------------------------------------------


def _boot_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    data = load_pilot(cfg)
    pc, r = pieces(cfg)
    lay = cfg["layout"]
    rng = np.random.default_rng([cfg["fit"]["seed"], task["chunk"]])
    n = len(data["m"])
    rows = []
    for _ in range(task["n"]):
        i = rng.integers(0, n, n)
        m, y = data["m"][i], data["y"][i]
        row = {"obs": R.record_stats(m, y, lay["burn"])}
        row["obs"]["baseline_p1"] = float(data["baseline"][i][:, lay["burn"] :].mean())
        for fam, x0 in (("decay", [0.12]), ("decay_first", [0.05, 0.12])):
            f = R.fit(m, y, pc, r, fam, x0)
            row[fam] = {
                "params": f.params,
                "pred_p_m1": R.predicted_p_m1(
                    pc,
                    r,
                    f.params["decay"],
                    f.params.get("decay_first"),
                    lay["length"],
                    lay["burn"],
                ),
            }
        rows.append(row)
    return {"rows": rows}


def bootstrap(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    nb, w = cfg["fit"]["n_boot"], cfg["experiment"]["workers"]
    chunks = [nb // w + (1 if c < nb % w else 0) for c in range(w)]
    tasks = [{"cfg": cfg, "chunk": c, "n": k} for c, k in enumerate(chunks)]
    if w <= 1:
        res = [_boot_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(w) as ex:
            res = list(ex.map(_boot_task, tasks))
    return [r for part in res for r in part["rows"]]


# -- the leakage comparator ---------------------------------------------------------------------


def leak_fit(cfg: dict[str, Any], data: dict[str, np.ndarray]) -> dict[str, Any]:
    """q75 leakage under measurement (``qmem.ancilla``: |1> -> |2> per measurement, seeping
    back to |1> per step, |2> read as 0) fitted to the same record: 2 parameters."""
    c0 = AN.config_from_snapshot(live_noise(cfg))
    base = AN.physical_step((rdir(cfg, "phase6_dir") / "step.qasm").read_text())
    steps, done = [], False
    for i in base:
        if i.name == "cz" and not done:
            steps.append(Instr("measure", (75,)))
            done = True
        steps.append(i)
    m, y = data["m"], data["y"]

    def proc(x: np.ndarray) -> W.LinearProcess:
        c = replace(c0, memory_levels=3, memory_leak=x[0], memory_seep=x[1], memory_read2=0.0)
        reg = AN.register_for(c, (59,), 2)
        return AN.to_process(AN.step_superops(steps, reg, c, record=[0, 1]), reg, compress=False)

    def nll(x: np.ndarray) -> float:
        if np.any(x < 0) or np.any(x >= 1):
            return np.inf
        p = proc(x)
        v = np.tile(p.init, (len(m), 1))
        tot = 0.0
        for t in range(m.shape[1]):
            v = np.einsum("nj,njk->nk", v, p.ops[2 * m[:, t] + y[:, t]])
            pr = (v @ p.final).real
            tot -= np.log(pr).sum()
            v = v / pr[:, None]
        return float(tot)

    x, val, _ = AN.nelder_mead(nll, np.array(cfg["fit"]["leak_x0"]), step=0.05, xtol=1e-4)
    p = proc(x)
    lw, burn = cfg["layout"]["length"], cfg["layout"]["burn"]
    v, pm = p.init, []
    for _ in range(lw):
        pm.append(float(((v @ (p.ops[2] + p.ops[3])) @ p.final).real))
        v = v @ p.ops.sum(0)
    return {
        "params": {"leak": float(x[0]), "seep": float(x[1]), "read2": 0.0},
        "n_params": 2,
        "nll": val,
        "p_m1_step0": pm[0],
        "p_m1_step1": pm[1],
        "p_m1": float(np.mean(pm[burn:])),
    }


# -- main ----------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main_fit(cfg: dict[str, Any]) -> None:
    out = new_run_dir(cfg["experiment"]["name"])
    meta = run_metadata(cfg, cfg["fit"]["seed"])
    save_json(out / "config_meta.json", meta)
    sd = rdir(cfg, "summary_dir")
    lay = cfg["layout"]
    data = load_pilot(cfg)
    pc, r = pieces(cfg)
    n = len(data["m"])

    fits = {}
    for fam in cfg["fit"]["families"]:
        k = len(R.FAMILIES[fam])
        f = R.fit(data["m"], data["y"], pc, r, fam, [cfg["fit"]["x0"]] * k or None)
        mod = R.model_of(fam, list(f.params.values()), pc, r)
        fits[fam] = {
            "params": f.params,
            "n_params": f.n_params,
            "nll": f.nll,
            "model": model_stats(mod, cfg),
            "p_m1_by_position": mod.marginals(lay["length"])["m"].tolist(),
        }
        print(fam, f)
    for a, b in (("calibrated", "decay"), ("decay", "decay_first")):
        lr = 2 * (fits[a]["nll"] - fits[b]["nll"])
        df = fits[b]["n_params"] - fits[a]["n_params"]
        fits[b][f"lr_vs_{a}"] = {"lr": lr, "df": df, "p": math.erfc(math.sqrt(lr / 2))}
    leak = leak_fit(cfg, data)
    print("leak", leak)

    boots = bootstrap(cfg)
    obs = R.record_stats(data["m"], data["y"], lay["burn"])
    obs["baseline_p1"] = float(data["baseline"][:, lay["burn"] :].mean())
    se_obs = {k: float(np.std([b["obs"][k] for b in boots], ddof=1)) for k in obs}
    params_b = {
        fam: {k: np.array([b[fam]["params"][k] for b in boots]) for k in fits[fam]["params"]}
        for fam in ("decay", "decay_first")
    }
    pred_b = {fam: np.array([b[fam]["pred_p_m1"] for b in boots]) for fam in params_b}

    # posterior checks: observed minus fitted, in units of the observed whole-shot SE
    check = []
    for k in obs:
        row = {"statistic": k, "observed": obs[k], "se": se_obs[k]}
        for fam in cfg["fit"]["families"]:
            mv = fits[fam]["model"].get(k)
            if k == "baseline_p1":
                mv = None
            row[fam] = mv
            row[f"z_{fam}"] = None if mv is None else (obs[k] - mv) / se_obs[k]
        check.append(row)

    prim = cfg["fit"]["primary"]
    se_pilot = se_obs["p_m1"] * math.sqrt(n / cfg["gate"]["pilot_a2_shots"])
    gate = {}
    for fam in ("decay", "decay_first"):
        sd_pred = float(np.std(pred_b[fam], ddof=1))
        tol = cfg["gate"]["z"] * math.sqrt(sd_pred**2 + se_pilot**2)
        gate[fam] = {
            "pred_p_m1_at_pilot25_calibration": fits[fam]["model"]["p_m1"],
            "sd_prediction_bootstrap": sd_pred,
            "se_500_shot_pilot": se_pilot,
            "tolerance": tol,
            "tolerance_rounded_up": math.ceil(tol * 1000) / 1000,
        }
    # the same refit at #16's snapshot calibration (the pre-registration's reference noise)
    snap = A.StepPieces.from_npz(np.load(rdir(cfg, "ablation_dir") / "pieces.npz"), "calibrated")
    s6 = json.loads((rdir(cfg, "phase6_dir") / "summary.json").read_text())
    r_snap = float(dict(s6["prepare"]["calibration"]["noise"]["readout"])[lay["step_memory"]])
    pp = fits[prim]["params"]
    snap_pred = {
        "calibrated": R.predicted_p_m1(snap, r_snap, 0.0, None, lay["length"], lay["burn"]),
        prim: R.predicted_p_m1(
            snap, r_snap, pp["decay"], pp.get("decay_first"), lay["length"], lay["burn"]
        ),
    }
    summary = {
        "meta": meta,
        "run_dir": str(out.relative_to(REPO_ROOT)),
        "shots": n,
        "readout_q75_live": r,
        "fits": fits,
        "leak_comparator": leak,
        "observed": obs,
        "observed_se": se_obs,
        "bootstrap": {
            "n_boot": len(boots),
            "params": {
                fam: {
                    k: {
                        "mean": float(v.mean()),
                        "sd": float(v.std(ddof=1)),
                        "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                    }
                    for k, v in ps.items()
                }
                for fam, ps in params_b.items()
            },
        },
        "posterior_check": check,
        "gate_c": gate,
        "primary": prim,
        "snapshot_calibration_p_m1": snap_pred,
    }
    save_json(out / "refit.json", summary)
    sd.mkdir(parents=True, exist_ok=True)
    save_json(sd / "refit.json", summary)
    write_csv(sd / "refit_check.csv", check)
    np.savez_compressed(
        sd / "refit_bootstrap.npz",
        **{f"{fam}__{k}": v for fam, ps in params_b.items() for k, v in ps.items()},
        **{f"{fam}__pred_p_m1": v for fam, v in pred_b.items()},
    )
    figure(cfg, data, fits, sd)
    print(json.dumps({"gate_c": gate, "bootstrap": summary["bootstrap"]}, indent=1))


def figure(cfg, data, fits, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = data["m"]
    t = np.arange(m.shape[1])
    se = np.sqrt(m.mean(0) * (1 - m.mean(0)) / len(m))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.errorbar(t, m.mean(0), se, fmt="o", ms=3, color="0.2", label="#25 pilot (500 shots)")
    styles = {"calibrated": ("C3", "--"), "decay": ("C0", ":"), "decay_first": ("C0", "-")}
    for fam, (c, ls) in styles.items():
        lab = f"{fam} ({fits[fam]['n_params']} par.)"
        ax.plot(t, fits[fam]["p_m1_by_position"], color=c, ls=ls, label=lab)
    ax.axvline(cfg["layout"]["burn"] - 0.5, color="0.7", lw=0.8)
    ax.set_xlabel("step t")
    ax.set_ylabel("P(m_t = 1), q75 record")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(sd / "refit_m_by_position.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export", action="store_true", help="pieces at #25's live calibration")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["experiment"]["workers"] > 2:
        sys.exit("at most 2 workers on the shared machine")
    if args.export:
        export(cfg)
    else:
        main_fit(cfg)


if __name__ == "__main__":
    main()
