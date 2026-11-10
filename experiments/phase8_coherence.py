"""Phase 8 Stage 2 (issue #32): coherence-specific test design on q75/q59, local only.

Can any feasible experiment make a classical hidden-state model with <= D states fail while
the trained instrument's fixed prediction passes? Parts (see the config's header):

* ``--part review``: #30's review points. (1) A fitted quantum model (``qmem.coherence``:
  the calibrated pieces with nine physical noise parameters) on every Stage 4c arm, its
  likelihood and BIC against #30's classical classes. (2) The pub-parity CV at d = 4, 5, 6 and
  at the fitted quantum model. (3) EM run to convergence (#30's fits all stopped at 800 steps).
* ``--part design``: the instrument's fixed predictions for the phase scan, inverse,
  composition and randomised-dephasing arms; population and coherence benchmarks of q75;
  Knee-style classical-disturbance controls; the arms under #21's and #22's alternatives; the
  budget.
* ``--part power``: data simulated from the fixed and the fitted quantum models at planned
  shots; classical program models (``qmem.classical.ProgramModel``, one free map per element)
  fitted at d = 4 and 6; held-out shots and held-out arms; power to exclude d <= 4, 6.
* ``--export-qasm`` (``hardware`` group, ``FakeKingston`` only): the circuits, the transpiler
  and scheduler checks of the physical phase insertions, QASM with hashes, device numbers.
* ``--plan``: the unsubmitted hardware plan.

No QPU, no IBM service. ``uv run python experiments/phase8_coherence.py --config
experiments/configs/phase8_coherence.toml --part review``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_classical as P8

from qmem import classical as C
from qmem import coherence as Q
from qmem import refit as R
from qmem.ancilla import nelder_mead
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def sdir(cfg: dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def jround(x: Any, nd: int = 7) -> Any:
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else x
    if isinstance(x, dict):
        return {k: jround(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jround(v, nd) for v in x]
    return x


# -- Stage 4c context ---------------------------------------------------------------------------

Q4C = {  # the quantum model's Stage 4c arms (A2's first measurement has its own decay)
    "baseline": C.ArmSpec("baseline"),
    "A2s_spec": C.ArmSpec("A2s_spec", (("spec",),)),
    "A1": C.ArmSpec("A1", (("reset",),)),
    "A2": C.ArmSpec("A2", (("meas0",), ("meas",)), "alternate"),
    "A3": C.ArmSpec("A3", (("I",), ("X",), ("Y",), ("Z",)), "random"),
}


class Ctx:
    """#27's data and #30's calibrated model (``phase8_classical``'s loaders)."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.ccfg = P8.load_config(REPO_ROOT / cfg["experiment"]["classical_config"])
        self.shots = P8.load_shots(self.ccfg)
        self.pt = P8.pauli_table(self.ccfg)
        self.quantum = P8.Quantum(self.ccfg)
        s6 = json.loads((REPO_ROOT / cfg["experiment"]["phase6_dir"] / "summary.json").read_text())
        noise = s6["prepare"]["calibration"]["noise"]
        t1 = {int(k): v for k, v in noise["t1"]}
        t2 = {int(k): v for k, v in noise["t2"]}
        self.t1_q75, self.t2_q75 = float(t1[1]), float(t2[1])

    def data(self, keep: dict[str, np.ndarray] | None = None) -> dict[str, C.ArmData]:
        return P8.arm_data(self.shots, self.pt, keep)

    def fold(self, parity: int) -> dict[str, np.ndarray]:
        return {a: (d["pub"] % 2) == parity for a, d in self.shots.items()}

    def calibrated(self) -> Q.NoiseParams:
        q = self.quantum
        return Q.NoiseParams(meas_decay=q.decay, meas_decay_first=q.decay_first)

    def model(self, p: Q.NoiseParams) -> Q.QuantumModel:
        q = self.quantum
        return Q.QuantumModel(q.pieces, q.readout, p, self.t1_q75, self.t2_q75)


def q4c_nll(ctx: Ctx, qm: Q.QuantumModel, data: dict[str, C.ArmData]) -> dict[str, float]:
    """Total ``-log L`` per Stage 4c arm (A2 jointly with its record; D1 #23's fixed model)."""
    out = {}
    for arm, a in data.items():
        if arm == "D1":
            out[arm] = float(ctx.quantum.d1.seq_nll(a.y).sum())
            continue
        d = C.program_data_from_arm(arm, a)
        ch = Q.first_step_choice(*d.y.shape) if arm == "A2" else None
        out[arm] = float(qm.nll(Q4C[arm], d, ch).sum())
    return out


def q4c_stats(ctx: Ctx, qm: Q.QuantumModel) -> dict[str, Any]:
    """Every Stage 4c statistic of a quantum model (as ``phase8_classical.Quantum.stats``)."""
    cfg = ctx.ccfg
    an = cfg["analysis"]
    length, burn = an["length"], an["burn"]
    out = {}
    for arm in ("baseline", "A2s_spec", "A1"):
        out[arm] = P8.process_stats(qm.averaged(Q4C[arm]), cfg)
    a2 = C.ArmSpec("A2", (("meas",),))
    out["A2"] = P8.process_stats(qm.averaged(a2), cfg)
    out["D1"] = P8.process_stats(ctx.quantum.d1, cfg)
    out["A3"] = P8.tv_stats([qm.time_varying(Q4C["A3"], length, row) for row in ctx.pt], cfg)
    first = np.stack([qm.step(("meas0",), m) for m in (0, 1)])
    ops = np.stack([qm.step(("meas",), m) for m in (0, 1)])
    rec = R.RecordModel(first, ops)
    out["A2"].update(rec.pair_stats(length, burn))
    marg = rec.marginals(length)["m"]
    out["A2"]["p_m1_step0"], out["A2"]["p_m1_step1"] = float(marg[0]), float(marg[1])
    return out


def fit_quantum(
    ctx: Ctx, data: dict[str, C.ArmData], start: Q.NoiseParams | None = None
) -> dict[str, Any]:
    """Nelder-Mead over ``[quantum_fit] params`` on the total NLL of the q75 arms."""
    fq = ctx.cfg["quantum_fit"]
    names = fq["params"]
    base = start or ctx.calibrated()
    pieces = ctx.quantum.pieces
    q75 = {k: v for k, v in data.items() if k != "D1"}
    evals = {"n": 0}

    def f(x: np.ndarray) -> float:
        evals["n"] += 1
        p = Q.from_vector(x, names, base)
        v = sum(q4c_nll(ctx, ctx.model(p), q75).values())
        return v if math.isfinite(v) else 1e30

    x = Q.to_vector(base, names, pieces)
    fx = f(x)
    hist = [fx]
    for _ in range(fq["restarts"] + 1):
        x, fx, _ = nelder_mead(f, x, step=fq["step"], max_evals=fq["max_evals"], ftol=1e-4)
        hist.append(fx)
    p = Q.from_vector(x, names, base)
    return {"params": p.to_dict(), "nll_q75": fx, "evals": evals["n"], "history": hist}


# -- tasks ----------------------------------------------------------------------------------------

_CTX: dict[str, Any] = {}


def _init_worker(cfg_path: str) -> None:
    cfg = load_config(Path(cfg_path))
    _CTX["ctx"] = Ctx(cfg)


def _pdata(ctx: Ctx, keep=None) -> dict[str, C.ProgramData]:
    return {k: C.program_data_from_arm(k, v) for k, v in ctx.data(keep).items()}


def _em(task: dict[str, Any], label: str, model: C.ProgramModel, arms, data, max_iters: int):
    """``fit_program_model`` with a checkpoint every ``checkpoint_every`` cycles (resumable)."""
    rv = _CTX["ctx"].cfg["review"]
    run = Path(task["run"])
    ck = run / f"ckpt_{label}.json"
    done = 0
    if ck.exists():
        c = json.loads(ck.read_text())
        model, done = C.ProgramModel.from_dict(c["model"]), int(c["iters"])
    hist: list[float] = []
    cycles = {"n": 0}

    def cb(it: int, m: C.ProgramModel, nps: float) -> None:
        hist.append(nps)
        cycles["n"] += 1
        if cycles["n"] % rv["checkpoint_every"] == 0:
            ck.write_text(json.dumps({"model": m.to_dict(), "iters": done + it, "nll": nps}))

    t0 = time.time()
    res = C.fit_program_model(arms, data, model, max(max_iters - done, 2), rv["tol"], callback=cb)
    return res, done + res.iters, hist, time.time() - t0


def _test_nll(model: C.ProgramModel, data: dict[str, C.ProgramData]) -> dict[str, float]:
    return {k: float(model.nll(C.STAGE4C_ARMS[k], a).sum()) for k, a in data.items()}


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    ctx: Ctx = _CTX["ctx"]
    rv = ctx.cfg["review"]
    run = Path(task["run"])
    res_path = run / f"task_{task['label']}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    t0 = time.time()
    kind = task["kind"]
    out: dict[str, Any] = {"label": task["label"], "kind": kind}
    if kind == "qfit":
        fold = task.get("fold")
        keep = None if fold is None else ctx.fold(fold)
        train = ctx.data(keep)
        fit = fit_quantum(ctx, train)
        out.update(fit)
        qm = ctx.model(Q.NoiseParams.from_dict(fit["params"]))
        out["nll_by_arm"] = q4c_nll(ctx, qm, ctx.data())
        if fold is not None:
            out["test_nll"] = q4c_nll(ctx, qm, ctx.data(ctx.fold(1 - fold)))
    elif kind == "cv_chain":
        fold = task["fold"]
        train = _pdata(ctx, ctx.fold(fold))
        test = _pdata(ctx, ctx.fold(1 - fold))
        rng = np.random.default_rng(rv["seed"] + fold)
        chain = []
        prev = None
        for d in task["dims"]:
            done_path = run / f"chain_{task['label']}_d{d}.json"
            if done_path.exists():
                c = json.loads(done_path.read_text())
                chain.append(c)
                prev = C.ProgramModel.from_dict(c["model"])
                continue
            if prev is None:
                start = C.ProgramModel.from_intervention(P8.quantum_start(ctx.ccfg, rv["smooth"]))
                if start.d != d:
                    raise ValueError("the chain starts at d = 4 (the quantum start)")
            else:
                start = C.smooth_program(C.split_program_state(prev, rng), rv["smooth"])
            fit, iters, hist, secs = _em(
                task, f"{task['label']}_d{d}", start, C.STAGE4C_ARMS, train, rv["cv_max_iters"]
            )
            prev = fit.model
            chain.append(
                {
                    "d": d,
                    "train_nll": fit.nll,
                    "train_symbols": fit.n_symbols,
                    "iters": iters,
                    "converged": fit.converged,
                    "seconds": secs,
                    "last_gain_per_symbol": (hist[-2] - hist[-1]) if len(hist) > 1 else None,
                    "test_nll": _test_nll(fit.model, test),
                    "test_symbols": {k: a.n_symbols for k, a in test.items()},
                    "model": fit.model.to_dict(),
                }
            )
            save_json(done_path, chain[-1])
        out["fold"] = fold
        out["chain"] = chain
    elif kind == "converge":
        s30 = json.loads((REPO_ROOT / ctx.cfg["experiment"]["classical_summary"]).read_text())
        cls = s30["classes"][f"general_d{task['d']}"]
        m30 = C.InterventionModel.from_dict(cls["model"])
        start = C.ProgramModel.from_intervention(m30)
        data = _pdata(ctx)
        nll0 = -C.program_loglik(start, C.STAGE4C_ARMS, data)
        fit, iters, hist, secs = _em(
            task, task["label"], start, C.STAGE4C_ARMS, data, rv["converge_max_iters"]
        )
        out.update(
            {
                "d": task["d"],
                "nll_800": nll0,
                "nll": fit.nll,
                "gain": nll0 - fit.nll,
                "iters_more": iters,
                "converged": fit.converged,
                "seconds": secs,
                "history_every_cycle": hist[:: max(1, len(hist) // 200)],
                "last_gain_per_symbol": (hist[-2] - hist[-1]) if len(hist) > 1 else None,
                "nll_by_arm_800": _test_nll(start, data),
                "nll_by_arm": _test_nll(fit.model, data),
                "n_params": start.n_params(),
                "n_symbols": fit.n_symbols,
                "model": fit.model.to_dict(),
            }
        )
    else:
        raise ValueError(kind)
    out["seconds_total"] = time.time() - t0
    save_json(res_path, out)
    return out


def run_tasks(tasks: list[dict], cfg_path: str, workers: int, out: Path) -> list[dict]:
    res = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg_path,)) as ex:
        for r in ex.map(_run_task, tasks):
            log(out, f"  done {r['label']} ({r.get('seconds_total', 0):.0f} s)")
            res.append(r)
    return res


# -- part: review ---------------------------------------------------------------------------


def part_review(cfg: dict[str, Any], cfg_path: str, out: Path, workers: int) -> dict[str, Any]:
    rv = cfg["review"]
    tasks = [{"kind": "qfit", "label": "qfit_all"}]
    tasks += [{"kind": "qfit", "label": f"qfit_fold{f}", "fold": f} for f in (0, 1)]
    tasks += [
        {"kind": "cv_chain", "label": f"cv_fold{f}", "fold": f, "dims": rv["cv_dims"]}
        for f in (0, 1)
    ]
    tasks += [
        {"kind": "converge", "label": f"converge_d{d}", "d": d}
        for d in sorted(rv["converge_dims"], reverse=True)
    ]
    for t in tasks:
        t["run"] = str(out)
    log(out, f"review: {len(tasks)} tasks with {workers} workers")
    res = {r["label"]: r for r in run_tasks(tasks, cfg_path, workers, out)}
    log(out, "review: summary")
    return summarise_review(cfg, res)


def summarise_review(cfg: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    ctx = Ctx(cfg)
    data = ctx.data()
    n_sym = sum(a.n_symbols for a in data.values())
    obs = P8.observed(ctx.ccfg, ctx.shots)
    z_fit = cfg["analysis"]["z_fit"]
    s30 = json.loads((REPO_ROOT / cfg["experiment"]["classical_summary"]).read_text())

    cal = ctx.model(ctx.calibrated())
    cal_nll = q4c_nll(ctx, cal, data)
    qa = res["qfit_all"]
    pq = Q.NoiseParams.from_dict(qa["params"])
    fq = ctx.model(pq)
    fq_nll = qa["nll_by_arm"]
    k_q = len(cfg["quantum_fit"]["params"])
    fq_rows = P8.compare(q4c_stats(ctx, fq), obs, ctx.ccfg)
    cal_rows = P8.compare(q4c_stats(ctx, cal), obs, ctx.ccfg)
    fq_total = sum(fq_nll.values())

    def entry(name, nll_total, k, by_arm, extra=None):
        e = {
            "class": name,
            "n_params": k,
            "nll": nll_total,
            "nll_per_symbol": nll_total / n_sym,
            "delta_nll_vs_fitted_quantum": nll_total - fq_total,
            "bic_vs_fitted_quantum": (k - k_q) * math.log(n_sym) + 2 * (nll_total - fq_total),
            "nll_by_arm": by_arm,
        }
        e.update(extra or {})
        return e

    table = [
        entry("quantum_fitted", fq_total, k_q, fq_nll),
        entry("quantum_calibrated", sum(cal_nll.values()), 0, cal_nll),
    ]
    for name, c in s30["classes"].items():
        if name in ("quantum_calibrated", "no_record", "drop_baseline_job1"):
            continue
        table.append(
            entry(f"{name} (#30, 800 steps)", c["nll_all_shots"], c["n_params"], c["nll_by_arm"])
        )
    conv = {}
    for d in cfg["review"]["converge_dims"]:
        r = res[f"converge_d{d}"]
        table.append(
            entry(
                f"general_d{d} (converged run)",
                r["nll"],
                r["n_params"],
                r["nll_by_arm"],
                {"converged": r["converged"], "iters_after_800": r["iters_more"]},
            )
        )
        conv[f"d{d}"] = {
            k: r[k]
            for k in (
                "nll_800",
                "nll",
                "gain",
                "iters_more",
                "converged",
                "last_gain_per_symbol",
                "seconds",
                "nll_by_arm_800",
                "nll_by_arm",
            )
        }
        conv[f"d{d}"]["gain_per_symbol"] = r["gain"] / n_sym
        conv[f"d{d}"]["a3_vs_fitted_quantum"] = r["nll_by_arm"]["A3"] - fq_nll["A3"]
        conv[f"d{d}"]["a3_vs_calibrated_quantum"] = r["nll_by_arm"]["A3"] - cal_nll["A3"]
        m = C.ProgramModel.from_dict(r["model"])
        conv[f"d{d}"]["verdict"] = P8.verdict(
            P8.compare(program_stats_4c(ctx, m), obs, ctx.ccfg), z_fit
        )

    cv = []
    for f in (0, 1):
        ch = res[f"cv_fold{f}"]["chain"]
        test = ctx.data(ctx.fold(1 - f))
        ns = {k: a.n_symbols for k, a in test.items()}
        tot = sum(ns.values())
        q_cal = q4c_nll(ctx, cal, test)
        q_fit = res[f"qfit_fold{f}"]["test_nll"]
        row = {
            "train_parity": f,
            "test_parity": 1 - f,
            "test_symbols": tot,
            "fitted_quantum_params": res[f"qfit_fold{f}"]["params"],
            "quantum_calibrated_per_symbol": sum(q_cal.values()) / tot,
            "quantum_fitted_per_symbol": sum(q_fit.values()) / tot,
            "fitted_minus_calibrated_per_symbol": (sum(q_fit.values()) - sum(q_cal.values())) / tot,
            "fitted_minus_calibrated_by_arm": {k: (q_fit[k] - q_cal[k]) / ns[k] for k in ns},
            "classical": {},
        }
        for c in ch:
            tn = c["test_nll"]
            row["classical"][f"d{c['d']}"] = {
                "iters": c["iters"],
                "converged": c["converged"],
                "per_symbol": sum(tn.values()) / tot,
                "minus_calibrated_per_symbol": (sum(tn.values()) - sum(q_cal.values())) / tot,
                "minus_fitted_per_symbol": (sum(tn.values()) - sum(q_fit.values())) / tot,
                "minus_calibrated_by_arm": {k: (tn[k] - q_cal[k]) / ns[k] for k in ns},
                "minus_fitted_by_arm": {k: (tn[k] - q_fit[k]) / ns[k] for k in ns},
            }
        cv.append(row)
    return {
        "n_symbols": n_sym,
        "quantum_fitted": {
            "params": qa["params"],
            "n_params": k_q,
            "evals": qa["evals"],
            "history": qa["history"],
            "nll_by_arm": fq_nll,
            "nll": fq_total,
            "minus_calibrated": fq_total - sum(cal_nll.values()),
            "minus_calibrated_by_arm": {k: fq_nll[k] - cal_nll[k] for k in fq_nll},
            "verdict": P8.verdict(fq_rows, z_fit),
            "rows": fq_rows,
        },
        "quantum_calibrated": {"nll_by_arm": cal_nll, "verdict": P8.verdict(cal_rows, z_fit)},
        "table": table,
        "convergence": conv,
        "cv": cv,
    }


def program_stats_4c(ctx: Ctx, m: C.ProgramModel) -> dict[str, Any]:
    """Stage 4c statistics of a classical ``ProgramModel`` (via ``InterventionModel``)."""
    im = C.InterventionModel(
        m.init,
        m.pre,
        m.post,
        m.maps["spec"],
        m.maps["reset"],
        m.labelled["meas"],
        np.stack([np.eye(m.d), m.maps["X"], m.maps["Y"], m.maps["Z"]]),
        m.whole["D1"],
    )
    return P8.classical_stats(im, ctx.ccfg, ctx.pt)


def write_review(cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    sd = sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    write_csv(
        sd / "review_classes.csv",
        [
            {k: jround(v) for k, v in r.items() if k != "nll_by_arm"}
            | {f"nll_{a}": round(v, 2) for a, v in r["nll_by_arm"].items()}
            for r in summary["table"]
        ],
    )
    write_csv(
        sd / "review_quantum_fitted_statistics.csv",
        [jround(r) for r in summary["quantum_fitted"]["rows"]],
    )
    slim = json.loads(json.dumps(summary))
    slim["quantum_fitted"].pop("rows")
    save_json(sd / "review.json", jround(slim, 9))


# -- main -------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--part",
        choices=["review", "design", "power", "null", "register", "contrast", "incommensurate"],
    )
    ap.add_argument("--export-qasm", action="store_true")
    ap.add_argument(
        "--export-controls", action="store_true", help="#34's controls tier (hardware group)"
    )
    ap.add_argument(
        "--ancillas",
        default=None,
        help="with --export-qasm: re-export RS, R2, R4 on this live pair, e.g. 13,50 (#39)",
    )
    ap.add_argument("--out-dir", default="results/summaries/phase8_hw/tier2/qasm")
    ap.add_argument("--snapshot", default=None, help="the live snapshot the pair came from")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--from-run", default=None, help="resume a run directory")
    ap.add_argument("--review-run", default=None, help="the review run (fitted quantum model)")
    args = ap.parse_args()
    cfg_path = str(Path(args.config).resolve())
    cfg = load_config(Path(cfg_path))
    workers = args.workers or cfg["experiment"]["workers"]
    if args.export_qasm:
        import phase8_coherence_hw as HW

        if args.ancillas:  # #39 (A5): the feed-forward arms on the live ancillas only
            pair = tuple(int(q) for q in args.ancillas.split(","))
            HW.export_live(cfg, pair, Path(args.out_dir), args.snapshot or "")
        else:
            HW.export(cfg)
        return
    if args.export_controls:
        import phase8_coherence_hw as HW

        HW.export_controls(cfg)
        return
    if args.plan:
        import phase8_coherence_design as DS

        DS.print_plan(cfg)
        return
    out = Path(args.from_run) if args.from_run else new_run_dir(cfg["experiment"]["name"])
    log(out, f"run dir {out}; part {args.part}")
    meta = run_metadata(cfg, cfg["review"]["seed"])
    if args.part == "review":
        summary = part_review(cfg, cfg_path, out, workers)
        summary["meta"] = meta
        save_json(out / "review_summary.json", summary)
        write_review(cfg, summary)
    elif args.part == "design":
        import phase8_coherence_design as DS

        review = Path(args.review_run) if args.review_run else None
        summary = DS.part_design(cfg, out, review, meta)
        summary["meta"] = meta
        save_json(out / "design_summary.json", summary)
    elif args.part == "power":
        import phase8_coherence_design as DS

        summary = DS.part_power(cfg, cfg_path, out, workers, args.review_run, meta)
        summary["meta"] = meta
        save_json(out / "power_summary.json", summary)
    elif args.part in ("null", "register"):
        import phase8_coherence_null as NL

        meta = run_metadata(cfg, cfg["null"]["seed"])
        part = NL.part_null if args.part == "null" else NL.part_register
        summary = part(cfg, cfg_path, out, workers, meta)
        save_json(out / f"{args.part}_summary.json", summary)
    elif args.part == "incommensurate":
        import phase8_coherence_incommensurate as IC

        summary = IC.part_incommensurate(
            cfg, cfg_path, out, workers, run_metadata(cfg, cfg["null"]["seed"])
        )
        save_json(out / "incommensurate_summary.json", summary)
    elif args.part == "contrast":
        import phase8_coherence_contrast as CT

        summary = CT.part_contrast(cfg, out, run_metadata(cfg, cfg["contrast"]["seed"]))
        save_json(out / "contrast_summary.json", summary)
    log(out, "done")


if __name__ == "__main__":
    main()
