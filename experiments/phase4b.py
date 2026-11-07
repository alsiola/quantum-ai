"""Phase 4b (issue #13): falsification arms for the candidate findings of #2.

One driver, three configs (one per finding):

* ``experiments/configs/phase4b_a.toml``: (a) A's parameter efficiency against D, D_diss
  and the HQMM (other D parameterisations, tuned budgets, an NLL-vs-parameters sweep and
  the HQMM restricted to A's parameter count);
* ``experiments/configs/phase4b_b.toml``: (b) weak measurement and retention in the
  feed-forward layout (fixed, annealed, penalised and curriculum strength; why weak
  helps; the general layout and n = 3);
* ``experiments/configs/phase4b_c.toml``: (c) the protected subspace (feed-forward
  variants that might protect, the HQMM and uMPS, and a training-free structural search).

Each ``[[arms]]`` entry is one model family with overrides, a list of tasks and a mode
(``standard``, ``penalty``, ``stages``, ``soft_source`` or ``warm``). An arm with ``lrs`` is
tuned first (every lr with the tuning seeds; the lowest median *validation NLL* wins), then
trained with the evaluation seeds; an arm with ``lr`` uses it as given (the config says
where it came from). Arms run in waves (``wave``, default 0), so ``warm`` arms can start
from the trained models of an earlier wave. Every run is saved (failures too).

Writes ``results/phase4b_<x>/<timestamp>/`` (records, parameters; gitignored) and the
config's ``summary_dir`` (tables, CSV, JSON, plots; committed).

``--from-run <dir>`` reuses saved runs and trains only the missing ones; ``--no-train``
only summarises; ``--arms a,b`` restricts training to those arms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from qmem.baselines import build_baseline, describe_model
from qmem.baselines.hmm import fit_em
from qmem.baselines.umps import UMPSModel
from qmem.benchmarks import feedforward_g_for_eta, marked_recall_metrics
from qmem.diagnostics import label_retention, stored_bit_alignment
from qmem.experiment import build_splits
from qmem.falsification import (
    Stage,
    protected_subspace_search,
    strength_prior_loss,
    train_stages,
    umps_instrument,
)
from qmem.metrics import operation_fidelity, operational_strength
from qmem.model import ModelConfig, QuantumMemoryModel, build_model
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import DISTRACTOR_A, DISTRACTOR_B, make_task, marked_recall_events
from qmem.train import TrainConfig, evaluate, strengths, train_with_restarts
from qmem.trainability import resolve_init_probs, restart_seed, soft_target_loss

LN2 = math.log(2)
MODES = ("standard", "penalty", "stages", "soft_source", "warm")


# -- configuration ---------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def task_entry(cfg: dict[str, Any], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    t = cfg["tasks"][name]
    return dict(t["spec"]), {**cfg["data"], **t.get("data", {})}


def n_symbols_of(spec: dict[str, Any]) -> int:
    return make_task(spec).n_symbols


def arm_model(cfg: dict[str, Any], arm: dict[str, Any], n_sym: int) -> dict[str, Any]:
    """The [model] dict of an arm for a task with ``n_sym`` symbols."""
    fam = cfg["families"][arm["family"]]
    kind = fam["kind"]
    model = {**fam.get("model", {}), **arm.get("model", {})}
    if kind == "A":
        model = {**cfg["a_model"], **model}
        n_out = (n_sym - 1).bit_length()
        model["n_output"] = n_out
        model.update(fam.get("by_n_output", {}).get(str(n_out), {}))
        model.update(arm.get("by_n_output", {}).get(str(n_out), {}))
        if arm.get("eta") is not None:
            model["strength"] = feedforward_g_for_eta(model["n_memory"], n_out, arm["eta"])
            model["strength_trainable"] = False
    if kind == "hqmm_gen" and "n_generators_by_symbols" in model:
        model["n_generators"] = model.pop("n_generators_by_symbols")[str(n_sym)]
    model.pop("n_generators_by_symbols", None)
    if kind in ("gru", "rnn") and "hidden_by_symbols" in model:
        model["hidden"] = model.pop("hidden_by_symbols")[str(n_sym)]
    return model


def run_key(arm: str, task: str, stage: str, lr: float | None, seed: int) -> str:
    lr_s = "fixed" if lr is None else f"lr{lr:g}"
    return f"{arm}/{task}/{stage}/{lr_s}/seed{seed}"


def make_spec(cfg, arm, task, stage, lr, seed, out_dir) -> dict[str, Any]:
    spec_t, data = task_entry(cfg, task)
    data = {**data, **arm.get("data", {})}
    n_sym = n_symbols_of(spec_t)
    fam = cfg["families"][arm["family"]]
    train_cfg = {**cfg["train"], **fam.get("train", {}), **arm.get("train", {})}
    if lr is not None:
        train_cfg["lr"] = lr
    spec = {
        "key": run_key(arm["name"], task, stage, lr, seed),
        "arm": arm["name"],
        "family": arm["family"],
        "kind": fam["kind"],
        "task_name": task,
        "task": spec_t,
        "data": data,
        "model": arm_model(cfg, arm, n_sym),
        "train": train_cfg,
        "method": arm.get("method", fam.get("method", "gradient")),
        "fit": {**fam.get("fit", {}), **arm.get("fit", {})},
        "mode": arm.get("mode", "standard"),
        "stages": arm.get("stages", []),
        "penalty": arm.get("penalty"),
        "warm": None,
        "eta": arm.get("eta"),
        "stage": stage,
        "lr": lr,
        "seed": seed,
        "analysis": cfg.get("analysis", {}),
        "structural_restarts": cfg.get("analysis", {}).get("subspace_restarts", 0),
    }
    if spec["mode"] not in MODES:
        raise ValueError(f"arm {arm['name']}: unknown mode {spec['mode']!r}")
    if spec["mode"] == "warm":
        w = dict(arm["warm"])
        w["path"] = str(out_dir / "runs" / (run_key(w["from"], task, "eval", None, seed) + ".json"))
        spec["warm"] = w
    return spec


# -- one run -------------------------------------------------------------------------
def instrument_of(model: Any) -> tuple[Any, torch.Tensor, int] | None:
    """(instrument, initial memory, n_memory qubits or None) of a quantum model, else None."""
    if isinstance(model, UMPSModel):
        inst, rho0 = umps_instrument(model)
        return inst, rho0, None
    if isinstance(model, QuantumMemoryModel) and hasattr(model, "instrument"):
        try:
            return model.instrument(), model.rho0, model.n_memory
        except Exception:  # D: the readout's instrument is not the model's update
            return None
    return None


def is_d(model: Any) -> bool:
    return type(model).__name__ == "SeparateUpdateReadoutModel"


@torch.no_grad()
def analyse(model: Any, task: Any, test: torch.Tensor, a: dict[str, Any], restarts: int):
    """Strength, disturbance and (on marked_recall) the retention diagnostics of #10."""
    out: dict[str, Any] = {}
    q = None if is_d(model) else instrument_of(model)
    if q is not None:
        inst, rho0, _ = q
        out["eta"] = float(operational_strength(inst))
        out["disturbance"] = 1 - float(operation_fidelity(inst))
    if task.params.get("kind") != "marked_recall":
        return out
    out["recall"] = marked_recall_metrics(model, task, test, a.get("delay_bins"))
    if q is None:
        return out
    inst, rho0, _ = q
    ev = marked_recall_events(test)
    max_lag = a.get("max_lag", 40)
    ret = label_retention(inst, rho0, test, ev["cue_value"], ev["lag"], max_lag)
    thr = a.get("holevo_threshold_bits", 0.1)
    out["horizon"] = next(
        (k for k in range(1, max_lag + 1) if k in ret and ret[k]["holevo_bits"] < thr),
        max_lag + 1,
    )
    out["cue_holevo"] = {k: v["holevo_bits"] for k, v in ret.items()}
    al = stored_bit_alignment(
        inst, rho0, test, ev["cue_value"], ev["lag"], [DISTRACTOR_A, DISTRACTOR_B]
    )
    rows = al["symbols"].values()
    out["commutator"] = statistics.fmean(r["commutator"] for r in rows)
    out["contraction"] = statistics.fmean(r["contraction"] for r in rows)
    out["prob_bias"] = statistics.fmean(abs(r["prob_bias"]) for r in rows)
    if restarts:
        q = 1 / (float(task.params["tau"]) + 1)
        out["subspace"] = protected_subspace_search(
            inst.effects(), [DISTRACTOR_A, DISTRACTOR_B], (3, 2), [(1 - q) / 2] * 2, restarts
        )
    return out


def _set_eta(model: Any, eta: float) -> None:
    for b in model.circuit.blocks:
        if hasattr(b, "set_strength"):
            n_mem, n_out = model.n_memory, model.n_output
            b.set_strength(feedforward_g_for_eta(n_mem, n_out, eta))
            return
    raise ValueError("model has no coupling strength to set")


def _load_state(path: str) -> dict[str, torch.Tensor]:
    return torch.load(Path(path).with_suffix(".pt"))


def run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate one run. Never raises: crashes are recorded."""
    start = time.perf_counter()
    rec: dict[str, Any] = {
        "meta": run_metadata({k: v for k, v in spec.items() if k != "key"}, spec["seed"]),
        **{k: spec[k] for k in ("key", "arm", "family", "task_name", "stage", "lr", "seed")},
    }
    try:
        task = make_task(spec["task"])
        splits = build_splits(task, spec["data"])
        n_sym = task.n_symbols
        model_cfg = resolve_init_probs(spec["model"], splits.train, n_sym)
        kind, seed = spec["kind"], spec["seed"]

        def make_model(r: int) -> Any:
            return build_baseline(kind, model_cfg, n_sym, restart_seed(seed, r))

        probe = make_model(0)
        rec["n_parameters"] = probe.n_parameters()
        tcfg = TrainConfig.from_dict(spec["train"])
        mode = spec["mode"]
        extra: dict[str, Any] = {}
        if spec["method"] == "em":
            model = probe
            res = fit_em(model, splits.train, splits.val, **spec["fit"])
            res["wall_seconds"] = res.pop("wall_seconds")
            result = {k: res[k] for k in ("status", "best_val_nll", "wall_seconds")}
            result["steps_run"] = len(res["curves"]["iter"])
        else:
            if mode in ("standard", "penalty", "soft_source"):
                loss = None
                if mode == "penalty":
                    loss = strength_prior_loss(spec["penalty"]["eta0"], spec["penalty"]["lam"])
                if mode == "soft_source":
                    loss = soft_target_loss(task.predictive)
                model, res = train_with_restarts(make_model, splits, tcfg, loss)
            elif mode == "stages":
                model = probe
                stages = []
                for st in spec["stages"]:
                    t_st = make_task({**spec["task"], **st.get("task", {})})
                    sp = build_splits(t_st, {**spec["data"], **st.get("data", {})})
                    eta = st.get("eta")
                    stages.append(
                        Stage(
                            splits=sp,
                            cfg=TrainConfig.from_dict({**spec["train"], **st.get("train", {})}),
                            frozen=st.get("frozen", []),
                            label=st.get("label", ""),
                            before=(lambda m, e=eta: _set_eta(m, e)) if eta is not None else None,
                        )
                    )
                staged = train_stages(model, stages)
                res = staged.result
                extra["stages"] = staged.stages
            else:  # warm: start from another arm's trained model (same task and seed)
                w = spec["warm"]
                model = probe
                model.load_state_dict(_load_state(w["path"]), strict=False)
                if w.get("eta") is not None:
                    _set_eta(model, w["eta"])
                frozen = [] if w.get("free_strength", False) else ["strength"]
                staged = train_stages(model, [Stage(splits, tcfg, frozen=frozen, label="warm")])
                res = staged.result
                extra["warm_from"] = w["path"]
            result = {
                "status": res.status,
                "steps_run": res.steps_run,
                "best_step": res.best_step,
                "best_val_nll": res.best_val_nll,
                "wall_seconds": res.wall_seconds,
                "strength_curve": [s for s in res.curves.get("strength", [])][::10],
            }
        rec.update(result)
        rec.update(extra)
        rec["train_seconds"] = rec.pop("wall_seconds")
        with torch.no_grad():
            rec["val_nll"] = float(-model.step_log_probs(splits.val).mean())
        rec["test"] = evaluate(model, splits.test, task)
        rec["model_info"] = describe_model(model)
        rec["strength_final"] = strengths(model) if isinstance(model, QuantumMemoryModel) else []
        rec["analysis"] = analyse(
            model, task, splits.test, spec["analysis"], spec["structural_restarts"]
        )
        rec["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    except Exception as exc:  # a crashed run is a result
        rec.update(status="error", message=repr(exc), traceback=traceback.format_exc())
    rec["wall_seconds"] = time.perf_counter() - start
    return rec


def run_and_save(spec: dict[str, Any], out_dir: str) -> dict[str, Any]:
    rec = run_spec(spec)
    path = Path(out_dir) / "runs" / (spec["key"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = rec.pop("state_dict", None)
    if sd is not None:
        torch.save(sd, path.with_suffix(".pt"))
    save_json(path, rec)
    return rec


def _worker_init() -> None:
    os.nice(10)
    torch.set_num_threads(1)


# -- orchestration ---------------------------------------------------------------------
def load_records(out_dir: Path) -> dict[str, dict[str, Any]]:
    recs = {}
    for p in sorted((out_dir / "runs").rglob("seed*.json")):
        r = json.loads(p.read_text())
        recs[r["key"]] = r
    return recs


def run_batch(specs, records, out_dir: Path, workers: int, label: str) -> None:
    todo = [s for s in specs if s["key"] not in records]
    print(f"[{label}] {len(specs)} runs, {len(todo)} to train", flush=True)
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_worker_init) as ex:
        futs = {ex.submit(run_and_save, s, str(out_dir)): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            records[r["key"]] = r
            gap = r.get("test", {}).get("gap_nats", math.nan)
            print(
                f"[{label}] {i}/{len(todo)} {r['key']} {r['status']} gap={gap:.4f} "
                f"({r['wall_seconds']:.0f}s, {time.time() - t0:.0f}s elapsed)",
                flush=True,
            )


def needs_tuning(arm: dict[str, Any], task: str) -> bool:
    return "lrs" in arm and task not in arm.get("lr_by_task", {})


def tuned_lr(cfg, arm, task, records) -> float | None:
    if task in arm.get("lr_by_task", {}):
        return arm["lr_by_task"][task]
    if "lrs" not in arm:
        return arm.get("lr")
    best = None
    for lr in arm["lrs"]:
        vals = [
            records[k]["val_nll"]
            for s in cfg["experiment"]["tuning_seeds"]
            if (k := run_key(arm["name"], task, "tune", lr, s)) in records
            and math.isfinite(records[k].get("val_nll", math.nan))
        ]
        if vals:
            m = statistics.median(vals)
            if best is None or m < best[0]:
                best = (m, lr)
    return None if best is None else best[1]


def run_all(cfg, records, out_dir: Path, workers: int, only: list[str] | None) -> None:
    arms = [a for a in cfg["arms"] if not only or a["name"] in only]
    for wave in sorted({a.get("wave", 0) for a in arms}):
        wave_arms = [a for a in arms if a.get("wave", 0) == wave]
        tune = [
            make_spec(cfg, a, t, "tune", lr, s, out_dir)
            for a in wave_arms
            for t in a["tasks"]
            if needs_tuning(a, t)
            for lr in a["lrs"]
            for s in cfg["experiment"]["tuning_seeds"]
        ]
        run_batch(tune, records, out_dir, workers, f"wave {wave} tune")
        evals = []
        for a in wave_arms:
            for t in a["tasks"]:
                lr = tuned_lr(cfg, a, t, records)
                for s in a.get("seeds", cfg["experiment"]["seeds"]):
                    spec = make_spec(cfg, a, t, "eval", lr, s, out_dir)
                    spec["key"] = run_key(a["name"], t, "eval", None, s)
                    evals.append(spec)
        run_batch(evals, records, out_dir, workers, f"wave {wave} eval")


# -- structural search (no training) ---------------------------------------------------
def structural(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Best protected read contrast of each readout layout, over a grid of strengths."""
    s = cfg.get("structural")
    if not s:
        return []
    rows = []
    for lay in s["layouts"]:
        shared = lay.get("shared", True)
        grid = [(g, g) for g in s["gs"]] if shared else [(a, b) for a in s["gs"] for b in s["gs"]]
        for g_type, g_value in grid:
            mc = ModelConfig(
                n_memory=lay["n_memory"],
                n_output=2,
                memory_depth=0,
                strength=g_type if shared else [g_type, g_value],
                strength_shared=shared,
            )
            inst = build_model(mc).instrument()
            for tau in s["taus"]:
                q = 1 / (tau + 1)  # marked_recall: P(cue-type) = q, each distractor (1-q)/2
                targets = None if tau == 0 else [(1 - q) / 2] * 2
                r = protected_subspace_search(
                    inst.effects(),
                    [DISTRACTOR_A, DISTRACTOR_B],
                    (3, 2),
                    flat_targets=targets,
                    restarts=s["restarts"],
                    steps=s["steps"],
                    seed=0,
                )
                rows.append(
                    {
                        "layout": lay["name"],
                        "tau": tau if tau else "any level",
                        "n_memory": lay["n_memory"],
                        "g_type": g_type,
                        "g_value": g_value,
                        "eta": float(operational_strength(inst)),
                        **r,
                    }
                )
                print(f"[structural] {rows[-1]}", flush=True)
    return rows


# -- summary -------------------------------------------------------------------------
def fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ms(v: list[float]) -> tuple[float, float]:
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return math.nan, math.nan
    if len(v) == 1:
        return v[0], math.nan
    return statistics.fmean(v), statistics.stdev(v) / math.sqrt(len(v))


def summarise(cfg, records) -> list[dict[str, Any]]:
    gap_ok = cfg["experiment"]["success_gap"]
    rows = []
    for arm in cfg["arms"]:
        for t in arm["tasks"]:
            recs = [
                r
                for s in arm.get("seeds", cfg["experiment"]["seeds"])
                if (r := records.get(run_key(arm["name"], t, "eval", None, s)))
            ]
            if not recs:
                continue
            ok = [r for r in recs if r["status"] != "error" and "test" in r]
            gaps = [r["test"]["gap_nats"] for r in ok]
            fails = sum(1 for r in recs if r not in ok or r["test"]["gap_nats"] >= gap_ok)
            row: dict[str, Any] = {
                "arm": arm["name"],
                "label": arm.get("label", arm["name"]),
                "family": arm["family"],
                "task": t,
                "n_parameters": recs[0].get("n_parameters"),
                "lr": ok[0]["lr"] if ok else None,
                "n_runs": len(recs),
                "n_errors": len(recs) - len(ok),
                "failures": fails,
                "gap_median": statistics.median(gaps) if gaps else math.nan,
                "gap_mean": ms(gaps)[0],
                "gap_se": ms(gaps)[1],
                "gap_q25": statistics.quantiles(gaps, n=4)[0] if len(gaps) > 1 else math.nan,
                "gap_q75": statistics.quantiles(gaps, n=4)[2] if len(gaps) > 1 else math.nan,
                "gap_min": min(gaps) if gaps else math.nan,
                "gaps": gaps,
                "val_nll_median": statistics.median([r["val_nll"] for r in ok]) if ok else None,
                "steps_mean": ms([r.get("steps_run", math.nan) for r in ok])[0],
                "train_seconds_mean": ms([r.get("train_seconds", math.nan) for r in ok])[0],
            }
            if ok:
                sel = min(ok, key=lambda r: r["val_nll"])
                row["selected_seed"] = sel["seed"]
                row["selected_gap"] = sel["test"]["gap_nats"]
            an = [r.get("analysis", {}) for r in ok]
            for key in ("eta", "disturbance", "horizon", "commutator", "contraction", "prob_bias"):
                row[f"{key}_mean"], row[f"{key}_se"] = ms([a.get(key, math.nan) for a in an])
            rc = [a["recall"] for a in an if "recall" in a]
            if rc:
                acc = [x["recall_accuracy"] for x in rc]
                row["recall_acc_mean"], row["recall_acc_se"] = ms(acc)
                row["gap_non_recall_mean"] = ms([x["gap_non_recall_nats"] for x in rc])[0]
                row["gap_value_recall_mean"] = ms([x["gap_value_recall_nats"] for x in rc])[0]
                row["retaining_seeds"] = sum(
                    1 for a in an if a.get("horizon", 0) > cfg["analysis"].get("max_lag", 40)
                )
                by = {}
                for x in rc:
                    for b in x["by_delay"]:
                        by.setdefault(b["delay_lo"], []).append(b["accuracy"])
                row["acc_by_delay"] = {k: statistics.fmean(v) for k, v in sorted(by.items())}
                sel_an = sel.get("analysis", {})
                row["selected_recall_acc"] = sel_an.get("recall", {}).get("recall_accuracy")
            subs = [a["subspace"]["contrast"] for a in an if "subspace" in a]
            if subs:
                row["subspace_contrast_mean"] = statistics.fmean(subs)
                row["subspace_contrast_max"] = max(subs)
            pairs = [(a["eta"], a["horizon"]) for a in an if "eta" in a and "horizon" in a]
            row["eta_horizon"] = pairs
            rows.append(row)
    return rows


def reach_rows(rows: list[dict[str, Any]], gap_ok: float) -> list[dict[str, Any]]:
    """Per (task, family): the smallest parameter count reaching the optimum."""
    out = []
    keys = sorted({(r["task"], r["family"]) for r in rows})
    for task, fam in keys:
        cells = sorted(
            [r for r in rows if r["task"] == task and r["family"] == fam],
            key=lambda r: r["n_parameters"] or 0,
        )
        med = next((r for r in cells if r["gap_median"] < gap_ok), None)
        sel = next((r for r in cells if r.get("selected_gap", math.inf) < gap_ok), None)
        out.append(
            {
                "task": task,
                "family": fam,
                "sizes": [r["n_parameters"] for r in cells],
                "median_reach": med["n_parameters"] if med else None,
                "median_reach_arm": med["arm"] if med else None,
                "selected_reach": sel["n_parameters"] if sel else None,
                "selected_reach_arm": sel["arm"] if sel else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = [c for c in rows[0] if not isinstance(rows[0][c], (list, dict))]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def markdown(cfg, rows, reach, struct, meta) -> str:
    out = [
        f"# {cfg['experiment']['title']}",
        "",
        "Generated by `experiments/phase4b.py` from "
        f"`{Path(cfg['_path']).relative_to(REPO_ROOT) if Path(cfg['_path']).is_absolute() else cfg['_path']}`. "  # noqa: E501
        f"{meta['n_runs']} evaluation runs ({meta['n_tune']} tuning runs), "
        f"{meta['n_errors']} errors, {meta['cpu_hours']:.1f} CPU-hours summed over runs. "
        f"Commit(s): {', '.join(meta['commits'])}.",
        "",
        "Gap: test NLL minus the source's NLL on the same sequences (nats/symbol); a run "
        f"fails if it crashed or its gap is ≥ {cfg['experiment']['success_gap']}. "
        "Median [IQR] over seeds; `sel` is the validation-selected run's gap.",
        "",
    ]
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    for t in tasks:
        tr = [r for r in rows if r["task"] == t]
        out += [f"## {t}", ""]
        has_rec = any("recall_acc_mean" in r for r in tr)
        head = "| arm | family | params | lr | fails | median gap [IQR] | sel | η |"
        sep = "|---|---|---|---|---|---|---|---|"
        if has_rec:
            head += " recall acc | non-recall gap | horizon | retain | [E_d,Δ] | contraction |"
            sep += "---|---|---|---|---|---|"
        out += [head, sep]
        for r in tr:
            line = (
                f"| {r['label']} | {r['family']} | {r['n_parameters']} | {fmt(r['lr'], 3)} | "
                f"{r['failures']}/{r['n_runs']} | {fmt(r['gap_median'])} "
                f"[{fmt(r['gap_q25'])}, {fmt(r['gap_q75'])}] | {fmt(r.get('selected_gap'))} | "
                f"{fmt(r['eta_mean'], 3)} |"
            )
            if has_rec:
                line += (
                    f" {fmt(r.get('recall_acc_mean'), 3)} ± {fmt(r.get('recall_acc_se'), 3)} | "
                    f"{fmt(r.get('gap_non_recall_mean'))} | {fmt(r.get('horizon_mean'), 1)} | "
                    f"{r.get('retaining_seeds', '-')} | {fmt(r.get('commutator_mean'), 3)} | "
                    f"{fmt(r.get('contraction_mean'), 3)} |"
                )
            out.append(line)
        out.append("")
        if has_rec:
            bins = sorted({k for r in tr for k in r.get("acc_by_delay", {})}, key=int)
            if bins:
                out += [
                    "Recall accuracy by realised delay (distractors between cue and recall; "
                    "bins at the left edges):",
                    "",
                    "| arm | " + " | ".join(f"d ≥ {b}" for b in bins) + " |",
                    "|---|" + "---|" * len(bins),
                ]
                for r in tr:
                    a = r.get("acc_by_delay", {})
                    out.append(
                        f"| {r['label']} | "
                        + " | ".join(fmt(a.get(b, a.get(str(b))), 3) for b in bins)
                        + " |"
                    )
                out.append("")
    if reach and len({r["family"] for r in rows}) > 1 and cfg.get("summary", {}).get("reach"):
        out += [
            "## Smallest parameter count reaching the optimum (gap < "
            f"{cfg['experiment']['success_gap']})",
            "",
            "| task | family | sizes run | median over seeds | validation-selected run |",
            "|---|---|---|---|---|",
        ]
        for r in reach:
            out.append(
                f"| {r['task']} | {r['family']} | {r['sizes']} | {fmt(r['median_reach'])} | "
                f"{fmt(r['selected_reach'])} |"
            )
        out.append("")
    if struct:
        out += [
            "## Structural search: a protected bit under the feed-forward readout",
            "",
            "Best read contrast (spread of `p(cue-1) - p(cue-0)` over a 2-dim subspace on "
            "which both distractor effects compress to multiples of I, residual < 1e-9; 0 = "
            "no such subspace found) for the bare coupling `D_y(g)` (V = W_y = I: V and W_y "
            "do not change existence). With a tau, both distractor levels are fixed at the "
            "source's (1 - q)/2, q = 1/(tau + 1); 'any level' leaves them free. An exact "
            "model needs contrast 2q(1 - 2 eps) on its stored bit.",
            "",
            "| layout | n | distractor level (tau) | g (type) | g (value) | η | contrast | "
            "feasible starts | best run: contrast, residual |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in struct:
            out.append(
                f"| {r['layout']} | {r['n_memory']} | {r['tau']} | {r['g_type']:.3f} | "
                f"{r['g_value']:.3f} | {r['eta']:.3f} | {r['contrast']:.4f} | "
                f"{r['n_feasible']}/{r['restarts']} | {r['best_run_contrast']:.4f}, "
                f"{r['best_run_residual']:.1e} |"
            )
        out.append("")
    return "\n".join(out)


def plots(cfg, rows, summary_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg.get("summary", {})
    if p.get("gap_vs_params"):
        tasks = list(dict.fromkeys(r["task"] for r in rows))
        fams = list(dict.fromkeys(r["family"] for r in rows))
        colors = {f: plt.cm.tab10(i % 10) for i, f in enumerate(fams)}
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
        for ax, t in zip(axes[0], tasks, strict=True):
            for f in fams:
                cells = sorted(
                    [r for r in rows if r["task"] == t and r["family"] == f],
                    key=lambda r: r["n_parameters"],
                )
                if not cells:
                    continue
                x = [r["n_parameters"] for r in cells]
                for r in cells:
                    g = [max(v, 1e-5) for v in r["gaps"]]
                    ax.scatter([r["n_parameters"]] * len(g), g, s=6, alpha=0.35, color=colors[f])
                ax.plot(
                    x, [max(r["gap_median"], 1e-5) for r in cells], "-o", ms=3,
                    color=colors[f], label=f,
                )  # fmt: skip
            ax.axhline(cfg["experiment"]["success_gap"], color="grey", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("trainable parameters")
        axes[0][0].set_ylabel("test gap (nats/symbol); dots = seeds")
        axes[0][-1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(summary_dir / "gap_vs_params.png", dpi=130)
        plt.close(fig)
    rec_rows = [r for r in rows if "recall_acc_mean" in r]
    if p.get("recall_bars") and rec_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(rec_rows) + 1.5))
        labels = [f"{r['label']} ({r['task']})" for r in rec_rows]
        y = range(len(rec_rows))
        axes[0].barh(
            y, [r["recall_acc_mean"] for r in rec_rows],
            xerr=[0 if not math.isfinite(r["recall_acc_se"]) else r["recall_acc_se"]
                  for r in rec_rows], color="tab:blue",
        )  # fmt: skip
        axes[0].axvline(0.95, color="grey", ls=":", lw=0.8)
        axes[0].axvline(0.5, color="grey", ls="--", lw=0.8)
        axes[0].set_xlim(0.45, 1.0)
        axes[0].set_yticks(list(y), labels, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("recall accuracy (optimum 0.95, chance 0.5)")
        for r in rec_rows:
            pts = r.get("eta_horizon", [])
            axes[1].scatter([e for e, _ in pts], [h for _, h in pts], s=8, label=None)
        axes[1].set_xlabel("operational strength η (per seed)")
        axes[1].set_ylabel("cue horizon (lags)")
        fig.tight_layout()
        fig.savefig(summary_dir / "recall.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--from-run", type=Path)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--arms", type=str)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--dry-run", action="store_true", help="count the runs and stop")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        n_t = sum(
            len(a["lrs"]) * len(cfg["experiment"]["tuning_seeds"])
            for a in cfg["arms"]
            for t in a["tasks"]
            if needs_tuning(a, t)
        )
        n_e = sum(
            len(a["tasks"]) * len(a.get("seeds", cfg["experiment"]["seeds"])) for a in cfg["arms"]
        )
        print(f"{n_t} tuning runs, {n_e} evaluation runs")
        return
    out_dir = args.from_run or new_run_dir(cfg["experiment"]["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config_meta.json", run_metadata(cfg, None))
    print(f"run dir: {out_dir}", flush=True)
    records = load_records(out_dir)
    if not args.no_train:
        only = args.arms.split(",") if args.arms else None
        run_all(cfg, records, out_dir, args.workers or cfg["experiment"]["workers"], only)
    struct = structural(cfg)
    rows = summarise(cfg, records)
    reach = reach_rows(rows, cfg["experiment"]["success_gap"])
    summary_dir = REPO_ROOT / cfg["experiment"]["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    allr = list(records.values())
    meta = {
        "n_runs": sum(r["stage"] == "eval" for r in allr),
        "n_tune": sum(r["stage"] == "tune" for r in allr),
        "n_errors": sum(r["status"] == "error" for r in allr),
        "cpu_hours": sum(r.get("wall_seconds", 0) for r in allr) / 3600,
        "commits": sorted({r["meta"]["git_commit"][:7] for r in allr if "meta" in r}),
        "run_dir": str(out_dir.relative_to(REPO_ROOT)) if out_dir.is_absolute() else str(out_dir),
    }
    save_json(
        summary_dir / "summary.json",
        {"meta": {**run_metadata(cfg, None), **meta}, "rows": rows, "reach": reach,
         "structural": struct},
    )  # fmt: skip
    write_csv(summary_dir / "main.csv", rows)
    write_csv(summary_dir / "reach.csv", reach)
    write_csv(summary_dir / "structural.csv", struct)
    (summary_dir / "summary.md").write_text(markdown(cfg, rows, reach, struct, meta))
    plots(cfg, rows, summary_dir)
    print(f"summary: {summary_dir}", flush=True)


if __name__ == "__main__":
    main()
