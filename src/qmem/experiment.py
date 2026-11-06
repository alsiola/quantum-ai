"""Multi-seed experiment runs: one function per run, a process pool, an aggregate.

A *run* is (task spec, data settings, model config, train config, model
seed). ``run_one`` builds everything from those plain dicts, trains, evaluates
on the test split and returns a JSON-ready record with ``run_metadata``. It
catches every exception, so crashed and diverged runs are recorded with their
curves like successful ones. ``run_many`` runs records in parallel (one torch
thread per worker) and writes each to ``<out_dir>/runs/<group>/seed<k>.json``
with the trained parameters next to it (``.pt``).

``aggregate`` groups records by ``group`` and reports mean ± standard error
and a seeded bootstrap 95% CI of the test NLL gap, plus the failure rate,
where a run *fails* if it crashed, diverged, or its test gap to the oracle is
at least ``success_gap`` nats/symbol.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .ablations import build_variant, validity_defect
from .data import Splits, make_splits
from .model import QuantumMemoryModel
from .records import run_metadata, save_json
from .tasks import HMMTask, make_task
from .train import TrainConfig, evaluate, strengths, train_with_restarts
from .trainability import resolve_init_probs, restart_seed


@dataclass
class DataConfig:
    n_train: int = 512
    n_val: int = 256
    n_test: int = 512
    length: int = 32
    seed: int = 0


@dataclass
class RunSpec:
    """Everything needed to reproduce one training run (plain, picklable data)."""

    group: str  # e.g. "even/m2" or "variants/even/lr=0.1"
    task: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    seed: int
    labels: dict[str, Any] = field(default_factory=dict)

    def config(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "task": self.task,
            "data": self.data,
            "model": self.model,
            "train": self.train,
            "labels": self.labels,
        }


def build_splits(task: HMMTask, data: dict[str, Any]) -> Splits:
    d = DataConfig(**data)
    return make_splits(task, d.n_train, d.n_val, d.n_test, d.length, d.seed)


def build_run_model(spec: RunSpec) -> QuantumMemoryModel:
    """Architecture A, or the ablation named by ``spec.model["variant"]`` (``qmem.ablations``)."""
    return build_variant({**spec.model, "seed": spec.seed})


def run_one(spec: RunSpec) -> dict[str, Any]:
    """Train and evaluate one run. Never raises; failures are part of the record."""
    start = time.perf_counter()
    record: dict[str, Any] = {
        "meta": run_metadata(spec.config(), spec.seed),
        "group": spec.group,
        "seed": spec.seed,
        "labels": spec.labels,
    }
    try:
        task = make_task(spec.task)
        splits = build_splits(task, spec.data)
        model_cfg = resolve_init_probs(spec.model, splits.train, task.n_symbols)
        record["model_resolved"] = model_cfg

        def make_model(r: int) -> QuantumMemoryModel:
            return build_variant({**model_cfg, "seed": restart_seed(spec.seed, r)})

        probe = make_model(0)
        if probe.n_symbols != task.n_symbols:
            raise ValueError(f"model alphabet {probe.n_symbols} != task alphabet {task.n_symbols}")
        record["task"] = task.describe()
        record["splits"] = splits.describe()
        record["n_parameters"] = probe.n_parameters()
        record["strength_init"] = strengths(probe)
        model, result = train_with_restarts(make_model, splits, TrainConfig.from_dict(spec.train))
        record["restart_screen"] = result.restart_screen
        record.update(
            status=result.status,
            message=result.message,
            steps_run=result.steps_run,
            best_step=result.best_step,
            best_val_nll=result.best_val_nll,
            curves=result.curves,
            train_seconds=result.wall_seconds,
        )
        record["test"] = evaluate(model, splits.test, task)
        record["train_metrics"] = evaluate(model, splits.train, task)
        record["strength_final"] = strengths(model)
        record["instrument_completeness_defect"] = validity_defect(model)
        record["diagnostics"] = dict(model.diagnostics)
        record["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    except Exception as exc:  # recorded, not hidden: a crashed run is a result
        record.update(status="error", message=repr(exc), traceback=traceback.format_exc())
    record["wall_seconds"] = time.perf_counter() - start
    return record


def _worker_init() -> None:
    torch.set_num_threads(1)


def _run_in_worker(runner: Callable[[Any], dict[str, Any]], spec: Any) -> dict[str, Any]:
    return runner(spec)


def save_record(record: dict[str, Any], out_dir: Path) -> Path:
    path = Path(out_dir) / "runs" / record["group"] / f"seed{record['seed']}.json"
    state = record.pop("state_dict", None)
    save_json(path, record)
    if state is not None:
        torch.save(state, path.with_suffix(".pt"))
        record["state_dict"] = state
    return path


def run_many(
    specs: list[Any],
    out_dir: Path,
    workers: int = 1,
    log: Any = print,
    runner: Callable[[Any], dict[str, Any]] = run_one,
) -> list[dict[str, Any]]:
    """Run every spec (in parallel if ``workers > 1``), saving each record as it lands.

    ``runner`` turns one spec into a record (default ``run_one``; Phase 3a passes
    ``qmem.baselines.runner.run_baseline``). Specs need ``group`` and ``seed``.
    """
    records: list[dict[str, Any]] = []

    def done(rec: dict[str, Any]) -> None:
        save_record(rec, out_dir)
        records.append(rec)
        gap = rec.get("test", {}).get("gap_nats", math.nan)
        log(
            f"  {rec['group']:<40s} seed {rec['seed']}: {rec['status']:<13s} "
            f"gap {gap:+.4f} nats  ({rec['wall_seconds']:.1f}s)"
        )

    if workers <= 1:
        for s in specs:
            done(runner(s))
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init) as ex:
            for fut in cf.as_completed([ex.submit(_run_in_worker, runner, s) for s in specs]):
                done(fut.result())
    order = {(s.group, s.seed): i for i, s in enumerate(specs)}
    records.sort(key=lambda r: order[(r["group"], r["seed"])])
    return records


def load_model(record: dict[str, Any]) -> QuantumMemoryModel:
    """Rebuild a trained model from a record (needs ``state_dict``)."""
    cfg = record["meta"]["config"]
    model_cfg = record.get("model_resolved", cfg["model"])
    model = build_variant({**model_cfg, "seed": record["seed"]})
    model.load_state_dict(record["state_dict"])
    return model


# -- aggregation --------------------------------------------------------------
def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    m = sum(values) / n
    if n == 1:
        return m, math.nan
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def bootstrap_ci(
    values: list[float], seed: int = 0, n_boot: int = 2000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (seeded)."""
    if len(values) < 2:
        return math.nan, math.nan
    v = torch.tensor(values, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(values), (n_boot, len(values)), generator=gen)
    means = v[idx].mean(-1)
    lo, hi = torch.quantile(means, torch.tensor([(1 - level) / 2, (1 + level) / 2]).double())
    return float(lo), float(hi)


def is_success(record: dict[str, Any], success_gap: float) -> bool:
    if record.get("status") not in ("completed", "early_stopped"):
        return False
    gap = record.get("test", {}).get("gap_nats", math.nan)
    return math.isfinite(gap) and gap < success_gap


def aggregate(records: list[dict[str, Any]], success_gap: float) -> list[dict[str, Any]]:
    """One summary row per group, in first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)
    rows = []
    for group, recs in groups.items():
        ok = [r for r in recs if "test" in r]
        gaps = [r["test"]["gap_nats"] for r in ok if math.isfinite(r["test"]["gap_nats"])]
        nlls = [r["test"]["nll_nats"] for r in ok if math.isfinite(r["test"]["nll_nats"])]
        succ = [r for r in recs if is_success(r, success_gap)]
        succ_gaps = [r["test"]["gap_nats"] for r in succ]
        first = ok[0]["test"] if ok else {}
        m_gap, se_gap = mean_se(gaps)
        m_nll, se_nll = mean_se(nlls)
        lo, hi = bootstrap_ci(gaps)
        rows.append(
            {
                "group": group,
                "labels": recs[0].get("labels", {}),
                "n_runs": len(recs),
                "n_success": len(succ),
                "failure_rate": 1 - len(succ) / len(recs),
                "statuses": {
                    s: sum(r["status"] == s for r in recs)
                    for s in sorted({r["status"] for r in recs})
                },
                "optimal_nll_nats": first.get("optimal_nll_nats", math.nan),
                "oracle_nll_nats": first.get("oracle_nll_nats", math.nan),
                "entropy_rate_nats": first.get("entropy_rate_nats", math.nan),
                "test_nll_mean": m_nll,
                "test_nll_se": se_nll,
                "gap_mean": m_gap,
                "gap_se": se_gap,
                "gap_ci95": [lo, hi],
                "gap_median": float(torch.tensor(gaps).median()) if gaps else math.nan,
                "gap_best": min(gaps, default=math.nan),
                "gap_worst": max(gaps, default=math.nan),
                "gap_mean_successful": mean_se(succ_gaps)[0],
                "accuracy_mean": mean_se([r["test"]["accuracy"] for r in ok])[0],
                "oracle_accuracy": first.get("oracle_accuracy", math.nan),
                "steps_mean": mean_se([r.get("steps_run", 0) for r in recs])[0],
                "wall_seconds_total": sum(r["wall_seconds"] for r in recs),
                "n_parameters": recs[0].get("n_parameters"),
            }
        )
    return rows
