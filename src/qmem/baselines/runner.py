"""One baseline run: build the model, fit it, evaluate it with the Phase 2 metrics.

A ``BaselineSpec`` is the baseline analogue of ``qmem.experiment.RunSpec``: a
task spec, data settings (the same splits as architecture A), a model family
with its settings, a fitting method and a seed. ``run_baseline`` never raises;
crashed runs are recorded like any other. ``family = "A"`` runs architecture A
through ``qmem.experiment.run_one`` itself, so A and the baselines go through
the same code where it matters.

Fitting methods:

* ``gradient``: ``qmem.train.train`` with ``spec.train`` (the trainer A uses);
* ``em``: Baum-Welch (``qmem.baselines.hmm.fit_em``) with ``spec.fit``;
* ``count``: n-gram counts with ``k`` and ``alpha`` chosen on validation.

**Soft targets** (issue #12; ``spec.teacher``, gradient method only, any family
including ``"A"``): ``{"kind": "source"}`` uses the task's exact predictive,
``{"kind": "record", "path": <run json>}`` a trained model. The model is first fitted
to the teacher's predictive with ``qmem.trainability.soft_target_loss`` (``spec.train``
updated by ``teacher["train"]``). ``mode = "fit"`` (default) stops there, a test of
whether the parameterisation can reach the teacher's function; ``mode = "warm"`` then
trains on the data NLL with ``spec.train``, a warm start. The record keeps the
teacher fit (``soft``: best validation loss, and the mean KL to the teacher on the
validation and test splits).
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..ablations import build_variant
from ..experiment import RunSpec, build_splits, run_one
from ..model import QuantumMemoryModel
from ..records import run_metadata
from ..tasks import make_task
from ..train import TrainConfig, evaluate, strengths, train, train_with_restarts
from ..trainability import resolve_init_probs, restart_seed, soft_target_loss, teacher_entropy
from .hmm import HMMModel, fit_em
from .hqmm import StiefelInstrumentModel
from .ngram import fit_ngram
from .rnn import RNNModel
from .umps import UMPSModel

FAMILIES = ("A", "ngram", "hmm", "gru", "rnn", "umps", "hqmm")


@dataclass
class BaselineSpec:
    group: str
    task: dict[str, Any]
    data: dict[str, Any]
    family: str
    model: dict[str, Any]
    method: str = "gradient"
    train: dict[str, Any] = field(default_factory=dict)
    fit: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    labels: dict[str, Any] = field(default_factory=dict)
    teacher: dict[str, Any] = field(default_factory=dict)

    def config(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "task": self.task,
            "data": self.data,
            "family": self.family,
            "model": self.model,
            "method": self.method,
            "train": self.train,
            "fit": self.fit,
            "labels": self.labels,
            "teacher": self.teacher,
        }


def build_baseline(family: str, cfg: dict[str, Any], n_symbols: int, seed: int):
    """A freshly initialised model of ``family`` (parameters drawn from ``seed``)."""
    c = dict(cfg)
    if family == "A":  # architecture A or an ablation, as in qmem.experiment
        model = build_variant({**c, "seed": seed})
        if model.n_symbols != n_symbols:
            raise ValueError(f"model alphabet {model.n_symbols} != task alphabet {n_symbols}")
        return model
    if family == "hmm":
        return HMMModel(c["n_states"], n_symbols, seed=seed, init_scale=c.get("init_scale", 1.0))
    if family in ("gru", "rnn"):
        return RNNModel(family, c["hidden"], n_symbols, seed=seed)
    if family == "umps":
        return UMPSModel(
            c["bond_dim"],
            n_symbols,
            purification=c.get("purification", 1),
            seed=seed,
            init_scale=c.get("init_scale", 1.0),
            init=c.get("init", "random"),
            init_probs=c.get("init_probs", "uniform"),
            fp_iters=c.get("fp_iters", 200),
            fp_grad_iters=c.get("fp_grad_iters", 20),
            fp_tol=c.get("fp_tol", 1e-8),
        )
    if family == "hqmm":
        n_output = (n_symbols - 1).bit_length()
        if 2**n_output != n_symbols:
            raise ValueError("the HQMM reads outcomes from qubits: alphabet must be a power of 2")
        return StiefelInstrumentModel(
            c["n_memory"], n_output, c.get("n_env", 0), seed=seed,
            init_scale=c.get("init_scale", 1.0), init=c.get("init", "random"),
            init_probs=c.get("init_probs", "uniform"),
        )  # fmt: skip
    raise ValueError(f"unknown baseline family {family!r}; choose from {FAMILIES}")


def describe_model(model: Any) -> dict[str, Any]:
    """Family, parameter count, state dimension and compute cost, for any model."""
    if hasattr(model, "describe"):
        return model.describe()
    if isinstance(model, QuantumMemoryModel):  # architecture A
        d, e = model.memory_dim, 2**model.n_env
        return {
            "family": "A",
            "n_symbols": model.n_symbols,
            "n_parameters": model.n_parameters(),
            "state_dim": d,
            "state_kind": "pure D-dim quantum state" if model.n_env == 0 else "D x D density",
            "flops_per_step": 4 * (2 * e * d**3 + model.n_symbols * d**2),
        }
    raise TypeError(f"cannot describe {type(model)}")


def _run_a(spec: BaselineSpec) -> dict[str, Any]:
    rec = run_one(
        RunSpec(spec.group, spec.task, spec.data, spec.model, spec.train, spec.seed, spec.labels)
    )
    rec["family"] = "A"
    rec["method"] = "gradient"
    if "state_dict" in rec:
        from ..experiment import load_model

        rec["model_info"] = describe_model(load_model(rec))
    return rec


def run_baseline(spec: BaselineSpec) -> dict[str, Any]:
    """Fit and evaluate one baseline. Never raises; failures are part of the record."""
    if spec.family == "A" and not spec.teacher:
        return _run_a(spec)
    start = time.perf_counter()
    record: dict[str, Any] = {
        "meta": run_metadata(spec.config(), spec.seed),
        "group": spec.group,
        "seed": spec.seed,
        "labels": spec.labels,
        "family": spec.family,
        "method": spec.method,
    }
    try:
        task = make_task(spec.task)
        splits = build_splits(task, spec.data)
        record["task"] = task.describe()
        record["splits"] = splits.describe()
        if spec.method == "count":
            if spec.family != "ngram":
                raise ValueError("method 'count' is for the ngram family")
            model, grid = fit_ngram(
                splits.train,
                splits.val,
                task.n_symbols,
                list(spec.model["orders"]),
                list(spec.model["alphas"]),
            )
            best = min(grid, key=lambda g: g["val_nll"])
            record.update(
                status="completed",
                message="",
                steps_run=0,
                best_step=0,
                best_val_nll=best["val_nll"],
                curves={},
                train_seconds=0.0,
                selection_grid=grid,
            )
            # Test NLL of every order (at its own best alpha), for the k-sweep plot.
            per_k = {}
            for k in spec.model["orders"]:
                rows = [g for g in grid if g["order"] == k]
                bk = min(rows, key=lambda g: g["val_nll"])
                mk, _ = fit_ngram(splits.train, splits.val, task.n_symbols, [k], [bk["alpha"]])
                per_k[k] = {**bk, "test": evaluate(mk, splits.test, task)}
            record["per_order"] = per_k
        else:
            model_cfg = resolve_init_probs(spec.model, splits.train, task.n_symbols)
            record["model_resolved"] = model_cfg

            def make_model(r: int) -> Any:
                seed = restart_seed(spec.seed, r)
                return build_baseline(spec.family, model_cfg, task.n_symbols, seed)

            model = make_model(0)
            if spec.method == "gradient" and spec.teacher:
                model, result = _fit_teacher(spec, task, splits, make_model, record)
            elif spec.method == "gradient":
                tcfg = TrainConfig.from_dict(spec.train)
                model, res = train_with_restarts(make_model, splits, tcfg)
                result = res.to_dict()
                result["train_seconds"] = result.pop("wall_seconds")
            elif spec.method == "em":
                if spec.family != "hmm":
                    raise ValueError("method 'em' is for the hmm family")
                result = fit_em(model, splits.train, splits.val, **spec.fit)
                result["train_seconds"] = result.pop("wall_seconds")
            else:
                raise ValueError(f"unknown method {spec.method!r}")
            record.update(result)
        record["model_info"] = describe_model(model)
        record["n_parameters"] = record["model_info"]["n_parameters"]
        record["test"] = evaluate(model, splits.test, task)
        record["train_metrics"] = evaluate(model, splits.train, task)
        if isinstance(model, QuantumMemoryModel):
            record["instrument_completeness_defect"] = float(
                model.instrument().completeness_defect().detach()
            )
        if spec.family == "umps":
            record["fixed_point_residual"] = model.last_fp_residual
        if spec.family == "A":
            record["strength_final"] = strengths(model)
        record["diagnostics"] = dict(model.diagnostics)
        record["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    except Exception as exc:  # recorded, not hidden: a crashed run is a result
        record.update(status="error", message=repr(exc), traceback=traceback.format_exc())
    record["wall_seconds"] = time.perf_counter() - start
    return record


def load_record(path: str | Path) -> dict[str, Any]:
    """A saved run record (``.json``) with its parameters (``.pt``) attached."""
    path = Path(path)
    rec = json.loads(path.read_text())
    rec["state_dict"] = torch.load(path.with_suffix(".pt"))
    return rec


def load_any(record: dict[str, Any]) -> Any:
    """Rebuild a trained model of any family (A through ``qmem.experiment.load_model``)."""
    if record.get("family", "A") == "A" and "family" not in record["meta"]["config"]:
        from ..experiment import load_model

        return load_model(record)
    return load_baseline(record)


def make_teacher(teacher: dict[str, Any], task: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    """The predictive ``q(. | y_<t)`` (``(B, T+1, Y)``) a soft-target fit aims at."""
    if teacher["kind"] == "source":
        return task.predictive
    if teacher["kind"] == "record":
        model = load_any(load_record(teacher["path"]))
        model.eval()
        return model.predictive_probs
    raise ValueError(f"unknown teacher kind {teacher['kind']!r}; choose 'source' or 'record'")


def _fit_teacher(spec, task, splits, make_model, record) -> tuple[Any, dict[str, Any]]:
    mode = spec.teacher.get("mode", "fit")
    if mode not in ("fit", "warm"):
        raise ValueError(f"unknown teacher mode {mode!r}; choose 'fit' or 'warm'")
    teacher = make_teacher(spec.teacher, task)
    loss = soft_target_loss(teacher)
    soft_cfg = TrainConfig.from_dict({**spec.train, **spec.teacher.get("train", {})})
    model, soft = train_with_restarts(make_model, splits, soft_cfg, loss_fn=loss)
    with torch.no_grad():
        kl_test = float(loss(model, splits.test)) - teacher_entropy(teacher, splits.test)
    record["soft"] = {
        "mode": mode,
        "status": soft.status,
        "steps_run": soft.steps_run,
        "best_val_loss": soft.best_val_nll,
        "kl_val": soft.best_val_nll - teacher_entropy(teacher, splits.val),
        "kl_test": kl_test,
        "eval_step": soft.curves["eval_step"],
        "val_loss": soft.curves["val_nll"],
        "grad_norm": soft.curves["grad_norm"],
        "restart_screen": soft.restart_screen,
        "train_seconds": soft.wall_seconds,
    }
    if mode == "fit":
        result = soft.to_dict()
        # best_val_nll must stay a data NLL: score the fitted model on validation.
        with torch.no_grad():
            result["best_val_nll"] = float(-model.step_log_probs(splits.val).mean())
    else:
        result = train(model, splits, TrainConfig.from_dict(spec.train)).to_dict()
        result["wall_seconds"] += soft.wall_seconds
    result["train_seconds"] = result.pop("wall_seconds")
    return model, result


def load_baseline(record: dict[str, Any]) -> Any:
    """Rebuild a trained gradient/EM baseline from a record (needs ``state_dict``)."""
    cfg = record["meta"]["config"]
    model_cfg = record.get("model_resolved", cfg["model"])
    model = build_baseline(cfg["family"], model_cfg, record["task"]["n_symbols"], record["seed"])
    model.load_state_dict(record["state_dict"])
    return model
