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
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ..experiment import RunSpec, build_splits, run_one
from ..model import QuantumMemoryModel
from ..records import run_metadata
from ..tasks import make_task
from ..train import TrainConfig, evaluate, train
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
        }


def build_baseline(family: str, cfg: dict[str, Any], n_symbols: int, seed: int):
    """A freshly initialised model of ``family`` (parameters drawn from ``seed``)."""
    c = dict(cfg)
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
            init_scale=c.get("init_scale", 1.0),
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
    if spec.family == "A":
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
            model = build_baseline(spec.family, spec.model, task.n_symbols, spec.seed)
            if spec.method == "gradient":
                result = train(model, splits, TrainConfig.from_dict(spec.train)).to_dict()
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
        record["diagnostics"] = dict(model.diagnostics)
        record["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    except Exception as exc:  # recorded, not hidden: a crashed run is a result
        record.update(status="error", message=repr(exc), traceback=traceback.format_exc())
    record["wall_seconds"] = time.perf_counter() - start
    return record


def load_baseline(record: dict[str, Any]) -> Any:
    """Rebuild a trained gradient/EM baseline from a record (needs ``state_dict``)."""
    cfg = record["meta"]["config"]
    model = build_baseline(cfg["family"], cfg["model"], record["task"]["n_symbols"], record["seed"])
    model.load_state_dict(record["state_dict"])
    return model
