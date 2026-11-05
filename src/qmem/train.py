"""Teacher-forced maximum-likelihood training and evaluation.

``train`` minimises the per-symbol teacher-forced NLL
``-(1/(B T)) sum_b sum_t log p(y_t | y_<t)`` with a torch optimiser, logging
every step (loss, gradient norm before clipping, whether it was clipped,
learning rate, the change in ``model.diagnostics``) and evaluating on the
validation split every ``eval_every`` steps. It never raises on a bad run: a
non-finite loss or gradient ends training with ``status = "diverged"`` and the
curves so far are returned, so failed runs are recorded like any other.

``evaluate`` reports test NLL (nats and bits per symbol), the paired gap to
the source's own log-probabilities on the same sequences (``oracle``), the
gap to the exact expected optimum, and next-symbol accuracy.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

from .data import Splits
from .model import QuantumMemoryModel
from .tasks import HMMTask

LN2 = math.log(2)
SCHEDULES = ("constant", "cosine", "step")


@dataclass
class TrainConfig:
    """Optimisation settings (one ``[train]`` table in an experiment config).

    * ``batch_size = 0`` means full batch. Minibatches are drawn without
      replacement per epoch from ``torch.Generator().manual_seed(batch_seed)``.
    * ``schedule``: ``constant``; ``cosine`` (from ``lr`` to ``lr * lr_min_factor``
      over ``steps``); ``step`` (multiply by ``gamma`` every ``step_every`` steps).
    * Early stopping: stop after ``patience`` evaluations without a validation
      improvement of at least ``min_delta`` nats/symbol (``patience = 0``: off).
      With ``restore_best`` the parameters of the best validation point are
      restored at the end.
    * ``clip_grad_norm``: clip the global gradient norm to this value (0: off).
      The pre-clip norm is always logged.
    """

    optimizer: str = "adam"
    lr: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    schedule: str = "constant"
    lr_min_factor: float = 0.05
    gamma: float = 0.5
    step_every: int = 100
    steps: int = 300
    batch_size: int = 0
    batch_seed: int = 0
    eval_every: int = 10
    patience: int = 10
    min_delta: float = 1e-4
    restore_best: bool = True
    clip_grad_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        cfg = cls(**d)
        if cfg.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {cfg.schedule!r}; choose from {SCHEDULES}")
        return cfg


def lr_at(cfg: TrainConfig, step: int) -> float:
    if cfg.schedule == "constant":
        return cfg.lr
    if cfg.schedule == "cosine":
        lo = cfg.lr * cfg.lr_min_factor
        frac = step / max(cfg.steps - 1, 1)
        return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * frac))
    return cfg.lr * cfg.gamma ** (step // cfg.step_every)


def _optimizer(cfg: TrainConfig, params) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; choose 'adam' or 'sgd'")


def nll_per_symbol(model: QuantumMemoryModel, sequences: torch.Tensor) -> torch.Tensor:
    """Mean teacher-forced NLL per symbol (nats), differentiable."""
    return -model.step_log_probs(sequences).mean()


def strengths(model: QuantumMemoryModel) -> list[float]:
    """Every coupling strength ``g`` in the circuit (output and env couplings)."""
    out: list[float] = []
    circuit = getattr(model, "circuit", None)  # baselines have no circuit
    for b in circuit.blocks if circuit is not None else []:
        if hasattr(b, "strength"):
            out.extend(float(x) for x in b.strength.detach())
    return out


@dataclass
class TrainResult:
    status: str  # "completed" | "early_stopped" | "diverged"
    steps_run: int
    best_step: int
    best_val_nll: float
    curves: dict[str, list]
    wall_seconds: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(model: QuantumMemoryModel, splits: Splits, cfg: TrainConfig) -> TrainResult:
    """Train ``model`` in place on ``splits.train``; early-stop on ``splits.val``."""
    start = time.perf_counter()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _optimizer(cfg, params)
    gen = torch.Generator().manual_seed(cfg.batch_seed)
    n_train = splits.train.shape[0]
    bs = cfg.batch_size if 0 < cfg.batch_size < n_train else n_train
    order = torch.randperm(n_train, generator=gen)
    cursor = 0

    curves: dict[str, list] = {
        k: [] for k in ("step", "loss", "grad_norm", "clipped", "lr", "diag", "elapsed")
    }
    curves.update({"eval_step": [], "val_nll": [], "strength": []})
    best = (math.inf, -1, copy.deepcopy(model.state_dict()))
    bad_evals = 0
    status, message = "completed", ""

    def run_eval(step: int) -> bool:
        nonlocal best, bad_evals
        with torch.no_grad():
            v = float(nll_per_symbol(model, splits.val))
        curves["eval_step"].append(step)
        curves["val_nll"].append(v)
        curves["strength"].append(strengths(model))
        if v < best[0] - cfg.min_delta:
            best = (v, step, copy.deepcopy(model.state_dict()))
            bad_evals = 0
        elif math.isfinite(v):
            bad_evals += 1
        return cfg.patience > 0 and bad_evals >= cfg.patience

    for step in range(cfg.steps):
        if step % cfg.eval_every == 0 and run_eval(step):
            status = "early_stopped"
            break
        if cursor + bs > n_train:
            order = torch.randperm(n_train, generator=gen)
            cursor = 0
        batch = splits.train[order[cursor : cursor + bs]]
        cursor += bs
        lr = lr_at(cfg, step)
        for group in opt.param_groups:
            group["lr"] = lr
        before = dict(model.diagnostics)
        opt.zero_grad()
        loss = nll_per_symbol(model, batch)
        loss_v = float(loss.detach())
        grad_norm = math.nan
        if math.isfinite(loss_v):
            loss.backward()
            grad_norm = float(
                torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
            )
        diag = {k: v - before.get(k, 0) for k, v in model.diagnostics.items()}
        diag = {k: v for k, v in diag.items() if v}
        clipped = bool(cfg.clip_grad_norm > 0 and grad_norm > cfg.clip_grad_norm)
        for k, v in (
            ("step", step),
            ("loss", loss_v),
            ("grad_norm", grad_norm),
            ("clipped", clipped),
            ("lr", lr),
            ("diag", diag),
            ("elapsed", time.perf_counter() - start),
        ):
            curves[k].append(v)
        if not (math.isfinite(loss_v) and math.isfinite(grad_norm)):
            status = "diverged"
            message = f"non-finite loss ({loss_v}) or gradient norm ({grad_norm}) at step {step}"
            break
        if clipped:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
        opt.step()
    else:
        run_eval(cfg.steps)
    if cfg.restore_best and best[1] >= 0:
        model.load_state_dict(best[2])
    return TrainResult(
        status=status,
        steps_run=len(curves["step"]),
        best_step=best[1],
        best_val_nll=best[0],
        curves=curves,
        wall_seconds=time.perf_counter() - start,
        message=message,
    )


def _accuracy(pred: torch.Tensor, seq: torch.Tensor) -> float:
    """Fraction of positions where the argmax of ``pred`` (B, T, Y) equals ``seq``."""
    return float((pred.argmax(-1) == seq).double().mean())


@torch.no_grad()
def evaluate(
    model: QuantumMemoryModel, sequences: torch.Tensor, task: HMMTask | None = None
) -> dict[str, Any]:
    """Test metrics for ``model`` on ``sequences`` (per symbol; SE over sequences).

    ``task`` may be any source with the ``HMMTask`` interface (including
    ``QuantumSourceTask``). If its exact optimum cannot be enumerated at this
    length, ``optimal_nll_nats`` is NaN with a note; the paired gap to the
    oracle is always exact.
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    n, length = seq.shape
    before = dict(model.diagnostics)
    per_seq = -model.step_log_probs(seq).mean(-1)
    pred = model.predictive_probs(seq)[:, :-1]
    out: dict[str, Any] = {
        "n_sequences": n,
        "length": length,
        "nll_nats": float(per_seq.mean()),
        "nll_se": float(per_seq.std() / math.sqrt(n)) if n > 1 else math.nan,
        "nll_bits": float(per_seq.mean()) / LN2,
        "accuracy": _accuracy(pred, seq),
        "min_p_observed": float(pred.gather(-1, seq[..., None]).min()),
    }
    if task is not None:
        oracle_seq = -task.log_probs(seq).mean(-1)
        diff = per_seq - oracle_seq
        try:
            optimum = task.optimal_nll(length)
        except RuntimeError as exc:  # non-unifilar source: belief enumeration too large
            optimum = math.nan
            out["optimal_nll_note"] = f"exact optimum unavailable at length {length}: {exc}"
        try:
            rate = task.entropy_rate()
        except RuntimeError as exc:
            rate = math.nan
            out["entropy_rate_note"] = str(exc)
        out.update(
            {
                "oracle_nll_nats": float(oracle_seq.mean()),
                "gap_nats": float(diff.mean()),
                "gap_se": float(diff.std() / math.sqrt(n)) if n > 1 else math.nan,
                "gap_bits": float(diff.mean()) / LN2,
                "optimal_nll_nats": optimum,
                "optimal_nll_bits": optimum / LN2,
                "gap_to_optimum_nats": float(per_seq.mean()) - optimum,
                "entropy_rate_nats": rate,
                "oracle_accuracy": _accuracy(task.predictive(seq)[:, :-1], seq),
            }
        )
    diag = {k: v - before.get(k, 0) for k, v in model.diagnostics.items()}
    out["diagnostics"] = {k: v for k, v in diag.items() if v}
    return out
