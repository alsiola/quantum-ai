"""Tools for falsifying the candidate findings (Phase 4b, issue #13).

* ``operational_strength_t``: the operational strength ``eta`` of
  ``qmem.metrics.operational_strength`` as a differentiable tensor, so it can enter a
  loss. ``eta = (sum_y lambda_max(E_y) - 1) / (min(Y, d) - 1)``.
* ``strength_prior_loss``: the NLL plus a quadratic prior ``lam (eta - eta0)^2`` on the
  operational strength. It works for any layout (feed-forward or general), because
  ``eta`` is a function of the effects only. Used for training *and* the validation
  score (a MAP objective); records also keep the plain validation NLL.
* ``train_stages``: train one model through several stages in sequence, each with its
  own splits (a curriculum), budget, loss and set of frozen parameters (an annealed
  strength: frozen first, free later). Each stage is ``qmem.train.train``, so early
  stopping and best-validation restore apply per stage.
* ``protected_subspace_search``: is there a 2-dimensional subspace ``S`` of the memory
  on which every "flat" effect (the distractors) acts as a multiple of the identity,
  while the "read" effects (the cue values) still tell the two states of ``S`` apart?
  ``P E_y P = c_y P`` is exactly the condition that the Kraus operators of ``y``
  restricted to ``S`` are proportional to isometries, i.e. outcome ``y`` neither reveals
  nor disturbs a bit stored in ``S`` (up to a unitary a feed-forward ``W_y`` can undo).
  The search maximises the read contrast with a large penalty on flatness, over many
  random starts. A structural (training-free) test of whether a readout layout *can*
  hold a protected bit.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .data import Splits
from .instrument import Instrument
from .linalg import DTYPE, REAL_DTYPE, dagger
from .train import TrainConfig, TrainResult, nll_per_symbol, train


# -- a differentiable operational strength ------------------------------------------
def operational_strength_t(inst: Instrument) -> torch.Tensor:
    """``eta`` as a 0-dim tensor that carries gradients (see ``qmem.metrics``)."""
    e = inst.effects()
    e = 0.5 * (e + dagger(e))
    top = min(inst.n_outcomes, inst.dim)
    if top <= 1:
        return torch.zeros((), dtype=REAL_DTYPE)
    lam = torch.linalg.eigvalsh(e)[:, -1]
    return (lam.sum() - 1) / (top - 1)


def strength_prior_loss(eta0: float, lam: float) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Loss ``NLL + lam (eta - eta0)^2`` (``eta``: the model's operational strength)."""

    def loss(model: Any, seqs: torch.Tensor) -> torch.Tensor:
        eta = operational_strength_t(model.instrument())
        return nll_per_symbol(model, seqs) + lam * (eta - eta0) ** 2

    return loss


# -- staged training ---------------------------------------------------------------
@dataclass
class Stage:
    """One stage of ``train_stages``. ``frozen``: parameter-name substrings to hold fixed."""

    splits: Splits
    cfg: TrainConfig
    frozen: Sequence[str] = ()
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor] | None = None
    label: str = ""
    before: Callable[[Any], None] | None = None  # e.g. set the strength before the stage


@dataclass
class StagedResult:
    result: TrainResult  # the last stage's result, with steps and wall time summed
    stages: list[dict[str, Any]] = field(default_factory=list)


def train_stages(model: Any, stages: Sequence[Stage]) -> StagedResult:
    """Train ``model`` in place through ``stages`` in order (module doc).

    Every parameter is trainable again after the last stage. Each stage starts a fresh
    optimiser. The returned result is the last stage's, with ``steps_run`` and
    ``wall_seconds`` summed over stages; ``stages`` keeps each stage's summary.
    """
    trainable = {n: p.requires_grad for n, p in model.named_parameters()}
    summaries, total_steps, total_wall, res = [], 0, 0.0, None
    try:
        for st in stages:
            if st.before is not None:
                st.before(model)
            for n, p in model.named_parameters():
                p.requires_grad_(trainable[n] and not any(f in n for f in st.frozen))
            res = train(model, st.splits, st.cfg, st.loss_fn)
            total_steps += res.steps_run
            total_wall += res.wall_seconds
            summaries.append(
                {
                    "label": st.label,
                    "status": res.status,
                    "steps_run": res.steps_run,
                    "best_step": res.best_step,
                    "best_val": res.best_val_nll,
                    "frozen": list(st.frozen),
                    "strength_end": res.curves["strength"][-1] if res.curves["strength"] else [],
                }
            )
            if res.status == "diverged":
                break
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(trainable[n])
    assert res is not None, "train_stages needs at least one stage"
    out = copy.copy(res)
    out.steps_run, out.wall_seconds = total_steps, total_wall
    return StagedResult(out, summaries)


# -- protected subspaces -----------------------------------------------------------
def _traceless_norm2(m: torch.Tensor) -> torch.Tensor:
    k = m.shape[-1]
    t = torch.diagonal(m, dim1=-2, dim2=-1).sum(-1) / k
    dev = m - t[..., None, None] * torch.eye(k, dtype=m.dtype)
    return (dev.abs() ** 2).sum((-2, -1))


def protected_subspace_search(
    effects: torch.Tensor,
    flat: Sequence[int],
    read: tuple[int, int],
    flat_targets: Sequence[float] | None = None,
    restarts: int = 32,
    steps: int = 200,
    penalties: Sequence[float] = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
    tol: float = 1e-9,
    seed: int = 0,
    dim: int = 2,
) -> dict[str, Any]:
    """Best read contrast on a subspace where the ``flat`` effects are proportional to I.

    ``effects``: ``(Y, d, d)``. For an isometry ``S`` (``d x dim``) the flatness residual
    is ``sum_{y in flat} ||S^+ E_y S - c_y I||_F^2``, with ``c_y`` the mean eigenvalue
    (any level) or ``flat_targets[y]`` (a fixed level, e.g. the source's distractor
    probability). The read contrast is the spread ``lambda_max - lambda_min`` of
    ``S^+ (E_a - E_b) S`` for ``read = (a, b)``: how much ``p(a) - p(b)`` can differ
    between two states of ``S``. From each of ``restarts`` random ``S``, Adam maximises
    ``contrast - mu * residual`` for ``steps`` steps at each penalty ``mu`` in turn (a
    continuation, so the end point is feasible if a feasible ``S`` is near). Returns the
    best contrast among the end points with residual below ``tol`` (0 if none), the
    number of feasible end points, and the best end point's contrast and residual.
    """
    gen = torch.Generator().manual_seed(seed)
    e = 0.5 * (effects + dagger(effects)).detach().to(DTYPE)
    d = e.shape[-1]
    diff = e[read[0]] - e[read[1]]
    flat_e = e[list(flat)]
    eye_k = torch.eye(dim, dtype=DTYPE)
    targets = None if flat_targets is None else torch.as_tensor(flat_targets, dtype=REAL_DTYPE)

    def measure(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Contrast and residual of every restart at once (``w``: ``(R, 2, d, dim)``)."""
        q, _ = torch.linalg.qr(torch.complex(w[:, 0], w[:, 1]))  # (R, d, dim)
        comp = dagger(q)[:, None] @ flat_e[None] @ q[:, None]  # (R, F, dim, dim)
        if targets is None:
            res = _traceless_norm2(comp).sum(-1)
        else:
            res = ((comp - targets[None, :, None, None] * eye_k).abs() ** 2).sum((-3, -2, -1))
        ev = torch.linalg.eigvalsh(dagger(q) @ diff @ q)
        return ev[:, -1] - ev[:, 0], res

    w = torch.randn(restarts, 2, d, dim, generator=gen, dtype=REAL_DTYPE)
    with torch.enable_grad():  # callers may sit inside no_grad (e.g. post-training analysis)
        _ascend(w, measure, penalties, steps)
    with torch.no_grad():
        con, res = measure(w)
    feasible = res < tol
    order = sorted(range(restarts), key=lambda i: (bool(feasible[i]), float(con[i])))
    best = order[-1]
    ev = torch.linalg.eigvalsh(diff)
    return {
        "contrast": float(con[feasible].max()) if bool(feasible.any()) else 0.0,
        "n_feasible": int(feasible.sum()),
        "best_run_contrast": float(con[best]),
        "best_run_residual": float(res[best]),
        "restarts": restarts,
        "max_contrast_unconstrained": float(ev[-1] - ev[0]),
    }


def _ascend(w: torch.Tensor, measure: Callable, penalties: Sequence[float], steps: int) -> None:
    w.requires_grad_(True)
    for mu in penalties:
        opt = torch.optim.Adam([w], lr=0.02)
        for _ in range(steps):
            con, res = measure(w)
            loss = (-con + mu * res).sum()  # restarts are independent: sum their losses
            opt.zero_grad()
            loss.backward()
            opt.step()
    w.requires_grad_(False)


def g_for_strength(eta: float) -> float:
    """The ``g`` with ``sin^2 g = eta`` (one coupling; ``eta = sin^2 g``, ``qmem.metrics``)."""
    return math.asin(math.sqrt(min(max(eta, 0.0), 1.0)))


# -- uMPS as an instrument ------------------------------------------------------------
@torch.no_grad()
def umps_instrument(model: Any) -> tuple[Instrument, torch.Tensor]:
    """A trained uMPS (``qmem.baselines.UMPSModel``) as an instrument and initial memory.

    With ``Sigma`` the fixed point of ``E*(X) = sum A^+ X A`` (eigenvalue ``lambda``),
    ``K_{y,k} = Sigma^{1/2} A[y,k] Sigma^{-1/2} / sqrt(lambda)`` is complete
    (``sum K^+ K = I``), and the left environment ``L`` maps to the memory
    ``rho = Sigma^{1/2} L Sigma^{1/2} / Tr(L Sigma)``, so ``p(y|L) = Tr(K_y rho K_y^+)``
    (``docs/architecture.md`` section 8; tested against ``predictive_probs``).
    """
    a, sigma = model.prepare()
    ev, vec = torch.linalg.eigh(0.5 * (sigma + dagger(sigma)))
    if float(ev.min()) <= 0:
        raise ValueError("uMPS fixed point is not positive definite")
    s_half = (vec * ev.sqrt().to(DTYPE)) @ dagger(vec)
    s_mhalf = (vec * (1 / ev.sqrt()).to(DTYPE)) @ dagger(vec)
    lam = float(torch.einsum("ykji,jl,yklm->im", a.conj(), sigma, a).trace().real) / float(
        sigma.trace().real
    )
    k = s_half @ a @ s_mhalf / math.sqrt(lam)
    l = model.boundary()
    rho0 = s_half @ (l[:, None] * l.conj()[None, :]) @ s_half
    rho0 = rho0 / rho0.trace().real
    return Instrument(k), rho0
