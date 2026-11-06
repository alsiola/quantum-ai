"""Initialisations, soft-target fitting and restarts for trainability (Phase 4a, issue #12).

Everything here is opt-in: the defaults of ``ModelConfig``, the baselines and
``TrainConfig`` reproduce the earlier phases exactly.

**Initial symbol probabilities.** The near-identity initialisations start from an
instrument whose effects are ``E_y = p_y I`` (exactly or nearly), so the predictive is
the memoryless ``p_y`` whatever the history. ``init_probs`` gives ``p_y``:
``"uniform"``, an explicit list, or ``"data"`` (the unigram frequencies of the
training split; ``resolve_init_probs`` substitutes them before the model is built,
and the record keeps the resolved list).

**Architecture A, ``init = "product"``** (``ModelConfig``). Needs ``entangler =
"cz"``. Every hardware-efficient angle is drawn uniformly on
``[-pi init_scale, pi init_scale]`` as usual (so a small ``init_scale`` makes it a
small perturbation), and the first joint layer's ``RY`` on output ancilla ``k`` gets
``2 asin(sqrt(P(bit_k = 1)))`` added, so that the ancillas are prepared in the product
of the bit marginals of ``p_y``. At zero perturbation and ``strength = 0`` the
circuit is ``(prep on the ancillas) x (a diagonal +-1 unitary)``: CZ layers square to
the identity and act on the memory only through phases, so every Kraus operator is
``K_y = c_y U_y`` with ``U_y`` unitary and ``E_y = |c_y|^2 I`` (tested). With CNOT
entanglers no setting of the angles does this, which is why CZ is required.

**HQMM and uMPS, ``init = "identity"``** (``qmem.baselines``): ``K_y = sqrt(p_y) I``
(``A_y`` for the uMPS) plus Gaussian noise of total size ``init_scale`` per column,
so the perturbation does not grow with the dimension.

**Soft targets** (``soft_target_loss``): cross-entropy between a teacher's predictive
``q(. | y_<t)`` and the model's, at every position of the given sequences. With the
source as teacher it is the expected NLL without sampling noise; with a trained
model as teacher it fits one parameterisation to another's solution (a reachability
test) or warm-starts a larger model from a smaller one's function.

**Restarts** (``TrainConfig.restarts``): ``train_with_restarts`` trains ``restarts``
initialisations for ``restart_steps`` each, keeps the one with the best validation
NLL and trains it with the full budget. Selection is on validation only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .linalg import REAL_DTYPE

INIT_MODES_A = ("random", "product")
INIT_MODES_BASELINE = ("random", "identity")


def unigram(sequences: torch.Tensor, n_symbols: int) -> list[float]:
    """Symbol frequencies of ``sequences`` (every position counted)."""
    counts = torch.bincount(torch.as_tensor(sequences).flatten(), minlength=n_symbols)
    return (counts.double() / counts.sum()).tolist()


def symbol_probs(spec: Any, n_symbols: int) -> torch.Tensor:
    """``p_y`` from ``"uniform"`` or an explicit list (``"data"`` must be resolved first)."""
    if spec is None or spec == "uniform":
        return torch.full((n_symbols,), 1.0 / n_symbols, dtype=REAL_DTYPE)
    if isinstance(spec, str):
        raise ValueError(f"init_probs {spec!r} must be resolved (resolve_init_probs) first")
    p = torch.as_tensor(spec, dtype=REAL_DTYPE)
    if p.shape != (n_symbols,) or (p < 0).any() or abs(float(p.sum()) - 1) > 1e-9:
        raise ValueError(f"init_probs must be {n_symbols} probabilities summing to 1")
    return p


def resolve_init_probs(cfg: dict[str, Any], train: torch.Tensor, n_symbols: int) -> dict[str, Any]:
    """``cfg`` with ``init_probs = "data"`` replaced by the training unigram (else unchanged)."""
    if cfg.get("init_probs") == "data":
        return {**cfg, "init_probs": unigram(train, n_symbols)}
    return cfg


def bit_marginals(p: torch.Tensor, n_bits: int) -> list[float]:
    """``P(bit_k = 1)`` for each output qubit ``k`` (big-endian: bit 0 is most significant)."""
    y = torch.arange(len(p))
    return [float(p[((y >> (n_bits - 1 - k)) & 1) == 1].sum()) for k in range(n_bits)]


def apply_product_init(model: Any, config: Any) -> None:
    """Turn a freshly built A (``build_model``) into the near-product start (module doc)."""
    from .circuits import HardwareEfficientBlock

    c = config
    if c.entangler != "cz":
        raise ValueError('init = "product" needs entangler = "cz" (CZ layers square to I)')
    if "ry" not in c.rotations:
        raise ValueError('init = "product" needs an "ry" rotation')
    n = c.n_memory + c.n_output + c.n_env
    joint = [
        b
        for b in model.circuit.blocks
        if isinstance(b, HardwareEfficientBlock) and b.qubits == tuple(range(n))
    ]
    if not joint:  # feed-forward / coupling-only layout: near-identity V and W_y only
        return
    p = symbol_probs(c.init_probs, 2**c.n_output)
    r = c.rotations.index("ry")
    with torch.no_grad():
        for k, pk in enumerate(bit_marginals(p, c.n_output)):
            joint[0].theta[0, c.n_memory + k, r] += 2 * math.asin(math.sqrt(min(max(pk, 0), 1)))


# -- soft targets -------------------------------------------------------------------
def soft_target_loss(
    teacher: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Loss ``-(1/BT) sum_{b,t,y} q_t(y) log p_t(y)`` against ``teacher(seqs)`` (B, T+1, Y).

    ``teacher`` is e.g. ``task.predictive`` or a trained model's ``predictive_probs``;
    it is evaluated without gradient. Terms with ``q = 0`` contribute 0 (so a model
    ``p = 0`` where the teacher is also 0 is not penalised).
    """

    def loss(model: Any, seqs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = teacher(seqs)[:, :-1]
        p = model.predictive_probs(seqs)[:, :-1]
        terms = torch.where(q > 0, q * torch.log(torch.where(q > 0, p, torch.ones_like(p))), 0)
        return -terms.sum(-1).mean()

    return loss


def teacher_entropy(teacher: Callable[[torch.Tensor], torch.Tensor], seqs: torch.Tensor) -> float:
    """Mean ``H(q_t)`` (nats): subtract from the soft-target loss to get the mean KL."""
    with torch.no_grad():
        q = teacher(seqs)[:, :-1]
        return float(-torch.where(q > 0, q * q.clamp_min(1e-300).log(), 0).sum(-1).mean())


# -- restarts -----------------------------------------------------------------------
def restart_seed(seed: int, r: int) -> int:
    """Model seed of restart ``r`` of evaluation seed ``seed`` (restart 0 is ``seed``)."""
    return seed + 10_007 * r


# -- diagnostics at initialisation ---------------------------------------------------
def init_statistics(
    make_model: Callable[[int], Any], seeds: Sequence[int], batch: torch.Tensor
) -> dict[str, Any]:
    """Gradient and predictive statistics of freshly initialised models, over ``seeds``.

    * ``grad_norm_mean/sd``: the NLL gradient norm on ``batch``;
    * ``grad_var``: ``Var_seeds[dL/dtheta_k]`` averaged over ``k`` (the barren-plateau
      quantity) and ``grad_var_max`` its largest coordinate;
    * ``nll``: the initial NLL; ``pred_entropy``: mean ``H(p(.|y_<t))``;
    * ``history_tv``: mean TV between ``p(.|y_<t)`` and its average over the batch at
      the same ``t`` (how much the history moves the prediction);
    * ``avg_purity``: purity of the outcome-averaged memory ``E^T(rho_0)`` (quantum
      models with a channel).
    """
    from .states import purity

    grads, norms, nll, ent, tv, pur = [], [], [], [], [], []
    for s in seeds:
        m = make_model(s)
        m.zero_grad()
        loss = -m.step_log_probs(batch).mean()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters() if p.requires_grad])
        grads.append(g)
        norms.append(float(g.norm()))
        nll.append(float(loss.detach()))
        with torch.no_grad():
            pp = m.predictive_probs(batch)[:, :-1]
            ent.append(float(-(pp * pp.clamp_min(1e-300).log()).sum(-1).mean()))
            tv.append(float(0.5 * (pp - pp.mean(0, keepdim=True)).abs().sum(-1).mean()))
            if hasattr(m, "instrument") and hasattr(m, "rho0"):
                pur.append(float(purity(m.unconditioned_memory(batch.shape[1])).real))
    g = torch.stack(grads)

    def ms(v: list[float]) -> tuple[float, float]:
        if not v:
            return math.nan, math.nan
        x = torch.tensor(v, dtype=REAL_DTYPE)
        return float(x.mean()), float(x.std()) if len(v) > 1 else math.nan

    return {
        "n_parameters": int(g.shape[1]),
        "n_seeds": len(seeds),
        "grad_norm_mean": ms(norms)[0],
        "grad_norm_sd": ms(norms)[1],
        "grad_var": float(g.var(0).mean()),
        "grad_var_max": float(g.var(0).max()),
        "nll_mean": ms(nll)[0],
        "nll_sd": ms(nll)[1],
        "pred_entropy_mean": ms(ent)[0],
        "history_tv_mean": ms(tv)[0],
        "avg_purity_mean": ms(pur)[0],
    }
