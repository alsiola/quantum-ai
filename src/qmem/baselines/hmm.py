"""Classical HMM with ``D`` hidden states (edge-emitting, the classical instrument).

The model is given by labelled transition matrices
``L[y, i, j] = P(emit y, move to j | state i)`` and an initial distribution
``pi``, the same representation as the sources in ``qmem.tasks``. This
edge-emitting (Mealy) form is the classical analogue of a quantum instrument:
``L[y]`` plays the role of ``rho -> K_y rho K_y^dagger``, and the belief state
plays the role of the memory. A 2-state model of this form represents the Even
process exactly (a state-emitting HMM needs 3).

Two ways to fit it:

* gradient training through the forward algorithm (``qmem.train.train``),
  with ``L`` and ``pi`` parameterised by softmax logits;
* Baum-Welch / EM (``fit_em``), from a random start drawn from the seed, with
  validation early stopping.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import nn

from ..linalg import REAL_DTYPE
from .base import RecurrentModel, check_sequences


class HMMModel(RecurrentModel):
    family = "hmm"

    def __init__(self, n_states: int, n_symbols: int, seed: int = 0, init_scale: float = 1.0):
        super().__init__(n_symbols)
        self.n_states = n_states
        gen = torch.Generator().manual_seed(seed)
        d, y = n_states, n_symbols
        self.trans_logits = nn.Parameter(
            init_scale * torch.randn(d, y * d, generator=gen, dtype=REAL_DTYPE)
        )
        self.init_logits = nn.Parameter(
            init_scale * torch.randn(d, generator=gen, dtype=REAL_DTYPE)
        )

    # -- parameters as probabilities -------------------------------------------
    def labelled(self) -> torch.Tensor:
        """``L[y, i, j]``, shape ``(Y, D, D)``."""
        d, y = self.n_states, self.n_symbols
        return torch.softmax(self.trans_logits, -1).reshape(d, y, d).permute(1, 0, 2)

    def initial(self) -> torch.Tensor:
        return torch.softmax(self.init_logits, -1)

    @torch.no_grad()
    def set_probabilities(self, labelled: torch.Tensor, initial: torch.Tensor) -> None:
        """Write probabilities back as logits (``log 0 = -inf`` keeps exact zeros)."""
        d, y = self.n_states, self.n_symbols
        flat = labelled.permute(1, 0, 2).reshape(d, y * d)
        self.trans_logits.copy_(torch.log(flat))
        self.init_logits.copy_(torch.log(initial))

    # -- recurrent interface ----------------------------------------------------
    def prepare(self) -> tuple[torch.Tensor, torch.Tensor]:
        lab = self.labelled()
        return lab, lab.sum(-1).T  # (Y, D, D), emit (D, Y)

    def init_state(self, ctx, batch: int) -> torch.Tensor:
        return self.initial().expand(batch, -1)

    def probs(self, ctx, state: torch.Tensor) -> torch.Tensor:
        return state @ ctx[1]

    def update(self, ctx, state: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        unnorm = torch.einsum("bi,bij->bj", state, ctx[0][y])
        z = unnorm.sum(-1, keepdim=True)
        possible = z > 0
        return torch.where(possible, unnorm / torch.where(possible, z, torch.ones_like(z)), state)

    # -- bookkeeping ------------------------------------------------------------
    def n_free_parameters(self) -> int:
        d, y = self.n_states, self.n_symbols
        return d * (y * d - 1) + (d - 1)

    def state_dim(self) -> int:
        return self.n_states

    def state_kind(self) -> str:
        return "belief over D classical states (D-1 real dof)"

    def flops_per_step(self) -> int:
        return self.n_states**2 + self.n_states * self.n_symbols

    def extra_info(self) -> dict[str, Any]:
        return {"n_free_parameters": self.n_free_parameters()}


@torch.no_grad()
def forward_backward(
    labelled: torch.Tensor, initial: torch.Tensor, seqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled forward-backward. Returns expected counts ``N[y,i,j]``, ``gamma_0`` sum, log-lik.

    ``a_t = P(s_t | y_<=t)``; ``c_t = P(y_t | y_<t)``;
    ``P(s_{t-1}=i, s_t=j | y) = a_{t-1}(i) L[y_t,i,j] b_t(j) / c_t``.
    """
    n, length = seqs.shape
    a = [initial.expand(n, -1)]
    cs = []
    for t in range(length):
        u = torch.einsum("bi,bij->bj", a[-1], labelled[seqs[:, t]])
        c = u.sum(-1)
        cs.append(c)
        a.append(u / c[:, None])
    counts = torch.zeros_like(labelled)
    b = torch.ones(n, labelled.shape[1], dtype=labelled.dtype)
    for t in range(length - 1, -1, -1):
        lt = labelled[seqs[:, t]]  # (B, D, D)
        xi = a[t][:, :, None] * lt * b[:, None, :] / cs[t][:, None, None]
        counts.index_add_(0, seqs[:, t], xi)
        b = torch.einsum("bij,bj->bi", lt, b) / cs[t][:, None]
    gamma0 = (initial[None] * b).sum(0)
    loglik = torch.log(torch.stack(cs, 1)).sum()
    return counts, gamma0, loglik


@torch.no_grad()
def nll_per_symbol(model: HMMModel, seqs: torch.Tensor) -> float:
    return float(-model.step_log_probs(seqs).mean())


@torch.no_grad()
def fit_em(
    model: HMMModel,
    train: torch.Tensor,
    val: torch.Tensor,
    max_iters: int = 500,
    patience: int = 20,
    min_delta: float = 1e-5,
) -> dict[str, Any]:
    """Baum-Welch on ``train`` from the model's current parameters, early-stopped on ``val``.

    Keeps the parameters with the best validation NLL (as the gradient trainer
    does). Returns a record with the curves, like ``qmem.train.TrainResult``.
    """
    start = time.perf_counter()
    train = check_sequences(train, model.n_symbols)
    lab, init = model.labelled().clone(), model.initial().clone()
    curves: dict[str, list] = {"iter": [], "train_nll": [], "val_nll": [], "elapsed": []}
    best = (math.inf, -1, lab.clone(), init.clone())
    bad = 0
    status = "completed"
    for it in range(max_iters):
        counts, gamma0, loglik = forward_backward(lab, init, train)
        model.set_probabilities(lab, init)
        v = nll_per_symbol(model, val)
        curves["iter"].append(it)
        curves["train_nll"].append(float(-loglik) / train.numel())
        curves["val_nll"].append(v)
        curves["elapsed"].append(time.perf_counter() - start)
        if not math.isfinite(v):
            status = "diverged"
            break
        if v < best[0] - min_delta:
            best = (v, it, lab.clone(), init.clone())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                status = "early_stopped"
                break
        # M-step. A state that is never visited keeps its old row (no 0/0).
        row = counts.sum(dim=(0, 2))  # (D,)
        visited = row > 0
        new = counts / torch.where(visited, row, torch.ones_like(row))[None, :, None]
        lab = torch.where(visited[None, :, None], new, lab)
        init = gamma0 / gamma0.sum()
    model.set_probabilities(best[2], best[3])
    return {
        "status": status,
        "steps_run": len(curves["iter"]),
        "best_step": best[1],
        "best_val_nll": best[0],
        "curves": curves,
        "wall_seconds": time.perf_counter() - start,
        "message": "",
    }
