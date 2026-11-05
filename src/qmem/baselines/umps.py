"""Complex uniform MPS (Born machine) sequence model, optionally locally purified.

Parameters: complex tensors ``A[y, k]`` of shape ``(chi, chi)`` for each symbol
``y`` and purification index ``k = 1..kappa`` (``kappa = 1``: a plain uMPS;
``kappa > 1``: a uniform locally purified state), and a complex left boundary
vector ``l``. Nothing is constrained: no isometry, no gauge fixing.

The probability of a prefix is the infinite-chain Born rule with the right
boundary at the dominant fixed point of the transfer map:

    P(y_1..y_t) = Tr(L_t Sigma) / (lambda^t Tr(L_0 Sigma)),
    L_0 = l l^dagger,  L_t = sum_k A[y_t,k] L_{t-1} A[y_t,k]^dagger,
    E*(Sigma) = sum_{y,k} A[y,k]^dagger Sigma A[y,k] = lambda Sigma.

So the next-symbol conditionals are

    p(y | L) = Tr(sum_k A[y,k] L A[y,k]^dagger Sigma) / sum_y' (same for y').

The denominator equals ``lambda Tr(L Sigma)`` when ``Sigma`` is the exact fixed
point; we normalise by the explicit sum over ``y`` so that the conditionals are
exactly normalised for any ``Sigma >= 0``, and the model is always a valid
autoregressive distribution. ``Sigma`` comes from power iteration on ``E*``
(``fp_iters`` steps without gradient, then ``fp_grad_iters`` differentiable
steps from there). If the final residual ``|E*(Sigma)/Tr - Sigma|`` exceeds
``fp_tol``, the call is counted in ``diagnostics["fixed_point_unconverged"]``
(the conditionals are then those of the model with that ``Sigma``, still
normalised, but no longer exactly the infinite-chain uMPS marginals).

With ``Sigma`` positive definite, the gauge ``A -> Sigma^{1/2} A Sigma^{-1/2}``
makes ``sum A^dagger A = lambda I``, so a uMPS of bond dimension ``chi`` is the
same model class as a quantum instrument with one Kraus operator per outcome
(``kappa`` Kraus operators per outcome for the purified version) on a
``chi``-dimensional memory. It is the "complex linear model of this dimension"
against which the circuit parameterisation of architecture A is compared.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..linalg import DTYPE, REAL_DTYPE, trace
from .base import RecurrentModel


class UMPSModel(RecurrentModel):
    family = "umps"

    def __init__(
        self,
        bond_dim: int,
        n_symbols: int,
        purification: int = 1,
        seed: int = 0,
        init_scale: float = 1.0,
        fp_iters: int = 200,
        fp_grad_iters: int = 20,
        fp_tol: float = 1e-8,
    ):
        super().__init__(n_symbols)
        self.bond_dim, self.purification = bond_dim, purification
        if purification > 1:
            self.family = "lps"
        self.fp_iters, self.fp_grad_iters, self.fp_tol = fp_iters, fp_grad_iters, fp_tol
        gen = torch.Generator().manual_seed(seed)
        shape = (n_symbols, purification, bond_dim, bond_dim)
        # Scale so that the transfer map has spectral radius about 1 at init.
        s = init_scale / math.sqrt(2 * n_symbols * purification * bond_dim)
        self.a_re = nn.Parameter(s * torch.randn(shape, generator=gen, dtype=REAL_DTYPE))
        self.a_im = nn.Parameter(s * torch.randn(shape, generator=gen, dtype=REAL_DTYPE))
        b = 1 / math.sqrt(2 * bond_dim)
        self.l_re = nn.Parameter(b * torch.randn(bond_dim, generator=gen, dtype=REAL_DTYPE))
        self.l_im = nn.Parameter(b * torch.randn(bond_dim, generator=gen, dtype=REAL_DTYPE))
        self.last_fp_residual = math.nan
        self._sigma_warm: torch.Tensor | None = None

    def tensors(self) -> torch.Tensor:
        """``A[y, k]``, complex ``(Y, kappa, chi, chi)``."""
        return torch.complex(self.a_re, self.a_im)

    def boundary(self) -> torch.Tensor:
        return torch.complex(self.l_re, self.l_im)

    @staticmethod
    def adjoint_map(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``E*(X) = sum_{y,k} A^dagger X A``."""
        return torch.einsum("ykji,jl,yklm->im", a.conj(), x, a)

    def _step(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = self.adjoint_map(a, x)
        y = 0.5 * (y + y.conj().transpose(-1, -2))
        return y / trace(y).real

    def fixed_point(self, a: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Dominant fixed point ``Sigma`` of ``E*`` (trace 1) and the final residual.

        Power iteration without gradient, warm-started from the previous call's
        ``Sigma`` (training changes ``A`` a little per step), stopping early once
        the residual is below ``fp_tol / 100``; then ``fp_grad_iters``
        differentiable iterations, so the gradient flows through ``Sigma``.
        """
        x = self._sigma_warm
        if x is None or x.shape[-1] != self.bond_dim:
            x = torch.eye(self.bond_dim, dtype=DTYPE) / self.bond_dim
        with torch.no_grad():
            ad = a.detach()
            for i in range(self.fp_iters):
                nx = self._step(ad, x)
                if i % 10 == 9 and float((nx - x).abs().max()) < self.fp_tol / 100:
                    x = nx
                    break
                x = nx
        for _ in range(self.fp_grad_iters):
            x = self._step(a, x)
        with torch.no_grad():
            residual = float((self._step(a, x) - x).abs().max())
            self._sigma_warm = x.detach().clone()
        return x, residual

    def prepare(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.tensors()
        sigma, residual = self.fixed_point(a)
        self.last_fp_residual = residual
        if not residual <= self.fp_tol:
            self.diagnostics["fixed_point_unconverged"] += 1
        return a, sigma

    # The state is the left environment L. For kappa = 1 it stays rank one,
    # L = psi psi^dagger, and we carry the vector psi (chi^2 work per step
    # instead of chi^3); for kappa > 1 we carry L itself.
    def _branches(self, a: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Unnormalised successors for every ``y``: ``A[y] psi`` ``(B, Y, chi)`` if
        ``kappa = 1``, else ``sum_k A[y,k] L A[y,k]^dagger`` ``(B, Y, chi, chi)``."""
        if self.purification == 1:
            return torch.einsum("yij,bj->byi", a[:, 0], state)
        return (a @ state[:, None, None] @ a.conj().transpose(-1, -2)).sum(2)

    def init_state(self, ctx, batch: int) -> torch.Tensor:
        _, sigma = ctx
        l = self.boundary()
        if self.purification == 1:
            state = l / torch.sqrt((l.conj() @ sigma @ l).real)
            return state.expand(batch, -1)
        rho = l[:, None] * l.conj()[None, :]
        rho = rho / torch.einsum("ij,ji->", rho, sigma).real
        return rho.expand(batch, -1, -1)

    def _weights(self, ctx, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Branches and their weights ``w_y = Tr(branch_y Sigma)`` (real, ``(B, Y)``)."""
        a, sigma = ctx
        br = self._branches(a, state)
        if self.purification == 1:
            w = torch.einsum("byi,ij,byj->by", br.conj(), sigma, br).real
        else:
            w = (br * sigma.T).sum((-1, -2)).real
        return br, w

    def probs(self, ctx, state: torch.Tensor) -> torch.Tensor:
        _, w = self._weights(ctx, state)
        return w / w.sum(-1, keepdim=True)

    def _select(self, br, w, state, y):
        idx = torch.arange(state.shape[0])
        wy = w[idx, y]
        possible = wy > 0
        norm = torch.where(possible, wy, torch.ones_like(wy))
        if self.purification == 1:
            new = br[idx, y] / torch.sqrt(norm)[:, None]
            return torch.where(possible[:, None], new, state)
        new = br[idx, y] / norm[:, None, None]
        return torch.where(possible[:, None, None], new, state)

    def update(self, ctx, state: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        br, w = self._weights(ctx, state)
        return self._select(br, w, state, y)

    def observe(self, ctx, state: torch.Tensor, y: torch.Tensor):
        br, w = self._weights(ctx, state)
        return w / w.sum(-1, keepdim=True), self._select(br, w, state, y)

    # -- bookkeeping ------------------------------------------------------------
    def state_dim(self) -> int:
        return self.bond_dim

    def state_kind(self) -> str:
        if self.purification == 1:
            return "pure complex chi-vector (left environment)"
        return "complex chi x chi PSD left environment"

    def flops_per_step(self) -> int:
        chi, yk = self.bond_dim, self.n_symbols * self.purification
        # complex multiply = 4 real: A L A^dagger for every (y, k), then traces with Sigma.
        return 4 * (2 * yk * chi**3 + self.n_symbols * chi**2)

    def extra_info(self) -> dict[str, Any]:
        return {
            "bond_dim": self.bond_dim,
            "purification": self.purification,
            "fixed_point_residual": self.last_fp_residual,
        }
