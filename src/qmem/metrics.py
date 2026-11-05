"""Operational information-gain and disturbance metrics for any ``Instrument``.

Every metric here is a function of the Kraus tensor only, so it applies to any
layout (``joint_depth > 0``, feed-forward, a hand-built or unconstrained HQMM
instrument). Definitions (``d`` = memory dimension, ``Y`` = number of outcomes,
``E_y = sum_e K_{y,e}^dagger K_{y,e}``); full derivations are in
``docs/architecture.md`` section 2.

Information gain (functions of the effects only, so blind to any unitary
applied after the readout):

* ``tv_gain``: ``max_{rho, sigma} TV(p(.|rho), p(.|sigma))``, the most the
  outcome distribution can differ between two inputs. Exact:
  ``max_S [lambda_max(E_S) - lambda_min(E_S)]`` over outcome subsets ``S``,
  with ``E_S = sum_{y in S} E_y`` (for ``Y = 2``: the spread of ``E_0``).
* ``estimation_fidelity`` ``G``: the mean fidelity between a Haar-random pure
  input and the best guess made from the outcome (Banaszek 2001),
  ``G = (d + sum_y lambda_max(E_y)) / (d (d + 1))``.
* ``operational_strength`` ``eta``: ``G`` rescaled to ``[0, 1]`` for the given
  ``Y`` and ``d``, ``eta = (sum_y lambda_max(E_y) - 1) / (min(Y, d) - 1)``.
  ``eta = 0`` iff every ``E_y`` is proportional to ``I`` (no information);
  ``eta = 1`` iff the effects are as sharp as ``Y`` outcomes on ``d`` levels
  allow. For the weak measurement ``D_y(g)`` with one coupling, ``eta = sin^2 g``.
* ``mutual_information``: ``I(R:Y)`` in nats between the outcome and a
  reference ``R`` purifying the memory state ``rho`` (Buscemi, Hayashi and
  Horodecki 2008):
  ``S(rho) - sum_y p_y S(sqrt(rho) E_y sqrt(rho) / p_y)``. Evaluated at the
  stationary state of the unconditioned channel by ``operational_summary``.

Disturbance (correction-invariant: the best unitary correction *per outcome*
is applied first, so unitary feed-forward ``K_y -> W_y K_y`` costs nothing):

* ``corrected_entanglement_fidelity`` ``F_e = max_{U_y} sum_{y,e}
  |Tr(U_y^dagger K_{y,e})|^2 / d^2``. With one Kraus operator per outcome
  this is ``sum_y ||K_y||_1^2 / d^2`` (``|Tr(U^dagger K)| <= ||K||_1``, with
  equality at the polar factor of ``K``). With several, a monotone
  minorise-maximise iteration over polar factors is used.
* ``operation_fidelity`` ``F = (d F_e + 1) / (d + 1)``: the mean fidelity of
  the (corrected) output with a Haar-random pure input (Banaszek's ``F``).
  ``disturbance = 1 - F``.

Trade-off: ``banaszek_bound(G, d)`` is the largest ``F`` any instrument on a
``d``-level system can have at estimation fidelity ``G`` (Banaszek 2001):
``sqrt(F - 1/(d+1)) <= sqrt(G - 1/(d+1)) + sqrt((d-1)(2/(d+1) - G))``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch

from .instrument import Instrument
from .linalg import DTYPE, dagger, eye, trace


def _evals(e: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(0.5 * (e + dagger(e)))


def _entropy(rho: torch.Tensor) -> float:
    ev = _evals(rho).clamp(min=0)
    ev = ev[ev > 1e-15]
    return float(-(ev * ev.log()).sum())


# -- information gain ---------------------------------------------------------------
def tv_gain(inst: Instrument) -> float:
    """``max_{rho,sigma} TV(p(.|rho), p(.|sigma))`` (exact, over outcome subsets)."""
    eff = inst.effects().detach()
    y = inst.n_outcomes
    best = 0.0
    for r in range(1, y):
        for subset in itertools.combinations(range(y), r):
            ev = _evals(eff[list(subset)].sum(0))
            best = max(best, float(ev[-1] - ev[0]))
    return best


def sum_max_effect_eigenvalues(inst: Instrument) -> float:
    """``sum_y lambda_max(E_y)``, between 1 (no information) and ``min(Y, d)``."""
    return float(_evals(inst.effects().detach())[:, -1].sum())


def estimation_fidelity(inst: Instrument) -> float:
    """Banaszek's ``G``: mean fidelity of the best state guess from the outcome."""
    d = inst.dim
    return (d + sum_max_effect_eigenvalues(inst)) / (d * (d + 1))


def operational_strength(inst: Instrument) -> float:
    """``eta in [0, 1]``: ``(sum_y lambda_max(E_y) - 1) / (min(Y, d) - 1)``."""
    top = min(inst.n_outcomes, inst.dim)
    if top <= 1:
        return 0.0
    return (sum_max_effect_eigenvalues(inst) - 1) / (top - 1)


def mutual_information(inst: Instrument, rho: torch.Tensor) -> float:
    """``I(R:Y)`` in nats for memory state ``rho`` (reference ``R`` purifies ``rho``)."""
    rho = rho.detach()
    ev, vec = torch.linalg.eigh(0.5 * (rho + dagger(rho)))
    sqrt_rho = (vec * ev.clamp(min=0).sqrt().to(DTYPE)) @ dagger(vec)
    out = _entropy(rho)
    for e in inst.effects().detach():
        sigma = sqrt_rho @ e @ sqrt_rho
        p = float(trace(sigma).real)
        if p > 1e-15:
            out -= p * _entropy(sigma / p)
    return out


# -- disturbance -----------------------------------------------------------------------
def _polar(m: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(m)
    return u @ vh


def best_correction(kraus_y: torch.Tensor, iters: int = 200) -> tuple[torch.Tensor, float]:
    """``argmax_U sum_e |Tr(U^dagger K_e)|^2`` for one outcome's Kraus set ``(E, d, d)``.

    One Kraus operator: the polar factor (exact). Several: minorise-maximise.
    Since ``|a|^2 >= |b|^2 + 2 Re(conj(b) (a - b))``, the objective is bounded
    below by ``2 Re Tr(U^dagger M) + const`` with ``M = sum_e conj(b_e) K_e``,
    ``b_e = Tr(U_old^dagger K_e)``, which ``U = polar(M)`` maximises; so the
    iteration ``U <- polar(M)`` never decreases the objective. It is started
    from each ``K_e``'s polar factor and the best end point is kept.
    """
    k = kraus_y.detach()

    def value(u: torch.Tensor) -> float:
        return float((torch.einsum("ji,eji->e", u.conj(), k).abs() ** 2).sum())

    if k.shape[0] == 1:
        u = _polar(k[0])
        return u, value(u)
    best_u, best_v = eye(k.shape[-1]), value(eye(k.shape[-1]))
    for start in k:
        if float(start.abs().max()) < 1e-14:
            continue
        u = _polar(start)
        for _ in range(iters):
            c = torch.einsum("ji,eji->e", u.conj(), k)  # Tr(U^dagger K_e)
            u_new = _polar(torch.einsum("e,eij->ij", c.conj(), k))
            if (u_new - u).abs().max() < 1e-13:
                u = u_new
                break
            u = u_new
        v = value(u)
        if v > best_v:
            best_u, best_v = u, v
    return best_u, best_v


def corrected_entanglement_fidelity(inst: Instrument) -> float:
    """``max_{U_y} sum_{y,e} |Tr(U_y^dagger K_{y,e})|^2 / d^2`` (outcome-wise correction)."""
    d = inst.dim
    return sum(best_correction(inst.kraus[y])[1] for y in range(inst.n_outcomes)) / d**2


def entanglement_fidelity(inst: Instrument) -> float:
    """Uncorrected ``sum_{y,e} |Tr K_{y,e}|^2 / d^2`` (counts feed-forward as disturbance)."""
    k = inst.kraus.detach()
    return float((trace(k).abs() ** 2).sum()) / inst.dim**2


def operation_fidelity(inst: Instrument, corrected: bool = True) -> float:
    """Banaszek's ``F = (d F_e + 1) / (d + 1)``: mean output-input fidelity, pure inputs."""
    d = inst.dim
    fe = corrected_entanglement_fidelity(inst) if corrected else entanglement_fidelity(inst)
    return (d * fe + 1) / (d + 1)


def banaszek_bound(g: float | torch.Tensor, d: int) -> torch.Tensor:
    """Largest operation fidelity ``F`` compatible with estimation fidelity ``G`` (Banaszek 2001).

    Valid for ``1/d <= G <= 2/(d+1)``.
    """
    g = torch.as_tensor(g, dtype=torch.float64)
    a = (g - 1 / (d + 1)).clamp(min=0).sqrt()
    b = ((d - 1) * (2 / (d + 1) - g)).clamp(min=0).sqrt()
    return (a + b) ** 2 + 1 / (d + 1)


# -- stationary state ---------------------------------------------------------------
def superoperator(inst: Instrument) -> torch.Tensor:
    """Matrix of the unconditioned channel on row-major ``vec(rho)``: ``sum K ⊗ conj(K)``."""
    k = inst.kraus.detach()
    d = inst.dim
    return torch.einsum("yeij,yekl->ikjl", k, k.conj()).reshape(d * d, d * d)


def stationary_state(inst: Instrument) -> tuple[torch.Tensor, dict[str, Any]]:
    """A fixed point ``rho = E(rho)`` of the unconditioned channel, plus diagnostics.

    Solved as a least-squares linear system with ``Tr rho = 1``. ``n_unit``
    counts eigenvalues of the channel within ``1e-8`` of 1; if it is above 1
    the fixed point is not unique and the returned one is the minimum-norm
    solution.
    """
    inst = Instrument(inst.kraus.detach())
    d = inst.dim
    s = superoperator(inst)
    a = torch.cat([s - eye(d * d), eye(d).reshape(1, -1)], 0)
    b = torch.zeros(d * d + 1, 1, dtype=DTYPE)
    b[-1] = 1.0
    x = torch.linalg.lstsq(a, b).solution
    rho = x.reshape(d, d)
    rho = 0.5 * (rho + dagger(rho))
    rho = rho / trace(rho)
    ev = torch.linalg.eigvals(s)
    info = {
        "residual": float((inst.channel(rho) - rho).abs().max()),
        "n_unit_eigenvalues": int(((ev - 1).abs() < 1e-8).sum()),
        "min_eigenvalue": float(_evals(rho)[0]),
    }
    return rho, info


# -- everything at once ---------------------------------------------------------------
def operational_summary(inst: Instrument, rho: torch.Tensor | None = None) -> dict[str, Any]:
    """Every metric above; ``mutual_information`` at ``rho`` (default: stationary state)."""
    inst = Instrument(inst.kraus.detach())
    info: dict[str, Any] = {}
    if rho is None:
        rho, info = stationary_state(inst)
    d = inst.dim
    g = estimation_fidelity(inst)
    f = operation_fidelity(inst)
    mi = mutual_information(inst, rho)
    mi_mixed = mutual_information(inst, eye(d) / d)
    return {
        "dim": d,
        "n_outcomes": inst.n_outcomes,
        "tv_gain": tv_gain(inst),
        "operational_strength": operational_strength(inst),
        "estimation_fidelity": g,
        "operation_fidelity": f,
        "operation_fidelity_uncorrected": operation_fidelity(inst, corrected=False),
        "disturbance": 1 - f,
        "corrected_entanglement_fidelity": corrected_entanglement_fidelity(inst),
        "banaszek_bound_F": float(banaszek_bound(g, d)),
        "mutual_information_stationary": mi,
        "mutual_information_mixed": mi_mixed,
        "max_mutual_information": math.log(min(inst.n_outcomes, d)),
        "stationary": info,
    }
