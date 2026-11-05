"""Exact memory requirements of a source: causal states, ``C_mu`` and ``C_q``.

For a *unifilar* ``HMMTask`` (state + symbol determine the next state):

* ``epsilon_machine`` merges predictively equivalent recurrent states by
  partition refinement (Moore/Hopcroft): two states are equivalent iff they
  give the same symbol distribution and, for every symbol, move to equivalent
  states. The blocks are the causal states; their stationary probabilities
  ``pi`` give the statistical complexity ``C_mu = H(pi)`` (bits) and the
  minimal classical state count (topological complexity ``D_mu = log2 n``).
* ``quantum_overlaps`` computes the Gram matrix of the quantum causal states
  of the standard (phaseless) unitary simulator [Binder2018 Eq. 5; Gu2012;
  Mahoney2016]: ``|s_j> -> sum_x sqrt(P(x|j)) |x> |s_lambda(j,x)>`` gives
  ``<s_j|s_k> = sum_x sqrt(P(x|j) P(x|k)) <s_lambda(j,x)|s_lambda(k,x)>``.
  Iterating that map from the identity yields exactly the overlaps of
  Mahoney's length-``L`` q-machine after ``L`` iterations, so the same loop
  gives ``C_q(L)`` and its limit ``C_q``.
* ``C_q = S(rho)``, ``rho = sum_j pi_j |s_j><s_j|``; its eigenvalues are those of
  ``sqrt(pi) G sqrt(pi)``. The quantum memory dimension is the rank of
  ``G`` (the span of the quantum causal states).

Phase-enhanced encodings [Liu2019] can do better than the phaseless ``C_q``;
the numbers here are the standard ones the cited papers report.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .instrument import Instrument
from .linalg import DTYPE, REAL_DTYPE
from .tasks import HMMTask, stationary_distribution

_ROUND = 10


def _bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def _successor(task: HMMTask) -> torch.Tensor:
    """``succ[s, y]`` = next state after emitting ``y`` from ``s`` (-1 if impossible)."""
    if not task.is_unifilar():
        raise ValueError(f"{task.name} is not unifilar; its causal states are not finite here")
    lab = task.labelled  # (Y, S, S)
    has = lab.sum(-1) > 0  # (Y, S)
    nxt = lab.argmax(-1)  # (Y, S)
    return torch.where(has, nxt, torch.full_like(nxt, -1)).T  # (S, Y)


def epsilon_machine(task: HMMTask) -> dict[str, Any]:
    """Causal states of a unifilar source (recurrent part, stationary weights).

    Returns ``labelled`` (the minimal unifilar machine, ``(Y, n, n)``), ``pi``,
    ``blocks`` (original state indices per causal state), ``n_states``,
    ``c_mu_bits`` and ``d_mu_bits = log2 n_states``.
    """
    succ = _successor(task)
    emit = task.labelled.sum(-1).T  # (S, Y)
    init = task.initial
    if torch.allclose(init @ task.transition, init, atol=1e-12):
        pi_all = init  # the start is stationary (e.g. a deterministic parity, where
    else:  # the stationary distribution is not unique)
        pi_all = stationary_distribution(task.transition)
    alive = (pi_all > 1e-14).nonzero().flatten().tolist()
    # initial partition: by emission distribution
    label = {s: tuple(round(x, _ROUND) for x in emit[s].tolist()) for s in alive}
    while True:
        ids = {k: i for i, k in enumerate(sorted(set(label.values())))}
        block = {s: ids[label[s]] for s in alive}
        new = {s: (block[s], *(block.get(int(n), -1) for n in succ[s].tolist())) for s in alive}
        if len(set(new.values())) == len(ids):
            break
        label = new
    n = len(ids)
    blocks = [[s for s in alive if block[s] == b] for b in range(n)]
    y_dim = task.n_symbols
    lab = torch.zeros(y_dim, n, n, dtype=REAL_DTYPE)
    for b, members in enumerate(blocks):
        s = members[0]
        for y in range(y_dim):
            if succ[s, y] >= 0:
                lab[y, b, block[int(succ[s, y])]] = emit[s, y]
    pi = torch.tensor([float(pi_all[m].sum()) for m in blocks], dtype=REAL_DTYPE)
    pi = pi / pi.sum()
    return {
        "labelled": lab,
        "pi": pi,
        "blocks": blocks,
        "n_states": n,
        "c_mu_bits": _bits(pi),
        "d_mu_bits": math.log2(n),
    }


def _machine_overlap_map(lab: torch.Tensor):
    """The map ``G -> G'`` of the phaseless quantum causal-state overlaps."""
    emit = lab.sum(-1).T  # (n, Y)
    y_dim = emit.shape[1]
    succ = torch.where(lab.sum(-1) > 0, lab.argmax(-1), torch.zeros_like(lab.argmax(-1))).T
    amp = torch.sqrt(emit)  # (n, Y)

    def step(g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(g)
        for y in range(y_dim):
            w = amp[:, y][:, None] * amp[:, y][None, :]
            s = succ[:, y]
            out += w * g[s][:, s]
        return out

    return step


def quantum_overlaps(task: HMMTask, max_length: int = 10_000, tol: float = 1e-14) -> dict[str, Any]:
    """Gram matrix of the quantum causal states and ``C_q(L)`` for ``L = 0, 1, ...``.

    Iterates the overlap map from ``G = I`` (``L = 0``: orthogonal classical
    states) until ``G`` changes by less than ``tol``. Returns ``gram``,
    ``cq_bits`` (the limit), ``cq_by_length`` (``C_q(L)`` for ``L = 0..``,
    stopping once it has converged), ``quantum_dim`` (rank of ``G``),
    ``converged_at`` and the causal-state ``pi``.
    """
    em = epsilon_machine(task)
    lab, pi = em["labelled"], em["pi"]
    step = _machine_overlap_map(lab)
    g = torch.eye(em["n_states"], dtype=REAL_DTYPE)
    cq = [_von_neumann_bits(g, pi)]
    converged_at = None
    for length in range(1, max_length + 1):
        g_new = step(g)
        delta = float((g_new - g).abs().max())
        g = g_new
        cq.append(_von_neumann_bits(g, pi))
        if delta < tol:
            converged_at = length
            break
    if converged_at is None:
        raise RuntimeError(f"overlaps did not converge in {max_length} iterations")
    # trailing entries are equal to tolerance; keep the sequence up to the last change
    last = len(cq) - 1
    while last > 0 and abs(cq[last - 1] - cq[-1]) < 1e-12:
        last -= 1
    return {
        "gram": g,
        "pi": pi,
        "cq_bits": cq[-1],
        "cq_by_length": cq[: max(last + 1, 2)],
        "cq_saturates_at": last,
        "quantum_dim": quantum_rank(g),
        "converged_at": converged_at,
    }


def _rho_eigenvalues(gram: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(pi)
    m = r[:, None] * gram * r[None, :]
    return torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0)


def _von_neumann_bits(gram: torch.Tensor, pi: torch.Tensor) -> float:
    return _bits(_rho_eigenvalues(gram, pi))


def quantum_rank(gram: torch.Tensor, tol: float = 1e-9) -> int:
    """Dimension spanned by states with Gram matrix ``gram``."""
    ev = torch.linalg.eigvalsh((gram + gram.T) / 2)
    return int((ev > tol * ev.max()).sum())


def memory_profile(task: Any) -> dict[str, Any]:
    """Every memory figure we can compute exactly for ``task`` (JSON-ready).

    For unifilar HMM tasks: causal-state count, ``C_mu``, ``D_mu``, ``C_q``,
    ``C_q(L)``, quantum dimension and qubits. For other sources the entry
    says why it is absent.
    """
    out: dict[str, Any] = {"name": task.name, "unifilar": bool(task.is_unifilar())}
    if not isinstance(task, HMMTask) or not task.is_unifilar():
        out["note"] = (
            "non-unifilar presentation: the causal states may be infinite and are not "
            "computed; see docs/benchmarks.md for the known figures"
        )
        return out
    em = epsilon_machine(task)
    q = quantum_overlaps(task)
    out.update(
        n_causal_states=em["n_states"],
        classical_bits=math.ceil(math.log2(em["n_states"])) if em["n_states"] > 1 else 0,
        c_mu_bits=em["c_mu_bits"],
        d_mu_bits=em["d_mu_bits"],
        cq_bits=q["cq_bits"],
        cq_by_length=q["cq_by_length"],
        cq_saturates_at=q["cq_saturates_at"],
        quantum_dim=q["quantum_dim"],
        qubits=math.ceil(math.log2(q["quantum_dim"])) if q["quantum_dim"] > 1 else 0,
    )
    return out


# -- explicit quantum models -------------------------------------------------------
def qmachine_instrument(task: HMMTask) -> tuple[Instrument, torch.Tensor]:
    """The phaseless unitary simulator of a unifilar source as an instrument.

    Writes the quantum causal states as columns of ``S`` (``r x n``, ``S^T S = G``,
    ``r`` = rank of ``G``) and solves ``M S = T`` with
    ``T_j = sum_x sqrt(P(x|j)) |x> ⊗ s_lambda(j,x)``; ``M`` is an isometry on
    the ``r``-dimensional span exactly when ``G`` is the overlap fixed point,
    and ``K_x = (<x| ⊗ I) M``. Returns the instrument (``r``-dimensional) and the
    stationary memory ``sum_j pi_j |s_j><s_j|``. Running it reproduces the
    source's statistics, which tests the quantum-memory figures independently.
    """
    em = epsilon_machine(task)
    q = quantum_overlaps(task)
    g, pi, lab = q["gram"], em["pi"], em["labelled"]
    w, v = torch.linalg.eigh((g + g.T) / 2)
    keep = w > 1e-9 * w.max()
    s = torch.sqrt(w[keep])[:, None] * v[:, keep].T  # (r, n)
    r, n = s.shape
    emit = lab.sum(-1).T  # (n, Y)
    succ = lab.argmax(-1).T  # (n, Y)
    y_dim = lab.shape[0]
    t = torch.zeros(y_dim, r, n, dtype=REAL_DTYPE)
    for j in range(n):
        for y in range(y_dim):
            if emit[j, y] > 0:
                t[y, :, j] = torch.sqrt(emit[j, y]) * s[:, succ[j, y]]
    m = t.reshape(y_dim * r, n) @ torch.linalg.pinv(s)
    kraus = m.reshape(y_dim, 1, r, r).to(DTYPE)
    rho = (s * pi[None, :]) @ s.T
    return Instrument(kraus), rho.to(DTYPE)


def pmd_qubit_instrument(period: int, gamma: float, v: float) -> tuple[Instrument, torch.Tensor]:
    """Wu et al. 2023 (Supplement S1, m = 1): a 1-qubit exact model of a PMD process.

    ``eta A_0`` has eigenvalues ``1, e^{i phi}`` (``phi = 2 pi / N``) with
    eigenvectors ``Z_0 = (1, 0)``, ``Z_1 = (alpha, 1)``; ``A_1 = zeta |s_0><v_1|``
    with ``v_1`` the second right-singular vector of ``A_0``. Choosing the
    largest singular value of ``A_0`` to be 1 fixes ``eta^2 = 1/Gamma`` and
    ``gamma' = 4 alpha^2 sin^2(phi/2)`` (S10); ``V = 4 alpha beta_0 beta_1`` and
    ``(beta_0 + alpha beta_1)^2 + beta_1^2 = 1`` (S12) fix ``|s_0>``. Returns
    the instrument and ``|s_0><s_0|`` (the memory just after a 1).
    """
    phi = 2 * math.pi / period
    x = 1 / gamma - 1  # eta^2 - 1
    gam = x * x / (1 + x)  # inverts S10
    alpha = math.sqrt(gam) / (2 * math.sin(phi / 2))

    def f(b: float) -> float:
        return (v / (4 * alpha * b) + alpha * b) ** 2 + b * b - 1

    grid = [k / 4000 for k in range(1, 4000)]
    vals = [f(b) for b in grid]
    i_min = min(range(len(grid)), key=vals.__getitem__)
    if vals[i_min] > 0:
        raise ValueError("no real (beta_0, beta_1) for these PMD parameters")
    lo, hi = 1e-12, grid[i_min]  # f(lo) > 0 >= f(hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    b1 = hi
    b0 = v / (4 * alpha * b1)
    eta = 1 / math.sqrt(gamma)
    e = complex(math.cos(phi), math.sin(phi))
    a0 = torch.tensor([[1, alpha * (e - 1)], [0, e]], dtype=DTYPE) / eta
    s0 = torch.tensor([b0 + b1 * alpha, b1], dtype=DTYPE)
    _, sv, vh = torch.linalg.svd(a0)
    if abs(float(sv[0]) - 1) > 1e-9:
        raise RuntimeError(f"largest singular value {float(sv[0])} != 1")
    zeta = math.sqrt(max(0.0, 1 - float(sv[1]) ** 2))
    a1 = zeta * torch.outer(s0, vh[1])  # |s_0><v_1|: the rows of vh are the bras <v_k|
    kraus = torch.stack([a0, a1])[:, None]
    return Instrument(kraus), torch.outer(s0, s0.conj())
