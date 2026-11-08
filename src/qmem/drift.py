"""Drift and nonstationary-mixture nulls for the Hankel witness (Phase 7 Stage 1, #19).

Builds on ``qmem.witness`` (whose functions it reuses and does not change) and is numpy only,
except for the exact differentiable statistics used by the adversarial search
(``exact_stats_t``), which use torch in ``float64``.

Contents:

* **Exact linear processes** for the drift nulls: ``mixture`` (a block-diagonal ``2K``-state
  process for ``K`` regimes chosen per shot), ``telegraph`` (a 2-state HMM whose parameters
  switch between regimes by a Markov fluctuator: a ``2R``-state HMM), and
  ``TimeVaryingProcess`` (operators that depend on the position in the shot, for drift within
  a shot). With a common start set every *stationary* 2-state constituent has ``w = 0``
  exactly; mixing does not preserve that, because the mixture's Hankel rank is up to ``2K``.
* **Samplers** that keep the time structure across shots: per-shot regime labels for mixtures,
  and a continuous-time fluctuator path over the whole run for the telegraph family.
* **Segment statistics**: per-segment witness, linearised (delta-method) SEs from contiguous
  batches, circular moving-block and segment-stratified bootstraps, effective sample size.
* **Stationarity tests**: chi-squared homogeneity of ``g(k)`` across blocks, per-block
  constancy of ``w(j)``, and an epidemic (two-change-point) scan with a permutation null.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from qmem import witness as W

# -- exact linear processes ------------------------------------------------------------------


def _block_diag(mats: Sequence[np.ndarray]) -> np.ndarray:
    dims = [m.shape[0] for m in mats]
    dtype = np.result_type(*mats)
    out = np.zeros((sum(dims), sum(dims)), dtype=dtype)
    o = 0
    for m, d in zip(mats, dims, strict=True):
        out[o : o + d, o : o + d] = m
        o += d
    return out


def mixture(procs: Sequence[W.LinearProcess], weights: Sequence[float]) -> W.LinearProcess:
    """Each shot comes from component ``c`` with probability ``weights[c]``: block-diagonal.

    ``P(y) = sum_c weights[c] * P_c(y)``. The result has ``sum_c D_c`` states, so a mixture of
    ``K`` 2-state HMMs is a ``2K``-state linear process (and an HMM when its parts are).
    """
    wts = np.asarray(weights, dtype=np.float64)
    if np.any(wts < 0) or not math.isclose(wts.sum(), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("mixture weights must be non-negative and sum to 1")
    init = np.concatenate([w * p.init for w, p in zip(wts, procs, strict=True)])
    ops = np.stack([_block_diag([p.ops[y] for p in procs]) for y in range(procs[0].ops.shape[0])])
    final = np.concatenate([p.final for p in procs])
    return W.LinearProcess(init, ops, final)


def telegraph(
    regimes: Sequence[tuple[np.ndarray, np.ndarray]],
    switch: np.ndarray,
    regime_init: np.ndarray,
) -> W.LinearProcess:
    """A ``D``-state HMM whose parameters follow a Markov regime ``r`` (a fluctuator).

    ``regimes[r] = (labelled_r, initial_r)`` with ``labelled_r[y, i, j] = P(y, j | i)``;
    ``switch[r, s]`` is the per-step regime transition matrix. Hidden state ``(r, i)``, big
    index ``r * D + i``: ``L[y, (r, i), (s, j)] = labelled_r[y, i, j] * switch[r, s]`` (emit
    with the current regime, then the regime moves). ``initial[(r, i)] = regime_init[r] *
    initial_r[i]``. Two regimes of a 2-state HMM give a 4-state HMM.
    """
    labs = [np.asarray(lab, dtype=np.float64) for lab, _ in regimes]
    n_r, d = len(labs), labs[0].shape[1]
    sw = np.asarray(switch, dtype=np.float64)
    ys = labs[0].shape[0]
    ops = np.zeros((ys, n_r * d, n_r * d))
    for r in range(n_r):
        for s in range(n_r):
            ops[:, r * d : (r + 1) * d, s * d : (s + 1) * d] = labs[r] * sw[r, s]
    init = np.concatenate(
        [regime_init[r] * np.asarray(regimes[r][1], dtype=np.float64) for r in range(n_r)]
    )
    return W.LinearProcess(init, ops, np.ones(n_r * d))


def switch_matrix(tau_s: float, occupancy: float, dt_s: float) -> np.ndarray:
    """Per-step transition matrix of a 2-level fluctuator with correlation time ``tau_s``.

    Rates ``k01 = occupancy / tau`` (into regime 1) and ``k10 = (1 - occupancy) / tau``; the
    regime autocorrelation decays as ``exp(-t / tau)``. Exact over a step of ``dt_s``.
    """
    q = float(occupancy)
    e = math.exp(-dt_s / tau_s)
    return np.array([[1 - q * (1 - e), q * (1 - e)], [(1 - q) * (1 - e), 1 - (1 - q) * (1 - e)]])


@dataclass
class TimeVaryingProcess:
    """``P(y_0..y_{T-1}) = init @ ops[0, y_0] @ ops[1, y_1] @ .. @ final``.

    The operators depend on the position ``t`` in the shot (drift within a shot); every shot
    is the same process. ``ops`` has shape ``(T, Y, D, D)``.
    """

    init: np.ndarray
    ops: np.ndarray
    final: np.ndarray

    @classmethod
    def constant(cls, proc: W.LinearProcess, length: int) -> TimeVaryingProcess:
        return cls(proc.init, np.broadcast_to(proc.ops, (length, *proc.ops.shape)), proc.final)

    @property
    def length(self) -> int:
        return self.ops.shape[0]

    def states(self) -> np.ndarray:
        """``(T, D)``: the (unnormalised) state before emitting at each position."""
        s, out = self.init, []
        for t in range(self.length):
            out.append(s)
            s = s @ self.ops[t].sum(0)
        return np.array(out)

    def marginals(self) -> np.ndarray:
        st = self.states()
        return np.array([(st[t] @ self.ops[t, 1] @ self.final).real for t in range(self.length)])

    def expected_windows(self, k_max: int, burn: int, common: bool = False) -> np.ndarray:
        """Exact expectation of ``W.per_shot_windows(samples, k_max, burn, common).mean(0)``."""
        st, length = self.states(), self.length
        out = np.zeros(k_max + 1)
        for k in range(k_max + 1):
            last = (length - k_max - 2) if common else (length - k - 2)
            vals = []
            for t in range(burn, last + 1):
                v = st[t] @ self.ops[t, 1]
                for u in range(t + 1, t + k + 1):
                    v = v @ self.ops[u, 0]
                vals.append((v @ self.ops[t + k + 1, 1] @ self.final).real)
            out[k] = np.mean(vals)
        return out

    def block_probs(self, starts: Sequence[int], size: int) -> np.ndarray:
        st = self.states()
        out = np.zeros(2**size)
        for t0 in starts:
            v = st[t0][None, :]
            for q in range(size):
                v = np.stack([v @ self.ops[t0 + q, 0], v @ self.ops[t0 + q, 1]], axis=1)
                v = v.reshape(-1, v.shape[-1])
            out += (v @ self.final).real
        return out / len(starts)

    def sample(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        s = np.tile(self.init, (n, 1))
        out = np.zeros((n, self.length), dtype=np.uint8)
        for t in range(self.length):
            nb = s @ self.ops[t, 1]
            p1 = (nb @ self.final).real / (s @ self.final).real
            y = rng.random(n) < p1
            out[:, t] = y
            s = np.where(y[:, None], nb, s @ self.ops[t, 0])
            s = s / (s @ self.final).real[:, None]
        return out


def drifting_two_state(
    start: tuple[Sequence[float], Sequence[float]],
    end: tuple[Sequence[float], Sequence[float]],
    init: np.ndarray,
    length: int,
) -> TimeVaryingProcess:
    """A 2-state HMM whose ``(p_stay, p_one)`` move linearly (in probability) from ``start``
    at ``t = 0`` to ``end`` at ``t = length - 1``."""
    a0, a1 = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    ops = []
    for t in range(length):
        f = t / max(length - 1, 1)
        ps, po = (1 - f) * a0 + f * a1
        ops.append(W.two_state_hmm(tuple(ps), tuple(po))[0])
    return TimeVaryingProcess(np.asarray(init, dtype=np.float64), np.array(ops), np.ones(2))


# -- samplers that keep the order of shots ------------------------------------------------------


def sample_mixture(
    comps: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    length: int,
    seed: int,
) -> np.ndarray:
    """Shot ``i`` from component ``labels[i]`` (a 2-state HMM ``(labelled, initial)``)."""
    out = np.zeros((len(labels), length), dtype=np.uint8)
    for c, (lab, ini) in enumerate(comps):
        m = labels == c
        if m.any():
            out[m] = W.hmm_sample(lab, ini, int(m.sum()), length, seed + 7919 * c)
    return out


def fluctuator_path(
    n_shots: int,
    length: int,
    tau_s: float,
    occupancy: float,
    dt_s: float,
    shot_period_s: float,
    seed: int,
) -> np.ndarray:
    """``(n_shots, length)`` regime (0/1) at each measurement in real time.

    Step ``t`` of shot ``i`` happens at ``i * shot_period_s + t * dt_s``; the regime is a
    continuous-time 2-level Markov process (correlation time ``tau_s``, stationary
    ``P(regime 1) = occupancy``) sampled at those times, so it persists across shots when
    ``tau_s`` exceeds the shot period.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(n_shots)[:, None] * shot_period_s + np.arange(length)[None, :] * dt_s
    total = float(times.max()) + dt_s
    k = {0: occupancy / tau_s, 1: (1 - occupancy) / tau_s}  # leave rate of each regime
    r = int(rng.random() < occupancy)
    edges, regs, t = [0.0], [r], 0.0
    # draw holding times in batches (fast also for tau ~ 10 us over ~11 s)
    while t < total:
        rate = k[r]
        t += rng.exponential(1.0 / rate) if rate > 0 else math.inf
        r = 1 - r
        edges.append(t)
        regs.append(r)
    edges_a, regs_a = np.array(edges), np.array(regs)
    pos = np.searchsorted(edges_a, times.ravel(), side="right") - 1
    return regs_a[pos].reshape(n_shots, length).astype(np.uint8)


def sample_switching(
    regimes: Sequence[tuple[np.ndarray, np.ndarray]],
    path: np.ndarray,
    seed: int,
    start_regime_init: bool = True,
) -> np.ndarray:
    """Hidden-state simulation of a 2-state HMM whose regime at ``(shot, t)`` is ``path``.

    The hidden state starts each shot from the initial state of the shot's first regime.
    """
    labs = np.stack([np.asarray(lab, dtype=np.float64) for lab, _ in regimes])  # (R, Y, D, D)
    n_r, ys, d, _ = labs.shape
    joint = labs.transpose(0, 2, 1, 3).reshape(n_r, d, ys * d)  # (r, i, (y, j))
    cum = np.cumsum(joint, axis=2)
    inits = np.stack([np.asarray(ini, dtype=np.float64) for _, ini in regimes])
    rng = np.random.default_rng(seed)
    n, length = path.shape
    p0 = inits[path[:, 0]]
    st = (rng.random(n)[:, None] * p0.sum(1, keepdims=True) >= np.cumsum(p0, 1)).sum(1)
    st = np.minimum(st, d - 1)
    out = np.zeros((n, length), dtype=np.uint8)
    for t in range(length):
        c_row = cum[path[:, t], st]  # (n, Y*D)
        u = rng.random(n) * c_row[:, -1]
        c = np.minimum((u[:, None] >= c_row).sum(1), ys * d - 1)
        out[:, t] = c // d
        st = c % d
    return out


# -- exact differentiable statistics (torch, float64) -----------------------------------------


def two_state_ops_t(p_stay: torch.Tensor, p_one: torch.Tensor) -> torch.Tensor:
    """``(..., 2, 2, 2)`` labelled ops ``L[y, i, j] = P(y|i) T[i, j]`` from ``(..., 2)``."""
    a, b = p_stay[..., 0], p_stay[..., 1]
    t = torch.stack([torch.stack([a, 1 - a], -1), torch.stack([1 - b, b], -1)], -2)  # (..., i, j)
    e1 = p_one  # (..., i)
    e = torch.stack([1 - e1, e1], -2)  # (..., y, i)
    return e[..., :, :, None] * t[..., None, :, :]


def exact_stats_t(
    init: torch.Tensor,
    ops: torch.Tensor,
    k_max: int,
    burn: int,
    block: int = 4,
) -> dict[str, torch.Tensor]:
    """Exact ``g(0..k_max)`` (common start set), marginals and aligned ``block``-statistics.

    ``init`` is ``(B, D)``; ``ops`` is ``(B, T, 2, D, D)`` (position-dependent; broadcast a
    constant one with ``expand``); the final vector is all ones (an HMM). Blocks start at
    ``burn, burn + block, ..`` as in ``qmem.witness.block_kl``.
    """
    length = ops.shape[1]
    chan = ops.sum(2)  # (B, T, D, D)
    s, sts = init, []
    for t in range(length):
        sts.append(s)
        s = torch.einsum("bd,bde->be", s, chan[:, t])
    st = torch.stack(sts, 1)  # (B, T, D)
    marg = torch.einsum("btd,btde->bt", st, ops[:, :, 1])
    ts = torch.arange(burn, length - k_max - 1)
    v = torch.einsum("bnd,bnde->bne", st[:, ts], ops[:, ts, 1])
    gs = []
    for k in range(k_max + 1):
        gs.append(torch.einsum("bnd,bnde->bn", v, ops[:, ts + k + 1, 1]).mean(1))
        v = torch.einsum("bnd,bnde->bne", v, ops[:, ts + k + 1, 0])
    starts = torch.arange(burn, length - block + 1, block)
    v = st[:, starts][:, :, None, :]  # (B, ns, patterns, D)
    for q in range(block):
        o = ops[:, starts + q]  # (B, ns, 2, D, D)
        v0 = torch.einsum("bnpd,bnde->bnpe", v, o[:, :, 0])
        v1 = torch.einsum("bnpd,bnde->bnpe", v, o[:, :, 1])
        v = torch.stack([v0, v1], 3).reshape(ops.shape[0], len(starts), -1, v.shape[-1])
    blocks = v.sum(-1).mean(1)
    return {"g": torch.stack(gs, 1), "marg": marg, "block": blocks}


def witness_t(g: torch.Tensor, j: int, size: int = 3) -> torch.Tensor:
    idx = j + torch.arange(size)[:, None] + torch.arange(size)[None, :]
    h = g[..., idx]
    return torch.linalg.det(h) / h.abs().amax((-1, -2)) ** size


# -- segment statistics and uncertainty ---------------------------------------------------------


def influence(per: np.ndarray, j: int) -> np.ndarray:
    """Per-shot linearised contribution ``u_i = grad w(j) . per_i`` (the delta method)."""
    return per @ W.witness_grad(per.mean(0), j)


def batch_means_se(x: np.ndarray, b: int) -> float:
    """SE of ``mean(x)`` from contiguous non-overlapping batches of about ``b`` items."""
    n = len(x)
    m = max(n // b, 2)
    parts = np.array_split(np.asarray(x, dtype=np.float64), m)
    sizes = np.array([len(p) for p in parts], dtype=np.float64)
    means = np.array([p.mean() for p in parts])
    mu = float(np.asarray(x).mean())
    return float(math.sqrt(np.sum(sizes**2 * (means - mu) ** 2) / n**2 * m / (m - 1)))


def block_se(per: np.ndarray, j: int, b: int) -> float:
    """Delta-method SE of ``w(j)`` with contiguous batches of ``b`` shots (``b = 1``: i.i.d.)."""
    u = influence(per, j)
    if b <= 1:
        return float(u.std(ddof=1) / math.sqrt(len(u)))
    return batch_means_se(u, b)


def effective_sample_size(per: np.ndarray, j: int, b: int) -> float:
    """``n * (SE_iid / SE_block)^2``: shots worth of i.i.d. information at batch size ``b``."""
    return len(per) * (block_se(per, j, 1) / block_se(per, j, b)) ** 2


def block_bootstrap_indices(n: int, b: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular moving-block bootstrap: ``(n_boot, n)`` indices in blocks of ``b`` shots.

    With ``b = 1`` the draws are identical to ``qmem.witness.bootstrap_indices``.
    """
    rng = np.random.default_rng(seed)
    nb = -(-n // b)
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    return idx.reshape(n_boot, nb * b)[:, :n]


def stratified_block_indices(
    segments: Sequence[np.ndarray], b: int, n_boot: int, seed: int
) -> np.ndarray:
    """Block bootstrap within each segment (index arrays), concatenated: segment sizes fixed."""
    out = []
    for s, seg in enumerate(segments):
        seg = np.asarray(seg)
        loc = block_bootstrap_indices(len(seg), min(b, len(seg)), n_boot, seed + 104729 * s)
        out.append(seg[loc])
    return np.concatenate(out, axis=1)


def bootstrap_se(per: np.ndarray, j: int, idx: np.ndarray) -> float:
    return float(np.std(W.witness(W.bootstrap_g(per, idx), j), ddof=1))


def contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the maximal runs of ``True`` in ``mask`` (in order)."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    cuts = np.nonzero(np.diff(idx) > 1)[0] + 1
    return np.split(idx, cuts)


def blocks_within(mask: np.ndarray, b: int) -> list[np.ndarray]:
    """Contiguous blocks of ``b`` shots inside each run of ``mask``; a run's remainder joins its
    last block (a run shorter than ``b`` is one block). No block straddles a gap."""
    out = []
    for run in contiguous_runs(mask):
        m = max(len(run) // b, 1)
        out += [run[i * b : (i + 1) * b] for i in range(m - 1)] + [run[(m - 1) * b :]]
    return out


@dataclass
class SegmentStats:
    name: str
    n: int
    p1: float
    g: np.ndarray
    w: dict[int, float]
    se_shot: dict[int, float]  # i.i.d. shot bootstrap
    se_block: dict[int, float]  # delta method, contiguous batches of `b` shots
    se_block_boot: dict[int, float]  # circular moving-block bootstrap
    ess: dict[int, float]


def segment_stats(
    name: str,
    per: np.ndarray,
    shots: np.ndarray,
    offsets: Sequence[int],
    b: int,
    n_boot: int,
    seed: int,
) -> SegmentStats:
    """Witness of one segment with shot-bootstrap and contiguous-block uncertainties."""
    n = len(per)
    g = per.mean(0)
    bb = min(b, max(n // 4, 1))
    idx_s = W.bootstrap_indices(n, n_boot, seed)
    idx_b = block_bootstrap_indices(n, bb, n_boot, seed + 1)
    w, s1, s2, s3, ess = {}, {}, {}, {}, {}
    for j in offsets:
        w[j] = W.witness(g, j)
        s1[j] = bootstrap_se(per, j, idx_s)
        s2[j] = block_se(per, j, bb)
        s3[j] = bootstrap_se(per, j, idx_b)
        ess[j] = n * (block_se(per, j, 1) / s2[j]) ** 2
    return SegmentStats(name, n, float(shots.mean()), g, w, s1, s2, s3, ess)


# -- stationarity tests --------------------------------------------------------------------------


def chi2_sf(x: float, dof: float) -> float:
    """Survival function of chi-squared (regularised upper incomplete gamma, via torch)."""
    if x <= 0:
        return 1.0
    xt = torch.tensor(x / 2.0, dtype=torch.float64)
    return float(torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64), xt))


def homogeneity_chi2(per: np.ndarray, blocks: Sequence[np.ndarray]) -> dict[str, float]:
    """Chi-squared test that every block has the same mean of each column of ``per``.

    Per column ``k``: ``X_k = sum_b n_b (mean_b - mean)^2 / s_k^2`` with the pooled
    within-block variance, ``~ chi2(B - 1)``. Joint: ``sum_b n_b d_b^T S^-1 d_b`` with the
    pooled within-block covariance, ``~ chi2((B - 1) K)``.
    """
    k = per.shape[1]
    mu = per.mean(0)
    within = np.zeros((k, k))
    dfw = 0
    stat_joint_terms = []
    means, sizes = [], []
    for blk in blocks:
        x = per[blk]
        means.append(x.mean(0))
        sizes.append(len(x))
        d = x - x.mean(0)
        within += d.T @ d
        dfw += len(x) - 1
    within /= dfw
    means_a, sizes_a = np.array(means), np.array(sizes, dtype=np.float64)
    dev = means_a - mu
    out: dict[str, float] = {"blocks": float(len(blocks))}
    for c in range(k):
        x2 = float(np.sum(sizes_a * dev[:, c] ** 2) / within[c, c])
        out[f"chi2_k{c}"] = x2
        out[f"p_k{c}"] = chi2_sf(x2, len(blocks) - 1)
    inv = np.linalg.pinv(within)
    stat_joint_terms = sizes_a * np.einsum("bi,ij,bj->b", dev, inv, dev)
    xj = float(stat_joint_terms.sum())
    rank = np.linalg.matrix_rank(within)
    out["chi2_joint"] = xj
    out["dof_joint"] = float((len(blocks) - 1) * rank)
    out["p_joint"] = chi2_sf(xj, (len(blocks) - 1) * rank)
    return out


def block_witness_constancy(
    per: np.ndarray, blocks: Sequence[np.ndarray], j: int
) -> dict[str, float]:
    """Is ``w(j)`` the same in every block? ``sum (w_b - w_bar)^2 / se_b^2 ~ chi2(B - 1)``
    with i.i.d. delta-method SEs inside each block and the inverse-variance mean ``w_bar``."""
    ws = np.array([W.witness(per[b].mean(0), j) for b in blocks])
    ses = np.array([block_se(per[b], j, 1) for b in blocks])
    wt = 1 / ses**2
    wbar = float(np.sum(wt * ws) / wt.sum())
    x2 = float(np.sum((ws - wbar) ** 2 * wt))
    return {
        "blocks": float(len(blocks)),
        "w_bar": wbar,
        "chi2": x2,
        "p": chi2_sf(x2, len(blocks) - 1),
        "w_min": float(ws.min()),
        "w_max": float(ws.max()),
    }


def epidemic_gain(x: np.ndarray, grid: int) -> tuple[float, int, int]:
    """Largest two-change-point gain of a segment ``[a, b)`` (on a grid of ``grid`` items)
    against a constant mean, in units of the sample variance (a likelihood-ratio statistic).

    Segments touching either end are included, so this also covers a single change.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    cut = np.unique(np.append(np.arange(0, n, grid), n))
    cs = np.concatenate([[0.0], np.cumsum(x - x.mean())])[cut]
    var = float(x.var())
    inside = cs[None, :] - cs[:, None]
    m = (cut[None, :] - cut[:, None]).astype(np.float64)
    ok = (m > 0) & (m < n)
    gain = np.where(ok, inside**2 / np.where(ok, m, 1) + inside**2 / np.where(ok, n - m, 1), -1)
    i, k = np.unravel_index(np.argmax(gain), gain.shape)
    return float(gain[i, k] / var), int(cut[i]), int(cut[k])


def changepoint_test(x: np.ndarray, grid: int, n_perm: int, seed: int) -> dict[str, float]:
    """Epidemic change-point scan with a permutation null (shots exchangeable under H0).

    ``p = (1 + #{T_perm >= T_obs}) / (1 + n_perm)``: exact size under exchangeability, and it
    accounts for the boundaries being chosen from the same data.
    """
    t_obs, a, b = epidemic_gain(x, grid)
    rng = np.random.default_rng(seed)
    null = np.array([epidemic_gain(rng.permutation(x), grid)[0] for _ in range(n_perm)])
    return {
        "stat": t_obs,
        "start": a,
        "stop": b,
        "p": float((1 + np.sum(null >= t_obs)) / (1 + n_perm)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
        "null_max": float(null.max()),
    }


def boundary_profile(x: np.ndarray, a: int, b: int, which: str, step: int) -> np.ndarray:
    """``(positions, LR)`` of the epidemic fit with one boundary moved, the other held."""
    x = np.asarray(x, dtype=np.float64)
    n, var = len(x), float(x.var())
    cs = np.concatenate([[0.0], np.cumsum(x - x.mean())])
    lo, hi = (0, b - step) if which == "start" else (a + step, n)
    pos = np.arange(lo, hi + 1, step)
    ins = cs[b] - cs[pos] if which == "start" else cs[pos] - cs[a]
    m = (b - pos) if which == "start" else (pos - a)
    m = m.astype(np.float64)
    ok = (m > 0) & (m < n)
    lr = np.where(ok, ins**2 / np.where(ok, m, 1) + ins**2 / np.where(ok, n - m, 1), 0) / var
    return np.stack([pos, lr], 1)


def ramp_fit(x: np.ndarray, centre: int, width: int, half: int) -> float:
    """Residual sum of squares of a linear ramp of ``width`` shots centred at ``centre``,
    fitted (least squares, two levels) to ``x`` within ``centre +- half``."""
    lo, hi = max(centre - half, 0), min(centre + half, len(x))
    t = np.arange(lo, hi, dtype=np.float64)
    if width <= 0:
        f = (t >= centre).astype(np.float64)
    else:
        f = np.clip((t - centre) / width + 0.5, 0.0, 1.0)
    a = np.stack([np.ones_like(f), f], 1)
    y = np.asarray(x[lo:hi], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(np.sum((y - a @ coef) ** 2))


# -- adversarial search over a null family -----------------------------------------------------


@dataclass
class Target:
    """Data a null must reproduce: means and SEs of ``g``, per-position marginals after the
    burn-in, and the aligned ``block``-statistics, with thresholds."""

    g: np.ndarray
    g_se: np.ndarray
    marg: np.ndarray  # positions burn..L-1
    marg_se: np.ndarray
    block: np.ndarray
    block_cov_inv: np.ndarray  # pseudo-inverse of the covariance of the block means
    z_max: float
    marg_chi2_max: float
    block_chi2_max: float


def violations_t(stats: dict[str, torch.Tensor], tgt: Target, burn: int) -> dict[str, torch.Tensor]:
    g = stats["g"]
    zg = (g - torch.as_tensor(tgt.g)) / torch.as_tensor(tgt.g_se)
    zm = (stats["marg"][:, burn:] - torch.as_tensor(tgt.marg)) / torch.as_tensor(tgt.marg_se)
    db = stats["block"] - torch.as_tensor(tgt.block)
    cb = torch.einsum("bi,ij,bj->b", db, torch.as_tensor(tgt.block_cov_inv), db)
    return {
        "max_abs_z_g": zg.abs().amax(1),
        "z_g": zg,
        "chi2_marg": (zm**2).sum(1),
        "chi2_block": cb,
    }


def penalty_t(v: dict[str, torch.Tensor], tgt: Target) -> torch.Tensor:
    hinge = torch.relu(v["z_g"].abs() - (tgt.z_max - 0.25)) ** 2
    pm = torch.relu(v["chi2_marg"] - 0.9 * tgt.marg_chi2_max) / tgt.marg_chi2_max
    pb = torch.relu(v["chi2_block"] - 0.9 * tgt.block_chi2_max) / tgt.block_chi2_max
    return hinge.sum(1) + 10 * pm + 10 * pb


def adversarial_search(
    family: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    n_params: int,
    tgt: Target,
    burn: int,
    w_scale: float,
    restarts: int,
    steps: tuple[int, int],
    seed: int,
    lr: float = 0.05,
) -> dict[str, np.ndarray]:
    """Most negative exact ``w(0)`` over a family, subject to reproducing ``tgt``.

    Two phases from ``restarts`` random starts in parameter space: (1) fit the data (minimise
    the constraint penalty), then (2) minimise ``w(0) / w_scale + mu * penalty`` with ``mu``
    rising from 1 to 1e3. Returns, per restart, the final parameters, exact ``g``, ``w(0..3)``
    and constraint values; and the phase-1 best-fit values.
    """
    gen = torch.Generator().manual_seed(seed)
    theta = (
        2.0 * torch.randn(restarts, n_params, generator=gen, dtype=torch.float64)
    ).requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps[0]):
        opt.zero_grad()
        loss = penalty_t(violations_t(family(theta), tgt, burn), tgt).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        st = family(theta)
        fit = {k: v.numpy().copy() for k, v in violations_t(st, tgt, burn).items()}
        fit["w0"] = witness_t(st["g"], 0).numpy().copy()
        fit["penalty"] = penalty_t(violations_t(st, tgt, burn), tgt).numpy().copy()
    opt = torch.optim.Adam([theta], lr=lr / 2)
    n2 = steps[1]
    for s in range(n2):
        mu = 10 ** (3 * s / max(n2 - 1, 1))
        opt.zero_grad()
        st = family(theta)
        loss = (
            witness_t(st["g"], 0) / w_scale + mu * penalty_t(violations_t(st, tgt, burn), tgt)
        ).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        st = family(theta)
        v = violations_t(st, tgt, burn)
        out = {k: x.numpy().copy() for k, x in v.items()}
        out["theta"] = theta.detach().numpy().copy()
        out["g"] = st["g"].numpy().copy()
        for j in range(4):
            out[f"w{j}"] = witness_t(st["g"], j).numpy().copy()
        out["feasible"] = (
            (v["max_abs_z_g"] <= tgt.z_max)
            & (v["chi2_marg"] <= tgt.marg_chi2_max)
            & (v["chi2_block"] <= tgt.block_chi2_max)
        ).numpy()
    out["fit"] = fit  # type: ignore[assignment]
    return out


# -- null families: parameter vector -> exact statistics (torch) and -> processes (numpy) --------


def _comp_t(th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(..., 5)`` logits ``(p_stay0, p_stay1, p_one0, p_one1, P(state 0 at t=0))``."""
    p = torch.sigmoid(th)
    lab = two_state_ops_t(p[..., 0:2], p[..., 2:4])
    init = torch.stack([p[..., 4], 1 - p[..., 4]], -1)
    return lab, init


def _comp_np(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = 1 / (1 + np.exp(-np.asarray(th, dtype=np.float64)))
    lab, _ = W.two_state_hmm((p[0], p[1]), (p[2], p[3]))
    return lab, np.array([p[4], 1 - p[4]])


@dataclass
class Family:
    """A null family: ``stats(theta)`` gives exact statistics for a batch of parameters,
    ``process(theta_row)`` the numpy linear process (``W.LinearProcess`` or
    ``TimeVaryingProcess``), and ``sampler(theta_row, n, seed)`` shots in run order."""

    name: str
    n_params: int
    stats: Callable[[torch.Tensor], dict[str, torch.Tensor]]
    process: Callable[[np.ndarray], object]
    sampler: Callable[[np.ndarray, int, int], np.ndarray]


def mixture_family(k: int, length: int, k_max: int, burn: int, block: int = 4) -> Family:
    """``k`` 2-state HMMs (each with its own initial state), one drawn per shot at random."""

    def stats(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        b = theta.shape[0]
        comps = theta[:, : 5 * k].reshape(b * k, 5)
        wts = torch.softmax(theta[:, 5 * k :], -1)  # (b, k)
        lab, init = _comp_t(comps)
        ops = lab[:, None].expand(b * k, length, 2, 2, 2)
        st = exact_stats_t(init, ops, k_max, burn, block)
        return {
            key: torch.einsum("bk,bk...->b...", wts, v.reshape(b, k, *v.shape[1:]))
            for key, v in st.items()
        }

    def parts(th: np.ndarray):
        z = np.asarray(th[5 * k :], dtype=np.float64)
        wts = np.exp(z - z.max())
        return [_comp_np(th[5 * c : 5 * c + 5]) for c in range(k)], wts / wts.sum()

    def process(th: np.ndarray) -> W.LinearProcess:
        comps, wts = parts(th)
        return mixture([W.LinearProcess.hmm(lab, ini) for lab, ini in comps], wts)

    def sampler(th: np.ndarray, n: int, seed: int) -> np.ndarray:
        comps, wts = parts(th)
        labels = np.random.default_rng(seed).choice(k, size=n, p=wts)
        return sample_mixture(comps, labels, length, seed + 1)

    return Family(f"mixture_K{k}", 6 * k, stats, process, sampler)


def continuous_family(
    n_grid: int, length: int, k_max: int, burn: int, block: int = 4, contiguous: bool = True
) -> Family:
    """2-state HMMs on a line between two endpoints (in probability), uniform over the line:
    a continuous drift of the parameters across the run (``n_grid`` points). With
    ``contiguous`` the run moves along the line in order; otherwise shots are shuffled."""
    f = np.linspace(0.0, 1.0, n_grid)

    def stats(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        b = theta.shape[0]
        p0, p1 = torch.sigmoid(theta[:, :5]), torch.sigmoid(theta[:, 5:10])
        ft = torch.as_tensor(f)[None, :, None]
        p = (1 - ft) * p0[:, None] + ft * p1[:, None]  # (b, n_grid, 5)
        p = p.reshape(b * n_grid, 5)
        lab = two_state_ops_t(p[:, 0:2], p[:, 2:4])
        init = torch.stack([p[:, 4], 1 - p[:, 4]], -1)
        st = exact_stats_t(init, lab[:, None].expand(-1, length, 2, 2, 2), k_max, burn, block)
        return {key: v.reshape(b, n_grid, *v.shape[1:]).mean(1) for key, v in st.items()}

    def comps(th: np.ndarray):
        p0 = 1 / (1 + np.exp(-np.asarray(th[:5], dtype=np.float64)))
        p1 = 1 / (1 + np.exp(-np.asarray(th[5:10], dtype=np.float64)))
        out = []
        for x in f:
            p = (1 - x) * p0 + x * p1
            lab, _ = W.two_state_hmm((p[0], p[1]), (p[2], p[3]))
            out.append((lab, np.array([p[4], 1 - p[4]])))
        return out

    def process(th: np.ndarray) -> W.LinearProcess:
        return mixture(
            [W.LinearProcess.hmm(a, b) for a, b in comps(th)], np.full(n_grid, 1 / n_grid)
        )

    def sampler(th: np.ndarray, n: int, seed: int) -> np.ndarray:
        labels = (np.arange(n) * n_grid) // n
        if not contiguous:
            labels = np.random.default_rng(seed).permutation(labels)
        return sample_mixture(comps(th), labels, length, seed + 1)

    name = "continuous_" + ("contiguous" if contiguous else "shuffled")
    return Family(name, 10, stats, process, sampler)


def telegraph_family(
    tau_s: float,
    dt_s: float,
    shot_period_s: float,
    length: int,
    k_max: int,
    burn: int,
    block: int = 4,
) -> Family:
    """Two 2-state regimes switched by a fluctuator of correlation time ``tau_s``.

    Parameters: two regimes (5 logits each, incl. their initial state) and the occupancy.
    Each shot starts its hidden state afresh, from the current regime's initial state; the
    regime runs on in real time, so it persists across shots when ``tau_s`` is long."""

    def stats(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        b = theta.shape[0]
        lab_a, ini_a = _comp_t(theta[:, 0:5])
        lab_b, ini_b = _comp_t(theta[:, 5:10])
        q = torch.sigmoid(theta[:, 10])
        e = math.exp(-dt_s / tau_s)
        s01, s10 = q * (1 - e), (1 - q) * (1 - e)
        sw = torch.stack(
            [torch.stack([1 - s01, s01], -1), torch.stack([s10, 1 - s10], -1)], -2
        )  # (b, r, s)
        labs = torch.stack([lab_a, lab_b], 1)  # (b, r, y, i, j)
        ops = torch.einsum("bryij,brs->byrisj", labs, sw).reshape(b, 2, 4, 4)
        init = torch.cat([(1 - q)[:, None] * ini_a, q[:, None] * ini_b], -1)
        return exact_stats_t(init, ops[:, None].expand(b, length, 2, 4, 4), k_max, burn, block)

    def regimes(th: np.ndarray):
        q = 1 / (1 + math.exp(-float(th[10])))
        return [_comp_np(th[0:5]), _comp_np(th[5:10])], q

    def process(th: np.ndarray) -> W.LinearProcess:
        reg, q = regimes(th)
        return telegraph(reg, switch_matrix(tau_s, q, dt_s), np.array([1 - q, q]))

    def sampler(th: np.ndarray, n: int, seed: int) -> np.ndarray:
        reg, q = regimes(th)
        path = fluctuator_path(n, length, tau_s, q, dt_s, shot_period_s, seed)
        return sample_switching(reg, path, seed + 1)

    return Family(f"telegraph_tau{tau_s:.0e}", 11, stats, process, sampler)


def within_shot_drift_family(length: int, k_max: int, burn: int, block: int = 4) -> Family:
    """A 2-state HMM whose parameters move linearly (in probability) over each shot, the same
    in every shot. Parameters: start ``(p_stay, p_one)``, end, and the initial state."""
    f = np.arange(length) / (length - 1)

    def stats(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        p0, p1 = torch.sigmoid(theta[:, 0:4]), torch.sigmoid(theta[:, 4:8])
        ft = torch.as_tensor(f)[None, :, None]
        p = (1 - ft) * p0[:, None] + ft * p1[:, None]  # (b, T, 4)
        lab = two_state_ops_t(p[..., 0:2], p[..., 2:4])  # (b, T, 2, 2, 2)
        q = torch.sigmoid(theta[:, 8])
        return exact_stats_t(torch.stack([q, 1 - q], -1), lab, k_max, burn, block)

    def process(th: np.ndarray) -> TimeVaryingProcess:
        p = 1 / (1 + np.exp(-np.asarray(th, dtype=np.float64)))
        return drifting_two_state(
            (p[0:2], p[2:4]), (p[4:6], p[6:8]), np.array([p[8], 1 - p[8]]), length
        )

    def sampler(th: np.ndarray, n: int, seed: int) -> np.ndarray:
        return process(th).sample(n, seed)

    return Family("within_shot_drift", 9, stats, process, sampler)


def hmm_family(d: int, length: int, k_max: int, burn: int, block: int = 4) -> Family:
    """A general stationary ``d``-state HMM (positive control: not a drift null).

    ``L[y, i, j] = softmax_{(y, j)}(theta_i)``; the initial state is a softmax too."""
    n_p = d * 2 * d + d

    def ops_t(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = theta.shape[0]
        rows = torch.softmax(theta[:, : 2 * d * d].reshape(b, d, 2 * d), -1)
        lab = rows.reshape(b, d, 2, d).permute(0, 2, 1, 3)  # (b, y, i, j)
        return lab, torch.softmax(theta[:, 2 * d * d :], -1)

    def stats(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        lab, init = ops_t(theta)
        ops = lab[:, None].expand(theta.shape[0], length, 2, d, d)
        return exact_stats_t(init, ops, k_max, burn, block)

    def process(th: np.ndarray) -> W.LinearProcess:
        lab, init = ops_t(torch.as_tensor(np.asarray(th, dtype=np.float64))[None])
        return W.LinearProcess.hmm(lab[0].numpy(), init[0].numpy())

    def sampler(th: np.ndarray, n: int, seed: int) -> np.ndarray:
        p = process(th)
        return W.hmm_sample(p.ops, p.init, n, length, seed)

    return Family(f"hmm_{d}state", n_p, stats, process, sampler)
