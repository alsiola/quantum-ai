"""Independent estimator and statistics of the Hankel dimension witness (Phase 7 audit, #18).

Numpy only, and written without the Phase 5a/6 helpers (``experiments/phase5a.py``
``empirical_windows``/``hankel_witness``, ``experiments/phase6.py`` ``witness_stats``) so that
it can audit them.

Definitions (as Phase 5a/6 state them):

* ``g(k) = P(1 0^k 1)``, estimated per shot as the fraction of start positions
  ``t in [burn, L - k - 2]`` with ``y_t = 1``, ``y_{t+1..t+k} = 0``, ``y_{t+k+1} = 1``, then
  averaged over shots. Each ``k`` has its own number of start positions, ``L - burn - k - 1``.
* ``w(j) = det H_j / max|H_j|^3`` with ``H_j[a, b] = g(j + a + b)``, ``a, b < 3``.

A process is represented linearly (an observable operator model): a row vector ``init``, one
matrix ``ops[y]`` per symbol and a column vector ``final`` with
``P(y_0 .. y_{T-1}) = init @ ops[y_0] @ .. @ ops[y_{T-1}] @ final``. An edge-emitting HMM
``L[y, i, j] = P(y, j | i)`` is ``(pi, L[y], 1)``; a qubit instrument is its superoperators on
``vec(rho)`` with ``final = vec(I)``. For a ``D``-dimensional representation every Hankel matrix
of ``g`` has rank ``<= D``, so ``w = 0`` for every 2-state HMM evaluated at a *fixed* state.

The exact expectation of the estimator (``expected_windows``) averages the state over each
``k``'s own start positions; from a non-stationary state this is *not* a fixed state for all
``k``, so it need not give ``w = 0`` for a 2-state HMM (the audit quantifies how much).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# -- decoding ---------------------------------------------------------------------------------


def unpack_shots(packed: np.ndarray, length: int) -> np.ndarray:
    """``(shots, length)`` uint8 tokens from ``np.packbits(arr, axis=1)`` (big bit order).

    Column ``t`` is bit ``7 - t % 8`` of byte ``t // 8`` (``packbits``' default ``"big"``
    order puts column 0 in the most significant bit of byte 0).
    """
    packed = np.asarray(packed, dtype=np.uint8)
    t = np.arange(length)
    return ((packed[:, t // 8] >> (7 - t % 8)) & 1).astype(np.uint8)


def load_shots(path: str) -> np.ndarray:
    """Shots of a Phase 6 ``shots_<stage>.npz``."""
    with np.load(path) as z:
        return unpack_shots(z["packed"], int(z["length"]))


# -- the estimator --------------------------------------------------------------------------


def n_starts(length: int, k_max: int, burn: int) -> np.ndarray:
    """Start positions per ``k``: ``t = burn .. length - k - 2``."""
    return np.array([length - burn - k - 1 for k in range(k_max + 1)], dtype=np.int64)


def window_counts(
    shots: np.ndarray, k_max: int, burn: int, last_start: int | None = None
) -> np.ndarray:
    """``(shots, k_max + 1)`` counts of ``1 0^k 1`` starting at ``t >= burn``.

    Counted from gaps between consecutive ones (each one at ``t >= burn`` is followed by its
    next one at ``t + k + 1``), a different algorithm from Phase 5a's window scan. With
    ``last_start``, only starts ``t <= last_start`` count (a common start set for every k).
    """
    y = np.asarray(shots).astype(bool)
    n, length = y.shape
    # next_one[:, t] = position of the first one strictly after t (length if none)
    next_one = np.full((n, length), length, dtype=np.int64)
    nxt = np.full(n, length, dtype=np.int64)
    for t in range(length - 1, -1, -1):
        next_one[:, t] = nxt
        nxt = np.where(y[:, t], t, nxt)
    out = np.zeros((n, k_max + 1), dtype=np.int64)
    hi = length - 1 if last_start is None else last_start
    for t in range(burn, hi + 1):
        gap = next_one[:, t] - t - 1
        ok = y[:, t] & (next_one[:, t] < length) & (gap <= k_max)
        np.add.at(out, (np.nonzero(ok)[0], gap[ok]), 1)
    return out


def per_shot_windows(shots: np.ndarray, k_max: int, burn: int, common: bool = False) -> np.ndarray:
    """Per-shot frequencies of ``1 0^k 1`` (``counts / starts``), shape ``(shots, k_max+1)``.

    ``common=True`` uses the same starts ``burn .. L - k_max - 2`` for every ``k``.
    """
    length = np.asarray(shots).shape[1]
    if common:
        last = length - k_max - 2
        c = window_counts(shots, k_max, burn, last_start=last)
        return c / float(last - burn + 1)
    return window_counts(shots, k_max, burn) / n_starts(length, k_max, burn)


def hankel(g: np.ndarray, j: int, size: int = 3) -> np.ndarray:
    """``H[..., a, b] = g[..., j + a + b]`` (batched over leading axes)."""
    g = np.asarray(g, dtype=np.float64)
    idx = j + np.arange(size)[:, None] + np.arange(size)[None, :]
    return g[..., idx]


def witness(g: np.ndarray, j: int, size: int = 3) -> np.ndarray | float:
    """``det H_j / max|H_j|^size`` (batched over leading axes of ``g``)."""
    h = hankel(g, j, size)
    m = np.abs(h).max(axis=(-1, -2))
    out = np.linalg.det(h) / m**size
    return float(out) if np.ndim(out) == 0 else out


def argmax_entry(g: np.ndarray, j: int, size: int = 3) -> np.ndarray:
    """Index ``j + a + b`` of the max |entry| of ``H_j`` (to see if it switches)."""
    g = np.asarray(g, dtype=np.float64)
    seg = np.abs(g[..., j : j + 2 * size - 1])
    return j + seg.argmax(axis=-1)


def witness_grad(g: np.ndarray, j: int, size: int = 3) -> np.ndarray:
    """``dw(j)/dg`` at ``g`` (the max entry held at its argmax), analytic."""
    g = np.asarray(g, dtype=np.float64)
    h = hankel(g, j, size)
    det = np.linalg.det(h)
    cof = det * np.linalg.inv(h).T  # d det / d H[a, b]
    km = int(argmax_entry(g, j, size))
    m = abs(g[km])
    grad = np.zeros_like(g)
    for a in range(size):
        for b in range(size):
            grad[j + a + b] += cof[a, b] / m**size
    grad[km] += -size * det / m ** (size + 1) * np.sign(g[km])
    return grad


# -- uncertainty -------------------------------------------------------------------------------


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n)`` shot indices drawn with replacement (numpy ``Generator``)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_g(per: np.ndarray, idx: np.ndarray, chunk: int = 50) -> np.ndarray:
    """``g`` of each bootstrap resample of *whole shots* (rows of ``per``)."""
    out = np.empty((len(idx), per.shape[1]))
    for s in range(0, len(idx), chunk):
        out[s : s + chunk] = per[idx[s : s + chunk]].mean(axis=1)
    return out


def jackknife_g(per: np.ndarray) -> np.ndarray:
    """Leave-one-shot-out ``g``, shape ``(n, K)``."""
    n = len(per)
    return (per.sum(0)[None, :] - per) / (n - 1)


def jackknife_se(theta: np.ndarray) -> float:
    n = len(theta)
    return float(math.sqrt((n - 1) / n * np.sum((theta - theta.mean()) ** 2)))


def delta_se(per: np.ndarray, j: int) -> float:
    g = per.mean(0)
    cov = np.cov(per, rowvar=False) / len(per)
    d = witness_grad(g, j)
    return float(math.sqrt(d @ cov @ d))


def batch_se(per: np.ndarray, j: int, n_batches: int) -> float:
    """SE from ``w`` of contiguous shot batches: ``sd(w_b) / sqrt(B)``."""
    ws = np.array([witness(b.mean(0), j) for b in np.array_split(per, n_batches)])
    return float(ws.std(ddof=1) / math.sqrt(n_batches))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile by bisection on ``erfc`` (numpy-only, ~1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bca_ci(
    est: float, boots: np.ndarray, jack: np.ndarray, level: float = 0.95
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987)."""
    boots = np.asarray(boots)
    frac = float(np.mean(boots < est))
    frac = min(max(frac, 0.5 / len(boots)), 1 - 0.5 / len(boots))
    z0 = _norm_ppf(frac)
    dj = jack.mean() - jack
    acc = float(np.sum(dj**3) / (6 * np.sum(dj**2) ** 1.5)) if np.any(dj) else 0.0
    out = []
    for q in ((1 - level) / 2, (1 + level) / 2):
        zq = _norm_ppf(q)
        p = _norm_cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
        out.append(float(np.quantile(boots, p)))
    return out[0], out[1]


@dataclass
class WitnessResult:
    g: np.ndarray
    w: dict[int, float]
    se: dict[int, float]
    z: dict[int, float]
    boots: dict[int, np.ndarray]

    @property
    def max_abs_z(self) -> float:
        return max(abs(v) for v in self.z.values())


def witness_stats(
    per: np.ndarray, offsets: Sequence[int], idx: np.ndarray, size: int = 3
) -> WitnessResult:
    """Point estimates and shot-bootstrap SEs (sample SD, ``ddof=1``) from per-shot windows."""
    g = per.mean(0)
    bg = bootstrap_g(per, idx)
    w, se, z, boots = {}, {}, {}, {}
    for j in offsets:
        w[j] = witness(g, j, size)
        b = witness(bg, j, size)
        boots[j] = b
        se[j] = float(np.std(b, ddof=1))
        z[j] = w[j] / se[j] if se[j] > 0 else math.nan
    return WitnessResult(g, w, se, z, boots)


# -- linear processes ---------------------------------------------------------------------------


@dataclass
class LinearProcess:
    """``P(y_0..y_{T-1}) = init @ ops[y_0] @ .. @ final`` (an HMM or a qubit instrument)."""

    init: np.ndarray  # (D,)
    ops: np.ndarray  # (Y, D, D)
    final: np.ndarray  # (D,)

    @classmethod
    def hmm(cls, labelled: np.ndarray, initial: np.ndarray) -> LinearProcess:
        lab = np.asarray(labelled, dtype=np.float64)
        return cls(np.asarray(initial, dtype=np.float64), lab, np.ones(lab.shape[1]))

    @classmethod
    def instrument(cls, superops: np.ndarray, rho0: np.ndarray) -> LinearProcess:
        """From ``S[y]`` acting on column-stacked ``vec(rho)`` (``vec(E_y(rho)) = S[y] vec``)."""
        s = np.asarray(superops, dtype=np.complex128)
        d = round(math.sqrt(s.shape[1]))
        vec = lambda m: np.asarray(m, dtype=np.complex128).reshape(-1, order="F")  # noqa: E731
        # row-vector convention: state' = state @ S^T
        return cls(vec(rho0), np.transpose(s, (0, 2, 1)), vec(np.eye(d)))

    @property
    def channel(self) -> np.ndarray:
        return self.ops.sum(0)

    def state_after(self, t: int) -> np.ndarray:
        s = self.init
        for _ in range(t):
            s = s @ self.channel
        return s

    def stationary(self, iters: int = 4096) -> np.ndarray:
        s = self.init
        c = self.channel
        for _ in range(iters):
            s = s @ c
        return s

    def marginals(self, length: int) -> np.ndarray:
        s, out = self.init, []
        for _ in range(length):
            out.append((s @ self.ops[1] @ self.final).real)
            s = s @ self.channel
        return np.array(out, dtype=np.float64)

    def windows_at(self, state: np.ndarray, k_max: int) -> np.ndarray:
        """``g(k)`` from a fixed (unnormalised) state."""
        cur, out = state @ self.ops[1], []
        for _ in range(k_max + 1):
            out.append((cur @ self.ops[1] @ self.final).real)
            cur = cur @ self.ops[0]
        return np.array(out, dtype=np.float64)

    def expected_windows(
        self, k_max: int, burn: int, length: int, common: bool = False
    ) -> np.ndarray:
        """Exact expectation of ``per_shot_windows(...).mean(0)``: each ``k`` averages the
        state over its own starts ``burn .. length - k - 2`` (or the common ones)."""
        states = [self.state_after(burn)]
        for _ in range(length - burn - 1):
            states.append(states[-1] @ self.channel)
        out = np.zeros(k_max + 1)
        for k in range(k_max + 1):
            last = (length - k_max - 2) if common else (length - k - 2)
            gs = [self.windows_at(states[t - burn], k_max)[k] for t in range(burn, last + 1)]
            out[k] = np.mean(gs)
        return out

    def block_probs(self, starts: Sequence[int], size: int) -> np.ndarray:
        """Distribution of the ``size``-block (big-endian index), averaged over ``starts``."""
        out = np.zeros(2**size)
        for t0 in starts:
            states = self.state_after(t0)[None, :]
            for _ in range(size):
                states = np.stack([states @ self.ops[0], states @ self.ops[1]], axis=1)
                states = states.reshape(-1, states.shape[-1])
            out += (states @ self.final).real
        return out / len(starts)

    def sample(self, n: int, length: int, seed: int) -> np.ndarray:
        """``(n, length)`` uint8 samples by sequential conditioning (real or complex ops)."""
        rng = np.random.default_rng(seed)
        s = np.tile(self.init, (n, 1))
        out = np.zeros((n, length), dtype=np.uint8)
        for t in range(length):
            nb = s @ self.ops[1]
            p1 = (nb @ self.final).real / (s @ self.final).real
            y = rng.random(n) < p1
            out[:, t] = y
            s = np.where(y[:, None], nb, s @ self.ops[0])
            s = s / (s @ self.final).real[:, None]
        return out

    def seq_nll(self, shots: np.ndarray) -> np.ndarray:
        """``-log P(y_0..y_{T-1})`` per shot (forward algorithm, renormalised)."""
        shots = np.asarray(shots)
        n, length = shots.shape
        s = np.tile(self.init, (n, 1))
        total = np.zeros(n)
        for t in range(length):
            nb = np.where(shots[:, t, None] == 1, s @ self.ops[1], s @ self.ops[0])
            z = (nb @ self.final).real
            total -= np.log(z)
            s = nb / z[:, None]
        return total


def hmm_sample(labelled: np.ndarray, initial: np.ndarray, n: int, length: int, seed: int):
    """Fast sampler of an edge-emitting HMM (hidden-state simulation)."""
    lab = np.asarray(labelled, dtype=np.float64)
    ys, d, _ = lab.shape
    joint = lab.transpose(1, 0, 2).reshape(d, ys * d)  # (i, (y, j))
    cum = np.cumsum(joint, axis=1)
    rng = np.random.default_rng(seed)
    st = rng.choice(d, size=n, p=np.asarray(initial) / np.sum(initial))
    out = np.zeros((n, length), dtype=np.uint8)
    for t in range(length):
        u = rng.random(n) * cum[st, -1]
        c = (u[:, None] >= cum[st]).sum(1)
        c = np.minimum(c, ys * d - 1)
        out[:, t] = c // d
        st = c % d
    return out


def two_state_hmm(
    p_stay: tuple[float, float], p_one: tuple[float, float], stationary: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """A 2-state HMM with state-dependent emission: ``L[y, i, j] = P(y|i) T[i, j]``.

    ``p_stay[i]`` is ``T[i, i]``, ``p_one[i]`` is ``P(1 | i)``. The transition eigenvalue is
    ``p_stay[0] + p_stay[1] - 1``. Returns ``(labelled, initial)``.
    """
    a, b = p_stay
    t = np.array([[a, 1 - a], [1 - b, b]])
    e = np.array([[1 - p_one[0], 1 - p_one[1]], [p_one[0], p_one[1]]])  # e[y, i]
    lab = e[:, :, None] * t[None, :, :]
    if stationary:
        pi = np.array([1 - b, 1 - a]) / (2 - a - b)
    else:
        pi = np.array([1.0, 0.0])
    return lab, pi


def block_kl(shots: np.ndarray, proc: LinearProcess, size: int, burn: int) -> float:
    """``D(empirical || proc) / size`` of aligned ``size``-blocks from position ``burn``."""
    length = shots.shape[1]
    starts = list(range(burn, length - size + 1, size))
    weights = 1 << np.arange(size - 1, -1, -1)
    idx = np.concatenate([shots[:, t : t + size].astype(np.int64) @ weights for t in starts])
    emp = np.bincount(idx, minlength=2**size) / len(idx)
    q = proc.block_probs(starts, size)
    m = emp > 0
    return float(np.sum(emp[m] * np.log(emp[m] / q[m])) / size)


# -- Baum-Welch (numpy, independent of qmem.baselines.hmm) -------------------------------------


def hmm_forward_backward(lab: np.ndarray, init: np.ndarray, shots: np.ndarray):
    """Expected counts ``C[y, i, j]``, initial posteriors and total log-likelihood."""
    n, length = shots.shape
    d = lab.shape[1]
    alpha = np.empty((length + 1, n, d))
    scale = np.empty((length, n))
    a = np.tile(init, (n, 1))
    alpha[0] = a
    for t in range(length):
        a = np.where(shots[:, t, None] == 1, a @ lab[1], a @ lab[0])
        z = a.sum(1)
        scale[t] = z
        a = a / z[:, None]
        alpha[t + 1] = a
    beta = np.ones((n, d))
    counts = np.zeros_like(lab)
    for t in range(length - 1, -1, -1):
        y = shots[:, t]
        a_t = alpha[t] / scale[t][:, None]
        one = (y == 1)[:, None]
        counts[1] += lab[1] * ((a_t * one).T @ beta)
        counts[0] += lab[0] * ((a_t * ~one).T @ beta)
        beta = np.where(one, beta @ lab[1].T, beta @ lab[0].T) / scale[t][:, None]
    gamma0 = (alpha[0] * beta).sum(0)
    return counts, gamma0, float(np.log(scale).sum())


def hmm_nll(lab: np.ndarray, init: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """Per-shot ``-log P`` of an HMM."""
    return LinearProcess.hmm(lab, init).seq_nll(shots)


def fit_em(
    lab: np.ndarray,
    init: np.ndarray,
    train: np.ndarray,
    val: np.ndarray | None = None,
    max_iters: int = 2000,
    tol: float = 1e-9,
    patience: int | None = None,
    min_delta: float = 1e-5,
) -> dict:
    """Baum-Welch from ``(lab, init)``.

    Without ``patience``: run until the train NLL per symbol improves by less than ``tol``
    (keeps the final parameters). With ``patience`` (and ``val``): Phase 6's rule, keep the
    best-validation parameters and stop after ``patience`` iterations without a ``min_delta``
    improvement.
    """
    lab, init = np.array(lab, dtype=np.float64), np.array(init, dtype=np.float64)
    n_sym = train.size
    prev = math.inf
    best = (math.inf, lab.copy(), init.copy(), -1)
    bad, it = 0, 0
    for it in range(max_iters):
        counts, gamma0, ll = hmm_forward_backward(lab, init, train)
        nll = -ll / n_sym
        if patience is not None:
            v = float(hmm_nll(lab, init, val).sum() / val.size)
            if v < best[0] - min_delta:
                best, bad = (v, lab.copy(), init.copy(), it), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        elif prev - nll < tol:
            break
        prev = nll
        row = counts.sum(axis=(0, 2))
        vis = row > 0
        lab = np.where(vis[None, :, None], counts / np.where(vis, row, 1)[None, :, None], lab)
        init = gamma0 / gamma0.sum()
    if patience is not None:
        lab, init = best[1], best[2]
    return {"labelled": lab, "initial": init, "iters": it + 1, "train_nll": nll}


def random_hmm(d: int, rng: np.random.Generator, n_symbols: int = 2):
    """Dirichlet(1) rows over ``(y, j)`` and a Dirichlet(1) initial distribution."""
    rows = rng.dirichlet(np.ones(n_symbols * d), size=d)  # (i, (y, j))
    lab = rows.reshape(d, n_symbols, d).transpose(1, 0, 2)
    return lab, rng.dirichlet(np.ones(d))


def renewal_init(shots: np.ndarray, d: int, eps: float = 1e-3):
    """Moment-based start: states = zeros since the last one, capped at ``d - 1``.

    The hazard ``P(1 | gap)`` is estimated from the data, so this is the best ``d``-state
    unifilar renewal model (lightly smoothed with ``eps`` so EM can move every entry).
    """
    y = np.asarray(shots)
    gap = np.zeros(len(y), dtype=np.int64)  # treat t=0 as just after a one
    ones, tot = np.zeros(d), np.zeros(d)
    first = np.zeros(d)
    for t in range(y.shape[1]):
        np.add.at(tot, gap, 1)
        np.add.at(ones, gap, y[:, t])
        if t == 0:
            np.add.at(first, gap, 1)
        gap = np.where(y[:, t] == 1, 0, np.minimum(gap + 1, d - 1))
    h = (ones + 0.5) / (tot + 1)
    lab = np.zeros((2, d, d))
    for i in range(d):
        lab[1, i, 0] = h[i]
        lab[0, i, min(i + 1, d - 1)] = 1 - h[i]
    lab = (1 - eps) * lab + eps / (2 * d)
    init = (1 - eps) * first / first.sum() + eps / d
    return lab, init / init.sum()


def markov_init(shots: np.ndarray, order: int, eps: float = 1e-3):
    """Moment-based start: an order-``m`` Markov chain as a ``2^m``-state HMM."""
    y = np.asarray(shots)
    d = 2**order
    ones, tot = np.zeros(d), np.zeros(d)
    for t in range(order, y.shape[1]):
        ctx = np.zeros(len(y), dtype=np.int64)
        for u in range(order):
            ctx = 2 * ctx + y[:, t - order + u]
        np.add.at(tot, ctx, 1)
        np.add.at(ones, ctx, y[:, t])
    p = (ones + 0.5) / (tot + 1)
    lab = np.zeros((2, d, d))
    for c in range(d):
        for s in (0, 1):
            lab[s, c, (2 * c + s) % d] = p[c] if s else 1 - p[c]
    lab = (1 - eps) * lab + eps / (2 * d)
    return lab, np.full(d, 1.0 / d)


def split_init(lab: np.ndarray, init: np.ndarray, rng: np.random.Generator, noise: float = 0.05):
    """A ``(D+1)``-state start from a ``D``-state fit: split the most visited state."""
    d = lab.shape[1]
    pi = LinearProcess.hmm(lab, init).stationary().real
    s = int(np.argmax(pi))
    new = np.zeros((2, d + 1, d + 1))
    new[:, :d, :d] = lab
    new[:, :d, d] = lab[:, :d, s] / 2
    new[:, :d, s] = lab[:, :d, s] / 2
    new[:, d, :] = new[:, s, :]
    new *= np.exp(noise * rng.standard_normal(new.shape))
    new /= new.sum(axis=(0, 2), keepdims=True)
    ini = np.append(init, init[s] / 2)
    ini[s] /= 2
    return new, ini / ini.sum()
