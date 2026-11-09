"""Phase 8 Stage 0A (issue #29): an independent re-analysis path for Phase 7 Stage 4c (#27).

Numpy only. Nothing here calls ``qmem.witness``, ``qmem.drift`` or ``qmem.followup``: it is a
second, minimal computation used to check them (``experiments/phase8_stage0.py`` compares the
two). It shares only the estimator's *definition* (``qmem.witness``'s docstring):

* ``g(k) = P(1 0^k 1)`` per shot, over the common start positions ``t = burn .. L - k_max - 2``
  (as Stage 3/4 analyse it, ``common=True``), averaged over shots;
* ``w(j) = det H_j / max|H_j|^3`` with ``H_j[a, b] = g(j + a + b)``, ``a, b < 3``.

The decoding (``np.unpackbits``), the window count (a sliding match per ``k`` rather than
``qmem.witness``'s gaps between ones), the 3 x 3 determinant (cofactors), the resampling, the
lag-1 excess and the decision arithmetic are all written again here. Also: Fieller's interval
for a ratio of independent normal estimates, TOST equivalence and its power, fixed- and
random-effects pooling with Cochran's Q, and the power of the vanish rule.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Z975 = 1.959963984540054  # two-sided 95%
Z95 = 1.6448536269514722  # one-sided 95% (TOST at alpha = 0.05 per bound)

# -- loading the committed shots -------------------------------------------------------------

KEY = re.compile(
    r"^j(?P<job>\d+)_p(?P<pub>\d+)_(?P<arm>.+?)(?:_i(?P<inst>\d+))?_c(?P<chunk>\d+)__(?P<reg>[a-z0-9_]+)$"
)


def parse_key(key: str) -> dict[str, Any] | None:
    """``j1_p004_A2_c0__m`` -> job 1, pub 4, arm A2, chunk 0, register m (None if not a pub)."""
    m = KEY.match(key)
    if m is None:
        return None
    d = m.groupdict()
    return {
        "job": int(d["job"]),
        "pub": int(d["pub"]),
        "arm": d["arm"],
        "instance": None if d["inst"] is None else int(d["inst"]),
        "chunk": int(d["chunk"]),
        "reg": d["reg"],
    }


def unpack(packed: np.ndarray, length: int) -> np.ndarray:
    """``np.packbits(axis=1)`` rows back to ``(shots, length)`` uint8 bits (big bit order)."""
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, count=length)


def load_jobs(paths: Sequence[str | Path], reg: str = "y") -> dict[str, dict[str, np.ndarray]]:
    """Every arm's ``reg`` shots in nominal time order (file order, then pub position), with
    job (the key's ``j<n>``), pub and instance labels; arms keep their order of first
    appearance."""
    out: dict[str, dict[str, list]] = {}
    for path in paths:
        with np.load(path) as z:
            length = int(z["length"])
            rows = []
            for k in z.files:
                p = parse_key(k)
                if p is not None and p["reg"] == reg:
                    rows.append((p["pub"], k, p))
            for _, k, p in sorted(rows):
                y = unpack(z[k], length)
                d = out.setdefault(p["arm"], {"shots": [], "jobs": [], "pubs": [], "inst": []})
                d["shots"].append(y)
                d["jobs"].append(np.full(len(y), p["job"]))
                d["pubs"].append(np.full(len(y), p["pub"]))
                d["inst"].append(np.full(len(y), -1 if p["instance"] is None else p["instance"]))
    return {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in out.items()}


# -- the estimator, counted again ------------------------------------------------------------


def window_freqs(shots: np.ndarray, k_max: int, burn: int) -> np.ndarray:
    """Per-shot frequency of ``1 0^k 1`` over the common starts ``burn .. L - k_max - 2``.

    A sliding match per ``k``: a one at ``t``, zeros at ``t+1 .. t+k``, a one at ``t+k+1``.
    """
    y = np.asarray(shots).astype(bool)
    length = y.shape[1]
    t = np.arange(burn, length - k_max - 1)
    out = np.empty((len(y), k_max + 1))
    for k in range(k_max + 1):
        hit = y[:, t] & y[:, t + k + 1]
        for d in range(1, k + 1):
            hit &= ~y[:, t + d]
        out[:, k] = hit.sum(1) / len(t)
    return out


def det3(h: np.ndarray) -> np.ndarray:
    """Determinant of ``(..., 3, 3)`` by cofactor expansion along the first row."""
    return (
        h[..., 0, 0] * (h[..., 1, 1] * h[..., 2, 2] - h[..., 1, 2] * h[..., 2, 1])
        - h[..., 0, 1] * (h[..., 1, 0] * h[..., 2, 2] - h[..., 1, 2] * h[..., 2, 0])
        + h[..., 0, 2] * (h[..., 1, 0] * h[..., 2, 1] - h[..., 1, 1] * h[..., 2, 0])
    )


def wit(g: np.ndarray, j: int) -> np.ndarray | float:
    """``w(j)`` of ``g`` (batched over leading axes)."""
    g = np.asarray(g, dtype=np.float64)
    h = np.stack([np.stack([g[..., j + a + b] for b in range(3)], -1) for a in range(3)], -2)
    out = det3(h) / np.abs(h).max(axis=(-1, -2)) ** 3
    return float(out) if np.ndim(out) == 0 else out


def lag1_excess(shots: np.ndarray, burn: int) -> float:
    """``P(1 | previous 1) - P(1 | previous 0)`` over pairs ``(t, t+1)``, ``t >= burn``."""
    y = np.asarray(shots)[:, burn:].astype(np.int64)
    a, b = y[:, :-1].ravel(), y[:, 1:].ravel()
    return float(b[a == 1].mean() - b[a == 0].mean())


def lag_stats(shots: np.ndarray, burn: int, k: int) -> np.ndarray:
    """Per shot ``(n, sx, sy, sxx, syy, sxy)`` of the pairs ``(y_t, y_{t+k})``, ``t >= burn``."""
    y = np.asarray(shots)[:, burn:].astype(np.float64)
    x, z = y[:, :-k], y[:, k:]
    n = np.full(len(y), x.shape[1], dtype=np.float64)
    return np.stack([n, x.sum(1), z.sum(1), (x * x).sum(1), (z * z).sum(1), (x * z).sum(1)], 1)


def corr_from(s: np.ndarray) -> np.ndarray:
    """Pearson correlation from pooled sums (rows of ``lag_stats`` summed; batched)."""
    s = np.atleast_2d(s)
    n, sx, sy, sxx, syy, sxy = (s[:, i] for i in range(6))
    cov = sxy / n - sx * sy / n**2
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    return cov / np.sqrt(vx * vy)


# -- resampling -------------------------------------------------------------------------------


def circular_blocks(n: int, b: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular moving-block resamples of ``range(n)``: ``ceil(n / b)`` uniform block starts
    per resample, each block ``b`` consecutive indices modulo ``n``, cut to ``n``."""
    rng = np.random.default_rng(seed)
    nb = -(-n // b)
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(b)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def label_strata(labels: np.ndarray) -> list[np.ndarray]:
    """Index arrays of each label, labels in order of first appearance."""
    labels = np.asarray(labels)
    seen: dict[Any, list[int]] = {}
    for i, v in enumerate(labels.tolist()):
        seen.setdefault(v, []).append(i)
    return [np.array(v) for v in seen.values()]


def stratified_blocks(
    strata: Sequence[np.ndarray], b: int, n_boot: int, seed: int, cap: bool = True
) -> np.ndarray:
    """Block resamples inside each stratum (sizes fixed), concatenated. ``cap``: a stratum's
    block is at most a quarter of it (#19's rule, at least 4 blocks per stratum). Stratum ``s``
    draws with ``seed + 104729 s``."""
    out = []
    for s, seg in enumerate(strata):
        bb = max(1, min(b, len(seg) // 4)) if cap else max(1, min(b, len(seg)))
        out.append(np.asarray(seg)[circular_blocks(len(seg), bb, n_boot, seed + 104729 * s)])
    return np.concatenate(out, axis=1)


def resample_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of the rows of ``x`` in each resample (multiplicities by ``bincount``)."""
    n = len(x)
    cnt = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)
    return cnt @ x / idx.shape[1]


def sd(x: np.ndarray) -> float:
    return float(np.std(np.asarray(x), ddof=1))


# -- one arm ----------------------------------------------------------------------------------


def arm_stats(
    shots: np.ndarray,
    jobs: np.ndarray,
    k_max: int,
    burn: int,
    offsets: Sequence[int],
    block: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Stage 3/4's per-arm statistics by this path: w(j); the shot and block (stratified by
    job) bootstrap SEs, the primary SE (the larger), z, the 95% CI of w(0) from the method that
    sets it; P(1) after the burn-in; the lag-1 excess and its SE; per-job w(0) and P(1)."""
    per = window_freqs(shots, k_max, burn)
    g = per.mean(0)
    st = label_strata(jobs)
    i_s = stratified_blocks(st, 1, n_boot, seed)
    i_b = stratified_blocks(st, block, n_boot, seed + 1)
    lag = lag_pair_counts(shots, burn)
    gs, gb = resample_means(per, i_s), resample_means(per, i_b)
    ls, lb = resample_means(lag, i_s), resample_means(lag, i_b)
    out: dict[str, Any] = {"n": len(shots), "g": g.tolist()}
    for j in offsets:
        w = wit(g, j)
        ws, wb = wit(gs, j), wit(gb, j)
        out[f"w{j}"] = w
        out[f"se_shot{j}"], out[f"se_block{j}"] = sd(ws), sd(wb)
        out[f"se{j}"] = max(out[f"se_shot{j}"], out[f"se_block{j}"])
        out[f"z{j}"] = w / out[f"se{j}"]
        if j == 0:
            src = wb if out["se_block0"] >= out["se_shot0"] else ws
            out["ci0"] = [float(np.quantile(src, 0.025)), float(np.quantile(src, 0.975))]
    out["p1"] = float(np.asarray(shots)[:, burn:].mean())
    out["lag1"] = lag1_excess(shots, burn)
    out["lag1_se"] = max(sd(_excess(ls)), sd(_excess(lb)))
    out["per_job"] = [
        {
            "job": int(jobs[s[0]]),
            "n": len(s),
            "w0": wit(per[s].mean(0), 0),
            "p1": float(np.asarray(shots)[s, burn:].mean()),
        }
        for s in st
    ]
    return out


def lag_pair_counts(shots: np.ndarray, burn: int) -> np.ndarray:
    """Per shot ``(#11, #1*, #01, #0*)`` over pairs ``(t, t+1)``, ``t >= burn``."""
    y = np.asarray(shots)[:, burn:].astype(np.int64)
    a, b = y[:, :-1], y[:, 1:]
    return np.stack([(a & b).sum(1), a.sum(1), ((1 - a) & b).sum(1), (1 - a).sum(1)], 1).astype(
        np.float64
    )


def _excess(c: np.ndarray) -> np.ndarray:
    c = np.atleast_2d(c)
    return c[:, 0] / c[:, 1] - c[:, 2] / c[:, 3]


# -- decision arithmetic ----------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (bisection; accurate to ~1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if norm_cdf(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def holm_reject(p: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down at family-wise ``alpha``."""
    order = sorted(p, key=lambda k: p[k])
    out = {k: False for k in p}
    for i, k in enumerate(order):
        if p[k] > alpha / (len(order) - i):
            break
        out[k] = True
    return out


def vanish_bound(w_a2: float, se_a2: float, w_base: float, z: float = Z975) -> float:
    """The pre-registered primary bound as a fraction of the baseline: ``(|w| + z SE) / |w_b|``
    (the baseline's own SE not propagated)."""
    return (abs(w_a2) + z * se_a2) / abs(w_base)


def call(bound: float, fraction: float = 0.10) -> str:
    """``vanishes`` if the bound is below the fraction; otherwise not decisive here."""
    return "vanishes" if bound < fraction else "ambiguous"


def ratio_delta(a: float, sa: float, b: float, sb: float) -> tuple[float, float]:
    """``a / b`` and its delta-method SE (independent estimates)."""
    r = a / b
    return r, abs(r) * math.sqrt((sa / a) ** 2 + (sb / b) ** 2)


def ratio_delta_abs(a: float, sa: float, b: float, sb: float) -> tuple[float, float]:
    """``a / b`` and the delta SE ``sqrt(sa^2 + r^2 sb^2) / |b|`` (finite as ``a -> 0``)."""
    r = a / b
    return r, math.sqrt(sa**2 + r**2 * sb**2) / abs(b)


def fieller(a: float, sa: float, b: float, sb: float, z: float = Z975) -> tuple[float, float]:
    """Fieller's interval for ``alpha / beta`` from independent ``a ~ N(alpha, sa^2)``,
    ``b ~ N(beta, sb^2)``: the ``rho`` with ``(a - rho b)^2 <= z^2 (sa^2 + rho^2 sb^2)``.
    Raises if the set is unbounded (``b`` not significant at ``z``)."""
    qa = b * b - z * z * sb * sb
    if qa <= 0:
        raise ValueError("Fieller set unbounded: the denominator is not significant")
    qb = -2.0 * a * b
    qc = a * a - z * z * sa * sa
    disc = qb * qb - 4 * qa * qc
    r1 = (-qb - math.sqrt(disc)) / (2 * qa)
    r2 = (-qb + math.sqrt(disc)) / (2 * qa)
    return min(r1, r2), max(r1, r2)


def tost(est: float, se: float, lo: float, hi: float, alpha: float = 0.05) -> dict[str, Any]:
    """Two one-sided z-tests of ``lo < theta < hi``: p of each bound and the decision."""
    p_lo = 1.0 - norm_cdf((est - lo) / se)
    p_hi = norm_cdf((est - hi) / se)
    p = max(p_lo, p_hi)
    return {"p_lower": p_lo, "p_upper": p_hi, "p": p, "equivalent": bool(p < alpha)}


def tost_power(true: float, se: float, lo: float, hi: float, alpha: float = 0.05) -> float:
    """P(TOST declares equivalence) for a normal estimate with SE ``se`` around ``true``."""
    z = norm_ppf(1 - alpha)
    a, b = lo + z * se, hi - z * se
    if a >= b:
        return 0.0
    return norm_cdf((b - true) / se) - norm_cdf((a - true) / se)


def vanish_power(mu: float, se: float, threshold: float, z: float = Z975) -> float:
    """P(|X| + z se < threshold), X ~ N(mu, se^2): the vanish rule with the baseline fixed."""
    c = threshold - z * se
    if c <= 0:
        return 0.0
    return norm_cdf((c - mu) / se) - norm_cdf((-c - mu) / se)


def vanish_power_mc(
    ratio: float,
    w_base: float,
    se_base: float,
    se_a2: float,
    fraction: float,
    gate_z: float,
    n: int,
    seed: int,
    z: float = Z975,
) -> float:
    """Monte Carlo P(gate and vanishes) for a replicate: ``w_b* ~ N(w_b, se_b)``,
    ``w_a* ~ N(ratio w_b, se_a)``, the SEs as given."""
    rng = np.random.default_rng(seed)
    wb = w_base + se_base * rng.standard_normal(n)
    wa = ratio * w_base + se_a2 * rng.standard_normal(n)
    ok = (
        (wb < 0)
        & (np.abs(wb) >= gate_z * se_base)
        & (np.abs(wa) + z * se_a2 < fraction * np.abs(wb))
    )
    return float(ok.mean())


def shots_for(
    criterion, n0: float, se0: float, n_lo: float = 100.0, n_hi: float = 1e9
) -> float | None:
    """Smallest ``n`` (shots) with ``criterion(se0 * sqrt(n0 / n))`` true, by bisection on
    ``log n``; SE scales as ``1 / sqrt(n)``. None if not reached by ``n_hi``."""
    if not criterion(se0 * math.sqrt(n0 / n_hi)):
        return None
    lo, hi = math.log(n_lo), math.log(n_hi)
    if criterion(se0 * math.sqrt(n0 / n_lo)):
        return n_lo
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if criterion(se0 * math.sqrt(n0 / math.exp(mid))):
            hi = mid
        else:
            lo = mid
    return math.exp(hi)


# -- pooling and heterogeneity ----------------------------------------------------------------


def chi2_sf(x: float, dof: int) -> float:
    """Survival function of chi-squared (series for the regularised lower gamma)."""
    if x <= 0:
        return 1.0
    a, t = dof / 2.0, x / 2.0
    if t > a + 40 * math.sqrt(a) + 40:
        return 0.0
    term = 1.0 / a
    s = term
    k = 1
    while term > 1e-17 * s:
        term *= t / (a + k)
        s += term
        k += 1
    lower = s * math.exp(-t + a * math.log(t) - math.lgamma(a))
    return max(0.0, 1.0 - lower)


def pool(est: Sequence[float], se: Sequence[float]) -> dict[str, float]:
    """Inverse-variance fixed effect, Cochran's Q (p, I^2), DerSimonian-Laird random effect."""
    x, s = np.asarray(est, dtype=float), np.asarray(se, dtype=float)
    w = 1 / s**2
    fe = float(np.sum(w * x) / w.sum())
    q = float(np.sum(w * (x - fe) ** 2))
    df = len(x) - 1
    c = float(w.sum() - np.sum(w**2) / w.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (s**2 + tau2)
    re_ = float(np.sum(wr * x) / wr.sum())
    return {
        "fe": fe,
        "fe_se": float(math.sqrt(1 / w.sum())),
        "q": q,
        "df": df,
        "q_p": chi2_sf(q, df) if df > 0 else math.nan,
        "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
        "tau2": tau2,
        "re": re_,
        "re_se": float(math.sqrt(1 / wr.sum())),
    }


# -- provenance -------------------------------------------------------------------------------


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(root: str | Path, rel: Iterable[str]) -> dict[str, str]:
    """``{relative path: sha256}`` of files under ``root``."""
    root = Path(root)
    return {r: sha256(root / r) for r in sorted(rel)}


__all__ = [
    "KEY",
    "Z95",
    "Z975",
    "arm_stats",
    "call",
    "chi2_sf",
    "circular_blocks",
    "corr_from",
    "det3",
    "fieller",
    "hash_files",
    "holm_reject",
    "label_strata",
    "lag1_excess",
    "lag_pair_counts",
    "lag_stats",
    "load_jobs",
    "norm_cdf",
    "norm_ppf",
    "p_two_sided",
    "parse_key",
    "pool",
    "ratio_delta",
    "ratio_delta_abs",
    "resample_means",
    "sd",
    "sha256",
    "shots_for",
    "stratified_blocks",
    "tost",
    "tost_power",
    "unpack",
    "vanish_bound",
    "vanish_power",
    "vanish_power_mc",
    "window_freqs",
    "wit",
]
