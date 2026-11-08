"""Classical readout, control and processing artefacts, and memoryless control circuits.

Phase 7 Stage 2D (issue #23). Numpy only; built on ``qmem.witness`` (``LinearProcess``,
``per_shot_windows``, ``witness``), which it does not modify.

Every artefact has two forms:

* an **exact** form: a transform of a ``LinearProcess`` (or a mixture of them over shots)
  whose ``expected_windows`` is the exact expectation of the estimator on artefact-laden data;
* a **data** form: the same artefact planted into sampled shots (explicit seeds), so the
  whole pipeline can be run end to end and checked against the exact form.

The artefacts act on the *recorded* bit ``r_t`` after the physical outcome ``y_t``:

* ``persistence``: the assignment depends on the previous *recorded* bit
  (``P(r_t | y_t, r_{t-1})``), e.g. discriminator hysteresis or a stuck readout buffer.
  The composite process has dimension ``2 D``; on an i.i.d. source it is a 2-state Markov
  chain, so ``w = 0`` exactly.
* ``drift``: the assignment matrix changes across shots (a discriminator threshold shift
  ``delta(n)``). The data are a mixture over shots, ``g = mean_n g_n``. For an i.i.d. source
  ``g(k) = E[p^2 (1 - p)^k]`` is a moment sequence, so every Hankel matrix is positive
  semidefinite and ``w(j) >= 0``: no mixture of i.i.d. shots gives the device's negative
  ``w``.
* ``burst``: a contiguous run of shots is corrupted (random bits, zeros, or a large
  assignment error): again a mixture over shots.
* ``reordering``: a permutation of shots leaves the pooled ``g`` unchanged exactly (the
  window counts are integers) but changes contiguous-block SEs when the data are not
  exchangeable.
* ``packing``: ``np.packbits`` / unpack mistakes (byte order, bit order) permute the columns.
* ``drop`` / ``duplicate``: losing or repeating whole shots leaves ``E[g]`` unchanged;
  duplicates shrink the whole-shot bootstrap SE.

``qubit_control`` gives the exact classical model of the memoryless controls D1 (reset every
step), D2 (no reset) and their noise: a qubit measured every step has no coherence left
after the measurement, so its outputs are an HMM on its true level (0, 1 and, with leakage,
2). Without leakage it has 2 states, so ``w = 0`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from qmem import witness as W

# -- elementary processes --


def iid_process(p1: float) -> W.LinearProcess:
    """A Bernoulli(``p1``) source as a 1-state ``LinearProcess``."""
    lab = np.array([[[1.0 - p1]], [[p1]]])
    return W.LinearProcess.hmm(lab, np.ones(1))


def readout_matrix(e01: float, e10: float) -> np.ndarray:
    """``M[y, r] = P(record r | outcome y)``: ``e01 = P(1|0)``, ``e10 = P(0|1)``."""
    return np.array([[1.0 - e01, e01], [e10, 1.0 - e10]])


def threshold_matrix(delta: float) -> np.ndarray:
    """A discriminator threshold shift: ``delta > 0`` adds ``P(1|0) = delta``, ``delta < 0``
    adds ``P(0|1) = -delta``."""
    return readout_matrix(max(delta, 0.0), max(-delta, 0.0))


def with_readout(proc: W.LinearProcess, m: np.ndarray) -> W.LinearProcess:
    """The recorded process after a memoryless assignment ``M[y, r]``: ``O'_r = sum_y M O_y``."""
    ops = np.einsum("yr,yij->rij", np.asarray(m, dtype=np.float64), proc.ops)
    return W.LinearProcess(proc.init, ops, proc.final)


def persistence_kernel(eps: float, kind: str = "sticky") -> np.ndarray:
    """``A[y, r_prev, r] = P(r_t = r | y_t = y, r_{t-1} = r_prev)``.

    * ``sticky``: with probability ``eps`` the record repeats the previous record, else it is
      ``y`` (a symmetric hysteresis; the misassignment rate is ``eps * P(r_prev != y)``).
    * ``after_one``: after a recorded 1, a 0 is recorded as 1 with probability ``eps``
      (a one-sided bias, e.g. a residual excitation of the readout chain).
    * ``after_zero``: the mirror image.
    """
    a = np.zeros((2, 2, 2))
    for y in (0, 1):
        for rp in (0, 1):
            if kind == "sticky":
                a[y, rp, rp] += eps
                a[y, rp, y] += 1.0 - eps
            elif kind == "after_one":
                if rp == 1 and y == 0:
                    a[y, rp, 1], a[y, rp, 0] = eps, 1.0 - eps
                else:
                    a[y, rp, y] = 1.0
            elif kind == "after_zero":
                if rp == 0 and y == 1:
                    a[y, rp, 0], a[y, rp, 1] = eps, 1.0 - eps
                else:
                    a[y, rp, y] = 1.0
            else:
                raise ValueError(f"unknown persistence kind {kind!r}")
    return a


def with_persistence(
    proc: W.LinearProcess, kernel: np.ndarray, r_prev0: int = 0
) -> W.LinearProcess:
    """The recorded process when the assignment depends on the previous record.

    State ``(s, r_prev)`` (index ``2 s + r_prev``): ``O'_r = sum_y O_y (x) B_{y, r}`` with
    ``B_{y, r}[rp, rp'] = A[y, rp, r] [rp' = r]``. Before the first symbol ``r_prev = r_prev0``
    (the data form uses the same convention).
    """
    d = proc.ops.shape[1]
    dt = np.result_type(proc.ops.dtype, np.float64)
    ops = np.zeros((2, 2 * d, 2 * d), dtype=dt)
    for r in (0, 1):
        for y in (0, 1):
            b = np.zeros((2, 2))
            b[:, r] = kernel[y, :, r]
            ops[r] += np.kron(proc.ops[y], b)
    e = np.zeros(2)
    e[r_prev0] = 1.0
    return W.LinearProcess(np.kron(proc.init, e), ops, np.kron(proc.final, np.ones(2)))


def expected_mixture(
    procs: Sequence[W.LinearProcess],
    weights: Sequence[float],
    k_max: int,
    burn: int,
    length: int,
    common: bool = True,
) -> np.ndarray:
    """Exact ``E[g]`` when a fraction ``weights[i]`` of the shots comes from ``procs[i]``."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sum(
        wi * p.expected_windows(k_max, burn, length, common=common)
        for wi, p in zip(w, procs, strict=True)
        if wi > 0
    )


# -- drift and bursts --


def drift_profile(n_shots: int, amplitude: float, kind: str = "ramp", levels: int = 41):
    """Per-shot threshold shift ``delta(n)``, quantised to ``levels`` values in
    ``[-amplitude, amplitude]`` (so the exact mixture has ``levels`` components).

    ``ramp``: linear in shot index; ``step``: ``-a`` then ``+a`` at half the run; ``sine``:
    one period; ``oneside``: a ramp from 0 to ``+a`` (a threshold walking one way).
    """
    x = np.arange(n_shots) / max(n_shots - 1, 1)
    if kind == "ramp":
        raw = 2 * x - 1
    elif kind == "step":
        raw = np.where(x < 0.5, -1.0, 1.0)
    elif kind == "sine":
        raw = np.sin(2 * np.pi * x)
    elif kind == "oneside":
        raw = x
    else:
        raise ValueError(f"unknown drift kind {kind!r}")
    lo = 0.0 if kind == "oneside" else -1.0
    grid = np.linspace(lo, 1.0, levels)
    q = grid[np.abs(raw[:, None] - grid[None, :]).argmin(1)]
    return amplitude * q


def expected_drift(proc, deltas, k_max: int, burn: int, length: int) -> np.ndarray:
    """Exact ``E[g]`` under per-shot threshold shifts ``deltas`` (a mixture over levels)."""
    vals, counts = np.unique(np.round(np.asarray(deltas), 15), return_counts=True)
    procs = [with_readout(proc, threshold_matrix(float(v))) for v in vals]
    return expected_mixture(procs, counts, k_max, burn, length)


def burst_slice(n_shots: int, frac: float, start: int) -> slice:
    """The contiguous corrupted run: ``round(frac * n)`` shots from ``start`` (moved back to
    fit)."""
    m = round(frac * n_shots)
    s = min(start, n_shots - m)
    return slice(s, s + m)


def burst_process(proc: W.LinearProcess, kind: str, param: float) -> W.LinearProcess:
    """The process of a corrupted shot: ``random`` (i.i.d. bits, ``P(1) = param``), ``zeros``,
    or ``assignment`` (an extra symmetric assignment error ``param``)."""
    if kind == "random":
        return iid_process(param)
    if kind == "zeros":
        return iid_process(0.0)
    if kind == "assignment":
        return with_readout(proc, readout_matrix(param, param))
    raise ValueError(f"unknown burst kind {kind!r}")


def expected_burst(proc, n_shots, frac, kind, param, k_max, burn, length) -> np.ndarray:
    sl = burst_slice(n_shots, frac, 0)
    m = sl.stop - sl.start
    return expected_mixture(
        [proc, burst_process(proc, kind, param)], [n_shots - m, m], k_max, burn, length
    )


# -- data forms (explicit seeds) --


def apply_readout(shots: np.ndarray, e01, e10, rng: np.random.Generator) -> np.ndarray:
    """Flip recorded bits: ``0 -> 1`` with ``e01``, ``1 -> 0`` with ``e10`` (scalars or per-shot
    arrays)."""
    y = np.asarray(shots, dtype=np.uint8)
    n = len(y)
    a = np.broadcast_to(np.asarray(e01, dtype=np.float64).reshape(-1, 1), (n, 1))
    b = np.broadcast_to(np.asarray(e10, dtype=np.float64).reshape(-1, 1), (n, 1))
    u = rng.random(y.shape)
    flip = np.where(y == 1, u < b, u < a)
    return (y ^ flip).astype(np.uint8)


def apply_threshold(shots: np.ndarray, deltas, rng: np.random.Generator) -> np.ndarray:
    d = np.asarray(deltas, dtype=np.float64)
    return apply_readout(shots, np.maximum(d, 0.0), np.maximum(-d, 0.0), rng)


def apply_persistence(
    shots: np.ndarray, kernel: np.ndarray, rng: np.random.Generator, r_prev0: int = 0
) -> np.ndarray:
    """Record ``r_t ~ A[y_t, r_{t-1}, .]`` sequentially (``r_{-1} = r_prev0``)."""
    y = np.asarray(shots, dtype=np.int64)
    n, length = y.shape
    out = np.zeros((n, length), dtype=np.uint8)
    prev = np.full(n, r_prev0, dtype=np.int64)
    for t in range(length):
        p1 = kernel[y[:, t], prev, 1]
        r = (rng.random(n) < p1).astype(np.int64)
        out[:, t] = r
        prev = r
    return out


def apply_burst(
    shots: np.ndarray, frac: float, start: int, kind: str, param: float, rng
) -> np.ndarray:
    out = np.array(shots, dtype=np.uint8, copy=True)
    sl = burst_slice(len(out), frac, start)
    seg = out[sl]
    if kind == "random":
        out[sl] = (rng.random(seg.shape) < param).astype(np.uint8)
    elif kind == "zeros":
        out[sl] = 0
    elif kind == "assignment":
        out[sl] = apply_readout(seg, param, param, rng)
    else:
        raise ValueError(f"unknown burst kind {kind!r}")
    return out


def permute_shots(shots: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Reorder whole shots: ``random``, ``reverse``, ``sort_p1`` (by each shot's P(1)), or
    ``interleave`` (even shots, then odd)."""
    y = np.asarray(shots)
    n = len(y)
    if kind == "random":
        idx = np.random.default_rng(seed).permutation(n)
    elif kind == "reverse":
        idx = np.arange(n)[::-1]
    elif kind == "sort_p1":
        idx = np.argsort(y.mean(1), kind="stable")
    elif kind == "interleave":
        idx = np.concatenate([np.arange(0, n, 2), np.arange(1, n, 2)])
    else:
        raise ValueError(f"unknown permutation {kind!r}")
    return y[idx]


def pack_shots(shots: np.ndarray) -> np.ndarray:
    """As Phase 6 saves them: ``np.packbits(arr, axis=1)`` (big bit order)."""
    return np.packbits(np.asarray(shots, dtype=np.uint8), axis=1)


_BITREV = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8)


def corrupt_packed(packed: np.ndarray, kind: str) -> np.ndarray:
    """Packing mistakes: ``none``, ``byte_reverse`` (bytes of a shot reversed), ``swap16``
    (adjacent byte pairs swapped, a 16-bit endianness mix-up), ``bit_little`` (bits of each
    byte reversed, i.e. decoded with the wrong bit order)."""
    p = np.asarray(packed, dtype=np.uint8)
    if kind == "none":
        return p.copy()
    if kind == "byte_reverse":
        return p[:, ::-1].copy()
    if kind == "swap16":
        if p.shape[1] % 2:
            raise ValueError("swap16 needs an even number of bytes")
        return p.reshape(len(p), -1, 2)[:, :, ::-1].reshape(p.shape).copy()
    if kind == "bit_little":
        return _BITREV[p]
    raise ValueError(f"unknown packing corruption {kind!r}")


def drop_shots(shots: np.ndarray, frac: float, seed: int) -> np.ndarray:
    y = np.asarray(shots)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.permutation(len(y))[: len(y) - round(frac * len(y))])
    return y[keep]


def duplicate_shots(shots: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Repeat ``round(frac * n)`` randomly chosen shots, each copy right after its original
    (as a re-read buffer would)."""
    y = np.asarray(shots)
    rng = np.random.default_rng(seed)
    dup = np.zeros(len(y), dtype=np.int64)
    dup[rng.permutation(len(y))[: round(frac * len(y))]] = 1
    return np.repeat(y, 1 + dup, axis=0)


def plant(shots: np.ndarray, spec: dict[str, Any], seed: int, n_shots_ref: int | None = None):
    """Apply one artefact ``spec`` (``{"kind": ..., ...}``) to sampled shots."""
    rng = np.random.default_rng(seed)
    k = spec["kind"]
    if k == "none":
        return np.array(shots, copy=True)
    if k == "chain":  # several artefacts in order (e.g. a burst, then a reordering)
        out = shots
        for i, sub in enumerate(spec["steps"]):
            out = plant(out, sub, seed + 1000 * i)
        return out
    if k == "persistence":
        return apply_persistence(
            shots, persistence_kernel(spec["strength"], spec.get("mode", "sticky")), rng
        )
    if k == "drift":
        d = drift_profile(
            len(shots), spec["strength"], spec.get("mode", "ramp"), spec.get("levels", 41)
        )
        return apply_threshold(shots, d, rng)
    if k == "burst":
        return apply_burst(
            shots,
            spec["strength"],
            spec.get("start", 0),
            spec.get("mode", "random"),
            spec.get("param", 0.5),
            rng,
        )
    if k == "reorder":
        return permute_shots(shots, spec.get("mode", "random"), seed)
    if k == "packing":
        length = np.asarray(shots).shape[1]
        return W.unpack_shots(corrupt_packed(pack_shots(shots), spec.get("mode", "none")), length)
    if k == "drop":
        return drop_shots(shots, spec["strength"], seed)
    if k == "duplicate":
        return duplicate_shots(shots, spec["strength"], seed)
    raise ValueError(f"unknown artefact {k!r}")


def exact_windows(
    proc: W.LinearProcess,
    spec: dict[str, Any],
    n_shots: int,
    k_max: int,
    burn: int,
    length: int,
) -> np.ndarray | None:
    """Exact ``E[g]`` under an artefact, or ``None`` where it has no closed form here
    (reordering, drop and duplicate leave ``E[g]`` unchanged; packing permutes columns)."""
    k = spec["kind"]
    if k in ("none", "reorder", "drop", "duplicate"):
        return proc.expected_windows(k_max, burn, length, common=True)
    if k == "persistence":
        a = persistence_kernel(spec["strength"], spec.get("mode", "sticky"))
        return with_persistence(proc, a).expected_windows(k_max, burn, length, common=True)
    if k == "drift":
        d = drift_profile(
            n_shots, spec["strength"], spec.get("mode", "ramp"), spec.get("levels", 41)
        )
        return expected_drift(proc, d, k_max, burn, length)
    if k == "burst":
        sl = burst_slice(n_shots, spec["strength"], spec.get("start", 0))
        m = sl.stop - sl.start
        bp = burst_process(proc, spec.get("mode", "random"), spec.get("param", 0.5))
        return expected_mixture([proc, bp], [n_shots - m, m], k_max, burn, length)
    return None


def lag1_excess(proc: W.LinearProcess, burn: int, length: int) -> float:
    """Exact ``P(1 | previous 1) - P(1 | previous 0)``, pairs from ``burn`` to the end.

    For a sticky record on an i.i.d. source it equals ``eps``.
    """
    b = proc.block_probs(range(burn, length - 1), 2)  # 00, 01, 10, 11
    return float(b[3] / (b[2] + b[3]) - b[1] / (b[0] + b[1]))


def lag1_excess_data(shots: np.ndarray, burn: int) -> float:
    """Data form of ``lag1_excess``: pairs ``(t, t + 1)`` with ``t >= burn``."""
    y = np.asarray(shots)[:, burn:]
    prev, cur = y[:, :-1].ravel(), y[:, 1:].ravel()
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())


# -- the pipeline and its guard rails --


def witness_summary(
    shots: np.ndarray,
    k_max: int = 7,
    burn: int = 16,
    offsets: Sequence[int] = (0, 1, 2, 3),
    n_blocks: int = 20,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """The Stage 0 pipeline: common-start windows, ``w(j)``, contiguous-block SE (primary),
    random-block SE and whole-shot bootstrap SE."""
    per = W.per_shot_windows(shots, k_max, burn, common=True)
    g = per.mean(0)
    rng = np.random.default_rng(seed)
    perm = per[rng.permutation(len(per))]
    idx = W.bootstrap_indices(len(per), n_boot, seed + 1)
    bg = W.bootstrap_g(per, idx)
    out: dict[str, Any] = {"n": len(per), "g": g.tolist()}
    for j in offsets:
        w = W.witness(g, j)
        se_b = W.batch_se(per, j, n_blocks)
        se_r = W.batch_se(perm, j, n_blocks)
        se_boot = float(np.std(W.witness(bg, j), ddof=1))
        out[f"w{j}"] = float(w)
        out[f"se_block{j}"] = se_b
        out[f"se_random{j}"] = se_r
        out[f"se_boot{j}"] = se_boot
        out[f"z_block{j}"] = float(w / se_b) if se_b > 0 else math.nan
    return out


def _wilson_hilferty_sf(chi2: float, dof: int) -> float:
    """Upper tail of chi^2 (Wilson-Hilferty normal approximation; numpy-only)."""
    if dof <= 0:
        return math.nan
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _dispersion(x: np.ndarray, se: np.ndarray) -> tuple[float, int, float]:
    """``chi^2`` of block values ``x`` about their inverse-variance mean."""
    w = 1 / np.maximum(se, 1e-300) ** 2
    m = np.sum(w * x) / np.sum(w)
    chi2 = float(np.sum(w * (x - m) ** 2))
    dof = len(x) - 1
    return chi2, dof, _wilson_hilferty_sf(chi2, dof)


def duplicate_rows(shots: np.ndarray) -> int:
    """Number of shots identical to an earlier shot."""
    packed = pack_shots(shots)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    return int(np.sum(counts - 1))


def expected_duplicates_iid(shots: np.ndarray) -> float:
    """Expected duplicate pairs if positions were independent with the observed marginals (a
    rough scale: for 64 symbols it is ~1e-10, so any duplicate is a red flag)."""
    p = np.asarray(shots).mean(0)
    coll = float(np.prod(p**2 + (1 - p) ** 2))
    n = len(shots)
    return n * (n - 1) / 2 * coll


def guard_rails(
    shots: np.ndarray,
    k_max: int = 7,
    burn: int = 16,
    n_blocks: int = 20,
    seed: int = 0,
    marginals: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Checks every Stage 3 analysis should run on a shot array (returned order).

    * ``values``: the array is 0/1 and rectangular;
    * ``duplicates``: shots identical to an earlier one (a re-read buffer, a resubmitted
      chunk); expected ~0 for 64-symbol shots;
    * block dispersion of ``P(1)`` (assignment or threshold drift across shots), of the
      lag-1 repeat rate ``P(r_t = r_{t-1})`` (persistence that changes across shots), and of
      ``g(1) - g(3)`` (#18's episode statistic), each as ``chi^2`` over contiguous blocks
      with within-block shot-level SEs;
    * ``order_inflation``: contiguous-block SE over random-block SE of ``w(0)`` and of P(1)
      (about 1 for exchangeable shots; its own sampling error is ~``1/sqrt(2(B-1))``);
    * ``positional_chi2``: the *shape* of the per-position P(1) against model ``marginals``
      (a fitted constant offset removed, since that is model mismatch), which a bit-order or
      packing error permutes; if given.

    ``flags`` lists the checks that fail their ``thresholds`` (p-value ``alpha``, inflation
    ``max_inflation``, ``max_duplicates``). The dispersion tests are the primary drift checks;
    the inflation ratios are a ratio of two noisy SDs (each ~16% at 20 blocks), so their
    threshold is 2, and ``w0``'s is unreliable when the Hankel matrix is near rank 1 (an
    i.i.d. source), where ``w`` is second order in the noise.
    """
    th = {"alpha": 1e-3, "max_inflation": 2.0, "max_duplicates": 0}
    th.update(thresholds or {})
    y = np.asarray(shots)
    out: dict[str, Any] = {"n": len(y), "length": int(y.shape[1])}
    out["values_ok"] = bool(np.isin(y, (0, 1)).all())
    out["duplicates"] = duplicate_rows(y)
    out["expected_duplicates_iid"] = expected_duplicates_iid(y)
    body = y[:, burn:].astype(np.float64)
    p_shot = body.mean(1)
    rep_shot = (y[:, burn:] == y[:, burn - 1 : -1]).mean(1) if burn > 0 else None
    per = W.per_shot_windows(y, k_max, burn, common=True)
    ep_shot = per[:, 1] - per[:, 3]
    blocks = np.array_split(np.arange(len(y)), n_blocks)

    def block_stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.array([v[b].mean() for b in blocks])
        s = np.array([v[b].std(ddof=1) / math.sqrt(len(b)) for b in blocks])
        return m, s

    for name, v in (("p1", p_shot), ("repeat", rep_shot), ("g1_minus_g3", ep_shot)):
        if v is None:
            continue
        m, s = block_stats(v)
        chi2, dof, p = _dispersion(m, s)
        out[f"block_{name}"] = m.tolist()
        out[f"dispersion_{name}"] = {"chi2": chi2, "dof": dof, "p": p}
    perm = np.random.default_rng(seed).permutation(len(y))
    se_c = W.batch_se(per, 0, n_blocks)
    se_r = W.batch_se(per[perm], 0, n_blocks)
    pb = np.array([p_shot[b].mean() for b in blocks])
    pr = np.array([p_shot[perm][b].mean() for b in np.array_split(np.arange(len(y)), n_blocks)])
    out["order_inflation"] = {
        "w0": se_c / se_r if se_r > 0 else math.nan,
        "p1": float(pb.std(ddof=1) / pr.std(ddof=1)) if pr.std(ddof=1) > 0 else math.nan,
        "expected_spread": 1 / math.sqrt(2 * (n_blocks - 1)),
    }
    if marginals is not None:
        # shape only: a constant offset of P(1) is model mismatch, not a decoding error
        obs = y.mean(0)
        exp = np.asarray(marginals, dtype=np.float64)
        var = exp * (1 - exp) / len(y)
        d = obs - exp
        off = np.sum(d / var) / np.sum(1 / var)
        out["positional_chi2"] = {
            "chi2": float(np.sum((d - off) ** 2 / var)),
            "dof": len(exp) - 1,
            "offset": float(off),
        }
    flags = []
    if not out["values_ok"]:
        flags.append("values")
    if out["duplicates"] > th["max_duplicates"]:
        flags.append("duplicates")
    for name in ("p1", "repeat", "g1_minus_g3"):
        d = out.get(f"dispersion_{name}")
        if d is not None and d["p"] < th["alpha"]:
            flags.append(f"dispersion_{name}")
    for name, v in out["order_inflation"].items():
        if name != "expected_spread" and v > th["max_inflation"]:
            flags.append(f"order_inflation_{name}")
    if marginals is not None:
        pc = out["positional_chi2"]
        if _wilson_hilferty_sf(pc["chi2"], pc["dof"]) < th["alpha"]:
            flags.append("positional_chi2")
    out["flags"] = flags
    return out


# -- strength needed --


def strength_for(fn, target: float, lo: float, hi: float, n_grid: int = 41, tol: float = 1e-7):
    """Smallest ``s`` in ``[lo, hi]`` with ``fn(s) >= target`` (grid scan, then bisection), or
    ``None`` if none on the grid. Returns ``(s, max_on_grid)``."""
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([fn(float(s)) for s in grid])
    hit = np.nonzero(vals >= target)[0]
    if len(hit) == 0:
        return None, float(np.nanmax(vals))
    i = int(hit[0])
    if i == 0:
        return float(grid[0]), float(np.nanmax(vals))
    a, b = float(grid[i - 1]), float(grid[i])
    while b - a > tol:
        m = 0.5 * (a + b)
        if fn(m) >= target:
            b = m
        else:
            a = m
    return b, float(np.nanmax(vals))


# -- the memoryless controls D1, D2, D3 --


def rotation_angle(p1: float) -> float:
    """``theta`` with ``sin^2(theta / 2) = p1`` (an ``rx(theta)`` from ``|0>``)."""
    return 2 * math.asin(math.sqrt(min(max(p1, 0.0), 1.0)))


def qubit_control(
    p_rot: float,
    reset: bool,
    e01: float = 0.0,
    e10: float = 0.0,
    reset_e01: float | None = None,
    reset_e10: float | None = None,
    decay: float = 0.0,
    leak: float = 0.0,
    seep: float = 0.0,
    read2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact classical HMM of a qubit rotated and measured every step (D1, D2).

    Hidden state: the true level after the step's measurement, ``m in {0, 1, 2}``. One step:

    1. idle in the post-measurement state: ``1 -> 0`` with ``decay`` (T1), ``2 -> 1`` with
       ``seep``;
    2. if ``reset``: IBM's reset, a measurement then ``X`` if it read 1. The reset reading
       uses ``reset_e01``/``reset_e10`` (default: the readout's), and level 2 reads 1 with
       ``read2`` (the ``X`` acts on 0/1 only, so 2 stays 2). A failed reset therefore
       depends on the previous outcome;
    3. a rotation with ``|<1|R|0>|^2 = |<0|R|1>|^2 = p_rot`` (level 2 untouched);
    4. a QND measurement with assignment errors ``e01``, ``e10``; level 2 reads 1 with
       ``read2``; a ``1`` leaks to ``2`` with ``leak`` during it (recorded as 1 first).

    Returns ``(labelled, initial)`` with ``L[r, m, m'] = P(r, m' | m)`` and ``m = 0`` at the
    start (``init_qubits``). Without leakage level 2 is unreachable: a 2-state HMM, ``w = 0``.
    """
    re01 = e01 if reset_e01 is None else reset_e01
    re10 = e10 if reset_e10 is None else reset_e10
    idle = np.array([[1, 0, 0], [decay, 1 - decay, 0], [0, seep, 1 - seep]], dtype=np.float64)
    if reset:
        # rows: level before, columns: level after. Level 0 misread as 1 (re01) -> X -> 1;
        # level 1 read as 1 (1 - re10) -> X -> 0, misread as 0 -> stays 1; level 2 stays.
        rs = np.array([[1 - re01, re01, 0], [1 - re10, re10, 0], [0, 0, 1]], dtype=np.float64)
    else:
        rs = np.eye(3)
    rot = np.array([[1 - p_rot, p_rot, 0], [p_rot, 1 - p_rot, 0], [0, 0, 1]], dtype=np.float64)
    pre = idle @ rs @ rot  # level just before the measurement
    lab = np.zeros((2, 3, 3))
    assign = {0: (1 - e01, e01), 1: (e10, 1 - e10), 2: (1 - read2, read2)}
    for m in range(3):
        for lv in range(3):
            p = pre[m, lv]
            if p == 0:
                continue
            for r in (0, 1):
                pr = p * assign[lv][r]
                if lv == 1:
                    lab[r, m, 1] += pr * (1 - leak)
                    lab[r, m, 2] += pr * leak
                else:
                    lab[r, m, lv] += pr
    return lab, np.array([1.0, 0.0, 0.0])


def product_and(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]):
    """Two independent HMMs recorded together, analysed as ``y = r_a AND r_b`` (D3)."""
    la, ia = a
    lb, ib = b
    da, db = la.shape[1], lb.shape[1]
    lab = np.zeros((2, da * db, da * db))
    for ra in (0, 1):
        for rb in (0, 1):
            lab[ra & rb] += np.kron(la[ra], lb[rb])
    return lab, np.kron(ia, ib)


def two_chain_and_sample(a, b, n: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample both records of D3: returns ``(r_a, r_b)``."""
    ra = W.hmm_sample(a[0], a[1], n, length, seed)
    rb = W.hmm_sample(b[0], b[1], n, length, seed + 1)
    return ra, rb


def qpu_seconds(shots: int, circuit_s: float, rep_delay_s: float = 250e-6, overhead_s: float = 2.0):
    """Phase 6's cost model: ``overhead + shots (rep_delay + circuit duration)``."""
    return overhead_s + shots * (rep_delay_s + circuit_s)
