"""Phase 8 Stage 0A (#29): the independent re-analysis path and its provenance freeze."""

from __future__ import annotations

import glob
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from qmem import drift as D
from qmem import reanalysis as R
from qmem import witness as W

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "experiments" / "configs" / "phase8_stage0.toml"
PROV = ROOT / "results" / "summaries" / "phase8_stage0" / "provenance.json"


def brute_force_g(shots: np.ndarray, k_max: int, burn: int) -> np.ndarray:
    """Per-shot ``1 0^k 1`` frequencies by matching strings at every common start."""
    length = shots.shape[1]
    starts = range(burn, length - k_max - 1)
    out = np.zeros((len(shots), k_max + 1))
    for i, row in enumerate(shots):
        s = "".join(str(int(v)) for v in row)
        for k in range(k_max + 1):
            pat = "1" + "0" * k + "1"
            out[i, k] = sum(s.startswith(pat, t) for t in starts) / len(starts)
    return out


def periodic(period: int, length: int, phase: int = 0) -> np.ndarray:
    return ((np.arange(length) + phase) % period == 0).astype(np.uint8)


# -- the counter -------------------------------------------------------------------------------


def test_counter_matches_brute_force_and_qmem_witness():
    rng = np.random.default_rng(1)
    shots = (rng.random((300, 64)) < 0.4).astype(np.uint8)
    mine = R.window_freqs(shots, 7, 16)
    assert np.array_equal(mine, brute_force_g(shots, 7, 16))
    assert np.array_equal(mine, W.per_shot_windows(shots, 7, 16, common=True))
    g = mine.mean(0)
    for j in range(4):
        assert R.wit(g, j) == pytest.approx(W.witness(g, j), rel=1e-12, abs=1e-18)


def test_counter_analytic_period_three():
    """Every shot ``100100...``: only g(2) is non-zero, H_0 is anti-diagonal, w(0) = -1."""
    shots = np.tile(periodic(3, 64), (50, 1))
    g = R.window_freqs(shots, 7, 16).mean(0)
    # starts 16..55 holding a one: 18, 21, ..., 54 -> 13 of 40
    assert g[2] == pytest.approx(13 / 40)
    assert np.count_nonzero(g) == 1
    assert R.wit(g, 0) == pytest.approx(-1.0)
    assert W.witness(g, 0) == pytest.approx(-1.0)


def test_counter_analytic_mixture_of_periods():
    """Half the shots period 2 (g(1) = a), half period 3 (g(2) = b): det H_0 = -b^3, so
    w(0) = -b^3 / max(a, b)^3."""
    shots = np.vstack([np.tile(periodic(2, 64), (40, 1)), np.tile(periodic(3, 64), (40, 1))])
    a = sum(1 for t in range(16, 56) if t % 2 == 0) / 40 / 2
    b = 13 / 40 / 2
    g = R.window_freqs(shots, 7, 16).mean(0)
    assert g[1] == pytest.approx(a) and g[2] == pytest.approx(b)
    assert R.wit(g, 0) == pytest.approx(-(b**3) / max(a, b) ** 3)
    assert R.wit(g, 0) == pytest.approx(W.witness(g, 0))


def test_counter_bernoulli_rank_one():
    """i.i.d. Bernoulli(p): E g(k) = p^2 (1-p)^k (rank 1), so w(0) = 0 within its SE."""
    p = 0.4
    rng = np.random.default_rng(2)
    shots = (rng.random((20000, 64)) < p).astype(np.uint8)
    per = R.window_freqs(shots, 7, 16)
    g = per.mean(0)
    exact = p**2 * (1 - p) ** np.arange(8)
    se_g = per.std(0, ddof=1) / math.sqrt(len(per))
    assert np.all(np.abs(g - exact) < 5 * se_g)
    assert abs(R.wit(g, 0)) < 5 * D.block_se(per, 0, 1)
    assert R.wit(exact, 0) == pytest.approx(0.0, abs=1e-12)


def test_unpack_and_keys():
    rng = np.random.default_rng(3)
    y = (rng.random((10, 64)) < 0.5).astype(np.uint8)
    packed = np.packbits(y, axis=1)
    assert np.array_equal(R.unpack(packed, 64), y)
    assert np.array_equal(R.unpack(packed, 64), W.unpack_shots(packed, 64))
    assert R.parse_key("j1_p004_A2_c0__m") == {
        "job": 1,
        "pub": 4,
        "arm": "A2",
        "instance": None,
        "chunk": 0,
        "reg": "m",
    }
    k = R.parse_key("j3_p010_A2s_spec_c2__y")
    assert k["arm"] == "A2s_spec" and k["chunk"] == 2
    k = R.parse_key("j0_p000_A3_i011_c0__y")
    assert k["arm"] == "A3" and k["instance"] == 11
    assert R.parse_key("length") is None


def test_det3_and_lag1():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(20, 3, 3))
    assert np.allclose(R.det3(h), np.linalg.det(h))
    shots = (rng.random((200, 64)) < 0.3).astype(np.uint8)
    from qmem import followup as F

    c = F.lag1_counts(shots, 16).sum(0)
    assert R.lag1_excess(shots, 16) == pytest.approx(c[1] / (c[0] + c[1]) - c[3] / (c[2] + c[3]))


def test_resampling_matches_qmem_drift():
    """Same draws as ``qmem.drift.block_bootstrap_indices`` (the definition, written again)."""
    assert np.array_equal(
        R.circular_blocks(1000, 37, 5, 9), D.block_bootstrap_indices(1000, 37, 5, 9)
    )
    segs = [np.arange(0, 600), np.arange(600, 1000)]
    assert np.array_equal(
        R.stratified_blocks(segs, 50, 4, 11, cap=False), D.stratified_block_indices(segs, 50, 4, 11)
    )


# -- the statistics ----------------------------------------------------------------------------


def test_tost_hand_case():
    """R = 1.0, SE 0.05, margin +-10%: each bound at z = 2, p = 1 - Phi(2) = 0.02275."""
    t = R.tost(1.0, 0.05, 0.9, 1.1, 0.05)
    assert t["p_lower"] == pytest.approx(0.0227501319, rel=1e-8)
    assert t["p_upper"] == pytest.approx(0.0227501319, rel=1e-8)
    assert t["equivalent"]
    # off-centre: R = 1.06, SE 0.03, margin 10%: upper z = -4/3, p = Phi(-1.3333) = 0.09121
    t = R.tost(1.06, 0.03, 0.9, 1.1, 0.05)
    assert t["p_upper"] == pytest.approx(0.0912112197, rel=1e-7)
    assert not t["equivalent"]
    # power at the centre: 2 Phi(delta / se - z) - 1
    z = R.norm_ppf(0.95)
    assert R.tost_power(1.0, 0.05, 0.9, 1.1) == pytest.approx(2 * R.norm_cdf(2 - z) - 1)


def test_fieller_against_brute_force():
    for a, sa, b, sb in [
        (-2.04e-4, 1.41e-4, -5.15e-3, 1.27e-4),
        (3.0, 1.0, 5.0, 1.5),
        (-1, 0.4, 2, 0.5),
    ]:
        lo, hi = R.fieller(a, sa, b, sb)
        grid = np.linspace(lo - 1.0 * (hi - lo), hi + 1.0 * (hi - lo), 400001)
        ok = np.abs(a - grid * b) <= R.Z975 * np.sqrt(sa**2 + grid**2 * sb**2)
        step = grid[1] - grid[0]
        assert grid[ok].min() == pytest.approx(lo, abs=2 * step)
        assert grid[ok].max() == pytest.approx(hi, abs=2 * step)
    with pytest.raises(ValueError):
        R.fieller(1.0, 0.1, 0.1, 0.1)


def test_pool_and_chi2():
    p = R.pool([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])
    assert p["fe"] == pytest.approx(1.0) and p["q"] == pytest.approx(0.0) and p["tau2"] == 0.0
    p = R.pool([0.0, 1.0], [1.0, 1.0])
    assert p["fe"] == pytest.approx(0.5) and p["q"] == pytest.approx(0.5)
    assert p["fe_se"] == pytest.approx(math.sqrt(0.5))
    for x, k in [(0.5, 1), (3.0, 3), (20.0, 7), (164.0, 3)]:
        assert R.chi2_sf(x, k) == pytest.approx(D.chi2_sf(x, k), rel=1e-6, abs=1e-14)


def test_holm_and_vanish():
    assert R.holm_reject({"a": 0.001, "b": 0.004, "c": 0.5}, 0.01) == {
        "a": True,
        "b": True,
        "c": False,
    }
    assert R.holm_reject({"a": 0.006, "b": 0.007}, 0.01) == {"a": False, "b": False}
    b = R.vanish_bound(-2.0e-4, 1.4e-4, -5.0e-3)
    assert b == pytest.approx((2.0e-4 + R.Z975 * 1.4e-4) / 5.0e-3)
    mc = R.vanish_power_mc(0.03, -5e-3, 1e-12, 1.4e-4, 0.1, 5.0, 400000, 1)
    assert mc == pytest.approx(R.vanish_power(0.03 * 5e-3, 1.4e-4, 5e-4), abs=0.003)


def test_shots_for():
    need = R.shots_for(lambda s: s <= 0.5, 100, 1.0)
    assert need == pytest.approx(400, rel=1e-6)


# -- provenance ----------------------------------------------------------------------------------


def test_provenance_frozen():
    """Fails if any committed Phase 7 Stage 4/4b/4c shot, QASM or calibration file changes,
    disappears, or a new one matching the globs appears."""
    frozen = json.loads(PROV.read_text())
    with open(CFG, "rb") as f:
        cfg = tomllib.load(f)
    now = sorted(
        str(Path(p).relative_to(ROOT))
        for g in cfg["provenance"]["globs"]
        for p in glob.glob(str(ROOT / g))
        if Path(p).is_file()
    )
    assert now == sorted(frozen["files"])
    for rel, row in frozen["files"].items():
        assert R.sha256(ROOT / rel) == row["sha256"], rel
    ids = {j["job_id"] for j in frozen["jobs"]}
    assert {
        "dborq95t7fqc7388b3h0",
        "dboseett7fqc7388c8p0",
        "dbotns0qtr0c738dee10",
        "dbouj7tt7fqc7388fphg",
        "dbocqblt7fqc73cmtd6g",
    } <= ids
