"""Phase 7 Stage 2D (#23): classical readout / processing artefacts and the D1-D3 controls.

Each artefact is checked against an analytic expectation (zero strength changes nothing;
rank arguments give w = 0 exactly; a permutation leaves pooled g unchanged), its exact form
against brute-force enumeration, and its data form against its exact form.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qmem import artefacts as A
from qmem import witness as W

K, BURN, L = 7, 16, 32


def _w(p: W.LinearProcess, j: int = 0, length: int = L) -> float:
    return W.witness(p.expected_windows(K, BURN, length, common=True), j)


def _source2():
    return W.LinearProcess.hmm(*W.two_state_hmm((0.9, 0.7), (0.3, 0.8)))


def _source3():
    lab, ini = W.random_hmm(3, np.random.default_rng(4))
    return W.LinearProcess.hmm(lab, ini)


def _seq_prob(p: W.LinearProcess, seq) -> float:
    s = p.init
    for y in seq:
        s = s @ p.ops[y]
    return float((s @ p.final).real)


def _sample_g(shots):
    per = W.per_shot_windows(shots, K, BURN, common=True)
    return per.mean(0), per.std(0, ddof=1) / math.sqrt(len(per))


# -- persistence ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sticky", "after_one", "after_zero"])
def test_persistence_kernel_is_stochastic_and_zero_is_identity(kind):
    a = A.persistence_kernel(0.3, kind)
    assert np.allclose(a.sum(-1), 1.0)
    a0 = A.persistence_kernel(0.0, kind)
    for y in (0, 1):
        for rp in (0, 1):
            assert a0[y, rp, y] == 1.0
    src = _source3()
    g0 = src.expected_windows(K, BURN, L, common=True)
    g1 = A.with_persistence(src, a0).expected_windows(K, BURN, L, common=True)
    assert np.allclose(g0, g1, atol=1e-15)
    shots = W.hmm_sample(src.ops, src.init, 50, L, seed=1)
    assert np.array_equal(A.apply_persistence(shots, a0, np.random.default_rng(0)), shots)


def test_persistence_matches_brute_force():
    """The 2D-dimensional composite against a sum over hidden outcome paths."""
    src = _source2()
    a = A.persistence_kernel(0.2, "sticky")
    comp = A.with_persistence(src, a)
    n = 5
    for rec in itertools.product((0, 1), repeat=n):
        tot = 0.0
        for ys in itertools.product((0, 1), repeat=n):
            p, prev = _seq_prob(src, ys), 0
            for y, r in zip(ys, rec, strict=True):
                p *= a[y, prev, r]
                prev = r
            tot += p
        assert _seq_prob(comp, rec) == pytest.approx(tot, abs=1e-14)


@pytest.mark.parametrize("kind", ["sticky", "after_one"])
def test_persistence_on_iid_is_two_state_so_w_is_zero(kind):
    """An i.i.d. source with a previous-record-dependent assignment is a Markov chain."""
    for eps in (0.01, 0.2, 0.7):
        p = A.with_persistence(A.iid_process(0.58), A.persistence_kernel(eps, kind))
        for j in range(4):
            assert abs(_w(p, j)) < 1e-14
    # ... but on a 2-state source it raises the dimension: w != 0
    p = A.with_persistence(_source2(), A.persistence_kernel(0.2, "sticky"))
    assert abs(_w(p)) > 1e-6


def test_persistence_data_form_matches_exact():
    src = _source2()
    a = A.persistence_kernel(0.3, "sticky")
    y = W.hmm_sample(src.ops, src.init, 6000, L, seed=3)
    r = A.apply_persistence(y, a, np.random.default_rng(4))
    g, se = _sample_g(r)
    ex = A.with_persistence(src, a).expected_windows(K, BURN, L, common=True)
    assert np.all(np.abs(g - ex) < 5 * se + 1e-12)
    # the sticky lag-1 excess on an i.i.d. source is exactly eps
    y = (np.random.default_rng(5).random((6000, L)) < 0.58).astype(np.uint8)
    r = A.apply_persistence(y, a, np.random.default_rng(6))
    prev, cur = r[:, :-1].ravel(), r[:, 1:].ravel()
    excess = cur[prev == 1].mean() - cur[prev == 0].mean()
    assert excess == pytest.approx(0.3, abs=0.02)


def test_readout_on_instrument_is_linear():
    """A complex (qubit-instrument) process passes through with_readout and persistence."""
    th = 0.7
    k0 = np.array([[1, 0], [0, 0]], dtype=complex)
    u = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=complex)
    # measure in the computational basis after a rotation, keep the post-measurement state
    sup = np.stack([np.kron((k @ u).conj(), k @ u) for k in (k0, np.eye(2) - k0)])
    p = W.LinearProcess.instrument(sup, np.diag([1.0, 0.0]).astype(complex))
    m = A.readout_matrix(0.0, 0.0)
    assert np.allclose(
        A.with_readout(p, m).expected_windows(K, BURN, L, True),
        p.expected_windows(K, BURN, L, True),
    )
    # the measured, unreset qubit is the 2-state chain with flip probability sin^2(theta)
    chain = W.LinearProcess.hmm(*A.qubit_control(math.sin(th) ** 2, False))
    assert np.allclose(
        p.expected_windows(K, BURN, L, True), chain.expected_windows(K, BURN, L, True)
    )
    assert abs(_w(p)) < 1e-14
    # persistence on it raises the dimension to 4: the same nonzero w both ways
    a = A.persistence_kernel(0.1)
    wq, wc = _w(A.with_persistence(p, a)), _w(A.with_persistence(chain, a))
    assert abs(wq) > 1e-6
    assert wq == pytest.approx(wc, rel=1e-9)


# -- drift --------------------------------------------------------------------------------------


def test_drift_zero_amplitude_changes_nothing():
    src = _source3()
    d = A.drift_profile(1000, 0.0, "ramp")
    assert np.allclose(
        A.expected_drift(src, d, K, BURN, L), src.expected_windows(K, BURN, L, True), atol=1e-15
    )
    y = W.hmm_sample(src.ops, src.init, 100, L, seed=0)
    assert np.array_equal(A.apply_threshold(y, d[:100], np.random.default_rng(1)), y)


def test_drift_on_iid_is_a_moment_sequence():
    """g(k) = E[p^2 (1-p)^k]: two levels give rank 2 (w = 0); more give w > 0 (PSD Hankel)."""
    src = A.iid_process(0.58)
    two = A.expected_drift(src, A.drift_profile(1000, 0.2, "step"), K, BURN, L)
    for j in range(4):
        assert abs(W.witness(two, j)) < 1e-15
    ramp = A.expected_drift(src, A.drift_profile(1000, 0.3, "ramp", levels=21), K, BURN, L)
    for j in range(4):
        h = W.hankel(ramp, j)
        assert np.linalg.eigvalsh(h).min() > -1e-15
        assert W.witness(ramp, j) > 0
    # closed form for the mixture
    d = A.drift_profile(1000, 0.3, "ramp", levels=21)
    ps = np.array([0.58 * (1 - max(-x, 0)) + 0.42 * max(x, 0) for x in d])
    closed = np.array([np.mean(ps**2 * (1 - ps) ** k) for k in range(K + 1)])
    assert np.allclose(ramp, closed, atol=1e-15)


def test_drift_data_form_matches_exact():
    src = _source2()
    n = 8000
    d = A.drift_profile(n, 0.2, "sine", levels=9)
    y = W.hmm_sample(src.ops, src.init, n, L, seed=7)
    g, se = _sample_g(A.apply_threshold(y, d, np.random.default_rng(8)))
    ex = A.expected_drift(src, d, K, BURN, L)
    assert np.all(np.abs(g - ex) < 5 * se + 1e-12)


# -- bursts -------------------------------------------------------------------------------------


def test_burst_analytic_cases():
    src = _source3()
    base = src.expected_windows(K, BURN, L, True)
    # zero fraction: nothing changes
    assert np.allclose(
        A.expected_burst(src, 1000, 0.0, "random", 0.5, K, BURN, L), base, atol=1e-15
    )
    # all-zero shots only scale g, and w is scale-invariant
    z = A.expected_burst(src, 1000, 0.3, "zeros", 0.0, K, BURN, L)
    assert np.allclose(z, 0.7 * base, atol=1e-15)
    for j in range(4):
        assert W.witness(z, j) == pytest.approx(W.witness(base, j), rel=1e-12)
    # an i.i.d. burst on an i.i.d. source: a two-component moment sequence, rank 2
    r = A.expected_burst(A.iid_process(0.58), 1000, 0.2, "random", 0.5, K, BURN, L)
    assert abs(W.witness(r, 0)) < 1e-15


def test_burst_data_form():
    src = _source2()
    y = W.hmm_sample(src.ops, src.init, 2000, L, seed=2)
    out = A.apply_burst(y, 0.25, 300, "zeros", 0.0, np.random.default_rng(0))
    assert np.array_equal(out[:300], y[:300]) and np.array_equal(out[800:], y[800:])
    assert not out[300:800].any()
    kept = np.concatenate([y[:300], y[800:]])
    g_out = W.per_shot_windows(out, K, BURN, common=True).mean(0)
    g_kept = W.per_shot_windows(kept, K, BURN, common=True).mean(0)
    assert W.witness(g_out, 0) == pytest.approx(W.witness(g_kept, 0), rel=1e-12)
    assert np.array_equal(A.apply_burst(y, 0.0, 300, "random", 0.5, np.random.default_rng(0)), y)


# -- reordering ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random", "reverse", "sort_p1", "interleave"])
def test_permutation_leaves_pooled_g_unchanged(kind):
    src = _source2()
    y = W.hmm_sample(src.ops, src.init, 3000, L, seed=9)
    y = A.apply_burst(y, 0.2, 1000, "random", 0.5, np.random.default_rng(1))
    p = A.permute_shots(y, kind, seed=3)
    c0 = W.window_counts(y, K, BURN, last_start=L - K - 2).sum(0)
    c1 = W.window_counts(p, K, BURN, last_start=L - K - 2).sum(0)
    assert np.array_equal(c0, c1)  # integer counts: exactly equal
    g0 = W.per_shot_windows(y, K, BURN, True).mean(0)
    g1 = W.per_shot_windows(p, K, BURN, True).mean(0)
    assert np.allclose(g0, g1, rtol=1e-13, atol=0)


def test_permutation_changes_block_se():
    """Contiguous blocks see the burst; a random order spreads it over every block."""
    src = _source2()
    y = W.hmm_sample(src.ops, src.init, 4000, L, seed=11)
    y = A.apply_burst(y, 0.2, 1500, "random", 0.5, np.random.default_rng(2))
    se = {}
    for kind in ("none", "random", "sort_p1"):
        p = y if kind == "none" else A.permute_shots(y, kind, seed=5)
        per = W.per_shot_windows(p, K, BURN, True)
        se[kind] = np.std([b[:, 0].mean() for b in np.array_split(per, 20)], ddof=1)
    assert se["none"] > 1.5 * se["random"]
    assert se["sort_p1"] > se["random"]


# -- packing ------------------------------------------------------------------------------------


def test_packing_round_trip_and_corruptions():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 64)) < 0.4).astype(np.uint8)
    pk = A.pack_shots(y)
    assert np.array_equal(W.unpack_shots(A.corrupt_packed(pk, "none"), 64), y)
    br = W.unpack_shots(A.corrupt_packed(pk, "byte_reverse"), 64)
    cols = np.concatenate([np.arange(8 * b, 8 * b + 8) for b in range(7, -1, -1)])
    assert np.array_equal(br, y[:, cols])
    bl = W.unpack_shots(A.corrupt_packed(pk, "bit_little"), 64)
    cols = np.concatenate([np.arange(8 * b + 7, 8 * b - 1, -1) for b in range(8)])
    assert np.array_equal(bl, y[:, cols])
    assert np.array_equal(
        np.unpackbits(pk, axis=1, bitorder="little"), bl
    )  # same as numpy's little bit order
    for kind in ("byte_reverse", "swap16", "bit_little"):
        twice = A.corrupt_packed(A.corrupt_packed(pk, kind), kind)
        assert np.array_equal(twice, pk)


def test_positional_guard_rail_catches_packing():
    """A process with a start-up transient: the wrong byte order fails the positional test."""
    src = W.LinearProcess.hmm(*W.two_state_hmm((0.97, 0.95), (0.2, 0.8), stationary=False))
    y = W.hmm_sample(src.ops, src.init, 20000, 64, seed=1)
    marg = src.marginals(64)
    ok = A.guard_rails(y, marginals=marg)
    bad = A.guard_rails(A.plant(y, {"kind": "packing", "mode": "byte_reverse"}, 0), marginals=marg)
    assert "positional_chi2" not in ok["flags"]
    assert "positional_chi2" in bad["flags"]


# -- drop and duplicate ------------------------------------------------------------------------


def test_drop_and_duplicate():
    src = _source2()
    y = W.hmm_sample(src.ops, src.init, 1000, L, seed=4)
    assert np.array_equal(A.drop_shots(y, 0.0, 1), y)
    assert np.array_equal(A.duplicate_shots(y, 0.0, 1), y)
    d = A.drop_shots(y, 0.3, 1)
    assert len(d) == 700
    dd = A.duplicate_shots(y, 1.0, 1)
    assert len(dd) == 2000 and np.array_equal(dd[0::2], y) and np.array_equal(dd[1::2], y)
    g0 = W.per_shot_windows(y, K, BURN, True).mean(0)
    g1 = W.per_shot_windows(dd, K, BURN, True).mean(0)
    assert np.allclose(g0, g1, rtol=1e-13, atol=0)
    assert A.duplicate_rows(dd) == A.duplicate_rows(y) + 1000


# -- guard rails ---------------------------------------------------------------------------------


def test_guard_rails_quiet_on_clean_and_loud_on_artefacts():
    src = W.LinearProcess.hmm(*W.two_state_hmm((0.9, 0.7), (0.3, 0.8)))
    y = W.hmm_sample(src.ops, src.init, 20000, 64, seed=2)
    clean = A.guard_rails(y, seed=1)
    assert clean["values_ok"] and clean["duplicates"] == 0
    assert clean["flags"] == []
    burst = A.guard_rails(A.plant(y, {"kind": "burst", "strength": 0.1, "start": 5000}, 3))
    assert "dispersion_p1" in burst["flags"]
    dup = A.guard_rails(A.plant(y, {"kind": "duplicate", "strength": 0.01}, 3))
    assert "duplicates" in dup["flags"]
    drift = A.guard_rails(A.plant(y, {"kind": "drift", "strength": 0.05, "mode": "step"}, 3))
    assert "dispersion_p1" in drift["flags"]


def test_strength_for():
    s, top = A.strength_for(lambda x: x**2, 0.25, 0.0, 1.0)
    assert s == pytest.approx(0.5, abs=1e-6) and top == pytest.approx(1.0)
    s, top = A.strength_for(lambda x: 0.1 * x, 0.25, 0.0, 1.0)
    assert s is None and top == pytest.approx(0.1)


# -- the controls D1-D3 -------------------------------------------------------------------------


def test_qubit_control_conserves_probability():
    lab, _ = A.qubit_control(0.58, True, 0.001, 0.007, decay=0.01, leak=0.01, seep=0.04, read2=0.9)
    assert np.allclose(lab.sum(axis=(0, 2)), 1.0)
    assert lab.min() >= 0


def test_d1_noiseless_is_iid_and_d2_is_a_markov_chain():
    p = 0.58
    d1 = W.LinearProcess.hmm(*A.qubit_control(p, True))
    g = d1.expected_windows(K, BURN, L, True)
    assert np.allclose(g, [p * p * (1 - p) ** k for k in range(K + 1)], atol=1e-15)
    f = 0.3
    lab, ini = A.qubit_control(f, False)
    d2 = W.LinearProcess.hmm(lab, ini)
    # stationary P(1) = 1/2; g(0) = P(1) (1 - f); g(k) = P(1) f (1 - f)^(k-1) f
    st = W.LinearProcess(np.array([0.5, 0.5, 0.0]), d2.ops, d2.final)
    gs = st.windows_at(st.init, K)
    exp = [0.5 * (1 - f)] + [0.5 * f * (1 - f) ** (k - 1) * f for k in range(1, K + 1)]
    assert np.allclose(gs, exp, atol=1e-15)


def test_controls_w_zero_without_leakage_nonzero_with():
    kw = {"e01": 0.001, "e10": 0.0068, "decay": 0.0104}
    for reset, p in ((True, 0.58), (False, 0.3)):
        noisy = W.LinearProcess.hmm(*A.qubit_control(p, reset, **kw))
        for j in range(4):
            assert abs(_w(noisy, j, 64)) < 1e-13
        # |2> sometimes read as 0: three states, w != 0 (tiny)
        leaky = W.LinearProcess.hmm(
            *A.qubit_control(p, reset, leak=0.01, seep=0.04, read2=0.9, **kw)
        )
        assert abs(_w(leaky, 0, 64)) > 1e-9
        # |2> always read as 1: a 0 is never emitted from |2>, the 0-block has rank 2, so
        # every Hankel matrix not touching g(0) at its corner has rank <= 2
        blind = W.LinearProcess.hmm(*A.qubit_control(p, reset, leak=0.01, seep=0.04, **kw))
        assert np.linalg.matrix_rank(blind.ops[0]) == 2
        for j in (1, 2, 3):
            assert abs(_w(blind, j, 64)) < 1e-13


def test_d1_reset_failure_is_a_lag1_correlation():
    """IBM's reset (measure, conditional X) fails after a 1 with e10 and after a 0 with e01."""
    p, e01, e10 = 0.58, 0.002, 0.01
    lab, ini = A.qubit_control(p, True, reset_e01=e01, reset_e10=e10)
    pr = W.LinearProcess.hmm(lab, ini)
    # P(y_t = 1 | y_{t-1} = m): start the next step from level m
    p1_after = [
        (np.eye(3)[m] @ lab[1] @ np.ones(3)) / (np.eye(3)[m] @ lab.sum(0) @ np.ones(3))
        for m in (0, 1)
    ]
    assert p1_after[0] == pytest.approx(p + e01 * (1 - 2 * p))
    assert p1_after[1] == pytest.approx(p + e10 * (1 - 2 * p))
    assert abs(_w(pr, 0, 64)) < 1e-14


def test_product_and_matches_brute_force():
    a = A.qubit_control(0.2, False, decay=0.01)
    b = A.qubit_control(0.35, False, decay=0.02)
    ab = W.LinearProcess.hmm(*A.product_and(a, b))
    pa, pb = W.LinearProcess.hmm(*a), W.LinearProcess.hmm(*b)
    n = 4
    for rec in itertools.product((0, 1), repeat=n):
        tot = 0.0
        for ra in itertools.product((0, 1), repeat=n):
            for rb in itertools.product((0, 1), repeat=n):
                if all((x & z) == r for x, z, r in zip(ra, rb, rec, strict=True)):
                    tot += _seq_prob(pa, ra) * _seq_prob(pb, rb)
        assert _seq_prob(ab, rec) == pytest.approx(tot, abs=1e-14)
    assert abs(_w(ab, 0, 64)) > 1e-4  # a known nonzero witness: the positive control


def test_qpu_seconds_matches_phase6():
    # Phase 6's main job estimate: 20,000 shots of a 308.316 us circuit
    assert A.qpu_seconds(20000, 308.316e-6) == pytest.approx(13.16632, abs=1e-5)


def test_lag1_excess_exact_and_data():
    p = A.with_persistence(A.iid_process(0.58), A.persistence_kernel(0.05))
    assert A.lag1_excess(p, BURN, 64) == pytest.approx(0.05, abs=1e-12)
    assert A.lag1_excess(A.iid_process(0.58), BURN, 64) == pytest.approx(0.0, abs=1e-14)
    y = (np.random.default_rng(1).random((5000, 64)) < 0.58).astype(np.uint8)
    r = A.plant(y, {"kind": "persistence", "strength": 0.05}, 2)
    assert A.lag1_excess_data(r, BURN) == pytest.approx(0.05, abs=0.006)
    chained = A.plant(y, {"kind": "chain", "steps": [{"kind": "reorder", "mode": "reverse"}]}, 0)
    assert np.array_equal(chained, y[::-1])
