"""Tests of the independent Hankel-witness estimator (``qmem.witness``, Phase 7 audit #18)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from qmem import witness as W

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "summaries" / "phase6"


def naive_counts(shots: np.ndarray, k_max: int, burn: int) -> np.ndarray:
    """Brute-force string search for ``1 0^k 1`` at starts ``t >= burn``."""
    out = np.zeros((len(shots), k_max + 1), dtype=np.int64)
    for i, row in enumerate(shots):
        s = "".join(map(str, row))
        for k in range(k_max + 1):
            pat = "1" + "0" * k + "1"
            out[i, k] = sum(1 for t in range(burn, len(s)) if s.startswith(pat, t))
    return out


def period3() -> W.LinearProcess:
    """Deterministic 0 -> 1 -> 2 -> 0 cycle emitting 1 on leaving state 0, uniform phase."""
    lab = np.zeros((2, 3, 3))
    lab[1, 0, 1] = lab[0, 1, 2] = lab[0, 2, 0] = 1.0
    return W.LinearProcess.hmm(lab, np.full(3, 1 / 3))


def qubit_instrument(theta: float, phi: float, eta: float) -> W.LinearProcess:
    """Rotate, then a weak Z measurement of strength ``eta`` (two Kraus operators)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    u = np.array([[c, -np.exp(1j * phi) * s], [np.exp(-1j * phi) * s, c]])
    m0 = np.diag([np.sqrt((1 + eta) / 2), np.sqrt((1 - eta) / 2)])
    m1 = np.diag([np.sqrt((1 - eta) / 2), np.sqrt((1 + eta) / 2)])
    sup = np.stack([np.kron((k @ u).conj(), k @ u) for k in (m0, m1)])
    return W.LinearProcess.instrument(sup, np.diag([1.0, 0.0]))


# -- decoding -----------------------------------------------------------------------------------


def test_unpack_inverts_packbits_and_detects_flips():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, size=(50, 64), dtype=np.uint8)
    packed = np.packbits(arr, axis=1)  # exactly as phase6.save_summary
    assert np.array_equal(W.unpack_shots(packed, 64), arr)
    assert np.array_equal(W.unpack_shots(packed, 64), np.unpackbits(packed, axis=1))
    little = np.packbits(arr, axis=1, bitorder="little")
    assert not np.array_equal(W.unpack_shots(little, 64), arr)
    assert not np.array_equal(W.unpack_shots(packed, 64)[:, ::-1], arr)


def test_decoding_pins_position_zero_and_time_asymmetry():
    """A process whose first symbol is always 1 and that then settles to P(1) = 0.2.

    Column 0 must decode to all ones and the column marginals must follow the transient
    forwards; a reversed or bit-reversed decoding fails both.
    """
    lab, _ = W.two_state_hmm((0.5, 0.95), (1.0, 0.0), stationary=False)  # starts in 'emit 1'
    p = W.LinearProcess.hmm(lab, np.array([1.0, 0.0]))
    shots = W.hmm_sample(lab, np.array([1.0, 0.0]), 4000, 64, seed=1)
    dec = W.unpack_shots(np.packbits(shots, axis=1), 64)
    assert dec[:, 0].min() == 1
    exact = p.marginals(64)
    se = np.sqrt(exact * (1 - exact) / len(dec)) + 1e-12
    assert np.max(np.abs(dec.mean(0) - exact) / se) < 5
    for bad in (dec[:, ::-1], W.unpack_shots(np.packbits(shots, axis=1, bitorder="little"), 64)):
        assert bad[:, 0].min() == 0 or np.max(np.abs(bad.mean(0) - exact) / se) > 5


@pytest.mark.skipif(not (SUMMARY / "circuit.qasm").exists(), reason="Phase 6 summary absent")
def test_phase6_circuit_measures_registers_in_step_order():
    qasm = (SUMMARY / "circuit.qasm").read_text()
    decl = re.findall(r"^bit\[1\] y(\d+);", qasm, flags=re.M)
    meas = re.findall(r"^y(\d+)\[0\] = measure \$(\d+);", qasm, flags=re.M)
    assert [int(t) for t in decl] == list(range(64))
    assert [int(t) for t, _ in meas] == list(range(64))  # y_t measured at step t, in order
    assert len({q for _, q in meas}) == 1  # always the same physical output qubit


# -- window counting ----------------------------------------------------------------------------


@pytest.mark.parametrize("burn", [0, 3, 16])
def test_window_counts_match_brute_force(burn):
    rng = np.random.default_rng(burn)
    shots = (rng.random((40, 30)) < 0.4).astype(np.uint8)
    assert np.array_equal(W.window_counts(shots, 7, burn), naive_counts(shots, 7, burn))


def test_window_counts_no_double_counting():
    # '1111': three 11's, no 101 (needs a zero); '11101': 11 at t=0,1 and 101 at t=2
    c = W.window_counts(np.array([[1, 1, 1, 1], [1, 1, 1, 0]]), 2, 0)
    assert c.tolist() == [[3, 0, 0], [2, 0, 0]]
    c = W.window_counts(np.array([[1, 1, 1, 0, 1]]), 3, 0)
    assert c.tolist() == [[2, 1, 0, 0]]
    assert W.n_starts(64, 7, 16).tolist() == [47, 46, 45, 44, 43, 42, 41, 40]


def test_wrong_burn_is_detected():
    """Ones everywhere before position 16, then the period-3 pattern '100'."""
    shots = np.ones((3, 64), dtype=np.uint8)
    shots[:, 16:] = np.tile([1, 0, 0], 16)[:48]
    per = W.per_shot_windows(shots, 7, 16)
    assert np.allclose(per, naive_counts(shots, 7, 16) / W.n_starts(64, 7, 16))
    assert per[0, 0] == 0  # no '11' at or after 16
    assert np.isclose(W.witness(per.mean(0), 0), -1.0, atol=0.05)
    assert W.per_shot_windows(shots, 7, 15)[0, 0] > 0  # one step earlier sees a '11'


# -- the witness on exact cases -------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_two_state_hmm_gives_zero_witness(seed):
    lab, pi = W.random_hmm(2, np.random.default_rng(seed))
    p = W.LinearProcess.hmm(lab, pi)
    g = p.windows_at(p.state_after(16), 7)
    for j in range(4):
        assert abs(W.witness(g, j)) < 1e-10


def test_two_state_hmm_common_starts_zero_even_off_stationarity():
    """Averaging the state over one start set keeps rank <= 2 exactly; per-k ranges do not."""
    lab, _ = W.two_state_hmm((0.995, 0.99), (0.9, 0.1), stationary=False)
    p = W.LinearProcess.hmm(lab, np.array([1.0, 0.0]))
    common = p.expected_windows(7, 16, 64, common=True)
    per_k = p.expected_windows(7, 16, 64)
    assert max(abs(W.witness(common, j)) for j in range(4)) < 1e-10
    assert max(abs(W.witness(per_k, j)) for j in range(4)) > 1e-8


def test_period3_witness_is_minus_one():
    p = period3()
    g = p.windows_at(p.stationary(), 7)
    assert np.allclose(g, [0, 0, 1 / 3, 0, 0, 0, 0, 0])  # 1 0^k 1 needs k zeros exactly
    assert np.isclose(W.witness(g, 0), -1.0)  # det of the anti-diagonal (1/3) / (1/3)^3
    assert W.witness(g, 1) == 0.0


def renewal_lab(h: list[float]) -> np.ndarray:
    """States = zeros since the last one, capped at ``len(h) - 1``; hazard ``h[state]``."""
    d = len(h)
    lab = np.zeros((2, d, d))
    for i in range(d):
        lab[1, i, 0] = h[i]
        lab[0, i, min(i + 1, d - 1)] = 1 - h[i]
    return lab


def test_three_state_renewal_witness_matches_closed_form():
    """g(k) = r f(k+1) with f the inter-arrival law and r = 1 / E[gap]; nonzero w."""
    h = [0.2, 0.7, 0.4]
    hz = lambda m: h[min(m, 2)]  # noqa: E731
    f = [hz(k) * np.prod([1 - hz(m) for m in range(k)]) for k in range(200)]
    rate = 1 / sum((k + 1) * fk for k, fk in enumerate(f))
    closed = rate * np.array(f[:8])
    p = W.LinearProcess.hmm(renewal_lab(h), np.array([1.0, 0, 0]))
    g = p.windows_at(p.stationary(), 7)
    assert np.allclose(g, closed, rtol=1e-12)
    for j in range(4):
        hj = closed[j + np.add.outer(np.arange(3), np.arange(3))]
        expect = np.linalg.det(hj) / np.abs(hj).max() ** 3
        assert np.isclose(W.witness(g, j), expect, rtol=1e-10, atol=1e-15)
    assert abs(W.witness(g, 0)) > 1e-3  # j >= 1 sees only the geometric tail (rank 2)
    # h(2) = h(1) would make it a 2-state renewal: w = 0 again
    p2 = W.LinearProcess.hmm(renewal_lab([0.2, 0.7, 0.7]), np.array([1.0, 0, 0]))
    assert abs(W.witness(p2.windows_at(p2.stationary(), 7), 0)) < 1e-12


def test_witness_scale_invariant_and_rank_bounds():
    q = qubit_instrument(1.1, 0.4, 0.7)
    g = q.windows_at(q.stationary(), 12)
    assert abs(W.witness(g, 0)) > 1e-4  # a qubit can beat every 2-state HMM
    assert np.isclose(W.witness(3.7 * g, 0), W.witness(g, 0))
    assert abs(np.linalg.det(W.hankel(g, 0, 5))) < 1e-14  # rank <= 4 = dim vec(rho)
    lab, pi = W.random_hmm(3, np.random.default_rng(3))
    p3 = W.LinearProcess.hmm(lab, pi)
    g3 = p3.windows_at(p3.stationary(), 12)
    assert abs(np.linalg.det(W.hankel(g3, 0, 4))) < 1e-15


def test_expected_windows_match_simulation():
    q = qubit_instrument(1.1, 0.4, 0.7)
    shots = q.sample(20000, 40, seed=3)
    per = W.per_shot_windows(shots, 7, 16)
    exact = q.expected_windows(7, 16, 40)
    se = per.std(0) / np.sqrt(len(per))
    assert np.max(np.abs(per.mean(0) - exact) / se) < 5
    assert np.max(np.abs(q.marginals(40) - shots.mean(0))) < 0.02


def test_witness_gradient_matches_finite_differences():
    q = qubit_instrument(0.9, 0.2, 0.6)
    g = q.windows_at(q.stationary(), 7)
    for j in range(4):
        num = np.zeros_like(g)
        for i in range(len(g)):
            e = np.zeros_like(g)
            e[i] = 1e-7
            num[i] = (W.witness(g + e, j) - W.witness(g - e, j)) / 2e-7
        assert np.allclose(W.witness_grad(g, j), num, rtol=1e-5, atol=1e-8)


# -- uncertainty ----------------------------------------------------------------------------------


def test_bootstrap_resamples_whole_shots():
    """Shots differ in their own P(1) (a per-shot latent): windows within a shot correlate.

    The shot-bootstrap SE of g(0) must match the spread over independent datasets; an
    implementation that resampled windows (or symbols) independently would be ~2x too small.
    """
    n, length = 400, 64

    def dataset(seed):
        rng = np.random.default_rng(seed)
        p = np.where(rng.random(n) < 0.5, 0.2, 0.8)
        return (rng.random((n, length)) < p[:, None]).astype(np.uint8)

    reps = np.array([W.per_shot_windows(dataset(s), 3, 16).mean(0)[0] for s in range(300)])
    per = W.per_shot_windows(dataset(1000), 3, 16)
    bg = W.bootstrap_g(per, W.bootstrap_indices(n, 1000, seed=0))
    se_boot = bg[:, 0].std(ddof=1)
    assert abs(se_boot / reps.std(ddof=1) - 1) < 0.2
    # the window-level (iid) SE that a wrong bootstrap would estimate
    counts = W.window_counts(dataset(1000), 3, 16)[:, 0]
    m = W.n_starts(length, 3, 16)[0]
    p_hat = counts.sum() / (n * m)
    se_windows = np.sqrt(p_hat * (1 - p_hat) / (n * m))
    assert se_boot > 2 * se_windows


def test_alternative_ses_agree_on_a_qubit_process():
    q = qubit_instrument(1.1, 0.4, 0.7)
    per = W.per_shot_windows(q.sample(4000, 64, seed=5), 7, 16)
    boots = W.witness(W.bootstrap_g(per, W.bootstrap_indices(len(per), 800, seed=1)), 0)
    jack = W.witness(W.jackknife_g(per), 0)
    se_b = boots.std(ddof=1)
    assert abs(W.jackknife_se(jack) / se_b - 1) < 0.15
    assert abs(W.delta_se(per, 0) / se_b - 1) < 0.15
    lo, hi = W.bca_ci(W.witness(per.mean(0), 0), boots, jack)
    assert lo < W.witness(per.mean(0), 0) < hi


def test_norm_ppf():
    for p, z in [(0.975, 1.959963985), (0.5, 0.0), (0.001, -3.090232306)]:
        assert abs(W._norm_ppf(p) - z) < 1e-6


# -- samplers and EM -------------------------------------------------------------------------------


def test_samplers_match_exact_marginals():
    lab, pi = W.random_hmm(3, np.random.default_rng(7))
    p = W.LinearProcess.hmm(lab, pi)
    exact = p.marginals(20)
    for shots in (W.hmm_sample(lab, pi, 20000, 20, 1), p.sample(20000, 20, 2)):
        assert np.max(np.abs(shots.mean(0) - exact)) < 0.015


def test_forward_backward_consistent_and_em_monotone():
    rng = np.random.default_rng(11)
    lab, pi = W.random_hmm(3, rng)
    shots = W.hmm_sample(lab, pi, 300, 32, 4)
    counts, gamma0, ll = W.hmm_forward_backward(lab, pi, shots)
    assert np.isclose(ll, -W.hmm_nll(lab, pi, shots).sum())
    assert np.isclose(counts.sum(), shots.size)
    assert np.isclose(gamma0.sum(), len(shots))
    lab0, pi0 = W.random_hmm(3, rng)
    prev = np.inf
    for iters in (1, 2, 5, 20):
        r = W.fit_em(lab0, pi0, shots, max_iters=iters, tol=-np.inf)
        nll = W.hmm_nll(r["labelled"], r["initial"], shots).sum()
        assert nll <= prev + 1e-9
        prev = nll


def test_moment_inits_are_valid_hmms():
    rng = np.random.default_rng(2)
    shots = (rng.random((100, 64)) < 0.5).astype(np.uint8)
    inits = [W.renewal_init(shots, 3), W.markov_init(shots, 2)]
    inits.append(W.split_init(*inits[0], rng))
    for lab, pi in inits:
        assert np.allclose(lab.sum(axis=(0, 2)), 1) and np.isclose(pi.sum(), 1)
        assert lab.min() > 0
