"""Tests for qmem.drift (Phase 7 Stage 1, #19): exact drift nulls, bootstraps, tests."""

import itertools

import numpy as np
import pytest
import torch

from qmem import drift as D
from qmem import witness as W

K_MAX, BURN, L = 7, 16, 64


def _hmm(p_stay, p_one, stationary=True):
    lab, ini = W.two_state_hmm(p_stay, p_one, stationary=stationary)
    return W.LinearProcess.hmm(lab, ini)


COMPS = [
    _hmm((0.9, 0.7), (0.3, 0.8)),
    _hmm((0.2, 0.4), (0.6, 0.1), stationary=False),
    _hmm((0.95, 0.5), (0.5, 0.9)),
]


# -- mixtures --------------------------------------------------------------------------------


def test_mixture_components_zero_and_mixture_equals_block_process():
    wts = np.array([0.5, 0.3, 0.2])
    for p in COMPS:  # every 2-state constituent: w = 0 with a common start set, any start
        g = p.expected_windows(K_MAX, BURN, L, common=True)
        for j in range(4):
            assert abs(W.witness(g, j)) < 1e-13
    mix = D.mixture(COMPS, wts)
    assert mix.ops.shape == (2, 6, 6)
    g_mix = mix.expected_windows(K_MAX, BURN, L, common=True)
    g_avg = sum(
        w * p.expected_windows(K_MAX, BURN, L, common=True) for w, p in zip(wts, COMPS, strict=True)
    )
    np.testing.assert_allclose(g_mix, g_avg, rtol=0, atol=1e-15)
    w_mix = W.witness(g_mix, 0)
    assert abs(w_mix) > 1e-6  # mixing breaks rank 2
    # brute force: sequence probabilities of the block process = weighted sum of components
    for seq in itertools.product((0, 1), repeat=6):
        p = mix.init
        for y in seq:
            p = p @ mix.ops[y]
        direct = sum(
            w * np.linalg.multi_dot([c.init, *[c.ops[y] for y in seq], c.final])
            for w, c in zip(wts, COMPS, strict=True)
        )
        assert abs(p @ mix.final - direct) < 1e-15


def test_mixture_family_torch_matches_numpy():
    fam = D.mixture_family(3, L, K_MAX, BURN)
    th = torch.randn(
        2, fam.n_params, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    st = fam.stats(th)
    for b in range(2):
        p = fam.process(th[b].numpy())
        np.testing.assert_allclose(
            st["g"][b].numpy(), p.expected_windows(K_MAX, BURN, L, common=True), atol=1e-14
        )
        np.testing.assert_allclose(st["marg"][b].numpy(), p.marginals(L), atol=1e-14)
        starts = list(range(BURN, L - 3, 4))
        np.testing.assert_allclose(st["block"][b].numpy(), p.block_probs(starts, 4), atol=1e-14)


def test_telegraph_limits_and_torch():
    a, b = COMPS[0], COMPS[2]
    regs = [(a.ops, a.init), (b.ops, b.init)]
    # no switching: the telegraph process is the mixture of its regimes
    tel = D.telegraph(regs, np.eye(2), np.array([0.6, 0.4]))
    mix = D.mixture([a, b], [0.6, 0.4])
    np.testing.assert_allclose(
        tel.expected_windows(K_MAX, BURN, L, common=True),
        mix.expected_windows(K_MAX, BURN, L, common=True),
        atol=1e-15,
    )
    # identical regimes: the telegraph process is the single HMM (w = 0)
    same = D.telegraph(
        [regs[0], regs[0]], D.switch_matrix(1e-5, 0.3, 4.85e-6), np.array([0.7, 0.3])
    )
    np.testing.assert_allclose(
        same.expected_windows(K_MAX, BURN, L, common=True),
        a.expected_windows(K_MAX, BURN, L, common=True),
        atol=1e-15,
    )
    fam = D.telegraph_family(3e-5, 4.85e-6, 567e-6, L, K_MAX, BURN)
    th = torch.randn(
        1, fam.n_params, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    st = fam.stats(th)
    p = fam.process(th[0].numpy())
    np.testing.assert_allclose(
        st["g"][0].numpy(), p.expected_windows(K_MAX, BURN, L, common=True), atol=1e-14
    )
    np.testing.assert_allclose(st["marg"][0].numpy(), p.marginals(L), atol=1e-14)


def test_switch_matrix_stationary_and_decay():
    s = D.switch_matrix(2e-5, 0.3, 5e-6)
    np.testing.assert_allclose(s.sum(1), 1)
    np.testing.assert_allclose(np.array([0.7, 0.3]) @ s, [0.7, 0.3], atol=1e-15)
    ev = sorted(np.linalg.eigvals(s).real)
    assert abs(ev[0] - np.exp(-5e-6 / 2e-5)) < 1e-14


def test_telegraph_sampler_matches_exact_fast_switching():
    fam = D.telegraph_family(1e-5, 4.85e-6, 567e-6, L, K_MAX, BURN)
    th = np.array([2.0, 0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 1.0, -1.5, 0.3, 0.4])
    exact = fam.process(th).expected_windows(K_MAX, BURN, L, common=True)
    shots = fam.sampler(th, 20000, 3)
    est = W.per_shot_windows(shots, K_MAX, BURN, common=True)
    se = est.std(0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(0) - exact) < 4.5 * se + 1e-12)


def test_fluctuator_path_occupancy_and_persistence():
    path = D.fluctuator_path(400, L, 1.0, 0.25, 4.85e-6, 567e-6, 5)
    assert np.all(path == path[:, :1])  # tau = 1 s: no switch within a 300 us shot
    fast = D.fluctuator_path(2000, L, 1e-5, 0.25, 4.85e-6, 567e-6, 6)
    assert abs(fast.mean() - 0.25) < 0.01


# -- time-varying processes ------------------------------------------------------------------


def _brute_windows(tv: D.TimeVaryingProcess, k_max: int, burn: int) -> np.ndarray:
    """Exact window frequencies by enumerating every sequence (short shots only)."""
    length = tv.length
    last = length - k_max - 2
    out = np.zeros(k_max + 1)
    for seq in itertools.product((0, 1), repeat=length):
        p = tv.init
        for t, y in enumerate(seq):
            p = p @ tv.ops[t, y]
        prob = float(p @ tv.final)
        arr = np.array(seq, dtype=np.uint8)[None, :]
        out += prob * W.per_shot_windows(arr, k_max, burn, common=True)[0]
    assert last >= burn
    return out


def test_time_varying_windows_brute_force_and_constant_case():
    tv = D.drifting_two_state(((0.9, 0.6), (0.2, 0.8)), ((0.3, 0.5), (0.7, 0.4)), [1.0, 0.0], 9)
    np.testing.assert_allclose(
        tv.expected_windows(2, 1, common=True), _brute_windows(tv, 2, 1), atol=1e-14
    )
    const = D.TimeVaryingProcess.constant(COMPS[1], L)
    np.testing.assert_allclose(
        const.expected_windows(K_MAX, BURN, common=True),
        COMPS[1].expected_windows(K_MAX, BURN, L, common=True),
        atol=1e-15,
    )


def test_within_shot_family_torch_matches_numpy_and_breaks_rank():
    fam = D.within_shot_drift_family(L, K_MAX, BURN)
    th = torch.tensor([[2.0, 1.0, -1.0, 1.0, -1.0, 2.0, 1.5, -1.0, 0.0]], dtype=torch.float64)
    st = fam.stats(th)
    p = fam.process(th[0].numpy())
    g = p.expected_windows(K_MAX, BURN, common=True)
    np.testing.assert_allclose(st["g"][0].numpy(), g, atol=1e-14)
    np.testing.assert_allclose(st["marg"][0].numpy(), p.marginals(), atol=1e-14)
    starts = list(range(BURN, L - 3, 4))
    np.testing.assert_allclose(st["block"][0].numpy(), p.block_probs(starts, 4), atol=1e-14)
    assert abs(W.witness(g, 0)) > 1e-8  # drift within a shot is not a fixed 2-state process


def test_continuous_family_torch_matches_numpy():
    fam = D.continuous_family(8, L, K_MAX, BURN)
    th = torch.randn(1, 10, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    p = fam.process(th[0].numpy())
    np.testing.assert_allclose(
        fam.stats(th)["g"][0].numpy(), p.expected_windows(K_MAX, BURN, L, common=True), atol=1e-14
    )


# -- bootstraps and segments -----------------------------------------------------------------


def test_block_bootstrap_b1_equals_shot_bootstrap():
    np.testing.assert_array_equal(
        D.block_bootstrap_indices(1234, 1, 50, 9), W.bootstrap_indices(1234, 50, 9)
    )
    idx = D.block_bootstrap_indices(1000, 100, 3, 2)
    assert idx.shape == (3, 1000)
    d = np.diff(idx, axis=1) % 1000
    assert np.all(d[:, :99] == 1)  # first block is contiguous


def test_stratified_bootstrap_keeps_segments():
    segs = [np.arange(0, 300), np.arange(300, 1000)]
    idx = D.stratified_block_indices(segs, 50, 20, 1)
    assert idx.shape == (20, 1000)
    assert np.all(idx[:, :300] < 300) and np.all(idx[:, 300:] >= 300)


def test_segment_estimate_equals_pooled_for_identical_segments():
    shots = COMPS[0].sample(500, L, 11)
    both = np.concatenate([shots, shots])
    per = W.per_shot_windows(both, K_MAX, BURN, common=True)
    pooled = D.segment_stats("all", per, both, range(4), 100, 50, 0)
    for half in (slice(0, 500), slice(500, 1000)):
        seg = D.segment_stats("half", per[half], both[half], range(4), 100, 50, 0)
        np.testing.assert_allclose(seg.g, pooled.g, atol=1e-15)
        for j in range(4):
            assert abs(seg.w[j] - pooled.w[j]) < 1e-15
    # exact: a mixture of identical segments has the segment's expectation
    mix = D.mixture([COMPS[0], COMPS[0]], [0.5, 0.5])
    np.testing.assert_allclose(
        mix.expected_windows(K_MAX, BURN, L, common=True),
        COMPS[0].expected_windows(K_MAX, BURN, L, common=True),
        atol=1e-15,
    )


def test_block_se_b1_is_delta_and_ess_iid():
    shots = COMPS[0].sample(4000, L, 12)
    per = W.per_shot_windows(shots, K_MAX, BURN, common=True)
    assert abs(D.block_se(per, 0, 1) / W.delta_se(per, 0) - 1) < 1e-3
    ess = D.effective_sample_size(per, 0, 200)
    assert 0.5 * 4000 < ess < 2 * 4000


def test_blocks_within_never_straddle():
    mask = np.ones(2000, bool)
    mask[530:800] = False
    blocks = D.blocks_within(mask, 100)
    for b in blocks:
        assert np.all(np.diff(b) == 1)
    assert sum(len(b) for b in blocks) == mask.sum()


# -- stationarity tests ------------------------------------------------------------------------


def test_chi2_sf_known_values():
    assert abs(D.chi2_sf(3.841458820694124, 1) - 0.05) < 1e-10
    assert abs(D.chi2_sf(18.307038053275146, 10) - 0.05) < 1e-10


def test_changepoint_finds_planted_change():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    x[1200:1800] += 0.4
    r = D.changepoint_test(x, 50, 199, 1)
    assert r["p"] <= 0.01
    assert abs(r["start"] - 1200) <= 100 and abs(r["stop"] - 1800) <= 100


def test_changepoint_nominal_size_without_change():
    rng = np.random.default_rng(1)
    rej = 0
    n_data = 300
    for i in range(n_data):
        x = rng.exponential(size=1000)  # skewed, like per-shot window counts
        rej += D.changepoint_test(x, 50, 99, 100 + i)["p"] <= 0.1
    rate = rej / n_data
    assert 0.05 <= rate <= 0.16  # nominal 0.10; binomial SD 0.017


def test_homogeneity_and_constancy_sizes():
    shots = COMPS[2].sample(6000, L, 13)
    per = W.per_shot_windows(shots, K_MAX, BURN, common=True)
    blocks = D.blocks_within(np.ones(6000, bool), 500)
    h = D.homogeneity_chi2(per, blocks)
    assert h["p_joint"] > 1e-3
    shifted = per.copy()
    shifted[:1000, 1] += 0.05
    assert D.homogeneity_chi2(shifted, blocks)["p_k1"] < 1e-6
    c = D.block_witness_constancy(per, D.blocks_within(np.ones(6000, bool), 1000), 0)
    assert c["blocks"] == 6 and 0 <= c["p"] <= 1


def test_epidemic_gain_is_lr_of_a_segment():
    x = np.zeros(400)
    x[100:200] = 1.0
    t, a, b = D.epidemic_gain(x, 50)
    assert t > 0
    assert (a, b) == (100, 200)
    prof = D.boundary_profile(x, 100, 200, "start", 10)
    assert prof[np.argmax(prof[:, 1]), 0] == 100


def test_ramp_fit_prefers_true_width():
    x = np.clip((np.arange(2000) - 1000) / 400 + 0.5, 0, 1)
    assert D.ramp_fit(x, 1000, 400, 800) < 1e-20
    assert D.ramp_fit(x, 1000, 0, 800) > 1.0


# -- adversarial search ------------------------------------------------------------------------


def test_adversarial_search_feasible_on_own_data():
    """A 2-state target is reproduced by a K = 1 'mixture', whose w(0) is then exactly 0."""
    p = COMPS[0]
    g = p.expected_windows(K_MAX, BURN, L, common=True)
    marg = p.marginals(L)[BURN:]
    starts = list(range(BURN, L - 3, 4))
    blk = p.block_probs(starts, 4)
    tgt = D.Target(
        g=g,
        g_se=np.full_like(g, 1e-3),
        marg=marg,
        marg_se=np.full_like(marg, 3e-3),
        block=blk,
        block_cov_inv=np.eye(16) / 1e-6,
        z_max=3.0,
        marg_chi2_max=80.0,
        block_chi2_max=30.0,
    )
    fam = D.mixture_family(1, L, K_MAX, BURN)
    res = D.adversarial_search(fam.stats, fam.n_params, tgt, BURN, 0.006, 8, (300, 200), 0)
    assert res["feasible"].any()
    assert np.all(np.abs(res["w0"]) < 1e-12)


@pytest.mark.parametrize("k", [2])
def test_mixture_family_can_produce_nonzero_witness(k):
    fam = D.mixture_family(k, L, K_MAX, BURN)
    th = torch.tensor(
        [[3.0, 3.0, -2.0, 2.0, 0.0, -3.0, -3.0, -2.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    assert abs(float(D.witness_t(fam.stats(th)["g"], 0)[0])) > 1e-6


def test_chi2_ppf_inverts_sf():
    assert abs(D.chi2_ppf(0.99, 15) - 30.577914166892494) < 1e-8


# -- step 6: physical modifications of the calibrated instrument (experiments/phase7_drift.py) --


def _p7():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "phase7_drift.py"
    spec = importlib.util.spec_from_file_location("phase7_drift", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_physical_superops_analytic():
    p7 = _p7()
    rho = np.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]])
    vec = rho.reshape(-1, order="F")
    unvec = lambda v: v.reshape(2, 2, order="F")  # noqa: E731
    u = np.diag([np.exp(-0.35j), np.exp(0.35j)])
    np.testing.assert_allclose(unvec(p7._rz_super(0.7) @ vec), u @ rho @ u.conj().T, atol=1e-15)
    d = unvec(p7._dephase_super(0.3) @ vec)
    np.testing.assert_allclose(d, [[0.6, 0.7 * rho[0, 1]], [0.7 * rho[1, 0], 0.4]], atol=1e-15)
    r = unvec(p7._relax_super(0.25) @ vec)
    assert abs(r[1, 1] - 0.3) < 1e-15 and abs(r[0, 1] - np.sqrt(0.75) * rho[0, 1]) < 1e-15
    z = np.load(_root() / "results/summaries/phase7_audit/instruments.npz")
    sup = z["calibrated"]
    for model, x0 in (("detuning", [0.0]), ("readout", [0.0, 0.0]), ("detuning_readout", [0, 0])):
        np.testing.assert_allclose(p7.physical_superops(sup, model, np.array(x0, float)), sup)
    ro = p7.physical_superops(sup, "readout", np.array([0.1, 0.05]))
    np.testing.assert_allclose(ro.sum(0), sup.sum(0), atol=1e-15)  # same channel


def test_joint_q59_model_reduces_to_calibrated():
    p7 = _p7()
    z = np.load(_root() / "results/summaries/phase7_audit/instruments.npz")
    cal = W.LinearProcess.instrument(z["calibrated"], z["rho0"])
    eps = float(
        np.load(_root() / "results/summaries/phase7_ablation/pieces.npz")["calibrated__eps_q59"]
    )
    j = p7.joint_process("q59_reset", np.array([eps, eps]))
    np.testing.assert_allclose(
        j.expected_windows(K_MAX, BURN, L, common=True),
        cal.expected_windows(K_MAX, BURN, L, common=True),
        atol=1e-12,
    )


def _root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]
