"""Phase 8 Stage 0B (#30): classical hidden-state/intervention models (``qmem.classical``)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qmem import classical as C
from qmem import witness as W

ARMS = ("baseline", "A2s_spec", "A1", "A2", "A3", "D1")


def _sample(model: C.InterventionModel, arm: str, n: int, length: int, seed: int):
    """Shots by explicit hidden-state simulation (s -> u -> v -> s'), independent of the
    linear-process code."""
    rng = np.random.default_rng(seed)
    d = model.d
    paulis = rng.integers(0, 4, size=(n, length)) if arm == "A3" else None
    y = np.zeros((n, length), dtype=np.uint8)
    m = np.zeros((n, length), dtype=np.uint8) if arm == "A2" else None
    for i in range(n):
        s = rng.choice(d, p=model.init)
        for t in range(length):
            if arm == "D1":
                row = model.d1[:, s, :].ravel()
                c = rng.choice(2 * d, p=row)
                y[i, t], s = c // d, c % d
                continue
            u = rng.choice(d, p=model.pre[s])
            if arm == "A2":
                row = model.meas[:, u, :].ravel()
                c = rng.choice(2 * d, p=row)
                m[i, t], v = c // d, c % d
            elif arm == "A3":
                v = rng.choice(d, p=model.pauli[paulis[i, t], u])
            else:
                v = rng.choice(d, p=model.slot(arm)[u])
            c = rng.choice(2 * d, p=model.post[:, v, :].ravel())
            y[i, t], s = c // d, c % d
    return C.ArmData(y, m, paulis)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("d", [2, 3])
def test_likelihood_matches_brute_force_enumeration(arm, d):
    rng = np.random.default_rng(10 * d + len(arm))
    model = C.random_model(d, rng)
    length = 3
    total = 0.0
    for ys in itertools.product((0, 1), repeat=length):
        ms_set = itertools.product((0, 1), repeat=length) if arm == "A2" else [None]
        for ms in ms_set:
            ps = rng.integers(0, 4, length) if arm == "A3" else None
            bf = C.brute_force_prob(model, arm, ys, ms, ps)
            y = np.array([ys], dtype=np.uint8)
            m = None if ms is None else np.array([ms], dtype=np.uint8)
            pa = None if ps is None else ps[None, :]
            assert np.exp(-model.nll(arm, y, m, pa)[0]) == pytest.approx(bf, rel=1e-12)
            total += bf if arm != "A3" else 0.0
    if arm != "A3":
        assert total == pytest.approx(1.0, abs=1e-12)  # a normalised distribution


def test_record_marginal_is_the_a2_process():
    """Summing A2's joint record over ``m`` gives the y-only A2 process (the slot map is
    ``M_0 + M_1``)."""
    model = C.random_model(3, np.random.default_rng(1))
    y = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    p = sum(
        np.exp(-model.nll("A2", y, np.array([ms], dtype=np.uint8))[0])
        for ms in itertools.product((0, 1), repeat=4)
    )
    assert p == pytest.approx(np.exp(-model.nll("A2", y)[0]), rel=1e-12)


def test_exact_statistics_match_simulation():
    model = C.random_model(3, np.random.default_rng(5), conc=2.0)
    for arm in ("baseline", "A2"):
        st = C.arm_stats(model, arm, 12, 2, 4, (0,))
        a = _sample(model, arm, 3000, 12, 7)
        assert a.y[:, 2:].mean() == pytest.approx(st["p1"], abs=0.02)
        per = W.per_shot_windows(a.y, 4, 2, common=True)
        np.testing.assert_allclose(per.mean(0), st["g"], atol=0.02)
    rs = C.record_stats_exact(model, 12, 2)
    assert a.m[:, 2:].mean() == pytest.approx(rs["p_m1"], abs=0.02)


@pytest.mark.parametrize("seed", range(5))
def test_two_state_models_give_zero_witness_in_every_stationary_arm(seed):
    """The rank argument in the intervention class: with d = 2 every stationary arm is a
    2-dimensional process, so w(j) = 0 exactly (common start set), whatever the maps."""
    model = C.random_model(2, np.random.default_rng(seed), conc=0.5)
    for arm in ("baseline", "A2s_spec", "A1", "A2", "D1"):
        st = C.arm_stats(model, arm, 64, 16, 7, (0, 1, 2, 3))
        for j in range(4):
            assert abs(st[f"w{j}"]) < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_rank_two_slot_map_gives_zero_witness_at_any_d(seed):
    """A2's vanishing needs no small hidden state: if the slot map has rank <= 2 (e.g. the Z
    measurement re-prepares one of two hidden-state distributions, a measure-and-prepare
    map), ``L[y] = P M Q[y]`` has rank <= 2 and w = 0 exactly, while the baseline keeps
    w != 0."""
    rng = np.random.default_rng(seed)
    model = C.random_model(4, rng, conc=0.3)
    r = rng.dirichlet(np.ones(4), size=2)  # the two post-measurement distributions
    resp = rng.dirichlet(np.ones(2), size=4)  # P(m | u)
    model.meas = np.stack([resp[:, mm, None] * r[mm][None, :] for mm in (0, 1)])
    st = C.arm_stats(model, "A2", 64, 16, 7, (0, 1, 2, 3))
    assert max(abs(st[f"w{j}"]) for j in range(4)) < 1e-12
    base = C.arm_stats(model, "baseline", 64, 16, 7, (0,))
    assert abs(base["w0"]) > 1e-8


def test_qnd_measurement_reproduces_the_baseline_exactly():
    """Theorem 1 of the framework: a measurement that is QND on the hidden state
    (``sum_m M_m = I``) leaves every statistic of A2's output equal to the baseline's."""
    rng = np.random.default_rng(3)
    model = C.random_model(4, rng)
    resp = rng.dirichlet(np.ones(2), size=4)
    model.meas = np.stack([np.diag(resp[:, mm]) for mm in (0, 1)])
    a2 = C.arm_stats(model, "A2", 64, 16, 7, (0, 1))
    base = C.arm_stats(model, "baseline", 64, 16, 7, (0, 1))
    for k in ("w0", "w1", "p1", "lag1"):
        assert a2[k] == pytest.approx(base[k], abs=1e-14)


def test_forward_backward_counts_sum_to_steps_and_match_nll():
    model = C.random_model(3, np.random.default_rng(2))
    a = _sample(model, "A2", 200, 10, 3)
    ops, idx, _ = C._classes(model, "A2", a, True)
    c, g0, ll = C.forward_backward(model.init, ops, idx)
    assert c.sum() == pytest.approx(a.y.size, rel=1e-10)
    assert g0.sum() == pytest.approx(len(a.y), rel=1e-10)
    assert -ll == pytest.approx(model.nll("A2", a.y, a.m).sum(), rel=1e-10)


@pytest.mark.parametrize(
    "st",
    [C.Structure(3), C.Structure(3, "kappa", invasive=0.4), C.Structure(4, "zpop", dh=2)],
    ids=lambda s: s.label(),
)
def test_em_is_monotone_and_keeps_the_class(st):
    truth = C.random_model(st.d, np.random.default_rng(0), conc=1.0)
    data = {arm: _sample(truth, arm, 60, 12, 11 + i) for i, arm in enumerate(ARMS)}
    model, kp = C.start_model(st, np.random.default_rng(1))
    prev = -np.inf
    for _ in range(15):
        model, kp, ll = C.em_step(model, data, st, kp)
        assert ll >= prev - 1e-8
        prev = ll
    np.testing.assert_allclose(model.pauli[0], np.eye(st.d))
    for mat in (model.pre, model.spec, model.reset, model.pauli[1]):
        np.testing.assert_allclose(mat.sum(1), 1, atol=1e-12)
    np.testing.assert_allclose(model.meas.sum((0, 2)), 1, atol=1e-12)
    if st.kind == "zpop":  # q75's operations act on z only
        for mat in (model.pre, model.spec, model.reset, *model.pauli, *model.meas):
            r = mat.reshape(2, 2, 2, 2)
            assert np.abs(r[:, 0, :, 1]).max() < 1e-14 and np.abs(r[:, 1, :, 0]).max() < 1e-14
    if st.kind == "kappa":  # QND on the hidden state in a fraction 1 - kappa of steps
        diag = np.diagonal(model.meas.sum(0))
        assert np.all(diag >= 1 - st.invasive - 1e-12)


def test_n_params_counts_free_entries():
    assert C.n_params(C.Structure(2)) == 1 + 2 * 2 * 3 + 5 * 2 + 2 * 3
    assert C.n_params(C.Structure(4, "zpop", dh=2)) == 3 + 2 * 4 * 7 + 5 * 2 + 2 * 3


# -- theorems used in docs/phase8-claim-framework.md ---------------------------------------------


def _random_eb_channel(rng: np.random.Generator, d: int = 2, k: int = 4):
    """A random entanglement-breaking (measure-and-prepare) channel: a random k-outcome POVM
    (``F_i = S^{-1/2} G_i S^{-1/2}``) and a random prepared state per outcome."""
    gs = []
    for _ in range(k):
        a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        gs.append(a @ a.conj().T)
    s = sum(gs)
    w, v = np.linalg.eigh(s)
    sm = v @ np.diag(w**-0.5) @ v.conj().T
    povm = [sm @ g @ sm for g in gs]
    states = []
    for _ in range(k):
        a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        r = a @ a.conj().T
        states.append(r / np.trace(r))
    return povm, states


@pytest.mark.parametrize("seed", range(40))
def test_entanglement_breaking_channel_bell_fidelity_at_most_half(seed):
    """Horodecki-Shor-Ruskai: an EB channel on half of a Bell pair leaves a separable state,
    and a separable two-qubit state has fidelity <= 1/2 with a maximally entangled state
    (Horodecki 1999). Checked on random measure-and-prepare channels against many
    maximally entangled targets, with the PPT test as a second check of separability."""
    rng = np.random.default_rng(seed)
    povm, states = _random_eb_channel(rng, k=int(rng.integers(2, 6)))
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1 / np.sqrt(2)
    # (id ⊗ Φ)(|Φ+><Φ+|) = sum_i F_i^T/2 ⊗ R_i for the measure-and-prepare channel Φ
    out = sum(np.kron(f.T / 2, r) for f, r in zip(povm, states, strict=True))
    assert np.trace(out).real == pytest.approx(1.0)
    # max over maximally entangled targets (I ⊗ U)|Φ+> (random unitaries): <= 1/2
    best = 0.0
    for _ in range(300):
        a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        u, _ = np.linalg.qr(a)
        v = np.kron(np.eye(2), u) @ phi
        best = max(best, float((v.conj() @ out @ v).real))
    assert best <= 0.5 + 1e-12
    # and the PPT test agrees: the output is separable (positive partial transpose)
    pt = out.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    assert np.linalg.eigvalsh(pt).min() >= -1e-12


def test_identity_channel_exceeds_the_half_bound():
    """The positive control: a perfect memory (identity) keeps F = 1 > 1/2, and full Z
    dephasing (an EB channel) gives exactly 1/2."""
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1 / np.sqrt(2)
    bell = np.outer(phi, phi.conj())
    assert (phi.conj() @ bell @ phi).real == pytest.approx(1.0)
    z = np.kron(np.eye(2), np.diag([1.0, -1.0]))
    deph = 0.5 * bell + 0.5 * z @ bell @ z
    assert (phi.conj() @ deph @ phi).real == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(20))
def test_stochastic_matrix_with_stochastic_inverse_is_a_permutation(seed):
    """Lemma (framework, coherence bar): a reversible stochastic map on a finite hidden state
    (its inverse also stochastic) is a permutation; a generic stochastic map's inverse has
    negative entries."""
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 7))
    perm = np.eye(d)[rng.permutation(d)]
    inv = np.linalg.inv(perm)
    assert inv.min() >= -1e-12 and np.allclose(inv.sum(1), 1)
    m = rng.dirichlet(np.ones(d), size=d)
    assert np.linalg.inv(m).min() < 0


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_divisible_phase_group_acts_trivially_on_a_finite_hidden_state(d):
    """Lemma: if phase rotations act on a d-state hidden variable as a group of reversible
    stochastic maps (``M_a M_b = M_{a+b}``), each ``M_phi = M_{phi/N}^N`` with ``N = d!`` is the
    identity, because every permutation of d states has order dividing d!."""
    n = math.factorial(d)
    for p in itertools.permutations(range(d)):
        m = np.eye(d)[list(p)]
        assert np.allclose(np.linalg.matrix_power(m, n), np.eye(d))


def test_klein_tetrahedron_embedding_reproduces_the_quantum_model():
    """``embedding`` is an exact change of coordinates: the (signed) 4-state model it gives
    reproduces the calibrated quantum model's statistics in every q75 arm, and the Paulis
    and the Z measurement are nonnegative in it (the only possible negativity is in the
    slot-to-slot step)."""
    from pathlib import Path

    from qmem import ablate_hw as A

    root = Path(__file__).resolve().parents[1]
    with np.load(root / "results/summaries/phase7_hw_4c/pieces_live.npz") as z:
        pc = A.StepPieces.from_npz(z, "live")
        r = float(z["readout_q75"])
    maps = C.effective_maps(pc, r)
    u = C.klein_tetrahedron(-1.45, -1.3)
    e = C.embedding(maps, u)
    for key in ("pauli", "meas", "reset"):
        assert e[key].min() >= -1e-12
    d1 = np.stack([0.4 * np.eye(4), 0.6 * np.eye(4)])
    model = C.InterventionModel(
        e["init"], np.eye(4), e["post"], np.eye(4), e["reset"], e["meas"], e["pauli"], d1
    )
    # exact for the baseline; A1 and A2 differ only by q59's relaxation while q75 is reset or
    # measured, which the q75-only maps leave out (q59 sits near |0> then)
    for arm, tol in (("baseline", 1e-12), ("A1", 5e-5), ("A2", 5e-5)):
        q = A.process(A.step_superops(arm, pc)).marginals(20)
        c = model.process(arm).marginals(20)
        np.testing.assert_allclose(c, q, atol=tol)
        gq = A.process(A.step_superops(arm, pc)).expected_windows(7, 4, 20, common=True)
        gc = model.process(arm).expected_windows(7, 4, 20, common=True)
        np.testing.assert_allclose(gc, gq.real, atol=tol)
    from qmem import refit as R

    rec = R.RecordModel.build(pc, r)
    assert C.record_stats_exact(model, 64, 16)["p_m1"] == pytest.approx(
        rec.pair_stats(64, 16)["p_m1"], abs=5e-5
    )
    assert C.embedding_negativity(maps, u) < 0  # the calibrated model is not 4-state classical
