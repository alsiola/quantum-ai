"""Feed-forward layout (issue #9) and operational information/disturbance metrics.

Analytic cases: the weak-measurement family ``D_y(g)``. Independent checks:
Monte Carlo over Haar-random pure states, random unitaries, and the
Banaszek bound on random instruments.
"""

import math

import pytest
import torch

from qmem import (
    Circuit,
    ControlledRotationBlock,
    Instrument,
    ModelConfig,
    OutcomeConditionedBlock,
    build_model,
)
from qmem.linalg import DTYPE, dagger, eye, ket_to_dm
from qmem.metrics import (
    banaszek_bound,
    best_correction,
    corrected_entanglement_fidelity,
    entanglement_fidelity,
    estimation_fidelity,
    mutual_information,
    operation_fidelity,
    operational_strength,
    operational_summary,
    stationary_state,
    strength_lower_bound,
    tv_gain,
)

ATOL = 1e-12
GS = [0.0, 0.2, math.pi / 4, 1.1, math.pi / 2]


def _haar_unitary(d, gen):
    z = torch.randn(d, d, generator=gen, dtype=torch.float64) + 1j * torch.randn(
        d, d, generator=gen, dtype=torch.float64
    )
    q, r = torch.linalg.qr(z.to(DTYPE))
    return q * (torch.diagonal(r) / torch.diagonal(r).abs())


def _haar_kets(n, d, gen):
    z = torch.randn(n, d, generator=gen, dtype=torch.float64) + 1j * torch.randn(
        n, d, generator=gen, dtype=torch.float64
    )
    z = z.to(DTYPE)
    return z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)


def _random_instrument(n_memory, n_output, n_env, gen):
    n = n_memory + n_output + n_env
    return Instrument.from_unitary(_haar_unitary(2**n, gen), n_memory, n_output, n_env)


def _weak(g, n_memory=1, n_output=1):
    n = n_memory + n_output
    pairs = [(k % n_memory, n_memory + k) for k in range(n_output)]
    block = ControlledRotationBlock(pairs, n, g, shared=True)
    return Instrument.from_unitary(Circuit([block], n).unitary(), n_memory, n_output)


def _ff_config(**kw):
    base = dict(n_memory=2, n_output=1, memory_depth=2, joint_depth=0, feedforward_depth=2, seed=3)
    return ModelConfig(**{**base, **kw})


# --- the feed-forward layout ---------------------------------------------------------


def test_defaults_unchanged():
    cfg = ModelConfig()
    assert cfg.feedforward_depth == 0 and not cfg.strength_shared
    names = [type(b).__name__ for b in build_model(cfg).circuit.blocks]
    assert names == ["HardwareEfficientBlock", "ControlledRotationBlock"]


@pytest.mark.parametrize("n_output,n_env", [(1, 0), (2, 0), (1, 1)])
def test_kraus_is_w_times_unfed_kraus(n_output, n_env):
    """K_y = W_y K_y^(no feed-forward), with the same V and g (same seed, block order)."""
    ff = build_model(_ff_config(n_output=n_output, n_env=n_env, env_strength=0.4))
    plain = build_model(
        _ff_config(n_output=n_output, n_env=n_env, env_strength=0.4, feedforward_depth=0)
    )
    w = next(b for b in ff.circuit.blocks if isinstance(b, OutcomeConditionedBlock))
    ws = w.outcome_unitaries().detach()
    k_ff = ff.instrument().kraus.detach()
    k_plain = plain.instrument().kraus.detach()
    assert torch.allclose(k_ff, torch.einsum("yij,yejk->yeik", ws, k_plain), atol=ATOL)
    for y in range(ws.shape[0]):
        assert torch.allclose(ws[y] @ dagger(ws[y]), eye(4), atol=ATOL)


@pytest.mark.parametrize("n_output", [1, 2])
def test_effects_invariant_under_any_w(n_output):
    model = build_model(_ff_config(n_output=n_output, strength=0.7))
    before = model.instrument().effects().detach()
    gen = torch.Generator().manual_seed(11)
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() == 3 and p is not model.circuit.blocks[0].theta:
                    p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 3)
        inst = model.instrument()
        assert torch.allclose(inst.effects().detach(), before, atol=ATOL)
        assert float(inst.completeness_defect().detach()) < ATOL


def test_feedforward_block_commutes_with_readout():
    """The controlled form equals measure-then-apply-W_y: the block is ancilla-diagonal."""
    gen = torch.Generator().manual_seed(0)
    b = OutcomeConditionedBlock([0, 1], [2, 3], 4, depth=2, generator=gen)
    u = b.unitary().detach().reshape(4, 4, 4, 4)  # (mem_out, y_out, mem_in, y_in)
    ws = b.outcome_unitaries().detach()
    for y in range(4):
        for y2 in range(4):
            expect = ws[y] if y == y2 else torch.zeros(4, 4, dtype=DTYPE)
            assert torch.allclose(u[:, y, :, y2], expect, atol=ATOL)


def test_g_zero_limit():
    """g = 0: E_0 = I (no information); the update is rho -> W_0 V rho V^dag W_0^dag."""
    model = build_model(_ff_config(n_output=2, strength=0.0, strength_trainable=False))
    inst = model.instrument()
    eff = inst.effects().detach()
    assert torch.allclose(eff[0], eye(4), atol=ATOL)
    assert eff[1:].abs().max() < ATOL
    v = model.circuit.blocks[0].unitary().detach()[::4, ::4]  # V on the memory
    w0 = model.circuit.blocks[2].outcome_unitaries().detach()[0]
    assert torch.allclose(inst.kraus.detach()[0, 0], w0 @ v, atol=ATOL)
    assert operational_strength(inst) == pytest.approx(0, abs=1e-12)
    assert tv_gain(inst) == pytest.approx(0, abs=1e-12)
    assert operation_fidelity(inst) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("n_output", [1, 2])
def test_g_half_pi_limit_is_projective(n_output):
    model = build_model(_ff_config(n_output=n_output, strength=math.pi / 2))
    eff = model.instrument().effects().detach()
    for e in eff:
        assert torch.allclose(e @ e, e, atol=1e-12)
    assert operational_strength(model.instrument()) == pytest.approx(1, abs=1e-12)


def test_shared_strength_matches_equal_per_pair():
    pairs = [(0, 2), (1, 3)]
    shared = ControlledRotationBlock(pairs, 4, 0.37, trainable=True, shared=True)
    per = ControlledRotationBlock(pairs, 4, 0.37, trainable=True)
    assert shared.strength.shape == (1,) and per.strength.shape == (2,)
    assert torch.allclose(shared.unitary(), per.unitary(), atol=ATOL)
    model = build_model(_ff_config(n_output=2, strength_shared=True, strength_trainable=True))
    assert model.circuit.blocks[1].strength.numel() == 1


def test_feedforward_model_trains_gradients_reach_w_and_g():
    model = build_model(_ff_config(strength_shared=True, strength_trainable=True))
    seqs = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])
    (-model.log_prob(seqs).sum()).backward()
    w = model.circuit.blocks[2]
    assert all(float(p.grad.abs().max()) > 0 for p in w.parameters())
    assert float(model.circuit.blocks[1].strength.grad.abs()) > 0


# --- metrics: closed forms for D_y(g) ------------------------------------------------


@pytest.mark.parametrize("g", GS)
def test_weak_qubit_closed_forms(g):
    s2, c = math.sin(g) ** 2, math.cos(g)
    inst = _weak(g)
    assert tv_gain(inst) == pytest.approx(s2, abs=1e-12)
    assert operational_strength(inst) == pytest.approx(s2, abs=1e-12)
    assert estimation_fidelity(inst) == pytest.approx((3 + s2) / 6, abs=1e-12)
    assert corrected_entanglement_fidelity(inst) == pytest.approx((1 + c) / 2, abs=1e-12)
    assert operation_fidelity(inst) == pytest.approx((2 + c) / 3, abs=1e-12)
    # D_y(g) is positive, so no correction helps: corrected == uncorrected
    assert entanglement_fidelity(inst) == pytest.approx((1 + c) / 2, abs=1e-12)
    # I(R:Y) at I/2: p0 = (1 + c^2)/2 with state diag(1, c^2)/(1 + c^2); p1 branch is pure
    p0, q = (1 + c * c) / 2, 1 / (1 + c * c)
    h = -sum(x * math.log(x) for x in (q, 1 - q) if x > 0)
    expect = math.log(2) - p0 * h
    assert mutual_information(inst, eye(2) / 2) == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize("g", GS)
def test_weak_two_qubit_memory_closed_forms(g):
    """Memory of 2 qubits; binary: one coupling; 4 symbols: shared g on both qubits."""
    s2, c = math.sin(g) ** 2, math.cos(g)
    b = _weak(g, n_memory=2, n_output=1)
    assert operational_strength(b) == pytest.approx(s2, abs=1e-12)
    assert tv_gain(b) == pytest.approx(s2, abs=1e-12)
    assert operation_fidelity(b) == pytest.approx((3 + 2 * c) / 5, abs=1e-12)
    q = _weak(g, n_memory=2, n_output=2)
    assert operational_strength(q) == pytest.approx((2 * s2 + s2 * s2) / 3, abs=1e-12)
    assert tv_gain(q) == pytest.approx(1 - c**4, abs=1e-12)
    assert corrected_entanglement_fidelity(q) == pytest.approx(((1 + c) / 2) ** 2, abs=1e-12)


# --- metrics: independent checks ------------------------------------------------------


def test_metrics_invariant_under_outcome_unitaries():
    gen = torch.Generator().manual_seed(1)
    inst = _random_instrument(2, 1, 1, gen)
    ws = torch.stack([_haar_unitary(4, gen) for _ in range(2)])
    moved = Instrument(torch.einsum("yij,yejk->yeik", ws, inst.kraus))
    a, b = operational_summary(inst), operational_summary(moved)
    for k in ("tv_gain", "operational_strength", "estimation_fidelity", "operation_fidelity"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_tv_gain_brute_force():
    gen = torch.Generator().manual_seed(2)
    inst = _random_instrument(1, 2, 0, gen)  # qubit memory, 4 outcomes
    kets = _haar_kets(40000, 2, gen)
    p = inst.probabilities(ket_to_dm(kets))
    tv = 0.5 * (p[:200, None, :] - p[None, :, :]).abs().sum(-1)
    best = float(tv.max())
    exact = tv_gain(inst)
    assert best <= exact + 1e-12
    assert best > exact - 0.02


def test_estimation_and_operation_fidelity_monte_carlo():
    gen = torch.Generator().manual_seed(3)
    inst = _random_instrument(1, 1, 1, gen)
    kets = _haar_kets(200000, 2, gen)
    rho = ket_to_dm(kets)
    eff = inst.effects()
    _, vec = torch.linalg.eigh(eff)
    guess = vec[:, :, -1]  # (Y, d): best guess per outcome, top eigenvector of E_y
    p = inst.probabilities(rho)  # (N, Y)
    overlap = (kets.conj() @ guess.T).abs() ** 2  # (N, Y)
    g_mc = float((p * overlap).sum(-1).mean())
    assert g_mc == pytest.approx(estimation_fidelity(inst), abs=3e-3)
    us = [best_correction(inst.kraus[y])[0] for y in range(inst.n_outcomes)]
    corrected = Instrument(torch.einsum("yji,yejk->yeik", torch.stack(us).conj(), inst.kraus))
    out = corrected.channel(rho)
    f_mc = float(torch.einsum("ni,nij,nj->n", kets.conj(), out, kets).real.mean())
    assert f_mc == pytest.approx(operation_fidelity(inst), abs=3e-3)


def test_best_correction_beats_random_unitaries():
    gen = torch.Generator().manual_seed(4)
    inst = _random_instrument(1, 1, 2, gen)  # 4 Kraus operators per outcome
    for y in range(inst.n_outcomes):
        k = inst.kraus[y]
        _, best = best_correction(k)
        for _ in range(300):
            u = _haar_unitary(2, gen)
            val = float((torch.einsum("ji,eji->e", u.conj(), k).abs() ** 2).sum())
            assert val <= best + 1e-10
    # single Kraus: the polar factor attains ||K||_1^2
    k1 = _random_instrument(2, 1, 0, gen).kraus[0]
    _, v = best_correction(k1)
    assert v == pytest.approx(float(torch.linalg.svdvals(k1[0]).sum()) ** 2, abs=1e-12)


def test_banaszek_bound_holds_for_random_instruments():
    gen = torch.Generator().manual_seed(5)
    assert float(banaszek_bound(0.5, 2)) == pytest.approx(1)
    assert float(banaszek_bound(2 / 3, 2)) == pytest.approx(2 / 3)
    assert float(banaszek_bound(0.25, 4)) == pytest.approx(1)
    for n_m, n_y, n_e in [(1, 1, 0), (1, 1, 1), (1, 2, 0), (2, 1, 0), (2, 2, 0), (2, 1, 1)]:
        for _ in range(4):
            inst = _random_instrument(n_m, n_y, n_e, gen)
            g, f = estimation_fidelity(inst), operation_fidelity(inst)
            assert 1 / inst.dim - 1e-12 <= g <= 2 / (inst.dim + 1) + 1e-12
            assert f <= float(banaszek_bound(g, inst.dim)) + 1e-9
    for g in GS:  # the weak family sits on or under the qubit bound
        inst = _weak(g)
        bound = float(banaszek_bound(estimation_fidelity(inst), 2))
        assert operation_fidelity(inst) <= bound + 1e-12


def test_stationary_state():
    gen = torch.Generator().manual_seed(6)
    inst = _random_instrument(2, 1, 0, gen)
    rho, info = stationary_state(inst)
    assert torch.allclose(inst.channel(rho), rho, atol=1e-10)
    assert info["residual"] < 1e-10 and info["min_eigenvalue"] > -1e-12
    # amplitude damping (gamma = 0.3) as a 2-outcome instrument: fixed point |0><0|
    gam = 0.3
    k = torch.zeros(2, 1, 2, 2, dtype=DTYPE)
    k[0, 0] = torch.tensor([[1, 0], [0, math.sqrt(1 - gam)]], dtype=DTYPE)
    k[1, 0] = torch.tensor([[0, math.sqrt(gam)], [0, 0]], dtype=DTYPE)
    rho, _ = stationary_state(Instrument(k))
    assert torch.allclose(rho, ket_to_dm(torch.tensor([1, 0], dtype=DTYPE)), atol=1e-10)


def test_mutual_information_limits():
    for d_q in (1, 2):
        d = 2**d_q
        assert mutual_information(_weak(0.0, d_q, 1), eye(d) / d) == pytest.approx(0, abs=1e-12)
    # projective on a full basis of a maximally mixed qubit: ln 2
    assert mutual_information(_weak(math.pi / 2), eye(2) / 2) == pytest.approx(math.log(2))
    # on a pure state there is nothing to learn about the reference
    plus = ket_to_dm(torch.tensor([1, 1], dtype=DTYPE) / math.sqrt(2))
    assert mutual_information(_weak(math.pi / 2), plus) == pytest.approx(0, abs=1e-12)


def test_strength_lower_bound():
    # iid needs no information; deterministic transitions both ways need eta = 1
    iid = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
    assert strength_lower_bound(iid, 4)["eta_lower_bound"] == pytest.approx(0)
    flip = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert strength_lower_bound(flip, 4)["eta_lower_bound"] == pytest.approx(1)
    # a weak instrument cannot exceed its ceiling: p(1|rho) <= sin^2 g for every rho
    g = 0.6
    inst = _weak(g, n_memory=2)
    gen = torch.Generator().manual_seed(7)
    p = inst.probabilities(ket_to_dm(_haar_kets(5000, 4, gen)))
    assert float(p[:, 1].max()) <= math.sin(g) ** 2 + 1e-12
    bound = strength_lower_bound(p, 4)["eta_lower_bound"]
    assert bound <= operational_strength(inst) + 1e-12
    assert bound > operational_strength(inst) - 0.02
