"""Phase 4b (issue #13): the falsification tools, each against an analytic case."""

from __future__ import annotations

import math

import torch

from qmem.ablations import build_variant
from qmem.baselines import build_baseline
from qmem.benchmarks import feedforward_readout
from qmem.circuits import PartialSwapBlock
from qmem.data import make_splits
from qmem.falsification import (
    Stage,
    operational_strength_t,
    protected_subspace_search,
    train_stages,
    umps_instrument,
)
from qmem.instrument import Instrument, kraus_from_unitary
from qmem.linalg import DTYPE
from qmem.metrics import operational_strength
from qmem.model import ModelConfig, build_model
from qmem.tasks import make_task
from qmem.train import TrainConfig

D_BASE = {"n_memory": 2, "memory_depth": 0, "strength_trainable": True, "variant": "D"}


def test_partial_swap_with_a_fresh_ancilla_is_amplitude_damping():
    b = PartialSwapBlock([(0, 1)], 2)
    theta = 0.7
    with torch.no_grad():
        b.theta.fill_(theta)
    k = kraus_from_unitary(b.unitary(), 1, 0, 1)[0]  # (E, 2, 2) on the memory qubit
    effects = torch.einsum("eji,ejk->eik", k.conj(), k)
    c2, s2 = math.cos(theta) ** 2, math.sin(theta) ** 2
    assert torch.allclose(effects[0], torch.diag(torch.tensor([1, c2], dtype=DTYPE)))
    assert torch.allclose(effects[1], torch.diag(torch.tensor([0, s2], dtype=DTYPE)))
    one = torch.diag(torch.tensor([0, 1], dtype=DTYPE))
    out = torch.einsum("eij,jk,elk->il", k, one, k.conj())
    assert torch.allclose(out, torch.diag(torch.tensor([s2, c2], dtype=DTYPE)))


def test_new_d_updates_match_d_diss_parameter_count_and_are_valid():
    for n_out, jd, count in ((1, 2, 25), (2, 1, 34)):
        base = {**D_BASE, "n_output": n_out, "joint_depth": jd}
        for extra in (
            {"update_env": 1},
            {"update_env": 1, "update_depth": 2, "update_rotations": ["ry"]},
            {"update_env": 1, "update_entangler": "cz"},
            {"update_env": 2, "update_coupling": "pswap"},
        ):
            m = build_variant({**base, **extra})
            assert m.n_parameters() == count, extra
            assert m.validity_defect() < 1e-12


def test_pswap_update_at_full_strength_erases_the_memory():
    m = build_variant({**D_BASE, "n_output": 1, "joint_depth": 2, "update_env": 2,
                       "update_coupling": "pswap"})  # fmt: skip
    with torch.no_grad():
        for blk in m.update_blocks:
            blk.blocks[1].theta.fill_(math.pi / 2)
    k = m.update_kraus()  # (Y, E, d, d)
    gen = torch.Generator().manual_seed(0)
    target = torch.zeros(4, 4, dtype=DTYPE)
    target[0, 0] = 1
    for _ in range(3):
        a = torch.randn(4, 4, generator=gen, dtype=torch.float64).to(DTYPE)
        rho = a @ a.conj().T
        rho = rho / rho.trace()
        for y in range(2):
            out = torch.einsum("eij,jk,elk->il", k[y], rho, k[y].conj())
            assert torch.allclose(out, target, atol=1e-12)


def test_generator_hqmm_is_a_valid_instrument_with_k_parameters():
    for n_sym, k in ((2, 21), (4, 26)):
        m = build_baseline("hqmm_gen", {"n_memory": 2, "n_generators": k}, n_sym, seed=3)
        assert m.n_parameters() == k
        assert torch.allclose(m.isometry(), m.v0)  # theta = 0: the random start
        with torch.no_grad():
            m.theta.normal_()
        inst = m.instrument()
        assert float(inst.completeness_defect().detach()) < 1e-12
        # brute force: p(y|rho) = sum_e Tr(K rho K^+) against the effects, and sums to 1
        a = torch.randn(4, 4, dtype=torch.float64).to(DTYPE)
        rho = a @ a.conj().T
        rho = rho / rho.trace()
        p = torch.einsum("yeij,jk,yelk->yil", inst.kraus, rho, inst.kraus.conj())
        p = torch.diagonal(p, dim1=-2, dim2=-1).sum(-1).real
        assert torch.allclose(p.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(p, inst.probabilities(rho[None])[0])
        seq = torch.randint(0, n_sym, (3, 6))
        (-m.step_log_probs(seq).mean()).backward()
        assert torch.isfinite(m.theta.grad).all() and m.theta.grad.abs().sum() > 0


def test_differentiable_strength_matches_metric_and_feedforward_formula():
    # Two couplings on two qubits: sum_y lambda_max(E_y) = (1 + s)^2, s = sin^2 g, so
    # eta = (2 s + s^2) / 3 and d eta / d g = (2 + 2 s) sin(2 g) / 3.
    for g in (0.3, 0.8, 1.2):
        inst = feedforward_readout(2, 2, g)
        s = math.sin(g) ** 2
        assert math.isclose(float(operational_strength_t(inst)), (2 * s + s * s) / 3)
        assert math.isclose(float(operational_strength_t(inst)), operational_strength(inst))
        m = build_model(ModelConfig(n_memory=2, n_output=2, memory_depth=0, strength=g,
                                    strength_trainable=True, strength_shared=True))  # fmt: skip
        operational_strength_t(m.instrument()).backward()
        grad = float(m.circuit.blocks[0].strength.grad)
        assert math.isclose(grad, (2 + 2 * s) * math.sin(2 * g) / 3, rel_tol=1e-9)


def test_train_stages_freezes_then_frees_the_strength():
    task = make_task({"kind": "markov", "order": 1, "probs": [[0.9, 0.1], [0.3, 0.7]]})
    splits = make_splits(task, 32, 16, 16, 12, 0)
    m = build_model(ModelConfig(n_memory=1, n_output=1, joint_depth=1, strength=0.5,
                                strength_trainable=True))  # fmt: skip
    cfg = TrainConfig(steps=5, eval_every=1, patience=0, lr=0.1, restore_best=False)
    blk = m.circuit.blocks[1]
    seen = []
    res = train_stages(
        m,
        [
            Stage(splits, cfg, frozen=["strength"], before=lambda mm: seen.append(1)),
            Stage(splits, cfg),
        ],
    )
    assert seen == [1]
    assert res.stages[0]["strength_end"] == [0.5]
    assert res.stages[1]["strength_end"] != [0.5]
    assert res.result.steps_run == 10
    assert all(p.requires_grad for p in m.parameters())
    assert math.isclose(float(blk.strength), res.stages[1]["strength_end"][0])


def _ff_effects(n, g, shared=True):
    cfg = ModelConfig(n_memory=n, n_output=2, memory_depth=0, strength=g, strength_shared=shared)
    return build_model(cfg).instrument().effects()


def test_protected_subspace_on_a_one_qubit_readout():
    # Distractor effects p I, cue effects reading Z: the whole qubit is protected and the
    # read contrast is the spread of q (|0><0| - |1><1|), i.e. 2 q.
    p, q = 0.4, 0.2
    eye = torch.eye(2, dtype=DTYPE)
    z0 = torch.diag(torch.tensor([1, 0], dtype=DTYPE))
    effects = torch.stack([p * eye, p * eye, q * (eye - z0), q * z0])
    r = protected_subspace_search(effects, [0, 1], (3, 2), restarts=4, steps=50)
    assert r["n_feasible"] == 4
    assert math.isclose(r["contrast"], 2 * q, rel_tol=1e-6)


def test_feedforward_two_qubits_has_a_protected_subspace_at_free_levels():
    # Analytic: x = (c|00> + |11>)/sqrt(1+c^2), y = (c|01> + |10>)/sqrt(1+c^2) (disjoint
    # supports) make both distractor compressions multiples of I, with read contrast
    # 2 s^4 / (1 + c^2) (s = sin g, c = cos g). The search must reach at least that.
    g = 0.6
    s2, c2 = math.sin(g) ** 2, math.cos(g) ** 2
    r = protected_subspace_search(_ff_effects(2, g), [0, 1], (3, 2), restarts=8, steps=100)
    assert r["n_feasible"] > 0
    assert r["contrast"] >= 2 * s2 * s2 / (1 + c2) - 1e-6


def test_protected_subspace_search_reports_infeasible_levels():
    # E_a + E_b <= I, so both compressions cannot be 0.6 I: nothing is feasible.
    r = protected_subspace_search(
        _ff_effects(2, 0.8), [0, 1], (3, 2), flat_targets=[0.6, 0.6], restarts=4, steps=50
    )
    assert r["n_feasible"] == 0 and r["contrast"] == 0.0
    assert r["best_run_residual"] > 1e-3


def test_umps_as_an_instrument_reproduces_its_predictive():
    m = build_baseline("umps", {"bond_dim": 3}, 2, seed=1)
    inst, rho = umps_instrument(m)
    assert float(inst.completeness_defect()) < 1e-6
    seq = torch.tensor([[0, 1, 1, 0, 1, 0, 0]])
    pred = m.predictive_probs(seq)[0]
    rho = rho[None]
    for t in range(seq.shape[1]):
        p = inst.probabilities(rho)[0]
        assert torch.allclose(p, pred[t], atol=1e-6)
        br = inst.branch(rho, seq[:, t])
        rho = br / torch.diagonal(br, dim1=-2, dim2=-1).sum(-1).real[:, None, None]


def test_generator_hqmm_instrument_class_is_hqmm():
    inst = build_baseline("hqmm_gen", {"n_memory": 1, "n_generators": 5}, 2, 0).instrument()
    assert isinstance(inst, Instrument) and inst.kraus.shape == (2, 1, 2, 2)
