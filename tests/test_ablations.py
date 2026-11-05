"""Ablation variants B, C_k, D and the hybrid control: validity and brute-force checks."""

import math

import pytest
import torch

from qmem import build_model
from qmem.ablations import (
    ClassicalFeedbackModel,
    ConditionedModel,
    OutcomeAveragedModel,
    ResetMemoryModel,
    SeparateUpdateReadoutModel,
    block_gates,
    build_variant,
    gate_resources,
    resource_counts,
    split_config,
    validity_defect,
)
from qmem.circuits import HardwareEfficientBlock
from qmem.data import all_sequences
from qmem.experiment import RunSpec, run_one
from qmem.gates import ry, rz
from qmem.instrument import kraus_from_unitary
from qmem.linalg import embed, kron, trace
from qmem.model import ModelConfig
from qmem.states import is_density_matrix, purity
from qmem.train import TrainConfig, train

ATOL = 1e-11

BIN = {"n_memory": 2, "n_output": 1, "memory_depth": 2, "joint_depth": 2, "strength": 0.9}
BIN_ENV = {**BIN, "n_env": 1, "env_strength": 0.6, "joint_depth": 1}
QUAD = {"n_memory": 2, "n_output": 2, "memory_depth": 1, "joint_depth": 1, "strength": 1.1}

# (id, model dict): every variant on binary, binary-with-env and 4-symbol layouts.
CASES = []
for name, base in (("bin", BIN), ("bin_env", BIN_ENV), ("quad", QUAD)):
    for variant, extra in (
        ("B", {}),
        ("C", {"reset_every": 1}),
        ("C", {"reset_every": 2}),
        ("D", {"memory_depth": 0, "update_depth": 1}),
        ("hybrid", {"feedback_lags": 2}),
    ):
        tag = f"{variant}{extra.get('reset_every', '')}-{name}"
        CASES.append((tag, {**base, **extra, "variant": variant}))


def _perturb(model, seed):
    """Make every parameter generic (the hybrid controller starts at zero)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.7 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _build(d, seed=5):
    return _perturb(build_variant({**d, "seed": seed}), seed + 100)


def _length(model):
    return 4 if model.n_symbols == 2 else 2


# -- an independent per-variant brute-force probability ---------------------------
def _hybrid_kraus(model, history):
    """Kraus operators for one controller input, from the full rotated unitary."""
    ang = torch.einsum("qai,i->qa", model.feedback_weight, history) + model.feedback_bias
    r = kron(*[ry(ang[q, 0]) @ rz(ang[q, 1]) for q in range(model.n_output)])
    anc = list(range(model.n_memory, model.n_memory + model.n_output))
    u = embed(r, anc, model.n_qubits) @ model.unitary()
    return kraus_from_unitary(u, model.n_memory, model.n_output, model.n_env)


def _brute_prob(model, seq):
    """p(seq) by explicit loops over plain matrices, separately for each variant."""
    seq = seq.tolist()
    rho = model.rho0
    prob = 1.0
    k = model.instrument().kraus
    if isinstance(model, SeparateUpdateReadoutModel):
        effects = [
            sum(k[y, e].conj().T @ k[y, e] for e in range(k.shape[1])) for y in range(k.shape[0])
        ]
        for y in seq:
            prob *= float(trace(effects[y] @ rho).real)
            v = model.update_blocks[y].unitary()
            rho = v @ rho @ v.conj().T
        return prob
    if isinstance(model, OutcomeAveragedModel):
        for y in seq:
            branch = sum(k[y, e] @ rho @ k[y, e].conj().T for e in range(k.shape[1]))
            prob *= float(trace(branch).real)
            rho = sum(
                k[z, e] @ rho @ k[z, e].conj().T
                for z in range(k.shape[0])
                for e in range(k.shape[1])
            )
        return prob
    if isinstance(model, ClassicalFeedbackModel):
        bits = model.token_bits
        h = torch.zeros(model.feedback_lags * model.n_output, dtype=torch.float64)
        for y in seq:
            kh = _hybrid_kraus(model, h)
            branch = sum(kh[y, e] @ rho @ kh[y, e].conj().T for e in range(kh.shape[1]))
            p = float(trace(branch).real)
            prob *= p
            rho = branch / p
            h = torch.cat([bits[y], h])[: h.numel()]
        return prob
    assert isinstance(model, ResetMemoryModel)
    # p(seq) = prod over blocks of k tokens of Tr(Phi_block(rho0)).
    kk = model.reset_every
    for start in range(0, len(seq), kk):
        rho = model.rho0
        for y in seq[start : start + kk]:
            rho = sum(k[y, e] @ rho @ k[y, e].conj().T for e in range(k.shape[1]))
        prob *= float(trace(rho).real)
    return prob


@pytest.mark.parametrize(("tag", "d"), CASES, ids=[c[0] for c in CASES])
def test_variant_is_a_normalised_sequence_model(tag, d):
    model = _build(d)
    assert validity_defect(model) < 1e-12
    seqs = all_sequences(model.n_symbols, _length(model))
    with torch.no_grad():
        logp = model.log_prob(seqs)
        pred = model.predictive_probs(seqs)
        steps = model.step_log_probs(seqs)
    assert math.isclose(float(logp.exp().sum()), 1.0, abs_tol=1e-11)
    assert (pred >= -1e-14).all()
    assert torch.allclose(pred.sum(-1), torch.ones_like(pred[..., 0]), atol=1e-12)
    gathered = pred[:, :-1].gather(-1, seqs[..., None]).squeeze(-1)
    assert torch.allclose(gathered.log(), steps, atol=1e-10)
    with torch.no_grad():
        for i in range(len(seqs)):
            assert math.isclose(float(logp[i].exp()), _brute_prob(model, seqs[i]), abs_tol=ATOL)


@pytest.mark.parametrize(("tag", "d"), CASES, ids=[c[0] for c in CASES])
def test_variant_memory_valid_and_batching_consistent(tag, d):
    model = _build(d)
    seqs = model.sample(12, 6, seed=3)
    with torch.no_grad():
        batched = model.log_prob(seqs)
        single = torch.stack([model.log_prob(s[None])[0] for s in seqs])
        states = model.memory_states(seqs)
    assert torch.allclose(batched, single, atol=1e-13)
    assert is_density_matrix(states.reshape(-1, model.memory_dim, model.memory_dim), atol=1e-9)


def _chi_square(counts, expected):
    """Pearson chi-square with cells pooled (in order) until each expects >= 5."""
    exp_cells, obs_cells, acc_e, acc_o = [], [], 0.0, 0.0
    for e, o in zip(expected.tolist(), counts.tolist(), strict=True):
        acc_e += e
        acc_o += o
        if acc_e >= 5:
            exp_cells.append(acc_e)
            obs_cells.append(acc_o)
            acc_e = acc_o = 0.0
    if acc_e > 0:
        exp_cells[-1] += acc_e
        obs_cells[-1] += acc_o
    stat = sum((o - e) ** 2 / e for o, e in zip(obs_cells, exp_cells, strict=True))
    return stat, len(exp_cells) - 1


@pytest.mark.parametrize(("tag", "d"), CASES, ids=[c[0] for c in CASES])
def test_sampler_matches_log_prob(tag, d):
    model = _build(d)
    length = _length(model)
    n = 40_000
    samples = model.sample(n, length, seed=2024)
    seqs = all_sequences(model.n_symbols, length)
    powers = model.n_symbols ** torch.arange(length - 1, -1, -1)
    counts = torch.bincount((samples * powers).sum(-1), minlength=len(seqs)).double()
    with torch.no_grad():
        expected = model.log_prob(seqs).exp() * n
    stat, dof = _chi_square(counts, expected)
    # 99.9th percentile of chi-square, Wilson-Hilferty approximation.
    z = 3.090
    crit = dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3
    assert stat < crit, (stat, dof, crit)
    assert torch.equal(model.sample(30, 5, seed=1), model.sample(30, 5, seed=1))


@pytest.mark.parametrize(("tag", "d"), CASES, ids=[c[0] for c in CASES])
def test_gradients_finite_and_training_runs(tag, d):
    model = _build(d)
    seqs = model.sample(16, 6, seed=9)
    loss = -model.step_log_probs(seqs).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    from qmem.data import Splits

    splits = Splits(train=seqs, val=seqs[:8], test=seqs[:8], seeds={})
    result = train(model, splits, TrainConfig(steps=3, eval_every=1, patience=0))
    assert result.status == "completed"


# -- the shared loop reproduces A -------------------------------------------------
@pytest.mark.parametrize("d", [BIN, BIN_ENV, QUAD], ids=["bin", "bin_env", "quad"])
def test_hooked_loop_reproduces_architecture_a(d):
    a = build_model(ModelConfig.from_dict({**d, "seed": 8}))
    h = ConditionedModel(a.n_memory, a.n_output, list(a.circuit.blocks), n_env=a.n_env)
    seqs = a.sample(20, 5, seed=4)
    with torch.no_grad():
        assert torch.allclose(h.log_prob(seqs), a.log_prob(seqs), atol=1e-13)
        assert torch.allclose(h.predictive_probs(seqs), a.predictive_probs(seqs), atol=1e-13)
        assert torch.allclose(h.final_memory(seqs).rho, a.final_memory(seqs).rho, atol=1e-13)
    assert torch.equal(h.sample(20, 5, seed=4), seqs)


def test_variant_a_is_build_model():
    a = build_variant({**BIN, "seed": 3})
    b = build_model(ModelConfig.from_dict({**BIN, "seed": 3}))
    assert type(a) is type(b)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb)


# -- B: the outcome-averaged control ------------------------------------------------
@pytest.mark.parametrize("d", [BIN, BIN_ENV, QUAD], ids=["bin", "bin_env", "quad"])
def test_b_equals_marginal_of_a_over_all_histories(d):
    """B's p(y_t) is A's marginal of y_t over every history, at the same parameters."""
    a = build_variant({**d, "seed": 6})
    b = build_variant({**d, "seed": 6, "variant": "B"})
    b.load_state_dict(a.state_dict())
    length = 5 if a.n_symbols == 2 else 3
    seqs = all_sequences(a.n_symbols, length)
    with torch.no_grad():
        pa = a.log_prob(seqs).exp()
        pred_b = b.predictive_probs(seqs)
    for t in range(length):
        marginal = torch.zeros(a.n_symbols, dtype=torch.float64).index_add_(0, seqs[:, t], pa)
        # B's prediction for y_t is the same for every history.
        assert torch.allclose(pred_b[:, t], marginal.expand_as(pred_b[:, t]), atol=ATOL)
    # And B's memory is A's unconditioned memory, whatever the tokens.
    with torch.no_grad():
        assert torch.allclose(
            b.final_memory(seqs).rho,
            a.unconditioned_memory(length).expand(len(seqs), -1, -1),
            atol=ATOL,
        )


# -- C_k: reset --------------------------------------------------------------------------
def test_c1_is_memoryless_and_ck_factorises_into_blocks():
    a = build_variant({**BIN, "seed": 7})
    c1 = build_variant({**BIN, "seed": 7, "variant": "C", "reset_every": 1})
    c3 = build_variant({**BIN, "seed": 7, "variant": "C", "reset_every": 3})
    seqs = all_sequences(2, 6)
    with torch.no_grad():
        p0 = a.instrument().probabilities(a.rho0)
        assert torch.allclose(c1.predictive_probs(seqs), p0.expand(len(seqs), 7, 2), atol=ATOL)
        blocks = a.log_prob(seqs[:, :3]) + a.log_prob(seqs[:, 3:])
        assert torch.allclose(c3.log_prob(seqs), blocks, atol=ATOL)
        long = build_variant({**BIN, "seed": 7, "variant": "C", "reset_every": 100})
        assert torch.allclose(long.log_prob(seqs), a.log_prob(seqs), atol=ATOL)


# -- D: separate update and readout -------------------------------------------------
def test_d_memory_stays_pure_and_identity_update_is_iid():
    d = build_variant({**BIN, "seed": 4, "variant": "D", "memory_depth": 0, "update_depth": 2})
    seqs = d.sample(10, 8, seed=1)
    with torch.no_grad():
        assert torch.allclose(purity(d.memory_states(seqs)), torch.ones(10, 9, dtype=torch.float64))
    iid = build_variant({**BIN, "seed": 4, "variant": "D", "memory_depth": 0, "update_depth": 0})
    with torch.no_grad():
        pred = iid.predictive_probs(seqs)
    assert torch.allclose(pred, pred[:1, :1].expand_as(pred), atol=ATOL)


def test_d_readout_does_not_update_memory():
    """D's memory depends only on the tokens and V_y, not on the readout parameters."""
    d1 = build_variant({**BIN, "seed": 4, "variant": "D", "memory_depth": 0})
    d2 = build_variant({**BIN, "seed": 4, "variant": "D", "memory_depth": 0})
    with torch.no_grad():
        for p in d2.circuit.parameters():
            p.add_(1.0)
    seqs = d1.sample(10, 6, seed=2)
    with torch.no_grad():
        assert torch.allclose(d1.memory_states(seqs), d2.memory_states(seqs), atol=1e-13)
        assert not torch.allclose(d1.log_prob(seqs), d2.log_prob(seqs))


@pytest.mark.parametrize(
    ("a_cfg", "d_extra", "n_params"),
    [
        ({**BIN, "joint_depth": 2, "strength_trainable": True}, {"joint_depth": 2}, 21),
        (
            {**QUAD, "memory_depth": 2, "joint_depth": 2, "strength_trainable": True},
            {"joint_depth": 1},
            26,
        ),
    ],
    ids=["binary", "quaternary"],
)
def test_d_matches_a_parameter_count_for_the_phase3b_configs(a_cfg, d_extra, n_params):
    a = build_variant(a_cfg)
    d = build_variant({**a_cfg, **d_extra, "variant": "D", "memory_depth": 0, "update_depth": 1})
    assert a.n_parameters() == d.n_parameters() == n_params


# -- the hybrid classical-feedback control --------------------------------------------
def test_hybrid_starts_as_a_and_its_controller_uses_history():
    a = build_variant({**BIN, "seed": 2})
    h = build_variant({**BIN, "seed": 2, "variant": "hybrid", "feedback_lags": 2})
    assert h.n_parameters() == a.n_parameters() + 2 * 2 + 2
    seqs = all_sequences(2, 5)
    with torch.no_grad():
        assert torch.allclose(h.log_prob(seqs), a.log_prob(seqs), atol=1e-13)
        h.feedback_weight.fill_(0.8)
        changed = h.predictive_probs(seqs)[:, -1]
    assert not torch.allclose(changed, a.predictive_probs(seqs)[:, -1].detach())


def test_hybrid_context_instruments_are_complete():
    h = _build({**QUAD, "variant": "hybrid", "feedback_lags": 2})
    inst = h.instrument()
    hist = torch.tensor([[1.0, -1, 0, 0], [-1, -1, 1, 1], [0, 0, 0, 0]], dtype=torch.float64)
    k = h.context_kraus(inst, hist).detach()
    eff = torch.einsum("byeji,byejk->bik", k.conj(), k)
    assert torch.allclose(eff, torch.eye(4, dtype=eff.dtype).expand_as(eff), atol=1e-12)
    for b in range(3):
        assert torch.allclose(k[b], _hybrid_kraus(h, hist[b]).detach(), atol=1e-12)


# -- configuration, resources, runner --------------------------------------------------
def test_config_split_and_unknown_keys():
    v, m = split_config({**BIN, "variant": "C", "reset_every": 3})
    assert v.variant == "C" and v.reset_every == 3 and m.n_memory == 2
    with pytest.raises(ValueError):
        split_config({**BIN, "variant": "E"})
    with pytest.raises(ValueError):
        split_config({**BIN, "bogus": 1})


def test_gate_resources_of_a_hardware_efficient_block():
    blk = HardwareEfficientBlock([0, 1, 2], 3, depth=2, rotations=("ry", "rz"), pattern="linear")
    r = gate_resources(block_gates(blk), [0, 1, 2])
    # per layer: 2 rotations on each of 3 qubits (depth 2), then CNOT(0,1), CNOT(1,2).
    assert r == {"depth": 8, "one_qubit_gates": 12, "two_qubit_gates": 4}
    rc = resource_counts(build_variant({**BIN, "variant": "C", "reset_every": 4}))
    assert rc["memory_resets_per_token"] == 0.5 and rc["mid_circuit_measurements_per_token"] == 1


def test_run_one_trains_a_variant():
    spec = RunSpec(
        group="t/D",
        task={"kind": "markov", "order": 1, "probs": [[0.9, 0.1], [0.3, 0.7]]},
        data={"n_train": 16, "n_val": 8, "n_test": 8, "length": 6, "seed": 0},
        model={**BIN, "variant": "D", "memory_depth": 0},
        train={"steps": 2, "eval_every": 1, "patience": 0},
        seed=0,
    )
    rec = run_one(spec)
    assert rec["status"] == "completed", rec.get("traceback")
    assert rec["instrument_completeness_defect"] < 1e-12
    assert rec["n_parameters"] == 20  # g fixed in BIN
