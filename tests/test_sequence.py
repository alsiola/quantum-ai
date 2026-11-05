"""Sequence-level tests: teacher forcing, brute-force checks, gradients, sampler."""

import logging
import math

import pytest
import torch

from qmem import ControlledRotationBlock, ModelConfig, QuantumMemoryModel, build_model
from qmem.data import all_sequences
from qmem.linalg import DTYPE, embed, ket_to_dm, trace
from qmem.states import is_density_matrix

ATOL = 1e-11


def _random_ket(n_qubits, seed):
    g = torch.Generator().manual_seed(seed)
    psi = torch.complex(
        torch.randn(2**n_qubits, generator=g, dtype=torch.float64),
        torch.randn(2**n_qubits, generator=g, dtype=torch.float64),
    )
    return psi / torch.linalg.vector_norm(psi)


def _kraus_product_prob(model, seq):
    """p(seq) = Tr(sum_e K_yT,eT ... K_y1,e1 rho0 K^dagger ... ) by explicit loops."""
    k = model.instrument().kraus.detach()
    rho = model.rho0
    for y in seq.tolist():
        rho = sum(k[y, e] @ rho @ k[y, e].conj().T for e in range(k.shape[1]))
    return float(trace(rho).real)


def _deferred_measurement_probs(model, psi0, length):
    """Joint p(y_1..y_T) from one unitary on memory + T fresh ancilla registers.

    All ancillas are measured at the end (deferred measurement principle);
    env ancillas and the memory are traced out.
    """
    u = model.unitary().detach()
    nm, no, ne = model.n_memory, model.n_output, model.n_env
    na = no + ne
    n_total = nm + length * na
    psi = torch.zeros(2**n_total, dtype=DTYPE)
    # |psi0>_mem ⊗ |0...0>_ancillas
    psi[torch.arange(2**nm) * 2 ** (length * na)] = psi0
    for t in range(length):
        qubits = list(range(nm)) + list(range(nm + t * na, nm + (t + 1) * na))
        psi = embed(u, qubits, n_total) @ psi
    amp2 = psi.abs().pow(2).reshape([2**nm] + [2**no, 2**ne] * length)
    dims_to_sum = [0] + [2 + 2 * t for t in range(length)]
    return amp2.sum(dim=dims_to_sum).reshape(-1)  # index = big-endian y_1..y_T


BRUTE_CONFIGS = [
    ModelConfig(n_memory=2, n_output=1, n_env=1, env_strength=0.6, strength=0.9, seed=11),
    ModelConfig(n_memory=2, n_output=2, memory_depth=2, joint_depth=1, seed=12),
    ModelConfig(n_memory=3, n_output=1, memory_depth=2, strength=0.5, seed=13),
]


@pytest.mark.parametrize(
    "cfg", BRUTE_CONFIGS, ids=lambda c: f"m{c.n_memory}o{c.n_output}e{c.n_env}"
)
def test_teacher_forced_matches_deferred_measurement(cfg):
    length = 3 if cfg.n_output == 1 else 2
    psi0 = _random_ket(cfg.n_memory, seed=cfg.seed)
    model = build_model(cfg)
    model = QuantumMemoryModel(
        cfg.n_memory, cfg.n_output, list(model.circuit.blocks), n_env=cfg.n_env, initial=psi0
    )
    seqs = all_sequences(model.n_symbols, length)
    with torch.no_grad():
        steps = model.step_log_probs(seqs)
        total = model.log_prob(seqs)
    assert torch.allclose(total, steps.sum(-1), atol=ATOL)
    brute = _deferred_measurement_probs(model, psi0, length)
    assert torch.allclose(total.exp(), brute, atol=ATOL)
    for i in (0, len(seqs) // 2, len(seqs) - 1):
        assert math.isclose(
            float(total[i].exp()), _kraus_product_prob(model, seqs[i]), abs_tol=ATOL
        )


@pytest.mark.parametrize("initial", ["zero", "plus", "mixed", "random_pure"])
def test_all_sequences_sum_to_one(model, initial):
    model = QuantumMemoryModel(
        model.n_memory,
        model.n_output,
        list(model.circuit.blocks),
        n_env=model.n_env,
        initial=initial,
        generator=torch.Generator().manual_seed(0),
    )
    length = 4 if model.n_symbols == 2 else 2
    with torch.no_grad():
        p = model.log_prob(all_sequences(model.n_symbols, length)).exp()
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-11)


def test_kraus_product_matches_for_mixed_initial_state():
    cfg = ModelConfig(n_memory=2, n_output=1, n_env=1, env_strength=0.8, initial="mixed", seed=21)
    model = build_model(cfg)
    seqs = all_sequences(2, 4)
    with torch.no_grad():
        p = model.log_prob(seqs).exp()
    for i in range(len(seqs)):
        assert math.isclose(float(p[i]), _kraus_product_prob(model, seqs[i]), abs_tol=ATOL)


def test_memory_stays_valid_and_batching_is_consistent(model):
    seqs = model.sample(16, 6, seed=3)
    with torch.no_grad():
        batched = model.log_prob(seqs)
        single = torch.stack([model.log_prob(s[None])[0] for s in seqs])
        mem = model.final_memory(seqs)
    assert torch.allclose(batched, single, atol=1e-13)
    assert is_density_matrix(mem.rho, atol=1e-9)


def test_predictive_probs_consistent_with_step_log_probs(model):
    seqs = model.sample(8, 5, seed=4)
    with torch.no_grad():
        pred = model.predictive_probs(seqs)
        steps = model.step_log_probs(seqs)
    assert pred.shape == (8, 6, model.n_symbols)
    assert torch.allclose(pred.sum(-1), torch.ones(8, 6, dtype=pred.dtype), atol=1e-12)
    gathered = pred[:, :-1].gather(-1, seqs[..., None]).squeeze(-1)
    assert torch.allclose(gathered.log(), steps, atol=1e-10)


def test_unconditioned_memory_is_average_of_conditioned_memories():
    model = build_model(ModelConfig(n_memory=2, n_output=1, n_env=1, env_strength=0.5, seed=31))
    length = 4
    seqs = all_sequences(2, length)
    with torch.no_grad():
        p = model.log_prob(seqs).exp()
        rho_cond = model.final_memory(seqs).rho
        avg = (p[:, None, None] * rho_cond).sum(0)
        assert torch.allclose(avg, model.unconditioned_memory(length), atol=ATOL)


def test_memory_carries_information_between_measurements():
    model = build_model(ModelConfig(n_memory=2, n_output=1, strength=0.8, seed=41))
    with torch.no_grad():
        pred = model.predictive_probs(torch.tensor([[0], [1]]))[:, 1]
    assert (pred[0] - pred[1]).abs().max() > 1e-3


def test_gradients_match_finite_differences():
    cfg = ModelConfig(
        n_memory=2,
        n_output=1,
        n_env=1,
        env_strength=0.4,
        memory_depth=2,
        joint_depth=1,
        strength=0.6,
        strength_trainable=True,
        seed=51,
    )
    model = build_model(cfg)
    seqs = model.sample(6, 5, seed=5)

    def loss():
        return -model.log_prob(seqs).mean()

    model.zero_grad()
    loss().backward()
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) > 10
    eps = 1e-6
    for p in params:
        flat = p.data.view(-1)
        grad = p.grad.view(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + eps
                up = loss().item()
                flat[i] = orig - eps
                down = loss().item()
                flat[i] = orig
            fd = (up - down) / (2 * eps)
            assert math.isclose(grad[i].item(), fd, rel_tol=1e-6, abs_tol=1e-8), (i, grad[i], fd)


def _chi_square(counts, expected):
    """Pearson chi-square with cells of expected count < 5 pooled; returns (stat, dof)."""
    order = torch.argsort(expected)
    exp_cells, obs_cells, acc_e, acc_o = [], [], 0.0, 0.0
    for i in order.tolist():
        acc_e += float(expected[i])
        acc_o += float(counts[i])
        if acc_e >= 5:
            exp_cells.append(acc_e)
            obs_cells.append(acc_o)
            acc_e = acc_o = 0.0
    if acc_e > 0:
        exp_cells[-1] += acc_e
        obs_cells[-1] += acc_o
    stat = sum((o - e) ** 2 / e for o, e in zip(obs_cells, exp_cells, strict=True))
    return stat, len(exp_cells) - 1


# 99.9th percentiles of chi-square for the degrees of freedom used below.
CHI2_999 = {7: 24.32, 15: 37.70}


@pytest.mark.parametrize(
    ("cfg", "length"),
    [
        (ModelConfig(n_memory=2, n_output=1, strength=0.9, seed=61), 4),
        (ModelConfig(n_memory=2, n_output=2, n_env=1, env_strength=0.7, seed=62), 2),
    ],
)
def test_sampler_frequencies_match_exact_probabilities(cfg, length):
    model = build_model(cfg)
    n = 40_000
    samples = model.sample(n, length, seed=2024)
    seqs = all_sequences(model.n_symbols, length)
    powers = model.n_symbols ** torch.arange(length - 1, -1, -1)
    counts = torch.bincount((samples * powers).sum(-1), minlength=len(seqs)).double()
    with torch.no_grad():
        expected = model.log_prob(seqs).exp() * n
    stat, dof = _chi_square(counts, expected)
    assert dof in CHI2_999, dof
    assert stat < CHI2_999[dof], (stat, dof)


def test_sampler_is_seeded():
    model = build_model(ModelConfig(seed=71))
    assert torch.equal(model.sample(50, 8, seed=1), model.sample(50, 8, seed=1))
    assert not torch.equal(model.sample(50, 8, seed=1), model.sample(50, 8, seed=2))


# --- probability safeguards -------------------------------------------------------


def _projective_one_qubit(**kwargs):
    block = ControlledRotationBlock([(0, 1)], n_qubits=2, strength=math.pi / 2)
    return QuantumMemoryModel(1, 1, [block], initial="zero", **kwargs)


def test_impossible_token_gives_minus_inf_without_nan(caplog):
    model = _projective_one_qubit()
    with torch.no_grad():
        steps = model.step_log_probs(torch.tensor([[1, 0, 0], [0, 0, 0]]))
    assert steps[0, 0] == -math.inf
    assert torch.isfinite(steps[0, 1:]).all()
    assert torch.allclose(steps[1], torch.zeros(3, dtype=steps.dtype))
    assert model.diagnostics["zero_prob_steps"] == 1
    assert model.diagnostics["floored_steps"] == 0


def test_prob_floor_is_explicit_counted_and_logged(caplog):
    model = _projective_one_qubit(prob_floor=1e-6)
    with caplog.at_level(logging.WARNING, logger="qmem"), torch.no_grad():
        steps = model.step_log_probs(torch.tensor([[1, 0]]))
    assert math.isclose(float(steps[0, 0]), math.log(1e-6))
    assert model.diagnostics["floored_steps"] == 1
    assert any("prob_floor" in r.message for r in caplog.records)


def test_tiny_probability_is_logged(caplog):
    g = 1e-6  # p(1 | |1>) = sin^2 g ≈ 1e-12 < tiny_prob
    block = ControlledRotationBlock([(0, 1)], n_qubits=2, strength=g)
    model = QuantumMemoryModel(1, 1, [block], initial=torch.tensor([0, 1], dtype=DTYPE))
    with caplog.at_level(logging.WARNING, logger="qmem"), torch.no_grad():
        steps = model.step_log_probs(torch.tensor([[1]]))
    assert math.isclose(float(steps[0, 0]), math.log(math.sin(g) ** 2), rel_tol=1e-9)
    assert model.diagnostics["tiny_prob_steps"] == 1
    assert caplog.records


def test_invalid_tokens_rejected():
    model = build_model(ModelConfig())
    with pytest.raises(ValueError):
        model.log_prob(torch.tensor([[0, 2]]))


def test_config_roundtrip():
    cfg = ModelConfig(n_memory=3, rotations=("rx", "rz"), prob_floor=1e-9)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"bogus": 1})


def test_initial_ket_matches_dm():
    psi = _random_ket(2, 0)
    model = QuantumMemoryModel(2, 1, [ControlledRotationBlock([(0, 2)], 3, 0.3)], initial=psi)
    assert torch.allclose(model.rho0, ket_to_dm(psi), atol=1e-14)
