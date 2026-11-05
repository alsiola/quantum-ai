"""Single-step instrument tests: structure and analytically solvable cases."""

import math

import pytest
import torch

from qmem import ControlledRotationBlock, FixedUnitaryBlock, Instrument, QuantumMemoryModel
from qmem.gates import CNOT
from qmem.linalg import DTYPE, eye, ket_to_dm
from qmem.states import is_density_matrix, min_eigenvalue, random_density_matrix

ATOL = 1e-12


def _random_states(n_memory, gen, n=6):
    states = [random_density_matrix(n_memory, gen) for _ in range(n - 2)]
    states.append(random_density_matrix(n_memory, gen, rank=1))
    states.append(torch.eye(2**n_memory, dtype=DTYPE) / 2**n_memory)
    return torch.stack(states)


# --- structural properties for random parameters ------------------------------


def test_unitary_is_unitary(model):
    u = model.unitary()
    assert torch.allclose(u @ u.conj().T, eye(u.shape[0]), atol=ATOL)


def test_completeness(model):
    inst = model.instrument()
    assert inst.n_outcomes == model.n_symbols
    assert inst.n_kraus == 2**model.n_env
    assert float(inst.completeness_defect().detach()) < ATOL


def test_each_outcome_is_trace_nonincreasing(model):
    eff = model.instrument().effects()
    ev = torch.linalg.eigvalsh(eff)
    assert (ev > -ATOL).all() and (ev < 1 + ATOL).all()


def test_channel_trace_preserving_and_positive(model, gen):
    rho = _random_states(model.n_memory, gen)
    out = model.instrument().channel(rho)
    assert is_density_matrix(out, atol=1e-11)


def test_probabilities_valid(model, gen):
    rho = _random_states(model.n_memory, gen)
    p = model.instrument().probabilities(rho)
    assert p.shape == (rho.shape[0], model.n_symbols)
    assert (p >= -ATOL).all() and (p <= 1 + ATOL).all()
    assert torch.allclose(p.sum(-1), torch.ones(rho.shape[0], dtype=p.dtype), atol=ATOL)


def test_conditional_states_valid_and_probs_consistent(model, gen):
    inst = model.instrument()
    rho = _random_states(model.n_memory, gen)
    p_all = inst.probabilities(rho)
    for y in range(inst.n_outcomes):
        state, p = inst.condition(rho, y)
        assert torch.allclose(p, p_all[:, y], atol=ATOL)
        ok = p > 1e-9
        assert is_density_matrix(state[ok], atol=1e-9)


def test_outcome_average_equals_channel(model, gen):
    inst = model.instrument()
    rho = _random_states(model.n_memory, gen)
    avg = torch.zeros_like(rho)
    for y in range(inst.n_outcomes):
        state, p = inst.condition(rho, y)
        avg = avg + p[:, None, None] * torch.nan_to_num(state)
    assert torch.allclose(avg, inst.channel(rho), atol=1e-11)
    assert torch.allclose(inst.branches(rho).sum(-3), inst.channel(rho), atol=1e-11)


def test_branch_accepts_per_item_outcomes(model, gen):
    inst = model.instrument()
    rho = _random_states(model.n_memory, gen)
    y = torch.arange(rho.shape[0]) % inst.n_outcomes
    batched = inst.branch(rho, y)
    for b in range(rho.shape[0]):
        assert torch.allclose(batched[b], inst.branch(rho[b], int(y[b])), atol=ATOL)


# --- analytically solvable cases ------------------------------------------------

PLUS = torch.tensor([1, 1], dtype=DTYPE) / math.sqrt(2)


def _weak_instrument(g):
    """One memory qubit, one output ancilla, controlled-RY(2g) memory -> ancilla."""
    block = ControlledRotationBlock([(0, 1)], n_qubits=2, strength=g)
    return Instrument.from_unitary(block.unitary(), n_memory=1, n_output=1)


def test_cnot_on_plus_collapses_projectively():
    u = FixedUnitaryBlock(CNOT, (0, 1), 2).unitary()
    inst = Instrument.from_unitary(u, 1, 1)
    rho = ket_to_dm(PLUS)
    p = inst.probabilities(rho)
    assert torch.allclose(p, torch.tensor([0.5, 0.5], dtype=p.dtype), atol=ATOL)
    s0, _ = inst.condition(rho, 0)
    s1, _ = inst.condition(rho, 1)
    assert torch.allclose(s0, torch.tensor([[1, 0], [0, 0]], dtype=DTYPE), atol=ATOL)
    assert torch.allclose(s1, torch.tensor([[0, 0], [0, 1]], dtype=DTYPE), atol=ATOL)
    # Unconditioned: full dephasing.
    assert torch.allclose(inst.channel(rho), torch.eye(2, dtype=DTYPE) / 2, atol=ATOL)


def test_projective_strength_equals_cnot():
    a = _weak_instrument(math.pi / 2).kraus
    b = Instrument.from_unitary(FixedUnitaryBlock(CNOT, (0, 1), 2).unitary(), 1, 1).kraus
    assert torch.allclose(a, b, atol=ATOL)


@pytest.mark.parametrize("g", [0.0, 0.2, 0.7, 1.2, math.pi / 2])
def test_weak_measurement_matches_hand_derivation(g):
    inst = _weak_instrument(g)
    k = inst.kraus[:, 0]
    c, s = math.cos(g), math.sin(g)
    assert torch.allclose(k[0], torch.tensor([[1, 0], [0, c]], dtype=DTYPE), atol=ATOL)
    assert torch.allclose(k[1], torch.tensor([[0, 0], [0, s]], dtype=DTYPE), atol=ATOL)

    rho = ket_to_dm(PLUS)
    p = inst.probabilities(rho)
    assert math.isclose(float(p[0]), (1 + c**2) / 2, abs_tol=ATOL)
    assert math.isclose(float(p[1]), s**2 / 2, abs_tol=ATOL)

    # Outcome 0: state ∝ |0> + cos g |1>.
    s0, _ = inst.condition(rho, 0)
    n = 1 + c**2
    expected0 = torch.tensor([[1 / n, c / n], [c / n, c**2 / n]], dtype=DTYPE)
    assert torch.allclose(s0, expected0, atol=ATOL)
    if s > 0:
        s1, _ = inst.condition(rho, 1)
        assert torch.allclose(s1, torch.tensor([[0, 0], [0, 1]], dtype=DTYPE), atol=ATOL)

    # Unconditioned channel: populations kept, coherence multiplied by cos g.
    out = inst.channel(rho)
    assert torch.allclose(out, torch.tensor([[0.5, c / 2], [c / 2, 0.5]], dtype=DTYPE), atol=ATOL)


def test_zero_strength_leaves_memory_untouched(gen):
    inst = _weak_instrument(0.0)
    rho = random_density_matrix(1, gen)
    p = inst.probabilities(rho)
    assert torch.allclose(p, torch.tensor([1.0, 0.0], dtype=p.dtype), atol=ATOL)
    s0, _ = inst.condition(rho, 0)
    assert torch.allclose(s0, rho, atol=ATOL)
    assert torch.allclose(inst.channel(rho), rho, atol=ATOL)


def test_four_symbol_readout_reads_computational_basis(gen):
    # CNOT from memory qubit k to output ancilla k: y is the memory bitstring.
    n = 4
    blocks = [FixedUnitaryBlock(CNOT, (0, 2), n), FixedUnitaryBlock(CNOT, (1, 3), n)]
    model = QuantumMemoryModel(2, 2, blocks)
    inst = model.instrument()
    rho = random_density_matrix(2, gen)
    p = inst.probabilities(rho)
    assert torch.allclose(p, torch.diagonal(rho).real, atol=ATOL)
    for y in range(4):
        s, _ = inst.condition(rho, y)
        expected = torch.zeros(4, 4, dtype=DTYPE)
        expected[y, y] = 1
        assert torch.allclose(s, expected, atol=ATOL)


def test_traced_env_ancilla_dephases_without_output(gen):
    # Output ancilla untouched (always y=0); env ancilla gets a CNOT and is traced:
    # two Kraus operators for outcome 0, the conditional map is full dephasing.
    blocks = [FixedUnitaryBlock(CNOT, (0, 2), 3)]
    model = QuantumMemoryModel(1, 1, blocks, n_env=1)
    inst = model.instrument()
    assert inst.n_kraus == 2
    rho = random_density_matrix(1, gen)
    p = inst.probabilities(rho)
    assert torch.allclose(p, torch.tensor([1.0, 0.0], dtype=p.dtype), atol=ATOL)
    s0, _ = inst.condition(rho, 0)
    assert torch.allclose(s0, torch.diag(torch.diagonal(rho)), atol=ATOL)
    assert float(min_eigenvalue(s0)) > -ATOL
