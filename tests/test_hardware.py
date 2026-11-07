"""Qiskit export of architecture A and the exact (noisy) circuit simulation (issue #14).

Needs the optional ``hardware`` group (``uv sync --group hardware``); skipped without it.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import transpile
from qiskit.quantum_info import Statevector

from qmem import hardware as hw
from qmem.model import ModelConfig, build_model

LAYOUTS = {
    "general_n1": dict(n_memory=1, n_output=1, memory_depth=2, joint_depth=2),
    "general_n2_4sym": dict(n_memory=2, n_output=2, memory_depth=2, joint_depth=2),
    "ff_n2": dict(n_memory=2, n_output=1, feedforward_depth=2, strength_shared=True, strength=0.5),
    "ff_n2_4sym": dict(n_memory=2, n_output=2, feedforward_depth=1, strength=0.7),
    "ff_n3_cz": dict(n_memory=3, n_output=1, feedforward_depth=2, entangler="cz", strength=0.9),
    "general_env": dict(n_memory=2, n_output=1, n_env=1, env_strength=0.4, joint_depth=1),
}


def model(name: str, seed: int = 3):
    return build_model(ModelConfig(seed=seed, **LAYOUTS[name]))


def all_seqs(n_symbols: int, length: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(n_symbols), repeat=length)))


@pytest.mark.parametrize("name", list(LAYOUTS))
@pytest.mark.parametrize("length", [1, 2, 3])
@torch.no_grad()
def test_export_matches_qmem_exactly(name: str, length: int) -> None:
    """Outcome distribution and conditional memory of the dynamic circuit equal qmem's."""
    m = model(name)
    seqs = all_seqs(m.n_symbols, length)
    p = m.log_prob(seqs).exp().numpy()
    rho = m.final_memory(seqs).rho.numpy()
    modes = ("dynamic", "coherent") if "ff" in name else ("dynamic",)
    for mode in modes:
        br = hw.sequence_branches(hw.export_circuit(m, length, mode), m)
        assert set(br) == {tuple(s) for s in seqs.tolist()}
        for i, s in enumerate(seqs.tolist()):
            pq, rq = br[tuple(s)]
            assert abs(pq - p[i]) < 1e-12
            if p[i] > 1e-12:
                assert np.abs(rq - rho[i]).max() < 1e-10


@pytest.mark.parametrize("name", ["general_n1", "ff_n2_4sym", "general_env"])
@torch.no_grad()
def test_deferred_measurement_statevector(name: str) -> None:
    """The deferred form (fresh ancillas, no measurement) gives the same joint distribution."""
    m = model(name)
    length = 3 if m.n_output == 1 else 2
    seqs = all_seqs(m.n_symbols, length)
    p = m.log_prob(seqs).exp().numpy()
    flat = [q for o in hw.output_qubits_deferred(m, length) for q in o]
    probs = Statevector(hw.export_deferred(m, length)).probabilities(flat)
    for i, s in enumerate(seqs.tolist()):
        bits = [(y >> (m.n_output - 1 - k)) & 1 for y in s for k in range(m.n_output)]
        assert abs(probs[sum(b << j for j, b in enumerate(bits))] - p[i]) < 1e-12


def test_circuit_structure() -> None:
    """Measurements, resets between steps and classical control as specified."""
    m = model("ff_n2_4sym")
    qc = hw.export_circuit(m, 4)
    ops = hw.count_ops(qc)
    assert ops["measure"] == 4 * m.n_output
    assert ops["reset"] == 3 * (m.n_output + m.n_env)
    assert ops["if_else"] == 4 * m.n_symbols
    assert [r.name for r in qc.cregs] == ["y0", "y1", "y2", "y3"]
    assert [r.name for r in qc.qregs] == ["mem", "out"]
    general = hw.export_circuit(model("general_n1"), 3)
    assert "if_else" not in hw.count_ops(general)
    assert hw.count_ops(hw.export_step(m))["reset"] == m.n_output


def test_aer_samples_the_dynamic_circuit() -> None:
    """Aer's density-matrix simulator agrees with qmem on the dynamic feed-forward circuit."""
    from qiskit_aer import AerSimulator

    m = model("ff_n2_4sym")
    qc = hw.export_circuit(m, 2)
    # shot branching is fast and exact without noise (it mis-applies mid-circuit readout
    # errors in qiskit-aer 0.18.2, so the noisy cross-check in phase5a.py does not use it)
    sim = AerSimulator(method="density_matrix", shot_branching_enable=True)
    n = 40_000
    counts = sim.run(transpile(qc, sim), shots=n, seed_simulator=5).result().get_counts()
    emp: dict[tuple[int, ...], int] = {}
    for k, v in counts.items():
        toks = tuple(int(r, 2) for r in reversed(k.split(" ")))
        emp[toks] = emp.get(toks, 0) + v
    seqs = all_seqs(m.n_symbols, 2)
    with torch.no_grad():
        p = m.log_prob(seqs).exp().numpy()
    chi2 = sum(
        (emp.get(tuple(s), 0) - n * p[i]) ** 2 / (n * p[i]) for i, s in enumerate(seqs.tolist())
    )
    assert chi2 < 50  # 15 dof: p ~ 1e-5


@torch.no_grad()
def test_noiseless_instrument_of_transpiled_step() -> None:
    """The instrument of the transpiled, routed step equals the model's."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        be = FakeKingston()
    for name in ("general_n2_4sym", "ff_n3_cz"):
        m = model(name)
        tq = transpile(hw.export_step(m), be, optimization_level=3, seed_transpiler=0)
        ini = tq.layout.initial_index_layout(filter_ancillas=True)
        fin = tq.layout.final_index_layout(filter_ancillas=True)
        cq, phys = hw.compact(tq)
        pos = {p: k for k, p in enumerate(phys)}
        inst = hw.noisy_instrument(
            cq,
            m.n_memory,
            m.n_output,
            None,
            [pos[ini[i]] for i in range(m.n_memory)],
            memory_qubits_out=[pos[fin[i]] for i in range(m.n_memory)],
        )
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(4, m.memory_dim, m.memory_dim, dtype=torch.complex128, generator=gen)
        rho = a @ a.mH
        rho = rho / rho.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        assert (inst.branches(rho) - m.instrument().branches(rho)).abs().max() < 1e-10
        assert inst.completeness_defect() < 1e-10


def _projective_qubit(g: float = math.pi / 2):
    """n = 1, no memory layers: only the coupling CRY(2g) memory -> ancilla."""
    return build_model(ModelConfig(n_memory=1, n_output=1, memory_depth=0, strength=g))


@torch.no_grad()
def test_readout_and_reset_errors_analytic() -> None:
    """Projective readout of a qubit: flips and a bad reset act as computed by hand."""
    m = _projective_qubit()
    step = hw.export_step(m)
    rho = torch.tensor([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=torch.complex128)
    eps = 0.05
    inst = hw.noisy_instrument(step, 1, 1, hw.StepNoise(readout=eps), [0])
    p = inst.probabilities(rho)
    assert abs(float(p[1]) - (0.3 * (1 - eps) + 0.7 * eps)) < 1e-12
    # recorded 1: (1 - eps) |1><1| rho |1><1| + eps |0><0| rho |0><0|
    b1 = inst.branch(rho, 1)
    want = torch.diag(torch.tensor([0.7 * eps, 0.3 * (1 - eps)], dtype=torch.complex128))
    assert (b1 - want).abs().max() < 1e-12
    # reset error r: the ancilla starts in |1>, CNOT flips it: recorded bit = memory bit XOR 1
    r = 0.1
    inst = hw.noisy_instrument(step, 1, 1, hw.StepNoise(reset=r), [0])
    assert abs(float(inst.probabilities(rho)[1]) - (0.3 * (1 - r) + 0.7 * r)) < 1e-12


@torch.no_grad()
def test_depolarising_gate_analytic() -> None:
    """Depolarising p after the coupling: Phi_y -> (1 - p) Phi_y + (p / 4) I."""
    m = _projective_qubit(0.6)
    p_dep = 0.08
    inst = hw.noisy_instrument(hw.export_step(m), 1, 1, hw.StepNoise(p2=p_dep), [0])
    rho = torch.tensor([[0.6, 0.3j], [-0.3j, 0.4]], dtype=torch.complex128)
    ideal = m.instrument()
    eye = torch.eye(2, dtype=torch.complex128)
    for y in (0, 1):
        want = (1 - p_dep) * ideal.branch(rho, y) + p_dep / 4 * eye
        assert (inst.branch(rho, y) - want).abs().max() < 1e-12


def test_relaxation_channel_analytic() -> None:
    t, t1, t2 = 2e-6, 100e-6, 60e-6
    ks = hw.relaxation_kraus(t, t1, t2)
    rho = np.array([[0.2, 0.3], [0.3, 0.8]], dtype=complex)
    out = sum(k @ rho @ k.conj().T for k in ks)
    assert abs(out[1, 1] - 0.8 * math.exp(-t / t1)) < 1e-12
    assert abs(out[0, 1] - 0.3 * math.exp(-t / t2)) < 1e-12
    assert abs(np.trace(out) - 1) < 1e-12


@torch.no_grad()
def test_idle_relaxation_during_readout() -> None:
    """The memory relaxes for the measurement and reset durations of each step."""
    m = build_model(ModelConfig(n_memory=1, n_output=1, memory_depth=0, strength=0.0))
    t1 = t2 = 50e-6
    noise = hw.StepNoise(t1=t1, t2=t2, measure_time=1e-6, reset_time=2e-6)
    inst = hw.noisy_instrument(hw.export_step(m), 1, 1, noise, [0])
    rho = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.complex128)
    out = inst.channel(rho)
    decay = math.exp(-3e-6 / t2)
    assert abs(float(out[0, 1].real) - 0.5 * decay) < 1e-12
    assert abs(float(out[1, 1].real) - 0.5 * math.exp(-3e-6 / t1)) < 1e-12


@torch.no_grad()
def test_instrument_model_reproduces_the_model() -> None:
    """``InstrumentModel`` on the noiseless exported step is the trained model."""
    m = model("ff_n2")
    inst = hw.noisy_instrument(hw.export_step(m), m.n_memory, m.n_output, None)
    im = hw.InstrumentModel(inst, m.n_memory, m.n_output)
    seqs = all_seqs(2, 6)
    assert (im.log_prob(seqs) - m.log_prob(seqs)).abs().max() < 1e-10
