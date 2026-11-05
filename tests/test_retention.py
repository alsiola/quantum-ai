"""Phase 3e (issue #10): the marked recall probe, its scoring and the retention diagnostics."""

import math

import pytest
import torch

from qmem.benchmarks import (
    feedforward_eta,
    feedforward_g_for_eta,
    feedforward_readout,
    feedforward_strength_floor,
    marked_recall_metrics,
)
from qmem.complexity import memory_profile, qmachine_instrument, qmachine_state
from qmem.data import all_sequences
from qmem.diagnostics import label_retention, stored_bit_alignment
from qmem.instrument import Instrument
from qmem.linalg import DTYPE
from qmem.model import ModelConfig, build_model
from qmem.tasks import (
    CUE0,
    CUE1,
    DISTRACTOR_A,
    DISTRACTOR_B,
    QuantumSourceTask,
    delayed_recall,
    delayed_recall_events,
    iid_task,
    make_task,
    marked_recall,
    marked_recall_events,
)

LN2 = math.log(2)


def _h(*p):
    return -sum(x * math.log(x) for x in p if x > 0)


class _SourceModel:
    """Wrap a source so it can be scored like a model."""

    def __init__(self, task):
        self.task = task

    def step_log_probs(self, s):
        return self.task.log_probs(s)

    def predictive_probs(self, s):
        return self.task.predictive(s)


@pytest.mark.parametrize("tau", [0, 1, 2, 4, 8, 16, 32, 2.5])
def test_geometric_marked_recall_has_three_causal_states(tau):
    for eps in (0.0, 0.05, 0.1):
        prof = memory_profile(marked_recall(tau, eps))
        assert prof["n_causal_states"] == 3
        assert prof["c_mu_bits"] == pytest.approx(1.5, abs=1e-12)  # pi = (1/2, 1/4, 1/4)


@pytest.mark.parametrize("tau", [0, 1, 3, 8])
def test_fixed_marked_recall_still_needs_a_clock(tau):
    assert (
        memory_profile(marked_recall(tau, 0.05, timing="fixed"))["n_causal_states"] == 2 * tau + 3
    )


@pytest.mark.parametrize("tau,eps", [(1, 0.0), (4, 0.05), (16, 0.1)])
def test_geometric_entropy_rate_closed_form(tau, eps):
    q = 1 / (tau + 1)
    # half the time in S (cue value uniform), half in W (recall value H(eps)); the type
    # is i.i.d. Bernoulli(q) and the distractor value is uniform everywhere
    rate = _h(q, 1 - q) + (1 - q) * LN2 + q * (LN2 + _h(eps, 1 - eps)) / 2
    assert marked_recall(tau, eps).entropy_rate() == pytest.approx(rate, abs=1e-10)


@pytest.mark.parametrize("timing", ["geometric", "fixed"])
def test_marked_optimum_matches_brute_force_and_monte_carlo(timing):
    task = marked_recall(2, 0.1, timing=timing)
    length = 5
    seqs = all_sequences(4, length)
    lp = task.log_probs(seqs).sum(-1)
    p = torch.exp(lp)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-12)
    joint = float(torch.where(p > 0, -p * lp, torch.zeros_like(p)).sum())
    assert task.optimal_nll(length) == pytest.approx(joint / length, abs=1e-12)
    # Monte Carlo of the expected NLL at a longer length
    length = 40
    sample = task.sample(4000, length, seed=3)
    per = -task.log_probs(sample).mean(-1)
    se = float(per.std()) / math.sqrt(len(per))
    assert abs(float(per.mean()) - task.optimal_nll(length)) < 4 * se


def test_marked_sampler_type_is_iid_and_delay_is_geometric():
    tau = 3.0
    seq = marked_recall(tau, 0.0).sample(400, 1000, seed=1)  # long: little censoring
    is_cue = seq >= CUE0
    assert float(is_cue.double().mean()) == pytest.approx(1 / (tau + 1), abs=0.005)
    ev = marked_recall_events(seq)
    delays = ev["delay"][ev["recall"]].double()
    assert float(delays.mean()) == pytest.approx(tau, rel=0.03)
    # with eps = 0 every recall repeats its cue
    rec_val = (seq[ev["recall"]] == CUE1).long()
    assert torch.equal(rec_val, ev["cue_value"][ev["recall"]])


def test_marked_recall_events_by_hand():
    a, b, c0, c1 = DISTRACTOR_A, DISTRACTOR_B, CUE0, CUE1
    seq = torch.tensor([[a, c1, a, b, c1, c0, c0, b]])
    ev = marked_recall_events(seq)
    assert ev["cue"].tolist() == [[False, True, False, False, False, True, False, False]]
    assert ev["recall"].tolist() == [[False, False, False, False, True, False, True, False]]
    assert ev["cue_value"].tolist() == [[-1, -1, 1, 1, 1, -1, 0, -1]]
    assert ev["lag"].tolist() == [[-1, -1, 1, 2, 3, -1, 1, -1]]
    assert ev["delay"].tolist() == [[-1, -1, -1, -1, 2, -1, 0, -1]]


def test_delayed_recall_events():
    seq = delayed_recall(2).sample(3, 9, seed=0)
    ev = delayed_recall_events(2, seq)
    assert ev["lag"][0].tolist() == [-1, 1, 2, 3, -1, 1, 2, 3, -1]
    assert torch.equal(ev["cue_value"][:, 3], seq[:, 0])
    assert torch.equal(ev["cue_value"][:, 3], seq[:, 3])  # eps = 0: the recall is the cue


def test_marked_recall_metrics_exact_and_memoryless():
    task = marked_recall(4, 0.05)
    seq = task.sample(400, 96, seed=2)
    exact = marked_recall_metrics(_SourceModel(task), task, seq)
    for k in ("gap_all_nats", "gap_type_nats", "gap_value_distractor_nats", "gap_non_recall_nats"):
        assert exact[k] == pytest.approx(0.0, abs=1e-12)
    assert exact["recall_value_nll_nats"] == exact["source_recall_value_nll_nats"]
    assert exact["recall_accuracy"] == pytest.approx(0.95, abs=0.02)
    # a memoryless model with the right marginals: chance at the recall, no type gap
    q = 1 / 5
    memoryless = iid_task([(1 - q) / 2, (1 - q) / 2, q / 2, q / 2])
    m = marked_recall_metrics(_SourceModel(memoryless), task, seq)
    assert m["recall_value_nll_nats"] == pytest.approx(LN2, abs=1e-12)
    assert m["gap_type_nats"] == pytest.approx(0.0, abs=1e-12)
    assert m["gap_value_distractor_nats"] == pytest.approx(0.0, abs=1e-12)
    assert m["gap_value_recall_nats"] == pytest.approx(LN2 - _h(0.05, 0.95), abs=0.03)
    total = sum(b["n"] for b in m["by_delay"])
    assert total == m["n_recalls"]


@pytest.mark.parametrize("tau", [1, 4, 16])
def test_qmachine_solves_marked_recall_on_three_levels(tau):
    task = marked_recall(tau, 0.05)
    inst, _ = qmachine_instrument(task)
    assert inst.dim == 3
    seq = task.sample(64, 64, seed=5)
    src = QuantumSourceTask(inst, initial=qmachine_state(task, 0))
    assert float((src.predictive(seq) - task.predictive(seq)).abs().max()) < 1e-10


def test_feedforward_eta_closed_forms():
    for g in (0.2, 0.7, 1.1, math.pi / 2):
        s2, c2 = math.sin(g) ** 2, math.cos(g) ** 2
        # two couplings on different memory qubits: sum lambda_max = (1 + s^2)^2, d = 4
        assert feedforward_eta(2, 2, g) == pytest.approx(((1 + s2) ** 2 - 1) / 3, abs=1e-12)
        assert feedforward_eta(3, 2, g) == pytest.approx(((1 + s2) ** 2 - 1) / 3, abs=1e-12)
        # both couplings on the one memory qubit: 1 + 2 c^2 s^2 + s^4 - 1 = 1 - c^4
        assert feedforward_eta(1, 2, g) == pytest.approx(1 - c2**2, abs=1e-12)
        assert feedforward_eta(2, 1, g) == pytest.approx(s2, abs=1e-12)
    for n in (1, 2, 3):
        for eta in (0.3, 0.75, 0.97):
            assert feedforward_eta(n, 2, feedforward_g_for_eta(n, 2, eta)) == pytest.approx(
                eta, abs=1e-9
            )


def test_feedforward_floor_closed_forms():
    # binary recall: sin^2 g >= 1 - eps (and cos^2 g <= eps), so eta_floor = 1 - eps
    for eps in (0.05, 0.1):
        out = feedforward_strength_floor(delayed_recall(4, eps), 2, 1)
        assert out["eta_floor"] == pytest.approx(1 - eps, abs=1e-8)
    # one memory qubit read by both ancillas: E_01 = E_10, and at g = pi/2 they vanish
    assert not feedforward_strength_floor(marked_recall(4, 0.05), 1, 2)["admissible"]
    assert feedforward_strength_floor(marked_recall(4, 0.05), 2, 2)["eta_floor"] < 0.5


def test_shared_strength_on_one_qubit_ties_the_mixed_outcomes():
    for g in (0.4, 1.0):
        cfg = ModelConfig(n_memory=1, n_output=2, feedforward_depth=2, strength=g, seed=3)
        cfg = ModelConfig(**{**cfg.to_dict(), "strength_shared": True, "rotations": ("ry", "rz")})
        e = build_model(cfg).instrument().effects()
        assert torch.allclose(e[1], e[2], atol=1e-12)


def test_projective_feedforward_on_two_qubits_forgets_everything():
    # g = pi/2 with every memory qubit read: each K_y is rank one, so the memory after
    # a step depends on the outcome only
    cfg = ModelConfig(n_memory=2, n_output=2, feedforward_depth=2, strength=math.pi / 2, seed=1)
    cfg = ModelConfig(**{**cfg.to_dict(), "strength_shared": True, "rotations": ("ry", "rz")})
    inst = build_model(cfg).instrument()
    gen = torch.Generator().manual_seed(0)
    for y in range(4):
        assert int(torch.linalg.matrix_rank(inst.kraus[y, 0])) == 1
        outs = []
        for _ in range(3):
            v = torch.randn(4, dtype=DTYPE, generator=gen)
            rho = torch.outer(v, v.conj()) / v.norm() ** 2
            post, p = inst.condition(rho, y)
            if p > 1e-9:
                outs.append(post)
        for post in outs[1:]:
            assert torch.allclose(post, outs[0], atol=1e-9)


def test_label_retention_of_the_exact_model_and_a_memoryless_one():
    task = marked_recall(3, 0.0)
    inst, _ = qmachine_instrument(task)
    seq = task.sample(300, 48, seed=4)
    ev = marked_recall_events(seq)
    ret = label_retention(inst, qmachine_state(task, 0), seq, ev["cue_value"], ev["lag"], 6)
    for k in range(1, 7):
        assert ret[k]["holevo_bits"] == pytest.approx(1.0, abs=1e-2)  # = H(cue frequencies)
        assert ret[k]["trace_distance"] == pytest.approx(1.0, abs=1e-9)
    # an instrument whose Kraus operators all reset to |0> remembers nothing
    k = torch.zeros(4, 1, 2, 2, dtype=DTYPE)
    for y in range(4):
        k[y, 0, 0, 0] = 0.5
        k[y, 0, 0, 1] = 0.5
    reset = Instrument(k)
    rho = torch.eye(2, dtype=DTYPE) / 2
    ret = label_retention(reset, rho, seq, ev["cue_value"], ev["lag"], 3)
    assert all(abs(r["holevo_bits"]) < 1e-12 for r in ret.values())


def test_stored_bit_alignment_of_a_protected_and_an_erasing_readout():
    # one qubit: the distractors are proportional to the identity (a decoherence-free
    # readout), the cues write and read the bit in Z
    q = 0.25
    k = torch.zeros(4, 1, 2, 2, dtype=DTYPE)
    k[DISTRACTOR_A, 0] = math.sqrt((1 - q) / 2) * torch.eye(2, dtype=DTYPE)
    k[DISTRACTOR_B, 0] = math.sqrt((1 - q) / 2) * torch.eye(2, dtype=DTYPE)
    k[CUE0, 0, 0, 0] = math.sqrt(q)
    k[CUE1, 0, 1, 1] = math.sqrt(q)
    protected = Instrument(k)
    assert float(protected.completeness_defect()) < 1e-12
    seq = torch.tensor(
        [[CUE1, DISTRACTOR_A, DISTRACTOR_B, CUE1], [CUE0, DISTRACTOR_B, DISTRACTOR_A, CUE0]]
    )
    ev = marked_recall_events(seq)
    rho0 = torch.eye(2, dtype=DTYPE) / 2
    out = stored_bit_alignment(
        protected, rho0, seq, ev["cue_value"], ev["lag"], [DISTRACTOR_A, DISTRACTOR_B]
    )
    assert out["trace_distance"] == pytest.approx(1.0, abs=1e-12)
    for y in (DISTRACTOR_A, DISTRACTOR_B):
        row = out["symbols"][y]
        assert row["commutator"] == pytest.approx(0.0, abs=1e-12)
        assert row["prob_bias"] == pytest.approx(0.0, abs=1e-12)
        assert row["contraction"] == pytest.approx(1.0, abs=1e-12)
    # an X-basis projective distractor erases the Z-stored bit and commutes with nothing
    k2 = k.clone()
    plus = torch.tensor([1.0, 1.0], dtype=DTYPE) / math.sqrt(2)
    minus = torch.tensor([1.0, -1.0], dtype=DTYPE) / math.sqrt(2)
    k2[DISTRACTOR_A, 0] = math.sqrt(1 - q) * torch.outer(plus, plus)
    k2[DISTRACTOR_B, 0] = math.sqrt(1 - q) * torch.outer(minus, minus)
    erasing = Instrument(k2)
    assert float(erasing.completeness_defect()) < 1e-12
    out = stored_bit_alignment(erasing, rho0, seq, ev["cue_value"], ev["lag"], [DISTRACTOR_A])
    row = out["symbols"][DISTRACTOR_A]
    assert row["contraction"] == pytest.approx(0.0, abs=1e-12)
    assert row["commutator"] > 0.1


def test_make_task_builds_marked_recall():
    task = make_task({"kind": "marked_recall", "tau": 8, "eps": 0.05})
    assert task.n_symbols == 4 and task.n_states == 3
    readout = feedforward_readout(2, 2, 1.0)
    assert float(readout.completeness_defect()) < 1e-12
