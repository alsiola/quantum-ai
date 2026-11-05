"""Phase 3 benchmark suite: exact optima, memory figures, literature numbers, diagnostics."""

import math

import pytest
import torch

from qmem import Instrument, ModelConfig, build_model
from qmem.benchmarks import (
    LITERATURE,
    literature_checks,
    perturbed_coin_cq,
    recall_metrics,
)
from qmem.complexity import (
    epsilon_machine,
    memory_profile,
    pmd_qubit_instrument,
    qmachine_instrument,
    quantum_overlaps,
)
from qmem.data import all_sequences
from qmem.diagnostics import (
    holevo_bits,
    memory_states,
    prediction_retention,
    reference_purification,
    retention_curve,
    stationary_memory,
)
from qmem.linalg import DTYPE
from qmem.states import is_density_matrix, von_neumann_entropy
from qmem.tasks import (
    HMMTask,
    QuantumSourceTask,
    channel_fixed_point,
    delayed_recall,
    ghafari_process,
    make_task,
    markov_task,
    mixed_timescales,
    nemo_process,
    parity_task,
    perturbed_coin,
    pmd_renewal,
    pmd_survival,
    probability_clock,
    recall_positions,
    rk_golden_mean,
    srinivasan_hmm,
    stern_gerlach,
    uniform_renewal,
    upset_gambler,
)
from qmem.train import evaluate

LN2 = math.log(2)

TASKS = {
    "coin": lambda: perturbed_coin(0.3),
    "ghafari": lambda: ghafari_process(0.3, 0.8),
    "pmd4": lambda: pmd_renewal(4, 0.5, 0.4),
    "uniform3": lambda: uniform_renewal(3),
    "gambler": lambda: upset_gambler(0.7, 0.8),
    "golden43": lambda: rk_golden_mean(4, 3, 0.505),
    "nemo": lambda: nemo_process(0.666),
    "recall2": lambda: delayed_recall(2),
    "parity3": lambda: parity_task(3, 0.1),
    "mixed": lambda: mixed_timescales(0.05, 0.9),
    "mixed_hidden": lambda: mixed_timescales(0.05, 0.9, marked=False),
    "srinivasan": srinivasan_hmm,
    "clock": probability_clock,
    "spin": stern_gerlach,
}


def _brute_joint_entropy(task, length):
    seqs = all_sequences(task.n_symbols, length)
    lp = task.log_probs(seqs).sum(-1)
    p = torch.exp(lp)
    terms = torch.where(p > 0, -p * lp, torch.zeros_like(p))
    return float(p.sum()), float(terms.sum())


def _short(task):
    return {2: 7, 4: 4}.get(task.n_symbols, 3)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_step_entropies_match_brute_force(name):
    task = TASKS[name]()
    length = _short(task)
    joint = [0.0]
    for t in range(1, length + 1):
        total, h = _brute_joint_entropy(task, t)
        assert total == pytest.approx(1.0, abs=1e-12)
        joint.append(h)
    expect = [joint[t] - joint[t - 1] for t in range(1, length + 1)]
    assert task.step_entropies(length).tolist() == pytest.approx(expect, abs=1e-10)


@pytest.mark.parametrize("name", sorted(set(TASKS) - {"srinivasan", "mixed_hidden"}))
def test_optimum_matches_long_sample_estimate(name):
    task = TASKS[name]()
    seqs = task.sample(3000, 24, seed=5)
    per_seq = -task.log_probs(seqs).mean(-1)
    est, se = float(per_seq.mean()), float(per_seq.std() / math.sqrt(len(per_seq)))
    assert abs(est - task.optimal_nll(24)) < 4 * se + 1e-12


@pytest.mark.parametrize("name", ["clock", "spin", "ghafari", "srinivasan", "mixed_hidden"])
def test_sampler_matches_predictive(name):
    """Average exact predictive at step 4 equals the empirical frequency of y_4."""
    task = TASKS[name]()
    seqs = task.sample(40000, 4, seed=3)
    pred = task.predictive(seqs[:, :3])[:, 3].mean(0)
    freq = torch.bincount(seqs[:, 3], minlength=task.n_symbols).double() / len(seqs)
    se = torch.sqrt(pred * (1 - pred) / len(seqs)) + 1e-12
    assert float(((freq - pred).abs() / se).max()) < 4.5


@pytest.mark.parametrize("name", ["coin", "gambler", "golden43", "parity3", "mixed", "recall2"])
def test_entropy_rate_is_the_limit_of_step_entropies(name):
    task = TASKS[name]()
    h = task.step_entropies(120)
    assert float(h[-12:].mean()) == pytest.approx(task.entropy_rate(), abs=1e-6)


def test_quantum_source_entropy_rates_converge():
    for task in (probability_clock(), stern_gerlach()):
        h = task.step_entropies(300)
        assert float(h[-1]) == pytest.approx(task.entropy_rate(), abs=1e-10)
        assert is_density_matrix(task.initial)
        assert torch.allclose(task.instrument.channel(task.initial), task.initial, atol=1e-12)
    # Spin: log 3 for the axis plus one fair bit unless the axis repeats (prob 1/3).
    assert stern_gerlach().entropy_rate() == pytest.approx(math.log(3) + 2 / 3 * LN2, abs=1e-12)


# -- exact optima of the retention probes ---------------------------------------------
@pytest.mark.parametrize("tau", [0, 1, 4, 16])
def test_delayed_recall_optimum(tau):
    task = delayed_recall(tau)
    length = 3 * (tau + 2) + 1
    pos = recall_positions(tau, length)
    n_other = length - len(pos)
    assert task.optimal_nll(length) == pytest.approx(n_other / length * LN2, abs=1e-12)
    assert task.entropy_rate() == pytest.approx((tau + 1) / (tau + 2) * LN2, abs=1e-12)
    seqs = task.sample(200, length, seed=1)
    assert (seqs[:, pos] == seqs[:, pos - tau - 1]).all()  # recall = cue
    pred = task.predictive(seqs)[:, :-1]
    p_true = pred.gather(-1, seqs[..., None]).squeeze(-1)
    assert torch.allclose(p_true[:, pos], torch.ones_like(p_true[:, pos]))
    other = torch.ones(length, dtype=torch.bool)
    other[pos] = False
    assert torch.allclose(pred[:, other], torch.full_like(pred[:, other], 0.5))
    m = memory_profile(task)
    assert m["n_causal_states"] == 2 * tau + 3  # phase 0, then (phase, cue)


def test_recall_metrics_for_exact_and_memoryless_predictors():
    task = delayed_recall(3)
    seqs = task.sample(100, 25, seed=2)

    class Exact:
        def step_log_probs(self, s):
            return task.log_probs(s)

        def predictive_probs(self, s):
            return task.predictive(s)

    class Coin:
        def step_log_probs(self, s):
            return torch.full(s.shape, -LN2, dtype=torch.float64)

        def predictive_probs(self, s):
            return torch.full((*s.shape[:1], s.shape[1] + 1, 2), 0.5, dtype=torch.float64)

    r = recall_metrics(Exact(), task, seqs)
    assert r["recall_nll_nats"] == pytest.approx(0.0, abs=1e-12)
    assert r["recall_accuracy"] == 1.0 and r["source_recall_accuracy"] == 1.0
    assert r["other_nll_nats"] == pytest.approx(LN2)
    r = recall_metrics(Coin(), task, seqs)
    assert r["recall_nll_nats"] == pytest.approx(LN2)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_parity_needs_the_whole_window(k):
    task = parity_task(k, 0.1)
    h_noise = -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))
    assert task.entropy_rate() == pytest.approx(h_noise, abs=1e-12)
    m = memory_profile(task)
    assert m["n_causal_states"] == 2**k and m["c_mu_bits"] == pytest.approx(k)
    # Marginalising out any one of the k context symbols leaves a fair coin.
    rows = task.labelled.sum(-1).T  # rows[c, y] = p(y | context c)
    for drop in range(k):
        for c in range(2**k):
            c2 = c ^ (1 << drop)
            assert float((rows[c] + rows[c2])[1] / 2) == pytest.approx(0.5)


def test_mixed_timescales_rates_and_persistence():
    s, f = 0.02, 0.9
    task = mixed_timescales(s, f)

    def h(x):
        return -(x * math.log(x) + (1 - x) * math.log(1 - x))

    assert task.entropy_rate() == pytest.approx(h(s) + h(f), abs=1e-12)
    h_t = task.step_entropies(6)
    assert float(h_t[0]) == pytest.approx(h(s) + LN2)
    assert h_t[1:].tolist() == pytest.approx([h(s) + h(f)] * 5)
    # The regime is the parity of the switch marks; with no marks it is 0 forever:
    # alternation is then predicted to be unlikely however long ago the start was.
    seq = torch.tensor([[0] * 40])
    assert float(task.predictive(seq)[0, -1, 1]) == pytest.approx((1 - s) * (1 - f))
    seq = torch.tensor([[0] * 20 + [2] + [1] * 19])  # one switch mark, then regime 1
    assert float(task.predictive(seq)[0, -1, 0]) == pytest.approx((1 - s) * f)
    hidden = mixed_timescales(s, f, marked=False)
    assert not hidden.is_unifilar() and hidden.n_symbols == 2


# -- complexity: epsilon-machines, C_mu, C_q -------------------------------------------
@pytest.mark.parametrize("p", [0.05, 0.2, 0.3, 0.45])
def test_perturbed_coin_cq_closed_form(p):
    m = memory_profile(perturbed_coin(p))
    assert m["cq_bits"] == pytest.approx(perturbed_coin_cq(p), abs=1e-12)
    assert m["c_mu_bits"] == pytest.approx(1.0) and m["quantum_dim"] == 2


def test_epsilon_machine_merges_redundant_states():
    # An order-2 table that only depends on y_{t-1} is an order-1 chain: 2 causal states.
    task = markov_task(2, [[0.9, 0.1], [0.3, 0.7], [0.9, 0.1], [0.3, 0.7]])
    em = epsilon_machine(task)
    assert em["n_states"] == 2
    order1 = markov_task(1, [[0.9, 0.1], [0.3, 0.7]])
    assert em["c_mu_bits"] == pytest.approx(memory_profile(order1)["c_mu_bits"])
    # Duplicating a state of the coin (same futures) changes nothing.
    lab = torch.zeros(2, 3, 3, dtype=torch.float64)
    p = 0.3
    for i, face in ((0, 0), (1, 1), (2, 0)):
        lab[face, i, 0 if face == 0 else 1] = 1 - p
        lab[1 - face, i, 2 if face == 1 else 1] = p
    dup = HMMTask(lab)
    assert epsilon_machine(dup)["n_states"] == 2
    assert memory_profile(dup)["cq_bits"] == pytest.approx(perturbed_coin_cq(p))


def _brute_qmachine_cq(task, length):
    """C_q(L) from explicit q-machine states |eta_j(L)> = sum_w sqrt(P(w|j)) |w>|lambda(j,w)>."""
    em = epsilon_machine(task)
    lab, pi = em["labelled"], em["pi"]
    n, y = lab.shape[1], lab.shape[0]
    words = all_sequences(y, length).tolist()
    vecs = torch.zeros(n, len(words) * n, dtype=torch.float64)
    for j in range(n):
        for wi, w in enumerate(words):
            state, prob = j, 1.0
            for x in w:
                row = lab[x, state]
                if row.sum() == 0:
                    prob = 0.0
                    break
                prob *= float(row.sum())
                state = int(row.argmax())
            if prob > 0:
                vecs[j, wi * n + state] = math.sqrt(prob)
    rho = torch.einsum("j,ja,jb->ab", pi, vecs, vecs)
    ev = torch.linalg.eigvalsh(rho).clamp(min=0)
    ev = ev[ev > 1e-15]
    return float(-(ev * torch.log2(ev)).sum())


@pytest.mark.parametrize("name", ["nemo", "golden43", "gambler", "uniform3"])
def test_cq_by_length_matches_explicit_qmachine(name):
    task = TASKS[name]()
    q = quantum_overlaps(task)
    series = q["cq_by_length"]
    for length in range(0, 5):
        expect = _brute_qmachine_cq(task, length) if length else memory_profile(task)["c_mu_bits"]
        got = series[min(length, len(series) - 1)]
        assert got == pytest.approx(expect, abs=1e-9)


@pytest.mark.parametrize("name", ["coin", "ghafari", "gambler", "uniform3", "nemo", "parity3"])
def test_qmachine_instrument_reproduces_the_source(name):
    task = TASKS[name]()
    inst, rho = qmachine_instrument(task)
    m = memory_profile(task)
    assert inst.dim == m["quantum_dim"]
    assert float(inst.completeness_defect()) < 1e-10
    assert float(von_neumann_entropy(rho)) == pytest.approx(m["cq_bits"], abs=1e-9)
    stationary = HMMTask(task.labelled)  # same source, stationary start
    seqs = stationary.sample(64, 30, seed=4)
    qsrc = QuantumSourceTask(inst, rho)
    assert torch.allclose(qsrc.predictive(seqs), stationary.predictive(seqs), atol=1e-9)


@pytest.mark.parametrize("period", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("gamma", [0.49, 0.57, 0.64])
def test_pmd_qubit_model_is_exact(period, gamma):
    inst, s0 = pmd_qubit_instrument(period, gamma, 0.4)
    assert inst.dim == 2 and float(inst.completeness_defect()) < 1e-12
    a0 = inst.kraus[0, 0]
    for n in range(2 * period + 2):
        an = torch.linalg.matrix_power(a0, n)
        phi = float(torch.trace(an @ s0 @ an.conj().T).real)
        assert phi == pytest.approx(pmd_survival(n, gamma, 0.4, period), abs=1e-12)
    task = pmd_renewal(period, gamma, 0.4)
    seqs = task.sample(32, 40, seed=6)
    assert torch.allclose(QuantumSourceTask(inst).predictive(seqs), task.predictive(seqs))


@pytest.mark.parametrize("key", [item.key for item in LITERATURE])
def test_literature_numbers(key):
    (row,) = literature_checks([key])
    assert row["agrees"], row


def test_make_task_builds_every_benchmark_kind():
    specs = [
        {"kind": "perturbed_coin", "p": 0.3},
        {"kind": "ghafari", "p": 0.5, "q": 0.5},
        {"kind": "pmd_renewal", "period": 5},
        {"kind": "uniform_renewal", "n": 4},
        {"kind": "upset_gambler"},
        {"kind": "rk_golden_mean", "r": 3, "k": 2},
        {"kind": "nemo"},
        {"kind": "srinivasan_hmm"},
        {"kind": "delayed_recall", "tau": 8},
        {"kind": "parity", "k": 2},
        {"kind": "mixed_timescales", "switch_prob": 0.01},
        {"kind": "probability_clock"},
        {"kind": "stern_gerlach"},
    ]
    for spec in specs:
        task = make_task(spec)
        assert task.params["kind"] == spec["kind"]
        n = task.n_symbols
        assert n & (n - 1) == 0, f"{spec['kind']}: alphabet {n} is not a power of 2"


# -- diagnostics ------------------------------------------------------------------------
def test_stationary_memory_and_retention_of_the_exact_coin_model():
    p = 0.3
    task = perturbed_coin(p)
    inst, rho = qmachine_instrument(task)
    seqs = task.sample(600, 50, seed=8)
    st = stationary_memory(inst, rho, seqs)
    assert st["entropy_bits"] == pytest.approx(perturbed_coin_cq(p), abs=5e-3)
    ret = retention_curve(inst, rho, seqs, [1, 2])
    # Lag 1: the memory is |s_y> exactly, so the Holevo information is C_q.
    assert ret[1]["holevo_bits"] == pytest.approx(perturbed_coin_cq(p), abs=5e-3)
    # Lag 2: rho_x = sum_y P(y|x) |s_y><s_y|.
    s = memory_states(inst, rho, torch.tensor([[0], [1]]))[:, 1]  # |s_0>, |s_1>
    rho0 = (1 - p) * s[0] + p * s[1]
    rho1 = p * s[0] + (1 - p) * s[1]
    expect = holevo_bits(torch.tensor([0.5, 0.5]), torch.stack([rho0, rho1]))
    assert ret[2]["holevo_bits"] == pytest.approx(expect, abs=5e-3)
    # Prediction-based MI of the exact source at lag 1 is I(Y_t; Y_{t+1}) = 1 - H(p) bits.
    mi = prediction_retention(task.predictive(seqs), seqs, [1, 3])
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert mi[1] == pytest.approx(1 - h, abs=5e-3)
    assert mi[3] < mi[1]


def test_memory_states_match_the_model_update():
    model = build_model(ModelConfig(n_memory=2, n_output=1, seed=3))
    seqs = torch.randint(0, 2, (5, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        states = memory_states(model.instrument(), model.rho0, seqs)
        assert torch.allclose(states[:, -1], model.final_memory(seqs).rho)


def test_reference_purification_limits():
    # A projective measurement of the memory qubit purifies the reference in one step.
    k = torch.zeros(2, 1, 2, 2, dtype=DTYPE)
    k[0, 0, 0, 0] = 1
    k[1, 0, 1, 1] = 1
    r = reference_purification(Instrument(k), n_memory=1, n=50, length=5, seed=0)
    assert r["entropy_mean_bits"][0] == pytest.approx(1.0)
    assert max(r["entropy_mean_bits"][1:]) < 1e-9 and r["fraction_purified"] == 1.0
    # Outcomes independent of the memory (K_y = U / sqrt 2) never purify it.
    u = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    k = torch.stack([u, u])[:, None] / math.sqrt(2)
    r = reference_purification(Instrument(k), n_memory=1, n=20, length=10, seed=0)
    assert r["entropy_mean_bits"] == pytest.approx([1.0] * 11)
    assert r["fraction_purified"] == 0.0
    # A weak measurement purifies gradually: entropy decreases on average.
    model = build_model(ModelConfig(n_memory=2, n_output=1, strength=0.3, seed=2))
    with torch.no_grad():
        r = reference_purification(model.instrument(), 2, n_reference=2, n=200, length=30)
    s = r["entropy_mean_bits"]
    assert s[0] == pytest.approx(2.0) and s[-1] < s[0]


def test_channel_fixed_point_is_stationary():
    inst = probability_clock().instrument
    rho = channel_fixed_point(inst)
    assert is_density_matrix(rho)
    assert torch.allclose(inst.channel(rho), rho, atol=1e-12)


def test_evaluate_reports_missing_optimum_for_non_unifilar_sources():
    task = srinivasan_hmm()
    model = build_model(ModelConfig(n_memory=1, n_output=3, memory_depth=1, seed=0))
    seqs = task.sample(8, 32, seed=0)
    out = evaluate(model, seqs, task)
    assert math.isnan(out["optimal_nll_nats"]) and "optimal_nll_note" in out
    assert math.isfinite(out["gap_nats"]) or out["gap_nats"] == math.inf
