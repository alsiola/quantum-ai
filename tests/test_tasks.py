"""Tasks: exact optima against brute-force enumeration and long-sample estimates."""

import math

import pytest
import torch

from qmem.data import all_sequences, make_splits
from qmem.tasks import (
    TASK_KINDS,
    HMMTask,
    even_process,
    golden_mean_process,
    iid_task,
    make_task,
    markov_task,
    register_task,
    stationary_distribution,
)

LN2 = math.log(2)

MARKOV2 = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]
TASKS = {
    "iid2": lambda: iid_task([0.8, 0.2]),
    "iid4": lambda: iid_task([0.4, 0.3, 0.2, 0.1]),
    "markov1": lambda: markov_task(1, [[0.9, 0.1], [0.3, 0.7]]),
    "markov1_4": lambda: markov_task(
        1, [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.2, 0.2, 0.2, 0.4], [0.5, 0.0, 0.0, 0.5]]
    ),
    "markov2": lambda: markov_task(2, MARKOV2),
    "even": even_process,
    "golden": golden_mean_process,
    "even_biased": lambda: even_process(0.3),
}


def _h(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def _brute_joint_entropy(task, length):
    seqs = all_sequences(task.n_symbols, length)
    lp = task.log_probs(seqs).sum(-1)
    p = torch.exp(lp)
    terms = torch.where(p > 0, -p * lp, torch.zeros_like(p))
    return float(p.sum()), float(terms.sum())


@pytest.mark.parametrize("name", sorted(TASKS))
def test_probabilities_sum_to_one(name):
    task = TASKS[name]()
    total, _ = _brute_joint_entropy(task, 5 if task.n_symbols == 2 else 4)
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_step_entropies_match_brute_force_chain_rule(name):
    task = TASKS[name]()
    length = 7 if task.n_symbols == 2 else 4
    h = task.step_entropies(length)
    joint = [0.0] + [_brute_joint_entropy(task, t)[1] for t in range(1, length + 1)]
    expect = [joint[t] - joint[t - 1] for t in range(1, length + 1)]
    assert h.tolist() == pytest.approx(expect, abs=1e-10)
    assert task.optimal_nll(length) == pytest.approx(joint[-1] / length, abs=1e-10)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_step_entropies_converge_to_entropy_rate(name):
    task = TASKS[name]()
    h = task.step_entropies(100)
    assert float(h[-1]) == pytest.approx(task.entropy_rate(), abs=1e-6)
    # Conditioning reduces entropy: H(Y_t | Y_<t) is non-increasing for stationary sources.
    assert (h[1:] <= h[:-1] + 1e-12).all()


@pytest.mark.parametrize("name", sorted(TASKS))
def test_optimum_matches_long_sample_estimate(name):
    task = TASKS[name]()
    seqs = task.sample(4000, 16, seed=7)
    per_seq = -task.log_probs(seqs).mean(-1)
    est, se = float(per_seq.mean()), float(per_seq.std() / math.sqrt(len(per_seq)))
    assert abs(est - task.optimal_nll(16)) < 4 * se


@pytest.mark.parametrize("name", ["markov2", "even", "markov1_4"])
def test_sampler_matches_predictive(name):
    """Empirical next-symbol frequencies given a short context match the exact predictive."""
    task = TASKS[name]()
    seqs = task.sample(20000, 6, seed=3)
    pred = task.predictive(seqs)[:, 5]  # p(y_6 | y_1..y_5)
    # group by the full prefix of 5 symbols; compare the average predictive with frequencies
    y = task.n_symbols
    idx = (seqs[:, :5] * (y ** torch.arange(4, -1, -1))).sum(-1)
    worst = 0.0
    for c in torch.unique(idx):
        mask = idx == c
        n = int(mask.sum())
        if n < 400:
            continue
        p = pred[mask][0]
        assert torch.allclose(pred[mask], p.expand_as(pred[mask]))
        freq = torch.bincount(seqs[mask, 5], minlength=y).double() / n
        z = ((freq - p).abs() / torch.sqrt(p * (1 - p) / n + 1e-12)).max()
        worst = max(worst, float(z))
    assert worst < 4.5


def test_iid_and_markov_entropy_rates_are_analytic():
    assert iid_task([0.8, 0.2]).entropy_rate() == pytest.approx(_h([0.8, 0.2]))
    rows = [[0.9, 0.1], [0.3, 0.7]]
    pi = [0.75, 0.25]  # stationary of rows
    expect = pi[0] * _h(rows[0]) + pi[1] * _h(rows[1])
    task = markov_task(1, rows)
    assert task.entropy_rate() == pytest.approx(expect)
    assert task.initial.tolist() == pytest.approx(pi)
    # iid: no history benefit, every step has the same entropy
    h = iid_task([0.4, 0.3, 0.2, 0.1]).step_entropies(5)
    assert h.tolist() == pytest.approx([_h([0.4, 0.3, 0.2, 0.1])] * 5)


def test_markov2_depends_on_two_steps_back():
    task = markov_task(2, MARKOV2)
    # contexts (y_{t-2}, y_{t-1}): (0,0)->0.1, (0,1)->0.8, (1,0)->0.4, (1,1)->0.9 for p(1)
    for ctx, p1 in (([0, 0], 0.1), ([0, 1], 0.8), ([1, 0], 0.4), ([1, 1], 0.9)):
        pred = task.predictive(torch.tensor([[1, 0, *ctx]]))[0, -1]
        assert float(pred[1]) == pytest.approx(p1)


def test_even_process_properties():
    task = even_process()
    assert task.entropy_rate() == pytest.approx(2 / 3 * LN2)
    assert task.is_unifilar()
    seqs = task.sample(500, 40, seed=1)
    from qmem.analysis import run_lengths

    runs = run_lengths(seqs)
    assert runs and all(k % 2 == 0 for k in runs)
    # After 0 then one 1 the next symbol must be 1; after 0 1 1 it is a fair coin.
    pred = task.predictive(torch.tensor([[0, 1, 1]]))[0]
    assert pred[2].tolist() == pytest.approx([0.0, 1.0])
    assert pred[3].tolist() == pytest.approx([0.5, 0.5])
    # A prefix with an odd run is impossible.
    assert task.log_probs(torch.tensor([[0, 1, 0]]))[0, 2] == -math.inf
    # Infinite Markov order: the optimum keeps falling with history length.
    h = task.step_entropies(12)
    assert (h[1:] < h[:-1]).all()


def test_golden_mean_properties():
    task = golden_mean_process()
    assert task.entropy_rate() == pytest.approx(2 / 3 * LN2)
    seqs = task.sample(500, 40, seed=1)
    assert not ((seqs[:, 1:] == 0) & (seqs[:, :-1] == 0)).any()
    # Markov order 1: H(Y_t | Y_<t) is constant from t = 2.
    h = task.step_entropies(8)
    assert h[1:].tolist() == pytest.approx([float(h[1])] * 7)


def test_stationary_distribution():
    t = torch.tensor([[0.9, 0.1], [0.3, 0.7]], dtype=torch.float64)
    pi = stationary_distribution(t)
    assert (pi @ t).tolist() == pytest.approx(pi.tolist())
    assert pi.tolist() == pytest.approx([0.75, 0.25])


def test_make_task_and_registry():
    task = make_task({"kind": "markov", "order": 2, "probs": MARKOV2, "name": "m2"})
    assert task.name == "m2" and task.params["kind"] == "markov"
    with pytest.raises(ValueError, match="unknown task kind"):
        make_task({"kind": "nope"})
    with pytest.raises(ValueError, match="needs 4 rows"):
        markov_task(2, [[0.5, 0.5]])
    with pytest.raises(ValueError, match="sum to 1"):
        HMMTask(torch.full((2, 1, 1), 0.6))

    def parity_like(name="flip"):
        lab = torch.zeros(2, 2, 2, dtype=torch.float64)
        lab[0, 0, 0] = lab[1, 0, 1] = lab[0, 1, 1] = lab[1, 1, 0] = 0.5
        return HMMTask(lab, name=name)

    register_task("test_flip", parity_like)
    try:
        assert make_task({"kind": "test_flip"}).entropy_rate() == pytest.approx(LN2)
        with pytest.raises(ValueError, match="already registered"):
            register_task("test_flip", parity_like)
    finally:
        TASK_KINDS.pop("test_flip")


def test_non_unifilar_entropy_rate_falls_back():
    # A noisy two-state HMM: each state emits either symbol (non-unifilar presentation).
    lab = torch.zeros(2, 2, 2, dtype=torch.float64)
    stay, emit = 0.9, 0.8
    for i in range(2):
        for j in range(2):
            pt = stay if i == j else 1 - stay
            lab[j, i, j] = pt * emit
            lab[1 - j, i, j] = pt * (1 - emit)
    task = HMMTask(lab)
    assert not task.is_unifilar()
    rate = task.entropy_rate(fallback_length=30)
    assert rate <= task.step_entropies(10)[-1] + 1e-12
    assert rate < LN2


def test_splits_are_deterministic_and_distinct():
    task = markov_task(1, [[0.9, 0.1], [0.3, 0.7]])
    a = make_splits(task, 20, 10, 10, 8, seed=5)
    b = make_splits(task, 20, 10, 10, 8, seed=5)
    assert torch.equal(a.train, b.train) and torch.equal(a.test, b.test)
    assert not torch.equal(a.train[:10], a.val)
    assert a.seeds["data_seed"] == 5
    assert a.describe()["shapes"]["train"] == [20, 8]
