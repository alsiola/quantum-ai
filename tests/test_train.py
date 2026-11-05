"""Trainer, evaluation, multi-seed runner and analyses."""

import json
import math

import pytest
import torch

from qmem import FixedUnitaryBlock, ModelConfig, QuantumMemoryModel, build_model
from qmem.analysis import (
    curve_pathologies,
    intervention_effects,
    markov_baseline_nll,
    memory_trajectories,
    ngram_distribution,
    run_lengths,
    sample_statistics,
    trace_distance,
)
from qmem.data import make_splits
from qmem.experiment import (
    RunSpec,
    aggregate,
    bootstrap_ci,
    load_model,
    mean_se,
    run_many,
    run_one,
)
from qmem.gates import H
from qmem.linalg import DTYPE
from qmem.tasks import iid_task, markov_task
from qmem.train import TrainConfig, evaluate, lr_at, train

MARKOV1 = [[0.9, 0.1], [0.3, 0.7]]


def _fair_coin_model(n_memory=1):
    """Memoryless fair coin: Hadamard on the output ancilla only, K_y = I / sqrt 2."""
    n = n_memory + 1
    return QuantumMemoryModel(n_memory, 1, [FixedUnitaryBlock(H, [n_memory], n)])


def _splits(task, seed=0):
    return make_splits(task, 64, 32, 64, 8, seed=seed)


def test_train_config_validation_and_schedules():
    with pytest.raises(ValueError, match="unknown train config"):
        TrainConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError, match="unknown schedule"):
        TrainConfig.from_dict({"schedule": "warp"})
    c = TrainConfig(lr=0.1, schedule="cosine", steps=11, lr_min_factor=0.1)
    assert lr_at(c, 0) == pytest.approx(0.1)
    assert lr_at(c, 10) == pytest.approx(0.01)
    assert lr_at(c, 5) == pytest.approx(0.055)
    s = TrainConfig(lr=0.1, schedule="step", step_every=10, gamma=0.5)
    assert [lr_at(s, k) for k in (0, 9, 10, 25)] == pytest.approx([0.1, 0.1, 0.05, 0.025])
    assert TrainConfig.from_dict(TrainConfig().to_dict()) == TrainConfig()


def test_evaluate_exact_model_has_zero_gap():
    task = iid_task([0.5, 0.5])
    seqs = task.sample(50, 6, seed=0)
    m = evaluate(_fair_coin_model(), seqs, task)
    assert m["nll_nats"] == pytest.approx(math.log(2))
    assert m["nll_bits"] == pytest.approx(1.0)
    assert m["gap_nats"] == pytest.approx(0.0, abs=1e-12)
    assert m["optimal_nll_nats"] == pytest.approx(math.log(2))
    assert m["gap_to_optimum_nats"] == pytest.approx(0.0, abs=1e-12)


def test_train_logs_every_step_and_improves():
    task = markov_task(1, MARKOV1)
    splits = _splits(task)
    model = build_model(ModelConfig(n_memory=2, joint_depth=1, seed=0))
    with torch.no_grad():
        start = float(-model.step_log_probs(splits.val).mean())
    cfg = TrainConfig(lr=0.05, steps=40, batch_size=16, eval_every=5, patience=0)
    res = train(model, splits, cfg)
    assert res.status == "completed" and res.steps_run == 40
    for key in ("step", "loss", "grad_norm", "clipped", "lr", "diag", "elapsed"):
        assert len(res.curves[key]) == 40
    assert res.curves["eval_step"] == [*range(0, 40, 5), 40]
    assert res.best_val_nll < start
    # restore_best: the model now has the best validation NLL
    with torch.no_grad():
        assert float(-model.step_log_probs(splits.val).mean()) == pytest.approx(res.best_val_nll)
    # same seeds -> identical run
    model2 = build_model(ModelConfig(n_memory=2, joint_depth=1, seed=0))
    res2 = train(model2, splits, cfg)
    assert res2.curves["loss"] == res.curves["loss"]


def test_early_stopping_and_clipping():
    task = iid_task([0.8, 0.2])
    splits = _splits(task)
    model = build_model(ModelConfig(seed=1))
    res = train(model, splits, TrainConfig(lr=0.0, steps=100, eval_every=2, patience=3))
    assert res.status == "early_stopped"
    assert res.steps_run == 6  # evals at 0 (best), 2, 4, 6 (third without improvement)
    model = build_model(ModelConfig(seed=1, strength_trainable=True))
    res = train(model, splits, TrainConfig(steps=5, clip_grad_norm=1e-6, patience=0))
    assert all(res.curves["clipped"])


def test_divergence_is_recorded_not_raised():
    # g = pi/2 projective readout of qubit 0, no memory circuit, start in |00>:
    # p(y=1) = 0 exactly, so any data containing a 1 has infinite NLL.
    task = iid_task([0.5, 0.5])
    splits = _splits(task)
    model = build_model(
        ModelConfig(memory_depth=0, strength=math.pi / 2, strength_trainable=True, seed=0)
    )
    res = train(model, splits, TrainConfig(steps=10, batch_size=0))
    assert res.status == "diverged"
    assert "non-finite" in res.message
    assert model.diagnostics["zero_prob_steps"] > 0


def test_run_one_records_errors_and_success(tmp_path):
    base = {
        "task": {"kind": "iid", "probs": [0.8, 0.2]},
        "data": {"n_train": 32, "n_val": 16, "n_test": 16, "length": 6, "seed": 0},
        "train": {"steps": 5, "eval_every": 5},
    }
    bad = RunSpec(group="bad", model={"n_output": 2}, seed=0, **base)
    rec = run_one(bad)
    assert rec["status"] == "error" and "alphabet" in rec["message"]
    good = RunSpec(group="good/x", model={"joint_depth": 1}, seed=3, **base)
    recs = run_many([good, bad], tmp_path, workers=1, log=lambda *_: None)
    assert [r["group"] for r in recs] == ["good/x", "bad"]
    saved = json.loads((tmp_path / "runs" / "good/x" / "seed3.json").read_text())
    assert saved["meta"]["seed"] == 3 and saved["meta"]["config"]["model"] == {"joint_depth": 1}
    assert "git_commit" in saved["meta"] and "test" in saved
    assert (tmp_path / "runs" / "good/x" / "seed3.pt").exists()
    model = load_model(recs[0])
    seqs = torch.tensor([[0, 1, 0, 0, 1, 0]])
    with torch.no_grad():
        assert float(-model.step_log_probs(seqs).mean()) > 0
    rows = aggregate(recs, success_gap=10.0)
    assert {r["group"]: r["failure_rate"] for r in rows} == {"good/x": 0.0, "bad": 1.0}


def test_aggregate_statistics():
    recs = [
        {
            "group": "g",
            "status": "completed",
            "wall_seconds": 1.0,
            "test": {"gap_nats": g, "nll_nats": 1 + g, "accuracy": 0.5},
        }
        for g in (0.001, 0.002, 0.003, 0.5)
    ]
    recs.append({"group": "g", "status": "diverged", "wall_seconds": 1.0})
    (row,) = aggregate(recs, success_gap=0.01)
    assert row["n_runs"] == 5 and row["n_success"] == 3
    assert row["failure_rate"] == pytest.approx(0.4)
    m, se = mean_se([0.001, 0.002, 0.003, 0.5])
    assert row["gap_mean"] == pytest.approx(m) and row["gap_se"] == pytest.approx(se)
    assert row["gap_mean_successful"] == pytest.approx(0.002)
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=0)
    assert 1.0 <= lo < 2.5 < hi <= 4.0
    assert mean_se([2.0]) == (2.0, pytest.approx(math.nan, nan_ok=True))


def test_ngrams_and_run_lengths():
    seqs = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 1]])
    assert run_lengths(seqs) == {2: 1, 1: 1}  # the trailing run is not closed
    p = ngram_distribution(seqs, 1, 2)
    assert p.tolist() == pytest.approx([3 / 9, 6 / 9])
    p2 = ngram_distribution(torch.tensor([[0, 1, 0]]), 2, 2)
    assert p2.tolist() == pytest.approx([0, 0.5, 0.5, 0])
    stats = sample_statistics(seqs, seqs, seqs, 2)
    assert stats["ngram_tv"][2] == 0 and stats["odd_run_fraction_model"] == 0.5


def test_markov_baseline_recovers_markov_optimum():
    task = markov_task(1, MARKOV1)
    splits = make_splits(task, 2000, 200, 2000, 16, seed=1)
    oracle = float(-task.log_probs(splits.test).mean())
    b0 = markov_baseline_nll(splits.train, splits.val, splits.test, 0, 2)
    b1 = markov_baseline_nll(splits.train, splits.val, splits.test, 1, 2)
    assert b1["test_nll"] == pytest.approx(oracle, abs=0.005)
    assert b0["test_nll"] > b1["test_nll"] + 0.05


def test_interventions_and_memory_trajectories():
    task = markov_task(1, MARKOV1)
    seqs = task.sample(200, 10, seed=2)
    coin = _fair_coin_model()
    rows = intervention_effects(coin, task, seqs, lags=[1, 2], prefix_len=8, seed=0)
    assert rows[0]["model_tv"] == pytest.approx(0.0, abs=1e-12)  # memoryless model
    assert rows[0]["source_tv"] == pytest.approx(0.6)  # |p(1|0) - p(1|1)| = 0.6
    assert rows[1]["source_tv"] == pytest.approx(0.0, abs=1e-12)  # order 1
    traj = memory_trajectories(build_model(ModelConfig(joint_depth=1, seed=0)), seqs)
    assert len(traj["conditional_purity"]) == 11 and len(traj["nll_conditioned"]) == 10
    # no env ancillas: the conditional memory stays pure
    assert traj["conditional_purity"] == pytest.approx([1.0] * 11)
    assert traj["distance_to_unconditioned"][0] == pytest.approx(0.0, abs=1e-12)


def test_trace_distance_and_pathologies():
    a = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    b = torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE))
    assert float(trace_distance(a, b)) == pytest.approx(1.0)
    assert float(trace_distance(a, (a + b) / 2)) == pytest.approx(0.5)
    curves = {
        "grad_norm": [1.0] * 30 + [50.0] + [1.0] * 5,
        "clipped": [False] * 36,
        "val_nll": [1.0, 0.5, 0.4999, 0.4998, 0.3],
        "eval_step": [0, 10, 20, 30, 40],
    }
    p = curve_pathologies(curves)
    assert p["grad_spikes"] == 1 and p["grad_norm_max"] == 50.0
    assert p["longest_plateau_steps"] == 20
