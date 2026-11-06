"""Phase 4a (issue #12): initialisations, soft targets and restarts.

Each initialisation is checked against its analytic form: at zero perturbation the
near-identity starts are exactly memoryless instruments with effects ``E_y = p_y I``.
"""

import math

import pytest
import torch

from qmem.baselines import StiefelInstrumentModel, UMPSModel
from qmem.baselines.runner import BaselineSpec, run_baseline
from qmem.experiment import RunSpec, build_splits, run_one
from qmem.model import ModelConfig, build_model
from qmem.tasks import make_task
from qmem.train import TrainConfig, train, train_with_restarts
from qmem.trainability import (
    bit_marginals,
    init_statistics,
    resolve_init_probs,
    restart_seed,
    soft_target_loss,
    teacher_entropy,
    unigram,
)

M1Q = {
    "kind": "markov",
    "order": 1,
    "probs": [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.2, 0.2, 0.2, 0.4], [0.5, 0, 0, 0.5]],
}
SMALL = {"n_train": 32, "n_val": 16, "n_test": 16, "length": 8, "seed": 0}


def product_probs(marg):
    """Distribution over big-endian bit strings with independent bits P(bit_k = 1) = marg[k]."""
    n = len(marg)
    out = []
    for y in range(2**n):
        p = 1.0
        for k in range(n):
            b = (y >> (n - 1 - k)) & 1
            p *= marg[k] if b else 1 - marg[k]
        out.append(p)
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize("n_memory,n_output,jd,md", [(2, 2, 2, 2), (3, 2, 2, 2), (2, 1, 3, 1)])
def test_product_init_is_memoryless_at_zero_perturbation(n_memory, n_output, jd, md):
    probs = [0.4, 0.3, 0.2, 0.1] if n_output == 2 else [0.8, 0.2]
    cfg = ModelConfig(
        n_memory=n_memory, n_output=n_output, memory_depth=md, joint_depth=jd, entangler="cz",
        strength=0.0, init="product", init_probs=probs, init_scale=0.0,
    )  # fmt: skip
    k = build_model(cfg).instrument().kraus  # (Y, 1, d, d)
    p = torch.tensor(probs, dtype=torch.float64)
    expected = product_probs(bit_marginals(p, n_output))
    d = 2**n_memory
    eye = torch.eye(d, dtype=k.dtype)
    for y in range(2**n_output):
        ky = k[y, 0]
        # K_y = c_y U_y with U_y unitary: K_y K_y^dagger = K_y^dagger K_y = |c_y|^2 I.
        assert torch.allclose(ky.conj().T @ ky, expected[y] * eye, atol=1e-12)
        assert torch.allclose(ky @ ky.conj().T, expected[y] * eye, atol=1e-12)


def test_product_init_matches_uniform_and_binary_marginals_exactly():
    # For a binary alphabet the bit marginal is the distribution itself.
    cfg = ModelConfig(
        n_memory=2, n_output=1, joint_depth=2, entangler="cz", strength=0.0,
        init="product", init_probs=[0.8, 0.2], init_scale=0.0,
    )  # fmt: skip
    m = build_model(cfg)
    seqs = torch.randint(2, (5, 7), generator=torch.Generator().manual_seed(0))
    pp = m.predictive_probs(seqs)
    assert torch.allclose(pp, torch.tensor([0.8, 0.2], dtype=torch.float64).expand_as(pp))


def test_product_init_needs_cz_and_small_noise_stays_close():
    with pytest.raises(ValueError, match="cz"):
        build_model(ModelConfig(joint_depth=2, init="product"))
    with pytest.raises(ValueError, match="unknown init"):
        ModelConfig.from_dict({"init": "bogus"})
    cfg = ModelConfig(
        n_memory=2, n_output=2, joint_depth=2, entangler="cz", strength=0.0, init="product",
        init_scale=0.01, seed=3,
    )  # fmt: skip
    eff = build_model(cfg).instrument().effects().detach()
    assert float((eff - 0.25 * torch.eye(4, dtype=eff.dtype)).abs().max()) < 0.1


def test_random_init_is_unchanged_by_the_new_options():
    a = build_model(ModelConfig(joint_depth=2, seed=5)).state_dict()
    b = build_model(ModelConfig(joint_depth=2, seed=5, init="random", init_probs=[0.5, 0.5]))
    for key, v in b.state_dict().items():
        assert torch.equal(v, a[key])


@pytest.mark.parametrize("n_memory,n_output,n_env", [(2, 2, 0), (3, 1, 0), (2, 1, 1)])
def test_hqmm_identity_init_is_sqrt_p_identity(n_memory, n_output, n_env):
    probs = torch.softmax(torch.arange(2**n_output, dtype=torch.float64), 0)
    m = StiefelInstrumentModel(
        n_memory, n_output, n_env, seed=1, init="identity", init_probs=probs.tolist(),
        init_scale=0.0,
    )  # fmt: skip
    k = m.instrument().kraus
    eye = torch.eye(2**n_memory, dtype=k.dtype)
    for y in range(2**n_output):
        assert torch.allclose(k[y, 0], math.sqrt(probs[y]) * eye, atol=1e-12)
        assert torch.allclose(k[y, 1:], torch.zeros_like(k[y, 1:]), atol=1e-12)
    assert float(m.instrument().completeness_defect()) < 1e-12


def test_hqmm_identity_noise_does_not_grow_with_dimension():
    dev = []
    for n in (1, 2, 3, 4):
        m = StiefelInstrumentModel(n, 2, 0, seed=0, init="identity", init_scale=0.1)
        eff = m.instrument().effects().detach()
        dev.append(float((eff - 0.25 * torch.eye(2**n, dtype=eff.dtype)).abs().max()))
    assert max(dev) < 0.1


@pytest.mark.parametrize("purification", [1, 2])
def test_umps_identity_init_predicts_p_everywhere(purification):
    probs = [0.4, 0.3, 0.2, 0.1]
    m = UMPSModel(4, 4, purification=purification, seed=0, init="identity", init_probs=probs,
                  init_scale=0.0)  # fmt: skip
    seqs = torch.randint(4, (6, 9), generator=torch.Generator().manual_seed(1))
    pp = m.predictive_probs(seqs)
    assert torch.allclose(pp, torch.tensor(probs, dtype=torch.float64).expand_as(pp), atol=1e-10)
    lp = m.log_prob(seqs)
    expect = torch.log(torch.tensor(probs, dtype=torch.float64))[seqs].sum(-1)
    assert torch.allclose(lp, expect, atol=1e-10)


def test_resolve_init_probs_uses_training_unigram():
    train_seqs = torch.tensor([[0, 1, 1, 3], [3, 3, 3, 0]])
    assert unigram(train_seqs, 4) == [0.25, 0.25, 0.0, 0.5]
    cfg = resolve_init_probs({"init_probs": "data", "x": 1}, train_seqs, 4)
    assert cfg == {"init_probs": [0.25, 0.25, 0.0, 0.5], "x": 1}
    assert resolve_init_probs({"init_probs": "uniform"}, train_seqs, 4) == {"init_probs": "uniform"}


def test_soft_target_loss_against_brute_force_and_self_kl():
    task = make_task(M1Q)
    seqs = task.sample(6, 5, seed=2)
    model = StiefelInstrumentModel(2, 2, 0, seed=0)
    loss = float(soft_target_loss(task.predictive)(model, seqs))
    q = task.predictive(seqs)[:, :-1]
    p = model.predictive_probs(seqs)[:, :-1].detach()
    brute = 0.0
    for b in range(6):
        for t in range(5):
            for y in range(4):
                if q[b, t, y] > 0:
                    brute -= float(q[b, t, y] * torch.log(p[b, t, y]))
    assert loss == pytest.approx(brute / 30, rel=1e-12)
    # KL(q || p) >= 0, and a model as its own teacher has KL exactly 0.
    assert loss - teacher_entropy(task.predictive, seqs) > 0
    self_loss = float(soft_target_loss(model.predictive_probs)(model, seqs))
    assert self_loss == pytest.approx(teacher_entropy(model.predictive_probs, seqs), abs=1e-12)


def test_soft_targets_from_the_source_train_towards_it():
    task = make_task({"kind": "iid", "probs": [0.8, 0.2]})
    splits = build_splits(task, SMALL)
    model = StiefelInstrumentModel(1, 1, 0, seed=0)
    loss = soft_target_loss(task.predictive)
    cfg = TrainConfig(lr=0.1, steps=150, patience=0)
    res = train(model, splits, cfg, loss_fn=loss)
    kl = res.best_val_nll - teacher_entropy(task.predictive, splits.val)
    assert 0 <= kl < 1e-3


def test_restarts_pick_the_best_screened_start_and_reduce_to_train():
    task = make_task(M1Q)
    splits = build_splits(task, SMALL)

    def make(r):
        return StiefelInstrumentModel(1, 2, 0, seed=restart_seed(4, r))

    cfg = TrainConfig(lr=0.05, steps=20, eval_every=5, patience=0, restarts=3, restart_steps=10)
    _model, res = train_with_restarts(make, splits, cfg)
    screen = res.restart_screen
    assert [s["restart"] for s in screen] == [0, 1, 2]
    chosen = min(screen, key=lambda s: s["best_val_nll"])
    assert [s["chosen"] for s in screen] == [s is chosen for s in screen]
    assert res.best_val_nll <= chosen["best_val_nll"] + 1e-12
    # restarts = 1 is exactly the plain trainer.
    one = TrainConfig(lr=0.05, steps=20, eval_every=5, patience=0)
    _m1, r1 = train_with_restarts(make, splits, one)
    m2 = make(0)
    r2 = train(m2, splits, one)
    assert r1.curves["loss"] == r2.curves["loss"] and r1.restart_screen == []
    assert restart_seed(7, 0) == 7


def test_runners_record_resolved_probs_and_restarts():
    tcfg = {"lr": 0.05, "steps": 10, "eval_every": 5, "patience": 0, "restarts": 2,
            "restart_steps": 5}  # fmt: skip
    model = {"n_memory": 1, "n_output": 2, "joint_depth": 2, "entangler": "cz",
             "init": "product", "init_probs": "data", "init_scale": 0.05}  # fmt: skip
    rec = run_one(RunSpec("t/a", M1Q, SMALL, model, tcfg, seed=0))
    assert rec["status"] == "completed", rec.get("traceback")
    assert len(rec["model_resolved"]["init_probs"]) == 4 and len(rec["restart_screen"]) == 2
    spec = BaselineSpec("t/h", M1Q, SMALL, "hqmm", {"n_memory": 1, "init": "identity",
                        "init_probs": "data", "init_scale": 0.1}, train=tcfg, seed=1)  # fmt: skip
    rec = run_baseline(spec)
    assert rec["status"] == "completed", rec.get("traceback")
    assert len(rec["restart_screen"]) == 2


def test_init_statistics_on_the_identity_start():
    batch = make_task(M1Q).sample(8, 6, seed=0)
    st = init_statistics(
        lambda s: StiefelInstrumentModel(2, 2, 0, seed=s, init="identity", init_scale=0.0),
        [0, 1, 2],
        batch,
    )
    # Exactly memoryless and uniform: H = ln 4, no history dependence, NLL = ln 4.
    assert st["pred_entropy_mean"] == pytest.approx(math.log(4), abs=1e-10)
    assert st["history_tv_mean"] == pytest.approx(0, abs=1e-10)
    assert st["nll_mean"] == pytest.approx(math.log(4), abs=1e-10)
    assert st["grad_var"] == pytest.approx(0, abs=1e-20)  # same start for every seed
