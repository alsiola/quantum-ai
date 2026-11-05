"""Phase 3a baselines: exact likelihoods against brute force, normalisation, sampling.

For every family: ``log_prob`` against an independent computation of the
sequence probability, the probabilities of all length-T sequences sum to 1,
predictive and step probabilities agree, and the sampler's empirical
distribution matches ``exp(log_prob)``.
"""

import itertools
import math

import pytest
import torch

from qmem.analysis import markov_baseline_nll
from qmem.baselines import (
    BaselineSpec,
    HMMModel,
    NGramModel,
    RNNModel,
    StiefelInstrumentModel,
    UMPSModel,
    fit_em,
    fit_ngram,
    matched_hidden_size,
    rnn_parameter_count,
    run_baseline,
)
from qmem.data import all_sequences
from qmem.experiment import build_splits, run_many
from qmem.instrument import kraus_from_unitary
from qmem.linalg import DTYPE, REAL_DTYPE
from qmem.tasks import even_process, markov_task

DATA = {"n_train": 256, "n_val": 128, "n_test": 128, "length": 16, "seed": 0}


def _ngram(n_symbols, order=2, alpha=0.5):
    task = markov_task(1, torch.full((n_symbols, n_symbols), 1.0 / n_symbols).tolist())
    train = task.sample(64, 12, seed=5)
    return NGramModel.fit(train, order, alpha, n_symbols)


MODELS = {
    "hmm3_y2": lambda: HMMModel(3, 2, seed=1),
    "hmm2_y4": lambda: HMMModel(2, 4, seed=2),
    "ngram_k2_y2": lambda: _ngram(2),
    "ngram_k1_y4": lambda: _ngram(4, order=1),
    "gru3_y2": lambda: RNNModel("gru", 3, 2, seed=3),
    "rnn2_y4": lambda: RNNModel("rnn", 2, 4, seed=4),
    "umps3_y2": lambda: UMPSModel(3, 2, seed=5),
    "umps2_y4": lambda: UMPSModel(2, 4, seed=6),
    "lps2_k2_y2": lambda: UMPSModel(2, 2, purification=2, seed=7),
    "hqmm_m1_y2": lambda: StiefelInstrumentModel(1, 1, seed=8),
    "hqmm_m2_e1_y2": lambda: StiefelInstrumentModel(2, 1, n_env=1, seed=9),
    "hqmm_m1_y4": lambda: StiefelInstrumentModel(1, 2, seed=10),
}


@pytest.fixture(params=sorted(MODELS))
def any_model(request):
    return MODELS[request.param]()


# -- interface properties shared by every family --------------------------------
def test_all_sequences_sum_to_one(any_model):
    seqs = all_sequences(any_model.n_symbols, 5)
    with torch.no_grad():
        total = any_model.log_prob(seqs).exp().sum()
    assert abs(float(total) - 1) < 1e-10


def test_predictive_matches_step_log_probs(any_model):
    seqs = torch.randint(0, any_model.n_symbols, (7, 6), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        pred = any_model.predictive_probs(seqs)
        step = any_model.step_log_probs(seqs)
    assert pred.shape == (7, 7, any_model.n_symbols)
    assert torch.allclose(pred.sum(-1), torch.ones(7, 7, dtype=pred.dtype), atol=1e-12)
    direct = torch.log(pred[:, :-1].gather(-1, seqs[..., None]).squeeze(-1))
    assert torch.allclose(direct, step, atol=1e-10)


def test_sampler_agrees_with_log_prob(any_model):
    length, n = 3, 40_000
    samples = any_model.sample(n, length, seed=11)
    assert samples.shape == (n, length)
    seqs = all_sequences(any_model.n_symbols, length)
    with torch.no_grad():
        p = any_model.log_prob(seqs).exp()
    idx = (samples * any_model.n_symbols ** torch.arange(length - 1, -1, -1)).sum(-1)
    freq = torch.bincount(idx, minlength=len(seqs)).to(REAL_DTYPE) / n
    # Total variation within a few standard errors of the multinomial noise.
    assert 0.5 * float((freq - p).abs().sum()) < 0.02
    assert torch.equal(samples, any_model.sample(n, length, seed=11))  # seeded


def test_describe_has_cost_fields(any_model):
    info = any_model.describe()
    for key in ("family", "n_parameters", "state_dim", "flops_per_step"):
        assert key in info
    assert info["n_parameters"] > 0 and info["state_dim"] >= 1


# -- independent brute-force likelihoods per family ------------------------------
def _seqs(n_symbols, length=4):
    return all_sequences(n_symbols, length)


def test_hmm_matches_matrix_product():
    m = HMMModel(3, 2, seed=1)
    lab, pi = m.labelled().detach(), m.initial().detach()
    seqs = _seqs(2)
    want = []
    for s in seqs:
        v = pi.clone()
        for y in s:
            v = v @ lab[y]
        want.append(v.sum())
    with torch.no_grad():
        got = m.log_prob(seqs).exp()
    assert torch.allclose(got, torch.stack(want), atol=1e-12)
    assert m.n_parameters() == 3 * (2 * 3) + 3  # logits, one redundant per softmax
    assert m.n_free_parameters() == 3 * (2 * 3 - 1) + 2


def test_hmm_set_probabilities_round_trip():
    m = HMMModel(2, 2, seed=0)
    lab = even_process().labelled
    pi = torch.tensor([2 / 3, 1 / 3], dtype=REAL_DTYPE)
    m.set_probabilities(lab, pi)
    assert torch.allclose(m.labelled(), lab, atol=1e-15)
    seqs = _seqs(2, 6)
    with torch.no_grad():
        ours, src = m.step_log_probs(seqs), even_process().log_probs(seqs)
    possible = torch.isfinite(src.sum(-1))
    assert torch.allclose(ours[possible], src[possible], atol=1e-12)
    assert torch.equal(ours.sum(-1)[~possible], src.sum(-1)[~possible])  # both -inf


def test_ngram_matches_phase2_count_model():
    task = markov_task(2, [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    splits = build_splits(task, DATA)
    for k in (0, 1, 3):
        ref = markov_baseline_nll(splits.train, splits.val, splits.test, k, 2, alphas=(0.5,))
        m = NGramModel.fit(splits.train, k, 0.5, 2)
        with torch.no_grad():
            got = float(-m.step_log_probs(splits.test).mean())
        assert abs(got - ref["test_nll"]) < 1e-12
        assert m.n_parameters() == sum(2**j for j in range(k + 1))


def test_ngram_selection_uses_validation():
    task = markov_task(2, [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    splits = build_splits(task, {**DATA, "n_train": 1024})
    m, grid = fit_ngram(splits.train, splits.val, 2, list(range(5)), [0.1, 1.0])
    assert len(grid) == 10
    assert m.order >= 2  # order 2 is needed
    assert min(g["val_nll"] for g in grid) == pytest.approx(
        float(-m.step_log_probs(splits.val).mean())
    )


def test_rnn_fused_matches_stepwise():
    for cell in ("gru", "rnn"):
        m = RNNModel(cell, 5, 3, seed=2)
        seqs = torch.randint(0, 3, (4, 9), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            assert torch.allclose(m.step_log_probs(seqs), m.step_log_probs_stepwise(seqs))
        assert m.n_parameters() == rnn_parameter_count(cell, 5, 3)


def test_vanilla_rnn_matches_equations():
    m = RNNModel("rnn", 3, 2, seed=4)
    r = m.rnn
    seq = torch.tensor([[1, 0, 1]])
    h = torch.zeros(3, dtype=REAL_DTYPE)
    x = torch.zeros(2, dtype=REAL_DTYPE)
    logp = 0.0
    for y in seq[0]:
        h = torch.tanh(r.weight_ih_l0 @ x + r.bias_ih_l0 + r.weight_hh_l0 @ h + r.bias_hh_l0)
        p = torch.softmax(m.readout.weight @ h + m.readout.bias, -1)
        logp += float(torch.log(p[y]).detach())
        x = torch.nn.functional.one_hot(y, 2).to(REAL_DTYPE)
    with torch.no_grad():
        assert abs(float(m.log_prob(seq)) - logp) < 1e-12


def test_matched_hidden_size():
    h = matched_hidden_size("gru", 2, 21)
    assert rnn_parameter_count("gru", h, 2) <= 21 < rnn_parameter_count("gru", h + 1, 2)


def _transfer_fixed_point(a):
    """Sigma and lambda from an eigendecomposition of the transfer matrix (independent)."""
    chi = a.shape[-1]
    # E*(X) = sum A^dagger X A  ->  vec(E*(X)) = sum (A^T ⊗ A^dagger) vec(X) (row-major vec)
    t = sum(torch.kron(ak.conj().T, ak.T) for ak in a.reshape(-1, chi, chi))
    vals, vecs = torch.linalg.eig(t)
    i = int(vals.abs().argmax())
    sigma = vecs[:, i].reshape(chi, chi)
    sigma = sigma / torch.trace(sigma)
    return sigma, vals[i]


@pytest.mark.parametrize("kappa", [1, 2])
def test_umps_matches_infinite_chain_marginals(kappa):
    m = UMPSModel(3, 2, purification=kappa, seed=3)
    a = m.tensors().detach()
    sigma, lam = _transfer_fixed_point(a)
    # The eigenvector check is itself independent of the power iteration.
    adj = torch.einsum("ykji,jl,yklm->im", a.conj(), sigma, a)
    assert torch.allclose(adj, lam * sigma, atol=1e-10)
    l = m.boundary().detach()
    l0 = torch.outer(l, l.conj())
    seqs = _seqs(2, 4)
    want = []
    for s in seqs:
        lt = l0
        for y in s:
            lt = sum(a[y, k] @ lt @ a[y, k].conj().T for k in range(kappa))
        want.append((torch.trace(lt @ sigma) / (lam ** len(s) * torch.trace(l0 @ sigma))).real)
    with torch.no_grad():
        got = m.log_prob(seqs).exp()
    assert torch.allclose(got, torch.stack(want), atol=1e-9)
    assert m.last_fp_residual < 1e-8 and not m.diagnostics


def test_umps_gradients_flow_through_fixed_point():
    m = UMPSModel(2, 2, seed=0)
    seqs = torch.randint(0, 2, (8, 5), generator=torch.Generator().manual_seed(0))
    loss = -m.step_log_probs(seqs).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_hqmm_isometry_and_kraus_from_its_unitary():
    for n_memory, n_output, n_env in ((1, 1, 0), (2, 1, 1), (1, 2, 0)):
        m = StiefelInstrumentModel(n_memory, n_output, n_env, seed=1)
        v = m.isometry()
        assert torch.allclose(v.conj().T @ v, torch.eye(v.shape[1], dtype=DTYPE), atol=1e-12)
        u = m.unitary()
        assert torch.allclose(u.conj().T @ u, torch.eye(u.shape[0], dtype=DTYPE), atol=1e-12)
        k = kraus_from_unitary(u, n_memory, n_output, n_env)
        assert torch.allclose(k, m.instrument().kraus, atol=1e-12)


def test_hqmm_matches_kraus_products():
    m = StiefelInstrumentModel(1, 1, n_env=1, seed=2)
    k = m.instrument().kraus.detach()  # (Y, E, d, d)
    seqs = _seqs(2, 3)
    want = []
    for s in seqs:
        total = 0.0
        # Sum over every unread Kraus index sequence: p = sum_e |K_e...K_e |0>|^2.
        for es in all_sequences(2, len(s)):
            psi = torch.tensor([1, 0], dtype=DTYPE)
            for y, e in zip(s, es, strict=True):
                psi = k[y, e] @ psi
            total += float((psi.conj() @ psi).real)
        want.append(total)
    with torch.no_grad():
        got = m.log_prob(seqs).exp()
    assert torch.allclose(got, torch.tensor(want, dtype=REAL_DTYPE), atol=1e-12)


# -- sanity: capacity statements from the issue ----------------------------------
def test_em_fits_even_process_with_two_states():
    task = even_process()
    splits = build_splits(task, {**DATA, "n_train": 512, "length": 32})
    gaps = []
    for seed in range(4):  # a few restarts; EM can stop in a local optimum
        m = HMMModel(2, 2, seed=seed)
        res = fit_em(m, splits.train, splits.val, max_iters=300, patience=30)
        with torch.no_grad():
            gap = float((-m.step_log_probs(splits.test) + task.log_probs(splits.test)).mean())
        gaps.append((res["best_val_nll"], gap))
    best = min(gaps)  # selected on validation
    assert best[1] < 0.01


def test_no_finite_order_markov_fits_even_process():
    """h_k = H(Y_{k+1} | Y_1..Y_k) > entropy rate for every k (computed exactly)."""
    task = even_process()
    rate = task.entropy_rate()

    def block_entropy(n):
        if n == 0:
            return 0.0
        s = all_sequences(2, n)
        lp = task.log_probs(s).sum(-1)
        p = lp.exp()
        return float(-(p[p > 0] * lp[p > 0]).sum())

    excess = [block_entropy(k + 1) - block_entropy(k) - rate for k in range(9)]
    assert all(e > 1e-4 for e in excess)
    assert all(a > b for a, b in itertools.pairwise(excess))  # decreasing in k


def test_run_baseline_every_family_and_run_many(tmp_path):
    task = {"kind": "even", "p": 0.5, "name": "even"}
    tr = {"steps": 3, "batch_size": 64, "eval_every": 2, "patience": 0}
    specs = [
        BaselineSpec("g/ngram", task, DATA, "ngram", {"orders": [0, 1, 2], "alphas": [0.5]},
                     method="count"),
        BaselineSpec("g/hmm_em", task, DATA, "hmm", {"n_states": 2}, method="em",
                     fit={"max_iters": 3}),
        BaselineSpec("g/hmm", task, DATA, "hmm", {"n_states": 2}, train=tr),
        BaselineSpec("g/gru", task, DATA, "gru", {"hidden": 3}, train=tr),
        BaselineSpec("g/rnn", task, DATA, "rnn", {"hidden": 3}, train=tr),
        BaselineSpec("g/umps", task, DATA, "umps", {"bond_dim": 2}, train=tr),
        BaselineSpec("g/hqmm", task, DATA, "hqmm", {"n_memory": 1, "n_env": 1}, train=tr),
        BaselineSpec("g/A", task, DATA, "A",
                     {"n_memory": 1, "n_output": 1, "joint_depth": 1}, train=tr),
    ]  # fmt: skip
    from qmem.baselines.runner import run_baseline as runner

    records = run_many(specs, tmp_path, workers=1, log=lambda *_: None, runner=runner)
    for r in records:
        assert r["status"] in ("completed", "early_stopped"), (r["group"], r.get("message"))
        assert math.isfinite(r["test"]["gap_nats"])
        assert r["model_info"]["state_dim"] >= 1
    assert (tmp_path / "runs" / "g" / "hqmm" / "seed0.json").exists()
    assert run_baseline(BaselineSpec("x", task, DATA, "nope", {}))["status"] == "error"
