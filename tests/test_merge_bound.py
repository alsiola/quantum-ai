"""Best coarse-grained (unifilar) classical model of fixed size: ``unifilar_merge_bound``."""

import math

import pytest
import torch

from qmem.complexity import epsilon_machine, set_partitions, unifilar_merge_bound
from qmem.linalg import REAL_DTYPE
from qmem.tasks import hmm_task, make_task

GHAFARI = {"kind": "ghafari", "p": 0.3, "q": 0.8}


def _kl_rate_mc(source, model, n=400, length=400, seed=0) -> float:
    """Monte Carlo KL rate (nats/symbol) of ``model`` against ``source`` on source samples."""
    seq = source.sample(n, length, seed)
    return float((source.log_probs(seq) - model.log_probs(seq)).mean())


def test_set_partitions_counts():
    # Stirling numbers of the second kind: S(4,1)+S(4,2) = 1+7, S(5,1..3) = 1+15+25
    assert len(list(set_partitions(4, 2))) == 8
    assert len(list(set_partitions(5, 3))) == 41
    assert len(list(set_partitions(3, 5))) == 5  # Bell number B_3


@pytest.mark.parametrize(
    "spec", [GHAFARI, {"kind": "pmd_renewal", "period": 4, "gamma": 0.5, "v": 0.4}]
)
def test_enough_states_gives_zero_and_one_state_gives_mutual_information(spec):
    task = make_task(spec)
    em = epsilon_machine(task)
    assert unifilar_merge_bound(task, em["n_states"])["bound_nats"] < 1e-12
    # One state: the best i.i.d. model; the KL is I(X; S) = H(X) - H(X | S).
    emit, pi = em["labelled"].sum(-1).T, em["pi"]
    marg = pi @ emit

    def h(p):
        p = p[p > 0]
        return float(-(p * torch.log(p)).sum())

    mi = h(marg) - sum(float(pi[s]) * h(emit[s]) for s in range(len(pi)))
    assert unifilar_merge_bound(task, 1)["bound_nats"] == pytest.approx(mi, abs=1e-12)


def test_ghafari_bound_is_reached_by_the_merged_machine():
    """Markov order 1: the minimising merge is unifilar and its KL rate equals the bound."""
    task = make_task(GHAFARI)
    res = unifilar_merge_bound(task, 2)
    assert res["realisable"]
    em = epsilon_machine(task)
    lab, pi, part = em["labelled"], em["pi"], res["partition"]
    emit, succ = lab.sum(-1).T, lab.argmax(-1).T
    y_dim, nb = lab.shape[0], max(part) + 1
    merged = torch.zeros(y_dim, nb, nb, dtype=REAL_DTYPE)
    for b in range(nb):
        members = [s for s in range(len(part)) if part[s] == b]
        w = pi[members] / pi[members].sum()
        for y in range(y_dim):
            p = float((w * emit[members, y]).sum())
            if p > 0:
                nxt = {part[int(succ[s, y])] for s in members if emit[s, y] > 0}
                merged[y, b, nxt.pop()] = p
    model = hmm_task(merged.tolist(), name="merged")
    assert _kl_rate_mc(task, model) == pytest.approx(res["bound_nats"], abs=4e-3)


def test_random_two_state_unifilar_models_never_beat_the_bound():
    """Brute force: random 2-state unifilar machines on the Ghafari source, KL by Monte Carlo."""
    task = make_task(GHAFARI)
    bound = unifilar_merge_bound(task, 2)["bound_nats"]
    gen = torch.Generator().manual_seed(0)
    y_dim = task.n_symbols
    for _ in range(12):
        emit = torch.rand(2, y_dim, generator=gen, dtype=REAL_DTYPE) + 0.05
        emit = emit / emit.sum(-1, keepdim=True)
        nxt = torch.randint(0, 2, (2, y_dim), generator=gen)
        lab = torch.zeros(y_dim, 2, 2, dtype=REAL_DTYPE)
        for s in range(2):
            for y in range(y_dim):
                lab[y, s, int(nxt[s, y])] = emit[s, y]
        model = hmm_task(lab.tolist(), name="random")
        kl = _kl_rate_mc(task, model, n=200, length=200, seed=1)
        assert kl > bound - 5e-3
    assert math.isfinite(bound) and bound > 0.01


# -- what a 2-state classical model can and cannot do (docs/phase3f-grid.md section 4) --------
@pytest.mark.parametrize("p,q", [(0.5, 0.5), (0.3, 0.8)])
def test_ghafari_is_a_two_state_non_unifilar_hmm(p, q):
    """The coin with "0 before 1 written as 2" is an edge-emitting HMM on the coin value.

    From coin 0: emit 0 and stay (1 - p), or emit 2 and flip (p). From coin 1: emit 1, and
    the next coin is 0 (q) or 1 (1 - q), which the symbol does not reveal (non-unifilar).
    So a 2-state classical model is exact, although every 2-state *unifilar* model is not
    (merge bound > 0).
    """
    from qmem.data import all_sequences

    task = make_task({"kind": "ghafari", "p": p, "q": q})
    lab = torch.zeros(task.n_symbols, 2, 2, dtype=REAL_DTYPE)
    lab[0, 0, 0], lab[2, 0, 1] = 1 - p, p
    lab[1, 1, 0], lab[1, 1, 1] = q, 1 - q
    model = hmm_task(lab.tolist(), initial=[q / (p + q), p / (p + q)], name="coin")
    seqs = all_sequences(task.n_symbols, 6)
    src, mod = task.log_probs(seqs).sum(-1), model.log_probs(seqs).sum(-1)
    ok = torch.isfinite(src)
    assert torch.equal(ok, torch.isfinite(mod))
    assert torch.allclose(src[ok], mod[ok], atol=1e-12)
    assert unifilar_merge_bound(task, 2)["bound_nats"] > 0.05


@pytest.mark.parametrize("period", [3, 4, 5, 6, 7, 8])
def test_pmd_needs_more_than_a_two_dimensional_real_model(period):
    """g(k) = P(1 0^k 1) of any 2-state HMM is pi T1 T0^k T1 1: a linear recurrence of order 2.

    PMD's survival function has three modes (Gamma^n and Gamma^n e^{+-2 pi i n / N}), so
    consecutive 3x3 Hankel determinants of g are non-zero: no 2-state HMM, and no real
    2-dimensional linear model of any sign, reproduces it. Wu et al.'s 1-qubit model does,
    with complex amplitudes (test_benchmarks.test_pmd_qubit_model_is_exact).
    """
    task = make_task({"kind": "pmd_renewal", "period": period, "gamma": 0.5, "v": 0.4})
    seqs = torch.zeros(12, 13, dtype=torch.long)
    for k in range(12):  # 1, k zeros, 1, then padding zeros (ignored below)
        seqs[k, 0], seqs[k, k + 1] = 1, 1
    lp = task.log_probs(seqs)
    g = torch.stack([lp[k, : k + 2].sum() for k in range(12)]).exp()
    for j in range(4):  # relative to the scale of the entries (g decays like Gamma^k)
        h = torch.stack([g[j + i : j + i + 3] for i in range(3)])
        assert abs(float(torch.linalg.det(h))) > 1e-6 * float(h.abs().max()) ** 3
    # Control: a 2-state HMM's g obeys the order-2 recurrence (Hankel determinants vanish).
    gen = torch.Generator().manual_seed(period)
    lab = torch.rand(2, 2, 2, generator=gen, dtype=REAL_DTYPE)
    lab = lab / lab.sum((0, 2), keepdim=True)
    lp_h = hmm_task(lab.tolist(), name="h2").log_probs(seqs)
    g_h = torch.stack([lp_h[k, : k + 2].sum() for k in range(12)]).exp()
    for j in range(4):
        h = torch.stack([g_h[j + i : j + i + 3] for i in range(3)])
        assert abs(float(torch.linalg.det(h))) < 1e-12
