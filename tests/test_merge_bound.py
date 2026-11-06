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
