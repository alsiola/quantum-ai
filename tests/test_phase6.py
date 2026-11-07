"""The Phase 6 analysis helpers (experiments/phase6.py) against brute-force computations."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import phase5a
import phase6

from qmem.model import ModelConfig, build_model
from qmem.tasks import make_task


@pytest.fixture(scope="module")
def pmd():
    return make_task({"kind": "pmd_renewal", "period": 3, "gamma": 0.5, "v": 0.4})


@pytest.fixture(scope="module")
def qmodel():
    cfg = ModelConfig(n_memory=1, memory_depth=2, joint_depth=2, seed=3)
    return build_model(cfg)


def all_seqs(length: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product((0, 1), repeat=length)), dtype=torch.long)


def brute_blocks(logp, starts: list[int], length: int) -> np.ndarray:
    """Block distribution by summing exact sequence probabilities over the prefix."""
    out = np.zeros(2**length)
    for t0 in starts:
        seqs = all_seqs(t0 + length)
        p = logp(seqs).exp().numpy()
        weights = 1 << np.arange(length - 1, -1, -1)
        idx = seqs[:, t0:].numpy() @ weights
        np.add.at(out, idx, p)
    return out / len(starts)


def test_hmm_proc_blocks_and_nll(pmd):
    p = phase6.Proc.hmm("pmd", pmd.labelled, pmd.initial)
    starts = [0, 2, 5]
    got = p.blocks(starts, 3)
    want = brute_blocks(lambda s: pmd.log_probs(s).sum(-1), starts, 3)
    assert np.allclose(got, want, atol=1e-12)
    seqs = pmd.sample(50, 12, seed=1)
    assert math.isclose(p.nll(seqs), float(-pmd.log_probs(seqs).mean()), rel_tol=1e-12)
    marg = p.marginals(6)
    brute = [
        float(pmd.log_probs(s).sum(-1).exp()[s[:, t] == 1].sum())
        for t in range(6)
        for s in [all_seqs(t + 1)]
    ]
    assert np.allclose(marg, brute, atol=1e-12)


def test_instrument_proc_matches_model(qmodel):
    with torch.no_grad():
        p = phase6.Proc("q", qmodel.instrument(), qmodel.rho0)
        starts = [1, 3]
        got = p.blocks(starts, 2)
        want = brute_blocks(qmodel.log_prob, starts, 2)
        assert np.allclose(got, want, atol=1e-12)
        seqs = qmodel.sample(40, 10, seed=2)
        assert math.isclose(p.nll(seqs), float(-qmodel.log_prob(seqs).mean() / 10), rel_tol=1e-10)
        g = p.windows(5, 4)
        assert torch.allclose(g, phase5a.window_probs(qmodel.instrument(), qmodel.rho0, 5, 4))


def test_block_counts_and_kl():
    seqs = np.array([[0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1]], dtype=np.uint8)
    c = phase6.block_counts(seqs, [0, 2, 4], 2)
    assert c.tolist() == [[0, 1, 1, 1], [1, 1, 0, 1]]
    p = np.array([0.5, 0.5, 0.0])
    assert phase6.kl(p, p) == 0.0
    assert phase6.kl(p, np.array([1.0, 0.0, 0.0])) == math.inf
    assert math.isclose(phase6.kl(p, np.array([0.25, 0.25, 0.5])), math.log(2))


def test_noise_json_roundtrip():
    hw = pytest.importorskip("qmem.hardware")
    noise = hw.StepNoise(
        p1={0: 1e-4, 1: 2e-4},
        p2={(0, 1): 3e-3},
        readout={0: 0.01, 1: 0.02},
        reset=0.01,
        t1={0: 1e-4, 1: 2e-4},
        t2=None,
        measure_time=2e-6,
    )
    back = phase6.noise_from_json(phase6.noise_to_json(noise))
    assert back == noise


def test_two_state_hmm_witness_vanishes():
    h = phase5a.control_hmm(5)
    p = phase6.Proc.hmm("c", h.labelled, h.initial)
    g = p.windows(7, 16)
    for j in range(4):
        assert abs(phase5a.hankel_witness(g, j)) < 1e-12
