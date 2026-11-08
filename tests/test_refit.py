"""Phase 7 Stage 4b (#26): q75's decay under mid-circuit measurement (``qmem.refit``).

Analytic cases (a one-qubit Markov chain with a closed-form stationary record), brute-force
enumeration of the joint record, the independent q59-qutrit simulator (``qmem.ancilla``) and
parameter recovery on simulated records.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import replace

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import ancilla as AN
from qmem import refit as R
from qmem.records import REPO_ROOT
from qmem.spectator import Instr

P6 = REPO_ROOT / "results" / "summaries" / "phase6"
ABL = REPO_ROOT / "results" / "summaries" / "phase7_ablation"
I2 = np.eye(2, dtype=complex)


@pytest.fixture(scope="module")
def cal() -> A.StepPieces:
    return A.StepPieces.from_npz(np.load(ABL / "pieces.npz"), "calibrated")


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def toy_pieces(theta: float) -> A.StepPieces:
    """q75 rotated by ry(theta) each step, q59 idle at |0> (y = 0 always), no noise."""
    one = np.array([I2])
    post = np.stack([np.eye(16, dtype=complex), np.zeros((16, 16), dtype=complex)])
    return A.StepPieces(
        pre=A.on_q75([ry(theta)]),
        post=post,
        relax_q75_reset=one,
        relax_q59_reset=one,
        relax_q59_measure=one,
        eps_q59=0.0,
        eps_q75=0.0,
        p1_q75=0.0,
    )


# -- the instrument -----------------------------------------------------------------------------


def test_meas_kraus_is_complete_and_decays_one_to_zero():
    for a in (0.0, 0.1, 0.7):
        k0, k1 = A.meas_kraus(a)
        tot = sum(k.conj().T @ k for k in k0 + k1)
        assert np.allclose(tot, I2)
        one = np.diag([0, 1.0]).astype(complex)
        p0 = sum(np.trace(k @ one @ k.conj().T).real for k in k0)
        assert math.isclose(p0, a, abs_tol=1e-15)
        post0 = sum(k @ one @ k.conj().T for k in k0)
        assert np.allclose(post0, np.diag([a, 0]))  # recorded 0 -> left in |0>


def test_a2_op_with_decay_is_damping_after_dephasing(cal):
    rng = np.random.default_rng(3)
    g = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = g @ g.conj().T
    rho /= np.trace(rho)
    a = 0.12
    s = A.op_superop("A2", cal, A.Hypothesis(meas_decay=a))
    out = (s @ rho.reshape(-1, order="F")).reshape(4, 4, order="F")
    q75 = A.ptrace_q59(out)
    q75_in = A.ptrace_q59(rho)
    assert math.isclose(q75[1, 1].real, (1 - a) * q75_in[1, 1].real, rel_tol=1e-12)
    assert abs(q75[0, 1]) < 1e-15
    # decay 0 is the old projective op exactly; summed records equal the A2 step
    assert np.array_equal(
        A.op_superop("A2", cal, A.Hypothesis(meas_decay=0.0)),
        A.op_superop("A2", cal, A.Hypothesis()),
    )
    rec = A.a2_record_superops(cal, 0.015, a)
    assert np.abs(rec.sum(0) - A.step_superops("A2", cal, A.Hypothesis(meas_decay=a))).max() < 1e-14


# -- analytic and brute-force cases --------------------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "a", "a0"), [(1.1, 0.0, None), (2.0, 0.15, 0.05), (0.6, 0.4, 0.4)]
)
def test_toy_chain_stationary_record_is_closed_form(theta, a, a0):
    s = math.sin(theta / 2) ** 2  # P(|1> before the measurement) from |0>; from |1>: 1 - s
    mod = R.RecordModel.build(toy_pieces(theta), 0.0, a, a0)
    pm = mod.marginals(200)["m"]
    first = a if a0 is None else a0
    assert math.isclose(pm[0], (1 - first) * s, abs_tol=1e-14)
    # pi = P(state 1 after the measurement) = P(m = 1): pi = (1 - a)[(1 - pi) s + pi (1 - s)]
    pi = (1 - a) * s / (1 - (1 - a) * (1 - 2 * s))
    assert math.isclose(pm[-1], pi, abs_tol=1e-12)
    # with a readout flip r the record is pi (1 - r) + (1 - pi) r
    r = 0.03
    pr = R.RecordModel.build(toy_pieces(theta), r, a, a0).marginals(200)["m"][-1]
    assert math.isclose(pr, pi * (1 - r) + (1 - pi) * r, abs_tol=1e-12)


def test_forward_nll_equals_enumeration(cal):
    mod = R.RecordModel.build(cal, 0.009, 0.12, 0.05)
    v0 = A.RHO0_JOINT.reshape(-1, order="F")
    tr = np.eye(4).reshape(-1, order="F")
    seqs = list(itertools.product(range(4), repeat=3))
    probs = []
    for seq in seqs:
        v = v0.astype(complex)
        for t, k in enumerate(seq):
            v = mod.step(t)[k // 2, k % 2] @ v
        probs.append((tr @ v).real)
    probs = np.array(probs)
    assert math.isclose(probs.sum(), 1.0, abs_tol=1e-13)
    assert probs.min() > 0
    arr = np.array(seqs)
    m, y = arr // 2, arr % 2
    assert np.allclose(mod.nll(m, y), -np.log(probs), atol=1e-12)
    # marginals by enumeration
    for t in range(3):
        assert math.isclose(mod.marginals(3)["m"][t], probs[m[:, t] == 1].sum(), abs_tol=1e-13)


def test_decay_zero_reproduces_stage2a_record_model(cal):
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import phase7_ablation as PA

    acfg = PA.load_config(REPO_ROOT / "experiments" / "configs" / "phase7_ablation.toml")
    ref = PA.a2_record(acfg, {"calibrated": cal})
    r = acfg["a2_record"]["readout_error"]
    stats = R.RecordModel.build(cal, r).marginals(64)
    assert math.isclose(stats["m"][16], ref["p_m1"], abs_tol=1e-12)
    pair = R.RecordModel.build(cal, r).pair_stats(64, 16)
    assert math.isclose(pair["p_m1"], ref["p_m1"], abs_tol=1e-6)  # stationary by step 16


def test_ancilla_simulator_memory_decay_matches(cal):
    """``qmem.ancilla`` (built from the circuit, qutrit-capable) against #20's pieces."""
    s6 = json.loads((P6 / "summary.json").read_text())
    cfg = AN.config_from_snapshot(s6["prepare"]["calibration"]["noise"])
    base = AN.physical_step((P6 / "step.qasm").read_text())
    steps, done = [], False
    for i in base:
        if i.name == "cz" and not done:
            steps.append(Instr("measure", (75,)))
            done = True
        steps.append(i)
    r = cfg.noise[75].readout
    reg = AN.Register((75, 59), (2, 2))
    an = {
        a: AN.step_superops(steps, reg, replace(cfg, memory_decay=a), record=[0, 1])
        for a in (0, 0.12)
    }
    mine = {a: A.a2_record_superops(cal, r, a).reshape(4, 16, 16) for a in (0, 0.12)}
    # the two simulators differ slightly already at decay 0 (q59's idling during q75's
    # readout is modelled differently in #20 and #21); the decay term itself agrees
    assert np.abs(an[0] - mine[0]).max() < 3e-5
    assert np.abs((an[0.12] - an[0]) - (mine[0.12] - mine[0])).max() < 5e-6

    def p_m1(p, n=16):
        v = p.init
        for _ in range(n):
            v = v @ p.ops.sum(0)
        return float(((v @ (p.ops[2] + p.ops[3])) @ p.final).real)

    # a qutrit memory with no leakage gives the same record as the qubit memory
    c3 = replace(cfg, memory_decay=0.12, memory_levels=3)
    reg3 = AN.Register((75, 59), (3, 2))
    p3 = AN.to_process(AN.step_superops(steps, reg3, c3, record=[0, 1]), reg3, compress=False)
    p2 = AN.to_process(an[0.12], reg, compress=False)
    assert math.isclose(p_m1(p3), p_m1(p2), abs_tol=1e-12)
    for a in (0, 0.12):
        pm = R.RecordModel.build(cal, r, a).marginals(17)["m"][16]
        assert math.isclose(p_m1(AN.to_process(an[a], reg, compress=False)), pm, abs_tol=1e-5)


# -- the fit ------------------------------------------------------------------------------------


def test_fit_recovers_simulated_parameters(cal):
    truth = {"decay_first": 0.05, "decay": 0.12}
    mod = R.RecordModel.build(cal, 0.009, truth["decay"], truth["decay_first"])
    m, y = mod.sample(3000, 32, seed=11)
    # the sampler against the exact marginals
    ex = mod.marginals(32)
    assert abs(m[:, 0].mean() - ex["m"][0]) < 4 * math.sqrt(0.25 / 3000)
    assert abs(m[:, 16:].mean() - ex["m"][16:].mean()) < 0.015
    f1 = R.fit(m, y, cal, 0.009, "decay_first", x0=[0.1, 0.1])
    assert abs(f1.params["decay"] - 0.12) < 0.015
    assert abs(f1.params["decay_first"] - 0.05) < 0.03
    f0 = R.fit(m, y, cal, 0.009, "calibrated")
    assert f0.nll > f1.nll + 50  # the calibrated model is strongly rejected on these data
    pred = R.predicted_p_m1(cal, 0.009, f1.params["decay"], f1.params["decay_first"], 32, 16)
    assert abs(pred - m[:, 16:].mean()) < 0.015
