"""Phase 7 Stage 2B (#21): the q59 ancilla error models against analytic and brute-force cases."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import ancilla as AN
from qmem import witness as W
from qmem.records import REPO_ROOT
from qmem.spectator import Instr

P6 = REPO_ROOT / "results" / "summaries" / "phase6"
AUDIT = REPO_ROOT / "results" / "summaries" / "phase7_audit"
ABL = REPO_ROOT / "results" / "summaries" / "phase7_ablation"
HAS_QISKIT = importlib.util.find_spec("qiskit") is not None


@pytest.fixture(scope="module")
def snap():
    s6 = json.loads((P6 / "summary.json").read_text())
    cfg = AN.config_from_snapshot(s6["prepare"]["calibration"]["noise"])
    return cfg, AN.physical_step((P6 / "step.qasm").read_text())


def _rand_rho(d, rng):
    m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    r = m @ m.conj().T
    return r / np.trace(r)


def _apply(s, rho):
    d = rho.shape[0]
    return (s @ rho.reshape(-1, order="F")).reshape(d, d, order="F")


# -- the register algebra ---------------------------------------------------------------------


def test_embed_matches_kron_and_permutes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 2)) + 0j
    b = rng.normal(size=(3, 3)) + 0j
    dims = (2, 3)
    assert np.allclose(AN.embed(a, [0], dims), np.kron(a, np.eye(3)))
    assert np.allclose(AN.embed(b, [1], dims), np.kron(np.eye(2), b))
    ab = np.kron(a, b)
    assert np.allclose(AN.embed(ab, [0, 1], dims), ab)
    # reversed site order: op on (site 1, site 0)
    ba = np.kron(b, a)
    assert np.allclose(AN.embed(ba, [1, 0], dims), ab)
    c = rng.normal(size=(9, 9)) + 0j
    full = AN.embed(c, [1, 2], (2, 3, 3))
    assert np.allclose(full, np.kron(np.eye(2), c))


# -- the calibrated model, rebuilt --------------------------------------------------------------


def test_defaults_reproduce_stage2a_pieces_and_audit_instrument(snap):
    cfg, instrs = snap
    z = np.load(ABL / "pieces.npz")
    au = np.load(AUDIT / "instruments.npz")
    for name, c in (("calibrated", cfg), ("noiseless", AN.noiseless_config())):
        ref = A.step_superops("baseline", A.StepPieces.from_npz(z, name))
        mine = AN.step_superops(instrs, AN.Register((75, 59), (2, 2)), c)
        assert np.abs(mine - ref).max() < 1e-13
        # qutrit ancilla with every new parameter at its default (eps = 0): same process
        p = AN.ancilla_process(instrs, c)
        q = W.LinearProcess.instrument(au[name], au["rho0"])
        assert np.abs(p.marginals(64) - q.marginals(64)).max() < 1e-12
        gp = p.expected_windows(7, 16, 64, common=True)
        gq = q.expected_windows(7, 16, 64, common=True)
        assert np.abs(gp - gq).max() < 1e-13
        sh = W.load_shots(str(P6 / "shots_main.npz"))[:300]
        assert np.abs(p.seq_nll(sh) - q.seq_nll(sh)).max() < 1e-11


def test_zero_leak_with_seepage_and_phase_is_calibrated(snap):
    cfg, instrs = snap
    base = AN.ancilla_process(instrs, cfg)
    c = AN.apply_params(cfg, {"seep": 0.3, "read2": 0.2, "leak_cz_phase": 1.0})
    p = AN.ancilla_process(instrs, c)
    assert np.abs(p.marginals(40) - base.marginals(40)).max() < 1e-14
    # symmetric e0 = e1 = eps is the calibrated reset exactly
    eps = cfg.noise[59].reset
    p2 = AN.ancilla_process(instrs, AN.apply_params(cfg, {"reset_e0": eps, "reset_e1": eps}))
    assert np.abs(p2.marginals(40) - base.marginals(40)).max() < 1e-15


def test_relaxation_kraus_matches_stage2a(snap):
    cfg, _ = snap
    z = np.load(ABL / "pieces.npz")
    ks = AN.relaxation_kraus(cfg.reset_time, cfg.noise[75].t1, cfg.noise[75].t2)
    assert np.allclose(A.superop(ks), A.superop(list(z["calibrated__relax_q75_reset"])))


def test_compress_is_exact(snap):
    cfg, instrs = snap
    c = AN.apply_params(cfg, {"leak": 0.01, "seep": 0.1, "reset_e0": 0.01, "reset_e1": 0.05})
    reg = AN.register_for(c, (59,), 3)
    sup = AN.step_superops(instrs, reg, c)
    full = AN.to_process(sup, reg, compress=False)
    small = AN.to_process(sup, reg)
    assert len(small.init) == 12 and len(full.init) == 36
    starts = list(range(3, 20, 4))
    assert np.allclose(full.block_probs(starts, 4), small.block_probs(starts, 4), atol=1e-14)


# -- trace preservation -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_error_models_conserve_trace(snap, seed):
    cfg, instrs = snap
    rng = np.random.default_rng(seed)
    vals = {
        "reset_e0": rng.uniform(0, 0.3),
        "reset_e1": rng.uniform(0, 0.3),
        "read10": rng.uniform(0, 0.1),
        "read01": rng.uniform(0, 0.1),
        "leak": rng.uniform(0, 0.2),
        "seep": rng.uniform(0, 0.5),
        "read2": rng.uniform(0, 1),
        "leak_cz_phase": rng.uniform(-3, 3),
        "flip01": rng.uniform(0, 0.1),
        "flip10": rng.uniform(0, 0.1),
    }
    c = AN.apply_params(cfg, vals)
    c = replace(c, memory_levels=3, memory_leak=0.05, memory_seep=0.1, memory_cz_phase=0.7)
    for step in (instrs, AN.a2_step(instrs), AN.control_step(instrs, "B2", c)):
        reg = AN.register_for(c, (59,), 3)
        sup = AN.step_superops(step, reg, c)
        rho = _rand_rho(reg.dim, rng)
        tot = sum(_apply(s, rho) for s in sup)
        assert abs(np.trace(tot) - 1) < 1e-12
        # completely positive: the Choi matrix of the channel is PSD
        d = reg.dim
        choi = np.zeros((d * d, d * d), dtype=complex)
        for i in range(d):
            for j in range(d):
                e = np.zeros((d, d), dtype=complex)
                e[i, j] = 1
                choi += np.kron(e, _apply(sum(sup), e))
        assert np.linalg.eigvalsh(choi).min() > -1e-10
    # two ancillas (B3's register)
    c3 = replace(c, noise={**c.noise, 74: c.noise[59]}, relax_idle_ancillas=True)
    reg = AN.register_for(c3, (59, 74), 3)
    sup = AN.step_superops(AN.relabel_output(instrs, 59, 74), reg, c3)
    rho = _rand_rho(reg.dim, rng)
    assert abs(np.trace(sum(_apply(s, rho) for s in sup)) - 1) < 1e-12


# -- the "never reset" limit: a classical Markov chain ------------------------------------------


def _memoryless_step(theta: float, phi: float) -> list[Instr]:
    """reset, a q59 rotation, measure: no gate touches the memory."""
    return [
        Instr("reset", (59,)),
        Instr("rz", (59,), phi),
        Instr("sx", (59,)),
        Instr("rz", (59,), theta),
        Instr("sx", (59,)),
        Instr("measure", (59,)),
    ]


def _unitary(theta, phi):
    sx = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    rz = lambda a: np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])  # noqa: E731
    return sx @ rz(theta) @ sx @ rz(phi)


@pytest.mark.parametrize("ro", [0.0, 0.03])
def test_never_reset_gives_the_predicted_markov_chain(ro):
    theta, phi = 0.9, 0.4
    u = _unitary(theta, phi)
    t = np.abs(u) ** 2  # t[b', b] = P(b' | b)
    c = AN.noiseless_config()
    c = replace(c, noise={59: AN.SiteNoise(readout=ro), 75: AN.SiteNoise()})
    c = replace(c, ancilla={59: AN.AncillaParams(never_reset=True)})
    p = AN.ancilla_process(_memoryless_step(theta, phi), c)
    # eps = 1 in the sense of the state-dependent reset (e0 = 0, e1 = 1) is the same thing
    c1 = replace(c, ancilla={59: AN.AncillaParams(reset_e0=0.0, reset_e1=1.0)})
    p1 = AN.ancilla_process(_memoryless_step(theta, phi), c1)
    # hidden Markov chain on q59's level, emitting a flipped bit with probability ro
    e = np.array([[1 - ro, ro], [ro, 1 - ro]])  # e[y, b]
    lab = np.einsum("yj,ji->yij", e, t)  # L[y, i, j] = P(j | i) P(y | j)
    hmm = W.LinearProcess.hmm(lab, np.array([1.0, 0.0]))
    starts = [0, 2, 5]
    for q in (p, p1):
        assert np.allclose(q.block_probs(starts, 5), hmm.block_probs(starts, 5), atol=1e-13)
        g = q.expected_windows(7, 4, 30, common=True)
        for j in range(4):
            assert abs(W.witness(g, j)) < 1e-12
    if ro == 0:  # a plain Markov chain: P(y_{t+1} | y_t) = t
        b2 = p.block_probs([3], 2).reshape(2, 2)
        cond = b2 / b2.sum(1, keepdims=True)
        assert np.allclose(cond, t.T, atol=1e-13)


def test_residual_reset_alone_keeps_rank_two():
    """A memoryless circuit with any state-dependent reset: a 2-state process, w = 0."""
    c = AN.noiseless_config()
    c = replace(c, noise={59: AN.SiteNoise(readout=0.01), 75: AN.SiteNoise()})
    for e0, e1 in ((0.01, 0.2), (0.3, 0.0), (0.5, 0.5)):
        cc = replace(c, ancilla={59: AN.AncillaParams(reset_e0=e0, reset_e1=e1)})
        g = AN.ancilla_process(_memoryless_step(1.1, 0.2), cc).expected_windows(7, 5, 40, True)
        assert max(abs(W.witness(g, j)) for j in range(4)) < 1e-12


def test_leakage_matches_brute_force_three_state_hmm():
    """Memoryless circuit + leakage: q59's level {0, 1, 2} after the readout is a 3-state HMM."""
    theta, phi, leak, seep, r2, e0, e1 = 0.8, 0.3, 0.07, 0.2, 0.85, 0.02, 0.05
    u = _unitary(theta, phi)
    pu = np.abs(u) ** 2
    c = AN.noiseless_config()
    c = replace(c, noise={59: AN.SiteNoise(), 75: AN.SiteNoise()})
    c = replace(
        c,
        ancilla={59: AN.AncillaParams(reset_e0=e0, reset_e1=e1, leak=leak, seep=seep, read2=r2)},
    )
    p = AN.ancilla_process(_memoryless_step(theta, phi), c)
    # brute force: state i = level after the readout (post-leak)
    lab = np.zeros((2, 3, 3))
    for i in range(3):
        pre = {0: [(0, 1.0)], 1: [(1, 1.0)], 2: [(2, 1 - seep), (1, seep)]}[i]  # seepage
        for lvl, w in pre:
            after = {0: [(0, 1 - e0), (1, e0)], 1: [(0, 1 - e1), (1, e1)], 2: [(2, 1.0)]}[lvl]
            for r, w2 in after:  # reset result r, then the rotation, then the readout
                outs = [(b, pu[b, r]) for b in (0, 1)] if r < 2 else [(2, 1.0)]
                for b, w3 in outs:
                    if b == 0:
                        lab[0, i, 0] += w * w2 * w3
                    elif b == 1:
                        lab[1, i, 1] += w * w2 * w3 * (1 - leak)
                        lab[1, i, 2] += w * w2 * w3 * leak
                    else:
                        lab[1, i, 2] += w * w2 * w3 * r2
                        lab[0, i, 2] += w * w2 * w3 * (1 - r2)
    hmm = W.LinearProcess.hmm(lab, np.array([1.0, 0, 0]))
    starts = [0, 1, 4, 9]
    assert np.allclose(p.block_probs(starts, 5), hmm.block_probs(starts, 5), atol=1e-13)


def test_non_qnd_flips_are_degenerate_with_reset_residue(snap):
    cfg, instrs = snap
    e0, e1, f01, f10 = 0.004, 0.02, 0.03, 0.01
    ca = AN.apply_params(cfg, {"reset_e0": e0, "reset_e1": e1, "flip01": f01, "flip10": f10})
    cb = AN.apply_params(
        cfg,
        {"reset_e0": (1 - f01) * e0 + f01 * e1, "reset_e1": (1 - f10) * e1 + f10 * e0},
    )
    reg = AN.register_for(ca, (59,), 3)
    sa, sb = AN.step_superops(instrs, reg, ca), AN.step_superops(instrs, reg, cb)
    # the flips act between the readout and the next reset: process a with the flip
    # applied to rho0 equals process b from rho0 (b's reset already contains the flip)
    pb = AN.to_process(sb, reg)
    flip = np.array([[1 - f01, f10], [f01, 1 - f10]])  # q59 = |0> initially
    rho = np.zeros((6, 6), dtype=complex)
    rho[0, 0], rho[1, 1] = flip[0, 0], flip[1, 0]  # q75 = 0, q59 in {0, 1}
    full = W.LinearProcess.instrument(sa, rho)
    k = AN.diag_subspace(reg)
    pa = W.LinearProcess(full.init[k], full.ops[:, k][:, :, k], full.final[k])
    starts = [0, 3, 10]
    assert np.allclose(pa.block_probs(starts, 4), pb.block_probs(starts, 4), atol=1e-13)


# -- time-inhomogeneous and post-selected statistics -----------------------------------------


def test_seq_process_matches_linear_process(snap):
    cfg, instrs = snap
    p = AN.ancilla_process(instrs, AN.apply_params(cfg, {"leak": 0.01, "seep": 0.1}))
    sp = AN.SeqProcess.homogeneous(p, 40)
    assert np.allclose(sp.windows(5, 8), p.expected_windows(5, 8, 40, common=True), atol=1e-14)
    assert np.allclose(sp.marginals(), p.marginals(40), atol=1e-14)
    starts = list(range(8, 40 - 4 + 1, 4))
    assert np.allclose(sp.block_probs(4, 8), p.block_probs(starts, 4), atol=1e-14)
    assert abs(sp.accept_prob() - 1) < 1e-12


def test_post_selection_matches_enumeration(snap):
    cfg, instrs = snap
    c = AN.apply_params(cfg, {"reset_e0": 0.05, "reset_e1": 0.2, "leak": 0.05, "seep": 0.2})
    reg = AN.register_for(c, (59,), 3)
    ops = AN.compress_ops(AN.step_superops(AN.control_step(instrs, "B2", c), reg, c), reg)
    init, final = AN.init_final(reg)
    length, k_max, burn = 6, 2, 0
    sp = AN.SeqProcess(init, [np.stack([ops[0], ops[1]])] * length, final)
    # enumerate all (f, y) sequences; keep all-f = 0
    tot, g = 0.0, np.zeros(k_max + 1)
    last = length - k_max - 2
    for seq in itertools.product(range(4), repeat=length):
        if any(o // 2 for o in seq):
            continue
        s = init
        for o in seq:
            s = s @ ops[o]
        pr = float((s @ final).real)
        y = [o % 2 for o in seq]
        tot += pr
        for t in range(burn, last + 1):
            for k in range(k_max + 1):
                if y[t] == 1 and y[t + k + 1] == 1 and not any(y[t + 1 : t + k + 1]):
                    g[k] += pr
    g /= tot * (last - burn + 1)
    assert abs(sp.accept_prob() - tot) < 1e-13
    assert np.allclose(sp.windows(k_max, burn), g, atol=1e-13)


def test_flag_stats_null_has_no_history_dependence(snap):
    cfg, instrs = snap
    for vals, zero in (({}, True), ({"reset_e0": 0.001, "reset_e1": 0.02}, False)):
        c = AN.apply_params(cfg, vals)
        reg = AN.register_for(c, (59,), 3)
        ops = AN.compress_ops(AN.step_superops(AN.control_step(instrs, "B2", c), reg, c), reg)
        init, final = AN.init_final(reg)
        f = AN.flag_stats(init, ops, final, 64, 16)
        contrast = f["p_flag_given_y1"] - f["p_flag_given_y0"]
        if zero:  # symmetric reset forgets q59: the flag is independent of y_{t-1}
            assert abs(contrast) < 1e-14
            # P(flag) = eps (left in |1>) * (1 - r) + (1 - eps) * r
            eps, r = cfg.noise[59].reset, cfg.noise[59].readout
            assert abs(f["p_flag"] - (eps * (1 - r) + (1 - eps) * r)) < 1e-12
        else:
            assert contrast > 0.01


def test_sample_outcomes_agree_with_exact(snap):
    cfg, instrs = snap
    p = AN.ancilla_process(instrs, AN.apply_params(cfg, {"leak": 0.02, "seep": 0.1}))
    o = AN.sample_outcomes(p.init, [p.ops] * 32, p.final, 4000, 3)
    m = o.mean(0)
    exact = p.marginals(32)
    se = np.sqrt(exact * (1 - exact) / 4000)
    assert np.all(np.abs(m - exact) < 5 * se)


# -- A2 with q75 leakage, and the controls ----------------------------------------------------


def test_a2_without_leakage_has_zero_witness(snap):
    cfg, instrs = snap
    step = AN.a2_step(instrs)
    for leak in (0.0, 0.02):
        c = replace(cfg, memory_levels=3, memory_leak=leak, memory_seep=0.05)
        reg = AN.register_for(c, (59,), 2)
        sup = AN.step_superops(step, reg, c, record=[1])
        g = AN.to_process(sup, reg).expected_windows(7, 16, 64, common=True)
        w0 = W.witness(g, 0)
        if leak == 0:
            assert abs(w0) < 1e-15
        else:
            assert abs(w0) > 1e-9  # a classical 3-level q75 can give w != 0


def test_double_reset_squares_the_state_dependence():
    e0, e1 = 0.01, 0.07
    s = A.superop(A.reset_kraus(e0, e1))
    two = s @ s
    for b in (0, 1):
        rho = np.diag([1.0 - b, b]).astype(complex)
        out = _apply(two, rho)
        p1 = e0 + (e0 if b == 0 else e1) * (e1 - e0)
        assert abs(out[1, 1].real - p1) < 1e-15
    # dependence on the input: (e1 - e0)^2
    d = (
        _apply(two, np.diag([0, 1.0]).astype(complex))[1, 1]
        - _apply(two, np.diag([1.0, 0]).astype(complex))[1, 1]
    )
    assert abs(d.real - (e1 - e0) ** 2) < 1e-15


def test_control_qasm_edits():
    text = (P6 / "circuit.qasm").read_text()
    b1 = AN.control_qasm(text, "B1")
    assert b1.count("reset $59;") == 2 * text.count("reset $59;")
    b2 = AN.control_qasm(text, "B2")
    assert b2.count("measure $59;") == text.count("measure $59;") + text.count("reset $59;")
    b3 = AN.control_qasm(text, "B3")
    assert b3.count("measure $74;") == 32 and b3.count("measure $59;") == 32
    assert b3.count("cz $75, $74;") == 32 * 3
    for v, ns in (("B4r", 553), ("B4m", 545)):
        q = AN.control_qasm(text, v)
        assert q.count(f"delay[{ns}dt] $59;") == text.count("reset $59;")


@pytest.mark.skipif(not HAS_QISKIT, reason="needs the hardware group")
def test_control_qasm_loads_and_is_native():
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    t = FakeKingston().target
    text = (P6 / "circuit.qasm").read_text()
    for v in ("B1", "B2", "B3", "B4r", "B4m"):
        qc = qasm3.loads(AN.control_qasm(text, v))
        for inst in qc.data:
            qs = tuple(qc.find_bit(q).index for q in inst.qubits)
            assert t.instruction_supported(inst.operation.name, qs), (v, inst.operation.name, qs)


def test_nelder_mead_finds_a_quadratic_minimum():
    f = lambda x: float((x[0] - 1.5) ** 2 + 3 * (x[1] + 0.5) ** 2 + 0.5 * x[0] * x[1])  # noqa: E731
    x, fx, _ = AN.nelder_mead(f, np.array([0.0, 0.0]), xtol=1e-8, ftol=1e-14, max_evals=2000)
    h = np.array([[2, 0.5], [0.5, 6]])
    xs = np.linalg.solve(h, np.array([3.0, -3.0]))
    assert np.allclose(x, xs, atol=1e-6)
    assert math.isfinite(fx)
