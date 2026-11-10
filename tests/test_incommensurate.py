"""Phase 8 Stage 2c (#35): the 1 rad echoes, Z_n registers, the free-permutation adversary."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from qmem import ablate_hw as A
from qmem import classical as C
from qmem import coherence as Q
from qmem import register as RG

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_coherence import pieces

TIER = {
    "RAD1": C.ArmSpec("RAD1", (("r1",),)),
    "RADM1": C.ArmSpec("RADM1", (("rm1",),)),
    "INV1": C.ArmSpec("INV1", (("r1", "rm1"),)),
    "COMP11": C.ArmSpec("COMP11", (("r1", "r1"),)),
    "RAD2": C.ArmSpec("RAD2", (("r2", "e0"),)),
    "CUBE1": C.ArmSpec("CUBE1", (("r1", "r1", "r1"),)),
    "RAD3": C.ArmSpec("RAD3", (("r3", "e0", "e0"),)),
    "E00": C.ArmSpec("E00", (("e0", "e0"),)),
    "P2": C.ArmSpec("P2", (("e2",),)),
    "A2": C.ArmSpec("A2", (("meas",),)),
}
ELEMENTS = ["e0", "e2", "r1", "rm1", "r2", "r3"]
FREE = ["r1", "rm1", "r2", "r3"]


def test_element_phase():
    assert Q.element_phase("e3") == pytest.approx(3 * math.pi / 4)
    assert Q.element_phase("r1") == 1.0 and Q.element_phase("rm1") == -1.0
    assert Q.element_phase("r3") == 3.0
    for name in ("reset", "meas", "ff01", "I", "X", "rx1", "p:1.0"):
        assert Q.element_phase(name) is None


@pytest.mark.parametrize("name", ["r1", "rm1", "r2", "r3"])
def test_radian_echo_is_rz_and_frame_neutral(name):
    phi = Q.element_phase(name)
    u = Q.echo_unitary(phi)
    rz = A.rz_unitary(phi)
    ph = np.vdot(rz.ravel(), u.ravel()) / 2  # global phase
    assert abs(abs(ph) - 1) < 1e-12
    assert np.allclose(u, ph * rz, atol=1e-12)
    a, b = Q.echo_angles(phi)
    assert a + b == 0.0


def test_tier_compositions_are_exact_in_the_quantum_model_with_pulse_noise():
    """INV(1, -1) = E00, COMP(1, 1) = RAD2 and Rz(1)^3 = RAD3 as superoperators (pulse
    depolarising noise commutes with Z rotations; the pulse counts are matched)."""
    qm = Q.QuantumModel(pieces(2), 0.01, Q.NoiseParams(dephase=0.05, detune=0.2))
    for a, b in (("INV1", "E00"), ("COMP11", "RAD2"), ("CUBE1", "RAD3")):
        assert np.allclose(qm.slot(TIER[a].programs[0]), qm.slot(TIER[b].programs[0]), atol=1e-13)
    # and an x amplitude error breaks them (what the refit's ``over`` absorbs)
    bad = Q.QuantumModel(pieces(2), 0.01, Q.NoiseParams(over=0.05))
    assert not np.allclose(bad.slot(TIER["INV1"].programs[0]), bad.slot(TIER["E00"].programs[0]))


def _brute_prob(pc: A.StepPieces, p: Q.NoiseParams, prog, ys) -> float:
    """``P(y)`` by explicit density-matrix evolution on the 4 x 4 (q75, q59) state: every echo
    applied as its Kraus operators (``x`` pulses with depolarising, the ``rz`` frame changes),
    the pieces as superoperators, the flip as a mixture of the physical outcomes."""
    reset = A.reset_superop(pc, A.Hypothesis())
    extra = A.kraus_kron(
        Q.compose(A.amp_damp_kraus(p.damp), A.dephase_kraus(p.dephase), [A.rz_unitary(p.detune)]),
        [np.eye(2)],
    )
    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    unvec = lambda v: v.reshape(4, 4, order="F")  # noqa: E731
    xk = Q.compose([Q.rx(math.pi)], A.depol_kraus(pc.p1_q75))

    def run(rho, ys_phys):
        for y in ys_phys:
            rho = unvec(reset @ vec(rho))
            rho = sum(k @ rho @ k.conj().T for k in extra)
            rho = unvec(pc.pre @ vec(rho))
            for e in prog:
                a = -Q.element_phase(e) / 2  # x, rz(a), x, rz(-a)
                for ks in (xk, [A.rz_unitary(a)], xk, [A.rz_unitary(-a)]):
                    kk = A.kraus_kron(ks, [np.eye(2)])
                    rho = sum(k @ rho @ k.conj().T for k in kk)
            rho = unvec(pc.post[y] @ vec(rho))
        return float(np.trace(rho).real)

    tot = 0.0
    for ys_phys in itertools.product((0, 1), repeat=len(ys)):
        w = 1.0
        for a, b in zip(ys, ys_phys, strict=True):
            w *= (1 - p.flip) if a == b else p.flip
        tot += w * run(A.RHO0_JOINT, ys_phys)
    return tot


@pytest.mark.parametrize("arm", ["RAD1", "RADM1", "INV1", "COMP11", "RAD2", "CUBE1", "RAD3"])
def test_tier_likelihood_against_brute_force_density_matrix(arm):
    pc = pieces(4)
    p = Q.NoiseParams(dephase=0.1, damp=0.05, detune=0.3, flip=0.03)
    qm = Q.QuantumModel(pc, 0.01, p)
    ys = np.array(list(itertools.product((0, 1), repeat=3)))
    nll = qm.nll(TIER[arm], C.ProgramData(ys))
    brute = np.array([_brute_prob(pc, p, TIER[arm].programs[0], y) for y in ys])
    assert np.allclose(np.exp(-nll), brute, atol=1e-12)
    assert np.exp(-nll).sum() == pytest.approx(1.0, abs=1e-12)


def test_radian_echoes_are_not_a_finite_phase_group():
    """Rz(1)^k is never the identity up to a global phase for 1 <= k <= 1000 (1 / 2 pi is
    irrational): the quantum response has no period a Z_n register could match. The nearest
    return is k = 710 (710 / 226 ~ pi), where 2 - |tr| ~ 9e-10."""
    u = Q.echo_unitary(1.0)
    v = np.eye(2, dtype=complex)
    gaps = []
    for _ in range(1000):
        v = u @ v
        gaps.append(2 - abs(np.trace(v)))
    assert min(gaps) > 1e-12
    assert int(np.argmin(gaps)) + 1 == 710


# -- Z_n registers ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 12, 16, 24, 32])
def test_nearest_shifts(n):
    for k in range(8):
        s = RG.echo_shift(f"e{k}", n)
        if (k * n) % 8 == 0:
            assert s == k * n // 8 and RG.shift_error(f"e{k}", n) < 1e-12
        assert RG.shift_error(f"e{k}", n) <= math.pi / n + 1e-12
    for e in ("r1", "r2", "r3"):
        assert RG.shift_error(e, n) <= math.pi / n + 1e-12
        assert RG.echo_shift("rm" + e[1:], n) == -RG.echo_shift(e, n)
    # INV(1, -1) is the identity in every Z_n (the Lemma's escape)
    for n_pop in (1, 2):
        lhs = RG.shift(RG.echo_shift("r1", n), n_pop, n) @ RG.shift(
            RG.echo_shift("rm1", n), n_pop, n
        )
        assert np.array_equal(lhs, np.eye(n * n_pop))
    assert RG.echo_shift("reset", n) is None


def test_z24_embeds_z8_and_approximates_one_radian():
    assert RG.echo_shift("r1", 24) == 4 and RG.shift_error("r1", 24) == pytest.approx(
        math.pi / 3 - 1, abs=1e-12
    )
    for k in range(8):
        assert RG.echo_shift(f"e{k}", 24) == 3 * k


@pytest.mark.parametrize("n", [12, 16, 24])
def test_prism_embeds_the_ideal_rotation_as_the_shift_at_any_n(n):
    u = RG.prism(2, 1 / math.cos(math.pi / n), n_phase=n)
    for k in range(n):
        a = 2 * math.pi * k / n
        t = np.eye(4)
        t[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        assert np.allclose(t @ u, u @ RG.shift(k, 2, n).T, atol=1e-12)


def _model(n: int, n_pop: int, seed: int, free=()) -> C.ProgramModel:
    rng = np.random.default_rng(seed)
    return RG.random_register(n_pop, ELEMENTS, rng, ["meas"], n_phase=n, free=free)


def test_register_echoes_and_free_echoes():
    m = _model(12, 1, 0)
    assert RG.is_register(m, 1, 12)
    assert np.array_equal(m.maps["r1"], RG.shift(2, 1, 12))
    assert np.array_equal(m.maps["rm1"], RG.shift(-2, 1, 12))
    r = _model(12, 1, 0, FREE)
    assert RG.echo_names(r, FREE) == ["e0", "e2"]
    assert not RG.is_permutation(r.maps["r1"]) and RG.is_register(r, 1, 12, FREE)
    assert not RG.is_register(r, 1, 12)
    assert RG.n_free_params(r, FREE) == RG.n_free_params(m) + 4 * 12 * 11


@pytest.mark.parametrize("free", [(), ("r1", "rm1", "r2", "r3")])
def test_zn_register_likelihood_against_path_enumeration(free):
    """Z_6 (d = 6; 1 rad -> shift 1, pi / 2 -> shift 2): the nearest-shift register and the
    relaxed one (1 rad echoes free) against every hidden path."""
    rng = np.random.default_rng(3)
    m = _model(6, 1, 1, free)
    assert RG.echo_shift("r1", 6) == 1 and RG.echo_shift("e2", 6) == 2
    for name in ("RAD1", "RADM1", "INV1", "COMP11", "CUBE1", "RAD3", "P2"):
        steps = 1 if free or len(TIER[name].programs[0]) > 2 else 2
        for _ in range(2):
            y = rng.integers(0, 2, steps)
            brute = C.brute_force_program_prob(m, TIER[name], list(y), None)
            nll = m.nll(TIER[name], C.ProgramData(y[None]))[0]
            assert math.isclose(math.exp(-nll), brute, rel_tol=1e-10)


def test_assignment_against_brute_force():
    rng = np.random.default_rng(0)
    for d in (1, 2, 3, 5, 6):
        for _ in range(10):
            a = rng.normal(size=(d, d))
            if d > 2:
                a[0, 1] = -np.inf  # a forbidden pair
            s = RG.assignment(a)
            assert sorted(s) == list(range(d))
            best = max(sum(a[i, p[i]] for i in range(d)) for p in itertools.permutations(range(d)))
            assert sum(a[i, s[i]] for i in range(d)) == pytest.approx(best)


def test_best_permutation_recovers_a_planted_permutation():
    rng = np.random.default_rng(1)
    d = 24
    sigma = rng.permutation(d)
    p = RG.permutation_matrix(sigma)
    noisy = 0.7 * p + 0.3 * rng.dirichlet(np.ones(d), size=d)
    noisy_inv = 0.7 * p.T + 0.3 * rng.dirichlet(np.ones(d), size=d)
    assert np.array_equal(RG.best_permutation(noisy, noisy_inv), sigma)
    assert np.array_equal(
        RG.permutation_matrix(sigma) @ RG.permutation_matrix(np.argsort(sigma)), np.eye(d)
    )


def test_frozen_free_permutation_survives_em_and_composes_exactly():
    """The strongest finite-permutation adversary: e(1) = sigma, e(-1) = sigma^-1 frozen, e(2)
    and e(3) free; EM increases the likelihood and INV(1, -1) stays E00's identity slot."""
    import phase8_coherence_null as NL

    truth = _model(8, 2, 4, FREE)
    rng = np.random.default_rng(5)
    arms = {k: TIER[k] for k in ("RAD1", "INV1", "COMP11", "RAD2", "E00", "P2", "A2")}
    data = {a: NL.simulate_program(truth, arm, 40, 10, rng) for a, arm in arms.items()}
    m = _model(8, 2, 6, FREE)
    sigma = rng.permutation(m.d)
    m.maps["r1"] = RG.permutation_matrix(sigma)
    m.maps["rm1"] = RG.permutation_matrix(sigma).T.copy()
    frozen = RG.echo_names(m, ["r2", "r3"])
    assert {"r1", "rm1", "e0", "e2"} <= set(frozen)
    fit = C.fit_program_model(arms, data, m, 10, frozen=frozen)
    assert -fit.nll >= C.program_loglik(m, arms, data) - 1e-9
    assert np.array_equal(fit.model.maps["r1"], m.maps["r1"])
    assert np.array_equal(fit.model.slot(("r1", "rm1")), np.eye(m.d))
    assert not np.array_equal(fit.model.maps["r2"], m.maps["r2"])


def test_tier_cost_adds_no_circuit_time():
    import phase8_coherence as PC
    import phase8_coherence_incommensurate as IC

    cfg = PC.load_config(
        Path(__file__).resolve().parents[1] / "experiments/configs/phase8_coherence.toml"
    )
    arms = Q.parse_arms(cfg["incommensurate"]["arms"])
    assert len(arms) == 7
    for arm in arms.values():
        assert IC.tier_circuit_s(cfg, arm) == cfg["budget"]["baseline_circuit_s"]
    merged = IC.merge_arms(cfg)["arms"]
    assert list(merged)[:18] == list(
        Q.parse_arms(
            PC.load_config(
                Path(__file__).resolve().parents[1] / "experiments/configs/phase8_coherence.toml"
            )["arms"]
        )
    )
