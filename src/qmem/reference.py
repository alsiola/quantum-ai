"""Phase 8 Stage 3 (issue #33): can q75's step channel be certified non-entanglement-breaking?

Feasibility only: theory, simulation and circuits for an unsubmitted plan. Nothing here talks
to an IBM service.

**The question.** The trained step instrument {M_y} acts on q75 (the memory) and emits y on
q59. Its unconditional channel is Phi = sum_y M_y; a branch is M_y for a predeclared record y.
A channel is entanglement-breaking (EB) iff it is measure-and-prepare, iff its Choi state
J = (id ⊗ Phi)(Phi+) is separable [Horodecki2003EB]. Two trusted tests follow:

* **Bell route.** Send half of a Bell pair through Phi^n and measure XX, YY, ZZ. A separable
  two-qubit state has fidelity <= 1/2 with every maximally entangled state, so a lower
  confidence bound above 1/2 certifies non-EB (trusted end measurements, an isolated
  reference).
* **Prepare-and-measure (P&M) route.** Prepare q75 in the six axis states (a 2-design), apply
  Phi^n, measure the projector onto the predeclared target V psi_k. The average score is
  F_avg = (2 F_e + 1) / 3, with F_e the Bell fidelity of J against (1 ⊗ V)|Phi+>, so the
  EB bound is 2/3 (trusted preparation and measurement, no reference qubit).

Both are linear in J. For a branch (a trace-decreasing CP map) the same bounds hold for the
*pooled ratio* (successes in the branch over shots in the branch, inputs equally allocated),
because an EB CP map still has a separable (unnormalised) Choi operator.

Conventions: two-qubit operators are (reference, q75), big-endian, reference first, as in
``tests/test_classical.py``. The step's superoperators come from ``qmem.ablate_hw`` on the
joint (q75, q59) memory (q75 first) and act on column-stacked ``vec(rho)``. numpy only, except
the circuit builders at the end, which import Qiskit lazily (the ``hardware`` group).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

from . import ablate_hw as A

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
PAULI = (I2, X, Y, Z)
PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
BELL = np.outer(PHI_PLUS, PHI_PLUS.conj())

_S2 = 1 / math.sqrt(2)
#: The six axis states (an octahedron, a 3-design): +Z, -Z, +X, -X, +Y, -Y.
AXIS_LABELS = ("+z", "-z", "+x", "-x", "+y", "-y")
AXIS_KETS = (
    np.array([1, 0], dtype=complex),
    np.array([0, 1], dtype=complex),
    np.array([_S2, _S2], dtype=complex),
    np.array([_S2, -_S2], dtype=complex),
    np.array([_S2, 1j * _S2], dtype=complex),
    np.array([_S2, -1j * _S2], dtype=complex),
)
#: The unitary that prepares each axis state from |0> (H, X·H, ...: written as matrices).
_H = np.array([[1, 1], [1, -1]], dtype=complex) * _S2
_SG = np.diag([1, 1j]).astype(complex)
AXIS_PREP = (I2, X, _H, _H @ X, _SG @ _H, _SG.conj() @ _H)


def dm(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


# -- two-qubit state functions ------------------------------------------------------------------


def correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """T_ij = tr(rho sigma_i ⊗ sigma_j), i, j in (x, y, z)."""
    return np.array(
        [[np.trace(rho @ np.kron(PAULI[i], PAULI[j])).real for j in (1, 2, 3)] for i in (1, 2, 3)]
    )


def target(v: np.ndarray | None = None) -> np.ndarray:
    """The maximally entangled target (1 ⊗ V)|Phi+> (V = 1: |Phi+>)."""
    return PHI_PLUS if v is None else np.kron(I2, v) @ PHI_PLUS


def bell_fidelity(rho: np.ndarray, v: np.ndarray | None = None) -> float:
    """<tau|rho|tau> / tr(rho) for the fixed target tau = (1 ⊗ V)|Phi+>."""
    t = target(v)
    return float((t.conj() @ rho @ t).real / np.trace(rho).real)


def max_bell_fidelity(rho: np.ndarray) -> float:
    """max over local unitaries on q75 of the fidelity with (1 ⊗ V)|Phi+> (closed form).

    F = (1 + tr(A R)) / 4 with A = D T, D = diag(1, -1, 1) (|Phi+>'s correlations) and R the
    rotation of V; its maximum over SO(3) is s1 + s2 + sign(det A) s3 (s: singular values).
    Over local unitaries on both sides it is the same (they compose to one rotation).
    """
    t = correlation_matrix(rho / np.trace(rho).real)
    a = np.diag([1.0, -1.0, 1.0]) @ t
    s = np.linalg.svd(a, compute_uv=False)
    return float((1 + s[0] + s[1] + np.sign(np.linalg.det(a)) * s[2]) / 4)


def rotation_to_unitary(r: np.ndarray) -> np.ndarray:
    """The SU(2) element V with V sigma_i V^dag = sum_j R_ji sigma_j (Shepperd's method)."""
    tr = np.trace(r)
    cands = [1 + tr, 1 + 2 * r[0, 0] - tr, 1 + 2 * r[1, 1] - tr, 1 + 2 * r[2, 2] - tr]
    i = int(np.argmax(cands))
    q = np.zeros(4)
    q[i] = math.sqrt(max(cands[i], 0.0)) / 2
    f = 4 * q[i]
    if i == 0:
        q[1], q[2], q[3] = (r[2, 1] - r[1, 2]) / f, (r[0, 2] - r[2, 0]) / f, (r[1, 0] - r[0, 1]) / f
    elif i == 1:
        q[0], q[2], q[3] = (r[2, 1] - r[1, 2]) / f, (r[0, 1] + r[1, 0]) / f, (r[0, 2] + r[2, 0]) / f
    elif i == 2:
        q[0], q[1], q[3] = (r[0, 2] - r[2, 0]) / f, (r[0, 1] + r[1, 0]) / f, (r[1, 2] + r[2, 1]) / f
    else:
        q[0], q[1], q[2] = (r[1, 0] - r[0, 1]) / f, (r[0, 2] + r[2, 0]) / f, (r[1, 2] + r[2, 1]) / f
    w, x, y, z = q / np.linalg.norm(q)
    return w * I2 - 1j * (x * X + y * Y + z * Z)


def adjoint_rotation(v: np.ndarray) -> np.ndarray:
    """R with V sigma_i V^dag = sum_j R_ji sigma_j."""
    return np.array(
        [
            [np.trace(PAULI[j] @ v @ PAULI[i] @ v.conj().T).real / 2 for i in (1, 2, 3)]
            for j in (1, 2, 3)
        ]
    )


def optimal_correction(rho: np.ndarray) -> np.ndarray:
    """The V that attains ``max_bell_fidelity``: fixed from the model, before any data."""
    t = correlation_matrix(rho / np.trace(rho).real)
    a = np.diag([1.0, -1.0, 1.0]) @ t
    u, _, wt = np.linalg.svd(a)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(wt.T @ u.T))])
    # maximise tr(A R) with A = U S W^T: R = W D U^T; the target's rotation is R^T (see below)
    r = wt.T @ d @ u.T
    # (1 ⊗ V)Phi+(1 ⊗ V^dag) has correlations D R_V^T, so tr(T^T D R_V^T) = tr(A R_V): R_V = R
    return rotation_to_unitary(r)


def partial_transpose(rho: np.ndarray) -> np.ndarray:
    """Partial transpose on the second qubit."""
    return rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def negativity(rho: np.ndarray) -> float:
    """Sum of |negative eigenvalues| of rho^Gamma (rho normalised); 0 iff separable (2 qubits)."""
    ev = np.linalg.eigvalsh(partial_transpose(rho / np.trace(rho).real))
    return float(-ev[ev < 0].sum())


def ppt_witness(rho: np.ndarray) -> tuple[np.ndarray, float]:
    """W = (|v><v|)^Gamma for the most negative eigenvector v of rho^Gamma; tr(W sigma) >= 0
    on separable sigma, and tr(W rho) = lambda_min(rho^Gamma)."""
    ev, vec = np.linalg.eigh(partial_transpose(rho / np.trace(rho).real))
    v = vec[:, 0]
    return partial_transpose(dm(v)), float(ev[0])


def werner(p: float) -> np.ndarray:
    """p |Phi+><Phi+| + (1 - p) 1/4: F = (1 + 3p)/4, correlators (p, -p, p)."""
    return p * BELL + (1 - p) * np.eye(4) / 4


def fidelity_from_correlators(cxx: float, cyy: float, czz: float) -> float:
    """<Phi+|rho|Phi+> = (1 + <XX> - <YY> + <ZZ>) / 4."""
    return (1 + cxx - cyy + czz) / 4


# -- channels -------------------------------------------------------------------------------------


def superop(kraus: Sequence[np.ndarray]) -> np.ndarray:
    return A.superop(kraus)


def apply(sup: np.ndarray, op: np.ndarray) -> np.ndarray:
    d = op.shape[0]
    return (sup @ op.reshape(-1, order="F")).reshape(d, d, order="F")


def choi(sup: np.ndarray) -> np.ndarray:
    """J = (id ⊗ Phi)(|Phi+><Phi+|) (reference first); tr J = tr Phi(1)/2."""
    j = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            e = np.zeros((2, 2), dtype=complex)
            e[a, b] = 1
            j += np.kron(e, apply(sup, e)) / 2
    return j


def apply_local(rho: np.ndarray, sup_b: np.ndarray, sup_a: np.ndarray | None = None) -> np.ndarray:
    """(Phi_a ⊗ Phi_b)(rho) on a two-qubit (a, b) operator (``sup_a=None``: identity)."""
    blocks = rho.reshape(2, 2, 2, 2)  # (a, i, a', i')
    out = np.zeros_like(blocks)
    for a in range(2):
        for b in range(2):
            out[a, :, b, :] = apply(sup_b, blocks[a, :, b, :])
    out = out.reshape(4, 4)
    if sup_a is None:
        return out
    sw = out.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    sw = apply_local(sw, sup_a)
    return sw.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)


def random_unitary(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_eb_kraus(rng: np.random.Generator, k: int, trace_scale: float = 1.0) -> list[np.ndarray]:
    """Rank-one Kraus operators of a random measure-and-prepare map.

    POVM {E_i} (k elements, random), output states sigma_i (random mixed); Kraus
    sqrt(p_ia lam_ij) |a_i><b_ij|. ``trace_scale < 1`` makes a trace-decreasing EB CP map
    (a branch) with E_i scaled.
    """
    g = [rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)) for _ in range(k)]
    e = [x @ x.conj().T for x in g]
    s = sum(e)
    w, v = np.linalg.eigh(s)
    s_inv = v @ np.diag(w**-0.5) @ v.conj().T
    e = [trace_scale * s_inv @ x @ s_inv for x in e]
    out = []
    for ei in e:
        lam, b = np.linalg.eigh(ei)
        h = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        sig = h @ h.conj().T
        sig /= np.trace(sig).real
        p, a = np.linalg.eigh(sig)
        for ia in range(2):
            for ib in range(2):
                c = max(p[ia], 0) * max(lam[ib], 0)
                if c > 0:
                    out.append(math.sqrt(c) * np.outer(a[:, ia], b[:, ib].conj()))
    return out


def f_avg_from_fe(fe: float, d: int = 2) -> float:
    return (d * fe + 1) / (d + 1)


def axis_average_fidelity(sup: np.ndarray, v: np.ndarray | None = None) -> float:
    """(1/6) sum_k <psi_k|V^dag Phi(psi_k) V|psi_k> / (mean trace): the P&M score, ideal SPAM."""
    v = I2 if v is None else v
    num = den = 0.0
    for ket in AXIS_KETS:
        out = apply(sup, dm(ket))
        t = v @ ket
        num += float((t.conj() @ out @ t).real)
        den += float(np.trace(out).real)
    return num / den


# -- the step channel ------------------------------------------------------------------------------

_P0 = np.diag([1.0, 0.0]).astype(complex)


def step_ops(
    pieces: A.StepPieces,
    variant: str = "baseline",
    hyp: A.Hypothesis | None = None,
    extra_dephasing: float = 0.0,
) -> np.ndarray:
    """(2, 16, 16): one step's branches on (q75, q59), with optional extra q75 dephasing.

    ``extra_dephasing`` p: a Z flip of q75 with probability p/2 per step before the step
    (``ablate_hw.dephase_kraus(p)``), a stand-in for unmodelled q75 decoherence (#22's
    readout-induced dephasing, T2 drift, the #29 baseline deficit).
    """
    s = A.step_superops(variant, pieces, hyp)
    if extra_dephasing:
        d = A.on_q75(A.dephase_kraus(extra_dephasing))
        s = np.stack([x @ d for x in s])
    return s


def q75_map(ops: Sequence[np.ndarray]) -> np.ndarray:
    """The 4x4 superop on q75 of the joint maps ``ops`` applied in order, q59 entering in |0>
    and traced out at the end."""
    out = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            e = np.zeros((2, 2), dtype=complex)
            e[a, b] = 1
            v = np.kron(e, _P0).reshape(-1, order="F")
            for m in ops:
                v = m @ v
            out[:, a + 2 * b] = A.ptrace_q59(v.reshape(4, 4, order="F")).reshape(-1, order="F")
    return out


def n_step_map(
    steps: np.ndarray, n: int, branch: Sequence[int] | None = None, first: np.ndarray | None = None
) -> np.ndarray:
    """q75's map after n steps: unconditional (``branch=None``) or the record ``branch``.

    ``first`` replaces the first step's branches (e.g. the A2 negative control).
    """
    if n == 0:
        return np.eye(4, dtype=complex)
    seq = []
    for t in range(n):
        s = first if (t == 0 and first is not None) else steps
        seq.append(s.sum(0) if branch is None else s[branch[t]])
    return q75_map(seq)


# -- SPAM, noise of the protocols ----------------------------------------------------------------


def depol_superop(p: float) -> np.ndarray:
    """rho -> (1 - p) rho + p 1/2."""
    return (1 - p) * np.eye(4, dtype=complex) + p * np.outer(
        I2.reshape(-1, order="F") / 2, I2.reshape(-1, order="F")
    )


def relax_superop(duration: float, t1: float, t2: float) -> np.ndarray:
    from .ancilla import relaxation_kraus

    return superop(relaxation_kraus(duration, t1, t2))


@dataclass
class Spam:
    """Trusted-device imperfections of q75 (P&M) and of the reference (Bell route).

    ``init_error``: |1> population after initialisation; ``prep_depol`` / ``meas_depol``:
    depolarising after the preparation / before the measurement (gate errors);
    ``p01`` = P(read 0 | 1), ``p10`` = P(read 1 | 0). Readout errors enter as an assignment
    matrix (classical noise after the measurement).
    """

    init_error: float = 0.0
    prep_depol: float = 0.0
    meas_depol: float = 0.0
    p01: float = 0.0
    p10: float = 0.0


def assignment(spam: Spam) -> np.ndarray:
    """A[read, true]."""
    return np.array([[1 - spam.p10, spam.p01], [spam.p10, 1 - spam.p01]])


def measurement_unitary(t: np.ndarray, label: int) -> np.ndarray:
    """W with W|t> = |label> (rows: <t| then <t_perp|, then X^label)."""
    t = t / np.linalg.norm(t)
    perp = np.array([-t[1].conj(), t[0].conj()])
    w = np.vstack([t.conj(), perp.conj()])
    return X @ w if label else w


def pm_probabilities(sup: np.ndarray, v: np.ndarray, spam: Spam) -> dict[str, np.ndarray]:
    """P&M outcome probabilities for every circuit (input k, success label b).

    ``sup`` may be trace-decreasing (a branch). Returns ``p_succ[k, b]`` = P(read b and the
    branch), ``p_branch[k, b]`` = P(branch), so the pooled score is sum p_succ / sum p_branch.
    """
    rho_init = np.diag([1 - spam.init_error, spam.init_error]).astype(complex)
    dpre, dmeas = depol_superop(spam.prep_depol), depol_superop(spam.meas_depol)
    am = assignment(spam)
    ps = np.zeros((6, 2))
    pb = np.zeros((6, 2))
    for k, u in enumerate(AXIS_PREP):
        rho = apply(dpre, u @ rho_init @ u.conj().T)
        out = apply(sup, rho)
        for b in (0, 1):
            w = measurement_unitary(v @ AXIS_KETS[k], b)
            fin = apply(dmeas, w @ out @ w.conj().T)
            pz = np.real(np.diag(fin))
            ps[k, b] = float(am[b] @ pz)
            pb[k, b] = float(pz.sum())
    return {"p_succ": ps, "p_branch": pb}


def pm_score(probs: dict[str, np.ndarray]) -> float:
    return float(probs["p_succ"].sum() / probs["p_branch"].sum())


@dataclass
class RefNoise:
    """The Bell route: preparation infidelity (Bell CZ + SWAP), the reference's idle relaxation
    per step, and the symmetric (twirled) readout errors of both qubits."""

    prep_infidelity: float = 0.0
    ref_t1: float = math.inf
    ref_t2: float = math.inf
    step_s: float = 0.0
    ref_readout: float = 0.0
    mem_readout: float = 0.0
    extra_ref_idle_s: float = 0.0


def bell_prep_state(eps: float) -> np.ndarray:
    """A Bell pair of fidelity 1 - eps (two-qubit depolarising)."""
    lam = 4 * eps / 3
    return (1 - lam) * BELL + lam * np.eye(4) / 4


def bell_final_state(sup_n: np.ndarray, n: int, noise: RefNoise) -> np.ndarray:
    """(D_ref ⊗ Phi_n)(rho_Bell); ``sup_n`` may be a branch (unnormalised output)."""
    rho = bell_prep_state(noise.prep_infidelity)
    t = n * noise.step_s + noise.extra_ref_idle_s
    d_ref = relax_superop(t, noise.ref_t1, noise.ref_t2) if math.isfinite(noise.ref_t1) else None
    return apply_local(rho, sup_n, d_ref)


def bell_parities(rho: np.ndarray, v: np.ndarray, noise: RefNoise) -> dict[str, float]:
    """P(even parity) per setting after the predeclared correction V^dag on q75, with twirled
    (symmetric) readout: each correlator shrinks by (1 - 2 e_ref)(1 - 2 e_mem)."""
    u = np.kron(I2, v.conj().T)
    r = u @ rho @ u.conj().T
    tr = float(np.trace(r).real)
    shrink = (1 - 2 * noise.ref_readout) * (1 - 2 * noise.mem_readout)
    out = {"trace": tr}
    for s, p in zip(("xx", "yy", "zz"), (X, Y, Z), strict=True):
        c = float(np.trace(r @ np.kron(p, p)).real) / tr * shrink
        out[s] = (1 + c) / 2
    out["fidelity"] = fidelity_from_correlators(*(2 * out[s] - 1 for s in ("xx", "yy", "zz")))
    return out


# -- thresholds under leakage (leakage as loss, scored worst case) ---------------------------------


def pm_threshold(leak: float) -> float:
    """EB bound of the P&M score when a fraction ``leak`` of shots leaves the qubit subspace and
    is scored as a success (worst case): (1 - l) 2/3 + l."""
    return 2 / 3 + leak / 3


def bell_threshold(leak: float) -> float:
    """EB bound of the XX/YY/ZZ fidelity when a fraction ``leak`` of shots has arbitrary
    outcomes: each correlator moves by at most 2 l, so F <= (1 - l)/2 + l = 1/2 + l/2."""
    return 0.5 + leak / 2


def leakage_bound(n: int, cz_per_step: int, leak_per_cz: float, extra_cz: int = 0) -> float:
    return min(1.0, (n * cz_per_step + extra_cz) * leak_per_cz)


# -- estimators, confidence bounds, power ---------------------------------------------------------

_N = NormalDist()


def z_of(alpha: float) -> float:
    return _N.inv_cdf(1 - alpha)


def pm_estimate(succ: np.ndarray, shots: np.ndarray) -> tuple[float, float]:
    """Equal-allocation mean of per-circuit success fractions and its stratified SE.

    Drift within a circuit's shots makes the binomial SE conservative (a mixture of
    Bernoullis has variance <= the Bernoulli at the mean)."""
    p = succ / shots
    est = float(p.mean())
    se = float(np.sqrt(np.sum(p * (1 - p) / shots)) / len(p))
    return est, se


def ratio_estimate(succ: np.ndarray, branch: np.ndarray) -> tuple[float, float]:
    """Pooled ratio sum succ / sum branch, SE by the binomial on the branch shots."""
    nb = float(branch.sum())
    if nb <= 0:
        return math.nan, math.inf
    r = float(succ.sum() / nb)
    return r, math.sqrt(max(r * (1 - r), 1e-300) / nb)


def bell_estimate(even: np.ndarray, shots: np.ndarray) -> tuple[float, float]:
    """F = (1 + c_xx - c_yy + c_zz)/4 from parity counts (order xx, yy, zz), and its SE."""
    p = even / shots
    c = 2 * p - 1
    f = fidelity_from_correlators(*c)
    se = math.sqrt(float(np.sum(4 * p * (1 - p) / shots))) / 4
    return f, se


def power_normal(effect: float, sd1: float, shots: float, alpha: float) -> float:
    """P(estimate - z_alpha SE > threshold) for a mean ``effect`` above it and per-shot SD."""
    if shots <= 0:
        return 0.0
    se = sd1 / math.sqrt(shots)
    return _N.cdf(effect / se - z_of(alpha))


def shots_for_power(effect: float, sd1: float, alpha: float, power: float) -> float:
    """Total shots for ``power`` (normal approximation; ``inf`` if ``effect <= 0``)."""
    if effect <= 0:
        return math.inf
    return ((z_of(alpha) + _N.inv_cdf(power)) * sd1 / effect) ** 2


def pm_sd1(probs: dict[str, np.ndarray]) -> float:
    """Per-shot SD of the P&M score with shots spread equally over the 12 circuits."""
    p = probs["p_succ"].ravel()
    return float(math.sqrt(np.mean(p * (1 - p))))


def ratio_sd1(probs: dict[str, np.ndarray]) -> float:
    """Per-shot SD (per total shot) of the pooled branch ratio."""
    r = probs["p_succ"].sum() / probs["p_branch"].sum()
    q = probs["p_branch"].mean()
    return float(math.sqrt(r * (1 - r) / q))


def bell_sd1(par: dict[str, float]) -> float:
    """Per-shot SD of the Bell fidelity estimate with shots spread equally over 3 settings."""
    v = sum(4 * par[s] * (1 - par[s]) for s in ("xx", "yy", "zz")) * 3 / 16
    return math.sqrt(v)


def mc_power_pm(
    probs: dict[str, np.ndarray],
    shots_per_circuit: int,
    thr: float,
    alpha: float,
    draws: int,
    seed: int,
    ratio: bool = False,
) -> float:
    """Monte Carlo power of ``pm_estimate`` (or the branch ratio) > ``thr`` at level alpha."""
    rng = np.random.default_rng(seed)
    ps, pb = probs["p_succ"].ravel(), probs["p_branch"].ravel()
    z = z_of(alpha)
    n = shots_per_circuit
    hits = 0
    for _ in range(draws):
        if ratio:
            u = rng.multinomial(n, np.stack([ps, pb - ps, 1 - pb], 1))
            est, se = ratio_estimate(u[:, 0], u[:, 0] + u[:, 1])
        else:
            est, se = pm_estimate(rng.binomial(n, ps), np.full(len(ps), n))
        hits += est - z * se > thr
    return hits / draws


def mc_power_bell(
    par: dict[str, float], shots_per_setting: int, thr: float, alpha: float, draws: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    p = np.array([par[s] for s in ("xx", "yy", "zz")])
    z = z_of(alpha)
    hits = 0
    for _ in range(draws):
        f, se = bell_estimate(rng.binomial(shots_per_setting, p), np.full(3, shots_per_setting))
        hits += f - z * se > thr
    return hits / draws


# -- a general linear witness from Pauli prepare-and-measure settings -----------------------------


def pauli_design(w: np.ndarray) -> tuple[float, np.ndarray]:
    """tr(W J) = c + sum_{k,b} alpha[k, b] m[k, b] for a trace-preserving channel, where
    m[k, b] = <sigma_b> on Phi(psi_k) (k: the six axis inputs, b: x, y, z).

    Uses tr[(sigma_a ⊗ sigma_b) J] = tr[sigma_b Phi(sigma_a^T)]/2 (reference first),
    Phi(sigma_a) = Phi(psi_a+) - Phi(psi_a-) and Phi(1) = mean over axes of the pair sums.
    """
    wc = np.array(
        [[np.trace(w @ np.kron(PAULI[a], PAULI[b])).real for b in range(4)] for a in range(4)]
    )
    alpha = np.zeros((6, 3))
    c = wc[0, 0] / 4  # b = 0, a = 0: tr Phi(1)/2 = 1; a != 0: tr Phi(sigma_a) = 0
    sign_t = {1: 1.0, 2: -1.0, 3: 1.0}  # sigma_a^T = +-sigma_a
    plus = {1: 2, 2: 4, 3: 0}  # index of psi_a+ in AXIS_KETS (x, y, z)
    for b in (1, 2, 3):
        for a in (1, 2, 3):
            coef = wc[a, b] / 4 * sign_t[a] / 2
            alpha[plus[a], b - 1] += coef
            alpha[plus[a] + 1, b - 1] -= coef
        coef0 = wc[0, b] / 4 / 2 / 3
        alpha[:, b - 1] += coef0
    return float(c), alpha


def pauli_expectations(sup: np.ndarray, spam: Spam) -> np.ndarray:
    """m[k, b] with SPAM: assignment errors as a twirled symmetric shrink (1 - p01 - p10)."""
    rho_init = np.diag([1 - spam.init_error, spam.init_error]).astype(complex)
    dpre, dmeas = depol_superop(spam.prep_depol), depol_superop(spam.meas_depol)
    shrink = 1 - spam.p01 - spam.p10
    m = np.zeros((6, 3))
    for k, u in enumerate(AXIS_PREP):
        out = apply(dmeas, apply(sup, apply(dpre, u @ rho_init @ u.conj().T)))
        for b in range(3):
            m[k, b] = float(np.trace(out @ PAULI[b + 1]).real) * shrink
    return m


def pauli_sd1(alpha: np.ndarray, m: np.ndarray) -> float:
    """Per-shot SD of c + sum alpha m with shots spread equally over the 18 settings."""
    return float(math.sqrt(18 * np.sum(alpha**2 * (1 - m**2))))


# -- the semiquantum signalling game (measurement-device-independent) [Rosset2018] -------------

_T = 1 / math.sqrt(3)
#: Bloch vectors of Rosset et al.'s tetrahedral inputs U tau U^dag, U in (1, X, Y, Z).
TETRA_BLOCH = (
    (_T, _T, _T),
    (_T, -_T, -_T),
    (-_T, _T, -_T),
    (-_T, -_T, _T),
)


def bloch_ket(r: Sequence[float]) -> np.ndarray:
    """The pure state with Bloch vector ``r`` (unit length)."""
    x, y, z = r
    th = math.acos(max(-1.0, min(1.0, z)))
    ph = math.atan2(y, x)
    return np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)], dtype=complex)


TETRA_KETS = tuple(bloch_ket(r) for r in TETRA_BLOCH)


def prep_unitary(ket: np.ndarray) -> np.ndarray:
    """A unitary taking |0> to ``ket``."""
    return measurement_unitary(ket, 0).conj().T


def _pauli_vec(op: np.ndarray) -> np.ndarray:
    return np.array(
        [np.trace(op @ np.kron(PAULI[a], PAULI[b])).real for a in range(4) for b in range(4)]
    )


def mdi_weights(v: np.ndarray | None = None) -> np.ndarray:
    """omega[x, y] with W = tau tau^dag - 1/2 = sum omega_xy xi_x^T ⊗ psi_y^T (Rosset et al.,
    Prop. 2 and Eq. 14; xi = psi = the tetrahedral states). The payoff sum omega_xy p(Phi+|x, y)
    equals tr(W J) = F_e - 1/2 for any channel, and is <= 0 for every EB channel whatever the
    Bell measurement actually does (only the inputs are trusted)."""
    t = target(v)
    w = dm(t) - np.eye(4) / 2
    rhos = [dm(k) for k in TETRA_KETS]
    m = np.stack([_pauli_vec(np.kron(rx.T, ry)) for rx in rhos for ry in [r.T for r in rhos]], 1)
    om, *_ = np.linalg.lstsq(m, _pauli_vec(w), rcond=None)
    return om.reshape(4, 4)


_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
U_BSM = np.kron(_H, I2) @ _CX  # on (memory output B, trusted input Y): Phi+ -> |00>


@dataclass
class MdiReset:
    """Amendment A4 (#38): ``reset q74`` immediately before psi_y. q74's pre-reset state (its
    initialisation error, ZZ-coupled to q75 during the steps) is discarded; after the reset q74
    is |1> with probability ``excited`` (the reset's error), whatever it was before. q75 idles
    for the reset's ``duration`` s (amplitude damping and dephasing at ``t1``, ``t2``)."""

    duration: float = 0.0
    t1: float = math.inf
    t2: float = math.inf
    excited: float = 0.0

    def idle(self) -> np.ndarray:
        """q75's superoperator over the reset (the identity at zero duration)."""
        if self.duration <= 0 or not math.isfinite(self.t1):
            return np.eye(4, dtype=complex)
        return relax_superop(self.duration, self.t1, self.t2)


def mdi_probabilities(
    sup: np.ndarray,
    spam_x: Spam,
    spam_y: Spam,
    bsm_depol: float,
    reset: MdiReset | None = None,
) -> np.ndarray:
    """p[x, y] = P(read 00 | x, y): xi_x on q75 through ``sup``, psi_y on the second qubit,
    the Bell measurement (CX, H) with two-qubit depolarising ``bsm_depol`` before readout and
    each qubit's assignment errors.

    ``reset`` (A4): q74 is reset before psi_y, so psi_y is prepared from
    diag(1 - excited, excited) instead of ``spam_y.init_error``, and q75 first idles through
    the reset (``MdiReset.idle``). Without it, the committed #33 model."""
    if reset is not None:
        sup = reset.idle() @ sup
        spam_y = Spam(reset.excited, spam_y.prep_depol, spam_y.meas_depol, spam_y.p01, spam_y.p10)

    def prep(ket, s):
        r0 = np.diag([1 - s.init_error, s.init_error]).astype(complex)
        u = prep_unitary(ket)
        return apply(depol_superop(s.prep_depol), u @ r0 @ u.conj().T)

    a_x, a_y = assignment(spam_x), assignment(spam_y)
    p00 = np.array([a_x[0, i] * a_y[0, j] for i in (0, 1) for j in (0, 1)])
    out = np.zeros((4, 4))
    for x, kx in enumerate(TETRA_KETS):
        bx = apply(sup, prep(kx, spam_x))
        for y, ky in enumerate(TETRA_KETS):
            r = U_BSM @ np.kron(bx, prep(ky, spam_y)) @ U_BSM.conj().T
            r = (1 - bsm_depol) * r + bsm_depol * np.trace(r) * np.eye(4) / 4
            out[x, y] = float(p00 @ np.real(np.diag(r)))
    return out


def mdi_payoff(omega: np.ndarray, p: np.ndarray) -> float:
    return float((omega * p).sum())


def mdi_sd1(omega: np.ndarray, p: np.ndarray) -> float:
    """Per-shot SD of sum omega p_hat with shots spread equally over the 16 input pairs."""
    return float(math.sqrt(16 * np.sum(omega**2 * p * (1 - p))))


def mdi_circuit(step: Any, n: int, x: int, y: int, reset: bool = False) -> Any:
    """MDI circuit on (q59, q75, q74) = (0, 1, 2): xi_x on q75, n steps (q74 idles in |0>, as
    in Phase 6), then psi_y on q74 and the Bell measurement CX(q75 -> q74), H(q75); Phi+ reads
    00. ``reset`` (amendment A4, #38): ``reset q74`` after the steps' barrier, immediately
    before psi_y (q75 idles through it)."""
    from qiskit import ClassicalRegister, QuantumCircuit

    regs = [ClassicalRegister(1, f"y{t}") for t in range(n)]
    fin = ClassicalRegister(2, "m")  # m[0]: q75, m[1]: q74
    qc = QuantumCircuit(3)
    for r in [*regs, fin]:
        qc.add_register(r)
    _append_1q(qc, prep_unitary(TETRA_KETS[x]), 1)
    qc.barrier()
    _steps(qc, step, n, regs, False, 1, 0, None)
    qc.barrier()
    if reset:
        qc.reset(2)
    _append_1q(qc, prep_unitary(TETRA_KETS[y]), 2)
    qc.cx(1, 2)
    qc.h(1)
    qc.measure(1, fin[0])
    qc.measure(2, fin[1])
    return qc


# -- reference candidates on the committed topology ------------------------------------------------


def bfs(adj: dict[int, set[int]], src: int) -> tuple[dict[int, int], dict[int, int]]:
    dist, prev = {src: 0}, {}
    q = deque([src])
    while q:
        u = q.popleft()
        for w in sorted(adj[u]):
            if w not in dist:
                dist[w], prev[w] = dist[u] + 1, u
                q.append(w)
    return dist, prev


def path_to(prev: dict[int, int], src: int, dst: int) -> list[int]:
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


@dataclass
class RefRule:
    """The predeclared rule for the reference qubit (applied to committed calibration only)."""

    min_dist_memory: int = 2  # no direct coupler to q75 (excludes q74, #22, #29)
    min_dist_output: int = 3  # no coupler to q59 nor to a neighbour of q59
    max_dist_memory: int = 3
    max_readout: float = 0.03
    min_t2_s: float = 50e-6
    max_cz_error: float = 0.01
    forbid_path: Sequence[int] = field(default_factory=tuple)  # e.g. the output qubit


def reference_candidates(
    device: dict[str, Any], memory: int, output: int, rule: RefRule
) -> list[dict[str, Any]]:
    """Every qubit within ``rule.max_dist_memory`` of q75 with its distances, its SWAP path to
    q75 and its calibration, and whether the rule admits it (and why not)."""
    adj: dict[int, set[int]] = {}
    for a, b in device["edges"]:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    dm_, pm_ = bfs(adj, memory)
    do_, _ = bfs(adj, output)
    rows = []
    for q in sorted(dm_):
        if q in (memory, output) or dm_[q] > rule.max_dist_memory:
            continue
        path = path_to(pm_, memory, q)  # memory ... q
        czs = []
        for a, b in itertools.pairwise(path):
            key = f"{min(a, b)}-{max(a, b)}"
            czs.append(float(device["cz"][key]["error"]))
        cal = device["qubits"][str(q)]
        swaps = dm_[q] - 1
        # Bell CZ on the edge next to q75, then SWAPs (3 CZ each) along the rest of the path
        cz_err = czs[0] + 3 * sum(czs[1:])
        why = []
        if dm_[q] < rule.min_dist_memory:
            why.append("coupler to the memory")
        if do_[q] < rule.min_dist_output:
            why.append("coupler to the output or its neighbour")
        if any(p in rule.forbid_path for p in path[1:-1]):
            why.append("path through the output")
        if cal["readout"] > rule.max_readout:
            why.append("readout")
        if cal["t2"] < rule.min_t2_s:
            why.append("T2")
        if max(czs) > rule.max_cz_error:
            why.append("CZ error on the path")
        rows.append(
            {
                "qubit": q,
                "dist_memory": dm_[q],
                "dist_output": do_[q],
                "path": "-".join(map(str, path)),
                "swaps": swaps,
                "cz_count": 1 + 3 * swaps,
                "cz_error_sum": cz_err,
                "max_cz_error": max(czs),
                "readout": cal["readout"],
                "t1_s": cal["t1"],
                "t2_s": cal["t2"],
                "admitted": not why,
                "excluded_because": "; ".join(why),
            }
        )
    return rows


# -- circuits (Qiskit; the hardware group) ---------------------------------------------------------


def euler_zsx(u: np.ndarray) -> list[tuple[str, float | None]]:
    """A 1-qubit unitary as rz/sx gates (qiskit's ZSX Euler basis), in circuit order."""
    from qiskit.synthesis import OneQubitEulerDecomposer

    qc = OneQubitEulerDecomposer("ZSX")(u)
    return [
        (i.operation.name, (float(i.operation.params[0]) if i.operation.params else None))
        for i in qc.data
    ]


def _append_1q(qc: Any, u: np.ndarray, q: int) -> None:
    for name, par in euler_zsx(u):
        if name == "rz":
            qc.rz(par, q)
        elif name == "sx":
            qc.sx(q)
        elif name == "x":
            qc.x(q)
        else:  # pragma: no cover - the ZSX basis has only these
            raise ValueError(name)


def _steps(
    qc: Any,
    step: Any,
    n: int,
    regs: Sequence[Any],
    negative: bool,
    mem: int,
    out: int,
    dur: A.Durations | None,
) -> None:
    """Append n copies of Phase 6's compact step (q[0] = q59, q[1] = q75) on (out, mem)."""
    for t in range(n):
        sc = step
        if negative and t == 0:
            sc = A.build_variant(step, "A2", 1, 0, dur, 1)
        clb = [regs[t][0], regs[-1][0]] if (negative and t == 0) else [regs[t][0]]
        qc.compose(sc, qubits=[out, mem], clbits=clb, inplace=True)


def pm_circuit(
    step: Any,
    n: int,
    k: int,
    label: int,
    v: np.ndarray,
    negative: bool = False,
    dur: A.Durations | None = None,
) -> Any:
    """P&M circuit on (q59, q75) = qubits (0, 1): prepare axis state k on q75, n steps, rotate
    V psi_k to |label>, measure q75 into ``m``. ``negative``: the first step is A2's (q75
    measured mid-step into ``neg``), an EB channel."""
    from qiskit import ClassicalRegister, QuantumCircuit

    regs = [ClassicalRegister(1, f"y{t}") for t in range(n)]
    if negative:
        regs.append(ClassicalRegister(1, "neg"))
    fin = ClassicalRegister(1, "m")
    qc = QuantumCircuit(2)
    for r in [*regs, fin]:
        qc.add_register(r)
    _append_1q(qc, AXIS_PREP[k], 1)
    qc.barrier()
    _steps(qc, step, n, regs, negative, 1, 0, dur)
    qc.barrier()
    _append_1q(qc, measurement_unitary(v @ AXIS_KETS[k], label), 1)
    qc.measure(1, fin[0])
    return qc


def bell_circuit(step: Any, n: int, setting: str, flip: int, v: np.ndarray, swaps: int = 1) -> Any:
    """Bell route on (q59, q75, path...) = (0, 1, 2, ..., 2 + swaps): a Bell pair on qubits
    (2, 1) (q74, q75), then SWAPs along the path so that the reference ends on qubit 2 + swaps
    and the bridge qubits are back in |0>; n steps; V^dag on q75; the basis change of
    ``setting`` on the reference and q75; ``flip``: X on both before readout (readout twirl)."""
    from qiskit import ClassicalRegister, QuantumCircuit

    regs = [ClassicalRegister(1, f"y{t}") for t in range(n)]
    fin = ClassicalRegister(2, "m")  # m[0]: reference, m[1]: memory q75
    ref = 2 + swaps
    qc = QuantumCircuit(ref + 1)
    for r in [*regs, fin]:
        qc.add_register(r)
    qc.h(2)
    qc.cx(2, 1)  # (q74, q75) in |Phi+>
    for a in range(2, ref):
        qc.swap(a, a + 1)  # the reference moves out along the path
    qc.barrier()
    _steps(qc, step, n, regs, False, 1, 0, None)
    qc.barrier()
    basis = {"xx": _H, "yy": _H @ _SG.conj(), "zz": I2}[setting]
    _append_1q(qc, basis @ v.conj().T, 1)
    _append_1q(qc, basis, ref)
    if flip:
        qc.x(1)
        qc.x(ref)
    qc.measure(ref, fin[0])
    qc.measure(1, fin[1])
    return qc


def q59_control_circuit(step: Any, n: int, ref_bit: int, swaps: int = 1) -> Any:
    """The q59-only control: the reference (qubit 2 + swaps) in |ref_bit>, q75 as in Phase 6,
    n steps; the reference is measured at the end. Any dependence of q59's record on ref_bit
    bounds the leakage of reference information into the output."""
    from qiskit import ClassicalRegister, QuantumCircuit

    regs = [ClassicalRegister(1, f"y{t}") for t in range(n)]
    fin = ClassicalRegister(1, "r")
    ref = 2 + swaps
    qc = QuantumCircuit(ref + 1)
    for r in [*regs, fin]:
        qc.add_register(r)
    if ref_bit:
        qc.x(ref)
    qc.barrier()
    _steps(qc, step, n, regs, False, 1, 0, None)
    qc.barrier()
    qc.measure(ref, fin[0])
    return qc


def noiseless_unconditional_rho(qc: Any, keep: Sequence[int]) -> np.ndarray:
    """The reduced state of qubits ``keep`` at the end of ``qc``, noiseless and unconditional:
    every measurement is a Z dephasing of its qubit (its outcome discarded), resets kept.
    Qiskit's little-endian order: ``keep[0]`` is the least significant."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace

    deph = Kraus([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    rho = DensityMatrix.from_label("0" * qc.num_qubits)
    buf = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            rho = rho.evolve(buf).evolve(deph, qargs=qs)
            buf = QuantumCircuit(qc.num_qubits)
            continue
        buf.append(inst.operation, [buf.qubits[i] for i in qs])
    rho = rho.evolve(buf)
    return np.asarray(partial_trace(rho, [q for q in range(qc.num_qubits) if q not in keep]).data)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
