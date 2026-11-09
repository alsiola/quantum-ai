"""Phase 8 Stage 0B (issue #30): classical hidden-state models with explicit intervention maps.

The adversarial alternative to the quantum reading of Phase 7 Stage 4c (#27). A classical
model of the PMD circuit and its q75 controls has a finite hidden state ``s in {0..d-1}``
(any physical reading: q75 in some basis, leakage, a TLS, the readout chain, the classical
control loop) at the start of every step, and one step of arm ``a`` is

    s --P--> u --M_a--> v --Q[y]--> s'

* ``P`` (d, d), stochastic: the part of the step before the intervention slot (in the circuit,
  q75's leading single-qubit gates while q59 is reset);
* ``M_a`` (d, d), stochastic: the arm's intervention, in the slot where the circuit puts it
  (the identity for the baseline);
* ``Q[y]`` (2, d, d), labelled (``Q[0] + Q[1]`` stochastic): the rest of the step, which
  emits q59's symbol ``y``.

In ``qmem.witness``'s row-vector convention the labelled step matrices are
``L_a[y] = P @ M_a @ Q[y]`` and an arm's output process is ``(init, L_a, 1)``. The arms:

=============  =========================================================================
``baseline``   ``M = I``
``A2s_spec``   ``M_spec`` (the q74 measurement and the delay)
``A1``         ``M_reset``
``A2``         ``M_m`` labelled by q75's recorded bit ``m`` (``M_0 + M_1`` stochastic); the
               joint record is ``L[m, y] = P @ M_m @ Q[y]`` (4 symbols, index ``2 m + y``)
``A3``         per step the instance's Pauli ``p_t``: ``M_I = I`` (a delay, as the
               baseline), ``M_X``, ``M_Y``, ``M_Z``
``D1``         a different circuit (no CZ, no q75 gate): its whole step is its own labelled
               map ``L_D1[y]`` on the same hidden state, from the same initial state
=============  =========================================================================

Every arm starts from the same ``init``. Classes (``Structure``):

* ``general``: every map free (the most adversarial class at a given ``d``);
* ``kappa`` (``invasive = κ``): A2's map is QND on the hidden state except in a fraction
  ``κ`` of steps, ``M_m = (1 - κ) diag(q(m|s)) + κ R_m``; ``κ = 0`` is a non-invasive
  (QND) readout of a classical variable, ``κ = 1`` the general class;
* ``zpop`` (``dz = 2``, ``dh``): the hidden state is ``(z, h)``, ``z`` a classical bit held in
  q75's Z populations and ``h`` a side memory with ``dh`` states that q75's own operations
  don't touch: ``P = P_z ⊗ I``, every q75 intervention ``K_a ⊗ I`` (``K`` free 2x2,
  measurement back-action allowed), ``Q`` free (the coupling step).

Numpy only: the model, its exact statistics, its likelihood (``nll``, checked against
brute-force path enumeration in the tests) and the maximum-likelihood fit, an exact EM
(Baum-Welch over the factored step: the expected counts of ``P``, ``M_a`` and ``Q`` follow
from each arm's transition counts and the posterior over the intermediate states ``u, v``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import drift as D
from . import witness as W

Q75_ARMS = ("baseline", "A2s_spec", "A1", "A2", "A3")
ARMS = (*Q75_ARMS, "D1")
PAULIS = ("I", "X", "Y", "Z")


# -- the model (numpy) ---------------------------------------------------------------------------


@dataclass
class InterventionModel:
    """A classical hidden-state model of every arm (see the module docstring)."""

    init: np.ndarray  # (d,)
    pre: np.ndarray  # (d, d)
    post: np.ndarray  # (2, d, d), labelled by y
    spec: np.ndarray  # (d, d)
    reset: np.ndarray  # (d, d)
    meas: np.ndarray  # (2, d, d), labelled by m
    pauli: np.ndarray  # (4, d, d), I X Y Z (I is the identity)
    d1: np.ndarray  # (2, d, d), D1's own step

    @property
    def d(self) -> int:
        return len(self.init)

    def slot(self, arm: str) -> np.ndarray:
        """``M_a`` (d, d) of a stationary q75 arm (A2: summed over the record)."""
        if arm == "baseline":
            return np.eye(self.d)
        if arm == "A2s_spec":
            return self.spec
        if arm == "A1":
            return self.reset
        if arm == "A2":
            return self.meas.sum(0)
        raise ValueError(f"no single slot map for {arm!r}")

    def ops(self, arm: str) -> np.ndarray:
        """Labelled step matrices ``L[y]`` (2, d, d) of a stationary arm."""
        if arm == "D1":
            return self.d1
        m = self.pre @ self.slot(arm)
        return np.stack([m @ self.post[y] for y in (0, 1)])

    def record_ops(self) -> np.ndarray:
        """A2's joint record ``L[2 m + y]`` (4, d, d)."""
        return np.stack([self.pre @ self.meas[m] @ self.post[y] for m in (0, 1) for y in (0, 1)])

    def pauli_ops(self) -> np.ndarray:
        """``L[p, y]`` (4, 2, d, d) of A3's step with Pauli ``p``."""
        return np.stack(
            [np.stack([self.pre @ self.pauli[p] @ self.post[y] for y in (0, 1)]) for p in range(4)]
        )

    def process(self, arm: str) -> W.LinearProcess:
        return W.LinearProcess.hmm(self.ops(arm), self.init)

    def record_process(self) -> W.LinearProcess:
        return W.LinearProcess.hmm(self.record_ops(), self.init)

    def a3_instance(self, paulis: np.ndarray) -> D.TimeVaryingProcess:
        ops = self.pauli_ops()[np.asarray(paulis)]
        return D.TimeVaryingProcess(self.init, ops, np.ones(self.d))

    # likelihood (numpy forward algorithm)
    def nll(
        self,
        arm: str,
        y: np.ndarray,
        m: np.ndarray | None = None,
        paulis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-shot ``-log P`` (nats) of ``y`` (shots, T); A2 with ``m`` uses the joint record;
        A3 needs ``paulis`` (shots, T), each shot's Pauli per step."""
        y = np.asarray(y)
        n, length = y.shape
        if arm == "A3":
            bank = self.pauli_ops()

            def step(t: int) -> np.ndarray:
                return bank[np.asarray(paulis)[:, t], y[:, t]]  # (n, d, d)

        elif arm == "A2" and m is not None:
            rec = self.record_ops()

            def step(t: int) -> np.ndarray:
                return rec[2 * np.asarray(m)[:, t] + y[:, t]]

        else:
            ops = self.ops(arm)

            def step(t: int) -> np.ndarray:
                return ops[y[:, t]]

        s = np.tile(self.init, (n, 1))
        out = np.zeros(n)
        for t in range(length):
            s = np.einsum("nd,nde->ne", s, step(t))
            z = s.sum(1)
            out -= np.log(z)
            s = s / z[:, None]
        return out

    def to_dict(self) -> dict[str, list]:
        return {k: getattr(self, k).tolist() for k in FIELDS}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionModel:
        return cls(**{k: np.asarray(d[k], dtype=np.float64) for k in FIELDS})


FIELDS = ("init", "pre", "post", "spec", "reset", "meas", "pauli", "d1")


def random_model(d: int, rng: np.random.Generator, conc: float = 1.0) -> InterventionModel:
    """Dirichlet(``conc``) rows for every map (I fixed to the identity)."""

    def stoch(k: int = 1) -> np.ndarray:
        rows = rng.dirichlet(np.full(k * d, conc), size=d)
        return rows.reshape(d, k, d).transpose(1, 0, 2) if k > 1 else rows

    paulis = np.stack([np.eye(d), stoch(), stoch(), stoch()])
    return InterventionModel(
        rng.dirichlet(np.full(d, conc)),
        stoch(),
        stoch(2),
        stoch(),
        stoch(),
        stoch(2),
        paulis,
        stoch(2),
    )


def brute_force_prob(
    model: InterventionModel,
    arm: str,
    y: Sequence[int],
    m: Sequence[int] | None = None,
    paulis: Sequence[int] | None = None,
) -> float:
    """``P(y_0..y_{T-1})`` (and the record ``m`` for A2) by explicit enumeration of every
    hidden path ``s_0, u_t, v_t, s_{t+1}`` (the tests' independent check of ``nll``)."""
    d = model.d
    T = len(y)
    total = 0.0

    def slot_map(t: int) -> tuple[np.ndarray, bool]:
        if arm == "A3":
            return model.pauli[paulis[t]], False
        if arm == "A2" and m is not None:
            return model.meas[m[t]], False
        return model.slot(arm), False

    def rec(t: int, s: int, w: float) -> float:
        if w == 0.0:
            return 0.0
        if t == T:
            return w
        acc = 0.0
        if arm == "D1":
            for s2 in range(d):
                acc += rec(t + 1, s2, w * model.d1[y[t], s, s2])
            return acc
        mp, _ = slot_map(t)
        for u in range(d):
            for v in range(d):
                for s2 in range(d):
                    acc += rec(t + 1, s2, w * model.pre[s, u] * mp[u, v] * model.post[y[t], v, s2])
        return acc

    for s0 in range(d):
        total += rec(0, s0, model.init[s0])
    return total


# -- statistics ----------------------------------------------------------------------------------


def lag1_from_blocks(b: np.ndarray) -> float:
    return float(b[3] / (b[2] + b[3]) - b[1] / (b[0] + b[1]))


def arm_stats(
    model: InterventionModel,
    arm: str,
    length: int,
    burn: int,
    k_max: int,
    offsets: Sequence[int],
    paulis: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact ``g``, ``w(j)`` (common start set), ``P(1)`` after the burn-in and the lag-1
    excess of an arm; A3 averaged over its instances ``paulis`` (n_inst, T), as
    ``experiments/phase7_hw.py``'s ``exact_arm``; A2 also its q75 record."""
    if arm == "A3":
        gs, p1s, bl = [], [], np.zeros(4)
        for row in paulis:
            tv = model.a3_instance(row)
            gs.append(tv.expected_windows(k_max, burn, common=True))
            p1s.append(tv.marginals()[burn:].mean())
            bl += tv.block_probs(range(burn, length - 1), 2)
        g, p1, lag = np.mean(gs, 0), float(np.mean(p1s)), lag1_from_blocks(bl / len(paulis))
    else:
        p = model.process(arm)
        g = p.expected_windows(k_max, burn, length, common=True)
        p1 = float(p.marginals(length)[burn:].mean())
        lag = lag1_from_blocks(p.block_probs(range(burn, length - 1), 2))
    out: dict[str, Any] = {f"w{j}": float(W.witness(g, j)) for j in offsets}
    out.update({"g": g.tolist(), "p1": p1, "lag1": lag})
    if arm == "A2":
        out.update(record_stats_exact(model, length, burn))
    return out


def record_stats_exact(model: InterventionModel, length: int, burn: int) -> dict[str, float]:
    """A2's record statistics as ``qmem.refit.record_stats`` computes them from shots."""
    rec = model.record_ops()  # [2m + y]
    s = model.init.copy()
    states = []
    for _ in range(length):
        states.append(s)
        s = s @ rec.sum(0)
    one = np.ones(model.d)
    pm = np.array([[[st @ rec[2 * a + b] @ one for b in (0, 1)] for a in (0, 1)] for st in states])
    # pm[t, m, y]
    out = {
        "p_m1_step0": float(pm[0, 1].sum()),
        "p_m1_step1": float(pm[1, 1].sum()),
        "p_m1": float(pm[burn:, 1].sum(1).mean()),
        "p_y1": float(pm[burn:, :, 1].sum(1).mean()),
        "p_y1_given_m1": float(pm[burn:, 1, 1].sum() / pm[burn:, 1].sum()),
        "p_y1_given_m0": float(pm[burn:, 0, 1].sum() / pm[burn:, 0].sum()),
    }
    m1 = rec[2] + rec[3]
    for a in (0, 1):
        for b in (0, 1):
            num = sum(states[t] @ rec[2 * a + b] @ m1 @ one for t in range(burn, length - 1))
            den = sum(states[t] @ rec[2 * a + b] @ one for t in range(burn, length - 1))
            out[f"p_m1_next_given_m{a}_y{b}"] = float(num / den)
    return out


# -- classes and the fit (EM) -------------------------------------------------------------------


@dataclass
class Structure:
    """The model class: ``general`` (``invasive = 1``), ``kappa`` (``0 <= invasive < 1``) or
    ``zpop`` (``dh`` side states; ``d = 2 dh``)."""

    d: int
    kind: str = "general"
    invasive: float = 1.0
    dh: int = 0
    fix: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        if self.kind == "zpop":
            return f"zpop_dh{self.dh}"
        if self.kind == "kappa":
            return f"kappa{self.invasive:g}_d{self.d}"
        return f"general_d{self.d}"


def n_params(st: Structure) -> int:
    """Free parameters (a row simplex of ``k`` entries has ``k - 1``; the Pauli I fixed)."""
    d = st.d
    common = (d - 1) + 2 * d * (2 * d - 1)  # init, Q, D1
    if st.kind == "zpop":
        dz = 2
        return common + 5 * dz * (dz - 1) + dz * (2 * dz - 1)  # P, spec, A1, X, Y, Z; meas
    meas = d * (2 * d - 1) + (d if st.kind == "kappa" else 0)  # R_m (+ q(m | s))
    return common + 5 * d * (d - 1) + meas


@dataclass
class ArmData:
    """One arm's shots (uint8 (n, T)); A2's q75 record ``m``; A3's per-shot Paulis."""

    y: np.ndarray
    m: np.ndarray | None = None
    paulis: np.ndarray | None = None

    def subset(self, keep: np.ndarray) -> ArmData:
        return ArmData(
            self.y[keep],
            None if self.m is None else self.m[keep],
            None if self.paulis is None else self.paulis[keep],
        )

    @property
    def n_symbols(self) -> int:
        return int(self.y.size + (0 if self.m is None else self.m.size))


def _classes(model: InterventionModel, arm: str, a: ArmData, use_record: bool):
    """The arm's candidate step matrices ``(K, d, d)``, each shot's choice per step (n, T), and
    per candidate ``(y, slot)``: q59's symbol and the slot map (``None``: the identity;
    ``("meas", m)``, ``("meas_sum",)``, ``("spec",)``, ...)."""
    y = a.y.astype(np.int64)
    if arm == "D1":
        return model.d1, y, None
    if arm == "A3":
        idx = 2 * np.asarray(a.paulis, dtype=np.int64) + y
        slots = [(yy, None if p == 0 else ("pauli", p)) for p in range(4) for yy in (0, 1)]
        return model.pauli_ops().reshape(8, model.d, model.d), idx, slots
    if arm == "A2" and use_record and a.m is not None:
        idx = 2 * a.m.astype(np.int64) + y
        slots = [(yy, ("meas", mm)) for mm in (0, 1) for yy in (0, 1)]
        return model.record_ops(), idx, slots
    slot = {"baseline": None, "A2s_spec": ("spec",), "A1": ("reset",), "A2": ("meas_sum",)}[arm]
    return model.ops(arm), y, [(yy, slot) for yy in (0, 1)]


def forward_backward(init: np.ndarray, ops: np.ndarray, idx: np.ndarray):
    """Expected transition counts ``C[k, i, j]`` of a process whose shots use the step matrix
    ``ops[idx[n, t]]`` at step ``t``; the initial posterior (summed over shots); total
    log-likelihood. (``qmem.witness.hmm_forward_backward`` for ``K`` candidates.)"""
    n, length = idx.shape
    k, d, _ = ops.shape
    cat = ops.transpose(1, 0, 2).reshape(d, k * d)
    alpha = np.empty((length, n, d))
    scale = np.empty((length, n))
    a = np.tile(init, (n, 1))
    rows = np.arange(n)
    for t in range(length):
        alpha[t] = a
        a = (a @ cat).reshape(n, k, d)[rows, idx[:, t]]
        z = a.sum(1)
        scale[t] = z
        a = a / z[:, None]
    beta = np.ones((n, d))
    cat_t = (
        ops.transpose(0, 2, 1).transpose(1, 0, 2).reshape(d, k * d)
    )  # [j, (k, i)] = ops[k, i, j]
    outer = np.zeros((k * d, d))  # sum over t, n of 1[idx = k] alpha_i b_j
    eye_k = np.eye(k)
    for t in range(length - 1, -1, -1):
        b = beta / scale[t][:, None]
        onehot = eye_k[idx[:, t]]  # (n, k)
        x = (onehot[:, :, None] * alpha[t][:, None, :]).reshape(n, k * d)
        outer += x.T @ b
        beta = (b @ cat_t).reshape(n, k, d)[rows, idx[:, t]]
    counts = ops * outer.reshape(k, d, d)
    gamma0 = (alpha[0] * beta).sum(0)
    return counts, gamma0, float(np.log(scale).sum())


def _slot_matrix(model: InterventionModel, slot) -> np.ndarray:
    if slot is None:
        return np.eye(model.d)
    if slot[0] == "meas":
        return model.meas[slot[1]]
    if slot[0] == "meas_sum":
        return model.meas.sum(0)
    if slot[0] == "pauli":
        return model.pauli[slot[1]]
    return {"spec": model.spec, "reset": model.reset}[slot[0]]


def _normalise(counts: np.ndarray, old: np.ndarray, labelled: bool = False) -> np.ndarray:
    """Rows of ``counts`` normalised (jointly over the label for a (Y, d, d) array); a row
    with no counts keeps ``old``."""
    if labelled:
        tot = counts.sum(axis=(0, 2))
        ok = tot > 0
        return np.where(ok[None, :, None], counts / np.where(ok, tot, 1)[None, :, None], old)
    tot = counts.sum(1)
    ok = tot > 0
    return np.where(ok[:, None], counts / np.where(ok, tot, 1)[:, None], old)


def _kron_sum(full: np.ndarray, dh: int) -> np.ndarray:
    """``N[a, b] = sum_h full[a dh + h, b dh + h]`` (counts of a ``K ⊗ I_h`` factor)."""
    dz = full.shape[-1] // dh
    r = full.reshape(*full.shape[:-2], dz, dh, dz, dh)
    return np.einsum("...ahbh->...ab", r)


def _kron(small: np.ndarray, dh: int) -> np.ndarray:
    eye = np.eye(dh)
    if small.ndim == 3:
        return np.stack([np.kron(x, eye) for x in small])
    return np.kron(small, eye)


@dataclass
class KappaParts:
    """``M_m = (1 - κ) diag(q[:, m]) + κ R[m]`` (the ``kappa`` class's A2 map)."""

    kappa: float
    q: np.ndarray  # (d, 2)
    r: np.ndarray  # (2, d, d), labelled

    def meas(self) -> np.ndarray:
        k = self.kappa
        return np.stack([(1 - k) * np.diag(self.q[:, m]) + k * self.r[m] for m in (0, 1)])


def em_step(
    model: InterventionModel,
    data: dict[str, ArmData],
    st: Structure,
    kappa: KappaParts | None = None,
    use_record: bool = True,
) -> tuple[InterventionModel, KappaParts | None, float]:
    """One EM iteration over every arm jointly; returns the new model and the old model's
    total log-likelihood."""
    d = model.d
    P, Q = model.pre, model.post
    nP = np.zeros((d, d))
    nQ = np.zeros((2, d, d))
    nmap = {k: np.zeros((d, d)) for k in ("spec", "reset", "X", "Y", "Z")}
    nmeas = np.zeros((2, d, d))
    nd1 = np.zeros((2, d, d))
    ninit = np.zeros(d)
    ll = 0.0
    for arm, a in data.items():
        ops, idx, slots = _classes(model, arm, a, use_record)
        c, g0, l_arm = forward_backward(model.init, ops, idx)
        ll += l_arm
        ninit += g0
        if arm == "D1":
            nd1 += c
            continue
        for k, (yy, slot) in enumerate(slots):
            if not c[k].any():
                continue
            lk = ops[k]
            r = np.divide(c[k], lk, out=np.zeros_like(lk), where=lk > 0)
            if slot is not None and slot[0] == "meas_sum":  # latent m: a mixture over M_m
                for mm in (0, 1):
                    mk = model.meas[mm]
                    nP += P * (r @ Q[yy].T @ mk.T)
                    nmeas[mm] += mk * (P.T @ r @ Q[yy].T)
                    nQ[yy] += Q[yy] * ((P @ mk).T @ r)
                continue
            mk = _slot_matrix(model, slot)
            nP += P * (r @ Q[yy].T @ mk.T)
            nQ[yy] += Q[yy] * ((P @ mk).T @ r)
            if slot is None:
                continue
            nm = mk * (P.T @ r @ Q[yy].T)
            if slot[0] == "meas":
                nmeas[slot[1]] += nm
            elif slot[0] == "pauli":
                nmap["XYZ"[slot[1] - 1]] += nm
            else:
                nmap[slot[0]] += nm
    new = InterventionModel(
        ninit / ninit.sum(),
        model.pre,
        _normalise(nQ, Q, True),
        model.spec,
        model.reset,
        model.meas,
        model.pauli.copy(),
        _normalise(nd1, model.d1, True),
    )
    if st.kind == "zpop":
        dh = st.dh
        sm = lambda n, old: _kron(_normalise(_kron_sum(n, dh), _kron_sum(old, dh) / dh), dh)  # noqa: E731
        new.pre = sm(nP, P)
        new.spec = sm(nmap["spec"], model.spec)
        new.reset = sm(nmap["reset"], model.reset)
        for i, key in enumerate("XYZ", start=1):
            new.pauli[i] = sm(nmap[key], model.pauli[i])
        small = _kron_sum(nmeas, dh)
        new.meas = _kron(_normalise(small, _kron_sum(model.meas, dh) / dh, True), dh)
        return new, kappa, ll
    new.pre = _normalise(nP, P)
    new.spec = _normalise(nmap["spec"], model.spec)
    new.reset = _normalise(nmap["reset"], model.reset)
    for i, key in enumerate("XYZ", start=1):
        new.pauli[i] = _normalise(nmap[key], model.pauli[i])
    if kappa is None:
        new.meas = _normalise(nmeas, model.meas, True)
        return new, None, ll
    # kappa class: split each M_m's counts between its QND diagonal and its R part
    k = kappa.kappa
    full = kappa.meas()
    ndiag = np.zeros((d, 2))
    nr = np.zeros((2, d, d))
    for mm in (0, 1):
        dg = (1 - k) * kappa.q[:, mm]
        diag = np.diag(full[mm])
        frac = np.divide(dg, diag, out=np.zeros(d), where=diag > 0)
        ndiag[:, mm] = np.diag(nmeas[mm]) * frac
        nr[mm] = nmeas[mm] - np.diag(ndiag[:, mm])
    tot = ndiag.sum(1)
    q = np.where((tot > 0)[:, None], ndiag / np.where(tot > 0, tot, 1)[:, None], kappa.q)
    kp = KappaParts(k, q, _normalise(nr, kappa.r, True) if k > 0 else kappa.r)
    new.meas = kp.meas()
    return new, kp, ll


@dataclass
class FitResult:
    structure: Structure
    model: InterventionModel
    nll: float  # total nats on the training data
    n_symbols: int
    iters: int
    converged: bool
    kappa: KappaParts | None = None


def start_model(st: Structure, rng: np.random.Generator, conc: float = 1.0):
    """A random start of the class (``random_model``; zpop: Kronecker maps; kappa: random
    ``q`` and ``R``)."""
    m = random_model(st.d, rng, conc)
    kp = None
    if st.kind == "zpop":
        small = random_model(2, rng, conc)
        dh = st.dh
        m.pre, m.spec, m.reset = (_kron(x, dh) for x in (small.pre, small.spec, small.reset))
        m.meas = _kron(small.meas, dh)
        m.pauli = np.stack([np.eye(st.d), *(_kron(small.pauli[i], dh) for i in (1, 2, 3))])
    elif st.kind == "kappa":
        kp = KappaParts(
            st.invasive, rng.dirichlet(np.ones(2), size=st.d), random_model(st.d, rng).meas
        )
        m.meas = kp.meas()
    return m, kp


_ROWS = {
    "init": "vec",
    "pre": "row",
    "post": "lab",
    "spec": "row",
    "reset": "row",
    "meas": "lab",
    "pauli": "pauli",
    "d1": "lab",
}


def _flatten(model: InterventionModel, kp: KappaParts | None) -> list[np.ndarray]:
    parts = [getattr(model, k) for k in FIELDS]
    if kp is not None:
        parts += [kp.q, kp.r]
    return parts


def _project(
    parts: list[np.ndarray], st: Structure, kp: KappaParts | None, floor: float = 1e-12
) -> tuple[InterventionModel, KappaParts | None]:
    """Back onto the class after an extrapolation: entries clipped at ``floor``, rows
    renormalised, zpop factors re-Kroneckered, kappa's map rebuilt from ``q`` and ``R``."""
    out = {}
    for k, x in zip(FIELDS, parts[: len(FIELDS)], strict=True):
        x = np.clip(x, floor, None)
        kind = _ROWS[k]
        if kind == "vec":
            x = x / x.sum()
        elif kind == "row":
            x = x / x.sum(-1, keepdims=True)
        elif kind == "lab":
            x = x / x.sum(axis=(0, 2))[None, :, None]
        else:
            x = x / x.sum(-1, keepdims=True)
            x[0] = np.eye(x.shape[-1])
        out[k] = x
    if st.kind == "zpop":
        dh = st.dh
        for k in ("pre", "spec", "reset"):
            sm = _kron_sum(out[k], dh) / dh
            out[k] = _kron(sm / sm.sum(-1, keepdims=True), dh)
        for i in (1, 2, 3):
            sm = _kron_sum(out["pauli"][i], dh) / dh
            out["pauli"][i] = _kron(sm / sm.sum(-1, keepdims=True), dh)
        sm = _kron_sum(out["meas"], dh) / dh
        out["meas"] = _kron(sm / sm.sum(axis=(0, 2))[None, :, None], dh)
    model = InterventionModel(**out)
    if kp is not None:
        q = np.clip(parts[len(FIELDS)], floor, None)
        r = np.clip(parts[len(FIELDS) + 1], floor, None)
        kp = KappaParts(
            kp.kappa, q / q.sum(1, keepdims=True), r / r.sum(axis=(0, 2))[None, :, None]
        )
        model.meas = kp.meas()
    return model, kp


def _loglik(model: InterventionModel, data: dict[str, ArmData], use_record: bool) -> float:
    return -float(
        sum(
            model.nll(arm, a.y, a.m if use_record else None, a.paulis).sum()
            for arm, a in data.items()
        )
    )


def fit_em_model(
    data: dict[str, ArmData],
    st: Structure,
    model: InterventionModel,
    kappa: KappaParts | None = None,
    max_iters: int = 2000,
    tol: float = 1e-9,
    use_record: bool = True,
    accel: bool = True,
) -> FitResult:
    """EM from ``model`` until a cycle improves the NLL per symbol by less than ``tol``.
    With ``accel``, SQUAREM (Varadhan & Roland 2008): two EM steps, a squared
    extrapolation projected back onto the class, one stabilising EM step, kept only if the
    likelihood did not fall. ``max_iters`` counts EM steps."""
    n_sym = sum(a.n_symbols if use_record else a.y.size for a in data.values())
    ll = -math.inf
    it, conv = 0, False
    while it < max_iters:
        m1, k1, ll0 = em_step(model, data, st, kappa, use_record)
        m2, k2, _ = em_step(m1, data, st, k1, use_record)
        it += 2
        cand, kc = m2, k2
        if accel:
            p0, p1, p2 = (_flatten(m, k) for m, k in ((model, kappa), (m1, k1), (m2, k2)))
            r = [b - a for a, b in zip(p0, p1, strict=True)]
            v = [c - 2 * b + a for a, b, c in zip(p0, p1, p2, strict=True)]
            nr = math.sqrt(sum(float(np.sum(x**2)) for x in r))
            nv = math.sqrt(sum(float(np.sum(x**2)) for x in v))
            if nv > 0:
                alpha = -max(1.0, nr / nv)
                pe = [a - 2 * alpha * x + alpha**2 * y for a, x, y in zip(p0, r, v, strict=True)]
                me, ke = _project(pe, st, kappa)
                ms, ks, ll_e = em_step(me, data, st, ke, use_record)
                it += 1
                if ll_e >= _loglik(m2, data, use_record):
                    cand, kc = ms, ks
        model, kappa = cand, kc
        if math.isfinite(ll) and (ll0 - ll) / n_sym < tol:
            conv = True
            break
        ll = ll0
    return FitResult(st, model, -_loglik(model, data, use_record), n_sym, it, conv, kappa)


def em_start(baseline: np.ndarray, d: int, seed: int, iters: int = 300) -> InterventionModel:
    """A start from ``qmem.witness``'s Baum-Welch on the baseline shots alone: ``Q`` the fitted
    ``d``-state HMM, ``P`` and the sham near the identity, D1 a copy of ``Q``."""
    lab, init = W.renewal_init(baseline, d)
    f = W.fit_em(lab, init, baseline, max_iters=iters, tol=1e-9)
    m = random_model(d, np.random.default_rng(seed), conc=5.0)
    near = 0.9 * np.eye(d) + 0.1 / d
    m.init, m.post, m.d1 = f["initial"], f["labelled"], f["labelled"].copy()
    m.pre, m.spec = near.copy(), near.copy()
    return m


# -- the calibrated quantum model in hidden-state coordinates --------------------------------------

_PAULI_M = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1.0, -1.0]).astype(complex),
)
_K0 = np.diag([1.0, 0.0]).astype(complex)


def _bloch(rho: np.ndarray) -> np.ndarray:
    return np.array([np.trace(p @ rho).real for p in _PAULI_M])


def _rho(b: np.ndarray) -> np.ndarray:
    return 0.5 * sum(c * p for c, p in zip(b, _PAULI_M, strict=True))


def _bloch_map(fn) -> np.ndarray:
    """(4, 4) real matrix of a linear map on q75's 2x2 operators, coordinates (tr, x, y, z)."""
    return np.stack([_bloch(fn(_rho(e))) for e in np.eye(4)], axis=1)


def effective_maps(pieces: Any, readout_error: float) -> dict[str, np.ndarray]:
    """The calibrated model (``qmem.ablate_hw``'s pieces) as maps on q75 alone, in Bloch
    coordinates (column convention): ``pre`` (reset of q59 + q75's leading gates),
    ``post`` (2, 4, 4) by y, the slot maps ``spec`` (identity), ``reset``, ``meas`` (2, 4, 4)
    by recorded m, ``pauli`` (4, 4, 4), and ``r0`` (|0>). q59 is reset every step, so the
    state after the reset is a product and ``post @ pre`` is the baseline step exactly; the
    slot maps leave out q59's relaxation while q75 is reset or measured (q59 is then near
    |0>; it moves A1's and A2's P(1) by < 2e-5, ``test_classical``)."""
    from . import ablate_hw as A

    vec = lambda m: m.reshape(-1, order="F")  # noqa: E731
    unvec = lambda v: v.reshape(4, 4, order="F")  # noqa: E731
    pt59 = lambda m4: np.einsum("aibi->ab", m4.reshape(2, 2, 2, 2))  # noqa: E731
    pt75 = lambda m4: np.einsum("iaib->ab", m4.reshape(2, 2, 2, 2))  # noqa: E731
    pre_full = pieces.pre @ A.reset_superop(pieces, A.Hypothesis())
    sig = pt75(unvec(pre_full @ vec(np.kron(_K0, _K0))))
    sig = sig / np.trace(sig)

    def pre(rho):
        return pt59(unvec(pre_full @ vec(np.kron(rho, _K0))))

    def post(y):
        return lambda rho: pt59(unvec(pieces.post[y] @ vec(np.kron(rho, sig))))

    def q75(kraus):
        return lambda rho: sum(k @ rho @ k.conj().T for k in kraus)

    k0, k1 = A.meas_kraus(0.0)
    r = readout_error
    phys = [q75(k0), q75(k1)]
    meas = [
        _bloch_map(lambda rho, a=a: (1 - r) * phys[a](rho) + r * phys[1 - a](rho)) for a in (0, 1)
    ]
    e = (pieces.eps_q75, pieces.eps_q75)
    paulis = [_bloch_map(q75([_PAULI_M[0]]))]
    for i in (1, 2):
        paulis.append(_bloch_map(q75([k @ _PAULI_M[i] for k in A.depol_kraus(pieces.p1_q75)])))
    paulis.append(_bloch_map(q75([_PAULI_M[3]])))
    return {
        "pre": _bloch_map(pre),
        "post": np.stack([_bloch_map(post(y)) for y in (0, 1)]),
        "spec": np.eye(4),
        "reset": _bloch_map(q75(A.reset_kraus(*e))),
        "meas": np.stack(meas),
        "pauli": np.stack(paulis),
        "r0": _bloch(_K0),
    }


def klein_tetrahedron(a: float, b: float, c: float = 1.0) -> np.ndarray:
    """(4, 4) columns ``(1, x, y, z)`` of the tetrahedron with vertices ``(a, b, c)`` and its
    images under the Pauli rotations (the only simplices the Paulis permute)."""
    return np.array(
        [[1, 1, 1, 1], [a, a, -a, -a], [b, -b, b, -b], [c, -c, -c, c]], dtype=np.float64
    )


def embedding(maps: dict[str, np.ndarray], u: np.ndarray) -> dict[str, np.ndarray]:
    """Every map of ``effective_maps`` in the vertex coordinates of the slot simplex ``u``
    (row convention, ``L[i, j]``: vertex i to vertex j), with the hidden state at the start
    of a step chosen as ``pre^{-1} u`` (so ``P = I``). The model is classical iff every
    entry is >= 0; ``post`` here is the slot-to-slot map ``u^{-1} pre post_y u``, which is
    ``P Q[y]`` for every choice of the step-start vertices, so its negativity is necessary."""
    ui = np.linalg.inv(u)
    t = lambda m: (ui @ m @ u).T  # noqa: E731
    return {
        "init": ui @ maps["pre"] @ maps["r0"],
        "post": np.stack([t(maps["pre"] @ maps["post"][y]) for y in (0, 1)]),
        "reset": t(maps["reset"]),
        "meas": np.stack([t(m) for m in maps["meas"]]),
        "pauli": np.stack([t(m) for m in maps["pauli"]]),
    }


def embedding_negativity(maps: dict[str, np.ndarray], u: np.ndarray) -> float:
    """The most negative entry over every map of ``embedding`` (>= 0: an exact classical
    model of the calibrated quantum model on the simplex ``u``)."""
    e = embedding(maps, u)
    return float(min(np.min(v) for v in e.values()))


def clipped_start(maps: dict[str, np.ndarray], u: np.ndarray, d1_p1: float) -> InterventionModel:
    """The nearest classical model to an embedding: negative entries set to 0, rows
    renormalised (an EM start that carries the calibrated model's structure)."""
    e = {k: np.clip(v, 0, None) for k, v in embedding(maps, u).items()}
    lab = lambda x: x / x.sum(axis=(0, 2))[None, :, None]  # noqa: E731
    row = lambda x: x / x.sum(1, keepdims=True)  # noqa: E731
    d = u.shape[1]
    d1 = np.stack([(1 - d1_p1) * np.eye(d), d1_p1 * np.eye(d)])
    paulis = np.stack([np.eye(d), *(row(e["pauli"][i]) for i in (1, 2, 3))])
    return InterventionModel(
        e["init"] / e["init"].sum(),
        np.eye(d),
        lab(e["post"]),
        np.eye(d),
        row(e["reset"]),
        lab(e["meas"]),
        paulis,
        d1,
    )


def smooth(model: InterventionModel, eps: float) -> InterventionModel:
    """Every map mixed with the uniform one by ``eps`` (EM cannot move a zero entry; the
    Pauli I stays the identity; zpop starts are smoothed in their small factors instead)."""
    d = model.d
    row = lambda x: (1 - eps) * x + eps / d  # noqa: E731
    lab = lambda x: (1 - eps) * x + eps / (2 * d)  # noqa: E731
    paulis = np.stack([np.eye(d), *(row(model.pauli[i]) for i in (1, 2, 3))])
    return InterventionModel(
        row(model.init),
        row(model.pre),
        lab(model.post),
        row(model.spec),
        row(model.reset),
        lab(model.meas),
        paulis,
        lab(model.d1),
    )


def split_state(model: InterventionModel, rng: np.random.Generator, noise: float = 0.05):
    """A ``d + 1``-state start: the most visited baseline state duplicated (each copy takes
    half the inflow), maps perturbed by ``noise`` (as ``qmem.witness.split_init``)."""
    d = model.d
    pi = model.process("baseline").stationary().real
    s = int(np.argmax(pi))

    def grow(m: np.ndarray, labelled: bool) -> np.ndarray:
        x = m if labelled else m[None]
        new = np.zeros((x.shape[0], d + 1, d + 1))
        new[:, :d, :d] = x
        new[:, :d, d] = x[:, :, s] / 2
        new[:, :d, s] = x[:, :, s] / 2
        new[:, d, :] = new[:, s, :]
        new *= np.exp(noise * rng.standard_normal(new.shape))
        new /= new.sum(axis=(0, 2), keepdims=True)
        return new if labelled else new[0]

    ini = np.append(model.init, model.init[s] / 2)
    ini[s] /= 2
    paulis = np.stack([np.eye(d + 1), *(grow(model.pauli[i], False) for i in (1, 2, 3))])
    return InterventionModel(
        ini / ini.sum(),
        grow(model.pre, False),
        grow(model.post, True),
        grow(model.spec, False),
        grow(model.reset, False),
        grow(model.meas, True),
        paulis,
        grow(model.d1, True),
    )


__all__ = [
    "ARMS",
    "FIELDS",
    "PAULIS",
    "Q75_ARMS",
    "ArmData",
    "FitResult",
    "InterventionModel",
    "Structure",
    "arm_stats",
    "brute_force_prob",
    "clipped_start",
    "effective_maps",
    "em_start",
    "em_step",
    "embedding",
    "embedding_negativity",
    "fit_em_model",
    "forward_backward",
    "klein_tetrahedron",
    "n_params",
    "random_model",
    "record_stats_exact",
    "smooth",
    "split_state",
    "start_model",
]
