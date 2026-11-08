"""Phase 7 Stage 4b (issue #26): q75 under repeated mid-circuit measurement, refit on a pilot.

#25's pilot recorded A2's q75 bit ``m_t`` and q59's output ``y_t`` in every step. The
calibrated model (#20's pieces) predicts P(m = 1) = 0.49 after the burn-in; the device gave
0.42. This module fits a small extension of the model to the joint record ``(m_t, y_t)``:

* ``decay``: in every mid-circuit measurement of q75 after the first, ``|1> -> |0>`` with this
  probability, recorded 0 (``qmem.ablate_hw.meas_kraus``);
* ``decay_first``: the same for the first measurement of a shot (step 0), a separate nuisance
  parameter (``None``: equal to ``decay``).

The joint record is a linear process with four symbols ``(m, y)`` on the (q75, q59) memory:
``P(m_0 y_0 .. m_T y_T) = tr(S[m_T, y_T] .. S[m_0, y_0] rho_0)``, with ``S[m, y]`` from
``qmem.ablate_hw.a2_record_superops``. Everything here is numpy; the fit is maximum
likelihood over whole shots and its uncertainty a whole-shot bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import ablate_hw as A

_VEC0 = A.RHO0_JOINT.reshape(-1, order="F")
_TRACE = np.eye(4).reshape(-1, order="F")


@dataclass
class RecordModel:
    """A2's joint record: ``first`` (2, 2, 16, 16) for step 0, ``ops`` for every later step."""

    first: np.ndarray
    ops: np.ndarray

    @classmethod
    def build(
        cls,
        pieces: A.StepPieces,
        readout_error: float,
        decay: float = 0.0,
        decay_first: float | None = None,
        hyp: A.Hypothesis | None = None,
    ) -> RecordModel:
        ops = A.a2_record_superops(pieces, readout_error, decay, hyp)
        d0 = decay if decay_first is None else decay_first
        first = ops if d0 == decay else A.a2_record_superops(pieces, readout_error, d0, hyp)
        return cls(first, ops)

    def step(self, t: int) -> np.ndarray:
        return self.first if t == 0 else self.ops

    def nll(self, m: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-shot negative log-likelihood (nats) of records ``m``, ``y`` (shots, T)."""
        n, length = m.shape
        v = np.tile(_VEC0, (n, 1))
        out = np.zeros(n)
        for t in range(length):
            s = self.step(t).reshape(4, 16, 16)[2 * m[:, t] + y[:, t]]
            v = np.einsum("nij,nj->ni", s, v)
            p = (v @ _TRACE).real
            if np.any(p <= 0):
                return np.full(n, np.inf)
            out -= np.log(p)
            v = v / p[:, None]
        return out

    def sample(self, n: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """``(m, y)``, each (n, length) uint8, by sequential conditioning."""
        rng = np.random.default_rng(seed)
        v = np.tile(_VEC0.astype(complex), (n, 1))
        m = np.zeros((n, length), dtype=np.uint8)
        y = np.zeros((n, length), dtype=np.uint8)
        for t in range(length):
            s = self.step(t).reshape(4, 16, 16)
            w = np.einsum("sij,nj->nsi", s, v)
            p = np.clip((w @ _TRACE).real, 0, None)
            c = np.cumsum(p, 1) / p.sum(1, keepdims=True)
            k = (rng.random(n)[:, None] > c[:, :-1]).sum(1)
            m[:, t], y[:, t] = k // 2, k % 2
            v = w[np.arange(n), k] / p[np.arange(n), k][:, None]
        return m, y

    def marginals(self, length: int) -> dict[str, np.ndarray]:
        """Exact P(m_t = 1) and P(y_t = 1) per position."""
        v = _VEC0.astype(complex)
        pm, py = [], []
        for t in range(length):
            s = self.step(t)
            pm.append(float((_TRACE @ (s[1].sum(0) @ v)).real))
            py.append(float((_TRACE @ (s[:, 1].sum(0) @ v)).real))
            v = s.sum((0, 1)) @ v
        return {"m": np.array(pm), "y": np.array(py)}

    def pair_stats(self, length: int, burn: int) -> dict[str, float]:
        """Exact position-averaged (t >= burn) statistics comparable with ``record_stats``."""
        v = _VEC0.astype(complex)
        acc = {k: 0.0 for k in ("m", "y", "m1y1", "m0y1")}
        trans = np.zeros((2, 2, 2))  # [m_t, y_t, m_{t+1}]
        n_pos = 0
        for t in range(length):
            s = self.step(t)
            if t >= burn:
                n_pos += 1
                p = np.array([[(_TRACE @ (s[a, b] @ v)).real for b in (0, 1)] for a in (0, 1)])
                acc["m"] += p[1].sum()
                acc["y"] += p[:, 1].sum()
                acc["m1y1"] += p[1, 1]
                acc["m0y1"] += p[0, 1]
                if t + 1 < length:
                    nxt = self.step(t + 1)
                    for a in (0, 1):
                        for b in (0, 1):
                            w = s[a, b] @ v
                            for c in (0, 1):
                                trans[a, b, c] += (_TRACE @ (nxt[c].sum(0) @ w)).real
            v = s.sum((0, 1)) @ v
        pm = acc["m"] / n_pos
        out = {"p_m1": pm, "p_y1": acc["y"] / n_pos}
        out["p_y1_given_m1"] = acc["m1y1"] / acc["m"]
        out["p_y1_given_m0"] = acc["m0y1"] / (n_pos - acc["m"])
        for a in (0, 1):
            for b in (0, 1):
                out[f"p_m1_next_given_m{a}_y{b}"] = trans[a, b, 1] / trans[a, b].sum()
        return out


def record_stats(m: np.ndarray, y: np.ndarray, burn: int) -> dict[str, float]:
    """The empirical counterparts of ``RecordModel.pair_stats`` (plus step 0 and step 1)."""
    mm, yy = m[:, burn:], y[:, burn:]
    out = {
        "p_m1_step0": float(m[:, 0].mean()),
        "p_m1_step1": float(m[:, 1].mean()),
        "p_m1": float(mm.mean()),
        "p_y1": float(yy.mean()),
        "p_y1_given_m1": float(yy[mm == 1].mean()),
        "p_y1_given_m0": float(yy[mm == 0].mean()),
    }
    for a in (0, 1):
        for b in (0, 1):
            sel = (mm[:, :-1] == a) & (yy[:, :-1] == b)
            out[f"p_m1_next_given_m{a}_y{b}"] = float(mm[:, 1:][sel].mean())
    return out


# -- the fit ------------------------------------------------------------------------------------

FAMILIES = {
    "calibrated": (),
    "decay": ("decay",),
    "decay_first": ("decay_first", "decay"),
}


def model_of(
    family: str, x: Sequence[float], pieces: A.StepPieces, readout_error: float
) -> RecordModel:
    p = dict(zip(FAMILIES[family], x, strict=True))
    return RecordModel.build(pieces, readout_error, p.get("decay", 0.0), p.get("decay_first"))


@dataclass
class Fit:
    family: str
    params: dict[str, float]
    nll: float
    n_params: int


def fit(
    m: np.ndarray,
    y: np.ndarray,
    pieces: A.StepPieces,
    readout_error: float,
    family: str,
    x0: Sequence[float] | None = None,
) -> Fit:
    """Maximum-likelihood fit of ``family`` to the joint records (parameters in [0, 1))."""
    names = FAMILIES[family]
    if not names:
        val = float(model_of(family, (), pieces, readout_error).nll(m, y).sum())
        return Fit(family, {}, val, 0)

    def f(x: np.ndarray) -> float:
        if np.any(x < 0) or np.any(x >= 1):
            return np.inf
        return float(model_of(family, x, pieces, readout_error).nll(m, y).sum())

    start = np.asarray(x0 if x0 is not None else [0.1] * len(names), dtype=float)
    res = minimize(f, start, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4})
    return Fit(family, dict(zip(names, map(float, res.x), strict=True)), float(res.fun), len(names))


def bootstrap_fits(
    m: np.ndarray,
    y: np.ndarray,
    pieces: A.StepPieces,
    readout_error: float,
    family: str,
    n_boot: int,
    seed: int,
    x0: Sequence[float] | None = None,
) -> np.ndarray:
    """(n_boot, n_params) refits on whole-shot resamples (``default_rng(seed)``)."""
    rng = np.random.default_rng(seed)
    n = len(m)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        out.append(list(fit(m[i], y[i], pieces, readout_error, family, x0).params.values()))
    return np.asarray(out)


def predicted_p_m1(
    pieces: A.StepPieces,
    readout_error: float,
    decay: float,
    decay_first: float | None,
    length: int,
    burn: int,
) -> float:
    """The pilot gate's (c) statistic under the model: mean P(m = 1) over positions >= burn."""
    mod = RecordModel.build(pieces, readout_error, decay, decay_first)
    return float(mod.marginals(length)["m"][burn:].mean())


__all__ = [
    "FAMILIES",
    "Fit",
    "RecordModel",
    "bootstrap_fits",
    "fit",
    "model_of",
    "predicted_p_m1",
    "record_stats",
]
