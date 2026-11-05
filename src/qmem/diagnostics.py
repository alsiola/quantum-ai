"""Memory diagnostics for any model whose memory is a quantum state.

Every function takes an ``Instrument`` and an initial memory ``rho0`` (and,
where it needs them, sequences), so it applies equally to architecture A
(``model.instrument(), model.rho0``), the ablations, and HQMM / uMPS
baselines that expose Kraus operators. ``*_for_model`` helpers wrap a
``QuantumMemoryModel``.

* ``memory_states``: the teacher-forced conditional memory ``rho_t`` after
  ``y_1..y_t`` (``t = 0..T``), the same update as ``QuantumMemoryModel.observe``.
* ``stationary_memory``: the time average of ``rho_t`` along trajectories and
  its von Neumann entropy (bits). For an exact model fed stationary data
  this average is ``sum_j pi_j |s_j><s_j|`` and its entropy is ``C_q`` of the
  model's memory encoding. Also returns the outcome-averaged channel's fixed point.
* ``retention_curve``: for lags ``tau``, the mutual information between the
  symbol ``tau`` steps back and the current memory, i.e. the Holevo quantity
  of the classical-quantum state ``sum_x p(x) |x><x| ⊗ rho_x``, with
  ``rho_x`` the average memory given that past symbol; plus the classical
  mutual information between that symbol and the next-symbol prediction
  (``prediction_retention``, which also works for the exact source).
* ``reference_purification``: Gullans & Huse's probe. A reference ``R`` is
  maximally entangled with ``n_reference`` memory qubits; the memory then
  runs the instrument along teacher-forced or self-generated outcomes and
  ``S(rho_R)`` is tracked. It starts at ``n_reference`` bits and falls as the
  measurements learn the initial state.

Lag convention: ``rho_t`` has seen ``y_1..y_t``; lag ``tau = 1`` is ``y_t``,
the most recent symbol, and in general the symbol is ``y_{t+1-tau}``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .instrument import Instrument
from .linalg import DTYPE, REAL_DTYPE, trace
from .states import purity, von_neumann_entropy
from .tasks import channel_fixed_point


def _entropy_bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


@torch.no_grad()
def memory_states(
    instrument: Instrument, rho0: torch.Tensor, sequences: torch.Tensor
) -> torch.Tensor:
    """Teacher-forced memories ``rho_0..rho_T``, shape ``(B, T+1, d, d)``.

    A step whose symbol has ``p(y) = 0`` leaves the memory unchanged (as in
    ``QuantumMemoryModel.observe``).
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    if seq.dim() == 1:
        seq = seq[None]
    rho = rho0.detach().to(DTYPE).expand(seq.shape[0], -1, -1)
    out = [rho]
    for t in range(seq.shape[1]):
        unnorm = instrument.branch(rho, seq[:, t])
        p = trace(unnorm).real
        ok = p > 0
        denom = torch.where(ok, p, torch.ones_like(p))[:, None, None]
        rho = torch.where(ok[:, None, None], unnorm / denom, rho)
        out.append(rho)
    return torch.stack(out, dim=1)


@torch.no_grad()
def stationary_memory(
    instrument: Instrument, rho0: torch.Tensor, sequences: torch.Tensor, burn_in: int = 0
) -> dict[str, Any]:
    """Time-averaged conditional memory over ``t = burn_in+1..T`` and all sequences."""
    states = memory_states(instrument, rho0, sequences)[:, burn_in + 1 :]
    rho_bar = states.mean(dim=(0, 1))
    out: dict[str, Any] = {
        "entropy_bits": float(von_neumann_entropy(rho_bar)),
        "purity": float(purity(rho_bar)),
        "mean_conditional_entropy_bits": float(von_neumann_entropy(states).mean()),
        "eigenvalues": torch.linalg.eigvalsh(rho_bar).clamp(min=0).flip(0).tolist(),
        "burn_in": burn_in,
        "n_states_averaged": int(states.shape[0] * states.shape[1]),
    }
    try:
        fp = channel_fixed_point(instrument)
        ev = torch.linalg.eigvalsh(rho_bar - fp)
        out["channel_fixed_point_entropy_bits"] = float(von_neumann_entropy(fp))
        out["trace_distance_to_fixed_point"] = float(0.5 * ev.abs().sum())
    except ValueError as exc:  # several fixed points: report, don't guess
        out["channel_fixed_point_note"] = str(exc)
    out["rho_bar"] = rho_bar
    return out


def holevo_bits(weights: torch.Tensor, states: torch.Tensor) -> float:
    """``S(sum_x p_x rho_x) - sum_x p_x S(rho_x)`` in bits (``states``: ``(X, d, d)``)."""
    w = torch.as_tensor(weights, dtype=REAL_DTYPE)
    avg = torch.einsum("x,xij->ij", w.to(states.dtype), states)
    return float(von_neumann_entropy(avg)) - float((w * von_neumann_entropy(states)).sum())


def _mutual_information_bits(joint: torch.Tensor) -> float:
    joint = joint / joint.sum()
    return (
        _entropy_bits(joint.sum(1)) + _entropy_bits(joint.sum(0)) - _entropy_bits(joint.flatten())
    )


@torch.no_grad()
def retention_curve(
    instrument: Instrument,
    rho0: torch.Tensor,
    sequences: torch.Tensor,
    lags: list[int],
    burn_in: int = 0,
    n_symbols: int | None = None,
) -> dict[int, dict[str, float]]:
    """Per lag ``tau``: Holevo information between ``y_{t+1-tau}`` and ``rho_t``.

    Pools every ``t`` with ``t > burn_in`` and ``t + 1 - tau >= 1``. Also
    returns the prediction-based mutual information (``prediction_mi_bits``)
    from the instrument's own next-symbol probabilities. Plug-in estimates:
    with ``N`` pooled samples they are biased upwards by about
    ``(X - 1)(Y - 1) / (2 N ln 2)`` bits for the classical one.
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    n_sym = n_symbols or instrument.n_outcomes
    states = memory_states(instrument, rho0, seq)  # (B, T+1, d, d)
    pred = instrument.probabilities(states)  # (B, T+1, Y)
    out: dict[int, dict[str, float]] = {}
    length = seq.shape[1]
    for tau in lags:
        t0 = max(burn_in + 1, tau)
        if t0 > length:
            continue
        ts = torch.arange(t0, length + 1)
        past = seq[:, ts - tau]  # y_{t+1-tau} is index t - tau (0-based)
        rho = states[:, ts]
        weights, cond = [], []
        for x in range(n_sym):
            mask = past == x
            n = int(mask.sum())
            if n == 0:
                continue
            weights.append(n)
            cond.append(rho[mask].mean(0))
        w = torch.tensor(weights, dtype=REAL_DTYPE)
        w = w / w.sum()
        out[tau] = {
            "holevo_bits": holevo_bits(w, torch.stack(cond)),
            "prediction_mi_bits": _prediction_mi(past, pred[:, ts], n_sym),
            "n_samples": int(past.numel()),
        }
    return out


def _prediction_mi(past: torch.Tensor, pred: torch.Tensor, n_symbols: int) -> float:
    onehot = torch.nn.functional.one_hot(past, n_symbols).to(REAL_DTYPE)
    joint = torch.einsum("...x,...y->xy", onehot, pred.to(REAL_DTYPE))
    return _mutual_information_bits(joint)


@torch.no_grad()
def prediction_retention(
    predictive: torch.Tensor,
    sequences: torch.Tensor,
    lags: list[int],
    burn_in: int = 0,
) -> dict[int, float]:
    """Mutual information (bits) between ``y_{t+1-tau}`` and the prediction of ``y_{t+1}``.

    ``predictive`` is ``(B, T+1, Y)`` with row ``t`` the prediction after
    ``y_1..y_t`` (as from ``task.predictive`` or ``model.predictive_probs``),
    so the exact source and any model are scored the same way.
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    length, n_sym = seq.shape[1], predictive.shape[-1]
    out = {}
    for tau in lags:
        t0 = max(burn_in + 1, tau)
        if t0 > length:
            continue
        ts = torch.arange(t0, length + 1)
        out[tau] = _prediction_mi(seq[:, ts - tau], predictive[:, ts], n_sym)
    return out


def _partial_trace_memory(rho_rm: torch.Tensor, d_r: int, d_m: int) -> torch.Tensor:
    r = rho_rm.reshape(*rho_rm.shape[:-2], d_r, d_m, d_r, d_m)
    return torch.einsum("...ajbj->...ab", r)


@torch.no_grad()
def reference_purification(
    instrument: Instrument,
    n_memory: int,
    sequences: torch.Tensor | None = None,
    n_reference: int = 1,
    n: int = 256,
    length: int = 64,
    seed: int = 0,
    threshold_bits: float = 0.01,
) -> dict[str, Any]:
    """Track ``S(rho_R)`` for a reference entangled with the initial memory.

    ``R`` (``n_reference`` qubits) starts in Bell pairs with memory qubits
    ``0..n_reference-1``; the other memory qubits start in ``|0>``. With
    ``sequences`` the outcomes are teacher-forced (steps with ``p(y) = 0`` on
    the joint state are left unchanged and counted); otherwise ``n``
    trajectories of ``length`` steps are drawn from the Born rule with
    ``torch.Generator().manual_seed(seed)``. Returns the mean entropy curve
    ``S_R(t)`` for ``t = 0..T`` and the purification times (first ``t`` with
    ``S_R < threshold_bits``; trajectories that never purify are censored).
    """
    if not 1 <= n_reference <= n_memory:
        raise ValueError("need 1 <= n_reference <= n_memory")
    d_m, d_r = 2**n_memory, 2**n_reference
    d_rest = d_m // d_r
    psi = torch.zeros(d_r, d_m, dtype=DTYPE)
    for i in range(d_r):
        psi[i, i * d_rest] = 1 / math.sqrt(d_r)  # |i>_R |i>_{first qubits} |0..0>_{rest}
    psi = psi.flatten()
    eye_r = torch.eye(d_r, dtype=DTYPE)
    kraus = torch.einsum("ab,yeij->yeaibj", eye_r, instrument.kraus).reshape(
        instrument.n_outcomes, instrument.n_kraus, d_r * d_m, d_r * d_m
    )
    joint = Instrument(kraus)
    if sequences is not None:
        seq = torch.as_tensor(sequences, dtype=torch.long)
        n, length = seq.shape
        gen = None
    else:
        seq = None
        gen = torch.Generator().manual_seed(seed)
    rho = torch.outer(psi, psi.conj()).expand(n, -1, -1)
    curves = [von_neumann_entropy(_partial_trace_memory(rho, d_r, d_m))]
    impossible = 0
    for t in range(length):
        if seq is None:
            p = joint.probabilities(rho).clamp(min=0)
            y = torch.multinomial(p, 1, generator=gen).squeeze(-1)
        else:
            y = seq[:, t]
        unnorm = joint.branch(rho, y)
        p_y = trace(unnorm).real
        ok = p_y > 0
        impossible += int((~ok).sum())
        denom = torch.where(ok, p_y, torch.ones_like(p_y))[:, None, None]
        rho = torch.where(ok[:, None, None], unnorm / denom, rho)
        curves.append(von_neumann_entropy(_partial_trace_memory(rho, d_r, d_m)))
    s = torch.stack(curves, dim=1)  # (n, T+1)
    below = s < threshold_bits
    first = torch.where(below.any(1), below.float().argmax(1), torch.full((n,), -1))
    purified = first >= 0
    times = first[purified].to(REAL_DTYPE)
    return {
        "mode": "teacher_forced" if seq is not None else "generated",
        "n_reference": n_reference,
        "entropy_mean_bits": s.mean(0).tolist(),
        "entropy_se_bits": (s.std(0) / math.sqrt(n)).tolist() if n > 1 else None,
        "threshold_bits": threshold_bits,
        "fraction_purified": float(purified.double().mean()),
        "purification_time_median": float(times.median()) if len(times) else math.nan,
        "purification_time_mean_purified": float(times.mean()) if len(times) else math.nan,
        "impossible_steps": impossible,
        "seed": None if seq is not None else seed,
    }


# -- model wrappers ---------------------------------------------------------------
def memory_report_for_model(
    model: Any,
    sequences: torch.Tensor,
    lags: list[int],
    burn_in: int = 0,
    purification_seed: int = 0,
    n_reference: int = 1,
) -> dict[str, Any]:
    """All three diagnostics for a ``QuantumMemoryModel`` on teacher-forced data.

    The purification probe is run both teacher-forced on ``sequences`` and on
    the model's own Born-rule outcomes (seeded).
    """
    with torch.no_grad():
        inst = model.instrument()
        stat = stationary_memory(inst, model.rho0, sequences, burn_in)
        stat.pop("rho_bar")
        b, t = torch.as_tensor(sequences).shape
        return {
            "stationary_memory": stat,
            "retention": retention_curve(inst, model.rho0, sequences, lags, burn_in),
            "purification_teacher_forced": reference_purification(
                inst, model.n_memory, sequences, n_reference=n_reference
            ),
            "purification_generated": reference_purification(
                inst, model.n_memory, None, n_reference, n=b, length=t, seed=purification_seed
            ),
        }
