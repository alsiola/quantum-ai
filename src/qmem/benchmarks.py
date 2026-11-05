"""The Phase 3 memory benchmark suite: task specs, literature numbers, recall scoring.

``LITERATURE`` lists every number we take from a paper, with its source and
the value we compute from the process definition (``literature_checks``
evaluates them; ``tests/test_benchmarks.py`` asserts them). Each entry's
``compute`` takes no arguments and returns our value; ``tol`` is the
tolerance (wider where the paper's number is read off a figure or is a
finite-sample estimate).

``recall_metrics`` scores a model at the recall positions of a
``delayed_recall`` task (NLL and accuracy there, against the exact source).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .complexity import epsilon_machine, memory_profile, pmd_qubit_instrument, quantum_overlaps
from .instrument import Instrument
from .metrics import operational_strength, strength_lower_bound
from .model import ModelConfig, build_model
from .states import von_neumann_entropy
from .tasks import (
    CUE0,
    CUE1,
    HMMTask,
    QuantumSourceTask,
    channel_fixed_point,
    delayed_recall,
    ghafari_process,
    marked_recall_events,
    mixed_timescales,
    nemo_process,
    parity_task,
    perturbed_coin,
    pmd_renewal,
    probability_clock,
    recall_positions,
    rk_golden_mean,
    srinivasan_hmm,
    stationary_distribution,
    stern_gerlach,
    uniform_renewal,
    upset_gambler,
)

LN2 = math.log(2)


def _bits_h2(x: float) -> float:
    return -sum(p * math.log2(p) for p in (x, 1 - x) if p > 0)


def perturbed_coin_cq(p: float) -> float:
    """Closed form (Gu et al. 2012): eigenvalues ``1/2 ± sqrt(p(1-p))`` of the stationary memory."""
    return _bits_h2(0.5 + math.sqrt(p * (1 - p)))


def description_accuracy(rate_nats: float, n_symbols: int) -> float:
    """Srinivasan et al. 2018 DA for per-symbol NLL ``rate_nats``: ``1 - h / ln s``."""
    return 1 - rate_nats / math.log(n_symbols)


def _hmm_rate_estimate(n: int = 200, length: int = 3000, burn: int = 1000, seed: int = 1) -> float:
    """Monte Carlo entropy rate of the (non-unifilar) Srinivasan HMM, nats; SE ~ 1e-3 nats."""
    task = srinivasan_hmm()
    lp = -task.log_probs(task.sample(n, length, seed))[:, burn:]
    return float(lp.mean())


def _pmd_qubit_reproduces(period: int, gamma: float, v: float) -> float:
    """Max |difference| between Wu's 1-qubit model and the PMD predictive (should be ~1e-15)."""
    inst, _ = pmd_qubit_instrument(period, gamma, v)
    task = pmd_renewal(period, gamma, v)
    seq = task.sample(64, 48, seed=11)
    return float((QuantumSourceTask(inst).predictive(seq) - task.predictive(seq)).abs().max())


@dataclass(frozen=True)
class LiteratureNumber:
    key: str
    source: str  # citation key in docs/references.bib + where in the paper
    quantity: str
    reported: float
    compute: Callable[[], float]
    tol: float
    how: str  # "text" (stated in the text), "figure" (read off a figure), "derived"


def _mp(task_fn: Callable[[], Any], key: str) -> Callable[[], float]:
    return lambda: float(memory_profile(task_fn())[key])


LITERATURE: list[LiteratureNumber] = [
    LiteratureNumber(
        "coin_cq_p0.2",
        "Gu2012, text: 'For p>0.2, Cq<0.5'",
        "C_q (bits) of the perturbed coin at p = 0.2 (< 0.5)",
        0.5,
        _mp(lambda: perturbed_coin(0.2), "cq_bits"),
        0.0,
        "text (upper bound)",
    ),
    LiteratureNumber(
        "coin_cq_p0.4",
        "Gu2012, text: 'For p>0.4, Cq<0.1'",
        "C_q (bits) of the perturbed coin at p = 0.4 (< 0.1)",
        0.1,
        _mp(lambda: perturbed_coin(0.4), "cq_bits"),
        0.0,
        "text (upper bound)",
    ),
    LiteratureNumber(
        "coin_cmu",
        "Gu2012, text: C_mu = 1 for p != 0.5",
        "C_mu (bits) of the perturbed coin, p = 0.3",
        1.0,
        _mp(lambda: perturbed_coin(0.3), "c_mu_bits"),
        1e-12,
        "text",
    ),
    LiteratureNumber(
        "ghafari_states",
        "Ghafari2019PRX Fig. 1b; Yang2021 Fig. 3a",
        "causal states of the asymmetric process",
        3,
        _mp(lambda: ghafari_process(0.3, 0.8), "n_causal_states"),
        0,
        "text",
    ),
    LiteratureNumber(
        "ghafari_qdim",
        "Ghafari2019PRX Eq. 2; Yang2021 ('achieved quantumly on a qubit')",
        "quantum memory dimension of the asymmetric process (p=0.3, q=0.8)",
        2,
        _mp(lambda: ghafari_process(0.3, 0.8), "quantum_dim"),
        0,
        "text",
    ),
    LiteratureNumber(
        "ghafari_cmu_peak",
        "Ghafari2019PRX Fig. 4b (p = q cut, peak at q = 0.5)",
        "C_mu (bits) at p = q = 0.5",
        1.5,
        _mp(lambda: ghafari_process(0.5, 0.5), "c_mu_bits"),
        0.02,
        "figure",
    ),
    LiteratureNumber(
        "ghafari_cq_mid",
        "Ghafari2019PRX Fig. 4b (p = q cut, minimum at q = 0.5)",
        "C_q (bits) at p = q = 0.5",
        0.82,
        _mp(lambda: ghafari_process(0.5, 0.5), "cq_bits"),
        0.03,
        "figure",
    ),
    LiteratureNumber(
        "upset_gambler_cmu",
        "Binder2018 Fig. 2 caption (p=0.7, q=0.8)",
        "C_mu (bits)",
        0.98,
        _mp(lambda: upset_gambler(0.7, 0.8), "c_mu_bits"),
        0.005,
        "text",
    ),
    LiteratureNumber(
        "upset_gambler_cq",
        "Binder2018 Fig. 2 caption (p=0.7, q=0.8)",
        "C_q (bits)",
        0.10,
        _mp(lambda: upset_gambler(0.7, 0.8), "cq_bits"),
        0.005,
        "text",
    ),
    LiteratureNumber(
        "upset_gambler_overlap",
        "Binder2018 Eq. 13: <s_A|s_B> = sqrt(p'q') / (1 - sqrt(pq))",
        "quantum causal-state overlap at p=0.7, q=0.8",
        math.sqrt(0.3 * 0.2) / (1 - math.sqrt(0.7 * 0.8)),
        lambda: float(quantum_overlaps(upset_gambler(0.7, 0.8))["gram"][0, 1]),
        1e-10,
        "text (formula)",
    ),
    *[
        LiteratureNumber(
            f"pmd_N{n}_states",
            "Wu2023 Fig. 1 caption / D_mu = log2 N",
            f"causal states of PMD, N={n}, Gamma=0.5, V=0.4",
            n,
            _mp(lambda n=n: pmd_renewal(n, 0.5, 0.4), "n_causal_states"),
            0,
            "text",
        )
        for n in range(3, 9)
    ],
    *[
        LiteratureNumber(
            f"pmd_N{n}_qubit_model",
            "Wu2023 Supplement S1 (1-qubit construction, m = 1)",
            f"max |p_qubit - p_PMD| over sampled histories, N={n}, Gamma=0.5, V=0.4 (0 = exact)",
            0.0,
            lambda n=n: _pmd_qubit_reproduces(n, 0.5, 0.4),
            1e-12,
            "derived",
        )
        for n in range(3, 9)
    ],
    LiteratureNumber(
        "uniform_renewal_states",
        "Yang2021 ('the classical topological complexity for U_N is N'), N=3",
        "causal states of the uniform renewal process, N = 3",
        3,
        _mp(lambda: uniform_renewal(3), "n_causal_states"),
        0,
        "text",
    ),
    LiteratureNumber(
        "golden_43_cryptic",
        "Mahoney2016 Fig. 2 / text: C_q(3) = C_q(L>3), cryptic order 3",
        "L at which C_q(L) stops decreasing, 4-3 golden mean (p = 0.505)",
        3,
        _mp(lambda: rk_golden_mean(4, 3, 0.505), "cq_saturates_at"),
        0,
        "text",
    ),
    LiteratureNumber(
        "golden_32_cmu",
        "Ho2020 Sec. IV.A (3-2 golden mean, p = 0.9): C_mu ≈ 1.435",
        "C_mu (bits)",
        1.435,
        _mp(lambda: rk_golden_mean(2, 3, 0.9, loop_symbol=1), "c_mu_bits"),
        0.001,
        "text",
    ),
    LiteratureNumber(
        "golden_32_cq",
        "Ho2020 Fig. 3c (exact C_q line, p = 0.9)",
        "C_q (bits)",
        0.613,
        _mp(lambda: rk_golden_mean(2, 3, 0.9, loop_symbol=1), "cq_bits"),
        0.002,
        "figure",
    ),
    LiteratureNumber(
        "nemo_cq_inf",
        "Mahoney2016 text: C_q(inf) ≈ 1.0332 (Fig. 3, p = 0.666)",
        "C_q (bits), Nemo p = 0.666",
        1.0332,
        _mp(lambda: nemo_process(0.666), "cq_bits"),
        0.001,
        "text",
    ),
    LiteratureNumber(
        "nemo_ho_cmu",
        "Ho2020 Sec. IV.B (p = 0.1, q = 0.9): C_mu ≈ 1.583",
        "C_mu (bits)",
        1.583,
        _mp(lambda: nemo_process(0.1, 0.9, loop_symbol=1), "c_mu_bits"),
        0.001,
        "text",
    ),
    LiteratureNumber(
        "nemo_ho_cq",
        "Ho2020 Fig. 4c (exact C_q line, p = 0.1, q = 0.9)",
        "C_q (bits)",
        1.53,
        _mp(lambda: nemo_process(0.1, 0.9, loop_symbol=1), "cq_bits"),
        0.03,
        "figure",
    ),
    LiteratureNumber(
        "clock_da",
        "Srinivasan2018 Table 2, true HQMM test DA 0.1632 (SD 0.0111)",
        "DA of the exact model (1 - h / ln 2)",
        0.1632,
        lambda: description_accuracy(probability_clock().entropy_rate(), 2),
        0.0111,
        "text (finite-sample)",
    ),
    LiteratureNumber(
        "spin_da",
        "Srinivasan2018 Table 4, true model test DA 0.1303 (SD 0.0047)",
        "DA of the exact model (1 - h / ln 6)",
        0.1303,
        lambda: description_accuracy(stern_gerlach().entropy_rate(), 6),
        0.0047,
        "text (finite-sample)",
    ),
    LiteratureNumber(
        "srinivasan_hmm_da",
        "Srinivasan2018 Table 5, true (6,6) HMM train DA 0.1838 (SD 0.0095)",
        "DA of the exact model (1 - h / ln 6; h by Monte Carlo)",
        0.1838,
        lambda: description_accuracy(_hmm_rate_estimate(), 6),
        0.0095,
        "text (finite-sample)",
    ),
]


def literature_checks(keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Evaluate each recorded literature number against our computation.

    Upper-bound entries pass when ours is below the bound; the others when
    ``|ours - reported| <= tol``.
    """
    out = []
    for item in LITERATURE:
        if keys is not None and item.key not in keys:
            continue
        ours = item.compute()
        if "upper bound" in item.how:
            ok = ours < item.reported
        else:
            ok = abs(ours - item.reported) <= item.tol
        out.append(
            {
                "key": item.key,
                "source": item.source,
                "quantity": item.quantity,
                "reported": item.reported,
                "ours": ours,
                "tol": item.tol,
                "how": item.how,
                "agrees": bool(ok),
            }
        )
    return out


# -- scoring at the recall positions -----------------------------------------------
@torch.no_grad()
def recall_metrics(model: Any, task: Any, sequences: torch.Tensor) -> dict[str, Any]:
    """NLL and accuracy at the recall positions of a ``delayed_recall`` task.

    The exact source predicts the recall symbol with certainty (NLL 0,
    accuracy 1); a model with no memory of the cue scores ``ln 2`` and 0.5.
    Also reports the per-symbol NLL over the other positions.
    """
    tau = int(task.params["tau"])
    seq = torch.as_tensor(sequences, dtype=torch.long)
    pos = recall_positions(tau, seq.shape[1])
    lp = model.step_log_probs(seq)  # (B, T)
    pred = model.predictive_probs(seq)[:, :-1]  # (B, T, Y)
    src_pred = task.predictive(seq)[:, :-1]
    other = torch.ones(seq.shape[1], dtype=torch.bool)
    other[pos] = False
    rec_nll = -lp[:, pos]
    return {
        "tau": tau,
        "n_recall_positions": len(pos) * seq.shape[0],
        "recall_nll_nats": float(rec_nll.mean()),
        "recall_nll_se": float(rec_nll.mean(1).std() / math.sqrt(seq.shape[0])),
        "recall_accuracy": float((pred[:, pos].argmax(-1) == seq[:, pos]).double().mean()),
        "recall_p_correct_mean": float(torch.exp(-rec_nll).mean()),
        "other_nll_nats": float(-lp[:, other].mean()),
        "source_recall_nll_nats": float(
            -torch.log(src_pred[:, pos].gather(-1, seq[:, pos, None])).mean()
        ),
        "source_recall_accuracy": float(
            (src_pred[:, pos].argmax(-1) == seq[:, pos]).double().mean()
        ),
        "memoryless_recall_nll_nats": LN2,
    }


def _type_value(pred: torch.Tensor, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``-log p(type)`` and ``-log p(value | type)`` of the observed symbols (``marked_recall``)."""
    p_cue = pred[..., CUE0] + pred[..., CUE1]
    is_cue = seq >= CUE0
    p_type = torch.where(is_cue, p_cue, 1 - p_cue)
    p_sym = pred.gather(-1, seq[..., None]).squeeze(-1)
    return -torch.log(p_type), -torch.log(p_sym / p_type)


@torch.no_grad()
def marked_recall_metrics(
    model: Any, task: Any, sequences: torch.Tensor, delay_bins: list[int] | None = None
) -> dict[str, Any]:
    """Scores of a model on a ``marked_recall`` task (issue #10), against the exact source.

    Each position's loss splits as ``-log p(y) = -log p(type) - log p(value | type)``
    (type: distractor or cue-type; value: a/b or 0/1). At the recall positions the
    *value* NLL and accuracy (cue-1 against cue-0) measure the memory and leave out
    when the recall comes: optimum ``H(eps)`` and ``1 - eps``; with no memory of the
    cue, ``ln 2`` and 0.5. The paired gaps (model - source, nats per position) are
    reported for the type (all positions) and for the value at distractor, cue and
    recall positions. With geometric timing the type is i.i.d. and independent of
    the history, so each of these averages is an expected KL and is ``>= 0``; a model
    that protects its memory by mispredicting distractors shows it in the type or
    distractor-value gap. ``by_delay``: value accuracy and NLL per realised delay
    (distractors between cue and recall), binned at the left edges ``delay_bins``.
    """
    seq = torch.as_tensor(sequences, dtype=torch.long)
    ev = marked_recall_events(seq)
    pred = model.predictive_probs(seq)[:, :-1]
    src = task.predictive(seq)[:, :-1]
    m_type, m_val = _type_value(pred, seq)
    s_type, s_val = _type_value(src, seq)
    rec, dis, cue = ev["recall"], ev["distractor"], ev["cue"]
    truth = seq == CUE1

    def acc(p: torch.Tensor) -> torch.Tensor:
        return ((p[..., CUE1] > p[..., CUE0]) == truth).double()

    m_acc, s_acc = acc(pred), acc(src)
    n_rec = int(rec.sum())
    v_nll = m_val[rec]
    gap_val = m_val - s_val
    out: dict[str, Any] = {
        "tau": task.params["tau"],
        "eps": task.params.get("eps", 0.0),
        "n_recalls": n_rec,
        "recall_value_nll_nats": float(v_nll.mean()),
        "recall_value_nll_se": float(v_nll.std() / math.sqrt(n_rec)),
        "recall_accuracy": float(m_acc[rec].mean()),
        "recall_accuracy_se": float(m_acc[rec].std() / math.sqrt(n_rec)),
        "recall_nll_nats": float((m_type + m_val)[rec].mean()),
        "source_recall_value_nll_nats": float(s_val[rec].mean()),
        "source_recall_accuracy": float(s_acc[rec].mean()),
        "source_recall_nll_nats": float((s_type + s_val)[rec].mean()),
        "memoryless_recall_value_nll_nats": LN2,
        "gap_all_nats": float((m_type + m_val - s_type - s_val).mean()),
        "gap_type_nats": float((m_type - s_type).mean()),
        "gap_value_distractor_nats": float(gap_val[dis].mean()),
        "gap_value_cue_nats": float(gap_val[cue].mean()),
        "gap_value_recall_nats": float(gap_val[rec].mean()),
        "fraction_recall": n_rec / seq.numel(),
        "fraction_distractor": float(dis.double().mean()),
        "fraction_cue": float(cue.double().mean()),
    }
    out["gap_non_recall_nats"] = (
        out["gap_all_nats"] - out["fraction_recall"] * out["gap_value_recall_nats"]
    ) / (1 - out["fraction_recall"])
    bins = delay_bins or [0, 1, 2, 4, 8, 16, 32, 64]
    delays = ev["delay"][rec]
    rows = []
    for i, lo in enumerate(bins):
        hi = bins[i + 1] if i + 1 < len(bins) else None
        m = delays >= lo if hi is None else (delays >= lo) & (delays < hi)
        n = int(m.sum())
        if n == 0:
            continue
        a, nll = m_acc[rec][m], v_nll[m]
        rows.append(
            {
                "delay_lo": lo,
                "delay_hi": hi,
                "n": n,
                "mean_delay": float(delays[m].double().mean()),
                "accuracy": float(a.mean()),
                "accuracy_se": float(a.std() / math.sqrt(n)) if n > 1 else math.nan,
                "value_nll_nats": float(nll.mean()),
            }
        )
    out["by_delay"] = rows
    return out


# -- one row of the suite catalogue ---------------------------------------------------
def entropy_rate_monte_carlo(
    task: Any, n: int = 200, length: int = 3000, burn_in: int = 1000, seed: int = 0
) -> dict[str, float]:
    """``-log p(y_t | y_<t)`` averaged over ``t > burn_in`` of long sampled sequences.

    For sources without a closed-form rate (non-unifilar HMMs); converges by
    Shannon-McMillan-Breiman. The SE is over the ``n`` sequences.
    """
    lp = -task.log_probs(task.sample(n, length, seed))[:, burn_in:]
    per = lp.mean(1)
    return {
        "nats": float(per.mean()),
        "se": float(per.std() / math.sqrt(n)),
        "n": n,
        "length": length,
        "burn_in": burn_in,
        "seed": seed,
    }


def describe_benchmark(task: Any, length: int, mc_seed: int = 0) -> dict[str, Any]:
    """Exact optimum, entropy rate and memory figures of one benchmark task (JSON-ready).

    Every figure is exact except ``entropy_rate_mc`` (only for sources whose
    rate has no closed form; then ``entropy_rate_nats`` is ``None``).
    """
    emitted = int((task_emission_mass(task) > 0).sum())
    row: dict[str, Any] = {
        "name": task.name,
        "params": task.params,
        "n_symbols": task.n_symbols,
        "n_symbols_emitted": emitted,
        "length": length,
    }
    if isinstance(task, QuantumSourceTask):
        row["generator"] = f"HQMM, quantum dim {task.dim}"
        row["generator_quantum_dim"] = task.dim
    else:
        row["generator"] = f"HMM, {task.n_states} states"
        row["generator_states"] = task.n_states
    try:
        row["optimal_nll_nats"] = task.optimal_nll(length)
    except RuntimeError as exc:
        row["optimal_nll_nats"] = None
        row["optimal_nll_note"] = str(exc)
    exact_rate = isinstance(task, QuantumSourceTask) or task.is_unifilar()
    if exact_rate:
        row["entropy_rate_nats"] = task.entropy_rate()
    else:
        row["entropy_rate_nats"] = None
        row["entropy_rate_upper_bound_nats"] = task.entropy_rate()
        row["entropy_rate_mc"] = entropy_rate_monte_carlo(task, seed=mc_seed)
    row["memory"] = memory_profile(task)
    if task.params.get("kind") == "pmd_renewal":
        p = task.params
        err = _pmd_qubit_reproduces(int(p["period"]), float(p["gamma"]), float(p["v"]))
        inst, _ = pmd_qubit_instrument(int(p["period"]), float(p["gamma"]), float(p["v"]))
        row["memory"]["qubit_model"] = {
            "source": "Wu2023 Supplement S1",
            "quantum_dim": 2,
            "max_predictive_error": err,
            "stationary_entropy_bits": float(von_neumann_entropy(channel_fixed_point(inst))),
        }
    return row


def task_emission_mass(task: Any) -> torch.Tensor:
    """Total probability weight each symbol can carry (zero for padding symbols)."""
    if isinstance(task, QuantumSourceTask):
        return task.instrument.kraus.abs().flatten(1).sum(1)
    return task.labelled.sum(dim=(1, 2))


# -- noise eps and the strength floor (for the measurement-strength sweep) ------------
def strength_floor(task: HMMTask, dim: int) -> dict[str, Any]:
    """``eta_floor`` of ``qmem.metrics.strength_lower_bound``, exact for a unifilar source.

    The bound needs ``max_h p(y|h)`` over all histories. Every causal state is
    reached by some history and every other belief is a mixture of them, so
    the maximum is over the epsilon-machine's emission rows: the bound is
    attained, not estimated. ``dim`` is the model's memory dimension.
    """
    em = epsilon_machine(task)
    rows = em["labelled"].sum(-1).T  # (n_causal_states, Y)
    out = strength_lower_bound(rows, dim)
    out["dim"] = dim
    return out


NOISY_PROBES: dict[str, Callable[[float], HMMTask]] = {
    "recall_tau1": lambda e: delayed_recall(1, eps=e),
    "recall_tau4": lambda e: delayed_recall(4, eps=e),
    "recall_tau8": lambda e: delayed_recall(8, eps=e),
    "recall_tau16": lambda e: delayed_recall(16, eps=e),
    "parity_k2": lambda e: parity_task(2, noise=e),
    "parity_k3": lambda e: parity_task(3, noise=e),
    "parity_k4": lambda e: parity_task(4, noise=e),
    "mixed_timescales": lambda e: mixed_timescales(0.02, 0.9, eps=e),
}


def noise_profile(
    eps_values: list[float], length: int, dims: list[int], probes: list[str] | None = None
) -> list[dict[str, Any]]:
    """Exact optimum, rate and ``eta_floor`` of each retention probe as a function of ``eps``."""
    rows = []
    for name in probes or list(NOISY_PROBES):
        for eps in eps_values:
            task = NOISY_PROBES[name](eps)
            emit = task.labelled.sum(-1)  # (Y, S): emission probabilities per state
            used = emit[:, stationary_or_initial(task) > 0]
            row = {
                "probe": name,
                "eps": eps,
                "optimal_nll_nats": task.optimal_nll(length),
                "entropy_rate_nats": task.entropy_rate(),
                "min_emission_prob": float(used.min()),
                "max_emission_prob": float(used.max()),
                "eta_floor": {d: strength_floor(task, d)["eta_lower_bound"] for d in dims},
                "length": length,
            }
            if task.params["kind"] == "delayed_recall":
                h = sum(-p * math.log(p) for p in (eps, 1 - eps) if p > 0)
                row["recall_optimal_nll_nats"] = h
                row["recall_optimal_accuracy"] = 1 - eps
            rows.append(row)
    return rows


def stationary_or_initial(task: HMMTask) -> torch.Tensor:
    """States with weight in the stationary distribution or at the start."""
    return stationary_distribution(task.transition) + task.initial


# -- the feed-forward layout's own strength window (issue #10) -------------------------
def feedforward_readout(n_memory: int, n_output: int, g: float) -> Instrument:
    """The bare coupling ``D_y(g)`` of architecture A (shared ``g``, ``V = W_y = I``).

    ``V`` and ``W_y`` don't change the spectra of the effects or ``eta``, so every
    feed-forward instrument with strength ``g`` has these effect eigenvalues.
    """
    cfg = ModelConfig(
        n_memory=n_memory, n_output=n_output, memory_depth=0, strength=g, strength_shared=True
    )
    return build_model(cfg).instrument()


def feedforward_eta(n_memory: int, n_output: int, g: float) -> float:
    """Operational strength ``eta`` of the feed-forward layout at shared strength ``g``."""
    with torch.no_grad():
        return float(operational_strength(feedforward_readout(n_memory, n_output, g)))


def feedforward_g_for_eta(n_memory: int, n_output: int, eta: float, tol: float = 1e-12) -> float:
    """The ``g`` in ``[0, pi/2]`` with ``feedforward_eta(g) = eta`` (bisection; monotone)."""
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0, 1]")
    lo, hi = 0.0, math.pi / 2
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if feedforward_eta(n_memory, n_output, mid) < eta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def feedforward_strength_floor(
    task: HMMTask, n_memory: int, n_output: int, tol: float = 1e-10
) -> dict[str, Any]:
    """Least shared ``g`` at which architecture A passes the per-outcome spectral test.

    An exact model needs ``lambda_min(E_y) <= p(y|h) <= lambda_max(E_y)`` for every
    outcome and history. In the feed-forward layout the spectra depend on ``g``
    alone, and both conditions get easier as ``g`` grows, so the least admissible
    ``g`` is found by bisection. The extreme ``p(y|h)`` are taken over the causal
    states (exact for a unifilar source, as in ``strength_floor``). Necessary, not
    sufficient: passing it does not mean the dynamics can be matched.
    """
    em = epsilon_machine(task)
    rows = em["labelled"].sum(-1).T  # (n_causal_states, Y)
    top, bottom = rows.max(0).values, rows.min(0).values

    def ok(g: float) -> bool:
        ev = torch.linalg.eigvalsh(feedforward_readout(n_memory, n_output, g).effects())
        return bool((ev[:, -1] >= top - 1e-12).all() and (ev[:, 0] <= bottom + 1e-12).all())

    if not ok(math.pi / 2):
        return {"g_floor": None, "eta_floor": None, "admissible": False}
    lo, hi = 0.0, math.pi / 2
    if ok(0.0):
        hi = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return {
        "g_floor": hi,
        "eta_floor": feedforward_eta(n_memory, n_output, hi),
        "admissible": True,
        "max_prob_per_outcome": top.tolist(),
        "min_prob_per_outcome": bottom.tolist(),
    }
