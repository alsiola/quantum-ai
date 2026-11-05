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

from .complexity import memory_profile, pmd_qubit_instrument, quantum_overlaps
from .states import von_neumann_entropy
from .tasks import (
    QuantumSourceTask,
    channel_fixed_point,
    ghafari_process,
    nemo_process,
    perturbed_coin,
    pmd_renewal,
    probability_clock,
    recall_positions,
    rk_golden_mean,
    srinivasan_hmm,
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
