"""Phase 3a baselines: classical and model-class comparators for architecture A.

Every model exposes the interface of ``qmem.QuantumMemoryModel`` (``step_log_probs``,
``log_prob``, ``predictive_probs``, ``sample``, ``n_parameters``, ``diagnostics``)
plus ``describe()`` (state dimension and compute cost), so it trains with
``qmem.train.train`` and is scored with ``qmem.train.evaluate`` on the same splits.
"""

from .base import RecurrentModel, SequenceModel
from .hmm import HMMModel, fit_em
from .hqmm import GeneratorInstrumentModel, StiefelInstrumentModel
from .ngram import NGramModel, fit_ngram
from .rnn import RNNModel, matched_hidden_size, rnn_parameter_count
from .runner import BaselineSpec, build_baseline, describe_model, run_baseline
from .umps import UMPSModel

__all__ = [
    "BaselineSpec",
    "GeneratorInstrumentModel",
    "HMMModel",
    "NGramModel",
    "RNNModel",
    "RecurrentModel",
    "SequenceModel",
    "StiefelInstrumentModel",
    "UMPSModel",
    "build_baseline",
    "describe_model",
    "fit_em",
    "fit_ngram",
    "matched_hidden_size",
    "rnn_parameter_count",
    "run_baseline",
]
