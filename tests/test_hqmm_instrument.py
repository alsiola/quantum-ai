"""The single-step instrument tests of ``test_instrument.py``, run on the Stiefel HQMM.

The HQMM reuses ``qmem.Instrument`` (and ``QuantumMemoryModel``'s teacher-forcing
and sampling code), so the same physics checks apply: unitarity of the dilation,
completeness, valid channels, probabilities and conditional states. Importing
the test functions here re-runs them with this module's ``model`` fixture.
"""

import pytest
from test_instrument import (  # noqa: F401  (collected again with the fixture below)
    test_branch_accepts_per_item_outcomes,
    test_channel_trace_preserving_and_positive,
    test_completeness,
    test_conditional_states_valid_and_probs_consistent,
    test_each_outcome_is_trace_nonincreasing,
    test_outcome_average_equals_channel,
    test_probabilities_valid,
    test_unitary_is_unitary,
)

from qmem.baselines import StiefelInstrumentModel

HQMM_CONFIGS = {
    "m1_o1": (1, 1, 0),
    "m2_o1_e1": (2, 1, 1),
    "m2_o2": (2, 2, 0),
    "m3_o1_e2": (3, 1, 2),
}


@pytest.fixture(params=sorted(HQMM_CONFIGS))
def model(request):
    n_memory, n_output, n_env = HQMM_CONFIGS[request.param]
    return StiefelInstrumentModel(n_memory, n_output, n_env, seed=len(request.param))
