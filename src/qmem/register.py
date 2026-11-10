"""Phase 8 Stage 2b (issue #34): a classical model with a cyclic phase register.

#32's Lemma: in a finite classical model, maps that compose exactly as ``Rz(pi/2) Rz(-pi/2) =
identity`` (and ``Rz(pi/2)^2 = Rz(pi)``) must be permutations. A classical model *with*
permutation maps escapes it. The natural one for a phase-scan design is a discretised Bloch
circle: a hidden phase index ``theta`` in ``Z_n`` (``n = 8`` for the scan's ``k pi / 4``
steps) that each echo ``e<k>`` shifts by ``k``, times a population register ``p`` of
``n_pop`` states. The echoes form exactly the phase group, so INV2 = E00, COMP22 = COMP40 and
ALT2 compose exactly, as they do for the quantum instrument.

The model is a ``qmem.classical.ProgramModel`` with ``d = n * n_pop`` hidden states, index
``s = p * n + theta``:

* ``e<k>``: the fixed permutation ``(theta, p) -> (theta + k mod n, p)`` (never fitted);
* the step's ``pre`` and ``post[y]`` (the CZ, q59's measurement and reset, the readout), the
  initial state, A2's labelled ``meas`` and each feed-forward element ``ff<b1><b2>``: free
  stochastic maps on all ``d`` states;
* the randomised dephasings R2 and R4 are not free either: each branch applies its recorded
  bits' permutations, so averaged over the bits the step is the bit-averaged mixture of the
  same permutations.

``fit`` it with ``qmem.classical.fit_program_model(..., frozen=echo_names(model))``: EM never
moves the permutations. Numpy only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from . import classical as C

N_PHASE = 8
_ECHO = re.compile(r"^e(\d+)$")


def echo_shift(name: str) -> int | None:
    """``k`` of an echo element ``e<k>`` (``None`` for any other element)."""
    m = _ECHO.match(name)
    return int(m.group(1)) if m else None


def shift(k: int, n_pop: int, n_phase: int = N_PHASE) -> np.ndarray:
    """(d, d) row-stochastic permutation ``(theta, p) -> (theta + k mod n, p)``."""
    d = n_phase * n_pop
    out = np.zeros((d, d))
    for p in range(n_pop):
        for t in range(n_phase):
            out[p * n_phase + t, p * n_phase + (t + k) % n_phase] = 1.0
    return out


def echo_names(model: C.ProgramModel) -> list[str]:
    """The model's echo elements (the fixed permutations: pass as ``frozen``)."""
    return sorted(k for k in model.maps if echo_shift(k) is not None)


def set_echoes(
    model: C.ProgramModel, n_pop: int, n_phase: int = N_PHASE, names: Sequence[str] = ()
) -> C.ProgramModel:
    """Every echo map of ``model`` (and ``names``) set to its exact shift (in place)."""
    for k in {*echo_names(model), *names}:
        model.maps[k] = shift(echo_shift(k), n_pop, n_phase)
    return model


def is_register(model: C.ProgramModel, n_pop: int, n_phase: int = N_PHASE) -> bool:
    return all(
        np.array_equal(model.maps[k], shift(echo_shift(k), n_pop, n_phase))
        for k in echo_names(model)
    )


def prism(n_pop: int, radius: float, height: float = 1.0, n_phase: int = N_PHASE) -> np.ndarray:
    """(4, d) Bloch-space vertices ``(1, x, y, z)`` of the register's states, in its index
    order: ``n_pop`` rings at ``z`` evenly spaced in ``[-height, height]`` (top ring first),
    ``n_phase`` points each at angle ``2 pi theta / n``. A rotation by ``2 pi k / n`` about z
    maps the vertex set to itself as ``shift(k)``. ``radius >= 1 / cos(pi / n)`` and
    ``height >= 1`` contain the Bloch ball."""
    zs = [height] if n_pop == 1 else np.linspace(height, -height, n_pop)
    cols = []
    for z in zs:
        for t in range(n_phase):
            a = 2 * math.pi * t / n_phase
            cols.append((1.0, radius * math.cos(a), radius * math.sin(a), float(z)))
    return np.array(cols).T


def register_start(
    maps: dict[str, np.ndarray],
    n_pop: int,
    elements: Sequence[str],
    labelled: Sequence[str] = (),
    radius: float | None = None,
    height: float = 1.0,
    eps: float = 0.01,
    n_phase: int = N_PHASE,
) -> C.ProgramModel:
    """A start carrying the quantum model's structure (``qmem.coherence.polytope_start`` on
    the prism), with the echoes set to their exact shifts."""
    from . import coherence as Q

    r = 1.0 / math.cos(math.pi / n_phase) if radius is None else radius
    u = prism(n_pop, r, height, n_phase)
    m = Q.polytope_start(maps, u, elements, labelled, eps)
    return set_echoes(m, n_pop, n_phase)


def random_register(
    n_pop: int,
    elements: Sequence[str],
    rng: np.random.Generator,
    labelled: Sequence[str] = (),
    n_phase: int = N_PHASE,
    conc: float = 1.0,
) -> C.ProgramModel:
    """Dirichlet(``conc``) rows for every free map; the echoes exact shifts."""
    m = C.random_program_model(n_phase * n_pop, elements, rng, labelled, conc=conc)
    return set_echoes(m, n_pop, n_phase)


def n_free_params(model: C.ProgramModel) -> int:
    return model.n_params(echo_names(model))


__all__ = [
    "N_PHASE",
    "echo_names",
    "echo_shift",
    "is_register",
    "n_free_params",
    "prism",
    "random_register",
    "register_start",
    "set_echoes",
    "shift",
]
