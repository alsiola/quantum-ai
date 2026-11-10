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

**Z_n (#35).** For the incommensurate tier's ``r<k>`` / ``rm<k>`` echoes (``Rz(+-k rad)``,
``qmem.coherence.element_phase``) no finite cyclic group is exact. ``n_phase`` may be any
n; each echo of ideal phase ``phi`` is the *nearest* shift, ``round(phi n / 2 pi) mod n``
(exact for ``e<k>`` when 8 divides n; ``rm<k>`` is ``r<k>``'s inverse, so INV(1, -1) is the
identity). ``free`` names echoes left as free stochastic maps (the relaxation), and
``best_permutation`` projects a fitted map pair onto the strongest finite-permutation
adversary of #32's Lemma: ``e(1) = sigma``, ``e(-1) = sigma^-1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import classical as C

N_PHASE = 8


def echo_angle(name: str) -> float | None:
    """The ideal phase (rad) of an echo element (``None`` for any other element)."""
    from . import coherence as Q

    return Q.element_phase(name)


def echo_shift(name: str, n_phase: int = N_PHASE) -> int | None:
    """The register shift of an echo element: ``round(phi n / 2 pi)`` (half up), not reduced
    mod n (``e<k>`` at n = 8: ``k``). ``None`` for any other element."""
    phi = echo_angle(name)
    if phi is None:
        return None
    return math.floor(phi * n_phase / (2 * math.pi) + 0.5 + 1e-9)


def shift_error(name: str, n_phase: int = N_PHASE) -> float:
    """``|phi - 2 pi shift / n|`` (rad): how far the nearest shift is from the echo."""
    return abs(echo_angle(name) - 2 * math.pi * echo_shift(name, n_phase) / n_phase)


def shift(k: int, n_pop: int, n_phase: int = N_PHASE) -> np.ndarray:
    """(d, d) row-stochastic permutation ``(theta, p) -> (theta + k mod n, p)``."""
    d = n_phase * n_pop
    out = np.zeros((d, d))
    for p in range(n_pop):
        for t in range(n_phase):
            out[p * n_phase + t, p * n_phase + (t + k) % n_phase] = 1.0
    return out


def echo_names(model: C.ProgramModel, free: Sequence[str] = ()) -> list[str]:
    """The model's echo elements but ``free`` (the fixed permutations: pass as ``frozen``)."""
    return sorted(k for k in model.maps if echo_shift(k) is not None and k not in free)


def set_echoes(
    model: C.ProgramModel,
    n_pop: int,
    n_phase: int = N_PHASE,
    names: Sequence[str] = (),
    free: Sequence[str] = (),
) -> C.ProgramModel:
    """Every echo map of ``model`` (and ``names``) but ``free`` set to its nearest shift (in
    place)."""
    for k in {*echo_names(model, free), *names} - set(free):
        model.maps[k] = shift(echo_shift(k, n_phase), n_pop, n_phase)
    return model


def is_register(
    model: C.ProgramModel, n_pop: int, n_phase: int = N_PHASE, free: Sequence[str] = ()
) -> bool:
    return all(
        np.array_equal(model.maps[k], shift(echo_shift(k, n_phase), n_pop, n_phase))
        for k in echo_names(model, free)
    )


def is_permutation(m: np.ndarray) -> bool:
    return bool(np.all((m == 0) | (m == 1)) and np.all(m.sum(0) == 1) and np.all(m.sum(1) == 1))


def permutation_matrix(sigma: Sequence[int]) -> np.ndarray:
    """Row-stochastic ``P[i, sigma[i]] = 1`` (state ``i`` goes to ``sigma[i]``)."""
    d = len(sigma)
    out = np.zeros((d, d))
    out[np.arange(d), np.asarray(sigma)] = 1.0
    return out


def assignment(score: np.ndarray) -> np.ndarray:
    """``sigma`` maximising ``sum_i score[i, sigma[i]]`` over permutations (the Hungarian
    algorithm, O(d^3); ``-inf`` entries allowed if a finite assignment exists)."""
    a = np.asarray(score, dtype=float)
    d = a.shape[0]
    finite = np.isfinite(a)
    big = (np.abs(a[finite]).max() if finite.any() else 1.0) * (d + 1) * 4 + 1.0
    cost = np.where(finite, -a, big)
    inf = math.inf
    u = np.zeros(d + 1)
    v = np.zeros(d + 1)
    p = np.zeros(d + 1, dtype=np.int64)  # p[j]: row matched to column j (1-based; 0 none)
    way = np.zeros(d + 1, dtype=np.int64)
    for i in range(1, d + 1):
        p[0] = i
        j0 = 0
        minv = np.full(d + 1, inf)
        used = np.zeros(d + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, 0
            for j in range(1, d + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(d + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    sigma = np.empty(d, dtype=np.int64)
    for j in range(1, d + 1):
        sigma[p[j] - 1] = j - 1
    return sigma


def best_permutation(
    fwd: np.ndarray, inv: np.ndarray | None = None, counts: bool = False
) -> np.ndarray:
    """The permutation ``sigma`` nearest a fitted pair of maps ``fwd`` (e(1)) and ``inv``
    (e(-1)) in log-likelihood: maximises ``sum_i log fwd[i, s(i)] + log inv[s(i), i]``
    (``counts``: the inputs are expected transition counts, scored linearly instead)."""
    f = np.asarray(fwd, dtype=float)
    g = np.zeros_like(f) if inv is None else np.asarray(inv, dtype=float).T
    if counts:
        return assignment(f + g)
    with np.errstate(divide="ignore"):
        return assignment(np.log(f) + (0.0 if inv is None else np.log(g)))


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
    free: Sequence[str] = (),
) -> C.ProgramModel:
    """A start carrying the quantum model's structure (``qmem.coherence.polytope_start`` on
    the prism), with the echoes but ``free`` set to their nearest shifts (``free`` echoes keep
    the polytope's projection of the quantum map)."""
    from . import coherence as Q

    r = 1.0 / math.cos(math.pi / n_phase) if radius is None else radius
    u = prism(n_pop, r, height, n_phase)
    m = Q.polytope_start(maps, u, elements, labelled, eps)
    return set_echoes(m, n_pop, n_phase, free=free)


def random_register(
    n_pop: int,
    elements: Sequence[str],
    rng: np.random.Generator,
    labelled: Sequence[str] = (),
    n_phase: int = N_PHASE,
    conc: float = 1.0,
    free: Sequence[str] = (),
) -> C.ProgramModel:
    """Dirichlet(``conc``) rows for every free map; the echoes but ``free`` nearest shifts."""
    m = C.random_program_model(n_phase * n_pop, elements, rng, labelled, conc=conc)
    return set_echoes(m, n_pop, n_phase, free=free)


def n_free_params(model: C.ProgramModel, free: Sequence[str] = ()) -> int:
    return model.n_params(echo_names(model, free))


__all__ = [
    "N_PHASE",
    "assignment",
    "best_permutation",
    "echo_angle",
    "echo_names",
    "echo_shift",
    "is_permutation",
    "is_register",
    "n_free_params",
    "permutation_matrix",
    "prism",
    "random_register",
    "register_start",
    "set_echoes",
    "shift",
    "shift_error",
]
