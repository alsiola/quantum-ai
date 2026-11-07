"""Unconstrained quantum instrument (HQMM) on the Stiefel manifold.

The Kraus operators ``K[y, e]`` (``Y`` outcomes, ``E`` Kraus operators per
outcome, memory dimension ``D``) are stacked into one ``(D Y E) x D`` matrix
``V`` with ``V^dagger V = I``, which is exactly instrument completeness
``sum_{y,e} K^dagger K = I``. ``V`` is the orthonormal factor of the QR
decomposition of an unconstrained complex matrix ``W`` (stored as two real
tensors), so every ``W`` of full column rank gives a valid instrument and the
optimiser moves freely.

``V`` is ordered like the first ``D`` columns of a unitary on memory ⊗ output ⊗
env (rows ``(i, y, e)``), so this is architecture A with the circuit replaced
by an arbitrary unitary: the same ``Instrument``, the same teacher-forcing and
sampling code (``QuantumMemoryModel`` is subclassed, only ``instrument()`` is
replaced), and the same physics tests apply. ``E = 2**n_env``: ``n_env = 0``
gives one Kraus operator per outcome (pure memory, the uMPS model class);
``n_env > 0`` allows mixed memory.

The memory starts in ``|0><0|``. That is no loss of generality among pure
starts: conjugating every ``K`` by a unitary keeps ``V`` on the manifold.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..instrument import Instrument
from ..linalg import DTYPE, REAL_DTYPE
from ..model import QuantumMemoryModel
from ..trainability import INIT_MODES_BASELINE, symbol_probs


def isometry_from_matrix(w: torch.Tensor) -> torch.Tensor:
    """Orthonormal ``Q`` of the reduced QR of ``w`` (tall), with ``diag(R) > 0``."""
    q, r = torch.linalg.qr(w, mode="reduced")
    d = torch.diagonal(r)
    phase = d / d.abs()
    return q * phase.conj()[None, :]


class StiefelInstrumentModel(QuantumMemoryModel):
    family = "hqmm"

    def __init__(
        self,
        n_memory: int,
        n_output: int,
        n_env: int = 0,
        seed: int = 0,
        init_scale: float = 1.0,
        init: str = "random",
        init_probs: Any = "uniform",
    ):
        super().__init__(n_memory, n_output, blocks=[], n_env=n_env, initial="zero")
        gen = torch.Generator().manual_seed(seed)
        d, dy, de = 2**n_memory, 2**n_output, 2**n_env
        rows = d * dy * de
        w_re = init_scale * torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE)
        w_im = init_scale * torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE)
        if init == "identity":
            # K[y, 0] = sqrt(p_y) I plus noise of total size init_scale per column
            # (qmem.trainability), so the start is near the memoryless E_y = p_y I.
            p = symbol_probs(init_probs, dy)
            base = torch.zeros(d, dy, de, d, dtype=REAL_DTYPE)  # (i, y, e, j)
            for y in range(dy):
                base[:, y, 0, :] = torch.sqrt(p[y]) * torch.eye(d, dtype=REAL_DTYPE)
            noise = 1 / math.sqrt(2 * rows)
            w_re = base.reshape(rows, d) + noise * w_re
            w_im = noise * w_im
        elif init != "random":
            raise ValueError(f"unknown init {init!r}; choose from {INIT_MODES_BASELINE}")
        self.init = init
        self.w_re = nn.Parameter(w_re)
        self.w_im = nn.Parameter(w_im)

    def isometry(self) -> torch.Tensor:
        return isometry_from_matrix(torch.complex(self.w_re, self.w_im))

    def unitary(self) -> torch.Tensor:
        """A unitary dilation: ``V`` in the ancilla-``|0,0>`` columns, completed by QR.

        Only the instrument is used by the model; this exists so the circuit-level
        physics tests (unitarity, ``kraus_from_unitary``) apply to the HQMM too.
        """
        v = self.isometry()
        n, d = v.shape
        full = torch.linalg.qr(torch.cat([v, torch.eye(n, dtype=DTYPE)], 1), mode="complete").Q
        # The first d columns of ``full`` equal v up to phases; the rest span its complement.
        complement = full[:, d:]
        stride = n // d  # ancilla-|0,0> columns sit at j * (2**n_output * 2**n_env)
        u = torch.zeros(n, n, dtype=DTYPE)
        cols = torch.arange(d) * stride
        rest = torch.tensor([c for c in range(n) if c % stride != 0], dtype=torch.long)
        u[:, cols] = v
        u[:, rest] = complement
        return u

    def instrument(self) -> Instrument:
        d, dy, de = self.memory_dim, self.n_symbols, 2**self.n_env
        v = self.isometry().reshape(d, dy, de, d)  # (i, y, e, j)
        return Instrument(v.permute(1, 2, 0, 3))

    # -- bookkeeping (same keys as the other baselines) -------------------------
    def n_manifold_dim(self) -> int:
        """Real dimension of the complex Stiefel manifold of ``V``."""
        rows, d = self.w_re.shape
        return 2 * rows * d - d * d

    def state_dim(self) -> int:
        return self.memory_dim

    def state_kind(self) -> str:
        return "pure D-dim quantum state" if self.n_env == 0 else "D x D density matrix"

    def flops_per_step(self) -> int:
        d, e = self.memory_dim, 2**self.n_env
        return 4 * (2 * e * d**3 + self.n_symbols * d**2)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_symbols": self.n_symbols,
            "n_parameters": self.n_parameters(),
            "n_manifold_dim": self.n_manifold_dim(),
            "state_dim": self.state_dim(),
            "state_kind": self.state_kind(),
            "flops_per_step": self.flops_per_step(),
            "n_kraus_per_outcome": 2**self.n_env,
        }


class GeneratorInstrumentModel(QuantumMemoryModel):
    """HQMM restricted to ``n_generators`` real parameters (Phase 4b, issue #13).

    ``V(theta) = exp(i sum_k theta_k H_k) V_0``: ``V_0`` is a Haar-random isometry
    (the same distribution as the full HQMM's start) and the ``H_k`` are fixed random
    Hermitian generators on the ``D Y E``-dimensional row space (GUE, unit Frobenius
    norm), both drawn from ``seed``. ``V`` stays on the Stiefel manifold for every
    ``theta``, so every parameter value is a valid instrument, ordered as in
    ``StiefelInstrumentModel``. ``theta`` starts at 0, so the start *is* a random full
    HQMM; training moves only along a ``K``-dimensional submanifold of instruments.

    The generators have no circuit structure (no locality, no qubit layout), so at
    ``K`` equal to A's parameter count this separates "the instrument class at A's
    parameter count" from "A's circuit".
    """

    family = "hqmm_gen"

    def __init__(
        self,
        n_memory: int,
        n_output: int,
        n_generators: int,
        n_env: int = 0,
        seed: int = 0,
    ):
        super().__init__(n_memory, n_output, blocks=[], n_env=n_env, initial="zero")
        gen = torch.Generator().manual_seed(seed)
        d, dy, de = 2**n_memory, 2**n_output, 2**n_env
        rows = d * dy * de
        w = torch.complex(
            torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE),
            torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE),
        )
        self.register_buffer("v0", isometry_from_matrix(w))
        a = torch.complex(
            torch.randn(n_generators, rows, rows, generator=gen, dtype=REAL_DTYPE),
            torch.randn(n_generators, rows, rows, generator=gen, dtype=REAL_DTYPE),
        )
        h = a + a.conj().transpose(-1, -2)
        h = h / torch.linalg.matrix_norm(h)[:, None, None]
        self.register_buffer("generators", h)
        self.theta = nn.Parameter(torch.zeros(n_generators, dtype=REAL_DTYPE))

    def isometry(self) -> torch.Tensor:
        h = torch.einsum("k,kij->ij", self.theta.to(DTYPE), self.generators)
        return torch.linalg.matrix_exp(1j * h) @ self.v0

    def instrument(self) -> Instrument:
        d, dy, de = self.memory_dim, self.n_symbols, 2**self.n_env
        v = self.isometry().reshape(d, dy, de, d)
        return Instrument(v.permute(1, 2, 0, 3))

    def state_dim(self) -> int:
        return self.memory_dim

    def state_kind(self) -> str:
        return "pure D-dim quantum state" if self.n_env == 0 else "D x D density matrix"

    def flops_per_step(self) -> int:
        d, e = self.memory_dim, 2**self.n_env
        return 4 * (2 * e * d**3 + self.n_symbols * d**2)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_symbols": self.n_symbols,
            "n_parameters": self.n_parameters(),
            "state_dim": self.state_dim(),
            "state_kind": self.state_kind(),
            "flops_per_step": self.flops_per_step(),
            "n_kraus_per_outcome": 2**self.n_env,
        }
