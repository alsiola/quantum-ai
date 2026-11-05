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

from typing import Any

import torch
from torch import nn

from ..instrument import Instrument
from ..linalg import DTYPE, REAL_DTYPE
from ..model import QuantumMemoryModel


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
    ):
        super().__init__(n_memory, n_output, blocks=[], n_env=n_env, initial="zero")
        gen = torch.Generator().manual_seed(seed)
        d = 2**n_memory
        rows = d * 2**n_output * 2**n_env
        self.w_re = nn.Parameter(init_scale * torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE))
        self.w_im = nn.Parameter(init_scale * torch.randn(rows, d, generator=gen, dtype=REAL_DTYPE))

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
