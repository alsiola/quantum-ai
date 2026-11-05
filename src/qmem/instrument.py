"""Quantum instruments on the memory register.

An instrument is a set of Kraus operators ``K[y, e]`` (outcome ``y``, unread
index ``e``) acting on the memory, stored as a tensor of shape
``(n_outcomes, n_kraus, d, d)``. Outcome ``y`` maps ``rho`` to the CP,
trace-nonincreasing ``sum_e K[y,e] rho K[y,e]^dagger``; completeness is
``sum_{y,e} K[y,e]^dagger K[y,e] = I``.

The four objects the brief asks us to keep apart map to:

1. unconditioned channel   -> ``Instrument.channel(rho)``
2. outcome probabilities   -> ``Instrument.probabilities(rho)``
3. conditional state       -> ``Instrument.condition(rho, y)`` (and the
   unnormalised ``Instrument.branch(rho, y)``)
4. persistent memory       -> ``qmem.model.Memory`` (carried by the model)

All methods accept ``rho`` of shape ``(..., d, d)``.
"""

from __future__ import annotations

import torch

from .linalg import eye, trace


def kraus_from_unitary(
    u: torch.Tensor, n_memory: int, n_output: int, n_env: int = 0
) -> torch.Tensor:
    """Kraus operators ``K[y, e] = (I ⊗ <y, e|) U (I ⊗ |0, 0>)``.

    The system is ordered ``memory, output ancillas, env ancillas`` (memory
    qubits most significant). Output ancillas are read in the computational
    basis (``y`` is the big-endian integer of their bits); env ancillas are
    traced out (``e`` indexes their basis). Ancillas start in ``|0>``.
    """
    dm, dy, de = 2**n_memory, 2**n_output, 2**n_env
    if u.shape != (dm * dy * de, dm * dy * de):
        raise ValueError("unitary size does not match the register sizes")
    k = u.reshape(dm, dy, de, dm, dy, de)[:, :, :, :, 0, 0]  # (i, y, e, j)
    return k.permute(1, 2, 0, 3)  # (y, e, i, j)


class Instrument:
    """A quantum instrument given by its Kraus tensor ``(n_outcomes, n_kraus, d, d)``."""

    def __init__(self, kraus: torch.Tensor):
        if kraus.dim() != 4 or kraus.shape[-1] != kraus.shape[-2]:
            raise ValueError("kraus must have shape (n_outcomes, n_kraus, d, d)")
        self.kraus = kraus

    @classmethod
    def from_unitary(
        cls, u: torch.Tensor, n_memory: int, n_output: int, n_env: int = 0
    ) -> Instrument:
        return cls(kraus_from_unitary(u, n_memory, n_output, n_env))

    @property
    def n_outcomes(self) -> int:
        return self.kraus.shape[0]

    @property
    def n_kraus(self) -> int:
        """Kraus operators per outcome."""
        return self.kraus.shape[1]

    @property
    def dim(self) -> int:
        return self.kraus.shape[-1]

    # -- structure -------------------------------------------------------------
    def effects(self) -> torch.Tensor:
        """POVM effects ``M_y = sum_e K^dagger K``, shape ``(n_outcomes, d, d)``."""
        return torch.einsum("yeji,yejk->yik", self.kraus.conj(), self.kraus)

    def completeness_defect(self) -> torch.Tensor:
        """``max |sum_y M_y - I|`` (zero for a valid instrument)."""
        return (self.effects().sum(0) - eye(self.dim)).abs().max()

    # -- (1) unconditioned channel ---------------------------------------------
    def channel(self, rho: torch.Tensor) -> torch.Tensor:
        """``E(rho) = sum_{y,e} K rho K^dagger`` (outcome discarded)."""
        return torch.einsum("yeij,...jk,yelk->...il", self.kraus, rho, self.kraus.conj())

    # -- (2) outcome probabilities ---------------------------------------------
    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        """``p(y | rho) = Tr(M_y rho)``, real tensor of shape ``(..., n_outcomes)``."""
        return torch.einsum("yij,...ji->...y", self.effects(), rho).real

    # -- (3) conditional post-measurement state ---------------------------------
    def branches(self, rho: torch.Tensor) -> torch.Tensor:
        """Unnormalised post-measurement states for every outcome, ``(..., Y, d, d)``."""
        return torch.einsum("yeij,...jk,yelk->...yil", self.kraus, rho, self.kraus.conj())

    def branch(self, rho: torch.Tensor, y: torch.Tensor | int) -> torch.Tensor:
        """Unnormalised ``sum_e K[y,e] rho K[y,e]^dagger`` for outcomes ``y`` of shape ``(...)``.

        Its trace is ``p(y | rho)``.
        """
        y = torch.as_tensor(y, dtype=torch.long).expand(rho.shape[:-2])
        k = self.kraus[y]  # (..., E, d, d)
        return torch.einsum("...eij,...jk,...elk->...il", k, rho, k.conj())

    def condition(
        self, rho: torch.Tensor, y: torch.Tensor | int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(K_y rho K_y^dagger / p(y|rho), p(y|rho))``.

        No safeguard: if ``p(y|rho) == 0`` the state is NaN. The sequence model
        handles that case explicitly (see ``QuantumMemoryModel``).
        """
        unnorm = self.branch(rho, y)
        p = trace(unnorm).real
        return unnorm / p[..., None, None], p
