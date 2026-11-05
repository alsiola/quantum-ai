import math

import torch

from qmem.gates import CNOT, I2, H, X, Y, Z, controlled, rx, ry, rz
from qmem.linalg import dagger, embed, kron

ATOL = 1e-12


def test_rotations_are_unitary_and_match_exponentials():
    for f, p in ((rx, X), (ry, Y), (rz, Z)):
        for theta in (0.0, 0.3, -1.7, math.pi):
            u = f(theta)
            assert torch.allclose(u @ dagger(u), I2, atol=ATOL)
            expected = torch.linalg.matrix_exp(-0.5j * theta * p)
            assert torch.allclose(u, expected, atol=ATOL)


def test_rotation_batching():
    theta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    u = ry(theta)
    assert u.shape == (3, 2, 2)
    assert torch.allclose(u[1], ry(0.2), atol=ATOL)


def test_embed_identity_order_matches_kron():
    a, b = rx(0.4), rz(1.3)
    assert torch.allclose(embed(kron(a, b), (0, 1), 2), kron(a, b), atol=ATOL)
    assert torch.allclose(embed(a, (1,), 3), kron(I2, a, I2), atol=ATOL)


def test_embed_reversed_cnot():
    # CNOT with control qubit 1 and target qubit 0 equals (H⊗H) CNOT (H⊗H).
    hh = kron(H, H)
    assert torch.allclose(embed(CNOT, (1, 0), 2), hh @ CNOT @ hh, atol=ATOL)


def test_embed_nonadjacent_controlled():
    # controlled-X from qubit 0 to qubit 2 in 3 qubits, checked on basis states.
    u = embed(controlled(X), (0, 2), 3)
    for idx in range(8):
        bits = [(idx >> (2 - k)) & 1 for k in range(3)]
        if bits[0]:
            bits[2] ^= 1
        out = bits[0] * 4 + bits[1] * 2 + bits[2]
        assert u[out, idx] == 1
