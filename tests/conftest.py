import math

import pytest
import torch

from qmem import ModelConfig, build_model

# Configurations exercised by the structural tests: binary and 4-symbol
# alphabets, traced-out env ancillas, joint layers, trainable strength.
CONFIGS = {
    "bin_2m": ModelConfig(n_memory=2, n_output=1, memory_depth=2, strength=0.7, seed=1),
    "bin_3m_env": ModelConfig(
        n_memory=3, n_output=1, n_env=1, env_strength=0.4, memory_depth=2, seed=2
    ),
    "quad_2m": ModelConfig(n_memory=2, n_output=2, memory_depth=2, strength=1.1, seed=3),
    "quad_3m_joint": ModelConfig(
        n_memory=3, n_output=2, memory_depth=1, joint_depth=2, pattern="ring", seed=4
    ),
    "bin_4m_env_joint": ModelConfig(
        n_memory=4,
        n_output=1,
        n_env=1,
        env_strength=1.0,
        memory_depth=2,
        joint_depth=1,
        entangler="cz",
        pattern="all",
        rotations=("rx", "ry", "rz"),
        seed=5,
    ),
    "bin_2m_projective": ModelConfig(n_memory=2, n_output=1, strength=math.pi / 2, seed=6),
}


@pytest.fixture(params=sorted(CONFIGS))
def model(request):
    return build_model(CONFIGS[request.param])


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)
