# Quantum AI

Can quantum measurement backaction serve as the memory-update mechanism of an
autoregressive sequence model? In this project, emitting a symbol and updating
a persistent quantum memory are the same operation: a trainable quantum
instrument measures a fresh ancilla coupled to the memory, the outcome is the
symbol, and the memory keeps the outcome-conditioned state.

This is a simulation-first research campaign (see issue #2). The `qmem` package
is a small, differentiable density-matrix simulator (PyTorch, complex128) for
2–4 memory qubits and 1–2 output ancillas.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Run the tests and checks

```sh
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Proof of concept

```sh
uv run python experiments/poc_memory.py
```

It builds a 2-memory-qubit, 1-ancilla model, shows that the next-symbol
distribution depends on past outcomes, and runs a few gradient steps on a
sticky Markov chain. It takes a few seconds on a laptop CPU.

## Minimal use

```python
from qmem import ModelConfig, build_model

model = build_model(ModelConfig(n_memory=2, n_output=1, strength=0.8, seed=0))
seqs = model.sample(n=128, length=16, seed=1)  # generation: sample and update
nll = -model.log_prob(seqs).mean()  # teacher forcing: condition on tokens
nll.backward()
```

See [docs/architecture.md](docs/architecture.md) for the maths, the code map and
design decisions.
