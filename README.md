# Quantum AI

Can quantum measurement backaction serve as the memory-update mechanism of an
autoregressive sequence model? In this project, emitting a symbol and updating
a persistent quantum memory are the same operation: a trainable quantum
instrument measures a fresh ancilla coupled to the memory, the outcome is the
symbol, and the memory keeps the outcome-conditioned state.

This is a simulation-first research campaign (see issue #2). **The findings, negative
results, limitations and every reproduction command are in the
[research report](docs/report.md).** The `qmem` package
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

## Phase 2: training and memory results

```sh
uv run python experiments/phase2.py --config experiments/configs/phase2.toml
```

It trains the model over 10 seeds on sources with exactly known optima (i.i.d.,
Markov, Even and Golden-mean processes), runs the variants and the memory analyses,
and writes `results/summaries/phase2/`. It takes about 20 minutes with 7 worker
processes. The results and the recommended training recipe are in
[docs/phase2-results.md](docs/phase2-results.md).

## Phase 3b: quantum ablations

```sh
uv run python experiments/phase3b.py --config experiments/configs/phase3b.toml
```

It trains ablations of the model on the Phase 2 tasks, 10 seeds each: an
outcome-averaged control (B), memory reset every k tokens (C_k), separate readout
and update (D: unitary, dissipative, matched and larger), and a hybrid with
classical feedback. It writes `results/summaries/phase3b/` in about an hour with
7 workers. The variants, their resource counts and the A-versus-D result are in
[docs/phase3b-ablations.md](docs/phase3b-ablations.md).

## Phase 3c: memory benchmark suite

```sh
uv run python experiments/phase3c.py --config experiments/configs/phase3c.toml
```

It describes every benchmark task exactly (optimum, entropy rate, causal states,
`C_mu`, `C_q`, quantum dimension), checks the numbers taken from the literature, and
trains the model on a few of the tasks as a sanity check. It writes
`results/summaries/phase3c/` in about 12 minutes with 7 workers (`--suite-only`
skips training). The tasks, their sources and assumptions are in
[docs/benchmarks.md](docs/benchmarks.md).

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
