# Project rules for agents

Research campaign: measurement-induced memory in quantum-native sequence models
(brief: issue #2). Simulation only: never call quantum cloud services.

## Setup and checks

- Setup (one line, needs `uv`): `uv sync`
- **Check, which must pass before every push to `main`:**
  `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- Proof of concept (must keep running in under a minute):
  `uv run python experiments/poc_memory.py`
- Phase 2 reproduction (about 20 min with 7 workers):
  `uv run python experiments/phase2.py --config experiments/configs/phase2.toml`
- Benchmark suite and its sanity runs (about 12 min with 7 workers; `--suite-only`: under a
  minute): `uv run python experiments/phase3c.py --config experiments/configs/phase3c.toml`
- Phase 3b ablations (A/B/C_k/D/hybrid; about an hour with 7 workers):
  `uv run python experiments/phase3b.py --config experiments/configs/phase3b.toml`
- Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Add one with
  `uv add <pkg>==<version>` (or `uv add --dev`) and commit both files.

## Layout

- `src/qmem/`: the package. `linalg` (dtypes, ordering, `embed`), `gates`, `circuits`
  (blocks → unitary), `instrument` (Kraus, channel, probabilities, conditional states),
  `model` (`QuantumMemoryModel`, `ModelConfig`, `build_model`), `states`, `data` (fixed
  splits), `records`, `tasks` (sources as labelled HMMs with exact optima; add Phase 3 tasks
  with `register_task`), `train` (`TrainConfig`, `train`, `evaluate`), `experiment`
  (multi-seed `RunSpec`/`run_many`/`aggregate`), `ablations` (Phase 3b variants B, C_k, D
  and hybrid on A's interface; `build_variant`; resource counts), `metrics` (operational
  information gain, disturbance, Banaszek trade-off for any `Instrument`), `analysis`
  (generation stats, interventions, Markov baselines, memory trajectories), `complexity`
  (causal states, `C_mu`, `C_q`, exact quantum models of a source), `benchmarks` (the Phase 3
  suite: literature numbers with sources, recall scoring, noise `eps` and strength floors),
  `diagnostics` (stationary memory, retention curves, reference purification for any
  `Instrument`).
- `tests/`: pytest. Every new physics feature gets a test against an analytic case or an
  independent brute-force computation.
- `docs/`: `architecture.md` (maths ↔ code); results notes `phase2-results.md`, `phase3d-strength.md`; `benchmarks.md` (the task suite). Don't edit `docs/literature-review.md`
  unless your issue is the literature review.
- `experiments/`: scripts, plus `experiments/configs/*.toml` (one config per run family).
- `results/`: raw outputs in `results/<experiment>/<timestamp>/` (gitignored); small curated
  summaries in `results/summaries/` (committed).
- `scratch/`: your scratch files (gitignored). Never commit scratch.

## Conventions

- dtype: `torch.complex128` for states/operators, `torch.float64` for parameters and
  probabilities (`qmem.DTYPE`, `qmem.REAL_DTYPE`). Don't mix in float32.
- Qubit order: memory, output ancillas, env ancillas; big-endian (qubit 0 most significant).
- Keep the four objects distinct: channel, outcome probabilities, conditional state,
  persistent memory. Keep teacher forcing (`log_prob`) and generation (`sample`) as separate
  code paths. Never substitute a classical hidden state for the quantum memory.
- No silent clamping of probabilities. Use `prob_floor` only deliberately; it is logged
  and counted in `model.diagnostics`.
- Seeding: no global RNG state. Pass explicit seeds (`ModelConfig.seed`, `sample(..., seed)`,
  `torch.Generator().manual_seed(s)`) and record them.
- Results: every saved result includes `qmem.records.run_metadata(config, seed)` (commit,
  dirty flag, environment, config, seed). Save with `save_json` into `new_run_dir(name)`.
  Experiments read their config from a committed TOML file; don't hard-code settings in scripts.
- Record unsuccessful runs too.
