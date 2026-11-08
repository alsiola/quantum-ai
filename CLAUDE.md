# Project rules for agents

Research campaign: measurement-induced memory in quantum-native sequence models
(brief: issue #2). Simulation only: never call quantum cloud services. The one exception is
`experiments/phase6.py` (IBM hardware run, approved by the owner on #16, free Open plan):
never submit a job outside an issue where the owner asks for one, always `--dry-run` first,
and keep within the plan's allowance. Credentials live in `.env` (gitignored; never commit it).

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
- Phase 4b falsification (one config per finding; 4–6 h each with 4 workers on a shared machine):
  `uv run python experiments/phase4b.py --config experiments/configs/phase4b_{a,b,c}.toml`
- Phase 5a hardware feasibility (optional `hardware` group: Qiskit, Aer, fake IBM backends;
  about 1 h with 4 workers on a shared machine):
  `uv run --group hardware python experiments/phase5a.py --config experiments/configs/phase5a.toml`.
  The main check does not need the group; `tests/test_hardware.py` skips without it.
- Phase 7 Stage 0 audit of the Phase 6 statistics (no QPU; about 15 min with 4 workers; the
  `--export-instruments` step, run once, needs the `hardware` group):
  `uv run python experiments/phase7_audit.py --config experiments/configs/phase7_audit.toml`
- Phase 7 Stage 2A q75 ablation controls (no QPU; `--export` once with the `hardware` group,
  then about 30 s with 2 workers; `--plan` prints the unsubmitted Stage 3 job):
  `uv run python experiments/phase7_ablation.py --config experiments/configs/phase7_ablation.toml`
- Phase 6 hardware run (PMD N = 3 on `ibm_kingston`; spends QPU time, see above):
  `uv run --group hardware python experiments/phase6.py --config experiments/configs/phase6.toml --dry-run`,
  then without `--dry-run`. `--from-run <dir>` resumes; a submitted job is re-fetched by id.
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
  `Instrument`), `witness` (independent numpy Hankel-witness estimator, its uncertainty, linear processes, Baum-Welch), `ablate_hw` (Phase 7 q75 destruction/dephasing controls: circuits built from
  Phase 6's, step pieces on the joint (q75, q59) memory, predictions; numpy except the circuit
  builders), `hardware` (Qiskit export of A as a dynamic circuit, exact branch
  simulation, noisy instruments from transpiled steps; needs the `hardware` group and is
  imported by nothing else).
- `tests/`: pytest. Every new physics feature gets a test against an analytic case or an
  independent brute-force computation.
- `docs/`: `architecture.md` (maths ↔ code); results notes `phase2-results.md`, `phase3d-strength.md`, `phase3e-retention.md`, `phase3f-grid.md` (cross-model grid, dimensional test), `phase4b-falsification.md` (verdicts on the candidate findings), `phase5a-hardware.md` (QPU feasibility, cost), `phase6-hardware-run.md` (the IBM run), `phase7-audit.md` (independent audit of the Phase 6 statistics), `phase7-ablation.md` (q75 ablation controls for Stage 3); `benchmarks.md` (the task suite); `report.md` (the campaign report; figures in `docs/figures/` from `experiments/report_figures.py`). Don't edit `docs/literature-review.md`
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
