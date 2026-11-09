# Project rules for agents

Research campaign: measurement-induced memory in quantum-native sequence models
(brief: issue #2). Simulation only: never call quantum cloud services. The exceptions are
`experiments/phase6.py` (IBM hardware run, approved by the owner on #16, free Open plan) and
`experiments/phase7_hw.py` (its pre-registered follow-up, #24; it submits only with the owner's
approving comment on #17 in its config): never submit a job outside an issue where the owner asks for one, always `--dry-run` first,
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
- Phase 7 Stage 2C spectators/crosstalk and control design (no QPU; about 30 s; the
  `--export-device` step, run once, needs the `hardware` group):
  `uv run python experiments/phase7_spectator.py --config experiments/configs/phase7_spectator.toml`
- Phase 7 Stage 2D classical readout/processing artefacts and the D1–D3 control designs (no
  QPU; about 5–10 min with 2 workers; `--export-qasm`, run once, needs the `hardware` group and
  uses `FakeKingston` only):
  `uv run python experiments/phase7_classical.py --config experiments/configs/phase7_classical.toml`
- Phase 7 Stage 2B q59 ancilla reset/leakage fits and B1–B4 control design (no QPU; about
  10 min with 2 workers; `--export` checks the QASM and needs the `hardware` group; `--plan`
  prints the unsubmitted Stage 3 job):
  `uv run python experiments/phase7_ancilla.py --config experiments/configs/phase7_ancilla.toml`
- Phase 7 Stage 1 drift and nonstationary-mixture nulls for the Phase 6 witness (no QPU; about
  50 min with 2 workers; `--from-run <dir>` resumes):
  `uv run python experiments/phase7_drift.py --config experiments/configs/phase7_drift.toml`
- Phase 7 Stage 3 pre-registered hardware follow-up (#24; q75 ablation on `ibm_kingston`):
  the local dry run, no network (about 40 min with 2 workers; the circuit and hash checks need
  the `hardware` group and are skipped without it):
  `uv run --group hardware python experiments/phase7_hw.py --config experiments/configs/phase7_hw.toml --dry-run-local`.
  Stage 4 only, after the owner approves on #17: `--dry-run` (live target, submits nothing),
  then `--submit`, which refuses unless `[ibm] approval_url` is set (`--stop-after-pilot` stops after the pilot gate for inspection); `--from-run <dir>` resumes.
- Phase 7 Stage 4b (#26): refit of q75 under mid-circuit measurement on #25's pilot (no QPU;
  `--export` once with the `hardware` group, then about 20 min with 2 workers):
  `uv run python experiments/phase7_refit.py --config experiments/configs/phase7_refit.toml`.
  The amended re-pilot and main jobs use `experiments/configs/phase7_hw_4b.toml` with
  `phase7_hw.py` (same modes as Stage 3/4); `--submit` also refuses unless the dated amendment
  (`docs/phase7-preregistration.md` §9) matches its `[amendment]` hash at its commit on `main`.
- Phase 7 Stage 4c (#27): gate (c) as a recorded diagnostic, the pilot's own q75 refit as the
  analysis's model; `experiments/configs/phase7_hw_4c.toml` with `phase7_hw.py` (same modes;
  `--dry-run-local` validates at both q75 decays, about 2.5 h with 2 workers). `--submit`
  refuses unless §9 and §10 both match their `[[amendment]]` hashes at their commits on `main`.
- Phase 8 Stage 0A (#29): independent re-analysis of Stage 4c's shots, fragility, drift,
  TOST (no QPU; about 20 s; `--freeze` wrote `results/summaries/phase8_stage0/provenance.json`
  once, and every run checks the frozen Phase 7 Stage 4/4b/4c files against it):
  `uv run python experiments/phase8_stage0.py --config experiments/configs/phase8_stage0.toml`
- Phase 8 Stage 0B (#30): the adversarial classical hidden-state/intervention model fitted
  jointly to every Stage 4c arm, and the Klein-tetrahedron test of the calibrated model (no
  QPU; about an hour with 2 workers on a shared machine):
  `uv run python experiments/phase8_classical.py --config experiments/configs/phase8_classical.toml`
- Phase 8 Stage 1 (#31): pair selection and the independent-pair replication design (no QPU,
  no IBM service, no submit path; `--export` once with the `hardware` group, FakeKingston only;
  `--dry-run-local` about 20 min with 2 workers; `--plan` prints the unsubmitted plan):
  `uv run python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --dry-run-local`
- Phase 8 Stage 2 (#32): coherence-specific test design on q75 (no QPU). `--part review`
  (fitted quantum model, CV at d = 4-6, EM convergence; about 2-3 h with 2 workers),
  `--part design` and `--part power --review-run <dir>` (about 1.5 h), all resumable with
  `--from-run <dir>`; `--export-qasm` needs the `hardware` group (`FakeKingston` only);
  `--plan` prints the unsubmitted Stage 2 job:
  `uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part review`
- Phase 8 Stage 3 (#33): whether q75's step channel can be certified non-entanglement-breaking
  (prepare-and-measure, Bell route, MDI game; power, budget and an unsubmitted plan; no QPU;
  about 5 s; `--export-qasm`, run once, needs the `hardware` group and uses `FakeKingston` only):
  `uv run python experiments/phase8_reference.py --config experiments/configs/phase8_reference.toml`
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
  `Instrument`), `witness` (independent numpy Hankel-witness estimator, its uncertainty, linear processes, Baum-Welch), `drift` (drift nulls as exact linear processes: across-shot mixtures, telegraph-modulated and within-shot-drifting 2-state HMMs; block and stratified bootstraps, segment statistics, stationarity and change-point tests, adversarial null search), `artefacts` (classical readout/processing artefacts, exact and planted into shots; pipeline guard rails; exact models of the D1–D3 controls), `ablate_hw` (Phase 7 q75 destruction/dephasing controls: circuits built from
  Phase 6's, step pieces on the joint (q75, q59) memory, predictions; numpy except the circuit
  builders), `ancilla` (numpy step simulator with a qutrit q59: state-dependent reset, leakage, non-QND and asymmetric readout; device fits; post-selected and alternating-ancilla statistics; B1–B4 QASM), `spectator` (numpy time-resolved step simulator with spectators, static ZZ, readout Stark shift and TLS; control circuits C1-C4), `refit` (Phase 7 Stage 4b: A2's joint q75/q59 record as a linear process; q75 decay under mid-circuit measurement, ML fit, bootstrap; numpy), `reanalysis` (Phase 8 Stage 0A: an independent numpy path for Stage 4c, sharing only the witness definition; Fieller, TOST and its power, fixed/random-effect pooling, vanish-rule power, provenance hashes), `classical` (Phase 8 Stage 0B: classical hidden-state models with one stochastic map per arm, their exact statistics and likelihood, a structured EM fit with SQUAREM, `ProgramModel` with one map per circuit element, the calibrated model in Klein-tetrahedron coordinates; numpy), `coherence` (Phase 8 Stage 2: a fitted quantum model with nine noise parameters, physical phase echoes, per-shot per-step randomised dephasings, arms as element programmes for `classical.ProgramModel`, Knee controls, benchmarks, power; numpy), `replication` (Phase 8 Stage 1: calibration-only pair selection, per-pair truth models from the step pieces, the per-pair rule with leave-one-job-out and TOST, the power engine, the arm x job rule; numpy), `reference` (Phase 8 Stage 3: the step channel's Choi state against n, Bell/P&M/MDI non-EB tests with SPAM and leakage thresholds, power, the reference-qubit rule; numpy, with lazy Qiskit circuit builders), `followup` (Phase 7 Stage 3: the pre-registered schedule, cost model, budget, calibration and submission guards, analysis pipeline and decision rule; numpy), `hardware` (Qiskit export of A as a dynamic circuit, exact branch
  simulation, noisy instruments from transpiled steps; needs the `hardware` group and is
  imported by nothing else).
- `tests/`: pytest. Every new physics feature gets a test against an analytic case or an
  independent brute-force computation.
- `docs/`: `architecture.md` (maths ↔ code); results notes `phase2-results.md`, `phase3d-strength.md`, `phase3e-retention.md`, `phase3f-grid.md` (cross-model grid, dimensional test), `phase4b-falsification.md` (verdicts on the candidate findings), `phase5a-hardware.md` (QPU feasibility, cost), `phase6-hardware-run.md` (the IBM run), `phase7-audit.md` (independent audit of the Phase 6 statistics), `phase7-classical.md` (classical artefacts, D1–D3 controls), `phase7-ablation.md` (q75 ablation controls for Stage 3), `phase7-ancilla.md` (q59 reset, leakage and residual memory; B1–B4 controls), `phase7-spectator.md` (spectators, crosstalk, replication design), `phase7-drift.md` (drift nulls, segment and SE policy), `phase7-preregistration.md` (the Stage 3 hardware follow-up, fixed before any data); `phase7-hardware.md` (Stage 4: the pilot and its gate; Stage 4b: the refit, the amended re-pilot and the outcome; Stage 4c: the pilot-fitted q75 model, the main jobs and the rung 4 outcome); `phase7-memory-confounders.md` (the Phase 7 synthesis and claim ladder); `phase8-stage0-audit.md` (Phase 8 Stage 0A: reproduction of Stage 4c, the fragility table, drift-conditional SEs, sham TOST margins); `phase8-claim-framework.md` (Phase 8 Stage 0B: the five-claim taxonomy, the adversarial classical model's fit, the coherence bar, the literature); `phase8-coherence.md` (Phase 8 Stage 2: #30's review points, the phase-scan and equivalent-dephasing design, the classical-d power study, go/no-go, prereg draft); `phase8-replication-prereg.md` (Phase 8 Stage 1: the pair selection, power study, budget options and the pre-registration draft of the replication); `phase8-reference.md` (Phase 8 Stage 3: non-EB certification feasibility of q75's channel, the reference choice, the capped unsubmitted plan); `benchmarks.md` (the task suite); `report.md` (the campaign report; figures in `docs/figures/` from `experiments/report_figures.py`). Don't edit `docs/literature-review.md`
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
