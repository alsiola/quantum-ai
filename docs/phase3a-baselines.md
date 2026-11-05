# Phase 3a: baselines against architecture A

Issue #6, part of the campaign in issue #2. This phase builds the classical and
model-class baselines and compares them with architecture A (Phase 2,
`docs/phase2-results.md`) on the Phase 2 tasks, with the same data splits, the same
evaluation seeds, the same trainer and the same metric.

Everything below comes from one command:

```sh
uv run python experiments/phase3a.py --config experiments/configs/phase3a.toml
```

The generated tables are in [`results/summaries/phase3a/summary.md`](../results/summaries/phase3a/summary.md),
with `main.csv` (every evaluated cell), `comparison.csv`, `tuning.csv` (every tuning
candidate), `ngram.csv` and `summary.json`. The raw per-run records (curves,
`run_metadata`, parameters) go to `results/phase3a/<timestamp>/` (gitignored).
`--from-run <dir>` resumes an interrupted run.

## 1. What is compared, and how

**Tasks, data, metric.** The seven Phase 2 tasks, read from
`experiments/configs/phase2.toml`: i.i.d. binary and 4-symbol, order-1 binary and
4-symbol Markov, order-2 binary Markov, Even, Golden mean. The splits are the same
(`data.seed = 0`: 512 train, 256 validation, 512 test sequences of length 32), and so
is the metric: the *gap* is test NLL minus the source's own NLL on the same
sequences, in nats per symbol. A run *fails* if it crashed, diverged or has a gap of
at least 0.01.

**Architecture A** is re-run inside this experiment with its Phase 2 recipe and
seeds 0–9, through the same `qmem.experiment.run_one`. Its per-seed test gaps
reproduce `results/summaries/phase2/summary.json` exactly (100 runs, max |difference| = 0).

**Baselines** (`src/qmem/baselines/`). Every baseline has A's interface
(`step_log_probs`, `log_prob`, `predictive_probs`, `sample`, `n_parameters`,
`diagnostics`) plus `describe()`, so `qmem.train.train` and `qmem.train.evaluate`
handle it exactly as they handle A. The tests (`tests/test_baselines.py`) check
each family against an independent brute-force sequence probability. They also
check that the probabilities of all length-5 sequences sum to 1, and that 40 000
samples of length 3 match `exp(log_prob)` to within total variation 0.02. The
instrument physics tests also run on the HQMM (`tests/test_hqmm_instrument.py`).

| family | sizes | state | fitted by |
|---|---|---|---|
| HMM (edge-emitting, `L[y,i,j] = P(y, j \| i)`) | D = 2, 4, 8, 16 | belief over D states | Baum-Welch/EM; and gradient through the forward algorithm (softmax logits) |
| order-k Markov, add-α smoothing | k = 0…8, α ∈ {0.01, 0.1, 0.5, 1} | last k symbols | counts; k and α chosen on validation |
| GRU, tanh RNN | "matched": the largest hidden size with at most A's parameter count; "large": H ∈ {16, 64} | real H-vector | gradient |
| complex uMPS (Born machine) | χ = 2, 4, 8, 16 | complex χ-vector (left environment) | gradient |
| locally purified uMPS (LPS), κ = 2 | χ = 2, 4 | χ×χ PSD environment | gradient |
| HQMM: Kraus operators on the Stiefel manifold | D = 2, 4, 8, 16 with 1 Kraus/outcome; D = 2, 4 with 2 | pure (E = 1) or mixed (E = 2) D-dim quantum state | gradient |

The HMM is edge-emitting, the classical analogue of an instrument: a 2-state model
of this form represents the Even process exactly.

The uMPS uses the infinite-chain Born rule. `p(y | L) ∝ Tr(Σ_k A_yk L A_yk† Σ)`,
where `Σ` is the dominant fixed point of `X ↦ Σ_yk A_yk† X A_yk`. That fixed point
is found by power iteration, warm-started and with gradients through its last 20
steps. The conditionals are normalised explicitly, so they are exact for any `Σ`.
An unconverged fixed point (residual above 1e-8) is counted in `diagnostics`.

The HQMM stacks its Kraus operators as `V = QR(W)` with `V†V = I`. It subclasses
`QuantumMemoryModel`, so its teacher forcing, sampling and `Instrument` are A's own
code. Section 8 of `docs/architecture.md` shows two equivalences:

- the uMPS and the 1-Kraus HQMM are the same model class under different
  parameterisations;
- the HQMM is A with the circuit replaced by an arbitrary unitary.

**Optimiser budget.** Every gradient baseline uses A's trainer and budget: Adam,
batch 128, at most 1500 steps, validation every 10 steps, early stopping after 30
evaluations without a 1e-4 improvement, and best-validation restore. EM runs at
most 500 iterations, with early stopping on validation (patience 20) and
best-validation restore. Each EM seed is one random restart, and all 10 are
recorded.

**Tuning budget, fixed in the config and selected on validation only.** For each
task and each gradient cell (family and size), every candidate is trained with
tuning seeds 100 and 101, which are disjoint from the evaluation seeds. The
candidate with the lowest median best-validation NLL wins. Candidates:

- the learning rate, from {0.01, 0.03, 0.1};
- for the large RNNs, the learning rate × H ∈ {16, 64}.

Cells with state dimension 16 reuse the learning rate chosen at dimension 8 for
the same family and task, to save compute. The winner is then trained with
evaluation seeds 0–9, and only those runs enter the tables. In all, 798 tuning
runs were made. Every candidate's score is in `tuning.csv`.

**"The better of EM and gradient" for the HMM** is chosen per task and D by the
lower validation NLL of each method's validation-selected run. Both methods are in
the full table.

## 2. Comparison with A

Entries are the **median test gap over 10 seeds** (nats/symbol), with the number of
failed runs out of 10 in brackets. "(EM)" means the EM fit was the better HMM fit on
validation for that cell. The n-gram model is deterministic, so it has a single run.
All numbers are from `results/summaries/phase3a/summary.md`.

| model | iid_binary | iid_quaternary | markov1_binary | markov1_quaternary | markov2_binary | even | golden_mean |
|---|---|---|---|---|---|---|---|
| **A, 2 qubits (D=4)** | 0.0003 (0) | 0.0013 (1) | 0.0006 (0) | **0.0199 (7)** | 0.0016 (0) | 0.0011 (0) | 0.0007 (0) |
| A, 3 qubits (D=8) | – | – | – | 0.0343 (8) | 0.0059 (1) | 0.0039 (1) | – |
| HMM D=2 | 0.0000 (0) EM | 0.0004 (0) | 0.0005 (0) | 0.1146 (10) | 0.0018 (0) | 0.0003 (0) | 0.0001 (0) EM |
| HMM D=4 | -0.0000 (0) EM | 0.0008 (0) | -0.0000 (0) EM | 0.0005 (0) EM | 0.0012 (0) EM | 0.0002 (0) EM | 0.0001 (0) EM |
| HMM D=16 | -0.0000 (0) EM | 0.0002 (0) EM | 0.0003 (0) EM | 0.0014 (0) EM | 0.0008 (0) EM | 0.0003 (0) EM | 0.0005 (0) |
| uMPS χ=2 | 0.0000 (0) | 0.0009 (0) | 0.0002 (0) | 0.0157 (10) | 0.0014 (0) | 0.0000 (0) | 0.0001 (0) |
| uMPS χ=4 | 0.0002 (0) | 0.0031 (1) | 0.0001 (0) | 0.0012 (0) | 0.0011 (0) | 0.0003 (0) | 0.0001 (0) |
| uMPS χ=16 | 0.0001 (0) | 0.0170 (10) | 0.0003 (0) | 0.0080 (5) | 0.0008 (0) | 0.0005 (0) | 0.0005 (0) |
| LPS χ=4, κ=2 | 0.0002 (0) | 0.0020 (0) | 0.0002 (0) | 0.0011 (0) | 0.0008 (0) | 0.0004 (0) | 0.0001 (0) |
| HQMM D=2 | 0.0002 (0) | 0.0008 (0) | 0.0001 (0) | 0.0157 (10) | 0.0010 (0) | -0.0000 (0) | 0.0001 (0) |
| **HQMM D=4** (A's model class) | 0.0012 (0) | 0.0072 (1) | 0.0011 (0) | **0.0029 (0)** | 0.0014 (0) | 0.0004 (0) | 0.0004 (0) |
| HQMM D=4, 2 Kraus/outcome | 0.0004 (0) | 0.0011 (0) | 0.0003 (0) | 0.0017 (0) | 0.0009 (0) | 0.0005 (0) | 0.0003 (0) |
| HQMM D=16 | 0.0078 (2) | 0.0157 (10) | 0.0070 (0) | 0.0275 (10) | 0.0084 (2) | 0.0047 (0) | 0.0064 (0) |
| GRU, ≤ A's parameters | 0.0000 (0) | 0.0006 (0) | 0.0000 (0) | 0.1043 (10) | 0.0012 (0) | 0.0058 (0) | 0.0003 (1) |
| GRU, large | 0.0002 (0) | 0.0010 (0) | 0.0001 (0) | 0.0017 (0) | 0.0007 (0) | 0.0011 (0) | 0.0004 (0) |
| RNN, ≤ A's parameters | 0.0001 (0) | 0.0005 (0) | 0.0006 (0) | 0.1043 (10) | 0.0003 (0) | 0.0007 (0) | 0.0003 (0) |
| RNN, large | 0.0002 (0) | 0.0011 (0) | 0.0005 (0) | 0.0028 (0) | 0.0006 (0) | 0.0007 (1) | 0.0007 (0) |
| n-gram, k on validation | 0.0001 | 0.0001 | -0.0001 | 0.0003 | 0.0000 | 0.0085 | 0.0003 |

The summary also gives each cell's **validation-selected** run: the seed with the
lowest best-validation NLL, and its test gap. Taking that run removes most of A's
seed sensitivity. On `markov1_quaternary` A's selected run has a gap of 0.0009,
against 0.0005 for HMM D=4 and 0.0023 for HQMM D=4. Every model that succeeds on a
task then sits in the same band of 0.0000–0.003.

**What the comparison says.**

1. **On these tasks, A has no advantage over any baseline of the same state
   dimension.** At D = 4, the HMM (EM), the uMPS, the LPS, the 2-Kraus HQMM and the
   large GRU/RNN all match or beat A's median gap on every task, with no failures.
   The median gaps are within about 0.001 nats of each other wherever all of them
   succeed. That is expected: every task is a finite HMM with at most 4 states, so
   these sources do not separate the model classes. Separating them needs the
   Phase 3c targets from literature review §4, where a classical model needs more
   states than the quantum one.
2. **A's one weak task is an optimisation problem, not a capacity limit.** On
   `markov1_quaternary`, A fails 7 of 10 seeds. The unconstrained instrument of the
   *same model class* (HQMM, D = 4, one Kraus operator per outcome, pure memory)
   fails 0 of 10. So does the uMPS χ = 4, which is gauge-equivalent to that class.
   The 4-dimensional pure-state instrument can represent this source and is easy to
   train when parameterised directly. A's circuit parameterisation (21–26 angles,
   with `joint_depth = 2`) is what makes it hard to reach. This is the clearest
   result of this phase, and it bears on #7 and #9: "circuit structure" currently
   costs trainability and buys nothing measurable.
3. **Model-class baselines behave as their capacity predicts.**
   - Every D = 2 model fails `markov1_quaternary` (HMM, uMPS, HQMM, LPS; 10 of 10
     each). The 4-state source needs dimension 4.
   - Every D = 2 model fits the Even process; the EM HMM reaches a gap of −0.0001.
   - The order-k Markov model approaches the Even process slowly. Its gap falls with
     k (0.156, 0.128, 0.073, …, 0.0085 at k = 8) and never reaches the D = 2 models.
     `tests/test_baselines.py` checks the exact statement: the excess conditional
     entropy `h_k − h` is positive for every k up to 8, and strictly decreasing.
   - The k = 8 model is just under the 0.01 failure threshold at T = 32 (gap 0.0085).
     So "0 failures" for the n-gram on Even reflects how lenient the threshold is,
     not a fit. It is still 8–80× worse than every finite-state model of dimension
     2 or more.
4. **The parameter-matched RNNs are tiny**: a GRU with H = 1 and an RNN with H = 1–2
   (15–29 parameters, at most A's 21–26). They fit every binary task and fail the
   4-symbol Markov task 10 of 10 (gap 0.10). With A's parameter count, a classical
   RNN cannot represent that source; the "large" arm, at H = 16, fits everything.

## 3. Baselines that look weak, and why we think so

A weak baseline is suspect. These cells did worse than a smaller model of the same
family, which capacity alone cannot explain:

- **HQMM D = 8 and D = 16**, and **uMPS χ = 8 and 16** on `iid_quaternary`. Gaps are
  0.013–0.018, with 9–10 of 10 runs failing, on the easiest 4-symbol task, which
  D = 2 solves. HQMM D = 16 is also 5–20× worse than HQMM D = 2 on every task.
  Likely causes:
  - at large D, a random isometry gives a near-maximally mixed predictive
    distribution with small gradients;
  - D = 16 reuses the learning rate chosen at D = 8 and was not tuned itself;
  - early stopping ends runs on slow plateaus (mean 316–364 steps).

  The large-D quantum-like baselines should therefore be read as *undertrained*
  upper bounds, not as evidence that more dimension hurts. A fix to try before
  using them in later phases: initialise close to a product or identity
  instrument, and give the large cells their own tuning.
- **A at 3 qubits** is worse than at 2 on every task where both ran, as Phase 2
  already found. This is the same pattern as above, in the circuit
  parameterisation.
- **GRU matched on `golden_mean`**: one seed out of 10 failed (gap 0.17), a local
  optimum. It is recorded, not removed.

No baseline produced zero or tiny probabilities (`zero_prob_steps` and
`tiny_prob_steps` are zero everywhere). The uMPS fixed point failed to converge to
1e-8 in 8 forward calls in total, out of hundreds of thousands. These were in
`iid_binary/umps/chi8` and `markov2_binary/umps/chi16`. They are counted in
`diagnostics`, and the conditionals stay exactly normalised.

## 4. Costs

| model (D = 4, binary tasks) | parameters | state | multiply-adds / symbol | train s / run |
|---|---|---|---|---|
| A, 2 qubits | 21 | pure 4-dim quantum state | 640 | 28–61 |
| HMM D=4 | 36 (35 free) | belief over 4 states | 24 | 1–5 (EM), 6–20 (grad) |
| uMPS χ=4 | 72 | complex 4-vector | 1152 | 16–71 |
| LPS χ=4, κ=2 | 136 | 4×4 PSD | 2176 | 23–43 |
| HQMM D=4 | 64 | pure 4-dim state | 640 | 15–38 |
| GRU H=16 / H=64 | 994 / 13186 | real 16- / 64-vector | 896 / 12800 | 7–37 |
| n-gram, k=8 | 511 | last 8 symbols | 1 (lookup) | < 1 |

The per-cell parameter counts, flop counts, mean steps, training time and tuning
time are in `main.csv` and the per-task tables of `summary.md`.

- **Multiply-adds** are analytic counts per symbol in teacher forcing, with complex
  multiplications counted as 4 real ones.
- **Wall times were measured on a shared machine**. Other agents' experiments kept
  the load average between 7 and 60 on 8 cores. Absolute times are therefore
  inflated, by up to about 4× against an idle machine, and noisy; compare them only
  within this run.
- **Totals:** the run took about 175 min of wall time in two sittings (128 min for the resumed part; see below), and
  18.8 CPU-hours summed over the 2585 runs: 100 A runs, 798 tuning runs and 1687
  evaluation runs.

## 5. Where the matching with A is imperfect

- **A was not re-tuned in this phase.** It uses its Phase 2 recipe. That recipe was
  chosen partly on test gaps (Phase 2 §8, worry 4) and was not drawn from the lr grid
  the baselines get. The baselines had 6–18 tuning runs per cell; A had none here.
  This favours A slightly in principle, and it does not change the conclusion.
- **A has no D = 2 or D = 16 runs.** Phase 2 trained only 2 and 3 memory qubits, so
  comparisons at matched dimension are made at D = 4 (and at D = 8 on three tasks).
- **Parameter matching and state-dimension matching cannot both hold.** The matched
  RNNs have about A's parameter count but a 1–2 dimensional state. HMM D = 4 has 36
  parameters, and HQMM D = 4 has 64 real parameters (48 on the Stiefel manifold) against A's
  21. Each table row states its parameter count and state dimension.
- **uMPS and HQMM (E = 1) are the same model class.** They differ only in
  parameterisation: unconstrained with a fixed-point normalisation, against an
  isometry. Their agreement here is a consistency check, not two independent
  baselines.
- **D = 16 cells reuse the D = 8 learning rate.** This was done to save compute, and
  section 3 suspects it.
- **The HMM's better-of-two choice is made on validation per cell**, so the headline
  HMM row mixes EM and gradient fits. Both fits are in `main.csv`, and EM was the
  better one in most cells. The gradient fit gets A's optimiser budget (1500 steps,
  same early stopping). EM gets 500 iterations; its 10 restarts are the 10 seeds.
- **The run was interrupted once.** The first session hosting it ended after 501
  records, and the rest was resumed with `--from-run`. It is the same code throughout
  (`2f298ec`). The `dirty` flag is set in the records because the docs were
  uncommitted while it ran; no code was uncommitted.

## 6. Recommendations for later phases

- Use **HMM (EM, D = 2ⁿ), uMPS/HQMM (χ = D = 2ⁿ) and a GRU with H = 16** as the
  standing baselines. They are cheap and reliable at D ≤ 4, and fit every Phase 2
  task.
- A's advantage, if there is one, has to be shown on sources where a classical
  model needs more states than the quantum model (Phase 3c, #8). There, the HMM must
  have D = 2ⁿ states, and the HQMM gives the ceiling for A's model class.
- Before relying on baselines at D ≥ 8, fix their initialisation and tuning
  (section 3).
- The HQMM-against-A gap on `markov1_quaternary` gives a direct target for #7 and
  #9: a parameterisation of A that trains as reliably as the unconstrained
  instrument.

