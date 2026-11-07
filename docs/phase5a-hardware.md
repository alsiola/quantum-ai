# Phase 5a: what it would take to run the best models on a QPU

Issue #14 (campaign #2, deliverable 8). **Simulation and desk study only**: local Qiskit
simulators and fake IBM backends; nothing was submitted to any quantum cloud service. The
generated tables, with every row, are in `results/summaries/phase5a/summary.md`, next to the
CSVs (`models`, `export`, `transpile`, `noise`, `thresholds`, `aer_check`, `cost`,
`experiment`, `witness`), `summary.json`, the plots `pmd_noise.png` and `recall_noise.png`, and
the exported circuits as OpenQASM 3 (`circuits/<model>_T4.qasm`).

Reproduce (needs the optional, pinned `hardware` group: qiskit 2.6.2, qiskit-aer 0.18.2,
qiskit-ibm-runtime 0.51.0 for the fake backends; about 15 min of training with 4 workers plus
45 min of analysis on a shared machine, most of it the noisy Aer cross-check and the witness):

```
uv run --group hardware python experiments/phase5a.py --config experiments/configs/phase5a.toml
```

`--from-run <dir>` reuses a run directory and `--stages train,export,transpile,noise,cost`
picks stages. The main check (`uv run pytest`) does not need the group:
`tests/test_hardware.py` skips without it.

## Verdict

**Feasible now.** Running a trained 1-qubit PMD model as a generator, and testing the
memory-dimension claim on its output, is a small experiment. Each symbol is 2 physical
qubits, 3 CZ gates, one mid-circuit measurement and one reset (about 5 µs). Under
FakeKingston's and FakeFez's calibrations the exact noisy model keeps a KL rate below the
best 2-state classical model at every N = 3–8, by 1.4× to 17×. A Hankel dimension witness
that is exactly zero for *every* 2-state HMM comes out at |z| ≈ 6.6–68 (N = 8 to 3) from
20,000 shots of 64 symbols, against |z| ≤ 1.1 for a 2-state HMM control. That is 13.5 s of
QPU time per model: about 22 USD at Pay-As-You-Go prices, or 2% of
the free Open plan's monthly allowance.

**Feasible, with care.** The n = 3 feed-forward retention model (5 qubits, 24 CZ per
symbol) needs dynamic circuits (classically conditioned `W_y`). It keeps teacher-forced recall
at 0.91–0.93 under the calibrations, against 0.94 noiseless, but loses it beyond about 12
distractors on FakeKingston. A device can only measure recall in its *own* samples, and there
it is 0.60/0.57 noisy and 0.68 noiseless, against the source's 0.95 and chance at 0.5.

**Not feasible now.**
- **Training on hardware.** Parameter shift needs 2,000–50,000 QPU hours (10–300 million
  USD); SPSA would still need about 9–14 h (50–80 thousand USD) per model.
- **Teacher-forced log-likelihood.** It needs post-selection on the device's own outcomes:
  about 10¹³ shots per test sequence of 32 symbols.
- **The n = 2 weak-η retention model and `markov1_quaternary`** (n = 2, general layout). At
  today's error rates they lose what made them interesting: recall falls to 0.72 / 0.52 and
  the gap rises from 0.002 to 0.037–0.043.
- **The coherent (deferred-measurement) feed-forward.** Routing makes it 27–75× more
  two-qubit gates than the dynamic form.

**Minimal hardware experiment to propose** (needs the owner's approval on #2; nothing has
been run): the six PMD models (N = 3–8) and `recall_n3` as generators, 20,000 shots of
T = 64 each on one Heron r2 device (ibm_kingston class), with no error mitigation. Measured:
the Hankel witness and the block KL D_L/L (device ‖ source) for L ≤ 8 (PMD), and generative
recall (retention). About 101 s of QPU time, about 160 USD at Pay-As-You-Go, or 17% of the
Open plan's 10 minutes. Details in section 6.

## 1. Models

The raw runs of #11 and #13, with their trained parameters, are gitignored and were gone. So
the script re-trains the **validation-selected run of each cell**: the same config, seed and
(tuned) learning rate, through the Phase 4b runner, which builds A exactly as
`qmem.experiment.run_one`. Every run reproduces its committed reference **bit for bit**
(validation NLL for PMD, test gap for the others; Δ = 0, `models.csv`).

| model | cell (source) | task | layout | n | qubits | params | test gap |
|---|---|---|---|---|---|---|---|
| pmd_N3 … pmd_N8 | #11 `A/D2` (selected seeds 8, 9, 7, 9, 1, 1) | `pmd_N3`–`pmd_N8` | general | 1 | 2 | 13 | 0.0003–0.0015 |
| mq_n2 | #13 (a) arm `A`, seed 2 | `markov1_quaternary` | general | 2 | 4 | 26 | 0.0019 |
| recall_n2_weak | #13 (b) arm `prior01` (weak η learned, η = 0.64), seed 2 | `marked_recall` τ = 4 | feed-forward, shared g | 2 | 4 | 41 | 0.0348 |
| recall_n3 | #13 (b) arm `n3_free` (#12 near-identity start, restarts), seed 2 | `marked_recall` τ = 4 | feed-forward, shared g | 3 | 5 | 61 | 0.0224 |

The PMD models are #11's standard-data cell. Its large-data cell fits better (long gap
≈ 8·10⁻⁵), but the issue asks for the result in `docs/phase3f-grid.md`, and its headline uses
both cells.

## 2. Circuit spec (`qmem.hardware`)

`export_circuit(model, T, feedforward)` writes the model as one Qiskit circuit for T tokens:

- **Registers in the project's order.** `mem` (n qubits), `out` (n_output), `env` (n_env);
  Qiskit qubit k is project qubit k.
- **One classical register per step,** `y0 … y{T-1}`. Output ancilla k is written to clbit
  `n_output−1−k`, so the register's integer value is the token (the project's big-endian `y`).
- **Per step, the trained gates one to one.** Memory HEA (RY/RZ, CNOT or CZ), `cry(2g)`
  couplings, and the joint HEA (general layout). Then a mid-circuit measurement of the output
  ancillas.
- **Feed-forward layout.** Each `W_y` follows, classically conditioned on the register: an
  `if/else` for two symbols, one `if` per outcome for four. `if_else` is supported more
  widely than `switch_case`; FakeNighthawk has no `switch_case`.
- **Reset between steps.** Every ancilla is reset before the next step. Env ancillas are
  reset unread, which is the partial trace. Only the `|0…0⟩` initial memory is exported.
- **Coherent alternative.** `feedforward="coherent"` gives the deferred-measurement form
  (multi-controlled `W_y` before the readout, no classical control).
- **Two more helpers.** `export_deferred` builds the measurement-free form with fresh
  ancillas per step. `export_step` builds one step with the reset in front, which is the
  repeating unit used below.

**Exactness** (`export.csv`). The outcome distribution of every token string and the
conditional memory state were compared with `qmem` (`log_prob`, `final_memory`) three ways:

1. **Branch simulation.** An independent exact simulator (`branch_simulate`) walks the
   Qiskit instructions with Qiskit's own gate matrices. It keeps one unnormalised density
   matrix per classical record, projects on measurement, applies reset as a channel and
   follows the classical control. It shares no code with `qmem`'s circuit or instrument.
2. **Deferred-measurement statevector** (Qiskit `Statevector`), up to 12 qubits.
3. **Aer** density-matrix simulation of the dynamic circuit, 100,000 shots.

For all 9 models and T = 1, 2, 3, 4, 6, 8 (binary) or T = 1–6 (four symbols: 4,096
strings): max |Δp| ≤ 4·10⁻¹⁶, max |Δρ| ≤ 4·10⁻¹², and the coherent and dynamic forms agree.
Aer's χ² p-values are 0.02–0.44.

**Tests** (`tests/test_hardware.py`, 29 tests, 2 s):
- export = `qmem` exactly, in six layouts (general n = 1, 4 symbols, feed-forward n = 2 and 3
  with CZ, 4-symbol feed-forward, an env ancilla), for T = 1–3;
- the deferred statevector; the circuit's structure; Aer sampling of the dynamic circuit;
- the noiseless instrument of a transpiled and routed step equals the model's;
- analytic noise cases: readout flips, a bad reset, a depolarising coupling
  `Φ_y → (1−p)Φ_y + (p/4) I`, the T1/T2 relaxation channel, idle relaxation during readout.

## 3. Transpilation

Fake backends need no account. FakeKingston is Heron r2 (156 qubits, heavy-hex, CZ);
FakeNighthawk is a square lattice (120 qubits). Settings: `optimization_level = 3`,
`seed_transpiler = 0`. Per symbol, from the T = 32 transpile (`transpile.csv` also has
T = 1 and 8):

| model | feed-forward | qubits | 2q/symbol | 1q/symbol | depth/symbol | meas, reset/symbol | step (µs) | T = 32: depth, 2q |
|---|---|---|---|---|---|---|---|---|
| pmd_N* (n = 1) | none | 2 | 3 | 12 | 21 | 1, 1 | 5.0 | 671, 96 |
| mq_n2 | none | 4 | 18 (16 on Nighthawk) | 51 | 50 | 2, 2 | 5.9 | 1608, 576 |
| recall_n2_weak | dynamic | 4 | 14 | 60 | 26 | 2, 2 | 10.1 | 832, 448 |
| recall_n2_weak | coherent | 4 | 376 (461) | 894 | 1427 | 2, 2 | 48 | 45677, 12030 |
| recall_n3 | dynamic | 5 | 24 | 74 | 28 | 2, 2 | 10.5 | 896, 768 |
| recall_n3 | coherent | 5–6 | 742 (794) | 1732 | 2584 | 2, 2 | 83 | 82671, 23737 |

How to read it:
- **Counts.** Two-qubit counts take the costlier branch of the classical control. The
  T-step counts are T times the step's, so the circuit grows linearly in T.
- **Step durations** are ASAP schedules with the backend's gate durations, its measurement
  (2.28 µs on Kingston, 1.56 µs on Fez) and reset (2.31 / 1.58 µs), and an assumed 1 µs
  feed-forward latency. No public IBM figure for that latency was found, so it is a config
  value (`feedforward_latency_s`).
- **Topology** barely matters for n ≤ 3. The heavy-hex and square-lattice counts are equal,
  or within 2 two-qubit gates per symbol (mq_n2).
- **The coherent feed-forward is the exception.** The multi-controlled `W_y` need all-to-all
  connectivity between the outputs and the memory, so routing multiplies its two-qubit count
  by 27–75. **Dynamic circuits are required**, and both backends support `if_else`.

## 4. Noise

**Method.** Every noisy number here is exact, with no shot noise:
- **One instrument per step.** With noise, each step is still an instrument on the memory,
  `Φ_y(ρ) = Tr_anc[step_y(ρ ⊗ ρ_anc)]`. The branch simulator builds it from the
  *transpiled* step on the backend's qubits, with the memory's input and output positions
  (routing can move it). Noise is added inside the simulation; the inputs are the basis
  operators |i⟩⟨j|.
- **Then `qmem` scores it.** The instrument is turned into Kraus operators (Choi
  decomposition) and wrapped as a `qmem` model (`InstrumentModel`). The KL rate,
  `evaluate`, the recall metrics and `sample` all apply unchanged.
- **Error model, from the backend's calibration** (`backend_noise`):
  - depolarising after every gate, with the reported average gate error (`p = e·d/(d−1)`);
    RZ is virtual and error-free;
  - a symmetric readout flip at the reported measurement error, with the post-measurement
    state the true one;
  - the same flip after a reset (an IBM reset is a measurement and a conditional X);
  - T1/T2 relaxation of every idling qubit for the measurement, reset and feed-forward
    durations.
- **Sweeps** (one error type at a time, on the FakeKingston transpiled step):
  - two-qubit depolarising p, with p₁ = p/10;
  - readout flip;
  - reset error;
  - idle T1 = T2;
  - every Kingston error times a factor.
- **Scoring.** PMD is scored as #11's long re-scoring (256 sequences of 1024 symbols, seed
  777). The noiseless rows reproduce #11's long gaps (pmd_N3 0.00041, identical).

### 4.1 PMD: the memory-dimension result

KL rate (nats/symbol; paired test NLL minus the source's, which estimates D(source ‖ model)):

| | N = 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| best 2-state classical model (#11) | 0.0137 | 0.0111 | 0.0068 | 0.0045 | 0.0029 | 0.0018 |
| unifilar 2-state block bound (#11) | 0.0076 | 0.0071 | 0.0036 | 0.0024 | 0.0016 | 0.0010 |
| 1-qubit A, noiseless | 0.0004 | 0.0004 | 0.0005 | 0.0008 | 0.0004 | 0.0010 |
| … FakeKingston calibration | 0.0011 | 0.0007 | 0.0004 | 0.0017 | 0.0018 | 0.0012 |
| … FakeFez calibration | 0.0016 | 0.0008 | 0.0006 | 0.0022 | 0.0021 | 0.0013 |

**The advantage over every 2-state classical model survives both calibrations.** The margin
is 12× at N = 3 and 17× at N = 5 on Kingston, falling to 1.4× at N = 7 and 8 on Fez.

Against the *unifilar* bound it survives at N = 3–6 on both, and fails at N = 7 (0.0018 and
0.0021 against 0.0016). At N = 8 the noiseless model already ties the
bound, as #11 found.

Noise sometimes *lowers* the rate slightly (N = 4, 5). A trained model is not the source,
and a little dephasing can move it closer.

**Error rates that keep it** (`thresholds.csv`; log-interpolated crossing of the best 2-state
classical model's rate, one error type at a time):

| error | N = 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| 2q depolarising p (p₁ = p/10) | 4.8% | 4.7% | 3.5% | 1.8% | 1.3% | 1.4% |
| readout flip | > 10% | > 10% | > 10% | > 10% | > 10% | > 10% |
| reset error | > 10% | > 10% | > 10% | 4.9% | 4.0% | 4.8% |
| idle T1 = T2 | < 10 µs | < 10 µs | < 10 µs | < 10 µs | 14 µs | < 10 µs |
| all Kingston errors × | > 3 | > 3 | > 3 | 2.1 | 1.4 | 1.5 |

The binding constraint is the two-qubit gate:
- **Depolarising parameter ≲ 1.3%** keeps the result at every N, which is an average CZ
  error of ≲ 1.0%. Kingston's CZ on these qubits is at 0.09%, Fez's at 0.21%.
- **Readout and reset errors matter little.** Readout corrupts only the record, not the
  memory; one bad reset per symbol enters like a gate error.
- **Against the unifilar bound** the depolarising thresholds are 0.8–3.5% (N = 3–7).

**Against T.** The KL rate over the first T symbols is roughly flat in T (T = 8 … 1024 in
`summary.md`). Each step's noise is the same channel, so the error per symbol settles; it
does not accumulate.

### 4.2 `markov1_quaternary` (n = 2, general layout)

- **Gap under the calibrations: 0.0019 → 0.037 (Kingston) and 0.043 (Fez).** That is 4× the
  campaign's 0.01 failure threshold.
- **What it needs.** Keeping it under 0.01 needs every error at ≲ 0.3× Kingston's. Taking
  one error type at a time, that is depolarising p ≲ 0.25%, readout ≲ 1% or T2 ≳ 300 µs.
- **Why.** It has 18 CZ per symbol, and idle dephasing during the two measurements and
  resets dominates: the T2 of its Kingston qubits is 89 µs.
- **So** the parameter-efficiency finding (#13 (a)) is a simulation result. Today's devices
  would not show it.

### 4.3 Retention (`marked_recall`, τ = 4)

| model | noise | teacher-forced recall | soft recall | generative recall |
|---|---|---|---|---|
| source | – | 0.95 | – | 0.949 |
| recall_n2_weak | noiseless | 0.901 | 0.620 | 0.612 |
| | FakeKingston | 0.720 | 0.551 | 0.559 |
| | FakeFez | 0.523 | 0.527 | 0.538 |
| recall_n3 | noiseless | 0.944 | 0.765 | 0.681 |
| | FakeKingston | 0.913 | 0.613 | 0.597 |
| | FakeFez | 0.925 | 0.588 | 0.569 |

The three recall measures:
- **Teacher-forced recall** is #13's metric: argmax accuracy at recall positions of the
  test split.
- **Soft recall** is the mean teacher-forced probability of the right value.
- **Generative recall** is the share of recalls in the model's *own* sampled sequences
  (4,096 × 64) that repeat their cue. It is the only one of the three a device can measure:
  teacher forcing would need post-selection (section 5).

The trained models are right but unsure: argmax accuracy is 0.94, while the probability on
the right value averages 0.77. Their sampled recall is therefore 0.68 even without noise.

What the noise does:
- **n = 3 keeps its argmax recall** to about 12 distractors under Kingston's calibration
  (0.89 at 8–11), then falls to 0.53–0.69. On Fez the qubits differ and it holds to 16.
- **The n = 2 weak-η model loses recall** beyond 2 distractors on Kingston, and almost
  entirely on Fez (0.52).
- **The two n = 2 effects are related.** The weak measurement that helped n = 2 in
  simulation protects the cue by disturbing it less. Idle dephasing at T2 ≈ 77–153 µs
  disturbs it anyway.

**Readout and reset errors give identical results for both retention models.** In the
feed-forward layout the readout is only `cry(2g)` from the memory into a fresh ancilla.
An ancilla left in |1⟩ by a bad reset gives the same Kraus operators, up to sign, as a
flipped record, and `W_y` follows the record either way. The general layout (mq_n2, PMD)
shows the two differ, as expected.

**Measurement-and-reset latency against T1/T2.**
- **Per symbol** the memory idles about 5.6 µs (measurement 2.3 µs + reset 2.3 µs +
  feed-forward ~1 µs) of a 10.5 µs step.
- **To lag 40** (#13's retention horizon) the cue must survive about 420 µs. That is 2.7–5.5
  times the T2 of the qubits the transpiler picked (median 77–153 µs), so it survives only in a
  protected, measurement-insensitive direction.
- **The idle sweep agrees.** recall_n3 keeps ≥ 0.94 teacher-forced recall only for
  T1 = T2 ≳ 300 µs; recall_n2_weak is still at 0.86 (0.90
  noiseless) with 1 ms.
- **Not modelled:** measurement-induced dephasing of spectator qubits (on top of idle
  T2) and leakage. Both would make this worse.

### 4.4 Cross-check against Aer's own noise model

`AerSimulator.from_backend(FakeKingston)` (Aer's model: thermal relaxation within gates, the
backend's asymmetric readout, no reset or idle error) was run on the transpiled circuits (T = 6 for pmd_N3, T = 3 for recall_n3, so that
n_symbols^T ≤ 4,096), 100,000 shots, without shot branching. TV distances of the T-symbol distribution:

| model | T | strings | vs noiseless | vs calibrated (this study) | vs calibrated, no idle relaxation | sampling scale ½√(K/N) |
|---|---|---|---|---|---|---|
| pmd_N3 | 6 | 64 | 0.0141 | 0.0187 | **0.0115** | 0.0126 |
| recall_n3 | 3 | 4,096 | 0.0667 | 0.0725 | **0.0601** | 0.101 |

For pmd_N3, Aer's model matches this study's calibrated instrument *without* idle
relaxation, within sampling (the scale column is an upper bound on the expected TV). Aer's
`from_backend` model applies no relaxation during measurement and reset, so it is the more
optimistic of the two, and the instruments used above are the more pessimistic. For
recall_n3 the 4,096-string distribution is too sparse at 100,000 shots to separate the
variants, but the ordering is the same.

**An Aer pitfall found on the way.** In qiskit-aer 0.18.2, `shot_branching_enable=True`
mis-applies readout errors to mid-circuit measurements. With a 10% flip, three successive
measure-and-reset rounds give flip rates of 0, 0 and 0.25 (density matrix) or 0, 0.125 and 0
(statevector), instead of 0.1 each. Without branching the rates are 0.096–0.102. It is exact
and 20–30× faster on noiseless circuits, so the exactness checks use it; the noisy
cross-check does not.

## 5. Shots, QPU time and cost

**Pricing**, read 2026-11-07 (https://www.ibm.com/quantum/pricing):
- **Pay-As-You-Go**: "Starts at $96 USD / minute", "Billed per second of usage".
- **Flex**: from 72 USD/min; **Premium**: from 48 USD/min.
- **Open (free)**: "up to 10 minutes of runtime per month". The plans overview says "per
  28-day rolling window" (https://quantum.cloud.ibm.com/docs/en/guides/plans-overview).

**Usage model**
(https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time, read 2026-11-07): about
2 s per sub-job plus (rep_delay + circuit length) per shot, with a default rep_delay of
250 µs. With the step durations above, a shot of T = 64 is 573 µs (PMD) to 921 µs
(recall_n3).

**What can be estimated, and with how many shots** (`cost.csv`):

- **Generated statistics: cheap.**
  - *Block KL.* D_L/L(device ‖ source) to ±0.001 nats/symbol takes 1,000–100,000 blocks of
    length L ≤ 10 for PMD (plug-in estimator, bias and 2σ each within half the precision).
    A shot of 64 symbols gives 8 disjoint blocks of 8 (6 of 10), so about 4,000 shots at
    L = 8 and 17,000 at L = 10.
  - *Block distributions.* TV ±0.01 of the 8-symbol block distribution takes about 640,000
    blocks.
  - *Recall.* Generative recall to ±0.01 (95%) takes about 1,300–1,500 sequences.
- **The block KL is infinite for mq_n2.** The source forbids two transitions the trained
  model still allows (p > 0 where the source has 0), so D(model ‖ source) = ∞ beyond L = 1.
  On hardware this direction is the wrong measure there; TV or the source-side cross-entropy
  is the usable one.
- **Teacher-forced log-likelihood: infeasible.** Conditioning the memory on a data token
  requires the measurement to *produce* that token, so p(y₁…y_T) can only be estimated by
  post-selection. To ±0.01 nats that takes (1−p)/(p·0.01²) shots per sequence, with
  p = e^{−T·NLL}:
  - **PMD:** 1.3·10⁵ (T = 4), 3.5·10⁸ (T = 16), 1.2·10¹³ (T = 32);
  - **retention:** 10²⁰ at T = 32.

  The NLL and gap metrics of the campaign are simulation-only.
- **Training on hardware: impractical.**
  - *Parameter shift.* Two evaluations per rotation occurrence and four per CRY, and every
    parameter occurs in every step, so a 64-step circuit needs 1,793 (PMD), 3,585 (mq_n2) to
    8,193 (recall_n3) circuits per gradient. At 10,000 shots each and the trained runs' own
    step counts (480–3,000), that is 1,800–51,000 QPU hours: 10–300 million USD at
    Pay-As-You-Go.
  - *SPSA.* It needs 3 circuits per step whatever the parameter count. At the same step
    counts that is about 9 h (≈ 50,000 USD) for pmd_N3 and 14 h (≈ 83,000 USD) for
    recall_n3, and SPSA usually needs more steps.
  - *Loss.* Either way the loss must be sample-based (block KL, MMD, recall), not the NLL,
    for the reason above.
  - **Train in simulation, run on hardware.**

## 6. The minimal hardware experiment (for the owner's approval; not run)

**What.** Use the trained pmd_N3 … pmd_N8 and recall_n3, exported as above
(`results/summaries/phase5a/circuits/`). Transpile them for one Heron r2 device
(ibm_kingston class; FakeKingston's layout) and generate 20,000 shots of T = 64 each. No
error mitigation and no training.

**Measure.**
1. **Hankel dimension witness (PMD).** Take g(k) = P(1 0ᵏ 1) from positions after 16 steps
   and the normalised 3 × 3 determinants det[g(j+a+b)].
   - Every 2-state HMM, from any initial state, gives exactly 0, because g obeys an order-2
     recurrence (#11, `tests/test_merge_bound.py`). A significantly non-zero value excludes
     every 2-state classical model of the device's output.
   - Predicted, from simulated 20,000-shot experiments with bootstrap errors over shots
     (`witness.csv`),
     the largest |z| over the four offsets.

     | | N = 3 | 4 | 5 | 6 | 7 | 8 |
     |---|---|---|---|---|---|---|
     | source (PMD itself) | 80 | 63 | 40 | 19 | 10 | 6.2 |
     | 1-qubit A, noiseless | 78 | 58 | 36 | 19 | 9.7 | 9.9 |
     | … FakeKingston calibration | 68 | 46 | 31 | 17 | 6.9 | 6.6 |
     | … FakeFez calibration | 64 | 44 | 29 | 17 | 8.0 | 6.2 |
     | control: a random 2-state HMM | 1.1 | | | | | |

     The witness shrinks with N, for the source itself as for the models, but even N = 8 is
     over 6σ at 20,000 shots. The exact witness of the noisy models agrees with the
     sampled one (`witness.csv`).
2. **Block KL** D_L/L(device ‖ source), L ≤ 8, against the simulated noisy prediction
   (PMD). With 160,000 blocks of 8 per model it is precise to about ±0.001.
3. **Generative recall (recall_n3).** Predicted 0.60 (Kingston calibration) against 0.5 for
   no memory and 0.95 for the source; its standard error at 20,000 shots is about 0.0013.

**Cost.** 13.5 s of QPU time per PMD model and 20.4 s for recall_n3, including 2 s of
sub-job overhead each: about **101 s in all**, which is **≈ 160 USD** at Pay-As-You-Go or 17%
of the Open plan's 10 minutes. N = 3 alone (the clearest case) is 13.5 s, about 22 USD.

**What it would and would not show.**
- **It would show** that a superconducting device running the trained 1-qubit instrument
  produces a process no 2-state classical model can produce exactly, and how close that
  process is to PMD.
- **It would not certify** that the device uses a single qubit of memory. Imperfect
  ancilla resets, leakage and spectator qubits can carry information between steps, so the
  device's effective memory is larger than one qubit.
- **It would not test** the KL comparison in #11's direction D(source ‖ model), which needs
  the likelihood.

## 7. Limits of this study

- **The calibrations are the fake backends' snapshots,** not today's devices.
  qiskit-ibm-runtime 0.51.0 says FakeNighthawk's error values are not typical, so it was
  used for topology only.
- **The noise model is a standard simplification:** gate depolarising at the reported error,
  symmetric readout, reset error equal to readout error, and idle T1/T2 during measurement,
  reset and control. It has no crosstalk, leakage, measurement-induced spectator dephasing,
  non-Markovian or coherent errors.
- **The feed-forward latency (1 µs) is an assumption.**
- **Prices are list prices** on 2026-11-07. Queue time is not billed and is not estimated.
