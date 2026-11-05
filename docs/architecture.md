# Architecture: measurement-conditioned quantum memory

This note describes the model that Phase 1b implements in `src/qmem/`, how
the maths maps onto the code, and why the simulation stack was chosen. The
research brief is issue #2; this work is issue #3.

## 1. The model

### Registers

| Register | Qubits | Lifetime |
| --- | --- | --- |
| memory `M` | `n_memory` (2–4 now) | persists across all steps |
| output ancillas `A` | `n_output` (1 → binary, 2 → 4 symbols) | fresh `|0>` each step, measured |
| env ancillas `E` | `n_env` (0+) | fresh `|0>` each step, traced out (never read) |

The joint system is ordered `M, A, E`, big-endian (qubit 0 is the most
significant bit of an index). Alphabet size is `|Y| = 2^n_output`.

### One step: a quantum instrument

At step `t` the memory is `rho_t`. A parameterised unitary `U(theta)` acts on
`rho_t ⊗ |0><0|_A ⊗ |0><0|_E`; `A` is measured in the computational basis,
giving `y`; `E` is discarded. The Kraus operators on the memory are

```
K_{y,e} = (I_M ⊗ <y|_A ⊗ <e|_E) U (I_M ⊗ |0>_A ⊗ |0>_E)
```

so outcome `y` acts by the CP, trace-nonincreasing map

```
Phi_y(rho) = sum_e K_{y,e} rho K_{y,e}^dagger
```

with `2^n_env` Kraus operators per outcome. Completeness

```
sum_{y,e} K_{y,e}^dagger K_{y,e} = I
```

holds because the columns of `U` with the ancillas in `|0>` are orthonormal.
The POVM effects are `M_y = sum_e K_{y,e}^dagger K_{y,e}`.

From the instrument we get four distinct objects (brief §4):

1. **Unconditioned channel** `E(rho) = sum_y Phi_y(rho)`: the memory when the outcome is
   discarded. Trace preserving.
2. **Outcome probabilities** `p(y | rho) = Tr Phi_y(rho) = Tr(M_y rho)`.
3. **Conditional post-measurement state** `rho_y = Phi_y(rho) / p(y | rho)`.
4. **Persistent memory**: the `rho_y` that is carried to the next step. The
   prediction (`p(y|rho)`) and the memory update (`rho -> rho_y`) come from the
   same operation.

By construction `sum_y p(y|rho) rho_y = E(rho)`.

### Sequences

For tokens `y_1..y_T`, `rho_0` the initial memory and `rho_t` the memory after
conditioning on `y_1..y_t`:

```
p(y_1..y_T) = prod_t p(y_t | rho_{t-1})
            = Tr( Phi_{y_T} ∘ ... ∘ Phi_{y_1}(rho_0) )
```

The second form (unnormalised Kraus products) is the independent check used in
the tests, along with a third: by the deferred measurement principle, the
joint distribution equals that of measuring `T` separate ancilla registers at
the end of one big unitary `U_T ... U_1` on `M ⊗ A_1E_1 ⊗ ... ⊗ A_TE_T`.

## 2. Measurement strength

The default circuit separates "what the memory does" from "how strongly it is
read":

```
U = J(theta_J) · C(g) · V(theta_V)
```

* `V(theta_V)`: hardware-efficient layers on the memory only (memory dynamics,
  and the basis in which the next coupling reads it).
* `C(g)`: for each output ancilla `k`, a controlled-`RY(2g)` from memory qubit
  `k mod n_memory` to the ancilla. Env ancillas get the same coupling from
  memory qubit `(n_output + j) mod n_memory` with `env_strength`.
* `J(theta_J)`: optional hardware-efficient layers on everything
  (`joint_depth`, default 0).

For one control qubit with the ancilla in `|0>`, `C(g)` gives

```
K_0 = |0><0| + cos g |1><1|,      K_1 = sin g |1><1|
```

* `g = 0`: `K_0 = I`, `K_1 = 0`. No information and no backaction.
* `g = pi/2`: a CNOT, i.e. a projective Z measurement of the control.
* In between: on `|+>`, `p(1) = sin^2(g)/2`; the unconditioned channel multiplies
  the coherence `rho_01` by `cos g`; after outcome 0 the state is
  `∝ |0> + cos g |1>`.

These cases are tested exactly (`tests/test_instrument.py`).

**Caveat:** with `joint_depth > 0` the readout passes through a general circuit,
so `g` no longer bounds the information extracted per step (Phase 2 found
several seeds learning `g ≈ 0` with a sharp instrument). `joint_depth = 0`
keeps `g` meaningful but can't fit even an i.i.d. source. Use the
feed-forward layout below for strength sweeps, and the operational metrics
(section 2.3) to compare any layouts.

### 2.1 The strength-faithful (feed-forward) layout

`feedforward_depth > 0` with `joint_depth = 0` (issue #9) gives

```
U = W · C(g) · V,      W = sum_y W_y(theta_y) ⊗ |y><y|_A ⊗ I_E
K_{y,e} = W_y · D_{y,e}(g) · V
```

where `D_{y,e}(g) = (I ⊗ <y,e|) C(g) (I ⊗ |0,0>)` is the weak-measurement
Kraus operator of section 2 and each `W_y` is its own `feedforward_depth`-layer
hardware-efficient unitary on the memory (`OutcomeConditionedBlock`).

**Effect invariance.** `W` is block-diagonal in the ancilla basis, so
`(I ⊗ <y,e|) W = W_y (I ⊗ <y,e|)`. Hence `K_{y,e} = W_y K^0_{y,e}`, where `K^0` are
the Kraus operators without `W`, and

```
E_y = sum_e K_{y,e}^† K_{y,e} = sum_e K^0_{y,e}^† W_y^† W_y K^0_{y,e} = V^† (sum_e D_{y,e}^† D_{y,e}) V
```

for every choice of `W`. So `p(y | rho) = Tr(E_y rho)`, and with it everything
one step's readout can reveal about the memory, is fixed by `g` and `V` alone.
`W_y` only steers the post-measurement memory `W_y D_y V rho V^† D_y^† W_y^† / p`,
and so what later steps will see. Tests: `tests/test_strength.py`
(`K_y = W_y K^0_y` exactly; effects unchanged when the `W` parameters are
redrawn; the block equals measure-then-apply-`W_y`).

**Limits** (tested). At `g = 0`, `E_0 = I` and the other effects vanish: no
information, and the update `rho -> W_0 V rho V^† W_0^†` is the same unitary
whatever the memory (only `y = 0` ever occurs). At `g = pi/2`, every `E_y` is a
projector: `V^† (|b><b| ⊗ I) V` on the coupled memory qubits, a projective
measurement in the basis `V` chooses.

**Probability cap.** Because `p(y|rho) <= lambda_max(E_y)`, a strength `g`
also caps the outcome probabilities: for a binary alphabet, `p(1|rho) <= sin^2 g`
for every memory state. A source that needs `p(1|h) > sin^2 g` after some
history `h` can't be represented at that `g` (`docs/phase3d-strength.md`; the
general form is the task floor `strength_lower_bound` in section 2.3).

**Hardware form.** Two equivalent circuits:
1. Mid-circuit measurement of the output ancillas, then a classically
   controlled `W_y` on the memory (dynamic circuits, feed-forward), then reset.
2. Before the measurement, the gates of `W_y` multi-controlled on the output
   ancillas being in `|y>` (for one output ancilla: `W_0` on `|0>`, `W_1` on
   `|1>`; equivalently `W_0`, then `W_1 W_0^†` controlled on the ancilla).
   Because the block is diagonal in the ancilla basis it commutes with the
   readout, so this is the deferred-measurement form of option 1.

**Not a classical hidden state.** `W_y` is selected by the one outcome just
read and acts once. Nothing classical persists to the next step: all memory of
the past is still carried by `rho`. The model is the same kind of object as
before (a time-homogeneous instrument with a fixed Kraus set); only the
parameterisation of the Kraus operators has changed.

### 2.2 Generalising the strength knob

With `n_output` output ancillas, the coupling reads memory qubits
`0..n_output-1`, one controlled-`RY(2g_k)` each. The `D_y` factorise:
`D_y = D_{y_1}(g_1) ⊗ D_{y_2}(g_2) ⊗ ...`. **Decision:** sweeps use one global
strength, `strength_shared = true`, a single `g` shared by every output coupling
(`ControlledRotationBlock(..., shared=True)`, one parameter). That `g` is the
"measurement strength" scalar a sweep sets. Per-coupling strengths remain
available (`strength_shared = false`, the default, as in Phase 2), but a sweep
over a vector of strengths has no single axis to report. Env couplings
(`env_strength`) are separate: they are decoherence, not readout.

Because `g` is only meaningful in this layout, results are reported in
operational units too. For `D_y(g)` with shared `g`, the operational strength
`eta` of section 2.3 is `sin^2 g` for one coupling (binary alphabet) and
`(2 sin^2 g + sin^4 g)/3` for two couplings (4 symbols); the TV gain is
`sin^2 g` and `1 - cos^4 g`.

### 2.3 Operational information gain and disturbance

`qmem.metrics` computes these for any `Instrument` (any Kraus tensor, so also
`joint_depth > 0` models and an unconstrained HQMM), with `d` the memory
dimension, `Y` the number of outcomes and `E_y` the effects.

*Information gain.* All three depend on the effects only, so they are blind to
any unitary applied after the readout.

* `tv_gain = max_{rho,sigma} TV(p(.|rho), p(.|sigma))`. Since
  `TV = max_S [p(S|rho) - p(S|sigma)]` over outcome subsets `S`, it equals
  `max_S [lambda_max(E_S) - lambda_min(E_S)]` with `E_S = sum_{y in S} E_y` (for
  `Y = 2`, `lambda_max(E_0) - lambda_min(E_0)`, which is half the spread of the
  eigenvalues of `E_0 - E_1`).
* Estimation fidelity `G` (Banaszek 2001): the mean fidelity between a
  Haar-random pure input `psi` and the best guess `phi_y` from the outcome. Using
  `∫ |psi><psi|^{⊗2} dpsi = (I + SWAP)/(d(d+1))`,
  `∫ |<phi|psi>|^2 <psi|E|psi> dpsi = (Tr E + <phi|E|phi>)/(d(d+1))`, so
  `G = (d + sum_y lambda_max(E_y)) / (d(d+1))`.
* **Operational strength** `eta = (sum_y lambda_max(E_y) - 1) / (min(Y, d) - 1)`,
  `G` rescaled to `[0, 1]`. Since `1 = lambda_max(sum E_y) <= sum lambda_max(E_y) <= min(Y, d)`,
  `eta = 0` exactly when every `E_y ∝ I`, and `eta = 1` when the effects are as
  sharp as `Y` outcomes on `d` levels allow (rank-1 projectors if `Y = d`). This is
  the scalar to report in place of `g`.
* `mutual_information(inst, rho)`: `I(R:Y)` between the outcome and a
  reference `R` purifying `rho`,
  `S(rho) - sum_y p_y S(sqrt(rho) E_y sqrt(rho)/p_y)` in nats (the information
  gain of Buscemi, Hayashi and Horodecki, PRL 100, 210504, 2008). It is evaluated at the
  stationary state of the unconditioned channel (`stationary_state`, a
  least-squares solve of `E(rho) = rho` with `Tr rho = 1`).

*Disturbance.* This is correction-invariant: before comparing, it applies
the best unitary `U_y` for each outcome, so feed-forward `K_y -> W_y K_y` is
not counted as disturbance.

* `F_e = max_{U_y} sum_{y,e} |Tr(U_y^† K_{y,e})|^2 / d^2`, the entanglement fidelity
  of the corrected channel. With one Kraus operator per outcome it is
  `sum_y ||K_y||_1^2 / d^2` (polar decomposition). With several, a monotone
  minorise-maximise iteration over polar factors is used (`best_correction`).
* Operation fidelity `F = (d F_e + 1)/(d + 1)`, the mean fidelity between
  output and input over Haar-random pure inputs (Horodecki, Horodecki and
  Horodecki, PRA 60, 1888, 1999; Nielsen, Phys. Lett. A 303, 249, 2002). This
  is Banaszek's `F`, and `disturbance = 1 - F`. The uncorrected value
  (`U_y = I`) is reported too.

*Trade-off.* For any instrument on `d` levels (Banaszek 2001, PRL 86, 1366):

```
sqrt(F - 1/(d+1)) <= sqrt(G - 1/(d+1)) + sqrt((d-1)(2/(d+1) - G))
```

`banaszek_bound(G, d)` returns the largest `F`. The bound allows any number of
outcomes, so with a binary readout on `d = 4` it is not tight. The weak family
`D_y(g)` on a qubit gives `G = (3 + sin^2 g)/6` and `F = (2 + cos g)/3`. It meets the
bound only at `g = 0` and `g = pi/2`, and lies below it in between, so a weak Z measurement is not an optimal
operation. Fuchs and Peres (PRA 53, 2038, 1996) give the corresponding
trade-off in information-theoretic terms for two non-orthogonal states. We use
Banaszek's fidelity form because it applies to any instrument without
choosing an ensemble.

| Weak family `D_y(g)`, one coupling | closed form |
| --- | --- |
| `tv_gain`, `eta` | `sin^2 g` |
| `G` (qubit / 2-qubit memory) | `(3 + sin^2 g)/6`, `(5 + sin^2 g)/20` |
| `F_e` (corrected = uncorrected, `D_y >= 0`) | `(1 + cos g)/2` |
| `F` (qubit / 2-qubit memory) | `(2 + cos g)/3`, `(3 + 2 cos g)/5` |
| `I(R:Y)` at `I/2` (qubit) | `ln 2 - p_0 h(1/(1 + cos^2 g))`, `p_0 = (1 + cos^2 g)/2` |

## 3. Teacher forcing vs generation vs postselection

* **Teacher forcing** (`log_prob`, `step_log_probs`, `predictive_probs`): the
  token comes from data; we evaluate `p(y_t | rho_{t-1})` and then condition the
  memory on that observed `y_t`. This is the computation of a *probability
  under the model*, used for the likelihood. It is not a physical procedure.
  On hardware, reproducing the "observed" branch would mean postselecting on
  the measurement outcome, which succeeds with probability `p(y_1..y_T)`, which
  is exponentially small in `T`. Any hardware cost estimate must account
  for this; the simulator does not hide it.
* **Generation** (`sample`): outcomes are drawn from `p(.|rho)` and the memory
  is updated on the drawn outcome. This is what a physical run of the
  instrument does (no postselection).

The two are separate code paths in `QuantumMemoryModel`, so a change to one
cannot silently alter the other; tests check the sampler's empirical
frequencies against teacher-forced exact probabilities.

## 4. Mapping to code

| Concept | Code |
| --- | --- |
| dtypes, qubit ordering, `embed` | `qmem/linalg.py` |
| rotations, CNOT/CZ, controlled gates | `qmem/gates.py` |
| `V`, `C(g)`, `W_y`, `J`, fixed unitaries; `U = B_last ... B_1` | `qmem/circuits.py`: `HardwareEfficientBlock`, `ControlledRotationBlock`, `OutcomeConditionedBlock`, `FixedUnitaryBlock`, `Circuit` |
| information gain, disturbance, Banaszek bound, stationary state | `qmem/metrics.py` (`operational_summary`; results: `docs/phase3d-strength.md`) |
| `K_{y,e}` from `U` | `qmem/instrument.py`: `kraus_from_unitary` |
| (1) channel, (2) probabilities, (3) conditional state | `Instrument.channel`, `Instrument.probabilities`, `Instrument.condition` / `branch` / `branches`, `Instrument.effects` |
| (4) persistent memory | `qmem.model.Memory` (batched `rho`, `(B, d, d)`) |
| sequence model | `QuantumMemoryModel`: `log_prob`, `step_log_probs`, `predictive_probs`, `final_memory`, `unconditioned_memory`, `sample` |
| default layout from a config | `ModelConfig`, `build_model` |
| initial states, validity checks, purity, entropy | `qmem/states.py` |
| toy data, fixed train/val/test splits | `qmem/data.py` |
| tasks (HMM and HQMM sources, exact optima, oracle log-probs) | `qmem/tasks.py` (suite: `docs/benchmarks.md`) |
| causal states, `C_mu`, `C_q(L)`, exact quantum models of a source | `qmem/complexity.py` |
| literature numbers, recall scoring, noise `eps`, strength floors | `qmem/benchmarks.py` |
| stationary memory, retention curves, reference purification | `qmem/diagnostics.py` |
| training, evaluation, multi-seed runs, analyses | `qmem/train.py`, `qmem/experiment.py`, `qmem/analysis.py` (results: `docs/phase2-results.md`) |
| run metadata / results files | `qmem/records.py` |

### Parameter layout

* `HardwareEfficientBlock.theta`: shape `(depth, n_block_qubits, n_rotations)`.
  Layer `l` applies, on each block qubit, the rotations in `rotations` order
  (default `RY` then `RZ`), then the entanglers on the `pattern` pairs
  (`linear`, `ring`, `all`, `none`; `cnot` or `cz`). Init is uniform on
  `[-pi·init_scale, pi·init_scale]` from a seeded generator.
* `ControlledRotationBlock.strength`: one `g` per (control, target) pair, or a
  single shared `g` (`strength_shared`); a parameter if `strength_trainable`,
  otherwise a buffer (set with `set_strength`).
* `OutcomeConditionedBlock.ws[y].theta`: the `W_y` layers, one
  `HardwareEfficientBlock` per outcome on the memory qubits.
* `build_model` draws all parameters, in block order, from
  `torch.Generator().manual_seed(config.seed)`.
* `model.rho0` is a buffer (the initial memory is not trained yet).

### Batching

States carry leading batch dimensions; the sequence API takes `(B, T)` long
tensors (equal-length sequences; no padding/masking yet). The instrument is
built once per call and shared across the batch and time steps. Measured on
an Apple-silicon laptop CPU: forward + backward for `B = 256`, `T = 32` takes
about 45 ms with 2 memory qubits and about 0.2 s with 4 memory qubits and 2 ancillas.

## 5. Numerical safeguards

* All complex tensors are `complex128` and all real tensors `float64`.
* No clamping by default. If an observed token has `p(y) = 0`, its step
  log-probability is `-inf` (the model's true answer) and that row's memory is
  left as it was so other rows and later steps stay finite. Counted in
  `model.diagnostics["zero_prob_steps"]`.
* `0 < p(y) < tiny_prob` (default `1e-10`) is counted
  (`tiny_prob_steps`) and logged on the `qmem` logger: there the normalisation
  `Phi_y(rho)/p` loses relative precision (absolute rounding error ~1e-16 in
  `Phi_y(rho)`).
* `prob_floor` (default off) replaces `log p` by `log max(p, prob_floor)` in
  the likelihood only. This changes the objective, so every use is counted
  (`floored_steps`) and logged. The memory is always normalised by the true `p`
  so it stays a density matrix.
* The sampler clamps rounding-level negative probabilities (`~ -1e-17`) to 0
  before drawing. A drawn outcome always has `p > 0`.
* Gradients through `p(y) -> 0` blow up as `1/p`; expect this near projective
  measurement of nearly deterministic outcomes.

## 6. Why a custom PyTorch density-matrix simulator

* **Differentiating the conditional update.** The objective is
  `sum_t log p(y_t|rho_{t-1})` with `rho_t = Phi_{y_t}(rho_{t-1}) / p`, unrolled
  over long sequences. In plain tensor code, autograd differentiates exactly
  this expression, normalisation included. Frameworks built around circuits
  and expectation values (PennyLane, Qiskit Aer) treat mid-circuit measurement
  either via deferred measurement (qubit count growing with `T`) or by
  sampling/postselection, and differentiating through outcome-conditioned
  normalisation is awkward or opaque.
* **Size.** At most ~6 qubits (`d <= 64`) per step, so building dense
  unitaries and Kraus tensors is trivial, and every intermediate can be
  inspected. Nothing here needs a circuit compiler.
* **PyTorch over JAX.** Both would work. PyTorch was chosen for mature
  complex128 autograd on CPU, an eager mode that is easy to debug and log
  (counting tiny probabilities is a Python side effect, which is awkward
  under `jit`), and `nn.Module`/optimisers that later training phases can use
  directly. If long-sequence speed becomes the bottleneck, the per-step loop is
  the obvious place for `torch.compile` or a move to JAX `lax.scan`.
* **Hardware translation** (brief §11) is a later, separate step: the circuit
  blocks are standard gates (RX/RY/RZ, CNOT/CZ, controlled-RY), so exporting to
  Qiskit with mid-circuit measurement and reset is straightforward.

## 7. Ablation variants (Phase 3b)

`qmem.ablations` holds the variants of A used to isolate what the conditioned
update contributes: B (memory follows the unconditioned channel `E`), C_k
(memory reset to `rho_0` every `k` tokens), D (readout effects `M_y` with the
post-measurement state discarded, then a token-controlled unitary `V_y`) and a
hybrid with a classical controller on the measurement basis. Each subclasses
`QuantumMemoryModel` and overrides three hooks of one shared per-step loop
(context, readout, update); with the default hooks the loop reproduces A
exactly. `build_variant` builds any of them from a `[model]` table with a
`variant` key, and `qmem.experiment` uses it for every run. The maths, the
matching and the physical caveats (D needs re-preparation to be faithful) are
in `docs/phase3b-ablations.md`.

## 8. Known limitations / for later phases

* Equal-length sequences only; padding/masking would go in `step_log_probs`.
* The instrument is time-homogeneous (one `U` for all steps); no input-dependent
  gates yet.
* Barren plateaus are not addressed; `init_scale` is the only knob so far.

## 8. Baselines (Phase 3a, `qmem.baselines`)

Every baseline has A's interface (`step_log_probs`, `log_prob`, `predictive_probs`,
`sample`, `n_parameters`, `diagnostics`) plus `describe()` (state dimension, analytic
multiply-adds per symbol), so `qmem.train.train` and `qmem.train.evaluate` treat it
exactly like A. Results and the comparison are in `docs/phase3a-baselines.md`.

| baseline | maths | code |
|---|---|---|
| HMM, D states | labelled transitions `L[y,i,j] = P(y, j \| i)`, belief `b' ∝ b L[y]` (the classical instrument) | `HMMModel`; `fit_em` (Baum-Welch) or gradient via softmax logits |
| order-k Markov | add-α smoothed counts of the last `min(t, k)` symbols | `NGramModel`, `fit_ngram` (reuses `analysis._markov_counts`) |
| GRU / tanh RNN | one-hot previous symbol → hidden `h` → softmax | `RNNModel` (float64 `nn.GRU` / `nn.RNN`) |
| complex uMPS / LPS | `p(y \| L) ∝ Tr(Σ_k A_yk L A_yk† Σ)`, `Σ` the fixed point of `Σ_yk A_yk† · A_yk` | `UMPSModel` (`purification = κ`) |
| HQMM | Kraus `K[y,e]` stacked as an isometry `V = QR(W)`, `V†V = I` | `StiefelInstrumentModel` (subclasses `QuantumMemoryModel`, so it uses `Instrument`) |

Two identities worth keeping in mind when reading the comparison:

* A uMPS of bond dimension χ with positive-definite `Σ` is gauge-equivalent
  (`A → Σ^{1/2} A Σ^{-1/2}`) to an instrument with one Kraus operator per outcome on a
  χ-dimensional pure memory, which is A's model class with `n_env = 0`. The uMPS and the
  HQMM with `E = 1` are therefore the same class under different parameterisations
  (unconstrained with a fixed-point normalisation, against the Stiefel manifold).
* The HQMM's `V` is the ancilla-`|0,0>` column block of an arbitrary unitary on
  memory ⊗ output ⊗ env (`StiefelInstrumentModel.unitary()` completes it), so the HQMM is
  A with the circuit replaced by an unrestricted unitary.
