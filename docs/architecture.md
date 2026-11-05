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
so `g` no longer bounds the information extracted per step. Measurement
strength sweeps should use `joint_depth = 0` (or analyse the full instrument,
e.g. via the effects `M_y`).

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
| `V`, `C(g)`, `J`, fixed unitaries; `U = B_last ... B_1` | `qmem/circuits.py`: `HardwareEfficientBlock`, `ControlledRotationBlock`, `FixedUnitaryBlock`, `Circuit` |
| `K_{y,e}` from `U` | `qmem/instrument.py`: `kraus_from_unitary` |
| (1) channel, (2) probabilities, (3) conditional state | `Instrument.channel`, `Instrument.probabilities`, `Instrument.condition` / `branch` / `branches`, `Instrument.effects` |
| (4) persistent memory | `qmem.model.Memory` (batched `rho`, `(B, d, d)`) |
| sequence model | `QuantumMemoryModel`: `log_prob`, `step_log_probs`, `predictive_probs`, `final_memory`, `unconditioned_memory`, `sample` |
| default layout from a config | `ModelConfig`, `build_model` |
| initial states, validity checks, purity, entropy | `qmem/states.py` |
| toy data | `qmem/data.py` |
| run metadata / results files | `qmem/records.py` |

### Parameter layout

* `HardwareEfficientBlock.theta`: shape `(depth, n_block_qubits, n_rotations)`.
  Layer `l` applies, on each block qubit, the rotations in `rotations` order
  (default `RY` then `RZ`), then the entanglers on the `pattern` pairs
  (`linear`, `ring`, `all`, `none`; `cnot` or `cz`). Init is uniform on
  `[-pi·init_scale, pi·init_scale]` from a seeded generator.
* `ControlledRotationBlock.strength`: one `g` per (control, target) pair; a
  parameter if `strength_trainable`, otherwise a buffer (set with
  `set_strength`).
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

## 7. Known limitations / for later phases

* Equal-length sequences only; padding/masking would go in `step_log_probs`.
* The instrument is time-homogeneous (one `U` for all steps); no input-dependent
  gates yet.
* Barren plateaus are not addressed; `init_scale` is the only knob so far.
