# Phase 3b: quantum ablations of architecture A

Issue #7, part of the campaign in issue #2 (§6). The question is *what* about
measurement-conditioned memory matters. Each variant below changes one thing
about architecture A and keeps everything else the same: the trainer, splits,
metrics, recipe and seeds of Phase 2 (`docs/phase2-results.md`). All of them
share A's interface (`log_prob`, `step_log_probs`, `predictive_probs`, `sample`,
`n_parameters`), and they live in `src/qmem/ablations.py`. The tests are in
`tests/test_ablations.py`.

Reproduce every number here with one command (about RUNTIME with 7 workers):

```sh
uv run python experiments/phase3b.py --config experiments/configs/phase3b.toml
```

The generated tables are in [`results/summaries/phase3b/summary.md`](../results/summaries/phase3b/summary.md),
with `main.csv`, `summary.json` and `gaps.png`. The raw per-run records go to
`results/phase3b/<timestamp>/` (gitignored). If a run is interrupted,
`--from-run <dir>` trains only the missing runs.

## 1. The variants

Notation as in `docs/architecture.md`. `U` is the per-step circuit on memory `M`
and output ancillas `A` (no env ancillas in the recipe). The instrument of `U` is
`Phi_y(rho) = sum_e K_{y,e} rho K_{y,e}^†` with effects `M_y`. `rho_0 = |00><00|`.

| variant | prediction for `y_t` | memory after `y_t` | in `qmem.ablations` |
|---|---|---|---|
| **A**, conditioned | `Tr(M_y rho_{t-1})` | `Phi_{y_t}(rho_{t-1}) / p(y_t)` | `build_model` (unchanged) |
| **B**, outcome-averaged *control* | `Tr(M_y rho_{t-1})` | `E(rho_{t-1}) = sum_y Phi_y(rho_{t-1})` | `OutcomeAveragedModel` |
| **C_k**, reset | `Tr(M_y rho_{t-1})` | A's update, then `rho_0` if `t` is a multiple of `k` | `ResetMemoryModel` |
| **D**, separate update / readout | `Tr(M^R_y rho_{t-1})` | `V_{y_t} rho_{t-1} V_{y_t}^†` | `SeparateUpdateReadoutModel` |
| **hybrid**, classical feedback | `Tr(M_y(h_t) rho_{t-1})` | `Phi_{y_t}^{(h_t)}(rho_{t-1}) / p(y_t)` | `ClassicalFeedbackModel` |

All variants are built by `qmem.ablations.build_variant` from the `[model]` table
plus a `variant` key and the variant's own settings (`reset_every`,
`update_depth`, `feedback_lags`). A, B, C_k and the hybrid start from identical
circuit parameters for a given seed.

### A: measurement-conditioned memory

This is the Phase 2 model and recipe, unchanged: 2 memory qubits, `memory_depth = 2`
memory-only hardware-efficient layers `V`, controlled-RY(2g) coupling `C(g)` with
trainable `g`, then `joint_depth = 2` layers `J` on memory plus output.
`U = J C(g) V`. Prediction and memory update are the same quantum operation.

### B: outcome-averaged control (not a competitor)

B uses the same instrument, but the memory follows the unconditioned channel
whatever token is observed, so

```
p_B(y_t | y_<t) = Tr(M_y E^{t-1}(rho_0)),
```

which does not depend on `y_<t`. **B ignores the observed history by
construction.** On a stationary source the best it can do is the best i.i.d.
model, and A ≥ B (in likelihood) is guaranteed. Following the literature review
(§3, item 4), B is reported only as a control, to check that the
conditioning is what A exploits.

*Identity tested* (`test_b_equals_marginal_of_a_over_all_histories`): at the same
parameters, B's distribution for `y_t` is A's marginal of `y_t` summed over every
history `y_<t`, and B's memory is A's `E^t(rho_0)`. This follows from
`sum_y Phi_y = E` and linearity:
`sum_{y_<t} Phi_{y_{t-1}} ... Phi_{y_1}(rho_0) = E^{t-1}(rho_0)`.

Two B numbers are reported. One is **B trained as B**. The other is **A's
outcome-averaged prediction**: each trained A, evaluated with its memory replaced
by `E^{t-1}(rho_0)`. The second is B at A's parameters (the Phase 2 preview,
now for every seed). It also gives the conditioned versus unconditioned
state-dynamics diagnostics of brief §9.

### C and C_k: reset memory

After the k-th, 2k-th, … token the memory is reset to `rho_0`. Within a block of
`k` tokens the update is A's. C = C_1 is memoryless: every prediction comes from
`rho_0`, so C is an i.i.d. model. C_2 and C_4 are finite-window retention probes.

- **Bound.** With resets every `k` tokens, `p(y_1..y_T)` factorises over blocks
  (tested: `test_c1_is_memoryless_and_ck_factorises_into_blocks`). The best C_k is
  therefore the source's own predictor restarted every `k` tokens. Its
  expectation is `H(Y_1..Y_k)/k` (`task.optimal_nll(k)`), because every source
  starts stationary. The summary reports each C_k against this bound on the same
  test sequences (paired), and against the oracle.
- **Reset state.** The reset state is fixed at `rho_0 = |00>`, not trained. The
  memory-only layers `V` act before every readout, so a trainable reset state would
  only re-parameterise `V rho_0` and would add no expressiveness for C_1.
- **On hardware**, a reset of the 2 memory qubits every `k` tokens.

### D: separate update and readout (the key comparison)

```
p_D(y_t | y_<t) = Tr(M^R_y rho_{t-1}),        rho_t = V_{y_t} rho_{t-1} V_{y_t}^†
```

- **Readout.** `M^R_y` are the effects of a readout instrument with A's layout
  (coupling `C(g)` and joint layers `J`, built by `build_model`). Its
  post-measurement state is **discarded**: the readout contributes only
  `p(y|rho)`, the Born rule with a separate effect.
- **Update.** `V_y` is a token-controlled unitary on the memory qubits only: one
  hardware-efficient layer per token value (RY, RZ on each memory qubit, then a
  CNOT), with its own parameters. This is an input-driven quantum RNN, in the
  template of the literature review §2.4.
- **Pure memory.** From `rho_0 = |00>` the memory stays pure and is a
  deterministic function of the token history,
  `|psi_t> = V_{y_t} ... V_{y_1} |00>` (tested).

**Parameter matching.** D uses no memory-only layers in its readout, because the
`V_y` already precede every readout:

| alphabet | A | D |
|---|---|---|
| binary | `V` 8 + `g` 1 + `J`(depth 2, 3 qubits) 12 = **21** | `g` 1 + `J`(depth 2) 12 + 2 × `V_y` 4 = **21** |
| 4 symbols | `V` 8 + `g` 2 + `J`(depth 2, 4 qubits) 16 = **26** | `g` 2 + `J`(depth 1) 8 + 4 × `V_y` 4 = **26** |

- **Matched:** memory dimension (2 qubits), number of trainable parameters,
  trainer, recipe, training budget (1500 steps maximum, the same early stopping),
  data and seeds.
- **Not matched:**
  - for 4 symbols, D's readout is shallower (one joint layer, not two) to keep
    the parameter count equal;
  - D's update is a unitary while A's is a general (non-unitary, normalised)
    Kraus map;
  - circuit depth and two-qubit gate counts differ (table in §2).
- **D+ check.** A second D, labelled **D+**, uses A's whole circuit as the readout
  and two-layer `V_y` (37 and 58 parameters). If D loses to A, D+ tells us whether
  it was only short of parameters.

**What D is physically.** The issue suggested D is "measure-then-feedforward, with
a classical control bit only one step long". That holds only with a caveat, and
it changes how D's resources should be read:

- **The cost of the readout.** Measuring `M^R_y` on a single copy of the memory
  disturbs it. Some Kraus operators `L_y` with `L_y^† L_y = M^R_y` act, and
  "discarding the post-measurement state" leaves no memory to update. A
  sequential single-copy device that measures and then feeds forward `V_y`
  implements `rho -> V_y L_y rho L_y^† V_y^† / p(y)`. That is itself an
  instrument, a special case of A's model class (Kraus operators `V_y L_y`), not D.
- **D proper needs a readout that leaves the memory untouched.** Because `rho_t`
  is a known, deterministic function of the tokens, this is possible by
  **re-preparation**: to predict `y_t`, prepare `|00>`, apply the classically
  controlled `V_{y_1} ... V_{y_{t-1}}`, and measure the readout.
  - Generating or scoring a length-`T` sequence then takes `T` circuits with
    `T(T-1)/2` controlled unitaries in total.
  - The controller must store the **entire token history** classically. The
    one-step control bit is an illusion: the classical side holds all the
    information the memory holds.
  - Estimating `p(y|rho)` for teacher forcing costs shots per step, as for A.
- **D is not a "classical hidden state" model in the code.** The state that
  carries information between steps is the quantum memory, and nothing classical
  is fed to the readout. But D's memory is a pure state fixed by the history, so
  as a model class D is a complex linear RNN with unitary transitions and a Born
  readout. A has a Kraus update with normalisation, which is nonlinear in the
  state.

In short, A versus D compares two **model classes** at matched memory dimension
and parameter count. It does not compare two equally cheap devices: D's
physically faithful implementation costs `O(T^2)` gates and `O(T)` classical
memory per sequence.

### Hybrid classical-feedback control (reported separately)

After Stefanski et al. (literature review §2.1.4), the hybrid is A plus a
classical controller.

- **The controller.** Before the output ancillas are measured, ancilla `q` gets
  `RY(a_q) RZ(b_q)`, a trainable measurement basis, with
  `(a, b) = W h_t + c`. `h_t` holds the bits of the last `k = 2` tokens as ±1,
  most recent first, and 0 before the sequence starts.
- **Equivalently**, `K_y(h) = sum_{y'} R(h)_{y y'} K_{y'}`.
- **Starting point.** `W` and `c` start at 0, so the hybrid starts as A
  (tested). It has 6 extra parameters for binary tasks (27 in total) and 20 for
  4-symbol tasks (46).
- **What it measures.** The buffer `h_t` is a classical hidden state, so the
  hybrid shows how much a classical side memory adds. **It is never evidence for
  backaction.**

## 2. Circuits and resources per token

From `qmem.ablations.resource_counts`.

- **Depth** is an as-soon-as-possible schedule with every gate one time step.
- **Two-qubit gates:** a controlled-RY counts as one (it compiles to 2 CNOTs and 2 RY).
- **Every variant** uses 2 memory qubits plus `log2 |Y|` output ancillas. Each
  token needs one mid-circuit measurement per output ancilla, and that ancilla
  is reset.

RESOURCES_TABLE

Notes:

- **B** has A's circuit. On hardware it would be A with the memory decohered by
  averaging over outcomes. Its teacher-forced likelihood does not depend on
  the data's history.
- **C_k** adds a reset of both memory qubits every `k` tokens.
- **D:**
  - the depth is readout plus one `V_y`, and the `V_y` gates are classically
    controlled by the last token;
  - this is the *per-step* cost only. The re-preparation that makes D faithful
    multiplies the update cost by `t` at step `t` (§1).
- **Hybrid:** two classically controlled single-qubit rotations per output
  ancilla, and a classical buffer of `k · log2 |Y|` bits.

## 3. Results

RESULTS

## 4. What this shows, and what it does not

CONCLUSIONS
