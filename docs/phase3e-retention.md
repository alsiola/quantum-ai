# Phase 3e: retention under repeated measurement (architecture A)

Issue #10 (campaign #2; literature review section 6, items 1 and 4). It builds on Phase 3d
(`docs/phase3d-strength.md`: the feed-forward layout, the operational strength `eta`)
and on #8's delayed recall (`docs/benchmarks.md` section 3.1). The generated tables are in
`results/summaries/phase3e/summary.md`, with the plots next to them.

Reproduce (about 3 h with 7 workers on a quiet machine; it took 2.9 h on a shared one):

```
uv run python experiments/phase3e.py --config experiments/configs/phase3e.toml
```

`--from-run <dir>` reuses saved runs and trains only the missing ones. `--no-train`
re-summarises saved runs only.

## What was built

* **A clock-free retention probe**, `qmem.tasks.marked_recall` (`docs/benchmarks.md`
  section 3.1b). Four symbols: distractor-a, distractor-b, cue-0 and cue-1. Cue-type
  symbols alternate between *cue* (uniform value) and *recall* (the cue's value, flipped
  with probability `eps`). With **geometric timing**, every symbol is cue-type with
  probability `q = 1/(tau+1)`, independently, so the delay is geometric with mean `tau`.
  The source has **3 causal states at every `tau`** (`C_mu` = 1.5 bits: the cue bit plus
  one phase bit) and quantum dimension 3. It needs no clock. With fixed timing, the
  marking alone does not remove the clock: the source still has `2 tau + 3` causal states
  (tested), as #8's `delayed_recall` has.
* **Retention diagnostics** (`qmem.diagnostics`): the cue's Holevo information in the
  memory as a function of the lag since the cue, counted only up to the recall
  (`label_retention`); stored-bit alignment of the distractor effects (`stored_bit_alignment`:
  commutator with the cue-conditioned memory difference, how much the distractor
  probabilities reveal the cue, and the contraction of cue distinguishability per
  distractor); reference-qubit purification along teacher-forced and generated trajectories.
* **A strength window for architecture A**: `feedforward_g_for_eta` (the shared `g` for a
  target `eta`) and `feedforward_strength_floor` (the least `g` that passes the
  per-outcome spectral test, which is necessary only).
* **An exact solution** for the capacity check: `qmachine_instrument`, the phaseless
  q-machine of the source (Binder et al. construction) as a general instrument on 3
  levels. It scores a gap of 0 to the source at every `tau` (summary, "Exact q-machine").
* `experiments/phase3e.py`: 820 training runs, 10 model seeds per cell, selection on
  validation only. Parameters are saved per run.

Tests: `tests/test_retention.py` (exact rate against Monte Carlo, causal-state counts,
q-machine exactness, the three structural facts below, diagnostics on analytic cases).

## Three structural facts about architecture A (proved and tested)

Write the feed-forward instrument as `K_y = W_y D_y(g) V`. Output ancilla `k` reads memory
qubit `k mod n` through `D(g)`: `K_0 = diag(1, cos g)` and `K_1 = sin g |1><1|` on that
qubit, in the frame after `V`. An outcome bit 1 **projects** the qubit it reads. An outcome
bit 0 measures it **weakly**. The first output bit is the type (0 = distractor) and the
second the value.

1. **At n = 2, projective readout forgets everything.** At `g = pi/2` both memory qubits
   are projected, so every `K_y` has rank 1. The memory after a step then depends only on
   the outcome, and the model is a first-order Markov chain. It recalls a cue that is
   immediately followed by its recall (delay 0), and is at chance for every delay of 1 or
   more (`test_projective_feedforward_on_two_qubits_forgets_everything`; measured: accuracy
   0.47–0.53 at every realised delay of 1 or more). At n = 2, any memory across a
   distractor has to live in the back-action of a weak readout.
2. **At n = 1 with a shared strength, two outcomes are tied.** Both ancillas read the
   same qubit with the same `g`, so `E_01 = E_10`, i.e. `p(distractor-b) = p(cue-0)` for
   every memory state (`test_shared_strength_on_one_qubit_ties_the_mixed_outcomes`). The
   source has `p(distractor-b) = (1-q)/2` everywhere and `p(cue-0)` in
   `{q/2, q(1-eps), q eps}`, so no `g` passes the per-outcome spectral test (the
   summary's `eta_floor(A, 1)` is `none`). With this alphabet, one qubit in A can't
   represent the task at any strength. That is a statement about A's coupling layout, not
   about one qubit: a 1-qubit instrument whose distractor effects are proportional to
   `I` keeps a cue indefinitely
   (`test_stored_bit_alignment_of_a_protected_and_an_erasing_readout`).
3. **At n = 2, every distractor weakly measures qubit 0.** Distractors have type bit 0, so
   they apply `diag(1, cos g)` to qubit 0. Distractor-b also projects qubit 1, and
   distractor-a reads it weakly. A cue that survives distractor-b must sit in qubit 0 (in
   `V`'s frame). If it is stored in qubit 0's `X` basis, so that the distractor
   probabilities don't reveal it (as the source requires), each distractor multiplies the
   coherence by `cos g`: after `k` distractors the two cue states overlap by
   `(1 - cos^2k g)/(1 + cos^2k g)`. If it is stored in the `Z` basis, nothing decays, but
   `p(type)` then depends on the cue, which the type gap measures. Every encoding in
   between trades distractor prediction against memory: the information–disturbance
   trade-off in its simplest form.

Fact 3 gives a reference retention curve, the **X-encoded qubit**: cue Holevo
`h((1 + o_k)/2)` with `o_k = (1 - cos^2k g)/(1 + cos^2k g)` after `k` distractors. It
falls below 0.1 bits after 6, 4, 3, 2 and 1 distractors at `eta` = 0.45, 0.6, 0.75, 0.9 and
0.99. In the summary's lag units (lag `k` = the memory that has seen `k - 1` distractors)
that is a horizon of 7, 5, 4, 3 and 2. This is a heuristic benchmark, not a proven optimum:
`W_y` and qubit 1 might do better.

## Results

All numbers: 2 memory qubits, `eps = 0.05`, mean ± SE over 10 seeds, unless stated. The
optimum recall accuracy is `1 - eps = 0.95`; chance is 0.5.

### Strength × delay (arm `grid`)

Recall accuracy (`accuracy_heatmap.png`):

| eta | tau = 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 0.45 | 0.926 | 0.80 ± 0.04 | 0.65 ± 0.03 | 0.57 ± 0.01 | 0.53 | 0.50 |
| 0.6 | 0.922 | 0.85 ± 0.03 | 0.77 ± 0.04 | 0.63 ± 0.02 | 0.54 | 0.52 |
| 0.75 | 0.918 | 0.87 ± 0.02 | 0.75 ± 0.02 | 0.64 ± 0.02 | 0.57 | 0.52 |
| 0.9 | 0.903 | 0.84 ± 0.01 | 0.74 ± 0.01 | 0.63 ± 0.01 | 0.57 | 0.53 |
| 1 (projective) | 0.731 | 0.66 | 0.60 | 0.56 | 0.54 | 0.51 |
| trained | 0.922 | 0.87 ± 0.02 | 0.76 ± 0.02 | 0.63 ± 0.02 | 0.56 | 0.52 |

The accuracy at mean delay `tau` mixes all realised delays. By realised delay, at
`tau = 4` (`accuracy_by_delay.png` shows `tau = 16`):

| eta | d = 0 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.94 | 0.87 | 0.75 | 0.77 | 0.71 | 0.66 | 0.58 |
| 0.9 | 0.95 | 0.95 | 0.86 | 0.67 | 0.51 | 0.50 | 0.51 |
| 1 | 0.96 | 0.50 | 0.50 | 0.53 | 0.50 | 0.47 | 0.49 |

At `eta = 0.45`, A is below its floor at `tau` ≤ 2 (`eta_floor(A, 2)` = 0.62 and 0.48) and
only just above it at `tau` ≥ 4 (0.35 to 0.40), so it fits the distractor statistics
worst (largest gaps).

### Falsification of "weak helps" (arm `falsify`, `tau = 4`)

Fixed `eta` ∈ {0.6, 0.75, 0.9}, learning rate ∈ {0.02, 0.05, 0.1}, twice the steps and
patience, four times the training data. Best learning rate per `eta`:

| eta | best lr | acc | horizon | acc at d = 4 / 6 |
|---|---|---|---|---|
| 0.6 | 0.1 | 0.843 ± 0.020 | 4.5 ± 0.3 | 0.84 / 0.74 |
| 0.75 | 0.1 | 0.783 ± 0.024 | 3.5 ± 0.2 | 0.71 / 0.57 |
| 0.9 | 0.1 | 0.759 ± 0.010 | 2.9 ± 0.1 | 0.58 / 0.50 |

The advantage of `eta = 0.6` over `0.9` grew from 0.03 ± 0.04 (grid recipe) to
0.08 ± 0.02 with tuning, more steps and more data. It holds at every learning rate tried
(0.795, 0.826, 0.843 against 0.728, 0.757, 0.759).

### Register size and the capacity check

| arm | n | tau | strength | acc | val-best acc | horizon | seeds that retain to lag 40 |
|---|---|---|---|---|---|---|---|
| A (`n1`) | 1 | 1, 4, 16 | 0.6, 1, trained | 0.49 to 0.51 | ≤ 0.51 | 1.0 to 1.9 | 0/10 |
| A (`n3`) | 3 | 2 | trained | 0.79 ± 0.02 | 0.919 | 7.2 ± 3.8 | 1/10 |
| A (`n3`) | 3 | 8 | eta = 1 | 0.62 ± 0.04 | 0.933 | 6.4 ± 3.9 | 1/10 |
| A (`n3`) | 3 | 8 | trained | 0.62 ± 0.02 | 0.642 | 6.2 ± 3.9 | 1/10 |
| general | 1 | 4 | trained | 0.85 ± 0.06 | 0.935 | 33.0 ± 5.3 | 8/10 |
| general | 1 | 16 | trained | 0.68 ± 0.07 | 0.928 | 17.0 ± 6.5 | 4/10 |
| general | 2 | 4 | trained | 0.92 ± 0.01 | 0.932 | 41 | 10/10 |
| general | 2 | 16 | trained | 0.60 ± 0.05 | 0.933 | 5.7 ± 4.0 | 1/10 |
| exact q-machine | dim 3 | 1 to 32 | eta 0.25 to 0.015 | 0.94 to 0.95 | — | — | exact |

"General" is a circuit that is not strength-faithful: memory block, couplings, then four
joint layers on every qubit, with trainable unshared `g`. At n = 3, A has one memory qubit
that no ancilla reads directly (ancilla `k` reads qubit `k mod 3`), and one seed in ten
uses it.

### Stored bit and the distractor effects

| model | [E_d, Delta] | abs(p(d\|0) − p(d\|1)) | contraction per distractor |
|---|---|---|---|
| A, n = 2, trained, tau = 1 | 0.24 ± 0.01 | 0.018 | 0.89 ± 0.03 |
| A, n = 2, trained, tau = 4 | 0.35 ± 0.02 | 0.008 | 0.47 ± 0.06 |
| A, n = 2, trained, tau = 16 | 0.34 ± 0.01 | 0.001 | 0.42 ± 0.04 |
| A, n = 2, eta = 0.6 / 0.9, tau = 4 | 0.30 / 0.38 | 0.013 / 0.008 | 0.72 / 0.37 |
| A, n = 2, eta = 0.45, tau = 1 | 0.15 ± 0.01 | **0.108** | **1.01** |
| general, n = 1 / 2, tau = 4 | **0.03 / 0.10** | 0.008 / 0.007 | **0.999 / 0.999** |
| general, n = 1 / 2, tau = 16 | 0.22 / 0.21 | 0.003 / 0.001 | 0.70 / 0.72 |

0 commutator means the distractor effects commute with the stored bit; contraction 1 means
a distractor leaves the two cue-conditioned memories as distinguishable as before.

### #8's implicit-clock recall at matched settings (arm `clock`)

#8's `delayed_recall` (fixed delay, binary alphabet, one output ancilla, `2 tau + 3`
causal states) on the same recipe and data sizes:

| tau | eta = 1 | trained (eta) | marked task, A trained, at realised delay `tau` | general, n = 2, at realised delay `tau` |
|---|---|---|---|---|
| 1 | 0.931 ± 0.002 | 0.932 ± 0.001 (0.87) | 0.94 | 0.92 |
| 2 | 0.877 ± 0.048 | 0.892 ± 0.028 (0.85) | 0.78 | 0.91 |
| 4 | 0.558 ± 0.022 | 0.571 ± 0.026 (0.78) | 0.63 (0.84 at eta = 0.6, tuned) | 0.91 |
| 8 | 0.508 ± 0.004 | 0.517 ± 0.014 (0.84) | 0.50 (0.62 at eta = 0.6, tuned) | 0.93 |
| 16, 32 | 0.49 to 0.51 | 0.49 to 0.50 | — | — |

The marked columns come from the models trained at mean delay 4 (realised-delay bins),
so they are a like-for-like comparison of memory across the same number of distractors,
not of the same training distribution.

## Answers to the four questions

**1. Is there an interior optimal strength for long `tau`?** In A at n = 2, yes, but it is
a modest effect at moderate delays and nothing helps at long ones.
*Shown:* projective is the worst setting at every `tau` ≤ 8, and provably so (fact 1):
0.60 against 0.77 at `tau = 4`. Among `eta` ∈ [0.6, 0.9], at the grid recipe the cells are
within 1–2 SE. With tuning, more steps and more data (falsification arm), `eta = 0.6`
beats `0.9` at `tau = 4` by 0.08 ± 0.02. The gain sits in the tail of the delay
distribution: `0.9` is best up to 2 distractors, `0.6` is best from 3 on (0.84 against
0.58 at 4 distractors). `eta = 0.45`, near A's floor, is worse again (0.65), so the optimum
is interior. At `tau` ≥ 16 every `eta` is within 0.07 of chance.
*Not like monitored QRC:* here the reason is fact 3, the cue's coherence decays by
`cos^2 g` per distractor. The exact q-machine (`eta` 0.10 at `tau = 4`, 0.015 at
`tau = 32`) and the retaining general-circuit seeds (`eta` 0.10–0.54, median 0.15, at
`tau = 4`; 0.05–0.11 at `tau = 16`, where five of the six forgetting n = 1 seeds sit at
0.71–0.90) are all weak, so on this task the solutions that work are weak ones.

**2. Does the trained `eta` depend on `tau`?** Weakly, and in the direction that hurts
retention. n = 2, `eps = 0.05`: 0.69 ± 0.02, 0.69 ± 0.04, 0.81 ± 0.05, 0.81 ± 0.04,
0.78 ± 0.03, 0.76 ± 0.04 at `tau` = 1, 2, 4, 8, 16, 32. A rise of about 0.1 from
`tau` ≤ 2 to `tau` ≥ 4 (about 2 SE), then flat; `eps = 0` (0.67 → 0.76 → 0.77) and
`eps = 0.1` (0.76 → 0.79 → 0.80) agree. Seeds spread widely (0.53–0.97 at `tau = 4`), and
within a cell the weak seeds retain longer (at `tau = 4`, horizon 4 at `eta` 0.66–0.68,
horizon 2 at 0.96–0.97). Training drifts stronger as recalls get rarer, plausibly because
recalls are only a fraction `q/2` of the loss while a stronger readout fits the distractor
positions better (non-recall gap 0.0055 nats at `eta = 0.9` against 0.0091 at 0.6, at
`tau = 4`). The clock task's trained `eta` shows no trend (0.78–0.87).

**3. How does the horizon scale with n and `eta`?** In A it is set by the per-distractor
contraction, not by the register size.
*n = 2:* horizon (first lag with cue Holevo below 0.1 bits) at `tau = 4`: 2.2 ± 0.4, 3.8 ±
0.3, 3.3 ± 0.2, 2.7 ± 0.2, 2.0 ± 0.0 at `eta` = 0.45, 0.6, 0.75, 0.9, 1; tuned: 4.5, 3.5,
2.9 at 0.6, 0.75, 0.9. The X-encoded reference gives 5, 4, 3 at 0.6, 0.75, 0.9, so trained
models sit about one lag below it and follow its slope (`eta = 0.45` falls well short of
its reference 7). *n = 1:* horizon 1 at every `eta` (fact 2). *n = 3:* most seeds have
horizon 2–5 at `tau` = 2 and 8, like n = 2; in 4 of the 6 cells one seed in ten keeps the
cue to lag 40.
*Purification:* median steps for a reference qubit to purify (generated trajectories,
below 0.01 bits) grow with n and weakness: n = 2: 1.0, 2.7–2.9, 3.5–3.9, 4.1–5.6, 6.2–6.9 at
`eta` = 1, 0.9, 0.75, 0.6, 0.45, i.e. of order `2^n = 4`; n = 3: 13–27 steps, with only
55–95 % of trajectories purified within the sequence, i.e. at least of order `2^n = 8`.
That matches the purification-time picture in scale [Fidkowski2021; Gullans2020PRX], but
the cue horizon at n = 3 did not grow with it: A forgets the cue faster than the register
purifies, because each distractor weakly measures the qubit that holds it.

**4. Do successful models learn a protected subspace?** A doesn't; the general circuit
does.
*A (n = 2):* the distractor effects do not commute with the stored bit (normalised
commutator 0.24–0.35), and each distractor shrinks cue distinguishability to 0.39–0.47 of
its value at `tau` ≥ 4. The trained models choose the X-like encoding (the distractor
probabilities reveal the cue by only 0.0005–0.008) and pay in memory. The one exception
shows the trade-off of fact 3 directly: at `tau = 1`, `eta = 0.45`, contraction is 1.01
(no decay) but `p(distractor)` reveals the cue by 0.108, with the largest type gap
(0.010 nats).
*General circuit:* at `tau = 4`, commutator 0.03 (n = 1) and 0.10 (n = 2), contraction
0.999, and the distractor probabilities don't reveal the cue (0.008). That is a protected
subspace, a decoherence-free readout of the issue, found by 8/10 (n = 1) and 10/10
(n = 2) seeds from random starts. At `tau = 16`, 4/10 and 1/10 seeds find it, and
selection on validation picks one (val-best accuracy 0.928 and 0.933).

## What #8's collapse was

Mostly **measurement-induced forgetting built into A's coupling layout**, with the clock
(capacity) a secondary cost at `tau` ≤ 4, and not an optimisation failure of A.

* *The clock costs something.* Across 4 distractors at n = 2, #8's task gets 0.57; the
  clock-free task gets 0.63 (trained `eta`) to 0.84 (`eta = 0.6`, tuned).
* *Removing the clock does not rescue A.* Across 8 distractors the clock-free task gets
  0.50–0.62, as #8's gets 0.52. At n = 3 it is the same for at least 9 seeds in 10.
* *The task fits in the register.* The marked task has quantum dimension 3, and the exact
  q-machine scores a zero gap. A general 2-qubit circuit trained on it holds the cue
  across 8 distractors (0.93) in 10/10 seeds at `tau = 4`.
* *A's failure does not look like optimisation.* Tuned learning rates, twice the steps
  and four times the data move the `tau = 4` accuracy by at most 0.08 and leave the
  horizons where fact 3 predicts them.

The general circuit's failures at `tau = 16` (6/10 and 9/10 seeds) are optimisation: some
seeds succeed with the same architecture and budget.

## Shown versus speculation

*Shown* (theorem, test, or measurement with 10 seeds): facts 1–3; the 3-state, `tau`-free
causal structure of the probe and the exactness of its dimension-3 q-machine; every table
above.

*Not shown:*
* That A at n = 2 **cannot** represent the task. Its per-outcome spectral test passes at
  `eta` ≥ 0.35–0.64 (necessary only), and fact 3's X-encoded qubit is a heuristic
  benchmark: `W_y` and qubit 1 might do better. What is shown is that training, tuned,
  does not find anything better, and that the measured horizons track the heuristic.
* That weak readout is **necessary** for retention in general. For an exact model the
  distractor probability must be `(1 - q)/2` on every reachable memory, so on the span of
  those memories the distractor effects are fixed and weak; a stronger readout of unused
  dimensions would raise `eta` without harming the task. The evidence that solutions are
  weak is the q-machine and the trained general circuits, not a proof.
* Larger n. Whether A's horizon grows with n when the unread qubits are used (one seed in
  ten at n = 3) is untested beyond n = 3.

## Judgement calls and what was pruned

* **Geometric delay instead of fixed.** A fixed delay needs `2 tau + 3` causal states
  even with the phase marked, which would bring the clock back. The geometric version
  has 3 at every `tau`. The fixed-delay variant is in the code (`timing = "fixed"`).
* **Symbol mapping (type, value)** on the two output bits, so that distractors share the
  type bit 0 and don't project the memory qubit read by the type ancilla (an outcome bit 1
  projects).
* **Grid pruned for compute** (the machine was shared): sequences of 64 for `tau` ≤ 8
  (128 otherwise); `eps = 0` and `0.1` at `tau` ∈ {1, 4, 16} and trained strength only;
  the clock arm at trained and projective strength only (its floor is 0.95); n = 3 at
  `tau` ∈ {2, 8}; n = 1 at `tau` ∈ {1, 4, 16}; falsification at `tau = 4` with
  `eta` ∈ {0.6, 0.75, 0.9} and three learning rates.
* **No soft-target "exact objective" optimiser.** The general-circuit arm (trained on the
  data) and the analytic q-machine play the role of the capacity check.
* **Records** span commits `a184170` and `7a404a2`, which differ in docs only; some runs
  carry `dirty = True` from that window.
