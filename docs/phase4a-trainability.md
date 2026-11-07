# Phase 4a: trainability of A and of the large-D quantum baselines

Issue #12 (campaign #2, sections 10 and 15). Three observations from Phases 2 and 3a looked
like one problem:

1. A at 3 memory qubits does worse than at 2 on every Phase 2 task where both ran
   (`markov1_quaternary` 8/10 failed at n = 3).
2. On `markov1_quaternary` A at n = 2 fails 7/10 seeds while the HQMM of the same model
   class (D = 4) fails 0/10.
3. HQMM and uMPS at D = 8 and 16 fail tasks that D = 2 solves.

The answer: they are **not one problem**. Item 1 and item 2 are a landscape problem of A's
circuit: local minima and saddles, not a budget shortfall and not a barren plateau. Screened
restarts fix it on the binary Phase 2 tasks and mostly fix it at n = 2; a deeper circuit
fixes most of it at n = 2. Item 3 is an initialisation problem of the unconstrained models.
A random Stiefel/uMPS start at large D has exponentially small gradients and an almost
uniform predictive. The near-identity start with the data unigram fixes it on
`iid_quaternary` but not on `markov1_quaternary` at D = 16.

All generated tables are in `results/summaries/phase4a/summary.md`, next to `cells.csv`,
`init_stats.csv`, `tuning.csv`, `summary.json` and `init_stats.png`.

Reproduce (4 workers; RUNTIME_PLACEHOLDER):

```
uv run python experiments/phase4a.py --config experiments/configs/phase4a.toml
```

`--from-run <dir>` reuses saved runs and `--no-train` only summarises them. Protocol: seeds
0–9 for every training cell, tuning seeds 100 and 101 on validation for the arms with an lr
grid, and the Phase 2 (A), Phase 3e (feed-forward A, `marked_recall`) and Phase 3a
(HQMM, uMPS) recipes unless an arm overrides them. A run **fails** if its test gap (test
NLL minus the source's NLL on the same sequences, nats/symbol) is ≥ 0.01, or if it crashes
or diverges. The grid was pruned for compute (the machine was shared and swapping while it
was planned): each question is answered with the cells that bear on it, not a full
model × n × task product.

New code, all behind config options with the old behaviour as the default
(`qmem.trainability`, tests in `tests/test_trainability.py`):

- `init = "product"` for A (CZ entanglers only): every gate starts at identity up to
  `init_scale` noise and the readout starts at strength `strength`, so the effects start
  near `p_y · I` (exactly `p_y · I` at zero strength and noise);
- `init = "identity"` for the HQMM and the uMPS: Kraus operators `sqrt(p_y) · I` (uMPS
  tensors likewise) plus noise that does not grow with D;
- `init_probs = "uniform" | "data" | [...]`: the `p_y` above, with `"data"` the training
  split's unigram;
- `train.restarts`, `train.restart_steps`: R starts screened for a few hundred steps,
  and the best by validation NLL is trained on;
- `teacher = {kind, mode}` in the runners: soft-target fits to the source's exact
  predictive or to a trained model (`mode = "fit"`), and warm starts from such a fit
  (`mode = "warm"`).

## Before and after

Failures out of 10 and the median test gap. For A, "before" is the Phase 2 recipe rerun
here (`m1q_A_recipe`, `p2_A_recipe`); it reproduces `docs/phase2-results.md` exactly on
every cell. For the HQMM, "before" is `docs/phase3a-baselines.md` (D = 16 reused D = 8's
lr); the per-D-tuned random start is in the table after this one.

| model | task | n (D) | before | after | the fix |
|---|---|---|---|---|---|
| A | markov1_quaternary | 2 (4) | 7/10, 0.0199 | **1/10, 0.0035** | `joint_depth = 3` (34 params) |
| A | markov1_quaternary | 2 (4) | 7/10, 0.0199 | 2/10, 0.0020 | 4 screened restarts |
| A | markov1_quaternary | 3 (8) | 8/10, 0.0343 | 5/10, 0.0078 | `joint_depth = 4` (54 params) |
| A | markov1_quaternary | 3 (8) | 8/10, 0.0343 | 6/10, 0.0267 | 4 screened restarts |
| A | markov2_binary | 2 (4) | 0/10, 0.0016 | 0/10, 0.0012 | restarts |
| A | markov2_binary | 3 (8) | 1/10, 0.0059 | **0/10, 0.0032** | restarts |
| A | even | 2 (4) | 0/10, 0.0011 | 0/10, 0.0005 | restarts |
| A | even | 3 (8) | 1/10, 0.0039 | **0/10, 0.0009** | restarts |
| HQMM | iid_quaternary | 4 (16) | 10/10, 0.0157 | **0/10, 0.0023** | identity init, data `p_y`, lr 0.003 |
| HQMM | iid_quaternary | 3 (8) | see #6 | 0/10, 0.0018 | identity init, data `p_y`, lr 0.003 |
| HQMM | markov1_quaternary | 4 (16) | 10/10, 0.0275 | 10/10, 0.0174 | identity init, data `p_y`, lr 0.003 (gap down, still fails) |
| HQMM | markov1_quaternary | 3 (8) | see #6 | 5/10, 0.0092 | identity init, data `p_y`, lr 0.003 |
UMPS_ROW
RET_ROWS

Things that did **not** help, each recorded as a cell in the summary:

| arm | cell | result |
|---|---|---|
| near-product start (`init = "product"`, data `p_y`) | A markov1_quaternary n = 2 / 3 | 9/10 and 10/10: **worse** than the recipe |
| near-product start | A markov2_binary n = 2 / 3, even n = 2 / 3 | 8, 6, 1, 3 out of 10: worse than the recipe on every cell |
| CZ entanglers, random init (the product start's control) | A markov1_quaternary n = 2 | 4/10, 0.0040 |
| lr tuned at n = 3 (0.03 chosen of 0.01/0.03/0.05), 6000 steps, no early stopping | A markov1_quaternary n = 3 | 10/10, 0.0230 |
| grow: fit the best n = 2 model's function into n = 3, then train | A markov1_quaternary n = 3 | 9/10, 0.0277 |
| per-D lr tuning alone (0.01 chosen of 0.003/0.01/0.03), random init | HQMM D = 16, iid / markov1_quaternary | 10/10 and 10/10, 0.0184 and 0.0275 |

## Answers

### 1. Plateau or barren landscape?

**Shown (init statistics, 20 random starts per row, `markov1_quaternary`).** The across-seed
variance of each gradient component falls with n for every model, but at very different
rates:

| model | Var[∂L/∂θ], n = 1 → 4 | factor per qubit | gradient norm n = 1 → 4 |
|---|---|---|---|
| HQMM (random Stiefel) | 9.8e-3 → 2.5e-6 | ≈ 16 | 0.50 → 0.07 |
| uMPS (random) | 1.1e-1 → 4.6e-4 | ≈ 6 | 1.86 → 0.97 |
| A general, jd = 2 | 5.7e-2 → 3.7e-3 | ≈ 2.5 | 0.90 → 0.34 |
| A general, CZ | 8.1e-2 → 1.6e-2 | ≈ 1.7 | 1.09 → 0.74 |
| A feed-forward | 9.7e-2 → 4.3e-4 | ≈ 6 (23 from n = 1 to 2, then 2–5) | 1.06 → 0.55 |
| HQMM identity init | 4.3e-3 → 2.4e-5 | ≈ 5.6 | 0.37 → 0.23 |
| uMPS identity init | 1.5e-2 → 2.9e-4 | ≈ 3.7 | 0.68 → 0.75 |

The HQMM's per-parameter variance falls as about 1/D², the signature of a random unitary
design. Its parameter count grows as D², so the full gradient norm still halves per qubit.
Under the identity init the norm stays flat (0.37 → 0.23) and the initial NLL drops from
1.40 to 1.15 at D = 16. A's circuit is shallow: its variance falls 2–3× per qubit, and its
gradient norm at n = 3 (0.57) is larger than the HQMM's at n = 2 (0.27), the model that
trains perfectly. **A at n = 2, 3 is not on a barren plateau.**

**Along training.** In every A cell the median gradient norm drops from 0.6–0.9 to 0.06–0.09
within 100 steps and then stays put. It is the same in failed and successful runs (final
0.066 against 0.072 at n = 2), so failure is not "stuck with a vanishing gradient". The
plateaus are minibatch-noise floors around a stationary point. The Hessian of the full
train NLL at the end of training (`m1q_A_recipe`, n = 2, finite differences of the exact
gradient) shows two kinds of failure. Seeds 1, 5, 6 and 8 have clearly negative eigenvalues
(−0.23, −7.5e-3, −1.2e-2 and −1.9e-3), so they are **saddles**. Seeds 4, 7 and 9 have no
eigenvalue below −4e-4, the size of the finite-difference noise that the successful seeds
2 and 3 also show, so they are **local minima** 0.02–0.05 nats above the optimum. Every seed
has one flat direction (the global phase).

**Speculative.** The saddles explain why restarts help and the lr budget doesn't: a screened
restart is a cheap way to land in a better basin. The large-D baselines' gradient decay is
the barren-plateau mechanism, but the identity init restores the gradient and still fails
`markov1_quaternary` at D = 16. Something beyond the start (lr 0.003 at the bottom of the
grid, and early stopping at about 400 steps) limits it there; that wasn't separated.

### 2. Initialisation

- **Near-identity for the HQMM and uMPS: yes, at large D.** HQMM D = 16 on
  `iid_quaternary` goes from 10/10 failed to 0/10 (gap 0.0157 → 0.0023), and D = 8 is
  0/10. On `markov1_quaternary` the D = 16 gap falls from 0.0275 to 0.0174 but every seed
  still fails; D = 8 fails 5/10. UMPS_INIT_SENTENCE
- **Near-product for A: no, it is worse.** The product start begins close to the unigram
  (initial NLL 1.30–1.32 against 1.41–1.58 for the random circuit; history TV 0.013
  against 0.08–0.15) and its gradient variance does not fall with n. Yet it fails 9/10 and 10/10 on
  `markov1_quaternary` and is worse than the recipe on all four binary cells. Its CZ-only
  control with a random start fails 4/10, so the CZ entangler is not the cause. The
  near-product point is a symmetric point of A's circuit (every gate near identity, a weak
  readout). Gradient descent leaves it along a few directions, and on these tasks it settles
  in poor minima. **Speculative:** the symmetric start breaks to the same few minima each
  time, so the seeds are less diverse than random starts.
- **Growing n = 2 into n = 3: no.** As implemented it *fits* a randomly started n = 3
  circuit to the best n = 2 model's predictive (soft targets, KL on the test split), then
  trains on the data. The fit itself fails: the median KL to the teacher is 0.033, and 9/10
  runs fail. **Not done:** an exact parameter embedding with the third qubit idle. A's
  brickwork at n = 3 doesn't contain the n = 2 circuit as a sub-circuit with the same
  angles, so that needs a different ansatz. The fit result shows only that the n = 3
  landscape is hard even with a function that is known to be representable as the target.

### 3. The predictive distribution at initialisation

**Shown.** For random starts the initial predictive entropy rises towards ln 4 = 1.386 with
n: HQMM 1.21 → 1.36, uMPS 1.22 → 1.36 and A 1.21 → 1.33 from n = 1 to 4. The purity of
the stationary memory `E^T(ρ0)` falls as about 1/D (HQMM 0.66 → 0.08), and the
history-dependence of the predictive (mean TV between histories) falls 0.14 → 0.08. A large
random instrument is a near-depolarising channel: the memory is nearly maximally mixed and
the predictive barely depends on the history. Every random start is 0.2–0.4 nats worse
than the unigram.

**Is it the mechanism?** For the large-D baselines, yes in part. The identity start removes
it (entropy 1.17–1.19 and NLL 1.15 at every D) and fixes `iid_quaternary`. For A, no. A
at n = 2, 3 has history TV 0.09–0.12, the same range as the HQMM at D = 4–8 that trains,
and the product start, which has none of this problem, trains worse.

### 4. Optimisation budget

**Not a budget shortfall for A.** At n = 3 with the lr tuned per n (0.03 of
{0.01, 0.03, 0.05}), 4× the steps (6000) and no early stopping, A fails 10/10 (median gap
0.0230, against 0.0343 with the recipe). The longer run lowers the gap but no seed reaches
0.01. **Not a tuning shortfall for the HQMM either.** Per-D tuning at D = 16 chose 0.01 (of
0.003/0.01/0.03) and still failed 10/10 on both tasks. The fix was the initialisation, and
with it the best lr moved to 0.003, the bottom of the grid. A smaller lr was not tried.

### 5. Circuit vs instrument

- **Capacity at depth: a large part of it.** One extra joint layer at n = 2 (`joint_depth = 3`,
  34 parameters, up from 26) cuts failures from 7/10 to 1/10. `joint_depth = 4` (42) gives
  2/10. At n = 3, `joint_depth = 4` (54) gives 5/10, against 8/10. The HQMM at D = 4 has
  128 real parameters.
- **Landscape: the rest.** Two soft-target fits at n = 2 remove sampling noise from the
  question:
  - A fitted to the source's exact predictive reaches KL 0.0001 in its best seed. The
    optimum is representable at 26 parameters, but half the seeds stop at KL 0.02–0.12
    (5/10 fail).
  - A fitted to the best D = 4 HQMM (0/10 failed, gap 0.0029) reaches KL 0.0027 at best
    and fails 6/10.

  So the 26-angle circuit can represent the solution and the HQMM's function, but from a
  random start gradient descent finds it in only about half the seeds, even with exact
  targets. With the Hessian result (saddles and local minima), that is a landscape problem
  of the circuit parameterisation. The HQMM's Stiefel parameterisation has no such minima
  at D = 4.

### 6. Purification vs retention

RETENTION_SECTION

## Recommended recipe

| model | n (D) | recipe |
|---|---|---|
| A, general layout | 2 | Phase 2 recipe + 4 restarts screened for 300 steps (`train.restarts = 4`, `restart_steps = 300`); on 4-symbol tasks also `joint_depth = 3` |
| A, general layout | 3 | as n = 2, with `joint_depth = 4` on 4-symbol tasks; expect about half the seeds to fail `markov1_quaternary` and select by validation |
| A, feed-forward | 2, 3 | Phase 3e recipe (strength trainable) + 4 screened restarts FF_NOTE |
| A, any | any | **not** `init = "product"` |
| HQMM | ≤ 2 (D ≤ 4) | Phase 3a recipe (random Stiefel init) |
| HQMM | ≥ 3 (D ≥ 8) | `init = "identity"`, `init_scale = 0.3`, `init_probs = "data"`, lr tuned with 0.003 in the grid |
| uMPS | χ ≥ 8 | UMPS_RECIPE |

The restart cost is 4 × 300 screening steps per run. Since the screened run then
early-stops sooner (mean 670–730 steps against 935–1075), the total cost is about twice the
recipe's.

## What is shown and what is not

- Shown, at 10 seeds per cell: the before/after counts above; the gradient-variance scaling
  at initialisation; that A's failures are not budget-limited at n = 3; that they include
  both saddles and local minima at n = 2; and that A's circuit can represent the source's
  and the HQMM's functions but rarely finds them.
- Not separated: whether HQMM D = 16 on `markov1_quaternary` needs a smaller lr, a longer
  run or more than the identity start; and whether an exact idle-qubit embedding would make
  n = 3 inherit n = 2's solution.
- The Hessian is for one cell (A, n = 2, `markov1_quaternary`) and from finite differences
  of the exact train-NLL gradient (step 1e-4).
- The grid is pruned; uMPS has one cell and A at n = 4 was not trained.
