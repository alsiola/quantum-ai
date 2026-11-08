# Phase 7 Stage 3: pre-registered hardware follow-up (q75 ablation)

Issue #24 (child of #17). This document fixes the design, the analysis and the decision rules
**before any data exist**. Nothing was submitted, and no IBM service was called. The only
mode run here is the local dry run: `FakeKingston`, the committed snapshots, exact
predictions, and a 50-shot local Aer pilot. Stage 4 executes the plan after the owner approves
it on #17, and reports in `docs/phase7-hardware.md`.

```
uv run --group hardware python experiments/phase7_hw.py --config experiments/configs/phase7_hw.toml --dry-run-local   # ~40 min, 2 workers
uv run --group hardware python experiments/phase7_hw.py --config experiments/configs/phase7_hw.toml --dry-run         # Stage 4: live target, no submission
uv run --group hardware python experiments/phase7_hw.py --config experiments/configs/phase7_hw.toml --submit          # Stage 4: needs approval_url
```

Code: `qmem.followup` (schedule, cost model, guards, analysis, decision rule; numpy) and
`experiments/phase7_hw.py`. The script imports Phase 6's job, resume-by-id, calibration and
saving code (`experiments/phase6.py`, refactored into `submit_or_fetch` and `finish_job`) and
#20's circuit builders. Tests: `tests/test_phase7_hw.py`. Dry-run outputs:
`results/summaries/phase7_hw_dryrun/`.

## 1. Question and claim ladder

The claim ladder after Stages 0–2 (#18–#23):

| rung | claim | status before Stage 3 |
|---|---|---|
| 1 | the output statistics have Hankel rank > 2 | supported |
| 2 | a stationary 2-state HMM is excluded, also as a drifting or mixed 2-state process (#19) | supported. A 3-state classical HMM reproduces the data |
| 3 | drift and classical artefacts do not produce it | model statements only: drift is excluded (#19); readout, processing and q59 residue are each < 10% (#21, #23); spectators are ≈ 5.5% (#22) |
| 4 | **preserving q75 is causally necessary** | **untested: this run** |
| 5 | a one-qubit physical memory dimension | not supported (#18); out of scope |

**Primary question.** Does the witness vanish when q75 is reduced to a classical bit each
step (A2), while a matched readout sham (A2s_spec) keeps it?

Three claims are kept apart throughout:
- the *output statistics* (what is measured);
- the *effective memory of the hardware* (rank ≥ 3);
- *memory attributable to q75*. Only A2 against A2s_spec, and A1 against A3, speak to this.

## 2. Arms, circuits, mapping and backend

The backend is `ibm_kingston` (Heron r2). Memory is q75, output q59, sham spectator q74
(q75's only other neighbour). Phase 6's pair is never re-mapped. Durations are Phase 6's live
baseline estimate (308.3 µs) plus each arm's FakeKingston excess.

| tier | arm | operation each step | from | shots | pubs per job | µs | H_q75 w(0) |
|---|---|---|---|---|---|---|---|
| 1 | baseline | Phase 6's circuit | #20 | 15,000 | 4 × 937–938 | 308.3 | −0.00690 |
| 1 | **A2** | measure q75 into `m[t]` | #20 | 10,000 | 3 × 833–834 | 310.6 | 0 (rank ≤ 2) |
| 1 | A2s_spec | delay on q75, measure q74 into `s[t]` | #20 | 15,000 | 4 × 937–938 | 310.5 | = baseline |
| 1 | A1 | reset q75 | #20 | 4,000 | 1 × 1,000 | 310.6 | 0 (i.i.d.) |
| 2 | A3 | random Pauli on q75, 64 fixed instances | #20 | 64 × 64 | 16 × 64 | 308.3 | −7.1e-5 |
| 2 | D1 | memoryless rx + reset on q59 alone | #23 | 10,000 | 3 × 833–834 | 308.1 | 0 |
| 3 | B2 | q59 reset, then a flag measurement (#21) | – | 15,000 | own job | 452.0 | – |
| 3 | D3 | classical 4-state positive control (#23) | – | 5,000 | own job | 308.1 | −0.0651 |
| 3 | C3 | q74 reset vs reset + X (#22) | – | 2 × 20,000 | 2 jobs | 310.7 | Δ = 0 |

The circuits are #20's ISA circuits, rebuilt exactly as #20 built them: level-0 transpile,
trivial layout. **The dry run reproduces all 68 committed SHA-256 hashes of #20** (baseline,
A1, A2, A2s_spec and the 64 A3 instances). D1 equals #23's committed QASM byte for byte.

Stage 4 re-generates **only the delays** from the live target:
- the sham delay (#20's measure duration, 545 dt in the snapshot);
- D1's pad (#23's rule, 98 dt).

The code checks that every change is a delay duration (`compare_retimed`). On FakeKingston's
target only A2s_spec changes (570 dt), and D1's re-computed pad is again 98 dt. **No level ≥ 1
transpile is ever run**, since it would merge A3's Paulis. The sampler runs without error
mitigation, as Phase 6. In the dry-run Aer pilot, A2's `m` register decodes with P(m = 1) =
0.46 at 50 shots, against 0.493 predicted.

**Dropped, with reasons** (#20–#23 give the numbers):

| arm | reason |
|---|---|
| B1 (double reset) | ≤ 7% predicted change; 1.5e5 shots per arm (211 s) |
| B3 (alternating ancilla) | only informative if B2's flags show leakage; ≈ 10 s per arm |
| B4m, B4r | references for B1/B2's post-selected witness; B2 is itself tier 3 |
| C1 (role swap) | tests the calibrated model, ≤ 2.7σ against spectators |
| C2 (another pair) | replication, not attribution; model spread 0.62–1.04 |
| C4 (echo) | weaker null than C3; 4.1σ even at upper strengths |
| D2 (known 2-state) | insensitive to every modelled mechanism (#23) |
| A1s, A2s, A3s (pure-delay shams) | predicted identical to the baseline, with the delay inside an idle window. A2s_spec already carries A2's delay |

**Power.** The decision needs the SE of A2's w(0) to be well below 10% of |w(0)|. At the
planned sizes, the primary SE under H_q75 is 1.30e-4 for the baseline, 8.6e-5 for A2, 1.29e-4 for A2s_spec, 1.0e-4 for A1, 1.4e-4 for A3 and 3e-6 for D1 (medians over the validation replicates). "Vanishes" therefore holds for any
|w_A2(0)| below ≈ 5.2e-4 (7.5% of the calibrated baseline). "Persists" at
25% sits at ≈ 20 SE. Moving shots between arms would buy nothing decisive, so the
issue's sizes are kept.

## 3. Schedule, pilot gate and calibration bounds

**Jobs.** There are four main jobs. Each holds every tier-1 and tier-2 arm, 14,524 shots in
31 pubs. Every arm is split evenly over the jobs, and each job's share is split into pubs of
at most 1,000 shots. The 64 A3 instances are dealt 16 per job by a seeded permutation. The
pub order within job j is `default_rng([2024, j]).permutation`; the config holds the seed,
and `schedule.csv` holds the resulting order. Jobs run in order 0–3.

**Expected spans.** One shot takes 0.558–0.561 ms. The pubs take:

| pub | span |
|---|---|
| baseline or A2s_spec (938 shots) | 0.52 s |
| A2 or D1 (834 shots) | 0.47 s |
| A1 (1,000 shots) | 0.56 s |
| A3 (64 shots) | 0.036 s |

A job takes 8.1 s of execution (`schedule.csv` has every pub's expected start). #19's episode
lasted 1.5 s, so it would cover 3–4 consecutive pubs.

**If the server reports one span per job** (as for Phase 6's jobs), the within-job order of
pubs cannot be verified:
- time resolution falls to the job, about 8 s;
- an episode can be localised only to its job;
- the per-pub and per-chunk analyses use the submitted order as nominal time.

The design still protects the contrasts. Every arm is in every job in a random order, so an
episode hits the arms in proportion to their shots. Four jobs give four independent
replications of every contrast. Per-pub spans, if reported, are saved and used.

**Pilot (one job, before anything else).** 500 baseline shots and 500 A2 shots, 1,000 shots
in total (2.6 s estimated). Go only if all of these hold:
- (a) the job completes, every pub decodes to (shots × 64) 0/1 arrays, and #23's integrity
  rail (values) passes. Any duplicate shots are inspected before going on;
- (b) the baseline's P(1) over positions ≥ 16 is within **0.02** (absolute) of the prediction
  from the live calibration. Stage 4 rebuilds #20's pieces from the live noise; the snapshot
  noise gives 0.5820. Phase 6's own episode moved P(1) by 0.027;
- (c) A2's q75 record has P(m = 1) within **0.03** of the live prediction (snapshot: 0.4925);
- (d) the execution-span seconds per shot are within **25%** of the cost model's
  (0.559 ms). Phase 6 measured 0.567 ms against 0.558 ms.

On any failure the script writes `pilot/gate_failed.json` and stops. It never retries;
re-running needs a new decision by the owner.

**Calibration acceptability.** Stage 4 captures the live calibration (Phase 6's
`calibration`, plus readout, T1 and T2 of q74 from the target) before and after **every** job.
It stops and asks the owner if any of these fails:
- the readout error, T1 or T2 of q59 or q75, or the q59–q75 CZ error, is outside a factor of
  2 of #16's snapshot (T1 and T2 either way; error rates may improve without limit);
- any measured qubit has readout error > 0.05, T1 < 50 µs or T2 < 30 µs. q74 is not in #16's
  snapshot, so only these absolute bounds apply to it.

On the stale FakeKingston calibration the check passes. The pair is never re-mapped silently.

## 4. Analysis and decision rules (fixed)

**Statistic.** w(0), with a common start set, burn 16 and k_max 7 (#18). w(1..3) are reported
for shape only.

**Uncertainty.** The primary SE is max(whole-shot bootstrap, circular block bootstrap with
b = 500 shots). Both are computed within job and stratified by job when pooled. The block is
capped at a quarter of a segment, as in #19's `segment_stats`. Each uses 1,000 resamples. The
95% CI comes from the resamples of whichever method gives the larger SE.

**Segments (#19).** For every job and arm:
- a change-point scan on g(1) − g(3), P(1) and the w(0) influence, with 1,000 permutations
  and a 50-shot grid;
- a χ² homogeneity test of g over 500-shot blocks;
- a w(0) constancy test over 1,000-shot blocks.

All are at α = 0.01. A job flagged in the baseline or A2 triggers the **segment
sensitivity**: the primary outcome is recomputed without the flagged jobs. If it changes, the
primary outcome is *ambiguous (segment-sensitive)*. Per-job and per-chunk w(0) and P(1) are
always reported.

**Guard rails (#23).** These run on every arm:
- malformed values invalidate the dataset; duplicate shots are a warning to inspect (§6);
- a positional χ² against the H_q75 marginals is a decoding warning;
- dispersion and order-inflation flags are reported with the segment results.

For A3, dispersion across its 64 different circuits is expected, so only integrity applies.

**Decision rule.**
- **Replication gate.** The baseline must give w(0) < 0 with |w(0)| ≥ 5 SE. Otherwise every
  ablation result is *uninterpretable* and is reported as such.
- **Primary outcome (A2).**
  - *Vanishes* if |w_A2| + 1.96 SE_A2 < 0.10 |w_base| (the upper end of the two-sided 95%
    interval of |w_A2|).
  - *Persists* if |w_A2| ≥ 5 SE_A2 and |w_A2| ≥ 0.25 |w_base|.
  - *Ambiguous* otherwise.
  - The baseline is the concurrent one, pooled over jobs.
- **Secondary family** (Holm, family-wise α = 0.01, two-sided z-tests):
  1. A2s_spec − baseline (with the ratio and its delta-method SE);
  2. A1 against 0;
  3. A3 against its exact realised-instance H_q75 value, −7.1e-5. #20 showed that this, not
     0, is A3's prediction, and the bias is ≈ 1 SE;
  4. D1's w(0) against 0;
  5. D1's lag-1 excess against its calibrated prediction, −0.0009.
- **Descriptive.** A1's and A3's outcomes under A2's thresholds, and A2's q75 record
  (P(m = 1), P(y | m)).

Every effect is reported with a 95% CI.

**Reading A2 with the other arms** (also fixed in advance):
- A2 vanishes and A2s_spec does not differ: **rung 4 supported**.
- A2 vanishes and A2s_spec is reduced: rung 4 supported, with q74 readout crosstalk on the
  causal path. The size of the reduction is reported.
- A2 persists while A1 and A3 vanish: the leading reading is incomplete dephasing by A2's
  measurement, not memory outside q75 (H_partial_c08).
- A2, A1 and A3 all persist: a memory outside q75 (H_outside).

## 5. Predictions and outcomes

Every number below is exact (no sampling), from the existing models: #20's joint pieces, #21's
q59-qutrit simulator, #19's fitted processes and #23's artefact transforms. Full
w(0), w(1), w(2), w(3), P(1) and lag-1 excess for every arm and hypothesis are in
`predictions.csv`. Notes on the hypotheses:
- **Drift.** The episode is #19's fitted regime: q75 −0.49 rad/step and q59 P(1|0) +0.060,
  in 13.5% of the shots. It is sampled contiguously from 40% of the run's time, across jobs
  1–2.
- **Spectators.** A reduced model on #20's pieces: a Stark phase over q59's two readout
  windows and a 5% thermal q74 with ZZ. It reproduces #22's Stark sweep to 0.01 (tested). At
  typical strengths it costs 10% of the baseline, against #22's full 5.5%, so it is
  conservative.

| hypothesis | baseline w(0) | A2 w(0) (ratio) | A2s_spec ratio | A1 w(0) | A3 w(0) | D1 lag-1 | P(1) base / A2 | predicted |
|---|---|---|---|---|---|---|---|---|
| H_q75 (calibrated) | -0.00690 | 0 (0.00) | 1.00 | 0 | -7.10e-5 | -0.0009 | 0.582 / 0.385 | vanishes |
| drift: #19 episode regime, 13.5% | -0.00593 | +8.32e-7 (-0.00) | 1.00 | 0 | -5.23e-5 | -0.0006 | 0.583 / 0.390 | vanishes |
| drift: #19 best K = 3 mixture | -0.00123 | -1.23e-3 (1.00) | 1.00 | -1.23e-3 | -1.23e-3 | -0.0009 | 0.582 / 0.582 | persists |
| q59 reset + leakage, #21 fit | -0.00579 | +3.35e-5 (-0.01) | 1.00 | 0 | +1.73e-4 | +0.0174 | 0.580 / 0.390 | vanishes |
| q59 leakage, #21 extreme | -0.00501 | +4.70e-5 (-0.01) | 1.00 | 0 | +2.79e-4 | +0.0499 | 0.612 / 0.415 | vanishes |
| spectators, typical (#22) | -0.00624 | 0 (-0.00) | 0.78 | 0 | -1.33e-5 | -0.0009 | 0.567 / 0.386 | vanishes |
| spectators, upper (#22) | -0.00467 | +5.19e-6 (-0.00) | 0.27 | 0 | -1.62e-5 | -0.0009 | 0.568 / 0.387 | vanishes |
| sticky record, eps = 0.027 | -0.00612 | +2.96e-5 (-0.00) | 1.00 | 0 | +1.49e-4 | +0.0261 | 0.582 / 0.385 | vanishes |
| sticky record, eps = 0.01 | -0.00660 | +1.31e-5 (-0.00) | 1.00 | 0 | +2.41e-5 | +0.0091 | 0.582 / 0.385 | vanishes |
| 3-state memory outside q75 | -0.00600 | -6.00e-3 (1.00) | 1.00 | -6.00e-3 | -6.00e-3 | -0.0009 | 0.579 / 0.579 | persists |
| A2 keeps 50% coherence | -0.00690 | -4.68e-4 (0.07) | 1.00 | 0 | -7.10e-5 | -0.0009 | 0.582 / 0.483 | ambiguous |
| A2 keeps 80% coherence | -0.00690 | -2.66e-3 (0.39) | 1.00 | 0 | -7.10e-5 | -0.0009 | 0.582 / 0.542 | persists |

The hypotheses map to these outcomes:
- **Vanishes:** H_q75; drift by episode; q59 residue and leakage, fitted or extreme;
  spectators, typical and upper; the sticky record. These differ in the secondary tests:
  - A2s_spec's ratio is 1.0 except under spectators (0.78 typical, 0.27 upper);
  - D1's lag-1 excess is −0.0009 except under the sticky record (+0.026, +0.009) and q59
    leakage (+0.017, +0.050);
  - A3 shifts by +1e-4 to +3e-4 under leakage and stickiness (1–4 SE).
- **Persists:** an unmodelled 3-state memory outside q75 (A1 and A3 persist too), and A2
  keeping 80% of q75's coherence (A1 and A3 vanish).
- **Ambiguous or uncertain cells:**
  - **A2 keeping 50% of the coherence** gives w_A2 = −4.7e-4, 6.8% of the baseline, at the
    vanish bound. In the validation the rule called it *vanishes* in 40% of the replicates and *ambiguous* in 60%. The rule cannot tell coherence kept at ≤ 50% from complete
    dephasing. A vanishing A2 therefore bounds the surviving coherence, not "none".
  - **#19's best drift mixture** gives w(0) = −0.00123 (21% of the episode-free device witness) in every PMD-cadence arm. In the
    replicates the gate passes, in 100% of the replicates (SE 6.9e-5), and A2 persists, in 100%. This null is
    already excluded by its shape (g misses by |z| ≈ 55, #19). Its w(1..3) ≈ 0 will also
    show it in the baseline's own data.

**What each outcome would and would not show:**

| outcome | rung 4 | would show | would not show |
|---|---|---|---|
| gate fails | – | the baseline does not replicate Phase 6 at 5σ: a replication result | anything about q75 |
| vanishes, sham equal | **supported** | the rank > 2 output statistics need q75's coherence (or its interplay with the classical loop) | "quantum memory" in a stronger sense; that no other path exists below the vanish bound (≈ 0.0005); rung 5 |
| vanishes, sham reduced | supported, with crosstalk | as above, plus q74's readout dephases q75 by the measured amount | that crosstalk explains the witness |
| persists, A1/A3 vanish | not supported as tested | A2's measurement is incomplete; q75's state still matters (A1/A3) | memory outside q75 |
| persists, all persist | **refuted** for q75 alone | effective hardware memory with a carrier outside q75's coherence | which carrier (q59, leakage, spectators, ≥ 3-regime drift) |
| ambiguous | no move | a partial loss or too little power; effect sizes only | a fraction of memory attributable to q75 (w is nonlinear in coherence) |

## 6. Validation on synthetic data

For each hypothesis, 200 synthetic datasets were drawn from its exact models (2,400 in
total). Each used the planned sizes and the exact schedule: 4 jobs, 124 pubs, pubs in their
seeded order, and the drift episode contiguous in time. Each dataset went through the full
pipeline: guard rails, per-arm statistics with the stratified shot and block bootstraps, the
segment policy per job, the segment sensitivity, the decision and the Holm family. The
validation used 200 resamples and 100 permutations per replicate. The *predicted* outcome is
the rule applied to the exact w(0) values with the median SEs. The error rate is the fraction
of replicates whose outcome differs from it. The secondary columns give each test's Holm
rejection rate, which is its size where its null holds and its power otherwise
(`validation.csv`, `validation_replicates.csv`).

| hypothesis | predicted | gate passed | vanishes | persists | ambiguous | **error rate** | segment-sensitive | spec ≠ base | A1 ≠ 0 | A3 ≠ pred | D1 w ≠ 0 | D1 lag-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_q75 (calibrated) | vanishes | 100.0% | 100.0% | 0 | 0 | **0** | 0 | 0.5% | 0 | 0 | 0 | 0 |
| drift: #19 episode regime, 13.5% | vanishes | 100.0% | 97.0% | 0 | 3.0% | **3.0%** | 3.0% | 2.0% | 0 | 0 | 0 | 0.5% |
| drift: #19 best K = 3 mixture | persists | 100.0% | 0 | 100.0% | 0 | **0** | 0 | 0 | 100.0% | 100.0% | 0 | 0.5% |
| q59 reset + leakage, #21 fit | vanishes | 100.0% | 77.5% | 0 | 22.5% | **22.5%** | 12.5% | 0 | 0 | 3.5% | 0.5% | 100.0% |
| q59 leakage, #21 extreme | vanishes | 100.0% | 44.5% | 0 | 55.5% | **55.5%** | 10.0% | 0.5% | 0 | 4.0% | 0.5% | 100.0% |
| spectators, typical (#22) | vanishes | 100.0% | 100.0% | 0 | 0 | **0** | 0 | 100.0% | 0 | 0 | 0 | 0.5% |
| spectators, upper (#22) | vanishes | 100.0% | 97.0% | 0 | 3.0% | **3.0%** | 2.5% | 100.0% | 0 | 0 | 0 | 0.5% |
| sticky record, eps = 0.027 | vanishes | 100.0% | 89.5% | 0 | 10.5% | **10.5%** | 3.5% | 0 | 0 | 3.0% | 0.5% | 100.0% |
| sticky record, eps = 0.01 | vanishes | 100.0% | 100.0% | 0 | 0 | **0** | 0 | 1.5% | 0 | 0 | 0 | 100.0% |
| 3-state memory outside q75 | persists | 100.0% | 0 | 100.0% | 0 | **0** | 0 | 0.5% | 100.0% | 100.0% | 0 | 0.5% |
| A2 keeps 50% coherence | ambiguous | 100.0% | 40.0% | 0 | 60.0% | **40.0%** | 3.0% | 0.5% | 0 | 0 | 0 | 0 |
| A2 keeps 80% coherence | persists | 100.0% | 0 | 100.0% | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0.5% |

- **The rule never gives the wrong decisive call.** No replicate says *persists* under any
  hypothesis where q75 carries the memory. No replicate says *vanishes* under the memory
  outside q75, or under A2 keeping 80% of the coherence. Every error is an *ambiguous*,
  which is conservative.
- **Where the power is short.**
  - Under q59 leakage, the fit (22.5%) and the extreme (55.5%), and under the ε = 0.027
    sticky record (10.5%), A2's SE about doubles (1.6–1.8e-4) and the baseline shrinks.
  - The vanish bound is then within 2 SE, so these cases fall to *ambiguous*.
  - The secondary tests still identify them: D1's lag-1 excess rejects in 100% of the
    replicates under every leakage and stickiness hypothesis.
  - Doubling A2 to 20,000 shots (+5.6 s estimated) would recover most of this power. That
    is an option for the owner; it is not in the default plan.
- **Size of the secondary tests under H_q75:** ≤ 0.5% for every test. Elsewhere it reaches 2% at most, for the
  sham under drift, where the episode hits arms unevenly.
- **Segment policy.** At least one job is flagged in 24% of the H_q75 replicates (5 tests ×
  2 arms × 4 jobs). It never changed the outcome. Under drift every replicate flags the
  episode jobs, and 3% become *ambiguous (segment-sensitive)*.
- **Guard rails.**
  - Drift flags fire in 11% of the H_q75 replicates. Under drift they fire in every
    replicate.
  - Under extreme leakage, the duplicates rail fired in every replicate, because leaked q59
    gives runs of all-ones shots. Duplicates are therefore a warning to inspect, not an
    invalidation; only malformed values invalidate a dataset.
- **A known bias.** For the near-i.i.d. arms (A1, D1), the mean of ŵ(0) lies 10 standard
  errors of the mean from the exact 0. This is second-order bias at rank ≈ 1, about 1.5e-6
  for D1 (0.5 SE) and 4.6e-5 for A1 (0.45 of A1's SE, 1.0e-4; corrected in #25 from an
  earlier "about 1e-5"). It is below half an SE, so no test is affected (rejection rates of
  0–0.5%).

## 7. Budget

The cost model is Phase 6's: 2 s per job plus (250 µs + duration) per shot. The upper bound
is 1.25 × the estimate + 1 s billing rounding per job; Phase 6 was billed 14 s for 13.2 s.

| item | jobs | shots | estimate (s) | upper bound (s) |
|---|---|---|---|---|
| pilot | 1 | 1,000 | 2.6 | 4.2 |
| each main job | 1 | 14,524 | 10.1 | 13.7 |
| tier 1 alone (4 jobs) | 4 | 44,000 | 32.6 | 44.8 |
| tier 2 (added to the same jobs) | 0 | 14,096 | +7.9 | +9.8 |
| **default plan: tiers 1 + 2** | 4 | 58,096 | **40.5** | **54.6** |
| **pilot + default plan** | 5 | 59,096 | **43.1** | **58.8** |
| tier 3: B2 | 1 | 15,000 | 12.5 | 16.7 |
| tier 3: D3 | 1 | 5,000 | 4.8 | 7.0 |
| tier 3: C3 (2 arms) | 2 | 40,000 | 26.4 | 35.0 |

- Every job's estimate is ≤ 20 s, Phase 6's `max_qpu_seconds`. The largest is 10.1 s, and
  the guard is kept per job.
- **Proposed hard cap: 60 QPU seconds** for the pilot and the default plan, about 10% of the
  ≈ 584 s left on the Open plan after #16. Tier 3 would need a separate approval (+59 s upper
  bound for all three).
- Before each submission the code refuses unless all of these hold:
  - the instance's plan is `open` (**never a paid plan**);
  - the live remaining allowance is known and at least the job's upper bound;
  - the seconds counted for the jobs already submitted, plus this job's upper bound, stay
    within the hard cap. Each submitted job counts max(IBM-reported charge, its upper bound);
    a charge not yet reported counts as the upper bound (added in #25);
  - no submitted job was charged more than its upper bound: if one was, the run stops for
    good (`usage_stop.json`) and is reported on #17; if any job's usage can't be read at all,
    the next job is refused (#25). The charge is re-read before every submission, from
    `job.usage()`, the job's metrics and the instance's usage delta, taking the largest;
  - the job's estimate is ≤ 20 s;
  - the calibration is within its bounds;
  - `approval_url` names the owner's comment, and `--submit` is given.
- A submitted job is re-fetched by its id and never resubmitted.

## 8. Data preservation and reproducibility

Stage 4 writes `results/phase7_hw/<timestamp>/`, one directory per job. Each directory holds:
- `job.json`: id, pubs, estimate and upper bound, usage before and after, the reported
  charge and what it counts against the cap (`charge`), the live
  calibration before and after, the calibration check, metrics, status, and the execution
  spans;
- `shots.npz`: every pub's bit-packed `y`, plus `m` (A2) and `s_0` (A2s_spec), in returned
  order.

The run directory also holds:
- `calibration_before.json`;
- `circuits.json` (the hash check, the re-timing and every SHA-256);
- `pilot_predictions.json`;
- `analysis.json`.

`analyse_run` copies all of these, with the log, to `results/summaries/phase7_hw/`. The
report goes in `docs/phase7-hardware.md`, and failed or stopped runs are reported too. Every
saved result carries `run_metadata`: commit, dirty flag, environment, config and seed.

Seeds:
- schedule 2024;
- pilot 1;
- analysis bootstrap 24;
- A3 instances 2020 (#20);
- validation 4242.

**Deviations a reader should know.**
- A3 is tested against its realised-instance prediction (−7.1e-5), not 0.
- The validation used 200 bootstrap resamples and 100 permutations per replicate; the real
  analysis uses 1,000 of each.
- The spectator hypothesis is a reduced model (§5).
- The hypotheses are exact models; none of them has been measured.

## 9. Amendment of 2026-11-08 (post hoc, after #25's pilot; #26)

**This section is post hoc.** It was written after #25's pilot (`results/summaries/phase7_hw/`,
`docs/phase7-hardware.md`) stopped at the gate of §3, and it uses that pilot's data. It was
committed to `main` before any new submission. `--submit` refuses unless the section's SHA-256
and its commit, recorded in `experiments/configs/phase7_hw_4b.toml` `[amendment]`, match the
section as committed on `main` (`phase7_hw.amendment_guard`). The owner approved this route
and its budget on #17 (https://github.com/alsiola/quantum-ai/issues/17#issuecomment-6720604627).

### 9.1 Reason

#25's pilot failed three of the four conditions of §3:
- (b) baseline P(1) 0.5517 ± 0.0048 against 0.5810 ± 0.02;
- (c) A2's q75 record P(m = 1) 0.4190 ± 0.0053 against 0.4925 ± 0.03;
- (d) execution span per shot 1.304 ms against 0.559 ms ± 25%.

(c) says the calibrated model of q75 under repeated mid-circuit measurement is incomplete.
(b) is a shift of the device's output on the scale of a job, the kind #19 describes. (d) compared
a span (with ≈ 0.7 s of fixed overhead) against a reference that is
`circuits_execution_time_ns / shots`: a mis-specified condition.

### 9.2 The refit (step 1 of #26)

`experiments/phase7_refit.py` (`qmem.refit`; config `experiments/configs/phase7_refit.toml`;
outputs in `results/summaries/phase7_hw_4b/refit*`). A2's joint record (q75's bit m_t and
q59's output y_t in each of 64 steps, 500 shots) is a linear process with four symbols on the
(q75, q59) memory, built from #20's pieces rebuilt under #25's live calibration. Families, fitted
by maximum likelihood on whole shots:

| family | parameters | ΔNLL vs calibrated | P(m=1) step 0 | P(m=1), t ≥ 16 | A2 P(1) |
|---|---|---|---|---|---|
| *#25's pilot* | | | 0.788 ± 0.018 | 0.4190 ± 0.0052 | 0.3817 ± 0.0036 |
| calibrated | 0 | 0 | 0.830 | 0.4925 | 0.386 |
| decay: q75 \|1⟩ → \|0⟩ in each measurement, recorded 0 | 1: a = 0.119 | −267.9 (LR 536) | 0.732 | 0.4211 | 0.380 |
| **decay, first measurement separate** | **2: a = 0.123 ± 0.008, a0 = 0.051 ± 0.022** | **−272.3** (LR 8.8 vs 1 par., p = 0.003) | **0.788** | **0.4191** | **0.380** |
| leakage \|1⟩ → \|2⟩ per measurement, \|2⟩ read 0, seeping back (`qmem.ancilla`) | 2: 0.079, seep 0.23 | −247.1 | 0.830 | 0.4210 | |

(± are whole-shot bootstrap SDs, 500 resamples.) The refit is the 2-parameter family: in every
mid-circuit measurement of q75 after the first, its |1⟩ decays to |0⟩ with probability
a = 0.123 (95% CI 0.108–0.137) early enough to be recorded 0. The first measurement of a shot
has a0 = 0.051 (0.011–0.095). a0 only shapes steps 0–2; it changes every statistic at t ≥ 16
by < 1e-12, so the stationary arm models use a alone. Leakage fits worse with as many
parameters and cannot move step 0.

**Checks.**
- The refit reproduces #25's q75 record within its bootstrap uncertainty: step 0 (z = 0.0),
  step 1 (z = −0.1) and the mean over t ≥ 16 (z = 0.0). A2's output P(1) (0.380, z = 0.5),
  P(y | m = 1) and P(y | m = 0) (|z| ≤ 1.0) also agree.
- **It is not complete.** The step-to-step record transitions P(m_{t+1} = 1 | m_t, y_t) miss by
  |z| = 3.0–3.4 (calibrated: 6–12). The device's q75 record is more persistent than either
  model. Under every q75 model in §5, A2's q75 is a classical bit each step, so A2's output
  is a 2-state process and its witness is 0 exactly whatever these transitions are. The
  decision-relevant prediction does not depend on this misfit.
- The term acts only on A2, the one arm that measures q75, so the baseline's prediction is
  unchanged (0.5810 at #25's calibration). #25's baseline (0.5517) is not explained by q75
  and is treated under (b).
- The new term is tested against a closed-form Markov chain, enumeration of the joint record,
  #20's record model at a = 0, #21's independent qutrit simulator (`StepConfig.memory_decay`),
  and parameter recovery on simulated records (`tests/test_refit.py`).
- At #16's snapshot calibration the refit gives P(m = 1) = 0.4201 (calibrated: 0.4925).

**Gate (c) prediction.** The refit's P(m = 1) over t ≥ 16 at the live calibration
(`live_pilot_predictions`, with a and a0 fixed from this fit and q75's live readout error).
Its uncertainty at #25's calibration is 0.0045 (SD over the bootstrap refits). A 500-shot pilot
adds 0.0052 (#25's whole-shot SE). The tolerance is 3 × √(0.0045² + 0.0052²) = 0.0206, fixed
at **0.021**.

### 9.3 Amended pilot conditions

The pilot itself is unchanged (one job; 500 baseline and 500 A2 shots; seed 1). Go only if all
of these hold:
- (a) unchanged.
- (b) **stop only if** the baseline's P(1) over positions ≥ 16 is more than **0.08** from the
  live prediction. Within 0.08, a deviation beyond the original 0.02 is **recorded as a
  consistency flag** and does not stop the run. Reasons:
  - the primary rule compares A2 with the *concurrent* baseline, interleaved in every job,
    and the replication gate (§4: baseline |w(0)| ≥ 5 SE) already refuses to interpret a
    baseline that does not reproduce the witness. A level shift in P(1) is not by itself a
    reason to stop;
  - the observed job-scale excursions are 0.02–0.04: Phase 6's episode +0.022, #25's pilot
    −0.029, and #25's two half-pilots 0.039 apart;
  - #19 showed that mean shifts up to ΔP(1) = 0.08 (three times the episode's), at weights
    0.135 or 0.5, give w(0) between 0 and +1.4e-5: they cannot produce the witness or hide its
    loss. Beyond 0.08 the shift is outside what #19 checked, and a broken circuit, decoding or
    mapping becomes the likelier reading, so the run stops.
- (c) A2's q75 record has P(m = 1), positions ≥ 16, within **0.021** of the **refit**
  prediction at the live calibration (§9.2: the record model with a0 and a fixed at #25's fit,
  the rest from the live noise). This is the out-of-sample test of the refit.
- (d) `circuits_execution_time_ns / shots` (IBM's job metric) is within **25%** of the cost
  model's per-shot time (0.559 ms). #25's pilot gave 0.568 ms (+1.5%).

On any failure the script writes `pilot/gate_failed.json` and stops: no retry, no re-mapping
and no second pilot without a new decision on #17.

### 9.4 Predictions and validation under the refit (step 2 of #26)

The local dry run under the refit (`--dry-run-local` with `experiments/configs/phase7_hw_4b.toml`;
outputs in `results/summaries/phase7_hw_4b/dryrun/`) re-checked all 68 circuit hashes of #20
(match) and recomputed every prediction of §5 and the validation of §6. The settings were
unchanged: 200 replicates per hypothesis, the planned schedule, 200 resamples and 100
permutations. The refit adds q75's measurement decay (a = 0.123) to A2 in every hypothesis
with a q75 model, both in #20's pieces and in #21's simulator. The hmm-kind hypotheses (#19's
mixture, the memory outside q75) are output-level and unchanged. No other arm changes.

**Predictions that move** (`predictions.csv`):
- **H_q75:** A2's w(0) stays 0 exactly. A2's P(1) is 0.379 (was 0.385; #25 measured 0.382).
  A2's median SE rises from 8.6e-5 to 9.3e-5. The A3 and D1 reference values are unchanged.
- **A2 keeping 50% of the coherence:** w_A2 = −3.0e-4 (was −4.7e-4), 4.4% of the baseline.
  It is now predicted *vanishes* (was *ambiguous*).
- **A2 keeping 80% of the coherence:** w_A2 = −1.71e-3 (was −2.66e-3), 24.8% of the baseline,
  just under the 25% persists threshold. It is now predicted *ambiguous* (was *persists*).
  The decay damps whatever coherence A2's measurement leaves, by √(1 − a) per step, and it
  moves population as well.
- The q59-residue and sticky-record cases keep |w_A2| ≈ 1–4e-5, slightly smaller than before.

**Validation** (error rate = the fraction of replicates whose outcome differs from the
predicted one; was → now):

| hypothesis | predicted | vanishes | persists | ambiguous | error rate |
|---|---|---|---|---|---|
| H_q75 (calibrated + refit) | vanishes | 100% | 0 | 0 | 0 → **0** |
| drift: #19 episode regime | vanishes | 96.5% | 0 | 3.5% | 3.0% → **3.5%** |
| drift: #19 best K = 3 mixture | persists | 0 | 100% | 0 | 0 → **0** |
| q59 reset + leakage, #21 fit | vanishes | 67.0% | 0 | 33.0% | 22.5% → **33.0%** |
| q59 leakage, #21 extreme | vanishes | 34.0% | 0 | 66.0% | 55.5% → **66.0%** |
| spectators, typical | vanishes | 99.5% | 0 | 0.5% | 0 → **0.5%** |
| spectators, upper | vanishes | 96.0% | 0 | 4.0% | 3.0% → **4.0%** |
| sticky record, ε = 0.027 | vanishes | 83.5% | 0 | 16.5% | 10.5% → **16.5%** |
| sticky record, ε = 0.01 | vanishes | 98.5% | 0 | 1.5% | 0 → **1.5%** |
| 3-state memory outside q75 | persists | 0 | 100% | 0 | 0 → **0** |
| A2 keeps 50% coherence | vanishes (was ambiguous) | 92.5% | 0 | 7.5% | 40% → **7.5%** |
| A2 keeps 80% coherence | ambiguous (was persists) | 0 | 41.0% | 59.0% | 0 → **41.0%** |

- **The decisive calls are unchanged in kind.** No replicate says *persists* where q75
  carries the memory. No replicate says *vanishes* under the memory outside q75 or under A2
  keeping 80% of the coherence. Every error is still an *ambiguous*, or, for the 80% case,
  a *persists* that its prediction (just under the threshold) calls ambiguous.
- **Power falls slightly.** A2's SE is ≈ 8% larger, so the cases near the vanish bound (q59
  leakage, the sticky record) fall to *ambiguous* 6–11 points more often. The secondary
  tests still identify them: D1's lag-1 excess rejects in 100% of those replicates. The gate
  passes in 100% of the replicates of every hypothesis, and the secondary sizes are ≤ 2%,
  as before.
- **The reading of a persisting A2 shifts.** With the decay, 80% coherence kept by A2's
  measurement gives 24.8% of the baseline's witness, at the persists threshold (it gave 39%
  before). So "A2 persists while A1 and A3 vanish" now needs more than about 80% of the
  coherence to survive. 50% kept now falls inside *vanishes* in 92.5% of the replicates (40%
  before), so a vanishing A2 bounds the surviving coherence less tightly. These are model
  statements under the refit. They do not change the rule.

### 9.5 Budget

The hard cap is **120 QPU seconds** in total for Phase 7 Stages 4 and 4b (#17). #25's pilot
counts 4.20 s against it (max of its 3 s charge and its upper bound;
`[budget] prior_counted_s`). The plan needs 4.20 s (pilot) + 54.6 s (four main jobs), as upper
bounds, so 63.0 s with #25's pilot. Every other guard of §7 is unchanged.

### 9.6 Unchanged

The decision rule and thresholds (§4), the arms and circuits (§2), the shot counts (pilot, then
tiers 1–2 with A2 at 10,000; no tier 3), the schedule and seeds (§3, §8), the mapping (q75 and
q59 on `ibm_kingston`, never re-mapped), the calibration bounds, the analysis settings, the
secondary family and its Holm correction, and the reading of A2 with the other arms. The
analysis is also reported under the original, un-refit predictions as a sensitivity row; the
decision uses the data alone, and the predictions only enter the positional χ² warning and
the A3 and D1 reference values, which the refit does not change.
