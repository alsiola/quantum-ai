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
planned sizes, the primary SE under H_q75 is VAL_SE_LINE. "Vanishes" therefore holds for any
|w_A2(0)| below VAL_VANISH_LIMIT (VAL_VANISH_PCT of the calibrated baseline). "Persists" at
25% sits at ≈ VAL_PERSIST_Z SE. Moving shots between arms would buy nothing decisive, so the
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
  rails (values, duplicates) pass;
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
- integrity failures (values, duplicates) invalidate the dataset;
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

VAL_PRED_TABLE

The hypotheses map to these outcomes:
- **Vanishes:** H_q75; drift by episode; q59 residue and leakage, fitted or extreme;
  spectators, typical and upper; the sticky record. These differ in the secondary tests:
  - A2s_spec's ratio is 1.0 except under spectators (0.79 typical, 0.27 upper);
  - D1's lag-1 excess is −0.0009 except under the sticky record (+0.026, +0.009) and q59
    leakage (+0.017, +0.050);
  - A3 shifts by +1e-4 to +3e-4 under leakage and stickiness (1–4 SE).
- **Persists:** an unmodelled 3-state memory outside q75 (A1 and A3 persist too), and A2
  keeping 80% of q75's coherence (A1 and A3 vanish).
- **Ambiguous or uncertain cells:**
  - **A2 keeping 50% of the coherence** gives w_A2 = −4.7e-4, 6.8% of the baseline, at the
    vanish bound. VAL_C05. The rule cannot tell coherence kept at ≤ 50% from complete
    dephasing. A vanishing A2 therefore bounds the surviving coherence, not "none".
  - **#19's best drift mixture** keeps 18% of the device witness in every arm. In the
    replicates the gate passes, VAL_MIX_GATE, and A2 persists, VAL_MIX_PRIMARY. This null is
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

VAL_INTRO

VAL_TABLE

VAL_NOTES

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
  - the cumulative upper bound stays within the hard cap;
  - the job's estimate is ≤ 20 s;
  - the calibration is within its bounds;
  - `approval_url` names the owner's comment, and `--submit` is given.
- A submitted job is re-fetched by its id and never resubmitted.

## 8. Data preservation and reproducibility

Stage 4 writes `results/phase7_hw/<timestamp>/`, one directory per job. Each directory holds:
- `job.json`: id, pubs, estimate and upper bound, usage before and after, the live
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
