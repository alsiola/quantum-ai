# Phase 7 Stage 4: the q75 ablation follow-up on ibm_kingston (#25)

**Outcome: a clean stop at the pilot gate.** The pre-registered pilot (one job, 1,000
shots) ran on `ibm_kingston` on 2026-11-08 and failed three of the four gate conditions.
Following the pre-registration (`docs/phase7-preregistration.md` §3) and the approval's terms
on #17, the main jobs were **not submitted**, and nothing was retried or re-mapped. The
primary outcome (A2 against the concurrent baseline) and the Holm-corrected secondary tests
were therefore never computed. **Rung 4 stays unresolved.**

QPU time used: **3 s charged**, against 2.56 s estimated and a 4.20 s upper bound. That is 3 s of
the 60 s cap, with 581 s left on the Open plan.

Raw outputs are in `results/phase7_hw/20261108T084425Z/` (gitignored). Curated copies, with
`run_metadata`, are in `results/summaries/phase7_hw/`:
- `job_pilot.json`: id, pubs, estimate, upper bound, the usage before and after, the charge,
  calibration before and after, metrics and spans;
- `shots_pilot.npz`, `gate_pilot.json` and `pilot_diagnostics.json`;
- `calibration_before.json`, `circuits.json`, `pilot_predictions.json` and `log.txt`;
- `live_dry_run.json` with its log.

## What ran

1. **Guards added before any submission (#25).** These landed on `main` before the pilot, in
   cce1039..733d6ef.
   - The approval URL was recorded in the config.
   - Usage guard: each job's IBM charge is read from `job.usage()`, the job metrics'
     `qpu_charge_time_seconds` and the instance's consumed-seconds delta, taking the
     largest. The charge is re-read before every later submission. The cap counts
     max(charged, upper bound) per job, and an unreported charge counts as the upper bound.
     If any job is charged above its upper bound, the run stops (`usage_stop.json`). If a
     job's usage can't be read at all, the next job is refused.
   - `--stop-after-pilot` was added, so the pilot could be inspected before the main jobs.
     A resume reuses the pilot gate's verdict and never skips the gate.
   - The local dry run was re-run. Hashes match, and the only change is H_ancilla_extreme's
     integrity rate, 1.0 → 0.0. The A1 bias nit was fixed in the pre-registration.
2. **Live dry run** (`results/phase7_hw_livedry/20261108T084140Z/`; nothing submitted):
   - **Plan and allowance:** plan `open`, 584 s remaining. Every job would pass the budget
     guard: the pilot's upper bound is 4.20 s and each of the four main jobs' is 13.66 s.
   - **Circuits:** the hash check matches, and the live re-timing changes delays only.
   - **Calibration:** no violations for q59, q75, q74 or the q59–q75 CZ:

     | | readout error (live) | #16 snapshot |
     |---|---|---|
     | q75 | 0.0088 | 0.0151 |
     | q59 | 0.0059 | 0.0039 |
     | q74 | 0.0078 | no snapshot |

     T1, T2 and the CZ error are unchanged from #16's snapshot.
   - **Live predictions:** baseline P(1) 0.5810; A2's P(m = 1) 0.4925.
3. **Pilot** (`--submit --stop-after-pilot`): job `dbo3cttt7fqc73akg8dg`, created 08:44:40Z,
   ran 08:46:04–08:46:09Z, 2 pubs of 500 shots each (baseline, A2).

## QPU time against the estimates

| job | shots | estimate (s) | upper bound (s) | charged (s) | counted against the cap (s) |
|---|---|---|---|---|---|
| pilot | 1,000 | 2.56 | 4.20 | **3** (all three sources agree) | 4.20 |
| job0–job3 | 14,524 each | 10.12 each | 13.66 each | not submitted | 0 |

IBM's `circuits_execution_time_ns` gives 0.568 s, which is 0.568 ms per shot against the cost model's 0.559 ms
(+1.5%). The 3 s charge is the 0.57 s of circuits plus about 2.4 s of per-job overhead. That
matches Phase 6's 2 s overhead model to within the 1 s billing rounding.

## The pilot gate

| condition | got | pre-registered | result |
|---|---|---|---|
| (a) decodes, integrity rail (values) | both pubs (500 × 64), no malformed values, **no duplicate shots** | must pass | pass |
| (b) baseline P(1), positions ≥ 16 | **0.5517** ± 0.0048 (whole-shot SE) | 0.5810 ± 0.02 | **fail** (−0.029, ≈ 6 SE) |
| (c) A2's P(m = 1), positions ≥ 16 | **0.4190** ± 0.0053 | 0.4925 ± 0.03 | **fail** (−0.073, ≈ 14 SE) |
| (d) execution-span s per shot | 1.304 ms (span 1.304 s) | 0.559 ms ± 25% | **fail** (see below) |

Guard warnings: the baseline flagged `dispersion_p1` and `order_inflation_p1`; A2 flagged none.

**(b) is a real change in the device's output.** Phase 6 measured this same circuit at
P(1) = 0.5791 (pilot) and 0.5809 (main) on 2026-11-07, matching today's live prediction of 0.5810.
Today's 500 shots give 0.552, well beyond the 0.02 tolerance and the pilot's sampling error.
The shift is not uniform within the job either: P(1) is 0.532 over the first 250 shots and
0.571 over the last 250, a gap of about 4 SE. This is the kind of within-job excursion that #19's
regime-B episode describes, and Phase 6's episode moved P(1) by 0.027. The calibration
snapshots before and after the job are identical, except for the time they were read.

**(c) does not come from how the bits were decoded.** Reading A2's q75 record `m` in the
reverse bit order gives 0.4275, which also fails. P(m = 1) is 0.79 at step 0
and then stays at 0.37–0.44 (mean 0.419), not the predicted 0.49. A2's output
P(1) is 0.382; there is no pre-registered value for it. The 50-shot Aer pilot on
FakeKingston gave 0.376–0.379, so A2's output agrees with the noise model while its
q75 record does not. A steady-state P(m = 1) that is 0.07 low is what extra relaxation of q75 towards
|0⟩ between its mid-circuit measurements would give. For example, measurement-induced
or leakage-assisted decay that the noise model doesn't include. This is a reading, not a
test.

**(d) fails by design, not because the device was slow.** The gate divides the execution
span by the shot count. But the pre-registration's reference, Phase 6's "0.567 ms per
shot", is `circuits_execution_time_ns / shots` (11.345 s / 20,000), not a span. A span
includes a fixed overhead of about 0.7 s. Phase 6's own 200-shot pilot span was 2 s, which
would also have failed (d), but Phase 6 did not have this gate. On the comparable quantity,
the pilot ran at 0.568 ms per shot, +1.5% from the model. Condition (d) can't be met by a
1,000-shot job. This is a flaw in how the gate was specified, found only now. It
doesn't change the verdict: (b) and (c) fail independently.

## What happened next

The script wrote `pilot/gate_failed.json` and stopped. As pre-registered, a stopped run
refuses to resume, and re-running needs a new decision by the owner. There was no retry, no
re-mapping and no main job.

## What this does and doesn't imply

- **For rung 4 (is q75's coherence causally necessary?): nothing either way.** The ablation
  arms were never run, so there is no A2 against concurrent-baseline contrast, no secondary
  family and no decision. Rung 4 stays unresolved.
- **For the Phase 6 result:** the pilot doesn't re-test the witness. It has 500 shots and
  no w(0) is pre-registered for it. But the baseline's output statistics today differ from
  #16's by about 6 SE in P(1), and drift within the job is visible in 500 shots. That
  supports #19's conclusion: the device's output is nonstationary on the scale of a job, and any
  hardware contrast must be concurrent and interleaved, as this design was.
- **For the noise model:** the q75 record in A2 sits 0.07 below the calibrated
  prediction. That suggests the calibrated model of q75 under repeated mid-circuit
  measurement is incomplete. That matters for interpreting any A2 result, because *vanishes*
  and *persists* were calibrated against that model.

## Options for the owner (each needs a fresh approval on #17)

1. **Accept the stop.** Publish the claim ladder with rung 4 unresolved, attaching this
   pilot and #24's plan.
2. **Repeat the pilot unchanged, with condition (d) fixed** (`circuits_execution_time_ns` per
   shot, which is what the reference measures). If (b) and (c) pass on another day, run the
   approved main jobs. About 3 s.
3. **Revise the predictions before any rerun:** refit q75's relaxation under mid-circuit
   measurement from this pilot's `m` record, then re-validate the decision rule (Stage 3
   style, local only). Only then would a new pilot be meaningful for (c).

Rerunning, changing a tolerance or changing the gate needs your decision. Stage 4 does not
make it.

# Stage 4b: refit, amended gate, re-pilot (#26)

The owner chose option 3 above and approved a 120 s cap on #17
(https://github.com/alsiola/quantum-ai/issues/17#issuecomment-6720604627).

## Refit and amendment (local, no QPU)

The full account is the dated, post hoc amendment in `docs/phase7-preregistration.md` §9.
Outputs are in `results/summaries/phase7_hw_4b/`: `refit.json`, `refit_check.csv`,
`refit_bootstrap.npz`, `refit_m_by_position.png`, and `dryrun/`.

- **Refit (2 parameters).** In each mid-circuit measurement of q75, |1⟩ decays to |0⟩ and is
  recorded 0 with probability a = 0.123 ± 0.008; the first measurement of a shot has
  a0 = 0.051 ± 0.022. Against #25's record:
  - P(m = 1) at step 0: 0.788 against 0.788 observed;
  - over t ≥ 16: 0.4191 against 0.4190;
  - A2's P(1): 0.380 against 0.382.

  The record's step-to-step transitions still miss at |z| ≈ 3, so the refit is not complete.
  A2's witness prediction (0) does not depend on that. Leakage fits worse.
- **Amended gate.**
  - (c) is the refit's P(m = 1) at the live calibration, ± 0.021 (0.419 at #25's calibration).
  - (b) stops only beyond ±0.08; ±0.02 is now a recorded consistency flag.
  - (d) is `circuits_execution_time_ns / shots`, ± 25%.
- **Unchanged:** the decision rule, the arms, the shots, the mapping and the analysis.
- **Validation under the refit.** No wrong decisive call. A2's SE is about 8% larger.
  *Ambiguous* rates rise under q59 leakage (fit 22.5% → 33%, extreme 55.5% → 66%) and the
  sticky record (10.5% → 16.5%). "A2 keeps 80% coherence" moves from *persists* to
  *ambiguous*.

## Hardware: a clean stop at the amended gate

**Outcome: the amended pilot failed gate (c).** The out-of-sample test of the refit failed, so
the main jobs were **not submitted**. Nothing was retried or re-mapped, and there was no second
pilot. The primary outcome and the secondary family were never computed. **Rung 4 stays
unresolved.**

**What ran** (after the coordinator's go on #26):
1. **Local dry run.** It was run under the refit at 13:35Z (`dryrun/`), and all 68 hashes
   match.
2. **Live `--dry-run`** at 13:41Z (`live_dry_run.json`):
   - plan `open`, with 581 s remaining;
   - calibration identical to #25's snapshot, with no violations;
   - every job within the guard.
3. **Pilot** (`--submit --stop-after-pilot`): job `dbo7oi0qtr0c73ae4hk0`, 1,000 shots
   (500 baseline, 500 A2).
   - It was created 13:42Z and queued until 14:15Z. It ran 14:15–14:46Z, as IBM's timestamps
     report.
   - **Charged 3 s** (all three sources agree), counted 4.20 s.

**Phase 7 Stage 4/4b QPU total:** 6 s charged and 8.40 s counted, against the 120 s cap; 578 s
are left on the Open plan.

| condition (amended, §9.3) | got | criterion | result |
|---|---|---|---|
| (a) decoding, integrity | both pubs 500 × 64, no flags | must pass | pass |
| (b) baseline P(1), t ≥ 16 | **0.5646** ± 0.0039 | stop beyond 0.5810 ± 0.08; flag beyond ± 0.02 | pass, **consistent** (−0.016) |
| (c) A2's P(m = 1), t ≥ 16 | **0.4927** ± 0.0041 | refit 0.4191 ± 0.021 | **fail** (+0.074, ≈ 18 SE) |
| (d) `circuits_execution_time_ns` / shot | 0.568 ms | 0.559 ms ± 25% | pass (+1.5%) |

**(c) failed in the opposite direction to #25.** The live calibration was identical to #25's,
so the refit predicted the same value as on #25's record. Today the record matches the
*original* calibrated model instead:
- P(m = 1) after the burn-in is 0.4927 against 0.4925 calibrated;
- step 0 is 0.800 against 0.830 calibrated, 0.788 refit;
- both halves of the pilot give 0.493.

Refitting this pilot, descriptively, gives a = 0 (decay_first: a0 = 0.037, a < 1e-10). On these
data the refit is 298 nats worse than the calibrated model. The step-to-step transitions,
which the refit missed on #25, are also close to the calibrated model today (`pilot_diagnostics.json`).
A2's output P(1) is 0.393 ± 0.004 (calibrated 0.386).

**Reading.** This is the null the issue named: *the refit is wrong, or A2 is unstable.* The
data favour instability. About five and a half hours apart (08:46Z and 14:15Z), with the same reported calibration, q75 under
repeated mid-circuit measurement behaved in two different ways:
- on #25's run, about 12% extra decay per measurement;
- today, none.

The calibration snapshot does not show the change. So it is a time-varying property of the
device, on the scale of hours, that IBM's calibration data do not track. A model fitted on one
pilot therefore cannot be expected to predict the next one. This is a reading, not a
pre-registered test.

**What it implies.**
- **For rung 4: nothing either way.** No ablation contrast was measured.
- **For a future attempt:**
  - The q75 record is not a stable calibration target, so a gate on its absolute level is
    fragile. Today's pilot would have passed #24's *original* (b) and (c) (0.5646 against
    0.5810 ± 0.02; 0.4927 against 0.4925 ± 0.03), and it fails only the refit.
  - The decision rule does not use the q75 record at all. A2's witness is compared with the
    concurrent baseline, and under every q75 model the predicted A2 witness is 0, whatever
    the decay.
  - A gate on decoding, timing and the baseline alone would have let today's pilot through.
    Choosing that needs a new decision on #17. Stage 4b does not make it.
- **For the device:** both baseline pilots (#25: 0.552; today: 0.565) sit below Phase 6's
  0.579–0.581. Today's sits within the 0.02 consistency band.

**Options for the owner** (each needs a new decision on #17):
1. **Accept the stop** and publish with rung 4 unresolved.
2. **Re-pilot under #24's original (b)/(c) with (d) fixed.** On today's evidence this would
   pass, but its outcome depends on which q75 regime the device is in. About 3 s.
3. **Drop (c) as a stop condition.** Keep it as a recorded diagnostic; A2's q75 record does
   not enter the decision rule. Run the pilot and the main jobs. About 3 s + 54.6 s (upper
   bounds) within the remaining 111.6 s of the cap.

# Stage 4c: gate (c) as a recorded diagnostic, the pilot's own q75 refit (#27)

The owner chose option 3 above on #17
(https://github.com/alsiola/quantum-ai/issues/17#issuecomment-6723526792). The full account is
the dated, post hoc amendment in `docs/phase7-preregistration.md` §10. The config is
`experiments/configs/phase7_hw_4c.toml`; outputs are in `results/summaries/phase7_hw_4c/`.

## Amendment and validation (local, no QPU)

- **Gate.** (a), (b) and (d) stop the run exactly as in Stage 4b. (c) is recorded against
  the calibrated model and #26's refit (± 0.021 each) and never stops it.
- **q75 model.** After the gate, q75's decay is refitted on the pilot's own A2 record and
  fixed in `pilot/refit_pilot.json` before any main job is submitted. It is the analysis's
  q75 model (A2's predicted P(1) and marginals, the partial-coherence reading). The calibrated
  model and #26's refit are sensitivity rows. The decision uses the data alone, so it is the
  same under all three.
- **Guard.** `--submit` checks both §9 and §10 against their recorded hashes at their commits
  on `main`.
- **Unchanged:** the decision rule, the arms, the shots, the schedule, the mapping, the
  analysis, the secondary family and the replication gate.
- **A simulated 500-shot pilot recovers a**: 0.0023 ± 0.0033 at a = 0, 0.1234 ± 0.0051 at
  a = 0.1225. It separates the two regimes in every replicate.
- **Validation** (`dryrun/validation.csv`): 12 hypotheses × 2 q75 decays (0 and #26's 0.1225)
  × 200 replicates, each analysed with its own simulated pilot's fit. **No wrong decisive call
  in any of the 4,800 replicates.** The replication gate passes in all of them. Every error is
  an *ambiguous*, except the 80%-coherence case at a = 0.1225, which sits at the persists
  threshold (43% *persists*, 57% *ambiguous*). *Ambiguous* rates under q59 leakage and the
  sticky record are 14–69%, as before. D1's lag-1 excess flags those cases in every replicate.
  All 68 circuit hashes match.

## Hardware

**Outcome: A2 *vanishes*; rung 4 is supported.** The pilot passed (a), (b) and (d), and the
four main jobs ran without a calibration violation or a usage stop. The pre-registered
analysis (§4, unchanged) gives *vanishes*. A2s_spec does not differ from the baseline, so the
reading is "A2 vanishes, sham equal". The decision is the same under all three q75 models.
Curated outputs: `results/summaries/phase7_hw_4c/` (`summary.json` is the run's
`analysis.json`; `gate_pilot.json`, `refit_pilot.json`, `job_*.json`, `shots_*.npz`, `log.txt`).

**What ran** (after the coordinator's go on #27), all at `13ef8f1`:
1. **Local dry run** and **live `--dry-run`** (2026-11-08 19:27Z): all 68 hashes match; no
   calibration violations; every job within the guard.
2. **Pilot** `dbocqblt7fqc73cmtd6g` (500 baseline + 500 A2 shots). Submitted 19:27Z, it
   queued for about 14 h and ran at 10:08Z on 2026-11-09. Charged 3 s, counted 4.20 s.
3. **Pilot refit** written to `pilot/refit_pilot.json` at 10:13Z, before any main job.
4. **Main jobs**, one after another, each with a live calibration check before it:

| job | id | ran (UTC, 2026-11-09) | shots | charged | counted | violations |
|---|---|---|---|---|---|---|
| job0 | `dborq95t7fqc7388b3h0` | 12:44–13:14 | 14,524 | 11 s | 13.66 s | none |
| job1 | `dboseett7fqc7388c8p0` | 14:41–14:42 | 14,524 | 11 s | 13.66 s | none |
| job2 | `dbotns0qtr0c738dee10` | 15:18–15:41 | 14,524 | 11 s | 13.66 s | none |
| job3 | `dbouj7tt7fqc7388fphg` | 15:42–16:10 | 14,524 | 11 s | 13.66 s | none |

Every job's `circuits_execution_time_ns` is 0.568 ms per shot, as in the pilot. The live
calibration was the same for all four jobs (q75 readout 0.0127, T1 253 µs, T2 132 µs; q59
readout 0.0020).

**QPU total.** Stage 4c: 47 s charged and 58.82 s counted. Stage 4/4b/4c: 53 s charged and
**67.22 s counted against the 120 s cap**, as planned. 531 s are left on the Open plan.

### The pilot gate

| condition (§10) | got | criterion | result |
|---|---|---|---|
| (a) decoding, integrity | both pubs 500 × 64, no flags | must pass | pass |
| (b) baseline P(1), t ≥ 16 | 0.5666 | 0.5810 ± 0.08 (flag at ± 0.02) | pass, consistent (−0.014) |
| (c) A2's P(m = 1), diagnostic | 0.4891 | calibrated 0.4925 ± 0.021; #26's refit 0.4191 ± 0.021 | within calibrated; outside refit26; does not stop |
| (d) execution time / shot | 0.568 ms | 0.559 ms ± 25% | pass (+1.5%) |

**go = true.**

**The pilot's own q75 refit** (`decay_first` family, 500 A2 shots, live pieces): decay
**a = 7.8e-9** (bootstrap 95% CI [2e-10, 0.011]) and decay_first 0.017 (CI [3e-8, 0.068]).
It improves on the calibrated model by 0.34 nats. So the q75 record looked like the
calibrated model again, as on #26's pilot. The ~12% decay per measurement on #25's run did
not come back. Across the main jobs A2's P(m = 1) is 0.4964, also close to the calibrated
model.

### The primary outcome and the replication gate

| arm | shots | P(1) | w(0) | SE | z | H_q75 w(0) |
|---|---|---|---|---|---|---|
| baseline | 15,000 | 0.5658 | −5.15e-3 | 1.27e-4 (block) | −40.5 | −6.90e-3 |
| A2 | 10,000 | 0.3818 | −2.04e-4 | 1.41e-4 | −1.45 | 0 |
| A2s_spec | 15,000 | 0.5768 | −5.38e-3 | 1.18e-4 | −45.5 | −6.90e-3 |
| A1 | 4,000 | 0.3474 | −1.29e-4 | 1.86e-4 | −0.70 | 0 |
| A3 | 64 × 64 | 0.3815 | −2.11e-5 | 9.5e-5 (block) | −0.22 | −7.10e-5 |
| D1 | 10,000 | 0.5787 | −6.2e-6 | 4.5e-6 | −1.37 | 0 |

- **Replication gate: passes.** The baseline's w(0) is −5.15e-3 at z = −40.5. It is 75% of
  the H_q75 prediction (−6.90e-3); its P(1) is 0.566 against 0.582.
- **Primary: *vanishes*.** |w_A2| + 1.96 SE = 4.80e-4, below 0.10 |w_base| = 5.15e-4. A2's
  witness is 4.0% of the baseline's (95% CI [0, 9.3%]). The margin is narrow: the bound is
  93% of the threshold.
- **Segment sensitivity was not triggered.** No job is flagged in the baseline or in A2, so
  the primary outcome on unflagged jobs is the same by construction. Per job, A2's w(0) is
  −3.0e-4, −1.3e-4, −2.4e-4 and −2.8e-4; the baseline's is −5.1e-3, −5.8e-3, −5.7e-3 and
  −5.6e-3.
- **A2's output P(1)** is 0.3818: 0.3852 predicted by the pilot fit and the calibrated model,
  0.3791 by #26's refit.

### Secondary family (Holm, α = 0.01)

| test | estimate | reference | difference | z | p | Holm |
|---|---|---|---|---|---|---|
| A2s_spec − baseline | −5.38e-3 | −5.15e-3 | −2.3e-4 (ratio 1.044 ± 0.034) | −1.32 | 0.19 | not rejected |
| A1 against 0 | −1.29e-4 | 0 | −1.29e-4 | −0.70 | 0.49 | not rejected |
| A3 against its realised H_q75 value | −2.11e-5 | −7.10e-5 | +5.0e-5 | +0.52 | 0.60 | not rejected |
| D1's w(0) against 0 | −6.2e-6 | 0 | −6.2e-6 | −1.37 | 0.17 | not rejected |
| D1's lag-1 excess against prediction | −0.0045 | −0.0009 | −0.0036 | −2.46 | 0.014 | not rejected |

Nothing in the family is rejected. A1 and A3, read descriptively under A2's thresholds, both
*vanish*. D1's lag-1 excess is the closest to the line, and it is in the **negative**
direction. The sticky record (+0.009 to +0.026) and q59 leakage (+0.017 to +0.050) predict a
positive excess, so neither is indicated.

### Guard rails and segments (reported, not in the rule)

- No integrity failures, no duplicate shots and no decoding warnings in any arm.
- **The baseline's P(1) moved between jobs**: 0.577, **0.541**, 0.573, 0.573. This sets its
  dispersion and order-inflation flags. Its w(0) stayed within −5.1e-3 to −5.8e-3, and
  neither its segment tests nor A2's flag a job. A2s_spec, run in the same jobs, does not
  show the dip in job1 (0.574).
- A2s_spec's segment tests flag jobs 1 and 3 and its dispersion of g(1) − g(3) is high. It
  is not part of the segment-sensitivity rule. Its pooled ratio to the baseline is 1.04.
- A3's flags come from its 64 different circuits, which the guard rails expect (§4: only
  integrity applies).

### The partial-coherence reading, by q75 model

The decision does not depend on the model (`decision_identical_across_models` is true). The
model only changes how A2's ratio is read as surviving coherence c. The observed ratio is
0.040 (95% CI [0, 0.093]).

| q75 model | decay a | A2 P(1) predicted | ratio at c = 0.5 | ratio at c = 0.8 | c consistent with the CI |
|---|---|---|---|---|---|
| **pilot fit** (the analysis's model) | 7.8e-9 | 0.3852 | 0.068 | 0.386 | 0 to 0.55 |
| calibrated (sensitivity) | 0 | 0.3852 | 0.068 | 0.386 | 0 to 0.55 |
| #26's refit (sensitivity) | 0.1225 | 0.3791 | 0.044 | 0.248 | 0 to 0.60 |

On the analysis's model, A2's measurement leaves at most about 55% of q75's coherence. That
rules out H_partial_c08 (80% kept). It cannot tell 50% kept from full dephasing, as §4
foresaw. The bound is 60% under #26's refit.

### What this does and doesn't imply

- **Supported (rung 4):** the rank > 2 output statistics need q75's coherence, or its
  interplay with the classical loop. Measuring q75 in Z every step (A2), resetting it (A1)
  or randomising it (A3) each removes the witness to below 10% of the baseline. A sham that
  delays q75 and measures q74 (A2s_spec) leaves it unchanged.
- **Not shown:** "quantum memory" in a stronger sense; that no other path exists below the
  vanish bound (about 5e-4 in w(0), with c ≤ 0.55); a physical memory dimension (rung 5).
- **Caveats:**
  - The primary bound is narrow, at 93% of the threshold.
  - The baseline witness replicates at 75% of the H_q75 prediction, with a dip in P(1)
    during job1.
  - Gate (c) and the pilot-fitted q75 model come from the post hoc amendments §9 and §10.
    The decision rule itself is the pre-registered one and does not use the q75 record.
