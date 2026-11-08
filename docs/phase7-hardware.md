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
- **Validation under the refit.** No wrong decisive call. A2's SE is about 8% larger. "A2 keeps
  80% coherence" moves from *persists* to *ambiguous*.

## Hardware

Pending the coordinator's go on #26.
