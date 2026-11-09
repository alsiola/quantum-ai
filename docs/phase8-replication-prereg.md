# Phase 8 Stage 1: independent-pair replication, pre-registration draft

Issue #31 (child of #28). Design and simulation only: no QPU, no IBM service, no submit path.
The device is FakeKingston's committed snapshot. Stage 4 executes it only after the owner
approves on #28.

```
uv run --group hardware python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --export
uv run python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --dry-run-local
uv run python experiments/phase8_replication.py --config experiments/configs/phase8_replication.toml --plan
```

Code: `qmem.replication`, `experiments/phase8_replication.py`; tests
`tests/test_replication.py`. Outputs: `results/summaries/phase8_replication/`.

Sections:

1. Hypotheses
2. Pairs and selection rule
3. Arms
4. Shots
5. Schedule
6. Primary statistics and decision rules
7. TOST margin and α
8. Multiplicity
9. Pilot gates
10. Drift and job-stratum policy
11. Ambiguous and failed pairs
12. Stop/go
13. Amendment log

## 1. Hypotheses

The question: does the q75/q59 result of #27 (rung 4 of `docs/phase7-memory-confounders.md`)
recur on pairs that share no qubit with it? Each chosen pair is tested on its own:

- **H1 (witness replicates).** The baseline's lag-0 witness is negative, w(0) < 0, and
  |w(0)| ≥ 5 drift-conditional SE.
- **H2 (memory attributable to the memory qubit).** A2, which measures and resets the memory
  each step, makes w(0) vanish (§6). The delay-only sham A2s is equivalent to the baseline on
  w(0) and lag1, and the spectator sham A2s_spec is equivalent on w(0) (TOST, §7).

A pair *replicates* when H1 and H2 both hold. g(k) is reported but is not part of H2: a ±10%
margin on it has no power under #27's drift (`sensitivity.csv`). These claims are about a
classical-or-quantum memory on the memory qubit, not quantumness; #30's classical model fits
every arm (`docs/phase8-claim-framework.md`).

## 2. Pairs and selection rule

The rule uses calibration and topology only, plus each pair's power from its own calibrated
model; no witness number enters (`Device.from_snapshot` keeps an allow-list of fields). Every
directed coupler of FakeKingston is a candidate. A candidate passes the screen when it is at
least 3 hops from q59, q75 and q74, and meets the T1, T2, readout, CZ and SX bounds in
`[selection]` of the config. It qualifies when its power P(H1 and H2) at the selection design
is at least 0.80. Two pairs are chosen in order of power, at least 3 hops apart.

Result (`candidates.csv`): 352 directed pairs, 87 pass the screen, 7 qualify. Chosen:

| pair (memory/output) | spectator | power at the design |
| --- | --- | --- |
| q67/q68 | q57 | 0.812 |
| q149/q148 | q150 | 0.809 |

#22's third candidate, q104/q105, passes the screen but has power 0.790, so it is not chosen.

Before Stage 4 submits anything, the live calibration of the chosen pairs is re-checked
against the same bounds. A pair that fails them is dropped, not replaced (§12).

## 3. Arms

Four arms per pair, each #20's circuit relabelled onto the pair (level 0, trivial layout, no
SWAPs; every circuit is DAG-equivalent to #20's after relabelling, `circuits.json`):

| arm | operation each step | step µs (FakeKingston) | calibrated w(0), q67/q68 · q149/q148 |
| --- | --- | --- | --- |
| baseline | Phase 6's circuit | 321.0 | −0.00743 · −0.00742 |
| A2 | measure the memory into `m[t]` | 323.3 | 0 · 0 |
| A2s_spec | delay on the memory, measure the spectator | 323.2 | = baseline |
| A2s | delay on the memory only (delay-only sham) | 323.2 | = baseline |

The truth model for power scales the baseline's w(0) to 0.75 of the calibrated value (#27:
0.747; Phase 6: 0.72). Predictions for every arm: `predictions.csv`. QASM and hashes:
`qasm/`, `circuits.json`.

**Parallel pairs.** In the recommended option both pairs run in the same circuits
(`qasm/parallel_*.qasm`). They are isolated as follows:

- *Coupling map.* The footprints {q67, q68, q57} and {q149, q148, q150} are 16 hops apart and
  share no neighbour. Each footprint is at least 6 hops from {q75, q59, q74}.
- *Spectator model.* In `qmem.spectator`, static ZZ and readout Stark shifts act only between
  coupled qubits. No coupler joins the two footprints, so the model's prediction for each pair
  in the parallel circuit equals its prediction alone.
- *Timing.* The parallel circuit keeps each pair's DAG unchanged (`pair_a_preserved`,
  `pair_b_preserved` and `only_pair_qubits` in `circuits.json`), with the same step durations as
  each pair alone, so readouts are simultaneous and per-pair timing does not move.

The model cannot exclude device-wide effects such as a shared readout line or a global drift.
So the pairs are reported separately, and a **cross-pair difference** is read as follows. For
each arm, a two-sided z-test of w(0, pair A) − w(0, pair B) at α = 0.01 uses the drift-
conditional SEs. A difference is reported but does not change either pair's call. When the two
pairs' calls differ, that is a pair-specific result, not a failure of replication. Correlated
job-level drift is tested by the correlation of the per-job baseline w(0) across the pairs. It
is reported, and the "both pairs" power in §4 assumes it is zero.

## 4. Shots

The cap for this tier is 120 s of QPU time, pilot included. That is a planning cap, not an
authorisation. Costs use Phase 7's model (`qmem.followup.CostModel`): upper bound =
1.25 × estimate + 1 s per job. All 108 options are in `options.csv`; the ones that matter:

| option | arms | shots (baseline, shams · A2) | jobs | upper bound incl. pilot | P(decisive) per pair | P(both pairs) |
| --- | --- | --- | --- | --- | --- | --- |
| **two_parallel (recommended)** | full | 35,000 · 20,000 | 7 | 116.9 s | 0.812 · 0.809 | 0.66 |
| two_parallel | no spectator sham | 45,000 · 20,000 | 7 | 106.3 s | 0.950 · 0.950 | 0.90 |
| one pair (q67/q68) | full | 35,000 · 20,000 | 7 | 116.9 s | 0.812 | – |
| one pair (q67/q68) | no spectator sham | 45,000 · 30,000 | 7 | 113.4 s | 0.959 | – |
| two_sequential | full | best within the cap: < 0.5 per pair | – | – | – | – |

"Decisive" means P(H1 and H2) under the truth model (§3), with true A2 ratio 0, sham ratio 1
and the drift-conditional SEs of §10. "Both pairs" is the product of the per-pair powers, so it
assumes the pairs are independent (§3).

**Generality against decisiveness.** Running in parallel costs no QPU time, because both pairs
share each shot. A single pair at the same cap therefore gets *no* more shots, and its power
stays 0.81; the full arm set at 45,000 shots needs 144.9 s. What buys decisiveness at a fixed
cap is the arm set. Dropping the spectator sham gives 0.95 per pair (one or two pairs alike),
but H2 then loses its readout-of-a-neighbour control, which #20 and #27 kept. Stage 4 chooses:

1. two pairs, full arms: generality, P(both decisive) 0.66;
2. two pairs, no spectator sham: P(both) 0.90 and a weaker H2;
3. one pair: only worth it if the second pair fails its live calibration (§12).

Sensitivity (`sensitivity.csv`): a true A2 ratio of 0.05 drops per-pair power to 0.40; a sham
ratio of 0.97 or 1.03 drops it to 0.58–0.62; a shortfall of 0.6 drops it to 0.67. #27's own
pair under this design would have 0.72.

## 5. Schedule

One pilot job (§9), then 7 main jobs. Every main job holds every arm. Each arm's shots are
split evenly over the jobs, and each job's share into pubs of at most 1,000 shots. Pub order
within job j is the permutation `default_rng([3131, j])` (`qmem.followup.make_schedule`), so
arms are interleaved within each job. The recommended option gives 126 pubs and 125,000 shots
per pair (`schedule.csv`). The pilot is 500 shots per arm in one job (`pilot_schedule.csv`).
`--plan` prints the unsubmitted execution plan. This issue has no submit path; Stage 4 builds one
on Phase 7's guards (`qmem.followup`).

## 6. Primary statistics and decision rules

The statistics of each arm (`qmem.replication.arm_statistics`): w(0) and g(0..7) as in Phase 7
(burn 16, length 64), and lag1, the lag-1 output autocorrelation. Each has a **drift-
conditional SE**, the larger of Phase 7's primary SE (max of the shot and block bootstrap,
block 500, 1,000 resamples, stratified by job) and #29's random-effect SE over pub chunks. The
same is computed for every leave-one-job-out (LOJO) subset.

The per-pair decision (`qmem.replication.decide`, config `[rule]`):

- **H1:** w_base(0) < 0 and |w_base(0)| ≥ 5 SE.
- **A2 vanishes:** |w_A2(0)| + 1.96 SE_A2 < 0.10 |w_base(0)|, on all jobs *and* on every
  LOJO subset (#29 finding 1).
- **A2 persists:** |w_A2(0)| ≥ 5 SE_A2 and |w_A2(0)| ≥ 0.25 |w_base(0)|. Otherwise A2 is
  ambiguous.
- **Shams equivalent:** A2s is equivalent to the baseline on the w(0) and lag1 ratios, and
  A2s_spec on the w(0) ratio (§7).
- **H2** = A2 vanishes and both shams are equivalent.

The reported call for each pair (`call_label`): *replicates* (H1 and H2); *failed* (not H1);
*ambiguous* (A2 vanishes on all jobs but not under LOJO, or §10's job rule); *partial* (H1
holds but H2 does not, with A2's call). Applied to #27's q75/q59 data, the rule gives
*ambiguous*: the all-jobs bound is 0.094, but leaving one job out raises it to 0.114.

## 7. TOST margin and α

Each sham is compared with the baseline on the ratio r = sham / baseline, with a delta-method
SE. A sham is equivalent on a statistic when the 90% CI of r lies inside [0.90, 1.10] (TOST,
α = 0.05 per bound; #29's recommended ±10%). It *differs* when the 95% CI excludes 1, which is
reported. g(0..7) differences from the baseline get two-sided z-tests, Holm over k at
α = 0.01 within each sham. They are reported, not part of H2 (§1).

## 8. Multiplicity

H2 is an intersection-union test: every component must pass at its own level, so there is no
inflation within a pair. Each pair gets its own call at the levels above, with no correction
across the two pairs. The campaign-level claim is stated per pair ("k of 2 pairs replicate"),
never as a pooled p-value. The cross-pair tests (§3) and the arm × job test (§10) run at
α = 0.01 each and only add flags.

## 9. Pilot gates

One pilot job, 500 shots per arm of every chosen pair, runs before any main job. The gates are
checked per pair (config `[pilot]`):

- (a) integrity: outcomes are 0/1, shapes are right, every pub returned;
- (b) baseline P(1) within 0.08 of the pair's live prediction (Phase 7 §9's value);
- (c) A2's memory-record P(m = 1) against the live prediction: recorded, never stops (Phase 7
  §10's treatment);
- (d) per-shot execution time within 25% of the cost model;
- (e) calibration: T1 and T2 within ×2 of the selection snapshot, readout ≤ 0.05 on every
  measured qubit.

A pair that fails (a), (b), (d) or (e) is stopped. There is no retry and no substitute pair.
In the parallel option, a stopped pair's qubits stay in the circuit (dropping them would change
the other pair's circuit), and its data are recorded but not analysed.

## 10. Drift and job-stratum policy

The SEs are drift-conditional (§6). For power, the inflation of each arm's SE over its per-shot
SD is calibrated on #27's arms and floored at 1 (`inflation.csv`: baseline 1.30, A2 1.79,
A2s_spec 1.69). The delay-only sham, which has no data yet, takes the larger of the baseline's
and A2s_spec's.

Arm × job (#29 finding 5, `arm_job_interaction`): for each sham and A2, and for each of P(1),
w(0) and lag1, the per-job difference from the baseline gets Cochran's Q. Where a P(1) or w(0)
test has p < 0.01, the job furthest from the pooled others (leave-one-out z) is flagged.

- One flagged job: the decision is recomputed without that job (every arm of it). If A2's call,
  H1 or H2 changes, or a P(1) or w(0) test is still heterogeneous, the pair is *ambiguous*.
- More than one flagged job: *ambiguous*.

LOJO in the vanish rule (§6) is applied whether or not a job is flagged.

## 11. Ambiguous and failed pairs

- *Failed* (not H1): the witness did not replicate on that pair. It is reported as a
  non-replication, with the pair's calibration, and is not explained away after the fact.
- *Ambiguous*: reported as such, with the LOJO bounds and flagged jobs. It counts as neither a
  replication nor a failure, and no extra shots are taken to resolve it.
- *Partial*: reported with the component that failed (A2 persists or ambiguous; which sham, on
  which statistic).
- A pair stopped at the pilot or by live calibration is *not run*, not failed.

The full pipeline was validated on simulated shots through the real analysis (`[validation]`,
40 replicates per pair and scenario, including #29's chunk-level drift and a planted arm × job
shift). Results are in the dry run's `summary.json`.

## 12. Stop/go

- **Before Stage 4:** the owner approves on #28 and chooses an option in §4. Live calibration
  of the chosen pairs passes §2's bounds. A pair that fails them is dropped, not replaced. If
  both fail, Stage 4 does not run, and the selection is redone as an amendment.
- **QPU time:** Phase 7's guards. Each job's upper bound is checked against the remaining
  allowance before submission (free Open plan, 600 s per 28 days, read live), and the run stops
  if a job is charged more than its upper bound.
- **After the pilot:** the gates of §9; a stopped pair does not continue.
- **No optional stopping:** every planned main job runs (unless a guard stops the run), and the
  analysis runs once, on all of them.

## 13. Amendment log

Any change after the first live data is a dated, post-hoc amendment appended here, with its
reason. It is fixed before the data it affects, in the manner of Phase 7 §§9–10.

- (none)
