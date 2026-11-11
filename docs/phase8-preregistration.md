# Phase 8 Stage 4: combined pre-registration of the three hardware tiers (#36)

Written before any Phase 8 hardware data exist. **Nothing here is approved or submitted.** Each
tier runs only after the owner approves that tier on #28 (device, qubits, circuits, upper bound,
hard cap) *and* a CLAUDE.md exception names the script that would contact IBM. Neither step is
part of this document.

This note fixes what each stage note left to Stage 4. It references the stage notes and does
not restate them:
- Tier 1: [phase8-reference.md](phase8-reference.md) (#33);
- Tier 2: [phase8-coherence.md](phase8-coherence.md) (#32, #34);
- Tier 3: [phase8-replication-prereg.md](phase8-replication-prereg.md) (#31);
- the claims: [phase8-claim-framework.md](phase8-claim-framework.md) §2 (#30);
- the drift and SE rules: [phase8-stage0-audit.md](phase8-stage0-audit.md) (#29).

The harness is `experiments/phase8_hw.py`, with config `experiments/configs/phase8_hw.toml`, and
the rules as code are in `qmem.prereg`. It has a `--dry-run-local` mode only: no live mode, no
submit path, and no `qiskit_ibm_runtime` import. Its outputs are in
`results/summaries/phase8_prereg/`: `tiers.csv`, `budget.csv`, `power_tier1.csv`,
`selection.json`, `reselection.csv`, `hashes.csv`, `dry_run.json` and `summary.json`. Tests:
`tests/test_prereg.py`.

```sh
uv run python experiments/phase8_hw.py --config experiments/configs/phase8_hw.toml --dry-run-local   # about 45 min, 2 workers
```

Sections:

1. Tier table
2. Tier 1: Stage 3, Claim 4
3. Tier 2: Stage 2, Claim 3
4. Tier 3: Stage 1, Claims 1–2 generality
5. Live re-selection rules
6. Order of the tiers and approvals
7. Budget
8. The dry run
9. Inconsistencies found when the stages were combined
10. Amendment log

## 1. Tier table

The cost model is `qmem.followup.CostModel`. Upper bounds are summed over each tier's jobs, and
"power" is at the planned shots under the stage's primary model.

| tier | claim | primary | jobs | upper bound | hard cap | power |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | P&M at n = 1; the record-00 branch at n = 2 (co-primary) | 2 (P&M; MDI optional) | 25.1 s | 30 s | 1.00 and 1.00 under every model (§2) |
| 2 | 3 | held-out NLL deficit, classical d ≤ D minus quantum, D = 4, 6; plus §4.1's contrast | 9 (A1) | 183.7 s (166.2 s as designed) | 225 s (30 + 150 + 45) | ≥ 0.999 at D = 6; contrast 0.97–1.00 (#32, #34) |
| 3 | 1–2 (generality) | per pair: H1 and H2 (#31 §6) | 8 (pilot + 7) | 116.9 s | 120 s | 0.81 per pair, 0.66 both (#31 §4) |

The three upper bounds total 325.7 s and the caps 375 s. Tier 3 is **deferrable**, and Tier 2
has a marked slot for #35's 1 rad extension (§3.9).

## 2. Tier 1: Stage 3 (#33), Claim 4

**2.1 Hypothesis.** One application of the trained step on q75 is not entanglement-breaking
(EB) under trusted SPAM. That is Claim 4 at n = 1. The co-primary is a weaker statement, about
the instrument and not the channel: the record-00 element M₀₀ at n = 2 is not EB. On its own
it is *ambiguous* for Claim 4 (#33 §3). It bears directly on #30's measure-and-prepare
alternative.

**2.2 Primary outcomes.**
- P&M score at n = 1: the equal-allocation mean over 12 circuits, with the stratified binomial
  SE (`R.pm_estimate`). EB is rejected if score − z SE > 2/3 + ℓ/3 = 0.6677.
- Record 00 at n = 2: the pooled branch ratio, with the binomial SE on the branch's shots
  (`R.ratio_estimate`). The threshold is 0.6687. It is tested only if the branch collects
  ≥ 500 shots.

The effect size is the score above its threshold.

**2.3 α and multiplicity.** The two co-primaries are tested with Holm at family-wise α = 0.05,
so the first step is at 0.025. The secondary family is #33 §3's other six tests: the
unconditional n = 2 test, records 0 and 1 at n = 1, and records 01, 10 and 11 at n = 2. They
are tested with Holm at 0.05 and are reported only. This is amendment A2 (§10): #33 had record
00 in the secondary family.

**2.4 Nulls already run.**
- Random EB channels and branches against all three bounds: `tests/test_reference.py`
  (`test_eb_*`, `test_mdi_*`), and `test_random_eb_channels_never_pass_the_pm_threshold_in_expectation`
  here.
- The EB negative control (A2's step) falsely passes in 0 of 4,000 Monte Carlo draws
  (`phase8_reference/summary.json` `mc_power`). Here it passes in 0 of 4,000 under every model
  (`power_tier1.csv`).

**2.5 Sham margins.** None. Tier 1 has no sham arm; its controls are the n = 0 gate and the
negative control.

**2.6 Drift and SE.** The SE is the stratified binomial one. Drift within a circuit only makes
it conservative, because a mixture of Bernoullis has variance at most that of the Bernoulli at
its mean (#33 §5). Pub order within the job is shuffled with a seed. A shot-order χ² over 10
blocks per pub is a warning, never an invalidation (`P.order_warnings`).

**2.7 Gates and stop/go.**
- P&M at n = 0 below 0.90: stop, and don't interpret n ≥ 1.
- The negative control's lower bound above its threshold: the n = 1 result is invalid.
- The MDI sub-tier runs only if the P&M job passed both gates and rejected EB at n = 1.

**2.8 Ambiguous.**
- Only the record-00 branch rejects EB: an instrument property, not Claim 4.
- A gate fails.
- The coherent-SPAM margin (#33 §5) is the stated limit: δ ≤ 0.022 per operation.

**2.9 Replication.** None within Tier 1. A pass is Claim 4 at one step on one pair. Claim 4 on
another pair needs its own tier.

**2.10 Circuits, shots, cost.**
- 112 QASM files in `results/summaries/phase8_reference/qasm/`, with SHA-256 in
  `manifest.json`. All match (`hashes.csv`). The V_n compiled into them equal the recomputed
  ones exactly (`summary.json` `targets_V_max_diff = 0`).
- P&M: 60 circuits, 36,000 shots, upper bound 15.2 s.
- MDI: 32 circuits, 20,000 shots, upper bound 9.9 s.
- Hard cap 30 s; per-job estimate limit 20 s.

**2.11 Power under Stage 2's fitted model** (task 3; `power_tier1.csv`). The planned shots are
12,000 for P&M n = 1 (12 × 1,000) and 6,000 for n = 2 (12 × 500); Holm's first step is
α = 0.025.

| model | P&M n = 1: score (effect) | power | shots for 80% | record 00, n = 2: score (effect) | expected branch shots | power | shots for 80% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 4c pieces | 0.712 (0.044) | 1.00 | 661 | 0.862 (0.193) | 1,793 | 1.00 | 84 |
| Stage 2 fixed | identical: its baseline step *is* the Stage 4c step (tested to 1e−14) | | | | | | |
| **Stage 2 fitted (9 parameters)** | **0.709 (0.041)** | **1.00** | **767** | **0.851 (0.182)** | 1,779 | **1.00** | 102 |
| deficit-matched (#33) | 0.703 (0.036) | 1.00 | 1,032 | 0.824 (0.156) | 1,811 | 1.00 | 155 |

**The fitted model does not break either primary.** It costs 0.003 in P&M score at n = 1 and
0.011 in record 00 at n = 2, and needs 767 and 102 shots for 80% power, against 12,000 and
6,000 planned. No shot change is needed. The n = 0 gate is predicted at 0.981 under every model
(threshold 0.90). The negative control is predicted at 0.577–0.578 (threshold 0.668), and its
false-pass rate is 0 of 4,000 under every model.

## 3. Tier 2: Stage 2 (#32, #34), Claim 3

**3.1 Hypothesis.** Claim 3 under the assumptions of phase8-coherence.md §5. A
population-preserving dephasing (R2, R4) removes w(0), while the matched feed-forward sham (RS)
keeps it within 20%. Element-wise classical models with d ≤ 6 hidden states lose to the quantum
prediction on held-out shots.

**It does not robustly exclude a classical phase register** (Z₈ × 2–4) or free d = 8. That
exclusion holds only if the 9-parameter quantum model is right to about 0.005 nats per shot
(#34 §3.1). The headline of any Tier 2 result must name the excluded class as phase8-coherence.md
§4.1 item 2 states it.

**3.2 Primary outcome.** The held-out NLL deficit per shot (classical best start minus the
quantum refit), with a block-bootstrap SE by job and 8-shot segment. "d ≤ D" is rejected at
z > 1.645, Holm over D = 4, 6 (§4 there). Claim 3 needs every leg (intersection–union, each at
α = 0.05):
- (i) population;
- (ii) phase response;
- (iii) equivalent dephasings;
- §4.1's contrast: vanish at f = 0.25 with LOJO, and RS non-inferior to E00 at m = 0.20.

**3.3 Nulls already run.**
- 0 false rejections of "d ≤ 6" in 12 replicates from two classical d = 6 generators: 95% CI
  0–0.26, z between −86 and −37 (`phase8_coherence/null.json`, `null_replicates.csv`).
- The phase register: `register.json`, `register_classes.csv`.

**3.4 Sham margins.**
- E0, E00 and RS against the baseline in P(1): ±0.01.
- E0 in lag 1: ±0.02.
- The composition pairs: the same margins.
- A2, R2 and R4 pairwise: ±0.01 in P(1) and lag 1; ±10% of the baseline's |w(0)| in w(0).
- RS against E00 in w(0): non-inferiority at m = 0.20.
- The controls tier: ±0.02 against e0 (descriptive).

**3.5 Drift and SE.**
- The legs use #31's drift-conditional SE (`REP.arm_statistics`): the larger of Phase 7's
  primary SE (shot and block bootstrap, block 500, stratified by job) and #29's random effect
  over pub chunks. Leave-one-job-out applies to the vanish rule.
- P(1)'s SE is the larger of the shot SE and the chunk random effect.
- The primary contrast uses the block bootstrap of §4 there.
- Under amendment A1 (§10), the analysis's "job" is the time block (three consecutive
  sub-jobs), as #32 designed it.

**3.6 Pilot gate and stop/go.** As in phase8-coherence.md §4: 9 arms × 2,000 shots in one job.
- E0, E00 and RS within ±0.02 of the baseline in P(1).
- P2 and P4 within ±0.10 of the fixed prediction.
- INV2 within ±0.03 of E00, and R2 within ±0.03 of A2, in P(1) and lag 1.

A failed gate stops the tier. Any rerun needs a dated amendment. Main runs only after the
pilot. Controls run only after main.

**3.7 Ambiguous.**
- *Partial:* the exclusion holds at D = 4 only.
- *Not supported:* the exclusion fails at D = 4, or (i), (iii) or the contrast fails. A failure
  of (iii) points to an implementation artefact.
- The register and free-d = 8 fits are a reported secondary, never part of the call.

**3.8 Replication.** None within Tier 2. The two time blocks are the drift strata, not
replicates.

**3.9 Circuits, shots, cost.**
- QASM hashes are in `phase8_coherence/plan.json`, and all 44 circuits are matched
  (`hashes.csv`):
  - #32 committed 10 of the 18 main-tier QASM files;
  - the baseline and A2 are #20's (`phase7_ablation/qasm/`);
  - the other six (P1, P3, P5, P6, P7, COMP40) were rebuilt here on FakeKingston (level 0,
    fixed layout) and reproduce `plan.json`'s hashes exactly. They are committed in
    `phase8_prereg/qasm/tier2/`.
- The cost by part:

  | part | arms | shots | upper bound | cap |
  | --- | --- | --- | --- | --- |
  | pilot | 9 | 18,000 | 16.7 s | 30 s |
  | main | 18 | 144,000 | 126.0 s (112.0 s as designed) | 150 s |
  | controls | 26 circuits | 76,000 | 41.0 s (37.5 s as designed) | 45 s |

- **The 1 rad slot (#35).** Six arms, about +34 s upper bound on main
  (`contrast.json` `separation`). It is **not folded in**: #35 was open when this was written.
  If #35 recommends it, the coordinator adds it as a dated pre-data amendment. Tier 2's cap
  would then be 259 s, and the budget table already prices it (`budget.csv`,
  `with_1rad_slot_at_remaining`).

## 4. Tier 3: Stage 1 (#31), Claims 1–2 generality; deferrable

- **Hypotheses, outcomes, rule:** phase8-replication-prereg.md §§1, 6.
  - H1: w(0) < 0 at ≥ 5 drift-conditional SE.
  - H2: A2 vanishes (all jobs and every LOJO subset), and both shams are equivalent.
- **α and multiplicity:** §§7–8 there. H2 is an intersection–union test with no correction
  across pairs, reported as "k of 2 pairs replicate".
- **Nulls run:** `phase8_replication/validation.csv`: 38/40 replicate under chunk drift, and 0/40
  flag a job.
- **Sham margins:** ±10% on the ratio, TOST at α = 0.05 per bound (#29's recommendation).
- **Drift and SE:** §10 there, the drift-conditional SE (#29).
- **Gates, stop/go and the ambiguous/failed labels:** §§9, 11 and 12 there.
- **Circuits and cost:**
  - `parallel_{baseline, A2, A2s_spec, A2s}.qasm`, hashed in `phase8_replication/circuits.json`;
    all match;
  - two_parallel, full arms, 125,000 shots per pair in 7 jobs plus a pilot;
  - upper bound 116.9 s, hard cap 120 s, per-job estimate limit 12 s.

**Deferrable.** If the allowance can't hold all three tiers with the reserve, Tier 3 is the one
deferred (#28). It can't move Claim 3, and P(both pairs decisive) is 0.66.

## 5. Live re-selection rules, fixed now and executed later

Each selection below was made on FakeKingston's stale snapshot (2026-04-15,
`phase7_spectator/device.json`). Each is now a deterministic function of a snapshot in
`qmem.prereg`, run before any approved tier on the live snapshot (same `device.json` shape).

| choice | rule (unchanged) | function | on FakeKingston | committed |
| --- | --- | --- | --- | --- |
| Tier 2 `ff` ancillas | distance ≥ 3 from q59 and q75, not adjacent to each other, lowest readout, then longest T1, qubits 0–75 | `select_ancillas` | **[50, 21]** | [50, 21] ✓ |
| Tier 1 Bell reference (Bell route only; not in Tier 1's plan) | #33 §6's rule, then the highest predicted n = 1 Bell fidelity | `select_reference` | **q79** (F = 0.546) | q79 ✓ |
| Tier 3 pairs | #31 §2's screen, P(H1 and H2) ≥ 0.80, ranked, ≥ 3 hops apart | `select_pairs` (`REP.select`) | **m67_o68 (0.839), m149_o148 (0.837)**; 87 screened | the same ✓ |

**If a live snapshot selects different qubits** (`P.reselection`, `reselection.csv`):

- **Tier 2 ancillas: replace.**
  1. Re-export RS, R2 and R4 on the live target with `phase8_coherence --export-qasm` (level 0,
     fixed layout).
  2. Re-hash, and record the new hashes in a dated pre-data amendment before submission.
  3. Re-check against the floor: the live ancillas' readout must be ≤ 0.02, and the R2/R4
     vanish power (f = 0.25) and the RS non-inferiority power (m = 0.20) must both be ≥ 0.80,
     taking the readout error as a conservative bound on the bit bias.
  4. If the floor fails, abort.

  In the dry run, a q50 readout of 0.05 moves the pick to [21, 68]. The vanish power is 1.00 and
  the preserve power 0.985, so the tier proceeds after re-export.
- **Tier 1 reference: replace.** This applies only if the Bell route is ever added, since
  neither P&M nor MDI uses a reference. Re-export `bell_*` and `q59ctrl_*`, re-hash, and check
  the Bell power at n = 1 (deficit-matched, 6,000 shots) against a floor of 0.80. If nothing is
  admitted, abort.
  - A q73 readout of 0.02 picks q73 (power 0.994).
  - A q79 T2 of 40 µs admits nothing, so the route aborts.
- **Tier 3 pairs: drop, never replace** (#31 §12).
  - A committed pair that fails the live screen or the power floor (0.80) is dropped. The live
    re-selection is recorded but not acted on.
  - If both committed pairs drop, Tier 3 aborts, and any reselection is an amendment.
  - A q68 readout of 0.03 drops m67_o68 and runs m149_o148 alone. Losing both pairs aborts.
  - A live run must rebuild each pair's pieces from the live snapshot (hardware group) before
    the power check. The dry run uses the committed FakeKingston pieces.

The snapshot hash (`P.snapshot_hash`) covers only the fields a rule reads. Witness numbers in a
snapshot can't reach a selection (tested).

## 6. Order of the tiers and approvals

1. **Tier 1, then Tier 2, then Tier 3.** This is #28's ranking by discriminating information per
   QPU second.
2. **Each tier runs only on its own approval.** The approval is one issue-comment URL in
   `[approval] tier<k>`. `P.Approval.check` refuses a URL for another tier. `P.freeze_approval`
   writes it into the run directory on first use and refuses a resume under a different one.
   An approval of one tier does not authorise the next.
3. A later tier may start before an earlier one finishes only if its own approval says so.
   Results of an earlier tier never change a later tier's pre-registered rules. Any change is
   an amendment (§10).
4. **Tier 3 is deferred first** whenever the allowance with its reserve can't hold it (§7).

## 7. Budget

- **Accounting.** Each job counts max(charged, upper bound) (`F.job_charge`). An unreadable
  charge counts as the upper bound and refuses the next job, and a charge above the upper bound
  stops the run (`F.usage_guard`).
- **Per-job estimate limits.** 20 s (Tiers 1 and 2) and 12 s (Tier 3).
- **Caps.** Each tier has its own hard cap.
- **Remaining allowance.** It is a parameter (`[allowance] remaining_s`). **The owner's
  dashboard reading on #28 (2026-11-10): 8.97 of 10 RU remaining, that is 538 s** (1 RU = 1 min
  of the Open plan's 10 min per rolling 28 days). #33's estimate had been 531 s.
- **Reserve.** 60 s is held back from every tier.
- **Admission.** A tier is admitted only if the cumulative hard caps, through that tier, fit in
  remaining − reserve.

| remaining allowance | Tier 1 (cap 30) | + Tier 2 (cap 225) | + Tier 3 (cap 120) |
| --- | --- | --- | --- |
| 600 s | yes | yes | yes |
| **538 s (reading, 2026-11-10)** | yes | yes | yes (375 ≤ 478; 409 ≤ 478 with the 1 rad slot) |
| 531 s (#33's estimate) | yes | yes | yes (375 ≤ 471) |
| 400 s | yes | yes | **defer** (375 > 340) |
| 300 s | yes | **no** (255 > 240; the upper bounds, 209 s, would fit) | defer |
| 200 s | yes | no | defer |
| unknown | **no tier runs** | | |

With #35's 1 rad slot, Tier 2's cap becomes 259 s, and all three tiers need 409 + 60 s.

The Open plan's 600 s is per rolling 28 days. The owner's dashboard reading, and its date, go
into the config before the first approved job.

## 8. The dry run

`dry_run.json` holds each tier at a quantum truth and a classical-alternative truth. Shots were
simulated job by job through `P.run_stage` on `LocalBackend`, with the guards, resume by id,
raw shots kept and usage counted, then run through the tier's own analysis and decision rule.

| tier | truth | call | counted |
| --- | --- | --- | --- |
| 1 | Stage 4c pieces | Claim 4 at n = 1 (P&M 0.714, record 00 0.871; MDI payoff 0.071, rejects EB) | 25.1 s ≤ 30 |
| 1 | Stage 2 fitted model | Claim 4 at n = 1 (P&M 0.709, record 00 0.859; MDI 0.038) | 25.1 s |
| 1 | EB (q75 measured every step) | not supported: nothing rejects EB (P&M 0.577; record 00 0.491) | 25.1 s |
| 2 | fixed quantum model | *supported*: the pilot passes; d ≤ 4 and d ≤ 6 rejected (deficit 0.239 and 0.063 nats per shot, z = 97 and 45); every leg passes (R2, R4 vanish with LOJO; RS/E00 = 1.01, lower bound 0.95); the Knee inequality is violated for e4, e2, A2 | 183.7 s ≤ 225 |
| 2 | classical d = 6 (the octahedron embedding of the instrument) | *stopped at the pilot* (E0, E00, RS, P2, P4 off); run on in simulation: *not supported* (no D rejected, deficit −1.02 nats per shot; every leg fails; no Knee violation) | 183.7 s |
| 3 | quantum (each pair's pieces, shortfall 0.75) | both pairs *replicate* | 116.9 s ≤ 120 |
| 3 | memory not on the memory qubit (A2's process = the baseline's) | both pairs *partial: H1 only (A2 persists)* | 116.9 s |

The harness refuses, as it must (`summary.json` `guards`):
- no approval;
- an approval for another tier;
- a changed approval on resume;
- the as-designed Tier 2 main job (44 s > 20 s);
- a charge over its upper bound.

A resume never resubmits.

**This classical truth is a weak adversary.** It is the octahedron embedding of the
instrument, a convenient d = 6 generator built from the same maps, and it sits far from the
quantum prediction. Here it checks the plumbing of the rule, not the test's size. The size
check against the closest d = 6 adversaries is #34's null calibration (§3.3).

One shot-order warning was raised, `duplicates` on P3 at the quantum truth. It is a warning,
not an invalidation: a low-entropy arm repeats whole 64-step records by chance.

**What the dry run shortens.** The Tier 2 primary contrast runs on the main tier's full shots
(8,000 per arm), but with one start per d (Klein for d = 4, octahedron for d = 6), 300 EM
steps, 200 refit evaluations and 200 bootstrap resamples. The real analysis uses four starts,
1,500 steps and 1,000 resamples. Fewer EM steps can only leave the classical fit worse, which
favours rejection. For the classical truth that bias doesn't matter, because the d = 6 start is
the generator itself. #34's null runs (§3.3) are the full-scale size check.

## 9. Inconsistencies found when the stages were combined

1. **Tier 2's jobs break the per-job guard every other tier uses.** #32's main jobs are 44.0 s
   each and its controls job 29.2 s. Phase 6/7 and Tier 1 use a 20 s limit, and Tier 3 uses
   12 s. `F.budget_check` would refuse them. → Amendment A1.
2. **#33's branch power used n = 1's shots for n = 2.** `phase8_reference.py` sizes every branch
   at 12 × 1,000 shots, but the planned n = 2 circuits have 500 shots each, so 6,000 in total
   and about 1,790 in record 00. At 6,000 the power is still 1.00, because 84 shots give 80%,
   so no design change is needed. The record-00 power in phase8-reference.md §3 is stated at
   twice the real shots.
3. **The record-00 branch is secondary in #33 but co-primary in #28.** → Amendment A2.
4. **Live re-selection against #31 §12.** #36 asks for live re-selection of the pairs, but #31
   drops a failing pair and never replaces it. #31's rule is kept (§5). If both pairs fail,
   re-selection needs an amendment.
5. **Six of Tier 2's main-tier circuits existed only as hashes.** They were rebuilt and match
   (§3.9). No design change.
6. **#33 leaves the MDI q74 reset open.** #33 §9 asks that a `reset` of q74 before ψ_y be
   decided before any MDI submission. It is not decided here: the committed MDI circuits have no
   reset, and adding one changes their hashes. **This is an owner or coordinator decision before
   the MDI sub-tier is approved.**
7. **Tier 3's selection power.** `candidates.csv` gives 0.839 and 0.837 (the selection design).
   #31 §2's table quotes 0.812 and 0.809 (the final option's power, §4 there). Both are #31's own
   numbers at two designs; the selection reproduces the former.

## 10. Amendment log

**Phase 7's pilot-gate changes were post hoc.** Phase 7 pre-registration §9 (2026-11-08, after
#25's pilot) and §10 (2026-11-08, after #26's pilot) changed the q75 model and the gate after
seeing pilot data, with the owner's approval. Phase 8 is **not a clean continuation** of a
pre-registered Phase 7:
- its designs were chosen after Phase 7's data;
- its models (Stage 4c pieces, the fitted quantum model, #30's classical fits) are fitted to
  them.

A Phase 8 result is pre-registered relative to Phase 8's own data only.

Every entry is dated and marked **pre-data** (before any Phase 8 hardware job) or **post hoc**.
Any change after a tier's first live data is post hoc and is appended here before the data it
affects, with the hash guard of Phase 7 §9 (`F.amendment_check`) on the live path.

- **A1, 2026-11-10, pre-data (Tier 2).** Each of #32's two main time blocks runs as 3
  consecutive sub-jobs (4,000 shots per arm per block, split 1,334/1,333/1,333, each sub-job
  holding every arm in a seeded order). The controls tier runs as 2 jobs.
  - The analysis's job is the time block, so the parity split, the block bootstrap and LOJO
    are unchanged.
  - Reason: §9 item 1. Largest sub-job estimate: 16.0 s.
  - Cost: main 112.0 → 126.0 s, controls 37.5 → 41.0 s, Tier 2 166.2 → 183.7 s, all within
    #32's caps.
  - The alternative, a per-tier limit of 60 s, is the owner's to choose instead.
- **A2, 2026-11-10, pre-data (Tier 1).** #28 promoted the record-00 branch at n = 2 from #33's
  secondary family to co-primary.
  - The two co-primaries are tested with Holm at 0.05, and the secondary family keeps #33's
    other six tests (Holm at 0.05).
  - Claim 4 still rests on P&M at n = 1 alone; a branch-only rejection is *ambiguous* (§2.8).
  - Power is unchanged at 1.00 (§2.11).
- **Slot, 2026-11-10, pre-data (Tier 2).** #35's 1 rad extension is not folded in (§3.9).
- **A3, 2026-11-10, pre-data (approvals, budget).** The owner gave a standing approval for the
  free Open plan on #28 (issuecomment-6782262170): "use the free plan however you see fit from
  now on". Its URL fills `[approval] tier1`, `tier2` and `tier3`.
  - §6's rules are otherwise unchanged. Tiers run in order. Each still passes its own
    admission check, gates, caps and live re-selection, and runs `--dry-run` first. A later
    tier's rules are never changed by an earlier tier's results without a dated amendment.
  - The allowance is the owner's reading (538 s, §7), so all three tiers are admitted, with
    the 1 rad slot too if #35 recommends it.
  - MDI (§2) is still excluded until its q74 reset is fixed by an amendment with new hashes.
- **A4, 2026-11-11, pre-data for MDI (Tier 1; after P&M's data).** This amendment is written
  before any MDI job. It comes *after* Tier 1's P&M data: job `dbprjt44148s739v32l0` (#37), which
  passed both gates and rejected EB at n = 1, so §2.7's precondition for MDI holds. A4 changes
  nothing about P&M, its analysis or its call. It fixes the MDI sub-tier (#38) as follows.
  - **The q74 reset (§9 item 6, closed).** The coordinator decided on #38: `reset q74`
    immediately before ψ_y. The game's soundness needs ψ_y uncorrelated with the channel's
    environment. Without the reset, q74's initialisation error (about 1%), ZZ-coupled to q75
    during the steps, is a known violation of that assumption.
  - **Circuits.** The 32 `mdi_*` circuits are re-exported: `phase8_reference --export-qasm`,
    FakeKingston, level 0, fixed layout (q59, q75, q74). The one change in each file is a
    `reset $74;` line after the steps' barrier and before ψ_y. The 80 P&M and Bell files are
    byte-identical (the P&M digest is still `4d17a450…`). Each circuit matches the numpy model
    noiselessly, with a worst difference of 1.8e-15 over 112 circuits. The new SHA-256s are in
    `phase8_reference/qasm/manifest.json` and `phase8_prereg/hashes.csv`. **The Tier 1 MDI
    digest** (the sorted `name sha256` lines of the 32 `mdi_*` rows) is
    `c36a119e950020da52e8635c9d5998ce06a1d980eeda3afffed1d07467102c0b`. #33's, without the
    reset, was `33c43250…`.
  - **Model.** `R.MdiReset`: q75 idles through q74's reset (2.312 µs on FakeKingston,
    2.212 µs live) at q75's T1 and T2. After the reset, q74 is excited with probability equal
    to its readout error, whatever its state before. The model is tested against a brute-force
    density matrix of (q75, q74), with a random joint channel for the steps. Random EB
    channels, chosen by q74's state and measured by any POVM element, never exceed payoff 0
    with the reset in place (`tests/test_reference.py`).
  - **Shots and cost (unchanged).** 250 per input pair at n = 0 and 1,000 per pair at n = 1,
    so 20,000 shots in one job of 32 circuits, in #33's seeded pub order. The estimate is
    7.18 s and the upper bound 9.97 s. That is within the 20 s per-job limit and within the
    14.84 s left of Tier 1's 30 s cap after P&M's 15.16 s counted. The ledger counts both
    cumulatively (`[tier1.mdi_live] prior_ledgers`).
  - **Power at the latest live calibration** (`ibm_kingston`, calibrated 2026-11-10 23:34 UTC,
    the snapshot of #37's submission; `phase8_hw/tier1_mdi_power_a4.json`). Payoff at n = 1
    with the reset, and power at 16,000 shots (one-sided α = 0.05, normal / Monte Carlo):

    | model | payoff | power | shots for 80% |
    | --- | --- | --- | --- |
    | Stage 4c pieces, committed / live | 0.055 / 0.052 | 0.94 / 0.92 | 9,596 / 10,815 |
    | **Stage 2 fitted, committed / live** | 0.050 / **0.047** | 0.90 / **0.87** | 11,352 / 13,010 |
    | deficit-matched, committed / live | 0.043 / 0.040 | 0.80 / 0.75 | 15,970 / 18,532 |

    The reset costs 0.009 in payoff (Stage 4c: 0.064 → 0.055). The fitted model at the live
    calibration has power 0.87 ≥ 0.80, so **the shots are not raised**. The n = 0 positive
    control is predicted at 0.44.
  - **Pre-data gate at the run's own calibration.** `--dry-run` and `--submit` recompute this
    check at their own live snapshot (`[tier1.mdi_live] power_gate`,
    `phase8_mdi_power.assess`). They refuse unless the fitted model's power at the live
    calibration is ≥ 0.80 (normal and Monte Carlo) at these shots, and the cost fits. If the
    gate fails, nothing is submitted. The check is repeated after each new calibration for up
    to 48 h. If it still fails, MDI is recorded as dropped. Raising the shots would need a
    further dated amendment before any MDI data.
  - **Analysis (#33 §2, as fixed there).** The payoff at n = 1 is Σ ω_xy p̂(Φ⁺ | x, y), where
    Φ⁺ reads 00 and ω = `R.mdi_weights(V_1)` with #33's committed V_1. Its SE is binomial per
    input pair. EB is rejected iff payoff − z₀.₉₅ SE > 0 (one-sided α = 0.05; `P.mdi_analyse`).
    - **Multiplicity:** a family of one. It sits in neither of P&M's Holm families, and it
      cannot change P&M's call.
    - **n = 0** is the positive control. It is reported, not a gate, because the game's
      soundness does not rest on the measurement.
    - **No leakage allowance** (threshold 0, as #33): the bound holds for any measurement, and
      leakage after ψ_y is part of the untrusted measurement.
    - **Shot-order warnings** are reported, never an invalidation (§2.6).
  - **Calls.**
    - The payoff rejects 0: Claim 4 at n = 1 in the MDI setting, trusting the input states
      only. Residual assumptions: the input-state calibration (ξ_x on q75, ψ_y on q74,
      including their leakage); the reset's residual excitation, which may depend on q74's
      state before it; and q74's ZZ with q75 while ψ_y is prepared.
    - The payoff does not reject 0: MDI is **inconclusive**. It does not undo the P&M result.
- **A5, 2026-11-11, pre-data for Tier 2 (after Tier 1's data).** This amendment is written
  before any Tier 2 job. It comes after Tier 1's two jobs, P&M (#37) and MDI (#38), and changes
  nothing about them. It fixes the live ancillas, circuits and cost of the Tier 2 **pilot**
  (#39). Main and the controls are not run under it (see the last item).
  - **Live ancillas (§5).** `phase8_hw.py --prepare --tier tier2` read `ibm_kingston`'s
    calibration of 2026-11-11 02:51 UTC at 03:06 UTC (`phase8_hw/tier2_prepare.json`,
    `tier2_prepare_calibration.json`). It is read-only and submits nothing. The snapshot hash
    is `c7aed81c…`, the same as #37's and #38's. `select_ancillas` picks **[13, 50]**: readout
    0.0020 and 0.0029, T1 413 and 335 µs, 6 hops from q59 and q75. FakeKingston's pick was
    [50, 21]. In the live circuits, b1 is read from q13 and b2 from q50.
  - **Re-export (§5's replace rule).** RS, R2 and R4 are re-exported on [13, 50] with
    `phase8_coherence --export-qasm --ancillas 13,50`. That uses #32's builder on
    FakeKingston, level 0, fixed layout, into `phase8_hw/tier2/qasm/` (with `manifest.json`).
    - Self-check: the same builder with [50, 21] reproduces #32's three hashes byte for byte.
    - Each new file has the same lines as #32's file with q50 → q13 and q21 → q50: 8 x per
      step on q75 and 256 `if_else` blocks.
    - The SHA-256s are RS `a2cab188…`, R2 `82ae2743…` and R4 `195aa5b7…` (full values in
      `[tier2.live] ancilla_sha256`). #32's files and `plan.json` are kept as the FakeKingston
      record.
    - The pilot's other seven arms (baseline, E0, E00, P2, P4, INV2, A2) are the committed
      files with `plan.json`'s hashes, unchanged.
    - **The Tier 2 pilot digest** (the sorted `name sha256` lines of the 9 pilot circuits) is
      `6a3fea5abb301fd872dcfb6e5fd425e4599d10512869e13d5dbc8a97c2180cef`.
  - **Floor (§5), at the same snapshot.** The bias bound is the larger readout error, 0.0029,
    which is ≤ 0.02. The R2/R4 vanish power (f = 0.25) is 1.00 and the RS non-inferiority power
    (m = 0.20) is 0.985, both ≥ 0.80 (`phase8_hw.ancilla_power`, fixed model, 8,000 shots per
    arm). The tier proceeds.
  - **The CZ duration (#37's `0.0`).** It is a rounding artefact in the summaries, not a Target
    quirk or a real zero. `phase8_hw.r6` rounded every float to 6 decimals, so the 68 ns CZ
    became 0.0. The raw snapshot (`calibration_live.json`) holds 6.8e-8 s, and the live cost
    always came from `estimate_duration` on the live Target at full precision. **No cost
    changes.**
    - Fixes: `r6` now keeps 6 significant digits below 1e-3.
    - `live_cost` refuses a missing, zero or non-finite circuit duration.
    - The live checks record each CZ among the tier's qubits and refuse one without a
      positive duration (`cz_check`). The q59–q75 CZ is 68 ns.
  - **Cost.** The feed-forward arms have no static duration (`if_else`). Their live duration
    is #32's design duration plus two live corrections:
    - the baseline's live duration less the design's;
    - per `ff` element and step, the live ancilla readout time less the design's 2.2 µs
      (`phase8_live.tier2_durations`).

    The pilot is 18,000 shots in one job. Its pre-registered estimate is 12.57 s and its upper
    bound 16.71 s. A preview at the prepare snapshot gives 12.57 s and 16.71 s. The guard uses
    max(pre-registered, live). Limits: per-job estimate 20 s, pilot cap 30 s
    (`[tier2.live] hard_cap_s`), Tier 2 cap 225 s.
  - **Admission (§7).** Tier 1's committed ledgers count 25.14 s (P&M 15.16 s, MDI 9.97 s).
    That is within the 25.2 s that `[allowance] prior_counted_s` already takes off the owner's
    reading; the live path refuses if the ledgers exceed it. Tier 2 is admitted only if its
    whole cap (225 s) fits in the remaining allowance less the 60 s reserve. The remaining
    allowance is the reading less what was counted, capped by IBM's live reading: 511 s at the
    prepare, so 225 ≤ 451.
  - **What the live run must reproduce, else stop and report.** `--dry-run` and `--submit`
    (`[tier2.live]`, `phase8_live.TIERS["tier2"]`) re-read the snapshot and re-run §5. They
    refuse if:
    - the rule picks other ancillas than [13, 50] (a new dated amendment is needed first);
    - the floor fails (abort);
    - any hash or the digest differs;
    - the level-0 transpile on the live target changes a file or touches a qubit outside
      {13, 50, 59, 75};
    - a CZ duration is missing;
    - the cost breaks a limit.
  - **Order and scope.** `--dry-run` first, then `--submit` with `stages = ["pilot"]` only. The
    run stops after the pilot by construction: no other stage is a live stage. The pilot gate
    is §3.6's, unchanged (`P.tier2_pilot_gate`, as the local dry run computes it):
    - point estimates of P(1) and lag 1 after the 16-step burn-in, over each arm's 2,000 shots;
    - E0, E00 and RS within ±0.02 of the baseline in P(1);
    - P2 and P4 within ±0.10 of Stage 2's fixed-model prediction;
    - INV2 within ±0.03 of E00 and R2 within ±0.03 of A2, in P(1) and lag 1.

    A failed gate stops the tier, and any rerun needs a dated amendment.
  - **Not covered.** Main (18 arms, cap 150 s) and the controls (cap 45 s) wait for #35's verdict
    on the 1 rad slot and the coordinator's review of the pilot. Each needs its own config
    stages and, if anything changes, an amendment.
    - R4 is re-exported here for main.
    - The controls tier's `bench_R2_*` circuits also use an ancilla (FakeKingston's q50). They
      are re-exported, with the live pick of their own day, before the controls run.
    - Tier 2's analysis rules are unchanged.
- **A6, 2026-11-11, pre-data for Tier 2 main and the controls (after the pilot's data).** This
  amendment is written before any main or controls job. It comes after Tier 2's pilot (#39, job
  `dbpuqd8qtr0c738f1c4g`, gate passed 9 of 9) and changes nothing about the pilot or its gate. It
  fixes main and the controls (#40). **It changes no analysis rule:** the pilot is data, so no
  margin, leg, SE policy or decision rule moves in light of it. The coordinator's review on #39
  notes that two legs of (ii) may fail as written (INV2 against E00 in TOST, and the harmonic
  fit against the fixed prediction). If they fail, that is the result.
  - **#35's verdict: drop the 1 rad slot.** #35 (phase8-coherence.md §3.5 and its §4.2
    amendment) found that the tier does not make the register exclusion robust. Against the
    other quantum model, Z_16 × 2 and Z_24 × 2 (nearest shift) and Z_8 × 3 with free 1 rad maps
    still win, by 0.009–0.019 nats per shot, and more shots do not change the sign. No 1 rad arm
    is added. Main stays at #32's 18 arms and **Tier 2's cap stays 225 s** (30 + 150 + 45).
    `[tier2.slot_1rad] included = false`.
  - **The day's ancillas (§5).** A fresh read-only `--prepare` (`--tier tier2_main`) read
    `ibm_kingston`'s calibration of 2026-11-11 16:57:55 UTC at 17:26 UTC
    (`phase8_hw/tier2_main_prepare.json`, `tier2_main_prepare_calibration.json`). The snapshot
    hash is `c7aed81c…`, the same as A5's. `select_ancillas` picks **[13, 50]** again: readout
    0.0020 and 0.0029, T1 413 and 335 µs. **Nothing is re-exported for main.** RS, R2 and R4 are
    A5's files with A5's hashes (`[tier2.main_live] ancilla_sha256`).
    - Floor at this snapshot: bias bound 0.0029 ≤ 0.02, vanish power (f = 0.25) 1.00 and
      preserve power (m = 0.20) 0.985, both ≥ 0.80. The tier proceeds.
    - **The main digest** (the sorted `name sha256` lines of main's 18 circuits: 15 `plan.json`
      hashes and A5's RS, R2, R4) is
      `76a51186aa53252d9e3101952376e65d4b649dcd3079fdbbe26a83004f1741a8`.
  - **Main's schedule (A1).** Six sub-jobs, run in order: time block 0 (`main_b0_s0`, `s1`,
    `s2`), then time block 1 (`main_b1_s0`, `s1`, `s2`). Each holds all 18 arms
    (1,334/1,333/1,333 shots per arm per block) in the harness's seeded pub order
    (`phase8_hw.tier2_schedules`, seed 3601), 144,000 shots in all.

    | sub-job | shots | estimate | upper bound |
    | --- | --- | --- | --- |
    | `main_b<b>_s0` | 24,012 | 16.00 s | 21.00 s |
    | `main_b<b>_s1`, `_s2` | 23,994 | 15.99 s | 20.99 s |
    | **main (6 sub-jobs)** | 144,000 | 95.98 s | **125.97 s** |

    Each estimate is ≤ the 20 s per-job limit. Main's cap is 150 s (`[tier2.main_live]
    hard_cap_s`), within the 208.29 s left of Tier 2's 225 s after the pilot's 16.71 s. The
    guard and the ledger use max(pre-registered, live) per sub-job (`phase8_live.live_cost`,
    with the feed-forward arms' live durations as in A5). The live path runs all six or none:
    `TIERS["tier2_main"]` refuses any other stage list.
  - **The controls.** The controls tier's two ancilla circuits, `bench_R2_0` and `bench_R2_1`,
    used FakeKingston's q50. They are re-exported on the day's first ancilla, **q13** (main's
    b1 qubit), with `phase8_coherence --export-controls --ancillas 13,50` into
    `phase8_hw/tier2/qasm/controls/` (with `manifest.json`). That is #34's builder on
    FakeKingston, level 0, fixed layout.
    - Self-check: the builder with q50 reproduces #34's two hashes byte for byte.
    - Each new file has #34's lines with q50 → q13, and acts only on q13 and q75.
    - The SHA-256s are `bench_R2_0` `8f19e966…` and `bench_R2_1` `1b3e27c1…` (full values in
      `[tier2.controls_live] ancilla_sha256`). The other 24 controls are #34's committed files,
      unchanged: no other control uses an ancilla.
    - **The controls digest** (the sorted `name sha256` lines of the 26) is
      `9b8df9499b73b5fab758db33af230a7508dbdf9d9161998508e1b57e5c705571`.
    - Cost: 2 jobs of 38,000 shots, each with an estimate of 15.62 s and an upper bound of
      20.52 s, so **41.04 s against the 45 s cap**. `bench_R2_*`'s live duration is
      `controls.json`'s design duration moved per repetition by the live ancilla readout
      time less the design's 2.2 µs.
    - The controls run only after main (their live entry needs main's committed ledger). They
      are descriptive (§3.4, phase8-coherence.md §4.1 item 4): never a gate, never part of the
      call.
    - The live check re-runs §5 on the controls' own snapshot. If it picks other ancillas
      than [13, 50], the controls stop, and a short dated **A6b** with the new re-export and
      hashes comes before they run.
  - **Admission (§7).** The committed ledgers count Tier 1 at 25.14 s (P&M 15.16, MDI 9.97) and
    the pilot at 16.71 s, 41.84 s in all. That is within `[allowance] prior_counted_s` = 41.9 s,
    which the live path checks (`earlier_ledgers`). The remaining allowance is 538 − 41.9 =
    496.1 s, and IBM's live reading at the prepare was 497 s. Tier 2's whole cap fits with the
    60 s reserve: 225 ≤ 436.1. Before the controls run, `prior_counted_s` adds main's
    counted seconds (bookkeeping only).
  - **Analysis (§3, unchanged).** Main's shots per arm are ordered by time block, then sub-job,
    then pub position. A random arm's program index per step is 2 b1 + b2, the circuit's
    feed-forward (element 1 follows b1, element 2 follows b2). The analysis's job is the time
    block (A1).
    - **Legs** (i), (ii) (the harmonic, return to baseline, composition), (iii) and §4.1's
      contrast (vanish at f = 0.25 with LOJO; RS non-inferior to E00 at m = 0.20) use
      `P.tier2_legs`, with `Tier2Rule`'s margins as pre-registered.
      - The statistics are `phase8_hw.tier2_main_estimates`, the code the dry run ran:
        §3.5's drift-conditional SE (`REP.arm_statistics`, block 500, 1,000 resamples,
        seed 3601), and P(1)'s SE as the larger of the shot SE and the pub-chunk random
        effect.
      - The predictions are Stage 2's fixed model's.
    - **The primary** is §3.2's held-out NLL deficit for d ≤ 4 and d ≤ 6 against the quantum
      refit, with Holm at 0.05. It uses #34's pipeline exactly as the null calibration ran it
      (`phase8_coherence_null.pipeline_tasks`):
      - the parity split within each time block;
      - d = 4 from the Klein and random starts, and d = 6 adding the octahedron and split
        starts, at 1,500 EM steps;
      - the refit with 600 Nelder–Mead evaluations;
      - the block bootstrap by time block and 8-shot segment, with 1,000 resamples.

      The random start is seeded by `[null] seed` 34 and source id 40
      (`[tier2.main_live] primary_source_id`). The deficit against the fixed model is
      reported, not used.
    - **The call** is `P.tier2_decide` (§3.7), and the headline names the excluded class as
      phase8-coherence.md §4.1 item 2 states it: element-wise classical models with d ≤ 6.
    - **The secondary** (reported, never part of the call) is the register Z_8 × 2, 3, 4
      (prism and random starts) and free d = 8 (cube and split), each against the refit and
      the fixed model.

    The code is `experiments/phase8_tier2.py` (`--part primary`, `secondary`, `decide`), on
    the committed shots.
  - **What the live run must reproduce, else stop and report.** `--dry-run --tier tier2_main`
    and `--submit` refuse if:
    - §5 picks other ancillas than [13, 50], or the floor fails;
    - any hash or the main digest differs;
    - the level-0 transpile changes a file or touches a qubit outside {13, 50, 59, 75};
    - a sub-job's cost differs from the table above, or the cost breaks a limit or a cap.

    `--dry-run` comes first. Then `--submit` runs the six sub-jobs in order, and a failed guard
    stops the run. **Any rerun of a main or controls job needs a dated amendment.** A resume
    re-fetches a submitted job by its id, so it is not a rerun.
- **A6b, 2026-11-11, pre-data for the controls (after main's data).** This amendment is written
  before any controls job, as A6 requires when the controls' own snapshot picks other ancillas.
  Main ran under A6 (#40: six sub-jobs, 18:04–18:54 UTC, 125.97 s counted). A6b changes nothing
  about main, its analysis or any rule, and the controls stay descriptive.
  - **The controls' snapshot.** The first `--submit --tier tier2_controls` read `ibm_kingston`'s
    calibration of 2026-11-11 21:32:02 UTC at 22:04 UTC (snapshot hash `0d81f343…`). §5's rule
    picks **[64, 56]**: q50's readout rose to 0.0117, and q64 and q56 read 0.0029. The live path
    refused before any submission, as A6 says it must.
    - The same snapshot shows q75's readout at 0.057, against 0.009 at main's. No Tier 2 rule
      gates the controls on it. It is recorded, and it bears on the controls' read of q75 only.
  - **Floor at that snapshot.** The bias bound is 0.0029 ≤ 0.02, the vanish power 1.00 and the
    preserve power 0.985, both ≥ 0.80.
  - **Re-export.** `bench_R2_0` and `bench_R2_1` use the first ancilla, so they are re-exported
    on **q64** with `phase8_coherence --export-controls --ancillas 64,56 --out-dir
    results/summaries/phase8_hw/tier2/qasm_a6b` (FakeKingston, level 0, fixed layout).
    - The builder with q50 reproduces #34's hashes.
    - Each file has #34's lines with q50 → q64, and acts only on q64 and q75.
    - The SHA-256s are `bench_R2_0` `30e7f0a6…` and `bench_R2_1` `671138c9…`.
    - **The controls digest** is
      `c645561e3100ddbe115574e4ab4101cade6c26d651b3d940bcbc178718ec9a5d`.
    - A6's q13 files stay as A6's record.
  - **Cost and admission are unchanged.** There are 2 jobs, each with an estimate of 15.62 s and
    an upper bound of 20.52 s, so 41.04 s against the 45 s cap. The allowance is 538 − 167.9
    counted = 370.1 s, so 225 ≤ 310.1 with the reserve.
  - **What the run must reproduce.** The pick [64, 56], the floor, the digest and the cost, as
    in A6. If the pick moves again before submission, the controls stop again and a further
    dated amendment is needed.

## Reproduction

The summaries in `results/summaries/phase8_prereg/` come from one full `--dry-run-local` (2,814 s
on the shared machine, 2 workers) at `2392151`. Its `git_dirty` flag is set only by this note
and the summaries themselves, which were uncommitted at the time; no code changed after
`2392151`. Seeds: `[tier2.dry_run] seed = 3601` (Tier 1 and 2 shots, schedules),
`[tier3.dry_run] seed = 3602`, `[reselection] seed = 3603`, `[tier1_power] seed = 3604`.
