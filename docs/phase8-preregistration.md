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
- **Remaining allowance.** It is a parameter (`[allowance] remaining_s`). **The live value is
  unknown until the owner reads the IBM dashboard (#28).** 531 s is #33's figure after #27, not
  a reading.
- **Reserve.** 60 s is held back from every tier.
- **Admission.** A tier is admitted only if the cumulative hard caps, through that tier, fit in
  remaining − reserve.

| remaining allowance | Tier 1 (cap 30) | + Tier 2 (cap 225) | + Tier 3 (cap 120) |
| --- | --- | --- | --- |
| 600 s | yes | yes | yes |
| 531 s (#33, unverified) | yes | yes | yes (375 ≤ 471) |
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

## Reproduction

The summaries in `results/summaries/phase8_prereg/` come from one full `--dry-run-local` (2,814 s
on the shared machine, 2 workers) at `2392151`. Its `git_dirty` flag is set only by this note
and the summaries themselves, which were uncommitted at the time; no code changed after
`2392151`. Seeds: `[tier2.dry_run] seed = 3601` (Tier 1 and 2 shots, schedules),
`[tier3.dry_run] seed = 3602`, `[reselection] seed = 3603`, `[tier1_power] seed = 3604`.
