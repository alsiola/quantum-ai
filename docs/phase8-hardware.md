# Phase 8 Stage 5: the live path and Tier 1 P&M on ibm_kingston (#37)

**Outcome: Claim 4 at n = 1, under trusted SPAM. One application of the trained step on q75 is
not entanglement-breaking.** Both co-primaries reject EB, with Holm at a family-wise α of 0.05:
- P&M at n = 1: 0.721 ± 0.004 against a threshold of 0.668. The one-sided lower bound is 0.715.
- Record 00 at n = 2: 0.874 ± 0.008 against 0.669.

Both gates pass. The n = 0 SPAM gate reads 0.987 (stop below 0.90). The EB negative control
reads 0.577, with a lower bound of 0.567 against its threshold of 0.668.

QPU time: IBM charged **12 s**, and the ledger counts **15.16 s** (max of charged and upper
bound) of the 20 s cap. That leaves 531 − 12 = 519 s on IBM's own reading, or 538 − 15.16 =
522.8 s on the owner's.

**MDI sub-tier (#38, §5): Claim 4 at n = 1 also holds trusting only the input states.** The
payoff is 0.067 ± 0.017, with a lower bound of 0.039 > 0. It cost 8 s charged and 9.97 s counted,
so Tier 1 has counted 25.14 s of its 30 s.

The pre-registration is [phase8-preregistration.md](phase8-preregistration.md): §2 for Tier 1,
§5 for live re-selection, §7 for the budget, and amendments A2 and A3 in §10.

## 1. The live path

`experiments/phase8_hw.py` gains two modes, `--dry-run` (the live target, submitting nothing)
and `--submit`. Both take `--tier` and `--from-run <dir>`. Their code is
`experiments/phase8_live.py`. It is the only Phase 8 code that imports `qiskit_ibm_runtime`, and
it does so inside functions, so the main check and `--dry-run-local` run without the `hardware`
group. Credentials are read from `.env` only.

```sh
uv run --group hardware python experiments/phase8_hw.py --config experiments/configs/phase8_hw.toml --dry-run --tier tier1
uv run --group hardware python experiments/phase8_hw.py --config experiments/configs/phase8_hw.toml --submit --tier tier1
```

It reuses the existing rules and does not restate them:
- **Approval:** `qmem.prereg.Approval.check` (that is, `F.submission_check` plus the tier
  name), then `freeze_approval` into the run directory. This happens before any network call.
- **Pre-registration:** each `[[tier1.live.prereg]]` section (§2 and §10) must hash to its
  recorded value, both in the working copy and at a161c11 on `origin/main`. The hashing uses
  `F.amendment_section` and `F.amendment_hash`, which is Phase 7's amendment rule. A2 and A3
  must be present.
  - `F.amendment_check` itself is not called. It needs a date in the section heading, and
    these headings have none.
- **Circuits:**
  - Every committed QASM hash must match its manifest and `phase8_prereg/hashes.csv`.
  - The digest of the 60 `pm_*` rows must equal the approved
    `4d17a450…0365276`. It is the SHA-256 of the sorted `name sha256` lines, each
    newline-terminated.
  - Each file must come back byte for byte unchanged from a level-0, fixed-layout transpile on
    the live target, and touch only q59 and q75.
- **Live re-selection (§5):** the three rules re-run on the live snapshot, which is saved. A
  change refuses only if it is in the tier's `gating_selections`, and that list is empty for
  P&M (see §2).
- **Budget:**
  - Admission is `P.budget_table`: the cumulative caps must fit in remaining − 60 s.
  - The per-job guard is `P.Ledger.guard` (that is, `F.budget_check`): the per-job estimate
    is at most 20 s; the plan must be in `allowed_plans` (the Open plan only); the allowance
    must be known; the spend plus the upper bound must be within the 20 s cap.
  - The allowance used is the owner's reading less what has already been counted, capped by
    IBM's live `usage_remaining_seconds`.
  - The guard and the ledger use the larger of the pre-registered cost and the cost recomputed
    with live circuit durations.
- **Jobs:** `P.run_stage` on `LiveBackend`, which is SamplerV2 with no error mitigation.
  - The job id is saved before waiting, and a resume re-fetches by id and never resubmits.
  - Raw shots are kept.
  - The charge is read three ways (job `usage()`, metrics, account delta), and
    `F.job_charge` counts max(charged, upper). A charge above the upper bound stops the run
    (`UsageExceeded`), and an unreadable one refuses the next job.
- **Analysis:** `P.tier1_analyse` with the config's `Tier1Rule`, on counts from
  `phase8_hw.tier1_pm_counts`. That is the same decoding the local dry run used, now factored
  out.

Tests: `tests/test_phase8_live.py` uses a fake service and sampler, with no network and no
Qiskit. It covers:
- refusal of a missing, other-tier or changed approval;
- refusal of a changed pre-registration section, or one whose commit is not on main;
- refusal of a digest or file hash mismatch;
- refusal of a paid or unknown plan, an unknown allowance, an over-cap job, and an estimate
  above 20 s;
- a submission once, then a resume by id;
- a stop on an over-bound charge;
- the upper bound counted for an unreported charge.

**Tiers 2 and 3** run on the same path. Each needs a `[tierN.live]` table and a
`phase8_live.TIERS` entry (its stages, decoding and pre-registered analysis), then its own
`--dry-run`.

## 2. The live dry run (2026-11-11, 00:03 UTC)

The run directory is `results/phase8_hw_livedry_tier1/20261111T000328Z/` (gitignored). The
committed copies are `results/summaries/phase8_hw/tier1_livedry.json`, `_calibration.json`
(the full live snapshot, in `device.json` shape) and `_log.txt`.

| check | result |
| --- | --- |
| plan | `open` |
| IBM usage | 69 s used, **531 s remaining** in the 28 days from 2026-10-14 (the owner's dashboard reading was 1.03 RU used, 538 s remaining; IBM's own seconds differ a little, and the guard uses the smaller) |
| calibration | `ibm_kingston`, last update 2026-11-10 23:34 UTC |
| pre-registration §2, §10 | unchanged since a161c11 (on `main`) |
| hashes | 169 of 169 ok and identical to `hashes.csv`; Tier 1 digest `4d17a450…` ✓ (60 files) |
| ISA | all 60 P&M circuits unchanged by the live transpile; only q59 and q75 |
| cost | estimate 11.33 s (live durations: 11.32 s), upper bound 15.16 s (15.14 s); cap 20 s; per-job limit 20 s |
| admission | 20 s of caps ≤ 531 − 60 s ✓ |
| per-job guard | would pass |

**Live re-selection (§5):**

| choice | committed (FakeKingston) | live | consequence |
| --- | --- | --- | --- |
| Tier 2 `ff` ancillas | [50, 21] | **[13, 50]** | Tier 2 only: re-export RS/R2/R4, re-hash, a dated amendment, and the power floor before Tier 2 submits |
| Tier 1 Bell reference | q79 | **q73** | Bell route only, which is not in Tier 1's plan. None for P&M |
| Tier 3 pairs | m67_o68, m149_o148 | the same (power 0.839, 0.837) | none |

P&M uses no re-selected qubit: its qubits are fixed at q75 and q59. So neither change gates
Tier 1 under §5. Both are recorded for the tiers they belong to.
- The live screen also admits 41 pairs that have no committed FakeKingston pieces. They get no
  power here.
- A live Tier 3 rebuilds every pair's pieces from the live snapshot first (§5). That cannot
  change the two committed pairs, which are dropped, never replaced.

**Calibration of the pair (diagnostic; Tier 1 pre-registers no calibration bound).**

| | live (2026-11-10) | #16 snapshot |
| --- | --- | --- |
| q75 readout, T1, T2 | 0.0093, 244 µs, 124 µs | 0.0151, 321 µs, 139 µs |
| q59 readout, T1, T2 | 0.0044, 371 µs, **41 µs** | 0.0039, 250 µs, 371 µs |
| CZ 59–75 | 0.0019 | 0.0011 |

**q59's T2 is 9 times shorter than at #16.** Phase 7's calibration bound (a factor of 2) would
have refused it, but Tier 1's pre-registration has no such bound. Its protections are the
n = 0 SPAM gate (0.90) and the EB negative control. A worse step channel can only lower the
P&M score, which makes the test conservative, not anti-conservative. The coordinator may still
prefer to wait for a better calibration.

### 2.1 Pre-data power check at the live calibration (the coordinator's condition)

Before any submission, #33's P&M prediction was re-computed with q59's and q75's live T1, T2
and readout. It used the Stage 4c noise model with those six numbers replaced; the pieces were
rebuilt as #27 built them, and the rebuild reproduces the committed pieces exactly (difference
0). q75's readout flips were scaled to its live readout. V stays #33's fixed correction, which
is what the QASM compiles. The rule: submit if the power of P&M at n = 1 is at least 0.8 at
12,000 shots; otherwise wait for a new calibration, for up to 48 h. Script:
`experiments/phase8_live_power.py` (hardware group); output
`results/summaries/phase8_hw/tier1_live_power.json`.

| model | P&M n = 1 (threshold 0.6677) | power at 12,000 (MC; Holm's 0.025) | shots for 80% | record 00, n = 2 (threshold 0.6687) | n = 0 gate (0.90) | negative control |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 4c (committed) | 0.7118 | 1.00 | 661 | 0.862 | 0.981 | 0.578 |
| **live T1/T2/readout** | **0.7108** | **1.00** | 691 | 0.861 | 0.981 | 0.578 |
| live T1/T2/readout + CZ/sx errors | 0.7102 | 1.00 | 716 | 0.859 | 0.981 | 0.578 |

**Go.** q59's short T2 costs about 0.001 in the score. q59 is measured and reset in every step,
so its coherence barely enters the channel; q75's is what matters, and its live T2 (124 µs) is
close to Stage 4c's (139 µs).

## 3. The job

| item | value |
| --- | --- |
| job id | `dbprjt44148s739v32l0` (SamplerV2, no error mitigation), on `ibm_kingston` |
| submitted | 2026-11-11 00:42:28 UTC |
| executed | 01:56:42–01:56:53 UTC, on the calibration of 01:16 UTC (q59 and q75 unchanged from §2) |
| pubs, shots | 60 circuits, 36,000 shots, in #33's seeded order |
| cost | estimate 11.33 s, upper bound 15.16 s |
| charge | **12 s** (metrics `qpu_charge_time_seconds` 12, account delta 12; `job.usage()` read 0); counted 15.16 s |
| ledger | 15.16 s of the 20 s cap; `[allowance] prior_counted_s` is now 15.2 for later tiers |

The session that submitted the job ended while the job was queued, after its id had been saved.
The run was resumed with `--from-run`. That re-ran every check on the new calibration (all
passed), re-fetched the job by id and never resubmitted it (`resumed: true` in the job record).
Execution spans total 11.4 s, and `circuits_execution_time_ns` is 9.64 s.

Raw outputs are in `results/phase8_hw_live_tier1/20261111T003800Z/` (gitignored). The committed
copies are in `results/summaries/phase8_hw/`:
- `tier1/pm_primary_shots.npz`: every shot. Per pub, the bits `y0..y{n-1}, m`; for `pm_neg`,
  the mid-step `neg` as well.
- `tier1/pm_primary_job.json`: the id, the guard's record and the charge.
- `tier1/live_jobs.json`: usage before and after, metrics and spans.
- `tier1/decision.json`, `tier1/ledger.json` and `tier1/log.txt`.
- `tier1_live.json`: every pre-submission check of the resumed run.
- The two live calibration snapshots: `tier1_live_calibration.json` (at submission) and
  `_resume.json` (01:16).

## 4. The §2 analysis, as pre-registered

`P.tier1_analyse` with `Tier1Rule(alpha=0.05, secondary_alpha=0.05, spam_gate=0.90,
min_branch_shots=500)`. The leakage-corrected thresholds are 2/3 + ℓ/3, with ℓ = 0.003 n.

| test | family | estimate ± SE | threshold | shots | EB rejected |
| --- | --- | --- | --- | --- | --- |
| **P&M, n = 1** | co-primary | **0.7208 ± 0.0037** | 0.6677 | 12,000 | **yes** (p < 10⁻¹⁵) |
| **record 00, n = 2** | co-primary | **0.8735 ± 0.0078** | 0.6687 | 1,795 | **yes** |
| P&M, n = 2 | secondary | 0.5935 ± 0.0058 | 0.6687 | 6,000 | no |
| record 0, n = 1 | secondary | 0.8877 ± 0.0037 | 0.6677 | 7,358 | yes |
| record 1, n = 1 | secondary | 0.4561 ± 0.0073 | 0.6677 | 4,642 | no |
| record 01, n = 2 | secondary | 0.4871 ± 0.0116 | 0.6687 | 1,854 | no |
| record 10, n = 2 | secondary | 0.4523 ± 0.0179 | 0.6687 | 776 | no |
| record 11, n = 2 | secondary | 0.4692 ± 0.0126 | 0.6687 | 1,575 | no |

| gate or check | value | rule | result |
| --- | --- | --- | --- |
| n = 0 SPAM gate | 0.9873 | ≥ 0.90 | pass |
| EB negative control (A2's step) | 0.5768 ± 0.0058, lower bound 0.5673 | lower bound ≤ 0.6677 | pass (does not falsely pass) |
| n = 3 model check | 0.537 | below 2/3 expected | as expected |
| shot-order χ² | 1 of 60 pubs warns (`pm_neg` c8, p = 4 × 10⁻⁴) | a warning, never an invalidation | recorded |

**Decision (§2.7–2.8): Claim 4 is supported at n = 1, so EB is rejected.** The record-00 element
M₀₀ at n = 2 is also certified not EB. That is an instrument property, as §2.1 states. The call
holds under trusted SPAM, with #33 §5's coherent-SPAM margin (δ ≤ 0.022 per operation) as its
stated limit. It is Claim 4 at one step, on one pair (q75, q59), on one day (§2.9). The one
order warning is on the EB control arm, and it doesn't touch the call.

**Against the pre-data predictions (§2.1):**

| | live-calibration prediction | observed |
| --- | --- | --- |
| P&M, n = 1 | 0.711 | 0.721 |
| record 00 | 0.861 | 0.874 |
| n = 0 | 0.981 | 0.987 |
| negative control | 0.578 | 0.577 |

The observed P&M scores are slightly above the predictions, by 1–2.5 SE. The negative control
matches.

**MDI** stays excluded until its q74 reset is fixed by an amendment (A3). §2.7's precondition for
it (P&M passed both gates and rejected EB at n = 1) is now met.

## 5. Stage 5b: the MDI sub-tier with a q74 reset (#38)

**Outcome: Claim 4 at n = 1 in the MDI setting, trusting the input states only.** The MDI payoff
at n = 1 is **0.067 ± 0.017**, with a one-sided 95% lower bound of **0.039 > 0** (p = 4 × 10⁻⁵).
EB is rejected. This removes P&M's trusted-measurement assumption, and with it #33's
coherent-SPAM margin (δ ≤ 0.022) as the stated limit. The P&M result (§4) is unchanged.

QPU time: IBM charged **8 s**, and the ledger counts **9.97 s** (the upper bound). Tier 1 has now
counted 15.16 + 9.97 = **25.14 s of its 30 s cap**. What remains is 519 − 8 = 511 s on IBM's
reading, or 538 − 15.2 − 9.97 = 512.8 s on the owner's.

The rules are amendment A4 ([phase8-preregistration.md](phase8-preregistration.md) §10),
written and landed on `main` (`17d213e`) before any MDI job and after P&M's data.

### 5.1 What A4 changed

- **The reset.** `reset q74` immediately before ψ_y, the coordinator's decision on #38. It
  removes the one known violation of the game's assumption: q74's initialisation error,
  ZZ-coupled to q75 during the steps, would otherwise correlate ψ_y with the channel's
  environment.
- **Circuits.** The 32 MDI circuits were re-exported with one `reset $74;` line each. The P&M and
  Bell files are byte-identical. The noiseless check against the numpy model is 1.8e-15. The MDI
  digest is `c36a119e…`.
- **Model.** `R.MdiReset`: q75 idles through the reset, and q74 is left excited with
  probability equal to its readout error. It is tested against a brute-force (q75, q74)
  density matrix. EB channels chosen by q74's state never beat payoff 0 with the reset, under
  any POVM.
- **Code.** `phase8_live.py` gains the sub-tier `tier1_mdi` (`[tier1.mdi_live]`):
  - it runs on Tier 1's approval;
  - its cap is Tier 1's 30 s less P&M's committed ledger, so 14.84 s;
  - it refuses unless P&M's committed decision meets §2.7;
  - it has a pre-data power gate at the run's own live snapshot (`phase8_mdi_power.py`).

  The guard tests are in `tests/test_phase8_live.py`, on a fake service.

### 5.2 Pre-data power, dry run and job

| model, at the live calibration (2026-11-11 02:32 UTC) | predicted payoff, n = 1 | power at 16,000 shots |
| --- | --- | --- |
| Stage 4c pieces | 0.052 | 0.92 |
| **Stage 2 fitted (the rule's model)** | **0.047** | **0.87** |
| deficit-matched | 0.040 | 0.75 |

- The reset costs 0.009 in payoff. The fitted model clears 0.80, so the shots stayed at #33's
  20,000 (250 per input pair at n = 0, 1,000 at n = 1). q59, q74 and q75 calibrated exactly as at
  #37's snapshot.
- **The dry run** (`tier1_mdi_livedry.json`, nothing submitted) passed every check:
  - A4's sections at their commits on `main`;
  - all 169 hashes, and the MDI digest;
  - no selection change;
  - the live transpile left the circuits unchanged;
  - estimate 7.18 s, upper bound 9.97 s;
  - the power gate passed, and the guard would pass.
- **The submission** re-ran the checks and the power gate at its own snapshot, with the same
  results.
- **The job.** `dbptemtt7fqc7389vmrg` on `ibm_kingston`, submitted 2026-11-11 02:47:56 local:
  32 circuits, 20,000 shots, SamplerV2, no mitigation. The usage readings were 8 s on all three
  (job usage, metrics, account delta).

### 5.3 The analysis, as pre-registered

The payoff is Σ ω_xy p̂(Φ⁺ | x, y) with #33's fixed V_n in the weights, with a binomial SE per
input pair. The test is one-sided at α = 0.05, a family of one (`P.mdi_analyse`).

| arm | payoff ± SE | one-sided 95% bound | predicted (fitted, live) | role |
| --- | --- | --- | --- | --- |
| **n = 1** | **0.0675 ± 0.0172** | lower **0.039** | 0.047 | the test: payoff > 0 |
| n = 0 | 0.4665 ± 0.0157 | lower 0.441 | 0.441 | positive control, reported (not a gate) |

A 90% two-sided interval for the n = 1 payoff is [0.039, 0.096]; the 95% one is [0.034, 0.101].
The observed payoff sits 1.2 SE above the fitted model's prediction and 0.9 SE above the Stage
4c pieces' prediction. No pub raised a shot-order warning. Raw shots, the job record, the
calibration before and after, the ledger and the decision are in
`results/summaries/phase8_hw/tier1_mdi/` and `tier1_mdi_live*.json`.

**Decision: Claim 4 at n = 1, trusting only the input states.** One application of the trained
step on q75 is not EB, whatever the Bell measurement on (q75, q74) actually implements. The
residual assumptions are:
- the input-state calibration: ξ_x on q75 and ψ_y on q74, including their leakage;
- the reset's residual excitation, which may depend on q74's state before it;
- q74's ZZ with q75 while ψ_y is prepared.

It is still one step, on one pair, on one day. It says nothing about whether the emitted-symbol
witness uses q75's coherence (Claim 3).
