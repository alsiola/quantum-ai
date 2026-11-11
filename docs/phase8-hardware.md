# Phase 8 Stage 5: the live path and Tier 1 P&M on ibm_kingston (#37)

**Status: the live dry run passed, the coordinator accepted the live code (#37,
issuecomment-6783342126), and the pre-data power check at the live calibration passed (§2.1).**
This note holds the job, the gates and Claim 4's decision in §3.

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

Not yet submitted. It will be one job: 60 pubs and 36,000 shots in #33's seeded order, with
an upper bound of 15.16 s.
