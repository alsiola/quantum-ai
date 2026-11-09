# Phase 8 Stage 0A: independent re-analysis of Phase 7 Stage 4c (#29)

Local analysis only: no QPU, no IBM service. This note re-derives the Stage 4c outcome (#27,
`docs/phase7-hardware.md`) from the committed shots, asks how fragile the *vanishes* call is,
and sets the sham's equivalence margins for Phase 8 (#28).

- Code: `qmem.reanalysis` (the independent path, numpy only) and
  `experiments/phase8_stage0.py`, with config `experiments/configs/phase8_stage0.toml`. The
  config's `[fragility]` set was committed (`be76c18`) before any fragility number was
  computed.
- Outputs: `results/summaries/phase8_stage0/`. That is `summary.json`, `provenance.json`,
  `fragility.csv`, `drift_chunks.csv`, `sham_jobs.csv`, `tost.csv`, `tost_plan.csv` and
  `power_shots.csv`. It takes about 20 s on one core.
- Tests: `tests/test_reanalysis.py`. They cover the counter against brute-force string
  matching and `qmem.witness`, with analytic cases (w(0) = −1 exactly, w(0) = −b³/max(a, b)³,
  and Bernoulli with rank 1); TOST on hand-computed cases; Fieller against a brute-force grid;
  and the provenance freeze.

## Outcome: **ambiguous** (reproduces exactly; one predeclared choice flips the call)

Every number reproduces. But one choice in the predeclared reasonable set gives a bound above
the threshold: **dropping job1**, the job with the baseline's P(1) dip. With job1 dropped,
A2's bound is **10.1%** of the baseline, against the 10% threshold. Under #29's rule that is
*ambiguous*. It is a first-class result, and it changes Stage 1's design (§3, §6).

## Negative findings first

1. **The vanish call does not survive dropping job1.** The bound is 0.1010 (A2: −2.29e-4 ±
   1.63e-4 on 7,500 shots; baseline −5.44e-3). The flip comes from losing a quarter of A2's
   shots (the SE grows by √(4/3)), not from a change in A2's witness. Dropping job2 instead,
   with no documented reason, gives 0.114. Dropping job0 or job3 gives 0.089 or 0.099. **The
   design has no margin for losing a quarter of its shots.**
2. **A replicate of the same design would come out *vanishes* only 91% of the time even if
   the true ratio were 0.** At a true ratio of 0.05 it would be 45%, and at the observed 0.040
   it would be 60%. The 93% bound was not a lucky draw at a true ratio of 0. It is what this
   design gives at a ratio of a few percent.
3. **#27's data do not establish sham equivalence at ±10%** (TOST, α = 0.05 per bound:
   p = 0.054). It is established at ±20% (p = 3e-6) and not at ±5%. The upper 90% bound on
   the sham ratio is 1.101 (delta), 1.103 (Fieller) or 1.106 (bootstrap). With the
   drift-conditional SE (§4) the bound is 1.121, and ±10% fails clearly (p = 0.12).
4. **The sham equals the baseline only in w(0).** Sham minus baseline is +0.015 in the lag-1
   excess (z = +8.3). Per job, the lag-1 autocorrelation differs in jobs 1–3 (z = 3.9–8.5),
   and the difference is heterogeneous across jobs (Q p = 1e-5). The sham's w(1) and w(2) are
   1.21× and 1.38× the baseline's (z = −3.6 and −4.6). w(3) agrees (z = −0.4). The A2s_spec
   operations change the output process; they leave its w(0) where it was. #27's family
   tested w(0) only. That pattern (w(0) barely moved, w(2) strongly) is what Stage 2C's
   spectator sweep gives for q74 held in |1⟩ with ZZ to q75 (§7).
5. **The baseline's job1 dip in P(1) is specific to the baseline arm. It is not a device
   drift common to all arms.** In job1, all four baseline chunks are at 0.539–0.543. The four
   A2s_spec chunks interleaved with them (pubs 1, 8, 17, 28 among the baseline's 10, 20, 22,
   24) are at 0.569–0.577, and D1 is at 0.578–0.581. Baseline minus sham in P(1) is −0.033 ±
   0.002 in job1 and −0.002 to −0.005 in the other jobs (Q p = 1e-14). Within each job the
   baseline's chunks agree (within-job Q p = 0.89). So it is a job-level shift of one arm.
   The existing models do not explain it (§7).
6. **The sham's witness is itself heterogeneous.** Its w(0) is not constant across jobs
   (p = 0.002) or chunks (p = 0.001). A change-point scan of its w(0) influence finds a
   segment (p = 0.004), and the random-effect SD across chunks is 5.7e-4 (11% of w(0)). The
   baseline's and A2's witnesses pass the same tests (p ≥ 0.11).

## 1. Provenance

`results/summaries/phase8_stage0/provenance.json` holds the SHA-256, size and adding commit of
51 files:
- every Stage 4/4b/4c shot file (`shots_*.npz`), job record, `calibration_before.json`,
  `circuits.json`, gate, refit and piece file;
- Stage 4c's `summary.json`;
- the Phase 7 QASM (`phase7_ablation/qasm/*`, `phase7_ancilla/qasm/*`,
  `phase7_classical/d1.qasm`, `phase6/step.qasm`) and #20's `pieces.npz`.

It also holds the IBM job ids, each with its stage, status, shots, submission time and
approval URL:
- Stage 4: `dbo3cttt7fqc73akg8dg` (pilot).
- Stage 4b: `dbo7oi0qtr0c73ae4hk0` (pilot).
- Stage 4c: `dbocqblt7fqc73cmtd6g` (pilot), plus `dborq95t7fqc7388b3h0`,
  `dboseett7fqc7388c8p0`, `dbotns0qtr0c738dee10` and `dbouj7tt7fqc7388fphg` (jobs 0–3).

The run commits are recorded too. Stage 4c ran at `13ef8f1` (clean). `test_provenance_frozen`
fails if any frozen file changes or disappears, or if a new file matches the globs. The
script refuses to run when the hashes differ, and `--freeze` refuses to overwrite a different
freeze.

## 2. Reproduction (task 2)

The independent path (`qmem.reanalysis`) has its own versions of each step:
- decoding with `np.unpackbits`;
- a sliding match per k for the windows (`qmem.witness` counts gaps between ones);
- the determinant by cofactors;
- its own circular block resampling, stratified by job (the same seeds as the pipeline:
  `24 + 11 i` per arm, `+104729` per stratum);
- its own lag-1 counts, z, p and Holm arithmetic.

| check | result |
|---|---|
| decoding vs `qmem.witness.unpack_shots`, all pubs | identical |
| per-shot window counts vs `qmem.witness.per_shot_windows(common=True)`, all arms | identical |
| per-arm w(0..3), shot/block/primary SE, z, P(1), lag-1 and its SE, the 95% CI, per-job w(0) and P(1) (6 arms) | worst relative difference 2.0e-11 (A3's w(2) = 8e-8, i.e. a 1.7e-18 absolute difference); everything else ≤ 1e-12 |
| the gate, primary, descriptive A1/A3 calls and the five Holm decisions | identical |
| the secondary family's estimates, references, SEs, z, p, sham ratio and its SE | ≤ 4e-13 relative |

At the printed precision every reported number matches:
- baseline w(0) = −5.15e-3 ± 1.27e-4, z = −40.5;
- A2 w(0) = −2.04e-4 ± 1.41e-4;
- ratio 4.0%, 95% CI [0, 9.3%], bound at 93% of the threshold;
- sham ratio 1.044 ± 0.034, z = −1.32;
- nothing in the Holm family rejected at α = 0.01.

**No difference, however small, beyond floating point.**

## 3. The narrow margin (task 3)

A2's 95% upper bound on |w_A2| / |w_base| under the predeclared choices (`fragility.csv`).
Choices marked † are descriptive only (not in the call).

| choice | A2 SE | bound | call |
|---|---|---|---|
| **pre-registered** (max(shot, block 500), stratified by job) | 1.41e-4 | **0.0931** | vanishes |
| shot bootstrap only / delta method, i.i.d. | 1.41e-4 / 1.38e-4 | 0.0931 / 0.0921 | vanishes |
| block 250 / 500 / 1,000 / 2,000, stratified by job (effective 250/500/625/625) | 1.41e-4 (block SE alone 1.06e-4 … 8.7e-5) | 0.0931 | vanishes |
| block 250 / 500 / 1,000 / 2,000, unstratified | 1.41e-4 (block SE alone 1.08e-4 … 6.3e-5) | 0.0931 | vanishes |
| block 500, stratified by pub chunk (effective 208) | 1.41e-4 | 0.0931 | vanishes |
| ratio: delta method including the baseline's SE | | 0.0931 | vanishes |
| ratio: Fieller 95% (interval [−0.014, 0.093]) | | 0.0933 | vanishes |
| ratio: percentile bootstrap, 97.5% (95% [−0.009, 0.097]) | | **0.0972** | vanishes |
| \|ratio\|: bootstrap, one-sided 95% | | 0.0880 | vanishes |
| fixed effect over jobs (baseline −5.56e-3) | 1.38e-4 | 0.0902 | vanishes |
| DerSimonian–Laird random effect over jobs (τ² = 0 for A2) | 1.38e-4 | 0.0903 | vanishes |
| **without job1** (the baseline's dip) | 1.63e-4 | **0.1010** | **ambiguous** |
| q75 model (pilot fit, calibrated, #26's refit): observed baseline | | 0.0931 (all three) | vanishes |
| q75 model: H_q75 baseline prediction (−6.90e-3, the same under all three) as denominator | | 0.0695 | vanishes |
| † without job0 / job2 / job3 | 1.48e-4 / 1.71e-4 / 1.61e-4 | 0.089 / **0.114** / 0.099 | –/ambiguous/– |
| † job0 / job1 / job2 / job3 alone | 2.3–3.5e-4 | 0.196 / 0.113 / 0.122 / 0.151 | ambiguous |

- **The block length cannot move the bound.** A2's primary SE is set by the shot bootstrap,
  and the block SE is smaller at every length (6.3e-5 to 1.08e-4). A2 has no positive
  shot-to-shot correlation to inflate it.
- **The q75 model cannot move it either.** The rule uses only the data. All three models
  give the same H_q75 baseline prediction, because q75's decay acts only under A2's
  measurement. Their ratios agree to 1e-12.
- **The ratio method moves it by at most +0.004** (the bootstrap, 0.097).
- **Pooling across jobs is conservative.** The per-job witnesses are −5.06, −5.84, −5.70 and
  −5.60e-3. The pooled g gives −5.15e-3, a 7% smaller magnitude than the fixed effect (§4).
- **The only flips come from dropping a job.** Losing a quarter of A2's shots leaves the
  bound at about 0.09–0.11, depending on which job goes.

**Power of a replicate** (same shots; A2 SE 1.41e-4; the baseline drawn around −5.15e-3 ±
1.27e-4, with the gate applied; Monte Carlo with 400,000 draws, which agrees with the analytic
value to 0.002):

| true \|w_A2\| / \|w_base\| | P(vanishes) |
|---|---|
| 0 | 0.910 |
| 0.040 (observed) | 0.599 |
| 0.05 | 0.449 |

**A2 shots for the bound at ≤ 70% of the threshold** (≤ 0.07; the baseline as observed; the SE
scaled from 1.41e-4 at 10,000 shots; `power_shots.csv`):

| criterion | true ratio 0 | true ratio 0.05 |
|---|---|---|
| observed \|w_A2\| = 2.04e-4 held fixed | 30,900 | n/a |
| expected bound ≤ 0.07 | 11,600 | 71,600 |
| P(bound ≤ 0.07) ≥ 0.8 | **16,000** | 146,000 |
| P(bound ≤ 0.07) ≥ 0.9 | **19,800** | 196,000 |
| (for comparison) P(bound < 0.10) ≥ 0.9 | 9,700 | 31,300 |

A true ratio of 0.05 is half the threshold. Resolving it with a bound at 70% needs about
15–20× the current A2 shots. Stage 1 should choose its true-ratio assumption first; the
shots follow from it.

## 4. Uncertainty conditional on the drift (task 4)

**What drifts.** The baseline's P(1) is 0.577, **0.541**, 0.573 and 0.573 by job (Q across
jobs: 432 on 3 df). `qmem.drift`'s change-point scan of the time-ordered baseline finds the
segment [3,750, 7,500), exactly job1 (p = 0.001). The g(k) are not homogeneous across jobs
(χ² p < 1e-15). The baseline's **w(0) is constant across jobs (p = 0.11) and chunks
(p = 0.16)**. A2 passes every test (change-point p ≥ 0.054; w(0) constancy p ≥ 0.98), and so
do D1 and A1. The sham fails its w(0) tests (finding 6).

**Balance.** Every arm has exactly a quarter of its shots in each job (the schedule's
`split_even`). A2's pubs sit earlier in jobs 2 and 3 (mean position 5.7 and 8.7, against the
baseline's 12.8 and 21.8). But the baseline's dip covers all of job1's baseline pubs, and
other arms in the same job do not show it (finding 5). So the position within a job is not
what matters, and the dip does not unbalance A2 against the baseline: A2 has no P(1) or w(0)
shift in job1 (P(1) 0.3805, against 0.3829, 0.3808 and 0.3829).

**SEs of w(0) conditional on the drift** (`summary.json` → `drift.se`):

| SE | baseline | A2 | A2s_spec |
|---|---|---|---|
| pre-registered | 1.27e-4 | 1.41e-4 | 1.18e-4 |
| block 500, stratified by chunk | 1.23e-4 | 1.37e-4 | 1.17e-4 |
| unstratified block 2,000 (job boundaries joined) | 1.38e-4 | 6.4e-5 | 1.94e-4 |
| random effect over jobs (τ) | 1.65e-4 (τ = 2.0e-4) | 1.38e-4 (τ = 0) | n/a |
| random effect over 16/12/16 chunks (τ) | 1.44e-4 (τ = 3.0e-4) | 1.32e-4 (τ = 0) | **1.87e-4 (τ = 5.7e-4)** |

A2's SE does not grow under any drift-conditional method; the largest is the pre-registered
one. The baseline's grows by up to 30%, but the rule does not propagate the baseline's SE.
Propagating it changes the bound by less than 1e-4 (§3, delta and Fieller).

**Could a drift-induced bias in the baseline cross 10%?** The bound reaches 10% if |w_base|
falls to 4.80e-3: 6.9% below the observed value, or 2.8 of its SEs.

| baseline used | \|w_base\| | bound |
|---|---|---|
| observed (pooled) | 5.15e-3 | 0.093 |
| fixed effect over jobs (the pooling effect removed) | 5.56e-3 | 0.086 |
| job1 replaced by the mean of the other jobs | 5.45e-3 | 0.088 |
| the weakest job (job0) | 5.06e-3 | 0.095 |
| observed less 1 between-job SD (3.4e-4) | 4.81e-3 | 0.0997 |
| observed less 2 between-job SD | 4.47e-3 | 0.107 |

The drift that is there biases the pooled baseline *toward zero*. Pooling g across jobs at
different P(1) shrinks |w(0)| by 4.0e-4. Removing that bias lowers the bound, so the observed
drift does not push the bound across 10%. A bias of one full between-job SD on top of the
observed value would bring it to 0.0997, and two SDs would cross. No mechanism in the data
gives a bias of that size in that direction.

## 5. Baseline against the sham, per job (task 5)

Sham minus baseline (`sham_jobs.csv`; SEs inside each job; Q across the four jobs):

| statistic | job0 | job1 | job2 | job3 | pooled (FE) | Q p across jobs |
|---|---|---|---|---|---|---|
| P(1) | +0.002 (0.8σ) | **+0.033 (16.7σ)** | +0.005 (2.5σ) | +0.004 (2.1σ) | +0.011 | 5e-15 |
| w(0) | +2e-5 (0.0σ) | −1.7e-4 (−0.4σ) | −4.8e-4 (−1.4σ) | −3.0e-4 (−0.8σ) | −2.4e-4 ± 1.9e-4 | 0.79 |
| lag-1 autocorrelation | +0.005 (1.4σ) | **+0.027 (8.5σ)** | +0.019 (5.6σ) | +0.013 (3.9σ) | +0.016 | 1e-5 |
| lag-2 autocorrelation | +0.007 (2.1σ) | +0.009 (2.5σ) | +0.005 (1.5σ) | +0.007 (2.1σ) | +0.007 | 0.87 |
| lag-3 autocorrelation | −0.000 | −0.007 (−1.9σ) | −0.002 | +0.005 (1.3σ) | −0.001 | 0.14 |

The two arms agree in w(0) in every job, with no heterogeneity. They differ in P(1) and in the
lag-1 and lag-2 autocorrelation, most in job1. The lag-2 difference is homogeneous across
jobs (+0.007, about 4σ pooled), so the sham changes the short-range correlations even
without the dip. Pooled over all shots, the sham's lag-1 excess is 0.193 against 0.178
(z = +8.3; H_q75 predicts 0.157 for both). Its w(1) and w(2) are larger in magnitude
(z = −3.6 and −4.6; finding 4).

## 6. Equivalence, not non-significance (task 6)

The sham ratio is R = w_sham / w_base = 1.044, with delta SE 0.0345. TOST at α = 0.05 per
bound (equivalently, a 90% CI inside the margin):

| margin | TOST p (delta) | 90% CI delta [0.988, 1.101] | Fieller [0.989, 1.103] | bootstrap [0.989, 1.106] | drift-conditional SE 0.047: [0.968, 1.121] |
|---|---|---|---|---|---|
| ±5% | 0.44 | not established | not established | not established | not established (p = 0.45) |
| ±10% | **0.054** | **not established** | not established | not established | not established (p = 0.12) |
| ±20% | 3e-6 | established | established | established | established (p = 4e-4) |

The smallest margin #27's data would establish is 10.1% (delta), 10.3% (Fieller) or 10.6%
(bootstrap). **#27 shows the sham is within ±20% of the baseline, not within ±10%.** At true
R = 1, the design had 79% power for ±10%, and this draw missed by a hair.

**Recommendation for Phase 8 (predeclared in the config before the TOST was run): margin
±10% on the ratio, TOST at α = 0.05 per bound.** The reasons are physical, not the observed
data:
- **The margin matches the primary's resolution.** The primary claim is that A2 removes at
  least 90% of the witness (the vanish fraction, 10%). A control that might itself remove up
  to 10% of the witness cannot carry A2's effect, and one that might remove 20% would
  explain a fifth of it. ±10% is the widest margin at which "sham equal" supports "the drop
  needs q75's measurement" at the resolution of the primary.
- **H_q75 predicts R = 1.000** for A2s_spec: the delay and the q74 measurement do not touch
  q75's state in the model. Stage 2C's spectator sweep (`docs/phase7-spectator.md` §2) gives
  the physical scale. At typical strengths (ZZ ≈ 10 kHz), thermal or frozen q74, q55 and
  q59–q75 ZZ move w(0) by 1–3% (0.973–1.02). A ≥ 10% change needs |ζ| ≳ 30 kHz to a thermal
  q74, ≳ 10 kHz to a q74 held in |1⟩, or a readout Stark shift of ≳ 10–30 kHz. So ±10% is
  where a spectator or crosstalk path becomes a material alternative to "the drop needs q75's
  measurement". A margin of ±5% would sit inside the typical-strength spread, which the model
  treats as negligible. It would test model error rather than an alternative path.
- **α = 0.05 per bound** is the standard TOST choice: a 90% CI inside the margin. The sham is
  a control, not a claim in the Holm family.

Shots per arm (baseline and sham equal) for ±10% at α = 0.05 (`tost_plan.csv`):

| SE model | true R | 80% power | 90% power |
|---|---|---|---|
| #27's primary SEs, scaled | 1.00 | 15,300 | 19,300 |
| #27's primary SEs, scaled | 0.97 or 1.03 | 22,600 | 31,200 |
| drift-conditional (random effect over chunks), scaled | 1.00 | **27,800** | **35,100** |
| drift-conditional, scaled | 0.97 or 1.03 | 41,100 | 56,800 |

Use the drift-conditional rows: the sham's witness was heterogeneous across chunks on this
device (finding 6). For ±5% at true R = 1 the shots are about 4× higher (61,000 / 77,000 with
the primary SEs). For ±20%, 3,800 / 4,800 are enough.

## 7. The baseline at 75% of its prediction (task 7)

| arm | w(0) observed | H_q75 | fraction (95% CI) | P(1) observed − predicted | lag-1 observed / predicted |
|---|---|---|---|---|---|
| baseline | −5.15e-3 | −6.90e-3 | **0.747 [0.711, 0.783]** | −0.016 | 0.178 / 0.157 |
| A2s_spec | −5.38e-3 | −6.90e-3 | 0.780 [0.747, 0.814] | −0.005 | 0.193 / 0.157 |

The CI propagates only the observed SE. The prediction has no SE. Phase 6 found the same
shortfall a day earlier (72% of the calibrated w(0), `docs/phase6-hardware-run.md`).

What the existing models can and cannot attribute it to, without fitting anything new:

- **q75 decay: cannot.** The three q75 models predict the same baseline w(0), −6.90e-3. The
  decay acts only under A2's mid-circuit measurement. The pilot's refit (a = 7.8e-9) and A2's
  P(m = 1) = 0.4964 (calibrated 0.4925; #26's refit 0.419) also say q75 did not decay here.
- **P(1) offset: cannot, and it points the wrong way.** The sham is close in P(1) (−0.005)
  and still at 78%. Across chunks, P(1) and w(0) are positively correlated (baseline r = 0.35,
  sham r = 0.59): a lower P(1) goes with a *more* negative w(0). The baseline's job1, with
  the lowest P(1), has the largest |w(0)| (−5.84e-3).
- **Readout: not beyond what is already in the model.** The calibrated model includes the
  live readout errors (q75 0.0127, q59 0.0020). D1, the memoryless control with q59's
  readout chain, gives w(0) = −6e-6 ± 5e-6 and a lag-1 excess *below* its prediction. So no
  readout-chain memory adds to or subtracts from the witness at this scale.
- **Stage 2C's spectator models: possibly, but untested.** Two rows of
  `docs/phase7-spectator.md` §2 reach this size. Extra pure dephasing on q75 during readout
  at Γ ≈ 1e4 s⁻¹ gives 0.83×. A readout Stark shift on q75 of 16 kHz gives 0.92×. That note
  already named the dephasing row as the one plausible account of Phase 6's deficit, and
  neither has been measured. A q74 held in |1⟩ with ZZ ≈ 10 kHz gives w(0) 1.02× but
  w(2) 1.73×. That is the same pattern as the sham against the baseline (w(0) 1.04×,
  w(2) 1.38×; finding 4), but this audit does not fit it.
- **What the data show instead:** the shape of g(k) differs from the model. g(1) is 0.87× the
  prediction, g(2) is 1.30× and g(5) is 1.38×, and the lag-1 excess is 13% (baseline) to 23%
  (sham) higher. The shortfall is a model-shape mismatch that persists across runs (Phase 6:
  72%; Stage 4c: 75%). It is not a drift and not q75 decay, P(1) or readout. The candidates
  among the existing models are readout-window dephasing or a Stark shift on q75. Testing
  them is Stage 1's business, not this audit's.

## What this means for Stage 1

- Size A2 for the bound at ≤ 70% of the threshold with ≥ 80% probability at a true ratio of 0:
  **about 16,000–20,000 A2 shots**. If a ratio of 0.05 must be resolved too, it takes about
  150,000–200,000. Pre-register a rule for losing a job (a minimum shot count, or a
  pre-specified per-job random effect) so that a single dropped job cannot flip the call.
- Pre-register the sham's equivalence as **TOST at ±10%, α = 0.05 per bound**, with about
  **28,000–35,000 shots per arm** (drift-conditional SEs, 80–90% power at R = 1). Test the
  sham at w(1), w(2) and the lag-1 excess as well, not only at w(0) (finding 4).
- Treat a job-level P(1) shift in one arm as a guard-rail event with its own pre-specified
  response. #27's baseline job1 dip was arm-specific and the guard rails flagged it, but the
  rule did not act on it (finding 5).
