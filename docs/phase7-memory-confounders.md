# Phase 7: is the Phase 6 witness memory in q75? Synthesis and claim ladder (#17)

Coordinator synthesis of Phase 7 (#17, children #18–#27). It asks whether the nonzero Hankel
witness from the `ibm_kingston` PMD N = 3 run (#16, [phase6-hardware-run.md](phase6-hardware-run.md))
comes from memory carried by the intended persistent qubit, q75, or from another physical or
statistical memory path. This note adds no new analysis. Each statement links to the child
note that establishes it, and those notes have the data, code and provenance.

Phase 7 used 53 s of QPU time (#25 and #26, 3 s each; #27, 47 s), 67.22 s counted against
the owner's 120 s cap. Everything else was analysis of the committed Phase 6 shots and local
simulation.

## The claim ladder

| rung | status | basis |
|---|---|---|
| 1. Nonzero higher-rank *output* statistics | **supported** | #18 |
| 2. A stationary 2-state HMM is excluded | **supported**, robust to drift and mixed 2-state regimes | #18, #19 |
| 3. Drift, nonstationary mixtures and control artefacts are excluded | **supported in models**; on hardware, only D1 and the A2s_spec sham were measured, and both agree | #19, #21–#23, #27 |
| 4. Leaving q75 undisturbed is causally necessary | **supported under the pre-registered analysis, at the edge of the design's resolution; not robust to leave-one-job-out**: A2 *vanishes*; sham equal in w(0) only, equivalent at ±20% (post hoc gate) | #20, #24–#27; #29, #30 |
| 5. A one-qubit physical memory-dimension claim | **not supported**, and out of reach of this design | #18, #19 |

**Verdict.** The device's output process on q59 has classical dimension above 2. That is
robust to every statistical and classical-artefact explanation tested. The pre-registered
ablation (#27) shows that the witness needs q75 to be left undisturbed: measuring, resetting
or randomising q75 every step removes it to below 10% of the baseline under the
pre-registered analysis, and a sham on spectator q74 leaves w(0) unchanged. That call sits at
the edge of the design's resolution and is not robust to leaving out one job (#29). Phase 7
has **not** shown that q75's *coherence* carries the memory, that the memory is quantum in
any stronger sense, or that the device uses one qubit of memory. A general 3-state classical
HMM reproduces the output statistics, and a 6-state classical model whose q75 measurement
disturbs its hidden state reproduces every Stage 4c arm (#30,
[phase8-claim-framework.md](phase8-claim-framework.md)).

*Revised after Phase 8 Stage 0 (#28): #29's re-analysis and #30's classical intervention
fit. Rung 4 previously read "supported: A2 vanishes, sham equal", and the verdict said the
witness "needs q75 to stay coherent".*

## Rung 1: higher-rank output statistics (supported)

- **Reproduced exactly.** An independent numpy estimator (`qmem.witness`) re-derives
  w(0) = −0.00494 from the raw shots ([phase7-audit.md](phase7-audit.md), #18). Tests fail on
  bit-order, window and bootstrap bugs.
- **The uncertainty was overstated.** The Phase 6 bootstrap treated shots as identically
  distributed, but shots 5,300–8,000 (in returned order) come from a different process. With
  SEs from contiguous blocks of shots, z(0) = −39 rather than −54. w(0) is the predeclared
  primary witness. The other offsets are reported separately: z(1..3) = −8.5, −4.7, −1.3.
- **Replicated (#27).** The concurrent baseline of the ablation, 15,000 shots on the same
  qubit pair on 2026-11-09, gives w(0) = −5.15e-3 at z = −40.5, which
  passes the pre-registered replication gate. It is 75% of the H_q75 prediction, with P(1)
  0.566 against 0.582 (a dip to 0.541 in one job).
- **Limitation.** One qubit pair on one device. Three later baseline measurements (#25, #26,
  #27) sit 0.013–0.029 below Phase 6's P(1).

## Rung 2: a stationary 2-state HMM is excluded (supported)

- Every 2-state HMM gives w(0) = 0 exactly. Data simulated from the 2-state HMM fitted to the
  device give |z| ≤ 1.7 (#18).
- **It holds within segments.** w(0) is −0.00585 (z = −51) outside the episode and −0.00531
  (z = −23) inside it. Across the 17 blocks of 1,000 shots outside the episode it is
  consistent with a single value (p = 0.90) ([phase7-drift.md](phase7-drift.md), #19).
- **Mixtures and drift don't produce it.** #19 searched adversarially over mixtures of 2 or 3
  regimes, continuous drift, drift within a shot, and telegraph fluctuators from 10 µs to
  1 s. No data-consistent null exists: every best fit misses g(k) by |z| ≥ 21. The largest
  witness any of them reaches is 21–24% of the observed one.
- **Limitation.** Whole-shot bootstraps alone don't exclude nonstationary mixtures. This rung
  rests on #19's explicit null families, which cover the classes listed, not every
  conceivable one. Shot order is IBM's returned order, which can't be verified as execution
  order from one execution span.

## Rung 3: drift and control artefacts are excluded (supported in models, not measured)

- **Classical readout and processing** ([phase7-classical.md](phase7-classical.md), #23).
  Any mixture of memoryless shots gives a positive semidefinite Hankel matrix, so w ≥ 0, the
  opposite sign to the device's. Readout persistence on a memoryless source gives w = 0
  exactly. On device-like 2-state sources, plausible artefacts reach a median of 0.03% of the
  witness. One atypical oscillating chain reaches 11%. On the calibrated model, every
  artefact *shrinks* |w|. PMD has no history-conditioned feed-forward, only measurement and
  reset.
- **q59 residual memory** ([phase7-ancilla.md](phase7-ancilla.md), #21). At plausible reset
  infidelity and leakage, it changes w(0) by ≤ 6%. Only leakage near an extreme 5e-3 per
  readout reaches 25%, and that would raise P(1) by 0.022, while the device sits 0.005 below
  the calibration. On its own, q59's memory only reduces |w|.
- **Spectators and crosstalk** ([phase7-spectator.md](phase7-spectator.md), #22). At
  published typical strengths (static ZZ, thermal population, readout Stark shift), together
  they change w(0) by 5.5%. With q75 reset every step, they leave |w| ≤ 4e-12. None
  reproduces the episode.
- **What remains.** The mismatch with the calibration is best described as a q75 phase offset
  of about 0.04 rad per step plus readout asymmetry (#19, #21), or as an effective sticky
  record with ε = 0.027 (#23). Neither needs a hidden memory. Readout persistence and Heron
  residual ZZ have no published values.
- **Measured on hardware (#27).** D1's lag-1 excess is −0.0036 (z = −2.46, not rejected
  under Holm). The sticky record and q59 leakage both predict a *positive* excess
  (+0.009 to +0.050), so neither is indicated. The A2s_spec sham, which delays q75 and
  measures spectator q74, matches the baseline (ratio 1.04 ± 0.03).
- **Limitation.** The other mechanisms remain models with assumed or literature parameters.
  B2, D3 and C3 (tier 3) were never run.

## Rung 4: is leaving q75 undisturbed causally necessary? (supported under the pre-registered analysis; not robust to leave-one-job-out)

**The design** ([phase7-ablation.md](phase7-ablation.md), #20; pre-registered in
[phase7-preregistration.md](phase7-preregistration.md), #24). The main arms are:
- A2: q75 measured in Z every step, which makes it a classical bit;
- A2s_spec: the same measurement on spectator q74, as a sham;
- A1: q75 reset every step;
- A3: random Paulis on q75;
- D1: a memoryless circuit with the same readout cadence.

They run interleaved with a concurrent baseline. The pre-registered rule gives:
- *vanishes* when the upper 95% bound on |w_A2(0)| is below 10% of the baseline's;
- *persists* when it is ≥ 5 SE from 0 and ≥ 25% of the baseline;
- *ambiguous* otherwise.

In #24's validation, no alternative hypothesis gave a wrong decisive call.

**What happened** ([phase7-hardware.md](phase7-hardware.md)):
- **Stage 4 (#25), owner-approved.** The pilot failed gate (b), baseline P(1) 0.552 against
  0.581 ± 0.02. It failed gate (c), A2's q75 record P(m = 1) 0.419 against 0.4925 ± 0.03,
  by about 14 SE. 3 s charged.
- **Stage 4b (#26), owner-approved, 120 s cap.**
  - A 2-parameter refit (extra decay of q75 under each mid-circuit measurement,
    a = 0.123) matched #25's record.
  - A dated, post hoc amendment (prereg §9) set (c) to the refit's prediction. Under it, (b)
    stops only beyond ±0.08, and (d) moved to `circuits_execution_time_ns`.
  - The new pilot failed (c) in the opposite direction: 0.4927 against the refit's
    0.4191 ± 0.021. It matched the *original* calibrated model (0.4925) under an identical
    reported calibration, about 5.5 h after #25. 3 s charged.
  - As pre-registered, both stages stopped with no retry, re-mapping or main job.
- **Stage 4c (#27), owner-approved, same cap.** A second dated, post hoc amendment
  (prereg §10) made (c) a recorded diagnostic that never stops the run. It took q75's model
  for the analysis from the pilot's own refit, fixed before any main job, with the
  calibrated model and #26's refit as sensitivity rows. The decision rule, arms, shots,
  schedule, mapping and secondary family were unchanged. Its validation (4,800 replicates)
  gave no wrong decisive call. The pilot passed (a), (b) and (d). Its q75 record matched the
  calibrated model (P(m = 1) 0.489, fitted decay a ≈ 0), so #25's decay did not return. The
  four main jobs ran with no calibration violation or usage stop. 47 s charged.

**The outcome (#27): A2 *vanishes*; the sham is equal in w(0) only, equivalent at ±20% (#29).**

| quantity | value |
|---|---|
| replication gate (baseline w(0), 15,000 shots) | −5.15e-3 ± 1.27e-4, z = −40.5: **passes** |
| A2 w(0) (10,000 shots) | −2.04e-4 ± 1.41e-4, **4.0%** of baseline (95% CI [0, 9.3%]) |
| vanish bound \|w\| + 1.96 SE against 0.10 \|w_base\| | 4.80e-4 < 5.15e-4: ***vanishes*** (bound at 93% of threshold) |
| A2s_spec − baseline | −2.3e-4, ratio 1.04 ± 0.03, z = −1.32: not rejected |
| A1 (reset) w(0) | −1.29e-4, z = −0.70: not rejected; vanishes descriptively |
| A3 (random Paulis) against its H_q75 value | z = +0.52: not rejected; vanishes descriptively |
| D1 lag-1 excess against prediction | −0.0036, z = −2.46, p = 0.014: not rejected (Holm, α = 0.01) |

The decision is the same under all three q75 models. They differ only in how much of q75's
coherence A2's measurement can leave (observed ratio 0.040):

| q75 model | decay a | surviving coherence c consistent with the CI |
|---|---|---|
| pilot fit (the analysis's model) | 7.8e-9 | 0 to 0.55 |
| calibrated (sensitivity) | 0 | 0 to 0.55 |
| #26's refit (sensitivity) | 0.1225 | 0 to 0.60 |

So 80% surviving coherence is ruled out, but 50% can't be told apart from full dephasing.

**What this does and doesn't imply.**
- **Supported, under the pre-registered analysis:** the rank > 2 output statistics need q75
  to be left undisturbed. A Z measurement, a reset or a random Pauli on q75 each step removes
  the witness; delaying q75 and measuring a spectator leaves w(0) unchanged.
- **Not robust (#29,** [phase8-stage0-audit.md](phase8-stage0-audit.md)**).** #27 reproduces
  exactly, but leaving out job1 puts A2's bound at 10.1% of the baseline, and leaving out job2
  at 11.4%: both *ambiguous*. A same-design replicate would call *vanishes* only 91% of the
  time at a true ratio of 0. The sham is equivalent to the baseline at ±20% (TOST) but not at
  ±10% (p = 0.054), and it differs from the baseline in the lag-1 excess (z = 8.3), w(1) and
  w(2). Job1's P(1) dip hits all four baseline pubs and none of the other arms' pubs
  interleaved with them; nothing yet explains it.
- **Not coherence (#30,** [phase8-claim-framework.md](phase8-claim-framework.md)**).** A
  6-state classical hidden-state model with per-arm intervention maps fits every Stage 4c
  arm, A2's q75 record included, within #27's CIs. Its q75 measurement disturbs the hidden
  state (the Leggett–Garg clumsiness loophole). Only classical memories that q75's measurement
  leaves undisturbed are excluded. So the result shows that q75 must not be disturbed, not
  that its coherence carries the memory.
- **Not shown:** that no other memory path exists below the vanish bound (about 5e-4 in
  w(0)); "quantum memory" in a stronger sense; a memory dimension (rung 5).
- **Caveats.**
  - **Narrow margin.** The vanish bound is 93% of the threshold.
  - **Post hoc gate.** Gate (c) as a diagnostic and the pilot-fitted q75 model come from the
    dated amendments §9 and §10, written after the earlier pilots. The decision rule is the
    original pre-registered one and doesn't use the q75 record, and A2's predicted witness is
    exactly 0 under every q75 model.
  - **Non-stationary q75.** q75's behaviour under repeated mid-circuit measurement changed
    by hours, invisibly to IBM's calibration data: about 12% extra decay per measurement on
    #25's run and none on #26's or #27's.
  - **Baseline below prediction.** The baseline replicates at 75% of the H_q75 witness, and
    its P(1) dipped to 0.541 in one main job, though its w(0) held and no segment test flagged
    a job.

## Rung 5: a one-qubit physical memory dimension (not supported)

- A general stationary 3-state classical HMM reproduces the episode-free witness within every
  CI (w(0) = −0.0062), and a 4-state one reproduces the episode (#19).
- On random or interleaved splits, 3–6 state HMMs predict held-out shots better than the
  calibrated 1-qubit model, by 0.0004–0.0005 nats/symbol (|z| = 9–15). The Phase 6 claim of
  the opposite was a train/test-split artefact (#18). Only the advantage over 2-state HMMs
  survives.
- #27's *vanishes* outcome shows only that q75 must not be disturbed. It does not
  bound the memory's dimension, because leaky resets, spectators and the readout chain can
  all add states.

## What would move the ladder

- **Rung 4, firmer:** A2 at more shots (≥ 16,000–20,000, #29) to widen the margin below the
  vanish bound, and a sham shown equivalent at ±10% in g(k) and lag-1 as well as w(0). Neither
  moves the question of coherence: #30's classical model reproduces A2's vanishing at any
  shot count. That needs reversible, composable phase insertions on q75 with held-out
  sequences ([phase8-claim-framework.md](phase8-claim-framework.md) §4), not a partial
  dephasing scan alone. Phase 8 (#28) designs these; any new QPU run needs a new approval from
  the owner, and Phase 7's 120 s cap does not carry over.
- **Rung 3, measured:** B2, D3 and C3 (tier 3) would measure q59's reset, the readout chain
  and crosstalk directly. They need separate approval.
- **Rung 1 replication:** a run on another qubit pair or device. The same pair has now
  replicated (#27).
- **Rung 5:** out of reach of output statistics from one qubit pair. It needs process
  tomography of the instrument, or a dimension witness that bounds every physical path.
