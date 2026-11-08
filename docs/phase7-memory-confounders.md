# Phase 7: is the Phase 6 witness memory in q75? Synthesis and claim ladder (#17)

Coordinator synthesis of Phase 7 (#17, children #18–#26). It asks whether the nonzero Hankel
witness from the `ibm_kingston` PMD N = 3 run (#16, [phase6-hardware-run.md](phase6-hardware-run.md))
comes from memory carried by the intended persistent qubit, q75, or from another physical or
statistical memory path. This note adds no new analysis. Each statement links to the child
note that establishes it, and those notes have the data, code and provenance.

Phase 7 used 6 s of QPU time (#25 and #26, 3 s each). Everything else was analysis of the
committed Phase 6 shots and local simulation.

## The claim ladder

| rung | status | basis |
|---|---|---|
| 1. Nonzero higher-rank *output* statistics | **supported** | #18 |
| 2. A stationary 2-state HMM is excluded | **supported**, robust to drift and mixed 2-state regimes | #18, #19 |
| 3. Drift, nonstationary mixtures and control artefacts are excluded | **supported in models, not measured on hardware** | #19, #21–#23 |
| 4. Preserving q75 is causally necessary | **unresolved**: the hardware ablation stopped twice at its pilot gate | #20, #24–#26 |
| 5. A one-qubit physical memory-dimension claim | **not supported**, and out of reach of this design | #18, #19 |

**Verdict.** The device's output process on q59 has classical dimension above 2. That is
robust to every statistical and classical-artefact explanation tested. Phase 7 has **not**
shown that this memory sits in q75, that it is quantum, or that the device uses one qubit of
memory. A general 3-state classical HMM reproduces the output statistics. The causal test of
q75 was designed, pre-registered and approved, but never reached its main jobs.

## Rung 1: higher-rank output statistics (supported)

- **Reproduced exactly.** An independent numpy estimator (`qmem.witness`) re-derives
  w(0) = −0.00494 from the raw shots ([phase7-audit.md](phase7-audit.md), #18). Tests fail on
  bit-order, window and bootstrap bugs.
- **The uncertainty was overstated.** The Phase 6 bootstrap treated shots as identically
  distributed, but shots 5,300–8,000 (in returned order) come from a different process. With
  SEs from contiguous blocks of shots, z(0) = −39 rather than −54. w(0) is the predeclared
  primary witness. The other offsets are reported separately: z(1..3) = −8.5, −4.7, −1.3.
- **Limitation.** This is one run of 20,000 shots on one qubit pair on one day. Two later
  500-shot baseline pilots (#25, #26) had P(1) of 0.552 and 0.565, against Phase 6's 0.579–0.581,
  and they were not sized to re-measure w(0).

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
- **Limitation.** Each of these is a model with assumed or literature parameters. None was
  measured on the device. The controls that would measure them (D1's lag-1 excess, A2s_spec,
  B2, C3) were designed and pre-registered but never run.

## Rung 4: is preserving q75 causally necessary? (unresolved)

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

**What this does and doesn't imply.**
- No ablation contrast was measured, so the stops bear on rung 4 in neither direction.
- q75's behaviour under repeated mid-circuit measurement changed by hours, invisibly to
  IBM's calibration data: about 12% extra decay per measurement on #25's run and none on
  #26's. Any physical reading of the original witness has to allow for a memory qubit whose
  noise isn't stationary between jobs.
- The decision rule doesn't use the q75 record. Under every q75 model, A2's predicted witness
  is exactly 0, because A2's q75 is a classical bit each step. A future run could gate on
  decoding, timing and the baseline alone. That is a new decision for the owner, and it would
  need its own dated amendment before submission.

**Flags from #26 for any future run.**
- **The refit is incomplete even on #25's data.** The record's step-to-step transitions
  P(m_{t+1} | m_t, y_t) are more persistent than either model, by |z| ≈ 3 for the refit and
  6–12 for the calibrated model. A third parameter wasn't added.
- **The partial-coherence reading depends on the q75 model.** Under the refit, "A2 keeps 80%
  of q75's coherence" predicts 24.8% of the baseline witness, at the *persists* threshold. Its
  prediction moves from *persists* to *ambiguous*. "Keeps 50%" moves to *vanishes*. So a
  *vanishes* or *persists* call is model-independent, but how much coherence it allows is not.
- **(b) is non-stopping within ±0.08 under the amendment,** a post hoc judgement resting on
  #19's bound on level shifts. The stricter alternative, a hard tolerance of about ±0.04
  derived from drift, would risk stopping on job-scale drift like #25's.
- **Ambiguous rates rise under the refit.** Under q59 leakage at its fit they go from 22.5%
  to 33%. Under the sticky record they go from 10.5% to 16.5%. D1's lag-1 test detects both.

## Rung 5: a one-qubit physical memory dimension (not supported)

- A general stationary 3-state classical HMM reproduces the episode-free witness within every
  CI (w(0) = −0.0062), and a 4-state one reproduces the episode (#19).
- On random or interleaved splits, 3–6 state HMMs predict held-out shots better than the
  calibrated 1-qubit model, by 0.0004–0.0005 nats/symbol (|z| = 9–15). The Phase 6 claim of
  the opposite was a train/test-split artefact (#18). Only the advantage over 2-state HMMs
  survives.
- Even a *vanishes* outcome for A2 would show only that q75 must stay coherent. It would not
  bound the memory's dimension, because leaky resets, spectators and the readout chain can
  all add states.

## What would move the ladder

- **Rung 4:** the pre-registered ablation (#24's tier 1 and tier 2, about 55 s upper bound),
  run under a gate that doesn't depend on q75's absolute record level, with q75's record
  reported as a diagnostic. Its validation and decision rule are already on `main`.
- **Rung 3, measured:** D1 (lag-1 excess to ±0.001) and A2s_spec in the same jobs. B2, D3 and
  C3 are tier 3 and would need separate approval.
- **Rung 1 replication:** a second full run, or the baseline arm of the run above, which the
  replication gate requires to give w(0) < 0 at |z| ≥ 5.
- **Rung 5:** out of reach of output statistics from one qubit pair. It needs process
  tomography of the instrument, or a dimension witness that bounds every physical path.
