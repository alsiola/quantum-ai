# Phase 8 Stage 2: a coherence-specific test on q75/q59 (#32, #34, #35)

Local theory, simulation, QASM and an unsubmitted plan only: no QPU, no IBM service. This note
asks whether any feasible experiment on q75/q59 makes a classical hidden-state model with ≤ D
states fail while the trained instrument's fixed prediction passes. That would support Claim 3
(coherence dependence, [phase8-claim-framework.md](phase8-claim-framework.md) §2) under stated
assumptions, against the bar of §4 there.

- Code: `qmem.coherence` (the fitted quantum model, physical phase echoes, randomised
  dephasings, arms as programmes of elements, statistics, Knee controls, benchmarks; numpy);
  `qmem.classical.ProgramModel` (one free stochastic map per element, EM with SQUAREM;
  backwards-compatible with #30's `InterventionModel`).
- Scripts: `experiments/phase8_coherence.py` (parts `review`, `design`, `power`, `--plan`),
  `experiments/phase8_coherence_design.py`, `experiments/phase8_coherence_hw.py`
  (`--export-qasm`, `hardware` group, `FakeKingston` only). Config:
  `experiments/configs/phase8_coherence.toml`.
- Outputs: `results/summaries/phase8_coherence/`: `review.json`, `review_classes.csv`,
  `review_quantum_fitted_statistics.csv`, `design.json`, `predictions.csv`, `phase_curve.csv`,
  `benchmarks.csv`, `alternatives.csv`, `budget_arms.csv`, `power.json`, `power_fits.csv`,
  `circuits.json`, `circuits.csv`, `device.json`, `qasm/` (hashes in `circuits.json`),
  `plan.json`.
- Tests: `tests/test_coherence.py` (listed in §6).
- Stage 2b (#34): `qmem.register` (the Z_8 × n_pop phase register); `qmem.classical`'s
  frozen-map EM; `qmem.coherence`'s §4 pipeline statistics (parity split, block bootstrap,
  Holm) and the Claim 3 contrast power. Scripts `experiments/phase8_coherence_null.py` (parts
  `null`, `register`), `experiments/phase8_coherence_contrast.py` (part `contrast`, with the
  separating options' prices), `experiments/phase8_coherence_hw.py --export-controls`. Outputs
  `null.json`, `null_replicates.csv`, `null_generator_b_match.csv`, `register.json`,
  `register_classes.csv`, `contrast.json`, `contrast_arms.csv`, `controls.json`,
  `controls.csv`, `qasm/controls/`, and the updated `plan.json`. Tests `tests/test_register.py`.
  The amendment is §4.1; the new results are §§3.1–3.4.
- Stage 2c (#35): `qmem.register` for any Z_n (nearest shifts, freed echoes, the Hungarian
  projection onto a free permutation); `qmem.coherence`'s 1 rad echoes `r<k>` / `rm<k>`.
  Script `experiments/phase8_coherence_incommensurate.py` (part `incommensurate`). Outputs
  `incommensurate.json`, `incommensurate_classes.csv`, `incommensurate_predictions.csv`,
  `incommensurate_null.csv`. Tests `tests/test_incommensurate.py`. The amendment is §4.2; the
  results are §3.5.

## Headline

**Stage 2c (#35; 2026-11-11, pre-data): drop the 1 rad tier. It does not make the register
exclusion robust.**
- **The tier has seven arms** (Rz(±1), INV(1, −1), Rz(1)² against Rz(2), Rz(1)³ against
  Rz(3)), not §3.2's six. Its composition contrasts are exactly zero under both quantum
  models. It would add 56,000 shots and 40 s of upper bound (§3.5).
- **Against the *other* quantum model, Z_16 × 2 and Z_24 × 2 registers (nearest shift) still
  win**, by 0.009–0.012 nats per shot (z −11 to −14). So does Z_8 × 3 with free 1 rad maps,
  by 0.016–0.019. That is the same gap #34 found. On the tier's own arms Z_16 and Z_24 also
  win (−0.001 to −0.007). Against the refit every class is rejected (z ≥ 8).
- **It excludes only what Z_8 cannot fit:** a Z_8 register with e(1) as the nearest shift or
  the best free permutation (#32's Lemma) loses by 0.065–0.106 even against the other model.
- **More shots do not help** (the gap is a per-shot bias; at 64,000 per arm, 320 s, every
  unexcluded class still wins). A second incommensurate angle is not expected to help either,
  because the misspecification, not the angle, sets the gap. **Size check:** SIZE_HEAD
- **The grid was cut** to Z_8 × 3 (three variants), Z_16 × 2 and Z_24 × 2. The run took about
  22 h on the shared machine, against the issue's ~12 h.
- §4.2 records the decision; `plan.json` is unchanged.

**Stage 2b (#34; 2026-11-10, pre-data): the Go stands at D = 6, but the exclusion is
narrower than the paragraphs below it suggest.**
- **The test holds its size.** On data from two classical d = 6 generators at the main tier's
  shots, §4's exact pipeline rejected "d ≤ 6" in **0 of 12 replicates (95% CI
  0–0.26)**. Its z never came near the threshold: z between −86 and −37 against 1.645. EM's search
  does not inflate the size (§3.3).
- **A classical phase register is not robustly excluded.** In this model a hidden phase index
  in Z_8 is shifted by each echo, alongside 2–4 free population states (d = 16–32). At 3 or 4
  population states it reproduces all 252 per-arm statistics within the plan's Bonferroni
  resolution. The held-out likelihood still separates it from the *refitted* quantum model at
  the plan's shots: 0.004–0.008 nats per shot, z = 8–17, 80% power at 1,600–7,000 shots in
  total. But that margin is smaller than the quantum model's own uncertainty. Scored against
  the *other* quantum model, the register wins by 0.014–0.018 nats per shot. Free d = 8 sits
  at the same edge (+0.0005 and −0.003 against the other model). **So Stage 2's classical
  exclusion holds robustly only for element-wise models with d ≤ 6. It does not hold for
  models with a phase-covariant hidden register, or for free d = 8, unless the 9-parameter
  quantum model is right to within about 0.005 nats per shot.** A 1 rad incommensurate-phase
  tier would separate them most cheaply (6 arms, upper bound +34 s, §3.2).
- **Claim 3's own contrast is now a leg (§4.1).** R2 and R4 vanish relative to RS (#31's
  rule, f = 0.25, power ≥ 0.999 at 8,000 shots per arm; f = 0.10 has power 0.75–1.00). RS is
  preserved relative to E00 (predicted ratio 0.935) at a non-inferiority margin of 0.20: power
  0.97–0.98 at the plan's shots. ±10% would need 50,000–54,000 shots per arm.
- **Controls tier.** It covers §2.5's population benchmark and §2.6's Knee circuits: 26
  circuits, 76,000 shots, one job. The upper bound is 38 s, with a proposed hard cap of 45 s.
  The plan's total upper bound is 166 s of the 600 s allowance.

**Stage 2 (#32):**

**Go, with D = 6, under the assumptions in §5.** The design adds 17 arms to the baseline:
- an 8-point physical phase scan on q75;
- inverse, composition and alternation arms;
- three equivalent complete dephasings (A2's measurement, and random Rz(0)/Rz(π) or
  Rz(kπ/2) drawn per shot and per step by two measured ancillas with feed-forward);
- a feed-forward sham.

In simulation, every classical model in `ProgramModel`'s class with d ≤ 6 states loses to the
quantum prediction on held-out shots. This holds when the data come from the fixed instrument
and when they come from the fitted one:
- 80% power needs at most 4,828 shots in total, even when the quantum prediction comes from the
  *other* quantum model;
- the plan has 8,000 shots per arm, 144,000 in all;
- the upper bound is 129 s of QPU time against the Open plan's 600 s.

The bar's three legs (§4 of the framework):
- (i) population invariance is built into the echoes;
- (ii) the phase response is large, periodic and non-monotone: P(1) runs from 0.05 to 0.62
  over φ;
- (iii) R2, R4 and A2 are the same channel, with TOST power at 8,000 shots per arm.

**What it would and would not show.** A pass supports Claim 3 for the class of classical
models in which each physical element (a pulse, a measurement, a feed-forward branch) acts as
one fixed stochastic map on at most 6 hidden states. It does not exclude a model with more
states. Nor does it exclude one in which a pulse's action depends on its context (§5).

**Review points first** (§1). The fitted quantum model (9 parameters) beats #30's
353-parameter d = 6 classical class on BIC by 2,325 (2,202 after #30's fit is run on to 4,800 steps). #30's A3 edge for the quantum
model shrinks with d but survives at d = 6 held out: 0.0037–0.0041 nats per symbol, against
0.0105 at d = 4. #30's fits, continued past their 800-step cap, gain
**347, 233 and 291 nats at d = 4, 5 and 6 (under 1e−4 nats per symbol). Only d = 5 converges; none of it changes an ordering or a conclusion**.

## 1. #30's review points

### 1.1 A fitted quantum model

`qmem.coherence.QuantumModel` takes #27's calibrated model of the joint (q75, q59) memory
(#20's pieces at the pilot's live calibration, and the pilot's q75 decay under measurement). It
adds nine physically motivated free parameters (`NoiseParams`), each 0 or `None` in the
calibrated model:

| parameter | meaning | fitted (all shots) | fold 0 | fold 1 |
| --- | --- | --- | --- | --- |
| `dephase` | extra q75 coherence loss per step (T2-type) | 0.0261 | 0.0378 | 0.0077 |
| `damp` | extra q75 amplitude damping per step (T1-type) | 0 | 0 | 7e−6 |
| `detune` | q75 Z rotation per step (rad; detuning-type) | 0.0433 | 0.0660 | 0.0067 |
| `flip` | symmetric flip of q59's recorded symbol | 1e−9 | 0 | 2e−6 |
| `reset_e0`, `reset_e1` | q59's reset leaves \|1⟩ from \|0⟩, \|1⟩ (#21) | 0.0020, 0.0186 | 0.0037, 0.0194 | 0, 0.0135 |
| `meas_decay` | A2's q75 \|1⟩→\|0⟩ per measurement (#26) | 3e−9 | 0 | 3e−9 |
| `spec_dephase`, `spec_phase` | the q74 sham's dephasing and ZZ/Stark phase on q75 | 0.0172, −0.052 | 0, −0.030 | 0.0459, −0.088 |

Every default reproduces #27's analysis model exactly. `test_default_model_reproduces_the_calibrated_step`
checks this against #30's likelihoods, and `test_fitted_quantum_likelihood_against_brute_force`
checks the likelihood on a small case. The fit is Nelder–Mead on the total NLL of the five q75
arms, restarted twice from its own optimum. D1 is a different circuit and stays at #23's fixed
model, as in #30.

| model | parameters | total NLL | ΔNLL vs fitted quantum | ΔBIC vs fitted quantum |
| --- | --- | --- | --- | --- |
| quantum, fitted | 9 | 2,839,420.9 | 0 | 0 |
| quantum, calibrated (#27) | 0 | 2,841,844.5 | +2,424 | +4,710 |
| classical d = 4 (#30, 800 steps) | 147 | 2,840,152.0 | +731 | +3,572 |
| classical d = 5 (#30, 800 steps) | 239 | 2,838,744.1 | −677 | +2,163 |
| classical d = 6 (#30, 800 steps) | 353 | 2,837,953.9 | −1,467 | +2,325 |
| classical d = 4 (continued, 4,002 more steps) | 159 | 2,839,804.7 | +384 | +3,061 |
| classical d = 5 (continued, converged after 3,063) | 259 | 2,838,510.8 | −910 | +2,002 |
| classical d = 6 (continued, 4,002 more steps) | 383 | 2,837,663.2 | −1,758 | +2,202 |

(Full table: `review_classes.csv`. BIC uses the total symbol count, 4,358,144; the parameter counts are those `review_classes.csv` records for each run.)

The fitted model gains 2,424 nats on the calibrated one. Most of the gain is on A2s_spec and
the baseline (the q74 sham's dephasing and q75's detuning). **The free d = 6 classical class is
still 1,467 nats better in likelihood (1,758 once continued). With 344 more parameters (374 continued), it loses on BIC by 2,325 (2,202).** On A3 the
fitted quantum model beats d = 6 by 1,035 nats (901 continued).

**The fold-wise parameters are not stable.** Fit on one pub-parity fold, `detune` is 0.066
against 0.007, and `dephase` is 0.038 against 0.008. The two folds are uneven per arm (#29's
parity split is not stratified by arm). The parameters trade against each other: dephasing,
detuning and the sham's phase all reduce the witness. So the fitted values are a description,
not a calibration. Stage 2's design therefore keeps both quantum models: the **fixed**
(calibrated) prediction is the pre-registered one, and the **fitted** one is the check that the
design's power does not rest on a particular noise model.

### 1.2 Cross-validation at d = 4, 5, 6

This repeats #30's pub-parity CV. A chain is trained on one fold: d = 4 from the quantum start
(the Klein-tetrahedron embedding), then d = 5 and d = 6 by splitting states. Each step runs
3,000 EM iterations; the last gain is 5e−10 to 3e−9 nats per symbol (below `tol` = 1e−10 per
iteration in total, but the cap was hit). The fitted quantum model is refitted on the same fold.
Held-out NLL per symbol, classical minus quantum:

| train fold | test symbols | d | − calibrated (all) | − fitted (all) | − calibrated (A3) | − fitted (A3) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2,028,288 | 4 | +0.00020 | +0.00038 | +0.0103 | +0.0105 |
| 0 | | 5 | −0.00009 | +0.00009 | +0.0062 | +0.0064 |
| 0 | | 6 | −0.00022 | −0.00004 | +0.0039 | +0.0041 |
| 1 | 2,329,856 | 4 | −0.00071 | −0.00020 | +0.0106 | +0.0107 |
| 1 | | 5 | −0.00105 | −0.00053 | +0.0039 | +0.0040 |
| 1 | | 6 | −0.00110 | −0.00059 | +0.0036 | +0.0037 |

(Positive: the quantum model predicts the held-out shots better.)

- **The A3 edge does not disappear at d = 6.** It falls from 0.0105 to about 0.004 nats per
  symbol, and stays positive in both folds. A3's per-shot Pauli sequences are where the quantum
  description still carries information that six classical states plus one learnt map per gate
  do not. That is Stage 2's rationale (reversible, composable insertions), so the rationale
  stands. But the edge shrinks with d, which is why D has to be stated.
- Over all arms, d ≥ 5 predicts held-out shots slightly better than either quantum model (by
  ≤ 0.0006 nats per symbol). The gain is mostly on the baseline in fold 1. Stage 4c's arms
  alone do not separate the two descriptions; they need new arms.
- The fitted quantum model generalises: held out, it beats the calibrated one in both folds
  (−0.00018 and −0.00052 nats per symbol), on every arm except A2 in fold 0 (+0.0006).

### 1.3 Convergence

Every #30 fit stopped at the 800-step cap. The fits continued here are #30's best at d = 4, 5
and 6, with up to 4,000 more SQUAREM steps:

| d | steps after 800 | converged | NLL gain | per symbol | last step, per symbol | ΔNLL vs fitted quantum | ΔBIC | A3: classical − fitted quantum | primary stats \|z\| > 1.96 (of 16; 0.8 expected) | all stats over Bonferroni (of 94) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 4,002 (cap) | no | 347 | 8.0e−5 | 3e−9 | +384 | +3,061 | +2,736 | 7 | 15 |
| 5 | 3,063 | yes | 233 | 5.4e−5 | 0 | −910 | +2,002 | +1,657 | 4 | 9 |
| 6 | 4,002 (cap) | no | 291 | 6.7e−5 | 1e−9 | −1,758 | +2,202 | +901 | 2 | 1 |

(`review.json` → `convergence`; the statistics are #30's 94 per-arm w, g, P(1) and lag 1 checks.)

- **The gap left by the 800-step cap is small:** at most 347 nats, under 1e−4 nats per symbol.
  At d = 4 and 6 the last steps gain 1e−9 to 3e−9 nats per symbol, so what remains is far below
  the BIC margins.
- **Nothing changes order.** The fitted quantum model still wins on BIC against every d (by at
  least 2,002) and on A3 against every d (by at least 901 nats). d = 6 still wins in raw
  likelihood.
- Continued to convergence, d = 6 now fits all 16 primary statistics within Bonferroni (two
  beyond 1.96, against 0.8 expected) and all but one of the 94 (A2s_spec g(4), z = +4.1). On
  Stage 4c's arms a 6-state classical model is an adequate description; the new arms are what
  separate it from the quantum one (§3).

## 2. The design

### 2.1 Physical phase insertions

IBM's `rz` is a frame change. The transpiler can cancel or commute Rz(φ)Rz(−φ), so a phase scan
built from `rz` alone could be pure software. A **physical** Z rotation is a frame-neutral echo
of two `x` pulses in time order `x, rz(a), x, rz(−a)`. Its unitary is Rz(−a) X Rz(a) X =
Rz(−2a) (up to a global phase), so a = −φ/2 gives Rz(φ), and the frame after it is unchanged
(`echo_kraus`, `test_echo_is_rz_up_to_phase_and_frame_neutral`). `e<k>` is the echo with
φ = kπ/4; `e0` (two `x` pulses) is the matched-timing sham.

On `FakeKingston` (`--export-qasm`, `circuits.csv`):
- **Fenced echoes keep every pulse at optimisation levels 0–3** (each echo between barriers;
  `all_physical_insertions_survive_levels_0_3 = true`). They add 64 ns per echo, inside q59's
  2.2 µs reset window, so the circuit duration is unchanged (321.0 µs).
- **Controls:** the same arms unfenced, and with a virtual `rz` scan, are merged or cancelled
  at level ≥ 1. This is the failure the issue warned of, and the plan transpiles at level 0
  with every echo fenced.

### 2.2 Arms

| arm | programme per step | quantum meaning |
| --- | --- | --- |
| baseline | — | Phase 6/7's circuit |
| E0 | e0 | the sham: two x pulses (identity) |
| P1–P7 | e1 … e7 | Rz(kπ/4) each step: the φ scan (E0 is φ = 0 = 2π) |
| E00 | e0 e0 | the 4-pulse sham |
| INV2 | e2 e6 | Rz(π/2) then Rz(−π/2): = E00 |
| COMP22, COMP40 | e2 e2; e4 e0 | Rz(π/2)² against Rz(π): equal |
| ALT2 | e2 on even steps, e6 on odd | alternating ±π/2 |
| RS | ff, then e0 e0 in every branch | the feed-forward sham |
| R2 | ff, then e4 if b₁ = 1 | random Rz(0)/Rz(π) per shot and step |
| R4 | ff, then e4 if b₁, e2 if b₂ | random Rz(kπ/2) per shot and step |
| A2 | q75 Z measurement | Phase 7's A2 |

`ff` is two fresh ancillas (q50 and q21) prepared in \|+⟩ and measured together. Their bits
are recorded and drive the echoes by feed-forward. The ancillas were chosen by a predeclared,
calibration-only rule (`circuits.json`): distance ≥ 3 from q59 and q75, not adjacent to each
other, lowest readout error, then longest T1. The rule is re-run on the live calibration at
Stage 4. The feed-forward arms run 393 µs instead of 321 µs: one ancilla measurement (2.2 µs)
plus 1.0 µs of conditional latency per step.

### 2.3 The randomised dephasing is a channel

Averaged over the bits b, which are independent of everything else, each step applies
Σ_b p(b) U_b ρ U_b†. This is a Z dephasing with coherence factor |Σ_b p(b) e^{iφ_b}|:
- 0 for R2 and R4 with fair bits;
- 2·|p − ½| for R2 with bias p.

`test_per_shot_per_step_random_phase_is_the_averaged_channel_by_enumeration` checks it against
every bit sequence. Pooled fixed patterns are not this channel
(`test_pooled_fixed_patterns_are_not_the_channel`). The bits are recorded, so the analysis can
also condition on them.

**Cost and crosstalk of the ancillas.** They cost 72 µs per shot of q75 idling, which the
fitted model's `dephase` term prices in. The RS sham carries the same idling and feed-forward,
so the RS–E00 contrast isolates their effect. `alternatives.csv` adds a 0.1 rad ZZ phase on
q75 per set bit and a 2% bias. Neither changes R2 against A2's w(·) (both 0), and the largest
P(1) shift is 0.015 (RS).

### 2.4 The instrument's fixed predictions

`predictions.csv` and `phase_curve.csv` (64 points over [0, 2π), both quantum models).

| | baseline / E0 | P2 (π/2) | P4 (π) | P5 | P7 | INV2 / E00 | COMP22 = COMP40 | R2 = R4 | A2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P(1) | 0.581 | 0.151 | 0.182 | 0.384 | 0.621 | 0.581 | 0.182 | 0.386 | 0.386 |
| lag 1 | 0.155 | 0.023 | −0.014 | 0.025 | 0.317 | 0.154 | −0.014 | 0.023 | 0.023 |
| w(0) | −0.0067 | −0.0107 | −0.084 | −0.091 | −0.0015 | −0.0066 | −0.084 | 0 | 0 |
| w(2) | −0.055 | −0.0016 | −0.020 | −0.338 | −0.0019 | −0.055 | −0.020 | 0 | 0 |

The response is periodic and strongly non-monotone (the extrema are in `design.json`):
- P(1) has a minimum of 0.05 near φ = 2.26 and a maximum of 0.62 near φ = 5.69;
- lag 1 changes sign twice;
- w(0) dips to −0.118 near φ = 3.73.

Under the quantum model:
- inverse and composition arms equal their partners exactly;
- R2, R4 and A2 agree to within 4e−5 in lag 1 and 2e−4 in P(1). A2 differs only by its 0.2%
  measurement-induced decay.

### 2.5 Population invariance and benchmarks

`benchmarks.csv`: q75 alone (FakeKingston's T1 = 321 µs, T2 = 139 µs), prepared in \|0⟩,
\|1⟩, \|+⟩ or \|+i⟩, then 0–8 repetitions of each intervention at the PMD step spacing.
- **The echoes (e0, e2, e4), RS, R2 and R4 leave the Z populations unchanged** relative to the
  matched sham (ΔP(1) = 0 exactly in the model).
- R2, R4 and A2 remove the coherence completely (Ramsey contrast −0.94 after one step).
- A2 adds 0.0013 to P(1) from \|1⟩ after 8 steps (its measurement-induced decay).
- A1 resets the populations.

A direct population benchmark to ±0.01 at 5σ needs about 125,000 shots per point. To ±0.02
with a TOST at 80% power it needs about 8,000. These short q75-only circuits are not in
`plan.json`. In the plan, (i) is tested on the arms themselves: E0, E00 and RS against the
baseline in P(1) and lag 1 (§4). **A Stage 4 proposal should add the direct benchmark.** About
20 s covers e0, e2, e4, R2 and A2 from \|0⟩ and \|1⟩ at 8 repetitions, to ±0.02.

*Stage 2b (#34):* it is now in `plan.json` as the controls tier (§3.4). Each circuit is tested
against the e0 sham at the same preparation (TOST, ±0.02, 80% power). That needs at most
5,315 shots (R2 from \|1⟩), so the tier runs 6,000 shots per circuit.

### 2.6 Knee-style classical-disturbance controls

q75 is prepared in step 0's slot in \|0⟩, \|1⟩, \|+⟩ or \|+i⟩, followed by one intervention
or the sham. The disturbance D_s is the total-variation distance between the first four
outputs' distributions with and without the intervention. Suppose a classical model's hidden
state after preparing a superposition is a mixture of its states after \|0⟩ and \|1⟩. Then
convexity gives **D₊ ≤ max(D₀, D₁)**, the predeclared inequality
(`test_knee_inequality_holds_for_mixture_preparations`). The quantum instrument violates it
(`test_quantum_instrument_violates_the_knee_inequality`):

| intervention | D₀ | D₁ | D₊ | D₊ᵢ | excess D₊ − max(D₀, D₁) | power (5σ) at 1,000 shots |
| --- | --- | --- | --- | --- | --- | --- |
| e4 (Rz(π)) | 0 | 0 | 0.79 | 0.26 | 0.79 | 1.00 |
| e2 (Rz(π/2)) | 0 | 0 | 0.38 | 0.44 | 0.38 | 1.00 |
| A2 | 7e−5 | 5e−5 | 0.40 | 0.13 | 0.40 | 1.00 |

(fixed model; the fitted model is within 0.01.)

The inequality's weak point is its premise. A classical model is free to give the \|+⟩
preparation its own hidden state rather than a mixture. The sham's own outputs test this: the
best mixture of the \|0⟩ and \|1⟩ sham distributions misses the \|+⟩ one by a TVD of 0.35. A
preparation-noncontextual classical model is therefore excluded outright. A contextual one is
not, which is why §5 lists noncontextuality as an assumption. *Stage 2b (#34):* the 16
circuits are now in `plan.json`'s controls tier at 1,000 shots each (§3.4).

### 2.7 Noise and leakage alternatives

`alternatives.csv` gives every arm's predictions under these alternatives, within
`qmem.coherence`'s 16-dimensional model:
- #21's fitted q59 reset asymmetry (0.0089, 0.0209) and q75 detuning (0.042 rad per step);
- #22's 16 kHz readout Stark shift (0.2 rad, on y = 1 only);
- a 1% x-pulse amplitude error;
- the ancillas' bias and ZZ phase.

The largest shift from the fixed prediction under each:

| alternative | w(0) | w(1) | P(1) | lag 1 |
| --- | --- | --- | --- | --- |
| #21 q59 reset asymmetry | +0.012 (P3) | +0.008 (P3) | −0.004 (P7) | −0.014 (P7) |
| #21 q75 detuning | +0.002 (P3) | −0.003 (baseline) | −0.012 (P4) | +0.014 (ALT2) |
| #22 Stark shift | −0.024 (P5) | −0.016 (baseline) | −0.010 (P6) | +0.034 (P1) |
| x amplitude error 1% | −0.043 (P3) | −0.007 (P3) | +0.008 (E00) | −0.025 (COMP22) |
| ancilla bias 2% | 0 | 0 | −0.008 (R2) | −0.003 (R2) |
| ancilla ZZ 0.1 rad | 0 | +0.009 (RS) | −0.015 (RS) | −0.027 (RS) |

Each shift is small next to the phase response (P(1) swings by 0.57). The power study's
"fitted" generator carries #21-like noise. The amplitude error breaks the phase group
(`test_over_rotation_breaks_the_group`): it is the one alternative that touches the inverse and
composition arms (COMP22's lag 1, −0.025). It is a pulse calibration, so the pilot checks INV2
against E00 before the main jobs.

### 2.8 Budget

`qmem.followup`'s cost model (Phase 6/7): 250 µs repetition delay, 2 s overhead per sub-job, a
25% margin and 1 s billing rounding.

- The echo arms cost 0.571 s per 1,000 shots (no added time).
- The feed-forward arms cost 0.643 s per 1,000 shots.
- A2 costs 0.573 s per 1,000 shots.

| option | shots per arm | estimate | upper bound | what it buys |
| --- | --- | --- | --- | --- |
| minimal | 2,000 | 23 s | 30 s | the classical exclusion only (§3); TOSTs underpowered |
| **recommended** | 8,000 (main) + a 2,000-shot pilot | 101 s | 129 s | the exclusion with a 4,000-shot held-out half; TOSTs for (iii) at 80% |
| generous | 20,000 | about 215 s | about 280 s | finer w(φ) for the periodicity statistic; half the 28-day allowance |

The illustrative 60–100 s in #28 is not an authorisation. Any Stage 2 job needs the owner's
explicit approval on #28.

## 3. The power study against the classical alternative

**Method** (`--part power`):
- Data are simulated from both quantum models (fixed and fitted) at 2,000 training and 2,000
  held-out shots per arm, for all 18 arms, with 64 steps each.
- `qmem.classical.ProgramModel` is fitted to the training shots at d = 4 and d = 6. Its
  hidden-state space and its pre, post, reset and readout maps are shared by every arm. Each
  element (`e0`…`e7`, `meas`, each `ff` branch) is one free stochastic map. The inverse,
  composition and alternation arms are data, not constraints.
- Starts: the Klein-tetrahedron embedding, a random start, an octahedron embedding at d = 6,
  and a d = 6 split of the best d = 4 fit. Each runs 1,500 SQUAREM steps.
- A d = 6 fit is never credited with a worse training NLL than d = 4's best.
- The quantum model is refitted to the same training shots (`refit`, its 9 noise parameters).
- *Held-out arms:* the fit also runs with INV2, COMP22, COMP40 and ALT2 removed from training,
  and those arms are predicted.
- **The statistic** is the held-out NLL deficit per shot, classical minus quantum, with its
  per-shot SD. Power is one-sided at α = 0.05 per d, Holm over d = 4 and 6.

The deficit is reported against three quantum predictions:
- *truth*, the generating model;
- *refit*, the quantum model refitted on the training half: the contrast the analysis would
  actually run;
- *other*, the other quantum model: the prediction is misspecified by the whole fixed/fitted
  difference.

Results (`power.json`, `power_fits.csv`):

Best start per cell (d = 6: the octahedron start; with arms held out, the split start). Deficit
in nats per shot, classical minus quantum, on all 18 arms:

| generator | d | in-sample (truth) | held out: truth | refit | other | SD per shot | held-out z (refit) | shots for 80% (total; worst evaluator) | stats over Bonferroni (of 252) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 4 | 0.227 | 0.235 | 0.235 | 0.214 | 0.66 | 67.7 | 53 | 86 |
| fixed | 6 | 0.045 | 0.050 | 0.050 | 0.030 | 0.30 | 31.5 | 710 | 25 |
| fitted | 4 | 0.198 | 0.198 | 0.197 | 0.176 | 0.63 | 59.2 | 93 | 77 |
| fitted | 6 | 0.032 | 0.035 | 0.033 | 0.013 | 0.29 | 22.0 | 4,828 | 15 |

With INV2, COMP22, COMP40 and ALT2 held out of training (deficit on all arms' test shots):

| generator | d | in-sample (truth) | held out: truth | refit | other | shots for 80% |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | 4 | 0.158 | 4.97 | 4.97 | 4.95 | 2 |
| fixed | 6 | 0.103 | 2.96 | 2.96 | 2.95 | 4 |
| fitted | 4 | 0.142 | 5.08 | 5.07 | 5.06 | 2 |
| fitted | 6 | 0.103 | 3.59 | 3.59 | 3.57 | 3 |

Power at every planned shot count (1,000 to 20,000 per arm) is ≥ 0.999 in every cell. No fit
converged at 1,500 steps; the last gains are ≤ 2e−6 nats per symbol.

- **(a) The minimal classical d that fits within CIs is above 6** at these shot counts. At
  d = 6, 25 (fixed generator) and 15 (fitted) of the 252 statistics (w, g, P(1) and lag 1 per arm) lie outside the
  Bonferroni level. The worst is COMP22's g(4): the composition arm is where the classical
  maps fail. This design does not locate the minimal d above 6.
- **(b) d ≤ 4 and d ≤ 6 are excluded** at 80% power with at most 4,828 shots in total
  (all arms together), against 144,000 planned. The margin is large enough that the exclusion
  survives a misspecified quantum prediction (*other*).
- **With the four composite arms held out,** d = 4 misses them by about 5 nats per shot. This
  is the Lemma at work: a finite classical model cannot make Rz(π/2)Rz(−π/2) the identity with
  non-permutation maps (`test_divisible_phase_maps_in_a_finite_model_must_be_permutations_to_compose_exactly`).
  At d = 6 the split fits miss them by 3.0 (fixed) and 3.6 (fitted) nats per shot: two more states do not let the maps compose.
- **The D the claim inherits is 6.** The deficit falls steeply from d = 4 to 6 (0.23 to 0.05
  nats per shot in-sample). An exclusion at d = 6 says nothing about d = 8 or more. Excluding
  larger d would need more φ settings or more composition arms, not more shots.

### 3.1 The phase register beyond d = 6 (#34, `--part register`)

**The model** (`qmem.register`). A hidden phase index θ ∈ Z_8 sits alongside a population
register of n_pop states, so d = 8 n_pop.
- Each echo e_k is the fixed permutation θ → θ + k, so INV2, COMP22 = COMP40 and ALT2 compose
  exactly.
- R2 and R4 are the bit-averaged mixtures of the same permutations, so they have no free
  parameters either.
- The free maps are the step's pre, post and reset maps, `meas`, the `ff` branches and the
  readout. That gives 2,207, 5,039 and 9,023 parameters at n_pop = 2, 3 and 4, against 527 for
  free d = 6, 975 for free d = 8, and about 4,100 for free d = 16.
- EM keeps the permutations frozen (`ProgramModel`'s frozen-map EM).
- Starts:
  - the *prism*, which embeds the ideal rotation as the shift (`test_prism_embeds_the_ideal_rotation_as_the_shift`);
  - a random start.
- Its likelihood matches path enumeration (`test_register_likelihood_against_path_enumeration`).

**Method.** The data come from #32's two quantum generators (rep 0) at the main tier's shots:
8,000 per arm in two jobs, with a parity split. Every class is run through §4's pipeline at
1,500 steps per start. Each class is credited with its best start, monotone in d. No fit
converged at 1,500 steps; the last gains were ≤ 6e−6 nats per symbol.

Results (`register_classes.csv`). Held-out deficit is in nats per shot, classical minus
quantum. z uses the block SE. "Stats out" counts the 252 per-arm statistics outside
Bonferroni:

| generator | class | params | held out: truth | refit | other | z (refit) | shots for 80% (refit, total) | in-sample (refit) | stats out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | free d ≤ 4 | 215 | 0.238 | 0.238 | 0.216 | 97.2 | 57 | 0.233 | 117 |
| fixed | free d ≤ 6 | 527 | 0.047 | 0.047 | 0.025 | 41.1 | 259 | 0.043 | 35 |
| fixed | free d ≤ 8 | 975 | 0.023 | 0.023 | 0.0005 | 29.8 | 540 | 0.020 | 16 |
| fixed | Z_8 × 2 (d = 16) | 2,207 | 0.0079 | 0.0079 | −0.014 | 16.7 | 1,665 | 0.0038 | 5 |
| fixed | Z_8 × 3 (d = 24) | 5,039 | 0.0051 | 0.0050 | −0.017 | 13.2 | 2,653 | −0.0024 | 0 |
| fixed | Z_8 × 4 (d = 32) | 9,023 | 0.0055 | 0.0055 | −0.016 | 13.3 | 2,425 | −0.0044 | 0 |
| fitted | free d ≤ 4 | 215 | 0.198 | 0.196 | 0.175 | 81.6 | 74 | 0.194 | 110 |
| fitted | free d ≤ 6 | 527 | 0.037 | 0.035 | 0.014 | 32.4 | 423 | 0.034 | 42 |
| fitted | free d ≤ 8 | 975 | 0.020 | 0.018 | −0.003 | 23.6 | 844 | 0.016 | 8 |
| fitted | Z_8 × 2 (d = 16) | 2,207 | 0.0075 | 0.0056 | −0.016 | 11.3 | 3,692 | 0.0010 | 0 |
| fitted | Z_8 × 3 (d = 24) | 5,039 | 0.0056 | 0.0037 | −0.018 | 8.2 | 6,970 | −0.0047 | 0 |
| fitted | Z_8 × 4 (d = 32) | 9,023 | 0.0070 | 0.0051 | −0.016 | 10.6 | 4,192 | −0.0075 | 0 |

**The outcome, plainly.**
- **The register matches the quantum prediction within the plan's resolution on every
  summary statistic.** At n_pop ≥ 3 none of the 252 statistics (w, g, P(1) and lag 1 per
  arm) is outside Bonferroni. In sample it beats even the generating model, which is
  overfitting with 5,000–9,000 parameters.
- **On held-out shots the refitted quantum model still wins** by 0.004–0.008 nats per shot
  (z = 8–17 at the plan's shots). §4's primary contrast would therefore reject the register
  too, *if the quantum model is correctly specified*.
- **That condition carries the result.** The fixed and fitted quantum models differ by about
  0.02 nats per shot. Scored against the *other* one, every register class beats the quantum
  prediction (z from −17 to −21). Free d = 8 is not rejected (z = 0.6, fixed generator) or
  wins (z = −2.9, fitted). In §3 the same
  check left d ≤ 6 excluded with a wide margin; here it does not. A device whose truth departs
  from the 9-parameter model by a fixed-versus-fitted amount would let a phase register (or
  free d = 8) win.
- **So the Stage 2 classical exclusion is restricted to models without a phase-covariant
  hidden register.** Robustly, it excludes element-wise models with d ≤ 6 (D = 6, §5).
  Excluding a register, or free d = 8, holds only conditional on the quantum model's
  specification.
- d = 32 does no better than d = 24 held out. More population states do not close the gap;
  the residual is the register's fixed Z_8 geometry, not its size.

### 3.2 What would separate the instrument from a phase register (not built)

Priced with `qmem.followup`'s cost model at the plan's 8,000 shots per arm (`contrast.json`
`separation`). The echoes add no circuit time.

| option | arms | shots | added upper bound (main tier) | own tier upper bound | what it tests |
| --- | --- | --- | --- | --- | --- |
| **incommensurate phase, 1 rad** | 6 | 48,000 | 34 s | 38 s | e(1), e(−1), INV(1, −1), COMP(1, 1) against e(2), e(1)³ against e(3). No finite permutation group contains Rz(1 rad). INV(1, −1) = identity forces e(1) to be a permutation (#32's Lemma), and its powers repeat with a period the quantum response does not have. |
| finer scan, 16 points | 10 | 80,000 | 57 s | 61 s | Z_8 cannot represent π/8, but Z_16 can: this raises D to 16 n_pop without an absolute exclusion |
| irrational scan, golden angle | 10 | 80,000 | 57 s | 61 s | phases outside any small cyclic group; a register needs n large enough to approximate all of them at the plan's resolution |
| dimension witness on the existing scan | 0 | 0 | 0 | 0 | the rank of the cross-arm Hankel matrix of the 8 scan points; a lower bound on d, weak (d = 16 has room) |

**Recommendation:** the 1 rad option. It is the cheapest, and it attacks the register's
structure rather than its size. With it the plan's total upper bound would be about 200 s of
600 s. It needs its own power study against a Z_n register (n ≤ 16) before it goes into a
pre-registration. That study is not run here.

### 3.3 Null calibration (#34, `--part null`)

**Generators.** Both are classical with d = 6, for all 18 arms (`null.json` `generators`):
- (a) #32's best d = 6 `ProgramModel` fit to the fixed quantum generator's simulated shots
  (`power_fixed_d6_octahedron`). This is the adversary closest to the quantum prediction.
- (b) #30's Stage 4c d = 6 fit (`general_d6`), with its maps frozen. The new elements' maps
  (e0…e7, `ff` branches) were fitted by EM to the same shots, starting at (1 − 0.02) I + 0.02/d.
  Its per-arm P(1) is within 0.02 of the quantum model's on every arm. Lag 1 is within
  0.034; w(0) differs at P3–P5 and in the composition arms (`null_generator_b_match.csv`).

**Pipeline.** §4's exactly, at the main tier's shots:
- 8,000 per arm in two jobs of 4,000, split by parity;
- the four starts at 1,500 steps, and the quantum refit;
- block bootstrap by job and 8-shot segment (1,000 resamples);
- Holm over D = 4, 6 at α = 0.05.

Each replicate draws new data from the generator.

**Replicates.** There are 6 per generator, 12 in all, fewer than the 20 the issue
asks for. Each replicate is five fits at 15–30 CPU-min each, about 86 CPU-min. At the
shared machine's 2 workers, 12 replicates took about 10 h. The z distribution below
makes the binomial bound the weaker of the two statements.

Results (`null.json`, `null_replicates.csv`). Deficit in nats per shot, classical minus
quantum, against the refit:

| generator | replicates | false rejections, d ≤ 6 (Holm) | 95% CI | z (d = 6): mean | SD | max | deficit (d = 6): mean | max | rejections, d ≤ 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) #32's d = 6 fit | 6 | 0 | 0–0.46 | −37.5 | 0.70 | −36.7 | −0.046 | −0.045 | 6 |
| (b) #30's d = 6, extended | 6 | 0 | 0–0.46 | −76.5 | 3.07 | −70.6 | −0.184 | −0.181 | 0 |
| both | 12 | **0** | **0–0.26** | | | −36.7 | | −0.045 | 6 |

Against the fixed quantum model instead of the refit, z is −38.8 (a) and −86.5 (b), with 0
rejections.

- **False rejections of "d ≤ 6": 0 of 12 (95% Clopper–Pearson CI 0–0.26).**
- The binomial interval is wide only because of the replicate count. The statistic itself is
  far below the threshold: 56 between-replicate SDs below it for (a), 25 for (b). Held out,
  the classical fit beats the quantum refit by 0.046 (a) and 0.18 (b) nats per shot.
- **Search failure would have to be very large to matter.** The best d = 6 start on (a)'s
  data beats the quantum refit by about 0.046 nats per shot. For a false rejection, EM would
  have to miss the optimum by more than that: about 3,300 nats on each replicate's 72,000
  training shots. The fits' last gains were ≤ 4e−6 nats per symbol.
- "d ≤ 4" was rejected in 6 of 6 replicates from (a). That is a correct rejection:
  (a) is genuinely d = 6. It was rejected in 0 of 6 from (b), whose d = 4 fit also
  beats the quantum refit.
- No pipeline fix is needed, so §3's power study stands at the pipeline as registered.

### 3.4 The controls tier in the plan (#34)

`--export-controls` (`hardware` group, `FakeKingston` only) writes 26 QASM files to
`qasm/controls/`, with hashes in `controls.json` and `plan.json`:
- §2.5's benchmark: 10 circuits (e0, e2, e4, R2 and A2 from \|0⟩ and \|1⟩, 8 repetitions)
  × 6,000 shots;
- §2.6's Knee circuits: 16 (\|0⟩, \|1⟩, \|+⟩, \|+i⟩ × e4, e2, A2, sham) × 1,000 shots.

That makes 76,000 shots in one job: estimate 29 s, upper bound 38 s, **proposed hard cap 45 s**.
The plan's tiers are now:

| tier | shots | jobs | estimate | upper bound | hard cap |
| --- | --- | --- | --- | --- | --- |
| pilot | 18,000 | 1 | 13 s | 17 s | 30 s |
| main | 144,000 | 2 | 88 s | 112 s | 150 s |
| controls | 76,000 | 1 | 29 s | 38 s | 45 s |
| **total** | 238,000 | 4 | 130 s | **166 s** | 225 s |

against the Open plan's 600 s. `plan.json` is UNSUBMITTED and needs the owner's explicit
approval on #28.

### 3.5 The 1 rad tier against Z_n phase registers (#35, `--part incommensurate`)

**The tier** (`[incommensurate.arms]`) has seven arms. The issue said six, but each of its two
composition tests needs both sides, so seven arms are needed to test the four relations of
§3.2: RAD1 = Rz(1), RADM1 = Rz(−1), INV1 = Rz(1)·Rz(−1), COMP11 = Rz(1)² against RAD2 = Rz(2),
and CUBE1 = Rz(1)³ against RAD3 = Rz(3). The elements `r<k>` and `rm<k>` are the physical echo
with Rz(±k rad), fenced and timed as `e<k>` (`qmem.coherence.element_phase`), so the tier adds
no circuit time. RAD2 and RAD3 pad with `e0` to the same pulse count as COMP11 and CUBE1.
Under both quantum models the three composition contrasts are exactly zero in P(1), lag 1 and
w(0): INV1 = E00, COMP11 = RAD2 and CUBE1 = RAD3 (`incommensurate.json` `predictions`;
`tests/test_incommensurate.py` checks them against brute-force density matrices). RAD1's P(1)
differs from the nearest Z_n shift angle's by 0.069 (Z_8, error 0.21 rad), 0.059 (Z_16,
0.18 rad), 0.016 (Z_12 and Z_24, 0.047 rad) and 0.006 (Z_32, 0.018 rad).

**The adversary** (`qmem.register` extended to any Z_n). Three variants:
- *nearest shift:* every echo, the 1 rad ones included, is the nearest Z_n shift;
- *1 rad maps free:* r1, rm1, r2 and r3 are free stochastic maps, warm-started from the
  nearest fit (an element-wise classical model, not a register in the strict sense);
- *e(1) a free permutation:* e(1) is the permutation nearest the relaxed fit (Hungarian),
  e(−1) is its inverse, so INV(1, −1) = identity exactly, and r2 and r3 are free. This is the
  strongest adversary #32's Lemma allows.

**The grid is cut** to fit the issue's budget (one EM step on the 25 arms costs about
0.25 + 0.045 d s). It covers Z_8 × 3 (all three variants), Z_16 × 2 and Z_24 × 2 (nearest
shift). Dropped: n = 12, because Z_24 contains Z_12's shifts and the same 0.047 rad
approximation of 1 rad; n = 32 (d ≥ 64 at n_pop = 2); n_pop = 1; and the free-permutation
variant above n = 8. A Z_24 nearest-shift register is itself an e(1) permutation with
INV(1, −1) = identity, so the free-permutation class at n = 24 does at least as well in
training as the row below. Even so, the full run took about 22 h on the shared machine:
13.4 h for power, then about 8.5 h for the size check. The issue's budget was about 12 h.

**Method.** #34's §4 pipeline unchanged: parity split, the four starts at D = 4 and 6, and
free d = 8 from the cube start. Registers start from the prism. Each fit runs 1,500 steps,
followed by the quantum refit, the block bootstrap by job and 8-shot segment, and Holm. The
data come from both quantum generators (rep 0) on the 18 + 7 arms at 8,000 shots per arm in
two jobs. The 18 arms' shots are #34's register run's exactly; the new arms are drawn after
them. No fit converged at 1,500 steps; the last gains were ≤ 3e−6 nats per symbol.

Results (`incommensurate_classes.csv`). Held-out deficit in nats per shot, classical minus
quantum, over all 25 arms. "Tier (other)" is the deficit against the other model on the seven
new arms alone. "Stats out" counts the 350 per-arm statistics outside Bonferroni:

| generator | class | params | truth | refit | other | z (refit) | z (other) | tier (other) | shots for 80% (refit, total) | stats out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | free d ≤ 4 | 263 | 0.230 | 0.230 | 0.207 | 109.7 | 102.7 | 0.160 | 59 | 160 |
| fixed | free d ≤ 6 | 647 | 0.076 | 0.076 | 0.053 | 60.7 | 41.5 | 0.055 | 172 | 71 |
| fixed | free d ≤ 8 | 1,199 | 0.026 | 0.026 | 0.0030 | 37.1 | 3.5 | −0.0021 | 458 | 25 |
| fixed | Z_8 × 3, nearest | 5,039 | 0.121 | 0.121 | 0.098 | 101.8 | 76.1 | 0.243 | 72 | 93 |
| fixed | Z_8 × 3, 1 rad maps free | 7,247 | 0.0071 | 0.0071 | **−0.0161** | 18.7 | −21.9 | −0.0227 | 1,773 | 2 |
| fixed | Z_8 × 3, e(1) free perm | 6,143 | 0.099 | 0.099 | 0.075 | 95.5 | 65.1 | 0.154 | 86 | 72 |
| fixed | Z_16 × 2, nearest | 9,023 | 0.0136 | 0.0136 | **−0.0096** | 24.9 | −11.2 | −0.0043 | 978 | 29 |
| fixed | Z_24 × 2, nearest | 20,447 | 0.0129 | 0.0129 | **−0.0103** | 25.8 | −12.7 | −0.0062 | 957 | 18 |
| fitted | free d ≤ 4 | 263 | 0.204 | 0.202 | 0.180 | 98.4 | 81.1 | 0.133 | 70 | 160 |
| fitted | free d ≤ 6 | 647 | 0.065 | 0.063 | 0.041 | 54.4 | 30.2 | 0.050 | 225 | 67 |
| fitted | free d ≤ 8 | 1,199 | 0.026 | 0.024 | 0.0023 | 32.2 | 2.3 | 0.0070 | 616 | 28 |
| fitted | Z_8 × 3, nearest | 5,039 | 0.129 | 0.128 | 0.106 | 90.3 | 63.1 | 0.257 | 95 | 91 |
| fitted | Z_8 × 3, 1 rad maps free | 7,247 | 0.0049 | 0.0030 | **−0.0190** | 8.2 | −25.6 | −0.0233 | 9,058 | 0 |
| fitted | Z_8 × 3, e(1) free perm | 6,143 | 0.089 | 0.087 | 0.065 | 87.1 | 51.0 | 0.131 | 103 | 65 |
| fitted | Z_16 × 2, nearest | 9,023 | 0.0146 | 0.0127 | **−0.0093** | 21.8 | −10.8 | −0.0014 | 1,292 | 34 |
| fitted | Z_24 × 2, nearest | 20,447 | 0.0121 | 0.0102 | **−0.0118** | 19.2 | −14.3 | −0.0072 | 1,674 | 15 |

**The outcome, plainly.**
- **The extension does not make the register exclusion robust.** Against the refit, every
  class is rejected (z ≥ 8). Against the *other* quantum model, Z_16 × 2 and Z_24 × 2 with
  nearest shifts still beat the quantum prediction by 0.009–0.012 nats per shot (z −11 to
  −14), and Z_8 × 3 with free 1 rad maps beats it by 0.016–0.019. That is the decisive column,
  and it has the same sign and size as #34's −0.014 to −0.018.
- **The tier's own arms do not break even against the other model.** On the seven new arms
  alone, Z_16 and Z_24 score −0.001 to −0.007 against the other model, and the relaxed class
  −0.023. The 1 rad response that the quantum model predicts is matched well enough by a Z_16
  or Z_24 ring (shift errors 0.18 and 0.047 rad), or by free maps. What remains is smaller than
  the fixed-versus-fitted gap.
- **What the tier does exclude:** a Z_8 register, whether its 1 rad echo is the nearest
  shift or the best free permutation (deficit against other 0.065–0.106, z ≥ 51). That
  confirms #32's Lemma for n = 8: e(1)'s powers repeat with Z_8's period, and COMP11 and CUBE1
  expose it: they score 0.38–0.70 nats per shot on those arms against the other model. It also lifts free d = 8 slightly:
  against other it goes from +0.0005 / −0.003 (#34) to +0.0030 / +0.0023, with z of 3.5 and
  2.3. Neither of these is the gap #34 asked the tier to close.
- **Which n stays unexcluded:** n = 16 and n = 24 at n_pop = 2 (nearest shift), and any n
  with free 1 rad maps (shown at Z_8 × 3). n = 12 and n = 32 were not fitted; n = 24
  contains Z_12, and n = 32 approximates 1 rad more closely (0.018 rad), so neither is
  expected to fare worse for the register.

**Options** (`incommensurate.json` `options`, priced with `qmem.followup`'s cost model).
"More 1 rad shots" keeps each arm's simulated held-out deficit per shot and its variance,
and reweights the tier's arms to that many shots per arm. The fits are not rerun, so this is
a projection, and a refit on more 1 rad data could move either side:

| option | shots | added upper bound (main tier) | Z_16 × 2 against other (fixed / fitted) | Z_24 × 2 against other | relaxed Z_8 × 3 against other |
| --- | --- | --- | --- | --- | --- |
| tier at the plan's 8,000 per arm | 56,000 | 40 s (own tier: 43 s) | −0.0096 / −0.0093 | −0.0103 / −0.0118 | −0.0161 / −0.0190 |
| tier at 16,000 per arm | 112,000 | 80 s | −0.0084 / −0.0076 | −0.0094 / −0.0108 | −0.0176 / −0.0199 |
| tier at 32,000 per arm | 224,000 | 160 s | −0.0072 / −0.0057 | −0.0084 / −0.0097 | −0.0191 / −0.0210 |
| tier at 64,000 per arm | 448,000 | 320 s | −0.0061 / −0.0041 | −0.0076 / −0.0087 | −0.0205 / −0.0219 |
| a second incommensurate angle (7 arms, not simulated) | 56,000 | 40 s | | | |

- **More shots do not close the gap.** The deficit against the other model is a bias per
  shot, not noise, so its sign does not change with shots. At 64,000 shots per arm (320 s,
  more than half the Open plan's 600 s on this tier alone) every unexcluded class still wins.
- **A second angle is not expected to close it either.** The tier's own arms already lose
  against the other model at 1 rad. A second angle gives a Z_24 or larger ring, or free maps,
  the same room: any angle lies within π/24 of a Z_24 shift. The gap is set by how well the
  9-parameter quantum model is specified (about 0.02 nats per shot), not by the angles. It
  was not simulated, so this is an argument, not a result.
- **What would close it** is a tighter quantum model, not a larger design: the register
  exclusion holds against the refit at z ≥ 8 in every class. A device characterisation that
  pins the quantum model to well within 0.01 nats per shot would make the refit column the
  operative one. That is outside Stage 2's scope.

**Size check** (`incommensurate.json` `null`, `incommensurate_null.csv`). The generator is the
best-fitting exact register on the fixed generator's data (smallest held-out deficit against
the refit, INV(1, −1) = identity): SIZE_GEN. SIZE_RESULT

**Recommendation: drop the tier.** It costs 40 s of upper bound in the main tier (43 s as
its own tier) and does not do what #34 asked of it. The register exclusion stays conditional
on the quantum model's specification, as §4.1 item 2 already states. `plan.json` is unchanged.

## 4. Pre-registration draft (Stage 4, if approved)

Everything below is fixed before any data. Amendments are dated and hashed as in Phase 7
(`docs/phase7-preregistration.md` §9).

- **Device and layout:** `ibm_kingston`; q75 memory, q59 output, ancillas from §2.2's rule on
  the live calibration (FakeKingston: q50, q21). Transpile at level 0 with the fixed layout and
  every echo fenced. The transpiled instructions are checked against the committed QASM hashes
  before each job.
- **Tiers** (`plan.json`):
  - *pilot:* 9 arms (baseline, E0, E00, P2, P4, INV2, RS, R2, A2) × 2,000 shots, one job,
    upper bound 17 s, hard cap 30 s;
  - *main:* all 18 arms × 8,000 shots in two jobs (two time blocks), upper bound 112 s, hard
    cap 150 s;
  - *controls* (§4.1): §2.5's benchmark and §2.6's Knee circuits, 26 circuits, 76,000 shots,
    one job, upper bound 38 s, hard cap 45 s.
  - Pub order is randomised within each job, and per-pub timing is recorded.
- **Pilot gate (stop rules).** The main jobs are submitted only if every check passes:
  - P(1) of E0, E00 and RS each within ±0.02 of the baseline's;
  - P(1) of P2 and P4 within ±0.10 of the fixed prediction (0.151, 0.182);
  - INV2 within ±0.03 of E00 in P(1) and lag 1;
  - R2 within ±0.03 of A2 in P(1) and lag 1;
  - the calibration and budget guards pass.

  A failed gate stops the stage; any rerun needs a dated amendment. Hard caps and the
  `max(charged, upper)` accounting are as in Phase 7.
- **Primary contrast (ii), the classical exclusion.**
  - Split the main tier's shots in each job by shot parity: even shots train, odd shots test.
  - On the training half, fit `ProgramModel` at d = 4 and d = 6 from the four §3 starts, at
    1,500 steps or convergence. Refit the quantum model's 9 noise parameters with φ fixed by
    design, not fitted.
  - The statistic is the held-out NLL deficit per shot, classical (best start) minus quantum,
    with a block bootstrap SE by job and 8-shot segment (#29/#31's drift-aware policy).
  - Reject "d ≤ D" if z > 1.645, Holm-adjusted over D = 4, 6. Report the held-out-arms fit as
    a secondary.
- **Coherent phase response (ii).**
  - Fit a first harmonic in φ to P(1; φ) and lag 1(φ) over the 8 scan points.
  - The amplitude and phase must agree with the fixed prediction within 3 SE plus a 0.05
    model margin.
  - Return to baseline: E0 equivalent to the baseline (TOST, margin 0.01 in P(1), 0.02 in
    lag 1).
  - Composition: COMP22 equivalent to COMP40, and INV2 to E00 (TOST, same margins).
- **Population invariance (i).** E0, E00 and RS are each equivalent to the baseline in P(1)
  (TOST, margin 0.01). §2.5's direct benchmark is in the controls tier (§4.1).
- **Equivalent dephasings (iii).** A2, R2 and R4 are pairwise equivalent in P(1) and lag 1
  (margin 0.01) and in w(0) (margin 10% of the baseline's |w(0)|). RS–E00 is the timing
  control in P(1) and lag 1; its w(0) TOST would need 86,000 shots per arm, so it is reported
  but not gated. (Superseded for w(0) by §4.1's Claim 3 contrast, a ratio with a 0.20 margin.)
- **Multiplicity.** Claim 3 needs every leg to pass, including §4.1's contrast
  (intersection–union, each at α = 0.05).
  The primary contrast is Holm over D. Everything else is descriptive.
- **Decision.**
  - *Supported (under §5):* all legs pass at D = 6. Per §4.1, the headline names the excluded
    class: element-wise classical models with d ≤ 6. It does not claim models with a
    phase-covariant register unless an amended design separates them (§3.2).
  - *Partial:* the exclusion passes only at D = 4.
  - *Not supported:* the exclusion fails at D = 4, or (i) or (iii) fails. A failure of (iii)
    would point to an implementation artefact (#25-type measurement disturbance) rather than
    to the absence of coherence.

### 4.1 Amendment of 2026-11-10 (#34; pre-data)

**Pre-data.** It was written before any Stage 2 job was approved or submitted, from simulation
only (§§3.1–3.4). It changes the draft above in four places.

1. **Null calibration.** The primary contrast is kept as registered. On classical d = 6 truth
   it rejected "d ≤ 6" in 0 of 12 replicates (§3.3). No change to the starts, the steps
   or the SE.
2. **Scope of the exclusion.** A Z_8 phase register is excluded only conditional on the
   quantum model's specification (§3.1). Stage 2's classical exclusion is stated for
   element-wise models with d ≤ 6 and for nothing larger. Report the register and free d = 8
   fits as a secondary, with the deficit against the refit and the fixed model. If the
   coordinator wants the register excluded, add §3.2's 1 rad tier with its own power study
   (not done here).
3. **The Claim 3 contrast, a new leg** (#28's taxonomy, item 3; `contrast.json`). On w(0),
   with #31's rule and the main tier's two jobs:
   - *Vanish:* R2 and R4 each vanish relative to RS:
     |w_X(0)| + 1.96 SE_X < f |w_RS(0)|, on all jobs and on each leave-one-job-out subset, with
     **f = 0.25**. Power at 8,000 shots per arm is ≥ 0.999 under both quantum models. f = 0.10
     is reported, not gated: power 0.95 / 0.85 (R2, fixed / fitted) and 1.00 / 0.75 (R4). The
     smallest f at 80% power is 0.085–0.096 (R2) and 0.059–0.104 (R4).
   - *Preserve:* RS is non-inferior to E00 in w(0): the ratio w_RS(0) / w_E00(0) has a lower
     one-sided 95% bound above 1 − m, with **m = 0.20**. The model predicts 0.935 (fixed) and
     0.937 (fitted). Power at 8,000 shots per arm is 0.98 (fixed) and 0.97 (fitted). The margin
     with 80% power at the plan is 0.153–0.159. The TOST form (ratio within 1 ± m) has the same
     power. Shots per arm for 80%: 3,400–3,900 at ±20%, 50,000–54,000 at ±10%, 1,100–1,300 at
     ±30%. Pooling RS against baseline, E0 and E00 lowers ±10% to 38,000–43,000.
   - *The trade* (`contrast.json` `trade`): ±10% is out of the tier's reach. Adding 40,000
     shots each to RS and E00 brings the 80% margin to 0.10, at a main-tier upper bound of
     173 s, 61 s more. Dropping the odd scan points P1, P3, P5 and P7 to triple RS and E00
     reaches only 0.12 and weakens §3's phase response. **Proposed: keep the plan's shots and
     m = 0.20.**
4. **Controls tier** (§3.4): added to `plan.json`, with QASM hashes, an upper bound of 38 s
   and a hard cap of 45 s. Its TOSTs (benchmark ±0.02 against e0) and the Knee inequality
   (§2.6) are descriptive checks of legs (i) and (iii), not gates.

### 4.2 Amendment of 2026-11-11 (#35; pre-data)

**Pre-data.** It was written before any Stage 2 job was approved or submitted, from
simulation only (§3.5). It changes nothing in the draft or in §4.1. It records a decision
not to add a tier.

1. **The 1 rad tier is not added.** §3.2's incommensurate-phase extension (seven arms,
   56,000 shots, 40 s upper bound in the main tier) was power-tested against Z_8, Z_16 and
   Z_24 registers. Against the refit, every register is rejected. Against the other quantum
   model, Z_16 × 2 and Z_24 × 2 (nearest shift) and Z_8 × 3 with free 1 rad maps still win, by
   0.009–0.019 nats per shot. More 1 rad shots (up to 64,000 per arm, 320 s) do not change
   the sign. `plan.json` is unchanged.
2. **§4.1 item 2 stands, with its last sentence replaced:** a phase register is excluded
   only conditional on the quantum model's specification. "Add §3.2's 1 rad tier" is
   withdrawn as the remedy, because §3.5 shows it does not remove the condition. A register
   exclusion that is robust to misspecification would need a quantum model pinned to well
   within 0.01 nats per shot, not more phase arms.
3. **Secondary report** (unchanged in kind from §4.1 item 2): the register and free d = 8
   fits on the 18 arms, with the deficit against the refit and the fixed model.

## 5. Assumptions the claim inherits

1. **D = 6.** Only classical models with ≤ 6 hidden states are excluded. Stage 2b (§3.1):
   a model with a phase-covariant hidden register (Z_8 × 2–4, d = 16–32) matches every
   summary statistic. It loses on held-out likelihood only by less than the quantum model's
   own misspecification margin, so it is *not* excluded. Stage 2c (§3.5): a 1 rad tier does not change
   this; Z_16 × 2 and Z_24 × 2 registers survive it against the other quantum model.
2. **Element-wise Markovianity.** Each physical element acts as the same stochastic map
   wherever it appears. A classical model whose pulse acts depending on context (the previous
   pulse, the step, the arm) is outside the class. This is what makes INV2 and the composition
   arms bite.
   - Partly testable: E0 against E00 against the baseline, and COMP40 = e4·e0, check that
     elements compose as single maps.
   - Not testable in general.
3. **Noncontextuality** for (iii) (Spekkens 2005; Schmid 2021). The three dephasings are
   represented by the same classical map. A contextual model may differ; agreement constrains
   it, but does not exclude it.
4. **Trust in q75's Z basis** as operationally defined by its measurement (Claim 3's basis
   dependence).
5. **Stationarity within a job.** Checked with #31's drift statistics across the two main
   jobs and within segments.
6. **Search completeness.** EM finds local optima. Four starts per d, with a monotone credit
   in d, is a heuristic, not a certificate. A better classical optimum at d = 6 would shrink
   the deficit. The power margin (about 30× in shots: 144,000 planned against 4,828 needed) is what protects against this.
   Stage 2b (§3.3): on classical d = 6 truth the pipeline never rejected, with z far below
   the threshold, so search incompleteness does not inflate the test's size.
7. **The bits are independent of the memory.** The ancillas are fresh, separate and measured
   before the echo. §2.3's crosstalk alternatives are the check.

## 6. Reproduction

```sh
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part review   # ~2-3 h, 2 workers, resumable
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part design --review-run <review run dir>
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part power --review-run <review run dir>   # ~1.5 h, resumable
uv run --group hardware python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --export-qasm
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --plan   # prints the UNSUBMITTED plan
uv run pytest tests/test_coherence.py
# Stage 2b (#34)
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part register   # ~6.5 h, 2 workers, resumable
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part null       # ~10 h, 2 workers, resumable
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part contrast   # < 1 min
uv run --group hardware python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --export-controls
uv run pytest tests/test_register.py
# Stage 2c (#35)
uv run python experiments/phase8_coherence.py --config experiments/configs/phase8_coherence.toml --part incommensurate   # ~22 h, 2 workers, resumable
uv run pytest tests/test_incommensurate.py
```

Stage 2c's run: `results/phase8_coherence/20261110T185852Z` (24 power tasks, then 8 size-check
tasks), started at 69da38e with a clean tree (29a273a, after it, added only tests). Seeds: `[null] seed = 34`, with `source_ids` fixed = 3,
fitted = 4 (so the 18 arms' shots are #34's register run's) and `reg = 5` for the size check.

Stage 2b's runs:
- `results/phase8_coherence/20261110T020530Z` (register, 26 fits);
- `…/20261110T082742Z` (null, 62 tasks).

The register run started at 5b0561e with a clean tree; the null run started at abf9f82 (its `git_dirty` flag is set only by the register part's summaries, written seconds earlier and untracked; no code changed after abf9f82).
Seeds: `[null] seed = 34` for both parts. Each task's data and starts derive from it, its
`source_ids` (a = 1, b = 2, fixed = 3, fitted = 4), the replicate, d and the start. The
contrast uses `[contrast] seed = 34`.

- `--from-run <dir>` resumes the review or power part from its task files and EM checkpoints.
- The machine was shared (load about 10) and the first runs were interrupted; they resumed
  from checkpoints.
- Seeds: `[review] seed = 32`, `[power] seed = 32` (each task's derived from it).
- The run directories here were `results/phase8_coherence/20261109T210012Z` (review) and
  `…/20261109T213520Z` (power).
- The curated files' metadata records the commit and config. All three runs started at
  7d39420; the review and power runs with uncommitted work in progress (`git_dirty = true`).
  30830cf changed only the `[plan]` config, `build_plan`, CLAUDE.md and this note, so the
  summaries were not regenerated. `design.json` and `power.json` got their runs' metadata
  afterwards (the script now writes it there itself).

Tests in `tests/test_coherence.py`:
- the echo is Rz(φ) and frame-neutral, and echoes compose as the phase group (with pulse
  noise); over-rotation breaks the group;
- the per-shot, per-step randomised channel equals complete dephasing (brute-force
  enumeration), and biased bits leave the predicted coherence; pooled fixed patterns are not
  the channel;
- the default model reproduces the calibrated step; the fitted quantum likelihood matches
  brute force on a small case;
- `ProgramModel`'s likelihood matches path enumeration, equals `InterventionModel` on the
  Stage 4c arms, increases under EM, and its statistics match sampling;
- the finite-group lemma;
- the Knee inequality for mixture preparations, and its violation by the instrument;
- the population and Ramsey benchmarks;
- the physical insertions survive transpilation (`hardware` group, skipped without it).
