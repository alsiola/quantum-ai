# Phase 8 Stage 2: a coherence-specific test on q75/q59 (#32)

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

## Headline

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
not, which is why §5 lists noncontextuality as an assumption. These short circuits, like §2.5's,
are not in `plan.json`.

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
    cap 150 s.
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
  (TOST, margin 0.01). Add §2.5's direct benchmark if it is approved.
- **Equivalent dephasings (iii).** A2, R2 and R4 are pairwise equivalent in P(1) and lag 1
  (margin 0.01) and in w(0) (margin 10% of the baseline's |w(0)|). RS–E00 is the timing
  control in P(1) and lag 1; its w(0) TOST would need 86,000 shots per arm, so it is reported
  but not gated.
- **Multiplicity.** Claim 3 needs every leg to pass (intersection–union, each at α = 0.05).
  The primary contrast is Holm over D. Everything else is descriptive.
- **Decision.**
  - *Supported (under §5):* all legs pass at D = 6.
  - *Partial:* the exclusion passes only at D = 4.
  - *Not supported:* the exclusion fails at D = 4, or (i) or (iii) fails. A failure of (iii)
    would point to an implementation artefact (#25-type measurement disturbance) rather than
    to the absence of coherence.

## 5. Assumptions the claim inherits

1. **D = 6.** Only classical models with ≤ 6 hidden states are excluded.
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
```

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
