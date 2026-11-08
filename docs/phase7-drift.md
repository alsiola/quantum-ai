# Phase 7 Stage 1: drift and nonstationary-mixture nulls for the Phase 6 witness

Issue #19 (child of #17). Existing data and simulation only: no QPU and no IBM service calls.
The reference numbers are #18's: the primary statistic is w(0) with a **common start set** for
every k (burn 16, k_max 7). Positions and shots are in **returned order** throughout (§1).

```
uv run python experiments/phase7_drift.py --config experiments/configs/phase7_drift.toml   # ~50 min, 2 workers
```

Code: `qmem.drift` (exact linear processes for the nulls, samplers that keep run order, block
and stratified bootstraps, segment statistics, χ² and change-point tests, adversarial search).
Tests: `tests/test_drift.py`. Outputs: `results/summaries/phase7_drift/`, with `segments.png`,
`se_vs_b.png`, `nulls.png` and `episode.png`.

## Verdict: robust (gate met), with a narrower reading

| criterion | result |
|---|---|
| w(0) ≠ 0 in each segment, \|z\| ≥ 5 under the primary SE | rest −0.00585 (z = −51); episode −0.00531 (z = −23) |
| per-1,000-shot-block w(0) in the rest is one constant | 17 blocks, −0.0065 to −0.0052, χ² p = 0.90 |
| no data-consistent drift null reaches 50% of the rest's \|w(0)\| | none is data-consistent at all; the best reach 21–24% while missing g by \|z\| ≥ 47 |

**What this does and does not show.** The witness is not produced by mixing, or drifting,
2-state regimes that are consistent with the data. Inside each segment the *output process*
has Hankel rank ≥ 3. It does **not** show quantum memory, or memory held in q75: a general
stationary **3-state classical HMM** reproduces the rest within every CI (w(0) = −0.0062), and a
4-state one reproduces the episode. This agrees with #18's NLL result.

## 1. Is returned order time order? (item 1)

The `qiskit_ibm_runtime` 0.51.0 source was read; no service was called.
- The client decodes each pub's `BitArray` from the server payload as is (`json.py`,
  `RuntimeDecoder`). It never reorders shots.
- `ExecutionSpans` guarantees only that the referenced data were collected inside the span's
  window. `DoubleSliceSpan` maps each pub to a shot slice. Neither the client nor the result
  format time-stamps individual shots or states their order.
- The main job has **one** span: 00:06:59–00:07:11 UTC, 20,000 shots, 11.35 s, so one shot
  every 567 µs. The pilot also has one span.

**So the order cannot be verified from these data.** All segments are in returned order. There
is indirect support: the episode is one contiguous block in returned order (p < 0.001 against
shuffled order, §3), which a scrambled order would not produce. But a server-side batch
structure would produce it too. If returned order is time order, the episode starts at 3.0 s,
lasts **1.53 s**, switches on within about 14 ms and switches off over about 57 ms.

**Order verification for Stage 3** (`[stage3_order_control]`, not submitted). Use Phase 6's
cost model: 2 s per job plus 558 µs per shot.

| design | QPU s | what it buys |
|---|---|---|
| one job, 20,000 shots (as run) | 13.2 | no time resolution inside 12 s |
| one job, 10 pubs × 2,000 shots | 13.2 | free; time labels only if the server reports one span per pub |
| **4 jobs × 5,000 shots** (recommended) | **19.2** | guaranteed ~3 s time labels from job spans; +6 s |
| standalone 4 jobs × 1,000 shots | 10.2 | an order check only |

Recommendation: run each Stage 3 configuration as at least 4 jobs, with several pubs per job.
Analyse per job.

## 2. Segment statistics and stationarity (items 2, 7)

These come from `segments.csv` (all, halves, quarters, deciles, episode, rest), `rolling.csv`,
`blocks_w.csv` and `segment_marginals.csv`. SEs are the primary SE (§7).

| segment | shots | P(1) | w(0) | w(1) | w(2) | w(3) | z(0) | ESS(0) / n |
|---|---|---|---|---|---|---|---|---|
| all | 20,000 | 0.581 | −0.00491 | −0.0184 | −0.0442 | −0.0044 | −52* | 0.08 |
| rest | 17,300 | 0.577 | −0.00585 | −0.0311 | −0.0526 | −0.0057 | −51 | ≈1 |
| episode | 2,700 | 0.604 | −0.00531 | −0.0073 | −0.206 | −0.056 | −23 | ≈0.7 |

\* Stratified by segment. Without stratification the contiguous-block SE at b = 500 is 3.3×
larger (z = −15).

- **Every episode-free decile** has w(0) between −0.0056 and −0.0061. Decile 2 contains the
  episode's start and has w(0) = −0.0040; decile 3 lies inside the episode, at −0.0051.
- **Tests specified before running (α = 0.01) in the rest.**
  - χ² homogeneity of g(0..7) across 500-shot blocks: joint p = 0.46, smallest per-k p 0.024.
  - Constancy of w(0..3) across 1,000-shot blocks: p = 0.90, 0.94, 0.82, 0.83.
  - The change-point scan is clean for P(1) (p = 0.04) and the w(0) influence (p = 0.10). For
    g(1) − g(3) it gives **p = 0.002**, located at shots 5200–5300. That is the episode's real
    start (§3): #18's boundary at 5300 is about 30 shots late.
- **Tests in the episode.**
  - Homogeneity: p = 0.12.
  - The g(1) − g(3) scan gives p = 0.001 at its last 50 shots, where the episode has already
    ended (§3).
  - Constancy of w(0) over its 2 blocks: p = 0.03.
- **With the refined boundaries** (a sensitivity check, since they are chosen from the data),
  every change-point p is ≥ 0.11 in the rest, excluding [5250, 8000), and ≥ 0.58 in the
  episode [5270, 7960). Homogeneity and constancy are unchanged.

So the rest is one stationary segment to the resolution of these tests. So is the episode,
once its edges are placed correctly.

## 3. Is the episode selected from the data? (item 3)

- **Selection-corrected significance.** Shuffling shot order gives a permutation null for the
  whole-run epidemic scan. The observed statistic is 3,164 for g(1) − g(3); the null's maximum
  in 1,000 permutations is 25. It is 308 for P(1) and 457 for the w(0) influence, against null
  maxima of 22 and 25. All three have p < 0.001, and all three put the boundaries at 5300/8000
  on the 100-shot grid.
- **Held out.**
  - Boundaries found on even shots alone are [5270, 7950). On the odd shots, g(1) − g(3) shifts
    by +0.078 inside them (z = 41).
  - Boundaries found from positions 16–39 of each shot are [5240, 7960). Positions 40–63 shift
    by +0.081 (z = 39).
- **Sensitivity.** Profile 95% CIs: start [5260, 5270], stop [7950, 7960] for g(1) − g(3). The
  union over the three statistics is [5250, 5290] and [7950, 8000]. The predeclared [5300, 8000)
  sits 30–40 shots late at both ends.
- **Abrupt or gradual.** The start is abrupt: a ramp of ≥ 50 shots is rejected (ΔLR ≥ 6.3).
  The end ramps over about 100 shots: a step is disfavoured (ΔLR = 9.2), and so is a ramp of
  ≥ 400 shots.

The episode is not a selection artefact.

## 4. Within-shot versus between-shot (item 4)

`within_shot.csv`:
- **g(k) from positions 16–39 and 40–63 agree** in every segment and at every k, paired over
  shots (|z| ≤ 1.9). The episode's g(1) = 0.156/0.158 and g(3) = 0.018/0.018 are the same in
  both halves.
- **The witness halves agree too:** the rest gives w(0) = −0.00564 ± 0.00018 and
  −0.00592 ± 0.00018.
- **Start-up transient (positions 0–15).** Its shape against the calibrated model, with the
  level removed, gives χ² 29 (rest) and 36 (episode) on 15 dof.

**What the data can tell apart.**
- **One step (< 5 µs).** A fluctuation this fast is just a different stationary process. That
  is a rank question, which the telegraph nulls at τ = 10 µs address (§5).
- **10–300 µs (within a shot).** Any drift would separate the halves. None is seen, at about
  1.5% of g(1).
- **0.6 ms – 11 s (between shots).** Visible in the block tests. The episode lives here: a
  ≥ 1 s change of the process that is constant within every shot in it.

## 5. Drift nulls (item 5)

**Data consistency.** A null is data-consistent when, for the target segment:
- every g(0..7) is within |z| ≤ 3 under the primary SE;
- the per-position marginals for positions 16–63 pass at the 99% χ² level;
- the aligned L = 4 block frequencies pass at the 99% χ² level.

These bounds are generous to the null. For each family, 64 restarts of a two-phase search
(fit the data, then minimise w(0) under a rising penalty) used exact differentiable statistics.
The searched member of each family was then simulated 20 times at 20,000 × 64 in run order,
through the same estimator and diagnostics (`nulls_adversarial.csv`,
`nulls_simulated_summary.csv`, `nulls.png`).

| family | target rest: best fit \|z\| (g) | largest \|w(0)\| / rest's | target episode: best fit \|z\| | largest \|w(0)\| / episode's |
|---|---|---|---|---|
| one 2-state HMM | 72 | 0 | 36 | 0 |
| K = 2, 3 mixtures (random assignment) | 55 | 0.21 | 23–25 | 0.42–0.45 |
| continuous drift across the run (16 points) | 72 | 0 | 28 | 0.23 |
| linear drift within each shot | 72 | 0 | 43 | 0.06 |
| telegraph-modulated, τ = 10 µs … 1 s | 54–57 | 0.20–0.22 | 21–23 | 0.40–0.49 |
| *control: general 3-state HMM* | **2.7 (fits)** | 1.07 | 11 | – |
| *control: general 4-state HMM* | **2.7 (fits)** | 1.07 | **2.4 (fits)** | 1.15 |

- **No drift family is data-consistent** for either target, or for the pooled run (best
  |z| ≥ 47). The largest |w(0)| in any member of these families is 0.0012–0.0013 on the rest,
  and those members miss g(1) by |z| ≈ 54.
- **Relaxing the bound.** With |z| ≤ 10 or 30 the K = 2, 3 mixtures still reach only
  |w(0)| = 0.0017 (29%). They reach 91% of the device's value only once g may miss by
  |z| ≈ 84 (the |z| ≤ 100 ladder).
- **Shape.** No family reproduces it. The mixtures' optima have w(1..3) ≈ 0, against the
  device's −0.031, −0.053 and −0.0057.
- **Block structure.** In simulation, contiguous K = 2 mixtures and telegraphs with
  τ ≥ 1 ms fail the homogeneity test in 75–100% of datasets, and their SE inflation is 1.3–12×.
  The device's rest shows neither. Random-assignment mixtures and fast telegraphs pass the block
  tests, but their exact w is the 21% above, at |z| ≈ 22–26 against 0.
- **Regimes fitted separately.** EM 2-state fits to the episode and to the rest, mixed
  contiguously as in the data, give exact w(0) = +2.5e-5: the wrong sign, and 0.4% of the
  device. Simulated, they give z(0) = +0.7 on average. Mean shifts up to 3× the episode's
  (ΔP(1) ≤ 0.08) at weight 0.135 or 0.5 give w(0) between 0 and +1.4e-5, always the wrong sign
  (`mean_shift.csv`).

**Analytic note.**
- **Across-shot mixtures.** For a 2-state HMM, O₀ = P(0, ·|·) is a non-negative 2 × 2 matrix,
  so its eigenvalues are real and g_c(k) = a μ₁ᵏ + b μ₂ᵏ. A mixture over shots of any K, with
  contiguous, random or continuous assignment, has g equal to a sum of ≤ 2K *real*
  exponentials. The device's g(k) has a period-3 revival: g(3) > g(2) and g(6) > g(5). In the
  calibrated model it comes from complex eigenvalues of O₀ at ±119°. Real exponentials cannot
  produce a period-3 revival, and #23 adds that i.i.d. regimes give w ≥ 0. A mixture can make
  w(0) < 0 only by mixing components that differ, at the cost of the shape. That is why the
  best fits miss g(1)/g(3) by |z| ≈ 55, and why w(1..3) of their optima vanish.
- **Telegraph-modulated 2-state HMMs** are 4-state HMMs.
  - For τ ≫ 300 µs the regime is frozen within a shot, so the family is exactly the K = 2
    mixture; the numbers agree to 1%. The slow end (τ ≥ 10 ms) also correlates neighbouring
    shots, which the block tests see.
  - For τ ≈ 10–30 µs, 2–6 steps, the fluctuator averages over a few steps. Its O₀ is a
    non-negative 4 × 4 block matrix that, in principle, could have complex eigenvalues. The
    search found none that helps.
- **Drift within a shot** that keeps the marginals flat leaves each start a near-stationary
  2-state process, so w ≈ 0. The data show no such drift anyway (§4).

## 6. Physical reading of the episode (item 6)

**Fit method** (`physics.csv`, `episode.png`).
- Maximum likelihood over every shot of each segment, starting from the calibrated 1-qubit
  instrument (`instruments.npz`) with extra channels composed in numpy.
- The q59 reset models use #20's joint (q75, q59) step pieces. With the calibrated reset they
  reproduce the 1-qubit instrument to 3e-14 (tested).
- #21's comparator, a q75 Z offset plus readout asymmetry, has 3 parameters. Every other model
  has at most 2.
- One phase step (δ) of 1 rad corresponds to 32.8 kHz, modulo 206 kHz.

Columns: ΔNLL is in nats/symbol against the calibrated model, and LR = 2ΔlogL.

| segment | model | parameters | ΔNLL | LR | g(1) | g(3) | P(1) | w(2) |
|---|---|---|---|---|---|---|---|---|
| rest | *data* | | | | 0.100 | 0.040 | 0.577 | −0.053 |
| rest | q75 offset + readout asymmetry (#21) | δ = +0.041 (1.3 kHz), e₀₁ = 0.002, e₁₀ = 0.011 | −0.00024 | 527 | 0.104 | 0.041 | 0.578 | −0.052 |
| rest | q75 detuning + dephasing | δ = +0.032, p = 0.028 | −0.00024 | 524 | 0.104 | 0.040 | 0.582 | −0.048 |
| rest | q75 detuning | δ = +0.036 | −0.00016 | 354 | 0.104 | 0.042 | 0.583 | −0.056 |
| rest | q59 reset only | e₀ = 0.008, e₁ = 0.018 | −0.00013 | 283 | 0.107 | 0.040 | 0.578 | −0.053 |
| rest | q59 readout only | e₀₁ = 0.007, e₁₀ = 0.010 | −0.00006 | 124 | 0.108 | 0.040 | 0.579 | −0.054 |
| episode | *data* | | | | 0.157 | 0.018 | 0.604 | −0.206 |
| episode | **q75 detuning + q59 readout (0→1)** | δ = −0.49 (−16 kHz), e₀₁ = 0.060 | **−0.0189** | **6513** | 0.146 | 0.023 | 0.590 | −0.041 |
| episode | #21's 3-parameter comparator | same; e₁₀ → 0 | −0.0189 | 6513 | | | | |
| episode | q75 detuning + q59 reset | δ = −0.47, e₀ → 0 | −0.0179 | 6175 | 0.143 | 0.024 | 0.566 | −0.125 |
| episode | q75 detuning | δ = −0.46 (−15 kHz) | −0.0177 | 6132 | 0.142 | 0.024 | 0.566 | −0.117 |
| episode | q59 readout only | e₀₁ = 0.20, e₁₀ = 0.11 | −0.0060 | 2089 | 0.126 | 0.029 | 0.601 | −0.002 |
| episode | q59 reset only | e₀ = 0.115, e₁ = 0 | −0.0044 | 1506 | 0.116 | 0.035 | 0.568 | −0.020 |
| episode | q75 dephasing + relaxation | γ = 0.75 | −0.0034 | 1158 | 0.119 | 0.027 | 0.587 | ≈ 0 |

- **The loss of the period-3 revival needs a change in the memory's per-step phase.** No
  q59-only change produces it. Readout or reset changes on q59 alone gain only a third of the
  likelihood. They leave g(3) at 0.029–0.035 against 0.018, and they drive the witness towards
  0. Extra dephasing or relaxation on q75 destroys the witness instead of reshaping it.
  - #22's χ² fit picked the +134 kHz branch (≈ −2.2 rad). My grid included it, but by
    likelihood the −15 kHz branch fits better.
- **The rise in P(1) needs the q59 term.** The detuning alone leaves P(1) at 0.566. Adding an
  extra 0→1 assignment error of 6% raises it to 0.590.
- **Same family or a separate mechanism?** The same family describes both segments: a q75 Z
  offset plus a q59 readout term.
  - The rest's best fit is #21's model, with an offset of +0.041 rad/step, matching #21's
    0.042.
  - The episode is that family with the offset moved by about −0.53 rad/step (about −17 kHz)
    *and* an extra 6% 0→1 assignment error, both for about 1.5 s.
  - The fit is incomplete, though. It still misses g(1) by 0.011, P(1) by 0.014 and w(2) by
    5×, while a general 4-state classical HMM fits the episode within CIs (§5). Two
    parameters of this family do not capture the episode.
- **Consistency only, not proof.** A q75 frequency shift of about 17 kHz lasting about 1.5 s
  (for example from a TLS) is *consistent with* the process running through q75's phase, and
  it needs a simultaneous q59 readout change. It does not prove that the memory is held in q75.
  Other explanations of the same effective phase:
  - a drift of the CZs' conditional phase (coupler or flux);
  - a q59 or readout-resonator frequency shift, which would also explain the readout term with
    a single cause;
  - a change in residual q59 coherence carried between steps;
  - phase drift in the control electronics.

  A spectator flip is excluded by #22, since q74's T1 is about 0.3 ms.

## 7. Uncertainty policy (item 7)

`se_vs_b.csv`, `se_vs_b.png`.
- **Within the rest,** the circular block-bootstrap SE of w(0) is 0.88–1.0× the shot-bootstrap
  SE for b = 100–2000. It is not inflated.
- **In the episode** it is 0.8–1.1×.
- **Pooled, without stratification,** it rises with b and never levels off: 1.8×, 2.4×, 3.4×,
  4.3× and 5.4× at b = 100–2000, because the episode is a level shift. Stratified by segment it
  stays at 0.8–1.0×.

**Predeclared for this and all later stages.**
1. **Primary SE** = max(whole-shot bootstrap SE, circular moving-block bootstrap SE with
   **b = 500 shots**), computed inside predeclared segments. Pooled numbers use the bootstrap
   stratified by segment. b = 500 is the largest size that leaves more than 5 blocks in a
   2,700-shot segment; within a stationary segment the SE does not depend on b.
2. **A whole-shot bootstrap cannot, by itself, rule out a nonstationary mixture.** A mixture
   across shots with random assignment is exchangeable, so the bootstrap is valid for it but
   cannot see it. A contiguous mixture is visible only to block methods. Only the drift nulls
   and the block tests above address H_drift.

## Recommendations for Stage 3

- **Fixed witness offset:** w(0), with a common start set, burn 16 and k_max 7. Report w(1..3)
  for shape only.
- **SE policy:** as in §7. Use b = 500 inside segments and stratify pooled numbers.
- **Segment policy.** For each job, before any comparison:
  - run the epidemic change-point scan with 1,000 permutations, on g(1) − g(3), P(1) and the
    w(0) influence, at α = 0.01;
  - run the χ² homogeneity test over 500-shot blocks;
  - run the w(0) constancy test over 1,000-shot blocks.

  If a scan fires, report the segments separately. Exclude the union of the boundary CIs from
  both segments, and compare controls against the episode-free value w(0) = −0.0059, not the
  pooled one.
- **Order verification is needed, and cheap:** 4 jobs × 5,000 shots (+6 s QPU per
  configuration), plus several pubs per job (free). Interleave baseline and control jobs
  (ABAB), so that an episode like this one cannot masquerade as a control effect.
