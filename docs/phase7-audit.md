# Phase 7 Stage 0: independent audit of the Phase 6 witness and NLL claims

Issue #18 (child of #17). No QPU was used. The audit re-derives every Phase 6 statistic from
the raw shots in `results/summaries/phase6/`, which it only reads, with a new estimator,
`qmem.witness`. That module is numpy only and imports none of the Phase 5a/6 helpers.

```
uv run --group hardware python experiments/phase7_audit.py --config experiments/configs/phase7_audit.toml --export-instruments   # once
uv run python experiments/phase7_audit.py --config experiments/configs/phase7_audit.toml   # 15 min, 4 workers
```

The export step rebuilds the calibrated, no-idle and noiseless 1-qubit instruments as
superoperators from `step.qasm` and the calibration snapshot. The result,
`instruments.npz`, is committed, so the audit itself does not need the `hardware` group. The
rebuilt instruments reproduce Phase 6's predicted marginals and g(k) to 1e-15, and its device
NLLs to 1e-16. Outputs: `results/summaries/phase7_audit/`. Tests:
`tests/test_witness_audit.py`.

## Verdict

**Analysis flaws found; the headline survives.**

1. **The witness is real.** w(0) = −0.00494 is reproduced exactly. It is nonzero under
   every check below, including in every block of 1,000 consecutive shots.
2. **The uncertainty is not calibrated against shot order.** The bootstrap treats shots as
   identically distributed, but shots 5,300–8,000 (in returned order) come from a different
   process. SEs from contiguous batches of shots are 1.4×, 4.0×, 5.8× and 11× the bootstrap
   SE at offsets 0–3. Under those SEs, z(0..3) = −39, −8.5, −4.7 and −1.3, against the
   reported −54, −33, −28 and −15.
3. **The claim "the device is noisier than its calibration" does not survive for the witness.**
   It comes mostly from that episode. Outside it, the witness is 85–98% of the calibrated
   prediction, not 58–76%.
4. **The claim "the calibrated 1-qubit model predicts better than any classical HMM fitted
   to the device" is false.** It is an artefact of the train/test split. With a random split,
   an interleaved split, or Phase 6's split with the episode removed from training, 3-, 4- and
   6-state HMMs beat the calibrated model by 0.0004–0.0005 nats/symbol (|z| = 9–15). Only the
   advantage over 2-state HMMs survives (0.0048–0.0052, |z| ≥ 30).

`docs/phase6-hardware-run.md` and `docs/report.md` §6.1 are corrected accordingly.

## 1. Reproduction, decoding, windows (issue items 1–3, 7)

- **witness.csv.** The independent estimator counts gaps between consecutive ones, a
  different algorithm from Phase 5a's window scan. With Phase 6's bootstrap draws, every
  number in the `device_main` and `device_pilot` rows matches to a relative 1e-12: w, SE, z
  and the 95% CI. g(k), its SE, the per-position marginals and P(1) all match exactly.
- **Pilot gate.** P(1) = 0.57906 against 0.58190 predicted (z = −0.65); the per-position
  χ² is 80.40 on 64 dof. Both match Phase 6.
- **Paired NLL table.** All six rows match Phase 6 to 1e-15.
- **Bit order.** `circuit.qasm` declares `y0..y63` in order and measures `y_t` at step t,
  always on physical qubit 59. `shots_from_result` reads register `y<t>` into column t, and
  `np.packbits` (big bit order) inverts exactly. The decoded order fits the calibrated
  model's start-up transient (positional χ² = 79.7). Time-reversed or little-endian
  decodings give χ² ≈ 137. The pattern 1 0ᵏ 1 is a palindrome, so the witness itself is
  nearly reversal-invariant (−0.00496 reversed). Tests fail on reversed or flipped decoding.
- **Windows and burn-in.** Each k has its own start count, L − 16 − k − 1 (47 down to 40),
  and every start is used exactly once (tested against brute force). Burn-in from 0 to 32,
  and a common start set for every k, change w(j) by less than 6%.
- **Normalisation.** The normalisation is scale-invariant. The max entry never switches
  between resamples on the main run. At j = 2 the top two entries, g(3) and g(2), are only
  6% apart, but no resample flipped them. On the pilot they flip in 0.5% of resamples. The
  bootstrap distributions are near-symmetric (|skew| < 0.09).
- **2-state HMMs with a non-stationary start.** For these, "w = 0 exactly" is not quite
  true. Each k averages the state over a different range of starts, so the Hankel entries
  are no longer the moments of one fixed state. With a transition eigenvalue of 0.985 and a
  pure starting state, the estimator's exact expectation is about 1e-9, six orders below the
  device. The test suite checks both the effect and the fix: one common start set gives
  exactly 0.

## 2. Uncertainty (items 4–6)

| j | w(j) | bootstrap SE | 95% percentile CI | 95% BCa CI | z (bootstrap) | z (contiguous batches) |
|---|---|---|---|---|---|---|
| 0 | −0.00494 | 0.000089 | [−0.00511, −0.00477] | [−0.00511, −0.00476] | −55 | **−39** |
| 1 | −0.0184 | 0.00054 | [−0.0195, −0.0173] | [−0.0195, −0.0173] | −34 | −8.5 |
| 2 | −0.0441 | 0.0016 | [−0.0472, −0.0409] | [−0.0473, −0.0409] | −27 | −4.7 |
| 3 | −0.0043 | 0.00029 | [−0.0049, −0.0038] | [−0.0049, −0.0038] | −15 | −1.3 |

- **The bootstrap resamples whole shots.** Its SE is stable to ±1.3% at 1,000 resamples
  across seeds.
- **Jackknife, delta method, and 20 randomly assigned batches** agree with it within 1%,
  1% and 12–23% respectively. The batch SE itself has about 16% Monte-Carlo error.
- **20 contiguous batches of 1,000 shots do not agree:** they give 1.4–11× the bootstrap SE.
  The cause is the episode (§3). With it removed, contiguous batches give 0.77–0.86× the
  bootstrap SE.
- **Primary statistic (predeclared): w(0) = −0.00494.** The 95% CI is [−0.00511, −0.00477];
  z = −55 with the bootstrap SE, and −39 with the conservative contiguous-batch SE.
- **Maximum of four |z|.** In 3,500 null datasets at 20,000 shots, max|z| never exceeded
  4.7, and its 99% quantile is at most 3.2 in every family. The device's 54 (or 39) has an
  empirical p < 1/3,501, and a Bonferroni-corrected normal p below 1e-300.

**Null calibration** (`null_summary.csv`, `null_z.png`). Stationary 2-state HMMs were
simulated at 20,000 × 64 (500 datasets per family) and 200 × 64 (1,000 per family), with
the same burn-in and estimator and 300 bootstrap resamples. The families: random HMMs;
Phase 6's device fit, from its stationary state and from its fitted initial state; transition
eigenvalues near 0, +1 and −1; and +1 from a pure state.

- **At j = 0 and 20,000 shots**, mean z is between −0.08 and +0.07 (Monte-Carlo error 0.05)
  and SD(z) is 0.95–1.07 (Monte-Carlo error 0.03) in every family but one. That meets the
  criterion of being within 10% of 1. The exception is eigenvalue ≈ 0 (SD 0.55, mean −0.30).
- **At j = 2–3, at 200 shots, and near rank 1, z is conservative** (SD 0.4–0.7, mean ≈ −0.3).
  When the null Hankel matrix is near rank 1, the determinant is second order in the noise,
  so ŵ is not Gaussian. A bootstrap z therefore overstates the null spread there; it never
  understates it.
- **95% percentile intervals cover 0 in 92–96% of datasets.**

So against a stationary 2-state null the z is calibrated or conservative. It is not
calibrated against non-stationarity across shots (§3).

## 3. The episode in shots 5,300–8,000

A two-change-point fit to g(1) − g(3) per shot finds one contiguous segment, shots
[5300, 8000), that is 13.5% of the run (`episode.png`). Inside it, the process changes
qualitatively:
- g(1) rises from 0.100 to 0.158 and g(3) falls from 0.040 to 0.018, so the period-3
  revival is largely gone;
- P(1) is 0.604 against 0.577;
- the calibrated NLL is 0.6737 against 0.6612.

| segment | shots | w(0) | w(1) | w(2) | w(3) | w(0..3) / calibrated |
|---|---|---|---|---|---|---|
| all | 20,000 | −0.00494 | −0.0184 | −0.0441 | −0.0043 | 0.72, 0.58, 0.76, 0.72 |
| episode | 2,700 | −0.00536 | −0.0070 | −0.191 | −0.054 | 0.78, 0.22, 3.3, 9.0 |
| rest | 17,300 | −0.00587 ± 0.00011 | −0.0312 | −0.0526 | −0.0057 | **0.85, 0.98, 0.91, 0.94** |

Pooling the two regimes dilutes the witness: |w(0)| for the pool is smaller than for either
segment. P(1) also cancels in the pool. The rest of the run sits below the calibrated 0.582
and the episode above it, so the pool matches the calibration in a way neither segment does.

The excess block KL against the calibrated model is therefore not the episode's doing. For
the rest of the run alone it is 0.00024 at L = 4 (null 0.000003) and 0.00054 at L = 8 (null
0.00015). Both are larger than the pooled 0.00009 and 0.00042.

**Stage 1 note.** An equal mixture of 2-state HMMs fitted to each tenth of the run gives
w(0) = 2.5e-5, 200 times below the device. Slow mixing across shots of 2-state processes at
the drift seen here does not produce the witness. The episode is a change in the process,
not only in its rate.

## 4. Held-out HMM fits (item 8)

The audit used its own Baum-Welch implementation (numpy). Starts for each size: 4 random
starts, a moment-based renewal start (zeros since the last one), an order-m Markov start for
2 and 4 states, and a state-splitting start from the best smaller fit. Fits ran to a train
tolerance of 1e-8 nats/symbol or 300 iterations, and the best was chosen on validation. Phase
6's own torch initialisations and early stopping were rerun for its winning restarts.

Held-out NLL minus the calibrated model's, in nats/symbol, paired over test shots
(`nll.csv`):

| split (train / val / test) | 2-state | 3-state | 4-state | 6-state |
|---|---|---|---|---|
| returned order (Phase 6: first 9,000 / next 1,000 / last 10,000) | +0.0061 | +0.0017 | +0.0018 | +0.0019 |
| same, Phase 6's procedure (reproduced) | +0.0060 | +0.0018 | +0.0018 | – |
| random shots (9,000 / 1,000 / 10,000) | +0.0049 | **−0.0004** | **−0.0004** | **−0.0004** |
| interleaved (even shots train/val, odd shots test) | +0.0048 | **−0.0005** | **−0.0005** | **−0.0005** |
| returned order, episode removed from train/val | +0.0052 | **−0.0005** | **−0.0005** | **−0.0005** |

SEs are 0.00016 for the 2-state rows and 0.00003–0.00007 for the others, so every entry has
|z| ≥ 9.

- **(a) EM.** More starts, moment-based starts and running to convergence do not change the
  returned-order result. Phase 6's fits were not stuck in poor local optima.
- **(b) The split.** It is confounded. The returned-order training set contains the whole
  episode: 2,700 of its 9,000 shots come from a different process. Every split that breaks
  this confound reverses the sign for 3 or more states.
- **(c) The claim.** "The calibrated 1-qubit model predicts better than any classical HMM
  fitted to the device" does not survive. What survives: the calibrated model, fitted to
  nothing from the device, beats every 2-state HMM fitted to it by about 0.005 nats/symbol. It
  also beats the PMD source and the noiseless model. A 3-state HMM fitted to device data
  beats it by about 0.0005.

## 5. Recommendations for #17

**Stage 1 (drift).**
- Make the episode the object of study. Is shot order time order? Do g(k), not just P(1),
  change? Does the episode recur in other jobs and calibration windows?
- Report every hardware statistic per contiguous block of shots as well as pooled. Use
  contiguous-block (or block-bootstrap) SEs as the primary uncertainty.
- Predeclare w(0) as the primary witness. Use a common start set for all k, so that a
  2-state HMM gives exactly 0 from any starting state.

**Stage 2 (physical controls).**
- Compare against the episode-free statistics: w(0) = −0.0059, 85% of the calibrated value.
  The pooled −0.0049 is not the right reference.
- For any prediction claim, split shots at random or interleaved, never by returned order.
  Include 3+ state HMMs: they beat the calibrated model on clean splits.
- The block-KL excess outside the episode (L = 4: 80× null) is the remaining unexplained
  mismatch with the calibration. Readout and reset error models are the first suspects,
  since the rest of the run's P(1) is 0.005 below the calibrated value.
