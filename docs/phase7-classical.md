# Phase 7 Stage 2D: classical readout, control and processing artefacts

Issue #23 (child of #17). No QPU and no IBM service was used. Inputs, read-only: the Phase 6
shots and calibration snapshot (`results/summaries/phase6/`) and the Stage 0 instruments
(`results/summaries/phase7_audit/instruments.npz`). References are #18's: w(0) with a common
start set, the episode-free device value w(0) = −0.00587 ± 0.00011, and contiguous-block SEs.

```
uv run --group hardware python experiments/phase7_classical.py --config experiments/configs/phase7_classical.toml --export-qasm   # once, FakeKingston only
uv run python experiments/phase7_classical.py --config experiments/configs/phase7_classical.toml   # ~10 min, 2 workers
```

Code: `qmem.artefacts` (exact and planted forms of each artefact, guard rails, the D1–D3
models). Tests: `tests/test_artefacts.py`. Outputs: `results/summaries/phase7_classical/`.

## Verdict

**No plausible classical readout, control or processing artefact can produce ≥ 10% of the
episode-free witness.** Every artefact modelled here either leaves the witness unchanged or
*shrinks* |w|; none can create the device's negative w(0) from a memoryless or a device-like
2-state source.

- **From a memoryless source, no classical artefact can produce the witness.** Readout
  persistence on an i.i.d. source is a 2-state Markov chain, so w = 0 exactly. Any mixture of
  i.i.d. shots (drift, bursts, dropped shots) gives g(k) = E[p²(1−p)^k], a moment sequence:
  every Hankel matrix is positive semidefinite and **w ≥ 0**, the opposite sign to the
  device's. Both results are proved in the tests and hold at every strength.
- **From the device's own 2-state fit, the largest |w(0)| at any strength tried (up to 0.5)
  is 8e-5,** 1.4% of 0.0059. Over 393 random *device-like* 2-state sources (P(1) within 0.05
  of 0.577), persistence at the generous bound (0.01) gives a median |w(0)| of 2e-6 (0.03%)
  and a 90% quantile of 3e-5 (0.5%). **One source reaches 6.3e-4 (11%)**: a strongly
  oscillating chain (transition eigenvalue −0.79), unlike the device. At the typical value
  (0.001) the worst case is 1.1%. Drift and bursts at their generous values stay below
  5e-4 (8%) for every device-like source.
- **On the calibrated PMD model, every artefact *shrinks* |w|.** A 10% shift (0.00059)
  needs persistence ≥ 0.020, a threshold drift ≥ 0.021–0.042, or random bits in ≥ 9% of
  the shots, which is 2–9× the generous plausible values (table, §2). The device's own block
  P(1) caps drift at 0.005–0.019 (§4), below every needed amplitude.
- **The device data do prefer an effective sticky record, ε̂ = 0.027** (held out on an
  interleaved split: −0.00026 ± 0.00003 nats/symbol, |z| = 8). That is half of the 0.0005
  by which 3-state HMMs beat the calibrated model. It would *lower* the predicted w(0) from
  −0.0069 to −0.0061, close to the episode-free −0.0059 (§4). So it may explain part of the
  15% shortfall, but it cannot create the witness. This is a statistic of the output
  process: from the data alone a classical readout persistence of 2.7% (4× the calibrated
  assignment error) and a physical q59 or q75 effect (#21, #20) are indistinguishable. D1
  measures it directly, to ±0.001.

## 1. Control flow (method item 1)

A text audit of the transpiled `circuit.qasm` (`summary.json` → `control_flow`):

- 1,926 lines: 64 one-bit registers `y0..y63`, 771 `rz`, 770 `sx`, 192 `cz`, 64 `measure`,
  63 `reset`. There are **no** `if`, `else`, `switch`, `while`, `for`, `box`, `def`,
  `delay` or `barrier` statements, and **no classical bit is read anywhere** except where
  its measurement writes it.
- Each `y_t` is measured once, on `$59`, in step order. For t < 63 it is immediately
  followed by `reset $59`. Step 0 has no reset: the shot starts from `init_qubits`. q75 is
  never measured or reset.

What the control electronics do with these:

- **`measure`**: a 2.18 µs dispersive readout pulse on q59 (`readout_length` in the
  snapshot). The returned trace is integrated (kernelled) and a discriminator classifies it
  as 0 or 1. SamplerV2's default `meas_type = "classified"` returns only that bit; `kerneled`
  would return the IQ point ([SamplerV2 execution options](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-sampler-execution-options-v2),
  [schema](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/ibm-quantum-schemas-executor-version-1-0-quantum-program-model)).
- **`reset`**: on IBM systems this is a measurement followed by an X conditioned on *its own*
  outcome ([Qiskit `ResetAfterMeasureSimplification`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.ResetAfterMeasureSimplification);
  [IBM, mid-circuit measurement](https://www.ibm.com/quantum/blog/quantum-mid-circuit-measurement)).
  It takes 2.212 µs in the snapshot (2.312 µs in `FakeKingston`), including the
  feedforward of that one bit; IBM quotes feedforward latencies of about 0.6–1 µs
  ([IBM, utility-scale dynamic circuits](https://www.ibm.com/quantum/blog/utility-scale-dynamic-circuits)).
  Its result is not returned. A conditional reset is limited by the measurement fidelity
  and cannot return a leaked qubit to |0⟩, so the hardware idles between shots
  ([IBM, Nighthawk r2](https://www.ibm.com/quantum/blog/nighthawk-r2)).
- **Between shots**: `init_qubits = True` (the default) resets every qubit before each shot;
  `rep_delay` defaults to 250 µs ([execution options](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-sampler-execution-options-v2)).
- **Processing**: `shots_from_result` stacks registers `y0..y63` into columns, and
  `np.packbits` (big bit order) stores them. #18 verified both; they are not re-verified here.

The only classically conditioned operation is the X inside each `reset`. Its condition is a
fresh measurement of q59 right after `y_t`, not the stored record `y_t`. **No stored record
feeds back**: nothing ever reads a `y` register. The mechanisms that could still couple a
later output to an earlier one are:

1. **The reset's conditional X** (physical). If its internal measurement misreads, q59
   starts the next step in |1⟩. Since q59 was just measured, that is more likely after
   `y_t = 1` (`P(0|1)` = 0.0068) than after `y_t = 0` (`P(1|0)` = 0.0010). This is a q59
   state effect, owned by Stage 2B (#21); `qubit_control` models it for D1 and D2.
2. **Readout-chain persistence** (classical): resonator ring-down, amplifier or ADC baseline,
   or discriminator hysteresis could make the assignment of `y_{t+1}` depend on `y_t`.
   Modelled here as `persistence`.
3. **Slow discriminator drift across shots** (classical), modelled as `drift`.
4. **Measurement back-action on q75**: q59's readout photons dephase or Stark-shift q75 in a
   way that depends on q59's state. That is crosstalk, owned by Stage 2C (#22).

Post-processing (packing, ordering, dropped or duplicated shots) cannot feed back; it can
only corrupt the data.

## 2. Artefact models, exact and end to end (item 2)

Each artefact acts on the *recorded* bit and has an exact form, a transform of a
`LinearProcess` whose `expected_windows(common=True)` is the estimator's exact expectation,
and a data form planted into sampled shots. Sources: (i) i.i.d. with P(1) = 0.58; (ii)
Phase 6's 2-state EM fit to the device; (iii) the calibrated PMD instrument (exact w(0) =
−0.00690).

| artefact | model | analytic check (tests) |
|---|---|---|
| persistence | `P(r_t \| y_t, r_{t−1})`: *sticky* (repeat the last record with ε), *after_one* / *after_zero* (one-sided) | ε = 0: no change; on i.i.d. sources w = 0 exactly; brute force over hidden paths |
| drift | per-shot threshold shift δ(n), ramp, step, sine or one-sided, 41 levels | δ = 0: no change; on i.i.d. sources a step gives w = 0 and a ramp gives w > 0 (PSD Hankel), equal to the closed form |
| burst | a contiguous run of shots from 5300: random bits, zeros, or assignment error 0.25 | fraction 0: no change; zeros only scale g, so w is unchanged; an i.i.d. burst on an i.i.d. source gives w = 0 |
| reordering | random, reverse, sorted by P(1), interleaved | window counts identical as integers; block SEs change |
| packing | byte order reversed, 16-bit swap, little bit order | round trip exact; each is a known column permutation and its own inverse |
| drop / duplicate | random shots lost, or repeated next to the original | fraction 0: no change; duplicating every shot leaves g identical |

**Strength needed** (`strengths.csv`). From (i) or (ii), the target is |w(0)| ≥ 0.0059. From
(iii), it is a shift of ≥ 0.00059 (10% of the episode-free value):

| artefact | (i) i.i.d. | (ii) 2-state fit | (iii) PMD, 10% shift | plausible (typical / generous) | generous ÷ needed |
|---|---|---|---|---|---|
| persistence, sticky | never (≡ 0) | never (max 2.4e-5) | **0.020** | 0.001 / 0.01 | 0.50 |
| persistence, after_one | never | never (5e-6) | 0.026 | 0.001 / 0.01 | 0.39 |
| persistence, after_zero | never | never (8e-5) | 0.092 | 0.001 / 0.01 | 0.11 |
| drift, ramp ±a | never (≤ 3e-7, > 0) | never (3e-5) | 0.042 | 0.002 / 0.01 | 0.24 |
| drift, step ±a | never (≡ 0) | never (7e-5) | **0.021** | 0.002 / 0.01 | 0.47 |
| drift, sine / one-sided | never | never | 0.033 / 0.021 | 0.002 / 0.01 | 0.30 / 0.47 |
| burst, random bits | never (≡ 0) | never (3e-8) | 9.1% of shots | 0 / 2% | 0.22 |
| burst, assignment 0.25 | never | never (2e-6) | 8.3% of shots | 0 / 2% | 0.24 |
| reorder, drop, duplicate | E[g] unchanged | unchanged | unchanged | – | – |

"Never" means not reached anywhere on the search range (persistence up to 0.5, drift up to
0.3, bursts up to 50%); the bracket gives the maximum reached. The plausible values compare
with q59's calibrated assignment errors, P(1|0) = 0.0010 and P(0|1) = 0.0068. A sticky ε
misassigns at a rate of ε·P(r_{t−1} ≠ y_t) ≈ ε/2, so ε = 0.01 is already 1.3× the mean
calibrated error. No published Heron number gives an outcome-to-outcome persistence (the
literature on measurement-induced transitions addresses leakage, not record hysteresis), so
the generous value is a judgement, and D1 is there to replace it with a measurement.

**End to end** (`sampled.csv`; 20,000 × 64, one dataset per row, the full pipeline). Every
planted artefact with an exact form agrees with it: |z| ≤ 1.1 against the contiguous-block
SE, in 51 rows. Further points:

- **Packing errors are not subtle.** On PMD every one of the three halves the witness
  (w(0) ≈ −0.0034), and the positional guard rail catches it. On the i.i.d. source it is
  invisible and harmless.
- **Reordering** leaves w unchanged but moves the block SE. Sorting the PMD shots by P(1)
  inflates the block SE of w(0) 15×. A burst followed by a random shuffle has the burst's
  pooled w but clean-looking block SEs. So order statistics can only be trusted on shots in
  time order, and every Stage 3 job should keep returned order and record it.
- **Duplicated shots** leave w unchanged; even 20 duplicates in 20,000 shots are flagged.
- **Drift and bursts** that would matter for the witness (§2 table) are flagged by the
  block-P(1) dispersion test long before that. A 2% random burst fails it at p < 1e-3 on
  every source, a ±0.01 ramp on the i.i.d. and PMD sources, and a ±0.05 ramp on all three.

## 3. Guard rails for Stage 3 (item 4)

`qmem.artefacts.guard_rails(shots, ..., marginals=model)` runs on returned-order shots and
returns `flags`. Every Stage 3 analysis should run it and report the flags, as #18
recommended for block statistics. The checks:

- 0/1 values; **exact duplicate shots** (expected ~1e-10 for 64-symbol shots);
- **block dispersion** χ² over 20 contiguous blocks, with within-block SEs, of P(1)
  (assignment and threshold drift), of the lag-1 repeat rate (changing persistence), and of
  g(1) − g(3) (#18's episode statistic);
- **order inflation**: the contiguous-block SE over the random-block SE, for w(0) and P(1).
  The threshold is 2, since each SD has about 16% sampling error. The w(0) ratio is
  unreliable near rank 1 (i.i.d. data), where ŵ is second order in the noise. It is the only
  rail that fired on any exchangeable synthetic dataset (the clean i.i.d. one, ratio 2.0), so
  ignore it for D1;
- **positional χ²** of the *shape* of P(1) by position against the model (the constant
  offset removed), which catches bit-order and packing errors.

On the device (`summary.json` → `device`): the full run fails the three dispersion tests
(P(1): χ² = 259 on 19 dof; repeat rate: 960) and order inflation of P(1) (3.9×). These flag
#18's episode. The episode-free run passes every check (positional shape χ² = 72.3 on 63 dof; order inflation 0.87 for w(0) and 0.98 for P(1)). There are no duplicate shots.

## 4. What the device's shots allow

- **Drift bound.** For the episode-free run, block P(1) has χ² = 16.9 on 19 dof (p = 0.59).
  Requiring the drift's non-centrality to stay below the observed excess plus 2√(2·19), a
  threshold drift is at most ±0.008 (ramp), ±0.0046 (step), ±0.0064 (sine) or +0.019
  (one-sided) (`drift_bound`). A 10% shift needs 0.042, 0.021, 0.033 and 0.021.
- **Persistence fit.** Calibrated model plus persistence, fitted on even episode-free shots
  and scored on odd ones (`device_persistence_fit.csv`):

| mode | ε̂ | held-out ΔNLL vs calibrated (nats/symbol) | w(0) of the fit | lag-1 excess (model) |
|---|---|---|---|---|
| sticky | 0.027 | **−0.00026 ± 0.00003** | −0.00612 | 0.177 |
| after_zero | 0.037 | −0.00025 ± 0.00003 | −0.00665 | 0.172 |
| after_one | 0.021 | −0.00006 ± 0.00002 | −0.00641 | 0.164 |
| (calibrated) | 0 | 0 | −0.00690 | 0.157 |

  The device's own lag-1 excess, P(1 | 1) − P(1 | 0), is 0.171 (episode-free), against
  0.157 calibrated: the outputs repeat more than the calibration predicts. If this were
  classical, ε = 0.027 would mean a history-dependent misassignment of about 1.3% per
  measurement, 3–4× q59's calibrated error. We rate that implausible for the readout chain
  alone, but it is not excluded. Stage 2B's q59 reset and Stage 2A's q75 models are the
  physical alternatives; D1 separates them from the readout chain.

## 5. Control circuits D1–D3 (item 3)

QASM: `results/summaries/phase7_classical/d{1,2,3}.qasm`, transpiled for `FakeKingston` on
q59 (D3: q59 and q75). Each step gets a per-step delay of 392 ns, so every control runs
within 0.26 µs of Phase 6's circuit on the same target (`controls_circuits.json`). None has
control flow; each `reset` is the only conditional, as in Phase 6. The exact predictions use
classical HMMs on q59's true level (`qubit_control`): after a measurement every step is
classical. They use the snapshot's q59 errors, P(1|0) = 0.00098 and P(0|1) = 0.0068 (q75:
0.0156 and 0.0146), T1 = 250 µs, and the reset as a measurement plus conditional X.

- **D1, memoryless** (q75 absent; `reset`, delay, rx(θ) with P(1) = 0.58, measure). Noiseless
  outputs are i.i.d. With calibrated noise the failed-reset correlation makes them a 2-state
  chain: **w = 0 exactly**, and the lag-1 excess is −0.0009 (= (1 − 2p)(P(0|1) − P(1|0))).
  Leakage that reads as 1 leaves w(0..3) at 0, and leakage at 1%/step with |2⟩ read as 1 at
  90% gives |w(0)| ≈ 1e-7. Leakage is a P(1) and lag-1 effect, not a witness effect. **D1's
  primary statistics are the lag-1 excess and block P(1)**: sticky persistence adds exactly
  ε to the lag-1 excess. D1 is distinct from 2A's A1, which keeps q75's step and resets q75.
- **D2, known 2-state process** (no reset, the reset's time as idle, rx with P(flip) = 0.30,
  measure). A Markov chain: w = 0 exactly, both noiseless and calibrated (T1 decay in the
  idle makes it asymmetric, P(1) = 0.494). Leakage gives at most |w(0)| = 3e-5 (1%/step,
  |2⟩ read as 1 at 90%), persistence at ε̂ = 0.027 gives 7e-8, and the lag-1 statistic is
  confounded with the rotation's calibration. **D2 tests rank ≤ 2 under the measurement
  without reset, but is insensitive to the mechanisms modelled here.** Stage 2B's leakage
  model, with its own parameters, should be substituted before D2 is run.
- **D3, classical positive control.** q59 and q75, neither reset, each with a fixed rotation
  (P(flip) = 0.80 and 0.85), both measured every step; the analysis uses y_t = a_t AND b_t.
  A 4-state classical chain (2 qubits × 2 levels) with a **known nonzero w(0) = −0.0651**
  (calibrated; −0.0665 noiseless), the device's sign. This is feasible and cheap: the same
  cadence, no feed-forward, and the predicted w also follows from each qubit's own record.
  It checks the pipeline end to end on hardware data. It cannot be a 3-state renewal: without
  feed-forward one measured qubit gives at most 2 classical states.

| control | prediction (calibrated) | shots to bound \|w(0)\| < 0.00059 at 95% | other sensitivity at 20,000 shots | QPU s |
|---|---|---|---|---|
| D1 | w = 0, P(1) = 0.576, lag-1 −0.0009 | 1,000 (q95 \|ŵ(0)\| = 7e-5) | lag-1 SE 0.0010: ε ≥ 0.005 at 5σ, ε̂ = 0.027 at 27σ | 13.2 (20,000) |
| D2 | w = 0, P(1) = 0.494 | 2,000 (q95 = 3.6e-4) | – | 7.6 (10,000) |
| D3 | w(0) = −0.0651 | – | z = −47 at 20,000; 5σ needs ~230 shots | 4.8 (5,000) |

The null quantiles come from 100 replicates per size (`control_null.csv`); D3's SD from 20.
QPU seconds use Phase 6's model (2 s + shots × (250 µs + 308.3 µs)). The configured set
(D1 20,000, D2 10,000, D3 5,000 shots) costs about 25.5 s of the ≈ 584 s left. The
`[hardware]` section of the config is marked `submitted = false`; nothing was sent.

## 6. Ranked recommendation for Stage 3

1. **D1 at 20,000 shots (13 s).** The one control that measures the readout chain directly:
   the lag-1 excess pins persistence to ±0.001, and block P(1) shows assignment drift with a
   fixed source. If D1's lag-1 excess is ≈ −0.001, the ε̂ = 0.027 seen in PMD is not
   classical, and Stage 2B/2A own it. If it is ≈ 0.026, the readout chain is implicated and
   the PMD witness is *under*-stated by about 11%. Run it in the same session as any PMD
   repeat, and consider `meas_type = "kerneled"` for D1, if the plan allows it, to see the IQ
   drift itself.
2. **D3 at 5,000 shots (5 s)**, as the hardware positive control: the pipeline must recover
   w(0) = −0.065 from a classical 4-state chain built from fixed gates.
3. **Run `guard_rails` on every Stage 3 dataset**, keeping shots in returned order, and treat
   any dispersion flag as #18 treated the episode: report per block and pooled.
4. **D2 (10,000 shots, 8 s)** only once Stage 2B has a leakage model worth testing; it adds
   little here.

Not done: no QPU, no account or backend query; `FakeKingston` durations (measure 2.28 µs,
reset 2.31 µs) differ from the live snapshot (2.18 µs and 2.21 µs), so the pad is computed on
the target used for the export, and Stage 3 should re-transpile against the live backend.

