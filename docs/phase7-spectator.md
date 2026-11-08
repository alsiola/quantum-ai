# Phase 7 Stage 2C: spectators, couplers, crosstalk, and the replication design

Issue #22 (child of #17). Local simulation only: no QPU and no IBM service calls. The
reference numbers are #18's (`docs/phase7-audit.md`). The primary statistic is w(0), with a
common start set for every k. Episode-free, the device gives w(0) = −0.00585, which is 85% of
the calibrated −0.00690.

```
uv run --group hardware python experiments/phase7_spectator.py --config experiments/configs/phase7_spectator.toml --export-device   # once
uv run python experiments/phase7_spectator.py --config experiments/configs/phase7_spectator.toml   # 25 s, 1 worker
```

- **Code:** `src/qmem/spectator.py`, a numpy-only, time-resolved branch simulator of the
  step. Its core is the pair (output, memory), with up to two spectators.
- **Tests:** `tests/test_spectator.py`.
- **Outputs:** `results/summaries/phase7_spectator/`, including `qasm/` with the unsubmitted
  control circuits.
- **Inputs:** FakeKingston's map, calibration and timing are in `device.json`. FakeKingston's
  calibration is dated 2026-04-15, so it is stale.

## Verdict

**At the published typical strengths, spectator and crosstalk effects cannot produce the
witness, cannot produce ≥ 10% of the episode-free witness, and cannot produce the episode.**
The typical strengths used are:
- static ZZ ≤ 10 kHz with the coupler off;
- 5% thermal population;
- a readout Stark shift on the neighbour of about 16 kHz.

All of these together change w(0) by 5.5% of the episode-free witness. The largest single
effect, the Stark shift, gives 9.6%. Only at the upper end of reported ZZ (50 kHz) does a
thermal q74 reach 10%.

None of these mechanisms creates a witness. With q75 reset every step, spectators leave
|w| ≤ 4e-12, against 0.0059 on the device. They only reshape q75's instrument. A spectator
flip cannot give the episode either. The episode's P(1) rise is not reproduced by any q75
mechanism.

## 1. Topology and idle times

On `ibm_kingston` (Heron r2, heavy hex, one tunable coupler per edge), both qubits have
degree 2.

| qubit | role | neighbours (couplers) | T1 / T2 (µs) |
|---|---|---|---|
| 75 | memory | 59, 74 | 321 / 139 (snapshot) |
| 59 | output | 55, 75 | 250 / 371 (snapshot) |
| 74 | spectator of q75 | 73, 75 | 323 / 205 (FakeKingston) |
| 55 | spectator of q59 | 54, 59 | 196 / 107 (FakeKingston) |

The couplers involved are 59–75 (the CZ pair), 74–75 and 55–59, with 73–74 and 54–55 in the
second ring.

Per step (4.852 µs, scheduled from `step.qasm` with the snapshot durations):
- **q59** is busy 4.79 µs: reset 2.212 µs, gates 0.46 µs, readout 2.18 µs.
- **q75** idles 4.392 µs: all of q59's readout and reset, which is 91% of the step. It is
  busy only during the 0.46 µs gate region.
- **q74 and q55** idle throughout: 310 µs per shot, plus the 250 µs repetition delay.

See `topology.csv` and `idle.csv`.

**Published strengths.** None of these is a Heron-specific measurement.
- **Static ZZ, coupler off:** below 10 kHz on IBM's tunable-coupler test devices [Stehlik
  et al., PRL 127, 080505 (2021)]. No Heron-specific number is published. One secondary
  report quotes < 50 kHz for Heron r2 (unconfirmed); this is used as the upper case. For
  comparison, fixed-coupling devices are at tens to hundreds of kHz [Kandala et al., PRL 127,
  130501 (2021)].
- **Readout crosstalk on a neighbour:** 1.7e-3 error per mid-circuit measurement, dominated
  by a Stark Z phase [Govia et al., NJP (2023), Falcon r8]. That is about 0.1 rad over their
  ~1 µs readout, or about 16 kHz; applied here through q59's readout and reset. On Heron r2 (ibm_marrakesh), the spectator loss during readout matches plain
  idling [Harper et al., arXiv:2504.07258].
- **TLS:** telegraphic frequency shifts of 5–140 kHz [Schlör et al., PRL 123, 190502 (2019)];
  switching on minute-to-hour scales [Klimov et al., PRL 121, 090502 (2018)].
- **Thermal population:** 3–9% [Haupt et al., arXiv:2506.09145]; 5% is used.

## 2. Models and sensitivity

The simulator schedules the step in time. Between events it applies:
- ZZ between any pair, `2π ζ n_a n_b`;
- static detunings;
- a Stark shift and extra dephasing on the memory while the output is read out or reset;
- spectator relaxation towards a thermal population (generalised amplitude damping);
- extra loss on the memory while a TLS is "on".

The persistent register is q75 plus the spectators, so a spectator is allowed to carry
memory.

**Tests.** The simulator was checked against analytic and brute-force cases:
- With zero coupling it reproduces #18's calibrated instrument to 7e-16, and Phase 6's step
  duration exactly.
- A decoupled spectator (thermal, frozen, or two of them) traces out to that instrument to
  1e-13.
- It agrees with `qmem.hardware`'s Qiskit branch simulator for another noise setting.
- Analytic cases: the ZZ phase and coherence, spectator relaxation, TLS loss, the Stark phase,
  and an XX echo cancelling a detuning.
- A frozen excited spectator equals a detuning, and a frozen mixture equals the mixture of
  processes.

w(0) as a ratio to the calibrated value (`sweep_*.csv`, `sensitivity.png`):

| mechanism | at typical strength | where w(0) changes by ≥ 10% |
|---|---|---|
| thermal q74, ZZ to q75 (p = 5%) | 10 kHz: 0.988 (−10 kHz: 0.973) | \|ζ\| ≳ 30 kHz (−30 kHz: 0.85) |
| q74 held in \|1⟩ (or a frozen TLS) | 10 kHz: 1.02, but w(2) 1.73 | \|ζ\| ≳ 10 kHz (−10 kHz: 0.82) |
| q55, ZZ to the output q59 | 10 kHz, p = 1: 0.98 | only at ≈ 100 kHz with p = 1 |
| residual ZZ q59–q75 | 10 kHz: 1.018 (w(1): 0.73) | ≳ 30–100 kHz |
| Stark shift on q75 during readout | 16 kHz: 0.92 | \|δ\| ≳ 10–30 kHz; non-monotonic |
| extra dephasing on q75 during readout | none published | Γ ≳ 5×10³ s⁻¹ (1e4: 0.83) |
| TLS loss on q75 when on | — | T1,TLS ≲ 100 µs with p = 1 |

**Can a spectator make a witness on its own?** No. `own_witness.csv` resets q75 every step,
so its memory is gone. Then:
- one spectator gives a 2-state process, and w = 0 to 1e-23;
- two spectators, thermal or frozen, at 100 kHz with p = 0.5 give |w(0)| ≤ 4e-12.

Spectators are classical here (diagonal, never driven), and their effect on the output per
step is small. The witness needs q75's memory.

**How the mechanisms differ.** Static ZZ to a thermal spectator adds a slow classical hidden
variable: it adds effective memory to the hardware, but not memory attributable to q75. The
Stark shift, the pair ZZ and dephasing during readout change q75's instrument without adding
any memory.

The extra-dephasing row is the one mechanism here that could plausibly explain the 15%
deficit outside the episode. It needs a readout-induced pure dephasing rate of about 8e3 s⁻¹
on q75, comparable to its 1/T2. This is untested, and the Heron r2 evidence [Harper et al.]
argues against a large excess.

## 3. The episode in shots [5300, 8000)

Its signature (common start set): g(1) 0.100 → 0.157, g(3) 0.040 → 0.018, P(1) 0.577 → 0.604.
Fits of 1–2 physical parameters to g(0..7) and P(1) are in `episode_fits.csv`, with a
diagonal χ² on 9 numbers.

| mechanism (episode) | χ² | g(1) | g(3) | P(1) |
|---|---|---|---|---|
| calibrated | 7229 | 0.107 | 0.041 | 0.582 |
| q75 static detuning (best +134 kHz) | 1022 | 0.174 | 0.021 | 0.583 |
| q75 TLS loss (T1,TLS ≥ 50 µs) | 6569 | 0.107 | 0.039 | 0.582 |
| detuning + loss | 986 | 0.174 | 0.021 | 0.584 |
| thermal q74, ZZ ±50 kHz, population free | 6259 | 0.119 | 0.035 | 0.567 |

- **Detuning.** A q75 frequency shift reproduces the loss of the period-3 revival and the
  rise of g(1). The +134 kHz best fit is a per-step phase of 4.09 rad, which is about
  −2.2 rad; the detuning is nearly periodic in 206 kHz. Such a shift lies at the top of
  reported TLS shifts and far above any coupler-off ZZ.
- **P(1).** No q75 mechanism raises P(1). The best fits leave it at 0.583 against 0.604.
- **A spectator flip** cannot produce the episode, for two reasons:
  - Its lifetime (T1 ≈ 0.3 ms) is about 5,000 times shorter than the episode (2,700 shots ×
    0.56 ms ≈ 1.5 s, if returned order is time order).
  - At ≤ 50 kHz, its effect has the wrong pattern.
- **A long-lived TLS** that shifts q75 by about 70–130 kHz is consistent with the g-pattern
  only, and needs a second, non-q75 change (readout or reset of q59) to give P(1). That
  combined fit is Stage 1's step 6 (#19). Consistency is not proof: a drift of q59's
  readout assignment, or a control-phase error, can also move g(1).
- **The rest of the run.** The best detuning is −1.2 kHz (χ² 748 → 551). A thermal q74 at
  −50 kHz with p = 0.1 does better (χ² 365), but χ² stays ≫ dof, and the fit lowers |w(0)|
  to −0.0050 against −0.0058 observed. The same pattern, extra 3-state structure, is why
  3-state HMMs beat the calibrated model (#18). This fit does not show that a spectator is
  present; C3 tests it directly.

## 4. Controls (ready, unsubmitted)

**Assumptions.**
- 20,000 shots per arm, both arms in one job (Phase 6's size).
- SE = 1.4 × the model's per-shot delta-method SD / √N. The model gives 0.0156 per shot,
  against 0.0155 on the device outside the episode.
- QPU time = 2 s + N × (250 µs + circuit), which is 24.3–24.4 s per two-arm job.
- Signals are relative to the null prediction for the same contrast (`controls_*.csv`,
  `detectable_zz.csv`).

| control | null prediction | signal at plausible strength (z at 20k/arm) | detects at 5σ, 20k/arm |
|---|---|---|---|
| **C3** q74 reset vs reset + X at shot start | Δw(0) = 0 exactly | ZZ 10 kHz: +0.00037 (1.7σ); −10 kHz: 5.7σ; all-plausible: 4.6σ; 50 kHz: 26σ | \|ζ(q74–q75)\| ≳ 10 kHz (negative), ≈ 15 kHz (positive) |
| **C4** XX echo on q75 through q59 readout + reset | Δw(0) = +0.00009, ΔP(1) = −0.008 | Stark 16 kHz: 2.5σ; ZZ 50 kHz: 2.8σ; all-upper: 4.1σ | Stark ≈ 25 kHz (extrapolated); ZZ to a thermal q74: < 5σ up to 100 kHz (max 4.8σ at −30 kHz) |
| **C1** role swap (memory q59, output q75) | Δw(0) = +0.00073 (+11%) from calibration alone | ≤ 2.7σ even at 50 kHz | nothing at plausible strength |
| **C2** another pair vs its own prediction | predicted spread across pairs: 0.62–1.04 × (10–90%) | w(0) moves by −10% to +11%, inside the ~15% model error | not attributable (below) |

**C4 on this backend.** Sampler's `dynamical_decoupling` option is incompatible with dynamic
circuits, so the echo is written into the circuit by hand: barriers and explicit
`delay`/`x` on $75 during each measure and reset of $59 (`qasm/c4_dd.qasm`). The delays
are on FakeKingston's 4 ns dt grid. qiskit-ibm-runtime's `PadDynamicalDecoupling` is the
supported alternative.

Under the null, C4 is itself a test of the idle model: flipping q75 halfway through the
window changes how T1 acts, which predicts ΔP(1) = −0.008, about 13σ at 20k shots. Stage 3
must check with the live target's durations and scheduling that the pulses land inside the
readout window.

**C2 pairs.** FakeKingston's calibration is used as a placeholder, and Stage 3 must redo the
ranking with live calibration. The search covers 160 directed pairs at least 3 hops from
{59, 75}, with readout ≤ 2%, chosen greedily and at least 3 hops from each other:

| memory / output | its other neighbours (alternate ancillas, for 2B) | predicted w(0) |
|---|---|---|
| q67 / q68 | 57, 66 | −0.00743 |
| q149 / q148 | 150 | −0.00742 |
| q104 / q105 | 103 | −0.00702 |

FakeKingston's own prediction for 59/75 is −0.00653, 5% off the snapshot's.

**Ranked recommendation** (if Stage 3 spends QPU time on spectators at all):
1. **C3**, about 24 s. It is the only control with an exact null, on the same qubits and the
   same calibration. It bounds the q74–q75 ZZ at about 10–15 kHz, or finds it.
2. **C4**, about 24 s. It tests coherent phases accumulated in the readout window (Stark,
   pair ZZ), and the idle model itself through P(1). Its null rests on the model, so its w(0)
   signal is weaker than C3's.
3. **C1**, about 24 s. It mostly tests the calibrated model under a role swap. It has little
   power against spectators.
4. **C2**, about 12–24 s per pair. It tests whether the witness replicates, not where it comes
   from.

Together C3 + C4 cost about 49 s of the ~584 s left.

## 5. Replication caution: decision table

A different pair changes T1, T2, readout, gate errors and neighbours all at once. Across the
160 candidates, calibration alone moves the predicted w(0) from 0.28× to 1.14× of the 59/75
value (median 0.90). On 59/75 the calibrated model is already 15% off outside the episode.

| outcome on pair P (block SE, P's calibrated model re-predicted with live calibration) | supports | does not support |
|---|---|---|
| w(0)/pred(P) within the model error seen on 59/75 (0.85 ± ~0.1) | the effect is generic to this circuit on Heron; the calibrated q-memory model transfers | anything about where the memory lives (q75 vs hardware); spectators are not excluded, since every pair has neighbours of similar strength |
| w(0)/pred(P) clearly outside 0.75–0.95 | the pair or device behaves differently | a spectator story, until the difference exceeds the calibration-only spread (re-predicted) and the model-error band. Run C3/C4 on P before attributing it |
| w(0)(P) ≈ 0 | the q75 result was local to that pair, or the model fails on P | that the 59/75 witness was a spectator effect; first check P's T2, readout and episode-like drift |

## Limits

- Spectators are modelled as classical (diagonal); coherent leakage into couplers is not
  modelled.
- Relaxation of q75 during its own gates follows Phase 6's model (none).
- The ZZ of q74/q55 to their own outer neighbours is ignored.
- The χ² in §3 ignores correlations between the g(k).
- All strengths other than the snapshot's T1/T2 are literature values, not measurements on
  ibm_kingston.
