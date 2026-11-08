# Phase 7 Stage 2A: q75 destruction and dephasing controls (design, simulated)

Issue #20 (child of #17). No QPU was used and no IBM service was called. Topology and native
operations come from `FakeKingston`, calibration from Phase 6's committed snapshot. Reference
numbers come from the Stage 0 audit (#18). Stage 1's SE policy (#19) and Stage 2B's reset
model (#21) had not landed, so this note uses #18's 1.4× SE inflation and its own reset model.

```
uv run --group hardware python experiments/phase7_ablation.py --config experiments/configs/phase7_ablation.toml --export  # once: circuits, QASM, pieces
uv run python experiments/phase7_ablation.py --config experiments/configs/phase7_ablation.toml          # predictions, about 30 s
uv run python experiments/phase7_ablation.py --config experiments/configs/phase7_ablation.toml --plan   # the Stage 3 job and its cost
```

Code: `qmem.ablate_hw`. Tests: `tests/test_ablate_hw.py`. Outputs:
`results/summaries/phase7_ablation/`, including `qasm/`, `predictions.csv` (w(0..3), g(0..7), P(1),
block KL), `blocks_L4.csv`, `shots.csv`, `scans.csv`, `a3_finite_m.csv` and `circuits.json`
(hardware support, durations, QASM hashes).

## 1. The controls

Each control adds one operation on q75 per step to Phase 6's transpiled circuit. Nothing else
changes. The operation goes after q75's leading single-qubit gates and before the step's
first CZ. In the schedule it runs while q59 is measured and then reset (4.4 µs, during which
q75 idles in the baseline), so it adds no duration. The exception is step 0, where there is
no readout to overlap it. Every circuit goes through the same level-0 transpile with a fixed
layout. The baseline that comes out is instruction-for-instruction Phase 6's `circuit.qasm`.

| circuit | operation on q75, every step | added per step | clbits | duration (FakeKingston) | ideal w |
|---|---|---|---|---|---|
| baseline | none | – | 64 | 321.0 µs | −0.0088 |
| A1 | `reset` | 1 reset | 64 | +2.31 µs | 0 |
| A2 | `measure` into unused bit `m[t]` | 1 measure | 128 | +2.28 µs | 0 |
| A3 | random Pauli, fixed per instance: `x` and/or `rz(π)`; I and Z pad with an 8 dt delay | 0.52 x, 0.58 rz, 0.48 delay | 64 | +0.03 µs | 0 (M → ∞) |
| A1s | `delay` 553 dt (2.212 µs, the live reset) | 1 delay | 64 | +2.21 µs | = baseline |
| A2s | `delay` 545 dt (2.18 µs, the live readout) | 1 delay | 64 | +2.18 µs | = baseline |
| A2s_spec | A2s's delay, plus `measure` of spectator q74 into `s[t]` | 1 delay, 1 measure (q74) | 128 | +2.18 µs | = baseline |
| A3s | `delay` 8 dt (an `x`) | 1 delay | 64 | +0.03 µs | = baseline |

No control adds a CZ or an `sx`. The circuit depth goes from 1,343 to 1,344.

**Hardware support (FakeKingston target).** `measure` and `reset` are calibrated on q59, q75
and q74. `x`, `delay` and `if_else` are native. Readout takes 2.28 µs and reset 2.312 µs.
That is exactly readout plus one `x` (32 ns), which fits a measure-and-conditional-X reset. The
live snapshot has 2.18 µs and 2.212 µs. The sham delays use the live values. Stage 3 must
regenerate them from the live target. The only neighbours of q75 are q59 and q74, so q74
neighbours q75 but not q59. That makes it the spectator for A2s_spec. Nothing limits the
extra 64 clbits.

## 2. Why A1, A2 and A3 give exactly w = 0

Phase 6's calibrated model resets q59 to a state that does not depend on what q59 held. So
the only carrier of memory between steps is q75. Write one A2 step as
`E_y = F_y ∘ D ∘ G`. Here `G` is q59's reset (with q75 idling) and q75's leading gates. `D` is
the Z measurement with its record discarded, `D(ρ) = Σ_m P_m ρ P_m`. `F_y` is the rest of the
step, ending with q59's readout y.

Just after `D`, the state of q75 is diagonal, `σ = Σ_m s_m |m⟩⟨m|`. From one step to the next,
`s` therefore evolves as an edge-emitting 2-state HMM with
`T_y[m, m'] = ⟨m'| (D ∘ G ∘ F_y)(|m⟩⟨m|) |m'⟩ ≥ 0`, and `Σ_{y,m'} T_y[m, m'] = 1`.

Every window probability g(k) from a fixed state, or averaged over one start set common to
all k, is then `u T_1 T_0^k T_1 1`, with u in ℝ². Every Hankel matrix therefore has rank at
most 2, its 3×3 determinant vanishes, and w(j) = 0 for every j. This holds for any noise that
is Markovian in time: gate depolarising, relaxation, q59's symmetric reset error, and the
readout error on the discarded m. It fails if memory has a second carrier (a q59 residual,
leakage, a spectator), if `D` is incomplete, or if the process is a mixture of regimes
(rank ≤ 2 per regime).

**A1.** The reset maps q75 to a fixed state, so the outputs are i.i.d. and g(k) = p²(1 − p)^k
has rank 1.

**A3.** Averaged over a uniform Pauli, q75 becomes I/2. Only X and Y carry the noise of the
`x` gate, which leaves a diagonal remainder of weight p₁/2, so the rank is still at most 2.

The tests check every one of these cases on brute-force random instruments, and on the
committed and the re-simulated instruments, all to below 1e-12.

## 3. Predictions

All predictions use exact superoperators: no Aer, no sampled shots. The step is factored on
the joint (q75, q59) memory as `Post_y ∘ Op ∘ Pre ∘ Reset`, and each control changes only
`Op`. The pieces come from `qmem.hardware.noisy_instrument` with the snapshot noise
(`pieces.npz`). The composed maps equal a direct simulation of each control's step circuit
to 2e-15.

The joint baseline reproduces Phase 6's predicted g to 2e-14 and its marginals to 1e-13 (noiseless: 4e-15).
It also reproduces #18's committed 1-qubit instrument. Rebuilt from the step, that instrument
matches #18's to 0.0. The added operations carry their own noise: q75's reset error
(0.0151), the readout dephasing, depolarising on the `x` (p₁ = 2.0e-4), and q59's relaxation
for the reset or readout time.

Witness values use #18's estimator: burn 16, a common start set, L = 64. The paired NLL is
NLL(Phase 6 calibrated) − NLL(own model), per symbol, on 20,000 shots drawn from the
control's model. The per-shot SD is the bootstrap SD of w(0) on 20,000 simulated shots ×
√N × 1.4.

| circuit | noise | w(0) | w(1) | w(2) | w(3) | P(1) | KL₄ vs Phase 6 | paired NLL | per-shot SD of w(0) |
|---|---|---|---|---|---|---|---|---|---|
| baseline, A1s, A2s, A2s_spec, A3s | noiseless | −0.00883 | −0.0492 | −0.0777 | −0.0089 | 0.587 | 0.0002 | – | – |
| same | calibrated | −0.00690 | −0.0317 | −0.0578 | −0.0060 | 0.582 | 0 | 0 | 0.020 |
| A1 | noiseless / calibrated | <1e-31 | <1e-31 | <1e-31 | <1e-31 | 0.341 / 0.346 | 0.101 | 0.106 ± 0.0003 | 0.0033 |
| A2 | noiseless / calibrated | <1e-17 | <1e-18 | <1e-19 | <1e-19 | 0.382 / 0.386 | 0.071 | 0.077 ± 0.0003 | 0.0104 |
| A3, M → ∞ | noiseless / calibrated | <1e-21 | <1e-27 | <1e-31 | <1e-33 | 0.383 / 0.386 | 0.073 | 0.080 ± 0.0003 | 0.0028 |
| A3, the 64 committed instances | calibrated | −7.1e-5 | | | | | | | |

The full g(0..7) and the 16 four-symbol block probabilities are in `predictions.csv` and
`blocks_L4.csv`. Moving the q75 operation after q59's gates, so that q59 idles in a live
state (the `*_serial` rows), changes P(1) by less than 0.001 and leaves w = 0.

**A3 at finite M** (`a3_finite_m.csv`). The realised circuits are fixed, so the device
converges to w of their M-instance mixture. Over random draws of the instances, the mean of
w(0) is −2.6e-3/M (−4.0e-5 at M = 64), with SD 5.7e-5. The committed 64 instances (seed
2020) give −7.1e-5, which is 1.2% of the device witness. That value, not 0, is A3's H_q75
prediction.

**A2's q75 record** (diagnostic only). The model predicts P(m = 1) = 0.493, P(y = 1 | m = 0) =
0.346 and P(y = 1 | m = 1) = 0.425. Given the recorded m_t, y_t depends on y_{t−1} only
through m's readout error: I(y_t; y_{t−1} | m_t) = 1.1e-6 nats. A clearly larger value would
mean a memory outside q75.

### H_other alternatives (calibrated noise; w(0); `scans.csv`)

| alternative | baseline | A1 | A2 | A3 | A2s_spec |
|---|---|---|---|---|---|
| H_q75 (calibrated model) | −0.00690 | 0 | 0 | 0 | −0.00690 |
| H1: residual q59 reset memory, e₀ = 0.001, e₁ = 0.0068 (snapshot readout) | −0.00690 | 2e-20 | +1.3e-6 | 3e-16 | −0.00690 |
| H1 × 10 | −0.00517 | −1e-18 | +1.2e-5 | 3e-14 | −0.00517 |
| H1 with q75's reset also state-dependent | −0.00690 | 7e-14 | +1.3e-6 | 3e-16 | −0.00690 |
| H2: 13.5% of shots at 3× noise (telegraph) | −0.00653 | 1e-21 | +3.6e-8 | 5e-12 | −0.00653 |
| H2: 13.5% of shots with q75 detuned 0.3 rad/step | −0.00632 | 3e-32 | +5.8e-7 | 1e-21 | −0.00632 |
| H3: A2's measurement keeps 5% of q75's coherence | – | – | +1.3e-5 | – | – |
| H3: the q74 readout dephases q75 by 5% / 25% per step | – | – | – | – | −0.00552 / −0.00204 |

Scans of one parameter at a time:

- **q59 residual, e₁ from 0.0068 to 0.9.** A2's w(0) stays between +1.3e-6 and +8.6e-5:
  at most 1.5% of the device's |w|, and positive. No residual q59 reset memory of any size
  reproduces the device witness once q75 is classical.
- **Coherence kept by A2's measurement** (c = 0.2, 0.5, 0.8). w(0) = −4.7e-5, −4.7e-4,
  −2.7e-3. The witness is strongly nonlinear in c.
- **Spectator dephasing** (1%, 5%, 10%). The A2s_spec witness keeps 96%, 80% and 63% of the
  baseline's.

## 4. Shots and QPU seconds

Cost model: 2 s per job + (250 µs + duration) per shot; 0.56 ms per shot here. "5σ"
uses the inflated per-shot SDs above. Device reference: w(0) = −0.00587 (#18, episode-free).
Under H_q75 a destructive control reads its prediction (§3) and a sham reads the device
baseline. All numbers are from `shots.csv`.

| circuit | test | shots at 5σ | QPU s | z² per QPU s |
|---|---|---|---|---|
| baseline | w(0) ≠ 0 | 301 | 0.17 | – |
| A1 | vs baseline (two arms) / persists at 100%, 50%, 20%, 5% | 309 / 8, 32, 202, 3,227 | 0.17 / … 1.8 | 1,382 |
| A2 | vs baseline / persists at 100%, 50%, 20%, 5% | 379 / 78, 311, 1,944, 31,098 | 0.21 / … 17.4 | 143 |
| A3 | vs baseline / persists at 100%, 50%, 20%, 5% | 315 / 6, 24, 164, 4,039 | 0.18 / … 2.3 | 1,837 |
| A1s, A2s, A2s_spec, A3s | 20% below the baseline (two arms) | 15,069 each arm | 8.4 | 3.0 |

"z² per QPU s" is for a witness persisting at 50% (destructive controls) or a 20% change
(shams). A1 and A3 have i.i.d.-like outputs, so their witness has a small SD and they are
cheapest per bit of evidence. A2 keeps a classical bit, which triples its SD. Every
concrete H_other in §3 changes A1's, A2's and A3's witness by less than 9e-5, so telling one
apart from H_q75 at a control takes about 4e5 shots or more for A2 (far more for the others),
beyond any feasible run. What a destructive control can do cheaply is tell "vanishes" from
"persists" (§5).

**Flags.**

- **A1, A2 and A3 cannot separate H_q75 from the concrete H_other alternatives tested
  (residual q59 memory, drift, incomplete dephasing).** All of them predict a control witness
  below 9e-5. A destructive control discriminates only against a memory path not modelled
  here, one that would make the witness persist at a sizeable fraction.
- **The shams are, by design, not separated from the baseline at ≤ 20,000 shots.** They are
  predicted equal to it. Seeing a 20% change needs about 15,000 shots per arm, about 8 s each.
- **A2s_spec may not be like for like.** If q74's readout dephases q75 by ≥ 5% per step, the
  sham itself loses ≥ 20% of the witness and becomes a partial A2. Measure it; don't assume it.
- **The destructive controls change the operating point, not the noise.** P(1) falls from 0.58
  to 0.35–0.39, and the process moves 0.07–0.11 nats/symbol from Phase 6's. q59's readout and
  reset are unchanged, but they now act on a different mix of 0s and 1s. The added
  operations' own noise moves nothing beyond 1e-3.
- **The small shot counts rely on a Gaussian SE.** #18 calibrated the estimator at 20,000
  shots (its z is conservative at 200). Use at least about 4,000 shots per circuit.

## 5. Interpretation

| control | outcome | what follows | what does **not** follow |
|---|---|---|---|
| A1 | vanishes (\|w\| < 5σ) | q75's state, or the operations on it, is in the causal path of the rank > 2 statistics | that q75 alone is the memory: a reset also drives q75's readout resonator and coupler, and every 2-state alternative (H1, H2) also predicts 0 |
| A1 | persists | the hardware has a memory with ≥ 3 effective states that survives destruction of q75 | which path it is (q59, leakage, spectators, drift with ≥ 3 regimes) |
| A1 | partial | q75 carries part; something else carries the rest, or the reset is imperfect (leakage) | a fraction attributable to q75 without A3's agreement |
| A2 | vanishes | with q75 reduced to a classical bit the output is consistent with a 2-state process: the extra rank needs q75's coherence, or its combination with the classical loop | that the coherence is "quantum memory" in any stronger sense, or that no other path exists below 9e-5 (§3) |
| A2 | persists | rank > 2 survives a classical q75: a memory outside q75's coherence (H_other). The record diagnostic I(y_t; y_{t−1} \| m_t) ≫ 1e-6 would locate it outside q75 | that q75 is irrelevant: it may still contribute a classical bit |
| A2 | partial | consistent with incomplete dephasing (w(0) reaches 7% only if > 50% of the coherence survives) or a weak second path | a coherence fraction, since w is nonlinear in it |
| A3 | vanishes, ≈ −7e-5 | A1's conclusion, reached with unitary gates only: no reset or readout on q75 | the same caveats as A1. A3 and A1 agreeing rules out "the reset itself injects the effect" |
| A3 | persists | as for A1 persisting; or the instance set is degenerate (check its exact prediction) | – |
| A2s_spec | equals baseline | a readout pulse and record beside q75 do not by themselves remove the witness, so **A2 vanishing while A2s_spec persists is the strongest support for coherence-borne memory in q75 that this design gives** | – |
| A2s_spec | reduced | q74's readout dephases q75 (5% per step costs 20%): crosstalk is on the causal path, and A2 and the shams must be read with it | that the q75 memory hypothesis is wrong |
| A1s, A2s, A3s | equal baseline / differ | timing and idle effects are, or are not, negligible. The model predicts equality, because the delay sits in an idle window | – |

In every row, keep three claims apart: statistics of the output process (what is measured),
effective memory of the hardware (rank ≥ 3), and memory attributable to q75 (only from A2
against A2s_spec, and A1 against A3).

## 6. Recommendation for Stage 3

One job, with the circuits interleaved by the sampler. Shots are the config's `[hardware]`
plan; the costs come from `--plan`.

1. **baseline, 15,000 shots, and A2, 10,000 (8.4 s + 5.6 s).** The decisive pair. A2 at
   10,000 shots resolves a persisting witness of 9% of the device's at 5σ.
2. **A2s_spec, 15,000 (8.4 s).** The sham that validates A2 and measures readout crosstalk
   onto q75. Against the baseline it resolves a 20% change.
3. **A1, 4,000 (2.2 s).** The cheapest causal-path check.
4. **A3, 64 instances × 64 shots (2.3 s).** Destroys the memory without a reset or readout,
   which separates "q75's state" from "the operations on q75".
5. Optional: the pure-delay shams A1s, A2s and A3s at 4,000 shots each (2.2 s each). The
   model predicts them identical to the baseline.

Total: 28.9 QPU s (36 s with the optional shams), against about 584 s remaining. Stage 3
should re-read the live target, regenerate the delays, check the QASM hashes in
`circuits.json` and submit the ISA circuits without re-transpiling. A level-1 transpile
would merge A3's Paulis into q75's gates.

**Limits.** Leakage is not modelled; that is Stage 2B's. The spectator q74 is not simulated.
Its effect enters only as the dephasing parameter. The operation's timing is modelled as
overlapping q59's readout and reset, as `estimate_duration` schedules it. The serial
sensitivity bounds the error if hardware serialises it.
