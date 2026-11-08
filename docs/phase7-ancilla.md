# Phase 7 Stage 2B: output-ancilla (q59) reset, leakage and residual memory

Issue #21 (child of #17). Local simulation only: no QPU, no IBM service call. Reference numbers
are #18's (`docs/phase7-audit.md`). The episode-free device run (17,300 shots, [5300, 8000)
removed) gives w(0) = −0.00585 and P(1) = 0.5774. The calibrated model gives −0.00690 and 0.5820.

```
uv run python experiments/phase7_ancilla.py --config experiments/configs/phase7_ancilla.toml            # 10 min, 2 workers
uv run --group hardware python experiments/phase7_ancilla.py --config experiments/configs/phase7_ancilla.toml --export   # QASM check
uv run python experiments/phase7_ancilla.py --config experiments/configs/phase7_ancilla.toml --plan     # unsubmitted Stage 3 job
```

- **Code:** `src/qmem/ancilla.py`, numpy only.
- **Tests:** `tests/test_ancilla.py`.
- **Outputs:** `results/summaries/phase7_ancilla/`:
  - `fits.csv`, `scans.csv`, `a2_leak.csv`, `controls.csv`, `shots.csv`, `summary.json`;
  - `qasm/` for B1–B4, and `circuits.json` (FakeKingston check, durations).

## Verdict

- **(i) Ancilla residue at physically plausible parameters cannot account for the block-KL
  excess.** The snapshot's own reset and readout numbers remove 14% of the L = 4 excess.
  Fits that remove up to 72% need leakage of 2–4% per readout, about 10× the literature, and
  still leave KL₄ at 4× its null.
- **(ii) Ancilla residue cannot produce or remove ≥ 10% of the episode-free witness at
  plausible parameters.**
  - Reset residue up to the readout-error ceiling changes w(0) by at most 2%.
  - Leakage at a typical 1e-3 per readout changes it by 6%.
  - Only leakage near the extreme 5e-3 reaches 25%, but it raises P(1) by 0.022, while the
    device sits 0.005 *below* the calibration.
  - On its own, q59's memory acts only by *reducing* |w|. In a memoryless circuit it gives
    rank ≤ 2 (tested).

## 1. The models

**How `reset` works on Heron.** IBM implements `reset` as a measurement followed by an X
conditioned on its outcome (Qiskit `ResetAfterMeasureSimplification`; IBM's mid-circuit
measurement blog). FakeKingston's reset lasts 2.312 µs, which is its readout (2.28 µs) plus
one `x` (32 ns). The snapshot gives 2.212 µs and 2.18 µs. q75 idles through the whole of it.
The failure modes follow from that implementation:

- from |0⟩ the reset leaves |1⟩ when its readout says 1 (e₀ ≈ P(1|0) = 0.00098; the snapshot's
  `init_error` is 0.00098);
- from |1⟩ it leaves |1⟩ when the readout says 0 *without* a decay (e₁ ≤ P(0|1) = 0.0068;
  a decay during the readout ends in |0⟩, which is a success);
- a leaked |2⟩ reads mostly as 1, because the discriminator is binary (Fernandes et al.,
  arXiv:2309.11303). The 0–1 X pulse then leaves it in |2⟩, so a reset does not remove leakage.
  That last step is our model; IBM publishes no statement on it.

**Exact instruments on q75 ⊗ q59** (q59 a qutrit) are built from `step.qasm`'s native gates:

- **(a)** state-dependent reset failure e₀, e₁. This reuses Stage 2A's `ablate_hw.reset_kraus`,
  lifted to a qutrit.
- **(b)** leakage |1⟩ → |2⟩ at rate ℓ per readout, seepage |2⟩ → |1⟩ at rate s per step, and
  P(record 1 | |2⟩) = r₂. A CZ gives |1⟩₇₅|2⟩₅₉ a free phase. Gates act as the identity on |2⟩.
- **(c)** non-QND flips after the record.
- **(d)** asymmetric assignment P(1|0), P(0|1).

With every default the simulator is Phase 6's calibrated model. It matches Stage 2A's joint
pieces to 9e-16, and #18's 1-qubit instrument to 2e-14 in g and 1e-13 in NLL. After every
step q59 is diagonal, so the process restricts exactly to a 12-dimensional subspace.

**Analytic checks.** `tests/test_ancilla.py`:

- The defaults (ε = 0) give the calibrated instrument exactly.
- "Never reset" (e₀ = 0, e₁ = 1) in a memoryless circuit gives the predicted Markov chain
  P(y′|y) = |⟨y′|U|y⟩|² (a 2-state HMM when the readout has errors), and w = 0.
- Leakage and every other model conserve trace and are completely positive (Choi spectrum).
- Leakage in a memoryless circuit equals a brute-force 3-state HMM.
- **(c) is exactly degenerate with (a).** q59 is always reset before reuse, so a flip between
  readout and reset changes only (e₀, e₁). Its scan is flat to 1e-16.
- **Double reset** squares the state dependence to (e₁ − e₀)².
- Post-selected windows match enumeration of all 4⁶ sequences.

## 2. Fit to the device

The free parameters of each family are fitted by maximum likelihood (Nelder–Mead) on the even
episode-free shots and scored on the odd ones (`fits.csv`). A random split agrees to within
the SEs. CIs are Wald CIs from the Hessian.

ΔNLL is held-out NLL minus the calibrated model's, in nats/symbol, paired over the 8,650
test shots. KL₄ null is 0.000019; KL₈ null is 0.00031.

| model | fitted values (95% CI) | ΔNLL | vs 3-state HMM | KL₄ | KL₈ | w(0) | P(1) |
|---|---|---|---|---|---|---|---|
| calibrated | – | 0 | +0.00038 | 0.000249 | 0.00073 | −0.00690 | 0.5820 |
| snapshot numbers (no fit) | e₀ = P(1\|0) = 0.00098, e₁ = P(0\|1) = 0.0068 | −0.000039(10) | +0.00035 | 0.000217 | 0.00069 | −0.00707 | 0.5788 |
| (a) reset | e₀ = 0.0057 [0.0035, 0.0092], e₁ = 0.0183 [0.0159, 0.0210] | −0.000130(21) | +0.00025 | 0.000206 | 0.00062 | −0.00644 | 0.5787 |
| (d) readout | P(1\|0) = 0.0076, P(0\|1) = 0.0125 | −0.000062(13) | +0.00032 | 0.000244 | 0.00069 | −0.00669 | 0.5785 |
| (a)+(d) | e₀ = 0.0089, e₁ = 0.0209, readout → 0 | −0.000139(21) | +0.00025 | 0.000197 | 0.00061 | −0.00649 | 0.5783 |
| (b) leakage | ℓ = 0.040 [0.031, 0.051], s = 0.54 | −0.000119(20) | +0.00027 | 0.000232 | 0.00064 | −0.00592 | 0.5860 |
| (b)+(d) | ℓ = 0.040, s = 0.41, P(0\|1) = 0.021 | −0.000258(31) | +0.00013 | 0.000106 | 0.00050 | −0.00567 | 0.5798 |
| (a)+(b)+(d) | e₁ = 0.016, ℓ = 0.024, s = 0.34, P(0\|1) = 0.016 | −0.000277(33) | +0.00011 | 0.000083 | 0.00047 | −0.00579 | 0.5795 |
| *comparator:* q75 Z offset + (d) | 0.042 rad/step [0.036, 0.047] (1.4 kHz), P(0\|1) = 0.014 | −0.000232(31) | +0.00015 | 0.000085 | 0.00050 | −0.00682 | 0.5769 |
| 3-state HMM | – | −0.000384(46) | 0 | 0.000029 | 0.00040 | −0.00620 | 0.5769 |
| 4-state HMM | – | −0.000404(44) | −0.00002 | 0.000034 | 0.00040 | −0.00625 | 0.5770 |

**Plausibility.**

- **Reset.** The fitted e₁ = 0.018 is 2.7× P(0|1), which is its ceiling, and e₀ = 0.006 is 6×
  the snapshot's `init_error`.
- **Leakage.** The fitted ℓ = 0.02–0.04 per readout is 5–10× the largest published rate. That
  rate, about 5e-3 per cycle on Google's measure qubits (Miao et al., arXiv:2211.04728), is
  itself an extreme. IBM leakage-RB data show 4e-5 per single-qubit Clifford (arXiv:2309.11303).
  The fitted seepage of 0.3–0.5 per step implies T1(|2⟩) of 10–15 µs, against the T1/2 ≈
  125 µs we expect.
- **Readout.** P(0|1) = 0.014–0.021 is 2–3× the snapshot.

No ancilla family closes the 0.0005 gap: the best ones reach 0.00028, and the 3-state HMM is
still 0.0001 better. The non-ancilla comparator, a q75 frequency offset of 1.4 kHz (plausible
drift, but not this stage's to claim), does about as well with plausible numbers.

**Witness added or removed** (interleaved fits):

- **Reset residue.** A memoryless "twin" keeps the same average reset error but drops its
  state dependence. It gives w(0) = −0.00638, against −0.00644 for (a). The memory part is
  6e-5, 1% of the device's witness. The 8% shift from the calibration comes from the
  *average* reset error, not from q59 memory.
- **Leakage.** In the (b) fit, leakage moves w(0) from −0.00690 to −0.00592, which removes 17%
  of the witness, but only at its implausible ℓ.

**Stage 2D's "sticky" readout.** #23 fits ε̂ = 0.027 on the same even/odd split. It gains
0.00026 ± 0.00003 nats/symbol and gives w(0) = −0.00612. The physical reset residual (a)
gains half of that (0.00013 ± 0.00002) and gives w(0) = −0.00644, even at implausible e₀
and e₁. A plausible q59 reset residual therefore does not reproduce the stickiness.
**The readout chain, or q75, is the likelier home.**

## 3. Scans at plausible parameters (`scans.csv`)

Δw(0) is the change from the calibrated model as a fraction of the device's |w(0)|. A
positive value means a weaker witness.

| parameter | value | Δw(0) | ΔP(1) |
|---|---|---|---|
| e₁ | 0 / 0.0068 (ceiling) / 0.02 / 0.05 | −2.0% / +1.4% / +7.7% / +20% | +0.0008 / −0.0006 / −0.0034 / −0.0095 |
| e₀ | 0.0005 / 0.005 / 0.02 | −1.7% / +0.5% / +7.8% | +0.0005 / −0.0001 / −0.0022 |
| ℓ (s = 0.038) | 1e-4 / 1e-3 / 5e-3 / 1e-2 | +0.6% / +5.8% / +26% / +44% | +0.0004 / +0.0045 / +0.022 / +0.041 |
| r₂ (ℓ = 1e-3) | 1 / 0.5 | +5.8% / +1.2% | +0.0045 / −0.0010 |
| CZ phase of \|1,2⟩ | 0 / π | identical to 1e-5 | – |
| non-QND flip | 0.002 / 0.01 | 0 (degenerate with (a)) | 0 |

**The coordinator's A2 question** (`a2_leak.csv`). Here q75 is measured every step and leaks
at ℓ₇₅ per measurement, with seepage 0.03 (T1/2) or 0.2, and a CZ phase of 0 or π. The
results:

| ℓ₇₅ | 1e-4 | 1e-3 | 5e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|
| A2 w(0) | 5.8e-8 | 5.8e-7 | 2.8e-6 | 5.4e-6 | 1.4e-5 |

These are the worst case over seepage and phase, at most 0.23% of the device's witness. A
3-level classical q75 does make w ≠ 0, as #17 expected, but it is far too small to matter. If
A2's witness persists, q75 leakage does not explain it.

## 4. Controls B1–B4

All five circuits are edits of Phase 6's `circuit.qasm` and are native on FakeKingston
(`circuits.json`):

| circuit | edit | FakeKingston duration |
|---|---|---|
| B1 | `reset; reset` | 466.7 µs |
| B2 | `reset` then `f[t] = measure $59`, 127 clbits | 464.7 µs |
| B3 | odd steps use q74, which is coupled to q75 (#22) | 161.7 µs |
| B4r | `delay[553dt]` after each reset (matches B1) | 460.4 µs |
| B4m | `delay[545dt]` after each reset (matches B2) | 458.4 µs |
| baseline | – | 321.0 µs |

B3 is shorter because ASAP runs each ancilla's reset during the other's step. It is modelled
that way, with q75 not idling through a reset. q74's calibration is FakeKingston's stale one.

**Hypotheses:**

- **null:** calibrated;
- **fit:** (a)+(d) and the full model, from §2;
- **max_reset:** e₀ = 0.005, e₁ = 0;
- **max_reset_rev:** e₀ = 0.0005, e₁ = 0.0068;
- **max_leak:** ℓ = 5e-3, s = 0.02.

**Shots and cost.** Each control is compared with its matched reference: B1 with B4r, B2 with
B4m, B3 and B4 with the baseline. Shots are per arm, at 5σ, using the per-shot SD of w(0)
(0.0159 for the baseline) × 1.4 (#18). QPU seconds use Phase 6's cost model. The Δ and %
columns are the predicted change of the control against its reference, minus the null's
change (`shots.csv`).

| control | null w(0) (noiseless / calibrated) | largest Δ at plausible parameters | shots per arm, QPU s | under the fits |
|---|---|---|---|---|
| B1 vs B4r | −0.00883 / −0.00644 | max_leak −4.0e-4 (6.7%); resets ≤ 2.9% | 1.5e5, 211 s | (a)+(d) 5.2%: 2.5e5 |
| B2 vs B4m, no post-selection | −0.00883 / −0.00644 | 0 under every reset model (1e-17); leakage 1e-5 | ∞ | – |
| B2 post-selected (all flags 0) vs B4m | −0.00666; acceptance 0.61 | max_leak −1.9e-3 (32%) | 1.3e4, 20 s | (a)+(d) 15%: 7.9e4, 112 s |
| B3 vs baseline | −0.00712 | max_leak −1.8e-3 (30%); resets ≤ 2% | 8.1e3, 9.9 s | full 5.5%: 2.4e5 |
| B4 vs baseline (null) | added idle: +7.9% | – | 1.1e5, 143 s | – |

**B2's flag is the cheap and direct test** (one arm, `shots.csv`). It records q59's state
after the reset.

- **Flag rate.** The null predicts P(f = 1) = 0.0078; the calibrated model's reset error is
  its readout error. The (a)+(d) fit predicts 0.016, max_reset 0.006 and max_leak 0.080.
- **History contrast.** P(f = 1 | y_{t−1} = 1) − P(f = 1 | y_{t−1} = 0) is 0 under the null.
  It is +0.012 for the fit, +0.0062 for max_reset_rev, −0.0049 for max_reset, and +0.12 for
  max_leak.
- **Shots.** At 5σ, 50–1,000 shots resolve each of these (2.0–2.7 QPU s, mostly job overhead).
  15,000 shots give a contrast SE of about 1.8e-4.
- **Residue against record.** Readout persistence (#23) would also produce a contrast, but a
  physical residue changes the *same step's* output. The predicted P(y_t = 1 | f_t = 0 / 1)
  is 0.581 / 0.643 under the null, 0.576 / 0.730 under the fit, and 0.581 / 0.968 under
  max_leak. A contrast with no P(y | f) signature points at the record, not at q59.

**What the backend cannot do.** IBM's discriminator is binary, and there is no
level-resolved readout. SamplerV2's `meas_type = "kerneled"` returns IQ points for every
register, mid-circuit ones included. We found no documented plan restriction, so an offline
|2⟩ discrimination is possible in principle (untested here). Dynamic circuits run on all
backends, the Open plan included. Reset or measure inside a conditional is not supported,
so B2 cannot do real-time rejection: post-selection is offline.

## 5. Recommendation for Stage 3 (ranked)

1. **B2, 15,000 shots (10.4 QPU s).** It measures q59's post-reset |1⟩ population and its
   dependence on the previous outcome directly, at a 5σ contrast of about 0.001. That settles
   H_ancilla against the null for any residue above about 15% of the readout-error ceiling.
   It separates a physical residue from readout persistence through P(y_t | f_t). Its
   post-selected analysis tests extreme leakage (a 32% change). This is the only control whose
   ancilla signal is affordable.
2. **B4m, 10,000 shots (7.0 s).** The delay-matched reference for B2's post-selected witness,
   in the same job. It also measures the added-idle effect predicted under the null (7.9%),
   at about 1.5σ.
3. **B3 (alternating q59/q74), 8,100 shots per arm (about 10 s), only if B2's flags look
   like leakage.** Leakage would show as a flag rate well above 0.01 with P(y = 1 | f = 1)
   near 1. B3 is the one witness-level control that responds to leakage. Re-rank q74 on live
   calibration first (#22).
4. **B1 is not recommended.** Its predicted changes (≤ 7%) need about 1.5e5 shots per arm.

The baseline is shared with Stage 2A's plan (#20). Total: 19.4 QPU s in one job
(`--plan`; `[hardware] submit = false`).

**Limits.**

- The leaked level is modelled simply: gates act as the identity on |2⟩, and the CZ phase is a
  parameter, which turns out not to matter.
- The fits keep q75's calibrated parameters, so the ancilla parameters may absorb q75
  miscalibration. The q75 comparator suggests they do.
- The Wald CIs assume independent shots, which holds once the episode is removed. The Hessian
  is singular where a readout parameter hits 0, and those CIs are reported as NaN.
- Step 0 is modelled with a reset, as #18's instrument is; the burn-in of 16 makes this
  irrelevant for w.
- Keep three claims apart: the output statistics, the hardware's effective memory, and memory
  attributable to q75. Nothing here supports more than "q59 is not the source".
