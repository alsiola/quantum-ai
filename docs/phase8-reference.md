# Phase 8 Stage 3: can q75's step channel be certified non-entanglement-breaking? (#33)

Local theory, simulation and circuits only. No QPU and no IBM service call. The hardware plan
at the end is **unsubmitted**. This note is the feasibility study for Claim 4 of
[phase8-claim-framework.md](phase8-claim-framework.md): "the relevant channel can preserve a
quantum resource".

- Code: `qmem.reference` (numpy; its circuit builders import Qiskit lazily),
  `experiments/phase8_reference.py`, config `experiments/configs/phase8_reference.toml`.
- Outputs: `results/summaries/phase8_reference/`:
  - `channel.csv`, `branches.csv`, `zero_branch.csv`, `power.csv`, `sensitivity.csv`,
    `ppt_witness.csv`, `references.csv`, `budget.csv`, `summary.json`;
  - `qasm/`: 112 circuits, with `manifest.json` (SHA-256, FakeKingston durations, and each
    circuit's check against the numpy model).
- Tests: `tests/test_reference.py`.

```sh
uv run python experiments/phase8_reference.py --config experiments/configs/phase8_reference.toml        # 4 s
uv run --group hardware python experiments/phase8_reference.py --config experiments/configs/phase8_reference.toml --export-qasm   # once, 6 s
```

## Verdict

**Feasible at n = 1 only, under trusted state preparation and measurement (SPAM).** Under the
Stage 4c calibrated instrument, the unconditional step channel on q75 is **not
entanglement-breaking (EB) after one step**:
- its Choi state's Bell fidelity is F_e = 0.580 against the predeclared target;
- the trusted prepare-and-measure (P&M) score is 0.712 against the EB bound of 0.668
  (2/3 plus the leakage allowance);
- 80% power needs 521 shots, and the planned 12,000 give power ≈ 1.

The Bell route, with the reference on q79 (chosen by the predeclared rule), and Rosset et al.'s
measurement-device-independent (MDI) game also pass at n = 1:
- Bell route: power 0.999 at 6,000 shots;
- MDI game: power 0.98 at 16,000 shots.

**From n = 2 every unconditional route is infeasible, and noise is not the reason.** Even
without noise, the trained instrument's unconditional channel has:
- F_e = 0.400 at n = 2: entangled (negativity 0.018), but below every fidelity witness;
- a PPT, and therefore EB, Choi state from n = 3.

Each step's emission on q59 extracts so much information about q75 that, averaged over the
record, q75's qubit does not survive three steps.

A predeclared conditional route works further (*ambiguous* by the issue's criteria):
- on the record y = 0 (probability 0.61), F_e = 0.844 at n = 1;
- on the record 00, F_e = 0.814 at n = 2;
- the all-zero branch keeps F_e = 0.74 at n = 4 (probability 0.069).

**What a success would show, and what it would not.** One application of the trained step
instrument, averaged over its output, transmits a qubit stored in q75. This is a capability of
the channel under trusted SPAM (Claim 4 at n = 1). It does **not** show that q75 holds quantum
information across the lags the output witness uses. The witness's memory runs through the
outcome-conditioned branches. The unconditional channel is EB from three steps by
construction, and that does not contradict a rank > 2 output process (§1).

**Recommendation.**
- **Primary:** the P&M test at n = 1. It is the strongest feasible option: the largest
  power per shot, no reference qubit, and no isolation question.
- **Complement:** the MDI game at n = 1, if the owner wants the measurement untrusted.
- Together they cost 25.1 s upper bound, within a proposed **30 s hard cap**. 531 s of the Open
  plan remain after #27.
- The Bell route is costed but not recommended. It needs two SWAPs through q74, and its
  reference is chosen on stale FakeKingston calibration.

## 1. The channel against n

`channel.csv`. Each step is the Stage 4c pieces (`results/summaries/phase7_hw_4c/pieces_live.npz`,
#27's analysis model) on the joint (q75, q59) memory. q59 starts each shot in |0⟩, and the
channel is traced over q59 at the end, so q59's reset residue is exact. The predeclared targets
V_n come from the same model (§2.2).

| n | F_e, fixed target (live) | max F_e (live) | negativity (live) | F_e (noiseless) | negativity (noiseless) |
|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 |
| 1 | **0.580** | 0.580 | 0.132 | 0.592 | 0.145 |
| 2 | 0.391 | 0.391 | 0.0080 | 0.400 | 0.018 |
| 3 | 0.308 | 0.308 | 0 (PPT) | 0.314 | 0 (PPT) |
| 4 | 0.276 | 0.276 | 0 | 0.279 | 0 |
| 8 | 0.251 | 0.251 | 0 | 0.251 | 0 |
| A2 step (negative control) | 0.372 | 0.372 | 0 | 0.376 | 0 |

- **Noise costs almost nothing at n = 1:** 0.592 noiseless, 0.580 live, 0.567 for the
  deficit-matched model (§5).
- **The decay is the instrument's own.** The trained step is a weak measurement of q75: q59's
  outcome carries information about q75's state, so the average over outcomes decoheres q75.
  - The branch y = 0 is close to a unitary (F_e,max = 0.868).
  - The branch y = 1 is close to a measure-and-prepare map (F_e,max = 0.519).
  - Their mixture is entanglement-preserving for one step, barely entangled after two, and
    separable after three.
- **For the Hankel witness** (#18, #27), this matters as follows. The output process has rank
  > 2 because the conditional maps M_y do not commute, not because the averaged channel keeps
  a qubit. That a channel is EB says nothing about the classical dimension of the process its
  branches generate. "Quantum memory of a process" and "a channel that transmits a stored
  qubit" are different properties [Giarmatzi2021; Rosset2018].
- `tests/test_reference.py::test_trained_instrument_breaks_entanglement_from_three_steps`
  checks the PPT point on the committed noiseless pieces.

## 2. Protocols and their nulls

### 2.1 Three tests, three trust models

| route | statistic | EB bound (null) | trusts | needs |
|---|---|---|---|---|
| **P&M** (primary) | mean over 6 axis inputs × 2 labels of P(measured V ψ_k) | F_avg ≤ 2/3 | preparation and measurement of q75 | q75 and q59 only |
| **Bell** | F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩)/4 after V_n† on q75 | F ≤ 1/2 | the end measurements; isolation of the reference | a reference qubit, 2 SWAPs |
| **MDI game** [Rosset2018] | Σ ω_xy P(Φ⁺ \| x, y), tetrahedral ξ_x on q75, ψ_y on q74 after the steps | payoff ≤ 0 | the input states only | q74 for the trusted input and Bell measurement |

**Why each null holds.**
- **EB channels.** Every EB channel is measure-and-prepare. Its Choi state
  J = (id ⊗ Φ)(Φ⁺) is separable [Horodecki2003EB; Rosset2018 §I]. A separable two-qubit state
  has fidelity ≤ 1/2 with every maximally entangled state [Horodecki1999Teleport].
- **P&M.** The six axis states form a 2-design, so F_avg = (2 F_e + 1)/3 exactly. That turns
  1/2 into 2/3.
- **Branches.** For a branch, a trace-decreasing CP map, the same bounds hold for the pooled
  ratio: successes in the branch over shots in the branch, with the inputs equally allocated.
  An EB CP map's Choi operator is still separable.
- **MDI game.** Rosset et al. (Prop. 2, Eqs. 13–15) decompose the witness
  W = ττ† − 1/2 on ξ_xᵀ ⊗ ψ_yᵀ, which gives p(Φ⁺ | x, y) = Tr[J (ξ_xᵀ ⊗ ψ_yᵀ)]. The payoff is
  then tr(W J) = F_e − 1/2, and it is ≤ 0 for an EB channel *whatever POVM element the Bell
  measurement actually implements*. That property is tested on random POVM elements.
- **Tests.** Random EB channels and branches are checked against all three bounds, with
  brute-force maxima over targets (`test_eb_*`, `test_mdi_*`). The 2-design identity is tested
  on random channels.

**Rosset et al.'s critique of the Bell route applies here.** Their §I lists its costs:
- the reference must be kept intact;
- a Bell or fidelity test is sufficient but not necessary;
- it is fragile to loss.

Their framework drops the reference entirely: trusted inputs are the minimal assumption for a
channel test, since the output lies in the future of the input. The P&M test is the
trusted-measurement special case of that framework.

### 2.2 Targets fixed before any data

- The target is (1 ⊗ V_n)|Φ⁺⟩. V_n is the rotation that maximises the *model's* Choi fidelity,
  computed in closed form: F_max = (1 + s₁ + s₂ + sign(det A) s₃)/4 with A = diag(1, −1, 1)·T.
- V_n is written to `summary.json` (`targets_V`) and compiled into every measurement rotation
  in the QASM. It is never refitted to data.
- At n = 1 the fixed target attains the live maximum to 1e-15. Under the dephasing and reset
  models it costs less than 1e-4 in F_e. Under a phase offset it costs 0.019 at 0.44 rad per
  step and 0.090 at 1 rad per step (§5).
- **Bounding the maximisation.** Even a target maximised on the test data keeps the null:
  F_max ≤ 1/2 for every separable state. Maximising on the data would only bias the estimate
  upward. Here no maximisation is done.

### 2.3 Design details

- **P&M.**
  - Inputs: the six axis states. ±z needs 0 or 2 pulses; ±x and ±y need one sx and virtual
    rz's.
  - After n steps, a rotation maps V_n ψ_k to |b⟩, then q75 is measured.
  - **Balanced labels:** b = 0 and b = 1 for every input (12 circuits per arm). Each input's
    success then needs a 0 half the time and a 1 half the time. Asymmetric readout errors then
    lower the EB bound to 2/3 − (e₀ + e₁)/6, so they are conservative (tested).
- **Bell route.**
  - A Bell pair is made on (q74, q75). SWAPs then move q74's half along 75–74–73–79, so the
    reference ends on q79 and q74 and q73 return to |0⟩, as in Phase 6.
  - Settings XX, YY, ZZ, each with and without an X on both qubits before readout (a readout
    twirl). Each correlator therefore shrinks by (1 − 2e_ref)(1 − 2e_mem), which is
    conservative (tested).
- **MDI game.**
  - Rosset et al.'s tetrahedral inputs (their §V.A) on q75 (ξ_x) and, after the steps, on q74
    (ψ_y).
  - Bell measurement CX(q75 → q74), then H(q75); Φ⁺ reads 00.
  - During the steps q74 idles in |0⟩, exactly as in Phase 6.
- **Step counts.**
  - Circuits are built for n ∈ {0, 1, 2, 3}.
  - n = 4 and n = 8 are simulated only. Their predicted scores are 0.516 and 0.501 against a
    bound of about 0.67, so the circuits would measure nothing the model doesn't already
    exclude.
  - n = 3 is kept as a model check: the noiseless channel is EB there, so its score must fall
    below 2/3.

## 3. Conditioning on q59's record (secondary)

**The predeclared rules.**
- The unconditional channel is the primary test.
- The secondary family is:
  - the unconditional n = 2 test;
  - every record of length 1 (two branches) and length 2 (four branches) at n = 1 and n = 2.

  That is 7 tests under Holm at α = 0.05, so the first step is at α = 0.0071.
- Branch analyses use the *same* circuits and the *unconditional* targets V_n. No circuit is
  added and nothing is fitted per branch.
- A branch is tested only if it collects ≥ 500 shots.
- Every branch is reported, whichever way it comes out. There is no success-only
  post-selection.

| n | record | P(record) | F_e (target V_n) | P&M score | power at planned shots (Holm) |
|---|---|---|---|---|---|
| 1 | 0 | 0.612 | **0.844** | 0.881 | 1.00 (41 shots for 80%) |
| 1 | 1 | 0.388 | 0.164 | 0.445 | 0 |
| 2 | 00 | 0.299 | **0.814** | 0.862 | 1.00 |
| 2 | 01 | 0.313 | 0.200 | 0.468 | 0 |
| 2 | 10 | 0.139 | 0.193 | 0.463 | 0 |
| 2 | 11 | 0.249 | 0.233 | 0.489 | 0 |

The all-zero branch alone (`zero_branch.csv`, simulation only) is non-monotone, which is the
instrument's period-3 structure:

| n | P(0ⁿ) | F_e (target V_n) | max F_e |
|---|---|---|---|
| 1 | 0.612 | 0.844 | 0.868 |
| 2 | 0.299 | 0.814 | 0.836 |
| 3 | 0.117 | 0.857 | 0.875 |
| 4 | 0.069 | 0.742 | 0.793 |
| 8 | 0.0039 | 0.621 | 0.700 |

**What a branch result means.** A branch result certifies that the instrument *element* M_0 (or
M_00) is not EB. That is a property of the instrument, not of the channel. It is the
"conditioning" case the issue classes as **ambiguous**. Taken alone, it should not be reported
as Claim 4.

## 4. Controls

- **Positive control (n = 0).** Prepare and measure with no step. It is predicted at 0.981 for
  P&M, 0.950 for Bell and +0.466 for the MDI game. These numbers are the SPAM characterisation.
  **Gate:** if the P&M score at n = 0 is below 0.90, SPAM is not as modelled. Stop and report;
  don't interpret n ≥ 1.
- **EB negative control.** Step 1 with q75 measured mid-step, which is A2's operation from #20,
  in the same position. The channel is then EB by construction. It is predicted at 0.578
  against the bound of 0.668, and it falsely passes in 0 of 4,000 Monte Carlo draws.
  **Gate:** if its lower bound exceeds the threshold, the trust assumptions have failed: a SPAM
  or leakage loophole, or an implementation error. Report the n = 1 result as invalid.
- **Leakage treated as loss.**
  - The binary discriminator cannot flag a leaked shot, so Rosset et al.'s erasure argument
    (their Prop. 3) does not apply as is.
  - Every leaked shot is therefore scored as if it had the most favourable outcome. This raises
    the bounds to 2/3 + ℓ/3 (P&M) and 1/2 + ℓ/2 (Bell).
  - ℓ = 10⁻³ per CZ × the number of CZs, about 10× IBM's leakage-RB rates (#21). At n = 1
    that is 0.6677 and 0.505; both are tested.
- **Crosstalk and the q59-only control (Bell route).**
  - The reference qubit (q79) is 3 hops from q75 and 4 from q59. After the SWAPs, its only
    path to either runs through q73 and q74, both in |0⟩.
  - A diagonal spectator in |0⟩ gives no conditional phase at first order (#22).
  - The control: q79 is prepared in |0⟩ or |1⟩ and q75 runs Phase 6's 8 steps,
    10,000 shots each. Under the null, q59's record is identical between the two.
  - The design detects |ΔP(1)| ≥ 0.0068 at 95%. This bounds the information about the
    reference that leaks into the output.
  - The P&M and MDI routes have no reference to leak.
- **Spectators of q75.** In every route q74 is in |0⟩ during the steps, as in Phase 6's
  baseline. The instrument's environment is therefore Phase 6's own.

## 5. Simulation, sensitivity and power

`sensitivity.csv`, `power.csv`. One-sided α = 0.05 for the primary test. The SE is the
stratified binomial one. Drift within a circuit only makes it conservative: a mixture of
Bernoullis has variance ≤ the Bernoulli at its mean. Power is computed from the normal
approximation and checked by Monte Carlo with 4,000 draws.

| model at n = 1 | F_e (fixed V₁) | P&M score | P&M shots for 80% | P&M power at 12,000 |
|---|---|---|---|---|
| Stage 4c live (primary prediction) | 0.580 | 0.712 | 521 | 1.00 (MC 1.000) |
| noiseless | 0.592 | 0.719 | 370 | 1.00 |
| Phase 6 snapshot | 0.580 | 0.712 | 516 | 1.00 |
| q59 reset fit (a) of #21 | 0.580 | 0.712 | 521 | 1.00 |
| **deficit-matched**: extra q75 dephasing p = 0.067 per step, making w(0) 0.747 of H_q75's (#29's baseline ratio) | 0.567 | 0.703 | 813 | 1.00 (MC 1.000) |
| extra dephasing p = 0.2 per step | 0.541 | 0.687 | 2,986 | 1.00 |
| extra dephasing p = 0.3 | 0.521 | 0.674 | 27,182 | 0.50 |
| extra dephasing p = 0.5 | 0.482 | 0.649 | — | 0 |
| q75 phase offset 0.042 rad/step (#21's comparator) | 0.580 | 0.712 | 524 | 1.00 |
| phase offset 0.44 rad (a 16 kHz Stark shift over q59's readout and reset, #22) | 0.561 | 0.700 | 1,013 | 1.00 |
| phase offset 1.0 rad/step | 0.490 | 0.654 | — | 0 |

**Every route at n = 1** (Stage 4c model; deficit-matched in brackets):

| route | effect above the bound | shots for 80% / 90% | planned shots | power |
|---|---|---|---|---|
| P&M | 0.044 (0.036) | 521 / 722 (813 / 1,126) | 12,000 | 1.00 (1.00) |
| Bell, reference q79 | 0.041 (0.030) | 1,701 / 2,356 (3,305 / 4,578) | 6,000 | 0.999 (0.957) |
| MDI game | 0.064 (0.051) | 7,004 / 9,702 (10,917 / 15,121) | 16,000 | 0.983 (0.914) |

- **The n at which it becomes infeasible is n = 2,** for every route and every model. At n = 2
  the effect is −0.078 (P&M), −0.132 (Bell) and −0.118 (MDI). The Choi state is still entangled
  (negativity 0.0080), but only a PPT witness could see it.
- **The predeclared PPT witness at n = 2** (`ppt_witness.csv`):
  - Its ideal value is λ_min = −0.0080 (live) and −0.018 (noiseless).
  - Measured with 18 Pauli P&M settings, SPAM moves it to −0.0016, and the leakage allowance
    alone (+0.0098) exceeds it.
  - Even noiselessly it needs 3.2 million shots. **Infeasible.**
- **Model sensitivity.** At the planned 12,000 shots the n = 1 test keeps ≥ 80% power up to:
  - an extra dephasing of p = 0.27 per step, 4× the deficit-matched model's;
  - a phase offset of |φ| = 0.76 rad per step, a detuning of about 25 kHz over the 4.85 µs
    step. That is 18× #21's comparator and 1.7× a 16 kHz Stark shift.

  Its power is 0.46 at 0.80 rad and 0 from 0.9 rad (scanned in steps of 0.05).
- **Stage 2's fitted quantum model** (#32) has not landed. When it does, its step pieces drop
  into `R.step_ops`, and the table should be recomputed before any submission.

**Coherent SPAM is the assumption we can't check locally.**
- Incoherent SPAM noise only lowers the scores (shown above).
- A coherent error of trace distance δ per operation can shift the P&M score by up to 2δ.
- The margin at n = 1 tolerates δ ≤ 0.022 (0.018 for the deficit-matched model).
- The worst case, with all of q75's sx RB error (1.0e-4) coherent and adversarially aligned
  over 2 pulses, is 0.0245. So the margin covers about 90% of the worst case, not all of it.
- This is the "trusted SPAM" assumption of the hypothesis. The MDI game removes trust in the
  measurement but still trusts the inputs.

## 6. Physical layout and the reference

`references.csv`. Every qubit within 3 hops of q75 on the committed map is a candidate
(`results/summaries/phase7_spectator/device.json`: FakeKingston, calibration dated 2026-04-15,
stale). The rule was fixed in the config before the candidates were evaluated:
- no coupler to q75 (this excludes q74: its ZZ to q75, #22, and its sham changes the process,
  #29);
- no coupler to q59 or to a neighbour of q59, and no SWAP path through q59;
- readout ≤ 3%, T2 ≥ 50 µs, and every CZ on the path ≤ 1%;
- then the highest predicted n = 1 Bell fidelity.

| qubit | path from q75 | hops to q75 / q59 | SWAPs | readout | T2 (µs) | Bell F, n = 1 | admitted |
|---|---|---|---|---|---|---|---|
| 74 | 75–74 | 1 / 2 | 0 | 0.008 | 205 | 0.560 | no: coupler to q75 |
| 73 | 75–74–73 | 2 / 3 | 1 | 0.041 | 271 | 0.539 | no: readout |
| 72 | 75–74–73–72 | 3 / 4 | 2 | 0.053 | 138 | 0.525 | no: readout |
| **79** | 75–74–73–79 | 3 / 4 | 2 | 0.0085 | 78 | **0.546** | **yes** |
| 55 | 75–59–55 | 2 / 1 | 1 | 0.012 | 107 | 0.548 | no: q59's neighbour, path through q59 |
| 54 | 75–59–55–54 | 3 / 2 | 2 | 0.037 | 194 | 0.537 | no: near q59, path through q59, readout |

- **The pick is q79.** Isolating the reference costs 0.014 in F against using q74 directly
  (0.546 against 0.560). q79's stale T2 of 78 µs is the shortest of the candidates; at n = 1
  the reference idles about 7.3 µs.
- **Before any Bell-route submission** the rule must be re-run on live calibration. q73 would
  be preferred (one SWAP) if its readout is now ≤ 3%.
- The P&M route uses only q75 and q59. The MDI route adds q74, and only after the steps.
- **No adequately isolated reference was needed.** The recommended route has none, so the
  issue's "no isolated reference" failure mode does not arise.

## 7. Budget and the capped, unsubmitted plan

`budget.csv`, `summary.json` (`plan`). The cost model is `qmem.followup.CostModel`:
- 2 s per job, 250 µs repetition delay, the upper bound at 1.25 × the estimate + 1 s;
- circuit durations: the longer of the analytic step schedule (4.852 µs per step, #22) and
  FakeKingston's schedule of the exported circuits;
- accounting by max(charged, upper bound); an unreadable charge counts as the upper bound.

Each option fits in one job under Phase 6's 20 s per-job limit. Pub order is shuffled with a
seed.

| option | circuits | shots | estimate (s) | upper bound (s) |
|---|---|---|---|---|
| **P&M** (n = 0, 1, 2, 3, negative control; 1,000 shots per circuit at n = 1, otherwise 500) | 60 | 36,000 | 11.3 | 15.2 |
| **MDI game** (n = 0 at 250 per pair, n = 1 at 1,000 per pair) | 32 | 20,000 | 7.1 | 9.9 |
| Bell route (n = 0, 1, 2 + the q59-only control) | 20 | 32,000 | 11.0 | 14.7 |

**Recommended:** P&M, then MDI, with a **hard cap of 30 s**. Their upper bounds total 25.1 s,
and the Open plan had 531 s left after #27. The guards (`F.budget_check`) pass in a dry run on
the P&M job.

**Stop and go:**
- Run the P&M job first.
- Stop after it if the n = 0 gate fails or the negative control passes (§4).
- Run the MDI job only if P&M passes.

Nothing is submitted. The plan carries `submitted: false` and `approval: null`. As #28 states,
any submission needs the owner's explicit approval of this stage, the device and qubits, the
circuits, the upper bound and the cap. Before submission, also:
- a dry run on live calibration;
- the reference rule re-run (Bell route only);
- `targets_V` re-derived only if the owner accepts a live-calibration model; otherwise the
  committed V_n stand.

## 8. Literature, as read for this note

- **[Rosset2018]**, PRX 8, 021033. Read: §I (Figs. 1–2), §IV (Def. 2), §V (Prop. 2, Eqs. 12–15;
  the depolarising example, Eqs. 16–20; Prop. 3 on erasure).
  - EB channels are exactly the measure-and-prepare channels.
  - Every non-EB channel is detectable by a semiquantum signalling game that trusts only the
    inputs.
  - Trusted inputs are minimal for a channel test (their §I and ref. 38).
  - Our MDI route is their §V.A game with the target rotated by V_n.
- **[Giarmatzi2021]**, Quantum 5, 440. Quantum memory *of a process*: a process matrix that is
  entangled across times, detected by a witness measured with CP maps at each time. It is a
  property of the multi-time process, not of a channel's ability to carry a qubit. Claim 4 is
  the latter.
- **IBM dynamic-circuit guide [IBMDynamic2026].**
  - `MidCircuitMeasure` (`measure_2`) is recommended where the backend supports it.
  - Mid-circuit measurement duration matters; resets and measurements are not allowed inside
    conditionals.
  - Estimator does not run dynamic circuits, so Sampler is used.
  - Our circuits keep Phase 6's plain `measure`/`reset` on q59, so that the step is the one
    #27 measured. They have no conditionals.
- **[Yuan2021]**, npj Quantum Inf. 7, 108. Read in full from the PDF the owner attached on
  #28; it was added after #33 closed, by the coordinator.
  - It defines the robustness of quantum memory, R(N): the least EB noise s for which
    (N + sM)/(1 + s) is EB. R = 0 exactly when N is EB.
  - R is the advantage of N over every EB channel in the best non-negative "quantum game"
    (its Thm 3). Any fixed game gives a lower bound on R, P(N, G) − 1.
  - For a qubit channel it is exact and computable: R(N) = max{0, 2 λ_max(Φ_N) − 1}, where Φ_N
    is the normalised Choi state (its Thm 4, using the reduction criterion).
  - Its IBM experiment trusts SPAM. Its channels are synthesised with ancillas or built from CX
    sequences (R from 2.67 for one CX to 2.50 for six). So certifying a non-EB channel on IBM
    hardware *is not new*. What is new here is the object certified: one step of a trained
    instrument with a mid-circuit measurement.
  - **Applied to §1's channel** (coordinator's check on the Stage 4c pieces, not in the
    committed outputs):

    | n | R, unconditional (live) | R, unconditional (noiseless) | R, all-zero record, normalised (live) |
    |---|---|---|---|
    | 1 | 0.265 | 0.289 | 0.947 |
    | 2 | 0.016 | 0.036 | 0.890 |
    | 3 | 0 | 0 | 0.751 |
    | 4 | 0 | 0 | 0.793 |

    At n = 2, λ_max = 0.508 while the best Bell fidelity is 0.391. The Choi state's top
    eigenvector is not maximally entangled, which is why every fidelity witness fails there.
    The optimal game would detect it, but only with R = 0.016, out of reach in shots like
    §4's PPT witness. So the paper confirms §1's verdict and doesn't change the plan.
  - In the noiseless model, the y = 0 branch is a single, invertible Kraus operator (λ_max = 1),
    so certifying it shows that q59's y = 0 emission is not a measure-and-prepare readout of
    q75. That is the quantum model's mechanism, under trusted SPAM.

## 9. Limits

- The predictions are models: Stage 4c's calibrated instrument plus the sensitivity models.
  The device's baseline sits at 0.747 of that model (#29). The deficit-matched row assumes the
  shortfall is q75 dephasing; a different mechanism would move F differently.
- q75 is a qubit here. Leakage of q75 or q59 enters only through the worst-case allowance, not
  through a qutrit simulation. #21's leaky-qutrit fits were implausible at their fitted rates.
- The reference's calibration is FakeKingston's, and stale.
- Coherent SPAM is bounded only by RB error rates (§5).
- **The MDI game's trusted input sits on q74,** which is ZZ-coupled to q75 during the steps.
  - The game needs ψ_y uncorrelated with the channel's environment. That holds if q74 is in
    |0⟩ throughout, since a diagonal eigenstate picks up no correlation.
  - If q74 starts excited (its initialisation error, about 1%), ψ_y is prepared from the wrong
    state, and in a way correlated with q75's phase.
  - A `reset` of q74 just before ψ_y would remove that correlation, at the cost of 2.2 µs more
    idle for q75. It is not in the committed circuits; decide before any MDI submission.
- A success at n = 1 is Claim 4 at one step. It is not evidence that the emitted-symbol
  witness uses q75's coherence (Claim 3), and not a dimension advantage (Claim 5).

## Reproduction

- `--export-qasm` (hardware group) builds every circuit against FakeKingston, with optimisation
  level 0 and the fixed layout. It then:
  - simulates each circuit noiselessly and unconditionally, with every measurement as a Z
    dephasing;
  - checks it against the numpy model: worst difference 1.8e-15 over 112 circuits;
  - writes the QASM with its SHA-256.
- `tests/test_reference.py` checks the committed files against the manifest. With the hardware
  group, it rebuilds four of them and compares their hashes.
- Seeds: `[analysis] seed = 33` (Monte Carlo power, pub order); `[hardware] seed_transpiler =
  0`.
- The curated summary records the commit and config in its `meta`. The analysis ran at
  `447b12d` and the QASM export at `2fab8e8`. Those are the pre-rebase SHAs of `2600472` and
  `939c170` on `main`. The circuit code is the same in both; the note
  and the summaries were uncommitted at the time, hence `git_dirty`.
