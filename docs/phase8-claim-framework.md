# Phase 8 Stage 0B: claim framework, adversarial classical models and literature (#30)

Local theory, simulation and literature only: no QPU, no IBM service. This note asks what
Phase 7's output witness and its q75 interventions (#27, [phase7-hardware.md](phase7-hardware.md))
actually show, and what evidence each stronger claim would need. It answers first the
adversarial question of the coordinator (#28): **does a classical hidden-state model with ≥ 3
states and an explicit intervention model explain both the output witness and every Stage 4c
intervention outcome?** Numbers on the re-analysis come from Stage 0A (#29,
[phase8-stage0-audit.md](phase8-stage0-audit.md)).

- Code: `qmem.classical` (the model, its likelihood, exact statistics and an EM fit; numpy),
  `experiments/phase8_classical.py`, config `experiments/configs/phase8_classical.toml`.
- Outputs: `results/summaries/phase8_classical/` (`summary.json`, `classes.csv`,
  `statistics.csv`). The fit takes 67 min with 2 workers on a shared machine (load about 10), somewhat over the issue's hour.
- Tests: `tests/test_classical.py`. The likelihood is checked against brute-force enumeration
  of hidden paths. Also tested: the zero-witness results (2 states; a rank-2 intervention), the
  QND theorem, EM monotonicity within each class, the Klein-tetrahedron embedding of the
  calibrated model, the entanglement-breaking bound F ≤ 1/2 on random EB channels, and the
  finite-group lemma behind the coherence bar (§4).

## Headline: a classical model fits every arm

**A 6-state classical hidden-state model with explicit intervention maps reproduces every
Stage 4c statistic within #27's CIs.** It is fitted jointly to every committed shot of every
arm, A2's q75 record included. The fit covers:
- the baseline's w(0..3), g(0..7), P(1) and lag-1;
- the vanishing of A2, A1 and A3;
- the sham, including the g(k) and lag-1 differences #29 found;
- D1's lag-1;
- A2's q75 record: P(m = 1) and its conditional statistics.

All 16 primary statistics are within 1.96 SE (max |z| = 1.9). All 88 statistics are within the
Bonferroni level (max |z| = 3.2, against 3.4). Its likelihood beats the calibrated quantum
model's by 3,891 nats (0.0010 nats per symbol). With 353 free parameters against 0, that is
ΔBIC = −2,385 in the classical model's favour.

In this model, q75's Z measurement reads a nearly deterministic classical bit of the hidden
state and **disturbs that state strongly**. The mean probability that the hidden state survives
the measurement is 0.17. This is the clumsiness loophole of Leggett–Garg tests, and Theorem 3
shows it is exactly what makes A2 vanish.

**What it means.**
- **Claim 2 stands but is narrowed.** Measuring, resetting or randomising q75 removes the
  witness and the sham doesn't. That shows *q75 must not be disturbed*. It does not show that
  q75's *coherence* carries the memory.
- **Excluded, by proof and by fit:** every classical model in which q75's Z measurement is
  non-invasive on the hidden state (Theorem 2; the κ = 0 fit is 27,837 nats worse).
- **Not excluded:** classical models whose hidden state the q75 measurement disturbs.
- **Where the quantum model does better.** The zero-parameter quantum model predicts A3's
  shots better than every classical fit: by 981 nats at d = 6, and by 0.010 nats per symbol held
  out. A3's per-instance Pauli sequences are where the quantum description carries information
  the classical maps have to learn. That points Stage 2 to *sequence-resolved, reversible*
  unitary insertions on q75 (§1.4, §4).

## 1. The adversarial model

### 1.1 The model class

A classical hidden state `s ∈ {0, …, d−1}` sits at the start of every step. It can have any
physical reading: q75 in some basis, leakage, a TLS, the readout chain or the classical control
loop. One step of arm `a` is

    s --P--> u --M_a--> v --Q[y]--> s'

- `P` is the part of the step before the intervention slot (in the circuit, q75's leading gates
  while q59 is reset).
- `M_a` is the arm's stochastic map, in the slot where the circuit puts its operation.
- `Q[y]` is the rest of the step. It emits q59's symbol y (`Q[0] + Q[1]` is stochastic).

The arms:
- the baseline: `M = I`;
- A2s_spec: `M_spec`;
- A1: `M_reset`;
- A2: `M_m`, labelled by q75's recorded bit m, so the joint record is `L[m, y] = P M_m Q[y]`;
- A3: the instance's Pauli at each step, with `M_I = I` (a delay) and `M_X`, `M_Y`, `M_Z` free;
- D1: a different circuit (no CZ, no q75 gate), so its whole step is its own map on the same
  hidden state.

Every arm starts from the same initial distribution. The classes fitted:

| class | A2's map | other maps | free parameters |
|---|---|---|---|
| general, d = 3…6 | any `M_m` | all free | 77 (d = 3) … 353 (d = 6) |
| kappa, d = 4 | `(1 − κ) diag(q(m\|s)) + κ R_m`: QND on the hidden state except in a fraction κ of steps | free | 151 |
| zpop, dh = 2, 3 | `K_m ⊗ I`: acts on a classical bit z in q75's Z populations, with any back-action on z | P and every q75 intervention act on z only; the side memory h (dh states) is untouched by q75's operations; Q free | 75 (dh = 2), 153 (dh = 3) |

The fit is maximum likelihood on every committed shot of every arm: 58,096 shots of 64 steps,
A2's q75 record included, 3.73 M symbols. It runs EM over the factored step (expected counts of
`P`, `M_a` and `Q` from each arm's forward–backward), accelerated by SQUAREM
[VaradhanRoland2008]. Each class has several starts: random ones; for d = 4 also the calibrated
quantum model in hidden-state coordinates (§1.3); for d > 4 a split of the best (d−1)-state
fit. A statistic "fits" when the model is within 1.96 of its SE of the data:
- w(j): #27's SEs;
- g(k): whole-shot SEs;
- P(1): the larger of the shot SE and the between-job SE;
- A2's record: a whole-shot bootstrap.

The *primary set* is the issue's list:
- the baseline's w(0), g(0..7) and P(1);
- w(0) of A2, A1, A3 and the sham;
- D1's lag-1;
- A2's P(m = 1).

The *full set* is every w(0..3), g(0..7), P(1) and lag-1 of every arm, plus A2's eight record
statistics: 88 in all. Its misses are counted against a Bonferroni level, z = 3.4.

### 1.2 What is proved

**Theorem 1 (2 states).** Every stationary arm of a d = 2 model gives w(j) = 0 exactly. Its
labelled matrices are 2 × 2, so every 3 × 3 Hankel matrix of g has rank ≤ 2. This holds for any
maps and any initial state on #27's common start set (`test_two_state_models_give_zero_witness_in_every_stationary_arm`,
and Phase 7's `test_two_state_hmm_gives_zero_witness`). A3 is time-varying, and its pooled g is
an average of rank-2 Hankel sequences, so A3 need not vanish exactly at d = 2.

**Theorem 2 (a non-invasive A2 is excluded).** If q75's measurement is QND on the hidden state,
`M_0 + M_1 = I` (the outcome is read and the state left alone), then A2's output process *is*
the baseline's. Every statistic of y is equal, w(j) included (`test_qnd_measurement_reproduces_the_baseline_exactly`).
#27 measured A2's w(0) at 4.0% of the baseline's, with 95% upper bound 9.3%, against a
prediction of 100%. **Excluded:** every classical model in which A2's Z measurement, outcome
discarded, leaves the hidden state as it was. That covers:
- a classical memory held in q75's Z populations with a QND readout;
- any memory outside q75 (readout chain, q59 leakage, a TLS, the control loop) that the q75
  measurement does not physically touch, if it carries the rank > 2 part of the process.

The same argument applies to A1 and A3: a memory that the reset or the Paulis leave untouched
would survive them.

**Theorem 3 (an invasive A2 vanishes exactly at any d).** If the measurement re-prepares one of
two hidden-state distributions according to its outcome (`M_m = p(m|·) r_mᵀ`, a classical
measure-and-prepare map), then `L[y] = P M Q[y]` has rank ≤ 2 and w_A2 = 0 exactly. The
baseline is not constrained by this (`test_rank_two_slot_map_gives_zero_witness_at_any_d`).
**Not excluded:** classical models in which q75's Z measurement disturbs the hidden state. This
is the Leggett–Garg *clumsiness loophole* [LeggettGarg1985; Emary2014; WildeMizel2012]. The
vanishing of A2, A1 and A3 is what such a model predicts. Measuring a qubit in Z and re-preparing
|m⟩ is exactly this kind of map on the qubit, so the quantum model's A2 is a 2-state process for
the same reason.

**Theorem 4 (the calibrated quantum model is not a 4-state classical model).** The calibrated
model is #27's analysis model: #20's pieces at the pilot's live calibration. In it, q75 is a
qubit and q59 is reset every step, so all its arms form a 4-dimensional linear process on q75's
operators. A d = 4 classical realisation would be related to it by a similarity
[MonrasWinter2016, Thm 9; Ito1992]. The hidden states at the intervention slot would then be
the vertices of a simplex in Bloch coordinates. The Paulis must permute those vertices (a
unitary map can't take a simplex strictly inside itself). The Z measurement's effects must be
≥ 0 on them, and its post-measurement states |0⟩ and |1⟩ must lie inside. That forces the
simplex to be a *Klein tetrahedron*, with vertices (±a, ±b, ±1) with an even number of minus
signs (`qmem.classical.klein_tetrahedron`). The slot-to-slot baseline step `pre ∘ post_y` must
be nonnegative in those coordinates for every choice of step-start vertices, since it equals
`P Q[y]`. A scan of (a, b) over [−4, 4]² gives a most-negative entry of at least
−0.158 (at a = 1.45, b = 1.30) < 0 (`test_klein_tetrahedron_embedding_reproduces_the_quantum_model`). **So no d = 4
classical model reproduces the calibrated model exactly.** Whether d = 5–8 can, or whether no
finite d can, is open:
- the calibrated step's no-click operator has a dominant real eigenvalue (0.487, against a
  complex pair of modulus 0.459), so the Perron–Frobenius obstruction to every finite HMM
  [Vandergraft1968; Fanizza2024] does not apply;
- with noise, finite HMMs approximate any such process [Fanizza2024, Thm 6].

Exact equality was never the test, though. The data have finite CIs, and the question is
whether a classical model fits them, which §1.3 answers.

**Lemma (reversible phases).** A stochastic map on a finite hidden state whose inverse is also
stochastic is a permutation. If phase rotations act on the hidden state as a group
(`M_a M_b = M_{a+b}`), then each `M_φ = (M_{φ/N})^N` with N = d! is the identity, since every
permutation of d states has order dividing d!. So a finite classical model cannot represent a
*reversible, composable* phase rotation of q75 nontrivially. It must represent such rotations by
irreversible maps whose composition only *looks* like a group on the observed statistics
(`test_divisible_phase_group_acts_trivially_on_a_finite_hidden_state`). §4 uses this.

### 1.3 The fits

All fits stop at the 800-step EM cap, and none meets the 1e-10 per-symbol tolerance.
- *Classical likelihoods are lower bounds* on what each class can reach. More steps could only
  improve the classical side, so the "fits" verdict is conservative.
- *The negative fits are not proofs.* Those are κ = 0, κ = 0.3, zpop and d = 3, each from one or
  two starts. κ = 0 has a proof anyway (Theorem 2).

The run took 67 min with 2 workers on a shared machine.

| class | free params | NLL per symbol | ΔNLL vs quantum (nats) | ΔBIC vs quantum | primary (16): max \|z\| (worst) | primary: \|z\| > 1.96 | all 88: \|z\| > 3.4 (worst, max \|z\|) |
|---|---|---|---|---|---|---|---|
| calibrated quantum (#27's model) | 0 | 0.652077 | 0 | 0 | 37.4 (baseline g1) | 10 | 34 (sham g1, 41.0) |
| general, d = 3 | 77 | 0.654367 | +9,982 | +21,141 | 54.2 (baseline g3) | 8 | 25 (sham g3, 58.6) |
| general, d = 4 | 147 | 0.651688 | −1,693 | −1,138 | 6.1 (sham w0) | 6 | 15 (A2 P(m₀ = 1), 14.9) |
| general, d = 5 | 239 | 0.651365 | −3,100 | −2,547 | 5.5 (sham w0) | 3 | 9 (sham w2, 6.8) |
| **general, d = 6** | 353 | **0.651184** | **−3,891** | **−2,385** | **1.9** (baseline g7) | **0** | **0** (sham g5, 3.2) |
| κ = 0 (QND A2), d = 4 | 151 | 0.658464 | +27,837 | +57,983 | 88.3 (baseline g0) | 14 | 47 (A2 g0, 161.5) |
| κ = 0.3, d = 4 | 151 | 0.652154 | +337 | +2,983 | 13.2 (baseline g4) | 11 | 34 (A2 g0, 31.6) |
| κ = 0.6, d = 4 | 151 | 0.651758 | −1,390 | −471 | 6.1 (sham w0) | 4 | 11 (sham w2, 8.3) |
| κ = 0.9, d = 4 | 151 | 0.651658 | −1,825 | −1,343 | 5.1 (sham w0) | 7 | 17 (A2 P(m₀ = 1), 11.7) |
| zpop, dh = 2 (d = 4) | 75 | 0.654789 | +11,820 | +24,786 | 55.8 (baseline g3) | 10 | 33 (sham g3, 58.7) |
| zpop, dh = 3 (d = 6) | 153 | 0.653717 | +7,150 | +16,640 | 12.4 (A2 P(m = 1)) | 12 | 43 (A2 P(m₀ = 1), 21.4) |

The calibrated quantum model misses the baseline's g(1) and g(2) by |z| ≈ 37 and the sham's
lag-1 by z = −29. Its baseline sits at 75% of the predicted witness (#27). Its parameters are
fixed by calibration, so this is a statement about the calibration, not about the quantum
class. A fitted quantum model, which this note did not build, would close part of the gap.
**The likelihood comparison is therefore between a free classical class and a fixed quantum
model, and it should be read as "the classical class is not disfavoured", not as evidence
against quantum memory.**

**The d = 6 model on the primary set.** Data is #27's (SE as defined in §1.1):

| statistic | data | d = 6 classical | z | calibrated quantum | z |
|---|---|---|---|---|---|
| baseline w(0) | −5.153e-3 ± 1.27e-4 | −5.011e-3 | +1.1 | −6.676e-3 | −12.0 |
| baseline g(1) | 0.09251 ± 0.00039 | 0.09209 | −1.1 | 0.10703 | +37.4 |
| baseline P(1) | 0.5658 ± 0.0084 (between-job) | 0.5668 | +0.1 | 0.5810 | +1.8 |
| A2 w(0) | −2.04e-4 ± 1.41e-4 | +1.6e-5 | +1.6 | 0 | +1.4 |
| A1 w(0) | −1.29e-4 ± 1.86e-4 | −8e-6 | +0.7 | 0 | +0.7 |
| A3 w(0) | −2.1e-5 ± 9.5e-5 | −2.5e-5 | −0.04 | −6.7e-5 | −0.5 |
| sham w(0) | −5.382e-3 ± 1.18e-4 | −5.485e-3 | −0.9 | −6.676e-3 | −10.9 |
| sham w(1), w(2), lag-1 | −0.0363, −0.0417, 0.1934 | −0.0356, −0.0402, 0.1909 | +0.5, +0.8, −1.8 | (= baseline) | |
| D1 lag-1 | −0.0045 ± 0.0015 | −0.0051 | −0.4 | −0.0009 | +2.5 |
| A2 P(m = 1) | 0.4964 ± 0.0010 | 0.4963 | −0.06 | 0.4925 | −3.9 |

Every row of every class is in `results/summaries/phase8_classical/statistics.csv`. The fitted
parameters are in `summary.json`.

**The fitted d = 6 model.**
- *A2's map* has a mean diagonal of 0.17, so the hidden state survives q75's measurement with
  probability 0.17 on average and 0 in two states. P(m = 1 | u) is close to 0 or 1 in five of
  the six states, so m is a near-deterministic readout of a classical bit that the measurement
  then scrambles.
- *The sham's map* is not the identity in these coordinates. The hidden-state basis is a gauge,
  and the data need the sham to differ (#29 finding 4).
- *Per-arm likelihood against the quantum model* (nats): baseline −2,369, sham −1,887,
  A2 −523, A1 −57, D1 −35, **A3 +981**.

**Invasiveness (the κ profile, d = 4).** A2's map is constrained to be QND on the hidden state
in a fraction 1 − κ of steps.
- κ = 0 (non-invasive) is 27,837 nats worse than the general class and misses A2's g(0) by
  z = 161.
- κ = 0.3 is still 337 nats worse than the quantum model.
- κ ≥ 0.6 matches the general class.

So *a classical model must let q75's Z measurement disturb its hidden state in at least about
half of the steps*. The general d = 4 and d = 6 fits are at the invasive end, with mean
disturbance 0.59 and 0.83.

**zpop: a classical bit in q75's Z populations plus side memory q75 doesn't touch.** Back-action
of the measurement on z is allowed. Neither dh = 2 nor dh = 3 fits.
- They miss A3 (z up to 20) and A2's record (z up to 21), and the sham by z = 12–46.
- They are 7,150–11,820 nats worse than the quantum model.

The A3 misses are telling. With the memory in z alone, X and Y only flip a bit, and the model
cannot reproduce the Pauli arm together with A2. *(Not a proof: one random start per dh,
800 steps.)* The physical reading "q75 is a classical population bit and the rest is side
memory" is disfavoured. "q75 carries a disturbable classical variable richer than its Z
population" is not.

**Does A2's q75 record constrain the class?** Yes, but only at small d.
- A d = 4 fit without the record predicts it badly: P(m = 1) at z = 229, and P(m = 1) at
  step 0 at z = −88. The y data alone don't determine the record.
- Fitting the record as well costs 482 nats of y-likelihood at d = 4 (2,453,128 against
  2,452,646).
- At d = 6, the joint fit's y-likelihood (2,451,065) beats the y-only d = 4 fit. The record is
  then absorbed at no cost.

**Cross-validation (d = 4, two folds by pub parity).** Held-out NLL of classical minus quantum:
+0.00029 and −0.00059 nats per symbol. That is mixed: the d = 4 class does not reliably
out-predict the calibrated quantum model on unseen pubs. Per arm, it is better on the baseline
(fold 2), the sham and A2, and **worse on A3 by 0.010–0.011 nats per symbol in both folds**.
A3's held-out pubs are unseen Pauli instances. The quantum model's exact unitaries generalise,
while the classical Pauli maps, learnt from 2,048 shots, do not. The d = 6 class was not
cross-validated.

**The job1 dip (#29, finding 5).** In job1, the baseline's P(1) drops by 0.033 and no
interleaved arm's does. No stationary model, classical or quantum, produces an arm × job
effect.
- Both models here treat it as unexplained. The P(1) SE in the fit test includes the
  between-job spread (0.0084 for the baseline), so P(1) "fits" in the main fit only because of
  that.
- The sensitivity fit without the baseline's job1 shots (d = 4) reaches 0.650576 nats per symbol
  on its own data. Its statistics are compared with the all-shot data, so its baseline misses
  (z ≈ 11) partly reflect job1 itself. Its fitted A2, A1, A3 and record statistics are as good
  as the main d = 4 fit's.
- The fit does not explain the dip. It also does not depend on it.

### 1.4 What would distinguish the classical model from the quantum one

The classical and quantum descriptions make the same *qualitative* predictions for every arm
Phase 7 ran: vanishing under invasive controls, persistence under the sham. They differ in
these ways:

1. **Sequence-resolved unitary insertions** (the A3 signal). With the Pauli or phase sequence
   known per shot, the quantum instrument predicts every instance's statistics with no free
   parameter. A classical model must learn one stochastic map per gate from data, and the
   learnt maps don't transfer to unseen sequences (held out, +0.010 nats per symbol on A3).
   - *Test:* predeclare a set of deterministic insertion sequences. Fit the classical class on
     half of them and predict the other half. Compare held-out likelihoods with the
     instrument's fixed prediction.
   - *Why it bites:* reversible gates compose exactly. By the Lemma, a finite classical model
     can't make Rz(φ)Rz(−φ) the identity unless both are permutations. Inverse-pair arms
     (X then X, Rz(φ) then Rz(−φ)) restore the baseline exactly in the quantum model. A
     classical model with invasive maps must restore it through a fitted coincidence.
2. **Coherent phase response** (§4): w(φ) periodic and non-monotone. A finite classical model
   needs more states as the response is resolved more finely.
3. **Entanglement with a reference** (Stage 3, Claim 4). This is outside the scope of any
   output-only classical model. It certifies the channel, not the witness's use of it.
4. **Knee-style classical-disturbance controls** (§4): the disturbance on |0⟩/|1⟩ inputs
   against superpositions, as an inequality.

What would *not* distinguish them:
- more A2 shots;
- more pairs;
- a dephasing dose–response alone.

The classical class reproduces all of these.

## 2. The claim taxonomy

The five claims of #28, from weakest to strongest. Each needs everything above it. "Quantum"
means something different in each.

### Claim 1: output memory

- **Statement.** The output process on q59 has rank > 2: no stationary 2-state classical HMM
  generates it.
- **Notion of "quantum".** None. Rank > 2 is a statement about classical dimension. Any 3-state
  classical HMM can have it.
- **Excludes:** stationary 2-state HMMs of any kind (Theorem 1), and mixtures of memoryless
  shots, which give w ≥ 0 (#23).
- **Doesn't exclude:** any ≥ 3-state classical process. That includes a nonstationary mixture of
  2-state regimes, which can give w ≠ 0 at finite length. #19 bounded that family to 21–24% of
  the witness, but did not exclude all such mixtures.
- **Trust assumptions:**
  - shot independence and stationarity within the analysis window (the shot order is IBM's
    returned order);
  - the readout as recorded (no SPAM model is needed, since the claim is about the recorded
    process);
  - no dimension assumption.
- **Falsification test.** w(0) consistent with 0 under the predeclared block and stratified SEs
  on a replicate, or a 2-state nonstationary null that reproduces w(0) and g(k) within their CIs.
- **Status: supported.**
  - #18: w(0) = −0.00494, block z = −39.
  - #27: baseline −5.15e-3 ± 1.27e-4, z = −40.5.
  - #29 reproduced the result exactly.
  - Limits: one pair and one device. Within the baseline, job1's P(1) dip is arm-specific
    (#29, finding 5).
- **Known counterexample class:** a nonstationary 2-state mixture at finite L. #19 bounded it
  but did not exclude it in general.

### Claim 2: causal q75 dependence

- **Statement.** Disturbing q75 every step removes the output witness, under matched controls.
  A spectator measurement plus the same delay does not.
- **Notion of "quantum".** None. This is a causal (interventionist) statement about a physical
  location.
- **Excludes** (Theorem 2):
  - memories that q75's Z measurement, its reset and the random Paulis all leave untouched: the
    q59 readout chain, q59's leakage, a TLS or loop memory not coupled through q75, and a
    classical memory in q75's Z populations read out QND;
  - memories that the q74 measurement and delay would disturb, given the sham (at ±20%; see
    Status).
- **Doesn't exclude:**
  - **any classical memory carried by q75 that its Z measurement physically disturbs**
    (Theorem 3, and the fit of §1.3);
  - a memory elsewhere that q75's operations disturb through crosstalk;
  - a memory below the vanish bound (about 5e-4 in w(0)).
- **Reading, narrowed by this issue.** "q75 must not be disturbed". *Not* "q75's coherence
  carries the memory".
- **Trust assumptions:**
  - the arms differ only in the intended operation (#20's circuit equivalence and hash checks);
  - stationarity across interleaved jobs; the arm-specific job1 dip shows this is imperfect;
  - the q75 record is not used by the rule.
- **Falsification test.** A2's w(0) ≥ 25% of the baseline's at ≥ 5 SE (*persists*) on a
  replicate. Or a sham that fails equivalence at the predeclared margin. Or the vanishing failing
  on another qubit pair under the same design (Stage 1).
- **Status: supported, narrowly.**
  - #27: A2 at 4.0%, 95% upper bound 9.3% (93% of the 10% threshold). The sham ratio is
    1.044 ± 0.034. A1 and A3 vanish descriptively. D1's lag-1 excess is −0.0036 (z = −2.46).
  - #29 calls the vanish result *ambiguous*. One predeclared choice, dropping job1, puts A2's
    bound at 10.1%. A replicate would come out *vanishes* only 91% of the time at a true ratio
    of 0.
  - **The sham is equivalent at ±20% (TOST p = 3e-6) but not at ±10% (p = 0.054)**, and it
    equals the baseline only in w(0). Its lag-1 excess is +0.015 (z = 8.3), and its w(1) and
    w(2) are 1.21× and 1.38× the baseline's (#29, findings 3–4). The sham does change the
    process.
- **Known counterexamples:**
  - the classical models of §1.3;
  - a "clumsy" classical q75 whose Z measurement re-prepares its hidden variable (Theorem 3).
    This is a Kochen–Specker-type hidden variable for q75's Bloch vector, updated on measurement
    [KochenSpecker1967; HarriganSpekkens2010].

### Claim 3: coherence dependence

- **Statement.** An intervention that selectively removes q75's off-diagonal coherence removes
  the witness, while q75's Z populations, the timing and the spectator and readout environment
  are held fixed.
- **Notion of "quantum".** Coherence in q75's Z basis, as a resource the process uses. That is
  basis-dependent and needs q75's Z basis to be operationally defined (by its measurement).
- **Excludes, if shown:** memories carried by q75's Z populations alone, *including* those the
  measurement disturbs. A population-preserving dephasing would leave them intact.
- **Doesn't exclude:** a classical hidden variable that encodes q75's transverse Bloch
  components and that the dephasing operation disturbs (§4). Dephasing a qubit and randomising a
  classical "phase" variable look the same in output statistics.
- **Trust assumptions:**
  - the dephasing is implemented as a per-step, per-shot channel, not a pooled set of fixed
    patterns (#28);
  - the populations are preserved, checked by an independent benchmark;
  - virtual-Z frame changes are not simplified away by the compiler;
  - timing and spectators are matched.
- **Falsification test.** A predeclared coherence factor c (Stage 2), with the witness
  following the instrument's predicted w(c) rather than a classical population-only prediction.
  Or w unchanged at c = 0 while populations are preserved. Either removes the claim.
- **Status: not demonstrated.** #27 bounds A2's surviving coherence at 0–0.55; 0.8 is ruled
  out.
- **Known counterexamples:**
  - **A pooled mixture of fixed Z patterns**: per-circuit fixed phases, pooled across shots.
    In expectation this is a dephasing channel. But each circuit is a fixed unitary, so the
    pooled data are a *mixture of processes* with a hidden instance variable. That can fake a
    dose–response curve or remove the witness by averaging rank-3 Hankel sequences (A3's finite-M
    bias, #20).
  - **A classical model that fakes a dose–response.** Take a classical hidden variable whose
    disturbance probability rises with the dephasing strength. With p_dist(c) = 1 − c and the
    κ-class map at κ = 1 − c, it gives any monotone w(c) curve. A dose–response is not a
    categorical proof (#28).

### Claim 4: quantum-state retention (a non-entanglement-breaking channel)

- **Statement.** The relevant channel can preserve a quantum resource. That channel is the step
  instrument averaged over q59's outcome, or a conditioned branch. Operationally: q75's
  entanglement with an isolated reference survives one or more steps.
- **Notion of "quantum".** A non-entanglement-breaking channel [Horodecki2003EB; Rosset2018].
  Every EB channel is measure-and-prepare, and its Choi state is separable.
- **Excludes, if shown:** every measure-and-prepare (classical-memory) description of that
  channel. An EB channel on half of a Bell pair leaves a separable state, and a separable
  2-qubit state has fidelity ≤ 1/2 with every maximally entangled state [Horodecki1999Teleport]
  (`test_entanglement_breaking_channel_bell_fidelity_at_most_half`). So a lower CI bound above
  1/2 certifies a non-EB channel, given trusted preparation and measurement.
- **Doesn't exclude:** a classical explanation of the *output witness*. A Bell pair that
  survives the channel shows a capability of that channel. It doesn't show that the emitted
  symbols use it.
  - The witness depends on the q59-emission instrument's branches. A channel can be non-EB on
    average while every branch that matters for the witness is near-classical.
  - The converse also holds: a channel can be EB on average while the conditional branches carry
    coherence.
- **Trust assumptions:**
  - trusted Bell-state preparation and end measurements (SPAM characterised);
  - no direct route from the reference to q59;
  - q74's isolation as reference, given its ZZ to q75 (#22, #29 finding 4);
  - leakage outside the qubit subspace treated as loss;
  - no post-selection on success.
  - A measurement-device-independent version [Rosset2018] trusts only the input states.
- **Falsification test.** Raw Bell fidelity lower CI ≤ 1/2 at every step count, with the
  zero-step positive control above 1/2 and a deliberately EB negative control at ≤ 1/2.
- **Status: not demonstrated.** Stage 3 is a feasibility study.
- **Known counterexamples:**
  - a Bell pair that survives in a channel that doesn't matter for the witness (above);
  - a reference that couples to q59 directly, which fakes retention through the output chain;
  - post-selection on q59 outcomes that picks out the coherent branch.

### Claim 5: memory-dimension advantage or device-independent proof

- **Statement.** Either of two statements:
  - (a) the device's memory is one qubit while every classical model of the same statistics
    needs more than two states;
  - (b) a device-independent witness bounds the memory's classical or quantum dimension.
- **Notion of "quantum".** A dimension separation: quantum dimension D_q against classical
  D_classical for the same process [Gallego2010; WolfPerezGarcia2009; Budroni2019].
- **Excludes, if shown:** every classical model with ≤ D_classical states, and with (b), without
  trusting the device's description.
- **Doesn't exclude:** a classical model with more states. And (a) needs the memory's quantum
  dimension bounded *physically*: leaky resets, spectators, a TLS and the readout chain all add
  states.
- **Trust assumptions:**
  - for (a): that the memory is q75 alone, a qubit with no leakage, no spectator and no
    readout-chain state;
  - for (b): the scenario's assumptions (independence of shots, no side channels, time-
    independent instruments [Hoffmann2018; Budroni2019]).
- **Falsification test.**
  - (a): a classical model with 2 states that fits; Theorem 1 already excludes it, so (a) stands
    or falls with the quantum dimension.
  - (b): a classical realisation within the bound.
- **Status: not supported, and out of reach of this design.**
  - A 3-state classical HMM fits the baseline (#19).
  - The joint classical model of §1.3 fits all arms at d = 6.
  - "Classical needs 3 states, the device uses a qubit" is a dimension advantage only if the
    device's memory is provably one qubit. With leakage, spectators and the q59 chain it isn't.
  - The literature's dimension advantages [Ghafari2019PRX; Elliott2020] compare against
    *causal* (predictive) classical models and take the quantum dimension from the
    construction, not from data.
- **Known counterexamples:**
  - a 3-state classical HMM of the output (#19);
  - the general d-state intervention model (§1.3);
  - a leaky qutrit q75 or q59 (#21), which makes "one qubit" false while the statistics look
    the same.

### Summary

| claim | "quantum" in it | status | what would move it |
|---|---|---|---|
| 1. output memory | none (classical dimension > 2) | **supported** (#18, #27; reproduced, #29) | replication on other pairs (Stage 1) |
| 2. causal q75 dependence | none (causal location) | **supported, narrowly**: A2 at 4.0%, bound 9.3% (*ambiguous* under one predeclared choice, #29); sham equivalent at ±20%, not ±10% (#29). **Narrowed: "q75 must not be disturbed"**; disturbed-classical-q75 models fit (§1.3) | ≥ 16,000–20,000 A2 shots, sham TOST at ±10% with ≈ 30,000 shots per arm (#29); a second pair |
| 3. coherence dependence | Z-basis coherence | **not demonstrated** | §4's bar: population-preserving, reversible phase controls |
| 4. non-EB channel | entanglement with a reference | **not demonstrated** | Stage 3 Bell-fidelity witness, F > 1/2 |
| 5. dimension advantage / DI | D_q < D_classical | **not supported**; out of reach | a DI temporal dimension witness with every physical path bounded |

## 3. Literature

Keys resolve in `docs/references.bib`. Read depth is recorded there: FT is full text, Abs the
abstract, Meta metadata only. Statements marked *(ours)* are this note's reasoning, not a cited
result.

**Classical and quantum stochastic simulators, causal states.**
- *Causal states.* The causal states of a process are its minimal *predictive* model, and
  their entropy is the statistical complexity C_μ [Crutchfield1989; Shalizi2001]. Minimality
  holds among rivals whose states are functions of the past. It does not hold among general
  (non-unifilar) HMMs, which can be smaller.
- *Quantum models.* Quantum causal models can store less, C_q < C_μ, whenever C_μ exceeds the
  excess entropy [Gu2012]. This is generalised to q-machines in [Mahoney2016] and given a
  unitary simulator with bounded memory in [Binder2018].
- *Dimension.* [Elliott2020] defines a dimensional (single-shot) measure D_q against D_μ and
  gives processes with D_q = 1 qubit and D_μ unbounded. *(ours, following the reader's note)*
  The gap is relative to causal classical models; the same process has a 2-state non-unifilar
  HMM.
- *Memoryless quantum dynamics.* [Korzekwa2021] shows it can simulate processes that need
  classical memory.
- *Our relation.* Our witness compares against *all* classical HMMs, not causal ones. So it is
  a weaker kind of separation, a classical dimension of at least 3. It is also a more robust
  one: it needs no predictive constraint.

**Experiments demonstrating quantum memory advantage.**
- [Palsson2017] shows an *entropic* advantage on the perturbed coin (photonic, C_q = 0.05
  against C_μ = 1). The dimension is 2 in both, with a trusted qubit and tomography.
- [Ghafari2019PRX] shows a *dimensional* advantage for the 3-symbol oracular coin, D_q = log₂ 2
  against D_μ = log₂ 3. The qubit dimension holds by construction (one photon's polarisation),
  not from a dimension witness. The classical figure is the optimal *causal* simulator. The
  protocol runs one step per causal state, with tomography.
- We found no hardware demonstration on a superconducting device.
- *Our relation.* None of these test a trained instrument with causal interventions on its
  memory, and none face the clumsiness problem. They don't need to: they trust the memory.

**Realisation theory (when is a process a finite classical HMM?).**
- Finite Hankel rank is necessary but not sufficient [Vidyasagar2011].
- A quasi-realisation has an equivalent positive (HMM) realisation iff a pointed polyhedral cone
  contains its state, is invariant under every operator, and has the final functional ≥ 0 on it
  [Dharmadhikari1963; Heller1965; MonrasWinter2016, Thm 6].
- Minimal quasi-realisations are related by similarity [Ito1992; MonrasWinter2016, Thm 9].
  Theorem 4 uses this.
- Finite-dimensional quantum processes exist with no finite HMM, when the peripheral eigenvalues
  are incommensurate with π [MonrasWinter2016, Ex. 2; Fanizza2024, Thm 5]. Under noise a finite
  HMM exists, with a size lower bound from where the eigenvalues lie [Fanizza2024, Thm 6].
- Hidden quantum Markov models can be smaller than classical ones [Monras2011].

**Dimension witnesses.**
- *Spatial and prepare-and-measure.* Device-independent dimension witnesses are Bell-type
  [Brunner2008] or prepare-and-measure [Gallego2010]. In the latter, classical and quantum
  dimension polytopes differ, and experiments followed [Hendrych2012; Ahrens2012].
- *Time series.* [WolfPerezGarcia2009] bound the dimension from the Hankel rank of one
  expectation-value sequence of a Markovian channel. A d-state classical chain's eigenvalues lie
  in the convex hull of roots of unity of order ≤ d, so damped oscillations need many classical
  states. This is the closest antecedent of our witness. Theirs uses one readout at time t,
  where ours uses sequential instruments.
- *Sequential measurements with bounded classical memory and arbitrary invasiveness.* A
  d-state machine is an HMM with input-dependent maps. A qubit's temporal correlations can
  exceed any classical bit's (Ω = 9/4 against 2.3557) [Budroni2019]. Qubit bounds against qutrit
  bounds give a dimension test [Hoffmann2018]. The memory cost of contextuality is studied in
  [Kleinmann2011]. Extremal temporal correlations are classically realisable with enough memory,
  which grows exponentially with sequence length [Spee2020]. Leggett–Garg bounds in multilevel
  systems are in [BudroniEmary2014].
- *Our relation.* This is the framework that fits our question. Our classical models are
  exactly Budroni et al.'s finite-state machines with an input (the arm). *(ours)* What is
  missing for a Claim-5 test is a witness built for our scenario, with the q75 arms as
  "inputs", whose classical bound is computed for d = 2, 3, …. Our d = 2 bound is Theorem 1;
  higher d is open.

**Resource theory of quantum memory; non-EB channel certification.**
- *What a quantum memory is.* A non-entanglement-breaking channel. Every non-EB channel can be
  certified by a semiquantum signalling game, with only the input states trusted, i.i.d. and
  characterised (measurement-device-independent in time) [Rosset2018].
- *EB channels.* EB ⇔ measure-and-prepare ⇔ separable Choi state [Horodecki2003EB]. A separable
  state has singlet fraction ≤ 1/d [Horodecki1999Teleport; Horodecki1999Reduction]. That gives
  the Bell-fidelity null F ≤ 1/2, and F_avg ≤ 2/3 for uniformly random qubit inputs
  [MassarPopescu1995], since f = (dF + 1)/(d + 1).
- *Robustness.* [Yuan2021] quantify quantum memory by its robustness and benchmark it on IBM
  hardware. Their inputs and measurements are trusted, and their channels are synthesised with
  ancillas.
- *Our relation.* Stage 3's natural framework. These certify a *channel*. None of them says the
  channel's quantumness is used by a particular output statistic. That gap is Claim 4's caveat.

**Quantum process memory and non-Markovianity.**
- *The process tensor.* It describes multi-time quantum processes under interventions
  [Pollock2018PRA]. Non-Markovianity is witnessed by dependence across a causal break
  [Pollock2018PRL]. It has been characterised on IBM processors without mid-circuit measurement
  [White2020].
- *Memory in general, not quantum memory.* These measures don't distinguish classical from
  quantum memory [MilzModi2021].
- *Classical memory in a process.* The environment applies EB channels, equivalently
  outcome-conditioned instruments. Its process matrix is separable, and an entanglement witness
  on it certifies quantum memory, given characterised interventions [Giarmatzi2021].
  [Taranto2024] establish a strict hierarchy, memoryless ⊊ mixed memoryless ⊊ classical memory
  ⊊ separable ⊊ quantum, for three or more times. [Backer2024] give a criterion from two known
  maps. In their sense, random-unitary dephasing is always classical.
- *Our relation.* These treat q75 as the *environment's* memory, probed by interventions on a
  system. Our A1/A2/A3 arms are interventions on the memory itself. *(ours)* Phase 7's controls
  are therefore closer to Leggett–Garg disturbance tests than to process-tensor witnesses. A
  process-tensor witness [Giarmatzi2021] on q59, with q75 as environment, would be a genuine
  Claim 4/5 alternative. It needs informationally complete, characterised instruments on q59,
  far beyond Phase 7's design.

**Leggett–Garg, invasiveness and the clumsiness loophole.**
- *Macrorealism plus non-invasive measurability.* It conflicts with quantum mechanics
  [LeggettGarg1985]. A macrorealist can always claim that the measurements disturbed the system;
  there is no "cast-iron" defence [Emary2014].
- *Remedies:*
  - adroit measurements, each checked to be non-invasive alone, which narrows clumsiness to a
    "collusion" loophole [WildeMizel2012];
  - ideal negative measurements [Robens2015];
  - control experiments that measure the classical disturbance on reliably distinguishable
    states and compare it with the quantum disturbance, on a superconducting flux qubit [Knee2016].
- *Our relation.* **This is the closest precedent for Phase 7's logic.** A2 is an invasive
  measurement. The sham is an adroitness control for a *different* qubit. Theorem 2 is the
  non-invasive half, and Theorem 3 is the clumsiness loophole. Knee et al.'s control design is
  what Stage 2 needs (§4).

**Contextuality and what "classical" means.**
- *Classical models always exist without constraints.* With no constraint on the hidden state, a
  classical (ontological) model always exists: the ψ-complete Beltrametti–Bugajski model, whose
  hidden state is the quantum state [HarriganSpekkens2010].
- *Kochen–Specker.* KS's qubit model reproduces projective measurements with a sphere-valued
  hidden variable [KochenSpecker1967].
- *So a constraint is needed.* Nonclassicality needs bounded memory, as in our d, or
  noncontextuality [Spekkens2005]. A prepare-and-measure fragment has a finite noncontextual
  model iff it is simplex-embeddable [Schmid2021]. *(ours)* Theorem 4 is a simplex-embedding
  test of the calibrated model's *transformations* at d = 4, a setting outside [Schmid2021]'s
  prepare-and-measure theorem.

**IBM dynamic circuits.** Mid-circuit measurements are tuned differently from terminal ones;
`MidCircuitMeasure` is preferred where available [IBMDynamic2026]. This bears on A2's q75
measurement, and on Stage 2's choice between measurement and dephasing.

**What is new here, and what isn't.**
- *Not new:*
  - rank > 2 output statistics as evidence of classical dimension > 2 [WolfPerezGarcia2009];
  - the clumsiness loophole [Emary2014];
  - non-EB certification by Bell fidelity [Horodecki2003EB; Rosset2018];
  - dimension advantage in stochastic simulation [Ghafari2019PRX].
- *New, narrowly:*
  - a joint maximum-likelihood fit of an explicit classical intervention model to every arm of a
    hardware ablation, including the ablated qubit's measurement record;
  - the exclusion and non-exclusion theorems for this specific design (Theorems 2–3);
  - the Klein-tetrahedron obstruction showing that the trained instrument's calibrated model is
    not a 4-state classical model (Theorem 4).

None of these is a quantum-memory proof.

## 4. What would count as evidence of coherence specifically

**The problem.** Every q75 control so far is *invasive*. A classical model in which the
intervention disturbs the hidden variable reproduces each vanishing (Theorem 3, §1.3). A
"selective dephasing" in Stage 2 has the same weakness. A classical model can posit a hidden
"phase" variable that random Z rotations randomise. Dephasing a qubit and randomising a
classical variable both give vanishing. So **"the witness vanishes under dephasing" is evidence
that q75 must not be disturbed, not evidence of coherence.**

**What would count, in increasing strength:**

1. **Population invariance, checked independently (necessary, not sufficient).** Z populations
   must be unchanged by the dephasing operation: a direct T1-type and population benchmark of
   q75 with and without it, at the per-step level. q59's readout and timing must also be
   unchanged. This removes every *population-carried* memory, including clumsy ones. It does not
   remove a classical variable for the transverse components.

2. **A coherent phase response, not only a dose response.** Insert a deterministic rotation
   Rz(φ) on q75 each step, physically implemented, not only as a virtual frame change the
   compiler could fold into later pulses. Scan φ over a full period.
   - The quantum instrument predicts w(φ), g(k; φ) and P(1; φ): periodic, and *non-monotone*,
     possibly with sign changes of w or shifts in g's oscillation frequency.
   - It must return to the baseline at φ = 2π, and Rz(φ₁)Rz(φ₂) must match Rz(φ₁ + φ₂)
     operationally.
   - **Why it helps.** By the Lemma (§1.2), a finite classical model cannot represent a
     reversible, composable phase group nontrivially. It must use irreversible maps that only
     mimic the group on the observed statistics. A fit to the φ scan, with the composition and
     inverse arms as constraints, gives a *lower bound on d*. That bound grows as the phase
     response is resolved more finely, in the spirit of [WolfPerezGarcia2009; Fanizza2024,
     Thm 6]. Each extra φ setting or composition arm tightens the bound. *(ours; the bound for
     this design is to be computed in Stage 2.)*
   - The dephasing dose–response then becomes the average over φ of this response, with a
     predicted shape, not a free monotone curve.

3. **Operationally equivalent implementations must agree** (preparation and transformation
   noncontextuality [Spekkens2005; Schmid2021]). Complete Z dephasing can be implemented as:
   - a Z measurement with the outcome discarded (A2);
   - a uniformly random Z rotation, drawn per shot and per step;
   - a random choice of Rz(0) or Rz(π), per shot and per step.

   Quantum mechanically these are the same channel. A noncontextual classical model must also
   represent them by the same stochastic map, while a contextual or clumsy one may not. Measured
   witnesses that agree under all of them, at matched timing, make the classical alternative pay
   with contextuality. Differences flag implementation artefacts: measurement-induced
   disturbance (#25), or pooled fixed patterns. *(The noncontextuality assumption must be
   stated; it is not a theorem about the device.)*

4. **Classical-disturbance controls in the Knee et al. style [Knee2016].** For states the Z
   measurement reveals reliably (q75 prepared in |0⟩ or |1⟩ at a fixed step), measure the
   disturbance the intervention causes to the *output* (a "classical disturbance"). Compare it
   with the disturbance for superposition inputs. A classical model with a Z-population memory
   predicts equal disturbance; coherence predicts more for superpositions. This is the closest
   available answer to the clumsiness loophole, and it is a quantitative inequality, not a
   vanishing.

5. **Entanglement with a reference (Stage 3).** This is the only route to a basis-independent,
   assumption-light statement about q75's *channel*: Claim 4, not Claim 3.

**Counterexamples the Stage 2 design must handle explicitly:**
- **Pooled fixed patterns.** A fixed Pauli or phase pattern per circuit instance, pooled across
  instances, is a *mixture of processes*, not a per-step channel. Its witness is the average of
  instance g-sequences, which can cancel w (A3's finite-M bias, #20) or produce a smooth
  dose–response from a hidden instance variable. The randomisation must be per shot and per
  step, or the analysis must model the instance as a hidden variable and fit per instance.
- **A dose–response from disturbance probability.** The κ class with κ = 1 − c gives any
  monotone w(c). A monotone curve alone can't separate it from coherence. Item 2 (non-monotone
  periodic response) or item 4 (a disturbance inequality) can.
- **Virtual Z commutation.** IBM's rz is a frame change. A scheduler or transpiler can commute
  or cancel it, and a classical controller "remembers" the frame. A phase scan must be checked
  on the transpiled instructions (#28).

**The bar.** Stage 2 supports Claim 3 only if all of the following hold:
- (i) population invariance is verified;
- (ii) a predeclared coherent phase response, not only a dephasing dose–response, matches the
  instrument's fixed prediction, and fits no classical model with d ≤ D for a predeclared D,
  computed by this note's fitting code extended to the φ arms;
- (iii) equivalent dephasing implementations agree within a predeclared margin.

Item (ii)'s D is a real assumption, and Claim 3 inherits it.

## 5. Recommendations for Stages 1–3

**Stage 1 (replication on other pairs).**
- Predeclare every arm's predictions under both the calibrated instrument and this note's best
  classical intervention model, refitted to each pair's own pilot. The classical model is the
  alternative hypothesis, not only the 2-state null.
- Keep A2 and the matched sham. Raise A2's shots to the 16,000–20,000 #29 computes. Predeclare
  the sham's TOST at ±10% (α = 0.05 per bound, about 28,000–35,000 shots per arm; #29).
- Compare the sham with the baseline on **g(k) and lag-1, not w(0) alone** (#29 finding 4).
- Treat the job as a stratum, and report any arm × job effect. #29's job1 dip in one arm is not
  explained by any stationary model, quantum or classical (§1.3).
- A replication supports Claim 1 and, narrowly, Claim 2. It cannot move Claim 3, since the
  classical model reproduces A2's vanishing on any pair.

**Stage 2 (coherence).**
- Design to §4's bar: a physical (not virtual) per-step Rz(φ) scan with inverse and composition
  arms; equivalent dephasing implementations (Z measurement, random-phase per shot and per step,
  random Rz(0)/Rz(π)); a population benchmark; Knee-style classical-disturbance controls.
- Before any hardware, compute with `qmem.classical` extended to the φ arms (each φ a free
  stochastic map in the slot, with composition constraints):
  - the minimal d at which a classical model fits the *predicted* quantum data at the planned
    shots;
  - the shots at which d ≤ 4, or d ≤ 6, is excluded.

  That is Stage 2's power study against the classical alternative. If no feasible shot count
  excludes small d, say so and don't run.
- The dephasing dose–response alone should not be run as the primary test.

**Stage 3 (entanglement with a reference).**
- The null is F ≤ 1/2 for EB channels (tested here). Predeclare the target Bell state, the
  local-rotation freedom (the maximum over local unitaries must be fixed before data or bounded
  with its own null), the step counts and the branches.
- Unconditional channel first. Conditioned branches only with a predeclared minimum count and
  no success-only post-selection.
- Positive control (0 steps) and an EB negative control (q75 measured mid-way).
- q74 as reference is compromised by its ZZ to q75 (#22, #29 finding 4). Prefer a reference
  with no coupling path to q59, or bound the leakage of reference information into q59's record
  with a q59-only control.
- A success certifies Claim 4 under trusted SPAM. It doesn't show the witness uses it: report it
  as a channel capability.
- A prepare-and-measure alternative trusts only the inputs and needs no reference qubit:
  average fidelity above 2/3 over a uniform set of q75 inputs, or a semiquantum signalling game
  [Rosset2018]. Do it if q74 is unusable.

**For every stage.** Fit the classical intervention class to every new dataset, with its
likelihood against the calibrated quantum model. State the class's limits: d, map classes, and
stationarity across jobs.

## 6. Reproduction

```sh
uv run python experiments/phase8_classical.py --config experiments/configs/phase8_classical.toml
uv run pytest tests/test_classical.py
```

- The run writes `results/phase8_classical/<timestamp>/` (every fit's model, its log and
  the full summary) and the curated `results/summaries/phase8_classical/`.
- The curated summary's metadata records the commit and config. This run's commit is
  `f25caf6`; the docs were uncommitted then.
- Seeds: `[fit] seed = 30` (each start's seed is derived from it); `[analysis] seed = 30` (the
  record bootstrap).
- Every start of every class is listed under `starts` in `summary.json`, including those that
  did worse.
