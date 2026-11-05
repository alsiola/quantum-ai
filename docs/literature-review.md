# Literature review: measurement-conditioned quantum memory for sequence models

Phase 1a of the research campaign (issue 2; this review is issue 4). Compiled 2026-11-05.

## How to read this document

**Provenance.** Every citation here was checked against a source fetched on 2026-11-05: an arXiv abstract page or API record, a publisher page, the Crossref API, PMLR, or a vendor page. Nothing is cited from memory. Where a venue or detail could not be confirmed, the text says so. Each reference carries a read-depth tag:

| Tag | Meaning |
|---|---|
| **[FT]** | full text read |
| **[FT-t]** | full text, targeted: the PDF was downloaded and the relevant passages (model, method, processes, numbers) were read, not the whole paper |
| **[Abs]** | abstract only |
| **[Meta]** | bibliographic record only, cited for a standard, well-known result |

Claims are attributed with keys such as `[Franceschetto2026]`, which resolve in the reference list (§7) and in `docs/references.bib`. Statements marked *(our analysis)* are this review's own reasoning, not a cited result.

**Model under review.** For reference, the model the campaign proposes (issue 2, §2) is:

- A memory density matrix ρ_t.
- A learned unitary couples the memory to a fresh ancilla, and the ancilla is measured.
- The outcome y is emitted with p(y) = Tr(K_y ρ K_y†).
- The memory is updated to ρ' = K_y ρ K_y† / p(y), with Σ_y K_y†K_y = I.
- Training uses the sequence negative log-likelihood (NLL), with a continuous measurement strength.

---

## 1. Summary

**What already exists.**

1. **The model class is old.**
   - An instrument {K_y} that both emits a symbol and updates a quantum memory is a *hidden quantum Markov model* (HQMM) [Monras2011]. Equivalently it is a quantum stochastic simulator in computational mechanics [Gu2012; Binder2018; Liu2019].
   - Our exact circuit form (persistent memory ⊗ fresh ancilla → unitary U → measure the ancilla, so K_y = ⟨y|U|0⟩) is the standard "unitary quantum causal model" of that literature [Binder2018; Liu2019].
   - It also appears as the Stinespring circuit for QHMMs with mid-circuit measurement [Markov2022].
   - It is the non-adaptive baseline of [Stefanski2025], trained on real limit-order-book sequences. Their "long-memory" extension adds a *classical* controller that conditions the measurement basis on past outcomes.
   - In tensor-network language it is a sequentially generated MPS / finitely correlated state [Schon2005; Fannes1992]. With mixed states or several Kraus operators per symbol it is a locally purified state [Glasser2019].
2. **Training it by likelihood is established.**
   - Kraus operators have been fitted by maximum likelihood with Givens-rotation search [Srinivasan2018], Stiefel-manifold gradient descent [Adhikary2020] and Newton–Schulz retraction [Ning2026].
   - [Yang2021] trains K_y = ⟨y|U|0⟩ at fixed qubit memory by minimising −log P(x₀:L) with Adam, builds U, and runs it on IBM hardware (ibmq_athens).
   - Recurrent quantum memories have also been trained from data with a recurrent parameter-shift rule [Wang2026] and variationally compressed [Lyu2026]. In the QRNN literature, [Bausch2020] generates tokens with outcome-induced collapse of a quantum cell state, and trains by postselecting on the target token.
3. **A quantum memory advantage is proven and demonstrated, for specific processes.**
   - Quantum models of some classical processes need strictly less memory than the optimal classical (ε-machine) model, sometimes unboundedly less. Examples are the perturbed coin, dual-Poisson and periodically modulated renewal processes [Gu2012; Elliott2020; Wu2023].
   - This has been shown on photonic hardware [Palsson2017; Ghafari2019PRX; Ghafari2019NC; Wu2023] and on IBM hardware [Yang2021].
4. **Measurement backaction as a tunable resource is established in quantum reservoir computing (QRC).**
   - [Franceschetto2026] (PRX, April 2026) shows that tuning measurement strength g in a weakly monitored reservoir improves memory and forecasting.
   - [MorguiSancho2026] (arXiv, July 2026) unifies projective, weak, partial and dissipative monitoring through indirect (ancilla) measurement, with the same Kraus update as ours. Its measurement-strength angle θ is essentially our parameterisation.
   - In both, however, the reservoir is **fixed**, only a linear readout is trained, the task is **input-driven** rather than generative, and the dynamics analysed are the **outcome-averaged** channel.
5. **The instrument model is classically simulable at our scale.**
   - With n memory qubits it is exactly a tensor network of bond dimension 2ⁿ [Schon2005; Glasser2019].
   - No computational quantum advantage can be claimed at 2–8 qubits.
   - The defensible quantum claims are resource claims: memory dimension or entropy versus classical models of equal size.

**Where the open questions are.**

1. **Measurement strength.** No paper we found treats measurement strength as a continuous, *trained* parameter of a *generative* instrument, or asks whether weak measurement improves learned memory.
2. **Ablations.** No paper compares the conditioned update (ablation A), its outcome-averaged channel (B) and reset (C) under matched training, as a test of whether backaction *learned for prediction* is useful memory. In a generative model, B is not a valid sequence model on its own; see §3.
3. **Retention.** How long a learned instrument's memory survives is untested. Monitored-circuit theory predicts that a small register purifies on a time of order its Hilbert-space dimension [Fidkowski2021; Gullans2020PRX].
4. **Learning beyond toy targets.** Learning *from data* (rather than compiling a known HMM) to reach the known quantum memory bounds has only been shown for 1–2 qubit memories [Yang2021; Wang2026]. Whether gradient training finds the provably minimal quantum models is open at 2–4 qubits.
5. **Long-horizon hardware runs.** Hardware runs with coherent memory persisting across many measured outputs are rare: 3 steps photonically [Ghafari2019NC]. Mid-circuit measurement and reset errors on spectator memory qubits are a known obstacle [Govia2023; Rudinger2022].

---

## 2. Per-topic review

### 2.1 The four seed references

All four exist, and their titles, venues and DOIs match the brief. All four were read in full. The ACM site blocks automated access, so the project owner supplied the ICAIF 2025 PDF.

#### 2.1.1 Harnessing Quantum Backaction for Time-Series Processing [Franceschetto2026] [FT]

**Citation.** G. Franceschetto, M. Płodzień, M. Lewenstein, A. Acín, P. Mujal. *Phys. Rev. X* **16**, 021002, published 2 April 2026. doi:10.1103/j7f9-hfsj; arXiv:2411.03979. The citation matches the brief, verified on Crossref and arXiv. Read from the arXiv v3 full text.

**Model.**
- A fixed reservoir of N = 6 spins with a random all-to-all transverse-field Ising Hamiltonian.
- Input s_k is encoded by resetting an input spin.
- Measurement is an indirect Gaussian-pointer measurement with **strength g**. The Kraus operator is ∝ e^{−(V−g)²/4}|0⟩⟨0| + e^{−(V+g)²/4}|1⟩⟨1|, which becomes projective at large g.

**The outcome does not condition the state.** The simulations use the outcome-*averaged* state, which the authors say is equivalent to a dephasing channel. Features are expectation values estimated over many shots. Only a linear readout is trained, by regression. g and h are grid-scanned.

**Tasks and results.**
- Tasks: Santa Fe laser forecasting and the short-term-memory (STM) capacity task.
- An "online" weak-measurement protocol improves on restart-based protocols. The best ratios are P_R = 2.18 (forecasting) and 27.3 (STM) at intermediate g (g* ≈ 0.36–0.49).
- With finite shots (1.5·10⁶ per point) the gains fall to 1.16 and 1.42.
- It beats a classical-feedback QRC baseline at long delays. The authors attribute this to retained coherences acting as quantum memory.
- Simulation only; no hardware.

**What the brief implies and what the paper says.** The brief cites this paper as evidence that measurement backaction can be harnessed for temporal processing, and that holds. However, the paper uses backaction only as *tunable dephasing of a fixed reservoir*. It has no trained instrument, no generation, and no single-shot conditioning of memory on outcomes.

#### 2.1.2 General Theory of Monitored Quantum Reservoir Computing [MorguiSancho2026] [FT]

**Citation.** O. Morguí-Sancho, G. Manzano, G. L. Giorgi, R. Zambrini. arXiv:2607.14311 (v1 15 July 2026, v2 23 July 2026). It is a preprint: no journal version was found on Crossref. The abstract wording differs slightly between v1 and v2. Read from the v2 full text.

**Model.**
- An indirect measurement: the system couples to a probe via U_PS, and the probe is measured projectively.
- This gives Ω_m = ⟨φ_m|U_PS|0⟩ and the update ρ^(m) = Ω_m ρ Ω_m†/Pr(m). That is exactly our update rule.
- Schemes covered:
  - tunable dephasing, CNOT·RY(θ), with **strength angle θ**;
  - amplitude damping (θ = π is measure-and-reset);
  - partial projective measurement, with or without reset.

**Theory.**
- The paper gives conditions on the *non-selective* channel for the echo-state property, fading memory and input separability. Unital maps are excluded.
- Theorem 1 gives conditions under which strict contractivity arises from composing maps that are not individually contractive.
- Table 1 classifies which pairs of scheme and dynamics are viable.

**Results.**
- Systems of N = 5 qubits, with STM capacity and NARMA-n tasks.
- Projective monitoring is best at short delays. Weak monitoring is best at long memory. Partial measurement with reset is worst.
- Projective measurement saturates at about 10⁶ shots; weak needs 10⁸–10⁹.
- The trained part is a linear readout; the reservoir is fixed. Simulation only.

**Relevance.** This paper gives the formalism and the θ-parameterisation we need for the measurement-strength sweep (§2.5). It also warns that different monitoring schemes are "qualitatively distinct routes to computational capability" rather than interchangeable knobs.

#### 2.1.3 Quantum Dimension Reduction of Hidden Markov Models [Sundar2026] [FT]

**Citation.** R. Sundar, T. J. Elliott. *Phys. Rev. A* **113**, 062461, published 25 June 2026. doi:10.1103/plt9-cm1x; arXiv:2601.16126. The citation matches the brief. Read from the v2 full text.

**Model.** A generative quantum HMM: an instrument {E_x} on a memory, with ρ_{t+1} = E_x(ρ_t)/Tr E_x(ρ_t). This is our model class.

**Method.** The paper *compresses an existing classical HMM*; it does not learn from data:
1. Dilate the HMM to a unifilar one with an auxiliary label alphabet.
2. Build the q-sample iMPS.
3. Variationally truncate the bond dimension.
4. Read off the Kraus operators.

**Results.**
- On an HMM fitted to Google Speech Commands, compressed quantum models at memory dimension ≲ 20 have much lower co-emission divergence than greedy classical state merging, "often by orders of magnitude".
- The authors flag that the classical baseline is deliberately simple. They list learning directly from data, and stronger classical baselines, as open.
- There is no measurement-strength parameter and no hardware.

**Relevance.** This is a direct competitor on the "smaller quantum memory than classical" claim, and a source of the co-emission divergence rate (CDR) metric.

#### 2.1.4 Adaptive Quantum Channels as Long-Memory Generative Models [Stefanski2025] [FT]

**Citation.** C. Stefanski, V. Markov, D. Novak (Wells Fargo) and V. Rastunkov (IBM Quantum). *Proc. 6th ACM International Conference on AI in Finance (ICAIF '25)*, pp. 492–500, 2025. doi:10.1145/3768292.3770440.
- The citation matches the brief, verified on Crossref, Semantic Scholar, OpenAlex and the IBM Research publication page.
- The paper is open access (CC BY-NC-ND). The ACM site blocks automated access, so the project owner supplied the PDF, which was read in full.
- There is no arXiv version.
- Some web-search summaries wrongly attribute the paper to other authors and to *Phil. Trans. R. Soc. A*. That attribution comes from a "related" sidebar.

**Base model (their §4).** This is the instrument model of issue 2 §2, implemented as a circuit:
- An n-qubit system register and an m-qubit ancilla.
- A fixed trained unitary U (layers of RX/RY/RZ, circular CNOTs and controlled phases), and a trained initial-state unitary U₀.
- Each ancilla qubit is measured *projectively* in a trainable rotated basis (θᵢ, φᵢ).
- The outcome bitstring is the emitted symbol. The ancilla is reset, and the system state becomes K_e ρ K_e†/P(e), with K_e = (I ⊗ ⟨e|) U (I ⊗ |e₀⟩).
- The authors present this as the existing "quantum channels as generative models" of [Markov2022] and [Markov2024], generalised by the trainable measurement basis.
- *(Our analysis)* A fixed rotated-basis measurement is equivalent to a local unitary on the ancilla followed by a computational-basis measurement. In the non-adaptive case it can therefore be absorbed into U.

**Adaptive extension (their §5).** A **classical** controller sets per-step corrections to the measurement-basis angles:
- The corrections are Δθᵗ, Δφᵗ = b + W·Hᵗ, plus optional second-order terms W̃·χᵗ (products of adjacent past outcomes, an "agreement detector").
- The inputs are the last k outcomes, re-encoded as ±1.
- The resulting Kraus operators depend explicitly on the classical history.
- The authors call these "long-memory" models because history enters through this classical channel, *in addition to* the quantum state.

**Analysis.** k-step "space–time density matrices" built from Choi operators, after [Milekhin2025] [Meta]. Two temporal-entanglement witnesses: p-imagitivity, and order-2 Tsallis entropy.

**Training.**
- The loss is a mean-squared error between empirical and model sequence probabilities, weighted by empirical probability, over all sequences up to length ℓ = 7: L = Σ P̂(o)(P̂(o) − P(o|P))².
- It is **not** a likelihood.
- The optimiser is described only as "classical optimization methods". The CCS concepts list evolutionary algorithms.

**Data and tasks.**
- AAPL Level-1 limit-order-book data at 1-second resolution. Trained on February 2025; evaluated on 3 March 2025.
- Three tasks:

  | Task | System qubits | Ancilla qubits | Controller lags |
  |---|---|---|---|
  | Binary mid-price direction | 3 | 1 | 6 |
  | Four volatility regimes | 4 | 2 | 6 |
  | Joint direction × regime | 5 | 3 | — |

**Results.** All results are reported as figures only. The authors report:
- the learned sequence distributions match the empirical ones (lengths 1–4, and 1–2 for larger alphabets);
- error falls as the controller's history length grows, and lag 0 (non-adaptive) is the worst;
- nonlinear controllers beat linear ones at every lag;
- the temporal-entanglement measures grow with lag.

**Limitations.**
- There are **no classical baselines**: no HMM, n-gram or RNN.
- Results are not reported across seeds, and there is no held-out likelihood.
- The only ablation is lag 0 versus lag > 0.
- There is no measurement-strength parameter; all measurements are projective.
- Simulation only. Hardware deployment is listed as future work.

**Relevance.**
- The paper's *non-adaptive baseline* (lag 0) is essentially our architecture A, already applied to real financial sequences, though trained with an MSE loss rather than NLL. It is further prior art for the model class (§3).
- Its *adaptive* model breaks our design rule of never substituting a classical hidden state for the quantum memory. Its gain over lag 0 is exactly the kind of result issue 2 §15 warns about. It shows that adding classical memory helps. It does not show that quantum backaction is a useful memory.
- For us, it is best treated as a hybrid **classical-feedback baseline**, comparable to [Kobayashi2024] and [Murauer2025] in QRC. It is not a competitor to the backaction-only hypothesis.

**Related work by the same group.**
- [Markov2022] [FT-t] (arXiv:2212.03796) gives the Stinespring/mid-circuit-measurement QHMM circuit. It trains with evolutionary structure search and derivative-free optimisers, and checks sequence probabilities on ibmq_montreal.
- [Markov2024] [Meta] (ICAIF '24) applies non-adaptive channels to mid-price movements.

### 2.2 Hidden quantum Markov models and their learning

- **Definition.** [Monras2011] [Abs] defines HQMMs by replacing the projective measurements of quantum finite-state generators with instruments. It links them to sequential read-out of matrix product states, and gives an example where the quantum description is more compact than the classical one.
- **Expressivity.**
  - [ONeill2012] [Abs]: 1-qubit HQMMs produce stronger correlations than 1-bit HMMs.
  - [Clark2015] [Abs]: open quantum systems with instantaneous feedback are HQMMs.
  - [Adhikary2020] [FT-t]: HQMMs are a subclass of observable operator models (OOMs) without the negative-probability problem. With one Kraus operator per symbol they are complex norm-observable operator models (NOOMs) [Zhao2010] [Meta].
  - [Srinivasan2020TN] [Abs] maps the hierarchy uMPS / Born machine / locally purified state onto PSR / HMM / NOOM / HQMM and onto weighted automata.
  - Existence and minimality: [MonrasWinter2016] [Abs] (the completely-positive realisation problem); [Fanizza2024] [Abs] (processes with no finite quantum model; qubit HQMMs with no finite classical model); [Riechers2025] [Abs] (minimal quantum dimension bounds).
- **Learning procedures** (all maximise sequence likelihood over Kraus operators that satisfy Σ K†K = I):

  | Paper | Method | Benchmarks and results |
  |---|---|---|
  | [Srinivasan2018] [FT-t] (AISTATS 2018, PMLR 84) | Kraus operators stacked into a matrix with orthonormal columns; random complex Givens rotations, angles fitted with fmincon. Slow. | "Probability clock": a 2-state HQMM matches the true model, while HMMs need about 8 states (DA 0.1630 vs 0.1631). Stern–Gerlach HQMM: a 6-state HMM is needed to match 2 quantum states. Metric is "description accuracy" (DA), a squashed per-symbol log-likelihood. |
  | [Adhikary2020] [FT-t] (AISTATS 2020, PMLR 108) | Autograd gradient of the log-likelihood with a Cayley retraction on the Stiefel manifold (COSM). | Much faster and better optima than Givens. Synthetic 6-state/6-symbol HMM and UCI Splice DNA. HQMMs beat HMMs at equal latent dimension but use more parameters. |
  | [Ning2026] [FT-t] (arXiv:2608.06554) | Newton–Schulz polar retraction. | Same benchmarks. Claims the first consistent HQMM win over EM-trained HMMs on non-quantum data (mean 38.5% over COSM). Single author, preprint: treat as unconfirmed. |

  Other learning work, all [Abs]:
  - [Srinivasan2018HSE]: kernel HQMMs.
  - [Javidian2023]: circular HQMMs as locally purified states.
  - [Li2024Quantum]: split HQMM from a conditional master equation.
  - [Ning2025]: robust derivative-free learning.
- **Gap relative to our proposal.** HQMM learning parameterises the instrument directly on the Stiefel manifold. It has no physical circuit or measurement-strength structure, no reset or outcome-averaged ablations, and no analysis of memory retention. Benchmarks are small (alphabets of 2–6 symbols, latent dimension 6 or less).

### 2.3 Quantum computational mechanics and quantum statistical memory

- **Memory advantage.** [Gu2012] [FT-t] (*Nat. Commun.* 3, 762; arXiv title "Occam's quantum razor") encodes causal states as non-orthogonal quantum states. The quantum memory C_q is less than the classical statistical complexity C_μ whenever the ε-machine is not already optimal.
  - Worked example, the **perturbed coin** with flip probability p: C_μ = 1 bit, while C_q = S(ρ) → 0 as p → ½. C_q < 0.1 bit for p > 0.4.
  - Refinements: q-machines with code length L, which saturate at the cryptic order [Mahoney2016] [FT-t]; and unbounded advantage for real-valued processes [Garner2017] [Abs].
- **Our architecture is the standard simulator.**
  - [Binder2018] [FT-t]: memory ⊗ fresh ancilla → U → measure the ancilla. The measurement emits the symbol and collapses the memory to the next quantum causal state. Example: the "upset gambler" process, with infinite Markov order, needs one qubit and one two-qubit unitary.
  - [Liu2019] [FT-t] shows "phase-enhanced" models are the most general unitary causal simulators. It finds that minimising entropy and minimising dimension conflict (symmetric 3-state quasi-cycle).
  - [Yang2018] [Abs]: quantum memory equals past–future entanglement of the associated MPS.
- **Dimensional advantage** (fewer qubits, not just less entropy):
  - [Ghafari2019PRX] [FT-t]: a 3-causal-state process fits in 1 qubit (photonic experiment).
  - [Elliott2020] [FT-t]: **dual-Poisson renewal**. Classical memory diverges as Δt → 0; the exact quantum model uses 1 qubit.
  - [Wu2023] [FT-t]: **periodically modulated decay (PMD)** renewal, Φ(n) = Γⁿ(1 − V sin²(nπ/N)). Classical D_μ = log₂N; quantum needs 1 qubit for all N. Photonic experiment for N = 3–8 with per-symbol KL < 10⁻² bits.
  - [Elliott2018] [FT-t]: continuous-time renewal processes.
  - [Elliott2021PRXQ] [FT-t]: quantum coarse-graining.
  - [Elliott2021PRA] [Abs]: non-deterministic HMMs.
  - [Yang2025npj] [Abs]: lossy q-sample compression.
- **Learning from data.**
  - [Ho2020] [FT-t]: robust inference of the quantum memory structure directly from time series.
  - **[Yang2021] [FT-t] is the closest prior work to Phase 2 of this campaign.** It fixes the qubit memory dimension, parameterises the Kraus operators A_x = ⟨x|U|0⟩, minimises −log P(x₀:L) with Adam, then constructs U. It reports better accuracy than the provably optimal classical model of equal dimension, including on ibmq_athens (d_q = 2, N = 3; 2.12σ). Targets: a 3-state asymmetric process, generalised quasi-cycles, uniform renewal.
  - [Wang2026] [FT-t] (arXiv:2603.24069): a recurrent circuit trained by a recurrent parameter-shift rule on co-emission distortion. Orders-of-magnitude better than Born machines in data-sparse regimes on uniform renewal processes. It targets coherent q-samples rather than measured prediction.
  - [Lyu2026] [FT-t] (arXiv:2603.09567): variational compression of recurrent quantum models. Cyclic random walks on 2ⁿ sites are compressed from 3–5 qubits to 1–2.
  - [Elliott2024] [Abs] embeds such simulators in continuous quantum trajectories. That is the nearest prior work to a continuous-measurement picture of the same models.
- **Experiments:**
  - [Palsson2017] [FT-t]: photonic, C_q = 0.05 ± 0.01 vs C = 1.
  - [Ghafari2019PRX] [FT-t]: photonic.
  - [Ghafari2019NC] [FT-t]: photonic, 3 coherent time steps, interference visibility 0.96.
  - [Wu2023] [FT-t]: photonic.
  - [Yang2021] [FT-t]: IBM superconducting.
  - [Markov2022] [FT-t]: IBM, sequence probabilities only.

### 2.4 Quantum reservoir computing and quantum recurrent networks

**Quantum reservoir computing (QRC).** QRC drives a fixed quantum system with the input sequence and trains only a linear readout on measured observables.

- **Origins and reviews:**
  - [Fujii2017] [Abs] proposed QRC; the published title is "Harnessing Disordered-*Ensemble* Quantum Dynamics…".
  - Review: [Mujal2021] [Abs].
  - Universality of dissipative reservoirs and IBM demonstrations: [Chen2020] [Abs].
  - Dissipation as a resource: [Sannia2024] [Abs].
  - Dynamical phase transitions: [MartinezPena2021] [Abs].
  - Quantumness versus short-term-memory (STM) capacity: [Gotting2023] [Abs]; the published title is "Exploring quantumness in quantum reservoir computing".
  - Sampling-noise limits and "eigentasks": [Hu2023] [Abs].
  - Non-Markovian memory enhancement: [Sannia2026] [Abs].
  - A memory–nonlinearity trade-off common to memory restriction, weak measurement, edge of chaos and dissipation: [Cindrak2026] [Abs].
- **Monitored, weak-measurement and backaction QRC:**
  - **[Mujal2023] [FT-t]** (*npj QI* 9, 16): 6-spin reservoir, online protocol with weak measurements of strength g.
    - Readout uses expectation values over 1.5·10⁶ trajectories.
    - With infinite shots the best is g → 0⁺. With finite shots, statistical error diverges as g → 0, so the optimal g grows. More shots means a weaker optimal measurement.
  - **[Yasuda2023] [FT-t]** (arXiv:2310.06706, unpublished): system plus ancilla on IBM hardware.
    - Each step does a mid-circuit QND measurement of the ancilla and resets it. Up to 120 qubits and 1000 steps.
    - The authors state the output depends only on the **outcome-averaged** state. That is precisely our ablation B.
    - Capacity peaks at an intermediate system–ancilla coupling.
  - **[Hu2024] [FT-t]** (NISQRC, *Nat. Commun.* 15, 7491) separates M persistent *memory* qubits from R *readout* qubits.
    - Readout qubits are measured projectively and reset every step.
    - Memory is shown to persist beyond the coherence time, on a 7-qubit device.
    - This is architecturally the closest QRC to our circuit, but the readout is trained, not the instrument.
  - **[Zhang2026] [Abs]** (*PRX* 16, 021027): with mid-circuit monitoring of a bath, information lifetime scales exponentially with system size. Without monitoring it grows at most linearly. Verified on IBM hardware. This bears directly on retention (§2.6).
  - Feedback variants:
    - [Kobayashi2024] [FT-t]: projective measurement of all qubits, with expectation values fed back classically. The memory is classical.
    - [Monomi2025] [FT-t]: weak measurement plus feedback, on ensemble systems.
    - [Murauer2025] [FT-t]: single-shot mid-circuit bits fed back classically, with qubits reset each step. The quantum memory is deliberately removed.
    - Partial-SWAP memory leaks on IBM: [Connerty2026a] [Abs].
    - Chaotic forecasting on NISQ hardware: [Connerty2026b] [Abs].
    - Few-atom cavity QRC with continuous measurement: [Zhu2025a; Zhu2025b] [Abs].
  - [Franceschetto2026] and [MorguiSancho2026] are covered in §2.1.
- **Memory-capacity metrics** used across this literature:
  - STM capacity C(τ), the squared correlation between the delayed input and the readout, summed to total capacity [Mujal2023; MorguiSancho2026].
  - Information processing capacity (IPC) [Dambre2012] [Meta].
  - Tasks: NARMA [Atiya2000] [Meta], Mackey-Glass [Mackey1977] [Meta], Santa Fe laser.
  - Shot-noise-aware resolvable capacity [Hu2023].
- **Takeaway for us.**
  - No monitored-QRC paper trains the instrument. All of them read out shot or ensemble averages, and the published analyses work with the outcome-averaged channel.
  - What QRC does give us:
    1. a weak/projective trade-off in memory capacity with an interior optimum;
    2. the warning that weak measurement is shot-expensive. For our model, NLL is computed exactly in simulation, but any hardware estimate of p(y) inherits this cost;
    3. capacity metrics that can be applied with a linear probe on ρ_t to compare the conditioned memory (A) with the averaged memory (B).

**Quantum recurrent neural networks (QRNNs).** These matter for ablation D: how each design splits memory update from readout.

| Model | Persistent memory | Measured / reset each step | Readout | Does the outcome condition the memory? | Training |
|---|---|---|---|---|---|
| [Takaki2021] [FT-t] | register A (3 qubits), never measured | register B: input encoding, measured, re-initialised | ⟨Z_B⟩ expectation (assumes an ensemble / NMR) | No: memory follows Tr_B, the averaged channel | BFGS on simulated density matrices; parameter-shift gradient in appendix |
| [Bausch2020] [FT-t] (NeurIPS 2020) | cell-state register | input/output lanes reset; neurons use repeat-until-success | token distribution from output-lane marginals; cross-entropy | **Yes, at inference:** measuring the output collapses the cell state ("we never actively modify the cell state"). In training, the output is **postselected on the target token**, i.e. teacher-forced conditioning. | PyTorch autograd on the simulation |
| [Chen2022QLSTM] [FT-t] (ICASSP 2022) | classical LSTM hidden and cell vectors | — | VQC expectation values | No quantum memory | — |
| [Li2023QRNN] [FT-t] (*Neural Networks* 166) | register H (history) | register D: measured and reset | P(first D qubit = 1), via partial trace | No | parameter shift + Adam |
| [Siemaszko2023] [FT-t] | memory qumodes (continuous variable) | I/O qumodes: measured, reset to vacuum | scaled measurement | No | — |
| [Nikoloska2023] [FT-t] | register A | register B, reset | "Stochastic QRNN" variant: single-shot projective outcomes that collapse the memory | Yes, at sampling | trained on per-step marginals of the *averaged* state ("pseudo-likelihood"), not the conditioned joint likelihood; input-driven |
| [Viqueira2025] [FT-t] | register M, never measured | exchange register E: measured, reset | expectations or shot outputs | No | density-matrix emulator; extended parameter-shift gradients and Hessians |
| [Balauca2025] [FT-t] (arXiv:2512.12710) | hidden register H | embedding register | separate U_pred; Z/ZZ expectations feed a classical linear head producing logits | No | SPSA on IBM hardware. Sampling bitstrings directly as tokens was tried and **abandoned as hard to train**. |

- [Bondarenko2023] [Abs] learns quantum-to-quantum processes with memory; it is not a classical-symbol generator.
- Further small QRNNs: [Chen2022RC] [Abs]; [Wu2026QRU] [Abs].

**Takeaway for us.**
1. Almost every QRNN separates a never-measured memory register from a measured-and-reset I/O register, and reads out expectation values. That is a ready-made template for **ablation D**:
   - the memory update U_upd(x_t) acts on H ⊗ E, then E is reset;
   - a *separate* readout U_read acts on H ⊗ O, and O's Born probabilities give the logits;
   - the sampled token re-enters only classically, as the next input;
   - during training and evaluation the readout must not condition H. Use the deferred-measurement / averaged state for H.
2. **[Bausch2020] is closer to our hypothesis than any QRC paper.** Its generative sampling uses outcome-induced collapse as the memory update, and training postselects on the target token, which amounts to the conditioned likelihood. Its focus, though, is quantum neurons with amplitude amplification, not instruments, measurement strength or memory analysis.
3. [Balauca2025] is a practical warning. Mapping tokens directly to measured bitstrings gave a hard optimisation landscape.

### 2.5 Weak measurement, instruments and the measurement-strength parameterisation

- **Instruments and Kraus form.** The instrument concept is due to Davies and Lewis [Davies1970] [Meta]; the operator-sum form to [Kraus1983] [Meta]. Textbook treatment: [Nielsen2010] [Meta]; trajectories and conditional states in [Wiseman2009] [Meta] and [Jacobs2014] [Meta].
- **Dilation.** Stinespring's theorem [Stinespring1955] [Meta] makes "unitary on memory ⊗ ancilla, then measure the ancilla" fully general. Every instrument arises this way, with an ancilla of dimension up to the number of Kraus operators. So the circuit construction in issue 2 §2 loses no generality, given a large enough ancilla.
- **Weak, continuous and repeated measurement.**
  - [JacobsSteck2006] [Abs] builds continuous measurement as the limit of repeated weak measurements, including the information–disturbance trade-off.
  - [Brun2002] [Abs] models trajectories with qubit probes that interact one at a time and are then measured. This is the closest textbook model of our architecture.
  - [OreshkovBrun2005] [Abs] shows that any generalised measurement decomposes into a sequence of weak ones. Restricting to weak steps therefore loses nothing in principle.
  - Trajectory methods: [Dalibard1992] [Meta]; [Carmichael1993] [Meta].
- **Information versus disturbance.** [FuchsPeres1996] [Abs] and [Banaszek2001] [Abs] give tight trade-offs between the information an outcome carries and the disturbance it causes.
- **How we parameterise measurement strength** *(our analysis, on the basis above).*
  - A coupling angle θ, such as a controlled-RY or partial swap between memory and ancilla (as in [MorguiSancho2026] and [Brun2002]), runs continuously from the identity (θ = 0: no information, no disturbance) to a projective measurement.
  - In a generative model, θ also sets how strongly the emitted symbol depends on the memory:
    - at small θ, emissions are nearly memory-independent;
    - at large θ, each emission nearly collapses the memory.
  - So θ should be implemented separately from the ancilla's local rotations, which set the marginal emission distribution. Otherwise "strength" and "emission bias" are confounded.
  - Note that Oreshkov–Brun universality means the set of reachable instruments does not grow with θ. θ is a *structural prior and diagnostic*, not added expressivity. Any benefit must show up in optimisation, generalisation or robustness.

### 2.6 Measurement-induced phase transitions: retention under repeated measurement

- **The transitions.** Monitored random circuits have an entanglement transition at a critical measurement rate [Li2018] [Abs]; [Skinner2019] [Abs]; review in [Fisher2023] [Abs]. It is also a **purification** transition [Gullans2020PRX] [Abs]:
  - in the pure phase, a mixed initial state purifies at a size-independent rate;
  - in the mixed phase, purification takes a time exponential in system size, with an error-protected subspace.
- **Scrambling helps retention.** Scrambling unitaries protect information against measurement acting as errors [Choi2020] [Abs].
- **Finite systems still forget.** [Fidkowski2021] [Abs] shows that even in the mixed phase a maximally mixed state purifies on a timescale of order the Hilbert-space dimension.
- **Monitoring can lengthen memory.** [Zhang2026] [Abs] (*PRX* 16, 021027) finds that mid-circuit monitoring of a bath makes information lifetime scale *exponentially* with system size, against at most linearly without monitoring. Measurement is not only a source of forgetting: what is measured matters.
- **A usable diagnostic.** The entropy of a reference qubit entangled with the system [Gullans2020PRL] [Abs] can be measured directly in our simulator.
- **Experiments:** [Noel2022] [Abs] (trapped ions); [GoogleQAI2023] [Abs] (superconducting, up to 70 qubits, using space–time duality to avoid mid-circuit measurement).
- **What this means for us** *(our analysis).*
  - These are thermodynamic-limit, random-circuit results; our registers are 2–8 qubits and learned. What carries over is a scale estimate: with strong measurement, information about the state from about D = 2ⁿ steps ago (4–16 steps for 2–4 qubits) should be largely lost.
  - Weak coupling (small θ) should lengthen this roughly as θ⁻², at the cost of each emission depending less on the memory. That scaling is a heuristic, not a cited result.
  - The mutual information between past and future emissions is also capped by the memory's Holevo capacity of n bits [Nielsen2010].
  - This gives a concrete, falsifiable prediction for the delayed-recall and retention sweeps (§5).

### 2.7 Quantum generative models for sequences, tensor networks, trainable channels

- **Born machines.** QCBMs [Benedetti2019] [Abs]; [Liu2018] [Abs] have no tractable likelihood and are trained with MMD or similar. Our instrument model has an exact likelihood in simulation.
- **MPS Born machines** [Han2018] [Abs]; uniform MPS for sequences [Miller2020] [Abs] (arXiv; venue not confirmed); [Bradley2020] [Abs].
- **The key classical-simulability results.**
  - Sequential coupling of an ancilla to a chain generates exactly the MPS [Schon2005] [Abs], i.e. finitely correlated states [Fannes1992] [Meta]. This holds equally for a memory register coupled to successive output qubits.
  - [Glasser2019] [FT-t] proves:
    - HMMs correspond to non-negative MPS;
    - Born machines are locally purified states (LPS) with purification dimension 1;
    - there are unbounded separations between these classes;
    - complex parameters can be arbitrarily more economical.
  - [FossFeig2021] [Abs] uses exactly the MPS–channel correspondence, with mid-circuit measurement and qubit reuse on trapped-ion hardware.
- **Consequence for us** *(our analysis).*
  - With a pure initial memory and one Kraus operator per symbol, P(x₁…x_T) = ‖K_{x_T}⋯K_{x₁}|ψ₀⟩‖². That is a complex MPS Born machine with bond dimension 2ⁿ.
  - With mixed states, or weak measurement giving several Kraus operators per symbol, it is an LPS / HQMM of bond dimension 2ⁿ.
  - Exact likelihood costs O(T·poly(2ⁿ)). The model is therefore *classically simulable* at the sizes we plan.
  - What the literature does support is a **representational** separation: a non-negative MPS (an HMM) may need many more states than a complex Born machine or LPS of the same bond dimension [Glasser2019; Gao2022].
  - A complex MPS / uMPS of bond dimension 2ⁿ is therefore the strongest "same model class, classical" baseline. It is essentially our model with the physical-circuit constraint removed.
- **Separations.** Gao et al. [Gao2022] [Abs] prove an unconditional expressivity separation between Bayesian networks and their minimal quantum extension, attributed to contextuality and non-locality.
- **Trainable channels.** Apart from the HQMM learning work in §2.2, [Markov2024] and [Stefanski2025], we found no verified paper on "variational quantum instruments" as such. Searches were rate-limited, so this gap is not conclusive. Quantum Boltzmann machines [Amin2018] [Meta] are not sequence models.

### 2.8 Hardware: mid-circuit measurement, reset, feedforward

All pages below were accessed on 2026-11-05.

**Papers.**
- **[Corcoles2021] [FT-t].** IBM superconducting dynamic circuits:
  - readout pulse 300 ns;
  - conditional reset applied at 585 ns;
  - 1.4 µs total latency for two measure+reset cycles, with residual reset error 0.01.
- **[Rudinger2022] [Abs].** Gate-set tomography of a mid-circuit measurement *as an instrument*:
  - total error 8.1 ± 1.4%;
  - post-measurement state fidelities 96.7% (outcome 0) and 93.7% (outcome 1).
- **[Govia2023] [Abs].** A benchmarking suite for mid-circuit measurements. On IBM hardware it detects errors that measurement induces on spectator qubits, and spectator–measured entangling errors. This is directly relevant because our memory qubits are spectators to the ancilla readout.
- **[Pino2021] [Abs].** Trapped-ion QCCD architecture with mid-circuit measurement.

**Vendor information.**
- **IBM.**
  - Qiskit dynamic circuits support mid-circuit `measure`, `reset`, `if_test` / `switch` and loops (IBM Quantum documentation, "classical feedforward and control flow").
  - The documentation's example measurement takes 2.7 µs with error 7.3·10⁻³; example T1 is about 175–200 µs and T2 about 120–180 µs.
  - IBM's Nov 2025 blog ("utility-scale dynamic circuits") reports about 600 ns feedforward latency, a faster `measure_2` instruction, and availability to Open Plan users.
  - The Open Plan gives up to 10 min of QPU time per 28 days.
- **Quantinuum H2** (product data sheet v4.00, Sep 2025):
  - 56 qubits, all-to-all connectivity;
  - mid-circuit measurement with conditional branching, and qubit reuse;
  - typical two-qubit error 1·10⁻³, SPAM 1·10⁻³, crosstalk on unmeasured qubits during mid-circuit measure/reset 5·10⁻⁶.
- **Quantinuum Helios** data sheet (June 2026): 98 qubits, two-qubit error 8·10⁻⁴.
- No operation durations were found on the Quantinuum pages.
- **Google Willow** (spec sheet, Dec 2024): repeated mid-circuit measurement error 0.77%. It is not publicly accessible.

**Feasibility for a ~20-token single-circuit run** with 2–4 memory qubits and one ancilla measured and reset each step *(our estimate)*.
- **IBM:** about 3–5 µs per step, so 60–100 µs in total, comparable to T2. Expect substantial decoherence of the memory and spectator errors by token 20. Mitigations: dynamical decoupling, and the controller's combined measure+reset instruction.
- **Trapped ions:** cleaner per step, roughly 5–10% total error over 20 steps. Wall-clock duration is not documented on the pages read.
- **Memory between jobs.** In every case the memory cannot persist between separate jobs, so a full sequence must be one circuit. This agrees with issue 2 §11.
- No hardware execution is proposed here. Issue 2 requires explicit human approval first.

---

## 3. Novelty assessment against the central hypothesis

**Hypothesis** (issue 2 §1): *a trainable quantum instrument can simultaneously produce a probability distribution over the next symbol and transform its persistent quantum memory into a useful state conditioned on the observed symbol.*

**Plain answer: the core idea is an HQMM under another name.** It is also the unitary quantum causal simulator of computational mechanics. It is *not* a monitored QRC: monitored QRC has a fixed reservoir, is input-driven, averages over outcomes and trains only a linear readout.

The parts of the proposal sort as follows.

| Component | Status | Evidence |
|---|---|---|
| **Physical mechanism:** measuring a coupled ancilla emits a symbol, and Kraus backaction updates the memory | **Existing.** Textbook quantum instruments and trajectories. | [Davies1970; Kraus1983; Brun2002; Wiseman2009] |
| **Model class:** instrument {K_y} as a generative sequence model, ρ' = K_yρK_y†/p(y) | **Existing.** This is the HQMM. | [Monras2011; Srinivasan2018; Adhikary2020; Sundar2026] |
| **Circuit form:** memory ⊗ fresh ancilla → U → measure the ancilla, K_y = ⟨y\|U\|0⟩ | **Existing.** Applied to real financial sequences in [Markov2024] and [Stefanski2025]. | [Binder2018; Liu2019; Markov2022; Yang2021; Stefanski2025] |
| **"Long memory" by conditioning the instrument on past outcomes through a *classical* controller** | **Existing** (a hybrid). This breaks our no-classical-hidden-state rule, so it is a baseline, not our hypothesis. | [Stefanski2025]; QRC analogues [Kobayashi2024; Murauer2025] |
| **Learning procedure:** maximise sequence likelihood under teacher forcing, gradients through exact simulation | **Existing.** Stiefel-manifold and Adam-based NLL training. A parameter-shift-style recurrent rule exists for a related cost. | [Adhikary2020; Yang2021; Ning2026; Wang2026] |
| **Generative QRNN sampling** in which the measured output is the token and its collapse is the memory update | **Existing**, though not framed as an instrument and with no strength or memory analysis | [Bausch2020]; [Nikoloska2023], stochastic QRNN variant |
| **Memory advantage over classical models** for specific processes | **Existing theory; demonstrated** on photonic and IBM hardware for 1–2 qubit memories | [Gu2012; Elliott2020; Wu2023; Palsson2017; Yang2021] |
| **Measurement strength as a tunable resource** for temporal memory | **Existing for fixed reservoirs** with outcome-averaged dynamics and a linear readout | [Franceschetto2026; MorguiSancho2026] |
| **Measurement strength as a continuous, *trained*, structural parameter of a *generative* instrument**, and whether weak measurement gives better learned memory than projective | **Not found.** Candidate new empirical finding. | Searches of all the above; [Stefanski2025] full text unread |
| **Matched ablations A / B / C / D**, isolating whether outcome-conditioned backaction learned for prediction is the useful memory | **Not found** as a controlled study. Candidate new empirical finding. | — |
| **Learned retention and forgetting curves vs θ and n**, tested against purification-time predictions | **Not found** for learned instruments. Candidate new empirical finding. | Theory: [Gullans2020PRX; Fidkowski2021] |
| **Gradient training reaching known minimal quantum memories** at 2–4 qubits, and trainability (barren plateaus, small-probability instability) of measured recurrent circuits | **Partially open.** Shown at 1–2 qubits. | [Yang2021; Wang2026; Lyu2026] |

**Consequences for what we may claim.**
1. **We must not claim a new architecture or a new learning procedure.** At most we can claim a particular *parameterisation*: an explicit coupling-strength angle in a physical dilation.
2. **We must not claim computational quantum advantage.** The model is a bond-dimension-2ⁿ tensor network and is classically simulable at our sizes (§2.7).
3. **What we can credibly add is empirical** (findings under controlled ablations), plus at most a modest methodological contribution.
4. **Ablation B needs care.** In a generative model, the outcome-averaged channel Σ_y K_y ρ K_y† gives a memory that ignores the observed history. Under teacher forcing, B therefore predicts the same marginal sequence regardless of the data's past. B is a *control for whether conditioning matters*. It is not a competing model, and any A-over-B gap is expected by construction. The informative comparisons are:
   - A against C (memory matters at all);
   - A against D (whether "update = readout" beats a separately parameterised update and readout);
   - A against classical HMMs and complex uMPS of equal state dimension (whether the quantum structure helps);
   - A as a function of θ (whether weak measurement helps).

   We flag this for the architecture issue.
5. **[Stefanski2025] does not change this assessment.** It has been read in full. Its quantum part is our model A, with a projective, rotated-basis measurement and an MSE loss. Its long-memory gains come from a classical history controller, and it reports no classical baselines.

---

## 4. Known theoretical results to test against or reproduce

These give exact targets: the classical and quantum memory sizes are known, so a learned model can be scored against an optimum rather than only against baselines.

| Process | Classical optimal memory | Quantum memory (exact) | Source | Suggested use |
|---|---|---|---|---|
| Perturbed coin, flip probability p | C_μ = 1 bit (2 causal states) | 1 qubit; C_q = S(ρ) → 0 as p → ½ (C_q < 0.1 for p > 0.4) | [Gu2012] | First sanity target. The learned model's memory entropy should approach C_q. |
| Switch-parity (two switches, parity output) | C = 1 bit | 1 qubit; photonic C_q = 0.05 ± 0.01 | [Palsson2017] | Hardware-validated target. |
| Ghafari 3-state post-processed coin; "asymmetric process" | D_μ = log₂3 (3 states) | 1 qubit (dimensional advantage) | [Ghafari2019PRX; Yang2021] | **Main dimensional test:** a 1-qubit learned instrument should beat any 2-state classical model. Directly reproduces [Yang2021]. |
| Upset gambler (infinite Markov order) | 2 causal states | 1 qubit, one 2-qubit unitary | [Binder2018] | Long-memory test at minimal size. |
| PMD renewal Φ(n) = Γⁿ(1 − V sin²(nπ/N)) | D_μ = log₂N | 1 qubit for all N | [Wu2023] | Scaling test: the classical state count needed grows with N; quantum stays at 1 qubit. Photonic data for N = 3–8. |
| Dual-Poisson renewal | → ∞ as Δt → 0 | 1 qubit | [Elliott2020] | Unbounded-advantage target (discretised). |
| Uniform renewal (ticks uniform on 0…N−1) | N states | lossy at small d_q | [Yang2021; Wang2026] | Matches IBM-run settings (N = 3, d_q = 2). |
| Cyclic random walk on N = 2ⁿ sites | — | exact needs n qubits; compressible to 1–2 | [Lyu2026] | Register-size / compression test. |
| R-k golden mean, Nemo | C_μ ≈ 1.435 for the R-k example in [Ho2020] | C_q(L) saturates at cryptic order (Nemo never saturates) | [Mahoney2016; Ho2020] | Infinite-Markov-order processes. |
| Probability clock (2 states) | HMM needs about 8 states to match | 2-dim HQMM | [Srinivasan2018] | Reproduces an HQMM-vs-HMM gap. |
| Stern–Gerlach HQMM (2 states, 6 symbols) | HMM needs 6 states | 2-dim HQMM | [Srinivasan2018] | As above, with a larger alphabet. |

**Other results to check against.**
- **Learning.** [Adhikary2020] and [Ning2026] report DA on a 6-state/6-symbol synthetic HMM and on UCI Splice. These numbers can be reproduced to validate our training loop against published HQMM learners.
- **Retention.** The prediction is a purification time of order 2ⁿ under strong measurement [Fidkowski2021], with reference-qubit entropy as the diagnostic [Gullans2020PRL].
- **Monitored QRC.** At short delays, projective monitoring wins on STM and NARMA; weak monitoring wins at long delays [MorguiSancho2026; Franceschetto2026]. This qualitative trade-off, now for a *learned* instrument, is the most direct test of the strength question.

Values not verified in this review (the even process and the simple non-unifilar source) are omitted on purpose. Their C_μ/C_q figures did not appear in the passages read.

---

## 5. Recommended benchmark processes and baselines

**Processes**, in the order to run them:

1. **Independent symbols** and **first-order Markov**: sanity checks, as in issue 2 §7.
2. **Perturbed coin** and **switch-parity**: 1-qubit targets with known C_q (§4).
3. **Ghafari / asymmetric 3-state process**: the dimensional-advantage test that reproduces [Yang2021].
4. **PMD renewal**, sweeping N from 3 to 8: register-scaling test against classical state count [Wu2023].
5. **Upset gambler** and **Nemo / R-k golden mean**: infinite-Markov-order processes for long memory.
6. **Delayed recall / copy at delay τ** and **parity of the last k symbols**: tasks that directly probe retention, to compare against the purification-time prediction (§2.6). These are not from the quantum literature; they come from issue 2 §7.
7. **Synthetic 6-state/6-symbol HMM** and **UCI Splice**: to compare against published HQMM learners [Adhikary2020; Ning2026].
8. Only then a small character-level corpus.

**Metrics.**
- Test NLL in bits per symbol, and its excess over the true entropy rate (known for every synthetic process above).
- KL divergence rate, or co-emission divergence rate as in [Sundar2026; Wang2026].
- For QRC-style comparisons, STM capacity per delay [MorguiSancho2026].
- Memory diagnostics: von Neumann entropy of the stationary memory state, compared with C_q; and reference-qubit purification time.

**Baselines.**
- **Classical, same state dimension:**
  - HMM with D = 2ⁿ states trained by EM with restarts;
  - n-gram;
  - small GRU and RNN.
- **Classical, same model class:** a complex uMPS / Born machine of bond dimension 2ⁿ, and a locally purified state [Glasser2019; Miller2020]. This is the most important baseline: it separates "quantum-circuit structure" from "complex linear model of this dimension".
- **Quantum, unconstrained instrument:** an HQMM on the Stiefel manifold trained as in [Adhikary2020]. This separates our physical-circuit parameterisation from a general instrument.
- **Hybrid classical-feedback baseline:** the non-adaptive channel plus a classical history controller, as in [Stefanski2025]. It measures how much a *classical* side memory adds on top of A, and should be reported apart from the quantum-only models.
- **Quantum ablations** (issue 2 §6):
  - A: conditioned;
  - B: outcome-averaged control (§3 caveat);
  - C: reset;
  - D: separate update and readout (see §2.4).
- **Provable classical optimum** at fixed dimension, where known [Yang2021]. This allows statements of the form "better than *any* classical model with d states", rather than only "better than our trained HMM".

---

## 6. Open questions this campaign could credibly address (ranked)

1. **Does measurement strength θ, treated as a continuous trained parameter, change what memory a learned generative instrument can acquire?**
   - Is the optimum interior (weak), as in fixed-reservoir QRC, or does it go to projective?
   - Does it depend on the required delay?
   - This is the clearest gap: monitored QRC studies strength only for fixed reservoirs, and HQMM learning has no strength parameter. It needs only 2–4 qubit simulations. The answer is publishable whichever way it comes out.
2. **Is "update = readout" a better inductive bias than a separately parameterised update and readout (A vs D), at matched parameters, state dimension and training budget?** And does A beat a complex uMPS of the same bond dimension, i.e. does the physical-circuit constraint help or hurt? This is the experiment that tests the hypothesis rather than the model class.
3. **Can gradient-trained instruments reach the known minimal quantum memories at 2–4 qubits from data alone?** Targets are the Ghafari, PMD and upset-gambler processes. Do they then beat the best classical model of equal dimension, extending [Yang2021] beyond 1 qubit?
4. **What are the retention and forgetting curves of learned instruments as a function of θ and n?** Do they follow the purification-time scale of order 2ⁿ, or can training find instruments with protected subspaces that remember longer, as the scrambling-protects-information picture suggests [Choi2020]?
5. **Trainability.** How do barren plateaus, near-zero p(y) instabilities and degenerate instruments scale with depth, n and sequence length for measured recurrent circuits? This is under-reported in the HQMM literature.
6. **Hardware feasibility for 20 or more tokens in one circuit.** How do the instrument errors measured by [Rudinger2022] and [Govia2023] translate into loss of the learned memory advantage? This is a paper-and-simulation question only, until hardware is approved.

---

## 7. References

Each entry gives the read-depth tag; arXiv IDs are given where they exist. Keys match `docs/references.bib`. Author lists were taken from the arXiv or Crossref record.

**Checked and corrected, or not confirmed:**
- The Fujii & Nakajima published title includes "Disordered-*Ensemble*".
- The published title of Götting et al. (PRA 2023) is "Exploring quantumness in quantum reservoir computing".
- The "Bondarenko & Feldmann" QRNN is actually Bondarenko, Salzmann & Schmiesing; Bondarenko & Feldmann wrote a quantum autoencoder paper.
- "Practical and scalable QRC" by Zhu et al. does not exist under that title. It corresponds to two separate papers, [Zhu2025a] and [Zhu2025b].
- Venues **not** confirmed, cited as arXiv: Miller et al. (u-MPS), Srinivasan–Downey–Boots (HSE-HQMM).
- No journal versions found: [Yang2021], [Markov2022], [Yasuda2023], [MorguiSancho2026].
- Not verified, so not cited: Naimark's original dilation paper, a separate "variational quantum instruments" literature, Quantinuum operation timings, and the C_μ/C_q values for the even process and the simple non-unifilar source.

- **[Adhikary2020]** [FT-t] Sandesh Adhikary, Siddarth Srinivasan, Geoff Gordon, Byron Boots. *Expressiveness and Learning of Hidden Quantum Markov Models*. AISTATS 2020, PMLR 108:4151-4161 (2020). arXiv:1912.02098
- **[Amin2018]** [Meta] Mohammad H. Amin, Evgeny Andriyash, Jason Rolfe, Bohdan Kulchytskyy, Roger Melko. *Quantum Boltzmann Machine*. Phys. Rev. X 8, 021050 (2018). doi:10.1103/PhysRevX.8.021050
- **[Atiya2000]** [Meta] A.F. Atiya, A.G. Parlos. *New results on recurrent network training: unifying the algorithms and accelerating convergence*. IEEE Trans. Neural Netw. 11, 697 (2000). doi:10.1109/72.846741
- **[Balauca2025]** [FT-t] Stefan Balauca, Ada-Astrid Balauca, Adrian Iftene. *Practical Hybrid Quantum Language Models with Observable Readout on Real Hardware*. arXiv preprint (2025). arXiv:2512.12710
- **[Banaszek2001]** [Abs] Konrad Banaszek. *Fidelity Balance in Quantum Operations*. Phys. Rev. Lett. 86, 1366 (2001). doi:10.1103/PhysRevLett.86.1366 · arXiv:quant-ph/0003123
- **[Bausch2020]** [FT-t] Johannes Bausch. *Recurrent Quantum Neural Networks*. Advances in Neural Information Processing Systems 33 (NeurIPS 2020) (2020). arXiv:2006.14619
- **[Benedetti2019]** [Abs] Marcello Benedetti, Delfina Garcia-Pintos, Oscar Perdomo, Vicente Leyton-Ortega, Yunseong Nam, Alejandro Perdomo-Ortiz. *A generative modeling approach for benchmarking and training shallow quantum circuits*. npj Quantum Inf. 5, 45 (2019). doi:10.1038/s41534-019-0157-8 · arXiv:1801.07686
- **[Binder2018]** [FT-t] Felix C. Binder, Jayne Thompson, Mile Gu. *Practical Unitary Simulator for Non-Markovian Complex Processes*. Phys. Rev. Lett. 120, 240502 (2018). doi:10.1103/PhysRevLett.120.240502 · arXiv:1709.02375
- **[Bondarenko2023]** [Abs] Dmytro Bondarenko, Robert Salzmann, Viktoria-S. Schmiesing. *Learning Quantum Processes with Memory -- Quantum Recurrent Neural Networks*. arXiv preprint (2023). arXiv:2301.08167
- **[Bradley2020]** [Abs] Tai-Danae Bradley, E. Miles Stoudenmire, John Terilla. *Modeling Sequences with Quantum States: A Look Under the Hood*. Mach. Learn.: Sci. Technol. 1, 035008 (2020). doi:10.1088/2632-2153/ab8731 · arXiv:1910.07425
- **[Brun2002]** [Abs] Todd A. Brun. *A simple model of quantum trajectories*. Am. J. Phys. 70, 719 (2002). doi:10.1119/1.1475328 · arXiv:quant-ph/0108132
- **[Carmichael1993]** [Meta] Howard Carmichael. *An Open Systems Approach to Quantum Optics*. Lecture Notes in Physics Monographs, Springer (1993). doi:10.1007/978-3-540-47620-7
- **[Chen2020]** [Abs] Jiayin Chen, Hendra I. Nurdin, Naoki Yamamoto. *Temporal Information Processing on Noisy Quantum Computers*. Phys. Rev. Applied 14, 024065 (2020). doi:10.1103/PhysRevApplied.14.024065 · arXiv:2001.09498
- **[Chen2022QLSTM]** [FT-t] Samuel Yen-Chi Chen, Shinjae Yoo, Yao-Lung L. Fang. *Quantum Long Short-Term Memory*. ICASSP 2022, pp. 8622-8626 (2022). doi:10.1109/ICASSP43922.2022.9747369 · arXiv:2009.01783
- **[Chen2022RC]** [Abs] Samuel Yen-Chi Chen, Daniel Fry, Amol Deshmukh, Vladimir Rastunkov, Charlee Stefanski. *Reservoir Computing via Quantum Recurrent Neural Networks*. arXiv preprint (2022). arXiv:2211.02612
- **[Choi2020]** [Abs] Soonwon Choi, Yimu Bao, Xiao-Liang Qi, Ehud Altman. *Quantum Error Correction in Scrambling Dynamics and Measurement-Induced Phase Transition*. Phys. Rev. Lett. 125, 030505 (2020). doi:10.1103/PhysRevLett.125.030505 · arXiv:1903.05124
- **[Cindrak2026]** [Abs] Saud Čindrak, Lara Giebeler, Niclas Götting, Christopher Gies, Kathy Lüdge. *Memory-Nonlinearity Trade-off across Quantum Reservoir Computing Frameworks*. arXiv preprint (2026). arXiv:2603.21371
- **[Clark2015]** [Abs] Lewis A. Clark, Wei Huang, Thomas M. Barlow, Almut Beige. *Hidden Quantum Markov Models and Open Quantum Systems with Instantaneous Feedback*. ISCS 2014, Emergence, Complexity and Computation vol. 14, Springer, p. 143 (2015). doi:10.1007/978-3-319-10759-2_16 · arXiv:1406.5847
- **[Connerty2026a]** [Abs] Erik L. Connerty, Ethan N. Evans. *Controllable Quantum Memory Capacity in Quantum Reservoir Networks with Tunable partial-SWAPs*. arXiv preprint (2026). arXiv:2605.12713
- **[Connerty2026b]** [Abs] Erik L. Connerty, Ethan N. Evans, Gerasimos Angelatos, Vignesh Narayanan. *Predicting chaotic dynamics on NISQ hardware with quantum reservoir networks*. Commun. Phys. 9, 268 (2026). doi:10.1038/s42005-026-02652-1 · arXiv:2505.06799
- **[Corcoles2021]** [FT-t] Antonio D. Córcoles, Maika Takita, Ken Inoue, Scott Lekuch, Zlatko K. Minev, Jerry M. Chow, Jay M. Gambetta. *Exploiting Dynamic Quantum Circuits in a Quantum Algorithm with Superconducting Qubits*. Phys. Rev. Lett. 127, 100501 (2021). doi:10.1103/PhysRevLett.127.100501 · arXiv:2102.01682
- **[Dalibard1992]** [Meta] Jean Dalibard, Yvan Castin, Klaus Mølmer. *Wave-function approach to dissipative processes in quantum optics*. Phys. Rev. Lett. 68, 580 (1992). doi:10.1103/PhysRevLett.68.580
- **[Dambre2012]** [Meta] Joni Dambre, David Verstraeten, Benjamin Schrauwen, Serge Massar. *Information Processing Capacity of Dynamical Systems*. Sci. Rep. 2, 514 (2012). doi:10.1038/srep00514
- **[Davies1970]** [Meta] E. B. Davies, J. T. Lewis. *An operational approach to quantum probability*. Commun. Math. Phys. 17, 239 (1970). doi:10.1007/BF01647093
- **[Elliott2018]** [FT-t] Thomas J. Elliott, Mile Gu. *Superior memory efficiency of quantum devices for the simulation of continuous-time stochastic processes*. npj Quantum Inf. 4, 18 (2018). doi:10.1038/s41534-018-0064-4 · arXiv:1704.04231
- **[Elliott2020]** [FT-t] Thomas J. Elliott, Chengran Yang, Felix C. Binder, Andrew J. P. Garner, Jayne Thompson, Mile Gu. *Extreme Dimensionality Reduction with Quantum Modeling*. Phys. Rev. Lett. 125, 260501 (2020). doi:10.1103/PhysRevLett.125.260501 · arXiv:1909.02817
- **[Elliott2021PRA]** [Abs] Thomas J. Elliott. *Memory compression and thermal efficiency of quantum implementations of nondeterministic hidden Markov models*. Phys. Rev. A 103, 052615 (2021). doi:10.1103/PhysRevA.103.052615 · arXiv:2105.06285
- **[Elliott2021PRXQ]** [FT-t] Thomas J. Elliott. *Quantum Coarse Graining for Extreme Dimension Reduction in Modeling Stochastic Temporal Dynamics*. PRX Quantum 2, 020342 (2021). doi:10.1103/PRXQuantum.2.020342 · arXiv:2105.06831
- **[Elliott2024]** [Abs] Thomas J. Elliott, Mile Gu. *Embedding memory-efficient stochastic simulators as quantum trajectories*. Phys. Rev. A 109, 022434 (2024). doi:10.1103/PhysRevA.109.022434 · arXiv:2402.04708
- **[Fanizza2024]** [Abs] Marco Fanizza, Josep Lumbreras, Andreas Winter. *Quantum theory in finite dimension cannot explain every general process with finite memory*. Commun. Math. Phys. 405, 50 (2024). doi:10.1007/s00220-023-04913-4 · arXiv:2209.11225
- **[Fannes1992]** [Meta] M. Fannes, B. Nachtergaele, R. F. Werner. *Finitely correlated states on quantum spin chains*. Commun. Math. Phys. 144, 443 (1992). doi:10.1007/BF02099178
- **[Fidkowski2021]** [Abs] Lukasz Fidkowski, Jeongwan Haah, Matthew B. Hastings. *How Dynamical Quantum Memories Forget*. Quantum 5, 382 (2021). doi:10.22331/q-2021-01-17-382 · arXiv:2008.10611
- **[Fisher2023]** [Abs] Matthew P. A. Fisher, Vedika Khemani, Adam Nahum, Sagar Vijay. *Random Quantum Circuits*. Annu. Rev. Condens. Matter Phys. 14, 335 (2023). doi:10.1146/annurev-conmatphys-031720-030658 · arXiv:2207.14280
- **[FossFeig2021]** [Abs] Michael Foss-Feig, David Hayes, Joan M. Dreiling, Caroline Figgatt, John P. Gaebler, Steven A. Moses, Juan M. Pino, Andrew C. Potter. *Holographic quantum algorithms for simulating correlated spin systems*. Phys. Rev. Research 3, 033002 (2021). doi:10.1103/PhysRevResearch.3.033002 · arXiv:2005.03023
- **[Franceschetto2026]** [FT] Giacomo Franceschetto, Marcin Płodzień, Maciej Lewenstein, Antonio Acín, Pere Mujal. *Harnessing Quantum Backaction for Time-Series Processing*. Phys. Rev. X 16, 021002 (2026). doi:10.1103/j7f9-hfsj · arXiv:2411.03979
- **[FuchsPeres1996]** [Abs] Christopher A. Fuchs, Asher Peres. *Quantum-state disturbance versus information gain: Uncertainty relations for quantum information*. Phys. Rev. A 53, 2038 (1996). doi:10.1103/PhysRevA.53.2038 · arXiv:quant-ph/9512023
- **[Fujii2017]** [Abs] Keisuke Fujii, Kohei Nakajima. *Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning*. Phys. Rev. Applied 8, 024030 (2017). doi:10.1103/PhysRevApplied.8.024030 · arXiv:1602.08159
- **[Gao2022]** [Abs] Xun Gao, Eric R. Anschuetz, Sheng-Tao Wang, J. Ignacio Cirac, Mikhail D. Lukin. *Enhancing Generative Models via Quantum Correlations*. Phys. Rev. X 12, 021037 (2022). doi:10.1103/PhysRevX.12.021037 · arXiv:2101.08354
- **[Garner2017]** [Abs] Andrew J. P. Garner, Qing Liu, Jayne Thompson, Vlatko Vedral, Mile Gu. *Provably unbounded memory advantage in stochastic simulation using quantum mechanics*. New J. Phys. 19, 103009 (2017). doi:10.1088/1367-2630/aa82df · arXiv:1609.04408
- **[Ghafari2019NC]** [FT-t] Farzad Ghafari, Nora Tischler, Carlo Di Franco, Jayne Thompson, Mile Gu, Geoff J. Pryde. *Interfering trajectories in experimental quantum-enhanced stochastic simulation*. Nat. Commun. 10, 1630 (2019). doi:10.1038/s41467-019-08951-2 · arXiv:1905.06953
- **[Ghafari2019PRX]** [FT-t] Farzad Ghafari, Nora Tischler, Jayne Thompson, Mile Gu, Lynden K. Shalm, Varun B. Verma, Sae Woo Nam, Raj B. Patel, Howard M. Wiseman, Geoff J. Pryde. *Dimensional Quantum Memory Advantage in the Simulation of Stochastic Processes*. Phys. Rev. X 9, 041013 (2019). doi:10.1103/PhysRevX.9.041013 · arXiv:1812.04251
- **[Glasser2019]** [FT-t] Ivan Glasser, Ryan Sweke, Nicola Pancotti, Jens Eisert, J. Ignacio Cirac. *Expressive power of tensor-network factorizations for probabilistic modeling*. Advances in Neural Information Processing Systems 32 (NeurIPS 2019) (2019). arXiv:1907.03741
- **[GoogleQAI2023]** [Abs] Google Quantum AI and Collaborators: Jesse C. Hoke, Matteo Ippoliti, Eliott Rosenberg, and others. *Measurement-induced entanglement and teleportation on a noisy quantum processor*. Nature 622, 481 (2023). doi:10.1038/s41586-023-06505-7 · arXiv:2303.04792
- **[Gotting2023]** [Abs] Niclas Götting, Frederik Lohof, Christopher Gies. *Exploring quantumness in quantum reservoir computing*. Phys. Rev. A 108, 052427 (2023). doi:10.1103/PhysRevA.108.052427 · arXiv:2302.03595
- **[Govia2023]** [Abs] L. C. G. Govia, P. Jurcevic, C. J. Wood, N. Kanazawa, S. T. Merkel, D. C. McKay. *A randomized benchmarking suite for mid-circuit measurements*. New J. Phys. 25, 123016 (2023). doi:10.1088/1367-2630/ad0e19 · arXiv:2207.04836
- **[Gu2012]** [FT-t] Mile Gu, Karoline Wiesner, Elisabeth Rieper, Vlatko Vedral. *Quantum mechanics can reduce the complexity of classical models*. Nat. Commun. 3, 762 (2012). doi:10.1038/ncomms1761 · arXiv:1102.1994
- **[Gullans2020PRL]** [Abs] Michael J. Gullans, David A. Huse. *Scalable Probes of Measurement-Induced Criticality*. Phys. Rev. Lett. 125, 070606 (2020). doi:10.1103/PhysRevLett.125.070606 · arXiv:1910.00020
- **[Gullans2020PRX]** [Abs] Michael J. Gullans, David A. Huse. *Dynamical Purification Phase Transition Induced by Quantum Measurements*. Phys. Rev. X 10, 041020 (2020). doi:10.1103/PhysRevX.10.041020 · arXiv:1905.05195
- **[Han2018]** [Abs] Zhao-Yu Han, Jun Wang, Heng Fan, Lei Wang, Pan Zhang. *Unsupervised Generative Modeling Using Matrix Product States*. Phys. Rev. X 8, 031012 (2018). doi:10.1103/PhysRevX.8.031012 · arXiv:1709.01662
- **[Ho2020]** [FT-t] Matthew Ho, Mile Gu, Thomas J. Elliott. *Robust inference of memory structure for efficient quantum modeling of stochastic processes*. Phys. Rev. A 101, 032327 (2020). doi:10.1103/PhysRevA.101.032327 · arXiv:1911.02767
- **[Hu2023]** [Abs] Fangjun Hu, Gerasimos Angelatos, Saeed A. Khan, Marti Vives, Esin Türeci, Leon Bello, Graham E. Rowlands, Guilhem J. Ribeill, Hakan E. Türeci. *Tackling Sampling Noise in Physical Systems for Machine Learning Applications: Fundamental Limits and Eigentasks*. Phys. Rev. X 13, 041020 (2023). doi:10.1103/PhysRevX.13.041020 · arXiv:2307.16083
- **[Hu2024]** [FT-t] Fangjun Hu, Saeed A. Khan, Nicholas T. Bronn, Gerasimos Angelatos, Graham E. Rowlands, Guilhem J. Ribeill, Hakan E. Türeci. *Overcoming the coherence time barrier in quantum machine learning on temporal data*. Nat. Commun. 15, 7491 (2024). doi:10.1038/s41467-024-51162-7 · arXiv:2312.16165
- **[Jacobs2014]** [Meta] Kurt Jacobs. *Quantum Measurement Theory and its Applications*. Cambridge University Press (2014). doi:10.1017/CBO9781139179027
- **[JacobsSteck2006]** [Abs] Kurt Jacobs, Daniel A. Steck. *A straightforward introduction to continuous quantum measurement*. Contemp. Phys. 47, 279 (2006). doi:10.1080/00107510601101934 · arXiv:quant-ph/0611067
- **[Javidian2023]** [Abs] Mohammad Ali Javidian, Vaneet Aggarwal, Zubin Jacob. *Learning Circular Hidden Quantum Markov Models: A Tensor Network Approach*. IEEE Trans. Quantum Eng. (2023). doi:10.1109/TQE.2023.3319254 · arXiv:2111.01536
- **[Kobayashi2024]** [FT-t] Kaito Kobayashi, Keisuke Fujii, Naoki Yamamoto. *Feedback-Driven Quantum Reservoir Computing for Time-Series Analysis*. PRX Quantum 5, 040325 (2024). doi:10.1103/PRXQuantum.5.040325 · arXiv:2406.15783
- **[Kraus1983]** [Meta] K. Kraus (A. Böhm, J. D. Dollard, W. H. Wootters, eds.). *States, Effects, and Operations: Fundamental Notions of Quantum Theory*. Lecture Notes in Physics 190, Springer (1983). doi:10.1007/3-540-12732-1
- **[Li2018]** [Abs] Yaodong Li, Xiao Chen, Matthew P. A. Fisher. *Quantum Zeno effect and the many-body entanglement transition*. Phys. Rev. B 98, 205136 (2018). doi:10.1103/PhysRevB.98.205136 · arXiv:1808.06134
- **[Li2023QRNN]** [FT-t] Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng, Guoqiang Zhong, Yongjian Gu. *Quantum recurrent neural networks for sequential learning*. Neural Networks 166, 148 (2023). doi:10.1016/j.neunet.2023.07.003 · arXiv:2302.03244
- **[Li2024Quantum]** [Abs] Xiao-Yu Li, Qin-Sheng Zhu, Yong Hu, Hao Wu, Guo-Wu Yang, Lian-Hui Yu, Geng Chen. *A new quantum machine learning algorithm: split hidden quantum Markov model inspired by quantum conditional master equation*. Quantum 8, 1232 (2024). doi:10.22331/q-2024-01-24-1232 · arXiv:2307.08640
- **[Liu2018]** [Abs] Jin-Guo Liu, Lei Wang. *Differentiable learning of quantum circuit Born machines*. Phys. Rev. A 98, 062324 (2018). doi:10.1103/PhysRevA.98.062324 · arXiv:1804.04168
- **[Liu2019]** [FT-t] Qing Liu, Thomas J. Elliott, Felix C. Binder, Carlo Di Franco, Mile Gu. *Optimal stochastic modeling with unitary quantum dynamics*. Phys. Rev. A 99, 062110 (2019). doi:10.1103/PhysRevA.99.062110 · arXiv:1810.09668
- **[Lyu2026]** [FT-t] Chufan Lyu, Ximing Wang, Mile Gu, Thomas J. Elliott, Chengran Yang. *Variational Quantum Dimension Reduction for Recurrent Quantum Models*. arXiv preprint (2026). arXiv:2603.09567
- **[Mackey1977]** [Meta] Michael C. Mackey, Leon Glass. *Oscillation and chaos in physiological control systems*. Science 197, 287 (1977). doi:10.1126/science.267326
- **[Mahoney2016]** [FT-t] J. R. Mahoney, C. Aghamohammadi, J. P. Crutchfield. *Occam's Quantum Strop: Synchronizing and Compressing Classical Cryptic Processes via a Quantum Channel*. Sci. Rep. 6, 20495 (2016). doi:10.1038/srep20495 · arXiv:1508.02760
- **[Markov2022]** [FT-t] Vanio Markov, Vladimir Rastunkov, Amol Deshmukh, Daniel Fry, Charlee Stefanski. *Implementation and Learning of Quantum Hidden Markov Models*. arXiv preprint (2022). arXiv:2212.03796
- **[Markov2024]** [Meta] Vanio Slavov Markov, Vladimir Rastunkov, Juan I Adame. *Quantum Generative Models of Mid-Price Movement in Limit Order Books*. Proc. 5th ACM Int. Conf. on AI in Finance (ICAIF '24), pp. 737-744 (2024). doi:10.1145/3677052.3698679
- **[MartinezPena2021]** [Abs] Rodrigo Martínez-Peña, Gian Luca Giorgi, Johannes Nokkala, Miguel C. Soriano, Roberta Zambrini. *Dynamical Phase Transitions in Quantum Reservoir Computing*. Phys. Rev. Lett. 127, 100502 (2021). doi:10.1103/PhysRevLett.127.100502 · arXiv:2103.05348
- **[Milekhin2025]** [Meta] Alexey Milekhin, Zofia Adamska, John Preskill. *Observable and computable entanglement in time*. arXiv preprint (2025). arXiv:2502.12240
- **[Miller2020]** [Abs] Jacob Miller, Guillaume Rabusseau, John Terilla. *Tensor Networks for Probabilistic Sequence Modeling*. arXiv preprint (venue not verified) (2020). arXiv:2003.01039
- **[Monomi2025]** [FT-t] Tomoya Monomi, Wataru Setoyama, Yoshihiko Hasegawa. *Feedback-enhanced quantum reservoir computing with weak measurements*. arXiv preprint (2025). arXiv:2503.17939
- **[Monras2011]** [Abs] Alex Monras, Almut Beige, Karoline Wiesner. *Hidden Quantum Markov Models and non-adaptive read-out of many-body states*. Appl. Math. Comput. Sci. 3, 93 (2011). arXiv:1002.2337
- **[MonrasWinter2016]** [Abs] Alex Monràs, Andreas Winter. *Quantum learning of classical stochastic processes: The completely positive realization problem*. J. Math. Phys. 57, 015219 (2016). doi:10.1063/1.4936935 · arXiv:1412.3634
- **[MorguiSancho2026]** [FT] Oriol Morguí-Sancho, Gonzalo Manzano, Gian Luca Giorgi, Roberta Zambrini. *General theory of monitored Quantum Reservoir Computing*. arXiv preprint (2026). arXiv:2607.14311
- **[Mujal2021]** [Abs] Pere Mujal, Rodrigo Martínez-Peña, Johannes Nokkala, Jorge García-Beni, Gian Luca Giorgi, Miguel C. Soriano, Roberta Zambrini. *Opportunities in Quantum Reservoir Computing and Extreme Learning Machines*. Adv. Quantum Technol. 4, 2100027 (2021). doi:10.1002/qute.202100027 · arXiv:2102.11831
- **[Mujal2023]** [FT-t] Pere Mujal, Rodrigo Martínez-Peña, Gian Luca Giorgi, Miguel C. Soriano, Roberta Zambrini. *Time-series quantum reservoir computing with weak and projective measurements*. npj Quantum Inf. 9, 16 (2023). doi:10.1038/s41534-023-00682-z · arXiv:2205.06809
- **[Murauer2025]** [FT-t] Jakob Murauer, Rajiv Krishnakumar, Sabine Tornow, Michaela Geierhos. *Feedback Connections in Quantum Reservoir Computing with Mid-Circuit Measurements*. IEEE QCE 2025, pp. 1646-1652 (2025). doi:10.1109/QCE65121.2025.00182 · arXiv:2503.22380
- **[Nielsen2010]** [Meta] Michael A. Nielsen, Isaac L. Chuang. *Quantum Computation and Quantum Information (10th anniversary ed.)*. Cambridge University Press (2010). doi:10.1017/CBO9780511976667
- **[Nikoloska2023]** [FT-t] Ivana Nikoloska, Osvaldo Simeone, Leonardo Banchi, Petar Veličković. *Time-warping invariant quantum recurrent neural networks via quantum-classical adaptive gating*. Mach. Learn.: Sci. Technol. 4, 045038 (2023). doi:10.1088/2632-2153/acff39 · arXiv:2301.08173
- **[Ning2025]** [Abs] Ning Ning. *Robust Iterative Learning Hidden Quantum Markov Models*. arXiv preprint (2025). arXiv:2510.23237
- **[Ning2026]** [FT-t] Ning Ning. *Newton-Schulz Retraction-Based Inference Enables Hidden Quantum Markov Models to Outperform Classical HMMs*. arXiv preprint (2026). arXiv:2608.06554
- **[Noel2022]** [Abs] Crystal Noel, Pradeep Niroula, Daiwei Zhu, Andrew Risinger, Laird Egan, Debopriyo Biswas, Marko Cetina, Alexey V. Gorshkov, Michael J. Gullans, David A. Huse, Christopher Monroe. *Measurement-induced quantum phases realized in a trapped-ion quantum computer*. Nat. Phys. 18, 760 (2022). doi:10.1038/s41567-022-01619-7 · arXiv:2106.05881
- **[ONeill2012]** [Abs] Ben O'Neill, Tom M. Barlow, Dominik Šafránek, Almut Beige. *Hidden Quantum Markov Models with one qubit*. AIP Conf. Proc. 1479, 667 (2012). arXiv:1207.4304
- **[OreshkovBrun2005]** [Abs] Ognyan Oreshkov, Todd A. Brun. *Weak Measurements Are Universal*. Phys. Rev. Lett. 95, 110409 (2005). doi:10.1103/PhysRevLett.95.110409 · arXiv:quant-ph/0503017
- **[Palsson2017]** [FT-t] Matthew S. Palsson, Mile Gu, Joseph Ho, Howard M. Wiseman, G. J. Pryde. *Experimentally modeling stochastic processes with less memory by the use of a quantum processor*. Sci. Adv. 3, e1601302 (2017). doi:10.1126/sciadv.1601302 · arXiv:1602.05683
- **[Pino2021]** [Abs] J. M. Pino, J. M. Dreiling, C. Figgatt, J. P. Gaebler, S. A. Moses, M. S. Allman, C. H. Baldwin, M. Foss-Feig, D. Hayes, K. Mayer, C. Ryan-Anderson, B. Neyenhuis. *Demonstration of the trapped-ion quantum CCD computer architecture*. Nature 592, 209 (2021). doi:10.1038/s41586-021-03318-4 · arXiv:2003.01293
- **[Riechers2025]** [Abs] Paul M. Riechers, Thomas J. Elliott. *Identifiability and minimality bounds of quantum and post-quantum models of classical stochastic processes*. arXiv preprint (2025). arXiv:2509.03004
- **[Rudinger2022]** [Abs] Kenneth Rudinger, Guilhem J. Ribeill, Luke C. G. Govia, Matthew Ware, Erik Nielsen, Kevin Young, Thomas A. Ohki, Robin Blume-Kohout, Timothy Proctor. *Characterizing Midcircuit Measurements on a Superconducting Qubit Using Gate Set Tomography*. Phys. Rev. Applied 17, 014014 (2022). doi:10.1103/PhysRevApplied.17.014014 · arXiv:2103.03008
- **[Sannia2024]** [Abs] Antonio Sannia, Rodrigo Martínez-Peña, Miguel C. Soriano, Gian Luca Giorgi, Roberta Zambrini. *Dissipation as a resource for Quantum Reservoir Computing*. Quantum 8, 1291 (2024). doi:10.22331/q-2024-03-20-1291 · arXiv:2212.12078
- **[Sannia2026]** [Abs] Antonio Sannia, Ricard Ravell Rodríguez, Gian Luca Giorgi, Roberta Zambrini. *Non-Markovianity and memory enhancement in quantum reservoir computing*. npj Quantum Inf. 12, 111 (2026). doi:10.1038/s41534-026-01257-4
- **[Schon2005]** [Abs] C. Schön, E. Solano, F. Verstraete, J. I. Cirac, M. M. Wolf. *Sequential Generation of Entangled Multiqubit States*. Phys. Rev. Lett. 95, 110503 (2005). doi:10.1103/PhysRevLett.95.110503 · arXiv:quant-ph/0501096
- **[Siemaszko2023]** [FT-t] Michał Siemaszko, Adam Buraczewski, Bertrand Le Saux, Magdalena Stobińska. *Rapid training of quantum recurrent neural networks*. Quantum Mach. Intell. 5, 31 (2023). doi:10.1007/s42484-023-00117-0 · arXiv:2207.00378
- **[Skinner2019]** [Abs] Brian Skinner, Jonathan Ruhman, Adam Nahum. *Measurement-Induced Phase Transitions in the Dynamics of Entanglement*. Phys. Rev. X 9, 031009 (2019). doi:10.1103/PhysRevX.9.031009 · arXiv:1808.05953
- **[Srinivasan2018]** [FT-t] Siddarth Srinivasan, Geoff Gordon, Byron Boots. *Learning Hidden Quantum Markov Models*. AISTATS 2018, PMLR 84:1979-1987 (2018). arXiv:1710.09016
- **[Srinivasan2018HSE]** [Abs] Siddarth Srinivasan, Carlton Downey, Byron Boots. *Learning and Inference in Hilbert Space with Quantum Graphical Models*. arXiv preprint (arXiv comment says NIPS 2018; not verified) (2018). arXiv:1810.12369
- **[Srinivasan2020TN]** [Abs] Siddarth Srinivasan, Sandesh Adhikary, Jacob Miller, Guillaume Rabusseau, Byron Boots. *Quantum Tensor Networks, Stochastic Processes, and Weighted Automata*. arXiv preprint (2020). arXiv:2010.10653
- **[Stefanski2025]** [FT] Charlee Stefanski, Vanio Markov, David Novak, Vladimir Rastunkov. *Adaptive Quantum Channels as Long-Memory Generative Models*. Proc. 6th ACM Int. Conf. on AI in Finance (ICAIF '25), pp. 492-500 (2025). doi:10.1145/3768292.3770440
- **[Stinespring1955]** [Meta] W. Forrest Stinespring. *Positive Functions on C*-Algebras*. Proc. Amer. Math. Soc. 6, 211 (1955). doi:10.2307/2032342
- **[Sundar2026]** [FT] Rishi Sundar, Thomas J. Elliott. *Quantum dimension reduction of hidden Markov models*. Phys. Rev. A 113, 062461 (2026). doi:10.1103/plt9-cm1x · arXiv:2601.16126
- **[Takaki2021]** [FT-t] Yuto Takaki, Kosuke Mitarai, Makoto Negoro, Keisuke Fujii, Masahiro Kitagawa. *Learning temporal data with a variational quantum recurrent neural network*. Phys. Rev. A 103, 052414 (2021). doi:10.1103/PhysRevA.103.052414 · arXiv:2012.11242
- **[Viqueira2025]** [FT-t] José Daniel Viqueira, Daniel Faílde, Mariamo M. Juane, Andrés Gómez, David Mera. *Density matrix emulation of quantum recurrent neural networks for multivariate time series prediction*. Mach. Learn.: Sci. Technol. 6, 015023 (2025). doi:10.1088/2632-2153/ad9431 · arXiv:2310.20671
- **[Wang2026]** [FT-t] Ximing Wang, Chengran Yang, Chidambaram Aditya Somasundaram, Jayne Thompson, Mile Gu. *Learning Quantum-Samplers for Stochastic Processes with Quantum Sequence Models*. arXiv preprint (2026). arXiv:2603.24069
- **[Wiseman2009]** [Meta] Howard M. Wiseman, Gerard J. Milburn. *Quantum Measurement and Control*. Cambridge University Press (2009). doi:10.1017/CBO9780511813948
- **[Wu2023]** [FT-t] Kang-Da Wu, Chengran Yang, Ren-Dong He, Mile Gu, Guo-Yong Xiang, Chuan-Feng Li, Guang-Can Guo, Thomas J. Elliott. *Implementing quantum dimensionality reduction for non-Markovian stochastic simulation*. Nat. Commun. 14, 2624 (2023). doi:10.1038/s41467-023-37555-0 · arXiv:2208.12744
- **[Wu2026QRU]** [Abs] Tzong-Daw Wu, Hsi-Sheng Goan. *Quantum Recurrent Unit*. arXiv preprint (2026). arXiv:2601.18164
- **[Yang2018]** [Abs] Chengran Yang, Felix C. Binder, Varun Narasimhachar, Mile Gu. *Matrix Product States for Quantum Stochastic Modeling*. Phys. Rev. Lett. 121, 260602 (2018). doi:10.1103/PhysRevLett.121.260602 · arXiv:1803.08220
- **[Yang2021]** [FT-t] Chengran Yang, Andrew Garner, Feiyang Liu, Nora Tischler, Jayne Thompson, Man-Hong Yung, Mile Gu, Oscar Dahlsten. *Provable superior accuracy in machine learned quantum models*. arXiv preprint (2021). arXiv:2105.14434
- **[Yang2025npj]** [Abs] Chengran Yang, Marta Florido-Llinàs, Mile Gu, Thomas J. Elliott. *Dimension reduction in quantum sampling of stochastic processes*. npj Quantum Inf. 11, 34 (2025). doi:10.1038/s41534-025-00978-2 · arXiv:2404.10338
- **[Yasuda2023]** [FT-t] Toshiki Yasuda, Yudai Suzuki, Tomoyuki Kubota, Kohei Nakajima, Qi Gao, Wenlong Zhang, Satoshi Shimono, Atsushi Matsuo, Janani Ananthanarayanan, Hendra I. Nurdin, Naoki Yamamoto. *Quantum reservoir computing with repeated measurements on superconducting devices*. arXiv preprint (2023). arXiv:2310.06706
- **[Zhang2026]** [Abs] Bingzhi Zhang, Fangjun Hu, Runzhe Mo, Tianyang Chen, Hakan E. Türeci, Quntao Zhuang. *Scaling Laws of Quantum Information Lifetime in Monitored Quantum Dynamics*. Phys. Rev. X 16, 021027 (2026). doi:10.1103/7717-1mw2 · arXiv:2506.22755
- **[Zhao2010]** [Meta] Ming-Jie Zhao, Herbert Jaeger. *Norm-Observable Operator Models*. Neural Computation 22, 1927 (2010). doi:10.1162/neco.2010.03-09-983
- **[Zhu2025a]** [Abs] Chuanzhou Zhu, Peter J. Ehlers, Hendra I. Nurdin, Daniel Soh. *Practical few-atom quantum reservoir computing*. Phys. Rev. Research 7, 023290 (2025). doi:10.1103/wsyq-jyxd · arXiv:2405.04799
- **[Zhu2025b]** [Abs] Chuanzhou Zhu, Peter J. Ehlers, Hendra I. Nurdin, Daniel Soh. *Minimalistic and scalable quantum reservoir computing enhanced with feedback*. npj Quantum Inf. 11, 195 (2025). doi:10.1038/s41534-025-01144-4 · arXiv:2412.17817
