# Phase 4b: falsifying the candidate findings

Issue #13 (campaign #2, sections 10 and 15). Each candidate finding listed on #2 was attacked
with the arms the issue asks for. Each ends as **survives**, **qualified** or **refuted**.
The generated tables cover every arm, including those that changed nothing. They are in
`results/summaries/phase4b/{a,b,c}/summary.md`, next to `main.csv`, `reach.csv`,
`summary.json` and the plots (`a/gap_vs_params.png`, `b/recall.png`, `c/recall.png`,
`c/structural.csv`).

Reproduce (one config per finding; 4 niced workers each on a shared machine, run in
parallel: (a) 4.0 h wall, 16.2 CPU-h; (b) 5.5 h, 16.4 CPU-h; (c) 5.9 h, 5.9 CPU-h; 1,200
evaluation runs plus 360 tuning runs, 0 errors):

```
uv run python experiments/phase4b.py --config experiments/configs/phase4b_a.toml
uv run python experiments/phase4b.py --config experiments/configs/phase4b_b.toml
uv run python experiments/phase4b.py --config experiments/configs/phase4b_c.toml
```

`--from-run <dir>` reuses saved runs. Protocol: seeds 0–9 for every cell. Gradient arms with
an lr grid are tuned on validation (tuning seeds 100 and 101). A run fails if its test gap
(test NLL minus the source's NLL on the same sequences, nats/symbol) is ≥ 0.01. The
`marked_recall` arms use the Phase 3e falsification recipe (4× data, 3000 steps, lr 0.1).
On that task every model "fails" the 0.01 gap, so the recall accuracy, the cue horizon
(first lag with cue Holevo information below 0.1 bits, capped at 40) and the number of
seeds that keep the cue to lag 40 ("retain") are the measures. #12's initialisation and
restarts (`qmem.trainability`) were used for n = 3; `docs/phase4a-trainability.md` had not
landed when these ran.

New code: `qmem.circuits.PartialSwapBlock` (learnable ancilla coupling for D),
`qmem.baselines.GeneratorInstrumentModel` (HQMM restricted to K generators), and
`qmem.falsification` (differentiable η and a prior on it, staged training, the
protected-subspace search, the uMPS as an instrument). Each is tested against an analytic case
in `tests/test_falsification.py`.

## Verdicts

| finding | verdict |
|---|---|
| (a) A is parameter-efficient compared with D | **qualified**: survives against every D on `markov1_quaternary`; on binary tasks a better D ties A, and the HQMM restricted to A's count ties A, so there the efficiency belongs to the instrument class |
| (b) weak measurement beats projective readout for retention (feed-forward) | **qualified**: true and structural at n = 2 (shared g), learnable with a weak prior or a τ curriculum; does not hold at n = 3 |
| (c) a protected subspace forms only in the general layout | **refuted**: feed-forward variants with per-coupling g or an unread qubit learn it, and so do the HQMM (D = 4) and the uMPS (χ = 4) |

## (a) Parameter efficiency of A against D

**Other D parameterisations** (25 parameters on binary tasks, 34 on 4 symbols; A has 21 and 26):

| task | A | D_diss (#7) | D_diss, 2 RY-only layers | D_diss, CZ | D, learnable ancilla coupling |
|---|---|---|---|---|---|
| markov1_binary | 0/10, 0.0008 | 10/10, 0.081 | **0/10, 0.0002** | 10/10, 0.085 | **0/10, 0.0021** |
| even | 0/10, 0.0010 | 10/10, 0.037 | **3/10, 0.0011** | 10/10, 0.064 | 10/10, 0.015 |
| markov1_quaternary | 4/10, 0.0056 | 10/10, 0.194 | 10/10, 0.091 | 10/10, 0.218 | 10/10, 0.110 |

(failures, median gap.) The A–D gap on the binary tasks is specific to #7's D: a deeper
RY-only update ties A at four more parameters, and the partial-SWAP (amplitude-damping) D
solves `markov1_binary`. On `markov1_quaternary` every D variant fails every seed.

**Budget.** 4× steps and patience with a wider lr grid change nothing: D 0.100 → 0.100,
D_diss 0.081 → 0.079 and partial-SWAP D 0.0021 → 0.0019 on `markov1_binary`, and the same
pattern on the other two tasks. The gap is not a budget shortfall.

**NLL against parameters** (`a/gap_vs_params.png`; dots are seeds; the legend omits D (red)
and the partial-SWAP D (green)). Smallest count whose median over seeds reaches the optimum:

| task | A | HQMM | HQMM, K generators | uMPS | HMM | GRU | D_diss |
|---|---|---|---|---|---|---|---|
| nemo | 13 | 16 | 13 | 20 | 36 | 42 | never (17, 25) |
| probability_clock | 13 | 16 | 13 | 20 | 36 | 42 | never (17, 25) |
| markov1_quaternary | 26 (4/10 fail) | 128 | 52 (val-selected only; 6/10 fail) | 136 | 68 | 60 | never (24–34) |

**The HQMM at A's count** (`hqmm_gen`: V = exp(i Σ θ_k H_k) V₀ with K random Hermitian
generators). At K = A's count it reaches the optimum with 0/10 failures on
`markov1_binary`, `even`, `nemo` and `probability_clock`, as A does. On `markov1_quaternary`
it fails 10/10 at K = 26 (0.078) and 6/10 at K = 52, where A fails 4/10 at 26.

**Shown.** (1) #7's D (unitary and dissipative) fails at matched parameters whatever the
budget. (2) A D with a different update ties A on the binary tasks. (3) On the binary and
qubit-dimension tasks a 13–21 parameter instrument with no circuit structure is as efficient
as A, so the efficiency there belongs to the instrument class, not to update = readout in a
circuit. (4) On `markov1_quaternary` A's circuit beats every D, the restricted HQMM, the
HMM (68) and the GRU (60) on parameter count. It is still unreliable (4/10 fail; see #12).

**Speculative.** Whether a better choice of K generators than random ones closes the
`markov1_quaternary` gap; only random generators were tried.

## (b) Weak measurement and retention (`marked_recall`, τ = 4)

**Can a weak η be learned?** (n = 2, feed-forward, shared g; recall accuracy ± s.e. over
seeds, learned η, gap on `markov1_quaternary` for the same arm where it was run):

| arm | recall acc | η | horizon | markov1_quaternary |
|---|---|---|---|---|
| η = 0.6 fixed | 0.843 ± 0.020 | 0.60 | 4.5 | 10/10 fail, 0.110 |
| η = 0.9 fixed | 0.759 ± 0.010 | 0.90 | 2.9 | 10/10 fail, 0.022 |
| η free | 0.771 ± 0.016 | 0.75 | 3.6 | 0/10, 0.0006 (η → 1.0) |
| annealed (0.6 frozen, then free) | 0.817 ± 0.020 | 0.69 | 3.8 | 0/10, 0.0007 |
| prior 0.1 (η − 0.6)² | **0.857 ± 0.009** | 0.64 | 4.0 | **0/10, 0.0007** (η → 1.0) |
| prior 1 (η − 0.6)² | 0.815 ± 0.029 | 0.61 | 4.1 | 10/10 fail, 0.062 |
| curriculum τ 1 → 2 → 4, free | **0.852 ± 0.013** | 0.66 | 4.4 | not run |

The weak prior (λ = 0.1) and the τ curriculum reach the hand-set weak-η retention. The weak
prior does so without hurting `markov1_quaternary`: there the data overrides it and η goes
to 1. A strong prior (λ = 1) hurts `markov1_quaternary`. Annealing helps only partly,
because η drifts back up once it is freed (0.69). The curriculum was not run on the other
task.

**Why it helps: disturbance, not optimisation.** Four tests point the same way:

- *Soft targets* (fitting the source's exact predictive, with no sampling noise): 0.839 at η
  = 0.6 against 0.780 at 0.9. The gap is in what each strength can represent, not in noise.
- *Swaps*: the trained η = 0.9 model retrained at 0.6 reaches 0.869; the η = 0.6 model
  retrained at 0.9 drops to 0.761. The final η decides the outcome, not the path taken to it.
- *Disturbance* (1 − corrected operation fidelity): 0.306 at η = 0.6, 0.474 at 0.9. The
  per-distractor contraction of cue distinguishability is 0.74 against 0.37, and the
  commutator ‖[E_d, Δ]‖ is 0.29 against 0.37. Across the n = 2 feed-forward arms, recall falls roughly as disturbance
  rises.
- *Plausibly*, training drifts to strong η because the recall term is a small share of the NLL: the
  non-recall gap is lower at η = 0.9 (0.0060 against 0.0076).

**Other layouts.**

| layout | weak | strong | free |
|---|---|---|---|
| general, τ = 4 (prior to 0.3 / 0.9) | **0.913** (η 0.30) | 0.775 (η 0.89) | 0.853 (η 0.13) |
| general, τ = 16 | **0.719** | 0.548 | 0.681 |
| feed-forward n = 3, #12 init (fixed 0.6 / 0.9) | 0.926 | 0.916 | **0.938** (η 0.95) |

In the general layout a moderate η (0.3) beats near-projective readout, and also beats the
very weak η that free training finds. At n = 3 in the feed-forward layout, strength stops
mattering and free training (η → 0.95) is best: the qubit that no ancilla reads holds the
cue (see (c)).

**Shown.** At n = 2 in the shared-g feed-forward layout and in the general layout, weaker
readout retains longer, because it disturbs the stored bit less. A weak prior on η recovers
this without cost on a task that wants a projective readout. **Not true** at n = 3 in the
feed-forward layout, where a protected qubit makes strong readout harmless.

## (c) The protected subspace

**Trained models** (`marked_recall`, τ = 4; "subspace" is the best read contrast on a
2-dimensional subspace where both distractor effects equal the source's level (1 − q)/2 · I,
found on each trained instrument; mean over seeds, 0 = none found):

| model | params | recall acc | horizon | retain | contraction | subspace |
|---|---|---|---|---|---|---|
| feed-forward, shared g (reference) | 41 | 0.771 | 3.6 | 0/10 | 0.55 | 0 |
| general circuit (reference) | 42 | 0.853 | 33.0 | 8/10 | 1.00 | 0.11 |
| **feed-forward, per-coupling g** (n = 2) | 42 | 0.916 | 34.2 | 8/10 | 0.94 | 0 |
| **feed-forward n = 3, qubit 2 unread** | 61 | 0.938 | 41.0 | 10/10 | 1.00 | 0.38 |
| feed-forward n = 3, per-coupling g | 62 | 0.946 | 41.0 | 10/10 | 0.98 | 0.37 |
| HQMM D = 2 | 32 | 0.773 | 21.4 | 5/10 | 0.82 | 0 |
| HQMM D = 4 | 128 | 0.944 | 41.0 | 10/10 | 1.00 | 0.16 |
| uMPS χ = 4 | 136 | 0.916 | 37.1 | 9/10 | 0.98 | 0.15 |

**Structural search** (training-free, on the bare coupling over a grid of g;
`c/structural.csv`). If the distractor levels are free, every feed-forward layout has a
protected 2-dimensional subspace, with read contrast 2 sin⁴g / (1 + cos²g) for shared g
(found to 1e-11). At the source's distractor levels (τ = 4 and 16), **no n = 2 layout has
one**: 0 feasible starts out of 16 in every one of the 60 cells, shared or per-coupling g.
With an unread third qubit one exists for g ≥ 0.9, and its contrast approaches 2q (0.398 at
τ = 4, 0.117 at τ = 16, g = 1.5). An exact model needs 2q(1 − 2ε) = 0.36 at τ = 4, which
only near-projective coupling provides. Trained n = 3 models do go to η ≈ 0.95.

**Shown.** The feed-forward layout learns to protect the cue when given one unread qubit
(+20 parameters, η stays one number) or per-coupling strengths (+1 parameter, but η becomes
a function of two strengths, and the reported η is their combined operational strength). The
HQMM at D = 4 and the uMPS at χ = 4 learn it too. The HQMM at D = 2 only partly does. The
finding as stated is refuted: protection is not specific to the general layout.

**Speculative.** The n = 2 per-coupling model retains to lag 40 in 8/10 seeds although no
exact protected subspace exists at the source's levels. Its protection is approximate
(contraction 0.94; the Holevo information decays slowly enough to stay above 0.1 bits for
40 lags). Why shared g does not find a similar approximate solution was not established.

## Claims the Phase 5 report may make

1. #7's D and D_diss, at A's parameter count and with up to 4× the budget, do not learn the
   Phase 2 tasks; A does. A different D update (deeper RY-only, or a learnable damping
   coupling) ties A on binary first-order Markov tasks but not on `markov1_quaternary`.
2. On qubit-dimension tasks (`nemo`, `probability_clock`), A reaches the optimum with 13
   parameters, against HQMM 16, uMPS 20, HMM 36 and GRU 42. An HQMM restricted to 13 random
   generators does too, so on these tasks the efficiency is a property of small quantum
   instruments, not of A's circuit.
3. On `markov1_quaternary`, A (26 parameters) reaches the optimum in 6/10 seeds; no other
   family tried does at fewer than 52 parameters.
4. In the n = 2 feed-forward layout (shared g) and in the general layout, weaker readout
   (η ≈ 0.3–0.6) retains a cue longer than near-projective readout. The cause is lower
   disturbance of the stored bit, shown by strength swaps and soft-target fits. A weak
   quadratic prior on η learns this at no cost on `markov1_quaternary`.
5. This does not generalise to n = 3 in the feed-forward layout, where a qubit no ancilla
   reads stores the cue and strong readout is best.
6. A protected memory subspace is learned by every model class tried that has at least one
   memory qubit, or one coupling, free of the shared readout: the general A, the feed-forward
   A with per-coupling strengths or an unread qubit, the HQMM at D = 4 and the uMPS at χ = 4.
   It is not specific to the general layout.

Not supported: that update = readout in a circuit is *the* source of parameter efficiency;
that weak measurement helps retention in general; that the feed-forward layout cannot
protect memory.
