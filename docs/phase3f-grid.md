# Phase 3f: cross-model grid at matched state dimension and matched parameter count

Issue #11 (campaign #2, sections 9 and 15). It runs every model class on the #8 suite
(`docs/benchmarks.md`), where some sources need more classical states than quantum
dimensions, and runs the 1-qubit vs 2-state dimensional test on Ghafari and PMD. The
generated tables, with every evaluated cell, its tuning and its counts, are in
`results/summaries/phase3f/summary.md`. CSVs (`main.csv`, `matched_dim.csv`,
`matched_params.csv`, `reach.csv`, `dimensional.csv`, `suspect.csv`, `tuning.csv`,
`retention.csv`), `summary.json` and the plots `gap_vs_params.png` and `gap_vs_dim.png` sit
next to it.

Reproduce (it took 16 h on a shared, saturated machine with 6 niced workers; 7,102 runs,
95.6 CPU-hours summed over runs, 0 errors):

```
uv run python experiments/phase3f.py --config experiments/configs/phase3f.toml
```

`--from-run <dir>` reuses saved runs and trains only the missing ones; `--no-train`
re-summarises; `--max-dim 2` stops after the D = 2 wave.

**Headline.**

1. **Dimensional advantage, PMD N = 3–8.** The trained 1-qubit models (A in both layouts,
   HQMM, uMPS) reach long-sequence KL rates of 0.0001–0.001 nats/symbol. The best 2-state
   classical model we could fit (EM with 100 restarts, gradient, 16× data) sits at 0.0137
   (N = 3) down to 0.0018 (N = 8): 10–140× the best 1-qubit model, and 2–33× the weakest
   (A, general). Every 1-qubit model is also below the provable bound for *unifilar*
   2-state models (block bound 0.0076 down to 0.0010), except A (general) at N = 8, which
   ties it (0.00104 ± 0.00009 against 0.00100). No 2-state HMM can be exact (a
   Hankel-rank test, `tests/test_merge_bound.py`). This is a claim about memory
   **dimension** in simulation, not about compute or hardware: A's multiply-adds per symbol
   are 10–60× an HMM's at the same D.
2. **No advantage on Ghafari over general 2-state HMMs.** The asymmetric process is
   exactly a 2-state *non-unifilar* HMM (the hidden coin value; tested). EM reaches the
   source (long gap 0.00000), as do 1-qubit A, HQMM and uMPS. The 1-qubit model beats
   only 2-state *unifilar* models (bound 0.131 and 0.061 nats/symbol), which is Yang et
   al.'s comparison class.
3. **Uniform renewal N = 3** (Yang et al.'s hardware setting): 1-qubit models 0.019 against
   the best 2-state HMM's 0.073 and the unifilar block bound 0.044. Neither class is exact
   at 2 dimensions, but the qubit beats every 2-state unifilar model.
4. **A is parameter-efficient where it trains.** It reaches parity_k3 at D = 8 with 29
   parameters (HMM 136, HQMM 256) and the clock, Nemo and PMD at D = 2 with 13. It fails
   Stern–Gerlach at D = 2 and 4 where the HQMM and uMPS succeed, and **neither A layout
   reaches the optimum on any retention task** (`recall_tau2`, `marked_recall`), where the
   HMM, uMPS and HQMM do at D = 4–8.
5. **D and D_diss fail almost every memory task** at every D, as in #7; recall_tau2 at
   D = 4 is the exception for their validation-selected runs.

## 1. What was run

**Tasks.** The #8 suite (`[suite.tasks]` of `experiments/configs/phase3c.toml`) with its
splits (`data.seed = 0`: 512 train, 256 validation, 512 test sequences of length 32), plus
`marked_recall` (ε = 0.05, geometric timing) at τ = 4 and 16, with the Phase 3e lengths
(64 and 128). The gap is test NLL minus the source's own NLL on the same sequences
(nats/symbol), and a run fails if it crashed or its gap is at least 0.01, as in #6.

**Pruned for compute** (the machine was shared and saturated; see Load below):

| pruned | why |
|---|---|
| D = 8 except on the 7 tasks needing more than 4 classical states (`pmd_N5`, `pmd_N8`, `golden_mean_4_3`, `probability_clock`, `stern_gerlach`, `recall_tau2`, `parity_k3`) | D = 4 suffices for every model class on the others |
| A, A_ff and D_diss at D = 8 except where the quantum dimension also exceeds 4 (`golden_mean_4_3`, `recall_tau2`, `parity_k3`) | PMD, the clock and Stern–Gerlach have 2-dimensional quantum models |
| D and uMPS at D = 8 | unitary D fails every memory task in #7; uMPS is the HQMM's model class (architecture §8) and its χ = 8 fits were unreliable in #6 |
| D = 4 on `perturbed_coin`, `upset_gambler`, `pmd_N4`, `pmd_N6`, `pmd_N7` | 2-state sources, or PMD periods that serve only the dimensional test |
| `uniform_renewal_N5`, `golden_mean_3_2_ho`, `nemo_ho` | parameter variants of kept tasks |
| `recall_tau1`, `recall_tau4`, `recall_tau8`, `recall_tau16`; `parity_k2`, `parity_k4` | τ = 2 is the one clocked recall that fits in D = 8 (7 causal states); τ ≥ 4 needs 11+; `parity_k2` is an order-2 Markov chain (Phase 2 has one); `parity_k4` needs 16 |
| `srinivasan_hmm6`, `mixed_timescales_hidden` | classical non-unifilar sources without an exact optimum (and 8 symbols for the first) |
| `marked_recall` at τ = 1, 2, 8, 32 | Phase 3e's general-vs-feed-forward arm used τ = 4, 16 |

**Models** (`experiments/configs/phase3f.toml`, `[families.*]`):

- **A, general**: the Phase 2 recipe (2 memory HEA layers, trainable coupling,
  `joint_depth = 2`) with `n_memory = n`.
- **A, feed-forward**: `joint_depth = 0`, `feedforward_depth = 2`, one shared trained `g`
  (#9, #10).
- **D** and **D_diss** (#7): readout without its backaction, then a token-controlled
  unitary (D) or a channel with one traced-out ancilla (D_diss); readout depth as #7 set it
  to match A's count.
- **HMM** (edge-emitting) by EM (each seed one restart) and by gradient; the table shows the
  better on validation. **uMPS** (complex, χ = D), **HQMM** (Stiefel, one Kraus per
  outcome, D = 2ⁿ), the **GRU** with the largest H at most A's count, and the **n-gram**
  model (k ≤ 8 and α on validation).

**Protocol.** As #6: every gradient cell, **A, D and D_diss included**, is tuned over
lr ∈ {0.01, 0.03, 0.1} with tuning seeds 100 and 101 (disjoint from the evaluation seeds),
ranked by median best-validation NLL, and the winner trained with seeds 0–9. Adam, batch 128,
at most 1500 steps, early stopping after 30 validation checks, best-validation restore. **Every
size is tuned on its own**: #12's trainability fixes had not landed, and there is no reuse
of the D = 8 learning rate. The grid runs in waves of increasing D (2, then the large-data
arm, then 4, then 8).

**Matching.**

- *Matched state dimension*: D = 2ⁿ for HMM states, χ, the HQMM's D and A's (and D's) memory.
- *Matched parameter count*: the budget is A (general, D = 4)'s count on each task (21 for
  binary tasks, 26 for 4 symbols, 34 for 8). Each family gets its largest run size with at
  most that many parameters. The HMM and the GRU get an extra cell at a non-power-of-2 size
  where that is closer (HMM D = 3 on binary tasks). Where a family's smallest size is
  already over budget (uMPS and HQMM at D = 2 on 4- and 8-symbol tasks), the table shows that
  size, marked.

Every row in the CSVs states its parameter count, state dimension and multiply-adds per
symbol (`describe()`; D and D_diss are counted by their own update channel).

**#12's trainability options** (`qmem.trainability`) reached `main` on 2026-11-06 at
17:38Z, after every wave had been tuned. They are opt-in, with no validated recipe yet, so
this grid uses the default initialisation and tunes each size on its own. The D = 8 HQMM
cells that underperform D = 2 (below) fit #12's early finding of barren starts at large D.

**Load.** Load average 70–175 on 8 cores with heavy swap throughout (other projects'
jobs); our 6 workers were niced and ran 4–15× slower than Phase 3a's. The run was resumed
once with `--from-run` (after a code fix to the large-data EM restarts, cut from 100 to 20;
the 100 already run on Ghafari are kept).


## 2. The dimensional-advantage test: method

On `ghafari`, `ghafari_p03_q08`, `pmd_N3`–`pmd_N8` (and `uniform_renewal_N3`, Yang et al.'s
IBM setting, where d = 2 is lossy for both classes):

- **Exact 1-qubit models**: `qmachine_instrument` (Ghafari; quantum dimension 2) and
  `pmd_qubit_instrument` (Wu et al.'s phased model), started at their channel fixed point.
- **Trained 1-qubit models**: A (both layouts), HQMM D = 2 and uMPS χ = 2 (the same model
  class as the 1-Kraus HQMM).
- **The best 2-state classical model**: the edge-emitting 2-state HMM by EM with 100
  restarts (2000 iterations, patience 100) and by gradient (tuned, 10 seeds), plus the 10
  grid EM restarts; the best of all on validation.
- **A large-data arm** (16× the training data, 8192 sequences) with the same models, to
  separate capacity from estimation error. Its gradient cells reuse each family's
  standard-data learning rate.
- **Long re-scoring**: each cell's validation-selected model scored on 256 fresh
  sequences of 1024 symbols (seed 777). There the paired gap estimates the KL rate to the
  source, with a standard error.

**The 2-state classical bound.** Yang et al. 2021 (Box 2) bound the distortion of the best
*unifilar* d-state model by merging causal states and optimising the merged futures. With
future length L it is computable exactly (`qmem.complexity.unifilar_merge_bound`); at
L = 1:

  min over partitions g of the causal states into ≤ 2 blocks of
  Σ_s π_s KL(P(·|s) ‖ Q_g(s)), with Q_b = Σ_{s∈b} π_s P(·|s) / π_b.

A unifilar model's state is a function of the past, so its prediction depends on the past
through a 2-valued label. For fixed predictions the best label of a past depends only on
its causal state, and for a fixed labelling the best predictions are the mixtures above. The
minimum is therefore a merge, and the formula is a lower bound for **every** 2-state
unifilar model of a unifilar source, at any Markov order. It is reached when the minimising
merge is itself unifilar. That is the case for both Ghafari settings (Markov order 1, where
Yang et al. prove tightness), but not for PMD or uniform renewal, where the bound may be
loose. With futures of length L the same argument applies to blocks of L symbols, giving a
per-symbol bound that can be tighter (`best_unifilar_merge_bound`, the best over
L = 1..10). The tables report both (the *one-step* and the *block* bound); the block bound is
tighter on PMD N = 3–6 and N = 8 (L = 2) and on uniform renewal (0.0436 at L = 2 against
0.0115 at L = 1). Tests: the merged Ghafari machine reaches the bound (Monte Carlo), and random
2-state unifilar machines never beat either bound.

The bound does **not** cover non-unifilar 2-state HMMs, which can do better, and on Ghafari
do: the process is exactly a 2-state edge-emitting HMM (the coin value is the hidden state
and a 0→1 flip emits 2; `test_ghafari_is_a_two_state_non_unifilar_hmm`). For PMD there is
a proof the other way: a 2-state HMM's output correlations obey an order-2 linear
recurrence (3×3 Hankel determinants vanish), and PMD's do not
(`test_pmd_needs_more_than_a_two_dimensional_real_model`), so no 2-state HMM, and no real
2-dimensional linear model, is exact. How close the best one gets is then empirical: the
fitted HMM is the evidence for general HMMs, the unifilar bound for unifilar ones.

We could not read Yang et al.'s numeric bounds (figures only; their asymmetric-process point
is p = 0.3, q = 0.8, our `ghafari_p03_q08`), so we compare with our recomputation of their
construction.

## 3. Results tables

Copied from `results/summaries/phase3f/summary.md` (generated). Entries are test gaps in nats/symbol; failures (gap ≥ 0.01 or crashed) of 10 seeds in brackets.

### Matched state dimension D = 2: median gap (failures)

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| perturbed_coin | 0.0003 (0) | 0.0004 (0) | 0.0727 (10) | 0.0418 (10) | 0.0002 (0) | 0.0003 (0) | 0.0004 (0) |
| upset_gambler | 0.0053 (0) | 0.0002 (0) | 0.0050 (0) | 0.0016 (0) | 0.0001 (0) | 0.0002 (0) | 0.0004 (0) |
| ghafari | 0.0001 (0) | 0.6170 (10) | 0.4241 (10) | 0.3615 (10) | 0.0004 (0) | 0.0003 (0) | 0.0003 (0) |
| ghafari_p03_q08 | 0.0025 (2) | 0.4867 (10) | 0.4596 (10) | 0.4713 (10) | 0.0001 (0) EM | 0.0001 (0) | 0.0000 (0) |
| pmd_N3 | 0.0010 (4) | 0.0003 (1) | 0.0261 (10) | 0.0056 (0) | 0.0108 (10) EM | 0.0002 (1) | 0.0003 (3) |
| pmd_N4 | 0.0045 (2) | 0.0008 (0) | 0.0173 (10) | 0.0041 (1) | 0.0104 (10) | 0.0005 (2) | 0.0045 (5) |
| pmd_N5 | 0.0015 (0) | 0.0012 (0) | 0.0073 (0) | 0.0015 (0) | 0.0058 (0) | 0.0004 (0) | 0.0031 (0) |
| pmd_N6 | 0.0019 (0) | 0.0009 (0) | 0.0067 (0) | 0.0066 (0) | 0.0067 (0) | 0.0009 (0) | 0.0031 (0) |
| pmd_N7 | 0.0030 (0) | 0.0022 (0) | 0.0030 (0) | 0.0031 (0) | 0.0030 (0) EM | 0.0018 (0) | 0.0028 (0) |
| pmd_N8 | 0.0025 (0) | 0.0017 (0) | 0.0027 (0) | 0.0026 (0) | 0.0022 (0) | 0.0020 (0) | 0.0024 (0) |
| uniform_renewal_N3 | 0.0188 (10) | 0.0185 (10) | 0.1212 (10) | 0.0723 (10) | 0.0700 (10) | 0.0187 (10) | 0.0186 (10) |
| golden_mean_4_3 | 0.2374 (10) | 0.2805 (10) | 0.3447 (10) | 0.3216 (10) | 0.3161 (10) EM | 0.2372 (10) | 0.2372 (10) |
| nemo | 0.0007 (0) | 0.1055 (10) | 0.1313 (10) | 0.0823 (10) | 0.0388 (10) EM | 0.0007 (0) | 0.0006 (0) |
| probability_clock | 0.0009 (1) | 0.0005 (0) | 0.0849 (10) | 0.0417 (10) | 0.0490 (10) | 0.0005 (0) | 0.0005 (0) |
| stern_gerlach | 0.1714 (10) | 0.4373 (10) | 0.4038 (10) | 0.2656 (10) | 0.1528 (10) | 0.0069 (2) | 0.0071 (3) |
| recall_tau2 | 0.1369 (10) | 0.1690 (10) | 0.1715 (10) | 0.1618 (10) | 0.1735 (10) EM | 0.1593 (10) | 0.1364 (10) |
| parity_k3 | 0.1993 (10) | 0.2755 (10) | 0.2816 (10) | 0.2532 (10) | 0.2933 (10) | 0.1989 (10) | 0.1992 (10) |
| mixed_timescales | 0.1752 (10) | 0.5986 (10) | 0.3506 (10) | 0.3194 (10) | 0.1821 (10) | 0.1656 (10) | 0.1652 (10) |
| marked_recall_tau4 | 0.0437 (10) | 0.1518 (10) | 0.0474 (10) | 0.0474 (10) | 0.0211 (10) | 0.0216 (10) | 0.0398 (10) |
| marked_recall_tau16 | 0.0134 (10) | 0.2846 (10) | 0.0134 (10) | 0.0135 (10) | 0.0062 (5) | 0.0131 (7) | 0.0127 (6) |

#### D = 2: validation-selected run's test gap

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| perturbed_coin | 0.0003 | 0.0007 | 0.0719 | 0.0424 | 0.0002 | 0.0003 | 0.0006 |
| upset_gambler | 0.0021 | 0.0000 | 0.0035 | 0.0016 | 0.0000 | 0.0002 | 0.0003 |
| ghafari | 0.0001 | 0.6171 | 0.3524 | 0.3079 | 0.0004 | 0.0002 | 0.0001 |
| ghafari_p03_q08 | 0.0025 | 0.4858 | 0.4300 | 0.4174 | 0.0000 | 0.0001 | 0.0000 |
| pmd_N3 | 0.0003 | 0.0001 | 0.0223 | 0.0056 | 0.0108 | 0.0002 | 0.0001 |
| pmd_N4 | 0.0011 | 0.0005 | 0.0158 | 0.0038 | 0.0104 | 0.0002 | 0.0004 |
| pmd_N5 | 0.0009 | 0.0001 | 0.0065 | 0.0011 | 0.0058 | 0.0002 | 0.0002 |
| pmd_N6 | 0.0012 | 0.0006 | 0.0062 | 0.0015 | 0.0050 | 0.0009 | 0.0010 |
| pmd_N7 | 0.0008 | -0.0000 | 0.0031 | 0.0000 | 0.0031 | 0.0002 | 0.0000 |
| pmd_N8 | 0.0015 | 0.0004 | 0.0029 | 0.0004 | 0.0022 | 0.0006 | 0.0007 |
| uniform_renewal_N3 | 0.0188 | 0.0186 | 0.1204 | 0.0723 | 0.0700 | 0.0187 | 0.0187 |
| golden_mean_4_3 | 0.2373 | 0.2805 | 0.3137 | 0.2892 | 0.3161 | 0.2372 | 0.2372 |
| nemo | 0.0005 | 0.1055 | 0.1326 | 0.0822 | 0.0388 | 0.0007 | 0.0005 |
| probability_clock | 0.0009 | 0.0013 | 0.0849 | 0.0419 | 0.0489 | 0.0005 | 0.0005 |
| stern_gerlach | 0.1764 | 0.4373 | 0.4075 | 0.2650 | 0.1527 | 0.0069 | 0.0059 |
| recall_tau2 | 0.1361 | 0.1643 | 0.1650 | 0.1615 | 0.1698 | 0.1364 | 0.1364 |
| parity_k3 | 0.1999 | 0.2712 | 0.2764 | 0.2522 | 0.2905 | 0.1995 | 0.1997 |
| mixed_timescales | 0.1672 | 0.5823 | 0.3389 | 0.3189 | 0.1844 | 0.1648 | 0.1658 |
| marked_recall_tau4 | 0.0260 | 0.1518 | 0.0472 | 0.0361 | 0.0211 | 0.0185 | 0.0179 |
| marked_recall_tau16 | 0.0134 | 0.2845 | 0.0135 | 0.0135 | 0.0057 | 0.0047 | 0.0044 |

### Matched state dimension D = 4: median gap (failures)

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| ghafari | 0.0041 (5) † | 0.0003 (0) | 0.4320 (10) † | 0.3197 (10) | 0.0005 (0) EM | 0.0004 (0) | 0.0011 (0) |
| ghafari_p03_q08 | 0.0025 (3) † | 0.0002 (0) | 0.5232 (10) † | 0.5465 (10) † | 0.0000 (0) EM | 0.0001 (0) | 0.0002 (0) |
| pmd_N3 | 0.0048 (4) † | 0.0011 (0) | 0.0266 (10) | 0.0142 (8) † | 0.0003 (0) EM | 0.0005 (0) | 0.0026 (0) † |
| pmd_N5 | 0.0092 (4) † | 0.0013 (0) | 0.0074 (0) | 0.0026 (0) † | 0.0011 (0) EM | 0.0011 (0) | 0.0037 (0) |
| pmd_N8 | 0.0030 (0) | 0.0016 (0) | 0.0030 (0) | 0.0026 (0) | 0.0010 (0) | 0.0014 (0) | 0.0033 (0) |
| uniform_renewal_N3 | 0.0065 (5) | 0.0008 (3) | 0.1119 (10) | 0.0639 (10) | 0.0005 (0) EM | 0.0007 (0) | 0.0016 (0) |
| golden_mean_4_3 | 0.1228 (10) | 0.1560 (10) | 0.2719 (10) | 0.2689 (10) | 0.1983 (10) EM | 0.0875 (10) | 0.0878 (10) |
| nemo | 0.0019 (0) † | 0.0004 (0) | 0.1362 (10) † | 0.0855 (10) † | 0.0003 (0) | 0.0003 (0) | 0.0005 (0) |
| probability_clock | 0.0023 (3) † | 0.0009 (0) | 0.0746 (10) | 0.0396 (10) | 0.0071 (4) EM | 0.0008 (0) | 0.0016 (0) † |
| stern_gerlach | 0.2047 (10) † | 0.3949 (10) | 0.4004 (10) | 0.3450 (10) † | 0.0011 (1) | 0.0048 (1) | 0.0045 (0) |
| recall_tau2 | 0.1268 (10) | 0.1277 (10) | 0.1349 (9) | 0.1201 (9) | 0.1341 (10) | 0.0702 (10) | 0.0704 (10) |
| parity_k3 | 0.0702 (10) | 0.0927 (10) | 0.2537 (10) | 0.2157 (10) | 0.1791 (10) | 0.0671 (10) | 0.0668 (10) |
| mixed_timescales | 0.0162 (8) | 0.1734 (10) | 0.4269 (10) † | 0.3638 (10) † | 0.0009 (2) EM | 0.0014 (0) | 0.0017 (0) |
| marked_recall_tau4 | 0.0438 (10) | 0.0385 (10) | 0.0479 (10) | 0.0477 (10) | 0.0002 (2) EM | 0.0021 (5) | 0.0019 (5) |
| marked_recall_tau16 | 0.0138 (10) | 0.0132 (10) | 0.4826 (10) † | 0.0136 (9) | 0.0057 (2) | 0.0130 (6) | 0.0135 (8) † |

#### D = 4: validation-selected run's test gap

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| ghafari | 0.0001 | 0.0003 | 0.3433 | 0.3067 | 0.0006 | 0.0003 | 0.0007 |
| ghafari_p03_q08 | 0.0003 | 0.0002 | 0.4266 | 0.4368 | 0.0001 | 0.0002 | 0.0002 |
| pmd_N3 | 0.0020 | 0.0011 | 0.0217 | 0.0050 | 0.0003 | 0.0005 | 0.0023 |
| pmd_N5 | 0.0031 | 0.0012 | 0.0071 | 0.0009 | 0.0010 | 0.0006 | 0.0015 |
| pmd_N8 | 0.0030 | 0.0013 | 0.0036 | 0.0006 | 0.0009 | 0.0010 | 0.0023 |
| uniform_renewal_N3 | 0.0011 | 0.0003 | 0.0957 | 0.0604 | 0.0005 | 0.0007 | 0.0013 |
| golden_mean_4_3 | 0.1175 | 0.1543 | 0.2152 | 0.2295 | 0.1284 | 0.0873 | 0.0878 |
| nemo | 0.0001 | 0.0002 | 0.0973 | 0.0711 | 0.0003 | 0.0003 | 0.0002 |
| probability_clock | 0.0013 | 0.0009 | 0.0637 | 0.0348 | 0.0066 | 0.0008 | 0.0006 |
| stern_gerlach | 0.1643 | 0.3835 | 0.3948 | 0.2716 | 0.0012 | 0.0029 | 0.0038 |
| recall_tau2 | 0.0857 | 0.1277 | 0.0005 | 0.0007 | 0.1332 | 0.0701 | 0.0703 |
| parity_k3 | 0.0680 | 0.0912 | 0.2207 | 0.2060 | 0.1770 | 0.0665 | 0.0667 |
| mixed_timescales | 0.0005 | 0.1726 | 0.3643 | 0.3599 | 0.0007 | 0.0011 | 0.0018 |
| marked_recall_tau4 | 0.0315 | 0.0384 | 0.0376 | 0.0254 | 0.0002 | 0.0004 | 0.0009 |
| marked_recall_tau16 | 0.0133 | 0.0130 | 0.0134 | 0.0054 | 0.0004 | 0.0049 | 0.0049 |

### Matched state dimension D = 8: median gap (failures)

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| pmd_N5 | - | - | - | - | 0.0010 (0) EM | - | 0.0101 (8) † |
| pmd_N8 | - | - | - | - | 0.0012 (0) | - | 0.0086 (2) † |
| golden_mean_4_3 | 0.0295 (9) | 0.0718 (10) | - | 0.2931 (10) † | 0.0003 (1) EM | - | 0.0006 (0) |
| probability_clock | - | - | - | - | 0.0024 (0) | - | 0.0073 (2) † |
| stern_gerlach | - | - | - | - | 0.0028 (0) EM † | - | 0.0390 (10) † |
| recall_tau2 | 0.1411 (10) † | 0.1144 (10) | - | 0.1491 (8) † | 0.0008 (3) | - | 0.0004 (0) |
| parity_k3 | 0.0013 (1) | 0.0011 (0) | - | 0.2342 (10) † | 0.0007 (4) EM | - | 0.0026 (0) |

#### D = 8: validation-selected run's test gap

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| pmd_N5 | - | - | - | - | 0.0006 | - | 0.0075 |
| pmd_N8 | - | - | - | - | 0.0012 | - | 0.0073 |
| golden_mean_4_3 | 0.0014 | 0.0312 | - | 0.2505 | 0.0006 | - | 0.0007 |
| probability_clock | - | - | - | - | 0.0022 | - | 0.0034 |
| stern_gerlach | - | - | - | - | 0.0024 | - | 0.0367 |
| recall_tau2 | 0.0879 | 0.0772 | - | 0.0005 | 0.0010 | - | 0.0002 |
| parity_k3 | 0.0011 | 0.0014 | - | 0.1970 | 0.0007 | - | 0.0019 |

### Matched parameter count: each family's largest size with at most A (general, D = 4)'s count

Entry: median gap (failures) [parameters]; * = no size within budget, smallest shown.

| task | budget | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM | GRU |
|---|---|---|---|---|---|---|---|---|---|
| perturbed_coin | 21 | 0.0003 (0) [13] D=2 | 0.0004 (0) [13] D=2 | 0.0727 (10) [13] D=2 | 0.0418 (10) [17] D=2 | 0.0002 (0) [21] D=3 | 0.0003 (0) [20] D=2 | 0.0004 (0) [16] D=2 | 0.0001 (1) [19] D=1 |
| upset_gambler | 21 | 0.0053 (0) [13] D=2 | 0.0002 (0) [13] D=2 | 0.0050 (0) [13] D=2 | 0.0016 (0) [17] D=2 | 0.0003 (0) [21] D=3 | 0.0002 (0) [20] D=2 | 0.0004 (0) [16] D=2 | 0.0006 (0) [19] D=1 |
| ghafari | 26 | 0.0041 (5) [26] † D=4 | 0.6170 (10) [21] D=2 | 0.4320 (10) [26] † D=4 | 0.3615 (10) [24] D=2 | 0.0004 (0) [18] D=2 | 0.0003 (0) [36] D=2 * | 0.0003 (0) [32] D=2 * | 0.0007 (4) [29] D=1 * |
| ghafari_p03_q08 | 26 | 0.0025 (3) [26] † D=4 | 0.4867 (10) [21] D=2 | 0.5232 (10) [26] † D=4 | 0.4713 (10) [24] D=2 | 0.0001 (0) EM [18] D=2 | 0.0001 (0) [36] D=2 * | 0.0000 (0) [32] D=2 * | 0.0006 (1) [29] D=1 * |
| pmd_N3 | 21 | 0.0048 (4) [21] † D=4 | 0.0003 (1) [13] D=2 | 0.0266 (10) [21] D=4 | 0.0056 (0) [17] D=2 | 0.0003 (0) [21] D=3 | 0.0002 (1) [20] D=2 | 0.0003 (3) [16] D=2 | 0.0010 (1) [19] D=1 |
| pmd_N4 | 21 | 0.0045 (2) [13] D=2 | 0.0008 (0) [13] D=2 | 0.0173 (10) [13] D=2 | 0.0041 (1) [17] D=2 | 0.0018 (0) [21] D=3 | 0.0005 (2) [20] D=2 | 0.0045 (5) [16] D=2 | 0.0042 (0) [19] D=1 |
| pmd_N5 | 21 | 0.0092 (4) [21] † D=4 | 0.0012 (0) [13] D=2 | 0.0074 (0) [21] D=4 | 0.0015 (0) [17] D=2 | 0.0014 (0) [21] D=3 | 0.0004 (0) [20] D=2 | 0.0031 (0) [16] D=2 | 0.0034 (0) [19] D=1 |
| pmd_N6 | 21 | 0.0019 (0) [13] D=2 | 0.0009 (0) [13] D=2 | 0.0067 (0) [13] D=2 | 0.0066 (0) [17] D=2 | 0.0017 (0) [21] D=3 | 0.0009 (0) [20] D=2 | 0.0031 (0) [16] D=2 | 0.0046 (0) [19] D=1 |
| pmd_N7 | 21 | 0.0030 (0) [13] D=2 | 0.0022 (0) [13] D=2 | 0.0030 (0) [13] D=2 | 0.0031 (0) [17] D=2 | 0.0010 (0) [21] D=3 | 0.0018 (0) [20] D=2 | 0.0028 (0) [16] D=2 | 0.0030 (0) [19] D=1 |
| pmd_N8 | 21 | 0.0030 (0) [21] D=4 | 0.0017 (0) [13] D=2 | 0.0030 (0) [21] D=4 | 0.0026 (0) [17] D=2 | 0.0011 (0) [21] D=3 | 0.0020 (0) [20] D=2 | 0.0024 (0) [16] D=2 | 0.0024 (0) [19] D=1 |
| uniform_renewal_N3 | 21 | 0.0065 (5) [21] D=4 | 0.0185 (10) [13] D=2 | 0.1119 (10) [21] D=4 | 0.0723 (10) [17] D=2 | 0.0005 (0) [21] D=3 | 0.0187 (10) [20] D=2 | 0.0186 (10) [16] D=2 | 0.0034 (2) [19] D=1 |
| golden_mean_4_3 | 21 | 0.1228 (10) [21] D=4 | 0.2805 (10) [13] D=2 | 0.2719 (10) [21] D=4 | 0.3216 (10) [17] D=2 | 0.2785 (10) EM [21] D=3 | 0.2372 (10) [20] D=2 | 0.2372 (10) [16] D=2 | 0.0860 (10) [19] D=1 |
| nemo | 21 | 0.0019 (0) [21] † D=4 | 0.1055 (10) [13] D=2 | 0.1362 (10) [21] † D=4 | 0.0823 (10) [17] D=2 | 0.0235 (8) EM [21] D=3 | 0.0007 (0) [20] D=2 | 0.0006 (0) [16] D=2 | 0.0131 (8) [19] D=1 |
| probability_clock | 21 | 0.0023 (3) [21] † D=4 | 0.0005 (0) [13] D=2 | 0.0746 (10) [21] D=4 | 0.0417 (10) [17] D=2 | 0.0202 (10) EM [21] D=3 | 0.0005 (0) [20] D=2 | 0.0005 (0) [16] D=2 | 0.0238 (10) [19] D=1 |
| stern_gerlach | 31 | 0.2047 (10) [31] † D=4 | 0.4373 (10) [37] D=2 * | 0.4038 (10) [27] D=2 | 0.2656 (10) [43] D=2 * | 0.1528 (10) [34] D=2 * | 0.0069 (2) [68] D=2 * | 0.0071 (3) [64] D=2 * | 0.1708 (10) [49] D=1 * |
| recall_tau2 | 21 | 0.1268 (10) [21] D=4 | 0.1690 (10) [13] D=2 | 0.1349 (9) [21] D=4 | 0.1618 (10) [17] D=2 | 0.1623 (10) EM [21] D=3 | 0.1593 (10) [20] D=2 | 0.1364 (10) [16] D=2 | 0.1693 (10) [19] D=1 |
| parity_k3 | 21 | 0.0702 (10) [21] D=4 | 0.2755 (10) [13] D=2 | 0.2537 (10) [21] D=4 | 0.2532 (10) [17] D=2 | 0.2611 (10) [21] D=3 | 0.1989 (10) [20] D=2 | 0.1992 (10) [16] D=2 | 0.2946 (10) [19] D=1 |
| mixed_timescales | 26 | 0.0162 (8) [26] D=4 | 0.5986 (10) [21] D=2 | 0.4269 (10) [26] † D=4 | 0.3194 (10) [24] D=2 | 0.1821 (10) [18] D=2 | 0.1656 (10) [36] D=2 * | 0.1652 (10) [32] D=2 * | 0.1765 (10) [29] D=1 * |
| marked_recall_tau4 | 26 | 0.0438 (10) [26] D=4 | 0.1518 (10) [21] D=2 | 0.0479 (10) [26] D=4 | 0.0474 (10) [24] D=2 | 0.0211 (10) [18] D=2 | 0.0216 (10) [36] D=2 * | 0.0398 (10) [32] D=2 * | 0.0082 (4) [29] D=1 * |
| marked_recall_tau16 | 26 | 0.0138 (10) [26] D=4 | 0.2846 (10) [21] D=2 | 0.4826 (10) [26] † D=4 | 0.0135 (10) [24] D=2 | 0.0062 (5) [18] D=2 | 0.0131 (7) [36] D=2 * | 0.0127 (6) [32] D=2 * | 0.0134 (9) [29] D=1 * |

#### Validation-selected run at matched parameter count

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM | GRU |
|---|---|---|---|---|---|---|---|---|
| perturbed_coin | 0.0003 | 0.0007 | 0.0719 | 0.0424 | 0.0001 | 0.0003 | 0.0006 | 0.0003 |
| upset_gambler | 0.0021 | 0.0000 | 0.0035 | 0.0016 | 0.0003 | 0.0002 | 0.0003 | 0.0002 |
| ghafari | 0.0001 | 0.6171 | 0.3433 | 0.3079 | 0.0004 | 0.0002 * | 0.0001 * | 0.0007 * |
| ghafari_p03_q08 | 0.0003 | 0.4858 | 0.4266 | 0.4174 | 0.0000 | 0.0001 * | 0.0000 * | 0.0005 * |
| pmd_N3 | 0.0020 | 0.0001 | 0.0217 | 0.0056 | 0.0002 | 0.0002 | 0.0001 | 0.0007 |
| pmd_N4 | 0.0011 | 0.0005 | 0.0158 | 0.0038 | 0.0020 | 0.0002 | 0.0004 | 0.0033 |
| pmd_N5 | 0.0031 | 0.0001 | 0.0071 | 0.0011 | 0.0012 | 0.0002 | 0.0002 | 0.0029 |
| pmd_N6 | 0.0012 | 0.0006 | 0.0062 | 0.0015 | 0.0016 | 0.0009 | 0.0010 | 0.0042 |
| pmd_N7 | 0.0008 | -0.0000 | 0.0031 | 0.0000 | 0.0010 | 0.0002 | 0.0000 | 0.0031 |
| pmd_N8 | 0.0030 | 0.0004 | 0.0036 | 0.0004 | 0.0011 | 0.0006 | 0.0007 | 0.0017 |
| uniform_renewal_N3 | 0.0011 | 0.0186 | 0.0957 | 0.0723 | 0.0005 | 0.0187 | 0.0187 | 0.0021 |
| golden_mean_4_3 | 0.1175 | 0.2805 | 0.2152 | 0.2892 | 0.2421 | 0.2372 | 0.2372 | 0.0659 |
| nemo | 0.0001 | 0.1055 | 0.0973 | 0.0822 | 0.0003 | 0.0007 | 0.0005 | 0.0081 |
| probability_clock | 0.0013 | 0.0013 | 0.0637 | 0.0419 | 0.0202 | 0.0005 | 0.0005 | 0.0184 |
| stern_gerlach | 0.1643 | 0.4373 * | 0.4075 | 0.2650 * | 0.1527 * | 0.0069 * | 0.0059 * | 0.1727 * |
| recall_tau2 | 0.0857 | 0.1643 | 0.0005 | 0.1615 | 0.1614 | 0.1364 | 0.1364 | 0.1694 |
| parity_k3 | 0.0680 | 0.2712 | 0.2207 | 0.2522 | 0.2490 | 0.1995 | 0.1997 | 0.2753 |
| mixed_timescales | 0.0005 | 0.5823 | 0.3643 | 0.3189 | 0.1844 | 0.1648 * | 0.1658 * | 0.0890 * |
| marked_recall_tau4 | 0.0315 | 0.1518 | 0.0376 | 0.0361 | 0.0211 | 0.0185 * | 0.0179 * | 0.0070 * |
| marked_recall_tau16 | 0.0133 | 0.2845 | 0.0134 | 0.0135 | 0.0057 | 0.0047 * | 0.0044 * | 0.0022 * |

### Smallest size reaching the optimum (gap < 0.01): D [parameters]; median over seeds / validation-selected run

| task | A, general | A, feed-forward | D | D_diss | HMM | uMPS | HQMM |
|---|---|---|---|---|---|---|---|
| perturbed_coin | 2 [13] / 2 [13] | 2 [13] / 2 [13] | none / none | none / none | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| upset_gambler | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [17] / 2 [17] | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| ghafari | 2 [18] / 2 [18] | 4 [41] / 4 [41] | none / none | none / none | 2 [18] / 2 [18] | 2 [36] / 2 [36] | 2 [32] / 2 [32] |
| ghafari_p03_q08 | 2 [18] / 2 [18] | 4 [41] / 4 [41] | none / none | none / none | 2 [18] / 2 [18] | 2 [36] / 2 [36] | 2 [32] / 2 [32] |
| pmd_N3 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | none / none | 2 [17] / 2 [17] | 3 [21] / 3 [21] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| pmd_N4 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | none / none | 2 [17] / 2 [17] | 3 [21] / 3 [21] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| pmd_N5 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [17] / 2 [17] | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| pmd_N6 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [17] / 2 [17] | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| pmd_N7 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [17] / 2 [17] | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| pmd_N8 | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [13] / 2 [13] | 2 [17] / 2 [17] | 2 [10] / 2 [10] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| uniform_renewal_N3 | 4 [21] / 4 [21] | 4 [25] / 4 [25] | none / none | none / none | 3 [21] / 3 [21] | 4 [72] / 4 [72] | 4 [64] / 4 [64] |
| golden_mean_4_3 | none / 8 [29] | none / none | none / none | none / none | 8 [136] / 8 [136] | none / none | 8 [256] / 8 [256] |
| nemo | 2 [13] / 2 [13] | 4 [25] / 4 [25] | none / none | none / none | 3 [21] / 3 [21] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| probability_clock | 2 [13] / 2 [13] | 2 [13] / 2 [13] | none / none | none / none | 4 [36] / 4 [36] | 2 [20] / 2 [20] | 2 [16] / 2 [16] |
| stern_gerlach | none / none | none / none | none / none | none / none | 4 [132] / 4 [132] | 2 [68] / 2 [68] | 2 [64] / 2 [64] |
| recall_tau2 | none / none | none / none | none / 4 [21] | none / 4 [25] | 8 [136] / 8 [136] | none / none | 8 [256] / 8 [256] |
| parity_k3 | 8 [29] / 8 [29] | 8 [37] / 8 [37] | none / none | none / none | 8 [136] / 8 [136] | none / none | 8 [256] / 8 [256] |
| mixed_timescales | none / 4 [26] | none / none | none / none | none / none | 4 [68] / 4 [68] | 4 [136] / 4 [136] | 4 [128] / 4 [128] |
| marked_recall_tau4 | none / none | none / none | none / none | none / none | 4 [68] / 4 [68] | 4 [136] / 4 [136] | 4 [128] / 4 [128] |
| marked_recall_tau16 | none / none | none / none | none / none | none / 4 [34] | 2 [18] / 2 [18] | none / 2 [36] | none / 2 [32] |

## 4. The dimensional test: results

Long-sequence KL rate (nats/symbol; 256 sequences of 1024, validation-selected run, paired
gap ± SE) of each trained 1-qubit model, the lowest of the 2-state HMM cells (standard and
16× data, EM with restarts and gradient), and the two unifilar bounds. The exact 1-qubit
models (`qmachine_instrument`, `pmd_qubit_instrument`) score 0.00000 on every Ghafari and
PMD task; uniform renewal has none. Per-cell tables, including the large-data arm and the
test-set gaps, are in `summary.md`.

| task | A | A_ff | HQMM | uMPS | best 2-state HMM | one-step bound | block bound |
|---|---|---|---|---|---|---|---|
| ghafari | 0.00007 ± 0.00001 | 0.62909 ± 0.00070 | 0.00007 ± 0.00002 | 0.00016 ± 0.00002 | 0.00000 ± 0.00000 (hmm-em/D2/restarts/large) | 0.13081 | 0.13081 |
| ghafari_p03_q08 | 0.00044 ± 0.00004 | 0.49458 ± 0.00129 | 0.00007 ± 0.00003 | 0.00009 ± 0.00003 | 0.00002 ± 0.00001 (hmm-em/D2/restarts/large) | 0.06136 | 0.06136 |
| pmd_N3 | 0.00041 ± 0.00006 | 0.00018 ± 0.00004 | 0.00017 ± 0.00004 | 0.00010 ± 0.00003 | 0.01365 ± 0.00029 (hmm-grad/D2/large) | 0.00736 | 0.00764 |
| pmd_N4 | 0.00045 ± 0.00006 | 0.00035 ± 0.00005 | 0.00018 ± 0.00003 | 0.00018 ± 0.00004 | 0.01106 ± 0.00028 (hmm-grad/D2/large) | 0.00037 | 0.00709 |
| pmd_N5 | 0.00047 ± 0.00006 | 0.00007 ± 0.00002 | 0.00023 ± 0.00004 | 0.00009 ± 0.00003 | 0.00681 ± 0.00022 (hmm-grad/D2/large) | 0.00298 | 0.00364 |
| pmd_N6 | 0.00076 ± 0.00007 | 0.00048 ± 0.00006 | 0.00089 ± 0.00008 | 0.00068 ± 0.00007 | 0.00454 ± 0.00017 (hmm-em/D2/restarts/large) | 0.00152 | 0.00241 |
| pmd_N7 | 0.00041 ± 0.00006 | 0.00027 ± 0.00004 | 0.00022 ± 0.00004 | 0.00065 ± 0.00007 | 0.00287 ± 0.00015 (hmm-em/D2/restarts/large) | 0.00157 | 0.00157 |
| pmd_N8 | 0.00104 ± 0.00009 | 0.00018 ± 0.00003 | 0.00023 ± 0.00005 | 0.00022 ± 0.00004 | 0.00183 ± 0.00012 (hmm-em/D2/restarts/large) | 0.00090 | 0.00100 |
| uniform_renewal_N3 | 0.01907 ± 0.00020 | 0.01895 ± 0.00018 | 0.01904 ± 0.00020 | 0.01904 ± 0.00019 | 0.07305 ± 0.00050 (hmm-em/D2/restarts/large) | 0.01154 | 0.04360 |

- **Ghafari (both settings): no advantage over general 2-state classical models.** The
  2-state HMM reaches the source with 16× data (0.00000 and 0.00002) and nearly with the
  standard data (EM 0.0004, gradient 0.0003). The 1-qubit models are far below the unifilar
  bound (0.131, 0.061), so the advantage over 2-state *unifilar* models (Yang et al.'s
  claim, and their comparison class) reproduces in simulation, but it is not an advantage
  over hidden-state models. A_ff fails at n = 1 for the structural reason of #10 (one shared
  `g` ties E_01 = E_10); it reaches the source at D = 4.
- **PMD N = 3–8: a dimensional advantage over every 2-state classical model we can fit,
  and over every 2-state unifilar model by proof.** The HMM's gap does not move with 16×
  data or 100 restarts (every restart converges to the same fit on N = 3, 4), so it is
  capacity, not estimation. It shrinks with N because PMD's hazard modulation becomes
  smoother (the bounds shrink with it). A 3-state HMM is enough (`reach`: N = 3, 4 at
  D = 3), so the advantage is one classical state.
- **Uniform renewal N = 3:** neither class is exact at 2 dimensions (all 1-qubit models
  0.019, fail); the qubit beats the 2-state HMM (0.073) and the unifilar block bound
  (0.044). At D = 3–4 every family except D and D_diss reaches it.
- Short test sets mislead here: D_diss on PMD N = 3 has a test gap of 0.0056 (a "success")
  but a long gap of 0.037. The long gap is the KL rate; the length-32 test gap includes the
  start-up transient.


## 5. Suspect cells

A cell is suspect when it is worse than a smaller cell of the same family on median gap,
failures or the selected run (`suspect.csv`; † in the tables). These are trainability
failures, not capacity limits, since the larger model contains the smaller one. They
cluster in **A at D = 4** (Ghafari, PMD N = 3, 5, Nemo, the clock, Stern–Gerlach), the
**HQMM at D = 8** (PMD N = 5, 8, the clock, Stern–Gerlach), and D and D_diss everywhere.
Phase 4a (#12) should start from these.


| task | cell | vs | median gap | vs | failures | vs | selected gap | vs |
|---|---|---|---|---|---|---|---|---|
| ghafari | A/D4 | A/D2 | 0.0041 | 0.0001 | 5 | 0 | 0.0001 | 0.0001 |
| ghafari | D/D4 | D/D2 | 0.4320 | 0.4241 | 10 | 10 | 0.3433 | 0.3524 |
| ghafari_p03_q08 | A/D4 | A/D2 | 0.0025 | 0.0025 | 3 | 2 | 0.0003 | 0.0025 |
| ghafari_p03_q08 | D/D4 | D/D2 | 0.5232 | 0.4596 | 10 | 10 | 0.4266 | 0.4300 |
| ghafari_p03_q08 | D_diss/D4 | D_diss/D2 | 0.5465 | 0.4713 | 10 | 10 | 0.4368 | 0.4174 |
| golden_mean_4_3 | D_diss/D8 | D_diss/D4 | 0.2931 | 0.2689 | 10 | 10 | 0.2505 | 0.2295 |
| marked_recall_tau16 | D/D4 | D/D2 | 0.4826 | 0.0134 | 10 | 10 | 0.0134 | 0.0135 |
| marked_recall_tau16 | hqmm/D4 | hqmm/D2 | 0.0135 | 0.0127 | 8 | 6 | 0.0049 | 0.0044 |
| mixed_timescales | D/D4 | D/D2 | 0.4269 | 0.3506 | 10 | 10 | 0.3643 | 0.3389 |
| mixed_timescales | D_diss/D4 | D_diss/D2 | 0.3638 | 0.3194 | 10 | 10 | 0.3599 | 0.3189 |
| nemo | A/D4 | A/D2 | 0.0019 | 0.0007 | 0 | 0 | 0.0001 | 0.0005 |
| nemo | D/D4 | D/D2 | 0.1362 | 0.1313 | 10 | 10 | 0.0973 | 0.1326 |
| nemo | D_diss/D4 | D_diss/D2 | 0.0855 | 0.0823 | 10 | 10 | 0.0711 | 0.0822 |
| parity_k3 | D_diss/D8 | D_diss/D4 | 0.2342 | 0.2157 | 10 | 10 | 0.1970 | 0.2060 |
| pmd_N3 | A/D4 | A/D2 | 0.0048 | 0.0010 | 4 | 4 | 0.0020 | 0.0003 |
| pmd_N3 | D_diss/D4 | D_diss/D2 | 0.0142 | 0.0056 | 8 | 0 | 0.0050 | 0.0056 |
| pmd_N3 | hqmm/D4 | hqmm/D2 | 0.0026 | 0.0003 | 0 | 3 | 0.0023 | 0.0001 |
| pmd_N5 | A/D4 | A/D2 | 0.0092 | 0.0015 | 4 | 0 | 0.0031 | 0.0009 |
| pmd_N5 | D_diss/D4 | D_diss/D2 | 0.0026 | 0.0015 | 0 | 0 | 0.0009 | 0.0011 |
| pmd_N5 | hqmm/D8 | hqmm/D2 | 0.0101 | 0.0031 | 8 | 0 | 0.0075 | 0.0002 |
| pmd_N8 | hqmm/D8 | hqmm/D2 | 0.0086 | 0.0024 | 2 | 0 | 0.0073 | 0.0007 |
| probability_clock | A/D4 | A/D2 | 0.0023 | 0.0009 | 3 | 1 | 0.0013 | 0.0009 |
| probability_clock | hqmm/D4 | hqmm/D2 | 0.0016 | 0.0005 | 0 | 0 | 0.0006 | 0.0005 |
| probability_clock | hqmm/D8 | hqmm/D2 | 0.0073 | 0.0005 | 2 | 0 | 0.0034 | 0.0005 |
| recall_tau2 | A/D8 | A/D2 | 0.1411 | 0.1369 | 10 | 10 | 0.0879 | 0.1361 |
| recall_tau2 | D_diss/D8 | D_diss/D4 | 0.1491 | 0.1201 | 8 | 9 | 0.0005 | 0.0007 |
| stern_gerlach | A/D4 | A/D2 | 0.2047 | 0.1714 | 10 | 10 | 0.1643 | 0.1764 |
| stern_gerlach | D_diss/D4 | D_diss/D2 | 0.3450 | 0.2656 | 10 | 10 | 0.2716 | 0.2650 |
| stern_gerlach | hmm-em/D8 | hmm-em/D4 | 0.0028 | 0.0010 | 0 | 0 | 0.0024 | 0.0005 |
| stern_gerlach | hmm-grad/D8 | hmm-grad/D4 | 0.0034 | 0.0011 | 0 | 1 | 0.0032 | 0.0012 |
| stern_gerlach | hqmm/D8 | hqmm/D2 | 0.0390 | 0.0071 | 10 | 3 | 0.0367 | 0.0059 |

## 6. What bears on the candidate findings in #2 (for Phase 4b)

Evidence only: Phase 4 does the targeted falsification.

**(a) Parameter efficiency of A against D, D_diss and the HQMM.**

- *For:* where A trains, it needs far fewer parameters than the HQMM at the same D and
  than the HMM at the D it needs. parity_k3: A at D = 8 with 29 parameters (median gap
  0.0013, 1/10 failed) against the HMM at 136 and the HQMM at 256. golden_mean_4_3: A's
  validation-selected D = 8 run reaches 0.0014 with 29 parameters (HQMM 0.0007 with 256),
  though 9/10 seeds fail. PMD, the clock and Nemo: A at D = 2 with 13 (HQMM 16, uMPS 20,
  HMM needs D = 3–4, 21–36). mixed_timescales: A's selected D = 4 run 0.0005 with 26
  (HMM 68, HQMM 128).
- *Against:* A fails Stern–Gerlach at D = 2 and D = 4 (0.17, 0.20) where the HQMM and
  uMPS reach it at D = 2 (0.007), and fails every retention task (c). Its seed medians are
  often far from its selected runs (golden_mean, mixed_timescales, Ghafari D = 4), so the
  efficiency comes with poor trainability. Parameters are not compute: A's multiply-adds
  per symbol are 10–60× the HMM's.
- *D and D_diss:* at matched parameters they fail nearly everything A reaches (Ghafari,
  Nemo, the clock, uniform renewal, parity). The one reversal is recall_tau2 at D = 4,
  where D's and D_diss's selected runs reach 0.0005–0.0007 and A's does not (0.086); their
  medians fail. So measurement backaction helps on most memory tasks and may hurt on
  clocked recall. That reversal is worth a targeted test.

**(c) General against feed-forward layout on retention-type tasks.**

- Neither layout reaches the optimum on `recall_tau2`, `marked_recall_tau4` or
  `marked_recall_tau16` at any D run (D = 2, 4; and 8 for recall_tau2), while the HMM,
  uMPS and HQMM reach marked_recall_tau4 at D = 4 and the HMM and HQMM reach recall_tau2
  at D = 8.
- Recall accuracy: recall_tau2 A 0.99 against A_ff 0.97 at D = 4 (optimum 1.0);
  marked_recall_tau4 A 0.72 against A_ff 0.77 at D = 4 (optimum 0.95); marked_recall_tau16
  both near chance (0.52, 0.56). At D = 2, A_ff sits at chance on marked_recall (the #10
  tie) while A does not.
- So this grid shows **no consistent retention advantage of either layout**. Both lose
  the cue between cue and recall, which is consistent with #10's finding that repeated
  measurement erodes retention. A_ff is better on PMD (long gaps 0.0001–0.0005 against A's
  0.0004–0.001), a renewal task where the memory is reset at each 1.

**(d) The dimensional advantage.** Section 4: established in simulation against 2-state
unifilar models on Ghafari, PMD and uniform renewal, and against every 2-state classical
model we could fit on PMD (with a proof that none is exact). **Not** an advantage over
2-state HMMs on Ghafari. Phase 4b's falsification target is the PMD gap to general
2-state HMMs: a better classical optimiser, or a proof of a lower bound for non-unifilar
2-state models, would settle it. It is a memory-dimension claim only.

## 7. Caveats

- **Pruned cells** (section 1) mean the reach table's "none" for A, A_ff, D, D_diss, uMPS
  at D = 8 on several tasks is "not run", not "failed" (`-` in the D = 8 tables).
- **Defaults only.** No #12 recipe was applied. The large-D cells, A at D = 4 and HQMM at
  D = 8 especially, may improve under it; they are the first rows to re-run.
- **Matching is imperfect.** On 4- and 8-symbol tasks the uMPS and HQMM at D = 2 are already
  over the parameter budget (`*` in the matched-parameter table). The GRU's state size is
  not comparable to the others' D.
- **Gaps below 0.001** are within a few standard errors of zero on the standard test set;
  the long re-scoring is the reliable comparison there.
