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

__VERDICT_DETAILS__

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

__SECTIONS__
