"""Phase 5a (issue #14): what it would take to run the best models on a real QPU.

Simulation and desk study only: local Qiskit simulators and fake backends (no account, no
job sent anywhere). Needs the optional ``hardware`` dependency group::

    uv run --group hardware python experiments/phase5a.py --config experiments/configs/phase5a.toml

Stages (all on by default; each reads the previous one's output from the run directory):

1. ``train``: re-train the validation-selected run of each cell named in ``[[models]]``
   through the Phase 4b runner (the raw runs of #11 and #13 are gitignored and gone) and
   check it against the committed ``reference``.
2. ``export``: export each model to a Qiskit dynamic circuit (``qmem.hardware``) and check
   the outcome distribution and the conditional memory state against ``qmem`` exactly, by an
   independent branch simulation of the circuit, by the deferred-measurement statevector, and
   by Aer's density-matrix simulator (sampled), for several sequence lengths.
3. ``transpile``: depth, two-qubit gates and measurement/reset counts per symbol and for a
   length-T sequence on fake IBM backends.
4. ``noise``: exact noisy instruments of the transpiled step under each fake backend's
   calibration and under swept depolarising, readout and reset errors: KL rate (PMD) and
   recall accuracy (retention) against T and error rate; an Aer cross-check on sampled runs.
5. ``cost``: shots for a target precision, QPU time and an indicative cost.

Writes ``results/phase5a/<timestamp>/`` (records, parameters; gitignored) and
``results/summaries/phase5a/`` (tables, CSV, JSON; committed). ``--from-run <dir>`` reuses a
run directory; ``--stages a,b`` runs only those stages.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
import time
import tomllib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4b

from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

STAGES = ("train", "export", "transpile", "noise", "cost")


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


# -- 1. train ----------------------------------------------------------------------------
def model_spec(cfg: dict[str, Any], m: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """A Phase 4b run spec for one ``[[models]]`` entry."""
    arm = {
        "name": m["name"],
        "family": m["family"],
        "model": m.get("model", {}),
        "train": m.get("train", {}),
        "mode": m.get("mode", "standard"),
        "penalty": m.get("penalty"),
    }
    spec = phase4b.make_spec(cfg, arm, m["task"], "eval", m["lr"], m["seed"], out_dir)
    spec["key"] = f"{m['name']}/seed{m['seed']}"
    return spec


def record_path(out_dir: Path, m: dict[str, Any]) -> Path:
    return out_dir / "runs" / f"{m['name']}/seed{m['seed']}.json"


def train_models(cfg: dict[str, Any], out_dir: Path, workers: int) -> None:
    todo = [m for m in cfg["models"] if not record_path(out_dir, m).exists()]
    print(f"[train] {len(cfg['models'])} models, {len(todo)} to train", flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(workers, initializer=phase4b._worker_init) as ex:
        futs = {
            ex.submit(phase4b.run_and_save, model_spec(cfg, m, out_dir), str(out_dir)): m
            for m in todo
        }
        for f in as_completed(futs):
            r = f.result()
            gap = r.get("test", {}).get("gap_nats", math.nan)
            print(
                f"[train] {r['key']} {r['status']} val={r.get('val_nll', math.nan):.6f} "
                f"gap={gap:.6f} ({r['wall_seconds']:.0f}s)",
                flush=True,
            )


def load_trained(cfg: dict[str, Any], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Each model's record, rebuilt model (``model``) and reproduction check (``repro``)."""
    from qmem.baselines import build_baseline
    from qmem.experiment import build_splits
    from qmem.tasks import make_task
    from qmem.trainability import resolve_init_probs

    out = {}
    for m in cfg["models"]:
        p = record_path(out_dir, m)
        rec = json.loads(p.read_text())
        if rec.get("status") == "error":
            raise RuntimeError(f"{m['name']} failed to train: {rec.get('message')}")
        spec = rec["meta"]["config"]
        task = make_task(spec["task"])
        splits = build_splits(task, spec["data"])
        model_cfg = resolve_init_probs(spec["model"], splits.train, task.n_symbols)
        model = build_baseline("A", model_cfg, task.n_symbols, spec["seed"])
        model.load_state_dict(torch.load(p.with_suffix(".pt")))
        model.eval()
        ref = m.get("reference", {})
        repro = {}
        if "val_nll" in ref:
            repro["val_nll"] = {"now": rec["val_nll"], "reference": ref["val_nll"]}
        if "test_gap" in ref:
            repro["test_gap"] = {"now": rec["test"]["gap_nats"], "reference": ref["test_gap"]}
        for v in repro.values():
            v["abs_diff"] = abs(v["now"] - v["reference"])
        out[m["name"]] = {
            "entry": m,
            "record": rec,
            "model": model,
            "task": task,
            "splits": splits,
            "repro": repro,
            "kind": kind_of(task),
        }
    return out


def kind_of(task: Any) -> str:
    k = task.params.get("kind")
    return {"pmd_renewal": "pmd", "marked_recall": "recall"}.get(k, "markov")


def layout_name(model: Any) -> str:
    blocks = [type(b).__name__ for b in model.circuit.blocks]
    return "feed-forward" if "OutcomeConditionedBlock" in blocks else "general"


def describe(name: str, t: dict[str, Any]) -> dict[str, Any]:
    m = t["model"]
    return {
        "model": name,
        "task": t["entry"]["task"],
        "layout": layout_name(m),
        "n_memory": m.n_memory,
        "n_output": m.n_output,
        "n_qubits": m.n_qubits,
        "n_parameters": m.n_parameters(),
        "seed": t["entry"]["seed"],
        "test_gap": t["record"]["test"]["gap_nats"],
        "steps_run": t["record"].get("steps_run"),
        **{f"repro_{k}_diff": v["abs_diff"] for k, v in t["repro"].items()},
    }


# -- 2. export ------------------------------------------------------------------------------
def all_sequences(n_symbols: int, length: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(n_symbols), repeat=length)), dtype=torch.long)


@torch.no_grad()
def export_checks(cfg: dict[str, Any], trained: dict[str, Any], summary_dir: Path):
    """Exactness of the exported circuits against qmem (three independent simulations)."""
    from qiskit import qasm3, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    from qmem import hardware as hw

    c = cfg["export"]
    rows = []
    (summary_dir / "circuits").mkdir(parents=True, exist_ok=True)
    for name, t in trained.items():
        m = t["model"]
        ff = layout_name(m) == "feed-forward"
        for length in c["lengths"]:
            if m.n_symbols**length > c["max_branches"]:
                continue
            seqs = all_sequences(m.n_symbols, length)
            p = m.log_prob(seqs).exp().numpy()
            rho = m.final_memory(seqs).rho.numpy()
            row = {"model": name, "length": length, "n_strings": len(seqs)}
            for mode in ("dynamic", "coherent") if ff else ("dynamic",):
                br = hw.sequence_branches(hw.export_circuit(m, length, mode), m)
                dp = max(abs(br[tuple(s)][0] - p[i]) for i, s in enumerate(seqs.tolist()))
                dr = max(
                    float(np.abs(br[tuple(s)][1] - rho[i]).max())
                    for i, s in enumerate(seqs.tolist())
                    if p[i] > 1e-12
                )
                row[f"{mode}_max_dp"] = dp
                row[f"{mode}_max_drho"] = dr
            row["prob_sum_defect"] = abs(float(p.sum()) - 1)
            n_def = m.n_memory + length * (m.n_output + m.n_env)
            if n_def <= c["deferred_max_qubits"]:
                sv = Statevector(hw.export_deferred(m, length))
                flat = [q for o in hw.output_qubits_deferred(m, length) for q in o]
                probs = sv.probabilities(flat)
                dd = 0.0
                for i, s in enumerate(seqs.tolist()):
                    bits = [(y >> (m.n_output - 1 - k)) & 1 for y in s for k in range(m.n_output)]
                    dd = max(dd, abs(probs[sum(b << j for j, b in enumerate(bits))] - p[i]))
                row["deferred_max_dp"] = dd
            if length == c["aer_length"]:
                qc = hw.export_circuit(m, length)
                sim = AerSimulator(method="density_matrix", shot_branching_enable=True)  # noiseless
                counts = (
                    sim.run(transpile(qc, sim), shots=c["aer_shots"], seed_simulator=c["aer_seed"])
                    .result()
                    .get_counts()
                )
                emp = counts_to_tokens(counts)
                n = c["aer_shots"]
                row["aer_shots"] = n
                row["aer_tv"] = 0.5 * sum(
                    abs(emp.get(tuple(s), 0) / n - p[i]) for i, s in enumerate(seqs.tolist())
                )
                row["aer_chi2_pvalue"] = chi2_pvalue(emp, seqs.tolist(), p, n)
            rows.append(row)
            print(f"[export] {name} T={length}: " + fmt_row(row), flush=True)
        qc = hw.export_circuit(m, c["qasm_length"])
        (summary_dir / "circuits" / f"{name}_T{c['qasm_length']}.qasm").write_text(qasm3.dumps(qc))
    return rows


def counts_to_tokens(counts: dict[str, int]) -> dict[tuple[int, ...], int]:
    """Aer counts (space-separated registers, last register first) to token tuples."""
    out: dict[tuple[int, ...], int] = {}
    for k, v in counts.items():
        toks = tuple(int(r, 2) for r in reversed(k.split(" ")))
        out[toks] = out.get(toks, 0) + v
    return out


def chi2_pvalue(emp, seqs, p, n) -> float:
    from scipy.stats import chi2

    stat, dof = 0.0, -1
    for i, s in enumerate(seqs):
        e = n * p[i]
        if e > 0:
            stat += (emp.get(tuple(s), 0) - e) ** 2 / e
            dof += 1
    return float(chi2.sf(stat, max(dof, 1)))


def fmt_row(row: dict[str, Any]) -> str:
    return " ".join(
        f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
        for k, v in row.items()
        if k not in ("model",)
    )


# -- 3. transpile ---------------------------------------------------------------------------
def get_backend(name: str) -> Any:
    from qiskit_ibm_runtime import fake_provider

    cls = "Fake" + "".join(w.capitalize() for w in name.removeprefix("fake_").split("_"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(fake_provider, cls)()


def step_duration(qc: Any, target: Any, feedforward_time: float) -> float:
    """ASAP duration (s) of a compact transpiled circuit; control flow costs its latency."""
    from qiskit.circuit.controlflow import IfElseOp, SwitchCaseOp

    from qmem import hardware as hw

    phys = getattr(qc, "_phys", list(range(qc.num_qubits)))
    clock = [0.0] * qc.num_qubits

    def dur(name: str, qs: list[int]) -> float:
        if name in ("rz", "barrier", "delay", "id"):
            return 0.0
        key = tuple(phys[q] for q in qs)
        try:
            props = target[name][key]
        except KeyError:
            props = target[name].get(tuple(reversed(key)))
        return (props.duration or 0.0) if props is not None else 0.0

    def walk(c: Any, qmap: list[int]) -> None:
        idx = {q: i for i, q in enumerate(c.qubits)}
        for item in hw.control_groups(c):
            if isinstance(item, list):  # mutually exclusive ifs: one latency, worst body
                bodies = [(i.operation.blocks[0], i.qubits) for i in item]
            elif isinstance(item.operation, IfElseOp | SwitchCaseOp):
                bodies = [(b, item.qubits) for b in item.operation.blocks]
            else:
                qs = [qmap[idx[q]] for q in item.qubits]
                t0 = max(clock[q] for q in qs)
                d = dur(item.operation.name, qs)
                for q in qs:
                    clock[q] = t0 + d
                continue
            start = max(clock) + feedforward_time  # wait for the readout everywhere
            saved = list(clock)
            ends = []
            for blk, qubits in bodies:
                qs = [qmap[idx[q]] for q in qubits]
                clock[:] = [max(x, start) if i in qs else x for i, x in enumerate(saved)]
                walk(blk, qs)
                ends.append(list(clock))
            clock[:] = [max(e[i] for e in ends) for i in range(len(clock))]

    walk(qc, list(range(qc.num_qubits)))
    return max(clock)


def transpile_model(m: Any, backend: Any, cfg_t: dict[str, Any], length: int, mode: str):
    from qiskit import transpile

    from qmem import hardware as hw

    qc = hw.export_circuit(m, length, mode, reset_first=(length == 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transpile(
            qc,
            backend,
            optimization_level=cfg_t["optimization_level"],
            seed_transpiler=cfg_t["seed_transpiler"],
        )


def transpile_stats(cfg: dict[str, Any], trained: dict[str, Any]) -> list[dict[str, Any]]:
    from qmem import hardware as hw

    c = cfg["transpile"]
    ff_time = cfg["noise"]["feedforward_latency_s"]
    rows = []
    for bname in c["backends"]:
        be = get_backend(bname)
        for name, t in trained.items():
            m = t["model"]
            modes = ("dynamic", "coherent") if layout_name(m) == "feed-forward" else ("dynamic",)
            for mode in modes:
                for length in c["lengths"]:
                    tq = transpile_model(m, be, c, length, mode)
                    cq, phys = hw.compact(tq)
                    cq._phys = phys
                    ops = hw.count_ops(tq)
                    two = hw.two_qubit_count(tq)
                    one = sum(v for k, v in ops.items() if k in ("sx", "x"))
                    dur = step_duration(cq, be.target, ff_time)
                    row = {
                        "backend": bname,
                        "model": name,
                        "feedforward": mode if layout_name(m) == "feed-forward" else "none",
                        "length": length,
                        "physical_qubits": len(phys),
                        "depth": tq.depth(),
                        "two_qubit": two,
                        "one_qubit": one,
                        "rz": ops.get("rz", 0),
                        "measure": ops.get("measure", 0),
                        "reset": ops.get("reset", 0),
                        "control_flow": ops.get("if_else", 0) + ops.get("switch_case", 0),
                        "duration_s": dur,
                    }
                    for k in ("depth", "two_qubit", "one_qubit", "measure", "reset", "duration_s"):
                        row[f"{k}_per_symbol"] = row[k] / length
                    rows.append(row)
                    print(
                        f"[transpile] {bname} {name} {mode} T={length}: depth={row['depth']} "
                        f"2q={two} meas={row['measure']} reset={row['reset']} "
                        f"dur={dur * 1e6:.2f}us",
                        flush=True,
                    )
    return rows


# -- 4. noise -------------------------------------------------------------------------------
def step_for_backend(m: Any, be: Any, cfg: dict[str, Any]):
    """Compact transpiled step (reset first) with its memory positions in and out."""
    from qmem import hardware as hw

    tq = transpile_model(m, be, cfg["transpile"], 1, "dynamic")
    ini = tq.layout.initial_index_layout(filter_ancillas=True)
    fin = tq.layout.final_index_layout(filter_ancillas=True)
    cq, phys = hw.compact(tq)
    pos = {p: k for k, p in enumerate(phys)}
    mem_in = [pos[ini[i]] for i in range(m.n_memory)]
    mem_out = [pos[fin[i]] for i in range(m.n_memory)]
    return cq, phys, mem_in, mem_out


def sweep_noise(kind: str, value: float, base: Any) -> Any:
    from qmem import hardware as hw

    if kind == "depolarising":
        return hw.StepNoise(p1=value / 10, p2=value)
    if kind == "readout":
        return hw.StepNoise(readout=value)
    if kind == "reset":
        return hw.StepNoise(reset=value)
    if kind == "idle":
        return hw.StepNoise(
            t1=value,
            t2=value,
            measure_time=base.measure_time,
            reset_time=base.reset_time,
            feedforward_time=base.feedforward_time,
        )
    if kind == "scaled":
        return hw.scale_noise(base, value)
    raise ValueError(f"unknown sweep {kind!r}")


@torch.no_grad()
def score(t: dict[str, Any], model: Any, cfg: dict[str, Any], ctx: dict[str, Any]):
    """The model's task metrics: KL rate (PMD, markov) or recall accuracy (retention)."""
    from qmem.benchmarks import marked_recall_metrics
    from qmem.tasks import marked_recall_events
    from qmem.train import evaluate

    task, kind = t["task"], t["kind"]
    out: dict[str, Any] = {}
    if kind in ("pmd", "markov"):
        seqs = ctx["long"]
        diff = (-model.step_log_probs(seqs)) - (-task.log_probs(seqs))
        for length in cfg["noise"]["lengths"]:
            if length <= seqs.shape[1]:
                per = diff[:, :length].mean(-1)
                out[f"kl_T{length}"] = float(per.mean())
                out[f"kl_T{length}_se"] = float(per.std() / math.sqrt(len(per)))
        out["kl_rate"] = out[f"kl_T{seqs.shape[1]}"]
        out["kl_rate_se"] = out[f"kl_T{seqs.shape[1]}_se"]
    if kind in ("markov", "recall"):
        out["test_gap"] = evaluate(model, t["splits"].test, task)["gap_nats"]
    if kind == "recall":
        rec = marked_recall_metrics(model, task, t["splits"].test, cfg["analysis"]["delay_bins"])
        out["recall_acc"] = rec["recall_accuracy"]
        out["recall_by_delay"] = {str(b["delay_lo"]): b["accuracy"] for b in rec["by_delay"]}
        test = t["splits"].test
        ev = marked_recall_events(test)
        pred = model.predictive_probs(test)[:, :-1]
        r = ev["recall"]
        p_cue = pred[..., 2:4]
        p_right = torch.where(test == 3, p_cue[..., 1], p_cue[..., 0]) / p_cue.sum(-1)
        out["recall_soft"] = float(p_right[r].mean())
        g = cfg["noise"]["generation"]
        out.update(generative_recall(model.sample(g["n"], g["length"], g["seed"])))
    return out


def generative_recall(gen: torch.Tensor) -> dict[str, float]:
    """Share of recalls in sampled sequences that repeat their cue (measurable on a device)."""
    from qmem.tasks import CUE1, marked_recall_events

    ev = marked_recall_events(gen)
    r = ev["recall"]
    ok = ((gen == CUE1).long() == ev["cue_value"]) & r
    return {
        "gen_recall_acc": float(ok.sum() / r.sum()),
        "gen_recalls_per_sequence": float(r.sum() / gen.shape[0]),
    }


def noise_study(cfg: dict[str, Any], trained: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Noise rows of every model (each model's rows cached in ``<run>/noise/<model>.json``)."""
    rows: list[dict[str, Any]] = []
    calib: list[dict[str, Any]] = []
    for name, t in trained.items():
        path = out_dir / "noise" / f"{name}.json"
        if path.exists():
            part = json.loads(path.read_text())
        else:
            part = noise_model(cfg, name, t)
            save_json(path, part)
        rows += part["rows"]
        calib += part["calibration"]
    aer = aer_crosscheck(cfg, trained)
    return {"rows": rows, "calibration": calib, "aer": aer}


def noise_model(cfg: dict[str, Any], name: str, t: dict[str, Any]) -> dict[str, Any]:
    from qmem import hardware as hw

    c = cfg["noise"]
    ff_time = c["feedforward_latency_s"]
    lt = c["long_test"]
    rows: list[dict[str, Any]] = []
    calib: list[dict[str, Any]] = []
    m, task = t["model"], t["task"]
    ctx = {}
    if t["kind"] in ("pmd", "markov"):
        ctx["long"] = task.sample(lt["n"], lt["length"], lt["seed"])
    base_row = {"model": name, "kind": t["kind"]}
    rows.append({**base_row, "noise": "none", "value": 0.0, **score(t, m, cfg, ctx)})
    print(f"[noise] {name} noiseless: {short(rows[-1])}", flush=True)
    if t["kind"] == "recall":
        g = c["generation"]
        src = generative_recall(task.sample(g["n"], g["length"], g["seed"]))
        rows.append({**base_row, "noise": "source", "value": 0.0, **src})
    steps = {}
    for bname in c["backends"]:
        be = get_backend(bname)
        cq, phys, mem_in, mem_out = step_for_backend(m, be, cfg)
        steps[bname] = (cq, mem_in, mem_out)
        noise, info = hw.backend_noise(be, phys, ff_time)
        inst = hw.noisy_instrument(
            cq, m.n_memory, m.n_output, noise, mem_in, memory_qubits_out=mem_out
        )
        im = hw.InstrumentModel(inst, m.n_memory, m.n_output)
        rows.append(
            {
                **base_row,
                "noise": "backend",
                "backend": bname,
                "value": 1.0,
                **score(t, im, cfg, ctx),
            }
        )
        calib.append({"model": name, **info})
        print(f"[noise] {name} {bname}: {short(rows[-1])}", flush=True)
        if bname == c["sweep_backend"]:
            base_noise = noise
    cq, mem_in, mem_out = steps[c["sweep_backend"]]
    for kind, values in c["sweeps"].items():
        for v in values:
            noise = sweep_noise(kind, v, base_noise)
            inst = hw.noisy_instrument(
                cq, m.n_memory, m.n_output, noise, mem_in, memory_qubits_out=mem_out
            )
            im = hw.InstrumentModel(inst, m.n_memory, m.n_output)
            rows.append({**base_row, "noise": kind, "value": v, **score(t, im, cfg, ctx)})
            print(f"[noise] {name} {kind}={v:g}: {short(rows[-1])}", flush=True)
    return {"rows": rows, "calibration": calib}


def short(row: dict[str, Any]) -> str:
    keys = ("kl_rate", "test_gap", "recall_acc", "gen_recall_acc")
    return " ".join(f"{k}={row[k]:.5f}" for k in keys if k in row)


@torch.no_grad()
def aer_crosscheck(cfg: dict[str, Any], trained: dict[str, Any]) -> list[dict[str, Any]]:
    """Aer with the fake backend's own noise model against the exact calibrated instrument."""
    from qiskit_aer import AerSimulator

    from qmem import hardware as hw

    a = cfg["noise"]["aer_check"]
    out = []
    for bname in cfg["noise"]["backends"][:1]:
        be = get_backend(bname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # No shot branching here: in qiskit-aer 0.18.2 it mis-applies readout errors to
            # mid-circuit measurements (docs/phase5a-hardware.md); noiseless runs are fine.
            sim = AerSimulator.from_backend(be, method="density_matrix")
        for name in a["models"]:
            t = trained[name]
            m = t["model"]
            length = a["length"]
            while m.n_symbols**length > cfg["export"]["max_branches"]:
                length -= 1
            tq = transpile_model(m, be, cfg["transpile"], length, "dynamic")
            counts = sim.run(tq, shots=a["shots"], seed_simulator=a["seed"]).result().get_counts()
            emp = counts_to_tokens(counts)
            seqs = all_sequences(m.n_symbols, length)
            cq, phys, mem_in, mem_out = step_for_backend(m, be, cfg)
            row = {"backend": bname, "model": name, "length": length, "shots": a["shots"]}
            for label, noise in (
                ("noiseless", None),
                (
                    "calibrated",
                    hw.backend_noise(be, phys, cfg["noise"]["feedforward_latency_s"])[0],
                ),
                ("calibrated_no_idle", hw.backend_noise(be, phys, 0.0)[0]),
            ):
                if label == "calibrated_no_idle":
                    noise.t1 = None
                if noise is None:
                    p = m.log_prob(seqs).exp().numpy()
                else:
                    inst = hw.noisy_instrument(
                        cq, m.n_memory, m.n_output, noise, mem_in, memory_qubits_out=mem_out
                    )
                    p = hw.InstrumentModel(inst, m.n_memory, m.n_output).log_prob(seqs).exp()
                    p = p.numpy()
                row[f"tv_aer_vs_{label}"] = 0.5 * sum(
                    abs(emp.get(tuple(s), 0) / a["shots"] - p[i])
                    for i, s in enumerate(seqs.tolist())
                )
            row["tv_sampling_scale"] = 0.5 * math.sqrt(len(seqs) / a["shots"])
            out.append(row)
            print(f"[aer] {fmt_row(row)}", flush=True)
    return out


def thresholds(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per PMD model and sweep: the error at which the KL rate reaches the 2-state references."""
    ref = pmd_references(cfg)
    out = []
    for name in sorted({r["model"] for r in rows if r["kind"] == "pmd"}):
        r0 = next(r for r in rows if r["model"] == name and r["noise"] == "none")
        refs = ref[name]
        for kind in cfg["noise"]["sweeps"]:
            pts = sorted(
                (r["value"], r["kl_rate"])
                for r in rows
                if r["model"] == name and r["noise"] == kind
            )
            if kind == "idle":  # larger T is better: sweep in 1/T
                pts = sorted((1 / v, k) for v, k in pts)
            row = {"model": name, "sweep": kind, "noiseless_kl": r0["kl_rate"], **refs}
            for label in ("hmm_long_gap", "block_bound"):
                row[f"threshold_{label}"] = crossing([(0.0, r0["kl_rate"]), *pts], refs[label])
                x = row[f"threshold_{label}"]
                if kind == "idle" and x is not None:  # 0 (fails noiseless) -> T = infinity
                    row[f"threshold_{label}"] = 1 / x if x > 0 else math.inf
            out.append(row)
    return out


def crossing(pts: list[tuple[float, float]], level: float) -> float | None:
    """First x where y reaches ``level`` (log-linear interpolation; None if never)."""
    if pts[0][1] >= level:
        return 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if y1 >= level:
            if x0 <= 0:
                return x1 * (level - y0) / (y1 - y0) if y1 > y0 else x1
            f = (level - y0) / (y1 - y0)
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))
    return None


def pmd_references(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    path = REPO_ROOT / cfg["noise"]["reference_csv"]
    rows = list(csv.DictReader(path.open()))
    out = {}
    for m in cfg["models"]:
        task = m["task"]
        if not task.startswith("pmd"):
            continue
        hmm = [float(r["long_gap"]) for r in rows if r["task"] == task and r["family"] == "hmm"]
        bound = next(float(r["block_bound_nats"]) for r in rows if r["task"] == task)
        out[m["name"]] = {"hmm_long_gap": min(hmm), "block_bound": bound}
    return out


# -- 5. shots and cost ----------------------------------------------------------------------
@torch.no_grad()
def cost_study(cfg, trained, transpiled, noise_rows) -> dict[str, Any]:
    c = cfg["cost"]
    be_name = cfg["transpile"]["backends"][0]
    price_s = c["price_usd_per_minute"] / 60
    rows = []
    for name, t in trained.items():
        m, task = t["model"], t["task"]
        tr = next(
            r
            for r in transpiled
            if r["backend"] == be_name
            and r["model"] == name
            and r["length"] == 1
            and r["feedforward"] in ("dynamic", "none")
        )
        step_s = tr["duration_s"]
        n_sym = m.n_symbols
        row = {"model": name, "step_time_s": step_s, "n_parameters": m.n_parameters()}
        # (a) block-KL D_L/L(model || source) from generated blocks, plug-in estimator
        for length in c["block_lengths"]:
            if n_sym**length > 2**12:
                continue
            seqs = all_sequences(n_sym, length)
            lp_m = m.log_prob(seqs)
            lp_s = task.log_probs(seqs).sum(-1)
            pm = lp_m.exp()
            ok = pm > 0
            lr = torch.where(ok, lp_m - lp_s, torch.zeros_like(lp_m))
            d = float((pm * lr).sum())
            var = float((pm * lr**2).sum()) - d**2
            k = n_sym**length
            delta = c["kl_precision"]
            n_bias = (k - 1) / (length * delta)  # bias (K-1)/(2N) <= delta L / 2
            n_var = (4 * math.sqrt(max(var, 0)) / (length * delta)) ** 2
            n_blocks = max(n_bias, n_var) if math.isfinite(d) else math.inf
            row[f"blockkl_L{length}"] = d / length
            row[f"blockkl_L{length}_blocks"] = n_blocks
        # (b) TV of L-block distributions: E TV <= 0.5 sqrt(K/N)
        for length in (4, 8):
            row[f"tv_blocks_L{length}"] = n_sym**length / (4 * c["tv_precision"] ** 2)
        # (c) teacher-forced log-likelihood by post-selection on the device's own outcomes
        nll = t["record"]["test"]["nll_nats"]
        for length in c["loglik_lengths"]:
            p = math.exp(-nll * length)
            row[f"loglik_shots_T{length}"] = (1 - p) / (p * c["loglik_rel_precision"] ** 2)
        # (d) generative recall accuracy
        if t["kind"] == "recall":
            r0 = next(r for r in noise_rows if r["model"] == name and r["noise"] == "none")
            a = r0["gen_recall_acc"]
            n_rec = (1.96 / c["recall_precision"]) ** 2 * a * (1 - a)
            row["recall_recalls_needed"] = n_rec
            row["recall_shots"] = n_rec / r0["gen_recalls_per_sequence"]
        # (e) training by parameter shift
        evals = param_shift_evals(m) * cfg["cost"]["experiment_length"]
        steps = t["record"].get("steps_run") or 1000
        shots = steps * (evals + 1) * c["train_shots_per_eval"]
        row["train_steps"] = steps
        row["train_circuits_per_step"] = evals + 1
        row["train_shots"] = shots
        shot_s = c["rep_delay_s"] + c["experiment_length"] * step_s
        row["shot_time_s"] = shot_s
        row["train_qpu_s"] = shots * shot_s + steps * (evals + 1) * c["sub_job_overhead_s"]
        row["train_cost_usd"] = row["train_qpu_s"] * price_s
        rows.append(row)
    experiment = minimal_experiment(cfg, trained, transpiled, rows)
    witness = dimension_witness(cfg, trained)
    return {"rows": rows, "experiment": experiment, "witness": witness}


@torch.no_grad()
def window_probs(inst: Any, rho0: torch.Tensor, k_max: int, burn: int) -> torch.Tensor:
    """``g(k) = P(1 0^k 1)`` from the state after ``burn`` unread steps, ``k = 0..k_max``."""
    rho = rho0
    for _ in range(burn):
        rho = inst.channel(rho)
    first = inst.branch(rho, 1)
    out = []
    cur = first
    for _k in range(k_max + 1):
        out.append(float(inst.branch(cur, 1).diagonal().sum().real))
        cur = inst.branch(cur, 0)
    return torch.tensor(out, dtype=torch.float64)


def control_hmm(seed: int) -> Any:
    """A random 2-state binary HMM (labelled transitions): the witness's null control."""
    from qmem.tasks import hmm_task

    gen = torch.Generator().manual_seed(seed)
    lab = torch.rand(2, 2, 2, generator=gen, dtype=torch.float64)
    lab = lab / lab.sum((0, 2), keepdim=True)
    return hmm_task(lab.tolist(), name="control_2state")


def hankel_witness(g: torch.Tensor, j: int) -> float:
    """``det [g(j+a+b)]_{a,b<3} / max|entry|^3``: 0 for every 2-state HMM (#11's proof)."""
    h = torch.stack([g[j + a : j + a + 3] for a in range(3)])
    return float(torch.linalg.det(h) / h.abs().max() ** 3)


def empirical_windows(seqs: torch.Tensor, k_max: int, burn: int) -> torch.Tensor:
    """Per sequence, the frequency of ``1 0^k 1`` over start positions ``>= burn``."""
    n, length = seqs.shape
    out = torch.zeros(n, k_max + 1, dtype=torch.float64)
    ones = seqs == 1
    for k in range(k_max + 1):
        span = k + 2
        starts = range(burn, length - span + 1)
        hit = torch.zeros(n, dtype=torch.float64)
        for t in starts:
            w = ones[:, t] & ones[:, t + k + 1]
            if k:
                w = w & ~ones[:, t + 1 : t + k + 1].any(-1)
            hit += w.double()
        out[:, k] = hit / len(starts)
    return out


@torch.no_grad()
def dimension_witness(cfg: dict[str, Any], trained: dict[str, Any]) -> list[dict[str, Any]]:
    """The Hankel dimension witness of each PMD model: exact, and from a simulated experiment.

    A 2-state HMM's ``g(k)`` obeys an order-2 linear recurrence whatever its initial state,
    so every 3x3 Hankel determinant of ``g`` vanishes (``tests/test_merge_bound.py``). A
    significantly non-zero determinant in a device's samples therefore excludes every
    2-state classical model. The z-score is the determinant over its bootstrap standard
    error (over shots) in ``experiment_shots`` sampled sequences of ``experiment_length``.
    """
    from qmem import hardware as hw

    w = cfg["cost"]["witness"]
    c = cfg["cost"]
    out = []
    for name, t in trained.items():
        if t["kind"] != "pmd":
            continue
        m, task = t["model"], t["task"]
        variants: list[tuple[str, Any]] = [
            ("source", task),
            ("control: 2-state HMM", control_hmm(w["seed"])),
            ("noiseless", m.instrument()),
        ]
        for bname in cfg["noise"]["backends"]:
            be = get_backend(bname)
            cq, phys, mem_in, mem_out = step_for_backend(m, be, cfg)
            noise, _ = hw.backend_noise(be, phys, cfg["noise"]["feedforward_latency_s"])
            inst = hw.noisy_instrument(cq, 1, 1, noise, mem_in, memory_qubits_out=mem_out)
            variants.append((bname, inst))
        for label, inst in variants:
            row: dict[str, Any] = {"model": name, "variant": label}
            if not hasattr(inst, "sample"):  # an instrument: exact witness, then sample it
                g = window_probs(inst, m.rho0, w["k_max"], w["burn"])
                row.update({f"exact_j{j}": hankel_witness(g, j) for j in w["offsets"]})
                sampler = hw.InstrumentModel(inst, 1, 1)
                seqs = sampler.sample(c["experiment_shots"], c["experiment_length"], w["seed"])
            else:  # a classical source: sample it directly
                seqs = inst.sample(c["experiment_shots"], c["experiment_length"], w["seed"])
            per = empirical_windows(seqs, w["k_max"], w["burn"])
            gen = torch.Generator().manual_seed(w["seed"])
            boots = []
            for _ in range(w["bootstrap"]):
                idx = torch.randint(len(per), (len(per),), generator=gen)
                boots.append(per[idx].mean(0))
            g_hat = per.mean(0)
            for j in w["offsets"]:
                est = hankel_witness(g_hat, j)
                se = statistics.stdev(hankel_witness(b, j) for b in boots)
                row[f"sampled_j{j}"] = est
                row[f"z_j{j}"] = est / se if se > 0 else math.nan
            out.append(row)
            print(f"[witness] {fmt_row(row)}", flush=True)
    return out


def param_shift_evals(m: Any) -> int:
    """Circuit evaluations per gradient for one step's parameters (each gate occurrence once).

    RX/RY/RZ angles: two-term shift rule (2 evaluations); controlled-RY: four-term rule. A
    parameter that occurs in every step of a length-T circuit needs this for each occurrence.
    """
    from qmem.circuits import ControlledRotationBlock

    n = 0
    for mod in m.modules():
        if isinstance(mod, ControlledRotationBlock):
            if isinstance(mod.strength, torch.nn.Parameter):
                n += 4 * len(mod.pairs)
        elif hasattr(mod, "theta") and isinstance(mod.theta, torch.nn.Parameter):
            n += 2 * mod.theta.numel()
    return n


def minimal_experiment(cfg, trained, transpiled, cost_rows) -> list[dict[str, Any]]:
    """QPU time and cost of the proposed generation-only runs."""
    c = cfg["cost"]
    price_s = c["price_usd_per_minute"] / 60
    out = []
    for r in cost_rows:
        shots = c["experiment_shots"]
        qpu = c["sub_job_overhead_s"] + shots * r["shot_time_s"]
        out.append(
            {
                "model": r["model"],
                "shots": shots,
                "length": c["experiment_length"],
                "blocks_L8": shots * (c["experiment_length"] // 8),
                "qpu_s": qpu,
                "cost_usd": qpu * price_s,
                "open_plan_fraction": qpu / (60 * c["open_plan_minutes"]),
            }
        )
    return out


# -- summary --------------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        flat.append({k: (json.dumps(v) if isinstance(v, dict | list) else v) for k, v in r.items()})
    keys: list[str] = []
    for r in flat:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(flat)


def g(x: Any, nd: int = 4) -> str:
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e6):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def markdown(cfg, models, export, transpiled, noise, thr, cost, meta) -> str:
    L = [f"# {cfg['experiment']['title']}", ""]
    L += [
        f"Generated by `experiments/phase5a.py` from `{cfg['_path']}`. Run dir "
        f"`{meta['run_dir']}`, commit `{meta['git_commit'][:7]}`"
        f"{' (dirty)' if meta['git_dirty'] else ''}. Simulation only.",
        "",
    ]
    L += ["## Models (re-trained, validation-selected seed)", ""]
    L += [
        "| model | task | layout | n | qubits | params | test gap | reproduction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in models:
        rep = ", ".join(f"{k[6:-5]} Δ {g(v, 2)}" for k, v in r.items() if k.startswith("repro_"))
        L.append(
            f"| {r['model']} | {r['task']} | {r['layout']} | {r['n_memory']} | "
            f"{r['n_qubits']} | {r['n_parameters']} | {g(r['test_gap'])} | {rep} |"
        )
    L += [
        "",
        "## Export: exactness against qmem",
        "",
        "Max |Δp| over every token string, max |Δρ| (conditional memory) and the Aer "
        "density-matrix check (sampled).",
        "",
        "| model | T | strings | dynamic Δp | dynamic Δρ | coherent Δp | deferred Δp | Aer TV | "
        "Aer χ² p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in export:
        L.append(
            f"| {r['model']} | {r['length']} | {r['n_strings']} | "
            f"{g(r['dynamic_max_dp'])} | {g(r['dynamic_max_drho'])} | "
            f"{g(r.get('coherent_max_dp'))} | {g(r.get('deferred_max_dp'))} | "
            f"{g(r.get('aer_tv'))} | {g(r.get('aer_chi2_pvalue'), 3)} |"
        )
    L += [
        "",
        "## Transpilation",
        "",
        "Per symbol from the T = 32 transpile (control flow: the costlier branch). Duration: "
        "ASAP schedule of one step with the backend's gate, measurement and reset durations "
        f"and an assumed {cfg['noise']['feedforward_latency_s'] * 1e6:g} µs feed-forward "
        "latency.",
        "",
        "| backend | model | feed-forward | qubits | depth/sym | 2q/sym | 1q/sym | meas/sym | "
        "reset/sym | step µs | T=32 depth | T=32 2q |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in transpiled:
        if r["length"] != 32:
            continue
        one = next(
            x
            for x in transpiled
            if x["backend"] == r["backend"]
            and x["model"] == r["model"]
            and x["feedforward"] == r["feedforward"]
            and x["length"] == 1
        )
        L.append(
            f"| {r['backend']} | {r['model']} | {r['feedforward']} | {r['physical_qubits']} | "
            f"{g(r['depth_per_symbol'], 1)} | {g(r['two_qubit_per_symbol'], 1)} | "
            f"{g(r['one_qubit_per_symbol'], 1)} | {g(r['measure_per_symbol'], 2)} | "
            f"{g(r['reset_per_symbol'], 2)} | {g(one['duration_s'] * 1e6, 2)} | "
            f"{r['depth']} | {r['two_qubit']} |"
        )
    rows = noise["rows"]
    L += [
        "",
        "## Noise: PMD KL rate (nats/symbol) against error rate",
        "",
        "Long re-scoring as #11 (256 x 1024, seed 777). *2-state HMM*: the best 2-state "
        "classical model's long gap; *bound*: the unifilar block bound (#11).",
        "",
    ]
    pmd = [n for n in dict.fromkeys(r["model"] for r in rows) if n.startswith("pmd")]
    refs = pmd_references(cfg)
    hdr = ["noise", "value", *pmd]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L.append("| 2-state HMM | | " + " | ".join(g(refs[n]["hmm_long_gap"]) for n in pmd) + " |")
    L.append("| unifilar bound | | " + " | ".join(g(refs[n]["block_bound"]) for n in pmd) + " |")
    for key in noise_keys(rows, pmd[0]):
        vals = [
            next((r for r in rows if r["model"] == n and noise_key(r) == key), None) for n in pmd
        ]
        L.append(
            f"| {key[0]} | {g(key[1], 4) if key[0] != 'backend' else key[1]} | "
            + " | ".join(g(v["kl_rate"]) if v else "-" for v in vals)
            + " |"
        )
    L += [
        "",
        "**KL rate against T** (per symbol over the first T symbols), backend noise:",
        "",
        "| model | noise | " + " | ".join(f"T={x}" for x in cfg["noise"]["lengths"]) + " |",
        "|---|---|" + "---|" * len(cfg["noise"]["lengths"]),
    ]
    for r in rows:
        if r["kind"] == "pmd" and r["noise"] in ("none", "backend"):
            lab = r.get("backend", "noiseless")
            L.append(
                f"| {r['model']} | {lab} | "
                + " | ".join(g(r.get(f"kl_T{x}")) for x in cfg["noise"]["lengths"])
                + " |"
            )
    L += [
        "",
        "**Thresholds**: error at which the KL rate reaches the 2-state references "
        "(log-interpolated; idle: T1 = T2 in seconds, larger is better; scaled: factor on "
        f"{cfg['noise']['sweep_backend']}'s calibration).",
        "",
        "| model | sweep | noiseless | HMM | → threshold | bound | → threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in thr:
        L.append(
            f"| {r['model']} | {r['sweep']} | {g(r['noiseless_kl'])} | "
            f"{g(r['hmm_long_gap'])} | {g(r['threshold_hmm_long_gap'])} | "
            f"{g(r['block_bound'])} | {g(r['threshold_block_bound'])} |"
        )
    oth = [n for n in dict.fromkeys(r["model"] for r in rows) if not n.startswith("pmd")]
    L += [
        "",
        "## Noise: markov1_quaternary gap and retention recall",
        "",
        "*recall*: teacher-forced recall value accuracy on the test split (source 0.95); "
        "*gen recall*: fraction of recalls in the model's own sampled sequences that repeat "
        "the cue (measurable on hardware); *soft recall*: mean teacher-forced probability of the "
        "right value at recalls (what sampling reproduces). Row *source*: the source's own "
        "sampled recall rate.",
        "",
        "| model | noise | value | test gap | KL rate | recall | soft recall | gen recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in oth:
        for r in rows:
            if r["model"] != n:
                continue
            v = r.get("backend") if r["noise"] == "backend" else g(r["value"], 4)
            L.append(
                f"| {n} | {r['noise']} | {v} | {g(r.get('test_gap'))} | "
                f"{g(r.get('kl_rate'))} | {g(r.get('recall_acc'), 3)} | "
                f"{g(r.get('recall_soft'), 3)} | "
                f"{g(r.get('gen_recall_acc'), 3)} |"
            )
    L += ["", "**Recall accuracy by delay** (teacher-forced; noiseless and backend noise):", ""]
    bins = cfg["analysis"]["delay_bins"]
    L += [
        "| model | noise | " + " | ".join(f"≥{b}" for b in bins) + " |",
        "|---|---|" + "---|" * len(bins),
    ]
    for r in rows:
        if r["kind"] == "recall" and r["noise"] in ("none", "backend"):
            bd = r.get("recall_by_delay", {})
            L.append(
                f"| {r['model']} | {r.get('backend', 'noiseless')} | "
                + " | ".join(g(bd.get(str(b)), 3) for b in bins)
                + " |"
            )
    L += [
        "",
        "**Aer cross-check** (the fake backend's own Aer noise model on the transpiled "
        "T-step circuit, sampled, against this study's exact instruments):",
        "",
        "| backend | model | T | shots | TV vs noiseless | TV vs calibrated | "
        "TV vs calibrated, no idle | sampling scale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in noise["aer"]:
        L.append(
            f"| {r['backend']} | {r['model']} | {r['length']} | {r['shots']} | "
            f"{g(r['tv_aer_vs_noiseless'])} | {g(r['tv_aer_vs_calibrated'])} | "
            f"{g(r['tv_aer_vs_calibrated_no_idle'])} | {g(r['tv_sampling_scale'])} |"
        )
    L += [
        "",
        "**Calibration used** (median over the step's qubits):",
        "",
        "| backend | model | qubits | sx error | 2q error | readout | T1 µs | T2 µs | "
        "measure µs | reset µs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in noise["calibration"]:
        med = lambda d: statistics.median(d.values()) if d else math.nan  # noqa: E731
        L.append(
            f"| {r['backend']} | {r['model']} | {r['physical_qubits']} | "
            f"{g(med(r['sx_error']), 5)} | {g(med(r['two_qubit_error']), 5)} | "
            f"{g(med(r['readout_error']), 5)} | {g(med(r['t1_s']) * 1e6, 0)} | "
            f"{g(med(r['t2_s']) * 1e6, 0)} | {g(r['measure_time_s'] * 1e6, 2)} | "
            f"{g(r['reset_time_s'] * 1e6, 2)} |"
        )
    c = cfg["cost"]
    L += [
        "",
        "## Shots, QPU time and cost",
        "",
        f"Price {c['price_usd_per_minute']:g} USD/min (Pay-As-You-Go, ibm.com/quantum/pricing, "
        f"read 2026-11-07); shot time = rep_delay {c['rep_delay_s'] * 1e6:g} µs + T x step; "
        f"{c['sub_job_overhead_s']:g} s per sub-job.",
        "",
        f"**Block KL** D_L/L(model ‖ source) of the noiseless model and the generated blocks "
        f"needed to estimate it to ±{c['kl_precision']:g} nats/symbol (plug-in estimator: "
        "bias (K-1)/2N and 2sigma both within half the precision):",
        "",
    ]
    Ls = [x for x in c["block_lengths"]]
    L += ["| model | " + " | ".join(f"L={x}" for x in Ls) + " |", "|---|" + "---|" * len(Ls)]
    for r in cost["rows"]:
        cells = []
        for x in Ls:
            if f"blockkl_L{x}" in r:
                cells.append(f"{g(r[f'blockkl_L{x}'])} ({g(r[f'blockkl_L{x}_blocks'], 0)})")
            else:
                cells.append("-")
        L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    L += [
        "",
        "**Teacher-forced log-likelihood by post-selection** (shots per sequence for "
        f"±{c['loglik_rel_precision']:g} nats on log p; p = exp(-T · test NLL)):",
        "",
        "| model | " + " | ".join(f"T={x}" for x in c["loglik_lengths"]) + " |",
        "|---|" + "---|" * len(c["loglik_lengths"]),
    ]
    for r in cost["rows"]:
        L.append(
            f"| {r['model']} | "
            + " | ".join(g(r[f"loglik_shots_T{x}"], 0) for x in c["loglik_lengths"])
            + " |"
        )
    L += [
        "",
        f"**Training on hardware** (parameter shift per gate occurrence x T = "
        f"{c['experiment_length']}, {c['train_shots_per_eval']} shots per circuit, the run's "
        "own step count):",
        "",
        "| model | params | step µs | shot µs | steps | circuits/step | shots | QPU hours | "
        "cost (USD) | recall shots |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cost["rows"]:
        L.append(
            f"| {r['model']} | {r['n_parameters']} | {g(r['step_time_s'] * 1e6, 2)} | "
            f"{g(r['shot_time_s'] * 1e6, 1)} | {r['train_steps']} | "
            f"{r['train_circuits_per_step']} | {g(float(r['train_shots']), 0)} | "
            f"{g(r['train_qpu_s'] / 3600, 1)} | {g(r['train_cost_usd'], 0)} | "
            f"{g(r.get('recall_shots'), 0)} |"
        )
    L += [
        "",
        f"**Generation-only experiment** ({c['experiment_shots']} shots of "
        f"T = {c['experiment_length']}):",
        "",
        "| model | L=8 blocks | QPU s | cost (USD) | share of Open plan month |",
        "|---|---|---|---|---|",
    ]
    for r in cost["experiment"]:
        L.append(
            f"| {r['model']} | {r['blocks_L8']} | {g(r['qpu_s'], 1)} | "
            f"{g(r['cost_usd'], 0)} | {g(r['open_plan_fraction'], 2)} |"
        )
    w = cfg["cost"]["witness"]
    js = w["offsets"]
    L += [
        "",
        "**Hankel dimension witness** det[g(j+a+b)]/max^3 with g(k) = P(1 0^k 1) after "
        f"{w['burn']} steps: exactly 0 for every 2-state HMM. *exact*: from the instrument; "
        f"*sampled (z)*: estimate from {c['experiment_shots']} simulated shots of "
        f"T = {c['experiment_length']} and its z-score (bootstrap SE over shots).",
        "",
        "| model | variant | "
        + " | ".join(f"exact j={j}" for j in js)
        + " | "
        + " | ".join(f"sampled j={j} (z)" for j in js)
        + " |",
        "|---|---|" + "---|" * (2 * len(js)),
    ]
    for r in cost["witness"]:
        L.append(
            f"| {r['model']} | {r['variant']} | "
            + " | ".join(g(r.get(f"exact_j{j}")) for j in js)
            + " | "
            + " | ".join(f"{g(r[f'sampled_j{j}'])} ({g(r[f'z_j{j}'], 1)})" for j in js)
            + " |"
        )
    return "\n".join(L) + "\n"


def noise_key(r: dict[str, Any]) -> tuple[str, Any]:
    return (r["noise"], r.get("backend") if r["noise"] == "backend" else r["value"])


def noise_keys(rows: list[dict[str, Any]], model: str) -> list[tuple[str, Any]]:
    return [noise_key(r) for r in rows if r["model"] == model]


# -- driver -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--from-run", type=Path)
    ap.add_argument("--stages", type=str, default=",".join(STAGES))
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    stages = args.stages.split(",")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    out_dir = args.from_run or new_run_dir(cfg["experiment"]["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config_meta.json", run_metadata(cfg, None))
    print(f"run dir: {out_dir}", flush=True)
    if "train" in stages:
        train_models(cfg, out_dir, args.workers or cfg["experiment"]["workers"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    trained = load_trained(cfg, out_dir)
    summary_dir = REPO_ROOT / cfg["experiment"]["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    models = [describe(n, t) for n, t in trained.items()]
    results: dict[str, Any] = {}
    cache = out_dir / "stages.json"
    if cache.exists():
        results = json.loads(cache.read_text())
    t0 = time.perf_counter()

    def done(stage: str) -> None:
        results.setdefault("stage_seconds", {})[stage] = time.perf_counter() - t0
        save_json(cache, results)

    if "export" in stages:
        results["export"] = export_checks(cfg, trained, summary_dir)
        done("export")
    if "transpile" in stages:
        results["transpile"] = transpile_stats(cfg, trained)
        done("transpile")
    if "noise" in stages:
        results["noise"] = noise_study(cfg, trained, out_dir)
        results["thresholds"] = thresholds(cfg, results["noise"]["rows"])
        done("noise")
    if "cost" in stages:
        results["cost"] = cost_study(cfg, trained, results["transpile"], results["noise"]["rows"])
        done("cost")
    missing = [k for k in ("export", "transpile", "noise", "cost") if k not in results]
    if missing:
        print(f"stages not yet run: {missing}; no summary written", flush=True)
        return
    meta = run_metadata(cfg, None)
    meta["run_dir"] = str(out_dir.relative_to(REPO_ROOT)) if out_dir.is_absolute() else str(out_dir)
    save_json(summary_dir / "summary.json", {"meta": meta, "models": models, **results})
    write_csv(summary_dir / "models.csv", models)
    write_csv(summary_dir / "export.csv", results["export"])
    write_csv(summary_dir / "transpile.csv", results["transpile"])
    write_csv(summary_dir / "noise.csv", results["noise"]["rows"])
    write_csv(summary_dir / "thresholds.csv", results["thresholds"])
    write_csv(summary_dir / "aer_check.csv", results["noise"]["aer"])
    write_csv(summary_dir / "cost.csv", results["cost"]["rows"])
    write_csv(summary_dir / "experiment.csv", results["cost"]["experiment"])
    write_csv(summary_dir / "witness.csv", results["cost"]["witness"])
    (summary_dir / "summary.md").write_text(
        markdown(
            cfg,
            models,
            results["export"],
            results["transpile"],
            results["noise"],
            results["thresholds"],
            results["cost"],
            meta,
        )
    )
    plots(cfg, results, summary_dir)
    print(f"summary: {summary_dir}", flush=True)


def plots(cfg: dict[str, Any], results: dict[str, Any], summary_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["noise"]["rows"]
    refs = pmd_references(cfg)
    pmd = [n for n in dict.fromkeys(r["model"] for r in rows) if n.startswith("pmd")]
    sweeps = [k for k in cfg["noise"]["sweeps"] if k != "idle"]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4 * len(sweeps), 3.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, kind in zip(axes, sweeps, strict=True):
        for i, n in enumerate(pmd):
            col = cmap(i / max(1, len(pmd) - 1))
            pts = sorted(
                (r["value"], r["kl_rate"]) for r in rows if r["model"] == n and r["noise"] == kind
            )
            ax.plot(*zip(*pts, strict=True), "o-", color=col, label=n, ms=3)
            ax.axhline(refs[n]["hmm_long_gap"], color=col, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            {
                "depolarising": "2-qubit depolarising p (p1 = p/10)",
                "readout": "readout flip probability",
                "reset": "reset error probability",
                "scaled": f"factor on {cfg['noise']['sweep_backend']} errors",
            }[kind]
        )
        ax.set_title(kind)
    axes[0].set_ylabel("KL rate (nats/symbol)")
    axes[-1].legend(fontsize=7, title="dotted: best 2-state HMM", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(summary_dir / "pmd_noise.png", dpi=130)
    plt.close(fig)

    rec = [n for n in dict.fromkeys(r["model"] for r in rows) if n.startswith("recall")]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4 * len(sweeps), 3.4), sharey=True)
    for ax, kind in zip(axes, sweeps, strict=True):
        for n, mk in zip(rec, ("o", "s"), strict=False):
            for key, ls in (("recall_acc", "-"), ("gen_recall_acc", "--")):
                pts = sorted(
                    (r["value"], r[key]) for r in rows if r["model"] == n and r["noise"] == kind
                )
                ax.plot(
                    *zip(*pts, strict=True),
                    mk + ls,
                    ms=3,
                    label=f"{n} ({'teacher-forced' if key == 'recall_acc' else 'generated'})",
                )
        ax.axhline(0.95, color="k", lw=0.8, ls=":")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(kind)
    axes[0].set_ylabel("recall accuracy")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(summary_dir / "recall_noise.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
