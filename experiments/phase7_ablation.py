"""Phase 7 Stage 2A (issue #20): q75 destruction / dephasing controls, designed and simulated.

Run: uv run python experiments/phase7_ablation.py --config experiments/configs/phase7_ablation.toml

No QPU and no IBM service of any kind. ``--export`` (needs the ``hardware`` group) builds every
control from Phase 6's committed transpiled circuits, checks it against ``FakeKingston``'s
target, exports its QASM and the superoperators of the step pieces
(``results/summaries/phase7_ablation/pieces.npz``). The default stage (numpy only) predicts
each control's witness, g, P(1), L-block statistics and paired NLL under H_q75 and the H_other
alternatives of the config, and the shots and QPU seconds each discrimination needs.
``--plan`` prints the Stage 3 job and its cost; nothing here can submit a job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import tomllib
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qmem import ablate_hw as A
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

# -- helpers -----------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def plain(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)


def sdir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def p6dir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


def phase6_summary(cfg) -> dict[str, Any]:
    return json.loads((p6dir(cfg) / "summary.json").read_text())


def settings(cfg) -> A.Settings:
    w = cfg["witness"]
    return A.Settings(w["length"], w["burn"], w["k_max"], tuple(w["offsets"]), w["block"])


def pauli_table(cfg) -> np.ndarray:
    """(M, length) Pauli indices of the A3 instances: instance i from default_rng([seed, i])."""
    a = cfg["a3"]
    n = cfg["circuits"]["length"]
    return np.stack(
        [np.random.default_rng([a["seed"], i]).integers(0, 4, n) for i in range(a["instances"])]
    )


# -- 1. export (hardware group) ----------------------------------------------------------------


def noise_from_json(nd: dict[str, Any]) -> Any:
    from qmem import hardware as hw

    kw = {}
    for k, v in nd.items():
        if isinstance(v, list):
            v = {(tuple(a) if isinstance(a, list) else a): b for a, b in v}
        kw[k] = v
    return hw.StepNoise(**kw)


def noise_levels(cfg, s6) -> dict[str, Any]:
    from qmem import hardware as hw

    cal = noise_from_json(s6["prepare"]["calibration"]["noise"])
    out: dict[str, Any] = {}
    for name in cfg["noise"]["levels"]:
        if name == "noiseless":
            out[name] = None
        elif name == "calibrated":
            out[name] = cal
        elif name == "calibrated_x3":
            out[name] = hw.scale_noise(cal, cfg["noise"]["x3_factor"])
        else:
            raise ValueError(name)
    return out


def durations(cfg, s6, target) -> A.Durations:
    lay = cfg["layout"]
    x = target["x"][(lay["memory"],)].duration
    if cfg["circuits"]["delays"] == "snapshot":
        n = s6["prepare"]["calibration"]["noise"]
        return A.Durations(n["reset_time"], n["measure_time"], x, target.dt)
    q = (lay["memory"],)
    return A.Durations(target["reset"][q].duration, target["measure"][q].duration, x, target.dt)


def isa(qc, backend, cfg):
    from qiskit import transpile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transpile(
            qc,
            backend,
            optimization_level=cfg["circuits"]["optimization_level"],
            seed_transpiler=cfg["circuits"]["seed_transpiler"],
            initial_layout=list(range(qc.num_qubits)),
        )


def signature(qc) -> list[tuple]:
    return [
        (
            i.operation.name,
            tuple(qc.find_bit(q).index for q in i.qubits),
            tuple(round(float(p), 12) for p in i.operation.params),
        )
        for i in qc.data
    ]


def per_step_counts(qc, base_counts: dict[str, int], length: int) -> dict[str, float]:
    c = dict(qc.count_ops())
    keys = set(c) | set(base_counts)
    return {k: (c.get(k, 0) - base_counts.get(k, 0)) / length for k in sorted(keys)}


def export(cfg: dict[str, Any]) -> None:
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    sd = sdir(cfg)
    (sd / "qasm").mkdir(parents=True, exist_ok=True)
    s6 = phase6_summary(cfg)
    backend = FakeKingston()
    target = backend.target
    lay = cfg["layout"]
    length = cfg["circuits"]["length"]
    dur = durations(cfg, s6, target)
    step = qasm3.loads((p6dir(cfg) / "step.qasm").read_text())
    full = qasm3.loads((p6dir(cfg) / "circuit.qasm").read_text())
    paulis = pauli_table(cfg)

    # -- hardware support (FakeKingston's target)
    mem, out_q, spec = lay["memory"], lay["output"], lay["spectator"]
    nbrs = sorted({b for a, b in backend.coupling_map.get_edges() if a == mem})
    support = {
        "backend": backend.name,
        "num_qubits": backend.num_qubits,
        "operations": sorted(target.operation_names),
        "dt_s": target.dt,
        "neighbours_of_memory": nbrs,
        "spectator_is_neighbour_of_memory_only": spec in nbrs
        and spec not in {b for a, b in backend.coupling_map.get_edges() if a == out_q},
        "qubits": {},
    }
    for q in (out_q, mem, spec):
        row = {}
        for name in ("measure", "reset", "x", "sx", "delay"):
            props = target[name].get((q,)) if name in target.operation_names else None
            row[name] = {
                "supported": (q,) in target[name] if name in target.operation_names else False,
                "duration_s": None if props is None else props.duration,
                "error": None if props is None else props.error,
            }
        qp = target.qubit_properties[q]
        row["t1_s"], row["t2_s"] = qp.t1, qp.t2
        support["qubits"][str(q)] = row
    m = support["qubits"][str(mem)]
    support["reset_is_measure_plus_x"] = {
        "reset_s": m["reset"]["duration_s"],
        "measure_plus_x_s": m["measure"]["duration_s"] + m["x"]["duration_s"],
    }
    support["dynamic_circuits"] = "if_else" in target.operation_names
    support["snapshot_live_durations_s"] = {
        "measure": s6["prepare"]["calibration"]["noise"]["measure_time"],
        "reset": s6["prepare"]["calibration"]["noise"]["reset_time"],
    }
    support["delays_used_s"] = {
        "reset": dur.reset,
        "measure": dur.measure,
        "x": dur.x,
        "dt_units": {
            "reset": dur.dt_units(dur.reset),
            "measure": dur.dt_units(dur.measure),
            "x": dur.dt_units(dur.x),
        },
    }

    # -- full circuits
    base_isa = isa(full, backend, cfg)
    support["baseline_isa_equals_phase6_circuit"] = signature(base_isa) == signature(full)
    base_counts = dict(base_isa.count_ops())
    base_dur = base_isa.estimate_duration(target, unit="s")
    rows, hashes = [], {}

    def record(name: str, qc, tq, write: bool) -> None:
        text = qasm3.dumps(tq)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        if write:
            (sd / "qasm" / f"{name}.qasm").write_text(text)
        d = tq.estimate_duration(target, unit="s")
        rows.append(
            {
                "circuit": name,
                "variant": name.split("_")[0] if name.startswith("A3_") else name,
                "qubits_used": sorted({tq.find_bit(q).index for i in tq.data for q in i.qubits}),
                "clbits": tq.num_clbits,
                "depth": tq.depth(),
                "duration_fake_s": d,
                "duration_minus_baseline_s": d - base_dur,
                "per_step_minus_baseline": per_step_counts(tq, base_counts, length),
                "transpile_preserved": signature(tq) == signature(qc)
                or dict(tq.count_ops()) == dict(qc.count_ops()),
                "measure_supported": all(
                    (tq.find_bit(i.qubits[0]).index,) in target["measure"]
                    for i in tq.data
                    if i.operation.name == "measure"
                ),
                "reset_supported": all(
                    (tq.find_bit(i.qubits[0]).index,) in target["reset"]
                    for i in tq.data
                    if i.operation.name == "reset"
                ),
            }
        )

    for v in cfg["circuits"]["variants"]:
        if v == "A3":
            for i in range(len(paulis)):
                ps = [A.PAULIS[k] for k in paulis[i]]
                qc = A.build_variant(full, v, mem, out_q, dur, length, spec, ps)
                record(f"A3_{i:03d}", qc, isa(qc, backend, cfg), i < cfg["a3"]["qasm_instances"])
            continue
        qc = A.build_variant(full, v, mem, out_q, dur, length, spec)
        record(v, qc, isa(qc, backend, cfg), True)

    # -- step pieces per noise level, and the serial-placement sensitivity
    smem, sout = lay["step_memory"], lay["step_output"]
    npz: dict[str, np.ndarray] = {}
    for name, nz in noise_levels(cfg, s6).items():
        npz.update(A.export_pieces(step, nz, smem, sout).to_dict(name))
    cal = noise_levels(cfg, s6)["calibrated"]
    for v in ("A1", "A2"):
        sc = A.build_variant(step, v, smem, sout, dur, 1, placement="serial", drop_reset=True)
        npz[f"serial__{v}"] = A.instrument_superops(sc, cal, [smem, sout], 1)
    npz["pauli_table"] = paulis
    np.savez(sd / "pieces.npz", **npz)
    (sd / "qasm" / "step_baseline.qasm").write_text(qasm3.dumps(step))
    for v in ("A1", "A2", "A3s"):
        sc = A.build_variant(step, v, smem, sout, dur, 1)
        (sd / "qasm" / f"step_{v}.qasm").write_text(qasm3.dumps(sc))
    meta = run_metadata(plain(cfg), cfg["a3"]["seed"])
    save_json(
        sd / "circuits.json",
        {
            "meta": meta,
            "baseline_duration_fake_s": base_dur,
            "hardware_support": support,
            "circuits": rows,
            "qasm_sha256": hashes,
        },
    )
    write_csv(
        sd / "durations.csv",
        [
            {
                "circuit": r["circuit"],
                "duration_fake_us": round(r["duration_fake_s"] * 1e6, 3),
                "minus_baseline_us": round(r["duration_minus_baseline_s"] * 1e6, 3),
                "depth": r["depth"],
                "clbits": r["clbits"],
                **{f"per_step_{k}": v for k, v in r["per_step_minus_baseline"].items()},
            }
            for r in rows
            if not r["circuit"].startswith("A3_") or r["circuit"] == "A3_000"
        ],
    )
    print(json.dumps(support, indent=1, default=str))


# -- 2. predictions (numpy) --------------------------------------------------------------------


def load_pieces(cfg) -> tuple[dict[str, A.StepPieces], Any]:
    path = sdir(cfg) / "pieces.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: run once with --export (hardware group)")
    z = np.load(path)
    return {n: A.StepPieces.from_npz(z, n) for n in cfg["noise"]["levels"]}, z


def reference(cfg) -> W.LinearProcess:
    """Phase 6's calibrated 1-qubit model (#18's committed superoperators)."""
    z = np.load(REPO_ROOT / cfg["experiment"]["audit_dir"] / "instruments.npz")
    return W.LinearProcess.instrument(z["calibrated"], z["rho0"])


def hypotheses(cfg) -> list[A.Hypothesis]:
    out = []
    for h in cfg["hypotheses"]:
        kw = dict(h)
        for k in ("q59_reset", "q75_reset"):
            if k in kw:
                kw[k] = tuple(kw[k])
        out.append(A.Hypothesis(**kw))
    return out


def variant_process(
    v: str, pieces: A.StepPieces, hyp: A.Hypothesis, z: Any = None, serial: bool = False
) -> W.LinearProcess:
    if serial:
        sup = z[f"serial__{v}"]
        rs = A.reset_superop(pieces, hyp)
        return A.process(np.stack([sup[y] @ rs for y in range(2)]))
    if v == "A3":  # M -> infinity: independent uniform Paulis average per step
        sup = np.mean([A.step_superops(v, pieces, hyp, p) for p in A.PAULIS], axis=0)
        return A.process(sup)
    return A.process(A.step_superops(v, pieces, hyp))


def drift_process(cfg, d: dict[str, Any], v: str, pieces: dict[str, A.StepPieces]):
    a = variant_process(v, pieces["calibrated"], A.Hypothesis())
    rb = dict(d["regime_b"])
    hb = A.Hypothesis(name="B", q75_phase=rb.get("q75_phase", 0.0))
    b = variant_process(v, pieces[rb.get("noise", "calibrated")], hb)
    return A.mixture([a, b], [1 - d["weight"], d["weight"]])


def task_predict(task: dict[str, Any]) -> dict[str, Any]:
    """One (variant, noise, hypothesis) prediction, with its per-shot SD and paired NLL."""
    cfg = task["cfg"]
    s = settings(cfg)
    pieces, z = load_pieces(cfg)
    ref = reference(cfg)
    v, noise, hname = task["variant"], task["noise"], task["hypothesis"]
    if task.get("drift"):
        d = next(x for x in cfg["drift"] if x["name"] == hname)
        proc = drift_process(cfg, d, v, pieces)
    else:
        hyp = next(h for h in hypotheses(cfg) if h.name == hname)
        proc = variant_process(v, pieces[noise], hyp, z, serial=task.get("serial", False))
    row: dict[str, Any] = {
        "variant": v + ("_serial" if task.get("serial") else ""),
        "noise": noise,
        "hypothesis": hname,
    }
    row.update(A.predict(proc, s, ref))
    if task.get("sd"):
        sh = cfg["shots"]
        sd = A.per_shot_sd(proc, s, sh["sim_shots"], sh["bootstrap"], sh["seed"])
        row.update({f"sd_w{j}": sd[j] for j in s.offsets})
    if task.get("nll"):
        sh = cfg["shots"]
        m, se = A.paired_nll(proc, ref, sh["nll_shots"], s.length, sh["nll_seed"])
        row["nll_ref_minus_own"], row["nll_ref_minus_own_se"] = m, se
    return row


def checks(cfg, pieces, z) -> dict[str, Any]:
    """Baseline vs Phase 6, and the A1/A2 analytic zeros."""
    s = settings(cfg)
    s6 = phase6_summary(cfg)
    ref = reference(cfg)
    out: dict[str, Any] = {}
    for noise in ("noiseless", "calibrated"):
        base = variant_process("baseline", pieces[noise], A.Hypothesis())
        pred = s6["analysis"]["predictions"][noise]
        g6 = base.windows_at(base.state_after(s.burn), s.k_max)
        out[noise] = {
            "max_abs_g_diff_vs_phase6": float(np.abs(g6 - np.array(pred["g"])).max()),
            "max_abs_marginal_diff_vs_phase6": float(
                np.abs(base.marginals(s.length) - np.array(pred["marginals"])).max()
            ),
        }
        for v in ("A1", "A2"):
            p = variant_process(v, pieces[noise], A.Hypothesis())
            g = p.expected_windows(s.k_max, s.burn, s.length, common=True)
            out[noise][f"{v}_max_abs_w"] = float(max(abs(W.witness(g, j)) for j in s.offsets))
    base = variant_process("baseline", pieces["calibrated"], A.Hypothesis())
    out["calibrated"]["max_abs_g_diff_vs_audit_instrument"] = float(
        np.abs(
            base.expected_windows(s.k_max, s.burn, s.length, True)
            - ref.expected_windows(s.k_max, s.burn, s.length, True)
        ).max()
    )
    return out


def a3_study(cfg, pieces, out: Path) -> list[dict[str, Any]]:
    """Finite-M bias and spread of A3's witness, from a pool of exact instance g's."""
    s = settings(cfg)
    a = cfg["a3"]
    rows = []
    for noise in ("noiseless", "calibrated"):
        bank = [A.process(A.step_superops("A3", pieces[noise], None, p)) for p in A.PAULIS]
        rng = np.random.default_rng([a["seed"], 10**6])
        pool = rng.integers(0, 4, size=(a["pool"], s.length))
        g_pool = A.instance_windows(bank, pool, s)
        g_cfg = A.instance_windows(bank, pauli_table(cfg), s).mean(0)
        log(out, f"A3 pool ({noise}) done")
        for m in a["m_grid"]:
            for j in s.offsets:
                mu, sd = A.finite_m_witness(g_pool, m, a["draws"], a["seed"] + m, j)
                rows.append({"noise": noise, "M": m, "j": j, "mean_w": mu, "sd_w": sd})
        for j in s.offsets:
            rows.append(
                {
                    "noise": noise,
                    "M": f"config ({a['instances']})",
                    "j": j,
                    "mean_w": float(W.witness(g_cfg, j)),
                    "sd_w": 0.0,
                }
            )
    return rows


DESTRUCTIVE = ("A1", "A2", "A3")
SHAMS = ("A1s", "A2s", "A2s_spec", "A3s")


def shots_table(
    cfg, preds: list[dict[str, Any]], circuits: dict[str, Any], a3_realised: float
) -> list[dict]:
    """Shots and QPU seconds for each discrimination, at the config's z and SE inflation.

    Per-shot SDs of w(0) come from simulated shots of each control's calibrated prediction,
    times #18's contiguous-block inflation. Under H_q75 a destructive control's device value
    is its noisy prediction (for A3 the exact value of the realised instance set) and a
    sham's is the device baseline, the episode-free w(0) of #18.
    """
    sh, b = cfg["shots"], cfg["budget"]
    infl, z = sh["se_inflation"], sh["z"]
    dev = sh["device_w0"]
    red = sh["sham_reduction"]
    cal = {(r["variant"], r["hypothesis"]): r for r in preds if r["noise"] == "calibrated"}
    sd_b = cal[("baseline", "H_q75")]["sd_w0"] * infl
    extra = {r["circuit"]: r["duration_minus_baseline_s"] for r in circuits["circuits"]}
    rows = []
    for v in cfg["circuits"]["variants"]:
        dur = b["phase6_circuit_duration_s"] + extra.get("A3_000" if v == "A3" else v, 0.0)
        per_shot = b["rep_delay_s"] + dur
        h0 = cal[(v, "H_q75")]
        sd_c = h0["sd_w0"] * infl
        role = "destructive" if v in DESTRUCTIVE else ("sham" if v in SHAMS else "baseline")
        w_h = (a3_realised if v == "A3" else h0["w0"]) if role == "destructive" else dev
        row: dict[str, Any] = {
            "variant": v,
            "role": role,
            "duration_us": dur * 1e6,
            "sd_w0_per_shot": sd_c,
            "w0_pred_H_q75": h0["w0"] if v != "A3" else a3_realised,
            "w0_device_expected_H_q75": w_h,
        }
        two_arm = math.sqrt(sd_c**2 + sd_b**2)
        if role == "baseline":
            n = A.shots_5sigma(dev, sd_b, z)
            row["shots_w0_from_0"] = n
            row["qpu_s"] = A.qpu_seconds(n, dur, b["rep_delay_s"], 0.0)
        elif role == "destructive":
            n = A.shots_5sigma(dev - w_h, two_arm, z)
            row["shots_vs_baseline"] = n
            row["qpu_s_vs_baseline"] = A.qpu_seconds(n, dur, b["rep_delay_s"], 0.0)
            for f in sh["persist_fractions"]:
                n = A.shots_5sigma(f * dev - w_h, sd_c, z)
                row[f"shots_persist_{f:g}"] = n
                row[f"qpu_s_persist_{f:g}"] = A.qpu_seconds(n, dur, b["rep_delay_s"], 0.0)
            row["z2_per_qpu_s"] = ((0.5 * dev - w_h) / sd_c) ** 2 / per_shot
            row["feasible"] = row["shots_vs_baseline"] <= sh["feasible_shots"]
        else:
            n = A.shots_5sigma(red * dev, two_arm, z)
            row[f"shots_reduction_{red:g}"] = n
            row[f"qpu_s_reduction_{red:g}"] = A.qpu_seconds(n, dur, b["rep_delay_s"], 0.0)
            row["z2_per_qpu_s"] = (red * dev / two_arm) ** 2 / per_shot
            row["feasible"] = n <= sh["feasible_shots"]
        for r in preds:  # concrete H_other alternatives against this control's H_q75 value
            if r["noise"] != "calibrated" or r["variant"] != v or r["hypothesis"] == "H_q75":
                continue
            h = r["hypothesis"]
            row[f"w0__{h}"] = r["w0"]
            row[f"shots_vs_H_q75__{h}"] = A.shots_5sigma(r["w0"] - h0["w0"], sd_c, z)
        rows.append(row)
    return rows


def scans(cfg, pieces) -> list[dict[str, Any]]:
    """How large must each alternative be to matter? w(0) along one parameter at a time."""
    s = settings(cfg)
    p = pieces["calibrated"]
    sc = cfg["scans"]
    rows = []
    for e1 in sc["q59_e1"]:
        h = A.Hypothesis(q59_reset=(sc["q59_e0"], e1))
        for v in ("baseline", "A2"):
            g = A.process(A.step_superops(v, p, h)).expected_windows(
                s.k_max, s.burn, s.length, True
            )
            rows.append(
                {"scan": "q59_residual_e1", "value": e1, "variant": v, "w0": W.witness(g, 0)}
            )
    for c in sc["a2_coherence"]:
        g = A.process(A.step_superops("A2", p, A.Hypothesis(op_coherence=c))).expected_windows(
            s.k_max, s.burn, s.length, True
        )
        rows.append(
            {"scan": "a2_kept_coherence", "value": c, "variant": "A2", "w0": W.witness(g, 0)}
        )
    for d in sc["spec_dephasing"]:
        proc = A.process(A.step_superops("A2s_spec", p, A.Hypothesis(spec_dephasing=d)))
        g = proc.expected_windows(s.k_max, s.burn, s.length, True)
        rows.append(
            {
                "scan": "spectator_dephasing",
                "value": d,
                "variant": "A2s_spec",
                "w0": W.witness(g, 0),
            }
        )
    return rows


def a2_record(cfg, pieces) -> dict[str, Any]:
    """Predicted statistics of A2's q75 record m_t (diagnostics only, never in the witness).

    In each step q75 is measured (m_t) before q59 (y_t). Under the model the hidden state is
    the classical bit q75 holds, so given the true m_t, y_t does not depend on y_{t-1}. The
    recorded m_t has q75's readout error, which leaves a small dependence.
    """
    s = settings(cfg)
    p = pieces["calibrated"]
    r = float(cfg["a2_record"]["readout_error"])
    proj = [np.diag([1.0, 0]).astype(complex), np.diag([0, 1.0]).astype(complex)]
    rs = A.reset_superop(p, A.Hypothesis())
    ops = {}
    for m in (0, 1):
        op = A.superop(A.kraus_kron([proj[m]], p.relax_q59_measure))
        for y in (0, 1):
            ops[m, y] = p.post[y] @ op @ p.pre @ rs
    rec = {}
    for mr in (0, 1):  # recorded m with a symmetric readout flip r
        for y in (0, 1):
            rec[mr, y] = (1 - r) * ops[mr, y] + r * ops[1 - mr, y]
    vec = A.RHO0_JOINT.reshape(-1, order="F")
    trace = np.eye(4).reshape(-1, order="F")
    st = vec
    chan = sum(ops.values())
    for _ in range(s.burn):
        st = chan @ st
    out: dict[str, Any] = {"readout_error": r}
    joint = {}
    for y0 in (0, 1):
        a = sum(rec[m, y0] for m in (0, 1)) @ st
        for m in (0, 1):
            for y in (0, 1):
                joint[y0, m, y] = float((trace @ (rec[m, y] @ a)).real)
    pm1 = sum(v for (y0, m, y), v in joint.items() if m == 1)
    out["p_m1"] = pm1
    for m in (0, 1):
        for y0 in (0, 1):
            num = joint[y0, m, 1]
            den = joint[y0, m, 0] + joint[y0, m, 1]
            out[f"p_y1_given_m{m}_yprev{y0}"] = num / den
    # conditional mutual information I(y_t ; y_{t-1} | m_t), recorded m
    cmi = 0.0
    tot = sum(joint.values())
    for (y0, m, y), v in joint.items():
        pm = sum(joint[a, m, b] for a in (0, 1) for b in (0, 1)) / tot
        py0m = sum(joint[y0, m, b] for b in (0, 1)) / tot
        pym = sum(joint[a, m, y] for a in (0, 1)) / tot
        pj = v / tot
        cmi += pj * math.log(pj * pm / (py0m * pym))
    out["cmi_y_yprev_given_m_nats"] = cmi
    return out


def run_pool(tasks, workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [task_predict(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(task_predict, tasks))


def analyse(cfg: dict[str, Any], from_run: Path | None) -> None:
    out = from_run or new_run_dir(cfg["experiment"]["name"])
    save_json(out / "config.json", {"meta": run_metadata(plain(cfg), cfg["shots"]["seed"])})
    pieces, z = load_pieces(cfg)
    log(out, f"run dir {out}")
    chk = checks(cfg, pieces, z)
    log(out, f"checks: {json.dumps(chk)}")

    hyps = [h["name"] for h in cfg["hypotheses"]]
    variants = cfg["circuits"]["variants"]
    tasks = []
    for v in variants:
        tasks.append(dict(cfg=cfg, variant=v, noise="noiseless", hypothesis="H_q75"))
        tasks.append(
            dict(cfg=cfg, variant=v, noise="calibrated", hypothesis="H_q75", sd=True, nll=True)
        )
        for h in hyps[1:]:
            tasks.append(dict(cfg=cfg, variant=v, noise="calibrated", hypothesis=h, sd=True))
        for d in cfg["drift"]:
            tasks.append(
                dict(cfg=cfg, variant=v, noise="calibrated", hypothesis=d["name"], drift=True)
            )
    for v in ("A1", "A2"):
        tasks.append(dict(cfg=cfg, variant=v, noise="calibrated", hypothesis="H_q75", serial=True))
    cache = out / "predictions.json"
    if cache.exists():
        preds = json.loads(cache.read_text())
        log(out, f"resumed {len(preds)} predictions")
    else:
        preds = run_pool(tasks, cfg["experiment"]["workers"])
        cache.write_text(json.dumps(preds))
    log(out, f"{len(preds)} predictions")

    a3_cache = out / "a3.json"
    if a3_cache.exists():
        a3 = json.loads(a3_cache.read_text())
    else:
        a3 = a3_study(cfg, pieces, out)
        a3_cache.write_text(json.dumps(a3))

    circuits = json.loads((sdir(cfg) / "circuits.json").read_text())
    realised = next(
        r["mean_w"]
        for r in a3
        if r["noise"] == "calibrated" and r["j"] == 0 and "config" in str(r["M"])
    )
    table = shots_table(cfg, preds, circuits, realised)
    scan_rows = scans(cfg, pieces)
    record = a2_record(cfg, pieces)
    log(out, f"A2 record diagnostics: {json.dumps(record)}")
    flat = []
    for r in preds:
        f = {k: v for k, v in r.items() if k not in ("g", "blocks")}
        f.update({f"g{k}": x for k, x in enumerate(r["g"])})
        flat.append(f)
    blocks = [
        {"variant": r["variant"], "noise": r["noise"], "hypothesis": r["hypothesis"]}
        | {format(i, "04b"): x for i, x in enumerate(r["blocks"])}
        for r in preds
    ]
    sd = sdir(cfg)
    for d in (out, sd):
        write_csv(d / "predictions.csv", flat)
        write_csv(d / "blocks_L4.csv", blocks)
        write_csv(d / "a3_finite_m.csv", a3)
        write_csv(d / "shots.csv", table)
        write_csv(d / "scans.csv", scan_rows)
    summary = {
        "meta": run_metadata(plain(cfg), cfg["shots"]["seed"]),
        "run_dir": str(out.relative_to(REPO_ROOT)),
        "checks": chk,
        "a2_record": record,
        "a3_realised_w0": realised,
        "shots": table,
    }
    save_json(out / "summary.json", summary)
    save_json(sd / "summary.json", summary)
    shutil.copy(out / "log.txt", sd / "log.txt")
    log(out, "done")


# -- 3. the Stage 3 plan (prints only) ---------------------------------------------------------


def plan(cfg: dict[str, Any]) -> None:
    hw, b = cfg["hardware"], cfg["budget"]
    circuits = json.loads((sdir(cfg) / "circuits.json").read_text())
    extra = {r["circuit"]: r["duration_minus_baseline_s"] for r in circuits["circuits"]}
    total = b["sub_job_overhead_s"]
    print(f"backend {hw['backend']} (submit = {hw['submit']}: this script never submits)")
    items = list(hw["shots_per_circuit"].items())
    if hw.get("a3_total_shots"):
        items.append(("A3", hw["a3_total_shots"]))
    for name, shots in items:
        key = "A3_000" if name == "A3" else name
        dur = b["phase6_circuit_duration_s"] + extra.get(key, 0.0)
        s = shots * (b["rep_delay_s"] + dur)
        total += s
        print(f"  {name:10s} {shots:6d} shots  {dur * 1e6:7.1f} us  {s:6.2f} s")
    print(f"one job, total {total:.2f} QPU s (cost model: 2 s + shots x (250 us + duration))")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export", action="store_true", help="build circuits and pieces (hardware)")
    ap.add_argument("--plan", action="store_true", help="print the Stage 3 job and its cost")
    ap.add_argument("--from-run", type=Path, default=None, help="resume a run directory")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["experiment"]["workers"] > 2:
        sys.exit("at most 2 workers on the shared machine (#20)")
    if args.export:
        export(cfg)
    elif args.plan:
        plan(cfg)
    else:
        analyse(cfg, args.from_run)


if __name__ == "__main__":
    main()
