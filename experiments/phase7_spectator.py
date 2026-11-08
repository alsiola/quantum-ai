"""Phase 7 Stage 2C (#22): spectators, couplers and crosstalk around q75/q59, and the design of
replication and spectator controls (C1-C4). Local simulation only: no QPU, no IBM service.

Two steps:

* ``--export-device`` (``hardware`` group, once): FakeKingston's coupling map, stale
  calibration and timing into ``results/summaries/phase7_spectator/device.json``.
* the analysis (numpy): topology and idle table, sensitivity of w(0..3), P(1), g to static
  ZZ, thermal spectators, TLS, readout Stark shift and dephasing; whether a spectator alone
  makes a witness; one- and two-parameter fits to the #18 episode; predictions, shots and QPU
  seconds for C1-C4; the QASM of every control arm (unsubmitted).

Everything uses ``qmem.spectator`` (time-resolved branch simulator) and ``qmem.witness``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tomllib
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qmem import spectator as S  # noqa: E402
from qmem import witness as W  # noqa: E402
from qmem.records import new_run_dir, run_metadata, save_json  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def log(out: Path, msg: str) -> None:
    print(msg, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(msg + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def sdir(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["summary_dir"]


def p6(cfg) -> Path:
    return REPO_ROOT / cfg["experiment"]["phase6_dir"]


# -- 0. FakeKingston export (hardware group) ---------------------------------------------------


def export_device(cfg) -> None:
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    b = FakeKingston()
    t = b.target
    edges = sorted({tuple(sorted(e)) for e in b.coupling_map.get_edges()})
    qubits = {}
    for q in range(b.num_qubits):
        qp = t.qubit_properties[q]
        qubits[q] = {
            "t1": qp.t1,
            "t2": qp.t2,
            "readout": t["measure"][(q,)].error,
            "sx": t["sx"][(q,)].error,
            "measure_time": t["measure"][(q,)].duration,
            "reset_time": t["reset"][(q,)].duration,
        }
    cz = {}
    for a, c in edges:
        p = t["cz"].get((a, c)) or t["cz"].get((c, a))
        cz[f"{a}-{c}"] = {"error": p.error, "duration": p.duration}
    out = {
        "meta": run_metadata({k: v for k, v in cfg.items() if not k.startswith("_")}, None),
        "backend": b.name,
        "calibration_date": str(b.properties().last_update_date),
        "num_qubits": b.num_qubits,
        "dt": t.dt,
        "granularity": t.granularity,
        "operations": sorted(t.operation_names),
        "edges": [list(e) for e in edges],
        "cz": cz,
        "qubits": qubits,
    }
    sd = sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    save_json(sd / "device.json", out)
    print(f"wrote {sd / 'device.json'}: {len(edges)} edges, {b.num_qubits} qubits")


def load_device(cfg) -> dict[str, Any]:
    path = sdir(cfg) / "device.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run once with --export-device (hardware group)")
    d = json.loads(path.read_text())
    d["qubits"] = {int(k): v for k, v in d["qubits"].items()}
    return d


def neighbours(dev) -> dict[int, list[int]]:
    nb: dict[int, set[int]] = {q: set() for q in dev["qubits"]}
    for a, c in dev["edges"]:
        nb[a].add(c)
        nb[c].add(a)
    return {q: sorted(v) for q, v in nb.items()}


def distances(nb, sources) -> dict[int, int]:
    dist = {s: 0 for s in sources}
    dq = deque(sources)
    while dq:
        u = dq.popleft()
        for v in nb[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


# -- models -----------------------------------------------------------------------------------


class Ctx:
    """Everything shared: the snapshot model, the step program, the device and settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.s6 = json.loads((p6(cfg) / "summary.json").read_text())
        self.cal = S.calibrated_model(self.s6["prepare"]["calibration"]["noise"])
        self.prog = S.parse_qasm((p6(cfg) / "step.qasm").read_text())
        self.dev = load_device(cfg)
        self.nb = neighbours(self.dev)
        lay = cfg["layout"]
        self.mem, self.out = lay["memory"], lay["output"]
        self.spec_m, self.spec_o = lay["spectator_memory"], lay["spectator_output"]
        w = cfg["witness"]
        self.kw = {"k_max": w["k_max"], "burn": w["burn"], "length": w["length"]}
        self.offsets = w["offsets"]
        self._base: S.Prediction | None = None

    # physical -> StepModel ----------------------------------------------------------------
    def snapshot_qubit(self, phys: int) -> S.CoreQubit:
        """The #16 snapshot for q59/q75 (compact 0 = q59, 1 = q75)."""
        return self.cal.output if phys == self.out else self.cal.memory

    def fake_qubit(self, phys: int) -> S.CoreQubit:
        q = self.dev["qubits"][phys]
        return S.CoreQubit(q["t1"], q["t2"], 2 * q["sx"], q["readout"], q["readout"])

    def spectator_t1(self, phys: int) -> float:
        return self.dev["qubits"][phys]["t1"]

    def model(
        self,
        out: int,
        mem: int,
        scenario: dict[str, Any],
        source: str = "snapshot",
    ) -> S.StepModel:
        """``StepModel`` with output ``out`` and memory ``mem`` (physical), under ``scenario``.

        ``scenario``: ``zz`` {(a, b): Hz} on physical qubits; ``spectators`` {phys: dict of
        ``Spectator`` fields}; ``detuning`` {phys: Hz}; ``stark`` (y=0, y=1) Hz on the memory
        during the output's readout; ``meas_dephasing_rate``; ``dd`` (bool).
        """
        if source == "snapshot":
            oq, mq = self.snapshot_qubit(out), self.snapshot_qubit(mem)
            p2, mt, rt = self.cal.p2, self.cal.measure_time, self.cal.reset_time
        else:
            oq, mq = self.fake_qubit(out), self.fake_qubit(mem)
            e = self.dev["cz"][f"{min(out, mem)}-{max(out, mem)}"]["error"]
            p2 = e * 4 / 3
            mt = self.dev["qubits"][out]["measure_time"]
            rt = self.dev["qubits"][out]["reset_time"]
        idx = {out: 0, mem: 1}
        specs = []
        for phys, kw in scenario.get("spectators", {}).items():
            if phys in idx:
                continue
            idx[phys] = 2 + len(specs)
            specs.append(S.Spectator(**kw))
        zz = {}
        for (a, b), z in scenario.get("zz", {}).items():
            if a in idx and b in idx:
                zz[(idx[a], idx[b])] = z
        det = {idx[q]: v for q, v in scenario.get("detuning", {}).items() if q in idx}
        return S.StepModel(
            output=oq,
            memory=mq,
            p2=p2,
            measure_time=mt,
            reset_time=rt,
            spectators=specs,
            zz_hz=zz,
            detuning_hz=det,
            stark_hz=tuple(scenario.get("stark", (0.0, 0.0))),
            meas_dephasing_rate=scenario.get("meas_dephasing_rate", 0.0),
            reset_memory=scenario.get("reset_memory", False),
        )

    def program(self, model: S.StepModel, dd: bool = False) -> list[S.Instr]:
        if not dd:
            return self.prog
        return S.add_step_dd(
            self.prog,
            1,
            model.measure_time,
            model.reset_time,
            grid=self.cfg["controls"]["dd_grid_s"],
        )

    def predict(self, model: S.StepModel, dd: bool = False) -> S.Prediction:
        return S.predict(S.process(self.program(model, dd), model), offsets=self.offsets, **self.kw)

    def proc(self, model: S.StepModel, dd: bool = False) -> W.LinearProcess:
        return S.process(self.program(model, dd), model)

    @property
    def base(self) -> S.Prediction:
        if self._base is None:
            self._base = self.predict(self.model(self.out, self.mem, {}))
        return self._base

    def row(self, pred: S.Prediction, **extra) -> dict[str, Any]:
        b = self.base
        r = dict(extra) | pred.row()
        for j in self.offsets:
            r[f"w{j}_ratio"] = pred.w[j] / b.w[j]
        r["dp1"] = pred.p1 - b.p1
        r["dg1"] = pred.g[1] - b.g[1]
        r["dg3"] = pred.g[3] - b.g[3]
        r["changes_w0_10pct"] = bool(abs(r["w0_ratio"] - 1) >= 0.1)
        return r


# -- 1. topology and idle table ------------------------------------------------------------------


def topology(ctx: Ctx) -> tuple[list[dict], list[dict]]:
    dev, nb = ctx.dev, ctx.nb
    rows = []
    core = (ctx.out, ctx.mem)
    for q in sorted({ctx.out, ctx.mem, *nb[ctx.out], *nb[ctx.mem]}):
        role = (
            "output"
            if q == ctx.out
            else "memory"
            if q == ctx.mem
            else f"spectator of q{ctx.mem}"
            if q in nb[ctx.mem]
            else f"spectator of q{ctx.out}"
        )
        fq = dev["qubits"][q]
        cpl = [
            f"{min(q, v)}-{max(q, v)} (cz err {dev['cz'][f'{min(q, v)}-{max(q, v)}']['error']:.2e})"
            for v in nb[q]
        ]
        rows.append(
            {
                "qubit": q,
                "role": role,
                "neighbours": " ".join(map(str, nb[q])),
                "couplers": "; ".join(cpl),
                "fake_t1_us": fq["t1"] * 1e6,
                "fake_t2_us": fq["t2"] * 1e6,
                "fake_readout": fq["readout"],
                "snapshot_t1_us": ctx.snapshot_qubit(q).t1 * 1e6 if q in core else None,
                "snapshot_t2_us": ctx.snapshot_qubit(q).t2 * 1e6 if q in core else None,
                "snapshot_readout": ctx.snapshot_qubit(q).readout if q in core else None,
            }
        )
    # idle per step (snapshot durations), from the schedule of the step program
    m = ctx.cal
    events, total = S.schedule(ctx.prog, m)
    busy = {0: 0.0, 1: 0.0}
    for e in events:
        if e.ins.name == "rz":
            continue
        for q in e.ins.qubits:
            busy[q] += e.end - e.t
    idle = []
    names = {0: ctx.out, 1: ctx.mem}
    for k, q in names.items():
        other_window = (
            (m.measure_time + m.reset_time) if k == 1 else 0.0
        )  # the memory idles through the output's readout and reset
        idle.append(
            {
                "qubit": q,
                "role": "output" if k == 0 else "memory",
                "step_us": total * 1e6,
                "busy_us": busy[k] * 1e6,
                "idle_us": (total - busy[k]) * 1e6,
                "idle_during_q59_readout_reset_us": other_window * 1e6,
                "idle_in_gate_region_us": (total - busy[k] - other_window) * 1e6
                if k == 1
                else (total - busy[k]) * 1e6,
            }
        )
    for q in (ctx.spec_m, ctx.spec_o):
        idle.append(
            {
                "qubit": q,
                "role": "spectator",
                "step_us": total * 1e6,
                "busy_us": 0.0,
                "idle_us": total * 1e6,
                "idle_during_q59_readout_reset_us": (m.measure_time + m.reset_time) * 1e6,
                "idle_in_gate_region_us": (total - m.measure_time - m.reset_time) * 1e6,
            }
        )
    return rows, idle


# -- 2. sensitivity ------------------------------------------------------------------------------


def sweeps(ctx: Ctx, out: Path) -> dict[str, list[dict]]:
    cfg, sw = ctx.cfg, ctx.cfg["sweep"]
    o, mm, sm, so = ctx.out, ctx.mem, ctx.spec_m, ctx.spec_o
    res: dict[str, list[dict]] = {}
    rows = []
    for z in sw["zz_hz"]:
        for p in sw["populations"]:
            for kind, t1 in (("thermal", ctx.spectator_t1(sm)), ("frozen", None)):
                sc = {"zz": {(mm, sm): z}, "spectators": {sm: {"t1": t1, "p_eq": p}}}
                rows.append(ctx.row(ctx.predict(ctx.model(o, mm, sc)), zz_hz=z, p=p, kind=kind))
    res["zz_memory_spectator"] = rows
    log(out, f"[sweep] q{mm}-q{sm} ZZ x population: {len(rows)} points")
    rows = []
    for z in sw["zz_hz"]:
        for p in sw["populations"]:
            sc = {"zz": {(o, so): z}, "spectators": {so: {"t1": ctx.spectator_t1(so), "p_eq": p}}}
            rows.append(ctx.row(ctx.predict(ctx.model(o, mm, sc)), zz_hz=z, p=p))
    res["zz_output_spectator"] = rows
    rows = [
        ctx.row(ctx.predict(ctx.model(o, mm, {"zz": {(o, mm): z}})), zz_hz=z)
        for z in sw["pair_zz_hz"]
    ]
    res["zz_pair"] = rows
    rows = []
    for s in sw["stark_hz"]:
        for kind, st in (("uniform", (s, s)), ("if_output_1", (0.0, s))):
            rows.append(
                ctx.row(ctx.predict(ctx.model(o, mm, {"stark": st})), stark_hz=s, kind=kind)
            )
    for g in sw["meas_dephasing_rate"]:
        rows.append(
            ctx.row(
                ctx.predict(ctx.model(o, mm, {"meas_dephasing_rate": g})),
                kind="dephasing",
                rate=g,
            )
        )
    res["readout_crosstalk"] = rows
    rows = []
    for t1 in sw["tls_loss_t1"]:
        for p in sw["populations"]:
            sc = {"spectators": {"tls": {"t1": None, "p_eq": p, "loss_rate": 1 / t1}}}
            rows.append(ctx.row(ctx.predict(ctx.model(o, mm, sc)), tls_t1_us=t1 * 1e6, p=p))
    for shift in cfg["literature"]["tls_shift_hz"]:
        for p in sw["populations"]:
            sc = {"zz": {(mm, "tls"): shift}, "spectators": {"tls": {"t1": None, "p_eq": p}}}
            rows.append(ctx.row(ctx.predict(ctx.model(o, mm, sc)), tls_shift_hz=shift, p=p))
    res["tls"] = rows
    log(out, "[sweep] output spectator, pair ZZ, readout crosstalk, TLS done")
    return res


def own_witness(ctx: Ctx) -> list[dict]:
    """q75 reset every step: can spectators alone produce w != 0?"""
    o, mm, sm, so = ctx.out, ctx.mem, ctx.spec_m, ctx.spec_o
    rows = []
    for z in ctx.cfg["sweep"]["own_zz_hz"]:
        for label, specs in (
            ("one thermal (memory side)", {sm: {"t1": ctx.spectator_t1(sm), "p_eq": 0.5}}),
            (
                "two thermal",
                {
                    sm: {"t1": ctx.spectator_t1(sm), "p_eq": 0.5},
                    so: {"t1": ctx.spectator_t1(so), "p_eq": 0.5},
                },
            ),
            ("two frozen", {sm: {"t1": None, "p_eq": 0.5}, so: {"t1": None, "p_eq": 0.5}}),
        ):
            sc = {
                "zz": {(mm, sm): z, (o, so): z, (o, mm): z},
                "spectators": specs,
                "reset_memory": True,
            }
            pr = ctx.predict(ctx.model(o, mm, sc))
            rows.append(
                {"zz_hz": z, "spectators": label, "p1": pr.p1}
                | {f"w{j}": pr.w[j] for j in ctx.offsets}
                | {"abs_w0_over_device_rest": abs(pr.w[0]) / 0.00587}
            )
    return rows


# -- 3. the episode ------------------------------------------------------------------------------


def device_segments(ctx: Ctx) -> dict[str, dict[str, Any]]:
    w = ctx.cfg["witness"]
    with np.load(p6(ctx.cfg) / "shots_main.npz") as z:
        arr = W.unpack_shots(z["packed"], int(z["length"]))
    per = W.per_shot_windows(arr, w["k_max"], w["burn"], common=True)
    a, b = w["episode"]
    inside = np.zeros(len(arr), bool)
    inside[a:b] = True
    out = {}
    for name, mask in (("all", np.ones(len(arr), bool)), ("episode", inside), ("rest", ~inside)):
        pm = per[mask]
        idx = W.bootstrap_indices(len(pm), w["bootstrap"], w["seed"])
        r = W.witness_stats(pm, ctx.offsets, idx)
        bg = W.bootstrap_g(pm, idx)
        p1_shot = arr[mask].mean(1)
        out[name] = {
            "shots": int(mask.sum()),
            "g": r.g,
            "g_se": bg.std(0, ddof=1),
            "p1": float(arr[mask].mean()),
            "p1_se": float(p1_shot.std(ddof=1) / math.sqrt(len(p1_shot))),
            "w": r.w,
            "w_se": r.se,
            "sd_w0_per_shot": r.se[0] * math.sqrt(len(pm)),
        }
    return out


def chi2(pred: S.Prediction, seg: dict[str, Any]) -> float:
    r = (pred.g - seg["g"]) / seg["g_se"]
    return float(np.sum(r**2) + ((pred.p1 - seg["p1"]) / seg["p1_se"]) ** 2)


def fit_episode(ctx: Ctx, segs, out: Path) -> list[dict]:
    """One/two-parameter fits of physical mechanisms to the episode and to the rest."""
    fcfg = ctx.cfg["fit"]
    o, mm = ctx.out, ctx.mem
    lo, hi = fcfg["detuning_hz"]
    dets = np.linspace(lo, hi, fcfg["grid"])
    rows = []

    def det_model(d, loss=0.0):
        sc: dict[str, Any] = {"detuning": {mm: d}}
        if loss:
            sc["spectators"] = {"tls": {"t1": None, "p_eq": 1.0, "loss_rate": loss}}
        return ctx.model(o, mm, sc)

    cache = {d: ctx.predict(det_model(d)) for d in dets}
    losses = np.linspace(*fcfg["loss_rate"], 21)
    loss_cache = {lr: ctx.predict(det_model(0.0, lr)) for lr in losses}
    for seg_name in ("episode", "rest"):
        seg = segs[seg_name]
        c0 = chi2(ctx.base, seg)
        rows.append(fit_row("calibrated (no extra parameter)", 0, ctx.base, seg, c0, {}))
        # (a) static detuning of q75: a TLS frequency shift, or a frozen spectator flip
        cs = {d: chi2(p, seg) for d, p in cache.items()}
        d1 = min(cs, key=cs.get)
        # refine
        fine = np.linspace(d1 - (hi - lo) / fcfg["grid"], d1 + (hi - lo) / fcfg["grid"], 41)
        fc = {d: chi2(ctx.predict(det_model(d)), seg) for d in fine}
        d1 = min(fc, key=fc.get)
        p = ctx.predict(det_model(d1))
        rows.append(fit_row("q75 static detuning", 1, p, seg, chi2(p, seg), {"detuning_hz": d1}))
        # (b) TLS loss: extra amplitude damping of q75
        cl = {lr: chi2(pp, seg) for lr, pp in loss_cache.items()}
        l1 = min(cl, key=cl.get)
        p = loss_cache[l1]
        rows.append(fit_row("q75 extra T1 loss (TLS)", 1, p, seg, cl[l1], {"loss_rate": l1}))
        # (c) detuning + loss
        best = (math.inf, 0.0, 0.0, None)
        for lr in losses[::2]:
            for d in np.linspace(d1 - 30e3, d1 + 30e3, 31):
                pp = ctx.predict(det_model(d, lr))
                c = chi2(pp, seg)
                if c < best[0]:
                    best = (c, d, lr, pp)
        rows.append(
            fit_row(
                "q75 detuning + loss",
                2,
                best[3],
                seg,
                best[0],
                {"detuning_hz": best[1], "loss_rate": best[2]},
            )
        )
        # (d) a thermal spectator at the upper ZZ, population free (the episode as p rising)
        sm = ctx.spec_m
        z = ctx.cfg["literature"]["zz_upper_hz"]
        bestp = (math.inf, 0.0, None)
        for sign in (1, -1):
            for pop in np.linspace(0, 1, 21):
                sc = {
                    "zz": {(mm, sm): sign * z},
                    "spectators": {sm: {"t1": ctx.spectator_t1(sm), "p_eq": float(pop)}},
                }
                pp = ctx.predict(ctx.model(o, mm, sc))
                c = chi2(pp, seg)
                if c < bestp[0]:
                    bestp = (c, sign * float(pop), pp)
        rows.append(
            fit_row(
                f"q74 thermal spectator, ZZ = ±{z / 1e3:.0f} kHz, population free",
                1,
                bestp[2],
                seg,
                bestp[0],
                {"signed_population": bestp[1]},
            )
        )
        log(out, f"[episode] {seg_name}: detuning fit {d1 / 1e3:.1f} kHz")
    return rows


def fit_row(name, k, pred, seg, c, params) -> dict[str, Any]:
    return {
        "mechanism": name,
        "params": k,
        "chi2": c,
        "dof": 9 - k,
        **{f"param_{a}": b for a, b in params.items()},
        "pred_p1": pred.p1,
        "pred_g1": pred.g[1],
        "pred_g3": pred.g[3],
        "pred_w0": pred.w[0],
        "pred_w2": pred.w[2],
        "obs_p1": seg["p1"],
        "obs_g1": float(seg["g"][1]),
        "obs_g3": float(seg["g"][3]),
        "obs_w0": seg["w"][0],
        "obs_w2": seg["w"][2],
        "segment_shots": seg["shots"],
    }


# -- 4. controls ---------------------------------------------------------------------------------


def scenarios(ctx: Ctx) -> dict[str, dict[str, Any]]:
    """Physical alternatives to the null, at the published strengths (Kingston pair)."""
    lit = ctx.cfg["literature"]
    o, mm, sm, so = ctx.out, ctx.mem, ctx.spec_m, ctx.spec_o
    p = lit["thermal_population"]
    t1m, t1o = ctx.spectator_t1(sm), ctx.spectator_t1(so)
    zlo, zhi = lit["zz_coupler_off_hz"], lit["zz_upper_hz"]
    return {
        "null": {},
        f"zz_q{mm}_q{sm}_{zlo / 1e3:.0f}k": {
            "zz": {(mm, sm): zlo},
            "spectators": {sm: {"t1": t1m, "p_eq": p}},
        },
        f"zz_q{mm}_q{sm}_{zhi / 1e3:.0f}k": {
            "zz": {(mm, sm): zhi},
            "spectators": {sm: {"t1": t1m, "p_eq": p}},
        },
        f"zz_pair_{zlo / 1e3:.0f}k": {"zz": {(o, mm): zlo}},
        f"stark_{lit['stark_meas_hz'] / 1e3:.0f}k": {
            "stark": (lit["stark_meas_hz"], lit["stark_meas_hz"])
        },
        "all_plausible": {
            "zz": {(mm, sm): zlo, (o, so): zlo, (o, mm): zlo},
            "spectators": {sm: {"t1": t1m, "p_eq": p}, so: {"t1": t1o, "p_eq": p}},
            "stark": (lit["stark_meas_hz"], lit["stark_meas_hz"]),
        },
        "all_upper": {
            "zz": {(mm, sm): zhi, (o, so): zhi, (o, mm): zhi},
            "spectators": {sm: {"t1": t1m, "p_eq": p}, so: {"t1": t1o, "p_eq": p}},
            "stark": (lit["stark_meas_hz"], lit["stark_meas_hz"]),
        },
    }


def translate(sc: dict[str, Any], mapping: dict[Any, Any]) -> dict[str, Any]:
    """Move a scenario to other physical qubits (role by role)."""
    out = dict(sc)
    if "zz" in sc:
        out["zz"] = {(mapping.get(a, a), mapping.get(b, b)): z for (a, b), z in sc["zz"].items()}
    if "spectators" in sc:
        out["spectators"] = {mapping.get(q, q): v for q, v in sc["spectators"].items()}
    return out


def circuit_duration(model: S.StepModel, prog, length: int) -> float:
    _, step = S.schedule(prog, model)
    return length * step - model.reset_time  # the first step has no reset


def controls(ctx: Ctx, out: Path) -> tuple[list[dict], list[dict], list[dict]]:
    cc, w = ctx.cfg["controls"], ctx.cfg["witness"]
    o, mm, sm = ctx.out, ctx.mem, ctx.spec_m
    infl, zt = w["se_inflation"], w["z_target"]
    n_arm = cc["shots_per_arm"]
    L = w["length"]
    hw = ctx.cfg["hardware"]
    scs = scenarios(ctx)
    sd_cache: dict[str, float] = {}

    def sd_of(key: str, model: S.StepModel, dd: bool = False) -> float:
        if key not in sd_cache:
            sd_cache[key] = S.per_shot_sd(
                ctx.proc(model, dd), 0, cc["sd_samples"], cc["sd_seed"], w["burn"]
            )
            log(out, f"[controls] per-shot SD of w(0), {key}: {sd_cache[key]:.4f}")
        return sd_cache[key]

    def cost(model, prog, extra_s=0.0, arms=2) -> float:
        dur = circuit_duration(model, prog, L) + extra_s
        return S.qpu_seconds(
            arms * n_arm, dur, jobs=1, rep_delay=hw["rep_delay_s"], job_s=hw["job_overhead_s"]
        )

    preds: list[dict] = []
    contrasts: list[dict] = []
    m_orig = ctx.model(o, mm, {})
    sd_orig = sd_of("original", m_orig)
    prog_orig = ctx.prog

    def add_pred(control, arm, scn, pr):
        preds.append({"control": control, "arm": arm, "scenario": scn} | pr.row())

    for scn, sc in scs.items():
        orig = ctx.predict(ctx.model(o, mm, sc))
        add_pred("reference", "original q75 memory / q59 output", scn, orig)
        # C1 role swap: memory q59, output q75 (same physical qubits)
        sw_m = ctx.model(mm, o, sc)
        swap = ctx.predict(sw_m)
        add_pred("C1", "role swap (memory q59, output q75)", scn, swap)
        # C3: spectator q74 prepared |0> (reset) vs |1> (reset + x) at the start of each shot
        ro = ctx.dev["qubits"][sm]["readout"]
        arms3 = {}
        for label, pin in (("q74 |0>", ro), ("q74 |1>", 1 - ro)):
            sc3 = dict(sc)
            specs = dict(sc.get("spectators", {}))
            base_kw = specs.get(sm, {"t1": ctx.spectator_t1(sm), "p_eq": 0.0})
            specs[sm] = dict(base_kw) | {"p_init": pin}
            sc3["spectators"] = specs
            arms3[label] = ctx.predict(ctx.model(o, mm, sc3))
            add_pred("C3", label, scn, arms3[label])
        # C4: XX echo on q75 through q59's readout + reset
        dd = ctx.predict(ctx.model(o, mm, sc), dd=True)
        add_pred("C4", "XX on q75 during q59 readout/reset", scn, dd)
        contrasts.append(
            {
                "scenario": scn,
                "orig_w0": orig.w[0],
                "C1_delta_w0": swap.w[0] - orig.w[0],
                "C3_delta_w0": arms3["q74 |1>"].w[0] - arms3["q74 |0>"].w[0],
                "C4_delta_w0": dd.w[0] - orig.w[0],
            }
        )
    null = contrasts[0]
    sd_swap = sd_of("C1 swap", ctx.model(mm, o, {}))
    sd_dd = sd_of("C4 dd", m_orig, dd=True)
    sc1 = {sm: {"t1": ctx.spectator_t1(sm), "p_eq": 0.0, "p_init": 1.0}}
    sd_c3 = sd_of("C3 |1>", ctx.model(o, mm, {"spectators": sc1}))
    for row in contrasts:
        for c, sds in (
            ("C1", (sd_swap, sd_orig)),
            ("C3", (sd_c3, sd_orig)),
            ("C4", (sd_dd, sd_orig)),
        ):
            d = row[f"{c}_delta_w0"] - null[f"{c}_delta_w0"]
            row[f"{c}_signal_vs_null"] = d
            row[f"{c}_shots_per_arm_5sigma"] = S.shots_for(d, *sds, z=zt, inflate=infl)
            row[f"{c}_z_at_{n_arm}"] = abs(d) / (
                infl * math.sqrt(sds[0] ** 2 + sds[1] ** 2) / math.sqrt(n_arm)
            )
    # costs per control (2 arms of n_arm shots in one job)
    dur_orig = circuit_duration(m_orig, prog_orig, L)
    dd_prog = ctx.program(m_orig, dd=True)
    costs = [
        {
            "control": "C1",
            "arms": 2,
            "circuit_us": dur_orig * 1e6,
            "qpu_s": cost(m_orig, prog_orig),
        },
        {
            "control": "C2 (per pair, + original)",
            "arms": 2,
            "circuit_us": dur_orig * 1e6,
            "qpu_s": cost(m_orig, prog_orig),
        },
        {
            "control": "C3",
            "arms": 2,
            "circuit_us": (dur_orig + 2.4e-6) * 1e6,
            "qpu_s": cost(m_orig, prog_orig, extra_s=2.4e-6),
        },
        {
            "control": "C4",
            "arms": 2,
            "circuit_us": circuit_duration(m_orig, dd_prog, L) * 1e6,
            "qpu_s": cost(m_orig, dd_prog),
        },
    ]
    for row in costs:
        row["shots_per_arm"] = n_arm
    # minimum detectable ZZ (q75-q74) for C3 and C4 at n_arm shots per arm
    zgrid = sorted({abs(z) for z in ctx.cfg["sweep"]["zz_hz"]} | {20e3, 50e3, 5e3})
    detect = []
    for z in zgrid:
        for sign in (1, -1):
            zs = sign * z
            sc_spec = lambda pin, zs=zs: {  # noqa: E731
                "zz": {(mm, sm): zs},
                "spectators": {
                    sm: {
                        "t1": ctx.spectator_t1(sm),
                        "p_eq": ctx.cfg["literature"]["thermal_population"],
                        "p_init": pin,
                    }
                },
            }
            a1 = ctx.predict(ctx.model(o, mm, sc_spec(1.0)))
            a0 = ctx.predict(ctx.model(o, mm, sc_spec(0.0)))
            th = ctx.predict(ctx.model(o, mm, sc_spec(None)))
            d4 = ctx.predict(ctx.model(o, mm, sc_spec(None)), dd=True)
            c3 = a1.w[0] - a0.w[0]
            c4 = (d4.w[0] - th.w[0]) - null["C4_delta_w0"]
            detect.append(
                {
                    "zz_hz": zs,
                    "C3_delta_w0": c3,
                    "C3_z": abs(c3) / (infl * math.hypot(sd_c3, sd_orig) / math.sqrt(n_arm)),
                    "C4_signal_w0": c4,
                    "C4_z": abs(c4) / (infl * math.hypot(sd_dd, sd_orig) / math.sqrt(n_arm)),
                    "thermal_w0_ratio": th.w[0] / ctx.base.w[0],
                }
            )
    log(out, "[controls] C1, C3, C4 predictions done")
    return preds, contrasts, [*costs, {"_detect": detect, "_sd": sd_cache}]


# -- C2: alternative pairs ----------------------------------------------------------------------


def pairs(ctx: Ctx, out: Path) -> tuple[list[dict], list[dict]]:
    cc, w = ctx.cfg["controls"], ctx.cfg["witness"]
    dev, nb = ctx.dev, ctx.nb
    dist0 = distances(nb, [ctx.out, ctx.mem])
    # FakeKingston's prediction for the Phase 6 pair itself (staleness of its calibration)
    ref_fake = ctx.predict(ctx.model(ctx.out, ctx.mem, {}, source="fake"))
    rows = []
    for a, b in dev["edges"]:
        for mem, outq in ((a, b), (b, a)):
            qa, qb = dev["qubits"][mem], dev["qubits"][outq]
            if not (qa["t1"] and qa["t2"] and qb["t1"] and qb["t2"]):
                continue
            if max(qa["readout"], qb["readout"]) > cc["c2_max_readout"]:
                continue
            if min(dist0.get(mem, 99), dist0.get(outq, 99)) < cc["c2_min_distance"]:
                continue
            pr = ctx.predict(ctx.model(outq, mem, {}, source="fake"))
            rows.append(
                {
                    "memory": mem,
                    "output": outq,
                    "distance_from_59_75": min(dist0[mem], dist0[outq]),
                    "memory_degree": len(nb[mem]),
                    "memory_other_neighbours": " ".join(str(v) for v in nb[mem] if v != outq),
                    "mem_t1_us": qa["t1"] * 1e6,
                    "mem_t2_us": qa["t2"] * 1e6,
                    "out_readout": qb["readout"],
                    "cz_error": dev["cz"][f"{min(a, b)}-{max(a, b)}"]["error"],
                    "pred_w0": pr.w[0],
                    "pred_p1": pr.p1,
                    "pred_w0_over_fake_59_75": pr.w[0] / ref_fake.w[0],
                }
            )
    rows.sort(key=lambda r: r["pred_w0"])  # most negative first
    chosen: list[dict] = []
    for r in rows:
        if len(chosen) >= cc["c2_n_pairs"]:
            break
        if any(
            r["memory"] in (c["memory"], c["output"]) or r["output"] in (c["memory"], c["output"])
            for c in chosen
        ):
            continue
        d = distances(nb, [r["memory"], r["output"]])
        if any(min(d[c["memory"]], d[c["output"]]) < cc["c2_min_distance"] for c in chosen):
            continue
        chosen.append(r)
    # predictions for chosen pairs: null and the translated spectator scenarios
    scs = scenarios(ctx)
    details = []
    for r in chosen:
        mem, outq = r["memory"], r["output"]
        sp_m = next(v for v in nb[mem] if v != outq)
        sp_o = next((v for v in nb[outq] if v != mem), None)
        mapping = {ctx.mem: mem, ctx.out: outq, ctx.spec_m: sp_m, ctx.spec_o: sp_o}
        sd = S.per_shot_sd(
            ctx.proc(ctx.model(outq, mem, {}, source="fake")),
            0,
            cc["sd_samples"],
            cc["sd_seed"],
            w["burn"],
        )
        null_w0 = None
        for scn, sc in scs.items():
            tsc = translate(sc, mapping)
            for q, kw in tsc.get("spectators", {}).items():
                kw["t1"] = dev["qubits"][q]["t1"]
            pr = ctx.predict(ctx.model(outq, mem, tsc, source="fake"))
            if scn == "null":
                null_w0 = pr.w[0]
            delta = pr.w[0] - null_w0
            details.append(
                {
                    "pair": f"q{mem} memory / q{outq} output",
                    "spectators": f"q{sp_m} (memory side), q{sp_o} (output side)",
                    "scenario": scn,
                    "w0": pr.w[0],
                    "w0_over_null": pr.w[0] / null_w0,
                    "p1": pr.p1,
                    "sd_w0_per_shot": sd,
                    "shots_5sigma_vs_model": S.shots_for(
                        delta, sd, 0.0, z=w["z_target"], inflate=w["se_inflation"]
                    ),
                    "shots_5sigma_nonzero": S.shots_for(
                        pr.w[0], sd, 0.0, z=w["z_target"], inflate=w["se_inflation"]
                    ),
                }
            )
    ws = np.array([r["pred_w0_over_fake_59_75"] for r in rows])
    spread = {
        "n_candidates": len(rows),
        "w0_over_fake_59_75_quantiles": {
            q: float(np.quantile(ws, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }
    chosen_txt = [(c["memory"], c["output"]) for c in chosen]
    log(out, f"[C2] {len(rows)} candidate directed pairs; chosen {chosen_txt}")
    return rows, chosen, details, ref_fake, spread  # type: ignore[return-value]


# -- 5. QASM for the controls ----------------------------------------------------------------


def control_qasm(ctx: Ctx, chosen: list[dict]) -> dict[str, str]:
    """Full 64-step circuits (physical qubits), derived from Phase 6's transpiled circuit.qasm."""
    src = (p6(ctx.cfg) / "circuit.qasm").read_text()
    o, mm, sm = ctx.out, ctx.mem, ctx.spec_m

    def remap(text: str, mapping: dict[int, int]) -> str:
        return re.sub(
            r"\$(\d+)", lambda m_: f"${mapping.get(int(m_.group(1)), int(m_.group(1)))}", text
        )

    out: dict[str, str] = {}  # the original is results/summaries/phase6/circuit.qasm
    out["c1_role_swap.qasm"] = remap(src, {o: mm, mm: o})
    for r in chosen:
        out[f"c2_mem{r['memory']}_out{r['output']}.qasm"] = remap(
            src, {mm: r["memory"], o: r["output"]}
        )
    lines = src.splitlines()
    first = next(
        i for i, ln in enumerate(lines) if not ln.startswith(("OPENQASM", "include", "bit"))
    )
    for label, prep in (
        ("c3_spectator0.qasm", [f"reset ${sm};"]),
        ("c3_spectator1.qasm", [f"reset ${sm};", f"x ${sm};"]),
    ):
        out[label] = (
            "\n".join(lines[:first] + prep + [f"barrier ${sm}, ${mm}, ${o};"] + lines[first:])
            + "\n"
        )
    # C4: XX echo on the memory through each readout (+ reset) of the output
    m = ctx.cal
    grid = ctx.cfg["controls"]["dd_grid_s"]
    dd = S.add_step_dd(
        [S.Instr("measure", (0,)), S.Instr("reset", (0,))],
        1,
        m.measure_time,
        m.reset_time,
        grid=grid,
    )
    meas_part = [i for i in dd[1:4]]
    reset_part = [i for i in dd[5:8]]

    def fmt(ins: S.Instr) -> str:
        if ins.name == "delay":
            return f"delay[{round(ins.param * 1e9)}ns] ${mm};"
        return f"x ${mm};"

    new = []
    for ln in lines:
        if re.search(rf"= measure \${o};", ln):
            new.append(f"barrier ${o}, ${mm};")
            new.append(ln)
            new += [fmt(i) for i in meas_part]
            continue
        new.append(ln)
        if ln.strip() == f"reset ${o};":
            new += [fmt(i) for i in reset_part]
            new.append(f"barrier ${o}, ${mm};")
    out["c4_dd.qasm"] = "\n".join(new) + "\n"
    return out


def statement(ctx: Ctx, contrasts, segs) -> dict[str, Any]:
    """Change of w(0) under each scenario, as a fraction of the episode-free device w(0)."""
    rest = abs(segs["rest"]["w"][0])
    null = contrasts[0]["orig_w0"]
    return {
        r["scenario"]: {
            "delta_w0": r["orig_w0"] - null,
            "fraction_of_rest_witness": abs(r["orig_w0"] - null) / rest,
            "ratio_to_calibrated": r["orig_w0"] / null,
        }
        for r in contrasts
    }


# -- plots -----------------------------------------------------------------------------------


def plot(ctx: Ctx, res, segs, sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    rows = [r for r in res["zz_memory_spectator"] if r["kind"] == "thermal"]
    for p in (0.01, 0.05, 0.2, 1.0):
        rr = sorted((r for r in rows if r["p"] == p), key=lambda r: r["zz_hz"])
        ax[0].plot(
            [r["zz_hz"] / 1e3 for r in rr],
            [r["w0_ratio"] for r in rr],
            "o-",
            ms=3,
            label=f"p = {p}",
        )
    ax[0].set_xscale("symlog", linthresh=1)
    ax[0].axhspan(0.9, 1.1, color="0.9")
    ax[0].axhline(
        segs["rest"]["w"][0] / ctx.base.w[0], color="k", ls="--", lw=0.8, label="device, rest"
    )
    ax[0].set_xlabel("ZZ q75-q74 (kHz)")
    ax[0].set_ylabel("w(0) / calibrated")
    ax[0].set_title("thermal spectator q74 (T1 from FakeKingston)")
    ax[0].legend(fontsize=7)
    for key, lab in (("zz_pair", "ZZ q59-q75"),):
        rr = sorted(res[key], key=lambda r: r["zz_hz"])
        ax[1].plot(
            [r["zz_hz"] / 1e3 for r in rr], [r["w0_ratio"] for r in rr], "o-", ms=3, label=lab
        )
    rr = sorted(
        (r for r in res["readout_crosstalk"] if r.get("kind") == "uniform"),
        key=lambda r: r["stark_hz"],
    )
    ax[1].plot(
        [r["stark_hz"] / 1e3 for r in rr],
        [r["w0_ratio"] for r in rr],
        "s-",
        ms=3,
        label="Stark on q75 in readout",
    )
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].axhspan(0.9, 1.1, color="0.9")
    ax[1].set_xlabel("strength (kHz)")
    ax[1].set_title("pair ZZ and readout Stark shift")
    ax[1].legend(fontsize=7)
    ks = np.arange(8)
    for name, c in (("rest", "C0"), ("episode", "C3")):
        s = segs[name]
        ax[2].errorbar(ks, s["g"], yerr=s["g_se"], fmt="o", color=c, ms=3, label=f"device {name}")
    ax[2].plot(ks, ctx.base.g, "k-", lw=0.8, label="calibrated")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("g(k)")
    ax[2].set_title("episode vs rest (common start set)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sd / "sensitivity.png", dpi=130)
    plt.close(fig)


# -- main ------------------------------------------------------------------------------------------


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating | np.integer):
        return x.item()
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--export-device", action="store_true")
    ap.add_argument(
        "--from-run", type=Path, default=None, help="reuse this run dir (cached sections)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export_device:
        export_device(cfg)
        return
    out = args.from_run or new_run_dir(cfg["experiment"]["name"])
    out.mkdir(parents=True, exist_ok=True)
    sd = sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(
        {k: v for k, v in cfg.items() if not k.startswith("_")}, cfg["witness"]["seed"]
    )
    ctx = Ctx(cfg)
    log(out, f"[start] run dir {out}")

    # exact-recovery check against the committed #18 instrument
    inst = np.load(REPO_ROOT / cfg["experiment"]["audit_dir"] / "instruments.npz")
    rec = float(np.abs(S.step_superops(ctx.prog, ctx.cal) - inst["calibrated"]).max())
    log(out, f"[check] |S - instruments.npz calibrated| = {rec:.2e}")

    topo, idle = topology(ctx)
    write_csv(sd / "topology.csv", topo)
    write_csv(sd / "idle.csv", idle)

    cache = out / "sweeps.json"
    if cache.exists():
        res = json.loads(cache.read_text())
    else:
        res = sweeps(ctx, out)
        cache.write_text(json.dumps(jsonable(res)))
    for k, rows in res.items():
        write_csv(sd / f"sweep_{k}.csv", rows)
    own = own_witness(ctx)
    write_csv(sd / "own_witness.csv", own)

    segs = device_segments(ctx)
    fits = fit_episode(ctx, segs, out)
    write_csv(sd / "episode_fits.csv", fits)

    preds, contrasts, extra = controls(ctx, out)
    detect = extra[-1]["_detect"]
    sds = extra[-1]["_sd"]
    costs = extra[:-1]
    write_csv(sd / "controls_predictions.csv", preds)
    write_csv(sd / "controls_contrasts.csv", contrasts)
    write_csv(sd / "controls_cost.csv", costs)
    write_csv(sd / "detectable_zz.csv", detect)

    cand, chosen, details, ref_fake, spread = pairs(ctx, out)
    write_csv(sd / "c2_candidates.csv", cand[:40])
    write_csv(sd / "c2_predictions.csv", details)

    qdir = sd / "qasm"
    qdir.mkdir(exist_ok=True)
    for name, text in control_qasm(ctx, chosen).items():
        (qdir / name).write_text(text)

    plot(ctx, res, segs, sd)

    hw = cfg["hardware"]
    summary = {
        "meta": meta,
        "exact_recovery_max_abs": rec,
        "calibrated": ctx.base.row(),
        "device_segments": {k: {kk: jsonable(vv) for kk, vv in v.items()} for k, v in segs.items()},
        "per_shot_sd_w0_model": sds,
        "fake_kingston_59_75": ref_fake.row(),
        "c2_chosen": chosen,
        "c2_calibration_spread": spread,
        "statement": statement(ctx, contrasts, segs),
        "hardware": {
            "submitted": hw["submitted"],
            "backend": hw["backend"],
            "note": "UNSUBMITTED. Stage 3 re-reads live calibration, redoes the C2 ranking and "
            "asks the owner before any job.",
            "jobs": costs,
        },
    }
    save_json(sd / "summary.json", jsonable(summary))
    save_json(out / "summary.json", jsonable(summary))
    log(out, f"[done] summaries in {sd}")


if __name__ == "__main__":
    main()
