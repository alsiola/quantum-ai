"""Phase 6 (issue #16): the trained 1-qubit PMD N = 3 model as a generator on an IBM QPU.

The only script in the campaign that calls a quantum cloud service, at the owner's request
on #16. Needs the optional ``hardware`` group and ``IBM_API_KEY`` / ``IBM_INSTANCE_CRN`` in
``.env``::

    uv run --group hardware python experiments/phase6.py \
        --config experiments/configs/phase6.toml --dry-run
    uv run --group hardware python experiments/phase6.py --config experiments/configs/phase6.toml

Stages (each saves to the run directory and is skipped when its output is there, so a run
can be resumed with ``--from-run``; a submitted job is fetched by its id, never resubmitted):

1. ``train``: re-train ``pmd_N3`` exactly as Phase 5a (``experiments/phase5a.py``).
2. ``prepare``: export the model for ``length`` symbols, transpile it against the live
   backend's target, and snapshot the calibration of the chosen qubits.
3. ``pilot`` and ``main``: submit one SamplerV2 job each (no error mitigation), after checking
   the estimated QPU time against ``max_qpu_seconds`` and the account's remaining allowance;
   save the job id at once, then every shot, the job's metadata, usage and metrics, and the
   calibration again after the job. ``pilot`` checks the output is sensible before ``main``.
4. ``analyse``: the Hankel dimension witness, per-position and block statistics, and
   classical HMMs fitted to the device's own shots, against exact predictions of the PMD
   source, the noiseless trained model and the noisy model built from the calibration.

``--dry-run`` replaces the QPU by a local Aer simulator of the live backend (its noise model
from the same calibration); nothing is submitted. Writes ``results/phase6/<timestamp>/``
(``results/phase6_dryrun/`` for a dry run) and, for a real run, ``results/summaries/phase6/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import tomllib
import traceback
import warnings
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5a

from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

STAGES = ("train", "prepare", "pilot", "main", "analyse")


def now() -> str:
    return datetime.now(UTC).isoformat()


def log(out_dir: Path, msg: str) -> None:
    """Print and append to the run's log (every stage logs here)."""
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(out_dir / "log.txt", "a") as f:
        f.write(line + "\n")


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    base = phase5a.load_config(REPO_ROOT / cfg["experiment"]["base_config"])
    base["models"] = [m for m in base["models"] if m["name"] == cfg["experiment"]["model"]]
    if len(base["models"]) != 1:
        raise SystemExit(f"model {cfg['experiment']['model']} not in the base config")
    return cfg, base


# -- IBM ----------------------------------------------------------------------------------
def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def service(cfg: dict[str, Any]) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    env = {**read_env(REPO_ROOT / cfg["ibm"]["env_file"]), **os.environ}
    return QiskitRuntimeService(
        channel=cfg["ibm"]["channel"], token=env["IBM_API_KEY"], instance=env["IBM_INSTANCE_CRN"]
    )


def usage(svc: Any) -> dict[str, Any]:
    u = dict(svc.usage())
    u.pop("instance_id", None)  # an account identifier; not needed in the records
    return u


def jsonable(x: Any) -> Any:
    """Best-effort conversion of Qiskit/IBM objects to JSON."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None or isinstance(x, bool | int | float | str):
        return x
    return repr(x)


# -- noise snapshots ------------------------------------------------------------------------
def noise_to_json(noise: Any) -> dict[str, Any]:
    out = {}
    for f in fields(noise):
        v = getattr(noise, f.name)
        if isinstance(v, dict):
            v = [[list(k) if isinstance(k, tuple) else k, val] for k, val in v.items()]
        out[f.name] = v
    return out


def noise_from_json(d: dict[str, Any]) -> Any:
    from qmem.hardware import StepNoise

    kw = {}
    for k, v in d.items():
        if isinstance(v, list):
            v = {(tuple(a) if isinstance(a, list) else a): b for a, b in v}
        kw[k] = v
    return StepNoise(**kw)


def calibration(backend: Any, phys: list[int], ff: float) -> dict[str, Any]:
    """The chosen qubits' calibration as the noise model sees it, plus the raw properties."""
    from qmem import hardware as hw

    noise, info = hw.backend_noise(backend, phys, ff)
    snap: dict[str, Any] = {"read_at": now(), "noise": noise_to_json(noise), "info": info}
    try:
        props = backend.properties(refresh=True)
        snap["last_update_date"] = jsonable(props.last_update_date)
        d = props.to_dict()
        snap["qubits"] = {p: jsonable(d["qubits"][p]) for p in phys}
        snap["gates"] = [jsonable(g) for g in d["gates"] if set(g.get("qubits", [])) <= set(phys)]
    except Exception as e:  # fake/Aer backends have no live properties
        snap["properties_error"] = repr(e)
    return jsonable(snap)


# -- 2. prepare -----------------------------------------------------------------------------
def transpile_on(m: Any, backend: Any, cfg: dict[str, Any], length: int, layout=None):
    from qiskit import transpile

    from qmem import hardware as hw

    qc = hw.export_circuit(m, length, "dynamic", reset_first=(length == 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transpile(
            qc,
            backend,
            optimization_level=cfg["circuit"]["optimization_level"],
            seed_transpiler=cfg["circuit"]["seed_transpiler"],
            initial_layout=layout,
        )


def prepare(cfg, m, backend, out_dir: Path) -> dict[str, Any]:
    """Transpile the T-symbol circuit, then its one-step unit on the same physical qubits."""
    from qiskit import qasm3, qpy

    from qmem import hardware as hw

    length = cfg["circuit"]["length"]
    tq = transpile_on(m, backend, cfg, length)
    ini = tq.layout.initial_index_layout(filter_ancillas=True)
    fin = tq.layout.final_index_layout(filter_ancillas=True)
    layout = [ini[i] for i in range(m.n_memory + m.n_output)]  # memory, output
    step = transpile_on(m, backend, cfg, 1, layout=layout)
    s_ini = step.layout.initial_index_layout(filter_ancillas=True)
    s_fin = step.layout.final_index_layout(filter_ancillas=True)
    cq, phys = hw.compact(step)
    pos = {p: k for k, p in enumerate(phys)}
    mem_in = [pos[s_ini[i]] for i in range(m.n_memory)]
    mem_out = [pos[s_fin[i]] for i in range(m.n_memory)]
    ff = cfg["circuit"]["feedforward_latency_s"]
    step_s = phase5a.step_duration(cq, backend.target, ff)
    try:
        sched_s = tq.estimate_duration(backend.target, unit="s")
    except Exception as e:
        sched_s = None
        print(f"estimate_duration failed: {e!r}")
    ops = hw.count_ops(tq)
    with open(out_dir / "circuit.qpy", "wb") as f:
        qpy.dump(tq, f)
    with open(out_dir / "step.qpy", "wb") as f:
        qpy.dump(cq, f)
    (out_dir / "circuit.qasm").write_text(qasm3.dumps(tq))
    (out_dir / "step.qasm").write_text(qasm3.dumps(cq))
    return {
        "backend": backend.name,
        "length": length,
        "layout_memory_output": layout,
        "final_layout_memory_output": [fin[i] for i in range(m.n_memory + m.n_output)],
        "step_physical": phys,
        "step_memory_in": mem_in,
        "step_memory_out": mem_out,
        "ops": ops,
        "two_qubit_gates": hw.two_qubit_count(tq),
        "depth": tq.depth(),
        "step_duration_s": step_s,
        "circuit_duration_s": sched_s if sched_s is not None else step_s * length,
        "calibration": calibration(backend, phys, ff),
    }


# -- 3. jobs --------------------------------------------------------------------------------
def estimate_qpu_seconds(cfg: dict[str, Any], prep: dict[str, Any], shots: int) -> float:
    b = cfg["budget"]
    return b["sub_job_overhead_s"] + shots * (b["rep_delay_s"] + prep["circuit_duration_s"])


def make_sampler(cfg: dict[str, Any], mode: Any) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    s = SamplerV2(mode=mode)
    o = cfg["sampler"]
    try:
        s.options.dynamical_decoupling.enable = o["dynamical_decoupling"]
        s.options.twirling.enable_gates = o["twirling_gates"]
        s.options.twirling.enable_measure = o["twirling_measure"]
    except AttributeError as e:  # local mode ignores some options
        print(f"sampler option not set: {e!r}")
    return s


def shots_from_result(res: Any, length: int) -> np.ndarray:
    """``(shots, length)`` tokens from a SamplerV2 result, register ``y<t>`` = token ``t``."""
    data = res[0].data
    cols = [np.asarray(getattr(data, f"y{t}").array)[:, 0] for t in range(length)]
    return np.stack(cols, axis=1).astype(np.uint8)


def run_job(stage: str, cfg, svc, backend, prep, out_dir: Path, dry_run: bool) -> np.ndarray:
    """Submit (or re-fetch) one job; save everything about it; return its shots."""
    from qiskit import qpy

    d = out_dir / stage
    d.mkdir(exist_ok=True)
    shots_path = d / "shots.npy"
    if shots_path.exists():
        log(out_dir, f"[{stage}] shots already saved; skipping")
        return np.load(shots_path)
    shots = cfg[stage]["shots"]
    if dry_run:  # local Aer is slow on many noisy mid-circuit shots; the code path is the same
        shots = min(shots, cfg["dry_run"]["max_shots"])
    length = cfg["circuit"]["length"]
    with open(out_dir / "circuit.qpy", "rb") as f:
        tq = qpy.load(f)[0]
    job_path = d / "job.json"
    rec: dict[str, Any] = json.loads(job_path.read_text()) if job_path.exists() else {}
    if dry_run:
        from qiskit_aer import AerSimulator

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mode = AerSimulator.from_backend(backend)
    else:
        mode = backend
    if "job_id" in rec and not dry_run:
        log(out_dir, f"[{stage}] re-fetching job {rec['job_id']}")
        job = svc.job(rec["job_id"])
    else:
        est = estimate_qpu_seconds(cfg, prep, shots)
        rec.update({"stage": stage, "shots": shots, "estimated_qpu_s": est, "dry_run": dry_run})
        if not dry_run:
            u = usage(svc)
            rec["usage_before"] = u
            remaining = u.get("usage_remaining_seconds")
            if est > cfg["budget"]["max_qpu_seconds"]:
                raise SystemExit(f"[{stage}] estimated {est:.1f} s > max_qpu_seconds; not sent")
            if remaining is not None and est > remaining:
                raise SystemExit(f"[{stage}] estimated {est:.1f} s > remaining {remaining} s")
        sampler = make_sampler(cfg, mode)
        log(out_dir, f"[{stage}] submitting {shots} shots (estimated {est:.2f} s QPU)")
        job = sampler.run([tq], shots=shots)
        rec["job_id"] = job.job_id()
        rec["submitted_at"] = now()
        save_json(job_path, rec)  # the id is on disk before we wait
        log(out_dir, f"[{stage}] job {rec['job_id']} submitted")
    t0 = time.perf_counter()
    res = job.result()
    rec["result_seconds_waited"] = time.perf_counter() - t0
    rec["fetched_at"] = now()
    arr = shots_from_result(res, length)
    np.save(shots_path, arr)
    rec["result_metadata"] = jsonable(dict(getattr(res, "metadata", {}) or {}))
    rec["pub_metadata"] = jsonable(dict(getattr(res[0], "metadata", {}) or {}))
    for attr in ("usage", "metrics", "status"):
        try:
            v = getattr(job, attr)()
            rec[attr] = jsonable(v if not hasattr(v, "name") else v.name)
        except Exception as e:
            rec[f"{attr}_error"] = repr(e)
    if not dry_run:
        rec["usage_after"] = usage(svc)
        rec["calibration_after"] = calibration(
            svc.backend(backend.name),
            prep["step_physical"],
            cfg["circuit"]["feedforward_latency_s"],
        )
    save_json(job_path, rec)
    log(out_dir, f"[{stage}] {arr.shape[0]} shots saved; usage {rec.get('usage')}")
    return arr


# -- 4. analysis ------------------------------------------------------------------------------
class Proc:
    """A binary process with a state: an instrument (density matrix) or an HMM (belief)."""

    def __init__(self, name: str, inst: Any = None, rho0=None, lab=None, init=None):
        self.name, self.inst, self.rho0, self.lab, self.init = name, inst, rho0, lab, init

    @classmethod
    def hmm(cls, name: str, lab: torch.Tensor, init: torch.Tensor) -> Proc:
        return cls(name, lab=lab.to(torch.float64), init=init.to(torch.float64))

    def start(self):
        return self.rho0 if self.inst is not None else self.init

    def branch(self, s, y: int):
        if self.inst is not None:
            return self.inst.branch(s, y)
        return s @ self.lab[y]

    def channel(self, s):
        return self.inst.channel(s) if self.inst is not None else s @ self.lab.sum(0)

    def mass(self, s) -> float:
        return float(s.diagonal().sum().real) if self.inst is not None else float(s.sum())

    def after(self, t: int):
        s = self.start()
        for _ in range(t):
            s = self.channel(s)
        return s

    def marginals(self, length: int) -> list[float]:
        s, out = self.start(), []
        for _ in range(length):
            out.append(self.mass(self.branch(s, 1)))
            s = self.channel(s)
        return out

    def windows(self, k_max: int, burn: int) -> torch.Tensor:
        cur, out = self.branch(self.after(burn), 1), []
        for _ in range(k_max + 1):
            out.append(self.mass(self.branch(cur, 1)))
            cur = self.branch(cur, 0)
        return torch.tensor(out, dtype=torch.float64)

    def blocks(self, starts: list[int], length: int) -> np.ndarray:
        """Distribution of the ``length``-block, averaged over the start positions."""
        p = np.zeros(2**length)
        for t0 in starts:
            states = [(self.after(t0), 0)]
            for _ in range(length):
                states = [(self.branch(s, y), (i << 1) | y) for s, i in states for y in (0, 1)]
            for s, i in states:
                p[i] += self.mass(s)
        return p / len(starts)

    def nll(self, seqs: torch.Tensor) -> float:
        """Mean ``-log p(y_t | y_<t)`` per symbol over ``seqs`` (forward algorithm)."""
        s = (
            self.rho0.expand(len(seqs), -1, -1)
            if self.inst is not None
            else self.init.expand(len(seqs), -1)
        )
        total = torch.zeros(len(seqs), dtype=torch.float64)
        for t in range(seqs.shape[1]):
            y = seqs[:, t]
            if self.inst is not None:
                b = torch.stack([self.inst.branch(s, k) for k in (0, 1)])  # (2, B, d, d)
                nb = b[y, torch.arange(len(seqs))]
                z = nb.diagonal(dim1=-2, dim2=-1).sum(-1).real
                s = nb / z[:, None, None]
            else:
                nb = torch.einsum("bi,bij->bj", s, self.lab[y])
                z = nb.sum(-1)
                s = nb / z[:, None]
            total -= torch.log(z)
        return float(total.sum() / seqs.numel())


def sample_proc(p: Proc, n: int, length: int, seed: int) -> torch.Tensor:
    from qmem import hardware as hw
    from qmem.tasks import HMMTask

    if p.inst is not None:
        return hw.InstrumentModel(p.inst, 1, 1, rho0=p.rho0).sample(n, length, seed)
    return HMMTask(p.lab, p.init).sample(n, length, seed)


def witness_stats(seqs: torch.Tensor, w: dict[str, Any]) -> dict[str, Any]:
    per = phase5a.empirical_windows(seqs, w["k_max"], w["burn"])
    g_hat = per.mean(0)
    gen = torch.Generator().manual_seed(w["seed"])
    boots = []
    for _ in range(w["bootstrap"]):
        idx = torch.randint(len(per), (len(per),), generator=gen)
        boots.append(per[idx].mean(0))
    out: dict[str, Any] = {"g": g_hat.tolist(), "g_se": per.std(0).div(len(per) ** 0.5).tolist()}
    for j in w["offsets"]:
        est = phase5a.hankel_witness(g_hat, j)
        bs = [phase5a.hankel_witness(b, j) for b in boots]
        se = statistics.stdev(bs)
        out[f"w_j{j}"] = est
        out[f"se_j{j}"] = se
        out[f"z_j{j}"] = est / se if se > 0 else math.nan
        out[f"ci95_j{j}"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    out["max_abs_z"] = max(abs(out[f"z_j{j}"]) for j in w["offsets"])
    return out


def block_counts(seqs: np.ndarray, starts: list[int], length: int) -> np.ndarray:
    """``(shots, 2**length)`` counts of each aligned block per shot."""
    weights = 1 << np.arange(length - 1, -1, -1)
    idx = np.stack([seqs[:, t : t + length] @ weights for t in starts], axis=1)
    out = np.zeros((len(seqs), 2**length))
    for c in range(idx.shape[1]):
        np.add.at(out, (np.arange(len(seqs)), idx[:, c]), 1)
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    if np.any(q[m] <= 0):
        return math.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def block_stats(seqs: np.ndarray, procs: list[Proc], a: dict[str, Any], label: str):
    """D_L/L(empirical || p) and D_L/L(p || empirical) for each prediction p, with bootstrap."""
    burn, length = a["witness"]["burn"], seqs.shape[1]
    rows = []
    rng = np.random.default_rng(a["simulated_seed"])
    for L in a["block_lengths"]:
        starts = list(range(burn, length - L + 1, L))
        counts = block_counts(seqs, starts, L)
        emp = counts.sum(0) / counts.sum()
        boots = []
        for _ in range(a["kl_bootstrap"]):
            idx = rng.integers(0, len(seqs), len(seqs))
            c = counts[idx].sum(0)
            boots.append(c / c.sum())
        for p in procs:
            q = p.blocks(starts, L)
            fwd = kl(emp, q) / L
            rev = kl(q, emp) / L
            fb = [kl(b, q) / L for b in boots]
            rows.append(
                {
                    "data": label,
                    "L": L,
                    "n_blocks": int(counts.sum()),
                    "prediction": p.name,
                    "kl_emp_vs_pred": fwd,
                    "kl_emp_vs_pred_se": float(np.std(fb)),
                    "kl_pred_vs_emp": rev,
                    "tv": 0.5 * float(np.abs(emp - q).sum()),
                }
            )
    return rows


def fit_hmms(seqs: torch.Tensor, a: dict[str, Any], log_fn) -> list[dict[str, Any]]:
    """Best-of-restarts EM fits of D-state HMMs to the first part of the shots."""
    from qmem.baselines.hmm import HMMModel, fit_em

    h = a["hmm"]
    n_tr = int(len(seqs) * h["train_fraction"])
    n_fit = int(n_tr * 0.9)
    train, val = seqs[:n_fit], seqs[n_fit:n_tr]
    out = []
    for d in h["states"]:
        best = None
        for r in range(h["restarts"]):
            model = HMMModel(d, 2, seed=h["seed"] + 1000 * d + r)
            rec = fit_em(model, train, val, max_iters=h["max_iters"], patience=h["patience"])
            v = rec["best_val_nll"]
            if best is None or v < best[0]:
                best = (v, r, model.labelled().detach().clone(), model.initial().detach().clone())
        log_fn(f"[hmm] D={d} best restart {best[1]} val NLL {best[0]:.6f}")
        out.append(
            {
                "states": d,
                "restart": best[1],
                "val_nll": best[0],
                "labelled": best[2].tolist(),
                "initial": best[3].tolist(),
                "n_free_parameters": d * (2 * d - 1) + d - 1,
            }
        )
    return out


@torch.no_grad()
def analyse(cfg, base, trained, prep, shots: dict[str, np.ndarray], out_dir: Path):
    from qiskit import qpy

    from qmem import hardware as hw

    a = cfg["analysis"]
    w = a["witness"]
    t = trained[cfg["experiment"]["model"]]
    m, task = t["model"], t["task"]
    length = cfg["circuit"]["length"]
    with open(out_dir / "step.qpy", "rb") as f:
        cq = qpy.load(f)[0]
    mem_in, mem_out = prep["step_memory_in"], prep["step_memory_out"]

    def noisy(cal: dict[str, Any], idle: bool = True) -> Any:
        noise = noise_from_json(cal["noise"])
        if not idle:
            noise.t1 = None
        return hw.noisy_instrument(cq, 1, 1, noise, mem_in, memory_qubits_out=mem_out)

    procs = [
        Proc.hmm("source", task.labelled, task.initial),
        Proc("noiseless", m.instrument(), m.rho0),
        Proc("calibrated", noisy(prep["calibration"]), m.rho0),
        Proc("calibrated_no_idle", noisy(prep["calibration"], idle=False), m.rho0),
    ]
    main_job = out_dir / "main" / "job.json"
    if main_job.exists():
        cal_after = json.loads(main_job.read_text()).get("calibration_after")
        if cal_after and "noise" in cal_after:
            procs.append(Proc("calibrated_after_job", noisy(cal_after), m.rho0))
    ctrl = phase5a.control_hmm(a["control_seed"])
    control = Proc.hmm("control_2state_hmm", ctrl.labelled, ctrl.initial)

    res: dict[str, Any] = {"predictions": {}, "device": {}, "simulated": {}}
    for p in [*procs, control]:
        g = p.windows(w["k_max"], w["burn"])
        res["predictions"][p.name] = {
            "g": g.tolist(),
            **{f"w_j{j}": phase5a.hankel_witness(g, j) for j in w["offsets"]},
            "marginals": p.marginals(length),
        }
    for stage, arr in shots.items():
        seqs = torch.as_tensor(arr, dtype=torch.long)
        n = len(seqs)
        log(out_dir, f"[analyse] {stage}: {n} shots")
        d: dict[str, Any] = {"shots": n}
        d["mean_one"] = float(arr.mean())
        d["marginals"] = arr.mean(0).tolist()
        d["marginals_se"] = (arr.std(0) / math.sqrt(n)).tolist()
        d["drift_first_vs_second_half"] = [float(arr[: n // 2].mean()), float(arr[n // 2 :].mean())]
        d["witness"] = witness_stats(seqs, w)
        d["blocks"] = block_stats(arr, procs, a, f"device_{stage}")
        d["nll"] = {p.name: p.nll(seqs) for p in procs}
        # chi^2-style z of each position's P(1) against each prediction
        for p in procs:
            pm = np.array(res["predictions"][p.name]["marginals"])
            se = np.sqrt(pm * (1 - pm) / n)
            z = (arr.mean(0) - pm) / se
            d[f"marginal_chi2_vs_{p.name}"] = float(np.sum(z**2))
        res["device"][stage] = d
        log(
            out_dir,
            f"[analyse] {stage}: P(1)={d['mean_one']:.4f} max|z|={d['witness']['max_abs_z']:.2f} "
            + " ".join(f"z_j{j}={d['witness'][f'z_j{j}']:.2f}" for j in w["offsets"]),
        )
    # Null levels: simulated datasets of the main run's size from each prediction.
    n_main = len(shots.get("main", shots.get("pilot")))
    for p in [*procs, control]:
        sim = sample_proc(p, n_main, length, a["simulated_seed"])
        s = {"witness": witness_stats(sim, w), "nll": {q.name: q.nll(sim) for q in procs}}
        if p.name in ("calibrated", "source"):
            s["blocks"] = block_stats(sim.numpy().astype(np.uint8), procs, a, f"sim_{p.name}")
        res["simulated"][p.name] = s
        log(out_dir, f"[analyse] simulated {p.name}: max|z|={s['witness']['max_abs_z']:.2f}")
    # Classical HMMs fitted to the device's own main-run shots, scored on held-out shots.
    if "main" in shots:
        seqs = torch.as_tensor(shots["main"], dtype=torch.long)
        fits = fit_hmms(seqs, a, lambda s: log(out_dir, s))
        n_tr = int(len(seqs) * a["hmm"]["train_fraction"])
        test = seqs[n_tr:]
        fit_procs = []
        for f in fits:
            p = Proc.hmm(
                f"hmm{f['states']}_fit",
                torch.tensor(f["labelled"], dtype=torch.float64),
                torch.tensor(f["initial"], dtype=torch.float64),
            )
            fit_procs.append(p)
            f["test_nll"] = p.nll(test)
            g = p.windows(w["k_max"], w["burn"])
            f.update({f"exact_w_j{j}": phase5a.hankel_witness(g, j) for j in w["offsets"]})
            sim = sample_proc(p, n_main, length, a["simulated_seed"])
            f["simulated_witness"] = witness_stats(sim, w)
        res["hmm_fits"] = fits
        res["test_nll"] = {p.name: p.nll(test) for p in procs} | {
            p.name: p.nll(test) for p in fit_procs
        }
        res["test_shots"] = len(test)
        res["device"]["main"]["blocks"] += block_stats(shots["main"], fit_procs, a, "device_main")
    return res


# -- output -----------------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)


def witness_rows(res: dict[str, Any], offsets: list[int]) -> list[dict[str, Any]]:
    rows = []
    for stage, d in res["device"].items():
        rows.append(
            {"data": f"device_{stage}", "shots": d["shots"], **flat_w(d["witness"], offsets)}
        )
    for name, s in res["simulated"].items():
        rows.append(
            {
                "data": f"simulated_{name}",
                "exact": [res["predictions"][name][f"w_j{j}"] for j in offsets],
                **flat_w(s["witness"], offsets),
            }
        )
    for f in res.get("hmm_fits", []):
        rows.append(
            {
                "data": f"simulated_hmm{f['states']}_fit",
                "exact": [f[f"exact_w_j{j}"] for j in offsets],
                **flat_w(f["simulated_witness"], offsets),
            }
        )
    return rows


def flat_w(wit: dict[str, Any], offsets: list[int]) -> dict[str, Any]:
    out = {}
    for j in offsets:
        out[f"w_j{j}"] = wit[f"w_j{j}"]
        out[f"se_j{j}"] = wit[f"se_j{j}"]
        out[f"z_j{j}"] = wit[f"z_j{j}"]
    out["max_abs_z"] = wit["max_abs_z"]
    return out


def save_summary(cfg, results: dict[str, Any], shots: dict[str, np.ndarray], out_dir: Path):
    sd = REPO_ROOT / cfg["experiment"]["summary_dir"]
    sd.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(cfg, None)
    meta["run_dir"] = str(out_dir.relative_to(REPO_ROOT))
    save_json(sd / "summary.json", {"meta": meta, **results})
    for stage, arr in shots.items():  # every shot, bit-packed (64 symbols = 8 bytes)
        np.savez_compressed(
            sd / f"shots_{stage}.npz", packed=np.packbits(arr, axis=1), length=arr.shape[1]
        )
    for stage in ("pilot", "main"):
        p = out_dir / stage / "job.json"
        if p.exists():
            save_json(sd / f"job_{stage}.json", json.loads(p.read_text()))
    for name in ("circuit.qasm", "step.qasm", "log.txt"):
        (sd / name).write_text((out_dir / name).read_text())
    an = results["analysis"]
    offsets = cfg["analysis"]["witness"]["offsets"]
    write_csv(sd / "witness.csv", witness_rows(an, offsets))
    blocks = [r for d in an["device"].values() for r in d["blocks"]]
    blocks += [r for s in an["simulated"].values() for r in s.get("blocks", [])]
    write_csv(sd / "blocks.csv", blocks)
    nll = [{"data": f"device_{k}", **v["nll"]} for k, v in an["device"].items()]
    nll += [{"data": f"simulated_{k}", **v["nll"]} for k, v in an["simulated"].items()]
    if "test_nll" in an:
        nll.append({"data": "device_main_heldout", **an["test_nll"]})
    write_csv(sd / "nll.csv", nll)
    marg = []
    for t in range(cfg["circuit"]["length"]):
        r: dict[str, Any] = {"t": t}
        for k, v in an["device"].items():
            r[f"device_{k}"] = v["marginals"][t]
            r[f"device_{k}_se"] = v["marginals_se"][t]
        for k, v in an["predictions"].items():
            r[k] = v["marginals"][t]
        marg.append(r)
    write_csv(sd / "marginals.csv", marg)
    plots(cfg, an, sd)
    print(f"summary: {sd}", flush=True)


def plots(cfg: dict[str, Any], an: dict[str, Any], sd: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = cfg["analysis"]["witness"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ks = list(range(w["k_max"] + 1))
    for name, style in (("source", "k--"), ("noiseless", "C0-"), ("calibrated", "C1-")):
        ax[0].plot(ks, an["predictions"][name]["g"], style, label=name)
    ax[0].plot(ks, an["predictions"]["control_2state_hmm"]["g"], "C7:", label="2-state HMM control")
    if "main" in an["device"]:
        dv = an["device"]["main"]["witness"]
        ax[0].errorbar(ks, dv["g"], yerr=[2 * s for s in dv["g_se"]], fmt="C3o", label="device")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("g(k) = P(1 0^k 1)")
    ax[0].legend(fontsize=8)
    labels, zs = [], []
    for stage, d in an["device"].items():
        labels.append(f"device {stage}")
        zs.append(d["witness"]["max_abs_z"])
    for name, s in an["simulated"].items():
        labels.append(f"sim {name}")
        zs.append(s["witness"]["max_abs_z"])
    for f in an.get("hmm_fits", []):
        labels.append(f"sim hmm{f['states']} fit")
        zs.append(f["simulated_witness"]["max_abs_z"])
    ax[1].barh(labels, zs, color=["C3" if lb.startswith("device") else "C0" for lb in labels])
    ax[1].axvline(3, color="k", ls=":", lw=1)
    ax[1].set_xscale("symlog")
    ax[1].set_xlabel("max |z| of the Hankel witness")
    fig.tight_layout()
    fig.savefig(sd / "witness.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--from-run", type=Path)
    ap.add_argument("--stages", type=str, default=",".join(STAGES))
    ap.add_argument("--dry-run", action="store_true", help="local Aer instead of the QPU")
    args = ap.parse_args()
    cfg, base = load_config(args.config)
    stages = args.stages.split(",")
    if set(stages) - set(STAGES):
        raise SystemExit(f"unknown stages; choose from {STAGES}")
    name = cfg["experiment"]["name"] + ("_dryrun" if args.dry_run else "")
    out_dir = args.from_run or new_run_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config_meta.json", run_metadata(cfg, None) | {"dry_run": args.dry_run})
    log(out_dir, f"run dir {out_dir} stages {stages} dry_run={args.dry_run}")
    try:
        if "train" in stages:
            phase5a.train_models(base, out_dir, cfg["experiment"]["workers"])
        torch.set_num_threads(min(4, torch.get_num_threads()))
        trained = phase5a.load_trained(base, out_dir)
        t = trained[cfg["experiment"]["model"]]
        log(out_dir, f"[train] reproduction {t['repro']}")
        if any(v["abs_diff"] > 1e-9 for v in t["repro"].values()):
            raise SystemExit("re-trained model does not reproduce its reference")
        svc = service(cfg)
        backend = svc.backend(cfg["ibm"]["backend"])
        prep_path = out_dir / "prepare.json"
        if "prepare" in stages and not prep_path.exists():
            prep = prepare(cfg, t["model"], backend, out_dir)
            save_json(prep_path, prep)
            log(out_dir, f"[prepare] {json.dumps({k: prep[k] for k in list(prep)[:11]})}")
        prep = json.loads(prep_path.read_text())
        shots: dict[str, np.ndarray] = {}
        for stage in ("pilot", "main"):
            if stage in stages or (out_dir / stage / "shots.npy").exists():
                shots[stage] = run_job(stage, cfg, svc, backend, prep, out_dir, args.dry_run)
            if stage == "pilot" and "pilot" in shots and "main" in stages:
                an = analyse(cfg, base, trained, prep, {"pilot": shots["pilot"]}, out_dir)
                save_json(out_dir / "pilot_analysis.json", an)
                if not pilot_ok(an, out_dir):
                    raise SystemExit("pilot check failed; main run not submitted")
        if "analyse" in stages and shots:
            an = analyse(cfg, base, trained, prep, shots, out_dir)
            results = {"prepare": prep, "reproduction": t["repro"], "analysis": an}
            save_json(out_dir / "analysis.json", results)
            if not args.dry_run:
                save_summary(cfg, results, shots, out_dir)
    except BaseException as e:
        log(out_dir, f"FAILED: {e!r}\n{traceback.format_exc()}")
        raise


def pilot_ok(an: dict[str, Any], out_dir: Path) -> bool:
    """Sensible output: P(1) and every position's P(1) consistent with the calibrated model."""
    d = an["device"]["pilot"]
    pred = np.array(an["predictions"]["calibrated"]["marginals"])
    n = d["shots"]
    z_mean = (d["mean_one"] - pred.mean()) / math.sqrt(pred.mean() * (1 - pred.mean()) / (n * 64))
    chi2 = d["marginal_chi2_vs_calibrated"]
    # 64 positions: chi^2 with 64 dof has mean 64, sd ~11.3. Allow a generous margin; the
    # overall P(1) uses all 64 symbols per shot (correlated, so also generous).
    ok = abs(z_mean) < 6 and chi2 < 64 + 6 * math.sqrt(128)
    log(
        out_dir,
        f"[pilot] P(1) {d['mean_one']:.4f} vs {pred.mean():.4f} (z~{z_mean:.2f}); "
        f"positional chi2 {chi2:.1f} (64 dof); ok={ok}",
    )
    return ok


if __name__ == "__main__":
    main()
