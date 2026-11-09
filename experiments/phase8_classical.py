"""Phase 8 Stage 0B (issue #30): a classical hidden-state model with explicit intervention maps,
fitted jointly to every arm of Phase 7 Stage 4c (#27).

Does a classical model with d >= 3 hidden states, an emission rule for q59 and one stochastic
map per arm (``qmem.classical``) reproduce the baseline's witness, A2's, A1's and A3's
vanishing, the sham, D1 and A2's q75 record, all at once? The fit uses the committed shots
of every arm (``results/summaries/phase7_hw_4c/shots_job*.npz``), A2's joint (m, y) record
included, by maximum likelihood (EM with SQUAREM, several starts per class). It reports:

* each class's best fit (general d = 3..6; kappa, the QND-except-kappa A2 map; zpop, q75 a
  classical Z bit plus side memory q75's operations don't touch) with every statistic's
  z against the data;
* the calibrated quantum model (#27's analysis model: #20's pieces at the pilot's live
  calibration, the pilot's fitted q75 decay) on the same shots and statistics;
* two-fold cross-validation by pub parity (held-out NLL, classical against quantum);
* the fit without A2's q75 record (does the record constrain the class?), and without the
  baseline's job1 shots (#29: an arm-specific P(1) dip no stationary model produces);
* whether the calibrated quantum model is itself exactly a 4-state classical model (the
  Klein-tetrahedron scan).

No QPU, no IBM service. ``uv run python experiments/phase8_classical.py --config
experiments/configs/phase8_classical.toml``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_ablation as PA

from qmem import ablate_hw as A
from qmem import artefacts as AR
from qmem import classical as C
from qmem import drift as D
from qmem import refit as R
from qmem import witness as W
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

KEY = re.compile(r"j(\d+)_p(\d+)_(.+?)(?:_i(\d+))?_c(\d+)__(y|m)$")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def rdir(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["experiment"][key]


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


# -- data -------------------------------------------------------------------------------------


def pauli_table(cfg: dict[str, Any]) -> np.ndarray:
    return PA.pauli_table(PA.load_config(REPO_ROOT / cfg["experiment"]["ablation_config"]))


def load_shots(cfg: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Every arm's shots (job, then pub order) with job, pub and A3 instance labels, and A2's
    q75 record ``m``, decoded with ``qmem.witness.unpack_shots`` from #27's npz keys."""
    raw: dict[str, dict[str, list]] = {}
    for j in range(cfg["experiment"]["n_jobs"]):
        with np.load(rdir(cfg, "data_dir") / f"shots_job{j}.npz") as z:
            length = int(z["length"])
            for k in sorted(z.files):
                mt = KEY.match(k)
                if not mt:
                    continue
                job, pub, arm, inst, _, reg = mt.groups()
                a = raw.setdefault(arm, {"y": [], "m": [], "inst": [], "job": [], "pub": []})
                arr = W.unpack_shots(z[k], length)
                if reg == "m":
                    a["m"].append(arr)
                    continue
                a["y"].append(arr)
                a["job"].append(np.full(len(arr), int(job)))
                a["pub"].append(np.full(len(arr), int(pub)))
                a["inst"].append(np.full(len(arr), -1 if inst is None else int(inst)))
    return {arm: {k: np.concatenate(v) for k, v in a.items() if v} for arm, a in raw.items()}


def arm_data(shots: dict[str, dict[str, np.ndarray]], pt: np.ndarray, keep=None):
    """``qmem.classical.ArmData`` per arm (``keep[arm]``: a boolean mask over its shots)."""
    out = {}
    for arm, a in shots.items():
        d = C.ArmData(a["y"], a.get("m"), pt[a["inst"]] if arm == "A3" else None)
        if keep is not None and arm in keep:
            d = d.subset(keep[arm])
        out[arm] = d
    return out


# -- observed statistics and their SEs ---------------------------------------------------------


def observed(cfg: dict[str, Any], shots: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Every arm's statistics from the shots, with SEs: w(j) from #27's summary (its
    pre-registered block/shot method); g(k) and P(1) from whole shots; P(1) also with the
    between-job spread (the drift-inclusive SE used for the fit test); lag-1 from #27; A2's
    record statistics by a whole-shot bootstrap."""
    an = cfg["analysis"]
    burn, k_max = an["burn"], an["k_max"]
    s27 = json.loads((rdir(cfg, "data_dir") / "summary.json").read_text())["arms"]
    rng = np.random.default_rng(an["seed"])
    out: dict[str, Any] = {}
    for arm, a in shots.items():
        y = a["y"]
        per = W.per_shot_windows(y, k_max, burn, common=True)
        g = per.mean(0)
        p1s = y[:, burn:].mean(1)
        jobs = np.unique(a["job"])
        pj = np.array([p1s[a["job"] == j].mean() for j in jobs])
        se_shot = float(p1s.std(ddof=1) / math.sqrt(len(y)))
        se_job = float(pj.std(ddof=1) / math.sqrt(len(jobs)))
        row = {
            "n": len(y),
            "g": g.tolist(),
            "g_se": (per.std(0, ddof=1) / math.sqrt(len(y))).tolist(),
            "p1": float(p1s.mean()),
            "p1_se_shot": se_shot,
            "p1_se_job": se_job,
            "p1_se": max(se_shot, se_job),
            "p1_by_job": pj.tolist(),
            "lag1": AR.lag1_excess_data(y, burn),
            "lag1_se": s27[arm]["lag1_se"],
        }
        for j in an["offsets"]:
            row[f"w{j}"] = float(W.witness(g, j))
            row[f"w{j}_se"] = s27[arm][f"se{j}"]
        if arm == "A2":
            m = a["m"]
            st = R.record_stats(m, y, burn)
            boots = []
            for _ in range(an["n_boot"]):
                i = rng.integers(0, len(y), len(y))
                boots.append(R.record_stats(m[i], y[i], burn))
            for k, v in st.items():
                row[k] = v
                row[f"{k}_se"] = float(np.std([b[k] for b in boots], ddof=1))
        out[arm] = row
    return out


# -- models' statistics --------------------------------------------------------------------------


def process_stats(proc: W.LinearProcess, cfg: dict[str, Any]) -> dict[str, Any]:
    an = cfg["analysis"]
    length, burn, k_max = an["length"], an["burn"], an["k_max"]
    g = proc.expected_windows(k_max, burn, length, common=True)
    out = {f"w{j}": float(W.witness(g, j)) for j in an["offsets"]}
    b = proc.block_probs(range(burn, length - 1), 2).real
    out.update(
        {
            "g": g.tolist(),
            "p1": float(proc.marginals(length)[burn:].mean()),
            "lag1": C.lag1_from_blocks(b),
        }
    )
    return out


def tv_stats(tvs: list[D.TimeVaryingProcess], cfg: dict[str, Any]) -> dict[str, Any]:
    an = cfg["analysis"]
    length, burn, k_max = an["length"], an["burn"], an["k_max"]
    gs, p1s, bl = [], [], np.zeros(4)
    for tv in tvs:
        gs.append(tv.expected_windows(k_max, burn, common=True))
        p1s.append(tv.marginals()[burn:].mean())
        bl += np.real(tv.block_probs(range(burn, length - 1), 2))
    g = np.mean(gs, 0)
    out = {f"w{j}": float(W.witness(g, j)) for j in an["offsets"]}
    out.update(
        {"g": g.tolist(), "p1": float(np.mean(p1s)), "lag1": C.lag1_from_blocks(bl / len(tvs))}
    )
    return out


def classical_stats(model: C.InterventionModel, cfg: dict[str, Any], pt: np.ndarray):
    an = cfg["analysis"]
    out = {}
    for arm in C.ARMS:
        out[arm] = C.arm_stats(
            model,
            arm,
            an["length"],
            an["burn"],
            an["k_max"],
            an["offsets"],
            pt if arm == "A3" else None,
        )
    return out


class Quantum:
    """#27's analysis model: #20's pieces at the pilot's live calibration (``pieces_live.npz``)
    with the pilot's own fitted q75 decay (``refit_pilot.json``); D1 #23's model."""

    def __init__(self, cfg: dict[str, Any]):
        d = rdir(cfg, "data_dir")
        with np.load(d / "pieces_live.npz") as z:
            self.pieces = A.StepPieces.from_npz(z, "live")
            self.readout = float(z["readout_q75"])
        rf = json.loads((d / "refit_pilot.json").read_text())
        self.decay, self.decay_first = float(rf["decay"]), float(rf["decay_first"])
        self.hyp = A.Hypothesis(meas_decay=self.decay)
        self.cfg = cfg
        import phase7_hw as PH

        hw = PH.load_config(REPO_ROOT / "experiments/configs/phase7_hw_4c.toml")
        self.d1 = PH.d1_process(hw)
        self.bank = [A.process(A.step_superops("A3", self.pieces, self.hyp, p)) for p in A.PAULIS]

    def proc(self, arm: str) -> W.LinearProcess:
        if arm == "D1":
            return self.d1
        return A.process(A.step_superops(arm, self.pieces, self.hyp))

    def record(self) -> R.RecordModel:
        return R.RecordModel.build(self.pieces, self.readout, self.decay, self.decay_first)

    def a3(self, row: np.ndarray) -> D.TimeVaryingProcess:
        ops = np.stack([self.bank[p].ops for p in row])
        b = self.bank[0]
        return D.TimeVaryingProcess(b.init, ops, b.final)

    def nll(self, data: dict[str, C.ArmData]) -> dict[str, float]:
        """Total ``-log L`` per arm; A2 by its joint record."""
        out = {}
        for arm, a in data.items():
            if arm == "A2" and a.m is not None:
                out[arm] = float(self.record().nll(a.m, a.y).sum())
            elif arm == "A3":
                out[arm] = float(tv_nll(self.bank, a.paulis, a.y).sum())
            else:
                out[arm] = float(self.proc(arm).seq_nll(a.y).sum())
        return out

    def stats(self, pt: np.ndarray) -> dict[str, Any]:
        out = {arm: process_stats(self.proc(arm), self.cfg) for arm in C.ARMS if arm != "A3"}
        out["A3"] = tv_stats([self.a3(row) for row in pt], self.cfg)
        an = self.cfg["analysis"]
        out["A2"].update(self.record().pair_stats(an["length"], an["burn"]))
        out["A2"]["p_m1_step0"] = float(self.record().marginals(an["length"])["m"][0])
        out["A2"]["p_m1_step1"] = float(self.record().marginals(an["length"])["m"][1])
        return out


def tv_nll(bank: list[W.LinearProcess], paulis: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-shot ``-log P`` with shot n using ``bank[paulis[n, t]]`` at step t."""
    ops = np.stack([b.ops for b in bank])  # (4, 2, D, D)
    n, length = y.shape
    s = np.tile(bank[0].init, (n, 1))
    out = np.zeros(n)
    for t in range(length):
        s = np.einsum("nd,nde->ne", s, ops[paulis[:, t], y[:, t]])
        z = (s @ bank[0].final).real
        out -= np.log(z)
        s = s / z[:, None]
    return out


# -- comparison with the data ---------------------------------------------------------------------

PRIMARY = [  # the issue's list: what the classical model must reproduce within #27's CIs
    ("baseline", "w0"),
    ("baseline", "p1"),
    ("A2", "w0"),
    ("A1", "w0"),
    ("A3", "w0"),
    ("A2s_spec", "w0"),
    ("D1", "lag1"),
    ("A2", "p_m1"),
] + [("baseline", f"g{k}") for k in range(8)]


def compare(model: dict[str, Any], obs: dict[str, Any], cfg: dict[str, Any]) -> list[dict]:
    """One row per (arm, statistic): model, data, SE, z."""
    rows = []
    for arm, o in obs.items():
        m = model[arm]
        keys = [f"w{j}" for j in cfg["analysis"]["offsets"]] + ["p1", "lag1"]
        keys += [k for k in o if k.startswith("p_") and not k.endswith("_se") and k in m]
        for k in keys:
            rows.append(_row(arm, k, m[k], o[k], o[f"{k}_se"]))
        for i, (gm, go, gs) in enumerate(zip(m["g"], o["g"], o["g_se"], strict=True)):
            rows.append(_row(arm, f"g{i}", gm, go, gs))
    prim = {(a, k) for a, k in PRIMARY}
    for r in rows:
        r["primary"] = (r["arm"], r["stat"]) in prim
    return rows


def _row(arm: str, stat: str, model: float, data: float, se: float) -> dict[str, Any]:
    return {
        "arm": arm,
        "stat": stat,
        "model": float(model),
        "data": float(data),
        "se": float(se),
        "z": float((model - data) / se) if se > 0 else math.nan,
    }


def verdict(rows: list[dict], z_fit: float) -> dict[str, Any]:
    """Fit summary: max |z| and misses over the primary set and over every statistic; a
    Bonferroni level for the full set (many statistics, so some |z| > 1.96 by chance)."""
    out = {}
    for name, sel in (("primary", [r for r in rows if r["primary"]]), ("all", rows)):
        zs = np.array([abs(r["z"]) for r in sel])
        zb = W._norm_ppf(1 - 0.025 / len(sel))
        worst = max(sel, key=lambda r: abs(r["z"]))
        out[name] = {
            "n": len(sel),
            "max_abs_z": float(zs.max()),
            "worst": f"{worst['arm']} {worst['stat']}",
            "n_over_z_fit": int((zs > z_fit).sum()),
            "expected_over_z_fit": float(len(sel) * 0.05),
            "bonferroni_z": float(zb),
            "n_over_bonferroni": int((zs > zb).sum()),
            "misses": [
                f"{r['arm']} {r['stat']} (z = {r['z']:+.1f})" for r in sel if abs(r["z"]) > zb
            ],
        }
    out["fits_primary"] = out["primary"]["n_over_z_fit"] == 0
    out["fits_all_bonferroni"] = out["all"]["n_over_bonferroni"] == 0
    return out


# -- fits (parallel) ------------------------------------------------------------------------------

_DATA: dict[str, Any] = {}


def _init_worker(cfg_path: str) -> None:
    cfg = load_config(Path(cfg_path))
    shots = load_shots(cfg)
    pt = pauli_table(cfg)
    _DATA.update({"cfg": cfg, "shots": shots, "pt": pt, "maps": None})


def _subset(task: dict[str, Any]) -> dict[str, C.ArmData]:
    shots, pt = _DATA["shots"], _DATA["pt"]
    sel = task.get("select")
    if sel is None:
        return arm_data(shots, pt)
    if sel[0] == "fold":  # train on pubs of one parity
        keep = {a: (d["pub"] % 2) == sel[1] for a, d in shots.items()}
    elif sel[0] == "drop_baseline_jobs":
        keep = {"baseline": ~np.isin(shots["baseline"]["job"], sel[1])}
    return arm_data(shots, pt, keep)


def quantum_start(cfg: dict[str, Any], smooth: float) -> C.InterventionModel:
    q = Quantum(cfg)
    maps = C.effective_maps(q.pieces, q.readout)
    emb = embedding_scan(cfg, maps)
    u = C.klein_tetrahedron(emb["a"], emb["b"])
    p1 = float(q.d1.marginals(cfg["analysis"]["length"])[cfg["analysis"]["burn"] :].mean())
    return C.smooth(C.clipped_start(maps, u, p1), smooth)


def embedding_scan(cfg: dict[str, Any], maps: dict[str, np.ndarray]) -> dict[str, Any]:
    e = cfg["embedding"]
    best = (-math.inf, 0.0, 0.0)
    for a in np.linspace(*e["a_grid"][:2], int(e["a_grid"][2])):
        for b in np.linspace(*e["b_grid"][:2], int(e["b_grid"][2])):
            if abs(a) < 1e-9 or abs(b) < 1e-9:
                continue
            v = C.embedding_negativity(maps, C.klein_tetrahedron(a, b))
            if v > best[0]:
                best = (v, float(a), float(b))
    return {"max_min_entry": best[0], "a": best[1], "b": best[2], "exact_classical": best[0] >= 0}


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = _DATA["cfg"]
    f = cfg["fit"]
    st = C.Structure(**task["structure"])
    data = _subset(task)
    rng = np.random.default_rng(task["seed"])
    kp = None
    src = task["start"]
    if src == "quantum":
        model = quantum_start(cfg, f["smooth"])
        if st.d != 4:
            raise ValueError("the quantum start has 4 states")
    elif src == "random":
        model, kp = C.start_model(st, rng)
    else:  # a previous fit's model: split, or kappa from it
        prev = C.InterventionModel.from_dict(task["from"])
        if src == "split":
            model = C.smooth(C.split_state(prev, rng), f["smooth"])
        elif src == "kappa":
            model = C.smooth(prev, f["smooth"])
            diag = np.stack([np.diag(model.meas[m]) for m in (0, 1)], 1) + 1e-3
            kp = C.KappaParts(st.invasive, diag / diag.sum(1, keepdims=True), model.meas)
            model.meas = kp.meas()
        else:
            model = C.smooth(prev, f["smooth"])
    t0 = time.time()
    res = C.fit_em_model(
        data,
        st,
        model,
        kp,
        max_iters=f["max_iters"],
        tol=f["tol"],
        use_record=task.get("use_record", True),
    )
    out = {
        "label": task["label"],
        "group": task["group"],
        "structure": task["structure"],
        "start": src,
        "select": task.get("select"),
        "use_record": task.get("use_record", True),
        "nll": res.nll,
        "n_symbols": res.n_symbols,
        "iters": res.iters,
        "converged": res.converged,
        "seconds": time.time() - t0,
        "model": res.model.to_dict(),
        "n_params": C.n_params(st),
    }
    if task.get("eval_fold") is not None:  # held-out NLL on the other parity
        shots, pt = _DATA["shots"], _DATA["pt"]
        keep = {a: (d["pub"] % 2) == task["eval_fold"] for a, d in shots.items()}
        test = arm_data(shots, pt, keep)
        out["test_nll"] = {
            arm: float(res.model.nll(arm, a.y, a.m, a.paulis).sum()) for arm, a in test.items()
        }
        out["test_n_symbols"] = {arm: a.n_symbols for arm, a in test.items()}
    return out


def run_round(tasks: list[dict], cfg_path: str, workers: int, out: Path) -> list[dict]:
    res = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg_path,)) as ex:
        for r in ex.map(_run_task, tasks):
            log(
                out,
                f"  {r['label']}: nll/sym {r['nll'] / r['n_symbols']:.6f}, "
                f"{r['iters']} steps, {r['seconds']:.0f} s, converged {r['converged']}",
            )
            res.append(r)
    save_json(out / f"fits_{len(list(out.glob('fits_*.json')))}.json", {"fits": res})
    return res


def best_of(results: list[dict], group: str) -> dict | None:
    sel = [r for r in results if r["group"] == group]
    return min(sel, key=lambda r: r["nll"]) if sel else None


# -- main -----------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg_path = str(Path(args.config).resolve())
    cfg = load_config(Path(cfg_path))
    workers = args.workers or cfg["experiment"]["workers"]
    out = new_run_dir(cfg["experiment"]["name"])
    log(out, f"run dir {out}")
    meta = run_metadata(cfg, cfg["fit"]["seed"])
    f = cfg["fit"]
    shots = load_shots(cfg)
    pt = pauli_table(cfg)
    data = arm_data(shots, pt)

    log(out, "observed statistics")
    obs = observed(cfg, shots)
    log(out, "calibrated quantum model: NLL, statistics, embedding scan")
    q = Quantum(cfg)
    q_nll = q.nll(data)
    q_stats = q.stats(pt)
    maps = C.effective_maps(q.pieces, q.readout)
    emb = embedding_scan(cfg, maps)
    log(
        out,
        f"  Klein-tetrahedron scan: least negative entry {emb['max_min_entry']:.4f} at "
        f"(a, b) = ({emb['a']:.3f}, {emb['b']:.3f})",
    )

    seed = f["seed"]
    gen = lambda d: {"d": d, "kind": "general"}  # noqa: E731
    r1: list[dict] = []
    for d in f["dims"][:2]:
        for i in range(f["random_starts"]):
            r1.append(
                {
                    "label": f"general_d{d}_rand{i}",
                    "group": f"general_d{d}",
                    "structure": gen(d),
                    "start": "random",
                    "seed": seed + 100 * d + i,
                }
            )
    r1.append(
        {
            "label": "general_d4_quantum",
            "group": "general_d4",
            "structure": gen(4),
            "start": "quantum",
            "seed": seed,
        }
    )
    for dh in f["zpop_dh"]:
        for i in range(f["random_starts"]):
            r1.append(
                {
                    "label": f"zpop_dh{dh}_rand{i}",
                    "group": f"zpop_dh{dh}",
                    "structure": {"d": 2 * dh, "kind": "zpop", "dh": dh},
                    "start": "random",
                    "seed": seed + 1000 + 10 * dh + i,
                }
            )
    dc = cfg["cv"]["d"]
    for fold in (0, 1):
        r1.append(
            {
                "label": f"cv_train{fold}",
                "group": "cv",
                "structure": gen(dc),
                "start": "quantum",
                "seed": seed,
                "select": ["fold", fold],
                "eval_fold": 1 - fold,
            }
        )
    r1.append(
        {
            "label": "no_record_d4",
            "group": "no_record",
            "structure": gen(4),
            "start": "quantum",
            "seed": seed,
            "use_record": False,
        }
    )
    sens = cfg["sensitivity"]
    r1.append(
        {
            "label": "drop_baseline_job1_d4",
            "group": "drop_baseline_job1",
            "structure": gen(sens["d"]),
            "start": "quantum",
            "seed": seed,
            "select": ["drop_baseline_jobs", sens["drop_baseline_jobs"]],
        }
    )
    log(out, f"round 1: {len(r1)} fits")
    results = run_round(r1, cfg_path, workers, out)

    b4 = best_of(results, f"general_d{f['kappa_d']}")
    r2: list[dict] = []
    prev = best_of(results, f"general_d{f['dims'][1]}")
    d5 = f["dims"][2]
    r2.append(
        {
            "label": f"general_d{d5}_split",
            "group": f"general_d{d5}",
            "structure": gen(d5),
            "start": "split",
            "from": prev["model"],
            "seed": seed + 5,
        }
    )
    for i in range(f["random_starts"]):
        r2.append(
            {
                "label": f"general_d{d5}_rand{i}",
                "group": f"general_d{d5}",
                "structure": gen(d5),
                "start": "random",
                "seed": seed + 100 * d5 + i,
            }
        )
    for kap in f["kappa_grid"]:
        r2.append(
            {
                "label": f"kappa{kap:g}_d{f['kappa_d']}",
                "group": f"kappa{kap:g}",
                "structure": {"d": f["kappa_d"], "kind": "kappa", "invasive": kap},
                "start": "kappa",
                "from": b4["model"],
                "seed": seed + 7,
            }
        )
    log(out, f"round 2: {len(r2)} fits")
    results += run_round(r2, cfg_path, workers, out)

    d6 = f["dims"][3]
    prev = best_of(results, f"general_d{d5}")
    r3 = [
        {
            "label": f"general_d{d6}_split",
            "group": f"general_d{d6}",
            "structure": gen(d6),
            "start": "split",
            "from": prev["model"],
            "seed": seed + 6,
        },
        {
            "label": f"general_d{d6}_rand0",
            "group": f"general_d{d6}",
            "structure": gen(d6),
            "start": "random",
            "seed": seed + 100 * d6,
        },
    ]
    log(out, f"round 3: {len(r3)} fits")
    results += run_round(r3, cfg_path, workers, out)

    log(out, "statistics of the best fit of every class")
    summary = summarise(cfg, results, obs, q_nll, q_stats, data, pt, emb)
    summary["meta"] = meta
    save_json(out / "summary.json", summary)
    write_outputs(cfg, out, summary)
    log(out, "done")


def summarise(cfg, results, obs, q_nll, q_stats, data, pt, emb) -> dict[str, Any]:
    z_fit = cfg["analysis"]["z_fit"]
    q_total = sum(q_nll.values())
    n_sym = sum(a.n_symbols for a in data.values())
    classes = {}
    q_rows = compare(q_stats, obs, cfg)
    classes["quantum_calibrated"] = {
        "nll": q_total,
        "nll_per_symbol": q_total / n_sym,
        "nll_by_arm": q_nll,
        "n_params": 0,
        "verdict": verdict(q_rows, z_fit),
        "rows": q_rows,
    }
    groups = sorted({r["group"] for r in results} - {"cv"})
    for g in groups:
        b = best_of(results, g)
        model = C.InterventionModel.from_dict(b["model"])
        stats = classical_stats(model, cfg, pt)
        rows = compare(stats, obs, cfg)
        sel = data
        if b["select"] is not None:  # drop_baseline_job1: compare only what was fitted
            sel = None
        by_arm = {
            arm: float(model.nll(arm, a.y, a.m if b["use_record"] else None, a.paulis).sum())
            for arm, a in data.items()
        }
        total = sum(by_arm.values())
        k = b["n_params"]
        entry = {
            "label": b["label"],
            "structure": b["structure"],
            "start": b["start"],
            "starts": {r["label"]: r["nll"] / r["n_symbols"] for r in results if r["group"] == g},
            "converged": b["converged"],
            "iters": b["iters"],
            "nll_fit": b["nll"],
            "nll_all_shots": total,
            "nll_by_arm": by_arm,
            "nll_per_symbol": total / n_sym,
            "n_params": k,
            "delta_nll_vs_quantum": total - q_total,
            "delta_nll_per_symbol_vs_quantum": (total - q_total) / n_sym,
            "aic_vs_quantum": 2 * k + 2 * (total - q_total),
            "bic_vs_quantum": k * math.log(n_sym) + 2 * (total - q_total),
            "verdict": verdict(rows, z_fit),
            "rows": rows,
            "model": b["model"],
            "stats": {a: {kk: v for kk, v in s.items() if kk != "g"} for a, s in stats.items()},
            "use_record": b["use_record"],
            "select": b["select"],
        }
        if sel is None:
            entry["note"] = "fitted without the baseline's job1 shots; NLL here is on all shots"
        if g == "no_record":
            entry["note"] = (
                "fitted to A2's y only (no q75 record); its record statistics are "
                "the fitted M_m's, unconstrained by data"
            )
        classes[g] = entry
    cv = [r for r in results if r["group"] == "cv"]
    cv_out = []
    qm = Quantum(cfg)
    shots_all = load_shots(cfg)
    for r in cv:
        fold = r["select"][1]
        keep = {a: (d["pub"] % 2) == (1 - fold) for a, d in shots_all.items()}
        test = arm_data(shots_all, pt, keep)
        qn = qm.nll(test)
        cn = r["test_nll"]
        ns = r["test_n_symbols"]
        cv_out.append(
            {
                "train_parity": fold,
                "test_parity": 1 - fold,
                "classical_nll_per_symbol": sum(cn.values()) / sum(ns.values()),
                "quantum_nll_per_symbol": sum(qn.values()) / sum(ns.values()),
                "delta_per_symbol": (sum(cn.values()) - sum(qn.values())) / sum(ns.values()),
                "delta_by_arm_per_symbol": {a: (cn[a] - qn[a]) / ns[a] for a in cn},
            }
        )
    return {
        "embedding": emb,
        "observed": obs,
        "classes": classes,
        "cv": cv_out,
        "n_symbols": n_sym,
        "primary_set": [f"{a} {k}" for a, k in PRIMARY],
    }


def write_outputs(cfg: dict[str, Any], out: Path, summary: dict[str, Any]) -> None:
    sd = rdir(cfg, "summary_dir")
    sd.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, c in summary["classes"].items():
        v = c["verdict"]
        rows.append(
            {
                "class": name,
                "n_params": c["n_params"],
                "nll_per_symbol": round(c["nll_per_symbol"], 7),
                "delta_nll_vs_quantum": round(c.get("delta_nll_vs_quantum", 0.0), 1),
                "bic_vs_quantum": round(c.get("bic_vs_quantum", 0.0), 1),
                "primary_max_abs_z": round(v["primary"]["max_abs_z"], 2),
                "primary_worst": v["primary"]["worst"],
                "primary_n_over_1.96": v["primary"]["n_over_z_fit"],
                "all_max_abs_z": round(v["all"]["max_abs_z"], 2),
                "all_worst": v["all"]["worst"],
                "all_n_over_bonferroni": v["all"]["n_over_bonferroni"],
                "fits_primary": v["fits_primary"],
                "fits_all_bonferroni": v["fits_all_bonferroni"],
            }
        )
    _csv(sd / "classes.csv", rows)
    stat_rows = []
    for name, c in summary["classes"].items():
        for r in c["rows"]:
            stat_rows.append(
                {
                    "class": name,
                    **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()},
                }
            )
    _csv(sd / "statistics.csv", stat_rows)
    slim = {k: v for k, v in summary.items() if k != "classes"}
    slim["classes"] = {
        n: {k: v for k, v in c.items() if k != "rows"} for n, c in summary["classes"].items()
    }
    save_json(sd / "summary.json", slim)


def _csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
