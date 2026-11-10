"""Phase 8 Stage 2b (issue #34): null calibration and the phase-register adversary.

Parts of ``phase8_coherence.py`` (local only: no QPU, no IBM service):

* ``--part null``: the pre-registered primary statistic of ``docs/phase8-coherence.md`` §4
  run on data simulated from two *classical* d = 6 generators at the plan's shot counts
  (8,000 shots per arm in two jobs, parity split): (a) #32's best d = 6 ``ProgramModel`` fit
  to the fixed quantum model's simulated data (the adversary closest to the quantum
  prediction), (b) #30's Stage 4c d = 6 fit with the new elements' maps fitted (its own maps
  frozen) to the same quantum data. The false-rejection rate of "d <= 6" (Holm over D = 4, 6)
  with its Clopper-Pearson interval and the distribution of the statistic.
* ``--part register``: the same pipeline on data from the two quantum generators (the power
  study at the plan's shot counts), plus the cyclic phase-register model of
  ``qmem.register`` (Z_8 x n_pop, n_pop = 2, 3, 4) and a free d = 8 model.

Every fit is a resumable task (``--from-run <dir>``); at most ``[experiment] workers``.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import phase8_coherence as PC
import phase8_coherence_design as DS

from qmem import classical as C
from qmem import coherence as Q
from qmem import register as RG
from qmem.ancilla import nelder_mead
from qmem.records import REPO_ROOT, save_json

FROZEN_B = ["init", "pre", "post", "meas", "spec", "reset", "X", "Y", "Z", "D1"]

# -- sources ----------------------------------------------------------------------------------


def split_of(cfg: dict[str, Any]) -> Q.PlanSplit:
    nc = cfg["null"]
    return Q.PlanSplit(nc["per_job"], nc["jobs"], nc["segment"])


def simulate_program(m: C.ProgramModel, arm: C.ArmSpec, n: int, length: int, rng) -> C.ProgramData:
    """Shots of an arm from a classical program model (random arms: the bits per shot)."""
    choice = arm.choices(n, length, rng)
    rec = bool({m.labelled_in(p) for p in arm.programs} - {None})
    bank = m.bank(arm, record=rec)
    sym = C.bank_sample(m.init, bank, np.ones(m.d), choice, rng)
    if rec:
        return C.ProgramData((sym % 2).astype(np.uint8), choice, (sym // 2).astype(np.uint8))
    return C.ProgramData(sym.astype(np.uint8), choice)


def source_model(run: Path, cfg, ctx, src: str):
    gens = DS.generators(ctx, cfg, None)
    if src in gens:
        return ctx.model(gens[src])
    path = run / f"gen_{src}.json"
    return C.ProgramModel.from_dict(json.loads(path.read_text())["model"])


def plan_data(run: Path, cfg, ctx, src: str, rep: int):
    """(arms, train, test) for one replicate: ``split.n`` shots per arm, in shot order."""
    arms = Q.parse_arms(cfg["arms"])
    sp = split_of(cfg)
    model = source_model(run, cfg, ctx, src)
    sid = cfg["null"]["source_ids"][src]
    rng = np.random.default_rng([cfg["null"]["seed"], sid, rep])
    length = cfg["analysis"]["length"]
    train, test = {}, {}
    for a, arm in arms.items():
        if isinstance(model, Q.QuantumModel):
            d = DS.simulate(model, arm, sp.n, length, rng)
        else:
            d = simulate_program(model, arm, sp.n, length, rng)
        train[a], test[a] = d.subset(sp.train()), d.subset(sp.test())
    return arms, train, test


# -- tasks ----------------------------------------------------------------------------------

_N: dict[str, Any] = {}


def _init(cfg_path: str) -> None:
    DS._init(cfg_path, None)
    _N["cfg"], _N["ctx"] = DS._W["cfg"], DS._W["ctx"]


def _em(run: Path, label: str, arms, data, start: C.ProgramModel, max_iters: int, frozen=()):
    ck = run / f"ckpt_{label}.json"
    done = 0
    if ck.exists():
        c = json.loads(ck.read_text())
        start, done = C.ProgramModel.from_dict(c["model"]), int(c["iters"])
    cyc = {"n": 0}

    def cb(it, m, nps):
        cyc["n"] += 1
        if cyc["n"] % 50 == 0:
            ck.write_text(json.dumps({"model": m.to_dict(), "iters": done + it, "nll": nps}))

    fit = C.fit_program_model(
        arms, data, start, max(max_iters - done, 2), 1e-10, callback=cb, frozen=frozen
    )
    return fit, done + fit.iters


def task_gen_a(task) -> dict[str, Any]:
    """#32's fixed_d6_octahedron, refitted exactly as #32 ran it (its data, start, steps)."""
    run = Path(task["run"])
    r = DS._power_task(
        {
            "label": "fixed_d6_octahedron",
            "generator": "fixed",
            "d": 6,
            "start": "octahedron",
            "run": str(run),
        }
    )
    return {"model": r["model"], "iters": r["iters"], "source": "#32 power: fixed_d6_octahedron"}


def task_gen_b(task) -> dict[str, Any]:
    """#30's general_d6 (init, pre, post, meas frozen) with e0..e7 and ff.. maps fitted by
    EM to #32's fixed-generator training shots (from near-identity maps)."""
    cfg, ctx = _N["cfg"], _N["ctx"]
    run = Path(task["run"])
    s30 = json.loads((REPO_ROOT / cfg["experiment"]["classical_summary"]).read_text())
    m30 = C.InterventionModel.from_dict(s30["classes"]["general_d6"]["model"])
    pm = C.ProgramModel.from_intervention(m30)
    gens = DS.generators(ctx, cfg, None)
    arms, qm, ds = DS.datasets(cfg, ctx, "fixed", gens["fixed"])
    els, lab = Q.arm_elements(arms)
    d, eps = pm.d, cfg["null"]["gen_b_eps"]
    for e in els:
        pm.maps[e] = (1 - eps) * np.eye(d) + eps / d
    for e in lab:
        if e not in pm.labelled:
            raise ValueError(f"#30's model has no labelled {e!r}")
    fit, iters = _em(run, "gen_b", arms, ds["train"], pm, cfg["null"]["gen_max_iters"], FROZEN_B)
    s = DS.settings(cfg)
    match = []
    for a, arm in arms.items():
        mc, mq = Q.model_arm_stats(fit.model, arm, s), Q.model_arm_stats(qm, arm, s)
        match.append(
            {"arm": a}
            | {f"{k}_{w}": v[k] for w, v in (("b", mc), ("q", mq)) for k in ("p1", "lag1", "w0")}
        )
    return {
        "model": fit.model.to_dict(),
        "iters": iters,
        "converged": fit.converged,
        "train_nll_per_shot": fit.nll / sum(len(x.y) for x in ds["train"].values()),
        "match": match,
        "source": "#30 general_d6 (frozen) + new maps by EM on #32's fixed-generator shots",
    }


def task_qref(task) -> dict[str, Any]:
    """Quantum evaluators' per-shot NLL on a replicate: ``refit`` (REFIT from the calibrated
    model, #32's Nelder-Mead budget), the fixed and fitted predictions; quantum sources also
    ``truth`` and ``other``."""
    cfg, ctx = _N["cfg"], _N["ctx"]
    run = Path(task["run"])
    src, rep = task["source"], task["rep"]
    arms, train, test = plan_data(run, cfg, ctx, src, rep)
    gens = DS.generators(ctx, cfg, None)
    evals = {"fixed": gens["fixed"], "fitted": gens["fitted"]}
    if src in gens:
        evals["truth"] = gens[src]
        evals["other"] = gens["fitted" if src == "fixed" else "fixed"]
    base = gens["fixed"]

    def f(x):
        m = ctx.model(Q.from_vector(x, DS.REFIT, base))
        v = sum(float(m.nll(arms[a], d).sum()) for a, d in train.items())
        return v if math.isfinite(v) else 1e30

    x0 = Q.to_vector(base, DS.REFIT, ctx.quantum.pieces)
    x, fx, n_ev = nelder_mead(f, x0, step=0.4, max_evals=cfg["power"]["refit_evals"], ftol=1e-4)
    evals["refit"] = Q.from_vector(x, DS.REFIT, base)
    arrays = {}
    for ev, p in evals.items():
        m = ctx.model(p)
        for split, data in (("train", train), ("test", test)):
            for a, d in data.items():
                arrays[f"{ev}|{split}|{a}"] = m.nll(arms[a], d)
    np.savez(run / f"{task['label']}.npz", **arrays)
    return {"refit_params": evals["refit"].to_dict(), "refit_nll_train": fx, "evals": n_ev}


def stat_rows(m: C.ProgramModel, arms, test, s: Q.StatSettings) -> list[dict[str, Any]]:
    rows = []
    for a, arm in arms.items():
        obs = Q.data_stats(test[a].y, s)
        mod = Q.model_arm_stats(m, arm, s)
        keys = [f"w{j}" for j in s.offsets] + ["p1", "lag1"]
        pairs = [(k, mod[k], obs[k], obs[f"{k}_se"]) for k in keys]
        pairs += [
            (f"g{i}", gm, go, gs)
            for i, (gm, go, gs) in enumerate(zip(mod["g"], obs["g"], obs["g_se"], strict=True))
        ]
        for k, mv, ov, se in pairs:
            rows.append(
                {
                    "arm": a,
                    "stat": k,
                    "model": mv,
                    "data": ov,
                    "se": se,
                    "z": (mv - ov) / se if se > 0 else 0.0,
                }
            )
    return rows


def task_fit(task) -> dict[str, Any]:
    cfg, ctx = _N["cfg"], _N["ctx"]
    run = Path(task["run"])
    src, rep = task["source"], task["rep"]
    arms, train, test = plan_data(run, cfg, ctx, src, rep)
    gens = DS.generators(ctx, cfg, None)
    qm = ctx.model(gens["fixed"])  # the starts embed the pre-registered (fixed) prediction
    els, lab = Q.arm_elements(arms)
    salt = {"klein": 1, "random": 2, "octahedron": 3, "split": 4, "cube": 5, "prism": 6}
    rng = np.random.default_rng(
        [cfg["null"]["seed"], cfg["null"]["source_ids"][src], rep, task["d"], salt[task["start"]]]
    )
    frozen: list[str] = []
    if task["family"] == "register":
        n_pop = task["n_pop"]
        if task["start"] == "prism":
            start = RG.register_start(Q.bloch_maps(qm, els, lab), n_pop, els, lab)
        else:
            start = RG.random_register(n_pop, els, rng, lab)
        frozen = RG.echo_names(start)
    elif task["start"] == "split":
        prev = C.ProgramModel.from_dict(
            json.loads((run / f"{task['from']}.json").read_text())["model"]
        )
        start = C.smooth_program(C.split_program_state(prev, rng), 0.01)
        while start.d < task["d"]:
            start = C.smooth_program(C.split_program_state(start, rng), 0.01)
    elif task["start"] == "cube":
        start = Q.polytope_start(Q.bloch_maps(qm, els, lab), Q.cube(0.9, 1.0), els, lab)
    else:
        start = DS.start_model(task, arms, qm, rng)
    fit, iters = _em(run, task["label"], arms, train, start, cfg["power"]["max_iters"], frozen)
    m = fit.model
    if frozen and not RG.is_register(m, task["n_pop"]):
        raise RuntimeError("EM moved a frozen permutation")
    arrays = {}
    for split, data in (("train", train), ("test", test)):
        for a, d in data.items():
            arrays[f"{split}|{a}"] = m.nll(arms[a], d)
    np.savez(run / f"{task['label']}.npz", **arrays)
    n_tr = sum(len(x.y) for x in train.values())
    out = {
        "iters": iters,
        "converged": fit.converged,
        "last_gain_per_symbol": (fit.history[-2] - fit.history[-1])
        if len(fit.history) > 1
        else None,
        "train_nll_per_shot": fit.nll / n_tr,
        "n_params": m.n_params(frozen),
        "model": m.to_dict(),
    }
    if task.get("stats"):
        out["stat_rows"] = stat_rows(m, arms, test, DS.settings(cfg))
    return out


KINDS = {"gen_a": task_gen_a, "gen_b": task_gen_b, "qref": task_qref, "fit": task_fit}


def _run(task: dict[str, Any]) -> dict[str, Any]:
    run = Path(task["run"])
    path = run / f"{task['label']}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    out = {k: v for k, v in task.items() if k != "run"} | KINDS[task["kind"]](task)
    out["seconds"] = time.time() - t0
    save_json(path, out)
    return out


def schedule(tasks: list[dict[str, Any]], cfg_path: str, workers: int, out: Path) -> dict:
    """Run tasks in list order as their ``deps`` (labels) complete; resumable."""
    for t in tasks:
        t["run"] = str(out)
    done: dict[str, dict] = {}
    for t in tasks:
        p = out / f"{t['label']}.json"
        if p.exists():
            done[t["label"]] = json.loads(p.read_text())
    todo = [t for t in tasks if t["label"] not in done]
    PC.log(out, f"{len(tasks)} tasks, {len(done)} done, {workers} workers")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg_path,)) as ex:
        running: dict[Any, str] = {}
        while todo or running:
            for t in list(todo):
                if len(running) >= workers:
                    break
                if all(d in done for d in t.get("deps", ())):
                    running[ex.submit(_run, t)] = t["label"]
                    todo.remove(t)
            if not running:
                raise RuntimeError(f"unsatisfiable deps: {[t['label'] for t in todo]}")
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                lab = running.pop(f)
                r = f.result()
                done[lab] = r
                PC.log(
                    out, f"  done {lab} ({r.get('seconds', 0):.0f} s, {r.get('iters', '-')} steps)"
                )
    return done


def pipeline_tasks(src: str, rep: int, stats: bool = False) -> list[dict[str, Any]]:
    """§4's pipeline on one replicate: the quantum refit and the four starts at D = 4, 6."""
    p = f"{src}_r{rep}"
    deps = [f"gen_{src}"] if src in ("a", "b") else []
    base = {
        "kind": "fit",
        "family": "free",
        "source": src,
        "rep": rep,
        "deps": deps,
        "stats": stats,
    }
    return [
        {"kind": "qref", "label": f"{p}_qref", "source": src, "rep": rep, "deps": deps},
        base | {"label": f"{p}_d4_klein", "d": 4, "start": "klein"},
        base | {"label": f"{p}_d6_octahedron", "d": 6, "start": "octahedron"},
        base | {"label": f"{p}_d4_random", "d": 4, "start": "random"},
        base
        | {
            "label": f"{p}_d6_split",
            "d": 6,
            "start": "split",
            "from": f"{p}_d4_klein",
            "deps": [*deps, f"{p}_d4_klein"],
        },
    ]


def register_tasks(cfg, src: str, rep: int = 0) -> list[dict[str, Any]]:
    rc = cfg["register"]
    p = f"{src}_r{rep}"
    out = []
    for n_pop in rc["n_pop"]:
        for start in rc["starts"]:
            out.append(
                {
                    "kind": "fit",
                    "family": "register",
                    "label": f"{p}_reg{n_pop}_{start}",
                    "source": src,
                    "rep": rep,
                    "d": RG.N_PHASE * n_pop,
                    "n_pop": n_pop,
                    "start": start,
                    "stats": True,
                }
            )
    base = {"kind": "fit", "family": "free", "source": src, "rep": rep, "d": 8, "stats": True}
    out.append(base | {"label": f"{p}_d8_cube", "start": "cube"})
    out.append(
        base
        | {
            "label": f"{p}_d8_split",
            "start": "split",
            "from": f"{p}_d6_octahedron",
            "deps": [f"{p}_d6_octahedron"],
        }
    )
    return out


# -- summaries --------------------------------------------------------------------------------


def _load(run: Path, label: str) -> dict[str, np.ndarray]:
    with np.load(run / f"{label}.npz") as z:
        return {k: z[k] for k in z.files}


def deficit(cfg, run: Path, fit_label: str, q: dict, ev: str, arms: list[str], split: str = "test"):
    """Pooled per-shot deficit (classical minus quantum evaluator ``ev``) with the block SE."""
    sp = split_of(cfg)
    c = _load(run, fit_label)
    keep = sp.test() if split == "test" else sp.train()
    blocks = sp.blocks(keep)
    jobs = sp.job()[keep]
    diffs = [c[f"{split}|{a}"] - q[f"{ev}|{split}|{a}"] for a in arms]
    seed = cfg["null"]["seed"] + 7
    r = Q.block_bootstrap(
        diffs, [blocks] * len(arms), [jobs] * len(arms), cfg["null"]["n_boot"], seed
    )
    r["z"] = r["mean"] / r["se_block"] if r["se_block"] > 0 else math.nan
    r["p"] = 1 - Q.norm_cdf(r["z"])
    return r


def best_fit(done: dict[str, dict], labels: list[str]) -> str:
    cand = [lab for lab in labels if lab in done]
    return min(cand, key=lambda lab: done[lab]["train_nll_per_shot"])


def pipeline_result(cfg, run: Path, done: dict, src: str, rep: int, evs=("refit", "fixed")):
    """§4's decision on one replicate: per D the best start (monotone in D), its held-out
    deficit against each quantum evaluator, z with the block SE, Holm over D."""
    p = f"{src}_r{rep}"
    arms = list(Q.parse_arms(cfg["arms"]))
    q = _load(run, f"{p}_qref")
    alpha = cfg["null"]["alpha"]
    fam = {4: [f"{p}_d4_klein", f"{p}_d4_random"]}
    fam[6] = fam[4] + [f"{p}_d6_octahedron", f"{p}_d6_split"]
    out: dict[str, Any] = {"source": src, "rep": rep}
    for ev in evs:
        res = {}
        for D, labs in fam.items():
            b = best_fit(done, labs)
            r = deficit(cfg, run, b, q, ev, arms)
            res[f"d{D}"] = {"fit": b, "train_nll_per_shot": done[b]["train_nll_per_shot"]} | r
        rej = Q.holm({k: v["p"] for k, v in res.items()}, alpha)
        for k in res:
            res[k]["reject_holm"] = rej[k]
            res[k]["reject_unadjusted"] = res[k]["p"] <= alpha
        out[ev] = res
    return out


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for ``k`` of ``n`` (bisection on the tails; no scipy)."""

    def cdf(x: int, p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))

    a = (1 - conf) / 2

    def root(f) -> float:  # f increasing in p, f(0) < 0 < f(1)
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else root(lambda p: (1 - cdf(k - 1, p)) - a)
    upper = 1.0 if k == n else root(lambda p: a - cdf(k, p))
    return lower, upper


# -- parts ------------------------------------------------------------------------------------


def null_tasks(cfg) -> list[dict[str, Any]]:
    nc = cfg["null"]
    tasks = [
        {"kind": "gen_a", "label": "gen_a"},
        {"kind": "gen_b", "label": "gen_b"},
    ]
    for rep in range(nc["replicates"]):  # interleave the sources (a partial run is balanced)
        for src in nc["sources"]:
            tasks += pipeline_tasks(src, rep)
    return tasks


def part_null(cfg, cfg_path: str, out: Path, workers: int, meta) -> dict[str, Any]:
    done = schedule(null_tasks(cfg), cfg_path, workers, out)
    summary = summarise_null(cfg, out, done)
    summary["meta"] = meta
    write_null(cfg, summary)
    return summary


def summarise_null(cfg, run: Path, done: dict) -> dict[str, Any]:
    nc = cfg["null"]
    alpha = nc["alpha"]
    za = cfg["power"]["z_alpha"]  # one-sided alpha = 0.05
    reps = []
    for src in nc["sources"]:
        for rep in range(nc["replicates"]):
            p = f"{src}_r{rep}"
            need = [
                f"{p}_qref",
                f"{p}_d4_klein",
                f"{p}_d4_random",
                f"{p}_d6_octahedron",
                f"{p}_d6_split",
            ]
            if all(n in done for n in need):
                reps.append(pipeline_result(cfg, run, done, src, rep))
    out: dict[str, Any] = {"alpha": alpha, "replicates": reps, "by_source": {}}
    for src in [*nc["sources"], "all"]:
        rs = [r for r in reps if src == "all" or r["source"] == src]
        if not rs:
            continue
        e: dict[str, Any] = {"n": len(rs)}
        for ev in ("refit", "fixed"):
            z6 = np.array([r[ev]["d6"]["z"] for r in rs])
            m6 = np.array([r[ev]["d6"]["mean"] for r in rs])
            k6 = sum(r[ev]["d6"]["reject_holm"] for r in rs)
            k4 = sum(r[ev]["d4"]["reject_holm"] for r in rs)
            lo, hi = clopper_pearson(k6, len(rs))
            mu, sd = float(z6.mean()), float(z6.std(ddof=1)) if len(z6) > 1 else math.nan
            e[ev] = {
                "false_rejections_d6_holm": int(k6),
                "rate_d6": k6 / len(rs),
                "rate_d6_cp95": [lo, hi],
                "rejections_d4_holm": int(k4),
                "z6_mean": mu,
                "z6_sd": sd,
                "z6_max": float(z6.max()),
                "deficit6_mean": float(m6.mean()),
                "deficit6_max": float(m6.max()),
                # P(z > z_alpha) if z6 is normal with the replicates' mean and SD
                "normal_rejection_prob_d6": 1 - Q.norm_cdf((za - mu) / sd) if sd > 0 else math.nan,
                "sds_from_threshold": (za - mu) / sd if sd > 0 else math.nan,
            }
        out["by_source"][src] = e
    gens = {}
    for g in ("gen_a", "gen_b"):
        if g in done:
            gens[g] = {k: v for k, v in done[g].items() if k not in ("model", "run")}
    out["generators"] = gens
    return out


def write_null(cfg, summary) -> None:
    sd = PC.sdir(cfg)
    rows = []
    for r in summary["replicates"]:
        for ev in ("refit", "fixed"):
            for D in ("d4", "d6"):
                x = r[ev][D]
                rows.append(
                    {"source": r["source"], "rep": r["rep"], "evaluator": ev, "D": D}
                    | {
                        k: x[k]
                        for k in (
                            "fit",
                            "mean",
                            "se_block",
                            "se_shot",
                            "z",
                            "p",
                            "reject_holm",
                            "train_nll_per_shot",
                        )
                    }
                )
    PC.write_csv(sd / "null_replicates.csv", [PC.jround(r) for r in rows])
    if "gen_b" in summary["generators"]:
        PC.write_csv(
            sd / "null_generator_b_match.csv",
            [PC.jround(r) for r in summary["generators"]["gen_b"]["match"]],
        )
    slim = {k: v for k, v in summary.items() if k != "replicates"}
    save_json(sd / "null.json", PC.jround(slim, 9))


def part_register(cfg, cfg_path: str, out: Path, workers: int, meta) -> dict[str, Any]:
    tasks = []
    for src in cfg["register"]["sources"]:
        tasks += pipeline_tasks(src, 0, stats=True)
        tasks += register_tasks(cfg, src, 0)
    done = schedule(tasks, cfg_path, workers, out)
    summary = summarise_register(cfg, out, done)
    summary["meta"] = meta
    write_register(cfg, summary)
    return summary


def summarise_register(cfg, run: Path, done: dict) -> dict[str, Any]:
    pw = cfg["power"]
    za, zb = pw["z_alpha"], pw["z_beta"]
    arms = list(Q.parse_arms(cfg["arms"]))
    rc = cfg["register"]
    out: dict[str, Any] = {"classes": [], "pipeline": {}}
    for src in rc["sources"]:
        p = f"{src}_r0"
        out["pipeline"][src] = pipeline_result(
            cfg, run, done, src, 0, ("refit", "truth", "other", "fixed")
        )
        q = _load(run, f"{p}_qref")
        fams = {
            "free d<=4": [f"{p}_d4_klein", f"{p}_d4_random"],
            "free d<=6": [f"{p}_d4_klein", f"{p}_d4_random", f"{p}_d6_octahedron", f"{p}_d6_split"],
        }
        fams["free d<=8"] = fams["free d<=6"] + [f"{p}_d8_cube", f"{p}_d8_split"]
        reg: list[str] = []
        for n_pop in rc["n_pop"]:
            reg = reg + [f"{p}_reg{n_pop}_{s}" for s in rc["starts"]]
            fams[f"register Z8 x {n_pop} (d = {8 * n_pop})"] = list(reg)
        for name, labs in fams.items():
            if not all(lab in done for lab in labs):
                continue
            b = best_fit(done, labs)
            f = done[b]
            row: dict[str, Any] = {
                "source": src,
                "class": name,
                "best": b,
                "n_params": f["n_params"],
                "iters": f["iters"],
                "converged": f["converged"],
                "train_nll_per_shot": f["train_nll_per_shot"],
            }
            for ev in ("truth", "refit", "other"):
                t = deficit(cfg, run, b, q, ev, arms)
                row[f"{ev}_heldout"] = t["mean"]
                row[f"{ev}_in_sample"] = deficit(cfg, run, b, q, ev, arms, "train")["mean"]
                row[f"{ev}_sd_per_shot"] = t["sd_per_shot"]
                row[f"{ev}_z_block"] = t["z"]
                row[f"{ev}_shots_80pct_total"] = Q.shots_for_power(
                    t["mean"], t["sd_per_shot"], za, zb
                )
            sr = f.get("stat_rows")
            if sr:
                zs = np.array([abs(x["z"]) for x in sr])
                zb_ = float(np.sqrt(2) * DS._erfinv(1 - 0.05 / len(sr)))
                worst = max(sr, key=lambda x: abs(x["z"]))
                row |= {
                    "stats_n": len(sr),
                    "stats_over_bonferroni": int((zs > zb_).sum()),
                    "stats_over_1.96": int((zs > 1.96).sum()),
                    "stats_max_abs_z": float(zs.max()),
                    "stats_worst": f"{worst['arm']} {worst['stat']}",
                }
                c = _load(run, b)
                row["by_arm_heldout_vs_refit"] = {
                    a: float((c[f"test|{a}"] - q[f"refit|test|{a}"]).mean()) for a in arms
                }
            out["classes"].append(row)
    out["fits"] = [
        {k: v for k, v in r.items() if k not in ("model", "stat_rows", "run", "deps")}
        for lab, r in done.items()
        if r.get("kind") == "fit"
    ]
    return out


def write_register(cfg, summary) -> None:
    sd = PC.sdir(cfg)
    PC.write_csv(
        sd / "register_classes.csv",
        [
            PC.jround({k: v for k, v in r.items() if not isinstance(v, dict)})
            for r in summary["classes"]
        ],
    )
    save_json(sd / "register.json", PC.jround(summary, 9))
