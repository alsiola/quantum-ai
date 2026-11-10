"""Phase 8 Stage 2c (issue #35): the incommensurate-phase (1 rad) tier against Z_n registers.

A part of ``phase8_coherence.py`` (``--part incommensurate``; local only: no QPU, no IBM
service, no submit path):

1. **Predictions.** The tier's seven arms (``[incommensurate.arms]``: Rz(1), Rz(-1),
   INV(1, -1), COMP(1, 1) against Rz(2), Rz(1)^3 against Rz(3); physical echoes ``r<k>`` /
   ``rm<k>`` fenced and timed as ``e<k>``) under the fixed and fitted quantum models, their
   composition contrasts, and the response at the nearest Z_n angles of 1 rad.
2. **Power.** #34's section 4 pipeline unchanged (parity split, the four starts at D = 4, 6,
   1,500 steps, the quantum refit, block bootstrap by job and 8-shot segment, Holm) on the
   18 + 7 arms at the main tier's shots, on data from both quantum generators, plus
   ``qmem.register``'s Z_n x n_pop classes (``[incommensurate] classes``): every echo its
   nearest shift; the 1 rad echoes relaxed to free maps; e(1) the nearest free permutation
   with e(-1) its inverse. The held-out deficit against truth, refit and *other*.
3. **Size.** Data from the best-fitting register (an exact one: INV(1, -1) = identity) run
   through the same register fit and refit; the false-rejection rate with a Clopper-Pearson
   interval.
4. **Options**, priced with ``qmem.followup``'s cost model: the tier at the plan's shots, more
   1 rad shots, a second incommensurate angle.

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
import phase8_coherence_null as NL

from qmem import classical as C
from qmem import coherence as Q
from qmem import followup as F
from qmem import register as RG
from qmem.records import save_json

SALT = {"klein": 1, "random": 2, "octahedron": 3, "split": 4, "cube": 5, "prism": 6}
VARIANT_ID = {"nearest": 1, "relaxed": 2, "permutation": 3}


def merge_arms(cfg: dict[str, Any]) -> dict[str, Any]:
    """``cfg["arms"]`` extended (in place) by the tier's arms, appended after the 18: the
    data of the 18 are drawn first, so they are #34's register run's shots exactly."""
    new = cfg["incommensurate"]["arms"]
    if not set(new) <= set(cfg["arms"]):
        cfg["arms"] = {**cfg["arms"], **new}
    cfg["null"]["source_ids"].setdefault("reg", cfg["incommensurate"]["null_source_id"])
    return cfg


def tier_arms(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["incommensurate"]["arms"])


# -- predictions ------------------------------------------------------------------------------

CONTRASTS = [("INV1", "E00"), ("COMP11", "RAD2"), ("CUBE1", "RAD3"), ("RAD1", "P1")]


def predictions(cfg: dict[str, Any], ctx) -> dict[str, Any]:
    """Each tier arm's statistics under both quantum models; the composition contrasts; the
    constant-echo arm's response at 1 rad against the nearest Z_n shift angles."""
    s = DS.settings(cfg)
    gens = DS.generators(ctx, cfg, None)
    arms = Q.parse_arms(merge_arms(cfg)["arms"])
    rows, contrasts, resolution = [], [], []
    for g, p in gens.items():
        qm = ctx.model(p)
        st = {a: Q.model_arm_stats(qm, arms[a], s) for a in [*tier_arms(cfg), "E00", "P1"]}
        for a in tier_arms(cfg):
            rows.append(DS.arm_stats_row(qm, arms[a], s) | {"generator": g})
        for a, b in CONTRASTS:
            contrasts.append(
                {"generator": g, "arm": a, "against": b}
                | {f"d_{k}": st[a][k] - st[b][k] for k in ("p1", "lag1", "w0")}
            )
        one = C.ArmSpec("x", (("p:1.0",),))
        ref = Q.model_arm_stats(qm, one, s)
        for n in sorted({c["n_phase"] for c in cfg["incommensurate"]["classes"]} | {12, 32}):
            k = RG.echo_shift("r1", n)
            phi = 2 * math.pi * k / n
            alt = Q.model_arm_stats(qm, C.ArmSpec("x", ((f"p:{phi!r}",),)), s)
            resolution.append(
                {
                    "generator": g,
                    "n_phase": n,
                    "shift": k,
                    "angle": phi,
                    "error_rad": RG.shift_error("r1", n),
                }
                | {f"d_{key}": alt[key] - ref[key] for key in ("p1", "lag1", "w0")}
            )
    return {"arms": rows, "contrasts": contrasts, "resolution": resolution}


# -- tasks ------------------------------------------------------------------------------------

_N: dict[str, Any] = {}


def _init(cfg_path: str) -> None:
    NL._init(cfg_path)
    merge_arms(NL._N["cfg"])  # the same dict as DS._W["cfg"]
    _N["cfg"], _N["ctx"] = NL._N["cfg"], NL._N["ctx"]


def source_id(cfg: dict[str, Any], src: str) -> int:
    return cfg["null"]["source_ids"][src]


def _model_of(run: Path, label: str) -> C.ProgramModel:
    return C.ProgramModel.from_dict(json.loads((run / f"{label}.json").read_text())["model"])


def _warm(m: C.ProgramModel, names, eps: float) -> C.ProgramModel:
    d = m.d
    for e in names:
        m.maps[e] = (1 - eps) * m.maps[e] + eps / d
    return m


def register_start(task, arms, train, qm, rng, run: Path) -> tuple[C.ProgramModel, list[str]]:
    """The start and the frozen echoes of a register fit (see ``[incommensurate] classes``)."""
    ic = _N["cfg"]["incommensurate"]
    n, n_pop, var = task["n_phase"], task["n_pop"], task["variant"]
    els, lab = Q.arm_elements(arms)
    free = [e for e in ic["free_rad"] if e in els]
    if var == "nearest":
        if task["start"] == "prism":
            m = RG.register_start(Q.bloch_maps(qm, els, lab), n_pop, els, lab, n_phase=n)
        else:
            m = RG.random_register(n_pop, els, rng, lab, n_phase=n)
        return m, RG.echo_names(m)
    if var == "relaxed":
        m = _warm(_model_of(run, task["from"]), free, ic["warm_eps"])
        return m, RG.echo_names(m, free)
    if var == "permutation":
        relaxed = _model_of(run, task["from"])
        cands = {
            "hungarian": RG.best_permutation(relaxed.maps["r1"], relaxed.maps["rm1"]),
            "nearest": (np.arange(relaxed.d) // n) * n
            + (np.arange(relaxed.d) % n + RG.echo_shift("r1", n)) % n,
        }
        best, best_ll = None, -math.inf
        for name, sigma in cands.items():
            m = C.ProgramModel.from_dict(relaxed.to_dict())
            p = RG.permutation_matrix(sigma)
            m.maps["r1"], m.maps["rm1"] = p, p.T.copy()
            ll = C.program_loglik(m, arms, train)
            if ll > best_ll:
                best, best_ll, task["sigma_from"] = m, ll, name
        rest = [e for e in free if e not in ("r1", "rm1")]
        return best, RG.echo_names(best, rest)
    raise ValueError(var)


def task_fit(task) -> dict[str, Any]:
    cfg, ctx = _N["cfg"], _N["ctx"]
    run = Path(task["run"])
    src, rep = task["source"], task["rep"]
    arms, train, test = NL.plan_data(run, cfg, ctx, src, rep)
    gens = DS.generators(ctx, cfg, None)
    qm = ctx.model(gens["fixed"])  # the starts embed the pre-registered (fixed) prediction
    els, lab = Q.arm_elements(arms)
    var = VARIANT_ID.get(task.get("variant", ""), 0)
    rng = np.random.default_rng(
        [35, cfg["null"]["seed"], source_id(cfg, src), rep, task["d"], SALT[task["start"]], var]
    )
    frozen: list[str] = []
    if task["family"] == "register":
        start, frozen = register_start(task, arms, train, qm, rng, run)
    elif task["start"] == "split":
        start = C.smooth_program(C.split_program_state(_model_of(run, task["from"]), rng), 0.01)
        while start.d < task["d"]:
            start = C.smooth_program(C.split_program_state(start, rng), 0.01)
    elif task["start"] == "cube":
        start = Q.polytope_start(Q.bloch_maps(qm, els, lab), Q.cube(0.9, 1.0), els, lab)
    else:
        start = DS.start_model(task, arms, qm, rng)
    fit, iters = NL._em(run, task["label"], arms, train, start, cfg["power"]["max_iters"], frozen)
    m = fit.model
    for k in frozen:
        if not np.array_equal(m.maps[k], start.maps[k]):
            raise RuntimeError(f"EM moved the frozen {k}")
    if task.get("variant") == "permutation" and not (
        RG.is_permutation(m.maps["r1"])
        and np.array_equal(m.maps["r1"] @ m.maps["rm1"], np.eye(m.d))
    ):
        raise RuntimeError("e(1) is not a permutation with e(-1) its inverse")
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
        "frozen": frozen,
        "sigma_from": task.get("sigma_from"),
        "model": m.to_dict(),
    }
    if task.get("stats"):
        out["stat_rows"] = NL.stat_rows(m, arms, test, DS.settings(cfg))
    return out


KINDS = {"qref": NL.task_qref, "fit": task_fit}


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
    if not todo:
        return done
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


def class_name(c: dict[str, Any]) -> str:
    v = {"nearest": "nearest shift", "relaxed": "1 rad maps free", "permutation": "e(1) free perm"}
    return f"Z{c['n_phase']} x {c['n_pop']} (d = {c['n_phase'] * c['n_pop']}; {v[c['variant']]})"


def class_label(prefix: str, c: dict[str, Any], start: str) -> str:
    return f"{prefix}_z{c['n_phase']}x{c['n_pop']}_{c['variant']}_{start}"


def register_tasks(cfg, src: str, rep: int, classes, starts) -> list[dict[str, Any]]:
    """The register fits of one replicate, longest first; ``relaxed`` warm-starts from the
    same (n, n_pop)'s nearest fit, ``permutation`` from the relaxed one."""
    p = f"{src}_r{rep}"
    out = []
    for c in sorted(classes, key=lambda c: -c["n_phase"] * c["n_pop"]):
        d = c["n_phase"] * c["n_pop"]
        base = {"kind": "fit", "family": "register", "source": src, "rep": rep, "d": d}
        base |= {k: c[k] for k in ("n_phase", "n_pop", "variant")} | {"stats": rep == 0}
        if c["variant"] == "nearest":
            for s in starts:
                out.append(base | {"label": class_label(p, c, s), "start": s})
        else:
            prev = "nearest" if c["variant"] == "relaxed" else "relaxed"
            dep = class_label(p, c | {"variant": prev}, "prism")
            out.append(
                base
                | {"label": class_label(p, c, "prism"), "start": "prism", "from": dep}
                | {"deps": [dep]}
            )
    return out


def free_tasks(cfg, src: str) -> list[dict[str, Any]]:
    ic = cfg["incommensurate"]
    tasks = NL.pipeline_tasks(src, 0, stats=True)
    p = f"{src}_r0"
    for s in ic["free_d8_starts"]:
        t = {"kind": "fit", "family": "free", "source": src, "rep": 0, "d": 8, "stats": True}
        tasks.append(t | {"label": f"{p}_d8_{s}", "start": s})
    return tasks


def power_tasks(cfg) -> list[dict[str, Any]]:
    ic = cfg["incommensurate"]
    reg, free = [], []
    for src in ic["sources"]:
        reg += register_tasks(cfg, src, 0, ic["classes"], ic["register_starts"])
        free += free_tasks(cfg, src)
    # the long register fits first (and their chains), then #34's pipeline
    return reg + free


# -- summaries --------------------------------------------------------------------------------


def families(cfg, src: str, rep: int, done: dict) -> dict[str, list[str]]:
    """Each class's fits (credited with the best training NLL): a class contains the classes
    it generalises (free d <= D: every smaller D; free permutation: the nearest shift;
    relaxed: both)."""
    ic = cfg["incommensurate"]
    p = f"{src}_r{rep}"
    fams: dict[str, list[str]] = {}
    if rep == 0 and src != "reg":
        fams["free d<=4"] = [f"{p}_d4_klein", f"{p}_d4_random"]
        fams["free d<=6"] = fams["free d<=4"] + [f"{p}_d6_octahedron", f"{p}_d6_split"]
        fams["free d<=8"] = fams["free d<=6"] + [f"{p}_d8_{s}" for s in ic["free_d8_starts"]]
    for c in ic["classes"]:
        labs = []
        for v in ("nearest", "permutation", "relaxed"):
            if v == "nearest":
                labs += [class_label(p, c | {"variant": v}, s) for s in ic["register_starts"]]
            else:
                labs.append(class_label(p, c | {"variant": v}, "prism"))
            if v == c["variant"]:
                break
        labs = [lab for lab in labs if lab in done]
        if labs:
            fams[class_name(c)] = labs
    return fams


def class_rows(cfg, run: Path, done: dict, src: str, rep: int, evs) -> list[dict[str, Any]]:
    pw = cfg["power"]
    za, zb = pw["z_alpha"], pw["z_beta"]
    arms = list(Q.parse_arms(cfg["arms"]))
    new = tier_arms(cfg)
    old = [a for a in arms if a not in new]
    q = NL._load(run, f"{src}_r{rep}_qref")
    rows = []
    for name, labs in families(cfg, src, rep, done).items():
        b = NL.best_fit(done, labs)
        f = done[b]
        row: dict[str, Any] = {
            "source": src,
            "rep": rep,
            "class": name,
            "best": b,
            "n_params": f["n_params"],
            "iters": f["iters"],
            "converged": f["converged"],
            "last_gain_per_symbol": f.get("last_gain_per_symbol"),
            "train_nll_per_shot": f["train_nll_per_shot"],
        }
        for ev in evs:
            t = NL.deficit(cfg, run, b, q, ev, arms)
            row[f"{ev}_heldout"] = t["mean"]
            row[f"{ev}_sd_per_shot"] = t["sd_per_shot"]
            row[f"{ev}_se_block"] = t["se_block"]
            row[f"{ev}_z_block"] = t["z"]
            row[f"{ev}_p"] = t["p"]
            row[f"{ev}_shots_80pct_total"] = Q.shots_for_power(t["mean"], t["sd_per_shot"], za, zb)
            row[f"{ev}_heldout_18"] = NL.deficit(cfg, run, b, q, ev, old)["mean"]
            row[f"{ev}_heldout_tier"] = NL.deficit(cfg, run, b, q, ev, new)["mean"]
            if rep == 0:
                row[f"{ev}_in_sample"] = NL.deficit(cfg, run, b, q, ev, arms, "train")["mean"]
        sr = f.get("stat_rows")
        if sr:
            zs = np.array([abs(x["z"]) for x in sr])
            zb_ = float(np.sqrt(2) * DS._erfinv(1 - 0.05 / len(sr)))
            worst = max(sr, key=lambda x: abs(x["z"]))
            row |= {
                "stats_n": len(sr),
                "stats_over_bonferroni": int((zs > zb_).sum()),
                "stats_max_abs_z": float(zs.max()),
                "stats_worst": f"{worst['arm']} {worst['stat']}",
            }
        if rep == 0:
            c = NL._load(run, b)
            row["by_arm"] = {
                ev: {
                    a: {
                        "mean": float((c[f"test|{a}"] - q[f"{ev}|test|{a}"]).mean()),
                        "var": float((c[f"test|{a}"] - q[f"{ev}|test|{a}"]).var(ddof=1)),
                    }
                    for a in arms
                }
                for ev in evs
            }
        rows.append(row)
    return rows


def weighted_option(row: dict, ev: str, shots: dict[str, float], za: float, zb: float) -> dict:
    """The pooled per-shot deficit and its shot-level z when arm ``a`` gets ``shots[a]`` test
    shots (the per-arm held-out means and variances of ``row``); the scale factor on every
    arm's shots for 80% power."""
    by = row["by_arm"][ev]
    n = sum(shots.values())
    mean = sum(shots[a] * by[a]["mean"] for a in shots) / n
    var = sum(shots[a] * by[a]["var"] for a in shots) / n**2
    z = mean / math.sqrt(var)
    scale = ((za + zb) / z) ** 2 if z > 0 else math.inf
    return {"deficit": mean, "z_shot": z, "scale_for_80pct": scale}


def cost_model(cfg) -> F.CostModel:
    bc = cfg["budget"]
    return F.CostModel(
        bc["sub_job_overhead_s"], bc["rep_delay_s"], bc["margin"], bc["billing_round_s"]
    )


def tier_circuit_s(cfg, arm: C.ArmSpec) -> float:
    """As ``DS.budget``: echoes inside q59's reset window add no time unless they overrun it."""
    ds = cfg["design"]
    n_x = max(2 * sum(Q.element_phase(e) is not None for e in prog) for prog in arm.programs)
    add = max(0.0, n_x * ds["x_s"] - ds["slot_window_s"])
    return cfg["budget"]["baseline_circuit_s"] + cfg["analysis"]["length"] * add


def options(cfg, rows: list[dict]) -> dict[str, Any]:
    """The tier's cost at the plan's shots, more 1 rad shots, a second incommensurate angle:
    each option's added main-tier upper bound, and its projected deficit against *other* for
    each class (per-arm held-out means and variances, re-weighted; shot-level SE)."""
    ic, pw = cfg["incommensurate"], cfg["power"]
    cm = cost_model(cfg)
    arms = Q.parse_arms(cfg["arms"])
    new = tier_arms(cfg)
    old = [a for a in arms if a not in new]
    n_plan = cfg["plan"]["shots_per_arm"]
    durs = {a: tier_circuit_s(cfg, arms[a]) for a in new}

    def added(n_new: int) -> dict[str, float]:
        est = sum(cm.shot_seconds(n_new, durs[a]) for a in new)
        return {"estimate_s": est, "upper_s": (1 + cm.margin) * est}

    out: dict[str, Any] = {"cost_model": cm.__dict__, "circuit_s": durs, "options": []}
    for n_new in [n_plan, *ic["extra_shots"]]:
        name = "tier at the plan's shots" if n_new == n_plan else f"tier at {n_new} shots per arm"
        opt = {"name": name, "arms": len(new), "shots_per_arm": n_new, "shots": len(new) * n_new}
        opt |= added(n_new)
        proj = []
        for r in rows:
            if "by_arm" not in r:
                continue
            shots = {a: n_plan / 2 for a in old} | {a: n_new / 2 for a in new}  # test halves
            for ev in ("refit", "other"):
                w = weighted_option(r, ev, shots, pw["z_alpha"], pw["z_beta"])
                proj.append({"source": r["source"], "class": r["class"], "evaluator": ev} | w)
        opt["projection"] = proj
        out["options"].append(opt)
    # a second incommensurate angle: the same seven arms at a second angle (not simulated)
    second = {"name": "second incommensurate angle (7 arms, not simulated)", "arms": len(new)}
    second |= {"shots_per_arm": n_plan, "shots": len(new) * n_plan} | added(n_plan)
    out["options"].append(second)
    own = cm.job([F.Pub(0, i, a, 0, n_plan) for i, a in enumerate(new)], durs)
    out["own_tier"] = {"estimate_s": own["estimate_s"], "upper_s": own["upper_s"]}
    return out


# -- the size check ---------------------------------------------------------------------------


def choose_generator(cfg, run: Path, done: dict) -> dict[str, Any]:
    """The best-fitting *exact* register (nearest shift or free permutation: INV(1, -1) =
    identity) on the fixed generator's data: the smallest held-out deficit against the
    refit."""
    rows = class_rows(cfg, run, done, "fixed", 0, ("refit",))
    rows = [r for r in rows if r["class"].startswith("Z") and "maps free" not in r["class"]]
    best = min(rows, key=lambda r: r["refit_heldout"])
    lab = best["best"]
    f = done[lab]
    gen = {
        "class": best["class"],
        "fit": lab,
        "n_phase": f["n_phase"],
        "n_pop": f["n_pop"],
        "variant": f["variant"],
        "refit_heldout": best["refit_heldout"],
        "model": f["model"],
    }
    save_json(run / "gen_reg.json", gen)
    return gen


def null_tasks(cfg, gen: dict[str, Any]) -> list[dict[str, Any]]:
    ic = cfg["incommensurate"]
    c = {k: gen[k] for k in ("n_phase", "n_pop", "variant")}
    chain = [c | {"variant": "nearest"}]
    if c["variant"] == "permutation":
        chain += [c | {"variant": "relaxed"}, c]
    tasks = []
    for rep in range(ic["null_replicates"]):
        tasks.append({"kind": "qref", "label": f"reg_r{rep}_qref", "source": "reg", "rep": rep})
        tasks += register_tasks(cfg, "reg", rep, chain, ic["register_starts"])
    return tasks


def summarise_null(cfg, run: Path, done: dict, gen: dict[str, Any]) -> dict[str, Any]:
    ic = cfg["incommensurate"]
    alpha = cfg["null"]["alpha"]
    rows = []
    for rep in range(ic["null_replicates"]):
        if f"reg_r{rep}_qref" not in done:
            continue
        for r in class_rows(cfg, run, done, "reg", rep, ("refit", "fixed")):
            if r["class"] == gen["class"]:
                rows.append(r | {"reject": r["refit_p"] <= alpha})
    k = sum(r["reject"] for r in rows)
    lo, hi = NL.clopper_pearson(k, len(rows)) if rows else (math.nan, math.nan)
    z = np.array([r["refit_z_block"] for r in rows])
    return {
        "generator": {k_: v for k_, v in gen.items() if k_ != "model"},
        "alpha": alpha,
        "replicates": [{k_: v for k_, v in r.items() if k_ != "by_arm"} for r in rows],
        "n": len(rows),
        "false_rejections": int(k),
        "rate": k / len(rows) if rows else math.nan,
        "rate_cp95": [lo, hi],
        "z_mean": float(z.mean()) if len(z) else math.nan,
        "z_sd": float(z.std(ddof=1)) if len(z) > 1 else math.nan,
        "z_max": float(z.max()) if len(z) else math.nan,
    }


# -- the part ---------------------------------------------------------------------------------


def part_incommensurate(cfg, cfg_path: str, out: Path, workers: int, meta) -> dict[str, Any]:
    merge_arms(cfg)
    ctx = PC.Ctx(cfg)
    pred = predictions(cfg, ctx)
    save_json(out / "predictions.json", pred)
    write(cfg, {"predictions": pred, "meta": meta})
    tasks = power_tasks(cfg)
    for src in cfg["incommensurate"]["sources"]:
        tasks.insert(0, {"kind": "qref", "label": f"{src}_r0_qref", "source": src, "rep": 0})
    done = schedule(tasks, cfg_path, workers, out)
    summary = summarise(cfg, out, done, pred, meta)
    write(cfg, summary)
    gen = choose_generator(cfg, out, done)
    PC.log(out, f"size check: generator {gen['class']} ({gen['fit']})")
    done |= schedule(null_tasks(cfg, gen), cfg_path, workers, out)
    summary["null"] = summarise_null(cfg, out, done, gen)
    write(cfg, summary)
    return summary


def summarise(cfg, run: Path, done: dict, pred: dict, meta) -> dict[str, Any]:
    ic = cfg["incommensurate"]
    rows, pipe = [], {}
    for src in ic["sources"]:
        rows += class_rows(cfg, run, done, src, 0, ("truth", "refit", "other", "fixed"))
        pipe[src] = NL.pipeline_result(cfg, run, done, src, 0, ("refit", "truth", "other"))
    fits = [
        {k: v for k, v in r.items() if k not in ("model", "stat_rows", "run", "deps")}
        for r in done.values()
        if r.get("kind") == "fit"
    ]
    return {
        "meta": meta,
        "arms": list(Q.parse_arms(cfg["arms"])),
        "tier_arms": tier_arms(cfg),
        "predictions": pred,
        "classes": rows,
        "pipeline": pipe,
        "options": options(cfg, rows),
        "fits": fits,
    }


def write(cfg, summary: dict[str, Any]) -> None:
    sd = PC.sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    pred = summary["predictions"]
    PC.write_csv(
        sd / "incommensurate_predictions.csv",
        [
            PC.jround({k: v for k, v in r.items() if not isinstance(v, list | dict)})
            for r in pred["arms"]
        ],
    )
    if "classes" in summary:
        PC.write_csv(
            sd / "incommensurate_classes.csv",
            [
                PC.jround({k: v for k, v in r.items() if not isinstance(v, dict)})
                for r in summary["classes"]
            ],
        )
    if summary.get("null", {}).get("replicates"):
        PC.write_csv(
            sd / "incommensurate_null.csv",
            [PC.jround(r) for r in summary["null"]["replicates"]],
        )
    save_json(sd / "incommensurate.json", PC.jround(summary, 9))
