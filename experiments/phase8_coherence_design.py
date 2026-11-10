"""Phase 8 Stage 2 (issue #32): the design and the power study (``phase8_coherence.py``).

``part_design``: the instrument's fixed predictions (calibrated and fitted quantum models) for
every arm and over a continuous phase grid; the equivalent-dephasing comparison; q75
population and coherence benchmarks; Knee-style controls; #21's and #22's alternatives; the
budget. ``part_power``: simulated data from both quantum models, classical program models
fitted at d = 4 and 6 (held-out shots and held-out arms), power and shots. ``print_plan``:
the unsubmitted hardware plan. Numpy only; no QPU, no IBM service.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import phase8_coherence as PC

from qmem import classical as C
from qmem import coherence as Q
from qmem import followup as F
from qmem.ancilla import nelder_mead
from qmem.records import REPO_ROOT, save_json

# -- shared ---------------------------------------------------------------------------------------


def settings(cfg: dict[str, Any]) -> Q.StatSettings:
    an = cfg["analysis"]
    return Q.StatSettings(an["length"], an["burn"], an["k_max"], tuple(an["offsets"]))


def ff_idle_s(cfg: dict[str, Any]) -> float:
    """q75's extra idle per feed-forward element: the ancillas' readout and the conditional
    latency, minus the part that fits in the q59 reset window the slot already runs in."""
    ds = cfg["design"]
    return max(0.0, ds["ff_measure_s"] + ds["ff_latency_s"] + 4 * ds["x_s"] - ds["slot_window_s"])


def fitted_params(cfg: dict[str, Any], review: Path | None = None) -> Q.NoiseParams:
    """The fitted quantum model of the review part (its run directory, else the summary)."""
    if review is not None and (review / "task_qfit_all.json").exists():
        p = json.loads((review / "task_qfit_all.json").read_text())["params"]
    else:
        p = json.loads((PC.sdir(cfg) / "review.json").read_text())["quantum_fitted"]["params"]
    return Q.NoiseParams.from_dict(p)


def generators(ctx: PC.Ctx, cfg: dict[str, Any], review: Path | None) -> dict[str, Q.NoiseParams]:
    """Design generators: the calibrated (#27) and fitted quantum models, with the design's
    feed-forward idle and one measurement decay for every step."""
    ff = ff_idle_s(cfg)
    cal = replace(ctx.calibrated(), meas_decay_first=None, ff_idle_s=ff)
    out = {"fixed": cal}
    try:
        out["fitted"] = replace(fitted_params(cfg, review), meas_decay_first=None, ff_idle_s=ff)
    except FileNotFoundError:
        pass
    return out


def alternatives(cfg: dict[str, Any], base: Q.NoiseParams) -> dict[str, Q.NoiseParams]:
    al = cfg["alternatives"]
    e0, e1 = al["reset_asym"]
    return {
        "null": base,
        "q59_reset_asym_21": replace(base, reset_e0=e0, reset_e1=e1),
        "q75_detuning_21": replace(base, detune=base.detune + al["q75_detuning"]),
        "stark_22": replace(base, stark=(0.0, al["stark_rad"])),
        "x_amplitude_error": replace(base, over=al["over"]),
        "ancilla_zz_phase": replace(base, ff_phase=(al["ff_phase"], al["ff_phase"])),
    }


def biased(arm: C.ArmSpec, bias: float) -> C.ArmSpec:
    """A random arm whose ancilla bits are 1 with probability 1/2 + ``bias`` each."""
    if arm.schedule != "random":
        return arm
    q = 0.5 + bias
    p = [(1 - q) * (1 - q), (1 - q) * q, q * (1 - q), q * q]
    return replace(arm, probs=tuple(p))


def arm_stats_row(model, arm: C.ArmSpec, s: Q.StatSettings) -> dict[str, Any]:
    st = Q.model_arm_stats(model, arm, s)
    return {k: v for k, v in st.items() if k != "g"} | {f"g{i}": v for i, v in enumerate(st["g"])}


# -- part: design ---------------------------------------------------------------------------------


def part_design(
    cfg: dict[str, Any],
    out: Path,
    review: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = PC.Ctx(cfg)
    s = settings(cfg)
    arms = Q.parse_arms(cfg["arms"])
    gens = generators(ctx, cfg, review)
    sd = PC.sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ff_idle_s": ff_idle_s(cfg), "generators": {}, "meta": meta}

    # 1. fixed predictions of every arm, and the continuous phase curve
    rows, curve = [], []
    for g, p in gens.items():
        summary["generators"][g] = p.to_dict()
        qm = ctx.model(p)
        for name, arm in arms.items():
            rows.append({"generator": g, "arm": name} | arm_stats_row(qm, arm, s))
        n = cfg["design"]["phi_grid"]
        for i in range(n + 1):
            phi = 2 * math.pi * i / n
            arm = C.ArmSpec("phi", ((f"p:{phi!r}",),))
            curve.append({"generator": g, "phi": phi} | arm_stats_row(qm, arm, s))
    PC.write_csv(sd / "predictions.csv", [PC.jround(r) for r in rows])
    PC.write_csv(sd / "phase_curve.csv", [PC.jround(r) for r in curve])
    summary["predictions"] = rows
    summary["phase_curve_extrema"] = curve_extrema(curve)
    PC.log(out, "design: predictions and phase curves")

    # 2. equivalent dephasings (iii): A2, RS, R2, R4 against each other
    summary["equivalence"] = equivalence(ctx, cfg, gens, arms, s, out)
    PC.log(out, "design: equivalent dephasings")

    # 3. benchmarks on q75 alone (population invariance, coherence suppression)
    summary["benchmarks"] = benchmarks(ctx, cfg)
    PC.write_csv(sd / "benchmarks.csv", [PC.jround(r) for r in summary["benchmarks"]["rows"]])
    PC.log(out, "design: benchmarks")

    # 4. Knee-style controls
    summary["knee"] = knee(ctx, cfg, gens, out)
    PC.log(out, "design: Knee controls")

    # 5. alternatives
    alt_rows = []
    for g, p in gens.items():
        for h, hp in alternatives(cfg, p).items():
            qm = ctx.model(hp)
            for name, arm in arms.items():
                alt_rows.append(
                    {"generator": g, "hypothesis": h, "arm": name} | arm_stats_row(qm, arm, s)
                )
        qm = ctx.model(p)
        for name, arm in arms.items():
            if arm.schedule == "random":
                b = biased(arm, cfg["design"]["ancilla_bias"])
                alt_rows.append(
                    {"generator": g, "hypothesis": "ancilla_bias", "arm": name}
                    | arm_stats_row(qm, b, s)
                )
    PC.write_csv(sd / "alternatives.csv", [PC.jround(r) for r in alt_rows])
    summary["alternatives"] = alt_rows
    PC.log(out, "design: alternatives")

    # 6. budget
    summary["budget"] = budget(cfg, arms)
    PC.write_csv(sd / "budget_arms.csv", [PC.jround(r) for r in summary["budget"]["arms"]])
    save_json(
        sd / "design.json",
        PC.jround(
            {k: v for k, v in summary.items() if k not in ("predictions", "alternatives")}, 9
        ),
    )
    return summary


def curve_extrema(curve: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for g in sorted({r["generator"] for r in curve}):
        rs = [r for r in curve if r["generator"] == g]
        e = {}
        for k in ("w0", "w1", "p1", "lag1"):
            v = np.array([r[k] for r in rs])
            ph = np.array([r["phi"] for r in rs])
            e[k] = {
                "min": float(v.min()),
                "phi_min": float(ph[v.argmin()]),
                "max": float(v.max()),
                "phi_max": float(ph[v.argmax()]),
                "at_0": float(v[0]),
                "at_2pi": float(v[-1]),
                "sign_changes": int(np.sum(np.diff(np.sign(v[:-1])) != 0)),
            }
        out[g] = e
    return out


def equivalence(ctx, cfg, gens, arms, s, out) -> dict[str, Any]:
    """Predicted differences between the dephasing implementations, and the shots for a TOST
    at a margin of 10% of the E00 sham's witness (per-shot SDs from simulated shots)."""
    res: dict[str, Any] = {}
    names = [a for a in ("A2", "RS", "R2", "R4", "E00") if a in arms]
    rng = np.random.default_rng(cfg["power"]["seed"] + 99)
    for g, p in gens.items():
        qm = ctx.model(p)
        st = {a: Q.model_arm_stats(qm, arms[a], s) for a in names}
        sims = {a: simulate(qm, arms[a], 4000, s.length, rng) for a in names}
        sd = {}
        for a in names:
            ds = Q.data_stats(sims[a].y, s)
            sd[a] = {k: ds[f"{k}_se"] * math.sqrt(ds["n"]) for k in ("w0", "p1", "lag1")}
        margin_w0 = 0.1 * abs(st["E00"]["w0"]) if "E00" in st else math.nan
        pairs = {}
        for a, b in (("R2", "R4"), ("RS", "E00"), ("A2", "R2"), ("A2", "R4")):
            if a not in st or b not in st:
                continue
            row = {}
            for k in ("w0", "w1", "p1", "lag1"):
                row[f"d_{k}"] = st[a][k] - st[b][k]
            for k, margin in (("w0", margin_w0), ("p1", 0.01), ("lag1", 0.01)):
                sdk = math.hypot(sd[a][k], sd[b][k])
                delta = margin - abs(row[f"d_{k}"])
                row[f"tost_margin_{k}"] = margin
                row[f"tost_shots_per_arm_{k}"] = (
                    Q.shots_for_power(delta, sdk, 1.6448536269514722, 1.2815515655446004)
                    if delta > 0
                    else math.inf
                )
            pairs[f"{a}-{b}"] = row
        res[g] = {
            "stats": {a: {k: v for k, v in st[a].items() if k != "g"} for a in names},
            "per_shot_sd": sd,
            "pairs": pairs,
        }
    return res


def benchmarks(ctx: PC.Ctx, cfg: dict[str, Any]) -> dict[str, Any]:
    """q75 alone: <X>, <Y>, <Z> after n interventions (calibrated T1, T2, p1; A2 with the
    pilot's decay), and the population change against the matched sham."""
    bc = cfg["benchmark"]
    pc = ctx.quantum.pieces
    b = Q.Benchmark(ctx.t1_q75, ctx.t2_q75, pc.p1_q75, bc["spacing_s"], ctx.quantum.readout)
    rows = []
    for name in bc["interventions"]:
        for st in bc["states"]:
            for n in bc["reps"]:
                ev = Q.benchmark(name, st, n, b, ctx.quantum.decay)
                sham = Q.benchmark("RS" if name in ("R2", "R4", "RS") else "e0", st, n, b)
                rows.append(
                    {"intervention": name, "state": st, "reps": n}
                    | ev
                    | {
                        "dp1_vs_sham": ev["p1"] - sham["p1"],
                        "coherence_vs_sham": ev["coherence"] - sham["coherence"],
                    }
                )
    # shots to see a population change of 0.01 at 5 sigma, and its binomial SE at 10k shots
    se10k = math.sqrt(0.25 / 10000)
    return {
        "rows": rows,
        "t1": ctx.t1_q75,
        "t2": ctx.t2_q75,
        "p1": pc.p1_q75,
        "spacing_s": bc["spacing_s"],
        "population_margin": 0.01,
        "binomial_se_10k_shots": se10k,
        "shots_5sigma_for_0.01": (5 * 0.5 / 0.01) ** 2 * 2,
    }


def knee(ctx: PC.Ctx, cfg: dict[str, Any], gens, out: Path) -> dict[str, Any]:
    kc = cfg["knee"]
    res = {}
    rng = np.random.default_rng(cfg["power"]["seed"] + 7)
    for g, p in gens.items():
        qm = ctx.model(p)
        d = Q.knee_disturbances(qm, kc["interventions"], kc["sham"], kc["size"])
        probs = d.pop("_probs")
        # power of the inequality D_+ - max(D_0, D_1) > 0 by multinomial replicates
        pw = {}
        for name in kc["interventions"]:
            for n in (1000, 3000, 10000):
                ex = []
                for _ in range(300):
                    est = {}
                    for st in ("0", "1", "+"):
                        a = rng.multinomial(n, probs[f"{name}|{st}"]) / n
                        b = rng.multinomial(n, probs[f"sham|{st}"]) / n
                        est[st] = Q.tvd(a, b)
                    ex.append(est["+"] - max(est["0"], est["1"]))
                ex = np.array(ex)
                pw[f"{name}_n{n}"] = {
                    "mean_excess": float(ex.mean()),
                    "sd_excess": float(ex.std(ddof=1)),
                    "power_5sigma": float(np.mean(ex > 5 * ex.std(ddof=1))),
                }
        res[g] = {"disturbance": d, "power": pw}
    return res


def budget(cfg: dict[str, Any], arms: dict[str, C.ArmSpec]) -> dict[str, Any]:
    """Per-arm circuit time and QPU seconds per 1000 shots with ``qmem.followup``'s model."""
    bc, ds = cfg["budget"], cfg["design"]
    cm = F.CostModel(
        bc["sub_job_overhead_s"], bc["rep_delay_s"], bc["margin"], bc["billing_round_s"]
    )
    ff = ff_idle_s(cfg)
    out = []
    for name, arm in arms.items():
        n_ff = max(sum(1 for e in prog if e.startswith("ff")) for prog in arm.programs)
        n_x = max(2 * sum(1 for e in prog if e.startswith("e")) for prog in arm.programs)
        meas = any("meas" in prog for prog in arm.programs)
        add = n_ff * ff + max(0.0, n_x * ds["x_s"] - ds["slot_window_s"])
        dur = bc["baseline_circuit_s"] + cfg["analysis"]["length"] * add
        if meas:
            dur += 2.28e-6  # Phase 7: A2 adds one measurement time to the whole circuit
        out.append(
            {
                "arm": name,
                "circuit_s": dur,
                "per_shot_s": cm.shot_seconds(1, dur),
                "qpu_s_per_1000_shots": cm.shot_seconds(1000, dur),
            }
        )
    return {"arms": out, "cost_model": cm.__dict__, "ff_idle_s": ff}


# -- simulation -----------------------------------------------------------------------------------


def simulate(qm: Q.QuantumModel, arm: C.ArmSpec, n: int, length: int, rng) -> C.ProgramData:
    """Shots of an arm from a quantum model (random arms: the bits drawn per shot and step)."""
    choice = arm.choices(n, length, rng)
    rec = any(qm.is_labelled(e) for prog in arm.programs for e in prog)
    bank = qm.bank(arm, record=rec)
    sym = C.bank_sample(qm.vec0, bank, qm.trace, choice, rng)
    if rec:
        return C.ProgramData((sym % 2).astype(np.uint8), choice, (sym // 2).astype(np.uint8))
    return C.ProgramData(sym.astype(np.uint8), choice)


# -- part: power ----------------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init(cfg_path: str, review: str | None) -> None:
    cfg = PC.load_config(Path(cfg_path))
    _W["cfg"] = cfg
    _W["ctx"] = PC.Ctx(cfg)
    _W["review"] = Path(review) if review else None


def datasets(cfg, ctx, gname: str, p: Q.NoiseParams):
    """The generator's train and test data for every arm (fixed seeds) and the truth's NLL."""
    pw = cfg["power"]
    arms = Q.parse_arms(cfg["arms"])
    qm = ctx.model(p)
    length = cfg["analysis"]["length"]
    seed = pw["seed"] + (0 if gname == "fixed" else 1000)
    out = {}
    for split, n, off in (("train", pw["n_train"], 0), ("test", pw["n_test"], 500)):
        rng = np.random.default_rng(seed + off)
        out[split] = {a: simulate(qm, arm, n, length, rng) for a, arm in arms.items()}
    return arms, qm, out


def start_model(task, arms, qm, rng) -> C.ProgramModel:
    els, lab = Q.arm_elements(arms)
    if task["start"] == "random":
        return C.random_program_model(task["d"], els, rng, lab)
    maps = Q.bloch_maps(qm, els, lab)
    if task["start"] == "klein":
        s30 = json.loads((REPO_ROOT / _W["cfg"]["experiment"]["classical_summary"]).read_text())
        e = s30["embedding"]
        u = C.klein_tetrahedron(e["a"], e["b"])
    elif task["start"] == "octahedron":
        u = Q.octahedron(0.9, 1.0)
    else:
        raise ValueError(task["start"])
    return Q.polytope_start(maps, u, els, lab)


REFIT = ["dephase", "damp", "detune", "flip", "reset_e0", "reset_e1", "meas_decay", "over"]


def _quantum_task(task: dict[str, Any]) -> dict[str, Any]:
    """Per generator: the quantum evaluators' per-shot NLL on its train and test data: the
    truth (the generator), the other generator (a misspecified quantum prediction), and the
    quantum model refitted to the training shots (``REFIT``; also without the held-out arms)."""
    cfg, ctx = _W["cfg"], _W["ctx"]
    run = Path(task["run"])
    res_path = run / f"quantum_{task['generator']}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    t0 = time.time()
    g = task["generator"]
    gens = generators(ctx, cfg, _W["review"])
    arms, _, ds = datasets(cfg, ctx, g, gens[g])
    held = set(cfg["design"]["held_out_arms"])
    evals = {"truth": gens[g]}
    for other in gens:
        if other != g:
            evals["other"] = gens[other]
    out: dict[str, Any] = {"label": f"quantum_{g}", "generator": g, "refit": {}}
    for name, keep in (("refit", None), ("refit_holdarms", held)):
        train = {a: d for a, d in ds["train"].items() if keep is None or a not in keep}

        def f(x, train=train):
            p = Q.from_vector(x, REFIT, gens[g])
            m = ctx.model(p)
            v = sum(float(m.nll(arms[a], d).sum()) for a, d in train.items())
            return v if math.isfinite(v) else 1e30

        base = replace(gens[g], over=0.0)
        if g != "fixed":  # start the refit from the calibrated model, not from the truth
            base = replace(gens["fixed"], ff_idle_s=gens[g].ff_idle_s)
        x0 = Q.to_vector(base, REFIT, ctx.quantum.pieces)
        x, fx, n_ev = nelder_mead(f, x0, step=0.4, max_evals=cfg["power"]["refit_evals"], ftol=1e-4)
        p = Q.from_vector(x, REFIT, base)
        evals[name] = p
        out["refit"][name] = {"params": p.to_dict(), "nll_train": fx, "evals": n_ev}
    arrays = {}
    for ev, p in evals.items():
        m = ctx.model(p)
        for split in ("train", "test"):
            for a, d in ds[split].items():
                arrays[f"{ev}|{split}|{a}"] = m.nll(arms[a], d)
    np.savez(run / f"quantum_{g}.npz", **arrays)
    out["evaluators"] = list(evals)
    out["seconds"] = time.time() - t0
    save_json(res_path, out)
    return out


def _power_task(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("kind") == "quantum":
        return _quantum_task(task)
    cfg, ctx = _W["cfg"], _W["ctx"]
    run = Path(task["run"])
    res_path = run / f"power_{task['label']}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    t0 = time.time()
    gens = generators(ctx, cfg, _W["review"])
    arms, qm, ds = datasets(cfg, ctx, task["generator"], gens[task["generator"]])
    held = set(cfg["design"]["held_out_arms"]) if task.get("hold_arms") else set()
    train = {a: d for a, d in ds["train"].items() if a not in held}
    tarms = {a: arms[a] for a in train}
    salt = {"klein": 1, "random": 2, "octahedron": 3, "split": 4}[task["start"]]
    rng = np.random.default_rng(cfg["power"]["seed"] + task["d"] * 10 + salt)
    if task["start"] == "split":
        prev = C.ProgramModel.from_dict(json.loads((run / task["from"]).read_text())["model"])
        start = C.smooth_program(C.split_program_state(prev, rng), 0.01)
        while start.d < task["d"]:
            start = C.smooth_program(C.split_program_state(start, rng), 0.01)
    else:
        start = start_model(task, arms, qm, rng)
    ck = run / f"ckpt_power_{task['label']}.json"
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
        tarms, train, start, max(cfg["power"]["max_iters"] - done, 2), 1e-10, callback=cb
    )
    m = fit.model
    out: dict[str, Any] = {
        "label": task["label"],
        "generator": task["generator"],
        "d": task["d"],
        "start": task["start"],
        "hold_arms": bool(task.get("hold_arms")),
        "iters": done + fit.iters,
        "converged": fit.converged,
        "last_gain_per_symbol": (fit.history[-2] - fit.history[-1])
        if len(fit.history) > 1
        else None,
        "n_params": m.n_params(),
        "model": m.to_dict(),
    }
    arrays = {}
    for split in ("train", "test"):
        for a, d in ds[split].items():
            arrays[f"{split}|{a}"] = m.nll(arms[a], d)
    np.savez(run / f"power_{task['label']}.npz", **arrays)
    s = settings(cfg)
    rows = []
    for a, arm in arms.items():
        obs = Q.data_stats(ds["test"][a].y, s)
        mod = Q.model_arm_stats(m, arm, s)
        for k in [f"w{j}" for j in s.offsets] + ["p1", "lag1"]:
            se = obs[f"{k}_se"]
            rows.append(
                {
                    "arm": a,
                    "stat": k,
                    "model": mod[k],
                    "data": obs[k],
                    "se": se,
                    "z": (mod[k] - obs[k]) / se if se > 0 else 0.0,
                }
            )
        for i, (gm, go, gs) in enumerate(zip(mod["g"], obs["g"], obs["g_se"], strict=True)):
            rows.append(
                {
                    "arm": a,
                    "stat": f"g{i}",
                    "model": gm,
                    "data": go,
                    "se": gs,
                    "z": (gm - go) / gs if gs > 0 else 0.0,
                }
            )
    out["stat_rows"] = rows
    out["seconds"] = time.time() - t0
    save_json(res_path, out)
    return out


def part_power(
    cfg: dict[str, Any],
    cfg_path: str,
    out: Path,
    workers: int,
    review: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pw = cfg["power"]
    tasks0, tasks1, tasks2 = [], [], []
    for g in pw["generators"]:
        tasks0.append({"kind": "quantum", "label": f"quantum_{g}", "generator": g})
        tasks1.append({"label": f"{g}_d4_klein", "generator": g, "d": 4, "start": "klein"})
        tasks1.append({"label": f"{g}_d4_random", "generator": g, "d": 4, "start": "random"})
        tasks1.append(
            {"label": f"{g}_d6_octahedron", "generator": g, "d": 6, "start": "octahedron"}
        )
        tasks1.append(
            {
                "label": f"{g}_d4_klein_holdarms",
                "generator": g,
                "d": 4,
                "start": "klein",
                "hold_arms": True,
            }
        )
        tasks2.append(
            {
                "label": f"{g}_d6_split",
                "generator": g,
                "d": 6,
                "start": "split",
                "from": f"power_{g}_d4_klein.json",
            }
        )
        tasks2.append(
            {
                "label": f"{g}_d6_split_holdarms",
                "generator": g,
                "d": 6,
                "start": "split",
                "from": f"power_{g}_d4_klein_holdarms.json",
                "hold_arms": True,
            }
        )
    for t in tasks0 + tasks1 + tasks2:
        t["run"] = str(out)
    res = []
    for round_, tasks in enumerate((tasks0 + tasks1, tasks2)):
        PC.log(out, f"power round {round_ + 1}: {len(tasks)} tasks, {workers} workers")
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg_path, review)) as ex:
            for r in ex.map(_power_task, tasks):
                PC.log(out, f"  {r['label']}: {r.get('iters', '-')} steps, {r['seconds']:.0f} s")
                res.append(r)
    summary = summarise_power(cfg, out, [r for r in res if "d" in r])
    summary["quantum"] = {r["generator"]: r for r in res if "d" not in r}
    summary["meta"] = meta
    write_power(cfg, summary)
    return summary


def _paired(c: dict[str, np.ndarray], q: dict[str, np.ndarray], split: str, arms) -> dict:
    diffs = [c[f"{split}|{a}"] - q[f"{split}|{a}"] for a in arms]
    n = sum(len(x) for x in diffs)
    tot = float(sum(x.sum() for x in diffs))
    var = sum(float(((x - x.mean()) ** 2).sum()) for x in diffs) / max(n - len(diffs), 1)
    sd = math.sqrt(var)
    return {
        "deficit_per_shot": tot / n,
        "sd_per_shot": sd,
        "shots": n,
        "z": tot / (sd * math.sqrt(n)) if sd > 0 else math.nan,
    }


def summarise_power(cfg: dict[str, Any], run: Path, res: list[dict[str, Any]]) -> dict[str, Any]:
    pw = cfg["power"]
    za, zb = pw["z_alpha"], pw["z_beta"]
    held = set(cfg["design"]["held_out_arms"])
    all_arms = list(Q.parse_arms(cfg["arms"]))
    qarr = {}
    for g in pw["generators"]:
        with np.load(run / f"quantum_{g}.npz") as z:
            qarr[g] = {k: z[k] for k in z.files}
    out: dict[str, Any] = {"fits": [], "best": {}}
    for r in res:
        with np.load(run / f"power_{r['label']}.npz") as z:
            c = {k: z[k] for k in z.files}
        g = r["generator"]
        evs = sorted({k.split("|")[0] for k in qarr[g]})
        row = {
            k: r[k] for k in ("label", "generator", "d", "start", "hold_arms", "iters", "converged")
        }
        row["n_params"] = r["n_params"]
        row["last_gain_per_symbol"] = r.get("last_gain_per_symbol")
        groups = (
            {"fitted": [a for a in all_arms if a not in held], "held_out_arms": sorted(held)}
            if r["hold_arms"]
            else {"all": all_arms}
        )
        for ev in evs:
            if ev == "refit_holdarms" and not r["hold_arms"]:
                continue
            if ev == "refit" and r["hold_arms"]:
                continue
            q = {k.split("|", 1)[1]: v for k, v in qarr[g].items() if k.startswith(ev + "|")}
            for split in ("train", "test"):
                for gname, arms in groups.items():
                    row[f"{ev}|{split}|{gname}"] = _paired(c, q, split, arms)
            row[f"{ev}|by_arm_test"] = {
                a: float((c[f"test|{a}"] - q[f"test|{a}"]).mean()) for a in all_arms
            }
        sr = r["stat_rows"]
        zs = np.array([abs(x["z"]) for x in sr])
        zbonf = float(np.sqrt(2) * _erfinv(1 - 0.05 / len(sr)))
        worst = max(sr, key=lambda x: abs(x["z"]))
        row.update(
            {
                "stats_n": len(sr),
                "stats_max_abs_z": float(zs.max()),
                "stats_worst": f"{worst['arm']} {worst['stat']}",
                "stats_over_1.96": int((zs > 1.96).sum()),
                "stats_bonferroni_z": zbonf,
                "stats_over_bonferroni": int((zs > zbonf).sum()),
            }
        )
        row["train_nll_per_shot"] = float(
            sum(c[f"train|{a}"].sum() for a in all_arms if not (r["hold_arms"] and a in held))
            / sum(len(c[f"train|{a}"]) for a in all_arms if not (r["hold_arms"] and a in held))
        )
        out["fits"].append(row)
    # the best start per (generator, d, hold_arms) by its training NLL; monotone in d (a
    # d = 4 model is a d = 6 model: d = 6 is never credited with a worse fit than d = 4's)
    for g in pw["generators"]:
        for hold in (False, True):
            prev = None
            for d in sorted(pw["dims"]):
                cand = [
                    f
                    for f in out["fits"]
                    if f["generator"] == g and f["d"] == d and f["hold_arms"] == hold
                ]
                if not cand:
                    continue
                b = min(cand, key=lambda f: f["train_nll_per_shot"])
                note = None
                if prev is not None and prev["train_nll_per_shot"] < b["train_nll_per_shot"]:
                    b, note = prev, f"the d = {prev['d']} fit is better: used for d = {d}"
                prev = b
                group = "held_out_arms" if hold else "all"
                n_arms = len(held) if hold else len(all_arms)
                entry: dict[str, Any] = {"label": b["label"], "note": note}
                for ev in ("truth", "other", "refit_holdarms" if hold else "refit"):
                    key = f"{ev}|test|{group}"
                    if key not in b:
                        continue
                    t = b[key]
                    tin = b[f"{ev}|train|{'fitted' if hold else 'all'}"]
                    entry[ev] = {
                        "held_out_deficit_per_shot": t["deficit_per_shot"],
                        "in_sample_deficit_per_shot": tin["deficit_per_shot"],
                        "sd_per_shot": t["sd_per_shot"],
                        "test_z": t["z"],
                        "shots_80pct_total": Q.shots_for_power(
                            t["deficit_per_shot"], t["sd_per_shot"], za, zb
                        ),
                        "power_at_planned": {
                            str(n): Q.power_one_sided(
                                t["deficit_per_shot"], t["sd_per_shot"], n * n_arms, za
                            )
                            for n in pw["planned_shots_per_arm"]
                        },
                    }
                entry["stats_over_bonferroni"] = b["stats_over_bonferroni"]
                entry["stats_max_abs_z"] = b["stats_max_abs_z"]
                entry["stats_worst"] = b["stats_worst"]
                out["best"][f"{g}_d{d}{'_holdarms' if hold else ''}"] = entry
    return out


def _erfinv(y: float) -> float:
    """Inverse error function by bisection (no scipy)."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def write_power(cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    sd = PC.sdir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    PC.write_csv(
        sd / "power_fits.csv",
        [
            PC.jround({k: v for k, v in r.items() if not isinstance(v, dict)})
            for r in summary["fits"]
        ],
    )
    save_json(sd / "power.json", PC.jround(summary, 9))


# -- the plan -------------------------------------------------------------------------------------


def build_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    """The unsubmitted plan: the arms and shots of ``[plan]``, each job's estimate and upper
    bound with ``qmem.followup``'s cost model, against the per-tier hard caps."""
    sd = PC.sdir(cfg)
    pl, bc = cfg["plan"], cfg["budget"]
    cm = F.CostModel(
        bc["sub_job_overhead_s"], bc["rep_delay_s"], bc["margin"], bc["billing_round_s"]
    )
    arms = Q.parse_arms(cfg["arms"])
    dur = {r["arm"]: r["circuit_s"] for r in budget(cfg, arms)["arms"]}
    circuits = json.loads((sd / "circuits.json").read_text())
    tiers = []
    for tier in pl["tiers"]:
        if tier.get("kind") == "controls":
            tiers.append(controls_tier(tier, sd, cm))
            continue
        n_jobs = tier["jobs"]
        n_arm = tier.get("shots_per_arm", pl["shots_per_arm"])
        per_job = [F.Pub(0, i, a, 0, n_arm // n_jobs) for i, a in enumerate(tier["arms"])]
        job = cm.job(per_job, dur)
        tiers.append(
            {
                "tier": tier["name"],
                "purpose": tier["purpose"],
                "arms": tier["arms"],
                "shots_per_arm": n_arm,
                "jobs": n_jobs,
                "per_job": job,
                "estimate_s": n_jobs * job["estimate_s"],
                "upper_s": n_jobs * job["upper_s"],
                "hard_cap_s": tier["hard_cap_s"],
                "within_cap": n_jobs * job["upper_s"] <= tier["hard_cap_s"],
            }
        )
    plan = {
        "status": "UNSUBMITTED: needs the owner's explicit approval on #28 (stage, device, "
        "circuits, upper bound, hard cap). Phase 7's approvals and cap do not apply.",
        "device": "ibm_kingston",
        "layout": {"memory": 75, "output": 59, "ancillas": circuits["ancillas"]["chosen"]},
        "transpile": "optimization_level = 0, fixed layout (as Phase 7); every echo fenced",
        "qasm_sha256": circuits["qasm_sha256"],
        "tiers": tiers,
        "total_upper_s": sum(t["upper_s"] for t in tiers),
        "allowance_s": bc["allowance_s"],
        "guards": [
            "qmem.followup.budget_check per job (Open plan only, live remaining allowance)",
            "live calibration snapshot and qmem.followup.calibration_check before each job",
            "pilot job first; stop on a failed pilot gate (prereg draft, docs/phase8-coherence.md)",
            "max(charged, upper bound) accounting; resume by job id; raw shots preserved",
            "pub order randomised within each job; per-pub timing recorded",
        ],
    }
    save_json(sd / "plan.json", PC.jround(plan, 9))
    return plan


def controls_tier(tier: dict[str, Any], sd: Path, cm: F.CostModel) -> dict[str, Any]:
    """#34's controls tier: the circuits of ``--export-controls`` (``controls.json``: shots,
    FakeKingston durations, QASM hashes), one pub per circuit, in ``jobs`` jobs."""
    ctl = json.loads((sd / "controls.json").read_text())
    rows = ctl["rows"]
    n_jobs = tier["jobs"]
    dur = {r["circuit"]: r["duration_s"] for r in rows}
    pubs = [F.Pub(0, i, r["circuit"], 0, r["shots"] // n_jobs) for i, r in enumerate(rows)]
    job = cm.job(pubs, dur)
    by_kind: dict[str, Any] = {}
    for r in rows:
        k = by_kind.setdefault(r["kind"], {"circuits": 0, "shots": 0, "shot_s": 0.0})
        k["circuits"] += 1
        k["shots"] += r["shots"]
        k["shot_s"] += cm.shot_seconds(r["shots"], r["duration_s"])
    return {
        "tier": tier["name"],
        "purpose": tier["purpose"],
        "circuits": {r["circuit"]: r["shots"] for r in rows},
        "by_kind": by_kind,
        "qasm_sha256": ctl["qasm_sha256"],
        "jobs": n_jobs,
        "per_job": job,
        "estimate_s": n_jobs * job["estimate_s"],
        "upper_s": n_jobs * job["upper_s"],
        "hard_cap_s": tier["hard_cap_s"],
        "within_cap": n_jobs * job["upper_s"] <= tier["hard_cap_s"],
    }


def print_plan(cfg: dict[str, Any]) -> None:
    plan = build_plan(cfg)
    print(json.dumps(plan, indent=2))
    print(
        "\nUNSUBMITTED. Nothing is sent: this script has no submission path. A Stage 2 job "
        "needs the owner's explicit approval on #28 of the stage, device, circuits, upper "
        "bound and hard cap (#28 Stage 4)."
    )
