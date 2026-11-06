"""Phase 4a (issue #12): trainability of architecture A and of the large-D quantum baselines.

One command runs ``experiments/configs/phase4a.toml``:

1. ``[init_stats]``: gradient norm and its variance over initialisations, initial NLL,
   predictive entropy, history dependence and the purity of the outcome-averaged
   memory, for each model and n, with no training;
2. ``[[arms]]`` in config order: tuning (if the arm has ``lrs``) on the tuning seeds,
   then the evaluation seeds; soft-target arms resolve their teacher from earlier arms;
3. the summary: failure counts and gaps per cell, gradient norms along training, the
   teacher-fit KLs, retention metrics for ``marked_recall`` cells, and the Hessian
   spectrum at the end of training for the arms in ``analysis.hessian_arms``.

Writes ``results/phase4a/<timestamp>/`` (records and parameters, gitignored) and
``<summary_dir>/`` (``summary.md``, CSVs, ``summary.json``, plots; committed).
``--from-run <dir>`` reuses saved records; ``--arms a,b`` runs only those arms;
``--no-train`` summarises saved records only; ``--skip-init-stats`` reuses them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from qmem.ablations import build_variant
from qmem.baselines.runner import BaselineSpec, build_baseline, load_any, run_baseline
from qmem.experiment import aggregate, build_splits, is_success, run_many
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import make_task
from qmem.trainability import init_statistics, resolve_init_probs

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase4a.toml"
sys.path.insert(0, str(Path(__file__).parent))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# -- configuration -------------------------------------------------------------------
class Setup:
    """Tasks, data and base recipes, gathered from the referenced configs."""

    def __init__(self, cfg: dict[str, Any]):
        exp = cfg["experiment"]
        self.cfg = cfg
        self.a = tomllib.loads((REPO_ROOT / exp["a_config"]).read_text())
        self.ff = tomllib.loads((REPO_ROOT / exp["ff_config"]).read_text())
        self.tasks: dict[str, dict[str, Any]] = {}
        for name, t in self.a["tasks"].items():
            self.tasks[name] = {
                "spec": t["spec"],
                "n_output": t["n_output"],
                "data": self.a["data"],
            }
        for name, t in cfg.get("extra_tasks", {}).items():
            data = {**self.ff["data"], "length": t.get("length", self.ff["data"]["length"])}
            self.tasks[name] = {"spec": t["spec"], "n_output": t["n_output"], "data": data}

    def model(self, family: str, task: str, n: int, extra: dict[str, Any]) -> dict[str, Any]:
        n_out = self.tasks[task]["n_output"]
        if family == "A":
            return {**self.a["model"], "n_output": n_out, "n_memory": n, **extra}
        if family == "A_ff":
            return {**self.ff["model"], "n_output": n_out, "n_memory": n, **extra}
        base = dict(self.cfg["baseline_model"][family])
        if family == "hqmm":
            base["n_memory"] = n
        else:
            base["bond_dim"] = 2**n
        return {**base, **extra}

    def train(self, family: str, extra: dict[str, Any]) -> dict[str, Any]:
        if family == "A":
            base = self.a["train"]
        elif family == "A_ff":
            base = self.ff["train"]
        else:
            base = {"lr": 0.03, **self.cfg["baseline_train"]}
        # [train_override] (absent in the committed config) caps budgets for smoke tests.
        return {**base, **extra, **self.cfg.get("train_override", {})}


def runner_family(family: str) -> str:
    return "A" if family in ("A", "A_ff") else family


def cells(arm: dict[str, Any]) -> list[tuple[str, int]]:
    return [(t, n) for t in arm["tasks"] for n in arm["ns"]]


def group_name(arm: str, task: str, n: int, lr: float | None = None) -> str:
    g = f"{arm}/{task}/n{n}"
    return g if lr is None else f"tune/{g}/lr={lr:g}"


def make_spec(
    setup: Setup, arm: dict[str, Any], task: str, n: int, seed: int, *,
    train_extra: dict[str, Any], group: str, teacher: dict[str, Any] | None = None,
) -> BaselineSpec:  # fmt: skip
    fam = arm["family"]
    t = setup.tasks[task]
    return BaselineSpec(
        group=group,
        task=t["spec"],
        data=t["data"],
        family=runner_family(fam),
        model=setup.model(fam, task, n, arm.get("model", {})),
        train=setup.train(fam, {**arm.get("train", {}), **train_extra}),
        seed=seed,
        labels={"arm": arm["name"], "family": fam, "task": task, "n": n, "state_dim": 2**n},
        teacher=teacher or {},
    )


# -- records -------------------------------------------------------------------------
def load_records(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for path in sorted((run_dir / "runs").rglob("seed*.json")):
        rec = json.loads(path.read_text())
        rec["_path"] = str(path)
        out[(rec["group"], rec["seed"])] = rec
    return out


def run_specs(specs, records, out, workers) -> None:
    missing = [s for s in specs if (s.group, s.seed) not in records]
    if not missing:
        return
    print(f"  training {len(missing)} runs", flush=True)
    for rec in run_many(missing, out, workers=workers, runner=run_baseline):
        rec.pop("state_dict", None)
        rec["_path"] = str(out / "runs" / rec["group"] / f"seed{rec['seed']}.json")
        records[(rec["group"], rec["seed"])] = rec


def median_val(recs: list[dict[str, Any]]) -> float:
    vals = [r.get("best_val_nll", math.nan) for r in recs]
    vals = [v if isinstance(v, float) and math.isfinite(v) else math.inf for v in vals]
    return statistics.median(vals) if vals else math.inf


def resolve_teacher(
    teacher: dict[str, Any], task: str, n: int, records: dict, cfg: dict[str, Any]
) -> dict[str, Any]:
    kind = teacher["kind"]
    if not kind.startswith("best:"):
        return teacher
    src_arm = kind.split(":", 1)[1]
    g = group_name(src_arm, task, teacher.get("n", n))
    cands = [r for (gg, _), r in records.items() if gg == g and "best_val_nll" in r]
    if not cands:
        raise RuntimeError(f"teacher arm {src_arm} has no runs for {g} yet")
    best = min(cands, key=lambda r: r["best_val_nll"])
    out = {k: v for k, v in teacher.items() if k not in ("kind", "n")}
    return {**out, "kind": "record", "path": best["_path"], "from": f"{g}/seed{best['seed']}"}


def run_arms(setup, cfg, records, out, workers, only: list[str] | None) -> list[dict[str, Any]]:
    exp = cfg["experiment"]
    tuning_rows = []
    for arm in cfg["arms"]:
        if only and arm["name"] not in only:
            continue
        print(f"arm {arm['name']}", flush=True)
        chosen: dict[tuple[str, int], float | None] = {c: None for c in cells(arm)}
        if arm.get("lrs"):
            specs = [
                make_spec(
                    setup,
                    arm,
                    task,
                    n,
                    s,
                    group=group_name(arm["name"], task, n, lr),
                    train_extra={"lr": lr, **arm.get("tune_train", {})},
                )
                for task, n in cells(arm)
                for lr in arm["lrs"]
                for s in exp["tuning_seeds"]
            ]
            run_specs(specs, records, out, workers)
            for task, n in cells(arm):
                scores = {}
                for lr in arm["lrs"]:
                    g = group_name(arm["name"], task, n, lr)
                    recs = [records[(g, s)] for s in exp["tuning_seeds"] if (g, s) in records]
                    scores[lr] = median_val(recs)
                best = min(scores, key=lambda lr: scores[lr])
                chosen[(task, n)] = best
                for lr, v in scores.items():
                    tuning_rows.append(
                        {"arm": arm["name"], "task": task, "n": n, "lr": lr,
                         "median_val_nll": v, "chosen": lr == best}
                    )  # fmt: skip
        specs = []
        for task, n in cells(arm):
            extra = {} if chosen[(task, n)] is None else {"lr": chosen[(task, n)]}
            teacher = None
            if arm.get("teacher"):
                teacher = resolve_teacher(arm["teacher"], task, n, records, cfg)
            specs += [
                make_spec(
                    setup,
                    arm,
                    task,
                    n,
                    s,
                    group=group_name(arm["name"], task, n),
                    train_extra=extra,
                    teacher=teacher,
                )
                for s in exp["seeds"]
            ]
        run_specs(specs, records, out, workers)
    return tuning_rows


# -- init statistics -----------------------------------------------------------------
def run_init_stats(setup: Setup, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ic = cfg["init_stats"]
    task_name = ic["task"]
    t = setup.tasks[task_name]
    task = make_task(t["spec"])
    splits = build_splits(task, t["data"])
    batch = splits.train[: ic["batch"]]
    rows = []
    for m in ic["models"]:
        fam = m["family"]
        for n in ic["ns"]:
            mcfg = setup.model(fam, task_name, n, m.get("model", {}))
            mcfg = resolve_init_probs(mcfg, splits.train, task.n_symbols)

            def make(seed: int, fam=fam, mcfg=mcfg):
                if fam in ("A", "A_ff"):
                    return build_variant({**mcfg, "seed": seed})
                return build_baseline(fam, mcfg, task.n_symbols, seed)

            t0 = time.perf_counter()
            st = init_statistics(make, list(range(ic["seeds"])), batch)
            row = {"model": m["name"], "family": fam, "n": n, "D": 2**n, **st}
            row["seconds"] = time.perf_counter() - t0
            print(
                f"  {m['name']:<28s} n={n} P={st['n_parameters']:5d} |g|={st['grad_norm_mean']:.4f}"
                f" Var={st['grad_var']:.2e} H={st['pred_entropy_mean']:.4f}"
                f" TV={st['history_tv_mean']:.4f}",
                flush=True,
            )
            rows.append(row)
    return rows


# -- analysis ------------------------------------------------------------------------
def grad_profile(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Gradient norm along training: mean and sd over runs at a few step windows."""
    out = {}
    for label, (lo, hi) in {"g_0": (0, 1), "g_100": (90, 110), "g_500": (490, 510)}.items():
        vals = []
        for r in recs:
            g = r.get("curves", {}).get("grad_norm", [])[lo:hi]
            g = [x for x in g if isinstance(x, float) and math.isfinite(x)]
            if g:
                vals.append(statistics.median(g))
        out[label] = statistics.mean(vals) if vals else math.nan
        out[label + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else math.nan
    return out


def med(values: list[float]) -> float:
    v = [x for x in values if math.isfinite(x)]
    return statistics.median(v) if v else math.nan


def final_grad(r: dict[str, Any]) -> float:
    g = [x for x in r.get("curves", {}).get("grad_norm", [])[-20:] if math.isfinite(x)]
    return statistics.median(g) if g else math.nan


def hessian_spectrum(rec: dict[str, Any], h: float = 1e-4) -> dict[str, Any]:
    """Eigenvalues of the train-NLL Hessian at the trained parameters (central differences)."""
    rec = {**rec, "state_dict": torch.load(Path(rec["_path"]).with_suffix(".pt"))}
    model = load_any(rec)
    cfg = rec["meta"]["config"]
    seqs = build_splits(make_task(cfg["task"]), cfg["data"]).train
    params = [p for p in model.parameters() if p.requires_grad]
    flat0 = torch.cat([p.detach().flatten() for p in params])

    def grad_at(flat: torch.Tensor) -> torch.Tensor:
        i = 0
        with torch.no_grad():
            for p in params:
                p.copy_(flat[i : i + p.numel()].view_as(p))
                i += p.numel()
        model.zero_grad()
        (-model.step_log_probs(seqs).mean()).backward()
        return torch.cat([p.grad.flatten() for p in params])

    cols = []
    for k in range(len(flat0)):
        e = torch.zeros_like(flat0)
        e[k] = h
        cols.append((grad_at(flat0 + e) - grad_at(flat0 - e)) / (2 * h))
    g0 = grad_at(flat0)
    hess = torch.stack(cols, 1)
    ev = torch.linalg.eigvalsh(0.5 * (hess + hess.T))
    return {
        "grad_norm": float(g0.norm()),
        "eig_min": float(ev[0]),
        "eig_max": float(ev[-1]),
        "n_negative": int((ev < -1e-6).sum()),
        "n_flat": int((ev.abs() <= 1e-6).sum()),
        "n_params": len(flat0),
    }


def retention(rec: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    import phase3e  # the Phase 3e per-model analysis (experiments/phase3e.py)

    rec = {**rec, "state_dict": torch.load(Path(rec["_path"]).with_suffix(".pt"))}
    model = load_any(rec)
    c = rec["meta"]["config"]
    task = make_task(c["task"])
    test = build_splits(task, c["data"]).test
    a = phase3e.analyse_model(model, task, test, cfg["analysis"])
    pur = a["purification_generated"]
    return {
        "eta": a["eta"],
        "recall_accuracy": a["recall"]["recall_accuracy"],
        "horizon": a["horizon"],
        "purify_steps_median": pur.get("purification_time_median", math.nan),
        "retains_to_max_lag": a["horizon"] > cfg["analysis"]["max_lag"],
    }


def summarise(cfg, records, tuning_rows):
    exp = cfg["experiment"]
    eval_recs: dict[str, list[dict[str, Any]]] = {}
    for (g, _), r in records.items():
        if not g.startswith("tune/"):
            eval_recs.setdefault(g, []).append(r)
    arm_order = {a["name"]: i for i, a in enumerate(cfg["arms"])}
    groups = sorted(
        eval_recs, key=lambda g: (arm_order.get(g.split("/")[0], 99), g.split("/")[1], g)
    )
    rows = []
    hess_arms = set(cfg["analysis"].get("hessian_arms", []))
    for g in groups:
        recs = sorted(eval_recs[g], key=lambda r: r["seed"])
        agg = aggregate(recs, exp["success_gap"])[0]
        lab = recs[0]["labels"]
        ok = [r for r in recs if is_success(r, exp["success_gap"])]
        bad = [r for r in recs if not is_success(r, exp["success_gap"]) and "test" in r]
        best = min((r for r in recs if "best_val_nll" in r), key=lambda r: r["best_val_nll"])
        row = {
            "group": g, "arm": lab["arm"], "family": lab["family"], "task": lab["task"],
            "n": lab["n"], "D": lab["state_dim"], "n_runs": agg["n_runs"],
            "n_failed": agg["n_runs"] - agg["n_success"], "gap_median": agg["gap_median"],
            "gap_mean": agg["gap_mean"], "gap_se": agg["gap_se"], "gap_best": agg["gap_best"],
            "gap_worst": agg["gap_worst"], "val_selected_gap": best["test"]["gap_nats"],
            "val_selected_seed": best["seed"], "n_parameters": agg["n_parameters"],
            "lr": recs[0]["meta"]["config"]["train"].get("lr"),
            "steps_mean": agg["steps_mean"],
            "cpu_minutes": agg["wall_seconds_total"] / 60,
            "final_grad_success": med([final_grad(r) for r in ok]),
            "final_grad_failed": med([final_grad(r) for r in bad]),
            "statuses": agg["statuses"],
            "seed_gaps": [r.get("test", {}).get("gap_nats", math.nan) for r in recs],
            **grad_profile(recs),
        }  # fmt: skip
        soft = [r["soft"] for r in recs if "soft" in r]
        if soft:
            row["soft_kl_test_median"] = statistics.median(s["kl_test"] for s in soft)
            row["soft_kl_test_min"] = min(s["kl_test"] for s in soft)
            row["soft_kl_test_max"] = max(s["kl_test"] for s in soft)
            row["teacher"] = recs[0]["meta"]["config"].get("teacher", {}).get("from", "source")
        screens = [r.get("restart_screen") for r in recs if r.get("restart_screen")]
        if screens:
            row["restart_chosen"] = [
                next(s["restart"] for s in sc if s["chosen"]) for sc in screens
            ]
        if lab["arm"] in hess_arms and lab["n"] == 2:
            print(f"  Hessian for {g}", flush=True)
            hs = []
            for r in recs:
                if "_path" in r and "test" in r:
                    hs.append({"seed": r["seed"], "success": r in ok, **hessian_spectrum(r)})
            row["hessian"] = hs
        if lab["task"].startswith("marked_recall"):
            print(f"  retention for {g}", flush=True)
            rets = [retention(r, cfg) for r in recs if "test" in r]
            row["recall_accuracy_mean"] = statistics.mean(x["recall_accuracy"] for x in rets)
            row["recall_accuracy_se"] = statistics.stdev(
                x["recall_accuracy"] for x in rets
            ) / math.sqrt(len(rets))
            row["horizon_mean"] = statistics.mean(x["horizon"] for x in rets)
            row["horizon_se"] = statistics.stdev(x["horizon"] for x in rets) / math.sqrt(len(rets))
            row["eta_mean"] = statistics.mean(x["eta"] for x in rets)
            row["seeds_retaining"] = sum(x["retains_to_max_lag"] for x in rets)
            row["purify_steps_median"] = statistics.median(
                x["purify_steps_median"] for x in rets
                if isinstance(x["purify_steps_median"], (int, float))
                and math.isfinite(x["purify_steps_median"])
            ) if rets else math.nan  # fmt: skip
            row["val_selected_recall_accuracy"] = rets[
                [r["seed"] for r in recs if "test" in r].index(best["seed"])
            ]["recall_accuracy"]
            row["retention"] = rets
        rows.append(row)
    return rows


# -- output --------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k, v in r.items() if k not in keys and not isinstance(v, (list, dict))]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def markdown(rows, init_rows, tuning_rows, meta, cpu_minutes) -> str:
    out = [
        "# Phase 4a summary (generated by `experiments/phase4a.py`)",
        "",
        f"Commit `{(meta['git_commit'] or '?')[:7]}` (dirty: {meta['git_dirty']}); "
        f"{cpu_minutes:.0f} CPU-minutes of training summed over runs. Failure: gap >= 0.01 "
        "nats/symbol, crash or divergence. Ten evaluation seeds (0-9) per cell.",
        "",
        "## Initialisation statistics (no training)",
        "",
        "| model | n | D | params | grad norm (sd) | Var[dL/dtheta] | initial NLL | "
        "pred. entropy | history TV | purity of E^T(rho0) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in init_rows:
        out.append(
            f"| {r['model']} | {r['n']} | {r['D']} | {r['n_parameters']} | "
            f"{r['grad_norm_mean']:.4f} ({r['grad_norm_sd']:.4f}) | {r['grad_var']:.2e} | "
            f"{r['nll_mean']:.4f} | {r['pred_entropy_mean']:.4f} | {r['history_tv_mean']:.4f} | "
            f"{fmt(r['avg_purity_mean'])} |"
        )
    out += [
        "",
        "## Training cells",
        "",
        "| cell | params | lr | failed | median gap | best / worst | val-selected gap | "
        "mean steps | grad norm at 0 / 100 / 500 | final grad (ok / failed) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| `{r['group']}` | {r['n_parameters']} | {fmt(r['lr'], 3)} | {r['n_failed']}/"
            f"{r['n_runs']} | {fmt(r['gap_median'])} | {fmt(r['gap_best'])} / "
            f"{fmt(r['gap_worst'])} | {fmt(r['val_selected_gap'])} | {r['steps_mean']:.0f} | "
            f"{fmt(r['g_0'], 3)} / {fmt(r['g_100'], 3)} / {fmt(r['g_500'], 3)} | "
            f"{fmt(r['final_grad_success'], 4)} / {fmt(r['final_grad_failed'], 4)} |"
        )
    soft = [r for r in rows if "soft_kl_test_median" in r]
    if soft:
        out += ["", "## Soft-target fits (KL to the teacher on the test split, nats/symbol)", "",
                "| cell | teacher | KL median | min | max | data gap median | failed |",
                "|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in soft:
            out.append(
                f"| `{r['group']}` | {r['teacher']} | {fmt(r['soft_kl_test_median'])} | "
                f"{fmt(r['soft_kl_test_min'])} | {fmt(r['soft_kl_test_max'])} | "
                f"{fmt(r['gap_median'])} | {r['n_failed']}/{r['n_runs']} |"
            )
    hess = [r for r in rows if r.get("hessian")]
    for r in hess:
        out += ["", f"## Hessian of the train NLL at the end of training: `{r['group']}`", "",
                "| seed | success | test gap | grad norm | min eig | max eig | negative | flat |",
                "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for h in r["hessian"]:
            gap = r["seed_gaps"][h["seed"]] if h["seed"] < len(r["seed_gaps"]) else math.nan
            out.append(
                f"| {h['seed']} | {h['success']} | {fmt(gap)} | {h['grad_norm']:.2e} | "
                f"{h['eig_min']:.2e} | {h['eig_max']:.2e} | {h['n_negative']} | {h['n_flat']} |"
            )
    ret = [r for r in rows if "recall_accuracy_mean" in r]
    if ret:
        out += ["", "## Retention (marked_recall, tau = 4)", "",
                "| cell | failed | recall acc | val-selected acc | horizon | eta | "
                "seeds retaining to lag 40 | median purification steps |",
                "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in ret:
            out.append(
                f"| `{r['group']}` | {r['n_failed']}/{r['n_runs']} | "
                f"{r['recall_accuracy_mean']:.3f} ± {r['recall_accuracy_se']:.3f} | "
                f"{r['val_selected_recall_accuracy']:.3f} | {r['horizon_mean']:.1f} ± "
                f"{r['horizon_se']:.1f} | {r['eta_mean']:.2f} | {r['seeds_retaining']}/"
                f"{r['n_runs']} | {fmt(r['purify_steps_median'], 1)} |"
            )
    if tuning_rows:
        out += [
            "",
            "## Tuning (median best-validation NLL over the tuning seeds)",
            "",
            "| arm | task | n | lr | median val NLL | chosen |",
            "|---|---|---|---|---|---|",
        ]
        for t in tuning_rows:
            out.append(
                f"| {t['arm']} | {t['task']} | {t['n']} | {t['lr']:g} | "
                f"{fmt(t['median_val_nll'])} | {'yes' if t['chosen'] else ''} |"
            )  # fmt: skip
    out += ["", "Per-seed gaps are in `summary.json` (`cells[].seed_gaps`)."]
    return "\n".join(out) + "\n"


def plots(summary_dir: Path, init_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in dict.fromkeys(r["model"] for r in init_rows):
        rs = [r for r in init_rows if r["model"] == name]
        ns = [r["n"] for r in rs]
        axes[0].semilogy(ns, [max(r["grad_var"], 1e-30) for r in rs], "o-", label=name)
        axes[1].plot(ns, [r["pred_entropy_mean"] for r in rs], "o-", label=name)
        axes[2].plot(ns, [r["history_tv_mean"] for r in rs], "o-", label=name)
    axes[0].set_title("Var over inits of dL/dθ (mean over theta)")
    axes[1].set_title("initial predictive entropy (nats; ln 4 = 1.386)")
    axes[2].set_title("initial history dependence (TV)")
    for ax in axes:
        ax.set_xlabel("memory qubits n")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(summary_dir / "init_stats.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--from-run", default=None)
    p.add_argument("--arms", default=None)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--skip-init-stats", action="store_true")
    args = p.parse_args()
    start = time.perf_counter()
    cfg = tomllib.loads(Path(args.config).read_text())
    exp = cfg["experiment"]
    setup = Setup(cfg)
    summary_dir = REPO_ROOT / exp["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.from_run) if args.from_run else new_run_dir(exp["name"])
    meta = run_metadata(cfg)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    save_json(out / f"config-{stamp}.json", {"meta": meta, "args": vars(args)})
    torch.set_num_threads(1)

    init_path = out / "init_stats.json"
    if args.skip_init_stats and init_path.exists():
        init_rows = json.loads(init_path.read_text())["rows"]
    else:
        print("init statistics", flush=True)
        init_rows = run_init_stats(setup, cfg)
        save_json(init_path, {"meta": meta, "rows": init_rows})

    records = load_records(out) if args.from_run else {}
    print(f"{len(records)} saved records in {out}", flush=True)
    only = args.arms.split(",") if args.arms else None
    if args.no_train:
        tuning_rows = json.loads((out / "tuning.json").read_text())["rows"] if (
            out / "tuning.json").exists() else []  # fmt: skip
    else:
        tuning_rows = run_arms(setup, cfg, records, out, args.workers or exp["workers"], only)
        if only and (out / "tuning.json").exists():
            old = json.loads((out / "tuning.json").read_text())["rows"]
            tuning_rows = [t for t in old if t["arm"] not in only] + tuning_rows
        save_json(out / "tuning.json", {"rows": tuning_rows})
    rows = summarise(cfg, records, tuning_rows)
    cpu = sum(r.get("wall_seconds", 0.0) for r in records.values()) / 60
    commits = sorted({r["meta"]["git_commit"] for r in records.values()})
    rel = str(out.relative_to(REPO_ROOT)) if out.is_relative_to(REPO_ROOT) else str(out)
    save_json(
        summary_dir / "summary.json",
        {"meta": meta, "run_commits": commits, "run_dir": rel, "init_stats": init_rows,
         "cells": rows, "tuning": tuning_rows, "cpu_minutes": cpu},
    )  # fmt: skip
    write_csv(summary_dir / "cells.csv", rows)
    write_csv(summary_dir / "init_stats.csv", init_rows)
    write_csv(summary_dir / "tuning.csv", tuning_rows)
    (summary_dir / "summary.md").write_text(markdown(rows, init_rows, tuning_rows, meta, cpu))
    plots(summary_dir, init_rows)
    print(f"wrote {summary_dir} and {out} ({(time.perf_counter() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
