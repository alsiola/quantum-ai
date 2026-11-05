"""Phase 3a (issue #6): baselines against architecture A on the Phase 2 tasks.

One command runs everything in ``experiments/configs/phase3a.toml``:

1. architecture A, re-run with its Phase 2 recipe and seeds (``run_a``);
2. the n-gram model (``k`` and ``alpha`` chosen on validation);
3. the tuning stage for every gradient-trained baseline cell;
4. the evaluation stage: each cell's chosen candidate with the 10 evaluation
   seeds, plus HMMs fitted by EM with the same seeds (each seed is one restart);

and writes

* ``results/phase3a/<timestamp>/``: every run record (``runs/<group>/seed<k>.json``
  with curves and ``run_metadata``, and the parameters ``.pt``), raw, gitignored;
* ``<summary_dir>/``: ``summary.md``, ``main.csv``, ``comparison.csv``,
  ``tuning.csv``, ``ngram.csv`` and ``summary.json`` (small, committed).

``--from-run <dir>`` reuses the records in ``<dir>`` and runs only the missing
ones, then redoes the selection and the summary. ``--tasks a,b`` restricts the
run to some tasks (for smoke tests; the committed summary uses all of them).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from qmem.baselines.rnn import matched_hidden_size
from qmem.baselines.runner import BaselineSpec, run_baseline
from qmem.experiment import aggregate, is_success, run_many
from qmem.model import ModelConfig, build_model
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase3a.toml"


# -- cells ---------------------------------------------------------------------------
@dataclass
class Cell:
    """One baseline configuration on one task: a family and size, tuned then evaluated."""

    task: str
    key: str  # e.g. "hmm-grad/D4", "gru/large"
    family: str
    method: str
    model: dict[str, Any]
    state_dim: int
    candidates: list[dict[str, Any]] = field(default_factory=list)  # {"tag", "model", "train"}
    reuse_from: str | None = None  # key of the cell whose choice this one reuses
    labels: dict[str, Any] = field(default_factory=dict)

    @property
    def tuned(self) -> bool:
        return len(self.candidates) > 1 and self.reuse_from is None


def lr_candidates(grid: list[float]) -> list[dict[str, Any]]:
    return [{"tag": f"lr={lr:g}", "model": {}, "train": {"lr": lr}} for lr in grid]


def a_parameter_count(a_cfg: dict[str, Any], task: dict[str, Any], n_memory: int = 2) -> int:
    m = {**a_cfg["model"], "n_memory": n_memory, "n_output": task["n_output"]}
    return build_model(ModelConfig.from_dict(m)).n_parameters()


def build_cells(cfg: dict[str, Any], a_cfg: dict[str, Any], tasks: list[str]) -> list[Cell]:
    lrs = cfg["tuning"]["lr_grid"]
    max_dim = cfg["tuning"]["max_tuned_state_dim"]
    cells: list[Cell] = []
    for name in tasks:
        t = a_cfg["tasks"][name]
        y = 2 ** t["n_output"]
        g = cfg["ngram"]
        cells.append(
            Cell(name, "ngram", "ngram", "count", {"orders": g["orders"], "alphas": g["alphas"]},
                 state_dim=0)
        )  # fmt: skip

        def tuned_cell(key, family, model, dim, reuse_key=None, name=name, **labels):
            # Above max_tuned_state_dim, reuse the choice of the same family at that size.
            reuse = reuse_key if dim > max_dim else None
            cells.append(
                Cell(name, key, family, "gradient", model, dim, lr_candidates(lrs),
                     reuse_from=reuse, labels=labels)
            )  # fmt: skip

        h = cfg["hmm"]
        for d in h["n_states"]:
            m = {"n_states": d, "init_scale": h["init_scale"]}
            cells.append(Cell(name, f"hmm-em/D{d}", "hmm", "em", m, d, labels={"fit": "em"}))
            tuned_cell(f"hmm-grad/D{d}", "hmm", m, d, f"hmm-grad/D{max_dim}", fit="gradient")
        for kind in ("umps", "lps"):
            u = cfg[kind]
            for chi in u["bond_dims"]:
                m = {"bond_dim": chi, **{k: u[k] for k in u if k != "bond_dims"}}
                tuned_cell(f"{kind}/chi{chi}", "umps", m, chi, f"{kind}/chi{max_dim}")
        hq = cfg["hqmm"]
        for nm, ne in hq["sizes"]:
            m = {"n_memory": nm, "n_env": ne, "init_scale": hq["init_scale"]}
            tuned_cell(f"hqmm/D{2**nm}E{2**ne}", "hqmm", m, 2**nm, f"hqmm/D{max_dim}E{2**ne}")
        budget = a_parameter_count(a_cfg, t)
        for cell in cfg["rnn"]["cells"]:
            hm = matched_hidden_size(cell, y, budget)
            tuned_cell(f"{cell}/matched", cell, {"hidden": hm}, hm, arm="matched",
                       a_parameters=budget)  # fmt: skip
            cands = [
                {"tag": f"h={hd},lr={lr:g}", "model": {"hidden": hd}, "train": {"lr": lr}}
                for hd in cfg["rnn"]["large_hidden"]
                for lr in lrs
            ]
            big = max(cfg["rnn"]["large_hidden"])
            cells.append(
                Cell(name, f"{cell}/large", cell, "gradient", {"hidden": big}, big, cands,
                     labels={"arm": "large"})
            )  # fmt: skip
    keys = {(c.task, c.key) for c in cells}
    for c in cells:  # no cell at max_tuned_state_dim to borrow from: tune this one
        if c.reuse_from is not None and (c.task, c.reuse_from) not in keys:
            c.reuse_from = None
    return cells


def _spec(cfg, a_cfg, cell: Cell, stage: str, cand: dict | None, seed: int) -> BaselineSpec:
    t = a_cfg["tasks"][cell.task]
    model = {**cell.model, **(cand["model"] if cand else {})}
    train = {**cfg["train"], **(cand["train"] if cand else {})}
    group = f"{stage}/{cell.task}/{cell.key}"
    if stage == "tune":
        group += "/" + cand["tag"].replace("/", "_")
    labels = {
        "task": cell.task,
        "cell": cell.key,
        "family": cell.family,
        "method": cell.method,
        "stage": stage,
        "candidate": cand["tag"] if cand else "",
        "state_dim": cell.state_dim,
        **cell.labels,
    }
    return BaselineSpec(
        group=group,
        task={**t["spec"], "name": cell.task},
        data=a_cfg["data"],
        family=cell.family,
        model=model,
        method=cell.method,
        train=train if cell.method == "gradient" else {},
        fit=cfg["hmm"]["em"] if cell.method == "em" else {},
        seed=seed,
        labels=labels,
    )


def tuning_specs(cfg, a_cfg, cells: list[Cell]) -> list[BaselineSpec]:
    return [
        _spec(cfg, a_cfg, c, "tune", cand, s)
        for c in cells
        if c.tuned
        for cand in c.candidates
        for s in cfg["experiment"]["tuning_seeds"]
    ]


def select(cells: list[Cell], records: list[dict]) -> tuple[dict, list[dict]]:
    """Choose each tuned cell's candidate by median best-validation NLL over tuning seeds."""
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    choice: dict[tuple[str, str], dict] = {}
    table = []
    for c in cells:
        if not c.tuned:
            continue
        scores = []
        for cand in c.candidates:
            recs = by_group.get(f"tune/{c.task}/{c.key}/{cand['tag']}", [])
            vals = [r.get("best_val_nll", math.inf) for r in recs]
            vals = [v if isinstance(v, float) and math.isfinite(v) else math.inf for v in vals]
            med = statistics.median(vals) if vals else math.inf
            secs = sum(r.get("wall_seconds", 0.0) for r in recs)
            scores.append((med, cand, secs, len(recs)))
        best = min(scores, key=lambda s: s[0])
        choice[(c.task, c.key)] = best[1]
        for med, cand, secs, n in scores:
            table.append(
                {
                    "task": c.task,
                    "cell": c.key,
                    "candidate": cand["tag"],
                    "median_val_nll": med,
                    "n_runs": n,
                    "wall_seconds": secs,
                    "chosen": cand is best[1],
                }
            )
    for c in cells:
        if c.reuse_from is not None:
            src = choice.get((c.task, c.reuse_from))
            if src is not None:
                choice[(c.task, c.key)] = src
    return choice, table


def eval_specs(cfg, a_cfg, cells: list[Cell], choice: dict) -> list[BaselineSpec]:
    specs = []
    for c in cells:
        if c.method == "count":
            specs.append(_spec(cfg, a_cfg, c, "eval", None, 0))  # deterministic: one run
            continue
        cand = choice.get((c.task, c.key))
        if cand is None and c.candidates:
            cand = c.candidates[0] if len(c.candidates) == 1 else None
            if cand is None:
                raise SystemExit(f"no tuning result for {c.task}/{c.key}")
        for s in cfg["experiment"]["seeds"]:
            specs.append(_spec(cfg, a_cfg, c, "eval", cand, s))
    return specs


def a_specs(a_cfg, tasks: list[str]) -> list[BaselineSpec]:
    specs = []
    for name in tasks:
        t = a_cfg["tasks"][name]
        for nm in t["n_memory"]:
            model = {**a_cfg["model"], "n_memory": nm, "n_output": t["n_output"]}
            for seed in a_cfg["experiment"]["seeds"]:
                specs.append(
                    BaselineSpec(
                        group=f"eval/{name}/A/m{nm}",
                        task={**t["spec"], "name": name},
                        data=a_cfg["data"],
                        family="A",
                        model=model,
                        train=a_cfg["train"],
                        seed=seed,
                        labels={
                            "task": name,
                            "cell": f"A/m{nm}",
                            "family": "A",
                            "method": "gradient",
                            "stage": "eval",
                            "candidate": "phase2 recipe",
                            "state_dim": 2**nm,
                        },
                    )
                )
    return specs


# -- running ----------------------------------------------------------------------------
def load_records(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((run_dir / "runs").rglob("seed*.json")):
        out.append(json.loads(path.read_text()))
    return out


def run_stage(specs, records, out_dir, workers, label) -> list[dict]:
    have = {(r["group"], r["seed"]) for r in records}
    for s in specs:
        r = next((r for r in records if (r["group"], r["seed"]) == (s.group, s.seed)), None)
        if r is not None and r["meta"]["config"]["model"] != s.config()["model"]:
            raise SystemExit(f"saved run {s.group} seed {s.seed} has a different config")
    todo = [s for s in specs if (s.group, s.seed) not in have]
    print(f"[{label}] {len(specs)} runs, {len(todo)} to do", flush=True)
    t0 = time.perf_counter()
    new = run_many(todo, out_dir, workers=workers, runner=run_baseline)
    for r in new:
        r.pop("state_dict", None)
    print(f"[{label}] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return records + new


# -- summarising ---------------------------------------------------------------------
def val_selected(recs: list[dict]) -> dict | None:
    ok = [r for r in recs if "test" in r and math.isfinite(r.get("best_val_nll", math.inf))]
    return min(ok, key=lambda r: r["best_val_nll"]) if ok else None


def summarise_eval(records: list[dict], tuning_table: list[dict], success_gap: float):
    evals = [r for r in records if r["group"].startswith("eval/")]
    rows = aggregate(evals, success_gap)
    by_group: dict[str, list[dict]] = {}
    for r in evals:
        by_group.setdefault(r["group"], []).append(r)
    tune_secs: dict[tuple[str, str], float] = {}
    for t in tuning_table:
        k = (t["task"], t["cell"])
        tune_secs[k] = tune_secs.get(k, 0.0) + t["wall_seconds"]
    for row in rows:
        recs = by_group[row["group"]]
        lab = recs[0]["labels"]
        info = next((r["model_info"] for r in recs if "model_info" in r), {})
        sel = val_selected(recs)
        row.update(
            task=lab["task"],
            cell=lab["cell"],
            family=lab["family"],
            method=lab["method"],
            candidate=lab.get("candidate", ""),
            state_dim=info.get("state_dim", lab.get("state_dim")),
            n_parameters=info.get("n_parameters", row.get("n_parameters")),
            flops_per_step=info.get("flops_per_step"),
            selected_seed=sel["seed"] if sel else None,
            selected_val_nll=sel["best_val_nll"] if sel else math.nan,
            selected_gap=sel["test"]["gap_nats"] if sel else math.nan,
            tuning_wall_seconds=tune_secs.get((lab["task"], lab["cell"]), 0.0),
            train_seconds_mean=statistics.mean(r.get("train_seconds", 0.0) for r in recs),
            diagnostics_total={
                k: sum(r.get("diagnostics", {}).get(k, 0) for r in recs)
                for k in sorted({k for r in recs for k in r.get("diagnostics", {})})
            },
        )
        if lab["family"] == "ngram" and recs[0].get("model_info"):
            row["candidate"] = f"k={info['order']}, alpha={info['alpha']:g}"
    return rows


HEADLINE = [  # (label, cell, how): models compared across tasks at state dimension 4
    ("A, 2 qubits (D=4)", "A/m2"),
    ("A, 3 qubits (D=8)", "A/m3"),
    ("HMM D=2 (better of EM/grad)", "hmm/D2"),
    ("HMM D=4 (better of EM/grad)", "hmm/D4"),
    ("HMM D=16 (better of EM/grad)", "hmm/D16"),
    ("uMPS chi=2", "umps/chi2"),
    ("uMPS chi=4", "umps/chi4"),
    ("uMPS chi=16", "umps/chi16"),
    ("LPS chi=4 (kappa=2)", "lps/chi4"),
    ("HQMM D=2", "hqmm/D2E1"),
    ("HQMM D=4", "hqmm/D4E1"),
    ("HQMM D=4, 2 Kraus/outcome", "hqmm/D4E2"),
    ("HQMM D=16", "hqmm/D16E1"),
    ("GRU, A's parameter count", "gru/matched"),
    ("GRU, large", "gru/large"),
    ("RNN, A's parameter count", "rnn/matched"),
    ("RNN, large", "rnn/large"),
    ("n-gram, k on validation", "ngram"),
]


def hmm_better(rows: list[dict], task: str, d: int) -> dict | None:
    """The HMM fit (EM or gradient) whose validation-selected run has lower val NLL."""
    cands = [
        r for r in rows if r["task"] == task and r["cell"] in (f"hmm-em/D{d}", f"hmm-grad/D{d}")
    ]
    return min(cands, key=lambda r: r["selected_val_nll"]) if cands else None


def comparison(rows: list[dict], tasks: list[str]) -> list[dict]:
    out = []
    for label, cell in HEADLINE:
        entry: dict[str, Any] = {"model": label, "cell": cell}
        for t in tasks:
            if cell.startswith("hmm/"):
                r = hmm_better(rows, t, int(cell.split("D")[1]))
            else:
                r = next((r for r in rows if r["task"] == t and r["cell"] == cell), None)
            entry[t] = (
                None
                if r is None
                else {
                    "median_gap": r["gap_median"],
                    "selected_gap": r["selected_gap"],
                    "n_success": r["n_success"],
                    "n_runs": r["n_runs"],
                    "n_parameters": r["n_parameters"],
                    "method": r["method"],
                    "cell": r["cell"],
                }
            )
        out.append(entry)
    return out


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def write_summary(out: Path, cfg, rows, tuning_table, comp, ngram_rows, tasks, meta, extra):
    out.mkdir(parents=True, exist_ok=True)
    main_cols = [
        "task", "cell", "family", "method", "candidate", "state_dim", "n_parameters",
        "flops_per_step", "n_runs", "n_success", "failure_rate", "oracle_nll_nats",
        "test_nll_mean", "gap_mean", "gap_se", "gap_median", "gap_best", "gap_worst",
        "selected_seed", "selected_val_nll", "selected_gap", "steps_mean",
        "train_seconds_mean", "wall_seconds_total", "tuning_wall_seconds",
    ]  # fmt: skip
    with open(out / "main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, main_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(tuning_table[0]) if tuning_table else ["task"])
        w.writeheader()
        w.writerows(tuning_table)
    with open(out / "ngram.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task", "order", "alpha", "val_nll", "test_nll", "gap"])
        w.writeheader()
        w.writerows(ngram_rows)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                *[f"{t} median gap" for t in tasks],
                *[f"{t} selected gap" for t in tasks],
                *[f"{t} success" for t in tasks],
            ]
        )
        for e in comp:
            w.writerow(
                [e["model"]]
                + [fmt(e[t]["median_gap"], 6) if e[t] else "" for t in tasks]
                + [fmt(e[t]["selected_gap"], 6) if e[t] else "" for t in tasks]
                + [f"{e[t]['n_success']}/{e[t]['n_runs']}" if e[t] else "" for t in tasks]
            )  # fmt: skip
    save_json(
        out / "summary.json",
        {"meta": meta, "rows": rows, "tuning": tuning_table, "comparison": comp,
         "ngram": ngram_rows, **extra},
    )  # fmt: skip

    lines = ["# Phase 3a summary (generated by experiments/phase3a.py)", ""]
    lines.append(
        f"Commit `{(meta.get('git_commit') or '?')[:7]}`, dirty: {meta.get('git_dirty')}. "
        f"Total run time {extra['wall_seconds'] / 60:.1f} min wall; "
        f"{extra['cpu_seconds'] / 3600:.2f} CPU-hours summed over runs."
    )
    n_seeds = len(cfg["experiment"]["seeds"])
    gap = cfg["experiment"]["success_gap"]
    lines += ["", f"## Comparison: median test gap over {n_seeds} seeds (nats/symbol), "
              f"and in brackets the number of failed runs (gap ≥ {gap})", ""]  # fmt: skip
    lines.append("| model | " + " | ".join(tasks) + " |")
    lines.append("|---" * (len(tasks) + 1) + "|")
    for e in comp:
        cells = []
        for t in tasks:
            v = e[t]
            if v is None:
                cells.append("-")
                continue
            fails = v["n_runs"] - v["n_success"]
            m = " (EM)" if v["method"] == "em" else ""
            cells.append(f"{fmt(v['median_gap'])} ({fails}){m}")
        lines.append(f"| {e['model']} | " + " | ".join(cells) + " |")
    lines += ["", f"## Validation-selected run (lowest best-validation NLL of the {n_seeds} "
              "seeds): its test gap", ""]  # fmt: skip
    lines.append("| model | " + " | ".join(tasks) + " |")
    lines.append("|---" * (len(tasks) + 1) + "|")
    for e in comp:
        cells = [fmt(e[t]["selected_gap"]) if e[t] else "-" for t in tasks]
        lines.append(f"| {e['model']} | " + " | ".join(cells) + " |")
    lines += ["", "## Every evaluated cell", ""]
    for t in tasks:
        lines += [f"### {t}", ""]
        lines.append(
            "| cell | method | chosen | D | params | flops/step | failed | median gap | "
            "mean ± SE | selected gap | steps | train s/run | tuning s |"
        )
        lines.append("|---" * 13 + "|")
        for r in [r for r in rows if r["task"] == t]:
            lines.append(
                f"| {r['cell']} | {r['method']} | {r['candidate']} | {r['state_dim']} | "
                f"{r['n_parameters']} | {r['flops_per_step']} | "
                f"{r['n_runs'] - r['n_success']}/{r['n_runs']} | {fmt(r['gap_median'])} | "
                f"{fmt(r['gap_mean'])} ± {fmt(r['gap_se'])} | {fmt(r['selected_gap'])} | "
                f"{r['steps_mean']:.0f} | {r['train_seconds_mean']:.1f} | "
                f"{r['tuning_wall_seconds']:.0f} |"
            )
        lines.append("")
    lines += ["## n-gram test gap by order (alpha chosen on validation per order)", ""]
    orders = sorted({r["order"] for r in ngram_rows})
    lines.append("| task | " + " | ".join(f"k={k}" for k in orders) + " | chosen |")
    lines.append("|---" * (len(orders) + 2) + "|")
    for t in tasks:
        rs = {r["order"]: r for r in ngram_rows if r["task"] == t}
        chosen = next((r["candidate"] for r in rows if r["task"] == t and r["cell"] == "ngram"),
                      "")  # fmt: skip
        lines.append(f"| {t} | " + " | ".join(fmt(rs[k]["gap"]) for k in orders) +
                     f" | {chosen} |")  # fmt: skip
    if extra.get("a_reproduction"):
        rep = extra["a_reproduction"]
        lines += ["", "## A reproduced against Phase 2", "",
                  "Per-seed test gaps compared with `results/summaries/phase2/summary.json`: "
                  f"{rep['n_compared']} runs, "
                  f"max |difference| = {rep['max_abs_diff']:.2e}."]  # fmt: skip
    lines += ["", "## Tuning", "",
              f"Learning-rate grid {cfg['tuning']['lr_grid']}, tuning seeds "
              f"{cfg['experiment']['tuning_seeds']}; candidates ranked by median best-validation "
              "NLL. Cells above state dimension "
              f"{cfg['tuning']['max_tuned_state_dim']} reuse the choice at that dimension. "
              "Full table in `tuning.csv`.", ""]  # fmt: skip
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def a_reproduction(rows_records: list[dict]) -> dict | None:
    """Compare re-run A gaps per seed with the committed Phase 2 summary."""
    path = REPO_ROOT / "results" / "summaries" / "phase2" / "summary.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    want = {}
    for a in old.get("analyses", []):
        _, task, m = a["group"].split("/")
        for pr in a["per_run"]:
            want[(task, m, pr["seed"])] = pr["gap_nats"]
    diffs = []
    for r in rows_records:
        if r.get("family") != "A" or "test" not in r:
            continue
        _, task, _, m = r["group"].split("/")
        if (task, m, r["seed"]) in want:
            diffs.append(abs(r["test"]["gap_nats"] - want[(task, m, r["seed"])]))
    if not diffs:
        return None
    return {"n_compared": len(diffs), "max_abs_diff": max(diffs)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--from-run", type=Path, default=None)
    ap.add_argument("--tasks", default=None, help="comma-separated subset (smoke tests)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summary-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.config.read_text())
    a_cfg = tomllib.loads((REPO_ROOT / cfg["experiment"]["a_config"]).read_text())
    tasks = list(a_cfg["tasks"]) if args.tasks is None else args.tasks.split(",")
    workers = args.workers or cfg["experiment"]["workers"]
    out_dir = args.from_run or new_run_dir(cfg["experiment"]["name"])
    summary_dir = args.summary_dir or REPO_ROOT / cfg["experiment"]["summary_dir"]
    meta = run_metadata({"phase3a": cfg, "a_config": a_cfg, "tasks": tasks}, None)
    save_json(out_dir / "config.json", meta)
    print(f"run dir: {out_dir}", flush=True)
    t0 = time.perf_counter()

    records = load_records(out_dir) if args.from_run else []
    cells = build_cells(cfg, a_cfg, tasks)
    if cfg["experiment"].get("run_a", True):
        records = run_stage(a_specs(a_cfg, tasks), records, out_dir, workers, "A")
    records = run_stage(tuning_specs(cfg, a_cfg, cells), records, out_dir, workers, "tuning")
    choice, tuning_table = select(cells, records)
    records = run_stage(eval_specs(cfg, a_cfg, cells, choice), records, out_dir, workers, "eval")

    success_gap = cfg["experiment"]["success_gap"]
    rows = summarise_eval(records, tuning_table, success_gap)
    comp = comparison(rows, tasks)
    ngram_rows = []
    for r in records:
        if r.get("family") == "ngram" and "per_order" in r:
            t = r["labels"]["task"]
            for k, v in sorted(r["per_order"].items(), key=lambda kv: int(kv[0])):
                ngram_rows.append(
                    {"task": t, "order": int(k), "alpha": v["alpha"], "val_nll": v["val_nll"],
                     "test_nll": v["test"]["nll_nats"], "gap": v["test"]["gap_nats"]}
                )  # fmt: skip
    extra = {
        "wall_seconds": time.perf_counter() - t0,
        "cpu_seconds": sum(r.get("wall_seconds", 0.0) for r in records),
        "n_records": len(records),
        "n_errors": sum(r.get("status") == "error" for r in records),
        "errors": [
            {"group": r["group"], "seed": r["seed"], "message": r.get("message")}
            for r in records
            if r.get("status") == "error"
        ],
        "a_reproduction": a_reproduction(records),
        "run_dir": str(out_dir.relative_to(REPO_ROOT)) if out_dir.is_relative_to(REPO_ROOT)
        else str(out_dir),
        "success_counts": {
            r["group"]: sum(is_success(x, success_gap) for x in records if x["group"] == r["group"])
            for r in rows
        },
    }  # fmt: skip
    save_json(out_dir / "aggregate.json", {"rows": rows, "comparison": comp, **extra})
    write_summary(summary_dir, cfg, rows, tuning_table, comp, ngram_rows, tasks, meta, extra)
    print(f"summary: {summary_dir}", flush=True)
    print((summary_dir / "summary.md").read_text().split("## Every")[0])


if __name__ == "__main__":
    torch.set_num_threads(1)
    sys.exit(main())
