"""Phase 3d (issue #9): the feed-forward layout and operational strength metrics.

Trains every (layout, task, seed) in the config on the Phase 2 tasks and data
splits, then scores each trained instrument with ``qmem.metrics`` (information
gain, disturbance, the Banaszek trade-off) and writes:

* ``results/phase3d/<timestamp>/``: every run record with ``run_metadata`` and
  its trained parameters (``.pt``), plus ``metrics.json`` (raw, gitignored);
* ``<summary_dir>/``: ``summary.md``, ``training.csv``, ``metrics.csv``,
  ``summary.json`` and the plots (small, committed).

``--from-run <dir>`` reuses saved records, trains only the missing runs, then
redoes the metrics, summary and plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from qmem.experiment import RunSpec, aggregate, is_success, load_model, mean_se, run_many
from qmem.metrics import operational_summary
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json

sys.path.insert(0, str(Path(__file__).parent))
import phase2
import phase3d_plots

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase3d.toml"
METRIC_KEYS = (
    "operational_strength",
    "tv_gain",
    "mutual_information_stationary",
    "estimation_fidelity",
    "operation_fidelity",
    "operation_fidelity_uncorrected",
    "disturbance",
)


def slug(name: str) -> str:
    return name.replace(" ", "").replace("/", "_").replace(",", "_")


def build_specs(cfg: dict[str, Any], tasks_cfg: dict[str, Any]) -> list[RunSpec]:
    exp = cfg["experiment"]
    specs = []
    for layout in cfg["layouts"]:
        for name, t in tasks_cfg["tasks"].items():
            model = {
                **cfg["model"],
                "n_memory": exp["n_memory"],
                "n_output": t["n_output"],
                **layout["model"],
            }
            for seed in exp["seeds"]:
                specs.append(
                    RunSpec(
                        group=f"{slug(layout['name'])}/{name}",
                        task={**t["spec"], "name": name},
                        data=tasks_cfg["data"],
                        model=model,
                        train=cfg["train"],
                        seed=seed,
                        labels={"task": name, "layout": layout["name"]},
                    )
                )
    return specs


def run_metrics(rec: dict[str, Any], success_gap: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": rec["group"],
        "seed": rec["seed"],
        "labels": rec["labels"],
        "status": rec["status"],
        "success": is_success(rec, success_gap),
        "gap_nats": rec.get("test", {}).get("gap_nats", math.nan),
        "strength_final": rec.get("strength_final"),
    }
    if "state_dict" in rec:
        with torch.no_grad():
            row.update(operational_summary(load_model(rec).instrument()))
    return row


def metric_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in per_run:
        groups.setdefault(r["group"], []).append(r)
    rows = []
    for group, rs in groups.items():
        ok = [r for r in rs if "operational_strength" in r]
        row: dict[str, Any] = {"group": group, "labels": rs[0]["labels"], "n_runs": len(rs)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in ok]
            row[f"{k}_mean"], row[f"{k}_se"] = mean_se(vals)
            row[f"{k}_min"] = min(vals, default=math.nan)
            row[f"{k}_max"] = max(vals, default=math.nan)
        succ = [r["operational_strength"] for r in ok if r["success"]]
        row["operational_strength_mean_successful"] = mean_se(succ)[0]
        g = [x for r in rs for x in (r["strength_final"] or [])]
        row["g_final_mean"] = mean_se(g)[0]
        row["g_final"] = g
        rows.append(row)
    return rows


# -- summary ---------------------------------------------------------------------------
fmt = phase2.fmt


def _cells(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], dict]:
    tasks = list(dict.fromkeys(r["labels"]["task"] for r in rows))
    layouts = list(dict.fromkeys(r["labels"]["layout"] for r in rows))
    return tasks, layouts, {(r["labels"]["task"], r["labels"]["layout"]): r for r in rows}


def training_table(rows: list[dict[str, Any]]) -> list[str]:
    tasks, layouts, cell = _cells(rows)
    lines = [
        "| task | optimum H/T | " + " | ".join(layouts) + " |",
        "|---|---|" + "---|" * len(layouts),
    ]
    for t in tasks:
        first = cell[(t, layouts[0])]
        cs = []
        for lay in layouts:
            r = cell.get((t, lay))
            cs.append(
                ""
                if r is None
                else f"{fmt(r['gap_mean'])} ± {fmt(r['gap_se'])}; median {fmt(r['gap_median'])}; "
                f"**{r['n_runs'] - r['n_success']}/{r['n_runs']}** failed"
            )
        lines.append(f"| {t} | {fmt(first['optimal_nll_nats'])} | " + " | ".join(cs) + " |")
    return lines


def metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| layout | task | eta (mean ± SE) [min, max] | eta, successful runs | TV gain "
        "| I(R:Y) stationary (nats) | disturbance 1-F (corrected) | 1-F uncorrected "
        "| learned g (mean) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['labels']['layout']} | {r['labels']['task']} "
            f"| {fmt(r['operational_strength_mean'], 3)} ± {fmt(r['operational_strength_se'], 3)} "
            f"[{fmt(r['operational_strength_min'], 2)}, {fmt(r['operational_strength_max'], 2)}] "
            f"| {fmt(r['operational_strength_mean_successful'], 3)} "
            f"| {fmt(r['tv_gain_mean'], 3)} "
            f"| {fmt(r['mutual_information_stationary_mean'], 3)} "
            f"| {fmt(r['disturbance_mean'], 3)} "
            f"| {fmt(1 - r['operation_fidelity_uncorrected_mean'], 3)} "
            f"| {fmt(r['g_final_mean'], 3)} |"
        )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        f = {
            k: (";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v)
            for k, v in r.items()
            if k not in ("labels", "statuses", "gap_ci95")
        }
        f.update({f"label_{k}": v for k, v in r["labels"].items()})
        if "gap_ci95" in r:
            f["gap_ci95_lo"], f["gap_ci95_hi"] = r["gap_ci95"]
        flat.append(f)
    keys = list(dict.fromkeys(k for f in flat for k in f))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def write_summary(summary_dir, cfg, train_rows, met_rows, per_run, meta, wall, commits) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "training.csv", train_rows)
    write_csv(summary_dir / "metrics.csv", met_rows)
    slim = [{k: v for k, v in r.items() if k != "stationary"} for r in per_run]
    save_json(
        summary_dir / "summary.json",
        {
            "meta": meta,
            "training": train_rows,
            "metrics": met_rows,
            "per_run": slim,
            "wall_seconds": wall,
        },
    )
    sg = cfg["experiment"]["success_gap"]
    run_seconds = sum(r["wall_seconds_total"] for r in train_rows)
    workers = cfg["experiment"]["workers"]
    stat = [r["stationary"] for r in per_run if "stationary" in r]
    lines = [
        "# Phase 3d summary (generated by `experiments/phase3d.py`)",
        "",
        f"Summary written at commit `{meta['git_commit']}` (dirty: {meta['git_dirty']}), "
        f"{meta['timestamp_utc']}. Runs trained at commit(s) "
        f"{', '.join(f'`{c[:7]}`' for c in sorted(set(commits)) if c)}; per-run wall times "
        f"sum to {run_seconds / 60:.0f} min (~{run_seconds / 60 / workers:.0f} min with "
        f"{workers} workers); this invocation took {wall / 60:.1f} min.",
        "",
        f"{cfg['experiment']['n_memory']} memory qubits, Phase 2 data splits, "
        f"{len(cfg['experiment']['seeds'])} seeds per cell. Gap = test NLL minus the "
        f"source's own NLL on the same sequences (nats/symbol); a run fails if it crashed, "
        f"diverged or its gap is >= {sg}.",
        "",
        "## Training: gap to oracle, mean ± SE; median; failed/runs",
        "",
        *training_table(train_rows),
        "",
        "## Operational metrics of the trained instruments (all runs)",
        "",
        "eta = operational strength (0: no information, 1: as sharp as the alphabet allows "
        "on 4 levels); TV gain = max over input pairs of the TV distance between outcome "
        "distributions; I(R:Y) = mutual information between outcome and a purification of "
        "the stationary memory state; 1-F = Banaszek operation infidelity after the best "
        "unitary correction per outcome (the uncorrected value is also given). See "
        "`docs/phase3d-strength.md` for definitions.",
        "",
        *metrics_table(met_rows),
        "",
        f"Stationary states: max residual {max(s['residual'] for s in stat):.1e}; "
        f"{sum(s['n_unit_eigenvalues'] > 1 for s in stat)} of {len(stat)} channels have a "
        "degenerate fixed point (minimum-norm fixed point used).",
        "",
    ]
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")


# -- entry point ------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--from-run", default=None, help="reuse records in this results directory")
    args = p.parse_args()
    start = time.perf_counter()
    cfg = tomllib.loads(Path(args.config).read_text())
    exp = cfg["experiment"]
    tasks_cfg = tomllib.loads((REPO_ROOT / exp["tasks_from"]).read_text())
    workers = args.workers or exp["workers"]
    specs = build_specs(cfg, tasks_cfg)

    if args.from_run:
        out = Path(args.from_run)
        records = phase2.load_records(out)
        phase2.check_records_match(records, specs)
        have = {(r["group"], r["seed"]) for r in records}
        missing = [s for s in specs if (s.group, s.seed) not in have]
        if missing:
            print(f"{len(records)} saved records; running {len(missing)} missing -> {out}")
            records += run_many(missing, out, workers=workers)
    else:
        out = new_run_dir(exp["name"])
        save_json(
            out / "config.json",
            {"meta": run_metadata(cfg), "config_path": args.config, "tasks": tasks_cfg},
        )
        print(f"{len(specs)} runs -> {out}")
        records = run_many(specs, out, workers=workers)
    records = phase2.order_records(records, specs)

    train_rows = aggregate(records, exp["success_gap"])
    print("scoring instruments ...")
    per_run = [run_metrics(r, exp["success_gap"]) for r in records]
    met_rows = metric_rows(per_run)
    wall = time.perf_counter() - start
    meta = run_metadata(cfg)
    save_json(out / "metrics.json", {"meta": meta, "per_run": per_run, "metrics": met_rows})
    summary_dir = REPO_ROOT / exp["summary_dir"]
    commits = [r["meta"]["git_commit"] for r in records]
    write_summary(summary_dir, cfg, train_rows, met_rows, per_run, meta, wall, commits)
    phase3d_plots.make_all(summary_dir, per_run)
    print("\n".join(training_table(train_rows)))
    print(f"\nwrote {summary_dir} and {out} ({wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
