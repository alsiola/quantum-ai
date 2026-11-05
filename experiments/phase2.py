"""Phase 2 (issue #5): train architecture A on sources with known optima, many seeds.

Runs every (task, memory size, seed) in the config, then the variants, then
the analyses (generation check, interventions, order-k Markov baselines,
memory trajectories, instrument spectra, training pathologies), and writes:

* ``results/phase2/<timestamp>/``: every run record (``runs/<group>/seed<k>.json``
  with curves and ``run_metadata``, and the trained parameters ``.pt``), the
  aggregates and the analyses (raw, gitignored);
* ``<summary_dir>/``: ``summary.md``, ``main.csv``, ``variants.csv``,
  ``summary.json`` and the plots (small, committed).

``--from-run <dir>`` reuses the records saved in ``<dir>``, trains only the
runs that are missing (e.g. after an interrupted run), then redoes the
aggregation, analyses and plots. Saved records must match the config.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from qmem.analysis import (
    context_table,
    curve_pathologies,
    instrument_summary,
    intervention_effects,
    markov_baseline_nll,
    memory_trajectories,
    prefix_predictions,
    sample_statistics,
)
from qmem.experiment import (
    RunSpec,
    aggregate,
    build_splits,
    is_success,
    load_model,
    run_many,
)
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import make_task

sys.path.insert(0, str(Path(__file__).parent))
import phase2_plots

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase2.toml"
LN2 = math.log(2)


# -- building the runs ----------------------------------------------------------
def main_specs(cfg: dict[str, Any]) -> list[RunSpec]:
    specs = []
    for name, t in cfg["tasks"].items():
        for nm in t["n_memory"]:
            model = {**cfg["model"], "n_memory": nm, "n_output": t["n_output"]}
            for seed in cfg["experiment"]["seeds"]:
                specs.append(
                    RunSpec(
                        group=f"main/{name}/m{nm}",
                        task={**t["spec"], "name": name},
                        data=cfg["data"],
                        model=model,
                        train=cfg["train"],
                        seed=seed,
                        labels={"task": name, "n_memory": nm, "variant": "recipe"},
                    )
                )
    return specs


def variant_specs(cfg: dict[str, Any]) -> list[RunSpec]:
    vs = cfg.get("variant_settings")
    if not vs:
        return []
    specs = []
    for name in vs["tasks"]:
        t = cfg["tasks"][name]
        for v in cfg["variants"]:
            model = {**cfg["model"], "n_memory": 2, "n_output": t["n_output"], **v.get("model", {})}
            train = {**cfg["train"], **v.get("train", {})}
            for seed in vs["seeds"]:
                specs.append(
                    RunSpec(
                        group=f"variants/{name}/{v['name'].replace('/', '_')}",
                        task={**t["spec"], "name": name},
                        data=cfg["data"],
                        model=model,
                        train=train,
                        seed=seed,
                        labels={"task": name, "n_memory": 2, "variant": v["name"]},
                    )
                )
    return specs


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "runs").rglob("seed*.json")):
        rec = json.loads(path.read_text())
        pt = path.with_suffix(".pt")
        if pt.exists():
            rec["state_dict"] = torch.load(pt)
        records.append(rec)
    return records


def check_records_match(records: list[dict[str, Any]], specs: list[RunSpec]) -> None:
    """Refuse to mix saved records made with a different config into this summary."""
    want = {(s.group, s.seed): s.config() for s in specs}
    for r in records:
        spec = want.get((r["group"], r["seed"]))
        got = r["meta"]["config"]
        if spec is not None and any(got[k] != spec[k] for k in ("task", "data", "model", "train")):
            raise SystemExit(f"saved run {r['group']} seed {r['seed']} has a different config")


def order_records(records: list[dict[str, Any]], specs: list[RunSpec]) -> list[dict[str, Any]]:
    by_key = {(r["group"], r["seed"]): r for r in records}
    return [by_key[(s.group, s.seed)] for s in specs if (s.group, s.seed) in by_key]


# -- analyses ---------------------------------------------------------------------
def pick_representative(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The run with the best *validation* NLL (never selected on test)."""
    ok = [r for r in recs if "state_dict" in r and math.isfinite(r.get("best_val_nll", math.inf))]
    return min(ok, key=lambda r: r["best_val_nll"]) if ok else None


def analyse_group(
    recs: list[dict[str, Any]], cfg: dict[str, Any], success_gap: float
) -> dict[str, Any]:
    a = cfg["analysis"]
    spec = recs[0]["meta"]["config"]
    task = make_task(spec["task"])
    splits = build_splits(task, spec["data"])
    length = splits.test.shape[1]
    out: dict[str, Any] = {"group": recs[0]["group"], "labels": recs[0]["labels"]}

    # pathologies and instruments for every run
    per_run = []
    for r in recs:
        row: dict[str, Any] = {
            "seed": r["seed"],
            "status": r["status"],
            "success": is_success(r, success_gap),
        }
        if "curves" in r:
            row.update(curve_pathologies(r["curves"]))
            row["min_p_observed_test"] = r["test"]["min_p_observed"]
            row["diagnostics"] = r.get("diagnostics", {})
            row["strength_final"] = r.get("strength_final")
            row["gap_nats"] = r["test"]["gap_nats"]
        if "state_dict" in r:
            model = load_model(r)
            inst = instrument_summary(model)
            eff_max = [max(e) for e in inst["effect_eigenvalues"]]
            row["never_emitted_outcomes"] = sum(m < 1e-3 for m in eff_max)
            row["kraus_ranks"] = inst["kraus_ranks"]
            row["memoryless_instrument"] = all(k == 1 for k in inst["kraus_ranks"])
            lag2 = intervention_effects(
                model,
                task,
                splits.test,
                [1, 2],
                a["intervention_prefix_len"],
                a["intervention_seed"],
            )
            row["intervention_lag1_tv"] = lag2[0]["model_tv"]
            row["intervention_lag2_tv"] = lag2[1]["model_tv"]
        per_run.append(row)
    out["per_run"] = per_run

    rep = pick_representative(recs)
    if rep is None:
        out["representative"] = None
        return out
    model = load_model(rep)
    out["representative"] = {
        "seed": rep["seed"],
        "test": rep["test"],
        "best_val_nll": rep["best_val_nll"],
    }

    # generation check: the separate sampling path vs the source
    gen = model.sample(a["n_samples"], length, seed=a["sample_seed"])
    src = task.sample(a["n_samples"], length, seed=a["source_sample_seed"])
    ref = task.sample(a["n_samples"], length, seed=a["reference_sample_seed"])
    out["generation"] = sample_statistics(gen, src, ref, task.n_symbols, max_n=3)
    out["generation"]["source_nll_of_model_samples"] = float(-task.log_probs(gen).mean())
    out["generation"]["impossible_model_samples"] = int(
        (~torch.isfinite(task.log_probs(gen).sum(-1))).sum()
    )

    with torch.no_grad():
        out["interventions"] = intervention_effects(
            model,
            task,
            splits.test,
            a["intervention_lags"],
            a["intervention_prefix_len"],
            a["intervention_seed"],
        )
        k = 2 if task.n_symbols == 2 else 1
        out["context_table"] = context_table(model, task, splits.test, k)
        out["memory"] = memory_trajectories(model, splits.test)
        out["instrument"] = instrument_summary(model)
        if spec["task"]["kind"] in ("even", "golden_mean"):
            # p(next | 0 then n ones): Even alternates (n odd -> must emit 1)
            pres = [[0, 0, 0] + [1] * n for n in range(11)]
            out["prefix_predictions"] = prefix_predictions(model, task, pres)
    out["markov_baselines"] = [
        markov_baseline_nll(splits.train, splits.val, splits.test, k, task.n_symbols)
        for k in a["markov_orders"]
    ]
    out["oracle_test_nll"] = float(-task.log_probs(splits.test).mean())
    out["optimal_nll"] = task.optimal_nll(length)
    return out


# -- summaries ---------------------------------------------------------------------
def fmt(x: float, nd: int = 4) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if k not in ("labels", "statuses", "gap_ci95")}
        f.update({f"label_{k}": v for k, v in r["labels"].items()})
        f["gap_ci95_lo"], f["gap_ci95_hi"] = r["gap_ci95"]
        f["statuses"] = ";".join(f"{k}={v}" for k, v in r["statuses"].items())
        flat.append(f)
    keys = list(dict.fromkeys(k for f in flat for k in f))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def main_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| task | mem qubits | optimum H/T (nats) | oracle test NLL | model test NLL "
        "(mean ± SE) | gap to oracle (mean ± SE) | gap 95% CI | median gap | failed / runs "
        "| accuracy (model / oracle) | params | wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["gap_ci95"]
        lines.append(
            f"| {r['labels']['task']} | {r['labels']['n_memory']} | {fmt(r['optimal_nll_nats'])} "
            f"| {fmt(r['oracle_nll_nats'])} | {fmt(r['test_nll_mean'])} ± {fmt(r['test_nll_se'])} "
            f"| {fmt(r['gap_mean'])} ± {fmt(r['gap_se'])} | [{fmt(lo)}, {fmt(hi)}] "
            f"| {fmt(r['gap_median'])} | {r['n_runs'] - r['n_success']} / {r['n_runs']} "
            f"| {fmt(r['accuracy_mean'], 3)} / {fmt(r['oracle_accuracy'], 3)} "
            f"| {r['n_parameters']} | {r['wall_seconds_total']:.0f} |"
        )
    return lines


def variant_table(rows: list[dict[str, Any]]) -> list[str]:
    tasks = list(dict.fromkeys(r["labels"]["task"] for r in rows))
    variants = list(dict.fromkeys(r["labels"]["variant"] for r in rows))
    cell = {(r["labels"]["task"], r["labels"]["variant"]): r for r in rows}
    lines = [
        "| variant | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for v in variants:
        cells = []
        for t in tasks:
            r = cell.get((t, v))
            if r is None:
                cells.append("")
                continue
            cells.append(
                f"{fmt(r['gap_median'])} ({r['n_runs'] - r['n_success']}/{r['n_runs']} failed)"
            )
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return lines


def analysis_lines(analyses: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Generation check (representative seed, `model.sample`)",
        "",
        "TV distance between n-gram distributions of 2000 model samples and 2000 source "
        "samples; the noise floor is the same distance for a second, independent source "
        "sample.",
        "",
        "| group | seed | TV 1-gram (floor) | TV 2-gram (floor) | TV 3-gram (floor) "
        "| odd 1-runs model / source | impossible samples |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in analyses:
        g = a.get("generation")
        if not g:
            continue
        tv, fl = g["ngram_tv"], g["ngram_tv_noise_floor"]
        odd = (
            f"{fmt(g.get('odd_run_fraction_model', math.nan), 3)} / "
            f"{fmt(g.get('odd_run_fraction_source', math.nan), 3)}"
        )
        lines.append(
            f"| {a['group']} | {a['representative']['seed']} | "
            + " | ".join(f"{fmt(tv[n], 3)} ({fmt(fl[n], 3)})" for n in (1, 2, 3))
            + f" | {odd} | {g['impossible_model_samples']} |"
        )
    lines += [
        "",
        "## Interventions: TV change of p(next) when the symbol at lag l is changed",
        "",
        "Model (representative seed) / exact source, on test prefixes of length 24.",
        "",
    ]
    lags = [r["lag"] for r in next(a for a in analyses if a.get("interventions"))["interventions"]]
    lines.append("| group | " + " | ".join(f"lag {k}" for k in lags) + " |")
    lines.append("|---|" + "---|" * len(lags))
    for a in analyses:
        if a.get("interventions"):
            lines.append(
                f"| {a['group']} | "
                + " | ".join(
                    f"{r['model_tv']:.3f} / {r['source_tv']:.3f}" for r in a["interventions"]
                )
                + " |"
            )
    lines += ["", "## Order-k Markov baselines on the same splits (test NLL, nats/symbol)", ""]
    ks = [b["k"] for b in analyses[0]["markov_baselines"]]
    lines.append("| group | oracle | model (rep.) | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|---|---|---|" + "---|" * len(ks))
    for a in analyses:
        if a.get("representative"):
            lines.append(
                f"| {a['group']} | {fmt(a['oracle_test_nll'])} | "
                f"{fmt(a['representative']['test']['nll_nats'])} | "
                + " | ".join(fmt(b["test_nll"]) for b in a["markov_baselines"])
                + " |"
            )
    lines += [
        "",
        "## Memory along test trajectories (representative seed)",
        "",
        "Mean over test sequences and steps t = 1..T. `NLL from E^t(rho0)` predicts each "
        "symbol from the outcome-averaged memory instead of the conditional one.",
        "",
        "| group | purity of rho_t | purity of E^t(rho0) | D(rho_t, E^t(rho0)) "
        "| NLL conditioned | NLL from E^t(rho0) |",
        "|---|---|---|---|---|---|",
    ]
    for a in analyses:
        m = a.get("memory")
        if not m:
            continue
        mean = lambda xs: sum(xs[1:]) / len(xs[1:])  # noqa: E731
        lines.append(
            f"| {a['group']} | {mean(m['conditional_purity']):.3f} | "
            f"{mean(m['unconditioned_purity']):.3f} | {mean(m['distance_to_unconditioned']):.3f} | "
            f"{sum(m['nll_conditioned']) / len(m['nll_conditioned']):.4f} | "
            f"{sum(m['nll_unconditioned']) / len(m['nll_unconditioned']):.4f} |"
        )
    lines += [
        "",
        "## Training pathologies (all runs in the group)",
        "",
        "| group | runs with grad spikes | max grad norm | median grad norm "
        "| longest val plateau (steps, median) | tiny/zero-p steps | memoryless instruments "
        "| never-emitted outcomes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in analyses:
        pr = [r for r in a["per_run"] if "grad_spikes" in r]
        if not pr:
            continue
        plateaus = sorted(r["longest_plateau_steps"] for r in pr)
        tiny = sum(sum(r["diagnostics"].values()) for r in pr)
        lines.append(
            f"| {a['group']} | {sum(r['grad_spikes'] > 0 for r in pr)}/{len(pr)} | "
            f"{max(r['grad_norm_max'] for r in pr):.2f} | "
            f"{sorted(r['grad_norm_median'] for r in pr)[len(pr) // 2]:.3f} | "
            f"{plateaus[len(plateaus) // 2]} | {tiny} | "
            f"{sum(bool(r.get('memoryless_instrument')) for r in pr)} | "
            f"{sum(r.get('never_emitted_outcomes', 0) for r in pr)} |"
        )
    return lines


def write_summary(
    summary_dir: Path, cfg: dict[str, Any], main_rows, var_rows, analyses, meta, wall
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "main.csv", main_rows)
    if var_rows:
        write_csv(summary_dir / "variants.csv", var_rows)
    slim = copy.deepcopy(analyses)
    for a in slim:
        for r in a.get("per_run", []):
            r.pop("kraus_ranks", None)
    save_json(
        summary_dir / "summary.json",
        {
            "meta": meta,
            "main": main_rows,
            "variants": var_rows,
            "analyses": slim,
            "wall_seconds": wall,
        },
    )
    sg = cfg["experiment"]["success_gap"]
    lines = [
        "# Phase 2 summary (generated by `experiments/phase2.py`)",
        "",
        f"Commit `{meta['git_commit']}` (dirty: {meta['git_dirty']}), {meta['timestamp_utc']}, "
        f"total wall time {wall / 60:.1f} min with {cfg['experiment']['workers']} workers.",
        "",
        f"All NLLs in nats per symbol on {cfg['data']['n_test']} test sequences of length "
        f"{cfg['data']['length']}. *Gap* = model test NLL - the source's own NLL on the same "
        f"sequences (paired; its expectation is the per-symbol KL). A run **fails** if it "
        f"crashed, diverged or its gap is >= {sg}. Mean ± SE and the bootstrap CI are over "
        "all runs, failures included.",
        "",
        "## Main results (recipe)",
        "",
        *main_table(main_rows),
        "",
    ]
    if var_rows:
        lines += [
            "## Variants (2 memory qubits): median gap (failed/runs)",
            "",
            *variant_table(var_rows),
            "",
        ]
    lines += analysis_lines(analyses)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")


# -- entry point --------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=None, help="override [experiment].workers")
    p.add_argument("--from-run", default=None, help="reuse records in this results directory")
    p.add_argument("--skip-variants", action="store_true")
    args = p.parse_args()
    start = time.perf_counter()
    cfg = tomllib.loads(Path(args.config).read_text())
    exp = cfg["experiment"]
    workers = args.workers or exp["workers"]
    specs_main = main_specs(cfg)
    specs_var = [] if args.skip_variants else variant_specs(cfg)

    if args.from_run:
        out = Path(args.from_run)
        records = load_records(out)
        check_records_match(records, specs_main + specs_var)
        have = {(r["group"], r["seed"]) for r in records}
        missing = [s for s in specs_main + specs_var if (s.group, s.seed) not in have]
        if missing:
            print(f"{len(records)} saved records; running {len(missing)} missing -> {out}")
            records += run_many(missing, out, workers=workers)
    else:
        out = new_run_dir(exp["name"])
        save_json(out / "config.json", {"meta": run_metadata(cfg), "config_path": args.config})
        print(f"{len(specs_main)} main runs + {len(specs_var)} variant runs -> {out}")
        records = run_many(specs_main + specs_var, out, workers=workers)
    rec_main = order_records(records, specs_main)
    rec_var = order_records(records, specs_var)

    main_rows = aggregate(rec_main, exp["success_gap"])
    var_rows = aggregate(rec_var, exp["success_gap"]) if rec_var else []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rec_main:
        groups.setdefault(r["group"], []).append(r)
    print("analysing ...")
    analyses = [analyse_group(recs, cfg, exp["success_gap"]) for recs in groups.values()]

    wall = time.perf_counter() - start
    meta = run_metadata(cfg)
    save_json(out / "aggregate.json", {"meta": meta, "main": main_rows, "variants": var_rows})
    save_json(out / "analyses.json", {"meta": meta, "analyses": analyses})
    summary_dir = REPO_ROOT / exp["summary_dir"]
    write_summary(summary_dir, cfg, main_rows, var_rows, analyses, meta, wall)
    phase2_plots.make_all(summary_dir, rec_main, rec_var, analyses, exp["success_gap"])
    print("\n".join(main_table(main_rows)))
    print(f"\nwrote {summary_dir} and {out} ({wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
