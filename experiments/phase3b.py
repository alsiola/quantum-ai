"""Phase 3b (issue #7): quantum ablations A / B / C_k / D / hybrid on the Phase 2 tasks.

Trains every (task, variant, seed) in the config with the Phase 2 recipe and
trainer, then analyses the trained models and writes:

* ``results/phase3b/<timestamp>/``: every run record (``runs/<group>/seed<k>.json``
  with curves and ``run_metadata``, and the trained parameters ``.pt``), and
  ``analysis.json`` (raw, gitignored);
* ``<summary_dir>/``: ``summary.md``, ``main.csv``, ``summary.json`` and
  ``gaps.png`` (small, committed).

Analyses: the gap of each variant to its own bound (the source restarted every
k tokens for C_k and B; the source itself otherwise), A-versus-D comparisons,
A's outcome-averaged prediction at its trained parameters (B at A's
parameters), intervention lag profiles and a generation check for the
representative seed (best validation NLL) of every group, conditioned vs
unconditioned memory diagnostics for A, and per-token resource counts.

``--from-run <dir>`` reuses the records in ``<dir>`` and trains only the missing runs.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from qmem.ablations import OutcomeAveragedModel, resource_counts
from qmem.analysis import intervention_effects, memory_trajectories, sample_statistics
from qmem.experiment import (
    RunSpec,
    aggregate,
    build_run_model,
    build_splits,
    load_model,
    mean_se,
    run_many,
)
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import HMMTask, make_task

sys.path.insert(0, str(Path(__file__).parent))
import phase2
import phase2_plots

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase3b.toml"
fmt = phase2.fmt


def variant_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def variant_model(cfg: dict[str, Any], v: dict[str, Any], n_output: int) -> dict[str, Any]:
    by = v.get("by_n_output", {}).get(str(n_output), {})
    return {**cfg["model"], "n_output": n_output, **v.get("model", {}), **by}


def make_specs(cfg: dict[str, Any]) -> list[RunSpec]:
    specs = []
    for name, t in cfg["tasks"].items():
        for v in cfg["variants"]:
            model = variant_model(cfg, v, t["n_output"])
            for seed in cfg["experiment"]["seeds"]:
                specs.append(
                    RunSpec(
                        group=f"{name}/{variant_key(v['name'])}",
                        task={**t["spec"], "name": name},
                        data=cfg["data"],
                        model=model,
                        train={**cfg["train"], **v.get("train", {})},
                        seed=seed,
                        labels={"task": name, "variant": v["name"], "bound": v["bound"]},
                    )
                )
    return specs


# -- bounds -------------------------------------------------------------------------
def block_oracle_nll(task: HMMTask, seqs: torch.Tensor, k: int) -> float:
    """Mean per-symbol NLL of the source's own predictor restarted every ``k`` tokens.

    For a stationary source this is the best any model with memory reset every
    ``k`` tokens can do on these sequences (in expectation, ``H(Y_1..Y_k)/k``).
    """
    t = seqs.shape[1]
    total = sum(-task.log_probs(seqs[:, s : s + k]).sum(-1) for s in range(0, t, k))
    return float((total / t).mean())


def bound_nll(task: HMMTask, seqs: torch.Tensor, bound: str) -> tuple[float, float]:
    """(test-set NLL of the bound, its exact expectation) for a variant's ``bound``."""
    if bound == "oracle":
        return float(-task.log_probs(seqs).mean()), task.optimal_nll(seqs.shape[1])
    k = int(bound.removeprefix("block"))
    return block_oracle_nll(task, seqs, k), task.optimal_nll(k)


# -- analyses -----------------------------------------------------------------------
@torch.no_grad()
def analyse(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    a = cfg["analysis"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)
    out: dict[str, Any] = {"groups": {}, "resources": {}, "averaged_a": {}}
    task_cache: dict[str, Any] = {}
    for group, recs in groups.items():
        spec = recs[0]["meta"]["config"]
        name = spec["task"]["name"]
        if name not in task_cache:
            task = make_task(spec["task"])
            task_cache[name] = (task, build_splits(task, spec["data"]))
        task, splits = task_cache[name]
        bound = recs[0]["labels"]["bound"]
        b_test, b_exact = bound_nll(task, splits.test, bound)
        g: dict[str, Any] = {
            "labels": recs[0]["labels"],
            "bound_test_nll": b_test,
            "bound_exact_nll": b_exact,
            "iid_bound_test_nll": block_oracle_nll(task, splits.test, 1),
        }
        ok = [r for r in recs if "test" in r]
        own = [r["test"]["nll_nats"] - b_test for r in ok]
        g["gap_to_own_bound_mean"], g["gap_to_own_bound_se"] = mean_se(own)
        g["gap_to_own_bound_per_seed"] = own

        # resources (one model per variant and alphabet)
        rkey = f"{recs[0]['labels']['variant']}|Y={task.n_symbols}"
        if rkey not in out["resources"]:
            out["resources"][rkey] = resource_counts(
                build_run_model(RunSpec("x", {}, {}, spec["model"], {}, 0))
            )

        rep = phase2.pick_representative(recs)
        if rep is not None:
            model = load_model(rep)
            g["representative"] = {"seed": rep["seed"], "test_gap": rep["test"]["gap_nats"]}
            g["interventions"] = intervention_effects(
                model,
                task,
                splits.test,
                a["intervention_lags"],
                a["intervention_prefix_len"],
                a["intervention_seed"],
            )
            length = splits.test.shape[1]
            gen = model.sample(a["n_samples"], length, seed=a["sample_seed"])
            src = task.sample(a["n_samples"], length, seed=a["source_sample_seed"])
            ref = task.sample(a["n_samples"], length, seed=a["reference_sample_seed"])
            stats = sample_statistics(gen, src, ref, task.n_symbols, max_n=3)
            g["generation"] = {
                "ngram_tv": stats["ngram_tv"],
                "ngram_tv_noise_floor": stats["ngram_tv_noise_floor"],
                "impossible_samples": int((~torch.isfinite(task.log_probs(gen))).sum()),
            }
            if rep["labels"]["variant"] == "A":
                traj = memory_trajectories(model, splits.test)
                g["memory_dynamics"] = {
                    k: sum(v) / len(v) for k, v in traj.items() if k != "conditional_entropy"
                }

        # A's outcome-averaged prediction at its trained parameters (B at A's parameters)
        if recs[0]["labels"]["variant"] == "A":
            vals = []
            for r in ok:
                a_model = load_model(r)
                b_model = OutcomeAveragedModel(
                    a_model.n_memory,
                    a_model.n_output,
                    copy.deepcopy(list(a_model.circuit.blocks)),
                    n_env=a_model.n_env,
                    initial=a_model.rho0,
                )
                vals.append(float(-b_model.step_log_probs(splits.test).mean()))
            m, se = mean_se(vals)
            out["averaged_a"][name] = {"nll_mean": m, "nll_se": se, "per_seed": vals}
        out["groups"][group] = g
    return out


def a_vs_d(records: list[dict[str, Any]], tasks: list[str], d_name: str) -> list[dict[str, Any]]:
    """Per task: gap(D) - gap(A) over seeds (unpaired; bootstrap CI), success and medians."""
    rows = []
    for t in tasks:
        a = [r for r in records if r["labels"]["task"] == t and r["labels"]["variant"] == "A"]
        d = [r for r in records if r["labels"]["task"] == t and r["labels"]["variant"] == d_name]
        ga = [r["test"]["gap_nats"] for r in a if "test" in r]
        gd = [r["test"]["gap_nats"] for r in d if "test" in r]
        if not ga or not gd:
            continue
        gen = torch.Generator().manual_seed(0)
        va, vd = torch.tensor(ga), torch.tensor(gd)
        ia = torch.randint(len(ga), (4000, len(ga)), generator=gen)
        id_ = torch.randint(len(gd), (4000, len(gd)), generator=gen)
        diffs = vd[id_].mean(-1) - va[ia].mean(-1)
        lo, hi = torch.quantile(diffs.double(), torch.tensor([0.025, 0.975]).double()).tolist()
        rows.append(
            {
                "task": t,
                "gap_A_mean": float(va.mean()),
                "gap_D_mean": float(vd.mean()),
                "diff_mean": float(vd.mean() - va.mean()),
                "diff_ci95": [lo, hi],
                "median_A": float(va.median()),
                "median_D": float(vd.median()),
                "success_A": sum(g < 0.01 for g in ga),
                "success_D": sum(g < 0.01 for g in gd),
                "n_A": len(ga),
                "n_D": len(gd),
            }
        )
    return rows


# -- summary ------------------------------------------------------------------------
def summary_lines(
    rows: list[dict[str, Any]], ana: dict[str, Any], cfg: dict[str, Any], comps: dict[str, Any]
) -> list[str]:
    tasks = list(cfg["tasks"])
    names = [v["name"] for v in cfg["variants"]]
    by = {(r["labels"]["task"], r["labels"]["variant"]): r for r in rows}
    gb = {(g["labels"]["task"], g["labels"]["variant"]): g for g in ana["groups"].values()}
    lines = ["## Test NLL (nats/symbol, mean ± SE over seeds) and failures", ""]
    lines.append("| task | oracle | i.i.d. bound | " + " | ".join(names) + " |")
    lines.append("|---|---|---|" + "---|" * len(names))
    for t in tasks:
        first = by.get((t, names[0]))
        g0 = gb.get((t, names[0]))
        cells = []
        for n in names:
            r = by.get((t, n))
            if r is None:
                cells.append("-")
                continue
            fails = r["n_runs"] - r["n_success"]
            cells.append(
                f"{fmt(r['test_nll_mean'])} ± {fmt(r['test_nll_se'])} ({fails}/{r['n_runs']})"
            )
        lines.append(
            f"| {t} | {fmt(first['oracle_nll_nats'])} | {fmt(g0['iid_bound_test_nll'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "In brackets: failed runs (test gap to the oracle ≥ "
        f"{cfg['experiment']['success_gap']} nats/symbol, or crashed/diverged) out of all seeds. "
        "The i.i.d. bound is the source's own predictor restarted every token (the best "
        "memoryless model) on the same test sequences.",
        "",
        "## Gap to each variant's own bound (nats/symbol, mean ± SE)",
        "",
        "Own bound: the oracle for A, D, D+ and the hybrid; the source restarted every token "
        "for B and C; every k tokens for C_k. A gap near 0 means the variant reached the best "
        "it can do in principle.",
        "",
        "| task | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for t in tasks:
        cells = []
        for n in names:
            g = gb.get((t, n))
            cells.append(
                "-"
                if g is None
                else f"{fmt(g['gap_to_own_bound_mean'])} ± {fmt(g['gap_to_own_bound_se'])}"
            )
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    for d_name, comp in comps.items():
        lines += [
            "",
            f"## A versus {d_name}: test gap to the oracle (nats/symbol)",
            "",
            "| task | A mean | D mean | D - A (95% bootstrap CI) | median A | median D | succeeded A | succeeded D |",  # noqa: E501
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in comp:
            lo, hi = c["diff_ci95"]
            lines.append(
                f"| {c['task']} | {fmt(c['gap_A_mean'])} | {fmt(c['gap_D_mean'])} | "
                f"{fmt(c['diff_mean'])} [{fmt(lo)}, {fmt(hi)}] | {fmt(c['median_A'])} | "
                f"{fmt(c['median_D'])} | {c['success_A']}/{c['n_A']} | {c['success_D']}/{c['n_D']} |"  # noqa: E501
            )

    lines += [
        "",
        "## A's outcome-averaged prediction at its trained parameters",
        "",
        "Test NLL when each trained A predicts from `E^{t-1}(rho_0)` (B evaluated at A's "
        "parameters, not trained as B), against A itself and the i.i.d. bound.",
        "",
        "| task | A | A, outcome-averaged | i.i.d. bound |",
        "|---|---|---|---|",
    ]
    for t in tasks:
        r, avg, g = by.get((t, "A")), ana["averaged_a"].get(t), gb.get((t, "A"))
        if r and avg and g:
            lines.append(
                f"| {t} | {fmt(r['test_nll_mean'])} | {fmt(avg['nll_mean'])} ± {fmt(avg['nll_se'])} | "  # noqa: E501
                f"{fmt(g['iid_bound_test_nll'])} |"
            )

    lags = cfg["analysis"]["intervention_lags"]
    lines += [
        "",
        "## Intervention lag profiles (representative seed, best validation NLL)",
        "",
        "Mean TV change of p(next | prefix) when the symbol at lag l is changed; "
        "the source's value is in the last row of each task.",
        "",
        "| task | variant | " + " | ".join(f"lag {lag}" for lag in lags) + " |",
        "|---|---|" + "---|" * len(lags),
    ]
    for t in tasks:
        src = None
        for n in names:
            g = gb.get((t, n))
            if g is None or "interventions" not in g:
                continue
            iv = g["interventions"]
            src = iv
            lines.append(f"| {t} | {n} | " + " | ".join(fmt(x["model_tv"], 3) for x in iv) + " |")
        if src:
            lines.append(
                f"| {t} | *source* | " + " | ".join(fmt(x["source_tv"], 3) for x in src) + " |"
            )

    lines += [
        "",
        "## Generation check (representative seed): 3-gram TV to the source",
        "",
        "Noise floor: the same distance between two independent source samples.",
        "",
        "| task | noise floor | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for t in tasks:
        floor = None
        cells = []
        for n in names:
            gen = gb.get((t, n), {}).get("generation")
            if gen is None:
                cells.append("-")
                continue
            floor = gen["ngram_tv_noise_floor"][3]
            cells.append(fmt(gen["ngram_tv"][3], 3))
        lines.append(f"| {t} | {fmt(floor, 3)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Conditioned vs unconditioned memory of A (representative seed, means over t)",
        "",
        "| task | purity rho_t | purity E^t(rho0) | D(rho_t, E^t rho0) | spread of rho_t | NLL conditioned | NLL from E^t(rho0) |",  # noqa: E501
        "|---|---|---|---|---|---|---|",
    ]
    for t in tasks:
        md = gb.get((t, "A"), {}).get("memory_dynamics")
        if md:
            lines.append(
                f"| {t} | {fmt(md['conditional_purity'], 3)} | {fmt(md['unconditioned_purity'], 3)} | "  # noqa: E501
                f"{fmt(md['distance_to_unconditioned'], 3)} | {fmt(md['spread'], 3)} | "
                f"{fmt(md['nll_conditioned'])} | {fmt(md['nll_unconditioned'])} |"
            )

    lines += [
        "",
        "## Resources per token",
        "",
        "Two-qubit gates count a controlled-RY as one gate (it compiles to 2 CNOTs). "
        "D's depth is readout plus update; see the note for its re-preparation cost.",
        "",
        "| variant | symbols | qubits | parameters | depth | 1q gates | 2q gates | mid-circuit measurements | memory resets | classically controlled gates |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, rc in ana["resources"].items():
        n, y = key.split("|")
        lines.append(
            f"| {n} | {y.removeprefix('Y=')} | {rc['qubits']} | {rc['n_parameters']} | {rc['depth']} | "  # noqa: E501
            f"{rc['one_qubit_gates']} | {rc['two_qubit_gates']} | "
            f"{rc['mid_circuit_measurements_per_token']} | {rc['memory_resets_per_token']:g} | "
            f"{rc['classically_controlled_gates_per_token']} |"
        )
    return lines


def write_summary(
    summary_dir: Path,
    run_dir: Path,
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    ana: dict[str, Any],
    comps: dict[str, Any],
    records: list[dict[str, Any]],
    config_path: Path,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(cfg, None)
    commits = sorted({r["meta"]["git_commit"] for r in records if r["meta"].get("git_commit")})
    dirty = any(r["meta"].get("git_dirty") for r in records)
    shown_dir = run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir
    head = [
        "# Phase 3b summary: quantum ablations (generated)",
        "",
        f"Generated by `experiments/phase3b.py --config {config_path}` at {meta['timestamp_utc']}.",
        f"Training commits: {', '.join(c[:7] for c in commits)}"
        + (" (some runs from a dirty tree)" if dirty else "")
        + f". Raw records: `{shown_dir}`.",
        f"Seeds per group: {len(cfg['experiment']['seeds'])}. Summed per-run wall time: "
        f"{sum(r['wall_seconds'] for r in records) / 60:.1f} min.",
        "",
        "Statuses: "
        + ", ".join(
            f"{s}={sum(r['status'] == s for r in records)}"
            for s in sorted({r["status"] for r in records})
        )
        + ".",
        "",
    ]
    (summary_dir / "summary.md").write_text(
        "\n".join(head + summary_lines(rows, ana, cfg, comps)) + "\n"
    )
    phase2.write_csv(summary_dir / "main.csv", rows)
    slim = {
        k: {kk: vv for kk, vv in g.items() if kk != "gap_to_own_bound_per_seed"}
        for k, g in ana["groups"].items()
    }
    save_json(
        summary_dir / "summary.json",
        {
            "meta": meta,
            "training_commits": commits,
            "rows": rows,
            "groups": slim,
            "averaged_a": ana["averaged_a"],
            "resources": ana["resources"],
            "a_vs_d": comps,
            "per_seed_gaps": {
                r["group"]: [
                    x["test"]["gap_nats"]
                    for x in records
                    if x["group"] == r["group"] and "test" in x
                ]
                for r in rows
            },
        },
    )
    phase2_plots.plot_gaps(summary_dir / "gaps.png", records, cfg["experiment"]["success_gap"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=None, help="override [experiment].workers")
    p.add_argument("--from-run", default=None, help="reuse records in this results directory")
    args = p.parse_args()
    config_path = Path(args.config)
    cfg = tomllib.loads(config_path.read_text())
    exp = cfg["experiment"]
    workers = args.workers or exp["workers"]
    specs = make_specs(cfg)

    if args.from_run:
        run_dir = Path(args.from_run)
        records = phase2.load_records(run_dir)
        phase2.check_records_match(records, specs)
        have = {(r["group"], r["seed"]) for r in records}
        todo = [s for s in specs if (s.group, s.seed) not in have]
    else:
        run_dir = new_run_dir(exp["name"])
        records, todo = [], specs
    save_json(
        run_dir / "config.json", {"meta": run_metadata(cfg, None), "config_path": str(config_path)}
    )
    print(f"phase3b: {len(specs)} runs ({len(todo)} to train) with {workers} workers -> {run_dir}")
    start = time.perf_counter()
    records += run_many(todo, run_dir, workers=workers)
    records = phase2.order_records(records, specs)
    print(f"training done in {(time.perf_counter() - start) / 60:.1f} min")

    rows = aggregate(records, exp["success_gap"])
    ana = analyse(records, cfg)
    tasks = list(cfg["tasks"])
    d_names = [v["name"] for v in cfg["variants"] if v.get("model", {}).get("variant") == "D"]
    comps = {n: a_vs_d(records, tasks, n) for n in d_names}
    save_json(run_dir / "analysis.json", {"rows": rows, "analysis": ana, "a_vs_d": comps})
    summary_dir = REPO_ROOT / exp["summary_dir"]
    write_summary(summary_dir, run_dir, cfg, rows, ana, comps, records, config_path)
    print(f"summary -> {summary_dir}")


if __name__ == "__main__":
    main()
