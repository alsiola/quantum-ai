"""Phase 3e (issue #10): retention under repeated measurement in architecture A.

Trains the feed-forward layout (shared strength) on the marked delayed-recall probe
(``qmem.tasks.marked_recall``) and on #8's implicit-clock recall, over the
``[[arms]]`` of the config: measurement strength (fixed ``eta`` or trained) x delay
``tau`` x memory qubits, plus a general-circuit capacity check and a falsification
arm (tuned learning rate, more steps and data). Every trained model is scored on
the test split: recall-position value NLL and accuracy (overall and by realised
delay), the decomposed gap to the source on the other positions, the cue retention
curve (Holevo information between the cue and the memory against the lag), the
reference-qubit purification, and how the distractor effects treat the stored bit.

Writes ``results/phase3e/<timestamp>/`` (run records and parameters, raw analyses;
gitignored) and ``<summary_dir>/`` (tables, CSV, JSON and plots; committed).

``--arms a,b`` trains only those arms; ``--from-run <dir>`` reuses saved records
(train what is missing, then summarise everything selected).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from qmem.benchmarks import (
    feedforward_g_for_eta,
    feedforward_strength_floor,
    marked_recall_metrics,
    recall_metrics,
    strength_floor,
)
from qmem.complexity import memory_profile, qmachine_instrument, qmachine_state
from qmem.diagnostics import label_retention, reference_purification, stored_bit_alignment
from qmem.experiment import RunSpec, build_splits, load_model, mean_se, run_many
from qmem.metrics import operational_strength
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import (
    DISTRACTOR_A,
    DISTRACTOR_B,
    QuantumSourceTask,
    delayed_recall_events,
    make_task,
    marked_recall_events,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase3e.toml"
LN2 = math.log(2)


def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# -- run specifications ------------------------------------------------------------
def arm_settings(arm: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(label, extra) for every strength setting of an arm."""
    out = [(f"eta={e:g}", {"eta": e}) for e in arm.get("etas", [])]
    if arm.get("trained", False):
        out.append(("trained", {"eta": None}))
    return out


def specs_for(cfg: dict[str, Any], arms: list[str] | None = None) -> list[RunSpec]:
    specs = []
    for arm in cfg["arms"]:
        if arms and arm["name"] not in arms:
            continue
        base_model = {**cfg["model"], **arm.get("model", {})}
        data = {**cfg["data"], **arm.get("data", {})}
        train = {**cfg["train"], **arm.get("train", {})}
        n_mems = arm.get("n_memory", [base_model["n_memory"]])
        eps_list = arm.get("eps", [arm["task"].get("eps", 0.0)])
        lrs = arm.get("lrs", [train["lr"]])
        for n_mem in n_mems:
            for eps in eps_list:
                for tau in arm["taus"]:
                    for label, extra in arm_settings(arm):
                        for lr in lrs:
                            model = {**base_model, "n_memory": n_mem}
                            if extra["eta"] is None:
                                model["strength_trainable"] = True
                            else:
                                model["strength_trainable"] = False
                                model["strength"] = feedforward_g_for_eta(
                                    n_mem, model["n_output"], extra["eta"]
                                )
                            task = {**arm["task"], "tau": tau, "eps": eps}
                            group = f"{arm['name']}/m{n_mem}/eps{eps:g}/tau{tau}/{label}"
                            if len(lrs) > 1:
                                group += f"/lr{lr:g}"
                            for seed in cfg["experiment"]["seeds"]:
                                specs.append(
                                    RunSpec(
                                        group=group,
                                        task=task,
                                        data=data,
                                        model=model,
                                        train={**train, "lr": lr},
                                        seed=seed,
                                        labels={
                                            "arm": arm["name"],
                                            "n_memory": n_mem,
                                            "eps": eps,
                                            "tau": tau,
                                            "setting": label,
                                            "eta_target": extra["eta"],
                                            "lr": lr,
                                            "task_kind": arm["task"]["kind"],
                                        },
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


# -- per-run analysis ---------------------------------------------------------------
def cue_events(task: Any, seq: torch.Tensor) -> dict[str, torch.Tensor]:
    if task.params["kind"] == "marked_recall":
        return marked_recall_events(seq)
    return delayed_recall_events(int(task.params["tau"]), seq)


def horizon(curve: dict[int, dict[str, float]], threshold: float, max_lag: int) -> int:
    """First lag whose cue Holevo is below ``threshold`` (``max_lag + 1`` if none)."""
    for k in range(1, max_lag + 1):
        if k in curve and curve[k]["holevo_bits"] < threshold:
            return k
    return max_lag + 1


def analyse_model(model: Any, task: Any, test: torch.Tensor, a: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with torch.no_grad():
        inst = model.instrument()
        out["eta"] = float(operational_strength(inst))
        if task.params["kind"] == "marked_recall":
            out["recall"] = marked_recall_metrics(model, task, test, a["delay_bins"])
        else:
            out["recall"] = recall_metrics(model, task, test)
        ev = cue_events(task, test)
        ret = label_retention(inst, model.rho0, test, ev["cue_value"], ev["lag"], a["max_lag"])
        out["cue_retention"] = ret
        out["horizon"] = horizon(ret, a["holevo_threshold_bits"], a["max_lag"])
        if task.params["kind"] == "marked_recall":
            out["alignment"] = stored_bit_alignment(
                inst, model.rho0, test, ev["cue_value"], ev["lag"], [DISTRACTOR_A, DISTRACTOR_B]
            )
        for mode, seqs in (("teacher_forced", test), ("generated", None)):
            pur = reference_purification(
                inst,
                model.n_memory,
                seqs,
                n_reference=a["n_reference"],
                n=test.shape[0],
                length=test.shape[1],
                seed=a["purification_seed"],
                threshold_bits=a["purification_threshold_bits"],
            )
            pur.pop("entropy_se_bits", None)
            out[f"purification_{mode}"] = pur
    return out


def analyse_run(rec: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group": rec["group"],
        "seed": rec["seed"],
        "status": rec["status"],
        "labels": rec["labels"],
    }
    if "state_dict" not in rec:
        out["message"] = rec.get("message")
        return out
    spec = rec["meta"]["config"]["task"]
    task = make_task(spec)
    test = build_splits(task, rec["meta"]["config"]["data"]).test
    model = load_model(rec)
    out.update(analyse_model(model, task, test, cfg["analysis"]))
    out["best_val_nll"] = rec["best_val_nll"]
    out["test_gap_nats"] = rec["test"]["gap_nats"]
    out["strength_final"] = rec["strength_final"]
    out["steps_run"] = rec["steps_run"]
    return out


class _SourceAsModel:
    """Score a source (e.g. an exact q-machine) with the model-facing metrics."""

    def __init__(self, source: Any):
        self.source = source

    def step_log_probs(self, s: torch.Tensor) -> torch.Tensor:
        return self.source.log_probs(s)

    def predictive_probs(self, s: torch.Tensor) -> torch.Tensor:
        return self.source.predictive(s)


# -- exact references (no training) --------------------------------------------------
def task_catalogue(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Causal states, C_mu, C_q, optimum and strength floors of every task in the arms."""
    seen, rows = set(), []
    length = cfg["data"]["length"]
    for arm in cfg["arms"]:
        n_out = {**cfg["model"], **arm.get("model", {})}["n_output"]
        for eps in arm.get("eps", [arm["task"].get("eps", 0.0)]):
            for tau in sorted(set(arm["taus"]) | {1, 2, 4, 8, 16, 32}):
                spec = {**arm["task"], "tau": tau, "eps": eps}
                key = json.dumps(spec, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                task = make_task(spec)
                mem = memory_profile(task)
                mem.pop("cq_by_length", None)
                row = {
                    "spec": spec,
                    "name": task.name,
                    "n_output": n_out,
                    "memory": mem,
                    "optimal_nll_nats": task.optimal_nll(length),
                    "entropy_rate_nats": task.entropy_rate(),
                    "eta_floor_any": {
                        d: strength_floor(task, d)["eta_lower_bound"] for d in (2, 4, 8)
                    },
                    "eta_floor_A": {
                        n: feedforward_strength_floor(task, n, n_out)["eta_floor"]
                        for n in (1, 2, 3)
                    },
                }
                rows.append(row)
    return rows


def qmachine_check(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The exact phaseless q-machine of each marked task scored like a trained model.

    A general instrument on ``quantum_dim`` levels (3 here, so 2 qubits) that solves
    the task: recall value NLL ``H(eps)`` at every delay.
    """
    rows = []
    for tau in (1, 4, 16, 32):
        spec = {"kind": "marked_recall", "tau": tau, "eps": 0.05}
        task = make_task(spec)
        inst, _ = qmachine_instrument(task)
        src = QuantumSourceTask(inst, initial=qmachine_state(task, 0), name="qmachine")
        test = build_splits(task, cfg["data"]).test

        m = marked_recall_metrics(_SourceAsModel(src), task, test, cfg["analysis"]["delay_bins"])
        rows.append(
            {
                "tau": tau,
                "quantum_dim": inst.dim,
                "recall_accuracy": m["recall_accuracy"],
                "recall_value_nll_nats": m["recall_value_nll_nats"],
                "source_recall_value_nll_nats": m["source_recall_value_nll_nats"],
                "gap_all_nats": m["gap_all_nats"],
                "eta": float(operational_strength(inst)),
            }
        )
    return rows


# -- aggregation --------------------------------------------------------------------
def _ms(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return mean_se(vals)


def summarise(analyses: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for an in analyses:
        groups.setdefault(an["group"], []).append(an)
    a = cfg["analysis"]
    out = []
    for group, ans in groups.items():
        ok = [x for x in ans if "recall" in x]
        lab = ans[0]["labels"]
        s: dict[str, Any] = {"group": group, **lab, "n_runs": len(ans), "n_analysed": len(ok)}
        s["statuses"] = {
            st: sum(x["status"] == st for x in ans) for st in {x["status"] for x in ans}
        }
        if not ok:
            out.append(s)
            continue
        marked = lab["task_kind"] == "marked_recall"
        acc_key = "recall_accuracy"
        nll_key = "recall_value_nll_nats" if marked else "recall_nll_nats"
        s["acc_mean"], s["acc_se"] = _ms([x["recall"][acc_key] for x in ok])
        s["recall_nll_mean"], s["recall_nll_se"] = _ms([x["recall"][nll_key] for x in ok])
        s["eta_mean"], s["eta_se"] = _ms([x["eta"] for x in ok])
        s["eta_min"] = min(x["eta"] for x in ok)
        s["eta_max"] = max(x["eta"] for x in ok)
        s["g_mean"] = _ms([x["strength_final"][0] for x in ok])[0]
        s["gap_mean"], s["gap_se"] = _ms([x["test_gap_nats"] for x in ok])
        s["horizon_mean"], s["horizon_se"] = _ms([float(x["horizon"]) for x in ok])
        best = min(ok, key=lambda x: x["best_val_nll"])
        s["best_val_seed"] = best["seed"]
        s["best_val_acc"] = best["recall"][acc_key]
        s["best_val_recall_nll"] = best["recall"][nll_key]
        s["best_val_horizon"] = best["horizon"]
        s["best_val_eta"] = best["eta"]
        s["acc_best_seed"] = max(x["recall"][acc_key] for x in ok)
        if marked:
            for k in (
                "gap_type_nats",
                "gap_value_distractor_nats",
                "gap_value_cue_nats",
                "gap_value_recall_nats",
                "gap_non_recall_nats",
            ):
                s[k] = _ms([x["recall"][k] for x in ok])[0]
            s["source_recall_value_nll"] = ok[0]["recall"]["source_recall_value_nll_nats"]
            bins: dict[int, list[float]] = {}
            for x in ok:
                for b in x["recall"]["by_delay"]:
                    bins.setdefault(b["delay_lo"], []).append(b["accuracy"])
            s["acc_by_delay"] = {lo: _ms(v)[0] for lo, v in sorted(bins.items())}
            for key in ("commutator", "prob_bias", "contraction"):
                vals = []
                for x in ok:
                    sym = x["alignment"]["symbols"]
                    vals.append(sum(abs(sym[y][key]) for y in (DISTRACTOR_A, DISTRACTOR_B)) / 2)
                s[f"distractor_{key}_mean"], s[f"distractor_{key}_se"] = _ms(vals)
        lags = sorted({int(k) for x in ok for k in x["cue_retention"]})
        s["holevo_by_lag"] = {
            k: _ms([_get(x["cue_retention"], k, "holevo_bits") for x in ok])[0] for k in lags
        }
        for mode in ("teacher_forced", "generated"):
            pur = [x[f"purification_{mode}"] for x in ok]
            s[f"purified_{mode}"] = _ms([p["fraction_purified"] for p in pur])[0]
            s[f"purification_median_{mode}"] = _ms([p["purification_time_median"] for p in pur])[0]
            curves = [p["entropy_mean_bits"] for p in pur]
            s[f"purification_curve_{mode}"] = [
                sum(c[t] for c in curves) / len(curves) for t in range(len(curves[0]))
            ]
        s["holevo_threshold_bits"] = a["holevo_threshold_bits"]
        out.append(s)
    return out


def _get(curve: dict, k: int, key: str) -> float | None:
    v = curve.get(k, curve.get(str(k)))
    return None if v is None else v[key]


# -- reports ------------------------------------------------------------------------
def tables_markdown(
    groups: list[dict[str, Any]],
    catalogue: list[dict[str, Any]],
    qm: list[dict[str, Any]],
    cfg: dict[str, Any],
    meta: dict[str, Any],
    cpu: float,
) -> str:
    seeds = cfg["experiment"]["seeds"]
    lines = [
        "# Phase 3e: retention under repeated measurement (architecture A)",
        "",
        "Generated by `experiments/phase3e.py` from `experiments/configs/phase3e.toml`"
        f" (commit `{(meta['git_commit'] or '?')[:7]}`, dirty: {meta['git_dirty']}).",
        f"{len(seeds)} model seeds per cell; data seed {cfg['data']['seed']}; training "
        f"{cpu / 3600:.1f} CPU-hours in total. Discussion: "
        "[`docs/phase3e-retention.md`](../../../docs/phase3e-retention.md).",
        "",
        "## Tasks (exact)",
        "",
        "`eta_floor(any, d)`: least operational strength of any instrument on `d` levels"
        " that reproduces the source; `eta_floor(A, n)`: least strength at which the"
        " feed-forward layout passes the per-outcome spectral test on `n` memory qubits"
        " (necessary only; `none` = fails at every `g`).",
        "",
        "| task | causal states | C_mu (bits) | C_q (bits) | dim_q | H/T (nats) | "
        "eta_floor(any, d=4) | eta_floor(A, n=1/2/3) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in catalogue:
        m = r["memory"]
        fa = " / ".join(
            fmt(r["eta_floor_A"][n], 2) if r["eta_floor_A"][n] is not None else "none"
            for n in (1, 2, 3)
        )
        lines.append(
            f"| {r['name']} | {m['n_causal_states']} | {fmt(m['c_mu_bits'])} | "
            f"{fmt(m['cq_bits'])} | {m['quantum_dim']} | {fmt(r['optimal_nll_nats'], 4)} | "
            f"{fmt(r['eta_floor_any'][4], 3)} | {fa} |"
        )
    lines += [
        "",
        "Exact q-machine (general instrument, Binder et al. construction) on the marked task,"
        " eps = 0.05, scored on the test split:",
        "",
        "| tau | memory dim | eta | recall acc | recall value NLL | source | gap (nats/symbol) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in qm:
        lines.append(
            f"| {r['tau']} | {r['quantum_dim']} | {fmt(r['eta'])} | {fmt(r['recall_accuracy'])} | "
            f"{fmt(r['recall_value_nll_nats'])} | {fmt(r['source_recall_value_nll_nats'])} | "
            f"{fmt(r['gap_all_nats'], 6)} |"
        )
    arms: dict[str, list[dict[str, Any]]] = {}
    for g in groups:
        arms.setdefault(g["arm"], []).append(g)
    for arm, gs in arms.items():
        lines += [
            "",
            f"## Arm `{arm}`",
            "",
            "acc / NLL: recall-position value accuracy and NLL (marked task: given that a"
            " cue-type symbol comes; clock task: the recall symbol), mean ± SE over seeds;"
            " `val-best`: the seed with the lowest validation NLL. Horizon: first lag with"
            f" cue Holevo < {cfg['analysis']['holevo_threshold_bits']} bits. Gaps in nats per"
            " position (model - source): type / distractor value / non-recall.",
            "",
            "| n | eps | tau | strength | eta (mean [min, max]) | acc | val-best acc | NLL | "
            "horizon | gap type / dis. value / non-recall | purified (gen) | median t |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in gs:
            if "acc_mean" not in g:
                lines.append(
                    f"| {g['n_memory']} | {g['eps']:g} | {g['tau']} | {g['setting']} | "
                    "no runs analysed |"
                )
                continue
            gaps = (
                f"{fmt(g.get('gap_type_nats'), 4)} / "
                f"{fmt(g.get('gap_value_distractor_nats'), 4)} / "
                f"{fmt(g.get('gap_non_recall_nats'), 4)}"
                if "gap_type_nats" in g
                else "n/a"
            )
            setting = g["setting"] + (f", lr {g['lr']:g}" if arm == "falsify" else "")
            lines.append(
                f"| {g['n_memory']} | {g['eps']:g} | {g['tau']} | {setting} | "
                f"{fmt(g['eta_mean'], 2)} [{fmt(g['eta_min'], 2)}, {fmt(g['eta_max'], 2)}] | "
                f"{fmt(g['acc_mean'])} ± {fmt(g['acc_se'])} | {fmt(g['best_val_acc'])} | "
                f"{fmt(g['recall_nll_mean'])} ± {fmt(g['recall_nll_se'])} | "
                f"{fmt(g['horizon_mean'], 1)} ± {fmt(g['horizon_se'], 1)} | {gaps} | "
                f"{fmt(g['purified_generated'], 2)} | "
                f"{fmt(g['purification_median_generated'], 1)} |"
            )
    marked = [g for g in groups if "distractor_commutator_mean" in g]
    if marked:
        lines += [
            "",
            "## The stored bit and the distractor effects",
            "",
            "Averaged over the two distractor outcomes and the seeds."
            " `[E_d, Delta]`: normalised commutator of the distractor effect with the"
            " difference of the cue-conditioned memories (0 = commutes);"
            " `|p(d|0) - p(d|1)|`: how much the distractor probability reveals the cue"
            " (source: 0); contraction: trace distance between the cue-conditioned memories"
            " after a distractor, relative to before (1 = protected).",
            "",
            "| arm | n | eps | tau | strength | [E_d, Delta] | abs(p(d|0) - p(d|1)) "
            "| contraction |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for g in marked:
            if g["arm"] == "falsify":
                continue
            lines.append(
                f"| {g['arm']} | {g['n_memory']} | {g['eps']:g} | {g['tau']} | {g['setting']} | "
                f"{fmt(g['distractor_commutator_mean'])} ± {fmt(g['distractor_commutator_se'])} | "
                f"{fmt(g['distractor_prob_bias_mean'], 4)} | "
                f"{fmt(g['distractor_contraction_mean'])} ± {fmt(g['distractor_contraction_se'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def make_plots(summary_dir: Path, groups: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [g for g in groups if g["arm"] == "grid" and "acc_mean" in g]
    if grid:
        taus = sorted({g["tau"] for g in grid})
        settings = [s for s in dict.fromkeys(g["setting"] for g in grid)]
        settings = sorted([s for s in settings if s != "trained"], key=lambda s: float(s[4:]))
        settings.append("trained")
        mat = torch.full((len(settings), len(taus)), math.nan)
        for g in grid:
            mat[settings.index(g["setting"]), taus.index(g["tau"])] = g["acc_mean"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(mat.numpy(), cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(taus)), [str(t) for t in taus])
        ax.set_yticks(range(len(settings)), settings)
        for i in range(len(settings)):
            for j in range(len(taus)):
                if math.isfinite(float(mat[i, j])):
                    ax.text(
                        j,
                        i,
                        f"{float(mat[i, j]):.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if mat[i, j] < 0.75 else "black",
                    )
        ax.set_xlabel("mean delay τ (distractors between cue and recall)")
        ax.set_ylabel("measurement strength")
        ax.set_title("Recall value accuracy, 2 memory qubits, ε = 0.05 (optimum 0.95, chance 0.5)")
        fig.colorbar(im, ax=ax, label="mean over 10 seeds")
        fig.tight_layout()
        fig.savefig(summary_dir / "accuracy_heatmap.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for tau, ax in zip((4, 16), axes, strict=True):
            for s in settings:
                g = next((x for x in grid if x["tau"] == tau and x["setting"] == s), None)
                if g is None:
                    continue
                lags = sorted(int(k) for k in g["holevo_by_lag"])
                ax.plot(
                    lags,
                    [g["holevo_by_lag"][k] for k in lags],
                    marker="o",
                    ms=2.5,
                    label=s,
                    ls="--" if s == "trained" else "-",
                )
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("lag k (steps since the cue)")
            ax.set_ylabel("Holevo χ(cue : memory) (bits)")
            ax.set_title(f"Cue retention, n = 2, τ = {tau} (source: 1 bit until recall)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(summary_dir / "retention_curves.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, n, eps, style in (
        ("grid", 2, 0.05, "o-"),
        ("eps", 2, 0.0, "s--"),
        ("eps", 2, 0.1, "^--"),
        ("n1", 1, 0.05, "x:"),
        ("n3", 3, 0.05, "d-"),
        ("clock", 2, 0.05, "v-."),
    ):
        sub = sorted(
            (
                g
                for g in groups
                if g["arm"] == arm
                and g["n_memory"] == n
                and g["eps"] == eps
                and g["setting"] == "trained"
                and "eta_mean" in g
            ),
            key=lambda g: g["tau"],
        )
        if sub:
            ax.errorbar(
                [g["tau"] for g in sub],
                [g["eta_mean"] for g in sub],
                yerr=[g["eta_se"] for g in sub],
                fmt=style,
                capsize=3,
                label=f"{'clock (#8)' if arm == 'clock' else 'marked'}, n={n}, ε={eps:g}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("delay τ")
    ax.set_ylabel("learned operational strength η")
    ax.set_title("Trained strength against delay (mean ± SE, 10 seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(summary_dir / "learned_eta.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g in grid:
        if g["tau"] != 16 or "acc_by_delay" not in g:
            continue
        d = sorted(int(k) for k in g["acc_by_delay"])
        ax.plot(
            d,
            [g["acc_by_delay"][k] for k in d],
            marker="o",
            ms=3,
            label=g["setting"],
            ls="--" if g["setting"] == "trained" else "-",
        )
    ax.axhline(0.95, color="grey", lw=0.8)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("realised delay (distractors, bin left edge)")
    ax.set_ylabel("recall value accuracy")
    ax.set_title("Accuracy by realised delay, n = 2, τ = 16")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(summary_dir / "accuracy_by_delay.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--from-run", default=None, help="reuse records in this results directory")
    p.add_argument("--arms", default=None, help="comma-separated arms to train (default: all)")
    p.add_argument("--no-train", action="store_true", help="summarise saved records only")
    args = p.parse_args()
    start = time.perf_counter()
    cfg = tomllib.loads(Path(args.config).read_text())
    exp = cfg["experiment"]
    summary_dir = REPO_ROOT / exp["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.from_run) if args.from_run else new_run_dir(exp["name"])
    meta = run_metadata(cfg)
    save_json(
        out / f"config-{time.strftime('%Y%m%dT%H%M%S')}.json", {"meta": meta, "args": vars(args)}
    )

    arms = args.arms.split(",") if args.arms else None
    specs = specs_for(cfg, arms)
    records = load_records(out) if args.from_run else []
    have = {(r["group"], r["seed"]) for r in records}
    missing = [] if args.no_train else [s for s in specs if (s.group, s.seed) not in have]
    print(f"{len(missing)} runs to train ({len(records)} saved) -> {out}")
    records += run_many(missing, out, workers=args.workers or exp["workers"])
    all_specs = specs_for(cfg)
    order = {(s.group, s.seed): i for i, s in enumerate(all_specs)}
    records = sorted(
        (r for r in records if (r["group"], r["seed"]) in order),
        key=lambda r: order[(r["group"], r["seed"])],
    )
    print(f"analysing {len(records)} runs ...")
    analyses = [analyse_run(r, cfg) for r in records]
    groups = summarise(analyses, cfg)
    catalogue = task_catalogue(cfg)
    qm = qmachine_check(cfg)
    cpu = sum(r.get("wall_seconds", 0.0) for r in records)
    save_json(out / "analyses.json", {"meta": meta, "analyses": analyses})
    run_commits = sorted({r["meta"]["git_commit"] for r in records})
    save_json(
        summary_dir / "summary.json",
        {
            "meta": meta,
            "run_commits": run_commits,
            "run_dir": str(out.relative_to(REPO_ROOT))
            if out.is_relative_to(REPO_ROOT)
            else str(out),
            "catalogue": catalogue,
            "qmachine": qm,
            "groups": groups,
        },
    )
    write_csv(summary_dir / "groups.csv", groups)
    (summary_dir / "summary.md").write_text(tables_markdown(groups, catalogue, qm, cfg, meta, cpu))
    make_plots(summary_dir, groups)
    print(f"wrote {summary_dir} and {out} ({(time.perf_counter() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
