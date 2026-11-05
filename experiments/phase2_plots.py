"""Static plots for the Phase 2 summary (PNG, light surface, two-colour encoding)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C1 = "#2a78d6"  # blue: model / successful runs
C2 = "#eb6834"  # orange: source / failed runs
LINTHRESH = 1e-4

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _short(group: str) -> str:
    return group.removeprefix("main/").removeprefix("variants/")


def _grid(n: int, ncols: int = 4, w: float = 3.0, h: float = 2.3):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _gap(r: dict[str, Any]) -> float:
    return r.get("test", {}).get("gap_nats", math.nan)


def plot_gaps(path: Path, records, success_gap: float) -> None:
    groups = list(dict.fromkeys(r["group"] for r in records))
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(groups) + 1.2))
    for i, g in enumerate(groups):
        recs = [r for r in records if r["group"] == g]
        for r in recs:
            gap = _gap(r)
            ok = r["status"] in ("completed", "early_stopped") and gap < success_gap
            if not math.isfinite(gap):
                ax.scatter([1.0], [i], marker="x", color=C2, s=40, zorder=3)
                continue
            ax.scatter(
                [gap],
                [i],
                s=36,
                color=C1 if ok else C2,
                edgecolor=SURFACE,
                linewidth=1,
                zorder=3,
                alpha=0.9,
            )
    ax.axvline(success_gap, color=MUTED, linestyle="--", linewidth=1)
    ax.text(success_gap, len(groups) - 0.4, f" success < {success_gap}", color=INK2, fontsize=8)
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_xlim(-3 * LINTHRESH, 1.5)
    ax.set_yticks(range(len(groups)), [_short(g) for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("test NLL gap to the source (nats/symbol, symlog)")
    ax.set_title("Per-seed test gap, recipe (blue: success, orange: failure)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(path: Path, records, success_gap: float) -> None:
    groups = list(dict.fromkeys(r["group"] for r in records))
    fig, axes = _grid(len(groups))
    for ax, g in zip(axes, groups, strict=True):
        for r in (r for r in records if r["group"] == g and "curves" in r):
            opt = r["test"]["optimal_nll_nats"]
            ok = r["status"] in ("completed", "early_stopped") and _gap(r) < success_gap
            ys = [v - opt for v in r["curves"]["val_nll"]]
            ax.plot(r["curves"]["eval_step"], ys, color=C1 if ok else C2, linewidth=1.2, alpha=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(_short(g))
        ax.set_xlabel("step")
    axes[0].set_ylabel("val NLL - optimum (nats)")
    fig.suptitle("Validation curves per seed (blue: success, orange: failure)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_markov_baselines(path: Path, analyses) -> None:
    sel = [
        a
        for a in analyses
        if a.get("representative")
        and a["labels"]["task"] in ("even", "golden_mean", "markov2_binary")
    ]
    if not sel:
        return
    fig, axes = _grid(len(sel), ncols=len(sel), w=3.2, h=2.6)
    for ax, a in zip(axes, sel, strict=True):
        ks = [b["k"] for b in a["markov_baselines"]]
        vals = [b["test_nll"] - a["oracle_test_nll"] for b in a["markov_baselines"]]
        ax.bar([str(k) for k in ks], vals, color=C2, width=0.6, label="order-k Markov")
        model = a["representative"]["test"]["nll_nats"] - a["oracle_test_nll"]
        ax.axhline(model, color=C1, linewidth=2, label="quantum model")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(_short(a["group"]))
        ax.set_xlabel("Markov order k")
    axes[0].set_ylabel("test NLL - oracle (nats)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_interventions(path: Path, analyses) -> None:
    sel = [a for a in analyses if a.get("interventions")]
    fig, axes = _grid(len(sel))
    for ax, a in zip(axes, sel, strict=True):
        lags = [r["lag"] for r in a["interventions"]]
        ax.plot(
            lags,
            [r["source_tv"] for r in a["interventions"]],
            color=C2,
            linestyle="--",
            marker="o",
            markersize=4,
            label="source",
        )
        ax.plot(
            lags,
            [r["model_tv"] for r in a["interventions"]],
            color=C1,
            marker="o",
            markersize=4,
            label="model",
        )
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylim(bottom=0)
        ax.set_title(_short(a["group"]))
        ax.set_xlabel("lag of the changed symbol")
    axes[0].set_ylabel("mean TV change of p(next)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_memory(path: Path, analyses) -> None:
    sel = [a for a in analyses if a.get("memory")]
    fig, axes = _grid(len(sel))
    for ax, a in zip(axes, sel, strict=True):
        m = a["memory"]
        t = range(1, len(m["nll_conditioned"]) + 1)
        ax.plot(t, m["nll_unconditioned"], color=C2, linestyle="--", label="predict from E^t(rho0)")
        ax.plot(t, m["nll_conditioned"], color=C1, label="conditional memory")
        ax.set_title(_short(a["group"]))
        ax.set_xlabel("step t")
    axes[0].set_ylabel("mean -log p(y_t) (nats)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Teacher-forced NLL per step: conditional vs outcome-averaged memory", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_memory_distance(path: Path, analyses) -> None:
    sel = [a for a in analyses if a.get("memory")]
    fig, axes = _grid(len(sel))
    for ax, a in zip(axes, sel, strict=True):
        m = a["memory"]
        t = range(len(m["distance_to_unconditioned"]))
        ax.plot(t, m["distance_to_unconditioned"], color=C1, label="D(rho_t, E^t(rho0))")
        ax.plot(t, m["spread"], color=C2, linestyle="--", label="D(rho_t, mean rho_t)")
        ax.set_ylim(0, 1)
        ax.set_title(_short(a["group"]))
        ax.set_xlabel("step t")
    axes[0].set_ylabel("trace distance")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_even_runs(path: Path, analyses) -> None:
    sel = [a for a in analyses if a["labels"]["task"] == "even" and a.get("generation")]
    if not sel:
        return
    fig, axes = _grid(len(sel), ncols=len(sel), w=4, h=2.6)
    for ax, a in zip(axes, sel, strict=True):
        g = a["generation"]
        lens = sorted(
            {int(k) for k in g["runs_of_1_model"]} | {int(k) for k in g["runs_of_1_source"]}
        )
        lens = [k for k in lens if k <= 14]
        tm, ts = g["n_runs_model"], g["n_runs_source"]
        xm = [g["runs_of_1_model"].get(str(k), 0) / tm for k in lens]
        xs = [g["runs_of_1_source"].get(str(k), 0) / ts for k in lens]
        ax.bar([k - 0.2 for k in lens], xs, width=0.38, color=C2, label="source")
        ax.bar([k + 0.2 for k in lens], xm, width=0.38, color=C1, label="model samples")
        ax.set_xticks(lens)
        ax.set_title(f"{_short(a['group'])}: lengths of 1-runs between 0s")
        ax.set_xlabel("run length")
    axes[0].set_ylabel("fraction of runs")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_variants(path: Path, records, success_gap: float) -> None:
    if not records:
        return
    tasks = list(dict.fromkeys(r["labels"]["task"] for r in records))
    variants = list(dict.fromkeys(r["labels"]["variant"] for r in records))
    fig, axes = _grid(len(tasks), ncols=len(tasks), w=3.0, h=0.3 * len(variants) + 1.2)
    for ax, t in zip(axes, tasks, strict=True):
        for i, v in enumerate(variants):
            for r in records:
                if r["labels"]["task"] != t or r["labels"]["variant"] != v:
                    continue
                gap = _gap(r)
                ok = r["status"] in ("completed", "early_stopped") and gap < success_gap
                ax.scatter(
                    [gap if math.isfinite(gap) else 1.0],
                    [i],
                    s=26,
                    color=C1 if ok else C2,
                    edgecolor=SURFACE,
                    linewidth=1,
                    zorder=3,
                    marker="o" if math.isfinite(gap) else "x",
                )
        ax.axvline(success_gap, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_xlim(-3 * LINTHRESH, 1.5)
        ax.set_yticks(range(len(variants)), variants if ax is axes[0] else [""] * len(variants))
        ax.invert_yaxis()
        ax.set_title(t)
        ax.set_xlabel("test gap (nats, symlog)")
    fig.suptitle(
        "Variants, 2 memory qubits: per-seed gap (blue: success, orange: failure)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_all(summary_dir: Path, rec_main, rec_var, analyses, success_gap: float) -> None:
    summary_dir = Path(summary_dir)
    plot_gaps(summary_dir / "gaps.png", rec_main, success_gap)
    plot_curves(summary_dir / "curves.png", rec_main, success_gap)
    plot_markov_baselines(summary_dir / "markov_baselines.png", analyses)
    plot_interventions(summary_dir / "interventions.png", analyses)
    plot_memory(summary_dir / "memory_nll.png", analyses)
    plot_memory_distance(summary_dir / "memory_distance.png", analyses)
    plot_even_runs(summary_dir / "even_runs.png", analyses)
    plot_variants(summary_dir / "variants.png", rec_var, success_gap)
