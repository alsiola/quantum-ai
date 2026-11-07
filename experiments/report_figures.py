"""Figures for the campaign report (`docs/report.md`), from committed summaries only.

Every figure reads `results/summaries/<phase>/*.csv` (committed) and writes a PNG to
`docs/figures/`. Nothing is trained or simulated, so it runs in a few seconds:

    uv run python experiments/report_figures.py

`--summaries` and `--out` override the input and output directories.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots in fixed order (validated: scripts/validate_palette.js, light mode).
# Three slots are below 3:1 contrast on the surface, so every series also has its own
# marker shape and a legend, and every number is in a table in docs/report.md.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
FAIL_GAP = 0.01  # the campaign's failure threshold (nats/symbol)

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
        "lines.markersize": 7,
        "legend.frameon": False,
    }
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def num(s: str) -> float:
    return float(s) if s not in ("", "nan", None) else math.nan


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


PMD = [f"pmd_N{n}" for n in range(3, 9)]


def classical_pmd(summaries: Path) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """Best fitted 2-state HMM, unifilar block bound, and each 1-qubit family (Phase 3f)."""
    rows = read(summaries / "phase3f" / "dimensional.csv")
    best_hmm, bound = [], []
    families = {"A": "A/D2", "A_ff": "A_ff/D2", "hqmm": "hqmm/D2", "umps": "umps/D2"}
    qubit: dict[str, list[float]] = {k: [] for k in families}
    for task in PMD:
        task_rows = [r for r in rows if r["task"] == task]
        best_hmm.append(min(num(r["long_gap"]) for r in task_rows if r["family"] == "hmm"))
        bound.append(num(task_rows[0]["block_bound_nats"]))
        for fam, cell in families.items():
            (r,) = [r for r in task_rows if r["cell"] == cell]
            qubit[fam].append(num(r["long_gap"]))
    return best_hmm, bound, qubit


def fig_dimension(summaries: Path, out: Path) -> None:
    best_hmm, bound, qubit = classical_pmd(summaries)
    ns = list(range(3, 9))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = {
        "A": "1-qubit A, general",
        "A_ff": "1-qubit A, feed-forward",
        "hqmm": "1-qubit HQMM",
        "umps": "uMPS, χ = 2",
    }
    for i, (fam, label) in enumerate(labels.items()):
        ax.plot(ns, qubit[fam], marker=MARKERS[i], color=SLOTS[i], label=label, lw=1.2)
    ax.plot(ns, best_hmm, marker=MARKERS[4], color=SLOTS[4], label="best fitted 2-state HMM", lw=2)
    ax.plot(ns, bound, color=INK2, ls="--", lw=1.5, label="bound for every 2-state unifilar model")
    ax.set_yscale("log")
    ax.set_xlabel("PMD period N")
    ax.set_ylabel("KL rate to the source (nats/symbol)")
    ax.set_title("Memory dimension on PMD: one qubit against two classical states")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7.5, ncol=3)
    save(fig, out / "dimension_pmd.png")


def fig_pmd_noise(summaries: Path, out: Path) -> None:
    best_hmm, bound, _ = classical_pmd(summaries)
    rows = [r for r in read(summaries / "phase5a" / "noise.csv") if r["kind"] == "pmd"]
    ns = list(range(3, 9))

    def series(noise: str, backend: str) -> list[float]:
        vals = []
        for task in PMD:
            (r,) = [
                r
                for r in rows
                if r["model"] == task and r["noise"] == noise and r["backend"] == backend
            ]
            vals.append(num(r["kl_rate"]))
        return vals

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (label, noise, backend) in enumerate(
        [
            ("1-qubit A, noiseless", "none", ""),
            ("… FakeKingston calibration", "backend", "fake_kingston"),
            ("… FakeFez calibration", "backend", "fake_fez"),
        ]
    ):
        ax.plot(ns, series(noise, backend), marker=MARKERS[i], color=SLOTS[i], label=label, lw=1.2)
    ax.plot(ns, best_hmm, marker=MARKERS[4], color=SLOTS[4], label="best fitted 2-state HMM", lw=2)
    ax.plot(ns, bound, color=INK2, ls="--", lw=1.5, label="2-state unifilar bound")
    ax.set_yscale("log")
    ax.set_xlabel("PMD period N")
    ax.set_ylabel("KL rate to the source (nats/symbol)")
    ax.set_title("The PMD result under IBM device calibrations (exact noisy instruments)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7.5, ncol=3)
    save(fig, out / "pmd_noise.png")


def fig_parameters(summaries: Path, out: Path) -> None:
    rows = read(summaries / "phase4b" / "a" / "main.csv")
    families = [
        ("A", "A"),
        ("hqmm_gen", "HQMM, K random generators"),
        ("hqmm", "HQMM (Stiefel)"),
        ("hmm", "HMM"),
        ("gru", "GRU"),
        ("D_diss", "D_diss (and variants)"),
    ]
    tasks = ["nemo", "probability_clock", "markov1_quaternary"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, task in zip(axes, tasks, strict=True):
        for i, (fam, label) in enumerate(families):
            fam_rows = [
                r
                for r in rows
                if r["task"] == task
                and (r["family"] == fam or (fam == "D_diss" and r["family"] in ("D", "D_pswap")))
                and not r["arm"].endswith("_long")
            ]
            pts = sorted(
                (int(r["n_parameters"]), max(num(r["gap_median"]), 1e-4)) for r in fam_rows
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            linestyle = "none" if fam == "D_diss" else "-"
            ax.plot(
                xs,
                ys,
                marker=MARKERS[i],
                color=SLOTS[i],
                ls=linestyle,
                lw=1.2,
                label=label,
            )
        ax.axhline(FAIL_GAP, color=INK2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([10, 20, 50, 100])
        ax.set_title(task)
        ax.set_xlabel("parameters")
    axes[0].set_ylabel("median test gap over 10 seeds (nats/symbol)")
    axes[0].text(11, FAIL_GAP * 1.15, "failure threshold", color=INK2, fontsize=7)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=7.5)
    fig.subplots_adjust(top=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "parameters.png", dpi=150)
    plt.close(fig)
    print("wrote docs/figures/parameters.png")


def _bars(ax, labels, means, ses, title) -> None:
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, means, xerr=ses, color=SLOTS[0], height=0.62, edgecolor=SURFACE, lw=2)
    for y, m, se in zip(ys, means, ses, strict=True):
        ax.text(m + se + 0.012, y, f"{m:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys, labels, fontsize=7.5)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.axvline(0.95, color=INK2, ls="--", lw=1)
    ax.set_xlim(0.4, 1.05)
    top = len(labels) - 0.35
    ax.text(0.5, top, "chance", fontsize=7, color=MUTED, ha="center")
    ax.text(0.95, top, "source", fontsize=7, color=INK2, ha="center")
    ax.set_xlabel("recall accuracy (mean ± s.e. over 10 seeds)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def fig_retention(summaries: Path, out: Path) -> None:
    b = {
        (r["arm"], r["task"]): r
        for r in read(summaries / "phase4b" / "b" / "main.csv")
        if r["task"].startswith("marked_recall")
    }
    c = {r["arm"]: r for r in read(summaries / "phase4b" / "c" / "main.csv")}
    b_arms = [
        ("eta06", "n = 2 ff, η = 0.6 fixed"),
        ("eta09", "n = 2 ff, η = 0.9 fixed"),
        ("free", "n = 2 ff, η free"),
        ("prior01", "n = 2 ff, prior on η"),
        ("curriculum", "n = 2 ff, τ curriculum"),
        ("gen_weak", "general, η ≈ 0.3"),
        ("gen_strong", "general, η ≈ 0.9"),
        ("n3_eta06", "n = 3 ff, η = 0.6"),
        ("n3_eta09", "n = 3 ff, η = 0.9"),
        ("n3_free", "n = 3 ff, η free"),
    ]
    c_arms = [
        ("ff", "feed-forward, shared g"),
        ("general", "general circuit"),
        ("ff_unshared", "feed-forward, per-coupling g"),
        ("ff_n3", "feed-forward n = 3, unread qubit"),
        ("hqmm_2", "HQMM D = 2"),
        ("hqmm_4", "HQMM D = 4"),
        ("umps_4", "uMPS χ = 4"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bb = [b[(arm, "marked_recall_tau4")] for arm, _ in b_arms]
    _bars(
        axes[0],
        [lab for _, lab in b_arms],
        [num(r["recall_acc_mean"]) for r in bb],
        [num(r["recall_acc_se"]) for r in bb],
        "Readout strength (marked_recall, τ = 4; ff = feed-forward)",
    )
    cc = [c[arm] for arm, _ in c_arms]
    _bars(
        axes[1],
        [f"{lab} [{c[arm]['retaining_seeds']}/10 retain]" for arm, lab in c_arms],
        [num(r["recall_acc_mean"]) for r in cc],
        [num(r["recall_acc_se"]) for r in cc],
        "Protected memory by model class (τ = 4)",
    )
    save(fig, out / "retention.png")


def fig_trainability(summaries: Path, out: Path) -> None:
    cells = {r["group"]: r for r in read(summaries / "phase4a" / "cells.csv")}
    rows = [
        ("A, markov1_quaternary, n = 2", "m1q_A_recipe/markov1_quaternary/n2", "joint depth 3",
         "m1q_A_jd3/markov1_quaternary/n2"),
        ("", "m1q_A_recipe/markov1_quaternary/n2", "4 screened restarts",
         "m1q_A_restarts/markov1_quaternary/n2"),
        ("", "m1q_A_recipe/markov1_quaternary/n2", "near-product start",
         "m1q_A_product/markov1_quaternary/n2"),
        ("A, markov1_quaternary, n = 3", "m1q_A_recipe/markov1_quaternary/n3", "joint depth 4",
         "m1q_A_jd4/markov1_quaternary/n3"),
        ("", "m1q_A_recipe/markov1_quaternary/n3", "4x budget, tuned lr",
         "m1q_A_budget/markov1_quaternary/n3"),
        ("A, markov2_binary, n = 3", "p2_A_recipe/markov2_binary/n3", "4 screened restarts",
         "p2_A_restarts/markov2_binary/n3"),
        ("A, even, n = 3", "p2_A_recipe/even/n3", "4 screened restarts",
         "p2_A_restarts/even/n3"),
        ("HQMM D = 16, iid_quaternary", "big_hqmm_tuned/iid_quaternary/n4",
         "identity start, data unigram", "big_hqmm_identity/iid_quaternary/n4"),
        ("HQMM D = 16, markov1_quaternary", "big_hqmm_tuned/markov1_quaternary/n4",
         "identity start, data unigram", "big_hqmm_identity/markov1_quaternary/n4"),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (cell, before, fix, after) in zip(ys, rows, strict=True):
        b, a = int(cells[before]["n_failed"]), int(cells[after]["n_failed"])
        ax.plot([b, a], [y, y], color=AXIS, lw=1.5, zorder=1)
        ax.scatter([b], [y], color=SLOTS[1], marker="s", s=48, zorder=2, edgecolor=SURFACE)
        ax.scatter([a], [y], color=SLOTS[0], marker="o", s=56, zorder=3, edgecolor=SURFACE)
        if a == b:  # keep the "before" square visible under an unchanged result
            ax.scatter([b], [y], facecolor="none", edgecolor=SLOTS[1], marker="s", s=110, zorder=4)
        ax.text(10.4, y, fix, va="center", fontsize=7.5, color=INK2)
        if cell:
            ax.text(-0.4, y + 0.42, cell, fontsize=7.5, color=INK, ha="left")
    ax.scatter([], [], color=SLOTS[1], marker="s", label="before (recipe)")
    ax.scatter([], [], color=SLOTS[0], marker="o", label="after the change")
    ax.set_yticks([])
    ax.set_xlim(-0.5, 15.5)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("failed seeds out of 10 (test gap ≥ 0.01)")
    ax.set_title("Trainability: what each change did (Phase 4a)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, out / "trainability.png")


def fig_witness(summaries: Path, out: Path) -> None:
    rows = read(summaries / "phase5a" / "witness.csv")
    ns = list(range(3, 9))

    def max_z(model: str, variant: str) -> float:
        (r,) = [r for r in rows if r["model"] == model and r["variant"] == variant]
        return max(abs(num(r[f"z_j{j}"])) for j in range(4))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    variants = [
        ("source", "PMD source"),
        ("noiseless", "1-qubit A, noiseless"),
        ("fake_kingston", "… FakeKingston calibration"),
        ("fake_fez", "… FakeFez calibration"),
        ("control: 2-state HMM", "control: a 2-state HMM"),
    ]
    for i, (variant, label) in enumerate(variants):
        ax.plot(
            ns,
            [max_z(f"pmd_N{n}", variant) for n in ns],
            marker=MARKERS[i],
            color=SLOTS[i],
            lw=1.2,
            label=label,
        )
    ax.axhline(3, color=INK2, ls=":", lw=1)
    ax.text(3.0, 3.3, "3 sigma", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("PMD period N")
    ax.set_ylabel("largest |z| of the Hankel witness")
    ax.set_title("Predicted dimension witness, 20,000 shots of 64 symbols (simulated)")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, out / "witness.png")


FIGURES = [
    fig_dimension,
    fig_pmd_noise,
    fig_parameters,
    fig_retention,
    fig_trainability,
    fig_witness,
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summaries", type=Path, default=ROOT / "results" / "summaries")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES:
        fig(args.summaries, args.out)


if __name__ == "__main__":
    main()
