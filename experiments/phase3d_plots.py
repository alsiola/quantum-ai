"""Static plots for the Phase 3d summary (same style as the Phase 2 plots)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from phase2_plots import GRID, INK2, MUTED, plt

from qmem import Circuit, ControlledRotationBlock, Instrument
from qmem.metrics import banaszek_bound, estimation_fidelity, operation_fidelity

COLOURS = ["#2a78d6", "#eb6834", "#1a9e77", "#7b5ea7"]


def weak_family(n_memory: int, n_output: int, gs: torch.Tensor) -> tuple[list[float], list[float]]:
    """(G, F) of ``D_y(g)`` (shared g, no V or W) on ``n_memory`` qubits."""
    n = n_memory + n_output
    pairs = [(k % n_memory, n_memory + k) for k in range(n_output)]
    gg, ff = [], []
    for g in gs:
        block = ControlledRotationBlock(pairs, n, float(g), shared=True)
        inst = Instrument.from_unitary(Circuit([block], n).unitary(), n_memory, n_output)
        gg.append(estimation_fidelity(inst))
        ff.append(operation_fidelity(inst))
    return gg, ff


def tradeoff(path: Path, per_run: list[dict[str, Any]]) -> None:
    gs = torch.linspace(0, math.pi / 2, 61)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # (a) one qubit: the closed-form weak family against the bound
    ax = axes[0]
    g2 = torch.linspace(0.5, 2 / 3, 200)
    ax.plot(g2, banaszek_bound(g2, 2), color=INK2, label="Banaszek bound, d = 2")
    s2, c = torch.sin(gs) ** 2, torch.cos(gs)
    ax.plot((3 + s2) / 6, (2 + c) / 3, color=COLOURS[0], label="weak family D_y(g), d = 2")
    for g in (math.pi / 8, math.pi / 4, 3 * math.pi / 8):
        x, y = (3 + math.sin(g) ** 2) / 6, (2 + math.cos(g)) / 3
        ax.plot(x, y, "o", color=COLOURS[0], ms=4)
        ax.annotate(
            f"g={g / math.pi:.3g}π",
            (x, y),
            (4, 4),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    ax.set_xlabel("estimation fidelity G")
    ax.set_ylabel("operation fidelity F (best unitary correction)")
    ax.set_title("(a) one qubit: weak measurement vs optimum")
    ax.legend(loc="lower left", fontsize=7)
    # (b) two memory qubits: trained models
    ax = axes[1]
    g4 = torch.linspace(0.25, 0.4, 200)
    ax.plot(g4, banaszek_bound(g4, 4), color=INK2, label="Banaszek bound, d = 4")
    for n_out, ls in ((1, "-"), (2, "--")):
        x, y = weak_family(2, n_out, gs)
        ax.plot(x, y, ls, color=GRID, lw=1.5, label=f"D_y(g), {2**n_out} symbols")
    layouts = list(dict.fromkeys(r["labels"]["layout"] for r in per_run))
    for i, lay in enumerate(layouts):
        for n_sym, marker in ((2, "o"), (4, "s")):
            rs = [
                r for r in per_run if r["labels"]["layout"] == lay and r.get("n_outcomes") == n_sym
            ]
            ok = [r for r in rs if r["success"]]
            bad = [r for r in rs if not r["success"]]
            col = COLOURS[i % len(COLOURS)]
            ax.scatter(
                [r["estimation_fidelity"] for r in ok],
                [r["operation_fidelity"] for r in ok],
                s=14,
                marker=marker,
                color=col,
                alpha=0.7,
                label=f"{lay} ({n_sym} sym)",
            )
            ax.scatter(
                [r["estimation_fidelity"] for r in bad],
                [r["operation_fidelity"] for r in bad],
                s=14,
                marker=marker,
                facecolors="none",
                edgecolors=col,
                alpha=0.7,
            )
    ax.set_xlabel("estimation fidelity G")
    ax.set_title("(b) trained 2-qubit memories (hollow: failed runs)")
    ax.legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strength(path: Path, per_run: list[dict[str, Any]]) -> None:
    tasks = list(dict.fromkeys(r["labels"]["task"] for r in per_run))
    layouts = list(dict.fromkeys(r["labels"]["layout"] for r in per_run))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.8 / len(layouts)
    gen = torch.Generator().manual_seed(0)
    for i, lay in enumerate(layouts):
        col = COLOURS[i % len(COLOURS)]
        for j, t in enumerate(tasks):
            rs = [
                r
                for r in per_run
                if r["labels"]["layout"] == lay
                and r["labels"]["task"] == t
                and "operational_strength" in r
            ]
            x0 = j - 0.4 + width * (i + 0.5)
            jit = (torch.rand(len(rs), generator=gen) - 0.5) * width * 0.6
            for r, dx in zip(rs, jit, strict=True):
                kw = {"color": col} if r["success"] else {"facecolors": "none", "edgecolors": col}
                ax.scatter(x0 + float(dx), r["operational_strength"], s=12, **kw)
        ax.scatter([], [], s=12, color=col, label=lay)
    ax.set_xticks(range(len(tasks)), tasks, rotation=15)
    ax.set_ylabel("operational strength η")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Operational strength of every trained instrument (hollow: failed runs)")
    ax.legend(fontsize=7, ncol=len(layouts), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(summary_dir: Path, per_run: list[dict[str, Any]]) -> None:
    tradeoff(summary_dir / "tradeoff.png", per_run)
    strength(summary_dir / "strength.png", per_run)
