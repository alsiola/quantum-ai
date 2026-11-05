"""Proof of concept for Phase 1b.

1. Builds a 2-memory-qubit, 1-ancilla model from a TOML config.
2. Shows memory across measurements: p(next symbol | past outcomes) depends on
   the past outcomes (teacher-forced conditioning on each prefix).
3. Runs a few Adam steps of teacher-forced NLL on a sticky Markov chain and
   checks the loss goes down.

Writes ``results/poc_memory/<timestamp>/summary.json`` with run metadata.
"""

from __future__ import annotations

import argparse
import math
import time
import tomllib
from pathlib import Path

import torch

from qmem import ModelConfig, build_model
from qmem.data import all_sequences, markov_sequences
from qmem.records import new_run_dir, run_metadata, save_json

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "poc_memory.toml"


def conditional_table(model, max_prefix: int) -> list[dict]:
    rows = []
    with torch.no_grad():
        for k in range(max_prefix + 1):
            prefixes = all_sequences(model.n_symbols, k) if k else torch.zeros(1, 0, dtype=int)
            pred = model.predictive_probs(prefixes)[:, -1]
            for prefix, p in zip(prefixes.tolist(), pred.tolist(), strict=True):
                rows.append({"prefix": prefix, "p_next": p})
    return rows


def binary_entropy_nats(p: float) -> float:
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    start = time.perf_counter()
    cfg = tomllib.loads(Path(args.config).read_text())
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = build_model(model_cfg)
    print(
        f"model: {model.n_memory} memory qubits, {model.n_output} output ancilla, "
        f"{model.n_parameters()} trainable parameters"
    )

    # -- 1. memory across measurements -----------------------------------------
    print("\np(next = 1 | past outcomes) at initialisation:")
    table = conditional_table(model, max_prefix=2)
    for row in table:
        print(f"  prefix {row['prefix']!s:10s} -> {row['p_next'][1]:.4f}")
    after_one = [r["p_next"][1] for r in table if len(r["prefix"]) == 1]
    spread = max(after_one) - min(after_one)
    print(f"  |p(1|0) - p(1|1)| = {spread:.4f} (nonzero => the memory carries the past outcome)")

    # -- 2. a few gradient steps on a trivial dataset ---------------------------
    d = cfg["data"]
    p_stay = d["p_stay"]
    transition = torch.tensor([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    data = markov_sequences(d["n_train"], d["length"], transition, seed=d["seed"])
    # Optimal teacher-forced NLL per symbol: first symbol log 2, then H(p_stay).
    optimum = (math.log(2) + (d["length"] - 1) * binary_entropy_nats(p_stay)) / d["length"]
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    losses = []
    for step in range(t["steps"]):
        opt.zero_grad()
        loss = -model.log_prob(data).mean() / d["length"]
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == t["steps"] - 1:
            print(f"  step {step:3d}  NLL/symbol = {loss.item():.4f} nats")
    with torch.no_grad():
        final = (-model.log_prob(data).mean() / d["length"]).item()
    print(
        f"\nNLL/symbol: start {losses[0]:.4f} -> end {final:.4f} "
        f"(data entropy rate bound {optimum:.4f}, memoryless bound {math.log(2):.4f})"
    )

    table_after = conditional_table(model, max_prefix=1)
    print("p(next = 1 | past outcomes) after training:")
    for row in table_after:
        print(f"  prefix {row['prefix']!s:10s} -> {row['p_next'][1]:.4f}")

    elapsed = time.perf_counter() - start
    out = new_run_dir("poc_memory")
    save_json(
        out / "summary.json",
        {
            "meta": run_metadata(config=cfg, seed=model_cfg.seed),
            "conditional_probs_init": table,
            "conditional_probs_trained": table_after,
            "losses": losses,
            "final_nll_per_symbol": final,
            "optimal_nll_per_symbol": optimum,
            "strength_final": [
                b.strength.tolist() for b in model.circuit.blocks if hasattr(b, "strength")
            ],
            "diagnostics": dict(model.diagnostics),
            "elapsed_seconds": elapsed,
        },
    )
    print(f"\nwrote {out / 'summary.json'} ({elapsed:.1f}s)")
    if not final < losses[0]:
        raise SystemExit("loss did not decrease")


if __name__ == "__main__":
    main()
