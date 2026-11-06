"""Phase 3f (issue #11): cross-model grid on the memory benchmark suite.

One command runs everything in ``experiments/configs/phase3f.toml``:

* every model family (A general, A feed-forward, D, D_diss, HMM by EM and by gradient,
  uMPS, HQMM, a parameter-matched GRU, the n-gram model) on every task of the pruned
  suite, at state dimension D = 2^n;
* gradient cells, A included, tuned on validation over the same learning-rate grid with
  disjoint tuning seeds, then evaluated with seeds 0-9;
* the dimensional-advantage test: 1-qubit quantum models (exact and trained) against the
  best 2-state classical model, with extra EM restarts, a large-data arm and a
  long-sequence re-scoring, compared with the 2-state unifilar bound
  (``qmem.complexity.unifilar_merge_bound``).

The grid runs in waves of increasing state dimension, each tuned and then evaluated.
Outputs:

* ``results/phase3f/<timestamp>/``: run records (``runs/<group>/seed<k>.json`` + ``.pt``),
  raw, gitignored, plus ``post.json`` (long-test and recall scores, cached);
* ``<summary_dir>/``: ``summary.md``, CSVs, ``summary.json`` and plots (committed).

``--from-run <dir>`` reuses saved records and runs only the missing ones.
``--no-train`` re-summarises saved records without training. ``--tasks a,b`` and
``--max-dim`` restrict the run (smoke tests).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import phase3a as p3a

from qmem.baselines.rnn import matched_hidden_size
from qmem.baselines.runner import BaselineSpec, build_baseline, load_baseline, run_baseline
from qmem.benchmarks import marked_recall_metrics, recall_metrics
from qmem.complexity import (
    epsilon_machine,
    pmd_qubit_instrument,
    qmachine_instrument,
    unifilar_merge_bound,
)
from qmem.experiment import build_splits, load_model
from qmem.records import REPO_ROOT, new_run_dir, run_metadata, save_json
from qmem.tasks import QuantumSourceTask, make_task

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "phase3f.toml"
CIRCUIT = ("A", "A_ff", "D", "D_diss")
GRID_FAMILIES = ("A", "A_ff", "D", "D_diss", "hmm", "umps", "hqmm")


# -- tasks and cells ---------------------------------------------------------------------
def resolve_tasks(cfg: dict, suite: dict, only: list[str] | None) -> dict[str, dict]:
    out = {}
    for name, t in cfg["tasks"].items():
        if only and name not in only:
            continue
        spec = dict(t["spec"]) if "spec" in t else dict(suite["suite"]["tasks"][t["suite"]])
        task = make_task(spec)
        n_out = max(1, (task.n_symbols - 1).bit_length())
        out[name] = {
            "spec": spec,
            "n_symbols": task.n_symbols,
            "n_output": n_out,
            "ns": list(t["ns"]),
            "circuit_ns": list(t.get("circuit_ns", t["ns"])),
            "data": {**cfg["data"], **t.get("data", {})},
        }
    return out


@dataclass
class Cell:
    task: str
    key: str  # e.g. "A/D4", "hmm-em/D2", "gru/matched", "hqmm/D2/large"
    family: str  # config family name: A, A_ff, D, D_diss, hmm, umps, hqmm, gru, ngram
    kind: str  # runner family: A, hmm, umps, hqmm, gru, ngram
    method: str  # gradient, em, count
    model: dict[str, Any]
    state_dim: int
    n_parameters: int
    wave: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reuse_from: str | None = None
    seeds: list[int] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    fit: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)

    @property
    def tuned(self) -> bool:
        return len(self.candidates) > 1 and self.reuse_from is None


def a_model(cfg: dict, fam: str, n: int, n_out: int) -> dict[str, Any]:
    f = cfg["families"][fam]
    m = {**cfg["a_model"], **f.get("model", {}), **f.get("by_n_output", {}).get(str(n_out), {})}
    return {**m, "n_memory": n, "n_output": n_out}


def count_parameters(kind: str, model: dict, n_symbols: int) -> int:
    if kind == "A":
        from qmem.ablations import build_variant

        return build_variant({**model, "seed": 0}).n_parameters()
    return build_baseline(kind, model, n_symbols, 0).n_parameters()


def hmm_model(cfg: dict, d: int) -> dict[str, Any]:
    return {"n_states": d, "init_scale": cfg["families"]["hmm"]["init_scale"]}


def build_cells(cfg: dict, tasks: dict[str, dict], max_dim: int | None) -> list[Cell]:
    lrs = cfg["tuning"]["lr_grid"]
    cands = p3a.lr_candidates(lrs)
    fams = cfg["families"]
    em_fit = fams["hmm"]["em"]
    dim_cfg = cfg["dimensional"]
    cells: list[Cell] = []
    for name, t in tasks.items():
        y, n_out = t["n_symbols"], t["n_output"]

        def add(key, family, kind, method, model, dim, wave=None, name=name, y=y, **kw):
            if max_dim is not None and dim > max_dim:
                return
            n_par = count_parameters(kind, model, y) if kind != "ngram" else 0
            c = Cell(name, key, family, kind, method, model, dim, n_par,
                     wave if wave is not None else dim, **kw)  # fmt: skip
            if method == "gradient" and not c.candidates:
                c.candidates = cands
            cells.append(c)

        add("ngram", "ngram", "ngram", "count", {"orders": cfg["ngram"]["orders"],
            "alphas": cfg["ngram"]["alphas"]}, 0, wave=2)  # fmt: skip
        for n in t["ns"]:
            d = 2**n
            for fam in CIRCUIT:
                if n in t["circuit_ns"] and n <= fams[fam].get("max_n", 99):
                    add(f"{fam}/D{d}", fam, "A", "gradient", a_model(cfg, fam, n, n_out), d)
            add(f"hmm-em/D{d}", "hmm", "hmm", "em", hmm_model(cfg, d), d, fit=em_fit)
            add(f"hmm-grad/D{d}", "hmm", "hmm", "gradient", hmm_model(cfg, d), d)
            if n <= fams["umps"].get("max_n", 99):
                add(f"umps/D{d}", "umps", "umps", "gradient",
                    {"bond_dim": d, **fams["umps"]["model"]}, d)  # fmt: skip
            if n <= fams["hqmm"].get("max_n", 99):
                add(f"hqmm/D{d}", "hqmm", "hqmm", "gradient",
                    {"n_memory": n, **fams["hqmm"]["model"]}, d)  # fmt: skip
        # matched-parameter arm: budget = the reference cell's count (A general, n = 2)
        ref_fam, ref_d = cfg["experiment"]["match_reference"].split("/D")
        budget = count_parameters("A", a_model(cfg, ref_fam, int(ref_d).bit_length() - 1, n_out), y)
        t["budget"] = budget
        h = matched_hidden_size("gru", y, budget)
        add("gru/matched", "gru", "gru", "gradient", {"hidden": h}, h, wave=2,
            labels={"a_parameters": budget})  # fmt: skip
        d_m = 1
        while count_parameters("hmm", hmm_model(cfg, d_m + 1), y) <= budget:
            d_m += 1
        if d_m not in {2**n for n in t["ns"]} and d_m > 1:
            add(f"hmm-em/D{d_m}", "hmm", "hmm", "em", hmm_model(cfg, d_m), d_m, wave=2,
                fit=em_fit)  # fmt: skip
            add(f"hmm-grad/D{d_m}", "hmm", "hmm", "gradient", hmm_model(cfg, d_m), d_m, wave=2)
        # the dimensional test
        if name in dim_cfg["tasks"]:
            restarts = list(range(dim_cfg["em_restarts"]))
            add("hmm-em/D2/restarts", "hmm", "hmm", "em", hmm_model(cfg, 2), 2, seeds=restarts,
                fit=dim_cfg["em"])  # fmt: skip
            big = dict(dim_cfg["large_data"])
            add("hmm-em/D2/restarts/large", "hmm", "hmm", "em", hmm_model(cfg, 2), 2, wave=3,
                seeds=restarts, fit=dim_cfg["em"], data=big)  # fmt: skip
            for fam in dim_cfg["large_data_families"]:
                src = next((c for c in cells if c.task == name and c.family == fam
                            and c.state_dim == 2 and c.method == "gradient"), None)  # fmt: skip
                if src is None:
                    continue
                add(f"{src.key}/large", fam, src.kind, "gradient", src.model, 2, wave=3,
                    reuse_from=src.key, data=big)  # fmt: skip
    for c in cells:
        c.labels = {**c.labels, "state_dim": c.state_dim, "family_cfg": c.family}
    return cells


def make_spec(cfg, tasks, cell: Cell, stage: str, cand: dict | None, seed: int) -> BaselineSpec:
    t = tasks[cell.task]
    train = {**cfg["train"], **(cand["train"] if cand else {})}
    group = f"{stage}/{cell.task}/{cell.key}"
    if stage == "tune":
        group += "/" + cand["tag"]
    labels = {
        "task": cell.task,
        "cell": cell.key,
        "family": cell.family,
        "method": cell.method,
        "stage": stage,
        "candidate": cand["tag"] if cand else "",
        **cell.labels,
    }
    return BaselineSpec(
        group=group,
        task={**t["spec"], "name": cell.task},
        data={**t["data"], **cell.data},
        family=cell.kind,
        model=cell.model,
        method=cell.method,
        train=train if cell.method == "gradient" else {},
        fit=cell.fit if cell.method == "em" else {},
        seed=seed,
        labels=labels,
    )


def tuning_specs(cfg, tasks, cells):
    return [
        make_spec(cfg, tasks, c, "tune", cand, s)
        for c in cells
        if c.tuned
        for cand in c.candidates
        for s in cfg["experiment"]["tuning_seeds"]
    ]


def select(cells: list[Cell], records: list[dict]) -> tuple[dict, list[dict]]:
    choice, table = p3a.select(cells, records)  # duck-typed: task, key, candidates, tuned
    for c in cells:
        if c.reuse_from is not None and (c.task, c.reuse_from) in choice:
            choice[(c.task, c.key)] = choice[(c.task, c.reuse_from)]
    return choice, table


def eval_specs(cfg, tasks, cells, choice):
    specs = []
    for c in cells:
        if c.method == "count":
            specs.append(make_spec(cfg, tasks, c, "eval", None, 0))
            continue
        cand = None
        if c.method == "gradient":
            cand = choice.get((c.task, c.key))
            if cand is None:
                raise SystemExit(f"no tuning result for {c.task}/{c.key}")
        for s in c.seeds if c.seeds is not None else cfg["experiment"]["seeds"]:
            specs.append(make_spec(cfg, tasks, c, "eval", cand, s))
    return specs


def run_and_save_params(out_dir: Path, spec: BaselineSpec) -> dict[str, Any]:
    """Run one spec in a worker and save its parameters there, returning a tensor-free record.

    Returning tensors from a worker goes through torch's shared-memory manager, which timed
    out under the shared machine's load ("Shared memory manager connection has timed out").
    """
    rec = run_baseline(spec)
    state = rec.pop("state_dict", None)
    if state is not None:
        path = Path(out_dir) / "runs" / rec["group"] / f"seed{rec['seed']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
    return rec


def _worker_init() -> None:
    torch.set_num_threads(1)


def run_stage(specs, records, out_dir: Path, workers: int, label: str) -> list[dict]:
    """``phase3a.run_stage`` with parameters saved in the workers (see above)."""
    have = {(r["group"], r["seed"]) for r in records}
    for s in specs:
        r = next((r for r in records if (r["group"], r["seed"]) == (s.group, s.seed)), None)
        if r is not None and r["meta"]["config"]["model"] != s.config()["model"]:
            raise SystemExit(f"saved run {s.group} seed {s.seed} has a different config")
    todo = [s for s in specs if (s.group, s.seed) not in have]
    print(f"[{label}] {len(specs)} runs, {len(todo)} to do", flush=True)
    t0 = time.perf_counter()
    new: list[dict] = []

    def done(rec: dict) -> None:
        save_json(Path(out_dir) / "runs" / rec["group"] / f"seed{rec['seed']}.json", rec)
        new.append(rec)
        gap = rec.get("test", {}).get("gap_nats", math.nan)
        print(f"  {rec['group']:<44s} seed {rec['seed']}: {rec['status']:<13s} "
              f"gap {gap:+.4f}  ({rec['wall_seconds']:.0f}s)", flush=True)  # fmt: skip

    if todo:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init) as ex:
            futs = [ex.submit(run_and_save_params, out_dir, s) for s in todo]
            for fut in cf.as_completed(futs):
                done(fut.result())
    print(f"[{label}] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return records + new


def run_waves(cfg, tasks, cells, records, out_dir, workers, train: bool):
    waves = sorted({c.wave for c in cells})
    choice, table = {}, []
    for w in waves:
        wc = [c for c in cells if c.wave == w]
        ts = tuning_specs(cfg, tasks, wc)
        if train:
            records = run_stage(ts, records, out_dir, workers, f"wave {w} tuning")
        choice, table = select(cells, records)
        es = eval_specs(cfg, tasks, [c for c in wc if c.method != "gradient"
                                     or (c.task, c.key) in choice], choice)  # fmt: skip
        if train:
            records = run_stage(es, records, out_dir, workers, f"wave {w} eval")
    return records, choice, table


# -- post-processing: reload models ---------------------------------------------------
def load_trained(out_dir: Path, rec: dict) -> Any:
    rec = dict(rec)
    rec["state_dict"] = torch.load(
        out_dir / "runs" / rec["group"] / f"seed{rec['seed']}.pt", weights_only=True
    )
    if rec["meta"]["config"].get("family") == "A" or "family" not in rec["meta"]["config"]:
        return load_model(rec)
    return load_baseline(rec)


def exact_one_qubit(task_spec: dict) -> Any:
    """The exact 1-qubit model of a dimensional-test task, as a source (None if none exists)."""
    task = make_task(task_spec)
    if task_spec["kind"] == "ghafari":
        inst, _ = qmachine_instrument(task)
    elif task_spec["kind"] == "pmd_renewal":
        inst, _ = pmd_qubit_instrument(task_spec["period"], task_spec["gamma"], task_spec["v"])
    else:
        return None
    if inst.dim != 2:
        return None
    return QuantumSourceTask(inst, name="exact")  # stationary start, as the source


def paired_gap(task, model_lp: torch.Tensor, seq: torch.Tensor) -> tuple[float, float]:
    d = (task.log_probs(seq) - model_lp).mean(-1)  # per sequence, nats/symbol
    return float(d.mean()), float(d.std() / math.sqrt(len(d)))


def post_process(cfg, tasks, cells, records, out_dir: Path) -> dict:
    """Long-test re-scoring (dimensional test) and recall metrics, cached in post.json."""
    path = out_dir / "post.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    cache.setdefault("long", {})
    cache.setdefault("recall", {})
    evals = [r for r in records if r["group"].startswith("eval/") and "test" in r]
    by_group: dict[str, list[dict]] = {}
    for r in evals:
        by_group.setdefault(r["group"], []).append(r)
    lt = cfg["dimensional"]["long_test"]
    with torch.no_grad():
        for name in cfg["dimensional"]["tasks"]:
            if name not in tasks:
                continue
            task = make_task(tasks[name]["spec"])
            seq = task.sample(lt["n"], lt["length"], lt["seed"])
            ex = exact_one_qubit(tasks[name]["spec"])
            k = f"{name}|exact"
            if ex is not None and k not in cache["long"]:
                g, se = paired_gap(task, ex.log_probs(seq), seq)
                cache["long"][k] = {"gap": g, "se": se}
            for c in cells:
                if c.task != name or c.state_dim != 2 or c.method == "count":
                    continue
                sel = p3a.val_selected(by_group.get(f"eval/{name}/{c.key}", []))
                if sel is None:
                    continue
                k = f"{name}|{c.key}|{sel['seed']}"
                if k in cache["long"]:
                    continue
                model = load_trained(out_dir, sel)
                g, se = paired_gap(task, model.step_log_probs(seq), seq)
                cache["long"][k] = {"gap": g, "se": se, "seed": sel["seed"]}
                save_json(path, cache)
        for name, t in tasks.items():
            kind = t["spec"]["kind"]
            if kind not in ("marked_recall", "delayed_recall"):
                continue
            task = make_task(t["spec"])
            test = build_splits(task, t["data"]).test
            for group, recs in by_group.items():
                if recs[0]["labels"]["task"] != name or recs[0]["labels"]["method"] == "count":
                    continue
                for r in recs:
                    k = f"{group}|{r['seed']}"
                    if k in cache["recall"]:
                        continue
                    model = load_trained(out_dir, r)
                    if kind == "marked_recall":
                        m = marked_recall_metrics(model, task, test, [0, 1, 2, 4, 8, 16, 32])
                        cache["recall"][k] = {
                            "accuracy": m["recall_accuracy"],
                            "value_nll": m["recall_value_nll_nats"],
                            "source_accuracy": m["source_recall_accuracy"],
                        }
                    else:
                        m = recall_metrics(model, task, test)
                        cache["recall"][k] = {
                            "accuracy": m["recall_accuracy"],
                            "value_nll": m["recall_nll_nats"],
                            "source_accuracy": m["source_recall_accuracy"],
                        }
                save_json(path, cache)
    save_json(path, cache)
    return cache


# -- summaries ------------------------------------------------------------------------
FAMILY_LABEL = {
    "A": "A, general", "A_ff": "A, feed-forward", "D": "D", "D_diss": "D_diss",
    "hmm": "HMM", "umps": "uMPS", "hqmm": "HQMM", "gru": "GRU", "ngram": "n-gram",
}  # fmt: skip


def fmt(x, nd=4):
    return p3a.fmt(x, nd)


def is_grid(r: dict) -> bool:
    """A cell of the main grid (not the dimensional test's extra restarts or large-data arm)."""
    return not r["cell"].endswith(("/restarts", "/large"))


def best_cell(rows: list[dict], task: str, family: str, dim: int | None = None,
              keys: set[str] | None = None) -> dict | None:  # fmt: skip
    """The family's grid cell at ``dim`` (HMM: better of EM/grad by validation-selected NLL)."""
    cands = [
        r for r in rows
        if r["task"] == task and r["family"] == family
        and (dim is None or r["state_dim"] == dim)
        and is_grid(r)
        and (keys is None or r["cell"] in keys)
    ]  # fmt: skip
    if not cands:
        return None
    return min(cands, key=lambda r: r["selected_val_nll"])


def cell_str(r: dict | None, params: bool = False) -> str:
    if r is None:
        return "-"
    s = f"{fmt(r['gap_median'])} ({r['n_runs'] - r['n_success']})"
    if r["family"] == "hmm":
        s += " EM" if r["method"] == "em" else ""
    if params:
        s += f" [{r['n_parameters']}]"
    if r.get("suspect"):
        s += " †"
    return s


def flag_suspects(rows: list[dict], margin: float) -> list[dict]:
    """Cells that do worse than a smaller cell of the same family on the same task."""
    out = []
    grid = [r for r in rows if is_grid(r)
            and r["family"] not in ("ngram", "gru")]  # fmt: skip
    for r in grid:
        smaller = [s for s in grid if s["task"] == r["task"] and s["family"] == r["family"]
                   and s["method"] == r["method"] and s["state_dim"] < r["state_dim"]]  # fmt: skip
        for s in smaller:
            worse_gap = r["gap_median"] > s["gap_median"] + margin
            worse_fail = (r["n_runs"] - r["n_success"]) > (s["n_runs"] - s["n_success"])
            if worse_gap or worse_fail:
                r["suspect"] = True
                out.append({
                    "task": r["task"], "cell": r["cell"], "vs": s["cell"],
                    "median_gap": r["gap_median"], "vs_median_gap": s["gap_median"],
                    "failures": r["n_runs"] - r["n_success"],
                    "vs_failures": s["n_runs"] - s["n_success"],
                    "selected_gap": r["selected_gap"], "vs_selected_gap": s["selected_gap"],
                })  # fmt: skip
                break
    return out


def matched_dim_tables(rows, tasks, dims=(2, 4, 8)):
    out = []
    for d in dims:
        for t in tasks:
            entry = {"task": t, "state_dim": d}
            for fam in GRID_FAMILIES:
                r = best_cell(rows, t, fam, d)
                entry[fam] = None if r is None else {k: r[k] for k in (
                    "cell", "method", "gap_median", "selected_gap", "n_success", "n_runs",
                    "n_parameters", "flops_per_step", "suspect")} | {"family": fam}  # fmt: skip
            if any(entry[f] for f in GRID_FAMILIES):
                out.append(entry)
    return out


def matched_param_table(rows, tasks_cfg, tasks):
    out = []
    for t in tasks:
        budget = tasks_cfg[t]["budget"]
        entry = {"task": t, "budget": budget}
        for fam in (*GRID_FAMILIES, "gru"):
            fam_rows = [r for r in rows if r["task"] == t and r["family"] == fam
                        and is_grid(r)]  # fmt: skip
            under = [r for r in fam_rows if r["n_parameters"] <= budget]
            if under:
                top = max(r["n_parameters"] for r in under)
                pick = min([r for r in under if r["n_parameters"] == top],
                           key=lambda r: r["selected_val_nll"])  # fmt: skip
                if fam == "hmm":  # better of EM / gradient at that size
                    same = [r for r in under if r["state_dim"] == pick["state_dim"]]
                    pick = min(same, key=lambda r: r["selected_val_nll"])
                entry[fam] = {**pick, "over_budget": False}
            elif fam_rows:
                low = min(fam_rows, key=lambda r: (r["n_parameters"], r["selected_val_nll"]))
                entry[fam] = {**low, "over_budget": True}
            else:
                entry[fam] = None
        out.append(entry)
    return out


def reach_table(rows, tasks, gap):
    out = []
    for t in tasks:
        for fam in GRID_FAMILIES:
            fam_rows = sorted(
                [r for r in rows if r["task"] == t and r["family"] == fam
                 and is_grid(r)],
                key=lambda r: (r["state_dim"], r["n_parameters"]),
            )  # fmt: skip
            if not fam_rows:
                continue
            med = next((r for r in fam_rows if r["gap_median"] < gap), None)
            sel = next((r for r in fam_rows if r["selected_gap"] < gap), None)
            out.append({
                "task": t, "family": fam,
                "dims_run": sorted({r["state_dim"] for r in fam_rows}),
                "median_dim": med["state_dim"] if med else None,
                "median_params": med["n_parameters"] if med else None,
                "selected_dim": sel["state_dim"] if sel else None,
                "selected_params": sel["n_parameters"] if sel else None,
            })  # fmt: skip
    return out


def dimensional_table(cfg, rows, tasks_cfg, post):
    out = []
    for t in cfg["dimensional"]["tasks"]:
        if t not in tasks_cfg:
            continue
        task = make_task(tasks_cfg[t]["spec"])
        mb = unifilar_merge_bound(task, 2)
        em = epsilon_machine(task)
        entry = {
            "task": t,
            "entropy_rate_nats": task.entropy_rate(),
            "causal_states": em["n_states"],
            "unifilar_bound_nats": mb["bound_nats"],
            "unifilar_bound_realisable": mb["realisable"],
            "exact": post["long"].get(f"{t}|exact"),
            "models": {},
        }
        for r in rows:
            if r["task"] != t or r["state_dim"] != 2 or r["family"] in ("ngram", "gru"):
                continue
            seed = r["selected_seed"]
            lg = post["long"].get(f"{t}|{r['cell']}|{seed}")
            entry["models"][r["cell"]] = {
                "family": r["family"], "method": r["method"], "n_runs": r["n_runs"],
                "n_success": r["n_success"], "median_gap": r["gap_median"],
                "selected_gap": r["selected_gap"], "selected_val_nll": r["selected_val_nll"],
                "selected_seed": seed, "n_parameters": r["n_parameters"],
                "long_gap": lg["gap"] if lg else None, "long_se": lg["se"] if lg else None,
            }  # fmt: skip
        for arm, suffix in (("standard", ""), ("large", "/large")):
            cl = {k: v for k, v in entry["models"].items() if v["family"] == "hmm"
                  and (k.endswith("/large") if suffix else not k.endswith("/large"))}  # fmt: skip
            if cl:
                best = min(cl, key=lambda k: cl[k]["selected_val_nll"])
                entry[f"best_classical_{arm}"] = best
        out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or (list(rows[0]) if rows else ["empty"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})  # fmt: skip


def plots(rows, tasks, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Categorical slots 1-8 of the validated reference palette, in fixed order, with a
    # distinct marker per family as the secondary encoding (main.csv is the table view).
    styles = {  # family: (color, marker)
        "A": ("#2a78d6", "o"), "A_ff": ("#eb6834", "s"), "D": ("#1baf7a", "v"),
        "D_diss": ("#eda100", "^"), "hmm": ("#e87ba4", "D"), "umps": ("#008300", "P"),
        "hqmm": ("#6250d6", "X"), "gru": ("#e34948", "*"),
    }  # fmt: skip
    floor = 1e-4
    for xkey, fname, xlabel in (("n_parameters", "gap_vs_params.png", "parameters"),
                                ("state_dim", "gap_vs_dim.png", "state dimension D")):  # fmt: skip
        n = len(tasks)
        ncol = 4
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, t in zip(axes.flat, tasks, strict=False):
            for fam, (col, mk) in styles.items():
                if xkey == "state_dim" and fam == "gru":
                    continue
                pts = []
                dims = sorted({r["state_dim"] for r in rows if r["task"] == t and r["family"] == fam
                               and is_grid(r)})  # fmt: skip
                for d in dims:
                    r = best_cell(rows, t, fam, d)
                    if r is not None and math.isfinite(r["gap_median"]):
                        pts.append((r[xkey], max(r["gap_median"], floor)))
                if pts:
                    pts.sort()
                    ax.plot(*zip(*pts, strict=True), color=col, marker=mk, ms=5, lw=1.2,
                            label=FAMILY_LABEL[fam])  # fmt: skip
            ax.axhline(0.01, color="#999999", lw=0.8, ls="--")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        for ax in axes.flat:
            h, lab = ax.get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, lab
        fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8, frameon=False)
        fig.supxlabel(xlabel, fontsize=9, y=0.03)
        fig.supylabel(f"median test gap (nats/symbol; floored at {floor:g}); dashed: failure "
                      "threshold", fontsize=9)  # fmt: skip
        fig.tight_layout(rect=(0.01, 0.05, 1, 1))
        fig.savefig(out / fname, dpi=110)
        plt.close(fig)


def write_summary(out, cfg, tasks_cfg, tasks, rows, tuning_table, suspects, mdim, mpar, reach,
                  dimtab, retention, meta, extra):  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    main_cols = [
        "task", "cell", "family", "method", "candidate", "state_dim", "n_parameters",
        "flops_per_step", "n_runs", "n_success", "failure_rate", "entropy_rate_nats",
        "gap_mean", "gap_se", "gap_median", "gap_best", "gap_worst", "selected_seed",
        "selected_val_nll", "selected_gap", "steps_mean", "train_seconds_mean",
        "tuning_wall_seconds", "suspect",
    ]  # fmt: skip
    write_csv(out / "main.csv", rows, main_cols)
    write_csv(out / "tuning.csv", tuning_table)
    write_csv(out / "suspect.csv", suspects)
    write_csv(out / "reach.csv", reach)
    flat_dim = []
    for e in mdim:
        for fam in GRID_FAMILIES:
            if e[fam]:
                flat_dim.append({"task": e["task"], "state_dim": e["state_dim"], **e[fam]})
    write_csv(out / "matched_dim.csv", flat_dim)
    flat_par = []
    for e in mpar:
        for fam in (*GRID_FAMILIES, "gru"):
            if e[fam]:
                r = e[fam]
                flat_par.append({"task": e["task"], "budget": e["budget"], "family": fam,
                                 **{k: r[k] for k in ("cell", "method", "state_dim", "n_parameters",
                                    "flops_per_step", "gap_median", "selected_gap", "n_success",
                                    "n_runs", "over_budget")}})  # fmt: skip
    write_csv(out / "matched_params.csv", flat_par)
    flat_dimtest = []
    for e in dimtab:
        for k, v in e["models"].items():
            flat_dimtest.append({"task": e["task"], "cell": k, **v,
                                 "entropy_rate_nats": e["entropy_rate_nats"],
                                 "unifilar_bound_nats": e["unifilar_bound_nats"]})  # fmt: skip
    write_csv(out / "dimensional.csv", flat_dimtest)
    write_csv(out / "retention.csv", retention)
    save_json(out / "summary.json", {
        "meta": meta, "rows": rows, "tuning": tuning_table, "suspects": suspects,
        "matched_dim": mdim, "matched_params": mpar, "reach": reach, "dimensional": dimtab,
        "retention": retention, "tasks": tasks_cfg, **extra,
    })  # fmt: skip

    gap = cfg["experiment"]["success_gap"]
    fams = GRID_FAMILIES
    L = ["# Phase 3f summary (generated by experiments/phase3f.py)", ""]
    L.append(
        f"Commit `{(meta.get('git_commit') or '?')[:7]}`, dirty: {meta.get('git_dirty')}. "
        f"{extra['n_records']} run records ({extra['n_errors']} errors); "
        f"{extra['cpu_seconds'] / 3600:.1f} CPU-hours summed over runs."
    )
    L += ["", f"Entries: median test gap over the evaluation seeds (nats/symbol), failed runs "
          f"(gap ≥ {gap} or crashed) in brackets. HMM: the better of EM and gradient on "
          "validation (EM marked). † = suspect: worse than a smaller cell of the same family "
          "(`suspect.csv`). - = not run (pruned).", ""]  # fmt: skip
    for d in (2, 4, 8):
        sub = [e for e in mdim if e["state_dim"] == d]
        if not sub:
            continue
        L += [f"## Matched state dimension D = {d}: median gap (failures)", ""]
        L.append("| task | " + " | ".join(FAMILY_LABEL[f] for f in fams) + " |")
        L.append("|---" * (len(fams) + 1) + "|")
        for e in sub:
            L.append(f"| {e['task']} | " + " | ".join(cell_str(e[f]) for f in fams) + " |")
        L += ["", f"### D = {d}: validation-selected run's test gap", ""]
        L.append("| task | " + " | ".join(FAMILY_LABEL[f] for f in fams) + " |")
        L.append("|---" * (len(fams) + 1) + "|")
        for e in sub:
            L.append(f"| {e['task']} | " + " | ".join(
                fmt(e[f]["selected_gap"]) if e[f] else "-" for f in fams) + " |")  # fmt: skip
        L.append("")
    pf = (*fams, "gru")
    L += ["## Matched parameter count: each family's largest size with at most A (general, "
          "D = 4)'s count", "",
          "Entry: median gap (failures) [parameters]; * = no size within budget, smallest "
          "shown.", ""]  # fmt: skip
    L.append("| task | budget | " + " | ".join(FAMILY_LABEL[f] for f in pf) + " |")
    L.append("|---" * (len(pf) + 2) + "|")
    for e in mpar:
        cells = []
        for f in pf:
            r = e[f]
            if r is None:
                cells.append("-")
                continue
            s = cell_str(r, params=True) + f" D={r['state_dim']}"
            cells.append(s + (" *" if r["over_budget"] else ""))
        L.append(f"| {e['task']} | {e['budget']} | " + " | ".join(cells) + " |")
    L += ["", "### Validation-selected run at matched parameter count", ""]
    L.append("| task | " + " | ".join(FAMILY_LABEL[f] for f in pf) + " |")
    L.append("|---" * (len(pf) + 1) + "|")
    for e in mpar:
        L.append(f"| {e['task']} | " + " | ".join(
            fmt(e[f]["selected_gap"]) + (" *" if e[f]["over_budget"] else "") if e[f] else "-"
            for f in pf) + " |")  # fmt: skip
    L += ["", f"## Smallest size reaching the optimum (gap < {gap}): D [parameters]; "
          "median over seeds / validation-selected run", ""]  # fmt: skip
    L.append("| task | " + " | ".join(FAMILY_LABEL[f] for f in fams) + " |")
    L.append("|---" * (len(fams) + 1) + "|")
    for t in tasks:
        cells = []
        for f in fams:
            r = next((x for x in reach if x["task"] == t and x["family"] == f), None)
            if r is None:
                cells.append("-")
                continue

            def one(d, p):
                return f"{d} [{p}]" if d else "none"

            cells.append(f"{one(r['median_dim'], r['median_params'])} / "
                         f"{one(r['selected_dim'], r['selected_params'])}")  # fmt: skip
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## Dimensional test: 1-qubit quantum vs 2-state classical", "",
          "Gaps in nats/symbol. *test*: median over seeds / validation-selected run on the "
          "standard test set. *long*: the validation-selected run re-scored on "
          f"{cfg['dimensional']['long_test']['n']} sequences of "
          f"{cfg['dimensional']['long_test']['length']} (paired gap ± SE). *bound*: least KL "
          "rate of any 2-state unifilar model (`unifilar_merge_bound`).", ""]  # fmt: skip
    for e in dimtab:
        ex = e["exact"]
        L += [
            f"### {e['task']} (h = {e['entropy_rate_nats']:.4f} nats, "
            f"{e['causal_states']} causal states; 2-state unifilar bound "
            f"{e['unifilar_bound_nats']:.5f}"
            f"{'' if e['unifilar_bound_realisable'] else ', not reached by a merge'}; exact "
            f"1-qubit model long gap "
            f"{(fmt(ex['gap'], 6) + ' ± ' + fmt(ex['se'], 6)) if ex else 'none'})",
            "",
        ]
        L.append("| model | params | failed | test median | test selected | long (selected) |")
        L.append("|---|---|---|---|---|---|")
        for k, v in sorted(e["models"].items()):
            best = (e.get("best_classical_standard"), e.get("best_classical_large"))
            star = " ← best 2-state classical" if k in best else ""
            lg = (
                f"{fmt(v['long_gap'], 5)} ± {fmt(v['long_se'], 5)}"
                if v["long_gap"] is not None
                else "-"
            )
            L.append(f"| {k}{star} | {v['n_parameters']} | {v['n_runs'] - v['n_success']}/"
                     f"{v['n_runs']} | {fmt(v['median_gap'])} | {fmt(v['selected_gap'])} | "
                     f"{lg} |")  # fmt: skip
        L.append("")
    if retention:
        L += ["## Recall accuracy on the retention tasks (mean over seeds; source optimum in "
              "brackets)", ""]  # fmt: skip
        L.append("| task | cell | accuracy mean | accuracy min-max | value NLL mean |")
        L.append("|---|---|---|---|---|")
        for r in retention:
            L.append(f"| {r['task']} | {r['cell']} | {fmt(r['accuracy_mean'], 3)} "
                     f"({fmt(r['source_accuracy'], 3)}) | {fmt(r['accuracy_min'], 3)}-"
                     f"{fmt(r['accuracy_max'], 3)} | {fmt(r['value_nll_mean'])} |")  # fmt: skip
        L.append("")
    L += ["## Suspect cells", "", "| task | cell | vs | median gap | vs | failures | vs | "
          "selected gap | vs |", "|---" * 9 + "|"]  # fmt: skip
    for s in suspects:
        L.append(f"| {s['task']} | {s['cell']} | {s['vs']} | {fmt(s['median_gap'])} | "
                 f"{fmt(s['vs_median_gap'])} | {s['failures']} | {s['vs_failures']} | "
                 f"{fmt(s['selected_gap'])} | {fmt(s['vs_selected_gap'])} |")  # fmt: skip
    L += ["", "## Every evaluated cell", ""]
    for t in tasks:
        L += [f"### {t}", "", "| cell | method | chosen | D | params | flops/step | failed | "
              "median gap | mean ± SE | selected gap | steps | train s/run |",
              "|---" * 12 + "|"]  # fmt: skip
        for r in [r for r in rows if r["task"] == t]:
            L.append(f"| {r['cell']} | {r['method']} | {r['candidate']} | {r['state_dim']} | "
                     f"{r['n_parameters']} | {r['flops_per_step']} | "
                     f"{r['n_runs'] - r['n_success']}/{r['n_runs']} | {fmt(r['gap_median'])} | "
                     f"{fmt(r['gap_mean'])} ± {fmt(r['gap_se'])} | {fmt(r['selected_gap'])} | "
                     f"{r['steps_mean']:.0f} | {r['train_seconds_mean']:.1f} |")  # fmt: skip
        L.append("")
    L += ["## Tuning", "", f"Learning-rate grid {cfg['tuning']['lr_grid']} for every gradient "
          f"cell (A, D and D_diss included), tuning seeds {cfg['experiment']['tuning_seeds']}, "
          "ranked by median best-validation NLL; every size tuned on its own. Large-data cells "
          "reuse the standard-data choice. Full table in `tuning.csv`.", ""]  # fmt: skip
    (out / "summary.md").write_text("\n".join(L) + "\n")


def retention_rows(rows, post) -> list[dict]:
    out = []
    for r in rows:
        vals = [v for k, v in post["recall"].items() if k.rsplit("|", 1)[0] == r["group"]]
        if not vals:
            continue
        acc = [v["accuracy"] for v in vals]
        out.append({
            "task": r["task"], "cell": r["cell"], "family": r["family"], "n": len(vals),
            "accuracy_mean": statistics.mean(acc), "accuracy_min": min(acc),
            "accuracy_max": max(acc),
            "value_nll_mean": statistics.mean(v["value_nll"] for v in vals),
            "source_accuracy": vals[0]["source_accuracy"],
        })  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--from-run", type=Path, default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma-separated subset (smoke tests)")
    ap.add_argument("--max-dim", type=int, default=None, help="skip cells above this D")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summary-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.config.read_text())
    suite = tomllib.loads((REPO_ROOT / cfg["experiment"]["suite_config"]).read_text())
    tasks_cfg = resolve_tasks(cfg, suite, args.tasks.split(",") if args.tasks else None)
    tasks = list(tasks_cfg)
    workers = args.workers or cfg["experiment"]["workers"]
    out_dir = args.from_run or new_run_dir(cfg["experiment"]["name"])
    summary_dir = args.summary_dir or REPO_ROOT / cfg["experiment"]["summary_dir"]
    meta = run_metadata({"phase3f": cfg, "tasks": tasks_cfg}, None)
    if not args.no_train:
        save_json(out_dir / "config.json", meta)
    print(f"run dir: {out_dir}", flush=True)
    t0 = time.perf_counter()

    records = p3a.load_records(out_dir) if args.from_run else []
    cells = build_cells(cfg, tasks_cfg, args.max_dim)
    print(f"{len(cells)} cells on {len(tasks)} tasks", flush=True)
    records, _choice, tuning_table = run_waves(
        cfg, tasks_cfg, cells, records, out_dir, workers, train=not args.no_train
    )
    cell_keys = {(c.task, c.key) for c in cells}
    records = [r for r in records if (r["labels"]["task"], r["labels"]["cell"]) in cell_keys]
    post = post_process(cfg, tasks_cfg, cells, records, out_dir)

    success_gap = cfg["experiment"]["success_gap"]
    rows = p3a.summarise_eval(records, tuning_table, success_gap)
    fam_of = {(c.task, c.key): c.family for c in cells}
    for r in rows:
        r["family"] = fam_of[(r["task"], r["cell"])]
        r["entropy_rate_nats"] = make_task(tasks_cfg[r["task"]]["spec"]).entropy_rate()
    suspects = flag_suspects(rows, cfg["analysis"]["suspect_margin"])
    mdim = matched_dim_tables(rows, tasks)
    mpar = matched_param_table(rows, tasks_cfg, tasks)
    reach = reach_table(rows, tasks, success_gap)
    dimtab = dimensional_table(cfg, rows, tasks_cfg, post)
    retention = retention_rows(rows, post)
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
        "run_dir": str(out_dir.relative_to(REPO_ROOT)) if out_dir.is_relative_to(REPO_ROOT)
        else str(out_dir),
        "n_cells": len(cells),
        "n_tuning_runs": sum(r["group"].startswith("tune/") for r in records),
        "n_eval_runs": sum(r["group"].startswith("eval/") for r in records),
    }  # fmt: skip
    write_summary(summary_dir, cfg, tasks_cfg, tasks, rows, tuning_table, suspects, mdim, mpar,
                  reach, dimtab, retention, meta, extra)  # fmt: skip
    plots(rows, tasks, summary_dir)
    print(f"summary: {summary_dir}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    sys.exit(main())
