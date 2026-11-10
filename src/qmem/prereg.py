"""Phase 8 Stage 4 (issue #36): the combined pre-registration's rules as code, numpy only.

Nothing here imports Qiskit or ``qiskit_ibm_runtime``, and nothing contacts IBM. The module holds
what ``docs/phase8-preregistration.md`` fixes before any data:

* **Live re-selection** (``select_ancillas``, ``select_reference``, ``select_pairs``,
  ``reselection``). Each qubit choice that Stages 1-3 made on FakeKingston's stale calibration
  is a deterministic function of a calibration snapshot (``phase7_spectator``'s
  ``device.json`` layout, or a live export of the same shape). ``reselection`` applies the
  predeclared consequences when a live snapshot picks different qubits: re-export the QASM,
  re-hash, re-check power against a stated floor, abort if the floor fails. Stage 1's pairs
  keep #31's own policy (a committed pair failing live calibration is dropped, not replaced).
* **Budget and approval** (``Approval``, ``freeze_approval``, ``TierBudget``, ``Ledger``,
  ``budget_table``). ``qmem.followup``'s cost model, per-job guard and max(charged, upper bound)
  accounting, per tier and in total, against a remaining allowance that is a *parameter*
  (unknown until the owner reads the dashboard) less a reserve. An approval names one tier and
  is frozen into the run directory the first time it is used; a resume with a different
  approval refuses.
* **Job execution with a pluggable backend** (``LocalBackend``, ``run_stage``). Resume by job id
  (an id on disk is never resubmitted), the id saved before waiting, raw shots preserved (never
  overwritten), usage read after each job and counted. Only the local simulator exists here; a
  live backend is a separate, reviewable addition with the same three methods.
* **The tiers' decision rules on data** (``tier1_*``, ``tier2_*``; Tier 3 uses
  ``qmem.replication.analyse_pair`` unchanged).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import followup as F
from . import reference as R
from . import replication as REP
from .reanalysis import Z95, norm_ppf

# -- calibration snapshots ------------------------------------------------------------------------

#: The calibration fields a selection may read (anything else in a snapshot is ignored).
SNAPSHOT_QUBIT_FIELDS = ("t1", "t2", "readout", "sx", "measure_time", "reset_time")


def snapshot_hash(dev: dict[str, Any]) -> str:
    """SHA-256 of the fields a selection reads (qubits, couplers, edges), canonical JSON."""
    body = {
        "qubits": {
            str(k): {f: q.get(f) for f in SNAPSHOT_QUBIT_FIELDS if f in q}
            for k, q in sorted(dev["qubits"].items(), key=lambda kv: int(kv[0]))
        },
        "cz": {k: dev["cz"][k] for k in sorted(dev["cz"])},
        "edges": sorted([sorted(map(int, e)) for e in dev["edges"]]),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def perturb(
    dev: dict[str, Any],
    qubits: dict[int, dict[str, float]] | None = None,
    cz: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """A copy of ``dev`` with some calibration values replaced (a stand-in for a live snapshot)."""
    out = copy.deepcopy(dev)
    for q, vals in (qubits or {}).items():
        out["qubits"][str(q)].update(vals)
    for k, vals in (cz or {}).items():
        out["cz"][k].update(vals)
    return out


def adjacency(dev: dict[str, Any]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {int(q): set() for q in dev["qubits"]}
    for a, b in dev["edges"]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def hops(adj: dict[int, set[int]], src: int) -> dict[int, int]:
    dist, dq = {src: 0}, deque([src])
    while dq:
        u = dq.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


# -- selection rules ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AncillaRule:
    """Stage 2's feed-forward ancillas (#32, ``phase8_coherence_hw.select_ancillas``): graph
    distance >= ``min_distance`` from the output and the memory, not adjacent to each other,
    lowest readout (measure) error, then longest T1, within Phase 6's register (qubits
    ``0 .. n_register - 1``)."""

    memory: int = 75
    output: int = 59
    min_distance: int = 3
    n_register: int = 76
    count: int = 2


def select_ancillas(dev: dict[str, Any], rule: AncillaRule) -> dict[str, Any]:
    """The rule on a snapshot. Deterministic: ties in (readout, -T1) break on the qubit index."""
    adj = adjacency(dev)
    d_out, d_mem = hops(adj, rule.output), hops(adj, rule.memory)
    rows = []
    for q in range(rule.n_register):
        c = dev["qubits"].get(str(q))
        if c is None or c.get("readout") is None:
            continue
        if min(d_out.get(q, 99), d_mem.get(q, 99)) < rule.min_distance:
            continue
        rows.append({"qubit": q, "measure_error": c["readout"], "t1": c.get("t1") or 0.0})
    rows.sort(key=lambda r: (r["measure_error"], -r["t1"], r["qubit"]))
    pick: list[int] = []
    for r in rows:
        if len(pick) == rule.count:
            break
        if all(r["qubit"] not in adj[p] for p in pick):
            pick.append(r["qubit"])
    near = min((min(d_out.get(q, 99), d_mem.get(q, 99)) for q in pick), default=None)
    return {
        "chosen": pick,
        "candidates": len(rows),
        "distance_to_core": near,
        "top10": rows[:10],
        "snapshot_sha256": snapshot_hash(dev),
    }


def select_reference(
    dev: dict[str, Any],
    memory: int,
    output: int,
    rule: R.RefRule,
    score: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Stage 3's reference rule (#33): ``qmem.reference.reference_candidates`` admits or
    excludes each qubit within ``rule.max_dist_memory`` of the memory; the pick is the admitted
    qubit with the highest ``score`` (the predicted n = 1 Bell fidelity), ties to fewer SWAPs,
    then the lower index."""
    cands = R.reference_candidates(dev, memory, output, rule)
    for c in cands:
        c["score"] = float(score(c))
    admitted = [c for c in cands if c["admitted"]]
    pick = (
        max(admitted, key=lambda c: (round(c["score"], 6), -c["swaps"], -c["qubit"]))
        if admitted
        else None
    )
    return {
        "chosen": [] if pick is None else [pick["qubit"]],
        "pick": pick,
        "candidates": cands,
        "snapshot_sha256": snapshot_hash(dev),
    }


def select_pairs(
    dev: dict[str, Any],
    rules: REP.SelectionRules,
    power: Callable[[dict[str, Any]], dict[str, float]],
) -> dict[str, Any]:
    """Stage 1's frozen algorithm (#31, ``qmem.replication.select``) on a snapshot."""
    sel = REP.select(REP.Device.from_snapshot(dev), rules, power)
    chosen = [(r["memory"], r["output"], r["spectator"]) for r in sel.chosen]
    return {
        "chosen": [f"m{m}_o{o}" for m, o, _ in chosen],
        "footprints": [list(c) for c in chosen],
        "power": {f"m{r['memory']}_o{r['output']}": r.get("power_joint") for r in sel.chosen},
        "screened": sum(1 for r in sel.rows if not r["screen_reasons"]),
        "qualified": sel.n_qualified,
        "candidates": len(sel.rows),
        "statement": sel.statement,
        "rows": sel.rows,
        "snapshot_sha256": snapshot_hash(dev),
    }


@dataclass(frozen=True)
class ReselectionPolicy:
    """What happens when the live snapshot's selection differs from the committed one.

    ``mode = "replace"``: the live choice is used, after the QASM of ``affected`` circuits is
    re-exported on the live target and re-hashed, and only if ``power`` at the live choice is
    at least ``floor``; otherwise the tier aborts. ``mode = "drop"`` (#31 section 12): each
    committed choice that fails the live screen or ``floor`` is dropped, never replaced; the
    live re-selection is recorded only; the tier aborts when every committed choice is dropped.
    """

    tier: str
    what: str
    mode: str
    floor: float
    floor_quantity: str
    affected: tuple[str, ...] = ()


def reselection(
    policy: ReselectionPolicy,
    committed: Sequence[Any],
    live: Sequence[Any],
    power: dict[Any, float],
    live_screen_ok: dict[Any, bool] | None = None,
) -> dict[str, Any]:
    """The predeclared consequence of a live selection (see ``ReselectionPolicy``).

    ``power`` maps each choice that needs a re-check (the live choice under ``replace``; each
    committed choice under ``drop``) to the re-computed quantity; ``live_screen_ok`` (``drop``)
    whether each committed choice passes the live calibration screen."""
    committed, live = list(committed), list(live)
    same = sorted(map(str, committed)) == sorted(map(str, live))
    out: dict[str, Any] = {
        "tier": policy.tier,
        "what": policy.what,
        "mode": policy.mode,
        "committed": committed,
        "live": live,
        "same": same,
        "actions": [],
        "abort": False,
        "use": committed,
    }
    if policy.mode == "replace":
        if same:
            out["actions"].append("none: the committed QASM and hashes stand")
            return out
        out["actions"] += [
            f"re-export the QASM of {list(policy.affected)} on the live target (level 0, "
            "fixed layout, the stage's export script)",
            "re-hash; the new hashes go into a dated pre-data amendment before any submission",
            f"re-check {policy.floor_quantity} at the live choice against the floor {policy.floor}",
        ]
        p = power.get(tuple(live)) if tuple(live) in power else power.get(str(live))
        out["power_live"] = p
        if p is None or not (p >= policy.floor):
            out["abort"] = True
            out["actions"].append("abort: the floor fails (no substitute, no submission)")
        else:
            out["use"] = live
        return out
    if policy.mode == "drop":
        kept = []
        for c in committed:
            ok = (live_screen_ok or {}).get(c, True)
            p = power.get(c)
            if ok and p is not None and p >= policy.floor:
                kept.append(c)
            else:
                why = "fails the live screen" if not ok else f"{policy.floor_quantity} {p}"
                out["actions"].append(f"drop {c}: {why} (never replaced, #31 section 12)")
        if not same:
            out["actions"].append(
                f"record only: the live re-selection would choose {live} (#31 does not replace)"
            )
        out["use"] = kept
        if not kept:
            out["abort"] = True
            out["actions"].append("abort: every committed choice dropped; reselect by amendment")
        elif same and len(kept) == len(committed):
            out["actions"].insert(0, "none: the committed choices stand")
        return out
    raise ValueError(f"unknown mode {policy.mode!r}")


# -- approval, budget, ledger ---------------------------------------------------------------------


@dataclass(frozen=True)
class Approval:
    """The owner's approval of exactly one tier (an issue-comment URL), or none."""

    tier: str
    url: str | None = None

    def check(self, tier: str, submit_flag: bool) -> None:
        """``qmem.followup.submission_check``, plus: the approval must name this tier."""
        if tier != self.tier:
            raise F.Refused(f"the approval is for {self.tier!r}, not {tier!r}")
        F.submission_check(self.url, submit_flag)


def freeze_approval(run_dir: Path, approval: Approval) -> Approval:
    """Write ``approval.json`` the first time; refuse a resume under a different approval."""
    path = run_dir / f"approval_{approval.tier}.json"
    if path.exists():
        old = Approval(**json.loads(path.read_text()))
        if old != approval:
            raise F.Refused(
                f"approval for {approval.tier} changed within a run: "
                f"{old.url!r} -> {approval.url!r}"
            )
        return old
    path.write_text(json.dumps(asdict(approval), indent=1))
    return approval


@dataclass
class TierBudget:
    """One tier: its jobs' costs (``qmem.followup.CostModel.job`` dicts), the per-job estimate
    limit and the hard cap."""

    tier: str
    jobs: list[dict[str, float]]
    hard_cap_s: float
    max_job_estimate_s: float
    labels: list[str] = field(default_factory=list)

    @property
    def estimate_s(self) -> float:
        return float(sum(j["estimate_s"] for j in self.jobs))

    @property
    def upper_s(self) -> float:
        return float(sum(j["upper_s"] for j in self.jobs))

    def problems(self) -> list[str]:
        out = []
        for i, j in enumerate(self.jobs):
            lab = self.labels[i] if i < len(self.labels) else f"job{i}"
            if j["estimate_s"] > self.max_job_estimate_s:
                out.append(
                    f"{self.tier} {lab}: estimate {j['estimate_s']:.2f} s > per-job limit "
                    f"{self.max_job_estimate_s} s"
                )
        if self.upper_s > self.hard_cap_s:
            out.append(
                f"{self.tier}: upper bound {self.upper_s:.2f} s > hard cap {self.hard_cap_s}"
            )
        return out


def budget_table(
    tiers: Sequence[TierBudget],
    allowance_s: float | None,
    reserve_s: float,
    prior_counted_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Tiers in order with cumulative upper bounds and caps against ``allowance_s - reserve_s``
    (``None``: unknown, so nothing fits)."""
    rows, cum_u, cum_c = [], prior_counted_s, prior_counted_s
    usable = None if allowance_s is None else allowance_s - reserve_s
    for t in tiers:
        cum_u += t.upper_s
        cum_c += t.hard_cap_s
        rows.append(
            {
                "tier": t.tier,
                "jobs": len(t.jobs),
                "estimate_s": t.estimate_s,
                "upper_s": t.upper_s,
                "hard_cap_s": t.hard_cap_s,
                "cumulative_upper_s": cum_u,
                "cumulative_cap_s": cum_c,
                "allowance_s": allowance_s,
                "reserve_s": reserve_s,
                "fits_upper": usable is not None and cum_u <= usable,
                "fits_cap": usable is not None and cum_c <= usable,
                "problems": "; ".join(t.problems()),
            }
        )
    return rows


class Ledger:
    """Charges per tier (``qmem.followup.Charge``), max(charged, upper bound) accounting."""

    def __init__(self) -> None:
        self.charges: dict[str, list[F.Charge]] = {}

    def add(self, tier: str, charge: F.Charge) -> None:
        self.charges.setdefault(tier, []).append(charge)

    def counted(self, tier: str | None = None) -> float:
        if tier is not None:
            return F.usage_guard(self.charges.get(tier, []))
        return float(sum(F.usage_guard(c) for c in self.charges.values()))

    def guard(
        self,
        tier: TierBudget,
        cost: dict[str, float],
        remaining_s: float | None,
        plan: str | None,
        reserve_s: float,
        allowed_plans: Sequence[str] = ("open",),
    ) -> None:
        """``qmem.followup.budget_check`` with this tier's spend and cap, and the reserve kept
        out of the remaining allowance. Raises ``Refused``/``UsageExceeded``."""
        spent = self.counted(tier.tier)
        rem = None if remaining_s is None else remaining_s - reserve_s
        F.budget_check(
            cost["estimate_s"],
            cost["upper_s"],
            tier.max_job_estimate_s,
            rem,
            plan,
            spent,
            tier.hard_cap_s,
            allowed_plans,
        )


# -- job execution with a pluggable backend -------------------------------------------------------


class Backend(Protocol):
    """What a job runner needs. ``LocalBackend`` simulates; a live backend is a separate file."""

    name: str

    def submit(self, stage: str, pubs: Sequence[F.Pub]) -> str: ...

    def result(self, job_id: str) -> dict[str, np.ndarray]: ...

    def usage(self, job_id: str) -> dict[str, Any]: ...


class LocalBackend:
    """Simulated jobs: ``simulate(stage, pubs)`` returns the raw arrays per pub key; ``charge``
    the seconds it reports (``None``: an unreadable charge). Results are kept in ``store`` (the
    simulator's side, like a provider's job store), so a resumed run in a new process re-fetches
    them by id; the simulator never runs a job twice."""

    name = "local-simulator"

    def __init__(
        self,
        simulate: Callable[[str, Sequence[F.Pub]], dict[str, np.ndarray]],
        charge: Callable[[str, Sequence[F.Pub]], float | None] | None,
        store: Path,
    ):
        self._sim, self._charge = simulate, charge
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.submissions = 0

    def submit(self, stage: str, pubs: Sequence[F.Pub]) -> str:
        self.submissions += 1
        jid = f"local-{stage}-{len(list(self.store.glob('*.npz'))) + 1}"
        arrays = self._sim(stage, pubs)
        np.savez_compressed(self.store / f"{jid}.npz", **arrays)
        v = None if self._charge is None else self._charge(stage, pubs)
        (self.store / f"{jid}.json").write_text(json.dumps({"charged_s": v}))
        return jid

    def result(self, job_id: str) -> dict[str, np.ndarray]:
        with np.load(self.store / f"{job_id}.npz") as z:
            return {k: z[k] for k in z.files}

    def usage(self, job_id: str) -> dict[str, Any]:
        v = json.loads((self.store / f"{job_id}.json").read_text())["charged_s"]
        return {"local": v if v is not None else {"error": "unreadable"}}


def keep_first(path: Path) -> Path:
    """``path``, or a timestamped sibling if it exists (raw data are never overwritten)."""
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}_{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}")


def run_stage(
    tier: str,
    stage: str,
    pubs: Sequence[F.Pub],
    backend: Backend,
    out: Path,
    cost: dict[str, float],
    ledger: Ledger,
    guard: Callable[[dict[str, Any]], None],
) -> dict[str, np.ndarray]:
    """One job, idempotent by id (``phase6.submit_or_fetch``'s semantics).

    An id in ``out/<stage>/job.json`` is re-fetched, never resubmitted. Otherwise ``guard(rec)``
    must pass, the job is submitted, and its id saved before the result is awaited. The raw
    arrays go to ``shots.npz`` (``keep_first``); the usage reading becomes a ``Charge`` in the
    ledger, and a charge above the upper bound stops the run (``UsageExceeded``)."""
    d = out / stage
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "job.json"
    rec: dict[str, Any] = json.loads(jp.read_text()) if jp.exists() else {}
    resumed = "job_id" in rec
    if not resumed:
        rec.update({"tier": tier, "stage": stage, "backend": backend.name, **cost})
        rec["pubs"] = [p.key for p in pubs]
        guard(rec)
        rec["job_id"] = backend.submit(stage, pubs)
        rec["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        jp.write_text(json.dumps(rec, indent=1, default=str))
    arrays = backend.result(rec["job_id"])
    raw = d / "shots.npz"
    if not raw.exists():
        np.savez_compressed(raw, **arrays)
    elif not resumed:
        np.savez_compressed(keep_first(raw), **arrays)
    readings = backend.usage(rec["job_id"])
    ch = F.job_charge(f"{tier}:{stage}", cost["upper_s"], readings)
    rec["charge"] = asdict(ch)
    rec["resumed"] = resumed
    jp.write_text(json.dumps(rec, indent=1, default=str))
    ledger.add(tier, ch)
    if ch.exceeded:
        raise F.UsageExceeded(f"{stage} charged {ch.charged_s} s > upper bound {ch.upper_s:.2f} s")
    return arrays


def order_warnings(bits: np.ndarray, blocks: int = 10, alpha: float = 1e-3) -> dict[str, Any]:
    """Shot-order warning for short circuits (Tier 1): chi^2 homogeneity of P(1) over
    ``blocks`` consecutive blocks of a pub's shots. A warning, never an invalidation."""
    bits = np.asarray(bits).astype(float).ravel()
    parts = np.array_split(bits, blocks)
    n = np.array([len(p) for p in parts], float)
    k = np.array([p.sum() for p in parts])
    p = k.sum() / n.sum()
    if p in (0.0, 1.0):
        return {"chi2": 0.0, "p": 1.0, "warn": False}
    chi2 = float(((k - n * p) ** 2 / (n * p * (1 - p))).sum())
    pv = _chi2_sf(chi2, blocks - 1)
    return {"chi2": chi2, "p": pv, "warn": bool(pv < alpha)}


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi^2_k (regularised upper incomplete gamma, series/continued)."""
    a, z = k / 2, x / 2
    if z <= 0:
        return 1.0
    if z < a + 1:
        s, t, n = 1 / a, 1 / a, 0
        while abs(t) > 1e-15 * abs(s) and n < 1000:
            n += 1
            t *= z / (a + n)
            s += t
        return max(0.0, 1 - s * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


# -- Tier 1 (Stage 3, #33): P&M at n = 1, the record-00 branch at n = 2 ---------------------------


@dataclass(frozen=True)
class Tier1Rule:
    """docs/phase8-preregistration.md section 2 (from #33 sections 3-5, amended pre-data)."""

    alpha: float = 0.05  # family-wise over the two co-primaries (Holm)
    secondary_alpha: float = 0.05  # Holm over the remaining secondary family
    spam_gate: float = 0.90  # P&M score at n = 0 must reach this
    min_branch_shots: int = 500
    leak_per_cz: float = 1e-3
    cz_per_step: int = 3


def one_sided_p(est: float, se: float, thr: float) -> float:
    if not math.isfinite(se) or se <= 0:
        return 1.0 if est <= thr else 0.0
    return 0.5 * math.erfc((est - thr) / se / math.sqrt(2))


def tier1_counts(
    steps: np.ndarray,
    n: int,
    v: np.ndarray,
    spam: R.Spam,
    shots: int,
    rng: np.random.Generator,
    first: np.ndarray | None = None,
    record: bool = False,
) -> dict[str, np.ndarray]:
    """Simulated counts of the 12 P&M circuits (index ``2 k + b``) after ``n`` steps.

    ``record``: also the q59 record per shot, as counts ``succ[r, c]`` and ``tot[r, c]`` over
    the 2^n records r (big-endian, step 1 first)."""
    recs = [()] if not record else [tuple(int(c) for c in f"{r:0{n}b}") for r in range(2**n)]
    cells = []
    for r in recs:
        sup = R.n_step_map(steps, n, list(r) if record else None, first)
        p = R.pm_probabilities(sup, v, spam)
        cells.append((p["p_succ"].ravel(), p["p_branch"].ravel()))
    ps = np.array([c[0] for c in cells])  # (records, 12)
    pb = np.array([c[1] for c in cells])
    succ = np.zeros_like(ps, dtype=np.int64)
    tot = np.zeros_like(ps, dtype=np.int64)
    for c in range(ps.shape[1]):
        probs = np.concatenate([ps[:, c], pb[:, c] - ps[:, c]])
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        draw = rng.multinomial(shots, probs)
        succ[:, c] = draw[: len(recs)]
        tot[:, c] = draw[: len(recs)] + draw[len(recs) :]
    return {"succ": succ, "tot": tot, "records": np.array(["".join(map(str, r)) for r in recs])}


def tier1_analyse(
    counts: dict[str, dict[str, np.ndarray]],
    rule: Tier1Rule,
    mdi: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tier 1's decision on counts.

    ``counts[arm]`` for ``pm_n0``, ``pm_n1``, ``pm_n2``, ``pm_n3``, ``pm_neg`` (``pm_n2`` with
    records). Co-primaries (Holm at ``alpha``): P&M at n = 1 (unconditional) and the record
    ``00`` branch at n = 2. Gates: P&M at n = 0 >= ``spam_gate``; the EB negative control's
    one-sided lower bound must not exceed its threshold. Secondary (Holm): the unconditional
    n = 2 test and records 0, 1 at n = 1 (from ``pm_n1``'s records if present) and 01, 10, 11
    at n = 2. n = 3 is a model check (expected below 2/3)."""

    def leak(n: int) -> float:
        return R.leakage_bound(n, rule.cz_per_step, rule.leak_per_cz)

    def uncond(arm: str) -> tuple[float, float]:
        c = counts[arm]
        return R.pm_estimate(c["succ"].sum(0), c["tot"].sum(0))

    def branch(arm: str, rec: str) -> tuple[float, float, int]:
        c = counts[arm]
        i = list(c["records"]).index(rec)
        est, se = R.ratio_estimate(c["succ"][i], c["tot"][i])
        return est, se, int(c["tot"][i].sum())

    out: dict[str, Any] = {"tests": {}}
    n0, _ = uncond("pm_n0")
    out["spam_gate"] = {"score": n0, "threshold": rule.spam_gate, "pass": n0 >= rule.spam_gate}
    neg, neg_se = uncond("pm_neg")
    z1 = norm_ppf(1 - rule.alpha)
    thr1 = R.pm_threshold(leak(1))
    neg_pass = neg - z1 * neg_se > thr1
    out["negative_control"] = {
        "score": neg,
        "se": neg_se,
        "threshold": thr1,
        "lower_bound": neg - z1 * neg_se,
        "falsely_passes": bool(neg_pass),
    }

    def test(name: str, est: float, se: float, thr: float, shots: int | None = None) -> None:
        row = {"estimate": est, "se": se, "threshold": thr, "p": one_sided_p(est, se, thr)}
        if shots is not None:
            row["branch_shots"] = shots
            row["testable"] = shots >= rule.min_branch_shots
            if not row["testable"]:
                row["p"] = 1.0
        out["tests"][name] = row

    e, s = uncond("pm_n1")
    test("pm_n1", e, s, thr1)
    e, s, nb = branch("pm_n2", "00")
    test("branch00_n2", e, s, R.pm_threshold(leak(2)), nb)
    # secondary family
    e, s = uncond("pm_n2")
    test("pm_n2", e, s, R.pm_threshold(leak(2)))
    sec = ["pm_n2"]
    if len(counts["pm_n1"]["records"]) > 1:
        for rec in ("0", "1"):
            e, s, nb = branch("pm_n1", rec)
            test(f"branch{rec}_n1", e, s, thr1, nb)
            sec.append(f"branch{rec}_n1")
    for rec in ("01", "10", "11"):
        e, s, nb = branch("pm_n2", rec)
        test(f"branch{rec}_n2", e, s, R.pm_threshold(leak(2)), nb)
        sec.append(f"branch{rec}_n2")
    e3, s3 = uncond("pm_n3")
    out["model_check_n3"] = {"score": e3, "se": s3, "below_two_thirds": e3 < 2 / 3}
    prim = F.holm({k: out["tests"][k]["p"] for k in ("pm_n1", "branch00_n2")}, rule.alpha)
    secd = F.holm({k: out["tests"][k]["p"] for k in sec}, rule.secondary_alpha)
    for k, v in {**prim, **secd}.items():
        out["tests"][k]["reject_eb"] = bool(v)
    valid = out["spam_gate"]["pass"] and not neg_pass
    out["valid"] = bool(valid)
    if not out["spam_gate"]["pass"]:
        out["call"] = "stopped: SPAM gate failed (n = 0 below 0.90); n >= 1 not interpreted"
    elif neg_pass:
        out["call"] = "invalid: the EB negative control passes (trust assumptions failed)"
    elif prim["pm_n1"]:
        out["call"] = "claim 4 at n = 1 under trusted SPAM: the step channel is not EB"
    elif prim["branch00_n2"]:
        out["call"] = (
            "ambiguous: only the record-00 element is certified not EB (an instrument "
            "property, not Claim 4)"
        )
    else:
        out["call"] = "not supported: no co-primary rejects EB"
    out["branch_call"] = (
        "instrument element M_00 (n = 2) not EB" if prim["branch00_n2"] and valid else "not shown"
    )
    if mdi is not None:
        out["mdi"] = mdi
    return out


def mdi_counts(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated Phi+ counts of the 16 MDI input pairs (``probs[x, y]``)."""
    return rng.binomial(shots, probs)


def mdi_analyse(
    hits: np.ndarray, shots: int, omega: np.ndarray, alpha: float = 0.05
) -> dict[str, Any]:
    """Payoff sum omega_xy p(Phi+ | x, y) > 0 at one-sided ``alpha`` (binomial SE)."""
    p = hits / shots
    est = float((omega * p).sum())
    se = float(math.sqrt(((omega**2) * p * (1 - p) / shots).sum()))
    return {
        "payoff": est,
        "se": se,
        "lower_bound": est - norm_ppf(1 - alpha) * se,
        "reject_eb": bool(est - norm_ppf(1 - alpha) * se > 0),
    }


# -- Tier 2 (Stage 2, #32/#34): legs of the Claim 3 bar on data -----------------------------------


@dataclass(frozen=True)
class Tier2Rule:
    """docs/phase8-coherence.md section 4 with its section 4.1 amendment."""

    alpha: float = 0.05
    pilot_sham_p1: float = 0.02  # E0, E00, RS within +-0.02 of the baseline in P(1)
    pilot_scan_p1: float = 0.10  # P2, P4 within +-0.10 of the fixed prediction
    pilot_pair: float = 0.03  # INV2 vs E00 and R2 vs A2 within +-0.03 (P(1), lag 1)
    harmonic_k: float = 3.0
    harmonic_margin: float = 0.05
    tost_p1: float = 0.01
    tost_lag1: float = 0.02
    deph_p1: float = 0.01
    deph_lag1: float = 0.01
    deph_w0_frac: float = 0.10
    vanish_fraction: float = 0.25
    vanish_z: float = 1.959963984540054
    preserve_margin: float = 0.20


def tost_diff(a: float, sa: float, b: float, sb: float, margin: float, alpha: float) -> bool:
    """TOST of a - b inside +-margin: the 1 - 2 alpha CI inside the margin."""
    d, s = a - b, math.hypot(sa, sb)
    z = norm_ppf(1 - alpha)
    return bool(d - z * s > -margin and d + z * s < margin)


def first_harmonic(phi: np.ndarray, y: np.ndarray, se: np.ndarray | None = None) -> dict[str, Any]:
    """Weighted least squares of y = c + a cos(phi) + b sin(phi): amplitude, phase and their
    delta-method SEs (``se is None``: the exact curve, SEs 0)."""
    phi, y = np.asarray(phi, float), np.asarray(y, float)
    x = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
    w = np.ones_like(y) if se is None else 1 / np.asarray(se, float) ** 2
    xtw = x.T * w
    cov = np.linalg.inv(xtw @ x)
    beta = cov @ xtw @ y
    a, b = beta[1], beta[2]
    amp = math.hypot(a, b)
    ph = math.atan2(b, a)
    out = {"offset": float(beta[0]), "amplitude": amp, "phase": ph}
    if se is None:
        out["amplitude_se"] = out["phase_se"] = 0.0
        return out
    ja = np.array([0, a / amp, b / amp])
    jp = np.array([0, -b / amp**2, a / amp**2])
    out["amplitude_se"] = float(math.sqrt(ja @ cov @ ja))
    out["phase_se"] = float(math.sqrt(jp @ cov @ jp))
    return out


def tier2_pilot_gate(
    st: dict[str, dict[str, float]], pred: dict[str, dict[str, float]], rule: Tier2Rule
) -> dict[str, Any]:
    """The pilot gate of docs/phase8-coherence.md section 4 on point estimates (``st[arm]``
    with ``p1``, ``lag1``; ``pred`` the fixed model's)."""
    checks = {}
    for a in ("E0", "E00", "RS"):
        checks[f"{a}_p1_vs_baseline"] = (
            abs(st[a]["p1"] - st["baseline"]["p1"]) <= rule.pilot_sham_p1
        )
    for a in ("P2", "P4"):
        checks[f"{a}_p1_vs_fixed"] = abs(st[a]["p1"] - pred[a]["p1"]) <= rule.pilot_scan_p1
    for a, b in (("INV2", "E00"), ("R2", "A2")):
        for k in ("p1", "lag1"):
            checks[f"{a}_{k}_vs_{b}"] = abs(st[a][k] - st[b][k]) <= rule.pilot_pair
    return {"checks": {k: bool(v) for k, v in checks.items()}, "go": all(checks.values())}


def tier2_legs(
    est: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, float]],
    phases: dict[str, float],
    rule: Tier2Rule,
) -> dict[str, Any]:
    """Every non-likelihood leg on the main tier's arms.

    ``est[arm]``: ``value``/``se`` over ``qmem.replication.STATS`` (drift-conditional SEs),
    ``lojo_value``/``lojo_se``, and ``p1``/``p1_se``. ``pred``: the fixed model's ``p1``,
    ``lag1`` per arm. ``phases``: scan arm -> phi."""
    iw, il = REP.STATS.index("w0"), REP.STATS.index("lag1")

    def v(a, k):
        if k == "p1":
            return est[a]["p1"], est[a]["p1_se"]
        i = iw if k == "w0" else il
        return float(est[a]["value"][i]), float(est[a]["se"][i])

    out: dict[str, Any] = {}
    # (ii) the phase response: first harmonic of P(1) and lag 1 against the fixed prediction
    names = list(phases)
    phi = np.array([phases[a] for a in names])
    harm = {}
    for k in ("p1", "lag1"):
        ys = np.array([v(a, k)[0] for a in names])
        ss = np.array([v(a, k)[1] for a in names])
        fd = first_harmonic(phi, ys, ss)
        fp = first_harmonic(phi, np.array([pred[a][k] for a in names]))
        dph = (fd["phase"] - fp["phase"] + math.pi) % (2 * math.pi) - math.pi
        ok_a = abs(fd["amplitude"] - fp["amplitude"]) <= (
            rule.harmonic_k * fd["amplitude_se"] + rule.harmonic_margin
        )
        ok_p = abs(dph) <= rule.harmonic_k * fd["phase_se"] + rule.harmonic_margin
        harm[k] = {"data": fd, "fixed": fp, "amplitude_ok": bool(ok_a), "phase_ok": bool(ok_p)}
    out["harmonic"] = harm
    tost = {}
    tost["E0_baseline_p1"] = tost_diff(
        *v("E0", "p1"), *v("baseline", "p1"), rule.tost_p1, rule.alpha
    )
    tost["E0_baseline_lag1"] = tost_diff(
        *v("E0", "lag1"), *v("baseline", "lag1"), rule.tost_lag1, rule.alpha
    )
    for a, b in (("COMP22", "COMP40"), ("INV2", "E00")):
        tost[f"{a}_{b}_p1"] = tost_diff(*v(a, "p1"), *v(b, "p1"), rule.tost_p1, rule.alpha)
        tost[f"{a}_{b}_lag1"] = tost_diff(*v(a, "lag1"), *v(b, "lag1"), rule.tost_lag1, rule.alpha)
    out["phase_tost"] = tost
    out["leg_ii_response"] = bool(
        all(h["amplitude_ok"] and h["phase_ok"] for h in harm.values()) and all(tost.values())
    )
    # (i) population invariance
    pop = {
        a: tost_diff(*v(a, "p1"), *v("baseline", "p1"), rule.tost_p1, rule.alpha)
        for a in ("E0", "E00", "RS")
    }
    out["population_tost"] = pop
    out["leg_i_population"] = bool(all(pop.values()))
    # (iii) equivalent dephasings
    wmar = rule.deph_w0_frac * abs(v("baseline", "w0")[0])
    deph = {}
    for a, b in (("A2", "R2"), ("A2", "R4"), ("R2", "R4")):
        deph[f"{a}_{b}_p1"] = tost_diff(*v(a, "p1"), *v(b, "p1"), rule.deph_p1, rule.alpha)
        deph[f"{a}_{b}_lag1"] = tost_diff(*v(a, "lag1"), *v(b, "lag1"), rule.deph_lag1, rule.alpha)
        deph[f"{a}_{b}_w0"] = tost_diff(*v(a, "w0"), *v(b, "w0"), wmar, rule.alpha)
    out["dephasing_tost"] = deph
    out["leg_iii_dephasings"] = bool(all(deph.values()))
    # section 4.1: the Claim 3 contrast (R2, R4 vanish against RS; RS non-inferior to E00)
    ws = abs(v("RS", "w0")[0])
    van = {}
    for a in ("R2", "R4"):
        wa, sa = v(a, "w0")
        ok = abs(wa) + rule.vanish_z * sa < rule.vanish_fraction * ws
        lo = est[a]["lojo_value"][:, iw]
        los = est[a]["lojo_se"][:, iw]
        lrs = np.abs(est["RS"]["lojo_value"][:, iw])
        lojo = bool(np.all(np.abs(lo) + rule.vanish_z * los < rule.vanish_fraction * lrs))
        van[a] = {"all_jobs": bool(ok), "lojo": lojo, "vanishes": bool(ok and lojo)}
    wr, sr = v("RS", "w0")
    we, se_ = v("E00", "w0")
    r = wr / we
    rse = abs(r) * math.sqrt((sr / wr) ** 2 + (se_ / we) ** 2)
    pres = {"ratio": r, "se": rse, "lower": r - Z95 * rse}
    pres["preserved"] = bool(pres["lower"] > 1 - rule.preserve_margin)
    out["claim3_contrast"] = {"vanish": van, "preserve": pres}
    out["leg_contrast"] = bool(all(x["vanishes"] for x in van.values()) and pres["preserved"])
    return out


def tier2_primary(deficits: dict[int, dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    """The primary contrast's decision: ``deficits[D]`` holds ``mean`` and ``se_block`` of the
    held-out NLL deficit per shot (classical d <= D minus quantum refit); reject "d <= D" at
    z > z_alpha, Holm over D."""
    p = {}
    rows = {}
    for d, x in deficits.items():
        z = x["mean"] / x["se_block"] if x["se_block"] > 0 else math.inf * np.sign(x["mean"])
        p[str(d)] = 0.5 * math.erfc(z / math.sqrt(2))
        rows[str(d)] = {**x, "z": float(z), "p": p[str(d)]}
    rej = F.holm(p, alpha)
    for k in rows:
        rows[k]["reject"] = bool(rej[k])
    return {"by_d": rows, "rejected": [int(k) for k, v in rej.items() if v]}


def tier2_decide(primary: dict[str, Any], legs: dict[str, Any], pilot_go: bool) -> str:
    rej = set(primary["rejected"])
    if not pilot_go:
        return "stopped: pilot gate failed (no main jobs; any rerun needs a dated amendment)"
    others = legs["leg_i_population"] and legs["leg_iii_dephasings"] and legs["leg_contrast"]
    if 6 in rej and 4 in rej and legs["leg_ii_response"] and others:
        return "supported (under section 5): element-wise classical models with d <= 6 excluded"
    if 4 in rej and legs["leg_ii_response"] and others:
        return "partial: the exclusion passes only at D = 4"
    return (
        "not supported: "
        + ", ".join(
            k
            for k, ok in (
                ("exclusion at D = 4", 4 in rej),
                ("(i) population", legs["leg_i_population"]),
                ("(ii) response", legs["leg_ii_response"]),
                ("(iii) dephasings", legs["leg_iii_dephasings"]),
                ("Claim 3 contrast", legs["leg_contrast"]),
            )
            if not ok
        )
        + " failed"
    )


__all__ = [
    "AncillaRule",
    "Approval",
    "Backend",
    "Ledger",
    "LocalBackend",
    "ReselectionPolicy",
    "Tier1Rule",
    "Tier2Rule",
    "TierBudget",
    "adjacency",
    "budget_table",
    "first_harmonic",
    "freeze_approval",
    "hops",
    "keep_first",
    "mdi_analyse",
    "mdi_counts",
    "one_sided_p",
    "order_warnings",
    "perturb",
    "reselection",
    "run_stage",
    "select_ancillas",
    "select_pairs",
    "select_reference",
    "snapshot_hash",
    "tier1_analyse",
    "tier1_counts",
    "tier2_decide",
    "tier2_legs",
    "tier2_pilot_gate",
    "tier2_primary",
    "tost_diff",
]
