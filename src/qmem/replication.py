"""Phase 8 Stage 1 (issue #31): independent-pair replication design, numpy only.

Everything a replication of Phase 7 Stage 4c (#27) on other physical (memory, output) pairs of
the same processor needs before any data exist:

* **Pair selection** (``Device``, ``SelectionRules``, ``screen``, ``select``). The inputs are
  calibration and topology only: T1, T2, CZ, ``sx`` and readout errors, measure and reset
  support and durations, the coupling map (neighbours, spectators, distance from the
  reference footprint q59/q75/q74 and between the chosen pairs), and the expected power that a
  caller computes from the pair's *own* calibrated noise model. ``Device.from_snapshot`` keeps
  an allow-list of calibration fields and drops everything else, so no output-witness number
  can reach the ranking (tested). Every directed edge is a candidate; each exclusion carries
  its reasons.
* **Per-pair models** (``truth_superops``, ``shortfall_dephasing``): the pair's step pieces
  (``qmem.ablate_hw.StepPieces``, exported per pair from the same Phase 6 step) give every
  arm's process; the design's "truth" adds a Z dephasing of the memory in the reset slot,
  sized so the baseline witness is a stated fraction of the calibrated prediction (Phase 6 and
  #27 saw 72% and 75%). It is the one account of the shortfall #22 and #29 left standing
  (readout-window dephasing), and it moves no arm's ideal ratio.
* **Statistics and their per-shot covariance** (``STATS``, ``shot_influence``,
  ``moments_from_shots``): w(0), g(0..7) and the lag-1 excess, with per-shot influence
  functions, so sham equivalence is tested on the whole short-range process, not w(0) alone
  (#29 finding 4).
* **The decision rule** (``Rule``, ``decide``, vectorised): the gate (H1); A2's vanish bound
  with every leave-one-job-out bound (#29: dropping one job flipped #27's call); TOST of each
  sham against the baseline on ratios at a margin, as an intersection-union family; H2.
* **The power engine** (``power_mc``): per-job normal draws from the per-shot covariance,
  drift-conditional SE inflation per arm (#29's random effect over chunks), the full rule.
* **The analysis of shots** (``arm_statistics``, ``analyse_pair``, ``arm_job_interaction``):
  stratified shot / block bootstrap SEs as Phase 7 (``qmem.followup``'s indices), a
  DerSimonian-Laird random effect over pub chunks (``qmem.reanalysis.pool``), leave-one-job-
  out, and the arm x job interaction test (#29 finding 5).
* **Simulation** (``simulate_arm``): shots of a pair's arm in a schedule, with chunk-level
  drift of the memory's dephasing and an optional arm-specific job effect.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import ablate_hw as A
from . import followup as F
from . import reanalysis as R
from . import witness as W

Z975 = R.Z975
Z95 = R.Z95

# -- the device: calibration and topology only ------------------------------------------------

#: The calibration fields a qubit may carry into the selection (everything else is dropped).
QUBIT_FIELDS = ("t1", "t2", "readout", "sx", "measure_time", "reset_time", "measure", "reset")
#: The fields of a coupler.
CZ_FIELDS = ("error", "duration")


@dataclass(frozen=True)
class Qubit:
    t1: float | None
    t2: float | None
    readout: float | None
    sx: float | None
    measure_time: float | None
    reset_time: float | None
    measure: bool = True  # measurement supported (mid-circuit)
    reset: bool = True


@dataclass
class Device:
    """A calibration snapshot reduced to what the selection may see."""

    backend: str
    date: str
    qubits: dict[int, Qubit]
    cz: dict[tuple[int, int], dict[str, float | None]]  # (min, max) -> error, duration
    edges: list[tuple[int, int]]

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> Device:
        """From ``phase7_spectator``'s ``device.json`` layout (or a live export of the same
        shape). Only ``QUBIT_FIELDS`` / ``CZ_FIELDS`` survive: any other key (a witness, a
        P(1), a pilot result) is ignored by construction."""
        qubits = {}
        for k, q in snap["qubits"].items():
            kw = {f: q.get(f) for f in QUBIT_FIELDS if f in q}
            kw.setdefault("measure", True)
            kw.setdefault("reset", True)
            qubits[int(k)] = Qubit(
                **{f: kw.get(f) for f in QUBIT_FIELDS[:6]},
                measure=bool(kw["measure"]),
                reset=bool(kw["reset"]),
            )
        cz = {}
        for k, v in snap["cz"].items():
            a, b = (int(x) for x in k.split("-"))
            cz[(min(a, b), max(a, b))] = {f: v.get(f) for f in CZ_FIELDS}
        edges = sorted({(min(a, b), max(a, b)) for a, b in snap["edges"]})
        return cls(
            str(snap.get("backend", "")), str(snap.get("calibration_date", "")), qubits, cz, edges
        )

    def neighbours(self) -> dict[int, list[int]]:
        nb: dict[int, set[int]] = {q: set() for q in self.qubits}
        for a, b in self.edges:
            nb[a].add(b)
            nb[b].add(a)
        return {q: sorted(v) for q, v in nb.items()}

    def distances(self, sources: Sequence[int]) -> dict[int, int]:
        """Hops from the nearest of ``sources`` (breadth-first search on the coupling map)."""
        nb = self.neighbours()
        dist = {s: 0 for s in sources}
        dq = deque(sources)
        while dq:
            u = dq.popleft()
            for v in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    def cz_error(self, a: int, b: int) -> float | None:
        c = self.cz.get((min(a, b), max(a, b)))
        return None if c is None else c.get("error")

    def couplers(self, qubits: Sequence[int]) -> list[tuple[int, int]]:
        """Every coupler touching ``qubits``."""
        s = set(qubits)
        return [e for e in self.edges if e[0] in s or e[1] in s]


@dataclass(frozen=True)
class SelectionRules:
    """The frozen pair-selection rules (``experiments/configs/phase8_replication.toml``)."""

    reference: tuple[int, ...] = (59, 75, 74)  # the Phase 6/7 footprint (output, memory, sham)
    min_distance_reference: int = 3  # hops from any reference qubit to the pair or spectator
    min_distance_pairs: int = 3  # hops between chosen footprints (also allows parallel runs)
    n_pairs: int = 2
    mem_t1_min: float = 150e-6
    mem_t2_min: float = 100e-6
    out_t1_min: float = 100e-6
    out_t2_min: float = 50e-6
    out_readout_max: float = 0.02
    mem_readout_max: float = 0.03
    spec_readout_max: float = 0.05
    cz_max: float = 0.005
    sx_max: float = 1e-3
    power_min: float = 0.8  # P(H1 and H2) at the selection design
    named: tuple[tuple[int, int], ...] = ((67, 68), (149, 148), (104, 105))


def spectator_for(dev: Device, mem: int, out: int, rules: SelectionRules) -> int | None:
    """The sham's measured spectator: the memory's neighbour other than the output with the
    lowest readout error within ``spec_readout_max`` (ties: lowest index). On heavy hex it is
    never a neighbour of the output."""
    nb = dev.neighbours()
    out_nb = set(nb[out])
    cands = []
    for s in nb[mem]:
        q = dev.qubits[s]
        if s == out or s in out_nb or q.readout is None or not q.measure:
            continue
        if q.readout <= rules.spec_readout_max:
            cands.append((q.readout, s))
    return min(cands)[1] if cands else None


def _low(why: list[str], v: float | None, lo: float, what: str) -> None:
    if v is None:
        why.append(f"{what} missing")
    elif v < lo:
        why.append(f"{what} {v * 1e6:.0f} us < {lo * 1e6:.0f} us")


def _high(why: list[str], v: float | None, hi: float, what: str) -> None:
    if v is None:
        why.append(f"{what} missing")
    elif v > hi:
        why.append(f"{what} {v:.4g} > {hi:.4g}")


def screen(dev: Device, rules: SelectionRules) -> list[dict[str, Any]]:
    """One row per directed edge (memory, output): calibration, topology and the reasons it is
    excluded before any power is computed (empty: it passes the screen)."""
    nb = dev.neighbours()
    d_ref = dev.distances(list(rules.reference))
    rows = []
    for a, b in dev.edges:
        for mem, out in ((a, b), (b, a)):
            qm, qo = dev.qubits[mem], dev.qubits[out]
            why: list[str] = []
            cz = dev.cz_error(mem, out)

            _low(why, qm.t1, rules.mem_t1_min, "memory T1")
            _low(why, qm.t2, rules.mem_t2_min, "memory T2")
            _low(why, qo.t1, rules.out_t1_min, "output T1")
            _low(why, qo.t2, rules.out_t2_min, "output T2")
            _high(why, qo.readout, rules.out_readout_max, "output readout")
            _high(why, qm.readout, rules.mem_readout_max, "memory readout")
            _high(why, cz, rules.cz_max, "CZ error")
            _high(why, qm.sx, rules.sx_max, "memory sx error")
            _high(why, qo.sx, rules.sx_max, "output sx error")
            if not (qo.measure and qo.reset):
                why.append("output lacks measure or reset")
            if not qm.measure:
                why.append("memory lacks mid-circuit measure (A2)")
            spec = spectator_for(dev, mem, out, rules)
            if spec is None:
                why.append("no spectator neighbour of the memory within the readout bound")
            foot = [mem, out] + ([spec] if spec is not None else [])
            dmin = min(d_ref.get(q, 10**6) for q in foot)
            if dmin < rules.min_distance_reference:
                why.append(
                    f"{dmin} hops from the reference footprint < {rules.min_distance_reference}"
                )
            rows.append(
                {
                    "memory": mem,
                    "output": out,
                    "spectator": spec,
                    "named_candidate": (mem, out) in rules.named,
                    "distance_from_reference": dmin,
                    "memory_t1_us": None if qm.t1 is None else qm.t1 * 1e6,
                    "memory_t2_us": None if qm.t2 is None else qm.t2 * 1e6,
                    "output_t1_us": None if qo.t1 is None else qo.t1 * 1e6,
                    "output_t2_us": None if qo.t2 is None else qo.t2 * 1e6,
                    "memory_readout": qm.readout,
                    "output_readout": qo.readout,
                    "spectator_readout": None if spec is None else dev.qubits[spec].readout,
                    "cz_error": cz,
                    "memory_sx": qm.sx,
                    "output_sx": qo.sx,
                    "measure_time_s": qo.measure_time,
                    "reset_time_s": qo.reset_time,
                    "memory_degree": len(nb[mem]),
                    "couplers": " ".join(f"{x}-{y}" for x, y in dev.couplers(foot)),
                    "screen_reasons": why,
                }
            )
    return rows


@dataclass
class Selection:
    rows: list[dict[str, Any]]  # every candidate, with status and reasons
    chosen: list[dict[str, Any]]
    n_qualified: int

    @property
    def statement(self) -> str:
        n = len(self.chosen)
        if n == 0:
            return "no pair qualifies"
        if n == 1:
            return "only one pair qualifies"
        return f"{n} pairs qualify and are chosen"


def select(
    dev: Device,
    rules: SelectionRules,
    power: Callable[[dict[str, Any]], dict[str, float]],
    rank_key: str = "power_joint",
    tie_key: str = "snr",
) -> Selection:
    """The frozen algorithm.

    1. ``screen`` every directed edge (calibration and topology thresholds).
    2. For each pair that passes, ``power(row)`` (the caller's computation from the pair's own
       calibrated noise model) returns at least ``power_joint`` (P(H1 and H2) at the selection
       design) and ``snr`` (the predicted |w(0)| over its per-shot SD). Pairs below
       ``power_min`` are excluded.
    3. Rank by ``power_joint`` (rounded to 0.01), then ``snr``, then (memory, output) ascending;
       take pairs greedily, skipping any that shares a qubit with, or is within
       ``min_distance_pairs`` hops of, a pair already chosen, until ``n_pairs``.

    Deterministic for a fixed snapshot and power function.
    """
    rows = screen(dev, rules)
    qualified = []
    for r in rows:
        r["status"] = "excluded" if r["screen_reasons"] else "screened"
        r["reasons"] = list(r["screen_reasons"])
        if r["screen_reasons"]:
            continue
        r.update(power(r))
        if r[rank_key] < rules.power_min:
            r["status"] = "excluded"
            r["reasons"].append(f"P(H1 and H2) {r[rank_key]:.3f} < {rules.power_min}")
            continue
        r["status"] = "qualified"
        qualified.append(r)
    qualified.sort(key=lambda r: (-round(r[rank_key], 2), -r[tie_key], r["memory"], r["output"]))
    chosen: list[dict[str, Any]] = []
    for rank, r in enumerate(qualified):
        r["rank"] = rank + 1
        foot = [r["memory"], r["output"], r["spectator"]]
        if len(chosen) >= rules.n_pairs:
            r["reasons"].append("not needed: n_pairs already chosen")
            continue
        clash = False
        for c in chosen:
            cf = [c["memory"], c["output"], c["spectator"]]
            d = dev.distances(cf)
            if set(foot) & set(cf):
                r["reasons"].append(f"shares a qubit with chosen q{c['memory']}/q{c['output']}")
                clash = True
                break
            if min(d.get(q, 10**6) for q in foot) < rules.min_distance_pairs:
                r["reasons"].append(
                    f"< {rules.min_distance_pairs} hops from chosen q{c['memory']}/q{c['output']}"
                )
                clash = True
                break
        if clash:
            continue
        r["status"] = "chosen"
        chosen.append(r)
    return Selection(rows, chosen, len(qualified))


# -- per-pair models ----------------------------------------------------------------------------

ARMS = ("baseline", "A2", "A2s_spec", "A2s")
SHAMS = ("A2s_spec", "A2s")


def truth_superops(
    arm: str, pieces: A.StepPieces, dephase: float = 0.0, hyp: A.Hypothesis | None = None
) -> np.ndarray:
    """``step_superops`` of ``arm`` with an extra Z dephasing ``dephase`` of the memory in the
    reset slot (where it idles during the output's reset and readout window)."""
    sup = A.step_superops(arm, pieces, hyp)
    if dephase:
        d = A.on_q75(A.dephase_kraus(dephase))
        sup = np.stack([s @ d for s in sup])
    return sup


def arm_process(
    arm: str, pieces: A.StepPieces, dephase: float = 0.0, hyp: A.Hypothesis | None = None
) -> W.LinearProcess:
    return A.process(truth_superops(arm, pieces, dephase, hyp))


def shortfall_dephasing(
    pieces: A.StepPieces, fraction: float, s: A.Settings, tol: float = 1e-10
) -> float:
    """The dephasing ``p`` at which the baseline's w(0) is ``fraction`` of ``p = 0``'s
    (bisection; w(0) moves monotonically towards 0 with ``p`` over the bracket)."""
    w_full = A.predict(arm_process("baseline", pieces), s)["w0"]
    target = fraction * w_full
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w = A.predict(arm_process("baseline", pieces, mid), s)["w0"]
        if abs(w) > abs(target):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# -- statistics and per-shot influence ----------------------------------------------------------

K_MAX, BURN, LENGTH = 7, 16, 64
STATS = ("w0", *(f"g{k}" for k in range(K_MAX + 1)), "lag1")


def lag1_parts(shots: np.ndarray, burn: int = BURN) -> np.ndarray:
    """Per shot ``(n10, n11, n00, n01)`` (``qmem.followup.lag1_counts``)."""
    return F.lag1_counts(shots, burn)


def lag1_of(c: np.ndarray) -> np.ndarray:
    c = np.atleast_2d(c)
    return c[:, 1] / (c[:, 0] + c[:, 1]) - c[:, 3] / (c[:, 2] + c[:, 3])


def stat_vector(per: np.ndarray, lc: np.ndarray) -> np.ndarray:
    """(w0, g0..g7, lag1) of pooled shots (``per``: per-shot windows; ``lc``: lag-1 counts)."""
    g = per.mean(0)
    return np.concatenate([[float(W.witness(g, 0))], g, lag1_of(lc.sum(0))])


def shot_influence(per: np.ndarray, lc: np.ndarray) -> np.ndarray:
    """(n, len(STATS)) influence functions: the delta method for w(0) and the lag-1 excess,
    the per-shot windows themselves for g(k). Their covariance / n is the stats' covariance."""
    g = per.mean(0)
    grad = W.witness_grad(g, 0)
    w_inf = (per - g) @ grad
    m = lc.mean(0)
    p1, p0 = m[1] / (m[0] + m[1]), m[3] / (m[2] + m[3])
    l_inf = (lc[:, 1] - p1 * (lc[:, 0] + lc[:, 1])) / (m[0] + m[1]) - (
        lc[:, 3] - p0 * (lc[:, 2] + lc[:, 3])
    ) / (m[2] + m[3])
    return np.column_stack([w_inf, per - g, l_inf])


@dataclass
class Moments:
    """Exact means of ``STATS`` for one arm and their per-shot covariance (from samples)."""

    mean: np.ndarray  # (S,)
    cov: np.ndarray  # (S, S), per shot

    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Moments:
        return cls(np.asarray(d["mean"]), np.asarray(d["cov"]))


def exact_stats(proc: W.LinearProcess, length: int = LENGTH, burn: int = BURN) -> np.ndarray:
    g = proc.expected_windows(K_MAX, burn, length, common=True)
    bl = proc.block_probs(range(burn, length - 1), 2)
    lag = bl[3] / (bl[2] + bl[3]) - bl[1] / (bl[0] + bl[1])
    return np.concatenate([[float(W.witness(g, 0))], g, [lag]])


def moments(proc: W.LinearProcess, n: int, seed: int) -> Moments:
    shots = proc.sample(n, LENGTH, seed)
    per = W.per_shot_windows(shots, K_MAX, BURN, common=True)
    inf = shot_influence(per, lag1_parts(shots))
    return Moments(exact_stats(proc), np.cov(inf, rowvar=False))


# -- the decision rule (vectorised over draws) -----------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """The pre-registered per-pair rule (``docs/phase8-replication-prereg.md`` section 4)."""

    gate_z: float = 5.0
    vanish_fraction: float = 0.10
    vanish_z: float = Z975
    persist_z: float = 5.0
    persist_fraction: float = 0.25
    margin: float = 0.10  # TOST margin on every sham / baseline ratio
    tost_alpha: float = 0.05  # per bound (a 90% CI inside the margin)
    lojo: bool = True  # every leave-one-job-out bound must also be below the threshold
    full_sham: str | None = "A2s"  # the sham tested on ``sham_stats`` (the delay-only sham)
    w0_sham: str | None = "A2s_spec"  # the sham tested on w(0) for H2 (reported on all)
    sham_stats: tuple[str, ...] = ("w0", "lag1")  # equivalence needed of ``full_sham``
    g_alpha: float = 0.01  # g(k) differences: two-sided, Holm over k within each sham

    @property
    def z_tost(self) -> float:
        return R.norm_ppf(1 - self.tost_alpha)


def _vanish(wa, sa, wb, rule: Rule):
    return np.abs(wa) + rule.vanish_z * sa < rule.vanish_fraction * np.abs(wb)


def _persist(wa, sa, wb, rule: Rule):
    return (np.abs(wa) >= rule.persist_z * sa) & (np.abs(wa) >= rule.persist_fraction * np.abs(wb))


def _holm_any(z: np.ndarray, alpha: float) -> np.ndarray:
    """Whether Holm's procedure at family-wise ``alpha`` rejects any of the two-sided z-tests
    along the last axis (equivalently: the smallest p is below ``alpha / m``)."""
    m = z.shape[-1]
    zc = R.norm_ppf(1 - alpha / (2 * m))
    return np.any(np.abs(z) > zc, axis=-1)


def tost_ratio(es, ss, eb, sb, rule: Rule):
    """TOST of ``es / eb`` in ``[1 - margin, 1 + margin]`` with the delta-method SE; returns
    (ratio, se, equivalent, differs). ``differs``: the 95% CI excludes 1."""
    r = es / eb
    se = np.abs(r) * np.sqrt((ss / es) ** 2 + (sb / eb) ** 2)
    z = rule.z_tost
    eq = (r - z * se > 1 - rule.margin) & (r + z * se < 1 + rule.margin)
    differs = np.abs(r - 1) > Z975 * se
    return r, se, eq, differs


def decide(est: dict[str, dict[str, Any]], rule: Rule) -> dict[str, Any]:
    """The per-pair decision.

    ``est[arm]`` holds ``value`` and ``se`` (arrays (..., S) over ``STATS``; the SE is the
    drift-conditional one) and, for ``baseline`` and ``A2``, ``lojo_value`` / ``lojo_se``
    (..., J, S) for every leave-one-job-out estimate. Works on single estimates (1-d) and on
    stacks of Monte Carlo draws. Returns booleans (or arrays of them) and the components.
    """
    iw = STATS.index("w0")
    for sham in (rule.full_sham, rule.w0_sham):
        if sham is not None and sham not in est:
            raise KeyError(f"the rule needs arm {sham!r}, which has no estimate")
    b, a2 = est["baseline"], est["A2"]
    wb, sb = b["value"][..., iw], b["se"][..., iw]
    wa, sa = a2["value"][..., iw], a2["se"][..., iw]
    h1 = (wb < 0) & (np.abs(wb) >= rule.gate_z * sb)
    van_all = _vanish(wa, sa, wb, rule)
    per_all = _persist(wa, sa, wb, rule)
    out: dict[str, Any] = {"h1": h1, "vanish_all_jobs": van_all, "persist": per_all}
    bound = (np.abs(wa) + rule.vanish_z * sa) / np.abs(wb)
    out["a2_bound"] = bound
    out["a2_ratio"] = wa / wb
    if rule.lojo and "lojo_value" in a2:
        wbl = b["lojo_value"][..., iw]
        wal, sal = a2["lojo_value"][..., iw], a2["lojo_se"][..., iw]
        lb = (np.abs(wal) + rule.vanish_z * sal) / np.abs(wbl)
        out["a2_lojo_max_bound"] = lb.max(-1)
        van = van_all & np.all(_vanish(wal, sal, wbl, rule), axis=-1)
    else:
        van = van_all
    out["vanish"] = van
    out["a2_call"] = np.where(van, "vanishes", np.where(per_all, "persists", "ambiguous"))
    for sham in SHAMS:
        if sham not in est:
            continue
        s = est[sham]
        r, se, eq, dif = tost_ratio(s["value"], s["se"], b["value"], b["se"], rule)
        out[f"{sham}_ratio"] = r
        out[f"{sham}_ratio_se"] = se
        out[f"{sham}_equiv"] = eq  # (..., S)
        out[f"{sham}_differs"] = dif
        out[f"{sham}_equiv_w0"] = eq[..., iw]
        sel = [STATS.index(k) for k in rule.sham_stats]
        out[f"{sham}_equiv_all"] = np.all(eq[..., sel], axis=-1)
        # g(k): difference tests with Holm over k (reported; not part of H2)
        gi = [STATS.index(f"g{k}") for k in range(K_MAX + 1)]
        dz = (s["value"][..., gi] - b["value"][..., gi]) / np.sqrt(
            s["se"][..., gi] ** 2 + b["se"][..., gi] ** 2
        )
        out[f"{sham}_g_z"] = dz
        out[f"{sham}_g_any_differs"] = _holm_any(dz, rule.g_alpha)
    full = out.get(f"{rule.full_sham}_equiv_all", True)
    w0s = out.get(f"{rule.w0_sham}_equiv_w0", True)
    out["h2_sham"] = np.asarray(full) & np.asarray(w0s)
    out["h2"] = van & out["h2_sham"]
    out["h1_and_h2"] = h1 & out["h2"]
    return out


def call_label(d: dict[str, Any]) -> str:
    """The reported per-pair outcome of one decision (not vectorised)."""
    if not bool(d["h1"]):
        return "failed: baseline witness not replicated (H1)"
    if bool(d["h2"]):
        return "replicates: H1 and H2 (A2 vanishes; shams equivalent)"
    if bool(d["vanish_all_jobs"]) and not bool(d["vanish"]):
        return "ambiguous: A2 vanishes on all jobs but not leaving one job out"
    a2 = str(d["a2_call"])
    if a2 == "vanishes":
        return "partial: A2 vanishes, sham equivalence not established"
    return f"partial: H1 only (A2 {a2})"


# -- power engine ---------------------------------------------------------------------------------


@dataclass
class Design:
    """Shots per arm, jobs, drift-conditional SE inflation per arm, and the true means."""

    shots: dict[str, int]
    n_jobs: int
    inflation: dict[str, float] = field(default_factory=dict)


def cov_sqrt(cov: np.ndarray) -> np.ndarray:
    """A square root ``L`` with ``L L^T = cov`` (symmetric eigendecomposition; tiny negative
    eigenvalues from rounding are set to 0, so a singular covariance is fine)."""
    c = 0.5 * (cov + cov.T)
    val, vec = np.linalg.eigh(c)
    return vec * np.sqrt(np.clip(val, 0.0, None))


def power_mc(
    mom: dict[str, Moments],
    design: Design,
    rule: Rule,
    n: int,
    seed: int,
    shift: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Monte Carlo probabilities of every component of ``decide``.

    Each arm's ``J`` job estimates are drawn from ``N(mean, cov * f^2 J / N)`` (``f``: the
    arm's inflation), the pooled estimate is their mean, its SE ``sd f / sqrt(N)``; the
    leave-one-job-out estimates are the means of the other ``J - 1`` jobs with SE scaled by
    ``sqrt(J / (J - 1))``. ``shift[arm]`` replaces an arm's mean (e.g. a true A2 ratio or a
    sham ratio other than 1). Common random numbers: the same ``seed`` gives the same draws.
    """
    rng = np.random.default_rng(seed)
    J = design.n_jobs
    est: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if arm not in design.shots or design.shots[arm] <= 0:
            continue
        m = mom[arm]
        mu = m.mean if shift is None or arm not in shift else np.asarray(shift[arm])
        f = design.inflation.get(arm, 1.0)
        N = design.shots[arm]
        chol = cov_sqrt(m.cov)
        z = rng.standard_normal((n, J, len(mu)))
        jobs = mu + f * math.sqrt(J / N) * (z @ chol.T)
        val = jobs.mean(1)
        se = np.broadcast_to(f * m.sd() / math.sqrt(N), val.shape)
        lv = (val[:, None, :] * J - jobs) / (J - 1)
        ls = np.broadcast_to(se[:, None, :] * math.sqrt(J / (J - 1)), lv.shape)
        est[arm] = {"value": val, "se": se, "lojo_value": lv, "lojo_se": ls}
    d = decide(est, rule)
    keys = ["h1", "vanish_all_jobs", "vanish", "persist", "h2_sham", "h2", "h1_and_h2"]
    keys += [k for k in d if k.endswith(("_equiv_w0", "_equiv_all", "_g_any_differs"))]
    out = {k: float(np.mean(d[k])) for k in keys}
    out["a2_bound_median"] = float(np.median(d["a2_bound"]))
    return out


def scaled_shift(mom: Moments, ratio_w0: float | None = None, ratio_all: float | None = None):
    """A mean vector with w(0) (or every statistic) scaled by a ratio."""
    mu = mom.mean.copy()
    if ratio_all is not None:
        mu = mu * ratio_all
    if ratio_w0 is not None:
        mu[STATS.index("w0")] *= ratio_w0
    return mu


# -- cost ------------------------------------------------------------------------------------------


def jobs_for(shot_seconds: float, cost: F.CostModel, max_job_s: float) -> int:
    """Fewest jobs with every job's estimate (overhead + its share) at most ``max_job_s``."""
    n = max(1, math.ceil(shot_seconds / max(max_job_s - cost.overhead_s, 1e-9)))
    while cost.overhead_s + shot_seconds / n > max_job_s:
        n += 1
    return n


def plan_cost(
    shots: dict[str, int],
    durations: dict[str, float],
    cost: F.CostModel,
    max_job_s: float,
    min_jobs: int,
) -> dict[str, float]:
    """Estimate and upper bound (``CostModel.job``'s rule per job) of a plan whose shots are
    spread evenly over the fewest jobs within ``max_job_s`` (at least ``min_jobs``)."""
    sec = sum(cost.shot_seconds(n, durations[a]) for a, n in shots.items())
    J = max(min_jobs, jobs_for(sec, cost, max_job_s))
    est = J * cost.overhead_s + sec
    ub = (1 + cost.margin) * est + J * cost.billing_round_s
    return {
        "jobs": J,
        "shots": int(sum(shots.values())),
        "shot_s": sec,
        "estimate_s": est,
        "upper_s": ub,
    }


# -- analysis of shots ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSettings:
    k_max: int = K_MAX
    burn: int = BURN
    block: int = 500
    n_boot: int = 1000
    seed: int = 31


def _counts(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n))
    for b in range(len(idx)):
        out[b] = np.bincount(idx[b], minlength=n)
    return out


def _boot_stats(cnt: np.ndarray, per: np.ndarray, lc: np.ndarray) -> np.ndarray:
    """``STATS`` of each resample given its multiplicities ``cnt`` (B, n)."""
    g = cnt @ per / cnt.sum(1, keepdims=True)
    w0 = np.asarray(W.witness(g, 0))
    lag = lag1_of(cnt @ lc)
    return np.column_stack([w0, g, lag])


def chunk_re_se(per: np.ndarray, lc: np.ndarray, chunks: Sequence[np.ndarray]) -> dict[str, Any]:
    """DerSimonian-Laird random effect over pub chunks of each statistic (#29's
    drift-conditional SE), from each chunk's estimate and its influence-function SE."""
    ests, ses = [], []
    for c in chunks:
        ests.append(stat_vector(per[c], lc[c]))
        inf = shot_influence(per[c], lc[c])
        ses.append(inf.std(0, ddof=1) / math.sqrt(len(c)))
    ests, ses = np.array(ests), np.array(ses)
    out = {"re_se": [], "tau": [], "q_p": []}
    for k in range(len(STATS)):
        p = R.pool(ests[:, k], ses[:, k])
        out["re_se"].append(p["re_se"])
        out["tau"].append(math.sqrt(p["tau2"]))
        out["q_p"].append(p["q_p"])
    return {k: np.array(v) for k, v in out.items()}


def arm_statistics(
    shots: np.ndarray, jobs: np.ndarray, pubs: np.ndarray, s: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """One arm: ``STATS`` with Phase 7's primary SE (max of the shot and the circular block
    bootstrap, within job and stratified by job; ``qmem.followup``'s indices), the
    drift-conditional SE (max of that and the random effect over pub chunks), and the same for
    every leave-one-job-out subset. ``jobs`` / ``pubs`` label each shot (nominal time order)."""
    shots = np.asarray(shots)
    jobs, pubs = np.asarray(jobs), np.asarray(pubs)
    per = W.per_shot_windows(shots, s.k_max, s.burn, common=True)
    lc = lag1_parts(shots, s.burn)

    def one(sel: np.ndarray, sd: int) -> dict[str, Any]:
        p, c, jb, pb = per[sel], lc[sel], jobs[sel], pubs[sel]
        segs = F.strata(jb)
        n = len(p)
        cs = _counts(F.stratified_indices(segs, 1, s.n_boot, sd), n)
        cb = _counts(F.stratified_indices(segs, s.block, s.n_boot, sd + 1), n)
        bs, bb = _boot_stats(cs, p, c), _boot_stats(cb, p, c)
        se_shot, se_block = bs.std(0, ddof=1), bb.std(0, ddof=1)
        primary = np.maximum(se_shot, se_block)
        re = chunk_re_se(p, c, F.strata(jb * 10**6 + pb))
        return {
            "n": n,
            "value": stat_vector(p, c),
            "se_shot": se_shot,
            "se_block": se_block,
            "se_primary": primary,
            "se_re_chunks": re["re_se"],
            "tau_chunks": re["tau"],
            "q_p_chunks": re["q_p"],
            "se": np.maximum(primary, re["re_se"]),
            "p1": float(shots[sel][:, s.burn :].mean()),
        }

    allj = one(np.arange(len(shots)), seed)
    labels = [int(j) for j in dict.fromkeys(jobs.tolist())]
    lo = [one(np.nonzero(jobs != j)[0], seed + 7919 * (k + 1)) for k, j in enumerate(labels)]
    allj["jobs"] = labels
    allj["lojo_value"] = np.array([x["value"] for x in lo])
    allj["lojo_se"] = np.array([x["se"] for x in lo])
    allj["lojo_se_primary"] = np.array([x["se_primary"] for x in lo])
    allj["per_job"] = [
        {
            "job": j,
            "n": int((jobs == j).sum()),
            "p1": float(shots[jobs == j][:, s.burn :].mean()),
            "se_p1": float(
                shots[jobs == j][:, s.burn :].mean(1).std(ddof=1)
                / math.sqrt(int((jobs == j).sum()))
            ),
            **{
                k: float(v)
                for k, v in zip(STATS, stat_vector(per[jobs == j], lc[jobs == j]), strict=True)
            },
            **{
                f"se_{k}": float(v)
                for k, v in zip(
                    STATS,
                    shot_influence(per[jobs == j], lc[jobs == j]).std(0, ddof=1)
                    / math.sqrt(int((jobs == j).sum())),
                    strict=True,
                )
            },
        }
        for j in labels
    ]
    return allj


def arm_job_interaction(
    arms: dict[str, dict[str, Any]], ref: str = "baseline", stats=("p1", "w0", "lag1")
) -> list[dict[str, Any]]:
    """#29 finding 5's test: for each other arm and statistic, the per-job difference from
    ``ref`` and Cochran's Q across jobs (a job-level shift of one arm shows as heterogeneity
    that a drift common to every arm cannot produce)."""
    out = []
    b = {r["job"]: r for r in arms[ref]["per_job"]}
    for arm, a in arms.items():
        if arm == ref:
            continue
        for st in stats:
            d, s = [], []
            for r in a["per_job"]:
                if r["job"] not in b:
                    continue
                rb = b[r["job"]]
                d.append(r[st] - rb[st])
                s.append(math.hypot(r[f"se_{st}"], rb[f"se_{st}"]))
            if len(d) < 2:
                continue
            p = R.pool(d, s)
            out.append(
                {
                    "arm": arm,
                    "vs": ref,
                    "stat": st,
                    "per_job": d,
                    "per_job_se": s,
                    "fe": p["fe"],
                    "q": p["q"],
                    "q_p": p["q_p"],
                }
            )
    return out


def loo_outlier(d: Sequence[float], se: Sequence[float]) -> tuple[int, float]:
    """The job whose difference is furthest from the pooled others (index, leave-one-out z)."""
    d, se = np.asarray(d, float), np.asarray(se, float)
    zs = []
    for j in range(len(d)):
        rest = np.arange(len(d)) != j
        p = R.pool(d[rest], se[rest])
        zs.append((d[j] - p["fe"]) / math.hypot(se[j], p["fe_se"]))
    k = int(np.argmax(np.abs(zs)))
    return k, float(zs[k])


def analyse_pair(arms: dict[str, dict[str, Any]], rule: Rule, alpha_armjob: float = 0.01):
    """The decision on analysed arms, with #29 finding 5's arm x job rule (prereg section 7).

    For every other arm and P(1) / w(0), Cochran's Q of its per-job difference from the
    baseline. Where Q's p < ``alpha_armjob``, the job with the largest leave-one-out |z| is
    flagged (one per test). One flagged job: the decision is recomputed without it (every arm
    of that job dropped); if A2's call, H1 or H2 changes, or a P(1) / w(0) test is still
    heterogeneous without it, the pair is *ambiguous*. More than one flagged job: ambiguous.
    """
    est = {
        a: {k: v[k] for k in ("value", "se", "lojo_value", "lojo_se") if k in v}
        for a, v in arms.items()
    }
    d = decide(est, rule)
    inter = arm_job_interaction(arms)
    flagged: set[int] = set()
    for t in inter:
        if t["stat"] in ("p1", "w0") and t["q_p"] < alpha_armjob:
            jobs = [r["job"] for r in arms[t["arm"]]["per_job"]]
            k, z = loo_outlier(t["per_job"], t["per_job_se"])
            t["outlier_job"], t["outlier_z"] = jobs[k], z
            flagged.add(jobs[k])
    out: dict[str, Any] = {
        "decision": d,
        "label": call_label(d),
        "interaction": inter,
        "flagged_jobs": sorted(flagged),
    }
    if len(flagged) > 1:
        out["label"] = "ambiguous: more than one job flagged (arm x job)"
    elif flagged:
        (job,) = flagged
        sub = {}
        for a, v in arms.items():
            drop = v["jobs"].index(job)
            sub[a] = {"value": v["lojo_value"][drop], "se": v["lojo_se"][drop]}
        d2 = decide(sub, Rule(**{**rule.__dict__, "lojo": False}))
        residual = []
        for t in inter:
            if t["stat"] not in ("p1", "w0"):
                continue
            jobs = [r["job"] for r in arms[t["arm"]]["per_job"]]
            keep = [i for i, j in enumerate(jobs) if j != job]
            if len(keep) >= 2:
                p = R.pool([t["per_job"][i] for i in keep], [t["per_job_se"][i] for i in keep])
                if p["q_p"] < alpha_armjob:
                    residual.append(f"{t['arm']} {t['stat']}")
        out["without_flagged"] = {
            "job": job,
            "a2_call": str(d2["a2_call"]),
            "h1": bool(d2["h1"]),
            "h2": bool(d2["h2"]),
            "residual_heterogeneity": residual,
        }
        if (
            bool(d2["vanish"]) != bool(d["vanish_all_jobs"])
            or bool(d2["h1"]) != bool(d["h1"])
            or bool(d2["h2_sham"]) != bool(d["h2_sham"])
        ):
            out["label"] = "ambiguous: the call changes without the flagged job"
        elif residual:
            out["label"] = "ambiguous: arm x job heterogeneity beyond one job"
    return out


# -- simulation ------------------------------------------------------------------------------------


def simulate_arm(
    procs: Sequence[W.LinearProcess],
    pubs: Sequence[F.Pub],
    level_of_chunk: Callable[[F.Pub], int],
    seed: int,
    flip_one_to_zero: dict[int, float] | None = None,
) -> dict[str, np.ndarray]:
    """Shots of one arm in schedule order: pub ``p`` comes from ``procs[level_of_chunk(p)]``
    (chunk-level drift); ``flip_one_to_zero[job]`` flips recorded 1s to 0 in that job (an
    arm-specific job effect like #27's job1 dip)."""
    ys, jobs, pos = [], [], []
    for k, p in enumerate(sorted(pubs, key=lambda q: (q.job, q.position))):
        y = procs[level_of_chunk(p)].sample(p.shots, LENGTH, seed + 1009 * k + 17 * p.job)
        if flip_one_to_zero and p.job in flip_one_to_zero:
            rng = np.random.default_rng([seed, p.job, p.position])
            y = np.where((y == 1) & (rng.random(y.shape) < flip_one_to_zero[p.job]), 0, y)
            y = y.astype(np.uint8)
        ys.append(y)
        jobs.append(np.full(p.shots, p.job))
        pos.append(np.full(p.shots, p.position))
    return {"shots": np.concatenate(ys), "jobs": np.concatenate(jobs), "pubs": np.concatenate(pos)}


__all__ = [
    "ARMS",
    "SHAMS",
    "STATS",
    "AnalysisSettings",
    "Design",
    "Device",
    "Moments",
    "Qubit",
    "Rule",
    "Selection",
    "SelectionRules",
    "analyse_pair",
    "arm_job_interaction",
    "arm_process",
    "arm_statistics",
    "call_label",
    "chunk_re_se",
    "decide",
    "exact_stats",
    "jobs_for",
    "loo_outlier",
    "moments",
    "plan_cost",
    "power_mc",
    "scaled_shift",
    "screen",
    "select",
    "shortfall_dephasing",
    "shot_influence",
    "simulate_arm",
    "spectator_for",
    "stat_vector",
    "tost_ratio",
    "truth_superops",
]
