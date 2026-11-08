"""Phase 7 Stage 3 (issue #24): the pre-registered hardware follow-up, numpy only.

Everything here is fixed before any data exist and is importable without Qiskit:

* **Schedule** (``make_schedule``): every arm is split into chunks (pubs) of at most
  ``max_chunk`` shots, spread over ``n_jobs`` jobs so that every job holds every tier-1 arm;
  within a job the pubs run in a seeded random order. A3's random-Pauli instances are dealt
  to the jobs by a seeded permutation, one pub per instance.
* **Budget and submission guards** (``job_cost``, ``budget_check``, ``submission_check``):
  Phase 6's cost model (overhead per job + shots x (rep_delay + duration)), an upper bound
  (25% on the estimate, plus whole-second billing), the per-job ``max_qpu_seconds``, the
  remaining allowance, the Open plan, the owner's hard cap, and the approval URL plus the
  explicit ``--submit`` flag. Each refusal raises ``Refused``. The usage guard (Stage 4,
  ``job_charge``, ``usage_guard``) reads each job's IBM-reported charge, counts
  max(charged, upper bound) against the cap, and stops (``UsageExceeded``) if a job was
  charged more than its upper bound.
* **Analysis** (``analyse_arm``, ``segment_policy``): the primary statistic w(0) with a
  common start set; the primary SE = max(shot bootstrap, circular block bootstrap at
  ``b`` shots), each within job and stratified by job when pooled (#19's policy); per-job
  and per-chunk witnesses; D1's lag-1 excess; #19's change-point, homogeneity and constancy
  tests per job.
* **Decision rule** (``decide``): the replication gate, the primary outcome on A2
  (vanishes / persists / ambiguous) with its segment sensitivity, and the secondary family
  with Holm's correction.

The bootstrap here draws the same indices as ``qmem.drift.stratified_block_indices`` and
computes each resample's ``g`` as a count-weighted mean (identical to
``qmem.witness.bootstrap_g``, tested), which is fast enough for the synthetic validation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import artefacts as AR
from . import drift as D
from . import witness as W

# -- schedule ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pub:
    """One pub: ``shots`` shots of ``arm`` (A3: of instance ``instance``), chunk ``chunk``."""

    job: int
    position: int
    arm: str
    chunk: int
    shots: int
    instance: int | None = None

    @property
    def key(self) -> str:
        inst = "" if self.instance is None else f"_i{self.instance:03d}"
        return f"j{self.job}_p{self.position:03d}_{self.arm}{inst}_c{self.chunk}"


def split_even(total: int, parts: int) -> list[int]:
    """``total`` split into ``parts`` integers differing by at most one (larger ones first)."""
    q, r = divmod(total, parts)
    return [q + (1 if i < r else 0) for i in range(parts)]


def make_schedule(
    arms: dict[str, int],
    n_jobs: int,
    max_chunk: int,
    seed: int,
    instanced: dict[str, tuple[int, int]] | None = None,
) -> list[list[Pub]]:
    """Jobs of pubs. ``arms``: shots per (non-instanced) arm. ``instanced``: arm -> (instances,
    shots per instance), one pub per instance (A3). Deterministic from ``seed``.

    Each non-instanced arm's shots are split evenly over the jobs, then each job's share into
    the fewest chunks of at most ``max_chunk`` shots (also split evenly). Instances are dealt
    to jobs round-robin after a seeded permutation. Pub order within job ``j`` is the
    permutation ``default_rng([seed, j])``.
    """
    if n_jobs < 1 or max_chunk < 1:
        raise ValueError("n_jobs and max_chunk must be positive")
    per_job: list[list[tuple[str, int, int, int | None]]] = [[] for _ in range(n_jobs)]
    for arm, total in arms.items():
        for j, share in enumerate(split_even(total, n_jobs)):
            if share == 0:
                continue
            n_chunks = -(-share // max_chunk)
            for c, s in enumerate(split_even(share, n_chunks)):
                per_job[j].append((arm, c, s, None))
    for arm, (n_inst, shots) in (instanced or {}).items():
        if shots > max_chunk:
            raise ValueError(f"{arm}: {shots} shots per instance > max_chunk")
        order = np.random.default_rng([seed, 10**6]).permutation(n_inst)
        for k, inst in enumerate(order):
            per_job[k % n_jobs].append((arm, 0, shots, int(inst)))
    jobs = []
    for j, items in enumerate(per_job):
        perm = np.random.default_rng([seed, j]).permutation(len(items))
        jobs.append(
            [
                Pub(j, pos, items[i][0], items[i][1], items[i][2], items[i][3])
                for pos, i in enumerate(perm)
            ]
        )
    return jobs


def schedule_rows(jobs: list[list[Pub]]) -> list[dict[str, Any]]:
    return [
        {
            "job": p.job,
            "position": p.position,
            "arm": p.arm,
            "chunk": p.chunk,
            "instance": p.instance,
            "shots": p.shots,
            "key": p.key,
        }
        for job in jobs
        for p in job
    ]


def check_schedule(
    jobs: list[list[Pub]],
    arms: dict[str, int],
    tier1: Sequence[str],
    max_chunk: int,
    instanced: dict[str, tuple[int, int]] | None = None,
    min_jobs: int = 4,
) -> list[str]:
    """Problems with a schedule (empty: it meets the design requirements)."""
    out = []
    if len(jobs) < min_jobs:
        out.append(f"{len(jobs)} jobs < {min_jobs}")
    for j, job in enumerate(jobs):
        present = {p.arm for p in job}
        for a in tier1:
            if a not in present:
                out.append(f"job {j} lacks tier-1 arm {a}")
        if [p.position for p in job] != list(range(len(job))):
            out.append(f"job {j}: positions not 0..n-1")
    for p in (p for job in jobs for p in job):
        if p.shots > max_chunk:
            out.append(f"{p.key}: {p.shots} > max_chunk {max_chunk}")
    for a, total in arms.items():
        got = sum(p.shots for job in jobs for p in job if p.arm == a)
        if got != total:
            out.append(f"{a}: {got} shots scheduled, {total} planned")
    for a, (n, s) in (instanced or {}).items():
        inst = sorted(p.instance for job in jobs for p in job if p.arm == a)
        if inst != list(range(n)):
            out.append(f"{a}: instances {len(inst)} != {n}")
        if any(p.shots != s for job in jobs for p in job if p.arm == a):
            out.append(f"{a}: shots per instance != {s}")
    return out


# -- cost, budget and submission guards ------------------------------------------------


class Refused(RuntimeError):
    """A guard refused a submission (nothing was sent)."""


@dataclass
class CostModel:
    """Phase 6's cost model plus this design's upper-bound rule."""

    overhead_s: float = 2.0
    rep_delay_s: float = 250e-6
    margin: float = 0.25  # the upper bound adds 25% to the estimate
    billing_round_s: float = 1.0  # IBM bills whole seconds per job (Phase 6: 13.2 s -> 14 s)

    def shot_seconds(self, shots: int, duration_s: float) -> float:
        return shots * (self.rep_delay_s + duration_s)

    def job(self, pubs: Sequence[Pub], durations: dict[str, float]) -> dict[str, float]:
        """Estimate and upper bound of one job (``durations``: arm -> circuit seconds)."""
        t = sum(self.shot_seconds(p.shots, durations[p.arm]) for p in pubs)
        est = self.overhead_s + t
        ub = (1 + self.margin) * est + self.billing_round_s
        return {"shots": sum(p.shots for p in pubs), "shot_s": t, "estimate_s": est, "upper_s": ub}


def budget_check(
    est_s: float,
    upper_s: float,
    max_qpu_seconds: float,
    remaining_s: float | None,
    plan: str | None,
    spent_upper_s: float,
    hard_cap_s: float,
    allowed_plans: Sequence[str] = ("open",),
) -> None:
    """Refuse a job unless every budget condition holds (``Refused`` otherwise).

    * the per-job estimate is at most ``max_qpu_seconds`` (Phase 6's guard);
    * the instance's plan is known and one of ``allowed_plans`` (the free Open plan only);
    * the live remaining allowance is known and at least the job's upper bound;
    * the upper bounds of the jobs already submitted plus this one stay within the hard cap.
    """
    if not est_s <= max_qpu_seconds:
        raise Refused(f"estimate {est_s:.2f} s > max_qpu_seconds {max_qpu_seconds}")
    if plan is None or plan.strip().lower() not in {p.lower() for p in allowed_plans}:
        raise Refused(f"plan {plan!r} is not one of {list(allowed_plans)}: never a paid plan")
    if remaining_s is None:
        raise Refused("remaining allowance unknown: refusing")
    if upper_s > remaining_s:
        raise Refused(f"upper bound {upper_s:.2f} s > remaining allowance {remaining_s} s")
    if spent_upper_s + upper_s > hard_cap_s:
        raise Refused(
            f"cumulative upper bound {spent_upper_s + upper_s:.2f} s > hard cap {hard_cap_s} s"
        )


class UsageExceeded(Refused):
    """A completed job was charged more than its upper bound: stop, report to the owner."""


@dataclass
class Charge:
    """What one completed (or submitted) job counts against the hard cap.

    ``readings`` maps each usage source to a number of QPU seconds (read), ``None`` (read but
    no value yet) or ``{"error": ...}`` (the read failed). ``charged_s`` is the largest
    positive reading (the most conservative source); ``status`` is ``charged``, ``unknown``
    (readable but nothing positive reported) or ``unreadable`` (every source failed).
    ``counted_s = max(charged_s, upper_s)``; an unknown or unreadable charge counts as
    ``upper_s``. ``exceeded``: the charge is above the job's upper bound.
    """

    stage: str
    upper_s: float
    readings: dict[str, Any]
    charged_s: float | None
    status: str
    counted_s: float
    exceeded: bool


def job_charge(stage: str, upper_s: float, readings: dict[str, Any]) -> Charge:
    """Classify one job's usage readings (see ``Charge``)."""
    nums = [
        float(v)
        for v in readings.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    pos = [v for v in nums if v > 0]
    readable = any(not isinstance(v, dict) for v in readings.values())
    if pos:
        charged, status = max(pos), "charged"
    elif readable:
        charged, status = None, "unknown"
    else:
        charged, status = None, "unreadable"
    counted = upper_s if charged is None else max(charged, upper_s)
    exceeded = charged is not None and charged > upper_s
    return Charge(stage, float(upper_s), dict(readings), charged, status, counted, exceeded)


def usage_guard(charges: Sequence[Charge]) -> float:
    """The seconds counted against the hard cap by the jobs already submitted.

    Raises ``UsageExceeded`` if any job was charged more than its upper bound (no further
    submissions), and ``Refused`` if any job's usage could not be read at all.
    """
    over = [c for c in charges if c.exceeded]
    if over:
        raise UsageExceeded(
            "reported usage above the job's upper bound, stop and report: "
            + ", ".join(f"{c.stage} charged {c.charged_s} s > {c.upper_s:.2f} s" for c in over)
        )
    blind = [c.stage for c in charges if c.status == "unreadable"]
    if blind:
        raise Refused(f"usage of {blind} could not be read: refusing the next job")
    return float(sum(c.counted_s for c in charges))


APPROVAL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/issues/\d+#issuecomment-\d+$")


def submission_check(approval_url: str | None, submit_flag: bool) -> None:
    """Refuse unless the config names the owner's approving comment and ``--submit`` is given."""
    if not submit_flag:
        raise Refused("no --submit flag: refusing to submit")
    if not approval_url or not APPROVAL_RE.match(approval_url.strip()):
        raise Refused(f"approval_url {approval_url!r} is not a GitHub issue-comment URL")


def amendment_section(doc: str, heading: str) -> str | None:
    """The text of the section whose heading line is ``heading``, up to the next heading of
    the same or a higher level (``None`` if absent). Trailing whitespace is dropped."""
    lines = doc.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    try:
        i = next(k for k, ln in enumerate(lines) if ln.rstrip() == heading.rstrip())
    except StopIteration:
        return None
    j = len(lines)
    for k in range(i + 1, len(lines)):
        m = re.match(r"^(#+) ", lines[k])  # a heading; "#25's ..." is not one
        if m and len(m.group(1)) <= level:
            j = k
            break
    return "\n".join(ln.rstrip() for ln in lines[i:j]).rstrip() + "\n"


def amendment_hash(section: str) -> str:
    import hashlib

    return hashlib.sha256(section.encode()).hexdigest()


def amendment_check(
    doc: str, heading: str, date: str, sha256: str, committed_doc: str | None
) -> None:
    """Refuse a submission unless the dated amendment is recorded and unchanged (#26).

    * ``heading`` (from the config) names the section and carries ``date``;
    * the section in the working copy ``doc`` hashes to the recorded ``sha256``;
    * ``committed_doc`` (the document at the recorded amendment commit, which must be on
      ``main``) holds the same section, so the amendment was committed as it now reads.
    """
    if not (date and sha256 and heading):
        raise Refused("no amendment recorded in the config: refusing to submit")
    if date not in heading:
        raise Refused(f"the amendment heading {heading!r} does not carry its date {date}")
    sec = amendment_section(doc, heading)
    if sec is None:
        raise Refused(f"amendment section {heading!r} not found in the pre-registration")
    if amendment_hash(sec) != sha256:
        raise Refused("the amendment section differs from its recorded hash")
    if committed_doc is None:
        raise Refused("the amendment commit is not on main (or not found)")
    old = amendment_section(committed_doc, heading)
    if old is None or amendment_hash(old) != sha256:
        raise Refused("the amendment at its recorded commit differs from its recorded hash")


@dataclass
class CalibrationBounds:
    """Acceptability of the live calibration against a reference snapshot.

    Ratios ``live / reference`` must lie in ``[1 / factor, factor]`` for T1 and T2, and be at
    most ``factor`` for error rates (an error may always improve). Absolute ceilings apply
    where there is no reference (q74 is not in #16's snapshot).
    """

    factor: float = 2.0
    abs_readout_max: float = 0.05
    abs_t1_min_s: float = 50e-6
    abs_t2_min_s: float = 30e-6


def calibration_check(
    live: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    bounds: CalibrationBounds,
) -> list[str]:
    """Violations (empty: acceptable). Keys: qubit or pair label -> {quantity: value}.

    Quantities ending in ``_error`` are error rates; ``t1``, ``t2`` are times in seconds.
    """
    out = []
    for q, vals in live.items():
        ref = reference.get(q, {})
        for k, v in vals.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(f"{q}.{k}: missing")
                continue
            r = ref.get(k)
            if k in ("t1", "t2"):
                lo = bounds.abs_t1_min_s if k == "t1" else bounds.abs_t2_min_s
                if v < lo:
                    out.append(f"{q}.{k} = {v:.3g} < {lo:.3g}")
                if r and not (r / bounds.factor <= v <= r * bounds.factor):
                    out.append(f"{q}.{k} = {v:.3g} outside x{bounds.factor} of {r:.3g}")
            elif k.endswith("_error"):
                if k == "readout_error" and v > bounds.abs_readout_max:
                    out.append(f"{q}.{k} = {v:.3g} > {bounds.abs_readout_max}")
                if r and v > r * bounds.factor:
                    out.append(f"{q}.{k} = {v:.3g} > x{bounds.factor} of {r:.3g}")
    return out


# -- analysis ----------------------------------------------------------------------------


@dataclass
class AnalysisSettings:
    """Fixed in advance (``docs/phase7-preregistration.md``)."""

    k_max: int = 7
    burn: int = 16
    offsets: tuple[int, ...] = (0, 1, 2, 3)
    block: int = 500  # circular block bootstrap, shots (#19)
    n_boot: int = 1000
    seed: int = 24
    alpha_segment: float = 0.01
    n_perm: int = 1000
    cp_grid: int = 50
    homog_block: int = 500
    const_block: int = 1000


def _counts(idx: np.ndarray, n: int) -> np.ndarray:
    """``(B, n)`` multiplicities of each shot in each resample."""
    out = np.zeros((len(idx), n))
    for b in range(len(idx)):
        out[b] = np.bincount(idx[b], minlength=n)
    return out


def boot_g(per: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``g`` of each resample (equals ``W.bootstrap_g(per, idx)``)."""
    return _counts(idx, len(per)) @ per / idx.shape[1]


def strata(labels: np.ndarray) -> list[np.ndarray]:
    """Index arrays of each label's shots, labels in order of first appearance."""
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    return [np.nonzero(labels == labels[i])[0] for i in sorted(first)]


def block_size(n: int, b: int) -> int:
    """#19's effective block: ``min(b, n // 4)`` so that a segment has at least 4 blocks."""
    return max(1, min(b, n // 4))


def stratified_indices(segs: Sequence[np.ndarray], b: int, n_boot: int, seed: int) -> np.ndarray:
    """``D.stratified_block_indices`` with each segment's block capped at a quarter of it."""
    out = []
    for s, seg in enumerate(segs):
        bb = block_size(len(seg), b)
        loc = D.block_bootstrap_indices(len(seg), bb, n_boot, seed + 104729 * s)
        out.append(np.asarray(seg)[loc])
    return np.concatenate(out, axis=1)


def lag1_counts(shots: np.ndarray, burn: int) -> np.ndarray:
    """Per shot ``(n10, n11, n00, n01)``: pairs (t, t+1), t >= burn, by previous and current."""
    y = np.asarray(shots)[:, burn:].astype(np.int64)
    prev, cur = y[:, :-1], y[:, 1:]
    return np.stack(
        [
            ((prev == 1) & (cur == 0)).sum(1),
            ((prev == 1) & (cur == 1)).sum(1),
            ((prev == 0) & (cur == 0)).sum(1),
            ((prev == 0) & (cur == 1)).sum(1),
        ],
        1,
    ).astype(np.float64)


def _lag1_from(c: np.ndarray) -> np.ndarray:
    c = np.atleast_2d(c)
    return c[:, 1] / (c[:, 0] + c[:, 1]) - c[:, 3] / (c[:, 2] + c[:, 3])


def _se_ci(est: float, boots: np.ndarray) -> tuple[float, list[float]]:
    se = float(np.std(boots, ddof=1))
    return se, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def analyse_arm(
    shots: np.ndarray, jobs: np.ndarray, pubs: np.ndarray, s: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """Witness statistics of one arm, shots in nominal time order with job and pub labels
    (``pubs``: the pub's position in its job, so each (job, pub) is one chunk).

    Returns w(j), the shot-bootstrap, block-bootstrap and primary SEs (stratified by job),
    the 95% CI of w(0) from the resamples of the method that sets the primary SE, z(0), P(1)
    after the burn-in, the lag-1 excess with its SE, and per-job / per-chunk w(0) and P(1).
    """
    shots = np.asarray(shots)
    per = W.per_shot_windows(shots, s.k_max, s.burn, common=True)
    g = per.mean(0)
    segs = strata(jobs)
    idx_s = stratified_indices(segs, 1, s.n_boot, seed)
    idx_b = stratified_indices(segs, s.block, s.n_boot, seed + 1)
    cnt_s, cnt_b = _counts(idx_s, len(per)), _counts(idx_b, len(per))
    gs, gb = cnt_s @ per / len(per), cnt_b @ per / len(per)
    out: dict[str, Any] = {"n": len(shots), "g": g.tolist()}
    for j in s.offsets:
        w = float(W.witness(g, j))
        ws, wb = W.witness(gs, j), W.witness(gb, j)
        se_s, ci_s = _se_ci(w, ws)
        se_b, ci_b = _se_ci(w, wb)
        out[f"w{j}"] = w
        out[f"se_shot{j}"] = se_s
        out[f"se_block{j}"] = se_b
        out[f"se{j}"] = max(se_s, se_b)
        out[f"z{j}"] = w / out[f"se{j}"] if out[f"se{j}"] > 0 else math.nan
        if j == 0:
            out["ci0"] = ci_b if se_b >= se_s else ci_s
            out["se0_method"] = "block" if se_b >= se_s else "shot"
    body = shots[:, s.burn :]
    out["p1"] = float(body.mean())
    lc = lag1_counts(shots, s.burn)
    out["lag1"] = float(_lag1_from(lc.sum(0))[0])
    ls, lb = _lag1_from(cnt_s @ lc), _lag1_from(cnt_b @ lc)
    out["lag1_se"] = max(float(np.std(ls, ddof=1)), float(np.std(lb, ddof=1)))
    out["per_job"] = [
        {
            "job": int(jobs[seg[0]]),
            "n": len(seg),
            "w0": float(W.witness(per[seg].mean(0), 0)),
            "se0": float(D.block_se(per[seg], 0, 1)),
            "p1": float(body[seg].mean()),
        }
        for seg in segs
    ]
    keys = np.asarray(jobs) * 10**6 + np.asarray(pubs)
    out["per_chunk"] = [
        {
            "job": int(jobs[seg[0]]),
            "pub": int(pubs[seg[0]]),
            "n": len(seg),
            "w0": float(W.witness(per[seg].mean(0), 0)),
            "p1": float(body[seg].mean()),
        }
        for seg in strata(keys)
    ]
    return out


def segment_policy(
    shots: np.ndarray, jobs: np.ndarray, s: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """#19's per-job tests at ``alpha_segment``: change-point scans on g(1) - g(3), P(1) and
    the w(0) influence; chi^2 homogeneity of g over ``homog_block``-shot blocks; constancy of
    w(0) over ``const_block``-shot blocks. Returns per-job results and the flagged jobs."""
    shots = np.asarray(shots)
    per = W.per_shot_windows(shots, s.k_max, s.burn, common=True)
    p_shot = shots[:, s.burn :].mean(1)
    res, flagged = [], []
    for k, seg in enumerate(strata(jobs)):
        x = per[seg]
        stats = {
            "d13": x[:, 1] - x[:, 3],
            "p1": p_shot[seg],
            "w0_influence": D.influence(x, 0),
        }
        r: dict[str, Any] = {"job": int(jobs[seg[0]]), "n": len(seg)}
        ps = []
        for name, v in stats.items():
            cp = D.changepoint_test(v, s.cp_grid, s.n_perm, seed + 31 * k)
            r[f"cp_{name}_p"] = cp["p"]
            ps.append(cp["p"])
        mask = np.ones(len(seg), dtype=bool)
        hb = D.blocks_within(mask, s.homog_block)
        if len(hb) >= 2:
            r["homogeneity_p"] = D.homogeneity_chi2(x, hb)["p_joint"]
            ps.append(r["homogeneity_p"])
        cb = D.blocks_within(mask, s.const_block)
        if len(cb) >= 2:
            r["constancy_p"] = D.block_witness_constancy(x, cb, 0)["p"]
            ps.append(r["constancy_p"])
        r["flag"] = bool(min(ps) < s.alpha_segment)
        if r["flag"]:
            flagged.append(r["job"])
        res.append(r)
    return {"jobs": res, "flagged": flagged}


def guard(shots: np.ndarray, marginals: np.ndarray | None, s: AnalysisSettings, seed: int):
    """#23's guard rails on an arm's shots (nominal time order); integrity failures separated."""
    g = AR.guard_rails(shots, s.k_max, s.burn, seed=seed, marginals=marginals)
    # Only malformed values invalidate a dataset. Duplicate shots are a warning to inspect: a
    # re-read buffer or resubmitted chunk duplicates whole pubs, but leaked q59 also produces
    # repeated all-ones shots (the validation saw both).
    g["integrity_fail"] = [f for f in g["flags"] if f == "values"]
    g["decoding_warn"] = [f for f in g["flags"] if f in ("positional_chi2", "duplicates")]
    g["drift_flags"] = [f for f in g["flags"] if f.startswith(("dispersion", "order"))]
    for k in ("block_p1", "block_repeat", "block_g1_minus_g3"):
        g.pop(k, None)
    return g


# -- decision rule ----------------------------------------------------------------------------


@dataclass
class DecisionRule:
    """The pre-specified thresholds (``docs/phase7-preregistration.md`` section 5)."""

    gate_z: float = 5.0
    vanish_fraction: float = 0.10
    vanish_quantile: float = 1.959963984540054  # two-sided 95%: upper end of |w| + 1.96 SE
    persist_z: float = 5.0
    persist_fraction: float = 0.25
    family_alpha: float = 0.01


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_sided_p(z: float) -> float:
    return 2 * norm_sf(abs(z)) if math.isfinite(z) else math.nan


def holm(pvals: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm's step-down procedure: which hypotheses are rejected at family-wise ``alpha``."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, stop = len(items), {}, False
    for i, (k, p) in enumerate(items):
        if not stop and p <= alpha / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out


def primary_outcome(base: dict[str, float], a2: dict[str, float], rule: DecisionRule) -> str:
    """``vanishes``, ``persists`` or ``ambiguous`` for A2 against the concurrent baseline."""
    wb, w, se = abs(base["w0"]), abs(a2["w0"]), a2["se0"]
    if w + rule.vanish_quantile * se < rule.vanish_fraction * wb:
        return "vanishes"
    if w >= rule.persist_z * se and w >= rule.persist_fraction * wb:
        return "persists"
    return "ambiguous"


def gate(base: dict[str, float], rule: DecisionRule) -> bool:
    return bool(base["w0"] < 0 and abs(base["w0"]) >= rule.gate_z * base["se0"])


@dataclass
class Secondary:
    """One secondary contrast: ``arm``'s statistic against ``ref`` (an arm or a number)."""

    name: str
    arm: str
    stat: str = "w0"  # "w0" or "lag1"
    ref_arm: str | None = None
    ref_value: float = 0.0


DEFAULT_SECONDARY = (
    Secondary("A2s_spec_vs_baseline", "A2s_spec", "w0", ref_arm="baseline"),
    Secondary("A1_vs_0", "A1", "w0"),
    Secondary("A3_vs_prediction", "A3", "w0"),
    Secondary("D1_w0_vs_0", "D1", "w0"),
    Secondary("D1_lag1_vs_prediction", "D1", "lag1"),
)


def secondary_tests(
    arms: dict[str, dict[str, Any]],
    tests: Sequence[Secondary],
    rule: DecisionRule,
    ref_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Two-sided z-tests of each contrast, Holm at ``family_alpha``; effect sizes with 95% CIs.

    ``ref_values`` overrides a test's ``ref_value`` (the pre-registered H_q75 predictions).
    Arms absent from ``arms`` are skipped (not counted in the family).
    """
    rows, pvals = {}, {}
    for t in tests:
        if t.arm not in arms or (t.ref_arm and t.ref_arm not in arms):
            continue
        a = arms[t.arm]
        se_key = "se0" if t.stat == "w0" else "lag1_se"
        est, se = a[t.stat], a[se_key]
        if t.ref_arm:
            r = arms[t.ref_arm]
            ref, ref_se = r[t.stat], r[se_key]
        else:
            ref, ref_se = (ref_values or {}).get(t.name, t.ref_value), 0.0
        d = est - ref
        se_d = math.sqrt(se**2 + ref_se**2)
        z = d / se_d if se_d > 0 else math.nan
        row = {
            "arm": t.arm,
            "stat": t.stat,
            "estimate": est,
            "reference": ref,
            "difference": d,
            "se": se_d,
            "ci95": [d - 1.959963984540054 * se_d, d + 1.959963984540054 * se_d],
            "z": z,
            "p": two_sided_p(z),
        }
        if t.ref_arm and t.stat == "w0" and ref != 0:
            ratio = est / ref
            # delta-method SE of the ratio
            row["ratio"] = ratio
            row["ratio_se"] = abs(ratio) * math.sqrt((se / est) ** 2 + (ref_se / ref) ** 2)
        rows[t.name] = row
        pvals[t.name] = row["p"]
    rej = holm(pvals, rule.family_alpha)
    for k in rows:
        rows[k]["reject_holm"] = rej[k]
    return rows


@dataclass
class Decision:
    gate: bool
    primary: str
    primary_all_jobs: str
    primary_unflagged: str | None
    flagged_jobs: list[int]
    ratio_a2: float
    secondary: dict[str, Any] = field(default_factory=dict)
    descriptive: dict[str, str] = field(default_factory=dict)  # A1, A3 under A2's thresholds

    @property
    def reported(self) -> str:
        """The pre-registered headline: uninterpretable without the gate."""
        return self.primary if self.gate else "uninterpretable"


def decide(
    arms: dict[str, dict[str, Any]],
    rule: DecisionRule,
    unflagged: dict[str, dict[str, Any]] | None = None,
    flagged_jobs: Sequence[int] = (),
    tests: Sequence[Secondary] = DEFAULT_SECONDARY,
    ref_values: dict[str, float] | None = None,
) -> Decision:
    """The full rule. ``unflagged``: baseline and A2 re-analysed without the jobs the segment
    policy or the drift guard rails flagged; if that changes the primary outcome, it is
    reported as ``ambiguous`` (segment-sensitive)."""
    base, a2 = arms["baseline"], arms["A2"]
    p_all = primary_outcome(base, a2, rule)
    p_unf = None
    final = p_all
    if unflagged is not None and flagged_jobs:
        p_unf = primary_outcome(unflagged["baseline"], unflagged["A2"], rule)
        if p_unf != p_all:
            final = "ambiguous"
    return Decision(
        gate=gate(base, rule),
        primary=final,
        primary_all_jobs=p_all,
        primary_unflagged=p_unf,
        flagged_jobs=list(flagged_jobs),
        ratio_a2=a2["w0"] / base["w0"] if base["w0"] else math.nan,
        secondary=secondary_tests(arms, tests, rule, ref_values),
        descriptive={a: primary_outcome(base, arms[a], rule) for a in ("A1", "A3") if a in arms},
    )


def expected_outcome(
    w_base: float, w_a2: float, se_base: float, se_a2: float, rule: DecisionRule
) -> str:
    """The rule applied to exact values with the SEs expected at the planned shots."""
    return primary_outcome({"w0": w_base, "se0": se_base}, {"w0": w_a2, "se0": se_a2}, rule)


__all__ = [
    "AnalysisSettings",
    "CalibrationBounds",
    "Charge",
    "CostModel",
    "Decision",
    "DecisionRule",
    "Pub",
    "Refused",
    "Secondary",
    "UsageExceeded",
    "amendment_check",
    "amendment_hash",
    "amendment_section",
    "analyse_arm",
    "boot_g",
    "budget_check",
    "calibration_check",
    "check_schedule",
    "decide",
    "expected_outcome",
    "gate",
    "guard",
    "holm",
    "job_charge",
    "make_schedule",
    "primary_outcome",
    "schedule_rows",
    "secondary_tests",
    "segment_policy",
    "split_even",
    "submission_check",
    "usage_guard",
]
