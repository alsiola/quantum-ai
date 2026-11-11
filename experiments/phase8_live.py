"""Phase 8 Stage 5 (issue #37): the live path of ``experiments/phase8_hw.py``.

``phase8_hw.py --dry-run`` (live target, nothing submitted) and ``--submit`` call this module.
It is tier-generic: a tier needs a ``[<tier>.live]`` config table and an entry in ``TIERS``
(its stages, circuits, decoding and pre-registered analysis). Tier 1 has two: ``tier1`` (P&M,
#33, run on #37) and its MDI sub-tier ``tier1_mdi`` (#38, amendment A4), whose table is
``[tier1.mdi_live]``. A sub-tier runs on its base tier's approval, counts against the base
tier's hard cap together with the base tier's committed ledgers (``prior_ledgers``), and
refuses unless its ``precondition`` (a committed decision) holds.

Guards, in order. Each refuses (``qmem.followup.Refused``) before anything is submitted:

1. **Approval** (``qmem.prereg.Approval``): ``[approval] <tier>`` must be an issue-comment URL,
   ``--submit`` must be given, and the approval is frozen into the run directory
   (``freeze_approval``). This check comes before any network call.
2. **Pre-registration** (``prereg_guard``): each ``[[<tier>.live.prereg]]`` section of the
   pre-registration must hash (``F.amendment_section``/``F.amendment_hash``) to its recorded
   value, both in the working copy and at its recorded commit, and that commit must be on
   ``origin/main``. This is Phase 7's amendment rule (``F.amendment_check``), applied to whole
   sections, whose headings carry no date. Each dated amendment named must be present.
3. **Circuits** (``circuit_check``): every committed QASM file hashes to its stage manifest
   and to ``phase8_prereg/hashes.csv``, and the tier's digest of sorted ``name sha256``
   lines matches the config. A level-0, fixed-layout transpile on the live target must leave
   each file byte for byte unchanged and touch only ``[<tier>.live] qubits``.
4. **Live re-selection** (section 5): the selection rules re-run on the live calibration
   snapshot, which is saved. A change in one of the tier's ``gating_selections`` refuses.
5. **Admission and per-job budget** (``admission``, ``job_guard``): the cumulative hard caps
   through this tier must fit in remaining − reserve (``P.budget_table``). Per job,
   ``P.Ledger.guard`` (that is, ``F.budget_check``) needs: the estimate ≤ the per-job limit,
   the plan in ``allowed_plans`` (the free Open plan only), a known allowance, the upper bound
   ≤ the allowance less the reserve, and spend plus the upper bound ≤ the cap. The allowance
   is the owner's reading less what is already counted, capped by the live reading when IBM
   reports one.

Jobs run through ``qmem.prereg.run_stage`` on ``LiveBackend``. An id on disk is re-fetched,
never resubmitted, and the id is saved before waiting. Raw shots are kept. Each job counts
max(charged, upper bound) (``F.job_charge``), and a charge above the upper bound stops the run.

``qiskit`` and ``qiskit_ibm_runtime`` are imported inside functions only, so the main check
and ``--dry-run-local`` never need the ``hardware`` group. Credentials come from ``[ibm]
env_file`` (``.env``) only, never from the environment.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qmem import followup as F
from qmem import prereg as P
from qmem import reference as R
from qmem.records import REPO_ROOT, run_metadata, save_json

# -- config helpers --------------------------------------------------------------------------------


def base_tier(tier: str) -> str:
    """The tier whose approval, cap and config a (sub-)tier uses: ``tier1_mdi`` -> ``tier1``."""
    return TIERS.get(tier, {}).get("base", tier)


def live_table(tier: str) -> str:
    return TIERS.get(tier, {}).get("table", "live")


def live_cfg(cfg: dict[str, Any], tier: str) -> dict[str, Any]:
    b, t = base_tier(tier), live_table(tier)
    try:
        return cfg[b][t]
    except KeyError:
        raise F.Refused(f"{tier} has no [{b}.{t}] table: no live path for it") from None


def prior_counted(cfg: dict[str, Any], tier: str) -> float:
    """Seconds the base tier has already counted in committed ledgers (``prior_ledgers``)."""
    total = 0.0
    for path in live_cfg(cfg, tier).get("prior_ledgers", []):
        led = json.loads((REPO_ROOT / path).read_text())
        total += float(F.usage_guard([F.Charge(**c) for c in led["charges"]]))
    return total


def read_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of the credentials file (Phase 6's format)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jsonable(x: Any) -> Any:
    """Best-effort conversion of Qiskit/IBM objects to JSON."""
    import datetime as _dt

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, _dt.datetime):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None or isinstance(x, bool | int | float | str):
        return x
    return repr(x)


# -- 1. approval -----------------------------------------------------------------------------------


def approval_guard(cfg: dict[str, Any], tier: str, submit: bool, out: Path | None) -> P.Approval:
    """``P.Approval.check`` for this tier (refuses a missing URL, another tier's approval, or no
    ``--submit``), then ``P.freeze_approval`` into the run directory."""
    b = base_tier(tier)
    appr = P.Approval(b, (cfg.get("approval") or {}).get(b) or None)
    appr.check(b, submit)
    return appr if out is None else P.freeze_approval(out, appr)


# -- 2. the pre-registration's sections ------------------------------------------------------------


def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)


def committed_doc(doc: str, commit: str) -> str | None:
    """``doc`` at ``commit`` if that commit is on ``origin/main`` (as last fetched), else None."""
    if not commit or _git("merge-base", "--is-ancestor", commit, "origin/main").returncode != 0:
        return None
    shown = _git("show", f"{commit}:{doc}")
    return shown.stdout if shown.returncode == 0 else None


def prereg_guard(
    cfg: dict[str, Any], tier: str, committed: Callable[[str, str], str | None] = committed_doc
) -> list[dict[str, Any]]:
    """Refuse unless every ``[[<tier>.live.prereg]]`` section is unchanged (working copy and its
    commit on main) and carries the dated amendments it names."""
    secs = live_cfg(cfg, tier).get("prereg") or []
    if not secs:
        raise F.Refused(f"no pre-registration sections recorded for {tier}: refusing")
    rows = []
    for s in secs:
        text = (REPO_ROOT / s["doc"]).read_text()
        sec = F.amendment_section(text, s["heading"])
        if sec is None:
            raise F.Refused(f"section {s['heading']!r} not found in {s['doc']}")
        if F.amendment_hash(sec) != s["sha256"]:
            raise F.Refused(f"section {s['heading']!r} differs from its recorded hash")
        old_doc = committed(s["doc"], s.get("commit", ""))
        if old_doc is None:
            raise F.Refused(f"the commit of {s['heading']!r} is not on main (or not found)")
        old = F.amendment_section(old_doc, s["heading"])
        if old is None or F.amendment_hash(old) != s["sha256"]:
            raise F.Refused(f"section {s['heading']!r} at its commit differs from its hash")
        for a in s.get("amendments", []):
            if a not in sec:
                raise F.Refused(f"amendment {a!r} not in {s['heading']!r}")
        rows.append({"heading": s["heading"], "sha256": s["sha256"], "commit": s["commit"]})
    return rows


# -- 3. circuits -----------------------------------------------------------------------------------


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def name_digest(rows: Sequence[tuple[str, str]]) -> str:
    """SHA-256 of the sorted ``name sha256`` lines, newline-terminated (#28's digest)."""
    return sha_text("".join(f"{ln}\n" for ln in sorted(f"{n} {h}" for n, h in rows)))


def prereg_hashes() -> dict[str, str]:
    path = REPO_ROOT / "results/summaries/phase8_prereg/hashes.csv"
    return {r["circuit"]: r["sha256"] for r in csv.DictReader(open(path))}


def circuit_check(cfg: dict[str, Any], tier: str, names: Sequence[str]) -> dict[str, Any]:
    """The tier's committed QASM texts by name, after the hash checks (no Qiskit): each file
    against its manifest and ``hashes.csv``; the digest of the tier's circuits against
    ``[<tier>.live] qasm_digest``. Refuses on any mismatch."""
    lc = live_cfg(cfg, tier)
    qd = REPO_ROOT / lc["qasm_dir"]
    man = json.loads((qd / "manifest.json").read_text())["circuits"]
    pre = prereg_hashes()
    tier_rows = sorted((n, h) for n, h in pre.items() if n.startswith(lc["circuit_prefix"]))
    if name_digest(tier_rows) != lc["qasm_digest"]:
        raise F.Refused(
            f"{tier}: the digest of hashes.csv's circuits differs from the approved one"
        )
    texts, bad = {}, []
    for n in names:
        f = qd / f"{n}.qasm"
        text = f.read_text() if f.exists() else ""
        h = sha_text(text)
        if not text or h != man.get(n, {}).get("sha256") or h != pre.get(n):
            bad.append(n)
        texts[n] = text
    if bad:
        raise F.Refused(f"{tier}: circuit hash mismatch: {bad}")
    if set(names) != {n for n, _ in tier_rows}:
        raise F.Refused(f"{tier}: the job's circuits are not exactly the approved set")
    return {"texts": texts, "digest": lc["qasm_digest"], "files": len(texts)}


def isa_check(texts: dict[str, str], backend: Any, qubits: Sequence[int]) -> dict[str, Any]:
    """Load each QASM, transpile at level 0 with a fixed trivial layout on the live ``backend``;
    refuse unless the result dumps to the committed text exactly and acts on ``qubits`` only.
    Returns the circuits and their durations on the live target (hardware group); ``None``
    for a circuit with control flow, which has no static duration."""
    from qiskit import qasm3, transpile

    circuits, dur, changed, stray = {}, {}, [], []
    for n, text in texts.items():
        qc = qasm3.loads(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tq = transpile(
                qc,
                backend,
                optimization_level=0,
                seed_transpiler=0,
                initial_layout=list(range(qc.num_qubits)),
            )
        if qasm3.dumps(tq) != text:
            changed.append(n)
        used = {
            tq.find_bit(q).index for i in tq.data if i.operation.name != "barrier" for q in i.qubits
        }
        if not used <= set(qubits):
            stray.append((n, sorted(used - set(qubits))))
        circuits[n] = tq
        try:
            dur[n] = float(tq.estimate_duration(backend.target, unit="s"))
        except Exception:  # control flow (if_else) has no static duration: the tier supplies one
            dur[n] = None
    if changed or stray:
        raise F.Refused(f"the live target changes the circuits: {changed}; stray qubits {stray}")
    return {"circuits": circuits, "duration_s": dur}


# -- 4. the live calibration snapshot --------------------------------------------------------------


def snapshot(backend: Any) -> dict[str, Any]:
    """The live calibration in ``phase7_spectator/device.json``'s shape (string qubit keys)."""
    t = backend.target
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    qubits = {}
    for q in range(backend.num_qubits):
        qp = t.qubit_properties[q]

        def err(op: str, q=q) -> float | None:
            p = t[op].get((q,)) if op in t else None
            return None if p is None else p.error

        def dur(op: str, q=q) -> float | None:
            p = t[op].get((q,)) if op in t else None
            return None if p is None else p.duration

        qubits[str(q)] = {
            "t1": qp.t1,
            "t2": qp.t2,
            "readout": err("measure"),
            "sx": err("sx"),
            "measure_time": dur("measure"),
            "reset_time": dur("reset"),
        }
    cz = {}
    for a, c in edges:
        p = t["cz"].get((a, c)) or t["cz"].get((c, a))
        cz[f"{a}-{c}"] = {
            "error": None if p is None else p.error,
            "duration": None if p is None else p.duration,
        }
    try:
        date = str(backend.properties(refresh=True).last_update_date)
    except Exception as e:
        date = f"unavailable: {e!r}"
    return jsonable(
        {
            "backend": backend.name,
            "calibration_date": date,
            "read_at": now(),
            "num_qubits": backend.num_qubits,
            "dt": t.dt,
            "granularity": t.granularity,
            "operations": sorted(t.operation_names),
            "edges": [list(e) for e in edges],
            "cz": cz,
            "qubits": qubits,
        }
    )


def usable_snapshot(dev: dict[str, Any]) -> dict[str, Any]:
    """Qubits and couplers with every field a selection reads (a faulty qubit lacks some); the
    selection rules then simply never see the unusable ones."""
    need = ("t1", "t2", "readout", "sx")
    qs = {k: v for k, v in dev["qubits"].items() if all(v.get(f) is not None for f in need)}
    cz = {k: v for k, v in dev["cz"].items() if v.get("error") is not None}
    edges = [
        e for e in dev["edges"] if str(e[0]) in qs and str(e[1]) in qs and f"{e[0]}-{e[1]}" in cz
    ]
    return {**dev, "qubits": qs, "cz": cz, "edges": edges}


# -- 5. admission and the per-job guard ------------------------------------------------------------


def remaining_allowance(
    cfg: dict[str, Any], ledger: P.Ledger, usage: dict[str, Any] | None
) -> float | None:
    """The owner's reading less what is already counted (earlier Phase 8 jobs and this run),
    capped by IBM's live remaining seconds when reported. ``None``: unknown, so refuse."""
    al = cfg["allowance"]
    if al.get("remaining_s") is None:
        return None
    rem = float(al["remaining_s"]) - float(al.get("prior_counted_s", 0.0)) - ledger.counted()
    live = (usage or {}).get("usage_remaining_seconds")
    if isinstance(live, int | float) and not isinstance(live, bool):
        rem = min(rem, float(live))
    return rem


def admission(
    cfg: dict[str, Any], tier: str, budgets: dict[str, P.TierBudget], remaining: float | None
) -> dict[str, Any]:
    """Section 7: the tier runs only if the cumulative hard caps through it fit in remaining −
    reserve (``P.budget_table``, tiers in ``[order]``)."""
    order = cfg["order"]["tiers"]
    rows = P.budget_table(
        [budgets[t] for t in order[: order.index(base_tier(tier)) + 1]],
        remaining,
        cfg["allowance"]["reserve_s"],
    )
    row = rows[-1]
    if not row["fits_cap"]:
        raise F.Refused(
            f"{tier} not admitted: cumulative caps {row['cumulative_cap_s']:.1f} s vs "
            f"remaining {remaining} s less the {cfg['allowance']['reserve_s']} s reserve"
        )
    return row


def earlier_counted(cfg: dict[str, Any], tier: str) -> dict[str, Any]:
    """The earlier tiers' counted seconds from their committed ledgers (``[<tier>.live]
    earlier_ledgers``). They are already out of the allowance through ``[allowance]
    prior_counted_s``; refuse if the config counts less than the ledgers do."""
    rows = {}
    for path in live_cfg(cfg, tier)["earlier_ledgers"]:
        led = json.loads((REPO_ROOT / path).read_text())
        rows[path] = float(F.usage_guard([F.Charge(**c) for c in led["charges"]]))
    total = sum(rows.values())
    prior = float(cfg["allowance"].get("prior_counted_s", 0.0))
    if total > prior + 1e-9:
        raise F.Refused(f"{tier}: earlier tiers counted {total:.2f} s > prior_counted_s {prior}")
    return {"ledgers": rows, "counted_s": total, "prior_counted_s": prior}


def admission_after(
    cfg: dict[str, Any], tier: str, budget: P.TierBudget, remaining: float | None
) -> dict[str, Any]:
    """Section 7's admission for a tier after finished ones: the earlier tiers count what their
    ledgers counted (already subtracted from ``remaining``), so this tier's whole hard cap must
    fit in remaining - reserve."""
    row = P.budget_table([budget], remaining, cfg["allowance"]["reserve_s"])[-1]
    if not row["fits_cap"]:
        raise F.Refused(
            f"{tier} not admitted: its cap {row['hard_cap_s']:.1f} s vs remaining {remaining} s "
            f"less the {cfg['allowance']['reserve_s']} s reserve"
        )
    return row


def cz_check(dev: dict[str, Any], qubits: Sequence[int]) -> dict[str, Any]:
    """#39: the CZ couplers among the tier's qubits, with their live durations. Refuses if one
    has no positive duration (the live cost would then miss the gate's time)."""
    qs = set(qubits)
    out = {k: v.get("duration") for k, v in dev["cz"].items() if set(map(int, k.split("-"))) <= qs}
    bad = {k: d for k, d in out.items() if d is None or not d > 0}
    if bad:
        raise F.Refused(f"CZ couplers without a positive live duration: {bad}")
    return {k: {"duration_s": d, "duration_ns": round(d * 1e9, 3)} for k, d in out.items()}


def plan_of(svc: Any) -> str | None:
    """The active instance's plan name (``None`` if it can't be read)."""
    try:
        crn = svc.active_instance()
        return next((i.get("plan") for i in svc.instances() if i.get("crn") == crn), None)
    except Exception:
        return None


def read_usage(svc: Any) -> dict[str, Any]:
    try:
        u = dict(svc.usage())
    except Exception as e:
        return {"error": repr(e)}
    u.pop("instance_id", None)  # an account identifier; not needed in the records
    return jsonable(u)


def job_guard(
    cfg: dict[str, Any],
    tier: str,
    tbud: P.TierBudget,
    ledger: P.Ledger,
    svc: Any,
    cost: dict[str, float],
) -> Callable[[dict[str, Any]], None]:
    """The guard ``P.run_stage`` calls before a submission: the plan, the live usage, then
    ``P.Ledger.guard`` (``F.budget_check``) with this tier's spend, cap, per-job limit and
    reserve."""

    def g(rec: dict[str, Any]) -> None:
        plan = plan_of(svc)
        u = read_usage(svc)
        rem = remaining_allowance(cfg, ledger, u)
        rec.update({"plan": plan, "usage_before": u, "remaining_s": rem, "guarded_at": now()})
        rec["approval_url"] = cfg["approval"][base_tier(tier)]
        ledger.guard(
            tbud, cost, rem, plan, cfg["allowance"]["reserve_s"], cfg["allowance"]["allowed_plans"]
        )

    return g


# -- the live backend ------------------------------------------------------------------------------


def make_sampler(cfg: dict[str, Any], backend: Any) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    s = SamplerV2(mode=backend)
    o = cfg["sampler"]
    s.options.dynamical_decoupling.enable = o["dynamical_decoupling"]
    s.options.twirling.enable_gates = o["twirling_gates"]
    s.options.twirling.enable_measure = o["twirling_measure"]
    return s


def bits_of(data: Any, reg: str) -> np.ndarray:
    """``(shots, width)`` uint8 of one classical register of a SamplerV2 pub result."""
    return np.asarray(getattr(data, reg).to_bool_array(order="little"), dtype=np.uint8)


class LiveBackend:
    """``qmem.prereg.Backend`` on IBM Runtime: SamplerV2 in job mode, one job per stage.

    ``circuit(pub)`` gives the ISA circuit of a pub, and ``decode(pub, data)`` its raw arrays.
    Set ``pubs`` to the stage's pubs before ``P.run_stage``. Each job's usage before the
    submission and its result metadata are kept in ``<out>/live_jobs.json``."""

    def __init__(
        self,
        cfg: dict[str, Any],
        svc: Any,
        backend: Any,
        circuit: Callable[[F.Pub], Any],
        decode: Callable[[F.Pub, Any], dict[str, np.ndarray]],
        out: Path,
        sampler: Callable[[Any], Any] | None = None,
    ):
        self.cfg, self.svc, self.backend = cfg, svc, backend
        self.name = backend.name
        self.circuit, self.decode, self.out = circuit, decode, Path(out)
        self.sampler = sampler or (lambda b: make_sampler(cfg, b))
        self.pubs: list[F.Pub] = []
        self.submissions = 0

    def _jobs(self) -> dict[str, Any]:
        p = self.out / "live_jobs.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, jid: str, **kw: Any) -> None:
        jobs = self._jobs()
        jobs.setdefault(jid, {}).update(jsonable(kw))
        save_json(self.out / "live_jobs.json", jobs)

    def submit(self, stage: str, pubs: Sequence[F.Pub]) -> str:
        before = read_usage(self.svc)
        job = self.sampler(self.backend).run([(self.circuit(p), None, p.shots) for p in pubs])
        self.submissions += 1
        jid = job.job_id()
        self._save(jid, stage=stage, pubs=[p.key for p in pubs], usage_before=before)
        return jid

    def result(self, job_id: str) -> dict[str, np.ndarray]:
        job = self.svc.job(job_id)
        res = job.result()
        if len(res) != len(self.pubs):
            raise RuntimeError(f"job {job_id}: {len(res)} pub results for {len(self.pubs)} pubs")
        meta: dict[str, Any] = {"fetched_at": now()}
        md = dict(getattr(res, "metadata", {}) or {})
        meta["execution_spans"] = repr((md.get("execution") or {}).get("execution_spans"))
        try:
            spans = md["execution"]["execution_spans"]
            meta["execution_seconds"] = float(
                sum((s.stop - s.start).total_seconds() for s in spans)
            )
        except Exception:
            meta["execution_seconds"] = None
        meta["result_metadata"] = {k: v for k, v in md.items() if k != "execution"}
        meta["pub_metadata"] = [dict(getattr(r, "metadata", {}) or {}) for r in res]
        for attr in ("status", "metrics"):
            try:
                v = getattr(job, attr)()
                meta[attr] = v if not hasattr(v, "name") else v.name
            except Exception as e:
                meta[f"{attr}_error"] = repr(e)
        self._save(job_id, **meta)
        arrays: dict[str, np.ndarray] = {}
        for p, r in zip(self.pubs, res, strict=True):
            arrays.update(self.decode(p, r.data))
        return arrays

    def usage(self, job_id: str) -> dict[str, Any]:
        """Phase 7's three readings: the job's ``usage()``, its metrics' QPU charge, and the
        instance's consumed-seconds delta across the job (``{"error"}`` when unreadable)."""
        out: dict[str, Any] = {}
        try:
            job = self.svc.job(job_id)
        except Exception as e:
            err = {"error": repr(e)}
            return {"job_usage": err, "job_metrics": err, "account_delta": err}
        try:
            v = job.usage()
            out["job_usage"] = None if v is None else float(v)
        except Exception as e:
            out["job_usage"] = {"error": repr(e)}
        try:
            m = job.metrics() or {}
            v = (m.get("usage") or {}).get("qpu_charge_time_seconds")
            out["job_metrics"] = None if v is None else float(v)
        except Exception as e:
            out["job_metrics"] = {"error": repr(e)}
        try:
            before = self._jobs()[job_id]["usage_before"]["usage_consumed_seconds"]
            after = read_usage(self.svc)["usage_consumed_seconds"]
            out["account_delta"] = float(after) - float(before)
        except Exception as e:
            out["account_delta"] = {"error": f"account delta unavailable: {e!r}"}
        self._save(job_id, usage_readings=out, usage_read_at=now())
        return out


# -- the tiers -------------------------------------------------------------------------------------


@dataclasses.dataclass
class LiveStage:
    stage: str
    pubs: list[F.Pub]
    names: dict[str, str]  # pub key -> circuit name (QASM file stem)
    cost: dict[str, float]  # the pre-registered cost of the job


def tier1_circuit_name(arm: str, chunk: int) -> str:
    """#33's export names. P&M: circuit ``c = 2 k + b``, k over ``R.AXIS_LABELS``. MDI:
    ``c = 4 x + y`` over the tetrahedral inputs (``mdi_n{n}_x{x}_y{y}``)."""
    if arm.startswith("mdi_"):
        x, y = divmod(chunk, 4)
        return f"{arm}_x{x}_y{y}"
    k, b = divmod(chunk, 2)
    ax = R.AXIS_LABELS[k].replace("+", "p").replace("-", "m")
    return f"{arm}_{ax}_b{b}"


def tier1_stages(cfg: dict[str, Any], tier: str = "tier1") -> list[LiveStage]:
    """#33's plan (``phase8_reference/summary.json``): one job per stage, pubs in its seeded
    order, with its committed cost."""
    plan = json.loads((REPO_ROOT / "results/summaries/phase8_reference/summary.json").read_text())[
        "plan"
    ]
    out = []
    for st in live_cfg(cfg, tier)["stages"]:
        if len(plan[st]["jobs"]) != 1:
            raise F.Refused(f"tier1 {st}: expected one job, found {len(plan[st]['jobs'])}")
        pubs = [
            F.Pub(0, i, q["arm"], q["circuit"], q["shots"])
            for i, q in enumerate(plan[st]["jobs"][0])
        ]
        for p, q in zip(pubs, plan[st]["jobs"][0], strict=True):
            if p.key != q["pub"]:
                raise F.Refused(f"pub key {p.key} != the plan's {q['pub']}")
        names = {p.key: tier1_circuit_name(p.arm, p.chunk) for p in pubs}
        out.append(LiveStage(st, pubs, names, plan[st]["costs"][0]))
    return out


def tier1_decode(pub: F.Pub, data: Any) -> dict[str, np.ndarray]:
    """P&M bits as ``phase8_hw.tier1_pm_counts`` reads them: ``(shots, n + 1)`` =
    ``y0..y{n-1}`` then ``m``; the negative control's mid-step ``neg`` is kept separately.
    MDI: ``(shots, n + 2)`` = ``y0..y{n-1}``, then ``m[0]`` (q75) and ``m[1]`` (q74)."""
    from phase8_hw import PM_STEPS

    if pub.arm.startswith("mdi_"):
        n = int(pub.arm.removeprefix("mdi_n"))
        m = bits_of(data, "m")
        cols = [bits_of(data, f"y{t}")[:, 0] for t in range(n)] + [m[:, 0], m[:, 1]]
        return {pub.key: np.stack(cols, 1).astype(np.uint8)}
    n = PM_STEPS[pub.arm]
    cols = [bits_of(data, f"y{t}")[:, 0] for t in range(n)] + [bits_of(data, "m")[:, 0]]
    out = {pub.key: np.stack(cols, 1).astype(np.uint8)}
    if pub.arm == "pm_neg":
        out[f"{pub.key}__neg"] = bits_of(data, "neg")
    return out


def tier1_analyse(cfg: dict[str, Any], stages: list[LiveStage], arrays: dict[str, dict]) -> dict:
    """Section 2's analysis, exactly as pre-registered (``P.tier1_analyse`` with the config's
    ``Tier1Rule``), plus the shot-order warnings (section 2.6, never an invalidation)."""
    import phase8_hw as H

    st = next(s for s in stages if s.stage == "pm_primary")
    arr = arrays["pm_primary"]
    counts = H.tier1_pm_counts(st.pubs, arr)
    res = P.tier1_analyse(counts, H.tier1_rule(cfg))
    res["order_warnings"] = {
        p.key: P.order_warnings(arr[p.key][:, -1])
        for p in st.pubs
        if P.order_warnings(arr[p.key][:, -1])["warn"]
    }
    res["counts"] = {
        a: {"succ": c["succ"].tolist(), "tot": c["tot"].tolist(), "records": c["records"].tolist()}
        for a, c in counts.items()
    }
    res["mdi"] = "excluded (amendment A3: its q74 reset is undecided)"
    return res


# -- Tier 1's MDI sub-tier (#38, amendment A4) -----------------------------------------------------


def mdi_precondition(cfg: dict[str, Any]) -> dict[str, Any]:
    """Section 2.7: the MDI sub-tier runs only if the P&M job passed both gates and rejected EB
    at n = 1. Read from the committed decision (``[tier1.mdi_live] precondition``)."""
    path = live_cfg(cfg, "tier1_mdi")["precondition"]
    d = json.loads((REPO_ROOT / path).read_text())
    ok = {
        "spam_gate": bool(d["spam_gate"]["pass"]),
        "negative_control": not d["negative_control"]["falsely_passes"],
        "valid": bool(d["valid"]),
        "pm_n1_rejects_eb": bool(d["tests"]["pm_n1"]["reject_eb"]),
    }
    if not all(ok.values()):
        raise F.Refused(f"tier1_mdi: P&M's decision does not meet section 2.7: {ok}")
    return {"decision": path, "call": d["call"], **ok}


def tier1_mdi_stages(cfg: dict[str, Any]) -> list[LiveStage]:
    mdi_precondition(cfg)
    return tier1_stages(cfg, "tier1_mdi")


def mdi_hits(st: LiveStage, arr: dict[str, np.ndarray], arm: str) -> tuple[np.ndarray, int]:
    """Phi+ counts (both final bits 0) per input pair ``[x, y]`` of ``arm``, and the shots per
    pair (equal by design; refuses otherwise)."""
    hits, shots = np.zeros((4, 4), dtype=np.int64), set()
    for p in st.pubs:
        if p.arm != arm:
            continue
        x, y = divmod(p.chunk, 4)
        b = arr[p.key]
        hits[x, y] = int(((b[:, -2] == 0) & (b[:, -1] == 0)).sum())
        shots.add(int(b.shape[0]))
    if len(shots) != 1:
        raise ValueError(f"{arm}: unequal shots per input pair {sorted(shots)}")
    return hits, shots.pop()


def tier1_mdi_analyse(cfg: dict[str, Any], stages: list[LiveStage], arrays: dict) -> dict:
    """Amendment A4's analysis (#33 section 3, docs/phase8-preregistration.md section 10): the
    payoff at n = 1 with #33's fixed V_1 in the weights, EB rejected if payoff - z SE > 0
    (one-sided ``[tier1] mdi_alpha``, a family of one: it is not in P&M's Holm families and
    cannot change P&M's call). n = 0 is reported as the positive control, not a gate. The
    shot-order warnings are reported (section 2.6, never an invalidation)."""
    v = {
        int(n): np.array(m["re"]) + 1j * np.array(m["im"])
        for n, m in json.loads(
            (REPO_ROOT / "results/summaries/phase8_reference/summary.json").read_text()
        )["targets_V"].items()
    }
    st = next(s for s in stages if s.stage == "mdi_game")
    arr = arrays["mdi_game"]
    a = cfg["tier1"]["mdi_alpha"]
    res: dict[str, Any] = {"alpha": a, "precondition": mdi_precondition(cfg)}
    for n in (0, 1):
        hits, shots = mdi_hits(st, arr, f"mdi_n{n}")
        r = P.mdi_analyse(hits, shots, R.mdi_weights(v[n]), a)
        r["upper_bound"] = r["payoff"] + (r["payoff"] - r["lower_bound"])
        res[f"n{n}"] = r | {"hits": hits.tolist(), "shots_per_pair": shots}
    res["order_warnings"] = {
        p.key: w
        for p in st.pubs
        if (w := P.order_warnings((arr[p.key][:, -2] | arr[p.key][:, -1]).astype(np.uint8)))["warn"]
    }
    if res["n1"]["reject_eb"]:
        res["call"] = (
            "claim 4 at n = 1 in the MDI setting (trusting the input states only): the step "
            "channel is not EB"
        )
    else:
        res["call"] = (
            "MDI inconclusive: the payoff does not reject 0; this does not undo the P&M result"
        )
    return res


# -- Tier 2 (#39, amendment A5): the pilot ---------------------------------------------------------

#: the arms that use the ``ff`` ancillas, re-exported on the live pair (section 5's replace rule)
TIER2_FF_ARMS = ("RS", "R2", "R4")
#: steps per Tier 2 circuit (``[analysis] length`` of phase8_coherence.toml; checked)
TIER2_LENGTH = 64


def tier2_ccfg(cfg: dict[str, Any]) -> dict[str, Any]:
    import phase8_coherence as PC

    return PC.load_config(REPO_ROOT / cfg["experiment"]["coherence_config"])


def tier2_stages(cfg: dict[str, Any]) -> list[LiveStage]:
    """Amendment A1's pilot: one job, 9 arms x 2,000 shots in the harness's seeded pub order
    (``phase8_hw.tier2_schedules``), costed with #32's design durations. Only ``pilot`` may be a
    live stage here: main and the controls wait for #35 and the coordinator (#39)."""
    import phase8_coherence_design as DS
    import phase8_hw as H

    from qmem import coherence as Q

    ccfg = tier2_ccfg(cfg)
    if ccfg["analysis"]["length"] != TIER2_LENGTH:
        raise F.Refused(f"tier2: length {ccfg['analysis']['length']} != {TIER2_LENGTH}")
    sched = dict(H.tier2_schedules(cfg, ccfg)["stages"])
    dur = {r["arm"]: r["circuit_s"] for r in DS.budget(ccfg, Q.parse_arms(ccfg["arms"]))["arms"]}
    plan = {t["tier"]: t for t in json.loads((REPO_ROOT / PLAN2).read_text())["tiers"]}
    out = []
    for st in live_cfg(cfg, "tier2")["stages"]:
        if st != "pilot":
            raise F.Refused(f"tier2: {st!r} is not a live stage (the pilot only; #39)")
        pubs = sched[st]
        cost = H.cost_model(cfg).job(pubs, dur)
        pj = plan["pilot"]["per_job"]
        arms = {p.arm for p in pubs}
        if (
            arms != set(plan["pilot"]["arms"])
            or cost["shots"] != pj["shots"]
            or abs(cost["upper_s"] - pj["upper_s"]) > 1e-6
            or any(
                sum(p.shots for p in pubs if p.arm == a) != plan["pilot"]["shots_per_arm"]
                for a in arms
            )
        ):
            raise F.Refused(f"tier2 pilot: the schedule or its cost differs from plan.json: {cost}")
        out.append(LiveStage(st, pubs, {p.key: p.arm for p in pubs}, cost))
    return out


PLAN2 = "results/summaries/phase8_coherence/plan.json"


def tier2_circuit_check(cfg: dict[str, Any], tier: str, names: Sequence[str]) -> dict[str, Any]:
    """The pilot's circuits by arm (no Qiskit). The arms without ancillas are the committed
    files (``phase8_hw.hash_rows``: plan.json's hash, found in the committed QASM directories,
    equal to ``hashes.csv``). The feed-forward arms are A5's re-exports in ``[tier2.live]
    qasm_dir``: each must hash to its manifest and to ``[tier2.live] ancilla_sha256``, and the
    manifest's ancillas must be ``[tier2.live] ancillas``. The digest of the sorted
    ``name sha256`` lines of the job's circuits must equal ``qasm_digest``."""
    import phase8_hw as H

    lc = live_cfg(cfg, tier)
    qd = REPO_ROOT / lc["qasm_dir"]
    man = json.loads((qd / "manifest.json").read_text())
    if man["ancillas"] != list(lc["ancillas"]):
        raise F.Refused(f"tier2: manifest ancillas {man['ancillas']} != A5's {lc['ancillas']}")
    rows = {
        r["circuit"]: r
        for r in H.hash_rows(cfg)
        if r["tier"].startswith("tier2:") and r["tier"] != "tier2:controls"
    }
    pre = prereg_hashes()
    texts, shas, bad = {}, {}, []
    for n in names:
        if n in TIER2_FF_ARMS:
            f, want = qd / f"{n}.qasm", lc["ancilla_sha256"].get(n)
            ok_ref = man["circuits"].get(n, {}).get("sha256") == want
        else:
            r = rows.get(n, {})
            f, want = REPO_ROOT / r.get("file", "/nonexistent"), r.get("sha256")
            ok_ref = r.get("status") == "ok" and pre.get(n) == want
        text = f.read_text() if f.is_file() else ""
        h = sha_text(text)
        if not text or not ok_ref or h != want:
            bad.append(n)
        texts[n], shas[n] = text, h
    if bad:
        raise F.Refused(f"{tier}: circuit hash mismatch: {bad}")
    if name_digest(sorted(shas.items())) != lc["qasm_digest"]:
        raise F.Refused(f"{tier}: the digest of the job's circuits differs from A5's")
    return {"texts": texts, "digest": lc["qasm_digest"], "files": len(texts), "sha256": shas}


def tier2_durations(
    cfg: dict[str, Any], dur: dict[str, float | None], dev: dict[str, Any]
) -> dict[str, float]:
    """Live circuit durations for the cost. A static circuit's is the live target's
    ``estimate_duration``. A feed-forward arm (``if_else``: no static duration) gets #32's
    design duration moved by the live differences: the baseline's live duration less the
    design's, plus, per ``ff`` element and step, the live ancilla readout time less the
    design's (``DS.ff_idle_s`` with the live measurement time)."""
    import phase8_coherence_design as DS

    from qmem import coherence as Q

    ccfg = tier2_ccfg(cfg)
    arms = Q.parse_arms(ccfg["arms"])
    design = {r["arm"]: r["circuit_s"] for r in DS.budget(ccfg, arms)["arms"]}
    ds = ccfg["design"]
    anc = live_cfg(cfg, "tier2")["ancillas"]
    meas = max(float(dev["qubits"][str(q)]["measure_time"]) for q in anc)
    live_ds = {**ccfg, "design": {**ds, "ff_measure_s": meas}}
    d_ff = DS.ff_idle_s(live_ds) - DS.ff_idle_s(ccfg)
    if dur.get("baseline") is None:
        raise F.Refused("tier2: the baseline has no live duration")
    d_base = float(dur["baseline"]) - ccfg["budget"]["baseline_circuit_s"]
    out = {}
    for n, d in dur.items():
        if d is not None:
            out[n] = float(d)
            continue
        n_ff = max(sum(1 for e in p if e.startswith("ff")) for p in arms[n].programs)
        out[n] = design[n] + d_base + TIER2_LENGTH * n_ff * d_ff
    return out


def tier2_floor(cfg: dict[str, Any], dev: dict[str, Any], pair: Sequence[int]) -> dict[str, Any]:
    """Section 5's floor for the live ancillas: readout <= ``tier2_ancilla_readout_max``, and
    the R2/R4 vanish power (f = 0.25) and RS non-inferiority power (m = 0.20) >= their floors,
    with the larger readout error as the bound on the bit bias (``phase8_hw.ancilla_power``)."""
    import phase8_hw as H

    rs = cfg["reselection"]
    ro = {str(q): float(dev["qubits"][str(q)]["readout"]) for q in pair}
    e = max(ro.values())
    ccfg = tier2_ccfg(cfg)
    pw = H.ancilla_power(cfg, ccfg, H.tier2_model_ctx(ccfg), e)
    ok = {
        "readout": e <= rs["tier2_ancilla_readout_max"],
        "vanish": pw["vanish_power"] >= rs["tier2_vanish_power_floor"],
        "preserve": pw["preserve_power"] >= rs["tier2_preserve_power_floor"],
    }
    return {
        "ancillas": list(pair),
        "readout": ro,
        "bias_bound": e,
        "readout_max": rs["tier2_ancilla_readout_max"],
        "vanish_power": pw["vanish_power"],
        "preserve_power": pw["preserve_power"],
        "floors": {
            "vanish": rs["tier2_vanish_power_floor"],
            "preserve": rs["tier2_preserve_power_floor"],
        },
        "w0": pw["w0"],
        "checks": ok,
        "pass": all(ok.values()),
    }


def tier2_selection_check(cfg: dict[str, Any], tier: str, dev: dict[str, Any], sel: dict) -> dict:
    """A5's selection gate: the live rule must pick A5's ancillas (else a new dated amendment
    first), and they must pass the floor (else abort, section 5)."""
    lc = live_cfg(cfg, tier)
    chosen = list(sel["tier2_ancillas"]["chosen"])
    if chosen != list(lc["ancillas"]):
        raise F.Refused(
            f"{tier}: the live rule picks ancillas {chosen}, A5 records {lc['ancillas']}: "
            "stop; any change needs a new dated amendment"
        )
    floor = tier2_floor(cfg, dev, chosen)
    if not floor["pass"]:
        raise F.Refused(f"{tier}: the live ancillas fail section 5's floor: abort ({floor})")
    return floor


def tier2_decode(pub: F.Pub, data: Any) -> dict[str, np.ndarray]:
    """``<key>__y``: ``(shots, 64)`` outputs ``y0..y63``; ``<key>__m`` (A2's q75 record) and
    ``<key>__b1``/``__b2`` (the ancilla bits, ``(shots, 64)``) when the circuit has them."""
    y = np.stack([bits_of(data, f"y{t}")[:, 0] for t in range(TIER2_LENGTH)], 1)
    out = {f"{pub.key}__y": y.astype(np.uint8)}
    for reg in ("m", "b1", "b2"):
        if hasattr(data, reg):
            b = bits_of(data, reg)
            if b.shape != y.shape:
                raise ValueError(f"{pub.key}: register {reg} has shape {b.shape}")
            out[f"{pub.key}__{reg}"] = b
    return out


def tier2_analyse(cfg: dict[str, Any], stages: list[LiveStage], arrays: dict) -> dict:
    """Section 3.6's pilot gate exactly as pre-registered and as the local dry run computes it
    (``phase8_hw.tier2_run``): point estimates of P(1) and lag 1 after the burn-in, per arm,
    against the baseline, the fixed model's prediction, or the paired arm
    (``P.tier2_pilot_gate``). A failed gate stops the tier. Shot warnings, the ancilla bits'
    bias and the fixed model's predictions are reported, never part of the gate."""
    import phase8_coherence_design as DS
    import phase8_hw as H

    from qmem import coherence as Q

    ccfg = tier2_ccfg(cfg)
    s = DS.settings(ccfg)
    t2 = H.tier2_model_ctx(ccfg)
    arms = Q.parse_arms(ccfg["arms"])
    qfix = t2["model"](t2["fixed"])
    st = next(x for x in stages if x.stage == "pilot")
    arr = arrays["pilot"]
    ys: dict[str, list] = {}
    bits: dict[str, list] = {}
    for p in sorted(st.pubs, key=lambda q: q.position):
        ys.setdefault(p.arm, []).append(arr[f"{p.key}__y"])
        for reg in ("b1", "b2"):
            if f"{p.key}__{reg}" in arr:
                bits.setdefault(f"{p.arm}:{reg}", []).append(arr[f"{p.key}__{reg}"])
    y = {a: np.concatenate(v) for a, v in ys.items()}
    pst = {
        a: {"p1": float(v[:, s.burn :].mean()), "lag1": H._lag1(v, s.burn), "shots": len(v)}
        for a, v in y.items()
    }
    pred = {a: Q.model_arm_stats(qfix, arms[a], s) for a in y}
    rule = P.Tier2Rule()
    gate = P.tier2_pilot_gate(pst, pred, rule)
    rows = []
    for a in ("E0", "E00", "RS"):
        rows.append(("p1", a, "baseline", pst["baseline"]["p1"], rule.pilot_sham_p1))
    for a in ("P2", "P4"):
        rows.append(("p1", a, "fixed", float(pred[a]["p1"]), rule.pilot_scan_p1))
    for a, b in (("INV2", "E00"), ("R2", "A2")):
        for k in ("p1", "lag1"):
            rows.append((k, a, b, pst[b][k], rule.pilot_pair))
    table = []
    for k, a, ref, rv, m in rows:
        key = f"{a}_{k}_vs_{'fixed' if ref == 'fixed' else ref}"
        table.append(
            {
                "check": key,
                "arm": a,
                "statistic": k,
                "value": pst[a][k],
                "reference": ref,
                "reference_value": rv,
                "difference": pst[a][k] - rv,
                "margin": m,
                "pass": gate["checks"][key],
            }
        )
    warn = {}
    for a, v in y.items():
        g = F.guard(v, None, F.AnalysisSettings(), cfg["tier2"]["dry_run"]["seed"])
        if g["decoding_warn"] + g["integrity_fail"]:
            warn[a] = g["decoding_warn"] + g["integrity_fail"]
    res: dict[str, Any] = {
        "gate": gate,
        "table": table,
        "statistics": pst,
        "fixed_prediction": {a: {k: float(pred[a][k]) for k in ("p1", "lag1")} for a in pred},
        "ancilla_bit_mean": {k: float(np.concatenate(v).mean()) for k, v in bits.items()},
        "shot_warnings": warn,
        "rule": dataclasses.asdict(rule),
        "burn": s.burn,
    }
    res["call"] = (
        "pilot gate passes: Tier 2 main may run after #35's verdict and the coordinator's review"
        if gate["go"]
        else "pilot gate fails: Tier 2 stops (section 3.6); any rerun needs a dated amendment"
    )
    return res


TIERS: dict[str, dict[str, Any]] = {
    "tier1": {"stages": tier1_stages, "decode": tier1_decode, "analyse": tier1_analyse},
    "tier2": {
        "stages": tier2_stages,
        "decode": tier2_decode,
        "analyse": tier2_analyse,
        "circuit_check": tier2_circuit_check,
        "durations": tier2_durations,
        "selection_check": tier2_selection_check,
        "admit_whole_tier": True,
    },
    "tier1_mdi": {
        "stages": tier1_mdi_stages,
        "decode": tier1_decode,
        "analyse": tier1_mdi_analyse,
        "base": "tier1",
        "table": "mdi_live",
    },
}


def tier_entry(tier: str) -> dict[str, Any]:
    if tier not in TIERS:
        raise F.Refused(f"no live plan for {tier} yet: add its stages, decoding and analysis")
    return TIERS[tier]


def tier_budget(cfg: dict[str, Any], tier: str, stages: list[LiveStage]) -> P.TierBudget:
    """This run's budget. Its cap is the smaller of ``[<table>] hard_cap_s`` and the base tier's
    hard cap less what the base tier's committed ledgers already counted, so a sub-tier's spend
    is cumulative with the base tier's (Tier 1: P&M's 15.16 s, then MDI, against 30 s)."""
    lc, b = live_cfg(cfg, tier), base_tier(tier)
    left = cfg[b]["hard_cap_s"] - prior_counted(cfg, tier)
    return P.TierBudget(
        tier,
        [s.cost for s in stages],
        min(lc.get("hard_cap_s", left), left),
        cfg[b]["max_job_estimate_s"],
        [s.stage for s in stages],
    )


def live_cost(cfg: dict[str, Any], st: LiveStage, dur: dict[str, float]) -> dict[str, float]:
    """The job's cost under the cost model with the live target's circuit durations, and the
    larger of it and the pre-registered cost (what the guard and the ledger use)."""
    import phase8_hw as H

    by_arm: dict[str, float] = {}
    for p in st.pubs:
        d = dur.get(st.names[p.key])
        # #39: a missing or zero duration (as #37's summary appeared to show for the CZ) would
        # make the live cost an underestimate; refuse rather than price a circuit at nothing
        if d is None or not np.isfinite(d) or d <= 0:
            raise F.Refused(f"{st.stage}: no positive live duration for {st.names[p.key]}: {d}")
        by_arm[p.arm] = max(by_arm.get(p.arm, 0.0), float(d))
    lv = H.cost_model(cfg).job(st.pubs, by_arm)
    return {
        "estimate_s": max(st.cost["estimate_s"], lv["estimate_s"]),
        "upper_s": max(st.cost["upper_s"], lv["upper_s"]),
        "shots": st.cost["shots"],
        "prereg_estimate_s": st.cost["estimate_s"],
        "prereg_upper_s": st.cost["upper_s"],
        "live_estimate_s": lv["estimate_s"],
        "live_upper_s": lv["upper_s"],
    }


# -- network: service and backend ------------------------------------------------------------------


def service(cfg: dict[str, Any]) -> Any:
    """``QiskitRuntimeService`` with the credentials in ``[ibm] env_file`` only."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    env = read_env(REPO_ROOT / cfg["ibm"]["env_file"])
    return QiskitRuntimeService(
        channel=cfg["ibm"]["channel"], token=env["IBM_API_KEY"], instance=env["IBM_INSTANCE_CRN"]
    )


# -- the run ---------------------------------------------------------------------------------------


def power_gate(cfg: dict[str, Any], tier: str, dev: dict[str, Any]) -> dict[str, Any]:
    """A4's pre-data power gate at this run's own live snapshot (``phase8_mdi_power.assess``,
    hardware group): refuse unless the fitted model's power reaches the target at the planned
    shots and the planned cost fits the per-job limit and the cap left."""
    if tier != "tier1_mdi":
        raise F.Refused(f"{tier}: no power gate is defined")
    import phase8_mdi_power as MP

    r = MP.assess(cfg, dev)
    keep = {
        "rule": r["rule"],
        "go": r["go"],
        "cost_ok": r["cost_ok"],
        "planned_cost": r["planned_cost"],
        "tier1_cap_left_s": r["tier1_cap_left_s"],
        "models": {k: m["n1"] for k, m in r["models"].items()},
    }
    if not r["go"]:
        raise F.Refused(f"{tier}: the pre-data power gate fails at this calibration: {keep}")
    return keep


def checks(cfg, tier: str, svc: Any, backend: Any, out: Path, workers: int, selection) -> dict:
    """Everything before a submission that needs the live target: the calibration snapshot,
    the live re-selection (section 5), every hash, the ISA check, the live cost, admission.
    ``selection(dev, out, workers)`` re-runs the selection rules (``phase8_hw``)."""
    import phase8_hw as H

    ent = tier_entry(tier)
    lc = live_cfg(cfg, tier)
    stages = ent["stages"](cfg)
    res: dict[str, Any] = {"tier": tier, "backend": backend.name, "read_at": now()}
    if tier != base_tier(tier):
        res["base_tier"] = base_tier(tier)
        res["prior_counted_s"] = prior_counted(cfg, tier)
    if tier == "tier1_mdi":
        res["precondition"] = mdi_precondition(cfg)
    res["prereg"] = prereg_guard(cfg, tier)
    names = sorted({n for s in stages for n in s.names.values()})
    cc = ent.get("circuit_check", circuit_check)(cfg, tier, names)
    rows = H.hash_rows(cfg)
    pre = prereg_hashes()
    res["hashes"] = {
        "all_ok": all(r["status"] in ("ok",) for r in rows),
        "n": len(rows),
        "same_as_prereg": all(pre.get(r["circuit"]) == r["sha256"] for r in rows),
        "tier_digest": cc["digest"],
        "tier_files": cc["files"],
    }
    if not (res["hashes"]["all_ok"] and res["hashes"]["same_as_prereg"]):
        raise F.Refused(f"a committed circuit hash changed: {res['hashes']}")
    dev = snapshot(backend)
    save_json(P.keep_first(out / "calibration_live.json"), dev)
    res["snapshot"] = {
        "calibration_date": dev["calibration_date"],
        "read_at": dev["read_at"],
        "sha256": P.snapshot_hash(usable_snapshot(dev)),
        "qubits": {q: dev["qubits"][str(q)] for q in (59, 75, 74)},
        "cz_59_75": dev["cz"].get("59-75"),
    }
    res["cz_durations"] = cz_check(dev, lc["qubits"])
    if lc.get("power_gate"):
        res["power_gate"] = power_gate(cfg, tier, dev)
    sel = selection(usable_snapshot(dev), out, workers)
    res["selection"] = sel
    changed = [k for k in lc.get("gating_selections", []) if not sel["reproduces"][k]]
    res["selection_gate"] = {"gating": lc.get("gating_selections", []), "changed": changed}
    if changed:
        raise F.Refused(f"{tier}: live re-selection changed {changed}: a dated amendment first")
    if "selection_check" in ent:
        res["floor"] = ent["selection_check"](cfg, tier, dev, sel)
    isa = isa_check(cc["texts"], backend, lc["qubits"])
    dur = ent["durations"](cfg, isa["duration_s"], dev) if "durations" in ent else isa["duration_s"]
    res["isa"] = {
        "unchanged": True,
        "duration_s": dict(sorted(dur.items())),
        "static": sorted(n for n, d in isa["duration_s"].items() if d is not None),
    }
    res["costs"] = {s.stage: live_cost(cfg, s, dur) for s in stages}
    return {"res": res, "stages": stages, "isa": isa, "dev": dev}


def prepare(cfg: dict[str, Any], args: Any, selection, svc_factory=service) -> dict[str, Any]:
    """``--prepare --tier tier2`` (#39): read the live calibration and run section 5's selection
    and the ancilla floor on it, before A5 is written. Read-only: no approval, no circuits, no
    sampler, nothing submitted (this function has no submit path). Its snapshot and pick are
    what A5 records; the later ``--dry-run`` and ``--submit`` must reproduce the pick."""
    import phase8_hw as H

    if args.tier != "tier2":
        raise F.Refused(f"--prepare is defined for tier2 only, not {args.tier}")
    out = (
        Path(args.from_run)
        if args.from_run
        else H.new_run_dir(f"{cfg['experiment']['name']}_prepare_{args.tier}")
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(P.keep_first(out / "config_meta.json"), run_metadata(cfg, None))
    svc = svc_factory(cfg)
    backend = svc.backend(cfg["ibm"]["backend"])
    res: dict[str, Any] = {"tier": args.tier, "backend": backend.name, "plan": plan_of(svc)}
    res["usage"] = read_usage(svc)
    dev = snapshot(backend)
    save_json(P.keep_first(out / "calibration_live.json"), dev)
    sel = selection(usable_snapshot(dev), out, args.workers or 2)
    pair = list(sel["tier2_ancillas"]["chosen"])
    res["snapshot"] = {
        "calibration_date": dev["calibration_date"],
        "read_at": dev["read_at"],
        "sha256": P.snapshot_hash(usable_snapshot(dev)),
        "qubits": {str(q): dev["qubits"][str(q)] for q in (59, 75, *pair)},
    }
    res["selection"] = sel
    res["ancillas"] = pair
    res["committed_ancillas"] = sel["committed"]["tier2_ancillas"]
    res["floor"] = tier2_floor(cfg, dev, pair)
    res["cz_durations"] = cz_check(dev, [59, 75, *pair])
    res["submitted"] = False
    save_json(out / "prepare.json", res)
    H.log(out, f"[{args.tier}] prepare: ancillas {pair}; floor pass {res['floor']['pass']}")
    return {"out": out, "res": res}


def run(
    cfg: dict[str, Any], args: Any, selection, svc_factory=service, sampler=None
) -> dict[str, Any]:
    """``--dry-run`` (everything but the submission) or ``--submit``, for ``args.tier``."""
    import phase8_hw as H

    tier, submit = args.tier, bool(args.submit)
    # before any network call: the approval (submit only) and the pre-registration
    if submit:
        approval_guard(cfg, tier, True, None)
    prereg_guard(cfg, tier)
    name = f"{cfg['experiment']['name']}_{'live' if submit else 'livedry'}_{tier}"
    out = (Path(args.from_run) if args.from_run else H.new_run_dir(name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(P.keep_first(out / "config_meta.json"), run_metadata(cfg, None))
    appr = approval_guard(cfg, tier, True, out) if submit else None
    svc = svc_factory(cfg)
    backend = svc.backend(cfg["ibm"]["backend"])
    plan = plan_of(svc)
    usage0 = read_usage(svc)
    H.log(out, f"[{tier}] backend {backend.name}; plan {plan!r}; usage {usage0}")
    ck = checks(cfg, tier, svc, backend, out, args.workers or 2, selection)
    res, stages = ck["res"], ck["stages"]
    res.update({"plan": plan, "usage": usage0, "approval": None if appr is None else appr.url})
    ledger = P.Ledger()
    for prior in sorted(out.glob("*/job.json")):  # a resume: what this run already counted
        rec = json.loads(prior.read_text())
        if "charge" in rec:
            ledger.add(rec.get("tier", tier), F.Charge(**rec["charge"]))
    tbud = tier_budget(cfg, tier, stages)
    tbud.jobs = [res["costs"][s.stage] for s in stages]  # max(pre-registered, live) per job
    prereg_budgets = H.tier_budgets(cfg, H.sub_cfgs(cfg))
    budgets = {t: prereg_budgets[t] for t in cfg["order"]["tiers"]}
    ent = tier_entry(tier)
    if tier == base_tier(tier) and not ent.get("admit_whole_tier"):
        budgets[tier] = tbud  # a sub-tier is admitted on its base tier's whole cap
    rem = remaining_allowance(cfg, ledger, usage0)
    if live_cfg(cfg, tier).get("earlier_ledgers"):
        res["earlier_tiers"] = earlier_counted(cfg, tier)
        res["admission"] = admission_after(cfg, tier, budgets[base_tier(tier)], rem)
    else:
        res["admission"] = admission(cfg, tier, budgets, rem)
    res["tier_budget"] = {
        "jobs": tbud.jobs,
        "hard_cap_s": tbud.hard_cap_s,
        "max_job_estimate_s": tbud.max_job_estimate_s,
        "problems": tbud.problems(),
    }
    if tbud.problems():
        raise F.Refused(f"{tier}: {tbud.problems()}")
    if not submit:
        verdicts = []
        for s in stages:
            rec: dict[str, Any] = {}
            try:
                job_guard(cfg, tier, tbud, ledger, svc, res["costs"][s.stage])(rec)
                v = "would pass"
            except F.Refused as e:
                v = f"would be refused: {e}"
            verdicts.append({"stage": s.stage, **res["costs"][s.stage], "guard": v, **rec})
        res["guards"] = verdicts
        res["submitted"] = False
        save_json(out / "live_dry_run.json", res)
        H.log(out, f"[{tier}] live dry run: {json.dumps(verdicts, default=str)}; nothing submitted")
        return {"out": out, "res": res}
    # submit, stage by stage (idempotent by job id)
    circ = ck["isa"]["circuits"]
    arrays: dict[str, dict[str, np.ndarray]] = {}
    lb = LiveBackend(cfg, svc, backend, lambda p: None, ent["decode"], out, sampler)
    for st in stages:
        lb.pubs = st.pubs
        lb.circuit = lambda p, st=st: circ[st.names[p.key]]
        cost = res["costs"][st.stage]
        guard = job_guard(cfg, tier, tbud, ledger, svc, cost)
        arrays[st.stage] = P.run_stage(tier, st.stage, st.pubs, lb, out, cost, ledger, guard)
        H.log(out, f"[{tier} {st.stage}] done; charge {ledger.charges[tier][-1]}")
    after = snapshot(backend)["qubits"]
    res["calibration_after"] = after["75"]
    res["calibration_after_qubits"] = {str(q): after[str(q)] for q in live_cfg(cfg, tier)["qubits"]}
    prior = prior_counted(cfg, tier)
    res["ledger"] = {
        "charges": [dataclasses.asdict(c) for c in ledger.charges.get(tier, [])],
        "counted_s": ledger.counted(tier),
        "allowance_reading_s": cfg["allowance"]["remaining_s"],
        "remaining_after_s": remaining_allowance(cfg, ledger, None),
    }
    if tier != base_tier(tier):
        res["ledger"] |= {
            "base_tier": base_tier(tier),
            "base_tier_prior_counted_s": prior,
            "base_tier_counted_s": prior + ledger.counted(tier),
            "base_tier_hard_cap_s": cfg[base_tier(tier)]["hard_cap_s"],
            "prior_ledgers": live_cfg(cfg, tier).get("prior_ledgers", []),
        }
    res["analysis"] = ent["analyse"](cfg, stages, arrays)
    res["submitted"] = True
    save_json(out / "live_run.json", res)
    H.log(out, f"[{tier}] {res['analysis'].get('call')}; counted {res['ledger']['counted_s']} s")
    return {"out": out, "res": res, "arrays": arrays, "stages": stages}
