"""Phase 8 Stage 5 (issue #37): the live path of ``experiments/phase8_hw.py``.

``phase8_hw.py --dry-run`` (live target, nothing submitted) and ``--submit`` call this module.
It is tier-generic: a tier needs a ``[<tier>.live]`` config table and an entry in ``TIERS``
(its stages, circuits, decoding and pre-registered analysis). Only Tier 1 (P&M, #33) has one.

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


def live_cfg(cfg: dict[str, Any], tier: str) -> dict[str, Any]:
    try:
        return cfg[tier]["live"]
    except KeyError:
        raise F.Refused(f"{tier} has no [{tier}.live] table: no live path for it") from None


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
    appr = P.Approval(tier, (cfg.get("approval") or {}).get(tier) or None)
    appr.check(tier, submit)
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
    Returns the circuits and their durations on the live target (hardware group)."""
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
        dur[n] = float(tq.estimate_duration(backend.target, unit="s"))
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
        [budgets[t] for t in order[: order.index(tier) + 1]],
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
        rec["approval_url"] = cfg["approval"][tier]
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
    """#33's export names: circuit ``c = 2 k + b``, k over ``R.AXIS_LABELS``."""
    k, b = divmod(chunk, 2)
    ax = R.AXIS_LABELS[k].replace("+", "p").replace("-", "m")
    return f"{arm}_{ax}_b{b}"


def tier1_stages(cfg: dict[str, Any]) -> list[LiveStage]:
    """#33's plan (``phase8_reference/summary.json``): one job per stage, pubs in its seeded
    order, with its committed cost."""
    plan = json.loads((REPO_ROOT / "results/summaries/phase8_reference/summary.json").read_text())[
        "plan"
    ]
    out = []
    for st in live_cfg(cfg, "tier1")["stages"]:
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
    ``y0..y{n-1}`` then ``m``; the negative control's mid-step ``neg`` is kept separately."""
    from phase8_hw import PM_STEPS

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


TIERS: dict[str, dict[str, Callable]] = {
    "tier1": {"stages": tier1_stages, "decode": tier1_decode, "analyse": tier1_analyse},
}


def tier_entry(tier: str) -> dict[str, Callable]:
    if tier not in TIERS:
        raise F.Refused(f"no live plan for {tier} yet: add its stages, decoding and analysis")
    return TIERS[tier]


def tier_budget(cfg: dict[str, Any], tier: str, stages: list[LiveStage]) -> P.TierBudget:
    lc = live_cfg(cfg, tier)
    return P.TierBudget(
        tier,
        [s.cost for s in stages],
        lc.get("hard_cap_s", cfg[tier]["hard_cap_s"]),
        cfg[tier]["max_job_estimate_s"],
        [s.stage for s in stages],
    )


def live_cost(cfg: dict[str, Any], st: LiveStage, dur: dict[str, float]) -> dict[str, float]:
    """The job's cost under the cost model with the live target's circuit durations, and the
    larger of it and the pre-registered cost (what the guard and the ledger use)."""
    import phase8_hw as H

    by_arm: dict[str, float] = {}
    for p in st.pubs:
        by_arm[p.arm] = max(by_arm.get(p.arm, 0.0), dur[st.names[p.key]])
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


def checks(cfg, tier: str, svc: Any, backend: Any, out: Path, workers: int, selection) -> dict:
    """Everything before a submission that needs the live target: the calibration snapshot,
    the live re-selection (section 5), every hash, the ISA check, the live cost, admission.
    ``selection(dev, out, workers)`` re-runs the selection rules (``phase8_hw``)."""
    import phase8_hw as H

    ent = tier_entry(tier)
    lc = live_cfg(cfg, tier)
    stages = ent["stages"](cfg)
    res: dict[str, Any] = {"tier": tier, "backend": backend.name, "read_at": now()}
    res["prereg"] = prereg_guard(cfg, tier)
    names = sorted({n for s in stages for n in s.names.values()})
    cc = circuit_check(cfg, tier, names)
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
    sel = selection(usable_snapshot(dev), out, workers)
    res["selection"] = sel
    changed = [k for k in lc.get("gating_selections", []) if not sel["reproduces"][k]]
    res["selection_gate"] = {"gating": lc.get("gating_selections", []), "changed": changed}
    if changed:
        raise F.Refused(f"{tier}: live re-selection changed {changed}: a dated amendment first")
    isa = isa_check(cc["texts"], backend, lc["qubits"])
    res["isa"] = {"unchanged": True, "max_duration_s": max(isa["duration_s"].values())}
    res["costs"] = {s.stage: live_cost(cfg, s, isa["duration_s"]) for s in stages}
    return {"res": res, "stages": stages, "isa": isa, "dev": dev}


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
    budgets = {t: prereg_budgets[t] for t in cfg["order"]["tiers"]} | {tier: tbud}
    rem = remaining_allowance(cfg, ledger, usage0)
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
    ent = tier_entry(tier)
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
    res["calibration_after"] = snapshot(backend)["qubits"]["75"]
    res["ledger"] = {
        "charges": [dataclasses.asdict(c) for c in ledger.charges.get(tier, [])],
        "counted_s": ledger.counted(tier),
        "allowance_reading_s": cfg["allowance"]["remaining_s"],
        "remaining_after_s": remaining_allowance(cfg, ledger, None),
    }
    res["analysis"] = ent["analyse"](cfg, stages, arrays)
    res["submitted"] = True
    save_json(out / "live_run.json", res)
    H.log(out, f"[{tier}] {res['analysis'].get('call')}; counted {res['ledger']['counted_s']} s")
    return {"out": out, "res": res, "arrays": arrays, "stages": stages}
