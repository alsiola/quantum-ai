"""Phase 8 Stage 2b (issue #34): the direct Claim 3 contrast (``--part contrast``).

Claim 3 (#28's taxonomy, item 3): a population-preserving dephasing removes the witness
*while the matched sham preserves it*. Two predeclared legs on w(0):

* **vanish:** R2 (and R4) vanishes relative to the feed-forward sham RS with #31's rule:
  ``|w_R(0)| + 1.96 SE_R < f |w_RS(0)|`` on all jobs and on every leave-one-job-out subset;
* **preserve:** RS is preserved relative to the 4-pulse sham E00: the ratio
  ``w_RS(0) / w_E00(0)`` equivalent to 1 (TOST, the 90% CI inside ``[1 - m, 1 + m]``), or,
  the weaker one-sided form, non-inferior (the lower 95% bound above ``1 - m``).

Power from the fixed and fitted quantum models' exact w(0) and per-shot SDs from simulated
shots; the trade against the scan points when the tier cannot power a leg. Numpy only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import phase8_coherence as PC
import phase8_coherence_design as DS

from qmem import coherence as Q
from qmem import followup as F
from qmem.records import save_json

Z95 = 1.6448536269514722


def part_contrast(cfg: dict[str, Any], out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    cc = cfg["contrast"]
    ctx = PC.Ctx(cfg)
    s = DS.settings(cfg)
    arms = Q.parse_arms(cfg["arms"])
    gens = DS.generators(ctx, cfg, None)
    names = cc["arms"]
    n_plan, jobs = cfg["plan"]["shots_per_arm"], cc["jobs"]
    rows, summary = [], {"meta": meta, "generators": {}}
    for g, p in gens.items():
        qm = ctx.model(p)
        rng = np.random.default_rng([cc["seed"], len(g)])
        st, sd = {}, {}
        for a in names:
            st[a] = Q.model_arm_stats(qm, arms[a], s)["w0"]
            ds = Q.data_stats(DS.simulate(qm, arms[a], cc["sd_shots"], s.length, rng).y, s)
            sd[a] = ds["w0_se"] * math.sqrt(ds["n"])
        e: dict[str, Any] = {"w0": st, "w0_sd_per_shot": sd}
        e["ratio_RS_E00"] = st["RS"] / st["E00"]
        e["ratio_RS_baseline"] = st["RS"] / st["baseline"]
        # vanish leg: R2 and R4 against RS
        van = {}
        for r in ("R2", "R4"):
            v: dict[str, Any] = {"w0": st[r], "ratio_to_RS": st[r] / st["RS"]}
            for f in cc["fractions"]:
                v[f"power_f{f}"] = {
                    str(n): Q.vanish_lojo_power(
                        st[r], sd[r], st["RS"], sd["RS"], n, jobs, f, 1.96, cc["n_mc"], cc["seed"]
                    )
                    for n in cc["shots_grid"]
                }
                v[f"shots_80_f{f}"] = _shots_vanish(st, sd, r, f, jobs, cc)
            v["fraction_80_at_plan"] = _fraction_vanish(st, sd, r, n_plan, jobs, cc)
            van[r] = v
        e["vanish"] = van
        # preserve leg: RS against E00 (and against the pooled population-invariant shams)
        pres = {}
        refs = {"E00": (st["E00"], sd["E00"], 1)}
        pool = cc["pooled_reference"]
        refs["pooled"] = (float(np.mean([st[a] for a in pool])), sd["E00"], len(pool))
        for ref, (wb, sb, k) in refs.items():
            for two in (True, False):
                key = f"{ref}_{'tost' if two else 'noninferiority'}"
                sbk = sb / math.sqrt(k)  # k arms' shots pooled for the reference

                def pw(n, m, wa=st["RS"], sa=sd["RS"], wb=wb, sbk=sbk, two=two):
                    return Q.ratio_power(wa, sa, wb, sbk, n, m, Z95, two)

                x: dict[str, Any] = {"ratio": st["RS"] / wb}
                for m in cc["margins"]:
                    x[f"power_plan_m{m}"] = pw(n_plan, m)
                    x[f"shots_80_m{m}"] = Q.ratio_shots(
                        st["RS"], sd["RS"], wb, sbk, m, Z95, 0.8, two
                    )
                x["margin_80_at_plan"] = _margin(pw, n_plan)
                pres[key] = x
        e["preserve"] = pres
        summary["generators"][g] = e
        for a in names:
            rows.append({"generator": g, "arm": a, "w0": st[a], "w0_sd_per_shot": sd[a]})
    summary["trade"] = trade(cfg, summary)
    sdir = PC.sdir(cfg)
    PC.write_csv(sdir / "contrast_arms.csv", [PC.jround(r) for r in rows])
    save_json(sdir / "contrast.json", PC.jround(summary, 9))
    return summary


def _shots_vanish(st, sd, r, f, jobs, cc) -> float:
    def p(n):
        return Q.vanish_lojo_power(
            st[r], sd[r], st["RS"], sd["RS"], n, jobs, f, 1.96, cc["n_mc"], cc["seed"]
        )

    lo, hi = 100.0, 1e8
    if p(hi) < 0.8:
        return math.inf
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if p(mid) >= 0.8 else (mid, hi)
    return hi


def _fraction_vanish(st, sd, r, n, jobs, cc) -> float:
    def p(f):
        return Q.vanish_lojo_power(
            st[r], sd[r], st["RS"], sd["RS"], n, jobs, f, 1.96, cc["n_mc"], cc["seed"]
        )

    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if p(mid) >= 0.8 else (mid, hi)
    return hi


def _margin(pw, n) -> float:
    lo, hi = 0.0, 5.0
    if pw(n, hi) < 0.8:
        return math.inf
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pw(n, mid) >= 0.8 else (mid, hi)
    return hi


def trade(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Reallocations of the main tier at a fixed total: drop scan points and give their shots
    to RS and E00 (and R2); the RS/E00 TOST margin at 80% power, and the cost."""
    cc = cfg["contrast"]
    n_plan = cfg["plan"]["shots_per_arm"]
    main = next(t for t in cfg["plan"]["tiers"] if t["name"] == "main")
    bc = cfg["budget"]
    cm = F.CostModel(
        bc["sub_job_overhead_s"], bc["rep_delay_s"], bc["margin"], bc["billing_round_s"]
    )
    dur = {r["arm"]: r["circuit_s"] for r in DS.budget(cfg, Q.parse_arms(cfg["arms"]))["arms"]}
    out: dict[str, Any] = {}
    for opt in cc["trade_options"]:
        arms = [a for a in main["arms"] if a not in opt["drop"]]
        freed = len(opt["drop"]) * n_plan
        boost = opt["boost"]
        shots = {a: n_plan for a in arms}
        for a in boost:
            shots[a] += freed // len(boost)
        for a, extra in opt.get("extra", {}).items():
            shots[a] = shots.get(a, n_plan) + extra
        jobs = main["jobs"]
        pubs = [F.Pub(0, i, a, 0, shots[a] // jobs) for i, a in enumerate(arms)]
        job = cm.job(pubs, dur)
        row: dict[str, Any] = {
            "drop": opt["drop"],
            "boost": boost,
            "shots": shots,
            "total_shots": sum(shots.values()),
            "upper_s": jobs * job["upper_s"],
            "estimate_s": jobs * job["estimate_s"],
        }
        for g, e in summary["generators"].items():
            st, sd = e["w0"], e["w0_sd_per_shot"]
            n_rs, n_e = shots["RS"], shots["E00"]
            sa_x = sd["RS"] * math.sqrt(n_plan / n_rs)
            sb_x = sd["E00"] * math.sqrt(n_plan / n_e)
            for two in (True, False):

                def pw(n, m, wa=st["RS"], sa=sa_x, wb=st["E00"], sb=sb_x, two=two):
                    return Q.ratio_power(wa, sa, wb, sb, n, m, Z95, two)

                row[f"{g}_margin80_{'tost' if two else 'noninf'}"] = _margin(pw, n_plan)
            sdx = dict(sd)
            for a in ("R2", "RS"):
                sdx[a] = sd[a] * math.sqrt(n_plan / shots[a])
            row[f"{g}_vanish_R2_fraction80"] = _fraction_vanish(st, sdx, "R2", n_plan, jobs, cc)
        out[opt["name"]] = row
    return out
