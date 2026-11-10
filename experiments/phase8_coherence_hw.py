"""Phase 8 Stage 2 (issue #32): the circuits of the coherence design (``hardware`` group).

``phase8_coherence.py --export-qasm``. ``FakeKingston`` only: no QPU, no IBM service. Builds
every arm of ``[arms]`` from Phase 6's committed transpiled circuit (``qmem.ablate_hw.
restructure``: the slot after q75's leading gates, before the step's first CZ, while q59 is
reset) and checks that the *physical* phase insertions survive the transpiler and the
scheduler:

* each element ``e<k>`` is the frame-neutral echo ``x, rz(a), x, rz(-a)`` with
  ``a = -k pi / 8``, fenced by single-qubit barriers on q75 so no pass can merge its pulses
  into one virtual ``rz``;
* at optimisation levels 0-3 the transpiled circuit keeps every ``x`` of every element (count
  per step), the rz angles of every slot and the frame neutrality (the slot's rz sum is 0);
* the controls: the same circuits with *virtual* ``rz(phi)`` (what a naive phase scan would
  run) and with unfenced echoes, to show what the transpiler does to them;
* the scheduler (``alap``) adds no time to an echo arm beyond the q59 reset window;
* the feed-forward arms: two ancillas prepared in ``|+>`` and measured each step, the echoes
  in ``if_else`` blocks with the same pulse count in both branches.

Writes ``circuits.json`` (checks, hashes, durations, the ancilla selection) and ``device.json``
(the numbers ``--part design`` uses) to the summary directory, and QASM 3 of the listed arms.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from typing import Any

import phase8_coherence as PC

from qmem import ablate_hw as A
from qmem import coherence as Q
from qmem.records import REPO_ROOT, run_metadata, save_json

MEMORY, OUTPUT = 75, 59


def _echo(qc, q, phi: float, fence: bool = True) -> None:
    a, b = Q.echo_angles(phi)
    if fence:
        qc.barrier(q)
    qc.x(q)
    if fence:
        qc.barrier(q)
    qc.rz(a, q)
    qc.x(q)
    if fence:
        qc.barrier(q)
    qc.rz(b, q)
    if fence:
        qc.barrier(q)


def _phase(name: str) -> float:
    return Q.echo_phase(int(name[1:]))


def make_op(arm, mode: str, regs: dict[str, Any], anc: tuple[int, int] | None):
    """``make_op(qc, t)`` for ``restructure``: the arm's slot in step ``t``.

    ``mode``: ``physical`` (fenced echoes), ``unfenced`` (echoes without barriers) or
    ``virtual`` (a bare ``rz(phi)`` per element, what a naive scan would run)."""

    def op(qc, t: int) -> None:
        q = qc.qubits[MEMORY]
        alt = arm.schedule == "alternate"
        prog = arm.programs[t % len(arm.programs)] if alt else arm.programs[0]
        if arm.schedule == "random":
            a1, a2 = (qc.qubits[i] for i in anc)
            for a in (a1, a2):
                qc.reset(a)
                qc.h(a)
            qc.measure(a1, regs["b1"][t])
            qc.measure(a2, regs["b2"][t])
            # program index 2 b1 + b2: element 1 depends on b1, element 2 on b2
            for pos, (reg, bit) in enumerate((("b1", 1), ("b2", 1))):
                e_if = arm.programs[2 * bit if pos == 0 else 1][1 + pos]
                e_else = arm.programs[0][1 + pos]
                with qc.if_test((regs[reg][t], 1)) as else_:
                    _echo(qc, q, _phase(e_if))
                with else_:
                    _echo(qc, q, _phase(e_else))
            return
        for e in prog:
            if e == "I":
                continue
            if e == "meas":
                qc.measure(q, regs["m"][t])
                continue
            phi = _phase(e)
            if mode == "virtual":
                qc.rz(phi, q)
            else:
                _echo(qc, q, phi, fence=(mode == "physical"))

    return op


def build(full, arm, length: int, mode: str = "physical", anc=None):
    from qiskit import ClassicalRegister

    regs: dict[str, Any] = {}
    extra = []
    if any("meas" in p for p in arm.programs):
        regs["m"] = ClassicalRegister(length, "m")
    if arm.schedule == "random":
        regs["b1"] = ClassicalRegister(length, "b1")
        regs["b2"] = ClassicalRegister(length, "b2")
    extra = list(regs.values())
    if arm.name == "baseline":
        return A.restructure(full, full.qubits[MEMORY], full.qubits[OUTPUT], None)
    return A.restructure(
        full,
        full.qubits[MEMORY],
        full.qubits[OUTPUT],
        make_op(arm, mode, regs, anc),
        extra_regs=extra,
    )


def isa(qc, backend, level: int, scheduling: str | None = None):
    from qiskit import transpile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transpile(
            qc,
            backend,
            optimization_level=level,
            seed_transpiler=0,
            initial_layout=list(range(qc.num_qubits)),
            scheduling_method=scheduling,
        )


def q75_ops(qc) -> list[tuple[str, tuple[float, ...]]]:
    """q75's instructions in order, control-flow blocks flattened (both branches, in order)."""
    out = []
    idx = qc.find_bit(qc.qubits[MEMORY]).index

    def walk(circ, qmap):
        for inst in circ.data:
            op = inst.operation
            qs = [qmap[circ.find_bit(x).index] for x in inst.qubits]
            if op.name == "if_else":
                for block in op.blocks:
                    if block is None:
                        continue
                    inner = {block.find_bit(b).index: qs[i] for i, b in enumerate(block.qubits)}
                    out.append(("block", ()))
                    walk(block, inner)
                continue
            if idx in qs:
                out.append((op.name, tuple(round(float(p), 10) for p in op.params)))

    walk(qc, {i: i for i in range(qc.num_qubits)})
    return out


def slot_checks(qc, expected_x_per_step: float, length: int) -> dict[str, Any]:
    """Per-step x count on q75, the rz sum between the fences (frame neutrality) and the
    number of fenced segments, from the q75 instruction stream."""
    ops = q75_ops(qc)
    c = Counter(n for n, _ in ops)
    # rz angles between barriers that enclose an x (the echo's inner rz) and after it
    segs, cur = [], []
    for n, p in ops:
        if n == "barrier":
            segs.append(cur)
            cur = []
        else:
            cur.append((n, p))
    segs.append(cur)
    return {
        "x_total": int(c.get("x", 0)),
        "x_per_step": c.get("x", 0) / length,
        "expected_x_per_step": expected_x_per_step,
        "barriers_q75": int(c.get("barrier", 0)),
        "rz_count": int(c.get("rz", 0)),
        "if_else_blocks": int(c.get("block", 0)),
    }


def select_ancillas(backend, n_circuit: int) -> dict[str, Any]:
    """Two ancillas for the feed-forward bits, by a predeclared rule on calibration only:
    graph distance >= 3 from q59 and q75 (no direct or next-neighbour coupling), not adjacent
    to each other, lowest measurement error, then longest T1; within Phase 6's register
    (``n_circuit`` qubits)."""
    edges = backend.coupling_map.get_edges()
    n = backend.num_qubits
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def dist_from(src: int) -> dict[int, int]:
        d, frontier = {src: 0}, [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        nxt.append(v)
            frontier = nxt
        return d

    d59, d75 = dist_from(OUTPUT), dist_from(MEMORY)
    tgt = backend.target
    rows = []
    for q in range(n_circuit):
        if min(d59.get(q, 99), d75.get(q, 99)) < 3:
            continue
        props = tgt["measure"].get((q,))
        err = None if props is None else props.error
        if err is None:
            continue
        rows.append({"qubit": q, "measure_error": err, "t1": tgt.qubit_properties[q].t1})
    rows.sort(key=lambda r: (r["measure_error"], -(r["t1"] or 0)))
    pick = [rows[0]["qubit"]]
    for r in rows[1:]:
        if r["qubit"] not in adj[pick[0]]:
            pick.append(r["qubit"])
            break
    nearest = min(min(d59.get(q, 99), d75.get(q, 99)) for q in pick)
    return {
        "rule": "distance >= 3 from q59 and q75; not adjacent to each other; lowest measure "
        "error, then longest T1; qubits 0-75 (Phase 6's register); FakeKingston's stale "
        "calibration, to be re-run on the live calibration at Stage 4",
        "chosen": pick,
        "distance_to_core": nearest,
        "top10": rows[:10],
        "candidates": len(rows),
    }


def export(cfg: dict[str, Any]) -> None:
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    sd = PC.sdir(cfg)
    (sd / "qasm").mkdir(parents=True, exist_ok=True)
    backend = FakeKingston()
    tgt = backend.target
    p6 = REPO_ROOT / cfg["experiment"]["phase6_dir"]
    full = qasm3.loads((p6 / "circuit.qasm").read_text())
    length = cfg["analysis"]["length"]
    arms = Q.parse_arms(cfg["arms"])
    anc = select_ancillas(backend, full.num_qubits)
    pair = tuple(anc["chosen"])
    base = isa(full, backend, 0)
    base_dur = base.estimate_duration(tgt, unit="s")
    base_x = sum(1 for n, _ in q75_ops(base) if n == "x")
    rows, hashes = [], {}
    write = set(cfg["export"]["qasm_arms"])
    controls = set(cfg["export"]["control_arms"])
    for name, arm in arms.items():
        qc = build(full, arm, length, "physical", pair)
        n_echo = max(sum(1 for e in p if e.startswith("e")) for p in arm.programs)
        exp_x = (2 * n_echo if arm.schedule != "random" else 2 * 2 * n_echo) + base_x / length
        row: dict[str, Any] = {"arm": name, "schedule": arm.schedule}
        for level in (0, 1, 2, 3):
            tq = isa(qc, backend, level)
            chk = slot_checks(tq, exp_x, length)
            row[f"L{level}_x_per_step"] = chk["x_per_step"]
            row[f"L{level}_barriers"] = chk["barriers_q75"]
            row[f"L{level}_q75_ops_equal_level0"] = None
            if level == 0:
                ref = q75_ops(tq)
                text = qasm3.dumps(tq)
                hashes[name] = hashlib.sha256(text.encode()).hexdigest()
                if name in write:
                    (sd / "qasm" / f"{name}.qasm").write_text(text)
                row["expected_x_per_step"] = exp_x
                row["L0_rz_count"] = chk["rz_count"]
                row["L0_if_else_blocks"] = chk["if_else_blocks"]
                try:
                    row["duration_s"] = tq.estimate_duration(tgt, unit="s")
                    row["duration_minus_baseline_s"] = row["duration_s"] - base_dur
                except Exception as exc:  # control flow: no static duration
                    row["duration_s"] = None
                    row["duration_note"] = f"{type(exc).__name__}: {exc}"[:160]
            else:
                row[f"L{level}_q75_ops_equal_level0"] = q75_ops(tq) == ref
        if arm.schedule != "random" and name in controls:
            try:
                sq = isa(qc, backend, 0, "alap")
                row["alap_duration_s"] = sq.estimate_duration(tgt, unit="s")
                row["alap_minus_baseline_s"] = row["alap_duration_s"] - base_dur
                row["alap_x_per_step"] = slot_checks(sq, exp_x, length)["x_per_step"]
            except Exception as exc:
                row["alap_note"] = f"{type(exc).__name__}: {exc}"[:160]
            for mode in ("unfenced", "virtual"):
                vq = build(full, arm, length, mode, pair)
                for level in (0, 1, 3):
                    tq = isa(vq, backend, level)
                    ops = q75_ops(tq)
                    row[f"{mode}_L{level}_x_per_step"] = sum(1 for n, _ in ops if n == "x") / length
                    row[f"{mode}_L{level}_rz_count"] = sum(1 for n, _ in ops if n == "rz")
        rows.append(row)
        print(f"export: {name}", flush=True)
    ok = all(
        r[f"L{lv}_x_per_step"] == r["expected_x_per_step"] for r in rows for lv in (0, 1, 2, 3)
    )
    q = lambda i: tgt.qubit_properties[i]  # noqa: E731
    device = {
        "backend": backend.name,
        "dynamic_circuits": "if_else" in tgt.operation_names,
        "q75": {
            "t1": q(MEMORY).t1,
            "t2": q(MEMORY).t2,
            "x_error": tgt["x"][(MEMORY,)].error,
            "x_duration_s": tgt["x"][(MEMORY,)].duration,
            "measure_s": tgt["measure"][(MEMORY,)].duration,
            "reset_s": tgt["reset"][(MEMORY,)].duration,
        },
        "ancillas": {
            str(a): {
                "t1": q(a).t1,
                "t2": q(a).t2,
                "measure_error": tgt["measure"][(a,)].error,
                "measure_s": tgt["measure"][(a,)].duration,
                "sx_error": tgt["sx"][(a,)].error,
            }
            for a in pair
        },
        "baseline_duration_s": base_dur,
    }
    save_json(sd / "device.json", device)
    save_json(
        sd / "circuits.json",
        {
            "meta": run_metadata(cfg, 0),
            "all_physical_insertions_survive_levels_0_3": ok,
            "ancillas": anc,
            "baseline_duration_s": base_dur,
            "baseline_x_on_q75_per_step": base_x / length,
            "rows": rows,
            "qasm_sha256": hashes,
        },
    )
    PC.write_csv(sd / "circuits.csv", [PC.jround(r) for r in rows])
    print(json.dumps({"survive": ok, "ancillas": anc["chosen"]}, indent=1))


# -- #34: the controls tier (direct q75 population benchmark, Knee circuits) ---------------------


def _prep(qc, q, state: str) -> None:
    """``qmem.coherence``'s ``prep<s>`` after a reset: |0>, |1>, |+> = Ry(pi/2)|0>,
    |+i> = Rx(-pi/2)|0>."""
    if state == "1":
        qc.x(q)
    elif state == "+":
        qc.ry(math.pi / 2, q)
    elif state == "+i":
        qc.rx(-math.pi / 2, q)


def benchmark_circuit(
    n_qubits: int, name: str, state: str, reps: int, spacing_s: float, anc, static: bool = False
):
    """q75 alone (``qmem.coherence.benchmark``): prepare ``state``, then ``reps`` times the
    intervention and ``spacing_s`` of idle, then read q75. ``e<k>`` fenced echoes; ``R2`` an
    ancilla in |+> measured, then ``e4`` if its bit is 1 else ``e0``, then ``e0`` (the arm's
    pulse count); ``A2`` a mid-circuit measurement of q75 (outcome recorded, not used).
    ``static``: R2's ``if_else`` replaced by its ``e0`` branch (a duration twin only)."""
    from qiskit import ClassicalRegister, QuantumCircuit

    regs = [ClassicalRegister(1, "out")]
    if name == "R2":
        regs.append(ClassicalRegister(reps, "b1"))
    if name == "A2":
        regs.append(ClassicalRegister(reps, "m"))
    qc = QuantumCircuit(n_qubits)
    for r in regs:
        qc.add_register(r)
    q = qc.qubits[MEMORY]
    _prep(qc, q, state)
    for r in range(reps):
        if name.startswith("e"):
            _echo(qc, q, _phase(name))
        elif name == "R2":
            a = qc.qubits[anc[0]]
            qc.reset(a)
            qc.h(a)
            qc.measure(a, regs[1][r])
            if static:
                _echo(qc, q, 0.0)
            else:
                with qc.if_test((regs[1][r], 1)) as else_:
                    _echo(qc, q, math.pi)
                with else_:
                    _echo(qc, q, 0.0)
            _echo(qc, q, 0.0)
        elif name == "A2":
            qc.measure(q, regs[1][r])
        else:
            raise ValueError(name)
        qc.delay(spacing_s, q, unit="s")
    qc.measure(q, regs[0][0])
    return qc


def knee_circuit(full, state: str, program, length: int):
    """The PMD circuit with q75 reset and prepared in ``state`` in step 0's slot, then the
    intervention ``program`` (``qmem.coherence.knee_disturbances``); later steps unchanged."""
    from qiskit import ClassicalRegister

    regs = {"m": ClassicalRegister(1, "m")} if "meas" in program else {}

    def op(qc, t: int) -> None:
        if t != 0:
            return
        q = qc.qubits[MEMORY]
        qc.reset(q)
        _prep(qc, q, state)
        for e in program:
            if e == "meas":
                qc.measure(q, regs["m"][0])
            else:
                _echo(qc, q, _phase(e))

    return A.restructure(
        full, full.qubits[MEMORY], full.qubits[OUTPUT], op, extra_regs=list(regs.values())
    )


def benchmark_shots(cfg: dict[str, Any], ctx) -> dict[str, Any]:
    """Shots per benchmark circuit: the TOST of each intervention against the e0 sham at the
    same preparation, margin ``[controls] benchmark_margin`` in P(1), 80% power, from the
    model's P(1) (binomial SDs); the maximum over circuits, rounded up to 1,000."""
    cc, bc = cfg["controls"], cfg["benchmark"]
    pc = ctx.quantum.pieces
    b = Q.Benchmark(ctx.t1_q75, ctx.t2_q75, pc.p1_q75, bc["spacing_s"], ctx.quantum.readout)
    z = 1.6448536269514722 + 1.2815515655446004  # TOST: alpha 0.05 per bound, 80% power
    rows, worst = [], 0.0
    for name in cc["benchmark_interventions"]:
        for st in cc["benchmark_states"]:
            p = Q.benchmark(name, st, cc["benchmark_reps"], b, ctx.quantum.decay)["p1"]
            p0 = Q.benchmark("e0", st, cc["benchmark_reps"], b)["p1"]
            sd = math.sqrt(p * (1 - p) + p0 * (1 - p0))
            delta = cc["benchmark_margin"] - abs(p - p0)
            n = (z * sd / delta) ** 2 if delta > 0 else math.inf
            rows.append({"intervention": name, "state": st, "p1": p, "p1_e0": p0, "shots_80": n})
            if name != "e0":
                worst = max(worst, n)
    shots = int(math.ceil(worst / 1000) * 1000)
    return {"rows": rows, "shots_per_circuit": shots}


def export_controls(cfg: dict[str, Any]) -> None:
    """QASM, hashes and durations of the controls tier (FakeKingston only)."""
    from qiskit import qasm3
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    sd = PC.sdir(cfg)
    qd = sd / "qasm" / "controls"
    qd.mkdir(parents=True, exist_ok=True)
    backend = FakeKingston()
    tgt = backend.target
    full = qasm3.loads((REPO_ROOT / cfg["experiment"]["phase6_dir"] / "circuit.qasm").read_text())
    circuits = json.loads((sd / "circuits.json").read_text())
    anc = tuple(circuits["ancillas"]["chosen"])
    cc, ds = cfg["controls"], cfg["design"]
    ctx = PC.Ctx(cfg)
    shots_b = benchmark_shots(cfg, ctx)
    built = []
    for name in cc["benchmark_interventions"]:
        for st in cc["benchmark_states"]:
            qc = benchmark_circuit(
                full.num_qubits, name, st, cc["benchmark_reps"], cfg["benchmark"]["spacing_s"], anc
            )
            twin = benchmark_circuit(
                full.num_qubits,
                name,
                st,
                cc["benchmark_reps"],
                cfg["benchmark"]["spacing_s"],
                anc,
                static=True,
            )
            built.append(
                ("benchmark", f"bench_{name}_{st}", qc, shots_b["shots_per_circuit"], twin)
            )
    for iv, prog in cc["knee_interventions"].items():
        for st in cc["knee_states"]:
            qc = knee_circuit(full, st, prog, cfg["analysis"]["length"])
            built.append(("knee", f"knee_{iv}_{st}", qc, cc["knee_shots"], qc))
    rows, hashes = [], {}
    for kind, label, qc, shots, twin in built:
        tq = isa(qc, backend, 0)
        text = qasm3.dumps(tq)
        hashes[label] = hashlib.sha256(text.encode()).hexdigest()
        (qd / f"{label}.qasm").write_text(text)
        row: dict[str, Any] = {"kind": kind, "circuit": label, "shots": shots}
        row["x_on_q75"] = sum(1 for n, _ in q75_ops(tq) if n == "x")
        try:
            row["duration_s"] = tq.estimate_duration(tgt, unit="s")
            row["duration_note"] = "FakeKingston estimate_duration (level 0)"
        except Exception:  # control flow (R2): the e0-only twin plus the feed-forward time
            # (conservative: the echo waits for the ancilla's readout and the latency)
            dt = isa(twin, backend, 0).estimate_duration(tgt, unit="s")
            ff = ds["ff_measure_s"] + ds["ff_latency_s"]
            row["duration_s"] = dt + cc["benchmark_reps"] * ff
            row["duration_note"] = (
                "twin without if_else plus reps x ([design] ff_measure_s + ff_latency_s)"
            )
        rows.append(row)
        print(f"controls: {label} {row['duration_s'] * 1e6:.1f} us", flush=True)
    save_json(
        sd / "controls.json",
        {
            "meta": run_metadata(cfg, 0),
            "backend": backend.name,
            "ancilla": anc[0],
            "benchmark_shots": shots_b,
            "rows": rows,
            "qasm_sha256": hashes,
        },
    )
    PC.write_csv(sd / "controls.csv", [PC.jround(r) for r in rows])
