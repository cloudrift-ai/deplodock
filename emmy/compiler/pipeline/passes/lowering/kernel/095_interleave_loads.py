"""Sink Loads in flat compute blocks to just before their first consumer.

When a register-tile matmul body emits its cells as a flat sequence (all
Loads hoisted to the top, then the FMAs), ptxas has to schedule loads/FMAs
across the whole cluster. The article SGEMM hand-emits each A-load right
before its consumer FMAs, which lets ptxas place the LDS adjacent to its
consumer without rescheduling across the block.

This pass approximates that pattern via a peephole "sink each Load to just
before its first consumer" reorder on every body that holds a Load + Assign
cluster. ``Stmt.defines`` / ``Stmt.deps`` carry the SSA dependency
information; we walk forward, identify each Load's first consumer position,
and re-emit:

- Loads with NO consumer in the body stay at the top in original order
  (they're consumed in a sibling / outer scope; sinking them would cross
  the body boundary).
- Loads with consumers in the body get placed immediately before their
  first consumer.
- Non-Load stmts keep their relative order — no semantic reorder of
  Assigns / Accums.

Observed impact: an IR-legibility transform; ptxas reschedules both forms to
nearly the same SASS at deployable opt levels, so ``INTERLEAVE_LOADS`` is
*not* a search dimension — only ``True`` is enumerated.
``EMMY_INTERLEAVE_LOADS=0`` remains a manual override.

Idempotent — a body already in interleaved form is a no-op.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.ir.stmt.leaves import Assign, Load
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import INTERLEAVE_LOADS

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    # Idempotence: the policy is recorded as the INTERLEAVE_LOADS knob, so a
    # re-scan of the rebound op skips here.
    if INTERLEAVE_LOADS.name in op.knobs:
        raise RuleSkipped("INTERLEAVE_LOADS already decided (idempotence via knob)")
    # Only ``True`` is enumerated, so the autotuner never forks on this knob;
    # ``EMMY_INTERLEAVE_LOADS=0`` still pins ``False``.
    if not INTERLEAVE_LOADS.narrow((True,))[0]:
        return KernelOp(body=op.body, name=op.name, knobs={**op.knobs, INTERLEAVE_LOADS.name: False})
    # Stamp the policy (True) even when no cluster benefits — the realized
    # config records that interleaving was enabled, keeping a uniform knob set.
    new_body, _changed = _walk(op.body)
    return KernelOp(body=new_body, name=op.name, knobs={**op.knobs, INTERLEAVE_LOADS.name: True})


def _walk(body: Body) -> tuple[Body, bool]:
    """Recurse into block stmts' nested bodies first, then check
    whether this body has a Load + Assign cluster of its own — if so,
    apply the sink-loads peephole."""
    new_stmts: list[Stmt] = []
    nested_changed = False
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                nested_changed = True
        new_stmts.append(s)
    rebuilt = Body(tuple(new_stmts))
    if not _has_load_and_assign(rebuilt):
        return rebuilt, nested_changed
    sunk = _sink_loads(rebuilt)
    return sunk, nested_changed or tuple(sunk) != tuple(rebuilt)


def _has_load_and_assign(body: Body) -> bool:
    has_load = False
    has_assign = False
    for s in body:
        if isinstance(s, Load):
            has_load = True
        elif isinstance(s, Assign):
            has_assign = True
        if has_load and has_assign:
            return True
    return False


def _sink_loads(body: Body) -> Body:
    """Re-emit ``body`` with each Load placed just before its first
    consumer. Preserves the relative order of all non-Load stmts and
    the original order of Loads that share the same first-consumer
    position."""
    stmts = tuple(body)
    if not stmts:
        return body

    # For each Load index, find its first consumer position (or None).
    # A consumer is any stmt downstream whose ``deps()`` references an
    # SSA name this Load ``defines()``.
    first_consumer: dict[int, int | None] = {}
    for i, s in enumerate(stmts):
        if not isinstance(s, Load):
            continue
        names = frozenset(s.defines())
        first_consumer[i] = None
        for j in range(i + 1, len(stmts)):
            if names.intersection(stmts[j].deps()):
                first_consumer[i] = j
                break

    emitted: set[int] = set()
    out: list[Stmt] = []

    # Loads with no consumer in this body stay at the top (consumed by
    # a sibling / outer scope; sinking them would cross boundaries).
    for i, s in enumerate(stmts):
        if isinstance(s, Load) and first_consumer.get(i) is None:
            out.append(s)
            emitted.add(i)

    # Walk non-Load stmts in original order. Before each, emit any
    # pending Loads whose first consumer is this position.
    for j, sj in enumerate(stmts):
        if isinstance(sj, Load):
            continue
        for i, si in enumerate(stmts):
            if i in emitted or not isinstance(si, Load):
                continue
            if first_consumer.get(i) == j:
                out.append(si)
                emitted.add(i)
        out.append(sj)

    # Safety: emit any unclaimed Loads at the end (shouldn't fire, but
    # guarantees we never drop a stmt).
    for i, si in enumerate(stmts):
        if isinstance(si, Load) and i not in emitted:
            out.append(si)
            emitted.add(i)

    assert len(out) == len(stmts), f"sink_loads dropped stmts: {len(out)} vs {len(stmts)}"
    return Body(tuple(out))
