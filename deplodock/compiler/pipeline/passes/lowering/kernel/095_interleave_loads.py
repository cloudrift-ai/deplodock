"""Sink Loads in flat compute blocks to just before their first consumer.

After ``010_split_register_axes`` fully unrolls the register tile, the
inner K-loop body emits the matmul cells as a flat sequence:

    for a_BK in 0..BK:
        Load in0  ← B[k, c0]      # FN B-loads
        Load in1  ← B[k, c1]
        Load in2  ← B[k, c2]
        Load in3  ← B[k, c3]
        Load in4  ← A[r0, k]      # FM A-loads
        Load in5  ← A[r1, k]
        ...
        Load in11 ← A[r7, k]
        Assign v0  = in0 * in4    # 32 FMAs
        Assign v1  = in0 * in5
        ...
        Assign v31 = in3 * in11
        Accum  acc0  <- v0        # 32 accumulates
        ...
        Accum  acc31 <- v31

Every Load is hoisted to the top of the block — ptxas has to figure
out load/FMA scheduling across the entire 32-FMA cluster. The article
SGEMM hand-emits each A-load RIGHT BEFORE its consumer FMAs:

    for a_BK in 0..BK:
        Load b0..b3                         # B-cells, used by every row
        Load a0
        v_0 = b0·a0; v_4 = b1·a0; v_8 = b2·a0; v_12 = b3·a0
        Load a1
        v_1 = b0·a1; v_5 = b1·a1; v_9 = b2·a1; v_13 = b3·a1
        ...

which lets ptxas place the LDS adjacent to its consumer FMA without
needing to reschedule across the whole block — the article's
diagnostic blamed the load-FMA scheduling distance for the 5 % gap
against cuBLAS's clustered-LDS template.

This pass approximates that pattern via a peephole "sink each Load to
just before its first consumer" reorder on every body that holds a
Load + Assign cluster. ``Stmt.defines`` / ``Stmt.deps`` carry the SSA
dependency information; we walk forward, identify each Load's first
consumer position, and re-emit:

- Loads with NO consumer in the body stay at the top in original order
  (they're consumed in a sibling / outer scope; sinking them would
  cross the body boundary).
- Loads with consumers in the body get placed immediately before their
  first consumer.
- Non-Load stmts keep their relative order — no semantic reorder of
  Assigns / Accums.

The result is the article's load-FMA-load-FMA interleave when the
underlying iteration order is FM-major; on the (currently default)
FN-major emission the pass still delivers a "B-loads-at-top + sunk
A-loads" pattern that closes most of the LDS-to-consumer-distance gap
because the first consumer of each A-load lands adjacent to its load.

The pass is wired between ``050_vectorize_loads`` and
``100_materialize_tile``: vectorization runs first (otherwise sunk
Loads break the consecutive-Load run vectorize_loads needs) and the
materializer runs after (it lowers the body to KernelOp; sinking has
to happen at TileOp / Loop level where the per-stmt SSA mutation is
still cheap).

A note on observed impact (RTX 5090, nvcc 13.0). The IR-level transform
produces visibly cleaner CUDA source (each smem load adjacent to its
first FMA-consumer rather than all-loads-up-front-then-the-FMAs). On a
2048x2048 fp32 SGEMM at the deployable opt level (``-Xcicc -O2`` / ``-O3``)
and the *optimal* wide tile (``FM=FN=8``, a 64-cell register tile) it is a
small but consistent **~2 %** win: ~431 us sunk vs ~440 us flat, reproduced
across ``-O2``, ``-O3``, and the default. The effect is tile- and
opt-level-dependent: at ``-Xcicc -O0`` the unoptimized path prefers the flat
layout (~5 % slower sunk), at ``-O1`` it is neutral, and on narrow tiles
(e.g. ``FN=4``) it washes out as the toolchain reschedules across the flat
LDS+FFMA window. So on the configuration you would actually ship it earns
~2 %, which (with the source-level legibility) is why the pass is **on by
default**. Deliberately-bad register-pressured tiles (128- to 512-cell grids)
do not change the picture: at deployable ``-O`` they stay in the ±2 % noise
band, and the only flicker of a win is a *noisy* ~3-4 % at ``-Xcicc -O0`` on a
moderately-pressured tile — gone by ``-O1`` and absent once the tile is large
enough to be spill-bound.

``cuobjdump`` shows why. The interleaved and hoisted forms carry the *same
instruction mix* at every level; at ``-Xptxas -O2`` / ``-O3`` ptxas reschedules
both to nearly the same SASS (a ~120-line cosmetic reorder of ~424
instructions), and only at ``-Xptxas -O0`` does the source order survive (the
unscheduled ~1096-instruction SASS keeps the loads where the source put them).
So, as with vectorization, the only flag that makes the transform change the
SASS is ``-Xptxas -O0``, which is never deployable. ptxas hoists loads early
and reorders by its own latency heuristics, so the source-emitted order is at
most a tie-breaker at ``-O2`` + (mechanism:
https://forums.developer.nvidia.com/t/ptx-instructions-are-reordered/197973 ).

``INTERLEAVE_LOADS`` is therefore *not* a search dimension — only ``True`` is
enumerated, so the autotuner never forks on it and the knob set stays small.
``DEPLODOCK_INTERLEAVE_LOADS=0`` remains as a manual override for inspecting
the flat layout or for a workload where it measures faster.

Idempotent — a body already in interleaved form is a no-op.
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.stmt.leaves import Assign, Load
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

INTERLEAVE_LOADS = Knob(
    "INTERLEAVE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Sink each Load to just before its first SSA-consumer in flat compute blocks.",
)


def rewrite(root: Node) -> TileOp | None:
    op: TileOp = root.op
    # Idempotence: the policy is recorded as the INTERLEAVE_LOADS knob (every path
    # stamps it now), so a re-scan of the rebound op skips here.
    if INTERLEAVE_LOADS.name in op.knobs:
        raise RuleSkipped("INTERLEAVE_LOADS already decided (idempotence via knob)")
    # Only ``True`` is enumerated, so the autotuner never forks on this knob;
    # ``DEPLODOCK_INTERLEAVE_LOADS=0`` still pins ``False`` (``narrow`` honours an env
    # pin authoritatively, even when it is not in the candidate set).
    if not INTERLEAVE_LOADS.narrow((True,))[0]:
        return TileOp(body=op.body, name=op.name, knobs={**op.knobs, INTERLEAVE_LOADS.name: False})
    # Stamp the policy (True) even when no cluster benefits — the realized config
    # records that interleaving was enabled, keeping a uniform knob set.
    new_body, _changed = _walk(op.body)
    return TileOp(body=new_body, name=op.name, knobs={**op.knobs, INTERLEAVE_LOADS.name: True})


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
