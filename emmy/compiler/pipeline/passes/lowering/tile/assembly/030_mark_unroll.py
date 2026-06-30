"""Mark small loop nests for ``#pragma unroll`` (R5 post-assemble).

For each ``SerialTile`` / ``StridedTile`` nest in the assembled ``TileOp`` body,
compute the **total trip count** (product of axis extents from the outermost loop
down through its inner loops). If the total is below ``_MAX_UNROLL_TRIPS`` mark
``unroll=True`` on every loop in that chain, so the CUDA renderer emits
``#pragma unroll``.

Runs after ``020_peel`` so the software-pipelined K loop's ``stage_inner`` reduce
(the FMA chain — e.g. the 32-iter BK reduce of the SGEMM hero tile) is unrolled,
which is what gives ptxas the register-resident operand reuse + FMA ILP the
hand-tuned kernels rely on (the article's ``TM=26`` kernel runs at 255 regs, not
the ~126 a rolled inner loop produces).

Ported back from the pre-block-DAG ``tile/090_mark_unroll`` (deleted in 7f764b26
with the legacy scheduling passes; the rewrite never re-added an equivalent, so
no kernel was getting ``#pragma unroll``). Deterministic, no fork: a per-nest
threshold, not a search dim.

Idempotent: a loop already ``unroll=True`` stays so; returns ``RuleSkipped`` when
nothing needs flipping.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.ir.tile.ir import GridTile, RegisterTile, SerialTile, StridedTile, ThreadTile, TileOp, WarpTile
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_MAX_UNROLL_TRIPS = 64


def rewrite(ctx: Context, root: Node) -> TileOp:  # noqa: ARG001 — ctx required by rule dispatch signature
    new_body, changed = _walk_body(root.op.body)
    if not changed:
        raise RuleSkipped(f"no Loop nest with total trips <= {_MAX_UNROLL_TRIPS} to mark")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk_body(body: Body) -> tuple[Body, bool]:
    new_body: list[Stmt] = []
    changed = False
    for s in body:
        new_s, c = _walk_stmt(s)
        new_body.append(new_s)
        changed = changed or c
    return Body(tuple(new_body)), changed


def _walk_stmt(s: Stmt) -> tuple[Stmt, bool]:
    if isinstance(s, (SerialTile, StridedTile)):
        should_unroll = _nest_trips(s) <= _MAX_UNROLL_TRIPS
        new_body, inner_changed = _walk_body(s.body)
        if should_unroll and not s.unroll:
            return replace(s, body=new_body, unroll=True), True
        if new_body != s.body:
            return replace(s, body=new_body), inner_changed
        return s, False
    if isinstance(s, (GridTile, ThreadTile, RegisterTile, WarpTile)):
        new_body, c = _walk_body(s.body)
        if c:
            return s.with_bodies((new_body,)), True
        return s, False
    nested = s.nested() if hasattr(s, "nested") else ()
    if nested:
        new_bodies, c = [], False
        for b in nested:
            nb, bc = _walk_body(b)
            new_bodies.append(nb)
            c = c or bc
        if c:
            return s.with_bodies(tuple(new_bodies)), True
    return s, False


def _nest_trips(loop) -> int:  # noqa: ANN001
    """Unrolled trip count: ``axis.extent`` × the sum of inner loop trips (siblings
    each run once per outer iteration). A symbolic extent reports a placeholder above
    the threshold so the loop is left rolled (no compile-time bound to amortize)."""
    if not loop.axis.extent.is_static:
        return 10**9
    total = loop.axis.extent.as_static()
    inner = sum(_nest_trips(s) for s in loop.body if isinstance(s, (SerialTile, StridedTile)))
    if inner:
        total *= inner
    return total
