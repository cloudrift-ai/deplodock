"""Mark small loop nests for ``#pragma unroll``.

Currently a no-op: the ``unroll`` annotation is stripped by
``canonicalize_free_axis_order`` (which rebuilds free Loops with
default ``unroll=False``), and re-applying it surfaces a CUDA codegen
issue (misaligned-address errors on certain TMA / coalesced-load
shapes). The pass stays in the pipeline as a structural placeholder so
restoring it is just a matter of re-enabling the body — see git
history for the original ``_walk_body`` logic.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_MAX_UNROLL_TRIPS = 64


def rewrite(root: Node) -> Graph | None:
    raise RuleSkipped("mark_unroll disabled — see module docstring")


def _walk_body(body: Body) -> tuple[Body, bool]:
    """Walk a body, marking small loop nests. Returns (new_body, changed)."""
    new_body: list[Stmt] = []
    changed = False
    for s in body:
        new_s, c = _walk_stmt(s)
        new_body.append(new_s)
        changed = changed or c
    return tuple(new_body), changed


def _walk_stmt(s: Stmt) -> tuple[Stmt, bool]:
    if isinstance(s, (Loop, StridedLoop)):
        trips = _nest_trips(s)
        should_unroll = trips <= _MAX_UNROLL_TRIPS
        new_body, inner_changed = _walk_body(s.body)
        if should_unroll and not s.unroll:
            return replace(s, body=new_body, unroll=True), True
        if new_body != s.body:
            return replace(s, body=new_body), inner_changed
        return s, False
    if isinstance(s, Tile):
        new_body, c = _walk_body(s.body)
        if c:
            return Tile(axes=s.axes, body=new_body), True
        return s, False
    if hasattr(s, "body") and hasattr(s, "nested"):
        # Cond and other block stmts — recurse uniformly via existing fields.
        # Rebuilding generically is awkward; keep the targeted Tile/Loop path
        # above and pass other stmts through (they don't carry unroll).
        return s, False
    return s, False


def _nest_trips(loop: Loop | StridedLoop) -> int:
    """Trip count when ``loop`` is unrolled: ``axis.extent`` × the sum
    of inner loop trip counts. Siblings sum (each runs once per outer
    iteration) and a single child reduces to plain product. Used to
    estimate the unrolled body size."""
    total = int(loop.axis.extent)
    inner_trips = sum(_nest_trips(s) for s in loop.body if isinstance(s, (Loop, StridedLoop)))
    if inner_trips:
        total *= inner_trips
    return total
