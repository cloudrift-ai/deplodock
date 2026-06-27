"""Schedule a ``LoopOp`` onto a tile grid → ``TileOp``.

Second of the two enumeration steps — recognition (``010_recognize``) normalized
the reduce carriers to ``Monoid``s; this only chooses the *schedule*. Two kinds
map onto the per-cell (one-thread-per-output-cell) tier:

- **no fold** (every axis free): the whole iteration space maps onto the thread
  grid; the per-cell body is just the leaf compute.
- **a fold** (a reduce ``Loop`` carrying a ``ReduceCarrier``): only the free axes
  *enclosing* the fold map onto the grid; the reduce loop — and any epilogue /
  output sweep that shares its accumulator — stays serial inside the per-cell
  body. One thread owns one output row's fold.

Both peel the same way (``_peel``): a leading loop-invariant prefix plus the
outer chain of single-child free loops, stopping at the first reduce loop or
branch — exactly where the per-cell body begins. Because the carrier is already
the unified ``Monoid`` (degenerate twist for a plain reduction, the max twist for
online softmax), the schedule and the downstream lowering never branch on which.

A contraction (``SEMIRING`` — matmul) is left un-lowered for now; its schedule
arrives with the matmul tier (see ``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Body, Loop
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(grid_axes, per_cell_stmts)``.

    A leading run of non-``Loop`` stmts (loop-invariant loads hoisted above the
    nest) is pulled into the per-cell body; then the outer chain of single-child
    **free** loops becomes ``grid_axes`` (the thread grid). The chain stops at the
    first reduce loop or branch — everything from there down is the per-cell body
    (the fold and its epilogue / output sweep), run serially by one thread. Each
    grid thread re-runs the invariant prefix (idempotent for constant / pure
    loads)."""
    stmts = list(body)
    i = 0
    while i < len(stmts) and not isinstance(stmts[i], Loop):
        i += 1
    prefix, rest = stmts[:i], stmts[i:]
    axes = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        axes.append(cur[0].axis)
        cur = list(cur[0].body)
    return axes, prefix + list(cur)


def _reduce_loops(stmts) -> list[Loop]:
    """Every reduce ``Loop`` reachable in ``stmts`` (deep)."""
    out: list[Loop] = []
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            out.append(s)
        for b in s.nested():
            out.extend(_reduce_loops(b))
    return out


def rewrite(match: Match, root: Node) -> TileOp | None:
    loop: LoopOp = root.op
    axes, cell = _peel(loop.body)
    folds = _reduce_loops(cell)
    if AlgebraKind.SEMIRING in {f.algebra_kind for f in folds}:
        raise RuleSkipped("contraction (semiring) — its schedule is not built yet")
    if not axes and not folds:
        raise RuleSkipped("no work to schedule")
    return TileOp(body=Body(tuple(cell)), name=loop.name, grid_axes=tuple(axes))
