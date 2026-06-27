"""Schedule a ``LoopOp`` onto a tile grid → ``TileOp``.

Reads the kernel's algebraic kind (``Loop.algebra_kind``) and picks a schedule.
The first kind the skeleton schedules is a kernel with no fold — every free axis
is parallel, so the schedule maps the whole iteration space onto the thread grid
(one thread per output cell). We strip the free ``Loop`` nest (those axes
*become* the grid) and keep the per-cell body — the leaf ``Load`` / ``Assign`` /
``Write`` plus any loop-invariant load — as the ``TileOp`` body; the algebra
stays implicit there and is read back via ``TileOp.algebra_kind``.

A kernel that carries a combine (a reduce ``Loop`` + ``ReduceCarrier``) is left
un-lowered for now — its schedule is added as the skeleton grows, reusing the
same op by supplying the combine (see ``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Body, Loop
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def _flatten(body: Body) -> tuple[list, list[Stmt]]:
    """Split a free-axis loop nest into ``(grid_axes, per_cell_stmts)``.

    Depth-first in source order: each free ``Loop`` contributes its axis (the
    thread grid) and we recurse into it; every other stmt — the per-cell body and
    any hoisted loop-invariant load — joins the flat cell body. Source order is
    preserved, so SSA defs precede uses; each thread re-runs an invariant
    (idempotent for constant loads / pure compute)."""
    axes = []
    cell: list[Stmt] = []

    def walk(b: Body) -> None:
        for s in b:
            if isinstance(s, Loop):
                axes.append(s.axis)
                walk(s.body)
            else:
                cell.append(s)

    walk(body)
    return axes, cell


def rewrite(match: Match, root: Node) -> TileOp | None:
    loop: LoopOp = root.op
    # Dispatch on the kernel's algebraic kind. ``reduce_axis_names`` is the
    # recursive set of fold axes (a contraction's reduce loop nests *under* the
    # free loops), so this rejects any kernel that carries a combine — a fold
    # axis must never be flattened into the parallel grid.
    if loop.reduce_axis_names:
        kinds = sorted({s.algebra_kind.value for s in loop.body.iter() if isinstance(s, Loop) and s.is_reduce})
        raise RuleSkipped(f"kernel carries a combine ({', '.join(kinds)}) — its schedule is not built yet")
    axes, cell = _flatten(loop.body)
    if not axes:
        raise RuleSkipped("no free-axis space to tile onto the thread grid")
    return TileOp(body=Body(tuple(cell)), name=loop.name, grid_axes=tuple(axes))
