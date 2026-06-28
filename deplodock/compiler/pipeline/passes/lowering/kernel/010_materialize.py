"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the ``TileOp``'s ``grid_axes`` to GPU threads — one thread per output cell
— by wrapping the per-cell body in a single :class:`Tile` kernel-IR primitive,
which emits the linear-thread decode (``_gid = blockIdx.x·blockDim.x +
threadIdx.x``, the bounds guard, the per-axis index decode) around the body.

The step is **algebra-generic**: it only realizes the geometry. A kernel that
carries a combine would arrive with the same shape — its fold (a serial ``Loop``
+ a carrier ``Accum`` / ``Mma`` / ``Monoid``) sits inside the per-cell body and renders through the
carrier unchanged — so other kernel kinds reuse this materializer once their
schedule is enumerated (see ``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.stmt import Body, Write
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.ir.tile.ops import lower
from deplodock.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", TileOp)]


def _has_write(stmts: list[Stmt]) -> bool:
    """Any ``Write`` reachable in ``stmts`` (deep — a projection's output sweep nests
    its ``Write`` inside a per-cell ``Loop``)."""
    for s in stmts:
        if isinstance(s, Write):
            return True
        if any(_has_write(list(b)) for b in s.nested()):
            return True
    return False


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    op = tile.op
    stmts = lower(op)
    # Output-store glue: a node that produces its finalized output as an SSA value
    # (``op.out``) — a bare reduction (Monoid / Semiring) or a flash-style projection Map
    # whose body is pure pointwise — needs that value stored to the output buffer at the
    # grid cell (index = the grid axes, one cell per output element). A kernel whose lowered
    # body already carries a ``Write`` (a pure pointwise Map, or a projection that writes
    # through its own output sweep) needs no glue.
    needs_store = not _has_write(stmts)
    if needs_store:
        index = tuple(Var(ax.name) for ax in tile.grid_axes)
        stmts = [*stmts, Write(output=root.output.name, index=index, value=op.out)]
    bound = Tile(axes=tuple(tile.grid_axes), body=Body(tuple(stmts)))
    return KernelOp(body=Body((bound,)), name=tile.name)
