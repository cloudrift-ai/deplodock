"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the ``TileOp``'s ``grid_axes`` to GPU threads — one thread per output cell
— by wrapping the per-cell body in a single :class:`Tile` kernel-IR primitive,
which emits the linear-thread decode (``_gid = blockIdx.x·blockDim.x +
threadIdx.x``, the bounds guard, the per-axis index decode) around the body.

The step is **algebra-generic**: it only realizes the geometry. A kernel that
carries a combine would arrive with the same shape — its fold (a serial ``Loop``
+ ``ReduceCarrier``) sits inside the per-cell body and renders through the
carrier unchanged — so other kernel kinds reuse this materializer once their
schedule is enumerated (see ``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.stmt import Body, Write
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.ir.tile.ops import Reduce, lower
from deplodock.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    stmts = lower(tile.op)
    # Output-store glue: a Reduce produces its finalized output as an SSA value
    # (``op.out``) — store it to the kernel's output buffer at the grid cell. The
    # output index IS the grid axes (one cell per output element); the buffer is the
    # graph node's output. A Map carries its own Write, so it needs no glue.
    if isinstance(tile.op, Reduce):
        index = tuple(Var(ax.name) for ax in tile.grid_axes)
        stmts = [*stmts, Write(output=root.output.name, index=index, value=tile.op.out)]
    bound = Tile(axes=tuple(tile.grid_axes), body=Body(tuple(stmts)))
    return KernelOp(body=Body((bound,)), name=tile.name)
