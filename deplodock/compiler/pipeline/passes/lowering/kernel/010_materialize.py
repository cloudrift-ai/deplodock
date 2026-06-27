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
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    bound = Tile(axes=tuple(tile.grid_axes), body=tile.body)
    return KernelOp(body=Body((bound,)), name=tile.name)
