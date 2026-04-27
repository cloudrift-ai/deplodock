"""Blockify thread axes — split each ``BIND_THREAD`` axis of a logical
``Tile`` into outer ``BIND_BLOCK`` + inner ``BIND_THREAD`` (extent
``BLOCK_TG``) pair, capped at ``MAX_THREADS_PER_BLOCK`` total threads.

Pre-rewrite::

    Tile(axes=(M:128 THREAD, N:2048 THREAD)):
        ... body ...

Post-rewrite (``BLOCK_TG=16``)::

    Tile(axes=(M_i:16 THREAD, M_o:8 BLOCK, N_i:16 THREAD, N_o:128 BLOCK)):
        ... body[M -> M_o*16 + M_i, N -> N_o*16 + N_i] ...

Idempotent (skips a Tile that already has any ``BIND_BLOCK`` axis).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

BLOCK_TG = 16
MAX_THREADS_PER_BLOCK = 256


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if tile.block_axes:
        return None  # idempotence
    new_tile = _blockify(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _blockify(tile: Tile) -> Tile | None:
    new_axes: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}
    threads = 1
    blockified = False
    for ba in tile.axes:
        if ba.bind != BIND_THREAD:
            new_axes.append(ba)
            continue
        ax = ba.axis
        ext = int(ax.extent)
        # Split when divisible by BLOCK_TG and the inner THREAD slice fits the budget.
        if ext > BLOCK_TG and ext % BLOCK_TG == 0 and threads * BLOCK_TG <= MAX_THREADS_PER_BLOCK:
            outer = Axis(f"{ax.name}_o", ext // BLOCK_TG)
            inner = Axis(f"{ax.name}_i", BLOCK_TG)
            new_axes.append(BoundAxis(axis=inner, bind=BIND_THREAD))
            new_axes.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
            sigma_map[ax.name] = Var(outer.name) * Literal(BLOCK_TG, "int") + Var(inner.name)
            threads *= BLOCK_TG
            blockified = True
        elif threads * ext <= MAX_THREADS_PER_BLOCK:
            # Small enough to fit whole — keep as THREAD without splitting.
            new_axes.append(ba)
            threads *= ext
        else:
            # Out of thread budget — demote whole axis to BLOCK.
            new_axes.append(BoundAxis(axis=ax, bind=BIND_BLOCK))

    if not blockified:
        return None

    sigma = Sigma(sigma_map)
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body)
    return Tile(axes=tuple(new_axes), body=new_body)


def _id(name: str) -> str:
    return name
