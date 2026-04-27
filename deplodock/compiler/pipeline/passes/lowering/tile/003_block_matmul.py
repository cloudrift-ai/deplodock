"""Generic blockify rule — splits each thread axis into outer-BLOCK +
inner-THREAD pair, enabling ``004_stage_inputs`` to cache reused operands
in smem.

Pre-rewrite::

    Tile(axes=(t1 THREAD, ..., tk THREAD)):
        ... body ...

Post-rewrite::

    Tile(axes=(t1_i THREAD, ..., tk_i THREAD, t1_o BLOCK, ..., tk_o BLOCK)):
        ... body with t<i> -> t<i>_o * _BLOCK_TG + t<i>_i ...

Each thread axis with extent divisible by ``_BLOCK_TG`` is split. Axes
that aren't divisible stay as THREAD axes whole (passthrough). The total
per-block thread count is capped at ``_MAX_THREADS_PER_BLOCK``; if the
budget is exceeded, additional axes are bound to BLOCK whole instead of
THREAD.

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.block_axes`` is empty (idempotence).
- At least one thread axis is divisible by ``_BLOCK_TG``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import (
    BIND_BLOCK,
    BIND_THREAD,
    Axis,
    BoundAxis,
)
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Stmt, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_BLOCK_TG = 16
_MAX_THREADS_PER_BLOCK = 1024


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
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
    splittable = [a for a in tile.thread_axes if int(a.extent) % _BLOCK_TG == 0 and int(a.extent) >= _BLOCK_TG]
    if not splittable:
        return None

    new_axes: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}
    threads_so_far = 1

    for axis in tile.thread_axes:
        if axis in splittable and threads_so_far * _BLOCK_TG <= _MAX_THREADS_PER_BLOCK:
            outer = Axis(f"{axis.name}_o", int(axis.extent) // _BLOCK_TG)
            inner = Axis(f"{axis.name}_i", _BLOCK_TG)
            new_axes.append(BoundAxis(axis=inner, bind=BIND_THREAD))
            new_axes.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
            sigma_map[axis.name] = Var(outer.name) * Literal(_BLOCK_TG, "int") + Var(inner.name)
            threads_so_far *= _BLOCK_TG
        elif threads_so_far * int(axis.extent) <= _MAX_THREADS_PER_BLOCK:
            new_axes.append(BoundAxis(axis=axis, bind=BIND_THREAD))
            threads_so_far *= int(axis.extent)
        else:
            new_axes.append(BoundAxis(axis=axis, bind=BIND_BLOCK))

    sigma = Sigma(sigma_map)
    new_body: tuple[Stmt, ...] = tuple(s.rewrite(_id, sigma) for s in tile.body)

    return Tile(axes=tuple(new_axes), body=new_body)


def _id(name: str) -> str:
    return name
