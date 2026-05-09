"""Launch-geometry pass.

Decides the THREAD/BLOCK partition of every parallel axis on a
``TileOp`` containing one ``Tile`` whose ``block_axes`` is still empty.
Output-dim free-Loop lifting happens earlier in ``001_tileify`` — by
the time this rule runs, every parallel axis is already in
``Tile.axes``.

**Apportion THREAD/BLOCK.** ``tuning.thread_tile_shape(tile)`` returns
the per-axis THREAD-tile widths the launch should emit, in innermost-
first order — typically ``(PAT, PAT)`` for matmul or
``(thread_budget,)`` otherwise.

For each of the innermost ``len(shape)`` axes ``ba`` with extent ``ext``
and target ``shape[i]``:

- ``ext == shape[i]`` — keep whole as THREAD.
- ``ext > shape[i]`` and ``ext % shape[i] == 0`` — split into
  ``axis_i:shape[i]`` (THREAD) + ``axis_o:ext/shape[i]`` (BLOCK), with
  body indices σ-rewritten ``axis → axis_o*shape[i] + axis_i``.
- ``ext < shape[i]`` — keep THREAD whole (smaller-than-target axis is
  fine; the launch loses some threads but stays correct).
- otherwise (``ext > shape[i]`` non-divisible) — bail with
  ``RuleSkipped``.

Outer axes beyond ``len(shape)`` go BLOCK whole.

PAT and the register-tile factor F are paired in ``tuning`` so
``PAT/F = 16``. After ``008_register_tile`` splits each PAT axis by F,
the final per-axis thread count is exactly 16 and the two output axes
together yield ``16² = 256 = thread_budget`` — no post-hoc rebalance
pass needed.

Idempotent: if no axis was split and every axis kept its original
bind, returns None.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import thread_tile_shape

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body):
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already partitioned (block_axes non-empty)")

    partitioned = _partition_threads(tile)
    if partitioned is None:
        raise RuleSkipped("partition already fits the requested tile shape")
    return body[:idx] + (partitioned,) + body[idx + 1 :]


def _partition_threads(tile: Tile) -> Tile | None:
    """Split the innermost axes per ``thread_tile_shape``. Outer axes →
    BLOCK whole. Bail on non-divisible oversized axes."""
    shape = thread_tile_shape(tile)
    axes = list(tile.axes)
    new_axes_inner_first: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}

    for i, ba in enumerate(reversed(axes)):
        if i >= len(shape):
            new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        target = shape[i]
        ext = int(ba.axis.extent)
        if ext == target or ext < target:
            new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
        elif ext % target == 0:
            inner = Axis(f"{ba.axis.name}_i", target)
            outer = Axis(f"{ba.axis.name}_o", ext // target)
            new_axes_inner_first.append(BoundAxis(axis=inner, bind=BIND_THREAD))
            new_axes_inner_first.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
            sigma_map[ba.axis.name] = Var(outer.name) * Literal(target, "int") + Var(inner.name)
        else:
            raise RuleSkipped(f"axis {ba.axis.name}:{ext} not divisible by tile-shape target {target}")

    new_axes_inner_first.reverse()
    new_axes = new_axes_inner_first

    # No-op short-circuit: identical axes, no split.
    if not sigma_map and len(new_axes) == len(axes):
        same = all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, axes, strict=True))
        if same:
            return None

    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=tuple(new_axes), body=new_body)


def _id(name: str) -> str:
    return name
