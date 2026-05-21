"""Pointwise launch-geometry pass — partitions a pointwise ``Tile``'s
output axes into ``(BLOCK_grid, THREAD_tile)`` per :func:`tuning.thread_tile_shape`.

Matmul kernels do NOT reach this pass: the planner stamps BLOCK / THREAD
tags on the matmul body, tileify lifts them into ``Tile.axes`` with the
appropriate ``BoundAxis.bind``, and 004 skips matmul bodies.

Idempotent: if a Tile already has ``block_axes`` set, or any axis is
already BIND_BLOCK (planner did the partition), the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import BodyInfo, thread_tile_shape

PATTERN = [Pattern("root", TileOp)]


def _logical_output_extents(tile: Tile) -> tuple[int, ...]:
    """Walk ``tile.axes`` folding adjacent BLOCK-then-THREAD pairs into a
    single extent. Returns sorted descending."""
    extents: list[int] = []
    i = 0
    while i < len(tile.axes):
        ba = tile.axes[i]
        ext = int(ba.axis.extent)
        if ba.bind == BIND_BLOCK and i + 1 < len(tile.axes) and tile.axes[i + 1].bind == BIND_THREAD:
            extents.append(ext * int(tile.axes[i + 1].axis.extent))
            i += 2
            continue
        extents.append(ext)
        i += 1
    return tuple(sorted(extents, reverse=True))


_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_WARP_SIZE = 32

BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul only)")


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already has block_axes — launch geometry already decided")
    # Planner-driven matmul path tags axes with BIND_BLOCK / BIND_THREAD
    # in the planner itself; 004 must not re-partition those.
    if any(ba.bind == BIND_BLOCK for ba in tile.axes):
        raise RuleSkipped("Tile already partitioned by planner (BIND_BLOCK present)")

    body_info = BodyInfo.of(tile.body)
    if body_info.has_matmul:
        raise RuleSkipped("matmul kernel — partition owned by planner")

    output_extents = _logical_output_extents(tile)
    new_axes, sigma_map = _plan_partition(tile, thread_tile_shape(output_extents, body_info))
    if _is_noop_plan(tile, new_axes, sigma_map):
        raise RuleSkipped("partition is a no-op — tile already fits one CTA")
    partitioned = _apply_partition(tile, new_axes, sigma_map)
    return TileOp(body=body[:idx] + (partitioned,) + body[idx + 1 :], name=root.op.name)


def _accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another Accum's
    running value. Retained for unit tests."""
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


# ---------------------------------------------------------------------------
# Shared partition machinery
# ---------------------------------------------------------------------------


def _plan_partition(tile: Tile, shape: tuple[int, ...]) -> tuple[tuple[BoundAxis, ...], dict[str, object]]:
    """Compute the new axis layout and σ-map for partitioning ``tile``
    under ``shape`` (innermost-first)."""
    new_axes_inner_first: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}

    for i, ba in enumerate(reversed(tile.axes)):
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
    return tuple(new_axes_inner_first), sigma_map


def _is_noop_plan(tile: Tile, new_axes: tuple[BoundAxis, ...], sigma_map: dict[str, object]) -> bool:
    if sigma_map:
        return False
    if len(new_axes) != len(tile.axes):
        return False
    return all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, tile.axes, strict=True))


def _apply_partition(tile: Tile, new_axes: tuple[BoundAxis, ...], sigma_map: dict[str, object]) -> Tile:
    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=new_axes, body=new_body)


def _id(name: str) -> str:
    return name
