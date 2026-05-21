"""Unified launch-geometry pass — decides ``Tile.axes`` shape for matmul
and pointwise bodies in one place.

Two paths, dispatched on body shape:

- **Matmul** (body has a matmul-shape reduce ``Loop``): partition each
  output axis into ``(BLOCK_grid, THREAD_tile)`` per a ``(BN, BM)``
  candidate; emit one ``TileOp`` variant per viable candidate
  (heuristic first; autotune explores the rest).
- **Pointwise** (everything else): deterministic ``thread_tile_shape``
  partition. Same behavior as the old non-matmul branch of
  ``blockify_launch``.

Pre-conditions: ``TileOp`` containing exactly one ``Tile`` with
``block_axes`` empty (post-tileify, before any launch decision).

Idempotent: if a Tile already has ``block_axes`` set or the partition
is a no-op (every axis fits one CTA), the pass skips.

Note: the cooperative-reduce launch path has been removed pending a
planner-driven replacement. The ``_accums_independent`` helper is kept
here for unit-test consumers (``tests/compiler/passes/test_reduction_rules.py``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Stmt, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import BodyInfo, thread_tile_shape

PATTERN = [Pattern("root", TileOp)]


def _logical_output_extents(tile: Tile) -> tuple[int, ...]:
    """Recover the pre-blockify output extents from a (possibly
    blockified) ``Tile``. Walks ``tile.axes`` folding adjacent
    BLOCK-then-THREAD pairs into a single extent."""
    from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD  # noqa: PLC0415

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

    body_info = BodyInfo.of(tile.body)
    output_extents = _logical_output_extents(tile)

    if body_info.has_matmul:
        # Planner-driven path: (BN, BM) already stamped on the parent
        # TileOp's knobs. Emit a single deterministic variant.
        if BN.name in root.op.knobs and BM.name in root.op.knobs:
            return _matmul_deterministic(body, idx, tile, root.op.name, root.op.knobs)
        variants = _matmul_variants(body, idx, tile, root.op.name, body_info, output_extents)
        if not variants:
            raise RuleSkipped("no viable (BN, BM) candidate")
        if len(variants) == 1:
            return variants[0]
        return variants

    # Pointwise / non-matmul: deterministic partition.
    new_axes, sigma_map = _plan_partition(tile, thread_tile_shape(output_extents, body_info))
    if _is_noop_plan(tile, new_axes, sigma_map):
        raise RuleSkipped("partition is a no-op — tile already fits one CTA")
    partitioned = _apply_partition(tile, new_axes, sigma_map)
    return TileOp(body=body[:idx] + (partitioned,) + body[idx + 1 :], name=root.op.name)


def _accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another Accum's
    running value. Permits multiple independent Accums; rejects online
    algorithms (online softmax, Welford). Retained for unit tests."""
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


# ---------------------------------------------------------------------------
# Matmul launch — fork over (BN, BM)
# ---------------------------------------------------------------------------


def _matmul_deterministic(body: tuple[Stmt, ...], idx: int, tile: Tile, name: str, parent_knobs: dict) -> TileOp:
    """Planner-driven matmul launch: read ``(BN, BM)`` from parent
    knobs, plan the partition once, no fork. The planner already
    enumerated the candidate space (:func:`partition_planner.
    _try_matmul_bn_bm_fork`)."""
    bn = int(parent_knobs[BN.name])
    bm = int(parent_knobs[BM.name])
    new_axes, sigma_map = _plan_partition(tile, (bn, bm))
    if _is_noop_plan(tile, new_axes, sigma_map):
        raise RuleSkipped("partition is a no-op — tile already fits one CTA")
    partitioned = _apply_partition(tile, new_axes, sigma_map)
    # No new knobs — BN, BM stamped on the parent LoopOp by the planner
    # and propagate via ``Candidate.apply``.
    return TileOp(body=body[:idx] + (partitioned,) + body[idx + 1 :], name=name)


def _matmul_variants(
    body: tuple[Stmt, ...],
    idx: int,
    tile: Tile,
    name: str,
    body_info: BodyInfo,
    output_extents: tuple[int, ...],
) -> list[TileOp]:
    """Enumerate viable ``(BN, BM)`` candidates. Heuristic shape is
    emitted first so deterministic compiles pick it."""
    if len(tile.axes) < 2:
        return []
    ext_inner = int(tile.axes[-1].axis.extent)
    ext_outer = int(tile.axes[-2].axis.extent)
    heuristic = thread_tile_shape(output_extents, body_info)

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(shape: tuple[int, int]) -> None:
        bn, bm = shape
        # Clamp oversized THREAD widths to extent; ``_plan_partition`` would
        # silently clamp anyway, so dedup after clamp to avoid aliased variants.
        bn = min(bn, ext_inner)
        bm = min(bm, ext_outer)
        if ext_inner > bn and ext_inner % bn != 0:
            return
        if ext_outer > bm and ext_outer % bm != 0:
            return
        shape = (bn, bm)
        if shape in seen:
            return
        seen.add(shape)
        ordered.append(shape)

    if len(heuristic) >= 2:
        _add((int(heuristic[0]), int(heuristic[1])))
    for bn in _TUNE_AXIS_CHOICES:
        for bm in _TUNE_AXIS_CHOICES:
            _add((bn, bm))

    variants: list[TileOp] = []
    for shape in ordered:
        try:
            new_axes, sigma_map = _plan_partition(tile, shape)
        except RuleSkipped:
            continue
        if _is_noop_plan(tile, new_axes, sigma_map):
            continue
        partitioned = _apply_partition(tile, new_axes, sigma_map)
        bn, bm = shape
        variants.append(
            TileOp(
                body=body[:idx] + (partitioned,) + body[idx + 1 :],
                name=name,
                knobs={BN.name: bn, BM.name: bm},
            )
        )
    return variants


# ---------------------------------------------------------------------------
# Shared partition machinery
# ---------------------------------------------------------------------------


def _plan_partition(tile: Tile, shape: tuple[int, ...]) -> tuple[tuple[BoundAxis, ...], dict[str, object]]:
    """Compute the new axis layout and σ-map for partitioning ``tile``
    under ``shape`` (innermost-first). Raises ``RuleSkipped`` if any
    axis is oversized and non-divisible."""
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
    """True iff applying ``new_axes`` / ``sigma_map`` to ``tile`` would
    leave it structurally unchanged."""
    if sigma_map:
        return False
    if len(new_axes) != len(tile.axes):
        return False
    return all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, tile.axes, strict=True))


def _apply_partition(tile: Tile, new_axes: tuple[BoundAxis, ...], sigma_map: dict[str, object]) -> Tile:
    """Build the partitioned ``Tile`` from a plan."""
    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=new_axes, body=new_body)


def _id(name: str) -> str:
    return name
