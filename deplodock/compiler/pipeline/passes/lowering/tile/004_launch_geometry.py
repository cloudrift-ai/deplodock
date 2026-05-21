"""Unified launch-geometry pass — decides ``Tile.axes`` shape for matmul,
cooperative-reduce, and pointwise bodies in one place.

Three paths, dispatched on body shape:

- **Matmul** (body has a matmul-shape reduce ``Loop``): partition each
  output axis into ``(BLOCK_grid, THREAD_tile)`` per a ``(BN, BM)``
  candidate; emit one ``TileOp`` variant per viable candidate
  (heuristic first; autotune explores the rest).
- **Cooperative reduce** (body has a reduce ``Loop`` with extent
  ≥ ``WARP_SIZE`` and independent Accums): leave the body alone, but
  rebind every output axis to ``BIND_BLOCK`` and add a synthetic
  ``t=THREAD`` axis sized to a power-of-two threads-per-CTA. The
  follow-up ``005_cooperative_reduce`` materializes the StridedLoop
  body rewrite and the post-reduce epilogue wrap.
- **Pointwise** (everything else): deterministic ``thread_tile_shape``
  partition. Same behavior as the old non-matmul branch of
  ``blockify_launch``.

Pre-conditions: ``TileOp`` containing exactly one ``Tile`` with
``block_axes`` empty (post-tileify, before any launch decision).

Idempotent: if a Tile already has ``block_axes`` set or the partition
is a no-op (every axis fits one CTA), the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BLOCK_SIZE, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import _has_matmul_reduce, thread_tile_shape

PATTERN = [Pattern("root", TileOp)]

_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_WARP_SIZE = 32

BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul only)")


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already has block_axes — launch geometry already decided")

    if _has_matmul_reduce(tile.body):
        # Planner-driven path: (BN, BM) already stamped on the parent
        # TileOp's knobs. Emit a single deterministic variant.
        if BN.name in root.op.knobs and BM.name in root.op.knobs:
            return _matmul_deterministic(body, idx, tile, root.op.name, root.op.knobs)
        variants = _matmul_variants(body, idx, tile, root.op.name)
        if not variants:
            raise RuleSkipped("no viable (BN, BM) candidate")
        if len(variants) == 1:
            return variants[0]
        return variants

    if _has_cooperative_stride_tag(tile):
        # Planner pre-tagged the cooperative reduce(s) and stamped BN.
        # The launch is now deterministic: read BN from the parent
        # TileOp's knobs and emit a single variant.
        bn = root.op.knobs.get(BN.name)
        if bn is None:
            raise RuleSkipped("cooperative-stride tag present but BN knob missing")
        return _emit_cooperative_launch_deterministic(body, idx, tile, root.op.name, int(bn), root.op.knobs)

    if _cooperative_viable(tile):
        result = _emit_cooperative_launch(body, idx, tile, root.op.name)
        if isinstance(result, list) and not result:
            raise RuleSkipped("no viable BN candidate for cooperative reduce")
        return result

    # Pointwise / non-matmul / non-cooperative: deterministic partition.
    new_axes, sigma_map = _plan_partition(tile, thread_tile_shape(tile))
    if _is_noop_plan(tile, new_axes, sigma_map):
        raise RuleSkipped("partition is a no-op — tile already fits one CTA")
    partitioned = _apply_partition(tile, new_axes, sigma_map)
    return TileOp(body=body[:idx] + (partitioned,) + body[idx + 1 :], name=root.op.name)


# ---------------------------------------------------------------------------
# Cooperative-reduce launch
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _effective_block_size(reduce_extent: int) -> int:
    """Threads/CTA for the cooperative reduce. Capped at the configured
    ``BLOCK_SIZE``; floored at ``WARP_SIZE`` so the cross-thread combine
    has a full warp."""
    return max(_WARP_SIZE, min(BLOCK_SIZE, _next_pow2(reduce_extent)))


def _accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another Accum's
    running value. Permits multiple independent Accums; rejects online
    algorithms (online softmax, Welford)."""
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


def _has_cooperative_stride_tag(tile: Tile) -> bool:
    """True iff any reduce Loop in ``tile.body`` carries the planner's
    ``Role.COOPERATIVE_STRIDE`` tag — signals the planner already
    elected cooperative launch and stamped ``BN``."""
    return any(lp.role is Role.COOPERATIVE_STRIDE for lp in tile.body.iter_of_type(Loop))


def _emit_cooperative_launch_deterministic(body: tuple[Stmt, ...], idx: int, tile: Tile, name: str, bn: int, parent_knobs: dict) -> TileOp:
    """Planner-driven cooperative launch: add ``t=THREAD`` axis sized to
    ``bn`` and rebind every output axis to ``BLOCK``. No fork — the
    planner already enumerated BN candidates."""
    t_axis = Axis("t", bn)
    new_axes = (
        BoundAxis(axis=t_axis, bind=BIND_THREAD),
        *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in tile.axes),
    )
    new_tile = Tile(axes=new_axes, body=tile.body)
    # No new knobs — BN was stamped on the parent LoopOp by the planner
    # and propagates via ``Candidate.apply``.
    del parent_knobs
    return TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=name)


def _cooperative_viable(tile: Tile) -> bool:
    """True iff this Tile's body is a cooperative-reduce candidate:
    has a reduce Loop with extent ≥ WARP_SIZE, every reduce Loop has at
    least one Accum, and Accums are independent."""
    if not tile.thread_axes:
        return False
    reduce_loops = [loop for loop in tile.body.of_type(Loop) if loop.is_reduce]
    if not reduce_loops:
        return False
    if int(reduce_loops[0].axis.extent) < _WARP_SIZE:
        return False
    for rl in reduce_loops:
        if not any(isinstance(s, Accum) for s in rl.body):
            return False
        if not _accums_independent(rl.body):
            return False
    return True


def _emit_cooperative_launch(body: tuple[Stmt, ...], idx: int, tile: Tile, name: str) -> list[TileOp] | TileOp:
    """Add a synthetic ``t=THREAD`` axis and rebind every existing output
    axis to ``BIND_BLOCK``. The body is left untouched —
    ``005_cooperative_reduce`` performs the StridedLoop rewrite and the
    naked-axis epilogue wrap.

    Forks over ``BN`` (cooperative thread count) so autotune can search
    the threads-per-CTA × per-thread-iter-count trade-off jointly with
    ``008_register_tile``'s ``FN``. Heuristic = the historical
    ``_effective_block_size`` so deterministic compiles pick option 0
    with no behavior change; smaller BN values follow."""
    reduce_loops = [loop for loop in tile.body.of_type(Loop) if loop.is_reduce]
    reduce_extent = int(reduce_loops[0].axis.extent)
    heuristic_bn = _effective_block_size(reduce_extent)
    # BN candidates: in _TUNE_AXIS_CHOICES, ≥ WARP_SIZE, ≤ heuristic_bn.
    # Capping at heuristic_bn keeps BN ≤ next_pow2(extent), so threads
    # never outnumber elements; otherwise we'd waste threads idle on
    # short rows.
    bn_candidates = sorted({bn for bn in _TUNE_AXIS_CHOICES if _WARP_SIZE <= bn <= heuristic_bn}, reverse=True)
    if heuristic_bn not in bn_candidates:
        bn_candidates.insert(0, heuristic_bn)
    # Move heuristic to the front so deterministic compiles pick it.
    bn_candidates = [heuristic_bn] + [bn for bn in bn_candidates if bn != heuristic_bn]

    variants: list[TileOp] = []
    for bn in bn_candidates:
        t_axis = Axis("t", bn)
        new_axes = (
            BoundAxis(axis=t_axis, bind=BIND_THREAD),
            *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in tile.axes),
        )
        new_tile = Tile(axes=new_axes, body=tile.body)
        variants.append(TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=name, knobs={BN.name: bn}))
    if len(variants) == 1:
        return variants[0]
    return variants


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


def _matmul_variants(body: tuple[Stmt, ...], idx: int, tile: Tile, name: str) -> list[TileOp]:
    """Enumerate viable ``(BN, BM)`` candidates. Heuristic shape is
    emitted first so deterministic compiles pick it."""
    if len(tile.axes) < 2:
        return []
    ext_inner = int(tile.axes[-1].axis.extent)
    ext_outer = int(tile.axes[-2].axis.extent)
    heuristic = thread_tile_shape(tile)

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
