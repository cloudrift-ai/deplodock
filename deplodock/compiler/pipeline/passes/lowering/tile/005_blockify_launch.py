"""Launch-geometry pass — forks over matmul (BN, BM) for autotuning.

Decides the THREAD/BLOCK partition of every parallel axis on a
``TileOp`` containing one ``Tile`` whose ``block_axes`` is still empty.
Output-dim free-Loop lifting happens earlier in ``001_tileify`` — by
the time this rule runs, every parallel axis is already in
``Tile.axes``.

**Apportion THREAD/BLOCK.** A per-axis THREAD-tile shape
``(BN, BM)`` (innermost-first) drives the split:

- ``ext == shape[i]`` — keep whole as THREAD.
- ``ext > shape[i]`` and ``ext % shape[i] == 0`` — split into
  ``axis_i:shape[i]`` (THREAD) + ``axis_o:ext/shape[i]`` (BLOCK), with
  body indices σ-rewritten ``axis → axis_o*shape[i] + axis_i``.
- ``ext < shape[i]`` — keep THREAD whole.
- otherwise (oversized, non-divisible) — bail with ``RuleSkipped``.

Outer axes beyond ``len(shape)`` go BLOCK whole.

**Matmul: autotune fork.** For matmul tiles the rule returns a *list*
of TileOp variants — one per ``(BN, BM)`` candidate that passes the
``register_tile_shape`` gate (THREAD extents share a value with the
heuristic class's ``(def_bn, def_bm)`` so ``008_register_tile`` can
apply a non-trivial F). Option 0 is the heuristic shape so deterministic
``run_pipeline`` callers behave exactly as before; the rest only get
explored under ``deplodock tune`` (or when an autotune-driven ``Search`` consumes
the queue). Non-matmul tiles return a single deterministic TileOp.

PAT and the register-tile factor F are paired in ``tuning`` so
``PAT/F = 16``. After ``008_register_tile`` splits each PAT axis by F,
the final per-axis thread count is exactly 16 and the two output axes
together yield ``16² = 256 = thread_budget`` — no post-hoc rebalance
pass needed.

Idempotent: if option 0's partition is identical to the current Tile
(no axis split, every bind unchanged), raises ``RuleSkipped``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import _has_matmul_reduce, thread_tile_shape

PATTERN = [Pattern("root", TileOp)]

# Candidate per-axis THREAD widths to fork over for matmul tiles.
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)
    # ``block_axes`` non-empty means either (a) this rule already
    # partitioned the tile, or (b) a prior pass (e.g. cooperative_reduce)
    # established a BLOCK layout we must not re-classify. Either way,
    # bail before ``_plan_partition`` rebinds existing BLOCK axes.
    if tile.block_axes:
        raise RuleSkipped("Tile already has block_axes — partition is owned by a prior pass")
    shape = thread_tile_shape(tile)

    if _has_matmul_reduce(tile.body):
        variants = _matmul_variants(body, idx, tile, root.op.name)
        if not variants:
            raise RuleSkipped("no viable (BN, BM) candidate")
        if len(variants) == 1:
            return variants[0]
        return variants

    new_axes, sigma_map = _plan_partition(tile, shape)
    if _is_noop_plan(tile, new_axes, sigma_map):
        raise RuleSkipped("partition is a no-op — tile already fits one CTA")
    partitioned = _apply_partition(tile, new_axes, sigma_map)
    return TileOp(body=body[:idx] + (partitioned,) + body[idx + 1 :], name=root.op.name)


def _matmul_variants(body, idx: int, tile: Tile, name: str) -> list[TileOp]:
    """Enumerate ``(BN, BM)`` candidates for a matmul Tile. Heuristic shape
    is emitted first so deterministic compiles pick it. Variants are
    filtered to those that pass the ``register_tile_shape`` gate (one
    THREAD extent must match the heuristic class's ``(def_bn, def_bm)``)
    so ``008_register_tile`` can apply non-trivial F."""
    if len(tile.axes) < 2:
        return []
    ext_inner = int(tile.axes[-1].axis.extent)
    ext_outer = int(tile.axes[-2].axis.extent)

    heuristic = thread_tile_shape(tile)

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(shape: tuple[int, int]) -> None:
        bn, bm = shape
        # Clamp oversized THREAD widths to the axis extent. ``_plan_partition``
        # silently clamps anyway, so ``bn > ext_inner`` would yield a kernel
        # identical to ``bn == ext_inner``; dedup via ``seen`` after clamping
        # so the autotuner doesn't spawn aliased variants. Clamping (vs the
        # old reject) gives tiny matmuls (both ext ≤ smallest _TUNE_AXIS_CHOICE)
        # at least one valid variant — e.g. SDPA's per-head 8×8 QK·V where
        # rejecting every choice left no candidate at all.
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

    # Heuristic first — deterministic-compile path unchanged. The
    # autotune search re-ranks unvisited candidates via ``TileOp.score``
    # so the rule's emission order doesn't bias exploration.
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
                knobs={"BN": bn, "BM": bm},
            )
        )
    return variants


def _plan_partition(tile: Tile, shape: tuple[int, ...]) -> tuple[tuple[BoundAxis, ...], dict[str, object]]:
    """Compute the new axis layout and σ-map for partitioning ``tile``
    under ``shape`` (innermost-first). Raises ``RuleSkipped`` if any
    axis is oversized and non-divisible. Pure: does not build a Tile."""
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
    leave it structurally unchanged — no axis split and every bind /
    axis identity preserved. Used both as the post-plan idempotence
    check on the deterministic path and the per-variant filter on the
    matmul fork."""
    if sigma_map:
        return False
    if len(new_axes) != len(tile.axes):
        return False
    return all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, tile.axes, strict=True))


def _apply_partition(tile: Tile, new_axes: tuple[BoundAxis, ...], sigma_map: dict[str, object]) -> Tile:
    """Build the partitioned ``Tile`` from a plan produced by
    ``_plan_partition``. σ-rewrites the body when any axis was split."""
    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=new_axes, body=new_body)


def _id(name: str) -> str:
    return name
