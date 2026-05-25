"""Promote ASYNC StageBundle to TMA on sm_90+.

For each ``StageBundle`` with ``policy == ASYNC`` inside a
``SerialTile(serial_outer)``, switch ``policy`` to ``TMA`` (keeping
``buffer_count`` / ``phase`` / ``pipeline_depth``; ``swizzle`` stays
NONE) when every member ``Stage`` is TMA-eligible:

- ``ctx.compute_capability >= (9, 0)`` (Hopper+).
- Exactly one ``Source`` on each member Stage. Multi-source bundles
  (matmul A+B) stay on cp.async — TMA's elected-thread + arrive-tx
  model doesn't trivially generalize to two sources without a
  fused-arrive-tx primitive.
- The source uses ``AffineAddressing`` (template addressing is a
  collapsed-reshape view that ``cuTensorMapEncodeTiled`` can't describe).
- ``addressing.dims`` is a strictly-increasing permutation; gap source
  dims (not swept by any cache axis) must be extent-1 singletons.
- Box-inner and source-inner extents both 16 B-aligned, with source
  inner at least 2× the box inner.
- Source rank ≤ 5 (TMA descriptor limit).

The pass is **all-or-nothing per tile**: if any ASYNC bundle in the
tile body is ineligible, leave the whole tile on cp.async (avoids
mixed-mode pipeline deadlock).

Idempotence: TMA-policy bundles are left alone.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SerialTile,
    Stage,
    StageBundle,
    StagePolicy,
    SwizzleMode,
    TileOp,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
_TMA_ALIGN_BYTES = 16


def rewrite(ctx: Context, match: Match, root: Node) -> TileOp | None:
    if ctx.compute_capability < _MIN_CAPABILITY:
        raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")

    # TMA descriptors bake the source shape statically into the cuTensorMap; bail
    # out cleanly if any input shape carries a symbolic dim.
    for nid, node in match.graph.nodes.items():
        for d in node.output.shape:
            if not d.is_static:
                raise RuleSkipped(f"TMA requires static shapes; node {nid!r} has symbolic dim {d!r}")
    src_shapes = {nid: tuple(d.as_static() for d in node.output.shape) for nid, node in match.graph.nodes.items()}

    body = root.op.body
    # All-or-nothing per tile (see module docstring).
    for s in body.iter():
        if not isinstance(s, StageBundle):
            continue
        if s.policy == StagePolicy.TMA:
            continue
        if s.policy == StagePolicy.ASYNC and not _bundle_eligible(s, src_shapes):
            raise RuleSkipped(
                f"ASYNC bundle on {list(s.local_decls())!r} not TMA-eligible; "
                "leaving the whole tile on cp.async (avoids mixed-mode pipeline deadlock)"
            )

    new_body, changed = _walk(body)
    if not changed:
        raise RuleSkipped("no ASYNC StageBundle inside SerialTile(serial_outer) eligible for TMA")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            new_kouter_body, sub = _promote_in_kouter(s.body)
            if sub:
                s = SerialTile(axis=s.axis, body=new_kouter_body, kind=s.kind, unroll=s.unroll)
                changed = True
            out.append(s)
            continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _promote_in_kouter(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, StageBundle) and s.policy == StagePolicy.ASYNC:
            out.append(_promote(s))
            changed = True
        else:
            out.append(s)
    return Body(tuple(out)), changed


def _promote(bundle: StageBundle) -> StageBundle:
    return StageBundle(
        stages=bundle.stages,
        body=bundle.body,
        policy=StagePolicy.TMA,
        buffer_count=bundle.buffer_count,
        phase=bundle.phase,
        pipeline_depth=bundle.pipeline_depth,
        swizzle=SwizzleMode.NONE,
    )


def _bundle_eligible(bundle: StageBundle, src_shapes: dict[str, tuple[int, ...]]) -> bool:
    """A bundle is TMA-eligible iff every member Stage is."""
    for stage in bundle.stages:
        if not _stage_eligible(stage, src_shapes):
            return False
    return True


def _stage_eligible(stage: Stage, src_shapes: dict[str, tuple[int, ...]]) -> bool:
    if len(stage.sources) != 1:
        return False
    (src,) = stage.sources
    if not isinstance(src.addressing, AffineAddressing):
        return False
    cache_axes = src.cache_axes
    if not cache_axes or len(cache_axes) > _MAX_RANK:
        return False
    src_shape = src_shapes.get(src.buf)
    if not src_shape or len(src_shape) > _MAX_RANK or len(src.origin) > _MAX_RANK:
        return False
    dims = src.addressing.dims
    if not dims or len(set(dims)) != len(dims) or list(dims) != sorted(dims):
        return False
    src_rank = len(src_shape)
    if dims[-1] != src_rank - 1:
        return False
    # Gap source dims (not swept by any cache axis) must be extent-1
    # singletons — the materializer drops those from the descriptor.
    dims_set = set(dims)
    for d in range(dims[0], src_rank):
        if d not in dims_set and src_shape[d] != 1:
            return False
    inner_extent = cache_axes[-1].extent.as_static()
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    if src_inner < inner_extent * 2:
        return False
    return True
