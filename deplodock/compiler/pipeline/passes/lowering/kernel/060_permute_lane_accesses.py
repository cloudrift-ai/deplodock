"""Chunk the N register-tile into ``LDS.128``-sized strips when ``FN > V``.

``V = ctx.lds128_bytes // stage.buf.dtype.nbytes`` — elements per
``LDS.128`` transaction at the staged buffer's dtype. ``V = 4`` for an
fp32 staged buffer, ``V = 8`` for fp16. The hardware constant (the
128-bit bus width) lives on ``Context``; the per-stage elem count is
looked up via ``match.graph.nodes[stage.buf].output.dtype.nbytes`` so
fp16 matmuls don't waste half the LDS.128 bandwidth.

After ``010_split_register_axes`` commits the canonical
``(THREAD-outer | RF-inner)`` ordering on the N axis, body Loads on the
B operand stage read column
``Var(lane) * FN + c`` for ``c=0..FN-1``. Two regimes:

* ``FN <= V`` (canonical default): lane ``l`` reads the contiguous strip
  ``[FN*l, FN*l+FN-1]``. nvcc auto-vectorizes the unrolled scalar reads
  into a single ``ld.shared.v4`` per K-iter — and within each LDS.128
  phase (8 lanes × V elems) the warp covers 32 distinct banks, so no
  conflicts. **No transform needed; pass skips.**

* ``FN > V`` (e.g. FN=8 at fp32): lane ``l`` reads ``[8l..8l+7]``. Each
  LDS.128 carries V elems, so the per-thread N-strip splits into ``FN/V``
  successive LDS.128. *Within* one LDS.128 phase, the 8 lanes are
  stride-``FN`` apart (lane 0 → cols 0-3, lane 1 → 8-11, …, lane 4
  wraps). 8 lanes × V elems = 32 elems of accesses, but with that stride
  they land on only 16 banks → 2-way conflict per phase, 8-way per
  LDS.128. ncu confirms 25 M+ conflicts on this shape.

Fix: keep each LDS.128 reading V *contiguous* elems (so the phase covers
32 distinct banks), but spread the ``FN/V`` LDS.128's *across the
N-tile* instead of stacking them at lane-stride FN. Lane ``l`` now owns
``FN/V`` chunks of V cols each, the chunks spaced ``V * lane_ext``
apart inside the BN-wide tile (which is exactly ``FN/V`` chunks ×
``V * lane_ext`` cols since ``BN = FN * lane_ext``). For the canonical
``FN=8, lane_ext=16`` (``BN=128``) shape at fp32, the chunk stride is 64.

Rewrite (applied to every Load + Write index expression containing
``Var(lane) * FN + Literal(c)``):

    Var(lane) * FN + Literal(c)
        →   V * Var(lane) + Literal((c // V) * (V * lane_ext) + (c % V))

For ``FN=8, lane_ext=16`` at fp32 (``V=4``):

    c=0..3 → ``4*lane + c``           (chunk 0, cols 0..63 of the tile)
    c=4..7 → ``4*lane + 64 + (c-4)``  (chunk 1, cols 64..127 of the tile)

Per LDS.128 phase: 8 lanes read 32 contiguous elems (one chunk), 32
distinct banks, **0 conflicts**. The Write to global memory follows
the same per-thread mapping, so each chunk's lane-to-output assignment
is also stride-1 (improved coalescing vs the FN-stride default).

Lane axis: the larger-extent THREAD axis (the N axis post-008; the M
axis sits at extent ``BM/FM``, N at ``BN/FN``, and ``BN >= BM`` by
convention). M-axis register tile is left alone — M loads broadcast
within a warp at this point and don't carry a fixable conflict.

Skips when:

* The picked axis has no body Load with stride F > V divisible by V.
  Captures the ``FN <= V`` no-op case and any non-canonical shape.
* Bank-conflict analysis (reusing ``070_pad_smem``'s analyzer) says
  ``post < pre`` doesn't hold.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.context import Context
from deplodock.compiler.diagnostics.bank_conflicts import lane_bank_distribution
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import StageBundle, StagePolicy, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    loads_reading,
    replace_thread_tile_body,
    single_tile,
    thread_tile_of,
)

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> Graph | None:
    knobs = root.op.knobs
    if "FN" not in knobs:
        raise RuleSkipped("no FN knob on TileOp (split_register_axes hasn't run)")
    F = int(knobs["FN"])
    new_body = _maybe_rewrite(root.op.body, lds128_bytes=ctx.lds128_bytes, F=F)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(knobs))


def _maybe_rewrite(body: Body, *, lds128_bytes: int, F: int) -> Body | None:
    idx, outer = single_tile(body)
    tile = thread_tile_of(outer)

    thread_axes = tile.axes
    if len(thread_axes) < 2:
        raise RuleSkipped("need >=2 THREAD axes (matmul-shaped tile)")

    # Lane axis: the innermost THREAD axis (``tt.axes[-1]``). By
    # convention that's the N axis post-split_register_axes — its extent is
    # ``BN/FN``, and its consumer Loads carry the ``Var(lane) * FN + c``
    # pattern we rewrite. Lane stride ``F = knobs["FN"]`` is the register-tile
    # factor stamped by ``003_register_tile``.
    lane = thread_axes[-1]
    vec_elems = _vec_elems_for_lane(tile, lane.name, lds128_bytes=lds128_bytes)
    if vec_elems is None:
        raise RuleSkipped("no Stage source dtype known for lane-axis consumers")
    if F <= vec_elems or F % vec_elems != 0:
        raise RuleSkipped(f"lane stride F={F} not > vec_elems={vec_elems} or not divisible")
    lane_ext = lane.extent.as_static()

    if not _swap_helps_any_stage(tile, lane.name, F, lane_ext, vec_elems=vec_elems):
        raise RuleSkipped("no Stage has a fixable lane-stride bank conflict")

    chunk_stride = vec_elems * lane_ext
    new_body = _rewrite_body(tile.body, lane.name, F, chunk_stride, vec_elems=vec_elems)
    new_tile = replace_thread_tile_body(outer, new_body)
    logger.debug(
        "chunk N register-tile on lane=%s F=%d -> stride=%d, chunks=%d (chunk_stride=%d, BN=%d)",
        lane.name,
        F,
        vec_elems,
        F // vec_elems,
        chunk_stride,
        lane_ext * F,
    )
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _vec_elems_for_lane(tile, lane_var: str, *, lds128_bytes: int) -> int | None:
    """Elements per LDS.128 at the dtype of the affected Stages. Reads
    each affected consumer Load's stamped ``load.dtype.nbytes``
    (``030_stamp_types`` populated it from the matching Stage source).

    Returns the minimum across affected Loads (conservative chunk width
    that still vectorizes every involved Stage), or ``None`` when no
    affected Load is stamped — caller treats that as "skip the rewrite"."""
    sizes: set[int] = set()
    for bundle in tile.body.iter():
        if not isinstance(bundle, StageBundle):
            continue
        for member in bundle.stages:
            for src in member.sources:
                loads = loads_reading(tile.body, src.name)
                for ld in loads:
                    if lane_var not in _load_free_vars(ld):
                        continue
                    if ld.dtype is None:
                        return None
                    sizes.add(lds128_bytes // ld.dtype.nbytes)
    if not sizes:
        return None
    return min(sizes)


def _load_free_vars(load: Load) -> frozenset[str]:
    out: set[str] = set()
    for e in load.index:
        out |= e.free_vars()
    return frozenset(out)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _stage_max_way(loads: list[Load], extents: tuple[int, ...], leading_phase: bool, tile) -> int | None:
    """Worst-case ``max_way`` across body ``loads`` of one Stage at the
    given (un-padded) cache extents. ``None`` when any Load can't be
    evaluated (rank mismatch). Auto-zero-binds non-thread free vars,
    so non-affine loop-dependent indices work out of the box."""
    worst = 1
    for ld in loads:
        cache_idx = ld.index[1:] if leading_phase else ld.index
        if len(cache_idx) != len(extents):
            return None
        dist = lane_bank_distribution(tuple(cache_idx), extents, tile.axes)
        if dist is None:
            return None
        worst = max(worst, dist.max_way)
    return worst


def _swap_helps_any_stage(tile, lane_var: str, F: int, lane_ext: int, *, vec_elems: int) -> bool:
    """At least one Stage has a fixable bank conflict that the chunked
    rewrite would drive lower. Reuses the shared bank kernel from
    ``compiler/diagnostics/bank_conflicts``.

    Note the kernel is a coarse predictor — it counts bank conflicts
    treating each Load as a single-element access (i.e. as if hardware
    issued LDS.32). For ``F=4`` with LDS.128 the hardware actually
    collapses 4 lanes' contiguous reads into one 32-bank phase (no
    conflict) but the kernel still scores ``F=4`` as 4-way. So we
    just check ``post < pre`` — if the rewrite at least doesn't worsen
    the model's score, the hardware will see a real improvement
    (no more cross-phase 8-way collisions for ``F>=8``).
    """
    for bundle in tile.body.iter():
        if not isinstance(bundle, StageBundle):
            continue
        # Non-SYNC bundles (BUFFERED / ASYNC / TMA) prepend a phase index
        # to consumer Loads of staged smem.
        leading_phase = bundle.policy != StagePolicy.SYNC
        for member in bundle.stages:
            for src in member.sources:
                loads = loads_reading(tile.body, src.name)
                if not loads:
                    continue
                extents = tuple(ax.extent.as_static() for ax in src.cache_axes)
                pre = _stage_max_way(loads, extents, leading_phase, tile)
                if pre is None or pre <= 1:
                    continue
                chunk_stride = vec_elems * lane_ext
                rewritten = [_rewrite_load_index(ld, lane_var, F, chunk_stride, vec_elems=vec_elems) for ld in loads]
                post = _stage_max_way(rewritten, extents, leading_phase, tile)
                if post is None:
                    continue
                if post < pre:
                    logger.debug("Stage %s: bank conflict %d -> %d under chunk rewrite", src.name, pre, post)
                    return True
    return False


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


def _rewrite_body(body: Body, lane_var: str, F: int, chunk_stride: int, *, vec_elems: int) -> Body:
    return body.map(lambda s: _rewrite_stmt(s, lane_var, F, chunk_stride, vec_elems=vec_elems))


def _rewrite_stmt(s: Stmt, lane_var: str, F: int, chunk_stride: int, *, vec_elems: int) -> Stmt:
    if isinstance(s, Load):
        return _rewrite_load_index(s, lane_var, F, chunk_stride, vec_elems=vec_elems)
    if isinstance(s, Write):
        new_idx = tuple(_chunk_expr(e, lane_var, F, chunk_stride, vec_elems=vec_elems) for e in s.index)
        if new_idx != s.index:
            return dc_replace(s, index=new_idx)
    return s


def _rewrite_load_index(ld: Load, lane_var: str, F: int, chunk_stride: int, *, vec_elems: int) -> Load:
    new_idx = tuple(_chunk_expr(e, lane_var, F, chunk_stride, vec_elems=vec_elems) for e in ld.index)
    if new_idx == ld.index:
        return ld
    return dc_replace(ld, index=new_idx)


def _chunk_expr(e: Expr, lane_var: str, F: int, chunk_stride: int, *, vec_elems: int) -> Expr:
    """Rewrite ``Var(lane_var) * F + Literal(c)`` into
    ``vec_elems * Var(lane_var) + Literal(off)`` where
    ``off = (c // vec_elems) * chunk_stride + (c % vec_elems)``. The
    bare-``Var(lane) * F`` form (the c=0 replica) lowers to
    ``vec_elems * Var(lane)``. ``chunk_stride = vec_elems * lane_ext``
    so the chunks tile the BN-wide N tile evenly. Other subtrees pass
    through.
    """
    if lane_var not in e.free_vars():
        return e

    if isinstance(e, BinaryExpr) and e.op == "+":
        for left, right in ((e.left, e.right), (e.right, e.left)):
            if _matches_lane_mul(left, lane_var, F) and isinstance(right, Literal) and isinstance(right.value, int):
                c = right.value
                chunk_idx, within = divmod(c, vec_elems)
                off = chunk_idx * chunk_stride + within
                return BinaryExpr(
                    "+",
                    BinaryExpr("*", Var(lane_var), Literal(vec_elems, "int")),
                    Literal(off, "int"),
                )
        return BinaryExpr(
            "+",
            _chunk_expr(e.left, lane_var, F, chunk_stride, vec_elems=vec_elems),
            _chunk_expr(e.right, lane_var, F, chunk_stride, vec_elems=vec_elems),
        )

    if isinstance(e, BinaryExpr) and e.op == "-":
        return BinaryExpr(
            "-",
            _chunk_expr(e.left, lane_var, F, chunk_stride, vec_elems=vec_elems),
            _chunk_expr(e.right, lane_var, F, chunk_stride, vec_elems=vec_elems),
        )

    if _matches_lane_mul(e, lane_var, F):
        return BinaryExpr("*", Var(lane_var), Literal(vec_elems, "int"))

    if isinstance(e, BinaryExpr):
        return BinaryExpr(
            e.op,
            _chunk_expr(e.left, lane_var, F, chunk_stride, vec_elems=vec_elems),
            _chunk_expr(e.right, lane_var, F, chunk_stride, vec_elems=vec_elems),
        )

    return e


def _matches_lane_mul(e: Expr, lane_var: str, F: int) -> bool:
    """``Var(lane_var) * Literal(F)`` (either operand order)."""
    if not isinstance(e, BinaryExpr) or e.op != "*":
        return False
    for var_side, lit_side in ((e.left, e.right), (e.right, e.left)):
        if isinstance(var_side, Var) and var_side.name == lane_var and isinstance(lit_side, Literal) and lit_side.value == F:
            return True
    return False
