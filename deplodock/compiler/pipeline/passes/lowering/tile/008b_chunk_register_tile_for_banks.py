"""Chunk the N register-tile into ``LDS.128``-sized strips when ``F_N > 4``.

After ``008_register_tile`` commits the canonical ``(THREAD-outer | RF-inner)``
ordering on the N axis, body Loads on the B operand stage read column
``Var(lane) * F_N + c`` for ``c=0..F_N-1``. Two regimes:

* ``F_N <= 4`` (canonical default): lane ``l`` reads the contiguous strip
  ``[F_N*l, F_N*l+F_N-1]``. nvcc auto-vectorizes the unrolled scalar reads
  into a single ``ld.shared.v4`` per K-iter — and within each LDS.128
  phase (8 lanes × 4 fp32) the warp covers 32 distinct banks, so no
  conflicts. **No transform needed; pass skips.**

* ``F_N > 4`` (e.g. F_N=8): lane ``l`` reads ``[8l..8l+7]``. Each LDS.128
  carries 4 fp32, so the per-thread N-strip splits into ``F_N/4``
  successive LDS.128. *Within* one LDS.128 phase, the 8 lanes are
  stride-``F_N`` apart (lane 0 → cols 0-3, lane 1 → 8-11, …, lane 4
  wraps). 8 lanes × 4 fp32 = 32 fp32 of accesses, but with that stride
  they land on only 16 banks → 2-way conflict per phase, 8-way per
  LDS.128. ncu confirms 25 M+ conflicts on this shape.

Fix: keep each LDS.128 reading 4 *contiguous* fp32 (so the phase covers
32 distinct banks), but spread the ``F_N/4`` LDS.128's *across the
N-tile* instead of stacking them at lane-stride F_N. Lane ``l`` now owns
``F_N/4`` chunks of 4 cols each, the chunks spaced ``4 * lane_ext``
apart inside the BN-wide tile (which is exactly ``F_N/4`` chunks ×
``4 * lane_ext`` cols since ``BN = F_N * lane_ext``). For the canonical
``F_N=8, lane_ext=16`` (``BN=128``) shape, the chunk stride is 64.

Rewrite (applied to every Load + Write index expression containing
``Var(lane) * F_N + Literal(c)``):

    Var(lane) * F_N + Literal(c)
        →   4 * Var(lane) + Literal((c // 4) * (4 * lane_ext) + (c % 4))

For ``F_N=8, lane_ext=16``:

    c=0..3 → ``4*lane + c``           (chunk 0, cols 0..63 of the tile)
    c=4..7 → ``4*lane + 64 + (c-4)``  (chunk 1, cols 64..127 of the tile)

Per LDS.128 phase: 8 lanes read 32 contiguous fp32 (one chunk), 32
distinct banks, **0 conflicts**. The Write to global memory follows
the same per-thread mapping, so each chunk's lane-to-output assignment
is also stride-1 (improved coalescing vs the F_N-stride default).

Lane axis: the larger-extent THREAD axis (the N axis post-008; the M
axis sits at extent ``BM/F_M``, N at ``BN/F_N``, and ``BN >= BM`` by
convention). M-axis register tile is left alone — M loads broadcast
within a warp at this point and don't carry a fixable conflict.

Skips when:

* The picked axis has no body Load with stride F > 4 divisible by 4.
  Captures the ``F_N <= 4`` no-op case and any non-canonical shape.
* Bank-conflict analysis (reusing ``013_pad_smem``'s analyzer) says
  ``post < pre`` doesn't hold.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var, affine_form
from deplodock.compiler.ir.stmt import Body, Load, Stmt, Tile, Write
from deplodock.compiler.ir.tile.ir import BufferedStage, Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    load_thread_axis_coeffs,
    loads_reading,
    max_bank_conflict,
    single_tile,
)

logger = logging.getLogger(__name__)

# ``LDS.128`` carries 4 fp32 per lane. The chunk size in this pass is
# the maximum-vector-load width; we keep each per-thread N-chunk that
# size so nvcc can keep emitting LDS.128 + the warp's 8-lane phase
# covers exactly 32 banks.
_LDS128_FLOATS = 4

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    thread_axes = tuple(ba.axis for ba in tile.axes if ba.bind == BIND_THREAD)
    if len(thread_axes) < 2:
        raise RuleSkipped("need >=2 THREAD axes (matmul-shaped tile)")

    # Pick the lane axis: the THREAD axis tied to the *N dim of B*. In
    # the canonical post-008 layout that's the larger-extent THREAD axis
    # (the M axis sits at extent BM/F_M, the N axis at BN/F_N, with BN
    # >= BM by convention). We confirm via the access pattern that the
    # picked axis appears in some Stage Load with stride F divisible by
    # ``_LDS128_FLOATS`` and > _LDS128_FLOATS — otherwise the chunked
    # rewrite is a no-op or doesn't apply.
    sorted_threads = sorted(thread_axes, key=lambda ax: int(ax.extent))
    candidates: list[tuple[Axis, int]] = []
    for ax in reversed(sorted_threads):  # try N (largest) first
        F = _infer_lane_stride(tile.body, ax.name)
        if F is None or F <= _LDS128_FLOATS or F % _LDS128_FLOATS != 0:
            continue
        candidates.append((ax, F))
    if not candidates:
        raise RuleSkipped(f"no THREAD axis has Load-stride F divisible by {_LDS128_FLOATS} and > {_LDS128_FLOATS}")
    lane, F = candidates[0]
    lane_ext = int(lane.extent)

    if not _swap_helps_any_stage(tile, thread_axes, lane.name, F, lane_ext):
        raise RuleSkipped("no Stage has a fixable lane-stride bank conflict")

    chunk_stride = _LDS128_FLOATS * lane_ext
    new_tile = Tile(axes=tile.axes, body=_rewrite_body(tile.body, lane.name, F, chunk_stride))
    logger.info(
        "chunk N register-tile on lane=%s F=%d -> stride=%d, chunks=%d (chunk_stride=%d, BN=%d)",
        lane.name,
        F,
        _LDS128_FLOATS,
        F // _LDS128_FLOATS,
        chunk_stride,
        lane_ext * F,
    )
    return body[:idx] + (new_tile,) + body[idx + 1 :]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _infer_lane_stride(body: Body, lane_var: str) -> int | None:
    """Coefficient of ``Var(lane_var)`` across all body Loads' index dims.

    Returns ``None`` when no Load depends on the lane var or the
    coefficients aren't uniform — same conservatism as the original
    swap pass, since a per-Load mixed-stride rewrite isn't produced
    by ``008``.
    """
    seen: set[int] = set()
    for s in body.iter():
        if not isinstance(s, Load):
            continue
        for e in s.index:
            if lane_var not in e.free_vars():
                continue
            af = affine_form(e, {lane_var})
            if af is None:
                return None
            _, coeffs = af
            c = coeffs.get(lane_var, 0)
            if c != 0:
                seen.add(c)
    if len(seen) != 1:
        return None
    (F,) = seen
    return F


def _swap_helps_any_stage(tile: Tile, thread_axes: tuple[Axis, ...], lane_var: str, F: int, lane_ext: int) -> bool:
    """At least one Stage has a fixable bank conflict that the chunked
    rewrite would drive lower. Reuses the affine analyzer from
    ``013_pad_smem``.

    Note the analyzer is a coarse predictor — it counts bank conflicts
    treating each Load as a single-element access (i.e. as if hardware
    issued LDS.32). For ``F=4`` with LDS.128 the hardware actually
    collapses 4 lanes' contiguous reads into one 32-bank phase (no
    conflict) but the analyzer still scores ``F=4`` as 4-way. So we
    just check ``post < pre`` — if the rewrite at least doesn't worsen
    the analyzer's score, the hardware will see a real improvement
    (no more cross-phase 8-way collisions for ``F>=8``).
    """
    for s in tile.body.iter():
        if not isinstance(s, Stage):
            continue
        loads = loads_reading(tile.body, s.name)
        if not loads:
            continue
        n_axes = len(s.axes)
        leading_phase = isinstance(s, BufferedStage)
        coeffs_pre = load_thread_axis_coeffs(loads, n_axes, thread_axes, leading_phase_dim=leading_phase)
        if coeffs_pre is None:
            continue
        extents = tuple(int(ax.extent) for ax in s.axes)
        pre = max_bank_conflict(coeffs_pre, extents, thread_axes)
        if pre <= 1:
            continue
        chunk_stride = _LDS128_FLOATS * lane_ext
        rewritten = [_rewrite_load_index(ld, lane_var, F, chunk_stride) for ld in loads]
        coeffs_post = load_thread_axis_coeffs(rewritten, n_axes, thread_axes, leading_phase_dim=leading_phase)
        if coeffs_post is None:
            continue
        post = max_bank_conflict(coeffs_post, extents, thread_axes)
        if post < pre:
            logger.debug("Stage %s: bank conflict %d -> %d under chunk rewrite", s.name, pre, post)
            return True
    return False


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


def _rewrite_body(body: Body, lane_var: str, F: int, chunk_stride: int) -> Body:
    return body.map(lambda s: _rewrite_stmt(s, lane_var, F, chunk_stride))


def _rewrite_stmt(s: Stmt, lane_var: str, F: int, chunk_stride: int) -> Stmt:
    if isinstance(s, Load):
        return _rewrite_load_index(s, lane_var, F, chunk_stride)
    if isinstance(s, Write):
        new_idx = tuple(_chunk_expr(e, lane_var, F, chunk_stride) for e in s.index)
        if new_idx != s.index:
            return dc_replace(s, index=new_idx)
    return s


def _rewrite_load_index(ld: Load, lane_var: str, F: int, chunk_stride: int) -> Load:
    new_idx = tuple(_chunk_expr(e, lane_var, F, chunk_stride) for e in ld.index)
    if new_idx == ld.index:
        return ld
    return dc_replace(ld, index=new_idx)


def _chunk_expr(e: Expr, lane_var: str, F: int, chunk_stride: int) -> Expr:
    """Rewrite ``Var(lane_var) * F + Literal(c)`` into
    ``_LDS128_FLOATS * Var(lane_var) + Literal(off)`` where
    ``off = (c // 4) * chunk_stride + (c % 4)``. The bare-``Var(lane)
    * F`` form (the c=0 replica) lowers to ``_LDS128_FLOATS *
    Var(lane)``. ``chunk_stride = _LDS128_FLOATS * lane_ext`` so the
    chunks tile the BN-wide N tile evenly. Other subtrees pass through.
    """
    if lane_var not in e.free_vars():
        return e

    if isinstance(e, BinaryExpr) and e.op == "+":
        for left, right in ((e.left, e.right), (e.right, e.left)):
            if _matches_lane_mul(left, lane_var, F) and isinstance(right, Literal) and isinstance(right.value, int):
                c = right.value
                chunk_idx, within = divmod(c, _LDS128_FLOATS)
                off = chunk_idx * chunk_stride + within
                return BinaryExpr(
                    "+",
                    BinaryExpr("*", Var(lane_var), Literal(_LDS128_FLOATS, "int")),
                    Literal(off, "int"),
                )
        return BinaryExpr("+", _chunk_expr(e.left, lane_var, F, chunk_stride), _chunk_expr(e.right, lane_var, F, chunk_stride))

    if isinstance(e, BinaryExpr) and e.op == "-":
        return BinaryExpr("-", _chunk_expr(e.left, lane_var, F, chunk_stride), _chunk_expr(e.right, lane_var, F, chunk_stride))

    if _matches_lane_mul(e, lane_var, F):
        return BinaryExpr("*", Var(lane_var), Literal(_LDS128_FLOATS, "int"))

    if isinstance(e, BinaryExpr):
        return BinaryExpr(e.op, _chunk_expr(e.left, lane_var, F, chunk_stride), _chunk_expr(e.right, lane_var, F, chunk_stride))

    return e


def _matches_lane_mul(e: Expr, lane_var: str, F: int) -> bool:
    """``Var(lane_var) * Literal(F)`` (either operand order)."""
    if not isinstance(e, BinaryExpr) or e.op != "*":
        return False
    for var_side, lit_side in ((e.left, e.right), (e.right, e.left)):
        if isinstance(var_side, Var) and var_side.name == lane_var and isinstance(lit_side, Literal) and lit_side.value == F:
            return True
    return False
