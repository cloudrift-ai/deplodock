"""Pick a TMA descriptor swizzle mode for each ``TmaBufferedStage``
and rewrite consumer ``Load`` indices with the matching XOR-swizzle.

Runs *after* ``014c_pad_smem_banks`` (which leaves ``TmaBufferedStage``
untouched — TMA box copies write rows back-to-back at the cache extent,
so ``+1`` padding doesn't apply). The swizzle mode is the descriptor-
side counterpart: instead of widening the smem layout, it tells the
TMA hardware to XOR-permute the per-row column index so consecutive
rows hit different banks.

Selection rule (NVIDIA CUDA Programming Guide on TMA swizzle, fp32):

- ``B128`` when the innermost cache dim is ≥ 128 B (32 × fp32) and a
  multiple of 128 B — XOR period 8 rows
- ``B64`` when ≥ 64 B and a multiple of 64 B — XOR period 4 rows
- ``B32`` when ≥ 32 B and a multiple of 32 B — XOR period 2 rows
- ``NONE`` otherwise

XOR rewrite (fp32, row index ``m``, col index ``k``):

    swizzled_k = k ^ ((m & (period - 1)) << 2)
               = k ^ ((m % period) * 4)

The hardware applies this same XOR when the descriptor's swizzle mode is
set, so consumer ``Load(name, [slot, m, k])`` becomes
``Load(name, [slot, m, k ^ ((m % period) * 4)])``. Without rewriting,
plain ``ld.shared.f32`` reads land on the *unswizzled* logical position
and miss the data.

Restricted to:

- 2D cache (``len(stage.axes) == 2``) — the only shape this rewriter
  handles. Higher-D caches need a per-dim flatten + swizzle, which is
  not implemented yet; they stay ``SwizzleMode.NONE``.
- Loads that read every row coord as an affine combination of thread-
  axis vars — required to keep the closed-form bank-conflict check
  meaningful for selection. Non-affine row exprs fall back to NONE.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, SwizzleMode, TileOp, TmaBufferedStage
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    load_thread_axis_coeffs,
    loads_reading,
    max_bank_conflict,
    single_tile,
)

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

# Swizzle byte-width thresholds + XOR row period (fp32). Inner cache dim's
# byte width must be ≥ threshold AND a multiple of it.
_SWIZZLE_TABLE: tuple[tuple[int, SwizzleMode, int], ...] = (
    (128, SwizzleMode.B128, 8),
    (64, SwizzleMode.B64, 4),
    (32, SwizzleMode.B32, 2),
)


def rewrite(graph: Graph, root: Node) -> Graph | None:
    # Opt-in via env var while we validate against ncu. Default off so
    # the cp.async-equivalent ``SwizzleMode.NONE`` path remains baseline.
    if os.environ.get("DEPLODOCK_TMA_SWIZZLE") != "1":
        raise RuleSkipped("TMA swizzle disabled (set DEPLODOCK_TMA_SWIZZLE=1 to enable)")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    thread_axes = tuple(ba.axis for ba in tile.axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise RuleSkipped("Tile has no THREAD axes — no bank-conflict layout to swizzle")

    # Decide swizzle mode per stage; collect (stage_name → mode, period)
    # so the body rewriter can XOR-rewrite the matching Loads.
    mode_by_stage: dict[str, tuple[SwizzleMode, int]] = {}
    new_tile_body = _rewrite_stages(tile.body, thread_axes, mode_by_stage)
    if not mode_by_stage:
        raise RuleSkipped("no TmaBufferedStage benefits from swizzle")

    # Rewrite every consumer Load reading a swizzled stage.
    rewritten = Body.coerce(new_tile_body).map(_make_load_rewriter(mode_by_stage))
    return body[:idx] + (Tile(axes=tile.axes, body=rewritten),) + body[idx + 1 :]


def _rewrite_stages(
    body: Body,
    thread_axes: tuple[Axis, ...],
    mode_by_stage: dict[str, tuple[SwizzleMode, int]],
) -> Body:
    """Walk body + free Loops; for each ``TmaBufferedStage`` decide a
    swizzle mode, mutate its ``swizzle`` field, and record the matching
    XOR period under its name."""
    new_body: list[Stmt] = list(body)
    changed = False
    for i, s in enumerate(body):
        if isinstance(s, TmaBufferedStage):
            decision = _decide_for_stage(s, body, thread_axes)
            if decision is not None:
                mode, period = decision
                # ``TmaBufferedStage.__post_init__`` asserts ``pad`` is empty,
                # so we round-trip via dc_replace which preserves it.
                new_body[i] = dc_replace(s, swizzle=mode)
                mode_by_stage[s.name] = (mode, period)
                changed = True
        elif isinstance(s, Loop):
            inner = _rewrite_stages(s.body, thread_axes, mode_by_stage)
            if inner is not s.body and inner != s.body:
                new_body[i] = dc_replace(s, body=inner)
                changed = True
    return tuple(new_body) if changed else body


def _decide_for_stage(
    stage: TmaBufferedStage,
    body: Body,
    thread_axes: tuple[Axis, ...],
) -> tuple[SwizzleMode, int] | None:
    """Return ``(mode, period)`` if a non-NONE swizzle is selected; else
    ``None`` (leave stage as-is)."""
    if len(stage.axes) != 2:
        # Higher-D caches need per-dim flatten before swizzle. Skip.
        return None

    loads = loads_reading(body, stage.name)
    if not loads:
        return None

    n = len(stage.axes)
    base_extents = tuple(int(ax.extent) for ax in stage.axes)
    per_load_coeffs = load_thread_axis_coeffs(loads, n, thread_axes, leading_phase_dim=True)
    if per_load_coeffs is None:
        logger.debug("TmaBufferedStage %s: non-affine consumer Loads; skipping swizzle", stage.name)
        return None

    base_conflict = max_bank_conflict(per_load_coeffs, base_extents, thread_axes)
    if base_conflict <= 1:
        # Already bank-conflict-free; no swizzle needed.
        return None

    # Inner-dim box width *must equal* the swizzle width (CUDA TMA
    # constraint): ``cuTensorMapEncodeTiled`` rejects descriptors where
    # inner_box_bytes > swizzle_bytes with ``CUDA_ERROR_INVALID_VALUE``.
    # Wider inner dims aren't swizzle-eligible until we add a swizzle-
    # tile-then-flatten step on the consumer side.
    inner_bytes = int(stage.axes[-1].extent) * BYTES_PER_ELEM
    for threshold, mode, period in _SWIZZLE_TABLE:
        if inner_bytes == threshold:
            logger.info(
                "TmaBufferedStage %s: %d-way bank conflict → swizzle=%s (period %d rows)",
                stage.name,
                base_conflict,
                mode.value,
                period,
            )
            return (mode, period)
    logger.warning(
        "TmaBufferedStage %s: %d-way bank conflict; no swizzle mode fits inner-dim width %d B",
        stage.name,
        base_conflict,
        inner_bytes,
    )
    return None


def _make_load_rewriter(mode_by_stage: dict[str, tuple[SwizzleMode, int]]):
    """Build a ``Body.map``-compatible rewriter that XOR-swizzles the
    inner-dim index of every ``Load`` reading a swizzled stage."""

    def fn(stmt: Stmt) -> Stmt:
        if not isinstance(stmt, Load):
            return stmt
        decision = mode_by_stage.get(stmt.input)
        if decision is None:
            return stmt
        _mode, period = decision
        # 2D cache + leading slot prefix → index = (slot, m, k); swizzle k.
        if len(stmt.index) != 3:
            return stmt
        slot, m_expr, k_expr = stmt.index
        new_k = _xor_swizzle(m_expr, k_expr, period)
        return dc_replace(stmt, index=(slot, m_expr, new_k))

    return fn


def _xor_swizzle(m: Expr, k: Expr, period: int) -> Expr:
    """``k ^ ((m % period) * 4)`` — the standard fp32 swizzle XOR."""
    mod = BinaryExpr("%", m, Literal(period, "int"))
    shift = BinaryExpr("*", mod, Literal(4, "int"))
    return BinaryExpr("^", k, shift)
