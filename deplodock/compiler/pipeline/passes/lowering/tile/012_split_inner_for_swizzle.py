"""Split the inner cache axis of a TMA stage to fit a swizzle width.

Runs after ``011_tma_copy`` and before ``013_async_copy``. Visits every
``TmaBufferedStage`` whose ``swizzle == NONE`` (left untouched by the
exact-match picker in ``011_tma_copy``) and whose inner cache extent is
a *multiple* of a swizzle width: 32 / 16 / 8 fp32 = 128 / 64 / 32 B.

When triggered, the pass:

1. Splits the inner axis ``ax_n: BN`` into two axes ``ax_n_outer: BN/IPS``
   and ``ax_n_inner: IPS``. The outer axis carries a fresh ``Axis`` name
   (``<orig>_o``); the inner keeps the original name. Both new axes map
   to the *same* source dim — ``AffineAddressing.dims`` gets the original
   dim repeated.
2. Sigma-rewrites every body ``Load`` reading the stage's smem buffer:
   the rank-N flat index ``[..., row, n_full]`` becomes a rank-(N+1)
   index ``[..., row, n_outer, n_inner_phys]`` where:

   - ``n_outer = n_full / IPS``
   - ``n_inner = n_full % IPS``
   - ``n_inner_phys = n_inner XOR ((swizzle_row & 7) * shift)`` —
     the inverse of the Hopper ``Sw<3, M, S>`` permutation.
   - ``swizzle_row = row * factor + n_outer`` with ``factor = BN/IPS``.
   - ``shift`` is 4 (B128) / 2 (B64) / 1 (B32).

3. Sets ``swizzle`` on the stage to the matching mode.

After this pass, the stage's ``axes`` is rank-(N+1), ``alloc_extents``
expands accordingly, and downstream passes (``013_async_copy``,
``014_pad_smem``, the materializer's TMA emit) see a consistent slab
shape — no special-case logic for the split.

Gated behind ``tuning._tma_swizzle_enabled``.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SwizzleMode,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

_SHIFT_FOR = {SwizzleMode.B128: 4, SwizzleMode.B64: 2, SwizzleMode.B32: 1}

# (row_mask, fp32_shift) per swizzle mode. The XOR amount on the inner
# element index = ``(swizzle_row & row_mask) * fp32_shift``.
#
# Why ``row_mask`` differs per mode: NVIDIA's TMA hardware swizzle XORs
# bits ``[M..M+B-1]`` of the byte address with bits ``[M+S..M+S+B-1]``,
# where ``M+S = 7`` for the fp32 16-byte basic block. The m-identity
# bits in the byte address start at bit ``log2(inner_extent_bytes)``:
# B128 (inner=128B) → m starts at byte bit 7, no overlap with XOR bits
# 7..9, so B=3 (row_mask=7). B64 (inner=64B) → m starts at bit 6, so
# byte bit 6 IS m's bit 0 — including it in the XOR would scramble m's
# identity, so hardware drops it (B=2, row_mask=6). B32 → only 1 bit
# of m available before scrambling (row_mask=4). The fp32 shift is
# always 4/2/1 matching the swizzle granularity.
_ROW_MASK_FOR = {SwizzleMode.B128: 7, SwizzleMode.B64: 6, SwizzleMode.B32: 4}


def rewrite(root: Node) -> Graph | None:
    from deplodock.compiler.tuning import _tma_swizzle_enabled  # noqa: PLC0415

    if not _tma_swizzle_enabled():
        raise RuleSkipped("DEPLODOCK_TMA_SWIZZLE not set")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _pick_split_swizzle(inner_extent_bytes: int) -> tuple[SwizzleMode, int] | None:
    """Largest swizzle width that divides the inner extent. Returns
    ``(mode, ips_fp32)`` or ``None`` if no width fits."""
    for width, mode in ((128, SwizzleMode.B128), (64, SwizzleMode.B64), (32, SwizzleMode.B32)):
        if inner_extent_bytes >= width and inner_extent_bytes % width == 0:
            return mode, width // BYTES_PER_ELEM
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    # Plan per-stage swizzle + optional split. ``factor == 1`` means the
    # inner extent already matches the swizzle width — set swizzle, no
    # axis split, but still rewrite body Loads to apply the XOR. ``factor
    # > 1`` means split the inner axis into ``(factor, IPS)`` and rewrite
    # body Loads to a rank-(N+1) index.
    plans: dict[str, tuple[SwizzleMode, int, int]] = {}  # name → (mode, IPS, factor)
    for s in tile.body.iter():
        if not isinstance(s, TmaBufferedStage) or s.swizzle != SwizzleMode.NONE or not s.axes:
            continue
        inner_extent = int(s.axes[-1].extent)
        inner_bytes = inner_extent * BYTES_PER_ELEM
        pick = _pick_split_swizzle(inner_bytes)
        if pick is None:
            continue
        mode, ips = pick
        factor = inner_extent // ips
        plans[s.name] = (mode, ips, factor)

    if not plans:
        raise RuleSkipped("no TMA stage with swizzle-fitting inner extent")

    new_tile_body = _process_body(tile.body, plans)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("split rewrite produced no change")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_body(
    body: Body,
    plans: dict[str, tuple[SwizzleMode, int, int]],
) -> Body:
    """Walk body: set swizzle (and optionally split axes) on eligible TMA
    stages; XOR-rewrite Loads on those stages. Recurses into Loop bodies."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, TmaBufferedStage) and s.name in plans:
            mode, ips, factor = plans[s.name]
            if factor > 1:
                out.append(_split_stage(s, mode, ips))
            else:
                out.append(dc_replace(s, swizzle=mode))
        elif isinstance(s, Load) and s.input in plans:
            mode, ips, factor = plans[s.input]
            out.append(_swizzle_decode_load(s, mode, ips, factor))
        elif isinstance(s, Loop):
            inner = _process_body(s.body, plans)
            if inner is not s.body and inner != s.body:
                out.append(dc_replace(s, body=inner))
            else:
                out.append(s)
        else:
            out.append(s)
    return Body(out)


def _split_stage(stage: TmaBufferedStage, mode: SwizzleMode, ips: int) -> TmaBufferedStage:
    """Replace ``stage.axes[-1]`` with two axes, the outer carrying the
    factor and the inner the swizzle width. Both map to the same source
    dim in the affine addressing."""
    if not isinstance(stage.addressing, AffineAddressing):
        raise RuleSkipped(f"stage {stage.name!r}: split requires AffineAddressing")
    orig = stage.axes[-1]
    factor = int(orig.extent) // ips
    if factor * ips != int(orig.extent):
        raise RuleSkipped(f"stage {stage.name!r}: inner extent {orig.extent} not divisible by IPS {ips}")
    outer_axis = Axis(f"{orig.name}_o", factor)
    inner_axis = Axis(orig.name, ips)
    new_axes = (*stage.axes[:-1], outer_axis, inner_axis)
    inner_dim = stage.addressing.dims[-1]
    new_dims = (*stage.addressing.dims, inner_dim)
    new_addressing = AffineAddressing(dims=new_dims)
    logger.info(
        "stage %s: split inner axis %s:%d → (%s:%d, %s:%d) for %s",
        stage.name,
        orig.name,
        orig.extent,
        outer_axis.name,
        outer_axis.extent,
        inner_axis.name,
        inner_axis.extent,
        mode.value,
    )
    return dc_replace(stage, axes=new_axes, addressing=new_addressing, swizzle=mode)


def _swizzle_decode_load(load: Load, mode: SwizzleMode, ips: int, factor: int) -> Load:
    """Rewrite a body Load to apply the swizzle XOR.

    For ``factor == 1`` (inner extent already matches IPS, no axis split):
    the Load index keeps its rank, the inner field becomes
    ``inner XOR ((row & 7) * shift)``.

    For ``factor > 1`` (axis was split into ``(factor, IPS)``):
    rank grows by 1. ``[..., row, n_full]`` → ``[..., row, n_outer,
    n_inner_phys]`` with ``n_outer = n_full / IPS``,
    ``n_inner_phys = (n_full % IPS) XOR ((row * factor + n_outer) & 7) * shift)``.
    """
    if len(load.index) < 2:
        raise RuleSkipped(f"Load on {load.input!r}: index rank < 2")
    shift = _SHIFT_FOR[mode]
    row_mask = _ROW_MASK_FOR[mode]
    row = load.index[-2]
    inner = load.index[-1]
    if factor == 1:
        row_bits = BinaryExpr("&", row, Literal(row_mask, "int"))
        xor_term: object = row_bits if shift == 1 else BinaryExpr("*", row_bits, Literal(shift, "int"))
        new_inner: object = BinaryExpr("^", inner, xor_term)
        return dc_replace(load, index=(*load.index[:-1], new_inner))
    n_outer: object = BinaryExpr("/", inner, Literal(ips, "int"))
    n_inner: object = BinaryExpr("%", inner, Literal(ips, "int"))
    swizzle_row: object = BinaryExpr("+", BinaryExpr("*", row, Literal(factor, "int")), n_outer)
    row_bits = BinaryExpr("&", swizzle_row, Literal(row_mask, "int"))
    xor_term = row_bits if shift == 1 else BinaryExpr("*", row_bits, Literal(shift, "int"))
    n_inner_phys = BinaryExpr("^", n_inner, xor_term)
    return dc_replace(load, index=(*load.index[:-1], n_outer, n_inner_phys))
