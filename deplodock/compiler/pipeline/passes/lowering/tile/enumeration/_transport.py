"""TMA-transport eligibility — the ``promote_transport`` move's legality oracle.

``plans/tile-ir-block-dag.md`` R5: ``promote_transport(read, →TMA)`` writes the
``Schedule.staged[edge]`` value. Eligibility reads only the **derived** projections
— the prospective smem ``Source`` ``assembly/_slab`` would build for each staged
read-site (its ``AffineAddressing`` + cache-axis box) plus the logical gmem
``Buffer.shape`` — never tower shape. This module is that pure predicate, ported
from the deleted legacy ``050_use_tma._source_eligible`` and adapted to the
block-DAG ``TileGraph``.

R5 scope: the **warp-tier** ``mma.sync`` matmul (an ``Atom`` is pinned), where the
staged operands feed ``ldmatrix`` and the slab is swizzled (B64 / B128). The scalar
cooperative-reduce / pointwise tiers stay on SYNC staging — their TMA promotion (and
its fp16-ring-slot alignment decline) rides a later follow-up; restricting here keeps
the scalar reduce correct via SYNC without the strict-slot-align gate.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.ir.expr import BinaryExpr, Literal
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SerialTile,
    Source,
    TileGraph,
)

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
# ``cuTensorMapEncodeTiled``: each ``boxDim[i] <= 256`` elements.
_TMA_MAX_BOX_DIM = 256
# NVIDIA's recommended TMA-destination alignment for box copies.
_TMA_ALIGN_BYTES = 128
# ``cuTensorMapEncodeTiled``: every global stride above the innermost dim must be a
# multiple of 16 bytes.
_TMA_STRIDE_ALIGN_BYTES = 16


def _inner_stride_aligned(dim, elem_bytes: int) -> bool:  # noqa: ANN001
    """True when a *symbolic* inner extent ``dim`` is a provable constant multiple
    whose byte width is a multiple of 16 — so the above-inner gmem stride stays
    16 B-aligned at every runtime value. A bare symbolic dim (no constant factor)
    is unaligned and stays off TMA."""
    expr = dim
    if not (isinstance(expr, BinaryExpr) and expr.op == "*"):
        return False
    factor = None
    if isinstance(expr.right, Literal) and isinstance(expr.right.value, int):
        factor = expr.right.value
    elif isinstance(expr.left, Literal) and isinstance(expr.left.value, int):
        factor = expr.left.value
    return factor is not None and (factor * elem_bytes) % _TMA_STRIDE_ALIGN_BYTES == 0


def _static_shape(shape) -> tuple[int, ...] | None:  # noqa: ANN001
    """A logical ``Buffer.shape`` (Expr tuple) as static ints for the compile-time
    box checks: static dims as-is, symbolic dims at their ``Dim`` hint, ``None``
    (cannot size the TMA box) for a symbolic dim with no hint."""
    dims: list[int] = []
    for d in shape:
        if d.is_static:
            dims.append(d.as_static())
        elif getattr(d, "hint", None) is not None:
            dims.append(int(d.hint))
        else:
            return None
    return tuple(dims)


def has_ringable_kloop(graph: TileGraph) -> bool:
    """True iff the block carries a ``serial_outer`` K loop with static extent ≥ 2
    — the prerequisite for the double-buffered TMA ring (a single-stage ``BK == K``
    whole-K slab has no K loop to pipeline, so it can't ring → stays SYNC)."""
    block = graph.blocks[0]
    for s in block.compute.iter():
        if isinstance(s, SerialTile) and s.kind == "serial_outer" and s.axis.extent.is_static and s.axis.extent.as_static() >= 2:
            return True
    return False


def _source_eligible(src: Source, src_shapes: dict, inner_symbolic_bufs: set[str]) -> bool:
    if not isinstance(src.addressing, AffineAddressing):
        return False
    if src.buf in inner_symbolic_bufs:
        return False
    cache_axes = src.cache_axes
    if not cache_axes or len(cache_axes) > _MAX_RANK:
        return False
    src_shape = src_shapes.get(src.buf)
    if not src_shape or len(src_shape) > _MAX_RANK or len(src.origin) > _MAX_RANK:
        return False
    dims = src.addressing.dims
    if not dims or list(dims) != sorted(dims):
        return False
    src_rank = len(src_shape)
    if dims[-1] != src_rank - 1:
        return False
    dims_set = set(dims)
    for d in range(dims[0], src_rank):
        if d not in dims_set and src_shape[d] != 1:
            return False
    # Collapsed box extent per source dim — product of (cache_extent × block) of
    # every cache axis mapping to that dim.
    block = src.addressing.block
    box_per_dim: dict[int, int] = {}
    for i, (d, ax) in enumerate(zip(dims, cache_axes, strict=True)):
        b = block[i] if block else 1
        box_per_dim[d] = box_per_dim.get(d, 1) * ax.extent.as_static() * b
    if any(ext > _TMA_MAX_BOX_DIM for ext in box_per_dim.values()):
        return False
    inner_extent = box_per_dim[dims[-1]]
    # Alignment checks use the lenient fp32 ``BYTES_PER_ELEM`` (matching the legacy
    # swizzle path): a 32-elem fp16 inner box (64 B true) reads as 128 B and stays
    # eligible — the swizzle atom keeps the ring-slot stride aligned.
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    if src_inner < inner_extent * 2:
        return False
    return True


def tma_eligible(graph: TileGraph, sources: list[Source], ctx: Context) -> bool:
    """All-or-nothing per tile: every prospective staged ``Source`` must be
    TMA-eligible, the device must be sm_90+, and the block must carry a ringable K
    loop. Mirrors the legacy ``050_use_tma`` gate over the block-DAG projections."""
    if ctx.compute_capability < _MIN_CAPABILITY:
        return False
    if not sources or not has_ringable_kloop(graph):
        return False
    src_shapes: dict[str, tuple[int, ...]] = {}
    for name, buf in graph.buffers.items():
        shp = _static_shape(buf.shape)
        if shp is None:
            return False
        src_shapes[name] = shp
    inner_symbolic_bufs = {
        name
        for name, buf in graph.buffers.items()
        if buf.shape and not buf.shape[-1].is_static and not _inner_stride_aligned(buf.shape[-1], buf.dtype.nbytes)
    }
    return all(_source_eligible(src, src_shapes, inner_symbolic_bufs) for src in sources)
