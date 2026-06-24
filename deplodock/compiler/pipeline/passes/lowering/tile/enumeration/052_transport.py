"""transport pass (enumeration fork) — ``promote_transport`` (R5).

``plans/tile-ir-block-dag.md`` R5: ``promote_transport(read, →TMA)`` writes the
``Schedule.staged[edge]`` value (``SYNC`` → ``TMA``); it edits **no body**. The smem
slab's swizzle + the double-buffered ``cp.async.bulk.tensor`` ring are synthesized later
by ``assembly/_slab`` from the annotation, and ``assembly/020_peel`` software-pipelines
the K loop. This pass is the genuine fork: ``TMA ∈ {True, False}`` is a ranked knob the
search benches.

Pre-assemble, over the fully-staged stored algorithm: it reads the prospective smem
``Source``s ``assemble`` would build (``assembly/_slab.prospective_sources`` — a derived
projection, no tower) and the TMA-eligibility oracle (:func:`tma_eligible` below:
sm_90+, affine box ≤ 256 / 16 B-aligned, a ringable K loop), then writes the chosen
``Transport.TMA`` straight into ``Schedule.staged``.

R5 scope: the **warp-tier** ``mma.sync`` matmul (an ``Atom`` is pinned). Greedy stays
byte-identical to today — the SYNC decision is offered first (option-0), so a cold compile
keeps SYNC staging; the TMA variant is the second offer the tuner explores (or a
``DEPLODOCK_TMA=1`` pin selects directly). Scalar / cooperative-reduce / pointwise tiers
skip here (``RuleSkipped`` → ``apply_off_defaults`` stamps ``TMA=False``).

The TMA-eligibility predicate (:func:`tma_eligible`, ported from the deleted legacy
``050_use_tma._source_eligible``) reads only the **derived** projections — the prospective
smem ``Source`` ``assembly/_slab`` would build for each staged read-site (its
``AffineAddressing`` + cache-axis box) plus the logical gmem ``Buffer.shape`` — never tower
shape. The staged operands feed ``ldmatrix`` and the slab is swizzled (B64 / B128); the
scalar cooperative-reduce / pointwise tiers stay on SYNC staging (their TMA promotion, and
its fp16-ring-slot alignment decline, rides a later follow-up).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SerialTile,
    Source,
    TileGraphOp,
    Transport,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import prospective_sources
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import STAGE, TMA

PATTERN = [Pattern("root", TileGraphOp)]

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


def _has_ringable_kloop(graph) -> bool:  # noqa: ANN001
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


def tma_eligible(graph, sources: list[Source], ctx: Context) -> bool:  # noqa: ANN001
    """All-or-nothing per tile: every prospective staged ``Source`` must be
    TMA-eligible, the device must be sm_90+, and the block must carry a ringable K
    loop. Mirrors the legacy ``050_use_tma`` gate over the block-DAG projections."""
    if ctx.compute_capability < _MIN_CAPABILITY:
        return False
    if not sources or not _has_ringable_kloop(graph):
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


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if STAGE.name not in op.knobs or TMA.name in op.knobs:
        raise RuleSkipped("transport runs once, after stage decided the staged read-sites (idempotence via the TMA knob)")
    if not op.tilegraph.schedule.staged:
        raise RuleSkipped("nothing staged — no read-site to promote")

    eligible = ctx.compute_capability >= _MIN_CAPABILITY and tma_eligible(op.tilegraph, prospective_sources(op.tilegraph), ctx)
    pin = TMA.raw()
    if pin is not None:
        # A pin is authoritative; an ineligible pinned-on shape declines gracefully to SYNC.
        decisions = [TMA.parse(pin) and eligible]
    else:
        # Greedy-safe: SYNC first (option-0, byte-identical to today), TMA second when eligible.
        decisions = [False, *([True] if eligible else [])]

    out: list[TileGraphOp] = []
    for use in decisions:
        if use:
            staged = {e: Transport.TMA for e in op.tilegraph.schedule.staged}
            tg = replace(op.tilegraph, schedule=replace(op.tilegraph.schedule, staged=staged))
        else:
            tg = op.tilegraph
        out.append(replace(op, tilegraph=tg, knobs={**op.knobs, TMA.name: use}))
    return out
