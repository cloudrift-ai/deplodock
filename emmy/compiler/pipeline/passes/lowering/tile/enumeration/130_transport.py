"""transport pass (enumeration fork) — ``promote_transport``.

``promote_transport(read, →TMA)`` writes the
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

Scope: any **staged matmul** with a ringable K loop — the **warp-tier** ``mma.sync`` atom
*and* the **scalar** register-tiled SGEMM (the matmul-optimization blogs' hero
``TM=26`` fp32 tile). The two differ only in how the slab is read back: the warp tier's
``ldmatrix`` consumer reads a hardware-swizzled deposit (B64 / B128), so ``assembly/_slab``
swizzle-stamps its sources; the scalar tier reads with plain affine ``Load``s, so its
deposit stays linear (``SwizzleMode.NONE``) — that branch lives in ``_slab._make_bundle``
(keyed on ``Block.atom``), not here. **TMA is offered first (option-0) when the tile is
eligible**, so a cold / greedy compile takes it: the bulk-async ring is 1.3-1.9x faster
than SYNC on every eligible tile and the choice is then deterministic regardless of the
loaded prior (the old SYNC-first default let a pinned config's transport flip with the
prior — see ``rewrite``). The tuner still explores SYNC (the second offer) for any tile
where it wins; a ``EMMY_TMA=0`` pin forces it directly. An ineligible tile stays SYNC.
Cooperative-reduce / pointwise tiers stage nothing ringable, so they fall out via the empty
``schedule.staged`` / no-ringable-K-loop guards (``RuleSkipped`` → ``apply_off_defaults``
stamps ``TMA=False``).

The TMA-eligibility predicate (:func:`tma_eligible`, ported from the deleted legacy
``050_use_tma._source_eligible``) reads only the **derived** projections — the prospective
smem ``Source`` ``assembly/_slab`` would build for each staged read-site (its
``AffineAddressing`` + cache-axis box) plus the logical gmem ``Buffer.shape`` — never tower
shape.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SerialTile,
    Source,
    TileGraphOp,
    Transport,
)
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile.assembly._slab import prospective_sources
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam

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


def _source_eligible(src: Source, src_shapes: dict, inner_symbolic_bufs: set[str], *, swizzled: bool) -> bool:
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
    # Slot / box 128 B alignment, tier-aware. The WARP tier's slab is hardware-swizzled
    # (``swizzled`` — ``_slab`` stamps a B64/B128 atom keyed on ``block.atom``), and the
    # swizzle keeps each ring slot aligned, so the lenient fp32 ``BYTES_PER_ELEM`` is
    # sound there: a 32-elem fp16 inner box (64 B true) reads as 128 B and stays eligible.
    # The SCALAR tier deposits LINEARLY (``SwizzleMode.NONE``, plain affine ``Load``s), so
    # a sub-128 B box genuinely lands the second ``RING`` slot at a misaligned offset and
    # ``cp.async.bulk.tensor`` faults with ``CUDA_ERROR_MISALIGNED_ADDRESS`` (the #244 fp16
    # ``BK=32`` = 64 B wedge). Size the check off the TRUE dtype width there so such a slab
    # declines TMA → cp.async.
    elem_bytes = BYTES_PER_ELEM if swizzled else (src.dtype.nbytes if src.dtype is not None else BYTES_PER_ELEM)
    if (inner_extent * elem_bytes) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * elem_bytes) % _TMA_ALIGN_BYTES != 0:
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
    # The slab is hardware-swizzled only on the warp/atom tier (``_slab`` keys the
    # swizzle on ``block.atom``); a scalar tile deposits linearly and needs a truly
    # 128 B-aligned box (see ``_source_eligible``).
    swizzled = graph.blocks[0].atom is not None
    return all(_source_eligible(src, src_shapes, inner_symbolic_bufs, swizzled=swizzled) for src in sources)


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    smem_edges = fam.smem_edges_without_xport(op.knobs)
    if not smem_edges:
        raise RuleSkipped("transport runs once, after stage placed smem read-sites (idempotence via PLACE :xport)")
    if not op.tilegraph.schedule.staged:
        raise RuleSkipped("nothing staged — no read-site to promote")

    eligible = ctx.compute_capability >= _MIN_CAPABILITY and tma_eligible(op.tilegraph, prospective_sources(op.tilegraph), ctx)
    pin = fam.pin_xport()
    if pin is not None:
        # A pin is authoritative; an ineligible pinned-on shape declines gracefully to SYNC.
        decisions = [pin and eligible]
    else:
        # TMA-first when eligible (option-0 = the cold / analytic / greedy default). The
        # strict ``tma_eligible`` gate only passes tiles that benefit, and the bulk-async
        # ring measures 1.3-1.9x faster than SYNC on every eligible matmul (square.1024
        # 51 vs 93 us, o_proj.s512 52 vs 97). Making it the default keeps the deployed
        # kernel DETERMINISTIC regardless of the loaded prior — the SYNC-first default let
        # a pinned golden's transport flip with the prior (TMA only when the learned prior
        # ranked it), so the same golden knobs benched 51 us (prior) vs 93 us (cold), and
        # the no-TMA cold/analytic path was needlessly slow. The tuner still explores SYNC
        # (the second offer) for any tile where it wins.
        decisions = [True, False] if eligible else [False]

    out: list[TileGraphOp] = []
    for use in decisions:
        if use:
            staged = {e: Transport.TMA for e in op.tilegraph.schedule.staged}
            tg = replace(op.tilegraph, schedule=replace(op.tilegraph.schedule, staged=staged))
            xport = "tma"
        else:
            tg = op.tilegraph
            xport = "sync"
        # Record the chosen transport on each staged ``PLACE@<edge>`` (``smem`` → ``smem:tma``
        # / ``smem:sync``); ``Schedule.staged`` already carries the Transport for codegen.
        place_updates = {k: fam.enc_place(fam.SMEM, xport) for k in smem_edges}
        out.append(replace(op, tilegraph=tg, knobs={**op.knobs, **place_updates}))
    return out
