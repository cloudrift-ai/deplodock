"""Narrow ``BufferedStage`` to ``TmaBufferedStage`` (TMA transport) on sm_90+.

All-or-nothing: if every ``BufferedStage`` in the Tile body meets TMA's
eligibility constraints, replace them all with ``TmaBufferedStage`` and
append a trailing ``AsyncWait`` so the synchronous-style invariant —
every async load dominated by a wait before its consumer — holds even
without pipelining. If any stage is ineligible, leave the whole tile
alone so ``013_async_copy`` puts every stage on cp.async uniformly.

Mixed TMA + cp.async pipelined K-loops force a per-iter
``cp.async.wait_group + __syncthreads`` (otherwise the per-CTA
``cp.async.commit_group`` tracker overflows past the 64-group HW limit
under back-to-back launches and deadlocks the device). The required
Sync defeats the latency hiding the pipeline was supposed to provide.
Falling back to all-cp.async on mixed kernels avoids both the deadlock
and the perf cliff.

Eligibility:

- compute capability >= sm_90 (Hopper+).
- ``AffineAddressing`` only — TMA's box-copy semantics need
  ``origin + decoded`` reconstruction; symbolic ``TemplateAddressing``
  (collapsed-reshape views) stays on cp.async.
- rank ≤ 5 (TMA descriptor limit).
- ``addressing.dims`` is a strictly-increasing permutation ending at
  ``src_rank - 1`` — the cache axes sweep a contiguous source-dim
  suffix in row-major order. A non-identity permutation means a
  transpose / reshape was fused into the load (e.g. SDPA's K projection
  read as ``[head_dim, seq]`` from a ``[seq, head_dim]`` source); TMA's
  ``(globalDim, globalStride, boxDim)`` triple cannot encode that.
- box and source inner extents are 16 B-aligned, with the source
  offering ≥ 2× the box inner extent of headroom (small matmuls where
  these clamp to the alignment boundary trip ``MISALIGNED_ADDRESS`` at
  the ``cp.async.bulk.tensor`` site).

The materializer expands ``TmaBufferedStage`` to a ``TmaDescriptor`` +
``MbarrierInit`` (hoisted to kernel prologue) plus
``Cond(tid==0, [MbarrierArriveExpectTx, TmaLoad])`` at the stage site.
The trailing ``AsyncWait`` lowers to ``MbarrierWait(phase)`` (no
``Sync`` — mbarrier arrival already publishes CTA-wide).

Idempotence: a ``TmaBufferedStage`` is left alone.
"""

from __future__ import annotations

import logging

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncBufferedStage,
    AsyncWait,
    BufferedStage,
    SwizzleMode,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline.engine import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

logger = logging.getLogger(__name__)


# Swizzle picking moved entirely to ``012_split_inner_for_swizzle``.
# This pass promotes eligible stages with ``swizzle=NONE``; the follow-up
# pass picks the mode (and optionally splits the inner axis to fit a
# wider swizzle) and rewrites body Loads in one place.


PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
_TMA_ALIGN_BYTES = 16


def rewrite(ctx: Context, match: Match, root: Node) -> Graph | None:
    graph = match.graph
    # TMA is on by default on sm_90+ (Hopper / Blackwell — the hardware
    # that has ``cp.async.bulk.tensor``). ``DEPLODOCK_TMA=0`` forces
    # the cp.async + ``+1`` padding baseline for A/B comparison.
    from deplodock.compiler.tuning import _tma_enabled  # noqa: PLC0415

    if not _tma_enabled():
        raise RuleSkipped("TMA disabled (DEPLODOCK_TMA=0 or compute capability < sm_90)")
    if ctx.compute_capability < _MIN_CAPABILITY:
        raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")

    src_ranks = {nid: len(node.output.shape) for nid, node in graph.nodes.items()}
    src_shapes = {nid: tuple(int(d) for d in node.output.shape) for nid, node in graph.nodes.items()}
    new_body = _maybe_rewrite(root.op.body, src_ranks, src_shapes)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(
    body: Body,
    src_ranks: dict[str, int],
    src_shapes: dict[str, tuple[int, ...]],
) -> Body | None:
    idx, tile = single_tile(body)
    # All-or-nothing: every BufferedStage in the tile body must be
    # TMA-eligible, otherwise leave the whole tile for cp.async. A
    # mixed TMA + cp.async pipelined K-loop needs a per-iter
    # ``CpAsyncWait + __syncthreads`` (otherwise the per-CTA
    # ``cp.async.commit_group`` tracker overflows past 64 under
    # back-to-back launches and deadlocks the device); the required
    # Sync also destroys the latency hiding the pipeline was supposed
    # to provide. Falling back to all-cp.async on mixed kernels avoids
    # both the hang and the perf cliff.
    for s in tile.body.iter():
        if isinstance(s, (AsyncBufferedStage, TmaBufferedStage)):
            continue
        if isinstance(s, BufferedStage) and not _eligible(s, src_ranks, src_shapes):
            raise RuleSkipped(
                f"stage {s.name!r} (buf={s.buf!r}) not TMA-eligible; "
                "leaving every stage in this tile to cp.async (avoids mixed-mode pipeline deadlock)"
            )
    new_tile_body = _process(tile.body, src_ranks, src_shapes)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no BufferedStage to convert")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(
    body: Body,
    src_ranks: dict[str, int],
    src_shapes: dict[str, tuple[int, ...]],
    k_var: str | None = None,
) -> Body:
    """Walk a body. Narrow eligible ``BufferedStage`` to ``TmaBufferedStage``,
    inserting a trailing ``AsyncWait`` so consumers see the loaded slab.
    Recurse into free Loops so Stages inside the K-outer chunk loop are
    processed.

    ``k_var`` is the immediate-parent ``Loop`` axis name (or ``None`` if
    the body sits at Tile scope). When set and a TMA stage's ``buffer_count
    >= 2``, the inserted ``AsyncWait`` carries the consumer-side mbar
    ``slot`` and ``phase`` (matching what ``015_pipeline_k_outer`` sets) so
    the materializer lowers the wait to ``MbarrierWait`` rather than the
    cp.async fallback (which doesn't actually wait for TMA bulk loads)."""
    new_body: list[Stmt] = []
    changed = False
    pending_wait: tuple[BufferedStage, ...] = ()  # consecutive TMA stages awaiting one shared wait
    i = 0
    while i < len(body):
        s = body[i]
        if (
            isinstance(s, BufferedStage)
            and not isinstance(s, (AsyncBufferedStage, TmaBufferedStage))
            and _eligible(s, src_ranks, src_shapes)
        ):
            new_body.append(
                TmaBufferedStage(
                    name=s.name,
                    buf=s.buf,
                    origin=s.origin,
                    axes=s.axes,
                    addressing=s.addressing,
                    pad=s.pad,
                    buffer_count=s.buffer_count,
                    phase=s.phase,
                    swizzle=SwizzleMode.NONE,
                )
            )
            pending_wait = (*pending_wait, s)
            changed = True
        else:
            if pending_wait:
                # Materialize emits ONE shared mbarrier with arrive count =
                # number of TMA stages. Coalesce all consecutive stages'
                # waits into a single AsyncWait so the materializer lowers
                # exactly one MbarrierWait that releases the consumer once
                # every stage has arrived. Per-stage waits would deadlock —
                # the first wait would block on count=N but only one stage
                # has arrived.
                new_body.append(_tma_async_wait(pending_wait[0], k_var))
                pending_wait = ()
            if isinstance(s, Loop):
                inner = _process(s.body, src_ranks, src_shapes, k_var=s.axis.name)
                if inner is not s.body and inner != s.body:
                    new_body.append(Loop(axis=s.axis, body=inner, unroll=s.unroll))
                    changed = True
                else:
                    new_body.append(s)
            else:
                new_body.append(s)
        i += 1
    if pending_wait:
        new_body.append(_tma_async_wait(pending_wait[0], k_var))
    return tuple(new_body) if changed else body


def _tma_async_wait(stage: BufferedStage, k_var: str | None) -> AsyncWait:
    """Build the ``AsyncWait`` that follows a synchronous-style TMA stage.

    With a parent K-outer loop, supply the consumer-side ring-buffer
    ``slot`` (``k_var % buffer_count``) and ``phase``
    (``(k_var / buffer_count) % 2``). Without a parent loop the slot is
    constant and the mbar is fresh — slot 0, phase 0 suffices for a
    single-shot TMA load."""
    bc = stage.buffer_count
    if k_var is None:
        return AsyncWait(keep=0, phase=Literal(0, "int"), slot=Literal(0, "int"))
    slot = BinaryExpr("%", Var(k_var), Literal(bc, "int"))
    phase = BinaryExpr("%", BinaryExpr("/", Var(k_var), Literal(bc, "int")), Literal(2, "int"))
    return AsyncWait(keep=0, phase=phase, slot=slot)


def _eligible(stage: BufferedStage, src_ranks: dict[str, int], src_shapes: dict[str, tuple[int, ...]]) -> bool:
    if not isinstance(stage.addressing, AffineAddressing):
        return False
    if len(stage.axes) > _MAX_RANK or not stage.axes:
        return False
    # The TMA descriptor encoder rejects source ranks > 5 (CUDA driver
    # limit). Stage's box rank == len(stage.origin) == source rank, but
    # for safety we also check the source buffer's actual shape rank
    # via the graph context — multi-head SDPA buffers can be 6-D and
    # reach this pass even after fusion / lifting.
    src_rank = src_ranks.get(stage.buf, len(stage.origin))
    if src_rank > _MAX_RANK or len(stage.origin) > _MAX_RANK:
        return False
    # Layout check: ``addressing.dims`` records which source dim each
    # cache axis sweeps. For TMA's box-copy semantics in a row-major
    # source, the cache axes must form a strictly-increasing contiguous
    # suffix of source dims, ending at the innermost (highest-index)
    # one. Any deviation means a transpose / reshape was fused into the
    # load (e.g. SDPA's V read with ``dims=(1, 3)`` skipping dim 2 from
    # a per-head slice) — TMA's ``(globalDim, globalStride, boxDim)``
    # triple cannot encode that, so the tile falls back to cp.async.
    # (A relaxed singleton-skip variant was tried — TMA-encoded but
    # deadlocked at seq=512 scale; the strict check is the proven gate.)
    dims = stage.addressing.dims
    if not dims or len(set(dims)) != len(dims):
        return False
    if list(dims) != sorted(dims):
        return False
    if dims[-1] != src_rank - 1:
        return False
    src_shape = src_shapes.get(stage.buf)
    if src_shape is None or not src_shape:
        return False
    # Permit non-contiguous suffixes only when every gap source dim is an
    # extent-1 singleton — those dims get dropped from the descriptor at
    # materialization time (see ``_001_materialize_tile.emit_tma_stage``)
    # so the encoded rank matches the swept rank, sidestepping the
    # rank-4-pipelined-TMA deadlock.
    dims_set = set(dims)
    for d in range(dims[0], src_rank):
        if d not in dims_set and int(src_shape[d]) != 1:
            return False
    inner_extent = int(stage.axes[-1].extent)
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    # CUDA TMA additionally requires the SOURCE's innermost-dim stride
    # (= source inner-dim extent × elemSize) and globalDim[0] × elemSize
    # to be 16-byte multiples. Our 016c smem-pad pass and the asymmetric
    # tile assume the source row stride is a clean multiple of the box
    # inner extent in bytes; for tiny matmuls (e.g. innermost=4 elem at
    # fp32 = exactly 16 B) the descriptor encodes successfully but the
    # generated kernel hits ``CUDA_ERROR_MISALIGNED_ADDRESS`` at the
    # ``cp.async.bulk.tensor`` site. Require both inner extents (box
    # and source) to be strict multiples of ``_TMA_ALIGN_BYTES`` AND the
    # source inner extent to be at least one full alignment quantum
    # past the box — i.e. the source must offer headroom beyond a single
    # box load.
    src_inner = int(src_shape[-1])
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    if src_inner < inner_extent * 2:
        return False
    return True
