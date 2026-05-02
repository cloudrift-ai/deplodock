"""Narrow ``BufferedStage`` to ``TmaBufferedStage`` (TMA transport) on sm_90+.

For each ``BufferedStage`` in the Tile body that meets TMA's eligibility
constraints, replace it with a ``TmaBufferedStage`` and append an
``AsyncWait(keep=0)`` so the synchronous-style invariant — every async
load dominated by a wait before its consumer — holds even without
pipelining.

Runs *before* ``014b_async_copy``: a Stage that's TMA-eligible is taken
by this pass; everything else falls through to cp.async.

Eligibility:

- compute capability >= sm_90 (Hopper+).
- ``AffineAddressing`` only — TMA's box-copy semantics need
  ``origin + decoded`` reconstruction; symbolic ``TemplateAddressing``
  (collapsed-reshape views) stays on cp.async.
- rank ≤ 5 (TMA descriptor limit).
- inner source dim is the contiguous one and its box extent in bytes
  is a multiple of 16 B (TMA alignment requirement).

The materializer expands ``TmaBufferedStage`` to a ``TmaDescriptor`` +
``MbarrierInit`` (hoisted to kernel prologue) plus
``Cond(tid==0, [MbarrierArriveExpectTx, TmaLoad])`` at the stage site.
The trailing ``AsyncWait`` lowers to ``MbarrierWait(phase)`` (no
``Sync`` — mbarrier arrival already publishes CTA-wide).

Idempotence: a ``TmaBufferedStage`` is left alone.
"""

from __future__ import annotations

import logging

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncBufferedStage,
    AsyncWait,
    BufferedStage,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import compute_capability, single_tile

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
_TMA_ALIGN_BYTES = 16


def rewrite(graph: Graph, root: Node) -> Graph | None:
    # TMA is on by default on sm_90+ (Hopper / Blackwell — the hardware
    # that has ``cp.async.bulk.tensor``). ``DEPLODOCK_TMA=0`` forces
    # the cp.async + ``+1`` padding baseline for A/B comparison.
    from deplodock.compiler.tuning import _tma_enabled  # noqa: PLC0415

    if not _tma_enabled():
        raise RuleSkipped("TMA disabled (DEPLODOCK_TMA=0 or compute capability < sm_90)")
    if compute_capability() < _MIN_CAPABILITY:
        raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {compute_capability()}")
    from deplodock.compiler.ir.base import ConstantOp  # noqa: PLC0415

    src_ranks = {nid: len(node.output.shape) for nid, node in graph.nodes.items()}
    # TMA only fires on stages reading from a transposed weight constant
    # (``004a_fold_into_constant`` recorded a ``TransposeOp`` in the
    # ``ConstantOp.load_ops`` chain). Activation stages have empty chains
    # and don't match the asymmetric (BN, BM) tile's KN expectation —
    # letting TMA load them produces silently wrong outputs (e.g. SDPA Q/K/V).
    tma_buf_ok = {nid for nid, node in graph.nodes.items() if isinstance(node.op, ConstantOp) and bool(node.op.load_ops)}
    new_body = _maybe_rewrite(root.op.body, src_ranks, tma_buf_ok)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body, src_ranks: dict[str, int], tma_buf_ok: set[str]) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process(tile.body, src_ranks, tma_buf_ok)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no BufferedStage eligible for TMA")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: Body, src_ranks: dict[str, int], tma_buf_ok: set[str], k_var: str | None = None) -> Body:
    """Walk a body. Narrow eligible ``BufferedStage`` to ``TmaBufferedStage``,
    inserting a trailing ``AsyncWait`` so consumers see the loaded slab.
    Recurse into free Loops so Stages inside the K-outer chunk loop are
    processed.

    ``k_var`` is the immediate-parent ``Loop`` axis name (or ``None`` if
    the body sits at Tile scope). When set and a TMA stage's ``buffer_count
    >= 2``, the inserted ``AsyncWait`` carries the consumer-side mbar
    ``slot`` and ``phase`` (matching what ``015_pipeline_async`` sets) so
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
            and _eligible(s, src_ranks, tma_buf_ok)
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
                inner = _process(s.body, src_ranks, tma_buf_ok, k_var=s.axis.name)
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


def _eligible(stage: BufferedStage, src_ranks: dict[str, int], tma_buf_ok: set[str]) -> bool:
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
    # Only weight-constant stages are TMA-safe — 004a transposed them
    # into the asymmetric tile's KN expectation. Activation stages keep
    # their natural layout and produce wrong outputs under TMA.
    if stage.buf not in tma_buf_ok:
        return False
    # Inner cache axis must be the innermost source dim — TMA box copies
    # require a contiguous inner-dim sweep.
    inner_dim = stage.addressing.dims[-1]
    expected_inner = max(stage.addressing.dims, default=-1)
    if inner_dim != expected_inner:
        return False
    inner_extent = int(stage.axes[-1].extent)
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    return True
