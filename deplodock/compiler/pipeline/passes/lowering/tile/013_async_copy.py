"""Narrow ``BufferedStage`` to ``AsyncBufferedStage`` (cp.async transport) on sm_80+.

For each ``BufferedStage`` in the Tile body, replace it with an
``AsyncBufferedStage`` (and append an explicit ``AsyncWait(keep=0)``
so the synchronous-style invariant — every async load dominated by a
wait before its consumer — holds even without pipelining) when:

- The target GPU's compute capability is sm_80 or higher (Ampere+).
- The slab footprint is large enough that the commit/wait overhead
  amortizes (currently ``≥ 4 elements per thread per Stage``).

The materializer expands ``AsyncBufferedStage`` to ``Smem`` +
cooperative ``CpAsyncCopy`` + ``CpAsyncCommit`` only (no implicit wait
or sync); the explicit ``AsyncWait`` lowers to ``CpAsyncWait(group=0)``
+ ``Sync()``. The pipelining pass (``015_pipeline_k_outer``) then drops
these synchronous-style waits and re-emits them at the pipelined
schedule positions.

Requires the upstream ``010_double_buffer`` pass to have promoted
plain ``Stage`` to ``BufferedStage``: cp.async without ``buffer_count >= 2``
gives no overlap, so a sync ``Stage`` is intentionally not eligible.

Idempotence: an ``AsyncBufferedStage`` is left alone.
"""

from __future__ import annotations

import logging

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, AsyncBufferedStage, AsyncWait, BufferedStage, TileOp, TmaBufferedStage
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (8, 0)
_MIN_BYTES_PER_THREAD = 16  # 4 fp32 elems


def rewrite(ctx: Context, root: Node) -> Graph | None:
    if ctx.compute_capability < _MIN_CAPABILITY:
        raise RuleSkipped(f"cp.async requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    n_threads = 1
    for ba in tile.axes:
        if ba.bind == BIND_THREAD:
            n_threads *= int(ba.axis.extent)

    new_tile_body = _process(tile.body, n_threads)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped(f"no Stage eligible for cp.async (need >= {_MIN_BYTES_PER_THREAD} bytes/thread)")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: Body, n_threads: int) -> Body:
    """Walk a body. Narrow eligible ``BufferedStage`` to
    ``AsyncBufferedStage``, inserting a trailing ``AsyncWait(drain_all)``
    so consumers see the loaded slab. Recurse into free Loops so Stages
    inside (e.g. the K-outer chunk loop) get processed."""
    new_body: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, BufferedStage) and not isinstance(s, (AsyncBufferedStage, TmaBufferedStage)) and _eligible(s, n_threads):
            new_body.append(
                AsyncBufferedStage(
                    name=s.name,
                    buf=s.buf,
                    origin=s.origin,
                    axes=s.axes,
                    addressing=s.addressing,
                    pad=s.pad,
                    body=s.body,
                    buffer_count=s.buffer_count,
                    phase=s.phase,
                )
            )
            new_body.append(AsyncWait(keep=0))
            changed = True
        elif isinstance(s, Loop):
            inner = _process(s.body, n_threads)
            if inner is not s.body and inner != s.body:
                new_body.append(Loop(axis=s.axis, body=inner, unroll=s.unroll))
                changed = True
            else:
                new_body.append(s)
        else:
            new_body.append(s)
    return tuple(new_body) if changed else body


def _eligible(stage: BufferedStage, n_threads: int) -> bool:
    slab_bytes = BYTES_PER_ELEM
    for ax in stage.axes:
        slab_bytes *= int(ax.extent)
    bytes_per_thread = max(BYTES_PER_ELEM, slab_bytes // max(1, n_threads))
    return bytes_per_thread >= _MIN_BYTES_PER_THREAD
