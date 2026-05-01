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
import os

from deplodock.compiler.graph import Graph, Node
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
    # Backend wiring (descriptor encoding, sm_90a compile flag, kernel-arg
    # binding) lands incrementally — gate the whole pass behind an env
    # var so the cp.async path remains the default until the TMA backend
    # is end-to-end ready. Set ``DEPLODOCK_TMA=1`` to opt in.
    if os.environ.get("DEPLODOCK_TMA") != "1":
        raise RuleSkipped("TMA path disabled (set DEPLODOCK_TMA=1 to enable)")
    if compute_capability() < _MIN_CAPABILITY:
        raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {compute_capability()}")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process(tile.body)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no BufferedStage eligible for TMA")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: Body) -> Body:
    """Walk a body. Narrow eligible ``BufferedStage`` to ``TmaBufferedStage``,
    inserting a trailing ``AsyncWait(0)`` so consumers see the loaded slab.
    Recurse into free Loops so Stages inside the K-outer chunk loop are
    processed."""
    new_body: list[Stmt] = []
    changed = False
    for s in body:
        if (
            isinstance(s, BufferedStage)
            and not isinstance(s, (AsyncBufferedStage, TmaBufferedStage))
            and _eligible(s)
        ):
            # The pad pass (``014c_pad_smem_banks``) runs AFTER this rule
            # and skips TmaBufferedStage by class, so this conversion sees
            # ``s.pad == ()`` and can pass it through. ``TmaBufferedStage``
            # also asserts ``pad`` is empty, which catches regressions.
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
            new_body.append(AsyncWait(keep=0))
            changed = True
        elif isinstance(s, Loop):
            inner = _process(s.body)
            if inner is not s.body and inner != s.body:
                new_body.append(Loop(axis=s.axis, body=inner, unroll=s.unroll))
                changed = True
            else:
                new_body.append(s)
        else:
            new_body.append(s)
    return tuple(new_body) if changed else body


def _eligible(stage: BufferedStage) -> bool:
    if not isinstance(stage.addressing, AffineAddressing):
        return False
    if len(stage.axes) > _MAX_RANK or not stage.axes:
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
