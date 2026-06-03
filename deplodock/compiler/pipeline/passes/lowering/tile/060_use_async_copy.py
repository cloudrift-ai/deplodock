"""Promote ``BufferedStage`` to ``AsyncBufferedStage`` (cp.async transport).

For each ``BufferedStage`` inside a ``SerialTile(serial_outer)``, swap
to ``AsyncBufferedStage(pipeline_depth=1)`` when the target supports
cp.async (sm_80+). All other fields (sources, body, buffer_count,
phase) pass through unchanged. The cooperative ``Load+Write`` becomes
``CpAsyncCopy``; the trailing sync becomes ``CpAsyncCommit +
CpAsyncWait(0) + Sync`` at the wrap boundary — emission lives in the
materializer's ``_emit_stage``.

``pipeline_depth = 1`` is the synchronous-style wait shape (every iter
issues, commits, waits before the consumer reads). M7
(``080_pipeline_stages``) bumps the depth on eligible
stages and expands the K-outer loop into prologue / main / epilogue
siblings whose waits sit at the pipelined schedule positions.

Requires the upstream ``040_use_ring_buffers`` to have run: cp.async
without ``buffer_count >= 2`` gives no producer-consumer overlap, so a
plain ``Stage`` is intentionally not eligible (the materializer's
async path also assumes ``is_buffered`` for its slab indexing).

Idempotence: an ``AsyncBufferedStage`` (or ``TmaBufferedStage``) is
left alone — already promoted past sync transport.
"""

from __future__ import annotations

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    SerialTile,
    StageBundle,
    StagePolicy,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (8, 0)

# Default on (sm_80+): promote double-buffered bundles to cp.async. Mirrors
# ``TMA`` in ``050_use_tma`` so the transport can be controlled explicitly.
# ``DEPLODOCK_ASYNC_COPY=0`` keeps the synchronous double-buffer — useful for
# A/B-benching sync-staged vs cp.async vs TMA on the same shape.
ASYNC_COPY = Knob(
    "ASYNC_COPY",
    KnobType.BOOL,
    hints=(True, False),
    help="Promote double-buffered (BUFFERED) bundles to cp.async (ASYNC). 0 = keep synchronous double-buffer.",
)


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    # Arch-gated default: cp.async needs sm_80+. Narrow the full hint tuple on
    # supported arch, ``(False,)`` otherwise; an explicit ``=0`` pin is honoured
    # at any arch.
    candidates = ASYNC_COPY.hints if ctx.compute_capability >= _MIN_CAPABILITY else (False,)
    if not ASYNC_COPY.narrow(candidates)[0]:
        if config.knob_raw(ASYNC_COPY.name) is not None:
            raise RuleSkipped("ASYNC_COPY=0 pinned")
        raise RuleSkipped(f"cp.async requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")

    body = root.op.body
    new_body, changed = _walk(body)
    if not changed:
        raise RuleSkipped("no BUFFERED StageBundle inside SerialTile(serial_outer) eligible for cp.async")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    """Recurse into wrappers; promote BUFFERED-policy bundles whose
    enclosing ``SerialTile(serial_outer)`` we're descending through."""
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            new_kouter_body, sub = _promote_in_kouter(s.body)
            if sub:
                s = SerialTile(axis=s.axis, body=new_kouter_body, kind=s.kind, unroll=s.unroll)
                changed = True
            out.append(s)
            continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _promote_in_kouter(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, StageBundle) and s.policy == StagePolicy.BUFFERED:
            out.append(_promote(s))
            changed = True
        else:
            out.append(s)
    return Body(tuple(out)), changed


def _promote(bundle: StageBundle) -> StageBundle:
    # cp.async.ca fires per-thread 4-byte copies; fp32 source layout is
    # naturally 4-byte aligned. fp16 sources fall back to the sync
    # cooperative-load path inside _emit_stage so we can over-promote
    # here without correctness risk.
    return StageBundle(
        sources=bundle.sources,
        body=bundle.body,
        compute=bundle.compute,
        policy=StagePolicy.ASYNC,
        buffer_count=bundle.buffer_count,
        phase=bundle.phase,
        pipeline_depth=1,
    )
