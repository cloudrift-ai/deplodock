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

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    AsyncBufferedStage,
    BufferedStage,
    SerialTile,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (8, 0)


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    if ctx.compute_capability < _MIN_CAPABILITY:
        raise RuleSkipped(f"cp.async requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")

    body = root.op.body
    new_body, changed = _walk(body)
    if not changed:
        raise RuleSkipped("no BufferedStage inside SerialTile(serial_outer) eligible for cp.async")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    """Recurse into wrappers; promote ``BufferedStage`` whose enclosing
    ``SerialTile(serial_outer)`` we're descending through."""
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
        if _is_promotable(s):
            out.append(_promote(s))
            changed = True
        else:
            out.append(s)
    return Body(tuple(out)), changed


def _is_promotable(s: Stmt) -> bool:
    if not isinstance(s, BufferedStage):
        return False
    if isinstance(s, (AsyncBufferedStage, TmaBufferedStage)):
        return False
    # cp.async.ca fires per-thread 4-byte copies; fp32 source layout
    # (the stage's smem slab is one element per thread per iter) is
    # naturally 4-byte aligned for any addressing the materializer
    # produces from a wrap-body Source. fp16 sources fall back to the
    # sync cooperative-load path inside _emit_stage — the CpAsyncCommit
    # / CpAsyncWait pair around an empty issue group is a no-op on the
    # hardware, so we can over-promote here without correctness risk.
    return True


def _promote(stage: BufferedStage) -> AsyncBufferedStage:
    return AsyncBufferedStage(
        sources=stage.sources,
        body=stage.body,
        buffer_count=stage.buffer_count,
        phase=stage.phase,
        pipeline_depth=1,
    )
