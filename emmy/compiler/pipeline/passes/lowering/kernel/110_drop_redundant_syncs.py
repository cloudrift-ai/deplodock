"""Drop redundant ``Sync`` stmts left by the materializer's templates.

A materializer that stages smem emits a defensive ``Sync()`` at several
template boundaries (stage prologue, combine, TMA wait). Many collapse: two
consecutive ``Sync``s are one, and a leading ``Sync`` before any smem access
fences nothing. This Kernel-IR peephole runs after materialize and before the
CUDA lowering.

Scope: the immediate body of each :class:`Tile` (the thread-schedule
wrapper). No descent into nested ``Loop`` / ``Cond`` bodies, where the syncs
are load-bearing.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp, Tile
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    Sync,
    TmaLoad,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

# Stmts that touch smem (so a following Sync has something to fence).
_SMEM_TOUCHING = (
    Smem,
    MbarrierInit,
    MbarrierArriveExpectTx,
    MbarrierWait,
    TmaLoad,
    CpAsyncCopy,
    CpAsyncCommit,
    CpAsyncWait,
    TreeHalve,
    WarpShuffle,
)


def _drop_redundant_syncs(body: tuple[Stmt, ...]) -> list[Stmt]:
    """Drop ``Sync`` stmts that are guaranteed no-ops at the body level:

    * Two consecutive ``Sync`` stmts collapse to one.
    * A leading ``Sync`` before any smem access is unnecessary — at kernel
      entry no thread can hold a stale view of smem because nothing has
      been written yet.

    Body-level only (no descent into nested ``Loop`` / ``Cond`` bodies);
    the bulk of redundant syncs come from materializer templates that emit
    a defensive ``Sync()`` at template boundaries, and those surface here.
    """
    smem_seen = False
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Sync):
            if not smem_seen:
                continue  # nothing for the sync to fence yet
            if out and isinstance(out[-1], Sync) and out[-1] == s:
                # Back-to-back identical syncs collapse. Equality (frozen
                # dataclass) covers barrier_id + count, so a default
                # ``Sync()`` and a named ``Sync(1, 192)`` placed adjacently
                # are preserved as distinct (different semantics).
                continue
        elif isinstance(s, _SMEM_TOUCHING):
            smem_seen = True
        out.append(s)
    return out


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    changed = False

    def clean_tile(t: Tile) -> Tile:
        nonlocal changed
        deduped = _drop_redundant_syncs(tuple(t.body))
        if list(deduped) != list(t.body):
            changed = True
            # ``with_bodies`` preserves ``block_threads`` — a bare ``Tile(axes, body)``
            # would drop it, reverting a cooperative / staged tile's ``blockDim``.
            return t.with_bodies((Body(deduped),))
        return t

    new_body: list[Stmt] = [clean_tile(s) if isinstance(s, Tile) else s for s in op.body]

    if not changed:
        raise RuleSkipped("no redundant syncs at the tile body level")
    return KernelOp(body=Body(new_body), name=op.name, knobs=dict(op.knobs))
