"""Drop redundant ``Sync`` stmts left by the materializer's templates.

``008_materialize_tile`` emits a defensive ``Sync()`` at several template
boundaries (stage prologue, combine, TMA wait). Many collapse: two
consecutive ``Sync``s are one, and a leading ``Sync`` before any smem
access fences nothing. This Kernel-IR peephole runs after materialize
and before the CUDA lowering.

Scope matches the materializer's original in-line cleanup exactly: the
immediate body of the (cooperative) ``ThreadTile`` — inside an optional
``GridTile`` wrapper. No descent into nested ``Loop`` / ``Cond`` bodies,
where the syncs are load-bearing.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    Sync,
    TmaLoad,
    TreeHalve,
    WarpShuffle,
)
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile
from deplodock.compiler.pipeline import Pattern, RuleSkipped

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
            if out and isinstance(out[-1], Sync):
                continue  # back-to-back sync collapses
        elif isinstance(s, _SMEM_TOUCHING):
            smem_seen = True
        out.append(s)
    return out


def rewrite(root: Node) -> Graph | None:
    op: KernelOp = root.op
    changed = False

    def clean_thread_tile(tt: ThreadTile) -> ThreadTile:
        nonlocal changed
        deduped = _drop_redundant_syncs(tuple(tt.body))
        if list(deduped) != list(tt.body):
            changed = True
        return ThreadTile(axes=tt.axes, body=Body(deduped))

    new_body: list[Stmt] = []
    for s in op.body:
        if isinstance(s, GridTile):
            new_children = [clean_thread_tile(c) if isinstance(c, ThreadTile) else c for c in s.body]
            new_body.append(GridTile(axes=s.axes, body=Body(new_children)))
        elif isinstance(s, ThreadTile):
            new_body.append(clean_thread_tile(s))
        else:
            new_body.append(s)

    if not changed:
        raise RuleSkipped("no redundant syncs at the tile body level")
    return KernelOp(body=new_body, name=op.name)
