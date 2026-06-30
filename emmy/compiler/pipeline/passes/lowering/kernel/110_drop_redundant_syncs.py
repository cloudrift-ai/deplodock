"""Drop redundant ``Sync`` stmts left by the materializer's templates.

``100_materialize_tile`` emits a defensive ``Sync()`` at several template
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

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.kernel.ir import (
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
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.ir.tile.ir import GridTile, ThreadTile, WarpTile
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


def rewrite(root: Node) -> Graph | None:
    op: KernelOp = root.op
    changed = False

    def clean_parallel_tile(tt: ThreadTile | WarpTile) -> ThreadTile | WarpTile:
        nonlocal changed
        deduped = _drop_redundant_syncs(tuple(tt.body))
        if list(deduped) != list(tt.body):
            changed = True
        if isinstance(tt, WarpTile):
            return WarpTile(axes=tt.axes, body=Body(deduped))
        return ThreadTile(axes=tt.axes, body=Body(deduped))

    new_body: list[Stmt] = []
    for s in op.body:
        if isinstance(s, GridTile):
            new_children = [clean_parallel_tile(c) if isinstance(c, (ThreadTile, WarpTile)) else c for c in s.body]
            new_body.append(GridTile(axes=s.axes, body=Body(new_children), swizzle_group_m=s.swizzle_group_m))
        elif isinstance(s, (ThreadTile, WarpTile)):
            new_body.append(clean_parallel_tile(s))
        else:
            new_body.append(s)

    if not changed:
        raise RuleSkipped("no redundant syncs at the tile body level")
    return KernelOp(body=new_body, name=op.name)
