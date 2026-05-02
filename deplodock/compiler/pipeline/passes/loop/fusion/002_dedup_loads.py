"""Drop duplicate ``Load`` stmts within each fused ``LoopOp`` body.

Runs after ``001_merge_loop_ops`` so the splicer sees un-deduped producer
bodies (its prepend-at-leaf worklist relies on the original Load
multiplicity for defined-before-use ordering). Once fusion is settled,
identical ``(input, index)`` Loads at the same scope are equivalent —
``dedup_loads`` keeps the first and rewires SSA references to it.

Subsequent Tile / Kernel lowering carries the deduped bodies forward, so
the emitted CUDA kernel issues each external read once even when the
original expression referenced the same tensor many times (e.g. tanh-GELU
fuses multiple ``x`` loads into one).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import dedup_loads
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = dedup_loads(root.op.body)
    if new_body == root.op.body:
        raise RuleSkipped("no duplicate (input, index) Loads to dedup")
    root.op = LoopOp(body=new_body)
    return None
