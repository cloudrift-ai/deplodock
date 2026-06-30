"""Drop duplicate ``Load`` stmts within each fused ``LoopOp`` body.

Runs after ``010_merge_loop_ops`` so the splicer sees un-deduped producer
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

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import dedup_loads
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(root: Node) -> Graph | None:
    new_body = dedup_loads(root.op.body)
    if new_body == root.op.body:
        raise RuleSkipped("no duplicate (input, index) Loads to dedup")
    return LoopOp(body=new_body)
