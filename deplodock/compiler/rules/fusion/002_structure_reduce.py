"""Recognize reduce-chain kernel shape inside a flat-prologue KernelOp.

Matches a KernelOp whose prologue contains one or more ``ReduceOp`` nodes
interleaved with elementwise ops. Rewrites the KernelOp so that:

- Ops before the first reduce stay in ``prologue``
- Each reduce plus its preceding inter-reduce elementwise chain becomes
  a ``ReduceStage``; stages collected into ``core: tuple[ReduceStage, ...]``
- Ops after the last reduce move to ``epilogue``

Handles single-reduce (row reduction, RMSNorm) and multi-reduce (softmax:
max → sub → exp → sum → div) uniformly. Runs after the contraction rule
so `sum(mul(a, b))` patterns are already claimed.

Pattern: ``_`` (wildcard — filter to KernelOp in rewrite since KernelOps
have variable input counts).
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import (
    ElementwiseOp,
    KernelOp,
    ReduceOp,
    ReduceStage,
)

PATTERN = "_"


def rewrite(graph: Graph, match: Match) -> Graph:
    node = graph.nodes.get(match.root_node_id)
    if node is None or not isinstance(node.op, KernelOp):
        return graph

    kernel = node.op
    if kernel.core is not None:
        return graph

    # Find all reduce ops in the prologue.
    reduce_indices = [i for i, n in enumerate(kernel.prologue) if isinstance(n.op, ReduceOp)]
    if not reduce_indices:
        return graph

    # Validate: ops between reduces (if any) must be pure elementwise chains.
    for i in range(len(reduce_indices) - 1):
        inter = kernel.prologue[reduce_indices[i] + 1 : reduce_indices[i + 1]]
        if any(not isinstance(n.op, ElementwiseOp) for n in inter):
            return graph

    # Build ReduceStages referencing the existing Node objects (no node moves).
    stages: list[ReduceStage] = []
    prev_idx = reduce_indices[0]
    stages.append(ReduceStage(pre_ops=(), reduce=kernel.prologue[prev_idx]))
    for ridx in reduce_indices[1:]:
        inter = kernel.prologue[prev_idx + 1 : ridx]
        stages.append(ReduceStage(pre_ops=tuple(inter), reduce=kernel.prologue[ridx]))
        prev_idx = ridx

    # Annotation-only restructure: leave prologue intact so backend compat
    # properties still see the flat node order. ``core`` is a classification
    # marker that references existing nodes; no nodes are moved.
    new_kernel = KernelOp(
        inputs=list(kernel.inputs),
        outputs=list(kernel.outputs),
        prologue=kernel.prologue,
        core=tuple(stages),
        epilogue=kernel.epilogue,
        kernel_source=kernel.kernel_source,
        external_shapes=dict(kernel.external_shapes),
    )

    g = graph.copy()
    new_id = g.add_node(
        op=new_kernel,
        inputs=list(node.inputs),
        output=g.nodes[match.root_node_id].output,
    )
    g.replace_node(match.root_node_id, new_id)
    g.remove_node(match.root_node_id)
    return g
