"""Absorb a producer pointwise KernelOp into a ContractionCore's a_chain.

Pattern: ``Kernel_contraction($producer_kernel, $B)`` where
``$producer_kernel`` is a KernelOp with ``core=None`` (pointwise) and
fan-out=1 feeding the contraction's A operand. The producer's prologue
moves into the downstream ContractionCore's ``a_chain`` so A is computed
per (row, k) inside the K-loop.

Enables ``mul(silu(gate), up) @ Wdown`` fusion.

be added before this rule can land.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ContractionCore, KernelOp
from deplodock.compiler.rules.fusion._assembly_helpers import fan_out_of

PATTERN = "_"


def rewrite(graph: Graph, match: Match) -> Graph:
    kid = match.root_node_id
    knode = graph.nodes.get(kid)
    if knode is None or not isinstance(knode.op, KernelOp):
        return graph
    kernel = knode.op
    if not isinstance(kernel.core, ContractionCore):
        return graph
    a_id = kernel.core.a.buffer_id
    producer_node = graph.nodes.get(a_id)
    if producer_node is None or not isinstance(producer_node.op, KernelOp):
        return graph
    producer = producer_node.op
    # Must be pure pointwise: core=None, epilogue empty.
    if producer.core is not None or producer.epilogue:
        return graph
    if fan_out_of(graph, a_id) != 1:
        return graph

    # ContractionCore doesn't have an a_chain field today; this rule is a
    # placeholder that will activate once the IR field is added in Commit 1b.
    # For now return graph unchanged — rule is a no-op.
    return graph
