"""Absorb an upstream fusible op (Elementwise / IndexMap) into a KernelOp.

Pattern: ``Kernel(..., $x, ...)`` where ``$x`` is an ``ElementwiseOp`` or
``IndexMapOp`` with fan-out=1 (only consumer is the KernelOp). The
producer moves into ``kernel.prologue`` (prepended) and the kernel's
input Port is rewired to the producer's inputs.

Doesn't handle ContractionCore's a_chain/b_chain yet — that's a later
rule (080_absorb_a_chain).
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ContractionCore, ElementwiseOp, IndexMapOp, KernelOp, Port
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    fan_out_of,
    rewrite_port_references,
    shape_of,
)

PATTERN = "_"  # wildcard; we need variable-arity input matching


def rewrite(graph: Graph, match: Match) -> Graph:
    kid = match.root_node_id
    knode = graph.nodes.get(kid)
    if knode is None or not isinstance(knode.op, KernelOp):
        return graph
    kernel = knode.op

    # Don't grow contractions through this rule — a_chain is a different fusion.
    if isinstance(kernel.core, ContractionCore):
        return graph

    # Find any input that is a fusible producer with fan-out=1.
    for i, port in enumerate(kernel.inputs):
        producer_id = port.buffer_id
        prod_node = graph.nodes.get(producer_id)
        if prod_node is None or not isinstance(prod_node.op, (ElementwiseOp, IndexMapOp)):
            continue
        if fan_out_of(graph, producer_id) != 1:
            continue

        # Prepend the producer to prologue. Rewire Port + kernel.inputs.
        snap = copy_node(prod_node)
        new_prologue = (snap,) + kernel.prologue
        # New kernel inputs: replace port at i with producer's inputs (in order).
        new_inputs_list: list[Port] = []
        for j, p in enumerate(kernel.inputs):
            if j == i:
                for pinp in prod_node.inputs:
                    if not any(q.buffer_id == pinp for q in new_inputs_list):
                        new_inputs_list.append(Port(buffer_id=pinp))
            else:
                if not any(q.buffer_id == p.buffer_id for q in new_inputs_list):
                    new_inputs_list.append(p)

        new_external = dict(kernel.external_shapes)
        new_external.pop(producer_id, None)
        for pinp in prod_node.inputs:
            new_external[pinp] = shape_of(graph, pinp)

        new_kernel = KernelOp(
            inputs=new_inputs_list,
            outputs=list(kernel.outputs),
            prologue=new_prologue,
            core=kernel.core,
            epilogue=kernel.epilogue,
            kernel_source=kernel.kernel_source,
            external_shapes=new_external,
        )

        g = graph.copy()
        new_kid = g.add_node(
            op=new_kernel,
            inputs=[p.buffer_id for p in new_inputs_list],
            output=g.nodes[kid].output,
        )
        g.nodes[new_kid].hints.merge(graph.nodes[kid].hints)
        g.nodes[new_kid].hints.merge(prod_node.hints)
        g.replace_node(kid, new_kid)
        rewrite_port_references(g, kid, new_kid)
        if kid in g.nodes:
            g.remove_node(kid)
        if producer_id in g.nodes and not g.consumers(producer_id):
            g.remove_node(producer_id)
        return g
    return graph
