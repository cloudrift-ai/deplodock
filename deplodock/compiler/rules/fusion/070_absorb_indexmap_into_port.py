"""Absorb a pure-view IndexMap into a KernelOp's input Port.

Pattern: ``Kernel(..., IndexMap($x), ...)`` where the IndexMap has
fan-out=1 and a single source. The IndexMap's coord_map moves onto the
corresponding Port's ``indexmap`` field; the IndexMap node is deleted.

This enables transpose-into-matmul: an upstream TransposeOp decomposes
to an IndexMapOp, this rule moves it onto Port.indexmap, and codegen
substitutes coords during A/B loads.

"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import IndexMapOp, KernelOp, Port
from deplodock.compiler.rules.fusion._assembly_helpers import fan_out_of, shape_of

PATTERN = "_"  # wildcard; filter to KernelOp in rewrite


def rewrite(graph: Graph, match: Match) -> Graph:
    kid = match.root_node_id
    knode = graph.nodes.get(kid)
    if knode is None or not isinstance(knode.op, KernelOp):
        return graph
    kernel = knode.op

    # Find any input Port whose producer is a single-source IndexMap with fan-out=1.
    for i, port in enumerate(kernel.inputs):
        if port.indexmap is not None:
            continue  # already has indexmap; don't layer
        producer_id = port.buffer_id
        prod_node = graph.nodes.get(producer_id)
        if prod_node is None or not isinstance(prod_node.op, IndexMapOp):
            continue
        if len(prod_node.op.sources) != 1:
            continue
        if fan_out_of(graph, producer_id) != 1:
            continue

        src = prod_node.op.sources[0]
        if src.select is not None:
            continue  # only pure-view IndexMaps

        inner_id = prod_node.inputs[src.input_idx]
        # The inner_id must still exist in the outer graph; otherwise this
        # IndexMap references a stale node and the rule should skip.
        if inner_id not in graph.nodes:
            continue

        # New Port: same buffer but carries IndexMap, pointing at the IndexMap's input.
        new_port = Port(buffer_id=inner_id, indexmap=prod_node.op)
        new_inputs_list = list(kernel.inputs)
        new_inputs_list[i] = new_port

        # Also update the core's a/b Port if this input feeds a contraction.
        # Preserve post_stages when rebuilding so combined cores stay intact.
        new_core = kernel.core
        if hasattr(new_core, "a") and new_core.a.buffer_id == producer_id:
            from deplodock.compiler.ops import ContractionCore

            new_core = ContractionCore(
                a=Port(buffer_id=inner_id, indexmap=prod_node.op),
                b=new_core.b,
                k_axis=new_core.k_axis,
                mul=new_core.mul,
                reduce=new_core.reduce,
                post_stages=new_core.post_stages,
            )
        elif hasattr(new_core, "b") and new_core.b.buffer_id == producer_id:
            from deplodock.compiler.ops import ContractionCore

            new_core = ContractionCore(
                a=new_core.a,
                b=Port(buffer_id=inner_id, indexmap=prod_node.op),
                k_axis=new_core.k_axis,
                mul=new_core.mul,
                reduce=new_core.reduce,
                post_stages=new_core.post_stages,
            )

        new_external = dict(kernel.external_shapes)
        new_external.pop(producer_id, None)
        new_external[inner_id] = shape_of(graph, inner_id)

        # Guard: every buffer_id in new_inputs_list must exist in the graph.
        if any(p.buffer_id not in graph.nodes for p in new_inputs_list):
            continue

        new_kernel = KernelOp(
            inputs=new_inputs_list,
            outputs=list(kernel.outputs),
            prologue=kernel.prologue,
            core=new_core,
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
        g.replace_node(kid, new_kid)
        if kid in g.nodes:
            g.remove_node(kid)
        if producer_id in g.nodes and not g.consumers(producer_id):
            g.remove_node(producer_id)
        return g

    return graph
