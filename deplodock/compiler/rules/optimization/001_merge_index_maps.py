"""Compose adjacent IndexMaps into one by coord-map substitution.

The broadcast-redundant unsqueeze elimination happens in
``decomposition/008_eliminate_broadcast_redundant_unsqueeze.py`` so it
can run BEFORE other view-op decompositions create rank-fixed IndexMaps.
This rule then collapses chains of IndexMaps left in the graph (e.g.
slice → transpose, or transpose composed with cat).
"""

from deplodock.compiler.coord_expr import compose_index_maps
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import IndexMapOp
from deplodock.compiler.shape_utils import propagate_shapes

GRAMMAR = [Production("root", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    outer_id = match.root_node_id
    root = graph.nodes[outer_id]
    inner_id = root.inputs[0]

    outer_node = graph.nodes[outer_id]
    inner_node = graph.nodes.get(inner_id)
    outer_op = outer_node.op
    inner_op = inner_node.op if inner_node else None

    if not isinstance(inner_op, IndexMapOp):
        return graph
    if len(outer_op.sources) != 1 or len(inner_op.sources) != 1:
        # Multi-source × multi-source not supported (would fan out the cross product).
        return graph

    g = graph.copy()
    merged = compose_index_maps(outer_op, inner_op)
    # The merged op reads from the inner's input directly.
    inner_input_id = inner_node.inputs[inner_op.sources[0].input_idx]
    new_id = g.add_node(
        op=merged,
        inputs=[inner_input_id],
        output=Tensor(outer_node.output.name, tuple(outer_op.out_shape), outer_node.output.dtype),
    )
    g.replace_node(outer_id, new_id)
    g.remove_node(outer_id)
    # Drop the inner if no other consumers remain.
    if inner_id in g.nodes and not g.consumers(inner_id):
        g.remove_node(inner_id)
    propagate_shapes(g, [new_id])
    return g
