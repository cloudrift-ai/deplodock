"""Replace Transpose with Reshape when one of the swapped dims has size 1.

A transpose that swaps a size-1 dim with another dim is physically a no-op
(memory layout unchanged), so it can become a Reshape — which the backend
turns into a free buffer alias instead of a copy kernel.
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ReshapeOp

PATTERN = "Transpose($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    t_id = match.root_node_id
    x_id = match.bindings["x"]

    t_node = graph.nodes[t_id]
    axes = t_node.op.axes
    x_shape = graph.nodes[x_id].output.shape
    ndim = len(x_shape)

    # Only handle the 2-axis swap form; full permutes need different handling.
    if len(axes) != 2:
        return graph

    a = axes[0] if axes[0] >= 0 else ndim + axes[0]
    b = axes[1] if axes[1] >= 0 else ndim + axes[1]
    if not (0 <= a < ndim and 0 <= b < ndim):
        return graph

    da, db = x_shape[a], x_shape[b]
    if not (isinstance(da, int) and da == 1) and not (isinstance(db, int) and db == 1):
        return graph

    g = graph.copy()
    out_shape = t_node.output.shape
    reshape_id = g.add_node(
        op=ReshapeOp(shape=tuple(out_shape)),
        inputs=[x_id],
        output=Tensor(name=t_node.output.name, shape=tuple(out_shape), dtype=t_node.output.dtype),
    )
    g.replace_node(t_id, reshape_id)
    g.remove_node(t_id)
    return g
