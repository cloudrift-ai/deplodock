"""Lower TransposeOp(x, axes) → IndexMapOp.

Per the tracer convention (``torch_trace.py``), ``TransposeOp.axes`` is
always a length-2 tuple describing two dims to swap — matching PyTorch's
``aten.transpose(dim0, dim1)``. coord_map is identity except positions a
and b read from each other.
"""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend import TransposeOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", TransposeOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = tuple(root.output.shape)
    ndim = len(in_shape)
    axes = root.op.axes

    if len(axes) != 2:
        return None  # only the 2-axis swap form is produced by the tracer
    a = axes[0] if axes[0] >= 0 else ndim + axes[0]
    b = axes[1] if axes[1] >= 0 else ndim + axes[1]
    if not (0 <= a < ndim and 0 <= b < ndim):
        return None
    coord_map = []
    for i in range(ndim):
        if i == a:
            coord_map.append(placeholder(b))
        elif i == b:
            coord_map.append(placeholder(a))
        else:
            coord_map.append(placeholder(i))
    coord_map = tuple(coord_map)

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
        inputs=[x_id],
        output=Tensor(root.output.name, out_shape, root.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
