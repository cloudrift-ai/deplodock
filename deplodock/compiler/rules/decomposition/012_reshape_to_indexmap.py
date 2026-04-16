"""Decompose ReshapeOp into an IndexMapOp with linearize→delinearize coord_map.

ReshapeOp changes logical shape without moving data. The coord_map
linearizes the output coordinates using output strides, then
delinearizes into the input coordinate space using input strides.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinOp, Literal, placeholder
from deplodock.compiler.ir.frontend import ReshapeOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", ReshapeOp, "1")]


def _reshape_coord_map(in_shape: tuple, out_shape: tuple):
    """Build coord_map entries that linearize output coords then delinearize to input coords.

    For reshape from in_shape to out_shape (same numel), each input
    coordinate is:  c_j = (flat // in_stride_j) % in_shape_j
    where flat = sum(out_coord_i * out_stride_i).
    """
    out_ndim = len(out_shape)
    in_ndim = len(in_shape)

    if in_shape == out_shape:
        return tuple(placeholder(d) for d in range(out_ndim))

    # Build flat = out_coord_0 * stride_0 + out_coord_1 * stride_1 + ...
    flat = None
    out_stride = 1
    for d in range(out_ndim - 1, -1, -1):
        term = placeholder(d) if out_stride == 1 else BinOp("*", placeholder(d), Literal(out_stride, "int"))
        flat = term if flat is None else BinOp("+", term, flat)
        out_stride *= int(out_shape[d])

    if flat is None:
        flat = Literal(0, "int")

    # Delinearize flat into input coords: c_j = (flat // in_stride_j) % in_shape_j
    coords = []
    in_stride = 1
    for j in range(in_ndim - 1, -1, -1):
        in_stride_j = in_stride
        coord = flat if in_stride_j == 1 else BinOp("/", flat, Literal(in_stride_j, "int"))
        dim_j = int(in_shape[j])
        if j > 0:
            coord = BinOp("%", coord, Literal(dim_j, "int"))
        coords.insert(0, coord)
        in_stride *= dim_j

    return tuple(coords)


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = root.op.infer_output_shape([in_shape])

    coord_map = _reshape_coord_map(in_shape, out_shape)
    indexmap = IndexMapOp(
        out_shape=out_shape,
        sources=(IndexSource(input_idx=0, coord_map=coord_map),),
    )

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    new_id = frag.add_node(
        op=indexmap,
        inputs=[x_id],
        output=Tensor(root.output.name, out_shape, root.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
