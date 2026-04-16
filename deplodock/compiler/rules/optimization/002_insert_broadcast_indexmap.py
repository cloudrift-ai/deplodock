"""Insert explicit IndexMapOp for broadcast reads.

When an ElementwiseOp has inputs with shapes smaller than its output
(broadcast), insert an IndexMapOp between the smaller input and the
elementwise op.
"""

from __future__ import annotations

from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, IndexSource, InputOp

GRAMMAR = [Production("root", ElementwiseOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    out_shape = tuple(root.output.shape)

    if not out_shape or not all(isinstance(d, int) for d in out_shape):
        return None

    # Check if any input needs broadcast.
    new_inputs: list[str | None] = [None] * len(root.inputs)
    needs_change = False
    broadcast_ops: list[tuple[str, IndexMapOp, Tensor]] = []

    for i, inp_id in enumerate(root.inputs):
        inp_node = graph.nodes.get(inp_id)
        if inp_node is None:
            continue
        inp_shape = tuple(inp_node.output.shape)
        if inp_shape == out_shape:
            continue
        if not inp_shape or not all(isinstance(d, int) for d in inp_shape):
            continue
        indexmap = _broadcast_indexmap(inp_shape, out_shape)
        if indexmap is None:
            continue
        bc_name = f"{inp_id}_bc"
        broadcast_ops.append((inp_id, indexmap, Tensor(bc_name, out_shape, inp_node.output.dtype)))
        new_inputs[i] = bc_name
        needs_change = True

    if not needs_change:
        return None

    # Build fragment: InputOps for all original inputs, broadcast IndexMapOps,
    # then the root op with updated inputs.
    frag = Graph()
    for inp_id in root.inputs:
        if inp_id not in frag.nodes:
            inp_node = graph.nodes[inp_id]
            frag.add_node(InputOp(), [], Tensor(inp_id, inp_node.output.shape, inp_node.output.dtype), node_id=inp_id)

    bc_ids: dict[int, str] = {}
    for inp_id, indexmap, tensor in broadcast_ops:
        bc_id = frag.add_node(indexmap, [inp_id], tensor)
        # Find which input index this broadcast replaces.
        for i, ni in enumerate(new_inputs):
            if ni == tensor.name:
                bc_ids[i] = bc_id

    final_inputs = [bc_ids.get(i, root.inputs[i]) for i in range(len(root.inputs))]
    out_id = frag.add_node(root.op, final_inputs, Tensor(root.output.name, root.output.shape, root.output.dtype))
    frag.outputs = [out_id]
    return frag


def _broadcast_indexmap(inp_shape: tuple, out_shape: tuple) -> IndexMapOp | None:
    out_ndim = len(out_shape)
    inp_ndim = len(inp_shape)
    if inp_ndim > out_ndim:
        return None
    offset = out_ndim - inp_ndim
    coord_map = []
    for d in range(inp_ndim):
        out_d = d + offset
        if inp_shape[d] == out_shape[out_d]:
            coord_map.append(placeholder(out_d))
        elif inp_shape[d] == 1:
            from deplodock.compiler.backend.ir.expr import Literal

            coord_map.append(Literal(0, "int"))
        else:
            return None
    return IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),))
