"""Insert explicit IndexMapOp for broadcast reads.

When an ElementwiseOp has inputs with shapes smaller than its output
(broadcast), insert an IndexMapOp between the smaller input and the
elementwise op. The coord_map maps output coords to input coords by
selecting only the axes that exist in the input shape.

Runs in the optimization pass (after decomposition, before fusion) so
the inserted IndexMapOps get absorbed into Port.indexmap by the fusion
rule's backward walk — no extra kernels emitted.
"""

from __future__ import annotations

from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, IndexSource

GRAMMAR = [Production("root", ElementwiseOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    root = graph.nodes[match.root_node_id]
    out_shape = tuple(root.output.shape)

    if not out_shape or not all(isinstance(d, int) for d in out_shape):
        return graph

    changed = False
    g = graph
    for i, inp_id in enumerate(root.inputs):
        inp_node = g.nodes.get(inp_id)
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

        if not changed:
            g = graph.copy()
            root = g.nodes[match.root_node_id]
            changed = True

        new_id = g.add_node(
            op=indexmap,
            inputs=[inp_id],
            output=Tensor(f"{inp_id}_bc", out_shape, inp_node.output.dtype),
        )
        root.inputs[i] = new_id

    return g


def _broadcast_indexmap(inp_shape: tuple, out_shape: tuple) -> IndexMapOp | None:
    """Build an IndexMapOp that broadcasts ``inp_shape`` to ``out_shape``.

    Uses NumPy-style right-aligned broadcasting: the input shape is
    aligned to the right of the output shape, and each axis is either
    size-matched or size-1 (broadcast).
    """
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

    return IndexMapOp(
        out_shape=out_shape,
        sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),),
    )
