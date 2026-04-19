"""Explicit-broadcast helper for decomposition rules.

When a decomposition rule needs to feed an input of smaller shape into an
ElementwiseOp of larger shape, the input must be wrapped in an IndexMapOp that
maps each output coord to the input coord — size-1 dims broadcast via
``Literal(0)``, size-matching dims pass through. This helper builds that
wrapper.

Centralizing this here lets every decomposition rule produce already-broadcast-
explicit IR, so ElementwiseOp can enforce the invariant that its inputs all
have shape equal to its output.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource


def broadcast_to(graph: Graph, node_id: str, target_shape: tuple) -> str:
    """Return a node id whose output has shape ``target_shape``.

    If ``node_id``'s output already has ``target_shape``, returns ``node_id``
    unchanged. Otherwise adds an IndexMapOp to ``graph`` that broadcasts the
    input to ``target_shape`` and returns the new node's id. Follows numpy's
    right-aligned broadcast rules: dims of size 1 broadcast via ``Literal(0)``,
    matching dims pass through via ``placeholder(out_d)``, and the input's rank
    must be ≤ the target rank.

    Raises ``ValueError`` when the broadcast is illegal (non-size-1 dim
    mismatch, or input rank exceeds target rank).
    """
    node = graph.nodes[node_id]
    inp_shape = tuple(node.output.shape)
    target_shape = tuple(target_shape)
    if inp_shape == target_shape:
        return node_id
    indexmap = _broadcast_indexmap(inp_shape, target_shape)
    if indexmap is None:
        raise ValueError(f"cannot broadcast shape {inp_shape} to {target_shape}: non-size-1 dim mismatch or rank exceeds target")
    # Use the tensor's semantic name (not the node id) for the broadcast output
    # name — fragment ids get rewritten on splicing, but tensor names carry
    # through, so anchoring the "_bc" suffix on the name avoids collisions when
    # two fragments internally use the same auto-generated id.
    return graph.add_node(
        op=indexmap,
        inputs=[node_id],
        output=Tensor(f"{node.output.name}_bc", target_shape, node.output.dtype),
    )


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
            coord_map.append(Literal(0, "int"))
        else:
            return None
    return IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),))
