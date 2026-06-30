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

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import Literal, placeholder
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource


def _is_symbolic(d) -> bool:
    return isinstance(d, Dim) and not d.is_static


def broadcast_to(graph: Graph, node: Node | str, target_shape: tuple) -> Node:
    """Return a node whose output has shape ``target_shape``.

    If ``node``'s output already has ``target_shape``, returns ``node``
    unchanged. Otherwise adds an IndexMapOp to ``graph`` that broadcasts the
    input to ``target_shape`` and returns the new node. Follows numpy's
    right-aligned broadcast rules: dims of size 1 broadcast via ``Literal(0)``,
    matching dims pass through via ``placeholder(out_d)``, and the input's rank
    must be ≤ the target rank.

    Raises ``ValueError`` when the broadcast is illegal (non-size-1 dim
    mismatch, or input rank exceeds target rank).
    """
    if not isinstance(node, Node):
        node = graph.nodes[node]
    inp_shape = tuple(node.output.shape)
    target_shape = tuple(target_shape)
    if inp_shape == target_shape:
        return node
    indexmap = _broadcast_indexmap(inp_shape, target_shape)
    if indexmap is None:
        # Symbolic dims that propagated to where another concrete dim was
        # expected are typically the symptom of a ``--dynamic`` value
        # colliding with another model dim (e.g. ``--seq-len 32`` on a
        # model where ``num_heads == 32`` rewrites both). Surface that hint
        # so the user doesn't dig through the decomposition pass.
        any_symbolic = any(_is_symbolic(d) for s in (inp_shape, target_shape) for d in s)
        hint = (
            " (note: shapes contain symbolic dims — if this came from "
            "``--dynamic``, the canonical value may collide with another "
            "model dim; try a non-colliding prime like 31 / 37 / 41 for "
            "``--seq-len``)"
            if any_symbolic
            else ""
        )
        raise ValueError(f"cannot broadcast shape {inp_shape} to {target_shape}: non-size-1 dim mismatch or rank exceeds target{hint}")
    # Use the tensor's semantic name (not the node id) for the broadcast output
    # name — fragment ids get rewritten on splicing, but tensor names carry
    # through, so anchoring the "_bc" suffix on the name avoids collisions when
    # two fragments internally use the same auto-generated id.
    nid = graph.add_node(
        op=indexmap,
        inputs=[node],
        output=Tensor(f"{node.output.name}_bc", target_shape, node.output.dtype),
    )
    return graph.nodes[nid]


def squeeze_axis(graph: Graph, node: Node | str, axis: int, out_name: str | None = None) -> Node:
    """Drop a single size-1 axis from ``node``'s output via an IndexMapOp.

    Used by decomposition rules (matmul, linear, etc.) that keep a reduction's
    reduced axis at size 1 for the ``ReduceOp`` itself, then squeeze it away
    for downstream ops that expect the old dropped-axis shape. Keeps the
    rank-preservation invariant local to the reduce while giving consumers the
    shape they declared.

    ``axis`` is the position of the size-1 dim to drop, given in terms of the
    input shape. Negative axes count from the end. ``out_name`` names the new
    tensor; defaults to ``"{input_name}_sq"``.
    """
    if not isinstance(node, Node):
        node = graph.nodes[node]
    inp_shape = tuple(node.output.shape)
    a = axis if axis >= 0 else len(inp_shape) + axis
    if a < 0 or a >= len(inp_shape):
        raise ValueError(f"squeeze_axis: axis {axis} out of range for shape {inp_shape}")
    if inp_shape[a] != 1:
        raise ValueError(f"squeeze_axis: cannot drop non-size-1 dim {a} of shape {inp_shape}")
    out_shape = tuple(inp_shape[:a]) + tuple(inp_shape[a + 1 :])
    coord_map = []
    out_d = 0
    for in_d in range(len(inp_shape)):
        if in_d == a:
            coord_map.append(Literal(0, "int"))
        else:
            coord_map.append(placeholder(out_d))
            out_d += 1
    nid = graph.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[node],
        output=Tensor(out_name or f"{node.output.name}_sq", out_shape, node.output.dtype),
    )
    return graph.nodes[nid]


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
