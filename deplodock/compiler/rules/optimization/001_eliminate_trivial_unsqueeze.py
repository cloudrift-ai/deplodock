"""Eliminate Reshape that just adds broadcast-redundant leading 1-dims.

Qwen2's ``apply_rotary_pos_emb`` does ``cos.unsqueeze(1)`` on cos/sin that
already have a leading 1-dim. After ``006_decompose_unsqueeze`` lowers
``Unsqueeze`` to ``Reshape``, the result is a Reshape that takes a shape
``(..., 1, ..., orig_dims)`` whose only difference from the input shape is
extra leading 1-dims.

Under NumPy broadcasting, leading 1-dims are added implicitly when the
other operand has higher rank, so the reshape is a no-op for any
elementwise consumer. Removing it lets the entire downstream chain drop a
spurious rank — which for Qwen collapses an entire 5D rotary + SDPA path
back to 4D.

Eligibility: ``Reshape(x, shape=s)`` is removable when ``s`` is exactly
``x.shape`` left-padded with one or more concrete 1's
(``s == (1,) * k + x.shape`` for some ``k > 0``). When ``k == 0`` the
reshape is identity and a separate identity-reshape pass would catch it;
this rule narrows to the leading-1-padding case.

When ineligible, returns the same ``graph`` object so the rewriter
treats it as a no-op (preventing the fixed-point loop from re-firing).
"""

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.shape_utils import propagate_shapes

PATTERN = "Reshape($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    rs_id = match.root_node_id
    x_id = match.bindings["x"]

    rs_node = graph.nodes[rs_id]
    out_shape = tuple(rs_node.op.shape)
    in_shape = tuple(graph.nodes[x_id].output.shape)

    if len(out_shape) <= len(in_shape):
        return graph
    pad = len(out_shape) - len(in_shape)
    if not all(isinstance(d, int) and d == 1 for d in out_shape[:pad]):
        return graph
    if out_shape[pad:] != in_shape:
        return graph

    g = graph.copy()
    if rs_id in g.outputs:
        g.outputs = [x_id if o == rs_id else o for o in g.outputs]
    g.replace_node(rs_id, x_id)
    g.remove_node(rs_id)
    propagate_shapes(g, [x_id])
    return g
