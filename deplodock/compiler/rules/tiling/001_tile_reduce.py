"""Split a reduce axis into outer and inner reduces for tiling."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ReduceOp

PATTERN = "Reduce{$f, $ax}($body)"

TILE_SIZE = 32


def rewrite(graph: Graph, match: Match) -> Graph:
    """Split Reduce{f, ax} into Reduce{f, ax_outer}(Reduce{f, ax_inner}(...))."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    reduce_fn = match.captured_constraints["f"]
    axis = match.captured_constraints["ax"]

    # Only tile if the axis is concrete and large enough.
    if isinstance(axis, str):
        # Symbolic axis — can't determine size, skip.
        return graph
    axis_size = _resolve_axis_size(root, axis)
    if axis_size is not None and axis_size <= TILE_SIZE:
        return graph

    ax_str = str(axis)
    inner_axis = f"{ax_str}_inner"
    outer_axis = f"{ax_str}_outer"

    # Inner reduce: same inputs as original, reduces over inner tile.
    inner_id = g.add_node(
        op=ReduceOp(fn=reduce_fn, axis=inner_axis),
        inputs=list(root.inputs),
        output=Tensor(
            name=f"{root.output.name}_inner",
            shape=root.output.shape,  # simplified — real impl would adjust
            dtype=root.output.dtype,
        ),
    )

    # Outer reduce: consumes inner reduce output.
    outer_id = g.add_node(
        op=ReduceOp(fn=reduce_fn, axis=outer_axis),
        inputs=[inner_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, outer_id)
    g.remove_node(match.root_node_id)
    return g


def _resolve_axis_size(root, axis):
    """Try to get the concrete size of the reduce axis from the input tensor."""
    if not root.inputs:
        return None
    # The axis size would come from the input tensor shape.
    # For now, return None (symbolic) — a full implementation would
    # look up the input tensor's shape at the given axis.
    return None
