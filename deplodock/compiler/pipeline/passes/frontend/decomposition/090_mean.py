"""Decompose MeanOp into sum + div by the reduced dimension size."""

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import MeanOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", MeanOp)]


def rewrite(match: Match, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    """Replace mean(x, axis) with sum(x, axis) / axis_size."""
    axis = root.op.axis
    x_shape = inp_x.output.shape
    dim_size = x_shape[axis % len(x_shape)] if isinstance(axis, int) and x_shape else None
    # The divisor is the reduced dimension's size. A STATIC dim bakes a literal count; a
    # SYMBOLIC dim (dynamic ``seq_len``) carries the ``Dim`` so the backend divides by the
    # RUNTIME extent (``float(seq_len)``) — without this a dynamic mean divides by the wrong
    # (trace-time) count.
    symbolic = isinstance(dim_size, Dim) and not dim_size.is_static
    count_value = float(dim_size.as_static()) if isinstance(dim_size, Dim) and dim_size.is_static else 1.0

    frag = open_fragment(graph, [inp_x])
    sum_id = frag.add_node(
        op=ReduceOp(op="sum", axis=axis),
        inputs=[inp_x],
        output=Tensor(f"{out.name}_sum", out.shape, out.dtype),
    )
    count_bc = const_bc(
        frag,
        name=f"{out.name}_count",
        value=None if symbolic else count_value,
        context_value=dim_size.expr if symbolic else None,
        target_shape=out.shape,
        dtype=out.dtype,
    )
    div_id = frag.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[sum_id, count_bc],
        output=Tensor(out.name, out.shape, out.dtype),
    )
    frag.outputs = [div_id]
    return frag
