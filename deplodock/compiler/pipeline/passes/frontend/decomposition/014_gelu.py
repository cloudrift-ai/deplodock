"""Decompose gelu(x) into 0.5 * x * (1 + erf(x / sqrt(2))).

Matches torch's default ``approximate='none'`` GELU. The lifted graph
has one elementwise leaf per stage (mul, add, erf, divide-by-constant)
so loop-fusion can collapse the whole chain into a single kernel
alongside surrounding pointwise ops — the same pattern silu uses.
"""

import math

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "gelu"})]


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    """Replace gelu(x) with 0.5 * x * (1 + erf(x * (1/sqrt(2))))."""
    frag = open_fragment(graph, [inp_x])

    inv_sqrt2 = const_bc(frag, name=f"{out.name}_inv_sqrt2", value=1.0 / math.sqrt(2.0), target_shape=out.shape, dtype=out.dtype)
    scaled_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, inv_sqrt2],
        output=Tensor(f"{out.name}_scaled", out.shape, out.dtype),
    )
    erf_id = frag.add_node(
        op=ElementwiseOp(op="erf"),
        inputs=[scaled_id],
        output=Tensor(f"{out.name}_erf", out.shape, out.dtype),
    )
    one_bc = const_bc(frag, name=f"{out.name}_one", value=1.0, target_shape=out.shape, dtype=out.dtype)
    plus_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[one_bc, erf_id],
        output=Tensor(f"{out.name}_plus1", out.shape, out.dtype),
    )
    half_bc = const_bc(frag, name=f"{out.name}_half", value=0.5, target_shape=out.shape, dtype=out.dtype)
    half_x_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, half_bc],
        output=Tensor(f"{out.name}_half_x", out.shape, out.dtype),
    )
    out_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[half_x_id, plus_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )

    frag.outputs = [out_id]
    return frag
