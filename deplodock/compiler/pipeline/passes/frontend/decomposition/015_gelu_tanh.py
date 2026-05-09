"""Decompose gelu_tanh(x) into the tanh approximation:

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

The tracer renames ``gelu`` to ``gelu_tanh`` when ``approximate='tanh'``
appears in the FX kwargs (``trace/torch.py``). This rule expands that
distinct op so eager-tanh and our decomposition match bit-for-bit on
the inner cubic + tanh chain.
"""

import math

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "gelu_tanh"})]

_C0 = math.sqrt(2.0 / math.pi)
_C1 = 0.044715


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    frag = open_fragment(graph, [inp_x])
    name = out.name

    x_sq = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, inp_x],
        output=Tensor(f"{name}_xsq", out.shape, out.dtype),
    )
    x_cu = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x_sq, inp_x],
        output=Tensor(f"{name}_xcu", out.shape, out.dtype),
    )
    c1_bc = const_bc(frag, name=f"{name}_c1", value=_C1, target_shape=out.shape, dtype=out.dtype)
    cubic = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x_cu, c1_bc],
        output=Tensor(f"{name}_cubic", out.shape, out.dtype),
    )
    inner_sum = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[inp_x, cubic],
        output=Tensor(f"{name}_inner", out.shape, out.dtype),
    )
    c0_bc = const_bc(frag, name=f"{name}_c0", value=_C0, target_shape=out.shape, dtype=out.dtype)
    scaled = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inner_sum, c0_bc],
        output=Tensor(f"{name}_scaled", out.shape, out.dtype),
    )
    tanh_id = frag.add_node(
        op=ElementwiseOp(op="tanh"),
        inputs=[scaled],
        output=Tensor(f"{name}_tanh", out.shape, out.dtype),
    )
    one_bc = const_bc(frag, name=f"{name}_one", value=1.0, target_shape=out.shape, dtype=out.dtype)
    plus_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[one_bc, tanh_id],
        output=Tensor(f"{name}_plus1", out.shape, out.dtype),
    )
    half_bc = const_bc(frag, name=f"{name}_half", value=0.5, target_shape=out.shape, dtype=out.dtype)
    half_x = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, half_bc],
        output=Tensor(f"{name}_half_x", out.shape, out.dtype),
    )
    out_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[half_x, plus_id],
        output=Tensor(name, out.shape, out.dtype),
    )

    frag.outputs = [out_id]
    return frag
