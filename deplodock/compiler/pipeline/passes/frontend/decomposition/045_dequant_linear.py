"""Decompose ``DequantLinearOp`` into unpack → dequant → transpose → matmul [→ bias].

The W4A16 frontend op expands here into the int4 weight-unpack + dequant cone
plus the matmul the rest of the pipeline lowers. Inputs, in order: ``x``,
``weight_packed`` (i32), ``weight_scale`` (f16), ``weight_zero_point`` (i32),
and — when ``has_bias`` — ``bias``. The format numbers ride as ``root.op.scheme``
metadata (a ``QuantScheme``), never as graph constants.

Mirrors ``040_linear``: ``dequant_decompose`` builds the weight cone + matmul,
this rule wires the optional bias add exactly as the linear rule does.
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import DequantLinearOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    dequant_decompose,
    open_fragment,
)

PATTERN = [Pattern("root", DequantLinearOp)]


def rewrite(
    match: Match, root: Node, inp_x: Node, inp_w: Node, inp_scale: Node, inp_zp: Node, inp_b: Node | None, out: Tensor
) -> Graph | None:
    graph = match.graph
    has_bias = root.op.has_bias
    scheme = root.op.scheme

    exts = [inp_x, inp_w, inp_scale, inp_zp] + ([inp_b] if has_bias else [])
    frag = open_fragment(graph, exts)

    matmul_name = f"{out.name}_mm" if has_bias else out.name
    mm = dequant_decompose(frag, inp_x, inp_w, inp_scale, inp_zp, scheme=scheme, matmul_name=matmul_name, out_dtype=out.dtype)

    if has_bias:
        bias_bc = broadcast_to(frag, inp_b, out.shape)
        add_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[mm, bias_bc],
            output=Tensor(out.name, out.shape, out.dtype),
        )
        frag.outputs = [add_id]
    else:
        frag.outputs = [mm.id]

    return frag
