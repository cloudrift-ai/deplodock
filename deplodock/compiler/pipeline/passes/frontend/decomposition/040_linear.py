"""Decompose LinearOp into transpose(weight) → matmul [→ add(bias)]."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import LinearOp, TransposeOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    matmul_decompose,
    open_fragment,
)

PATTERN = [Pattern("root", LinearOp)]


def rewrite(match: Match, inp_x: Node, inp_w: Node, inp_b: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    exts = [inp_x, inp_w] + ([inp_b] if inp_b else [])
    frag = open_fragment(graph, exts)

    w_shape = inp_w.output.shape
    wt_shape = (w_shape[-1], w_shape[-2]) if len(w_shape) >= 2 else w_shape
    wt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[inp_w],
        output=Tensor(f"{out.name}_wt", wt_shape, out.dtype),
    )

    matmul_name = f"{out.name}_mm" if inp_b else out.name
    mm = matmul_decompose(frag, inp_x, wt_id, name=matmul_name)

    if inp_b:
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
