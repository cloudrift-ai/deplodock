"""Decompose LinearOp into transpose(weight) → matmul [→ add(bias)]."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import LinearOp, TransposeOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    matmul_decompose,
    open_fragment,
)

PATTERN = [Pattern("root", LinearOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    x_id = root.inputs[0]
    w_id = root.inputs[1]
    b_id = root.inputs[2] if len(root.inputs) > 2 else None
    out = root.output

    ext_ids = {x_id, w_id} | ({b_id} if b_id else set())
    frag = open_fragment(graph, ext_ids)

    w_shape = graph.nodes[w_id].output.shape
    wt_shape = (w_shape[-1], w_shape[-2]) if len(w_shape) >= 2 else w_shape
    wt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[w_id],
        output=Tensor(f"{out.name}_wt", wt_shape, out.dtype),
    )

    matmul_name = f"{out.name}_mm" if b_id else out.name
    mm_id = matmul_decompose(frag, x_id, wt_id, name=matmul_name, dtype=out.dtype)

    if b_id:
        bias_bc = broadcast_to(frag, b_id, out.shape)
        add_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[mm_id, bias_bc],
            output=Tensor(out.name, out.shape, out.dtype),
        )
        frag.outputs = [add_id]
    else:
        frag.outputs = [mm_id]

    return frag
