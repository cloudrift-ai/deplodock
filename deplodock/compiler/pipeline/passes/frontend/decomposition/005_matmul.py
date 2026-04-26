"""Decompose MatmulOp into unsqueeze(A) * unsqueeze(B) → reduce_sum [→ add(bias)]."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    matmul_decompose,
    open_fragment,
)

PATTERN = [Pattern("root", MatmulOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    a_id = root.inputs[0]
    b_id = root.inputs[1]
    bias_id = root.inputs[2] if len(root.inputs) > 2 else None
    out = root.output

    ext_ids = {a_id, b_id} | ({bias_id} if bias_id else set())
    frag = open_fragment(graph, ext_ids)

    matmul_name = f"{out.name}_mm" if bias_id else out.name
    mm_id = matmul_decompose(frag, a_id, b_id, name=matmul_name, dtype=out.dtype)

    if bias_id:
        bias_bc = broadcast_to(frag, bias_id, out.shape)
        add_id = frag.add_node(op=ElementwiseOp(op="add"), inputs=[mm_id, bias_bc], output=Tensor(out.name, out.shape, out.dtype))
        frag.outputs = [add_id]
    else:
        frag.outputs = [mm_id]

    return frag
