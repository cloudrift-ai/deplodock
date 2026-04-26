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
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    ext_ids = {a_id, b_id} | ({bias_id} if bias_id else set())
    frag = open_fragment(graph, ext_ids)

    matmul_name = f"{name}_mm" if bias_id else name
    mm_id = matmul_decompose(frag, a_id, b_id, name=matmul_name, dtype=dtype)

    if bias_id:
        bias_bc = broadcast_to(frag, bias_id, shape)
        add_id = frag.add_node(op=ElementwiseOp(op="add"), inputs=[mm_id, bias_bc], output=Tensor(name, shape, dtype))
        frag.outputs = [add_id]
    else:
        frag.outputs = [mm_id]

    return frag
