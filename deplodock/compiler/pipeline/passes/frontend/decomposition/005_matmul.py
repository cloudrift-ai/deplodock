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


def rewrite(graph: Graph, inp_a: Node, inp_b: Node, inp_bias: Node | None, out: Tensor) -> Graph | None:
    exts = [inp_a, inp_b] + ([inp_bias] if inp_bias else [])
    frag = open_fragment(graph, exts)

    matmul_name = f"{out.name}_mm" if inp_bias else out.name
    mm_id = matmul_decompose(frag, inp_a.id, inp_b.id, name=matmul_name, dtype=out.dtype)

    if inp_bias:
        bias_bc = broadcast_to(frag, inp_bias.id, out.shape)
        add_id = frag.add_node(op=ElementwiseOp(op="add"), inputs=[mm_id, bias_bc], output=Tensor(out.name, out.shape, out.dtype))
        frag.outputs = [add_id]
    else:
        frag.outputs = [mm_id]

    return frag
