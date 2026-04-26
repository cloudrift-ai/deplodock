"""Decompose softmax(x, dim) into max → sub → exp → sum → div."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import SoftmaxOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, softmax_decompose

PATTERN = [Pattern("root", SoftmaxOp)]


def rewrite(graph: Graph, root: Node, inp_x: Node | None, out: Tensor) -> Graph | None:
    if inp_x is None:
        return None
    frag = open_fragment(graph, [inp_x])
    out_node = softmax_decompose(frag, inp_x, root.op.axis, name=out.name)
    frag.outputs = [out_node.id]
    return frag
