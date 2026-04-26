"""Decompose softmax(x, dim) into max → sub → exp → sum → div."""

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.frontend.ir import SoftmaxOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, softmax_decompose

PATTERN = [Pattern("root", SoftmaxOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    if not root.inputs:
        return None
    x_id = root.inputs[0]
    frag = open_fragment(graph, [x_id])
    out_id = softmax_decompose(frag, x_id, root.op.axis, name=root.output.name, dtype=root.output.dtype)
    frag.outputs = [out_id]
    return frag
