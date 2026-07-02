"""Decompose softmax(x, dim) into max → sub → exp → sum → div."""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.frontend.ir import SoftmaxOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, softmax_decompose

PATTERN = [Pattern("root", SoftmaxOp)]


def rewrite(match: Match, root: Node, inp_x: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    if inp_x is None:
        raise RuleSkipped("softmax has no input node bound")
    frag = open_fragment(graph, [inp_x])
    out_node = softmax_decompose(frag, inp_x, root.op.axis, name=out.name)
    frag.outputs = [out_node.id]
    return frag
