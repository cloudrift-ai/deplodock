"""Decompose pow(x, 2) into mul(x, x) to enable RMSNorm fusion."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "pow"})]


def rewrite(match: Match, inp_x: Node, inp_exp: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    """Replace pow(x, 2) with mul(x, x) — enables RMSNorm pattern matching."""
    if inp_exp and isinstance(inp_exp.op, ConstantOp) and inp_exp.op.value != 2.0:
        raise RuleSkipped(f"only pow(x, 2) is decomposed; got exponent {inp_exp.op.value}")

    frag = open_fragment(graph, [inp_x])
    mul_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, inp_x],
        output=Tensor(name=out.name, shape=out.shape, dtype=out.dtype),
    )
    frag.outputs = [mul_id]
    return frag
