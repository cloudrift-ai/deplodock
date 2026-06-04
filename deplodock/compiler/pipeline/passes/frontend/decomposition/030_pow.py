"""Decompose constant-exponent ``pow`` into the matching elementwise op.

``pow(x, 2)`` → ``mul(x, x)`` (enables RMSNorm mean-of-squares fusion),
``pow(x, -0.5)`` → ``rsqrt(x)`` and ``pow(x, 0.5)`` → ``sqrt(x)`` (the
normalization factor — Gemma's RMSNorm uses ``torch.pow(ms, -0.5)`` instead
of ``torch.rsqrt`` for Torch/JAX parity). Other / unknown exponents are left
as ``pow`` (lowered via ``powf``).

The exponent reaches the ``pow`` op as a *broadcast* of a ``ConstantOp``
(``pow_c1_bc = pow_c1[k]``), not the constant directly, so we resolve it by
following the single-input broadcast/reshape chain back to the constant. The
previous ``isinstance(inp_exp.op, ConstantOp)`` check never matched the
broadcast, so it fell through and squared *every* pow — silently miscompiling
``pow(x, -0.5)`` (rsqrt) into ``x * x``.
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "pow"})]

# Constant exponent → the single-arg elementwise op it's equivalent to.
_EXPONENT_OPS: dict[float, str | None] = {
    2.0: "multiply",  # x * x (special-cased below — needs x twice)
    -0.5: "rsqrt",
    0.5: "sqrt",
}


def _resolve_const_value(node: Node | None, graph: Graph) -> float | None:
    """Follow a single-input broadcast / reshape chain back to the underlying
    ``ConstantOp`` and return its scalar value, or ``None`` if it isn't a
    statically-known constant."""
    seen: set[str] = set()
    while node is not None and node.id not in seen:
        seen.add(node.id)
        if isinstance(node.op, ConstantOp):
            return node.op.value
        if len(node.inputs) != 1:
            return None
        node = graph.nodes.get(node.inputs[0])
    return None


def rewrite(match: Match, inp_x: Node, inp_exp: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    exp_val = _resolve_const_value(inp_exp, graph)
    op_name = _EXPONENT_OPS.get(exp_val) if exp_val is not None else None
    if op_name is None:
        raise RuleSkipped(f"pow exponent {exp_val!r} not a decomposable constant (2 / -0.5 / 0.5)")

    frag = open_fragment(graph, [inp_x])
    # ``x ** 2`` needs the base twice; ``rsqrt`` / ``sqrt`` are unary.
    inputs = [inp_x, inp_x] if op_name == "multiply" else [inp_x]
    out_id = frag.add_node(
        op=ElementwiseOp(op=op_name),
        inputs=inputs,
        output=Tensor(name=out.name, shape=out.shape, dtype=out.dtype),
    )
    frag.outputs = [out_id]
    return frag
