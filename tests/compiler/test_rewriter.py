"""Tests for the rewrite engine."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp
from deplodock.compiler.pattern import parse_pattern
from deplodock.compiler.rewriter import Pass, Rule


def _make_silu_graph():
    """Build graph with silu(x) for decomposition testing."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("N",)), node_id="x")
    g.inputs = [x]
    silu = g.add_node(op=ElementwiseOp(fn="silu"), inputs=[x], output=Tensor("out", ("N",)), node_id="out")
    g.outputs = [silu]
    return g


def _load_decomp_rule():
    from pathlib import Path

    return Rule.from_file(
        Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "decomposition" / "002_decompose_silu.py"
    )


def test_decompose_silu():
    """Decomposition rule replaces silu with primitive ops."""
    g = _make_silu_graph()
    result = Pass(name="decomp", rules=[_load_decomp_rule()]).apply(g)

    fns = [n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)]
    assert "silu" not in fns
    assert "neg" in fns
    assert "exp" in fns


def test_decomposed_output_is_correct():
    """Output of decomposed graph points to the last op."""
    g = _make_silu_graph()
    result = Pass(name="decomp", rules=[_load_decomp_rule()]).apply(g)

    assert len(result.outputs) == 1
    assert isinstance(result.nodes[result.outputs[0]].op, ElementwiseOp)


def test_fixed_point_no_change():
    """Applying decomposition to an already-decomposed graph is a no-op."""
    g = _make_silu_graph()
    p = Pass(name="decomp", rules=[_load_decomp_rule()])
    decomposed = p.apply(g)
    decomposed2 = p.apply(decomposed)
    assert len(decomposed2.nodes) == len(decomposed.nodes)


def test_pass_with_no_matching_rules():
    """A pass whose rules don't match returns the graph unchanged."""
    g = _make_silu_graph()
    rule = Rule(name="noop", pattern=parse_pattern("Scan{sum, $ax}($x)"), rewrite=lambda graph, match: graph)
    result = Pass(name="noop", rules=[rule]).apply(g)
    assert len(result.nodes) == len(g.nodes)
