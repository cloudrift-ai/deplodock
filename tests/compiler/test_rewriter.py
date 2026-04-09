"""End-to-end tests for the rewrite engine."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import (
    ElementwiseOp,
    FusedReduceElementwiseOp,
    InputOp,
    ReduceOp,
)
from deplodock.compiler.pattern import parse_pattern
from deplodock.compiler.rewriter import Pass, Rule

# ---- helpers ----


def _make_matmul_graph():
    """C = reduce_sum(elementwise_mul(A, B))."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("K", "N")), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a, b],
        output=Tensor("AB", ("M", "K", "N")),
        node_id="ew",
    )
    red = g.add_node(
        op=ReduceOp(fn="sum", axis=1),
        inputs=[ew],
        output=Tensor("C", ("M", "N")),
        node_id="red",
    )
    g.outputs = [red]
    return g


def _import_rewrite_fn(rule_path: str):
    """Import a rewrite function from a rule file."""
    return Rule.from_file(__import__("pathlib").Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / rule_path)


# ---- tests ----


def test_fuse_reduce_elementwise():
    """Naive matmul graph → fused node, no intermediate."""
    g = _make_matmul_graph()
    rule = _import_rewrite_fn("fusion/001_fuse_reduce_elementwise.py")
    fusion_pass = Pass(name="fusion", rules=[rule])
    result = fusion_pass.apply(g)

    # The elementwise and reduce nodes should be gone.
    op_types = [type(n.op).__name__ for n in result.nodes.values()]
    assert "ElementwiseOp" not in op_types
    assert "ReduceOp" not in op_types
    assert "FusedReduceElementwiseOp" in op_types

    # The fused node should consume A and B directly.
    fused_nodes = [n for n in result.nodes.values() if isinstance(n.op, FusedReduceElementwiseOp)]
    assert len(fused_nodes) == 1
    fused = fused_nodes[0]
    assert fused.op.reduce_fn == "sum"
    assert fused.op.elementwise_fn == "mul"
    assert set(fused.inputs) == {"A", "B"}


def test_fused_graph_output_is_correct():
    """Output of fused graph should point to the fused node."""
    g = _make_matmul_graph()
    rule = _import_rewrite_fn("fusion/001_fuse_reduce_elementwise.py")
    fusion_pass = Pass(name="fusion", rules=[rule])
    result = fusion_pass.apply(g)

    assert len(result.outputs) == 1
    out_node = result.nodes[result.outputs[0]]
    assert isinstance(out_node.op, FusedReduceElementwiseOp)


def test_fixed_point_no_change():
    """Applying fusion to an already-fused graph produces no change."""
    g = _make_matmul_graph()
    rule = _import_rewrite_fn("fusion/001_fuse_reduce_elementwise.py")
    fusion_pass = Pass(name="fusion", rules=[rule])
    fused = fusion_pass.apply(g)

    # Apply again — should be a no-op.
    fused2 = fusion_pass.apply(fused)
    assert len(fused2.nodes) == len(fused.nodes)
    assert set(type(n.op).__name__ for n in fused2.nodes.values()) == set(type(n.op).__name__ for n in fused.nodes.values())


def test_pass_with_no_matching_rules():
    """A pass whose rules don't match returns the graph unchanged."""
    g = _make_matmul_graph()
    # Use a pattern that won't match anything in this graph.
    rule = Rule(
        name="noop",
        pattern=parse_pattern("Scan{sum, $ax}($x)"),
        rewrite=lambda graph, match: graph,
    )
    p = Pass(name="noop_pass", rules=[rule])
    result = p.apply(g)
    assert len(result.nodes) == len(g.nodes)
