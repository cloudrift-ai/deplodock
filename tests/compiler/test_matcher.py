"""Tests for graph pattern matching."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.pattern import parse_pattern

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


# ---- tests ----


def test_match_matmul_pattern():
    g = _make_matmul_graph()
    pattern = parse_pattern("Reduce{sum, $k}(Elementwise{mul}($A, $B))")
    matches = match_pattern(g, pattern)
    assert len(matches) == 1
    m = matches[0]
    assert m.root_node_id == "red"
    assert m.bindings["A"] == "A"
    assert m.bindings["B"] == "B"
    assert m.captured_constraints["k"] == 1


def test_no_match_wrong_fn():
    g = _make_matmul_graph()
    pattern = parse_pattern("Reduce{max, $k}(Elementwise{mul}($A, $B))")
    matches = match_pattern(g, pattern)
    assert len(matches) == 0


def test_no_match_wrong_op():
    g = _make_matmul_graph()
    pattern = parse_pattern("Scan{sum, $k}(Elementwise{mul}($A, $B))")
    matches = match_pattern(g, pattern)
    assert len(matches) == 0


def test_commutativity_via_alternatives():
    g = _make_matmul_graph()
    # A,B order matches first alternative.
    pattern = parse_pattern("Reduce{sum,$k}(Elementwise{mul}($A,$B)) | Reduce{sum,$k}(Elementwise{mul}($B,$A))")
    matches = match_pattern(g, pattern)
    assert len(matches) == 1


def test_wildcard_matches_any_node():
    g = _make_matmul_graph()
    pattern = parse_pattern("Reduce{sum, $k}(_)")
    matches = match_pattern(g, pattern)
    assert len(matches) == 1
    assert matches[0].root_node_id == "red"


def test_fan_out_same_variable():
    """$x appearing twice must match the same node."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    g.inputs = [x]
    # exp(x) * exp(x) — same input
    exp_node = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[x], output=Tensor("expX", (4,)), node_id="exp")
    mul_node = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[exp_node, exp_node],
        output=Tensor("sq", (4,)),
        node_id="mul",
    )
    g.outputs = [mul_node]

    # Pattern: mul($a, $a) — same var, must point to same node.
    pattern = parse_pattern("Elementwise{mul}($a, $a)")
    matches = match_pattern(g, pattern)
    assert len(matches) == 1
    assert matches[0].bindings["a"] == "exp"


def test_fan_out_different_nodes_no_match():
    """$x appearing twice must NOT match if they point to different nodes."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    y = g.add_node(op=InputOp(), inputs=[], output=Tensor("Y", (4,)), node_id="y")
    g.inputs = [x, y]
    mul_node = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x, y],
        output=Tensor("XY", (4,)),
        node_id="mul",
    )
    g.outputs = [mul_node]

    # Pattern: mul($a, $a) — should NOT match since inputs are different.
    pattern = parse_pattern("Elementwise{mul}($a, $a)")
    matches = match_pattern(g, pattern)
    assert len(matches) == 0


def test_variable_captures_subgraph():
    """$body captures whatever subtree feeds into the reduce."""
    g = _make_matmul_graph()
    pattern = parse_pattern("Reduce{$f, $ax}($body)")
    matches = match_pattern(g, pattern)
    assert len(matches) == 1
    assert matches[0].bindings["body"] == "ew"
    assert matches[0].captured_constraints["f"] == "sum"
