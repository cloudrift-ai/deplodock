"""Tests for the chain-pattern matcher."""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline.engine import Pattern, match_pattern


def _simple_graph() -> Graph:
    """a, b -> mul -> sum."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["a", "b"], output=Tensor("m", (4, 8, 4)), node_id="m")
    g.add_node(op=ReduceOp("sum", 1), inputs=["m"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def test_match_single_pattern():
    g = _simple_graph()
    matches = match_pattern(g, [Pattern("root", ElementwiseOp)])
    assert len(matches) == 1
    assert matches[0].root_node_id == "m"


def test_match_reduce():
    g = _simple_graph()
    matches = match_pattern(g, [Pattern("root", ReduceOp)])
    assert len(matches) == 1
    assert matches[0].root_node_id == "o"


def test_match_with_constraint():
    g = _simple_graph()
    matches = match_pattern(g, [Pattern("root", ElementwiseOp, {"fn": "multiply"})])
    assert len(matches) == 1


def test_constraint_rejects():
    g = _simple_graph()
    matches = match_pattern(g, [Pattern("root", ElementwiseOp, {"fn": "add"})])
    assert len(matches) == 0


def test_two_node_chain():
    """Producer→consumer pattern consumes mul+sum as one chain."""
    g = _simple_graph()
    matches = match_pattern(
        g,
        [
            Pattern("ew", ElementwiseOp),
            Pattern("red", ReduceOp),
        ],
    )
    assert len(matches) == 1
    assert matches[0].nodes == {"ew": "m", "red": "o"}
    assert matches[0].consumed == {"m", "o"}


def test_chain_fails_on_fanout():
    """Producer with multiple consumers can't extend chain."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["x"], output=Tensor("m", (4,)), node_id="m")
    # Two consumers of m, so the chain can't extend.
    g.add_node(op=ReduceOp("sum", 0), inputs=["m"], output=Tensor("r1", (1,)), node_id="r1")
    g.add_node(op=ReduceOp("sum", 0), inputs=["m"], output=Tensor("r2", (1,)), node_id="r2")
    g.inputs = ["x"]
    g.outputs = ["r1", "r2"]
    matches = match_pattern(g, [Pattern("ew", ElementwiseOp), Pattern("red", ReduceOp)])
    assert matches == []
