"""Tests for the grammar-based graph matcher."""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.matcher import Production, match_grammar


def _simple_graph() -> Graph:
    """x -> mul -> sum (matmul pattern)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8, 4)), node_id="m")
    g.add_node(op=ReduceOp("sum", 1), inputs=["m"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def test_match_single_production():
    g = _simple_graph()
    matches = match_grammar(g, [Production("root", ElementwiseOp, "1")])
    assert len(matches) == 1
    assert matches[0].root_node_id == "m"


def test_match_reduce():
    g = _simple_graph()
    matches = match_grammar(g, [Production("root", ReduceOp, "1")])
    assert len(matches) == 1
    assert matches[0].root_node_id == "o"


def test_match_with_constraint():
    g = _simple_graph()
    matches = match_grammar(g, [Production("root", ElementwiseOp, "1", {"fn": "mul"})])
    assert len(matches) == 1


def test_constraint_rejects():
    g = _simple_graph()
    matches = match_grammar(g, [Production("root", ElementwiseOp, "1", {"fn": "add"})])
    assert len(matches) == 0


def test_chain_consumes_forward():
    """A 2-production grammar consumes mul+sum as one chain."""
    g = _simple_graph()
    matches = match_grammar(
        g,
        [
            Production("ew", ElementwiseOp, "+"),
            Production("red", ReduceOp, "?"),
        ],
    )
    assert len(matches) == 1
    assert matches[0].get("ew") == ["m"]
    assert matches[0].get("red") == ["o"]
