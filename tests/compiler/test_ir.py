"""Tests for the tensor IR: Tensor, Node, Graph."""

import pytest

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp

# ---- helpers ----


def _make_matmul_graph():
    """Build: C[M,N] = reduce_sum(elementwise_mul(A[M,K], B[K,N]), axis=1)."""
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


def test_add_node_and_lookup():
    g = Graph()
    nid = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 4)), node_id="x")
    assert nid == "x"
    assert "x" in g.nodes
    assert g.nodes["x"].output.name == "X"


def test_duplicate_id_raises():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    with pytest.raises(ValueError, match="already exists"):
        g.add_node(op=InputOp(), inputs=[], output=Tensor("Y", (4,)), node_id="x")


def test_missing_input_raises():
    g = Graph()
    with pytest.raises(ValueError, match="does not exist"):
        g.add_node(
            op=ElementwiseOp(fn="add"),
            inputs=["nonexistent"],
            output=Tensor("Y", (4,)),
        )


def test_topological_order():
    g = _make_matmul_graph()
    order = g.topological_order()
    assert order.index("A") < order.index("ew")
    assert order.index("B") < order.index("ew")
    assert order.index("ew") < order.index("red")


def test_consumers():
    g = _make_matmul_graph()
    assert g.consumers("A") == ["ew"]
    assert g.consumers("B") == ["ew"]
    assert g.consumers("ew") == ["red"]
    assert g.consumers("red") == []


def test_replace_node():
    g = _make_matmul_graph()
    # Add a new node and rewire red's consumers to it.
    new_id = g.add_node(
        op=ReduceOp(fn="sum", axis=0),
        inputs=["ew"],
        output=Tensor("C2", ("N",)),
        node_id="red2",
    )
    g.replace_node("red", new_id)
    assert "red2" in [o for o in g.outputs]


def test_remove_node():
    g = _make_matmul_graph()
    g.outputs = []
    g.remove_node("red")
    assert "red" not in g.nodes


def test_copy_is_independent():
    g = _make_matmul_graph()
    g2 = g.copy()
    g2.remove_node("red")
    assert "red" in g.nodes
    assert "red" not in g2.nodes


def test_fan_out():
    """One node consumed by two different nodes."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    g.inputs = [x]
    a = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[x], output=Tensor("expX", (4,)), node_id="a")
    b = g.add_node(op=ElementwiseOp(fn="neg"), inputs=[x], output=Tensor("negX", (4,)), node_id="b")
    g.outputs = [a, b]
    assert sorted(g.consumers("x")) == ["a", "b"]
