"""Tests for the tensor IR: Tensor, Node, Graph."""

import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp

# ---- helpers ----


def _make_matmul_graph():
    """Build: C[M,1,N] = reduce_sum(elementwise_mul(A[M,K,N], B[M,K,N]), axis=1).

    Uses keepdim reduction and matching-shape elementwise inputs to stay on
    the Tensor IR rank-preservation invariant (see pipeline/passes/frontend/decomposition/_broadcast.py for how
    decomposition rules insert explicit IndexMapOps when shapes differ).
    """
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K", "N")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("M", "K", "N")), node_id="B")
    g.inputs = [a, b]

    ew = g.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[a, b],
        output=Tensor("AB", ("M", "K", "N")),
        node_id="ew",
    )
    red = g.add_node(
        op=ReduceOp(op="sum", axis=1),
        inputs=[ew],
        output=Tensor("C", ("M", 1, "N")),
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
            op=ElementwiseOp(op="add"),
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
        op=ReduceOp(op="sum", axis=0),
        inputs=["ew"],
        output=Tensor("C2", (1, "K", "N")),
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
    a = g.add_node(op=ElementwiseOp(op="exp"), inputs=[x], output=Tensor("expX", (4,)), node_id="a")
    b = g.add_node(op=ElementwiseOp(op="negative"), inputs=[x], output=Tensor("negX", (4,)), node_id="b")
    g.outputs = [a, b]
    assert sorted(g.consumers("x")) == ["a", "b"]


def test_to_dict_serializes_composite_shape_dim():
    """A node whose output shape carries a COMPOSITE Dim (``BinaryExpr``-backed —
    e.g. the demoted symbolic-N B operand's TMA-padded ``round_up(seq_len, 64)``
    inner extent, or a CatOp output) must not crash ``to_dict``: the dump path
    serializes it to its pretty expr string. Atomic dims (int / Var name) still
    return their scalar value for the ``run --ir`` round-trip."""
    from deplodock.compiler.dim import Dim

    g = Graph()
    padded = Dim((Dim("seq_len").ceil_div(64) * 64).expr, hint=512)  # ((seq_len + 63) // 64) * 64
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (128, padded)), node_id="xnb")
    g.inputs, g.outputs = ["xnb"], ["xnb"]

    d = g.to_dict()  # must not raise (pre-fix: Dim.value raised on the composite)
    shape = d["nodes"]["xnb"]["output"]["shape"]
    assert shape[0] == 128, "static inner-rank dim keeps its int value"
    assert isinstance(shape[1], str) and "seq_len" in shape[1] and "64" in shape[1], (
        f"composite dim must serialize to its pretty expr string, got {shape[1]!r}"
    )
