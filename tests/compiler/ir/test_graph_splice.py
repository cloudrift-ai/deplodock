"""Tests for ``Graph.splice`` — single- and multi-output forms.

``splice`` is the engine's only graph-rewrite primitive (every rule that
returns a ``Graph`` fragment is applied through it). The single-output form
redirects one node's consumers to ``fragment.outputs[0]``; the multi-output
form (``output={old_id: frag_output_id}``) redirects several at once — used to
inline one producer into all its consumers in a single rewrite.
"""

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp


def _make_fanout_graph() -> Graph:
    """``x -> p; a=f(p); b=g(p); ua=h(a); ub=h(b)`` with p shared by a and b."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("p", (8,)), node_id="p")
    g.add_node(ElementwiseOp("exp"), ["p"], Tensor("a", (8,)), node_id="a")
    g.add_node(ElementwiseOp("reciprocal"), ["p"], Tensor("b", (8,)), node_id="b")
    g.add_node(ElementwiseOp("negative"), ["a"], Tensor("ua", (8,)), node_id="ua")
    g.add_node(ElementwiseOp("negative"), ["b"], Tensor("ub", (8,)), node_id="ub")
    g.inputs, g.outputs = ["x"], ["ua", "ub"]
    return g


def test_splice_single_output_back_compat():
    """``output`` as a str keeps the original behavior: one node redirected,
    a single new id returned."""
    g = _make_fanout_graph()
    frag = Graph()
    frag.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    frag.add_node(ElementwiseOp("exp"), ["x"], Tensor("a", (8,)), node_id="fa")
    frag.outputs = ["fa"]

    new_id = g.splice(frag, consumed={"a"}, output="a")
    assert isinstance(new_id, str)
    assert new_id == "a"  # promoted back to the friendly output name
    assert g.nodes["ua"].inputs == ["a"]  # downstream rewired to the new node
    assert g.nodes["a"].inputs == ["x"]  # the fragment node reads x directly
    assert "p" in g.nodes  # p not consumed here — still feeds b


def test_splice_multi_output_redirects_each_consumer():
    """``output`` as a dict redirects several consumers to distinct fragment
    outputs and dissolves the shared producer — all in one splice."""
    g = _make_fanout_graph()

    # Fragment: replace a and b (dissolving p) with fa, fb that each read x directly.
    frag = Graph()
    frag.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    frag.add_node(ElementwiseOp("exp"), ["x"], Tensor("a", (8,)), node_id="fa")
    frag.add_node(ElementwiseOp("reciprocal"), ["x"], Tensor("b", (8,)), node_id="fb")
    frag.outputs = ["fa", "fb"]

    result = g.splice(frag, consumed={"p", "a", "b"}, output={"a": "fa", "b": "fb"})

    # Returns the {old: new} map; both promoted to their friendly names.
    assert result == {"a": "a", "b": "b"}
    # Shared producer dissolved.
    assert "p" not in g.nodes
    # Each downstream consumer rewired to its own replacement.
    assert g.nodes["ua"].inputs == ["a"]
    assert g.nodes["ub"].inputs == ["b"]
    # The replacements read x directly (producer inlined away).
    assert g.nodes["a"].inputs == ["x"]
    assert g.nodes["b"].inputs == ["x"]
    # Graph outputs unchanged in identity (ua/ub kept their ids).
    assert g.outputs == ["ua", "ub"]
