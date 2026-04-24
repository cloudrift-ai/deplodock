"""Tests for the pattern-based rewrite engine (``run_pass``)."""

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import run_pass

_DECOMP_DIR = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "pipeline" / "passes" / "frontend" / "decomposition"


def _make_silu_graph() -> Graph:
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("N",)), node_id="x")
    g.inputs = [x]
    silu = g.add_node(op=ElementwiseOp(op="silu"), inputs=[x], output=Tensor("out", ("N",)), node_id="out")
    g.outputs = [silu]
    return g


def test_decompose_silu():
    """Running the decomposition pass replaces silu with primitive ops."""
    g = _make_silu_graph()
    result = run_pass(g, _DECOMP_DIR)
    fns = [n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)]
    assert "silu" not in fns
    assert "neg" in fns
    assert "exp" in fns


def test_decomposed_output_preserved():
    """Graph outputs still point at an ElementwiseOp after decomposition."""
    g = _make_silu_graph()
    result = run_pass(g, _DECOMP_DIR)
    assert len(result.outputs) == 1
    assert isinstance(result.nodes[result.outputs[0]].op, ElementwiseOp)


def test_fixed_point_is_idempotent():
    """Applying a pass to its own output is a no-op."""
    g = _make_silu_graph()
    once = run_pass(g, _DECOMP_DIR)
    twice = run_pass(once, _DECOMP_DIR)
    assert len(twice.nodes) == len(once.nodes)
