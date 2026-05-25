"""Symbolic-extent ``Dim`` round-trips through trace → lift → ``LoopOp.forward``.

Plan: ``plans/dynamic-shapes.md``. M1 milestone: free-axis extents stay
symbolic through lifting; ``LoopOp.forward`` binds them from input array
shapes at execute time and specializes the body before C++ rendering.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler import dtype as dt
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import Pipeline


def _symbolic_elementwise_graph() -> Graph:
    """``y = exp(x)`` where ``x`` has shape ``(1, S, 2048)`` — S symbolic."""
    g = Graph()
    sym_shape = (Dim(1), Dim("seq_len"), Dim(2048))
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", sym_shape, dt.F32), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("y", sym_shape, dt.F32), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _symbolic_reduce_graph() -> Graph:
    """``y = sum(x, axis=-1)`` where ``x`` has shape ``(1, S, 2048)`` — S symbolic, reduce axis static."""
    g = Graph()
    in_shape = (Dim(1), Dim("seq_len"), Dim(2048))
    out_shape = (Dim(1), Dim("seq_len"), Dim(1))  # keepdim
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", in_shape, dt.F32), node_id="x")
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("y", out_shape, dt.F32), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_lift_elementwise_preserves_symbolic_free_axes():
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_nodes = [n for n in lifted.nodes.values() if isinstance(n.op, LoopOp)]
    assert loop_nodes, "expected at least one LoopOp after lifting"
    extents = {ax.name: ax.extent for n in loop_nodes for ax in n.op.axes}
    assert Dim("seq_len") in extents.values(), f"symbolic seq_len axis lost: {extents}"


def test_lift_reduce_allows_symbolic_free_axis_static_reduce():
    graph = _symbolic_reduce_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_nodes = [n for n in lifted.nodes.values() if isinstance(n.op, LoopOp)]
    assert loop_nodes, "expected at least one LoopOp after lifting"
    extents = {ax.name: ax.extent for n in loop_nodes for ax in n.op.axes}
    assert Dim("seq_len") in extents.values(), f"free symbolic axis lost: {extents}"
    assert Dim(2048) in extents.values(), f"static reduce axis missing: {extents}"


def test_loop_forward_binds_symbolic_axis_from_input_shape():
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_node = next(n for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    x = np.random.RandomState(0).standard_normal((1, 7, 2048)).astype(np.float32)
    out = loop_node.op.forward(x)
    assert out.shape == (1, 7, 2048)
    np.testing.assert_allclose(out, np.exp(x), rtol=1e-5, atol=1e-6)


def test_loop_forward_same_kernel_different_seq_lens():
    """Same symbolic LoopOp, two distinct runtime ``seq_len`` values both run cleanly."""
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_op = next(n.op for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    for s in (3, 11):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        out = loop_op.forward(x)
        assert out.shape == (1, s, 2048)
        np.testing.assert_allclose(out, np.exp(x), rtol=1e-5, atol=1e-6)
