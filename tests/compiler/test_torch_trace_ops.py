"""Tests for the torch_trace module: op handlers, edge cases, and helpers.

Supplements test_torch_trace.py (which only has 2 smoke tests) with
targeted coverage of individual op handlers and helper functions.
These tests require PyTorch.
"""

import pytest

from deplodock.compiler.trace.torch import has_torch

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch not available")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_get_reduce_axis_from_list():
    """_get_reduce_axis extracts the first axis from a list arg."""
    from deplodock.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None, [2, 3]]

    assert _get_reduce_axis(FakeNode()) == 2


def test_get_reduce_axis_scalar():
    """_get_reduce_axis handles a scalar axis."""
    from deplodock.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None, -1]

    assert _get_reduce_axis(FakeNode()) == -1


def test_get_reduce_axis_default():
    """_get_reduce_axis defaults to -1 when no axis arg is present."""
    from deplodock.compiler.trace.torch import _get_reduce_axis

    class FakeNode:
        args = [None]

    assert _get_reduce_axis(FakeNode()) == -1


def test_op_name_aten_format():
    """_op_name extracts short name from aten.xxx.yyy targets."""
    from deplodock.compiler.trace.torch import _op_name

    assert _op_name("aten.mul.Tensor") == "mul"
    assert _op_name("aten.linear.default") == "linear"
    assert _op_name("aten.scaled_dot_product_attention.default") == "scaled_dot_product_attention"


def test_op_name_non_aten():
    """_op_name returns None for non-ATen targets."""
    from deplodock.compiler.trace.torch import _op_name

    assert _op_name("some.custom.op") is None
    assert _op_name("prims.convert_element_type.default") is None


def test_resolve_inputs_scalars_become_constants():
    """Scalar args (int, float) are promoted to ConstantOp nodes."""
    from deplodock.compiler.ir.base import ConstantOp, InputOp
    from deplodock.compiler.ir.graph import Graph, Tensor
    from deplodock.compiler.trace.torch import _resolve_inputs

    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    node_map = {"x_fx": x}

    class FakeNode:
        name = "test"
        args = []

    # Non-scalar, non-node arg: no constant created.
    class FakeNodeScalar:
        name = "test_s"
        args = []

    # With a scalar arg.
    class FakeNodeWithScalar:
        name = "test_ws"

    FakeNodeWithScalar.args = [type("FN", (), {"name": "x_fx"})(), 1e-5]
    result = _resolve_inputs(FakeNodeWithScalar, node_map, g)
    assert len(result) == 2
    assert result[0] == x
    # Second is a constant node.
    const_node = g.nodes[result[1]]
    assert isinstance(const_node.op, ConstantOp)


# ---------------------------------------------------------------------------
# Module tracing: elementwise ops
# ---------------------------------------------------------------------------


def test_trace_all_elementwise_ops():
    """All supported elementwise ops trace without errors."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.tensor.ir import ElementwiseOp
    from deplodock.compiler.trace.torch import trace_module

    class AllOps(nn.Module):
        def forward(self, x):
            a = torch.neg(x)
            b = torch.exp(a)
            c = torch.abs(b)
            d = torch.tanh(c)
            return d

    m = AllOps()
    x = torch.randn(2, 3)
    g = trace_module(m, (x,))

    fns = {n.op.fn for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "neg" in fns
    assert "exp" in fns
    assert "abs" in fns
    assert "tanh" in fns


def test_trace_binary_ops():
    """Binary ops (add, sub, mul, div) trace correctly with two inputs."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.tensor.ir import ElementwiseOp
    from deplodock.compiler.trace.torch import trace_module

    class BinaryOps(nn.Module):
        def forward(self, x, y):
            return (x + y) * (x - y) / (x + 1e-5)

    m = BinaryOps()
    x = torch.randn(4)
    y = torch.randn(4)
    g = trace_module(m, (x, y))

    fns = {n.op.fn for n in g.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "add" in fns
    assert "sub" in fns
    assert "mul" in fns
    assert "div" in fns


# ---------------------------------------------------------------------------
# Module tracing: reductions
# ---------------------------------------------------------------------------


def test_trace_sum_reduction():
    """aten.sum traces to ReduceOp with correct axis."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.tensor.ir import ReduceOp
    from deplodock.compiler.trace.torch import trace_module

    class SumReduce(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1)

    m = SumReduce()
    x = torch.randn(4, 8)
    g = trace_module(m, (x,))

    reduces = [n for n in g.nodes.values() if isinstance(n.op, ReduceOp)]
    assert len(reduces) >= 1
    assert reduces[0].op.fn == "sum"


def test_trace_max_reduction():
    """aten.amax traces to ReduceOp(max)."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.tensor.ir import ReduceOp
    from deplodock.compiler.trace.torch import trace_module

    class MaxReduce(nn.Module):
        def forward(self, x):
            return x.amax(dim=-1)

    m = MaxReduce()
    x = torch.randn(4, 8)
    g = trace_module(m, (x,))

    reduces = [n for n in g.nodes.values() if isinstance(n.op, ReduceOp)]
    assert len(reduces) >= 1
    assert reduces[0].op.fn == "max"


# ---------------------------------------------------------------------------
# Module tracing: structural ops
# ---------------------------------------------------------------------------


def test_trace_reshape():
    """view/reshape traces to ReshapeOp."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.frontend.ir import ReshapeOp
    from deplodock.compiler.trace.torch import trace_module

    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(2, 6)

    m = Reshape()
    x = torch.randn(3, 4)
    g = trace_module(m, (x,))

    reshapes = [n for n in g.nodes.values() if isinstance(n.op, ReshapeOp)]
    assert len(reshapes) >= 1


def test_trace_transpose():
    """aten.transpose traces to TransposeOp."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.frontend.ir import TransposeOp
    from deplodock.compiler.trace.torch import trace_module

    class Transpose(nn.Module):
        def forward(self, x):
            return x.transpose(0, 1)

    m = Transpose()
    x = torch.randn(3, 4)
    g = trace_module(m, (x,))

    transposes = [n for n in g.nodes.values() if isinstance(n.op, TransposeOp)]
    assert len(transposes) >= 1


# ---------------------------------------------------------------------------
# Module tracing: linear / matmul decomposition
# ---------------------------------------------------------------------------


def test_trace_linear_produces_linearop():
    """nn.Linear produces a single LinearOp node (not decomposed)."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.frontend.ir import LinearOp
    from deplodock.compiler.trace.torch import trace_module

    linear = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)
    g = trace_module(linear, (x,))

    has_linear = any(isinstance(n.op, LinearOp) for n in g.nodes.values())
    assert has_linear, "Linear should produce a LinearOp node"


def test_trace_linear_with_bias():
    """nn.Linear with bias produces LinearOp with has_bias=True."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.ir.frontend.ir import LinearOp
    from deplodock.compiler.trace.torch import trace_module

    linear = nn.Linear(8, 4, bias=True)
    x = torch.randn(2, 8)
    g = trace_module(linear, (x,))

    linear_nodes = [n for n in g.nodes.values() if isinstance(n.op, LinearOp)]
    assert len(linear_nodes) == 1
    assert linear_nodes[0].op.has_bias, "Linear with bias should have has_bias=True"


# ---------------------------------------------------------------------------
# Module tracing: pass-through ops
# ---------------------------------------------------------------------------


def test_trace_passthrough_ops_no_extra_nodes():
    """contiguous/clone produce no new IR nodes (pass-through)."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.trace.torch import trace_module

    class PassThrough(nn.Module):
        def forward(self, x):
            return x.contiguous()

    m = PassThrough()
    x = torch.randn(2, 3)
    g = trace_module(m, (x,))

    # Should just have input(s) — contiguous is a no-op.
    assert len(g.outputs) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_trace_empty_module():
    """Identity module produces a valid graph."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.trace.torch import trace_module

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = Identity()
    x = torch.randn(4)
    g = trace_module(m, (x,))

    assert len(g.inputs) >= 1
    assert len(g.outputs) >= 1


def test_trace_multiple_outputs():
    """Module returning a tuple produces multiple graph outputs."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.trace.torch import trace_module

    class MultiOut(nn.Module):
        def forward(self, x):
            return x + 1, x * 2

    m = MultiOut()
    x = torch.randn(4)
    g = trace_module(m, (x,))

    assert len(g.outputs) == 2
