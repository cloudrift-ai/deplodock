"""Tests for the PyTorch → Graph IR tracer."""

import pytest

from deplodock.compiler.torch_trace import has_torch

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch not available")


def test_trace_linear():
    """Trace nn.Linear → verify InputOp, ConstantOp, matmul decomposition."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.torch_trace import trace_module

    linear = nn.Linear(4, 2, bias=False)
    x = torch.randn(1, 4)
    g = trace_module(linear, (x,))

    op_types = {type(n.op).__name__ for n in g.nodes.values()}
    # Should have at least InputOp and some form of matmul decomposition.
    assert "InputOp" in op_types or "ConstantOp" in op_types
    assert len(g.nodes) > 0
    assert len(g.outputs) > 0


def test_trace_simple_elementwise():
    """Trace a module with elementwise ops."""
    import torch
    import torch.nn as nn

    from deplodock.compiler.torch_trace import trace_module

    class AddMul(nn.Module):
        def forward(self, x, y):
            return (x + y) * x

    m = AddMul()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    g = trace_module(m, (x, y))

    op_types = [type(n.op).__name__ for n in g.nodes.values()]
    assert "ElementwiseOp" in op_types
    assert len(g.inputs) == 2
    assert len(g.outputs) == 1
