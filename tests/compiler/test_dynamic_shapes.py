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


def test_cuda_symbolic_elementwise_one_kernel_multiple_seq_lens():
    """M2 validation slice: same symbolic graph compiles to ONE CudaOp
    whose kernel signature carries ``int seq_len``; running it at two
    different ``seq_len`` values resolves the launch geometry from the
    actual input shape without recompiling."""
    pytest = __import__("pytest")
    cupy = pytest.importorskip("cupy")
    del cupy
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp

    graph = _symbolic_elementwise_graph()
    backend = CudaBackend()
    compiled = backend.compile(graph)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert len(cuda_ops) == 1, f"expected one CudaOp, got {len(cuda_ops)}"
    op = cuda_ops[0]
    assert "seq_len" in op.runtime_args, f"runtime_args missing 'seq_len': {op.runtime_args}"
    assert "int seq_len" in op.kernel_source, f"kernel signature missing int seq_len param:\n{op.kernel_source}"

    cached_source = op.kernel_source
    for s in (5, 13):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"x": x})
        y = result.outputs["y"]
        assert y.shape == (1, s, 2048)
        np.testing.assert_allclose(y, np.exp(x), rtol=1e-5, atol=1e-5)
    # Sanity-check: nothing in the lowering path swapped the kernel source between runs.
    assert op.kernel_source == cached_source


def test_symbolic_sdpa_traces_and_decomposes():
    """M4 validation: a single-layer causal SDPA + reshape traced from
    torch, free seq_len dim rewritten to ``Dim('seq_len')``, decomposes
    + lifts cleanly. Execution still requires M5 (symbolic reduce
    axis); this test stops at the loop-lifted graph and asserts the
    surviving op shapes carry symbolic dims end-to-end."""
    import torch

    from deplodock.compiler.ir.frontend.ir import ReshapeOp
    from deplodock.compiler.trace.torch import trace_module

    class AttnBlock(torch.nn.Module):
        def forward(self, q, k, v):
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            B, H, S, D = attn.shape
            return attn.reshape(B, S, H * D)

    b, h, s, d = 1, 4, 16, 64
    graph = trace_module(AttnBlock(), (torch.randn(b, h, s, d), torch.randn(b, h, s, d), torch.randn(b, h, s, d)))
    # Rewrite the seq_len position (index 2 of every shape, value 16) and ReshapeOp.shape.
    for node in graph.nodes.values():
        node.output.shape = tuple(Dim("seq_len") if (i in (1, 2) and dim == s) else dim for i, dim in enumerate(node.output.shape))
        if isinstance(node.op, ReshapeOp):
            node.op.shape = tuple("seq_len" if (i in (1, 2) and dim == s) else dim for i, dim in enumerate(node.op.shape))

    decomp = Pipeline.build(["frontend/decomposition", "frontend/optimization"]).run(graph)
    # SDPA decomposition stamps softmax (max + exp + sum) over the kv_len axis.
    # That axis is symbolic — the reduce ops survive the decomposition pass but
    # are accepted because M0 widened ``ReduceOp.axis`` handling.
    assert any(Dim("seq_len") in n.output.shape for n in decomp.nodes.values()), (
        "expected at least one node to carry symbolic seq_len after decomposition"
    )

    lifted = Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting"]).run(graph)
    loop_op_count = sum(1 for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    assert loop_op_count >= 1, "expected at least one LoopOp after lifting"


def test_cuda_symbolic_rmsnorm_traced_and_run():
    """End-to-end on a real ``torch.nn.RMSNorm``: trace at one seq_len,
    rewrite the traced tensors' middle dim to ``Dim("seq_len")``, compile
    once, run at two distinct seq_len values, compare to torch eager."""
    pytest = __import__("pytest")
    cupy = pytest.importorskip("cupy")
    del cupy
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    m = torch.nn.RMSNorm(2048)
    graph = trace_module(m, (torch.randn(1, 32, 2048),))
    for node in graph.nodes.values():
        node.output.shape = tuple(Dim("seq_len") if (i == 1 and d == 32) else d for i, d in enumerate(node.output.shape))
    backend = CudaBackend()
    compiled = backend.compile(graph)

    weight = m.weight.detach().numpy().astype(np.float32)
    for s in (8, 32):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"x": x, "p_weight": weight})
        y = next(iter(result.outputs.values()))
        with torch.no_grad():
            y_ref = torch.nn.functional.rms_norm(torch.from_numpy(x), (2048,), m.weight, eps=m.eps).numpy()
        assert y.shape == (1, s, 2048)
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
