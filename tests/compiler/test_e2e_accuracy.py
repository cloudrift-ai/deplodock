"""End-to-end accuracy tests: compare deplodock output against PyTorch eager.

Uses actual PyTorch tensor data (not pseudo-random) so the numerical
comparison is meaningful. Requires a GPU.
"""

import pytest
import torch

from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _compile_and_run(graph: Graph, input_data: dict[str, list[float]]) -> dict[str, list[float]]:
    """Full pipeline: compile_graph → CudaBackend.compile → run."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline import compile_graph

    result = compile_graph(graph)
    program = CudaBackend().compile(
        result.kernels,
        buf_shapes=result.buf_shapes,
        graph_inputs=result.graph_inputs,
        graph_outputs=result.graph_outputs,
        graph_constants=result.graph_constants,
    )
    run_result = CudaBackend().run(program, input_data=input_data)
    return run_result.outputs


def _assert_close(actual: list[float], expected: list[float], *, rtol: float = 1e-4, atol: float = 1e-5):
    assert len(actual) == len(expected), f"length mismatch: {len(actual)} vs {len(expected)}"
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    mean_diff = sum(abs(a - e) for a, e in zip(actual, expected, strict=True)) / len(actual)
    assert max_diff < atol + rtol * max(abs(e) for e in expected), f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


@requires_cuda
def test_e2e_pointwise_add():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (8,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    outputs = _compile_and_run(g, {"x": x, "y": y})
    expected = [a + b for a, b in zip(x, y, strict=True)]
    _assert_close(list(outputs.values())[0], expected)


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


@requires_cuda
def test_e2e_reduce_sum():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (4,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    torch.manual_seed(0)
    x_t = torch.randn(4, 8)
    x_data = x_t.flatten().tolist()
    expected = x_t.sum(-1).flatten().tolist()
    outputs = _compile_and_run(g, {"x": x_data})
    _assert_close(list(outputs.values())[0], expected)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------


@requires_cuda
def test_e2e_matmul():
    from deplodock.compiler.ir.frontend import MatmulOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 3)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (4, 3)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    torch.manual_seed(0)
    a_t = torch.randn(4, 8)
    b_t = torch.randn(8, 3)
    expected = (a_t @ b_t).flatten().tolist()
    outputs = _compile_and_run(g, {"a": a_t.flatten().tolist(), "b": b_t.flatten().tolist()})
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


@requires_cuda
def test_e2e_rmsnorm():
    """RMSNorm: x * rsqrt(sum(x^2) + eps) * w."""
    torch.manual_seed(42)
    rows, dim = 8, 64
    eps = 1e-6

    x_t = torch.randn(rows, dim).cuda()
    w_t = torch.randn(dim).cuda()
    sq_sum = x_t.pow(2).sum(-1, keepdim=True)
    ref = x_t * torch.rsqrt(sq_sum + eps) * w_t
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    g.add_node(ConstantOp(name="eps", value=eps), [], Tensor("eps", (1,)), node_id="eps")
    g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.inputs = ["X", "w"]

    g.add_node(ElementwiseOp("mul"), ["X", "X"], Tensor("sq", (rows, dim)), node_id="sq")
    g.add_node(ReduceOp("sum", axis=-1), ["sq"], Tensor("red", (rows, 1)), node_id="red")
    g.add_node(ElementwiseOp("add"), ["red", "eps"], Tensor("ae", (rows, 1)), node_id="ae")
    g.add_node(ElementwiseOp("rsqrt"), ["ae"], Tensor("rsq", (rows, 1)), node_id="rsq")
    g.add_node(ElementwiseOp("mul"), ["X", "rsq"], Tensor("norm", (rows, dim)), node_id="norm")
    g.add_node(ElementwiseOp("mul"), ["norm", "w"], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = ["out"]

    outputs = _compile_and_run(
        g,
        {
            "X": x_t.cpu().flatten().tolist(),
            "w": w_t.cpu().flatten().tolist(),
            "eps": [eps],
        },
    )
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


@requires_cuda
def test_e2e_softmax():
    """Softmax: max → sub → exp → sum → div."""
    torch.manual_seed(0)
    rows, cols = 4, 8
    x_t = torch.randn(rows, cols).cuda()
    expected = torch.softmax(x_t, dim=-1).cpu().flatten().tolist()

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = ["x"]

    g.add_node(ReduceOp("max", -1), ["x"], Tensor("mx", (rows, 1)), node_id="mx")
    g.add_node(ElementwiseOp("sub"), ["x", "mx"], Tensor("sub", (rows, cols)), node_id="sub")
    g.add_node(ElementwiseOp("exp"), ["sub"], Tensor("exp", (rows, cols)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (rows, 1)), node_id="sm")
    g.add_node(ElementwiseOp("div"), ["exp", "sm"], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = ["out"]

    outputs = _compile_and_run(g, {"x": x_t.cpu().flatten().tolist()})
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)
