"""End-to-end accuracy tests: compare deplodock output against PyTorch eager.

Uses actual PyTorch tensor data (not pseudo-random) so the numerical
comparison is meaningful.  Requires a GPU and transformers.
"""

import math

import pytest
import torch

from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.plan import plan_graph

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _compile_and_run_with_data(graph: Graph, input_data: dict[str, list[float]]) -> dict[str, list[float]]:
    """Full pipeline with actual input data."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    fused = auto_fuse(graph)
    plan = plan_graph(fused)
    backend = CudaBackend()
    program = backend.compile(plan)
    result = backend.run(program, input_data=input_data)
    return result.outputs


@requires_cuda
def test_e2e_rmsnorm_matches_torch():
    """RMSNorm: deplodock matches PyTorch RMSNorm with same inputs."""
    torch.manual_seed(42)
    rows, dim = 8, 64
    eps = 1e-6

    x_t = torch.randn(rows, dim).cuda()
    w_t = torch.randn(dim).cuda()

    # PyTorch reference: rsqrt(sum(x^2) + eps) * x * w
    # Note: LlamaRMSNorm uses sum (not mean) in the traced decomposition.
    sq_sum = x_t.pow(2).sum(-1, keepdim=True)
    ref = x_t * torch.rsqrt(sq_sum + eps) * w_t
    expected = ref.cpu().flatten().tolist()

    # Deplodock graph — same decomposition as torch.export trace
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    eps_n = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    w = g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.inputs = [x, w]

    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (rows, dim)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=-1), [sq], Tensor("red", (rows, 1)), node_id="red")
    ae = g.add_node(ElementwiseOp("add"), [red, eps_n], Tensor("ae", (rows, 1)), node_id="ae")
    rsq = g.add_node(ElementwiseOp("rsqrt"), [ae], Tensor("rsq", (rows, 1)), node_id="rsq")
    norm = g.add_node(ElementwiseOp("mul"), [x, rsq], Tensor("norm", (rows, dim)), node_id="norm")
    out = g.add_node(ElementwiseOp("mul"), [norm, w], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "X": x_t.cpu().flatten().tolist(),
        "w": w_t.cpu().flatten().tolist(),
        "eps": [eps],
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    assert len(actual) == len(expected)
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"RMSNorm max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_matmul_matches_torch():
    """Matmul: deplodock matches torch.mm with same inputs."""
    torch.manual_seed(42)
    m, k, n = 16, 32, 64

    a_t = torch.randn(m, k).cuda()
    b_t = torch.randn(k, n).cuda()

    ref = torch.mm(a_t, b_t)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("C", (m, n)), node_id="C")
    g.outputs = [out]

    input_data = {
        "A": a_t.cpu().flatten().tolist(),
        "B": b_t.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"Matmul max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_softmax_matches_torch():
    """Softmax: deplodock matches torch.softmax with same inputs."""
    torch.manual_seed(42)
    rows, cols = 8, 32

    x_t = torch.randn(rows, cols).cuda()
    ref = torch.softmax(x_t, dim=-1)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (rows, cols)), node_id="X")
    g.inputs = [x]

    mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (rows, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (rows, cols)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (rows, cols)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (rows, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = [out]

    input_data = {"X": x_t.cpu().flatten().tolist()}
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-4, f"Softmax max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_matmul_softmax_matches_torch():
    """Matmul + softmax: deplodock matches torch with same inputs."""
    torch.manual_seed(42)
    m, k, n = 8, 16, 32

    a_t = torch.randn(m, k).cuda()
    b_t = torch.randn(k, n).cuda()
    scale = 1.0 / math.sqrt(k)

    scores = torch.mm(a_t, b_t) * scale
    ref = torch.softmax(scores, dim=-1)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (m, n)), node_id="mm")
    scaled = g.add_node(ElementwiseOp("mul"), [mm, sc], Tensor("sc", (m, n)), node_id="sc")
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (m, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("sub", (m, n)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (m, n)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (m, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (m, n)), node_id="out")
    g.outputs = [out]

    input_data = {
        "A": a_t.cpu().flatten().tolist(),
        "B": b_t.cpu().flatten().tolist(),
        "scale": [scale],
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-3, f"Matmul+softmax max diff = {max_diff:.6f}"
