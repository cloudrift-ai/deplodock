"""Tests for CUDA backend: codegen, lowering, and end-to-end GPU execution."""

import pytest

from deplodock.compiler.cuda.codegen import emit_kernel
from deplodock.compiler.cuda.ir import (
    AugAssign,
    BinOp,
    CudaBuiltin,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    Var,
    VarDecl,
)
from deplodock.compiler.cuda.lower import lower_graph
from deplodock.compiler.cuda.runner import has_cuda_gpu, has_nvcc, run_kernel
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.rewriter import Pass, Rule

# ---- helpers ----


def _make_matmul_graph(m, k, n):
    """Build naive matmul graph: C[M,N] = reduce_sum(ew_mul(A[M,K], B[K,N]))."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (m, k)), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a, b],
        output=Tensor("AB", (m, k, n)),
        node_id="ew",
    )
    red = g.add_node(
        op=ReduceOp(fn="sum", axis=1),
        inputs=[ew],
        output=Tensor("C", (m, n)),
        node_id="red",
    )
    g.outputs = [red]
    return g


def _fuse(graph):
    """Apply fusion pass to graph."""
    from pathlib import Path

    rule_path = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion" / "001_fuse_reduce_elementwise.py"
    rule = Rule.from_file(rule_path)
    return Pass(name="fusion", rules=[rule]).apply(graph)


def _python_matmul(a, b, m, k, n):
    """Pure-Python matrix multiply for verification."""
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i * k + kk] * b[kk * n + j]
            c[i * n + j] = s
    return c


# ---- codegen unit tests (no GPU needed) ----


def test_emit_simple_kernel():
    """Emit a minimal kernel and check it contains expected CUDA constructs."""
    kernel = KernelDef(
        name="test_kernel",
        params=[KernelParam("float*", "X"), KernelParam("int", "N")],
        body=[
            VarDecl("int", "i", BinOp("+", BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")), CudaBuiltin("threadIdx.x"))),
            IfStmt(
                BinOp("<", Var("i"), Var("N")),
                [VarDecl("float", "val", Literal(1.0))],
            ),
        ],
    )
    source = emit_kernel(kernel)
    assert "__global__ void test_kernel" in source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "float val = 1.0f;" in source


def test_emit_matmul_kernel():
    """Emit a matmul kernel from a lowered graph and verify structure."""
    g = _make_matmul_graph(4, 3, 2)
    fused = _fuse(g)
    kernel = lower_graph(fused)
    source = emit_kernel(kernel)

    assert "__global__ void fused_matmul" in source
    assert "float* A" in source
    assert "float* B" in source
    assert "float* C" in source
    assert "int M" in source
    assert "int N" in source
    assert "int K" in source
    assert "float acc = 0.0f;" in source
    assert "for (int k = 0;" in source
    assert "acc +=" in source
    assert "A[" in source
    assert "B[" in source
    assert "C[" in source


def test_emit_for_loop():
    """Test ForLoop emission with aug-assign body."""
    kernel = KernelDef(
        name="loop_test",
        params=[KernelParam("float*", "X")],
        body=[
            VarDecl("float", "s", Literal(0.0)),
            ForLoop(
                "i",
                Literal(0, dtype="int"),
                Literal(10, dtype="int"),
                [AugAssign("s", "+=", Var("i"))],
            ),
        ],
    )
    source = emit_kernel(kernel)
    assert "for (int i = 0; i < 10; i++)" in source
    assert "s += i;" in source


# ---- lowering unit tests (no GPU needed) ----


def test_lower_matmul_produces_kernel():
    """Lower a fused matmul graph and check the KernelDef structure."""
    g = _make_matmul_graph(4, 3, 2)
    fused = _fuse(g)
    kernel = lower_graph(fused)

    assert kernel.name == "fused_matmul"
    param_names = [p.name for p in kernel.params]
    assert "A" in param_names
    assert "B" in param_names
    assert "C" in param_names
    assert "M" in param_names
    assert "N" in param_names
    assert "K" in param_names


# ---- GPU integration test ----


requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


@requires_cuda
def test_matmul_end_to_end():
    """Full pipeline: graph → fuse → lower → codegen → compile → run → verify."""
    M, K, N = 4, 3, 2

    # Build and fuse the graph.
    g = _make_matmul_graph(M, K, N)
    fused = _fuse(g)

    # Lower to CUDA IR and generate source.
    kernel = lower_graph(fused)
    source = emit_kernel(kernel)

    # Test data.
    a_data = [float(i + 1) for i in range(M * K)]  # 1..12
    b_data = [float(i + 1) for i in range(K * N)]  # 1..6

    # Run on GPU.
    result = run_kernel(
        kernel=kernel,
        kernel_source=source,
        inputs={"A": a_data, "B": b_data},
        output_name="C",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
    )

    # Verify against Python reference.
    expected = _python_matmul(a_data, b_data, M, K, N)
    assert len(result) == len(expected)
    for got, exp in zip(result, expected, strict=True):
        assert abs(got - exp) < 1e-4, f"Mismatch: got {got}, expected {exp}"


@requires_cuda
def test_matmul_larger():
    """Test with a larger matrix to exercise multiple thread blocks."""
    M, K, N = 33, 17, 25  # intentionally not multiples of block size (16)

    g = _make_matmul_graph(M, K, N)
    fused = _fuse(g)
    kernel = lower_graph(fused)
    source = emit_kernel(kernel)

    # Simple test data: A=1s, B=1s → each C[i,j] should equal K.
    a_data = [1.0] * (M * K)
    b_data = [1.0] * (K * N)

    result = run_kernel(
        kernel=kernel,
        kernel_source=source,
        inputs={"A": a_data, "B": b_data},
        output_name="C",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
    )

    for i, val in enumerate(result):
        assert abs(val - K) < 1e-4, f"C[{i}] = {val}, expected {K}"
