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
    assert "__global__" in source
    assert "void test_kernel" in source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "float val = 1.0f;" in source


def test_emit_matmul_kernel():
    """Emit a matmul kernel from a lowered graph and verify structure."""
    g = _make_matmul_graph(4, 3, 2)
    fused = _fuse(g)
    kernel = lower_graph(fused)
    source = emit_kernel(kernel)

    assert "__global__" in source
    assert "void fused_matmul" in source
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
    assert len(result.output) == len(expected)
    for got, exp in zip(result.output, expected, strict=True):
        assert abs(got - exp) < 1e-4, f"Mismatch: got {got}, expected {exp}"
    assert result.kernel_time_ms is not None
    assert result.kernel_time_ms >= 0


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

    for i, val in enumerate(result.output):
        assert abs(val - K) < 1e-4, f"C[{i}] = {val}, expected {K}"


@requires_cuda
def test_run_benchmark_surfaces_cuda_oom():
    """Bench harness must NOT silently report 0.000 ms when the launch fails.

    Regression for the bug where requesting an enormous matmul (here 65536^2 with
    batch=8 ≈ 100 GB of FP32) would OOM at allocation time, the runner would
    catch the failure inside the generated bench binary, and the Python wrapper
    would happily produce a row with kernel_time_ms=0 and infinite GFLOPS.

    With the new `cudaPeekAtLastError()` checks the bench binary exits non-zero
    and `run_benchmark` raises `RuntimeError`. This test asserts that path.
    """

    from deplodock.compiler.benchmark import (
        run_adaptive_benchmark_suite,
    )
    from deplodock.compiler.cuda.tuning import default_matmul_strategy_map
    from deplodock.compiler.ir import Graph as _Graph
    from deplodock.compiler.ir import Tensor as _Tensor
    from deplodock.compiler.ops import (
        FusedReduceElementwiseOp as _FRE,
    )
    from deplodock.compiler.ops import (
        InputOp as _Input,
    )
    from deplodock.compiler.rewriter import Rewriter as _Rewriter

    g = _Graph()
    a = g.add_node(_Input(), [], _Tensor("A", ("M", "K")))
    b = g.add_node(_Input(), [], _Tensor("B", ("K", "N")))
    c = g.add_node(_FRE("sum", "mul", 1), [a, b], _Tensor("C", ("M", "N")))
    g.inputs = [a, b]
    g.outputs = [c]

    strategy_map, _ = default_matmul_strategy_map()
    # Pick a size that's guaranteed to OOM on any consumer GPU: 65536^2 FP32
    # is 16 GB per matrix; ×3 matrices ×8 batches = 384 GB.
    huge = [{"M": 65536, "N": 65536, "K": 65536}]
    import dataclasses

    huge_map = [(t, dataclasses.replace(c, batch_count=8, k_splits=1)) for t, c in strategy_map]

    suite = run_adaptive_benchmark_suite(
        graph=g,
        rewriter=_Rewriter(),
        strategy_map=huge_map,
        sizes=huge,
        num_iterations=2,
        description="oom regression",
    )
    # The runner should record an error rather than producing a "successful"
    # 0.000 ms row.
    assert suite.error is not None, (
        f"Expected the OOM to surface as suite.error, but got results: {[(r.dimensions, r.kernel_time_ms) for r in suite.results]}"
    )
    # And no successful row should have been recorded.
    assert all(r.kernel_time_ms > 0 for r in suite.results), "Bench produced a 0-ms row — cudaGetLastError check is not working"


# All matmul lowering strategies that the dispatcher in lower_graph() recognizes.
# Each entry: (strategy name, MatmulConfig overrides). The configs are the
# minimum settings each strategy needs to compile — many have hardcoded
# assumptions about block / coarsening dims and won't accept arbitrary values.
ALL_STRATEGIES: list[tuple[str, dict]] = [
    ("naive", {}),
    ("tma_db", {"block_k": 32, "thread_m": 8}),
    ("tma_db_tf32", {}),
    ("tma_db_fma_tf32", {}),
]


@pytest.mark.parametrize("strategy_name,overrides", ALL_STRATEGIES, ids=[s[0] for s in ALL_STRATEGIES])
def test_lower_every_strategy(strategy_name, overrides):
    """Every strategy listed in `lower_graph` must produce a valid KernelDef.

    Pure lowering — no nvcc, no GPU. Catches dispatcher / template errors after
    refactors like the `block_m`/`block_n` → `threads_y`/`threads_x` rename.
    """
    from deplodock.compiler.cuda.lower import MatmulConfig

    # Use a size that's a multiple of common tile factors so any per-strategy
    # divisibility assertion passes.
    M, K, N = 64, 64, 64

    g = _make_matmul_graph(M, K, N)
    fused = _fuse(g)
    cfg = MatmulConfig(strategy=strategy_name, **overrides)
    kernel = lower_graph(fused, config=cfg)
    source = emit_kernel(kernel)
    assert kernel.name
    assert source
    # Sanity: the launched block dim should match what the strategy advertises.
    bx, by, _bz = kernel.block_size
    assert bx > 0 and by > 0


@requires_cuda
@pytest.mark.parametrize(
    "strategy_name,overrides",
    [
        ("naive", {}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_run_every_strategy_correctness(strategy_name, overrides):
    """Every non-TMA strategy must compile, run, and produce numerically-correct
    results. TMA is excluded here because it requires `cuTensorMapEncodeTiled`
    setup that the lightweight `run_kernel` helper doesn't perform — it's
    covered separately by the bench harness in `run_benchmark`.
    """
    from deplodock.compiler.cuda.lower import MatmulConfig

    M, K, N = 32, 32, 32
    g = _make_matmul_graph(M, K, N)
    fused = _fuse(g)
    cfg = MatmulConfig(strategy=strategy_name, **overrides)
    kernel = lower_graph(fused, config=cfg)
    source = emit_kernel(kernel)

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
    for i, val in enumerate(result.output):
        assert abs(val - K) < 1e-3, f"{strategy_name}: C[{i}] = {val}, expected {K}"


@requires_cuda
def test_run_tma_db_strategy_via_bench():
    """TMA needs the bench harness's TMA descriptor setup; verify it via run_benchmark."""
    from deplodock.compiler.cuda.lower import MatmulConfig
    from deplodock.compiler.cuda.runner import run_benchmark

    cfg = MatmulConfig(strategy="tma_db", block_k=32, thread_m=8)
    M = N = K = 256  # multiple of TMA tile (224x128 → use 256 to keep things simple)
    g = _make_matmul_graph(M, K, N)
    fused = _fuse(g)
    kernel = lower_graph(fused, config=cfg)
    source = emit_kernel(kernel)
    result = run_benchmark(
        kernel=kernel,
        kernel_source=source,
        dim_args={"M": M, "N": N, "K": K, "k_splits": 1},
        num_iterations=2,
    )
    assert result.kernel_time_ms > 0
