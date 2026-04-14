"""Tests for CUDA backend: codegen, lowering, and end-to-end GPU execution."""

import pytest

from deplodock.compiler.backend.cuda.generators import analyze, lower_tiled
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc, run_kernel
from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel
from deplodock.compiler.backend.ir.kernel_ir import (
    AugAssign,
    BinOp,
    Builtin,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    Var,
    VarDecl,
)
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, InputOp, ReduceOp

# ---- helpers ----


def _make_matmul_graph(m=None, k=None, n=None, **matmul_hints):
    """Build a matmul graph with optional hints. The general interface."""
    m_dim = m if m is not None else "M"
    k_dim = k if k is not None else "K"
    n_dim = n if n is not None else "N"
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m_dim, k_dim)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k_dim, n_dim)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("AB", (m_dim, k_dim, n_dim)), node_id="ew")
    c = g.add_node(ReduceOp(fn="sum", axis=1), [ew], Tensor("C", (m_dim, n_dim)), node_id="C")
    g.outputs = [c]
    for key, val in matmul_hints.items():
        g.hints.set(f"cuda.matmul.{key}", val)
    return g


def _lower_matmul(m=None, k=None, n=None, **matmul_hints):
    """Build graph, set hints, lower via analyze → lower_tiled."""
    g = _make_matmul_graph(m, k, n, **matmul_hints)
    out_node = g.nodes[g.outputs[0]]
    ew_node = g.nodes[out_node.inputs[0]]
    input_a = g.nodes[ew_node.inputs[0]]
    input_b = g.nodes[ew_node.inputs[1]]
    a_name, b_name, c_name = input_a.output.name, input_b.output.name, out_node.output.name
    region = FusedRegionOp(
        region_ops=[(ew_node.id, ew_node.op, [a_name, b_name]), (out_node.id, out_node.op, [ew_node.id])],
        input_names=[a_name, b_name],
        output_names=[c_name],
    )
    shapes = {
        a_name: input_a.output.shape,
        b_name: input_b.output.shape,
        ew_node.id: ew_node.output.shape,
        c_name: out_node.output.shape,
    }
    strategy = matmul_hints.get("strategy", "naive")
    kernel_def, _loop_prog, _sched = lower_tiled(region, "fused_matmul", shapes, analyze(region, shapes), strategy=strategy)
    return kernel_def


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
            VarDecl("int", "i", BinOp("+", BinOp("*", Builtin("blockIdx.x"), Builtin("blockDim.x")), Builtin("threadIdx.x"))),
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
    """Emit a matmul kernel from lowering and verify structure."""
    kernel = _lower_matmul(4, 3, 2)
    source = emit_kernel(kernel)

    assert "__global__" in source
    assert "void fused_matmul" in source
    assert "A" in source and "B" in source and "C" in source
    assert "int M" in source
    assert "int N" in source
    assert "int K" in source
    # Coarsened: 8×4 outputs per thread with register accumulators
    assert "c00" in source  # first accumulator
    assert "for (int k = 0; k < K; k++)" in source
    assert "A[" in source
    assert "B[" in source


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
    """Lower a matmul and check the KernelDef structure."""
    kernel = _lower_matmul()

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
    """Full pipeline: lower → codegen → compile → run → verify."""
    M, K, N = 4, 3, 2

    kernel = _lower_matmul()
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

    kernel = _lower_matmul()
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
    """run_benchmark raises RuntimeError when the kernel launch OOMs.

    Regression: requesting an enormous matmul (65536^2 × batch=8 ≈ 384 GB)
    OOMs at allocation time. The bench binary must exit non-zero and
    run_benchmark must raise, not silently report 0 ms.
    """
    from deplodock.compiler.backend.cuda.runner import run_benchmark
    from deplodock.compiler.backend.cuda.tuning import default_matmul_strategy_map

    strategy_map, _ = default_matmul_strategy_map()
    hints = {**strategy_map[0][1], "batch_count": 8, "k_splits": 1}
    kernel = _lower_matmul(**hints)
    source = emit_kernel(kernel)

    huge = {"M": 65536, "N": 65536, "K": 65536, "batch": 8}

    with pytest.raises(RuntimeError):
        run_benchmark(kernel=kernel, kernel_source=source, dim_args=huge, num_iterations=2)


# All matmul lowering strategies that lower_matmul() recognizes.
# Each entry: (strategy name, matmul hint overrides). The hints are the
# minimum settings each strategy needs to compile — many have hardcoded
# assumptions about block / coarsening dims and won't accept arbitrary values.
ALL_STRATEGIES: list[tuple[str, dict]] = [
    ("naive", {}),
    ("smem", {"block_k": 32, "thread_m": 4}),
    ("tma_db", {"block_k": 32, "thread_m": 8}),
    ("tma_db_tf32", {}),
    ("tma_db_fma_tf32", {}),
]


@pytest.mark.parametrize("strategy_name,overrides", ALL_STRATEGIES, ids=[s[0] for s in ALL_STRATEGIES])
def test_lower_every_strategy(strategy_name, overrides):
    """Every strategy listed in lower_matmul must produce a valid KernelDef.

    Pure lowering — no nvcc, no GPU. Catches dispatcher / template errors after
    refactors like the `block_m`/`block_n` → `threads_y`/`threads_x` rename.
    """
    kernel = _lower_matmul(strategy=strategy_name, **overrides)
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
        ("smem", {"block_k": 32, "thread_m": 4}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_run_every_strategy_correctness(strategy_name, overrides):
    """Every non-TMA strategy must compile, run, and produce numerically-correct
    results. TMA is excluded here because it requires `cuTensorMapEncodeTiled`
    setup that the lightweight `run_kernel` helper doesn't perform — it's
    covered separately by the bench harness in `run_benchmark`.
    """
    M, K, N = 32, 32, 32
    kernel = _lower_matmul(strategy=strategy_name, **overrides)
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
    from deplodock.compiler.backend.cuda.runner import run_benchmark

    M = N = K = 256  # multiple of TMA tile (224x128 → use 256 to keep things simple)
    kernel = _lower_matmul(strategy="tma_db", block_k=32, thread_m=8)
    source = emit_kernel(kernel)
    result = run_benchmark(
        kernel=kernel,
        kernel_source=source,
        dim_args={"M": M, "N": N, "K": K, "k_splits": 1},
        num_iterations=2,
    )
    assert result.kernel_time_ms > 0


@requires_cuda
@pytest.mark.parametrize(
    "strategy_name,overrides",
    [
        ("naive", {}),
        ("smem", {"block_k": 32, "thread_m": 4}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_rectangular_correctness(strategy_name, overrides):
    """Non-square matrix (M=4, K=3, N=2), verified element-by-element against reference."""
    M, K, N = 4, 3, 2
    kernel = _lower_matmul(strategy=strategy_name, **overrides)
    source = emit_kernel(kernel)

    a_data = [float(i + 1) for i in range(M * K)]  # 1..12
    b_data = [float(i + 1) for i in range(K * N)]  # 1..6

    result = run_kernel(
        kernel=kernel,
        kernel_source=source,
        inputs={"A": a_data, "B": b_data},
        output_name="C",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
    )

    expected = _python_matmul(a_data, b_data, M, K, N)
    for idx, (got, exp) in enumerate(zip(result.output, expected, strict=True)):
        assert abs(got - exp) < 1e-4, f"{strategy_name} [{idx}] got {got}, expected {exp}"


# ---- Matmul + epilogue GPU tests ----


def _lower_matmul_with_epilogue(epilogue_ops, extra_inputs, extra_shapes):
    """Build a contraction region with epilogue ops, lower via naive strategy."""
    M, K, N = 4, 3, 2
    region_ops = [
        ("ew", ElementwiseOp(fn="mul"), ["A", "B"]),
        ("red", ReduceOp(fn="sum", axis=1), ["ew"]),
        *epilogue_ops,
    ]
    input_names = ["A", "B"] + list(extra_inputs)
    # Use the last epilogue op's node_id as the output.
    out_id = epilogue_ops[-1][0]
    region = FusedRegionOp(
        region_ops=region_ops,
        input_names=input_names,
        output_names=[out_id],
    )
    shapes = {
        "A": (M, K),
        "B": (K, N),
        "ew": (M, K, N),
        "red": (M, N),
        out_id: (M, N),
        **extra_shapes,
    }
    # Add shapes for intermediate epilogue ops (all (M, N)).
    for nid, _, _ in epilogue_ops[:-1]:
        shapes[nid] = (M, N)

    analysis = analyze(region, shapes)
    kernel_def, _loop_prog, _sched = lower_tiled(region, "matmul_epi", shapes, analysis, strategy="naive")
    return kernel_def, M, K, N


@requires_cuda
def test_matmul_bias_end_to_end():
    """Matmul + bias add on GPU, verified element-by-element against reference."""
    M, K, N = 4, 3, 2
    epilogue_ops = [("ba", ElementwiseOp("add"), ["red", "bias"])]
    kernel, _, _, _ = _lower_matmul_with_epilogue(
        epilogue_ops,
        ["bias"],
        {"bias": (N,)},
    )
    source = emit_kernel(kernel)

    a_data = [float(i + 1) for i in range(M * K)]  # 1..12
    b_data = [float(i + 1) for i in range(K * N)]  # 1..6
    bias_data = [10.0, 20.0]

    result = run_kernel(
        kernel=kernel,
        kernel_source=source,
        inputs={"A": a_data, "B": b_data, "bias": bias_data},
        output_name="ba",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
    )

    # Reference: C = A @ B + bias
    expected = _python_matmul(a_data, b_data, M, K, N)
    for i in range(M):
        for j in range(N):
            expected[i * N + j] += bias_data[j]

    assert len(result.output) == len(expected)
    for idx, (got, exp) in enumerate(zip(result.output, expected, strict=True)):
        assert abs(got - exp) < 1e-4, f"[{idx}] got {got}, expected {exp}"


@requires_cuda
def test_matmul_bias_relu_end_to_end():
    """Matmul + bias add + ReLU on GPU, verified element-by-element against reference."""
    M, K, N = 4, 3, 2
    epilogue_ops = [
        ("ba", ElementwiseOp("add"), ["red", "bias"]),
        ("out", ElementwiseOp("relu"), ["ba"]),
    ]
    kernel, _, _, _ = _lower_matmul_with_epilogue(
        epilogue_ops,
        ["bias"],
        {"bias": (N,)},
    )
    source = emit_kernel(kernel)

    a_data = [float(i + 1) for i in range(M * K)]  # 1..12
    b_data = [float(i + 1) for i in range(K * N)]  # 1..6
    # Large negative bias on col 0 to trigger ReLU clipping.
    bias_data = [-1000.0, 20.0]

    result = run_kernel(
        kernel=kernel,
        kernel_source=source,
        inputs={"A": a_data, "B": b_data, "bias": bias_data},
        output_name="out",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
    )

    # Reference: C = relu(A @ B + bias)
    expected = _python_matmul(a_data, b_data, M, K, N)
    for i in range(M):
        for j in range(N):
            expected[i * N + j] += bias_data[j]
            expected[i * N + j] = max(0.0, expected[i * N + j])

    assert len(result.output) == len(expected)
    for idx, (got, exp) in enumerate(zip(result.output, expected, strict=True)):
        assert abs(got - exp) < 1e-4, f"[{idx}] got {got}, expected {exp}"
    # Verify ReLU clipping actually occurred (col 0 should be 0).
    for i in range(M):
        assert result.output[i * N + 0] == 0.0, f"Row {i} col 0 should be 0 (ReLU clipped)"
