"""Tests for the kernel generator — verifies generated CUDA kernels produce correct output.

Each test loads a pattern fixture (precomputed from PyTorch), builds the
primitive op graph, runs it through auto_fuse + kernel_gen, and compares
the GPU output to the reference values.

Fixtures are in tests/compiler/fixtures/patterns/*.json with:
  inputs: dict of name → flat float list
  output: flat float list (reference from PyTorch)
  dims: dimension values
  params: optional op-specific parameters (eps, scale, etc.)
"""

import json
from pathlib import Path

import pytest

from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "patterns"


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / f"{name}.json") as f:
        return json.load(f)


def _assert_close(actual: list[float], expected: list[float], tol: float = 1e-3, label: str = ""):
    """Assert two float lists are element-wise close."""
    assert len(actual) == len(expected), f"{label}: length mismatch {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert abs(a - e) < tol, f"{label}[{i}]: got {a}, expected {e}, diff={abs(a - e)}"


# ---------------------------------------------------------------------------
# Matmul: C[M,N] = A[M,K] @ B[K,N]
# Primitive decomposition: ReduceOp(sum)(ElementwiseOp(mul)(A, B))
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_matmul():
    """Generated matmul kernel produces correct output."""
    fixture = _load_fixture("matmul")
    # TODO: build primitive graph, run through auto_fuse + kernel_gen
    # For now, verify the fixture is valid
    m, n, k = fixture["dims"]["M"], fixture["dims"]["N"], fixture["dims"]["K"]
    assert len(fixture["inputs"]["A"]) == m * k
    assert len(fixture["inputs"]["B"]) == k * n
    assert len(fixture["output"]) == m * n

    # --- This is the test that will pass once kernel_gen is implemented ---
    # graph = _build_matmul_graph(m, n, k)
    # fused = auto_fuse(graph)
    # plan = plan_graph(fused)
    # backend = CudaBackend()  # uses kernel_gen for FusedRegionOp
    # program = backend.compile(plan)
    # result = backend.run(program)
    # _assert_close(result.outputs["C"], fixture["output"], label="matmul")
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight
# Primitive decomposition: mul(x,x) → reduce_sum → div(N) → add(eps) →
#   rsqrt → mul(x) → mul(weight)
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_rmsnorm():
    """Generated RMSNorm kernel produces correct output."""
    fixture = _load_fixture("rmsnorm")
    rows, dim = fixture["dims"]["rows"], fixture["dims"]["dim"]
    assert len(fixture["inputs"]["x"]) == rows * dim
    assert len(fixture["inputs"]["weight"]) == dim
    assert len(fixture["output"]) == rows * dim

    # graph = _build_rmsnorm_graph(rows, dim, eps)
    # fused = auto_fuse(graph)
    # ... run and compare
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
# Primitive decomposition: reduce_max → sub → exp → reduce_sum → div
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_softmax():
    """Generated softmax kernel produces correct output."""
    fixture = _load_fixture("softmax")
    rows, cols = fixture["dims"]["rows"], fixture["dims"]["cols"]
    assert len(fixture["inputs"]["x"]) == rows * cols
    assert len(fixture["output"]) == rows * cols

    # graph = _build_softmax_graph(rows, cols)
    # fused = auto_fuse(graph)
    # ... run and compare
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# SiLU + Mul: out = silu(gate) * up = gate / (1 + exp(-gate)) * up
# Primitive decomposition: neg → exp → add(1) → recip → mul(gate) → mul(up)
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_silu_mul():
    """Generated SiLU+Mul kernel produces correct output."""
    fixture = _load_fixture("silu_mul")
    n = fixture["dims"]["n"]
    assert len(fixture["inputs"]["gate"]) == n
    assert len(fixture["inputs"]["up"]) == n
    assert len(fixture["output"]) == n

    # graph = _build_silu_mul_graph(n)
    # fused = auto_fuse(graph)
    # ... run and compare
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# Flash Attention: out = softmax(Q @ K^T * scale) @ V
# Primitive decomposition:
#   mul(Q, K^T) → reduce_sum → mul(scale) →
#   reduce_max → sub → exp → reduce_sum → div →
#   mul(softmax, V) → reduce_sum
#
# The kernel generator must discover that this requires a TILED ONLINE
# algorithm (flash attention) — not sequential execution.
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_attention():
    """Generated attention kernel produces correct output via tiled online softmax.

    This is the key test: the kernel generator must automatically discover
    the flash attention algorithm from the primitive op graph. The N×N
    scores matrix must NOT be materialized in global memory.
    """
    fixture = _load_fixture("attention")
    batch = fixture["dims"]["batch"]
    heads = fixture["dims"]["heads"]
    seq_len = fixture["dims"]["seq_len"]
    head_dim = fixture["dims"]["head_dim"]
    total = batch * heads * seq_len * head_dim

    assert len(fixture["inputs"]["Q"]) == total
    assert len(fixture["inputs"]["K"]) == total
    assert len(fixture["inputs"]["V"]) == total
    assert len(fixture["output"]) == total

    # graph = _build_attention_graph(batch, heads, seq_len, head_dim, scale)
    # fused = auto_fuse(graph)
    #
    # # The fused graph should have ONE FusedRegionOp for the entire attention
    # # (QK^T + softmax + @V fused together) because the N×N scores matrix
    # # is the largest intermediate.
    # fused_ops = [n for n in fused.nodes.values() if isinstance(n.op, FusedRegionOp)]
    # assert len(fused_ops) == 1, "Attention should fuse into a single region"
    #
    # plan = plan_graph(fused)
    # backend = CudaBackend()
    # program = backend.compile(plan)
    # result = backend.run(program)
    # _assert_close(result.outputs["out"], fixture["output"], tol=1e-2, label="attention")
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# Triple matmul: Q, K, V = A @ Wq, A @ Wk, A @ Wv
# Three matmuls sharing the same input A.
# The fusion algorithm should discover that A is read 3 times and fuse
# the three matmuls to read A once.
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_triple_matmul():
    """Generated triple matmul reads shared input once."""
    # Small dims for testing
    m, k, nq, nk, nv = 4, 8, 6, 4, 4

    # Reference from PyTorch
    import torch

    torch.manual_seed(42)
    a = torch.randn(m, k)
    wq = torch.randn(k, nq)
    wk = torch.randn(k, nk)
    wv = torch.randn(k, nv)
    _q_ref = (a @ wq).flatten().tolist()
    _k_ref = (a @ wk).flatten().tolist()
    _v_ref = (a @ wv).flatten().tolist()

    # graph = _build_triple_matmul_graph(m, k, nq, nk, nv)
    # fused = auto_fuse(graph)
    # # Should fuse into one region (shared input A)
    # ... run and compare Q, K, V outputs
    pytest.skip("kernel_gen not implemented yet")


# ---------------------------------------------------------------------------
# Matmul + residual add: out = A @ B + residual
# The fusion should combine the matmul epilogue with the add.
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_matmul_residual_add():
    """Generated matmul+add kernel fuses the residual into the store."""
    import torch

    torch.manual_seed(42)
    m, n, k = 4, 6, 8
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    residual = torch.randn(m, n)
    _ref = (a @ b + residual).flatten().tolist()

    # graph = _build_matmul_residual_add_graph(m, n, k)
    # fused = auto_fuse(graph)
    # ... run and compare
    pytest.skip("kernel_gen not implemented yet")
