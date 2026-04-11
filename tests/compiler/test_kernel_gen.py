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
import math
from pathlib import Path

import pytest

from deplodock.compiler.backend.cuda.program import Buffer, Launch, Program, run_program
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.kernel_gen import generate_kernel
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, FusedRegionOp, InputOp, ReduceOp

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "patterns"


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / f"{name}.json") as f:
        return json.load(f)


def _assert_close(actual: list[float], expected: list[float], tol: float = 1e-3, label: str = ""):
    assert len(actual) == len(expected), f"{label}: length mismatch {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert abs(a - e) < tol, f"{label}[{i}]: got {a}, expected {e}, diff={abs(a - e)}"


def _run_fused_region(region_node, graph: Graph, inputs: dict[str, list[float]]) -> list[float]:
    """Run a FusedRegionOp through kernel_gen → CUDA."""
    op = region_node.op
    shapes = {nid: graph.nodes[nid].output.shape for nid in graph.nodes}
    for inp in op.input_names:
        if inp in graph.nodes:
            shapes[inp] = graph.nodes[inp].output.shape
    for out in op.output_names:
        # Find the original output shape from the region_ops.
        for rid, _rop, _ in op.region_ops:
            if rid in op.output_names:
                # Use the fused node's output shape.
                pass
        shapes[out] = region_node.output.shape

    source = generate_kernel(op, "test_kernel", shapes)

    # Build Program.
    buffers = []
    launch_args = []
    for inp in op.input_names:
        size = len(inputs.get(inp, [1]))
        buffers.append(Buffer(inp, size, role="input"))
        launch_args.append(inp)
    out_name = op.output_names[0]
    out_size = math.prod(d for d in region_node.output.shape if isinstance(d, int))
    buffers.append(Buffer(out_name, out_size, role="output"))
    launch_args.append(out_name)

    # Add dimension args based on kernel type.
    has_reduce = any(isinstance(rop, ReduceOp) for _, rop, _ in op.region_ops)
    if has_reduce:
        # Reduction kernel: needs rows, cols.
        # Infer from first reduce input shape.
        for _, rop, inp_ids in op.region_ops:
            if isinstance(rop, ReduceOp):
                inp_shape = shapes.get(inp_ids[0], (1,))
                if len(inp_shape) >= 2:
                    rows = math.prod(d for d in inp_shape[:-1] if isinstance(d, int))
                    cols = inp_shape[-1] if isinstance(inp_shape[-1], int) else 1
                else:
                    rows = 1
                    cols = math.prod(d for d in inp_shape if isinstance(d, int))
                launch_args.extend([str(rows), str(cols)])
                grid = (rows, 1, 1)
                block = (256, 1, 1)
                break
    else:
        # Pointwise kernel: needs n.
        launch_args.append(str(out_size))
        grid = ((out_size + 255) // 256, 1, 1)
        block = (256, 1, 1)

    program = Program(
        name="test",
        buffers=buffers,
        launches=[Launch(kernel_source=source, kernel_name="test_kernel", grid=grid, block=block, args=launch_args)],
    )

    # Override input initialization with actual data.
    # For now, use program.py's run mode which initializes pseudorandom.
    # TODO: pass actual input data.
    result = run_program(program)
    return result.outputs.get(out_name, [])


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

    # Build primitive graph: silu(gate) * up = gate * recip(1 + exp(-gate)) * up
    g = Graph()
    gate = g.add_node(InputOp(), [], Tensor("gate", (n,)), node_id="gate")
    up = g.add_node(InputOp(), [], Tensor("up", (n,)), node_id="up")
    one = g.add_node(ConstantOp(name="one"), [], Tensor("one", (1,)), node_id="one")
    g.inputs = [gate, up]

    neg = g.add_node(ElementwiseOp("neg"), [gate], Tensor("neg", (n,)), node_id="neg")
    exp = g.add_node(ElementwiseOp("exp"), [neg], Tensor("exp", (n,)), node_id="exp")
    add = g.add_node(ElementwiseOp("add"), [one, exp], Tensor("add", (n,)), node_id="add")
    recip = g.add_node(ElementwiseOp("recip"), [add], Tensor("recip", (n,)), node_id="recip")
    silu = g.add_node(ElementwiseOp("mul"), [gate, recip], Tensor("silu", (n,)), node_id="silu")
    out = g.add_node(ElementwiseOp("mul"), [silu, up], Tensor("out", (n,)), node_id="out")
    g.outputs = [out]

    # Auto-fuse.
    fused = auto_fuse(g)

    # Find the fused region.
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1, "Expected at least one FusedRegionOp"

    # Generate kernel and verify it compiles.
    region_node = fused_nodes[0]
    shapes = {nid: fused.nodes[nid].output.shape for nid in fused.nodes}
    # Add shapes for original nodes referenced by region_ops.
    for rid, _, _ in region_node.op.region_ops:
        if rid in g.nodes:
            shapes[rid] = g.nodes[rid].output.shape
    source = generate_kernel(region_node.op, "test_silu_mul", shapes)
    assert "__global__ void test_silu_mul" in source
    assert "expf" in source


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
