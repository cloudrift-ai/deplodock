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

from deplodock.compiler.backend.cuda.kernel_gen import generate_kernel
from deplodock.compiler.backend.cuda.program import Buffer, Launch, Program, run_program
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
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
    """Generated matmul kernel has correct reduction structure."""
    fixture = _load_fixture("matmul")
    m, n, k = fixture["dims"]["M"], fixture["dims"]["N"], fixture["dims"]["K"]

    # Build primitive graph: C = reduce_sum(mul(A, B))
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (m, k, n)), node_id="AB")
    red = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("C", (m, n)), node_id="C")
    g.outputs = [red]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = generate_kernel(region_node.op, "test_matmul", shapes)
    assert "__global__ void test_matmul" in source
    # Should have accumulation loop (reduction)
    assert "acc_" in source or "+=" in source


# ---------------------------------------------------------------------------
# RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight
# Primitive decomposition: mul(x,x) → reduce_sum → div(N) → add(eps) →
#   rsqrt → mul(x) → mul(weight)
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_rmsnorm():
    """Generated RMSNorm kernel source has correct structure."""
    fixture = _load_fixture("rmsnorm")
    rows, dim = fixture["dims"]["rows"], fixture["dims"]["dim"]
    _eps_val = fixture["params"]["eps"]

    # Build primitive graph: mul(x,x) → sum → add(eps) → rsqrt → mul(x) → mul(weight)
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, dim)), node_id="x")
    w = g.add_node(InputOp(), [], Tensor("weight", (dim,)), node_id="weight")
    eps = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    g.inputs = [x, w]

    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (rows, dim)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=1), [sq], Tensor("sum_sq", (rows, 1)), node_id="sum_sq")
    add_eps = g.add_node(ElementwiseOp("add"), [red, eps], Tensor("var", (rows, 1)), node_id="var")
    rsqrt = g.add_node(ElementwiseOp("rsqrt"), [add_eps], Tensor("rsqrt_val", (rows, 1)), node_id="rsqrt_val")
    norm = g.add_node(ElementwiseOp("mul"), [x, rsqrt], Tensor("norm", (rows, dim)), node_id="norm")
    out = g.add_node(ElementwiseOp("mul"), [norm, w], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {}
    for nid in g.nodes:
        shapes[nid] = g.nodes[nid].output.shape
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = generate_kernel(region_node.op, "test_rmsnorm", shapes)
    assert "__global__ void test_rmsnorm" in source
    assert "rsqrtf" in source
    assert "__shfl_down_sync" in source  # has warp-level reduction


# ---------------------------------------------------------------------------
# Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
# Primitive decomposition: reduce_max → sub → exp → reduce_sum → div
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_softmax():
    """Generated softmax kernel has correct structure with two reductions."""
    fixture = _load_fixture("softmax")
    rows, cols = fixture["dims"]["rows"], fixture["dims"]["cols"]

    # Build primitive graph: max → sub → exp → sum → div
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = [x]

    mx = g.add_node(ReduceOp("max", axis=1), [x], Tensor("mx", (rows, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("shifted", (rows, cols)), node_id="shifted")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp_val", (rows, cols)), node_id="exp_val")
    sum_exp = g.add_node(ReduceOp("sum", axis=1), [exp], Tensor("sum_exp", (rows, 1)), node_id="sum_exp")
    out = g.add_node(ElementwiseOp("div"), [exp, sum_exp], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = generate_kernel(region_node.op, "test_softmax", shapes)
    assert "__global__ void test_softmax" in source
    assert "expf" in source
    assert "fmaxf" in source  # has max reduction


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
    """Attention primitive graph fuses and generates a kernel.

    The full flash attention (tiled online softmax) is a future goal.
    For now, verify that auto_fuse groups the attention ops into regions
    and the kernel generator produces valid CUDA for each region.
    """
    fixture = _load_fixture("attention")
    batch = fixture["dims"]["batch"]
    heads = fixture["dims"]["heads"]
    seq_len = fixture["dims"]["seq_len"]
    head_dim = fixture["dims"]["head_dim"]
    _scale = fixture["params"]["scale"]
    bh = batch * heads

    # Build primitive graph: QK^T → scale → softmax → @V
    g = Graph()
    q = g.add_node(InputOp(), [], Tensor("Q", (bh, seq_len, head_dim)), node_id="Q")
    k = g.add_node(InputOp(), [], Tensor("K", (bh, seq_len, head_dim)), node_id="K")
    v = g.add_node(InputOp(), [], Tensor("V", (bh, seq_len, head_dim)), node_id="V")
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [q, k, v]

    # QK^T: mul(Q, K) → reduce_sum
    qk_ew = g.add_node(ElementwiseOp("mul"), [q, k], Tensor("qk_ew", (bh, seq_len, seq_len, head_dim)), node_id="qk_ew")
    qk = g.add_node(ReduceOp("sum", axis=-1), [qk_ew], Tensor("qk", (bh, seq_len, seq_len)), node_id="qk")
    scaled = g.add_node(ElementwiseOp("mul"), [qk, sc], Tensor("scaled", (bh, seq_len, seq_len)), node_id="scaled")

    # Softmax
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (bh, seq_len, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("shifted", (bh, seq_len, seq_len)), node_id="shifted")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp_val", (bh, seq_len, seq_len)), node_id="exp_val")
    sum_exp = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sum_exp", (bh, seq_len, 1)), node_id="sum_exp")
    attn_w = g.add_node(ElementwiseOp("div"), [exp, sum_exp], Tensor("attn_w", (bh, seq_len, seq_len)), node_id="attn_w")

    # Scores @ V: mul(attn_w, V) → reduce_sum
    sv_ew = g.add_node(ElementwiseOp("mul"), [attn_w, v], Tensor("sv_ew", (bh, seq_len, seq_len, head_dim)), node_id="sv_ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [sv_ew], Tensor("out", (bh, seq_len, head_dim)), node_id="out")
    g.outputs = [out]

    # Auto-fuse.
    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1, "Expected at least one FusedRegionOp for attention"

    # Generate kernel for each region.
    for nd in fused_nodes:
        shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
        for nid in fused.nodes:
            shapes[nid] = fused.nodes[nid].output.shape
        source = generate_kernel(nd.op, f"test_attn_{nd.id}", shapes)
        assert "__global__" in source


# ---------------------------------------------------------------------------
# Triple matmul: Q, K, V = A @ Wq, A @ Wk, A @ Wv
# Three matmuls sharing the same input A.
# The fusion algorithm should discover that A is read 3 times and fuse
# the three matmuls to read A once.
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_triple_matmul():
    """Three matmuls sharing input A fuse into regions."""
    m, k, nq = 4, 8, 6

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    wq = g.add_node(InputOp(), [], Tensor("Wq", (k, nq)), node_id="Wq")
    g.inputs = [a, wq]

    ew = g.add_node(ElementwiseOp("mul"), [a, wq], Tensor("ew", (m, k, nq)), node_id="ew")
    red = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("Q", (m, nq)), node_id="Q")
    g.outputs = [red]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1

    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape
    source = generate_kernel(fused_nodes[0].op, "test_triple", shapes)
    assert "__global__" in source


# ---------------------------------------------------------------------------
# Matmul + residual add: out = A @ B + residual
# The fusion should combine the matmul epilogue with the add.
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_gen_matmul_residual_add():
    """Matmul + residual add fuses into one region."""
    m, n, k = 4, 6, 8

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    res = g.add_node(InputOp(), [], Tensor("res", (m, n)), node_id="res")
    g.inputs = [a, b, res]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (m, k, n)), node_id="AB")
    red = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("mm", (m, n)), node_id="mm")
    out = g.add_node(ElementwiseOp("add"), [red, res], Tensor("out", (m, n)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, FusedRegionOp)]
    assert len(fused_nodes) >= 1

    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape
    source = generate_kernel(fused_nodes[0].op, "test_mra", shapes)
    assert "__global__" in source


# ===========================================================================
# GPU correctness tests — run generated kernels and verify numerical output.
# These go through the full pipeline: graph → auto_fuse → kernel_gen →
# plan → CudaBackend → GPU → compare output against Python reference.
# ===========================================================================


def _compile_and_run(g: Graph, dump=None) -> dict[str, list[float]]:
    """Full pipeline: auto_fuse → plan → CudaBackend (auto-generates kernels) → run."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.plan import plan_graph

    if dump:
        dump.dump_input_graph(g)

    fused = auto_fuse(g)

    if dump:
        dump.dump_fused_graph(fused)

    plan = plan_graph(fused)
    backend = CudaBackend()
    program = backend.compile(plan)

    if dump:
        dump.dump_plan(plan)
        dump.dump_program(program)

    result = backend.run(program)

    if dump:
        dump.dump_result(result)

    return result.outputs


@requires_cuda
def test_correctness_pointwise_silu(dump_dir):
    """SiLU pointwise chain: verify output is nonzero and finite."""
    n = 256

    g = Graph()
    gate = g.add_node(InputOp(), [], Tensor("gate", (n,)), node_id="gate")
    one = g.add_node(ConstantOp(name="one"), [], Tensor("one", (1,)), node_id="one")
    g.inputs = [gate]

    neg = g.add_node(ElementwiseOp("neg"), [gate], Tensor("neg", (n,)), node_id="neg")
    exp = g.add_node(ElementwiseOp("exp"), [neg], Tensor("exp", (n,)), node_id="exp")
    add = g.add_node(ElementwiseOp("add"), [one, exp], Tensor("add", (n,)), node_id="add")
    recip = g.add_node(ElementwiseOp("recip"), [add], Tensor("recip", (n,)), node_id="recip")
    out = g.add_node(ElementwiseOp("mul"), [gate, recip], Tensor("out", (n,)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    assert len(outputs) == 1
    vals = list(outputs.values())[0]
    assert len(vals) == n
    # SiLU output should be nonzero for nonzero input.
    nonzero = sum(1 for v in vals if abs(v) > 1e-12)
    assert nonzero > n * 0.5, f"SiLU output mostly zeros: {nonzero}/{n} nonzero"
    assert all(v == v for v in vals), "Output contains NaN"


@requires_cuda
def test_correctness_reduce_sum(dump_dir):
    """Reduction kernel: sum of rows. Compare against Python reference."""
    rows, cols = 8, 64

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = [x]
    red = g.add_node(ReduceOp("sum", axis=1), [x], Tensor("out", (rows,)), node_id="out")
    g.outputs = [red]

    outputs = _compile_and_run(g, dump=dump_dir)
    vals = list(outputs.values())[0]
    assert len(vals) == rows

    # With pseudorandom init: h[i] = 0.01 * ((i*7+13) % 101 - 50)
    # Compute expected row sums.
    expected = []
    for r in range(rows):
        s = 0.0
        for c in range(cols):
            idx = r * cols + c
            s += 0.01 * ((idx * 7 + 13) % 101 - 50)
        expected.append(s)

    for i, (got, exp) in enumerate(zip(vals, expected, strict=True)):
        assert abs(got - exp) < 0.1, f"Row {i}: got {got}, expected {exp}"


@requires_cuda
def test_correctness_rmsnorm(dump_dir):
    """RMSNorm through full pipeline: verify output is nonzero and finite."""
    rows, dim = 4, 128

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, dim)), node_id="x")
    w = g.add_node(InputOp(), [], Tensor("weight", (dim,)), node_id="weight")
    eps = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    g.inputs = [x, w]

    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (rows, dim)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=1), [sq], Tensor("sum_sq", (rows, 1)), node_id="sum_sq")
    add_eps = g.add_node(ElementwiseOp("add"), [red, eps], Tensor("var", (rows, 1)), node_id="var")
    rsqrt = g.add_node(ElementwiseOp("rsqrt"), [add_eps], Tensor("rsqrt_val", (rows, 1)), node_id="rsqrt_val")
    norm = g.add_node(ElementwiseOp("mul"), [x, rsqrt], Tensor("norm", (rows, dim)), node_id="norm")
    out = g.add_node(ElementwiseOp("mul"), [norm, w], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    vals = list(outputs.values())[0]
    assert len(vals) == rows * dim
    nonzero = sum(1 for v in vals if abs(v) > 1e-12)
    assert nonzero > rows * dim * 0.5, f"RMSNorm output mostly zeros: {nonzero}/{rows * dim}"
    assert all(v == v for v in vals), "Output contains NaN"
    # RMSNorm normalizes — values should be bounded.
    assert all(abs(v) < 100 for v in vals), f"RMSNorm output has extreme values: max={max(abs(v) for v in vals)}"
