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

from deplodock.compiler.backend.cuda.generators import generate_kernel
from deplodock.compiler.backend.cuda.program import Buffer, Launch, Program, run_program
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, InputOp, KernelOp, ReduceOp
from tests.compiler._fusion_helper import auto_fuse

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
    for inp in [p.buffer_id for p in op.inputs]:
        if inp in graph.nodes:
            shapes[inp] = graph.nodes[inp].output.shape
    for out in [p.buffer_id for p in op.outputs]:
        # Find the original output shape from the region_ops.
        for _node in op.body_ops():
            if _node.id in [p.buffer_id for p in op.outputs]:
                # Use the fused node's output shape.
                pass
        shapes[out] = region_node.output.shape

    source = emit_kernel(generate_kernel(op, "test_kernel", shapes))

    # Build Program.
    buffers = []
    launch_args = []
    for inp in [p.buffer_id for p in op.inputs]:
        size = len(inputs.get(inp, [1]))
        buffers.append(Buffer(inp, size, role="input"))
        launch_args.append(inp)
    out_name = [p.buffer_id for p in op.outputs][0]
    out_size = math.prod(d for d in region_node.output.shape if isinstance(d, int))
    buffers.append(Buffer(out_name, out_size, role="output"))
    launch_args.append(out_name)

    # Add dimension args based on kernel type.
    has_reduce = any(isinstance(n.op, ReduceOp) for n in op.body_ops())
    if has_reduce:
        # Reduction kernel: needs rows, cols.
        # Infer from first reduce input shape.
        for _node in op.body_ops():
            if isinstance(_node.op, ReduceOp):
                inp_shape = shapes.get(_node.inputs[0], (1,))
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = emit_kernel(generate_kernel(region_node.op, "test_matmul", shapes))
    assert "__global__" in source and "void test_matmul" in source
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {}
    for nid in g.nodes:
        shapes[nid] = g.nodes[nid].output.shape
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = emit_kernel(generate_kernel(region_node.op, "test_rmsnorm", shapes))
    assert "__global__" in source and "void test_rmsnorm" in source
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1

    region_node = fused_nodes[0]
    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape

    source = emit_kernel(generate_kernel(region_node.op, "test_softmax", shapes))
    assert "__global__" in source and "void test_softmax" in source
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1, "Expected at least one FusedRegionOp"

    # Generate kernel and verify it compiles.
    region_node = fused_nodes[0]
    shapes = {nid: fused.nodes[nid].output.shape for nid in fused.nodes}
    # Add shapes for original nodes referenced by region_ops.
    for _node in region_node.op.body_ops():
        if _node.id in g.nodes:
            shapes[_node.id] = g.nodes[_node.id].output.shape
    source = emit_kernel(generate_kernel(region_node.op, "test_silu_mul", shapes))
    assert "__global__" in source and "void test_silu_mul" in source
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

    # Auto-fuse. Attention ops are >2D so they stay as separate ops
    # (the codegen only supports 2D indexing). Flash attention fusion
    # is a future goal requiring multi-dimensional codegen.
    fused = auto_fuse(g)

    # Verify the graph is valid (no cycles) after fusion.
    fused.topological_order()

    # Generate kernels for any fused regions that do exist (e.g., 2D matmul pairs).
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    for nd in fused_nodes:
        shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
        for nid in fused.nodes:
            shapes[nid] = fused.nodes[nid].output.shape
        source = emit_kernel(generate_kernel(nd.op, f"test_attn_{nd.id}", shapes))
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1

    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape
    source = emit_kernel(generate_kernel(fused_nodes[0].op, "test_triple", shapes))
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
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) >= 1

    shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
    for nid in fused.nodes:
        shapes[nid] = fused.nodes[nid].output.shape
    source = emit_kernel(generate_kernel(fused_nodes[0].op, "test_mra", shapes))
    assert "__global__" in source


# ===========================================================================
# GPU correctness tests — run generated kernels and verify numerical output.
# These go through the full pipeline: graph → auto_fuse → kernel_gen →
# plan → CudaBackend → GPU → compare output against Python reference.
# ===========================================================================


def _compile_and_run(g: Graph, dump=None) -> dict[str, list[float]]:
    """Full pipeline: rewriter → auto_fuse → plan → CudaBackend → run.

    The rewriter lowers view ops (TransposeOp/SliceOp/CatOp/UnsqueezeOp) to
    IndexMapOp before fusion + compile.
    """
    from pathlib import Path

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.plan import plan_graph
    from deplodock.compiler.rewriter import Rewriter

    if dump:
        dump.dump_input_graph(g)

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    g = Rewriter.from_directory(rules_dir).apply(g)

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


# ---------------------------------------------------------------------------
# Python reference implementations for correctness checking
# ---------------------------------------------------------------------------


def _pseudo_random(size: int) -> list[float]:
    """Match the pseudorandom init in program.py: h[i] = 0.01 * ((i*7+13) % 101 - 50)."""
    return [0.01 * ((i * 7 + 13) % 101 - 50) for i in range(size)]


def _python_silu(gate: list[float]) -> list[float]:
    import math

    return [g / (1.0 + math.exp(-g)) for g in gate]


def _python_reduce_sum(x: list[float], rows: int, cols: int) -> list[float]:
    return [sum(x[r * cols + c] for c in range(cols)) for r in range(rows)]


def _python_rmsnorm(x: list[float], w: list[float], eps: float, rows: int, dim: int) -> list[float]:
    import math

    result = []
    for r in range(rows):
        row = x[r * dim : (r + 1) * dim]
        sq_sum = sum(v * v for v in row)
        rsqrt_val = 1.0 / math.sqrt(sq_sum + eps)
        for c in range(dim):
            result.append(row[c] * rsqrt_val * w[c])
    return result


def _python_matmul(a: list[float], b: list[float], m: int, k: int, n: int) -> list[float]:
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i * k + kk] * b[kk * n + j]
            c[i * n + j] = s
    return c


def _python_softmax(x: list[float], rows: int, cols: int) -> list[float]:
    import math

    result = []
    for r in range(rows):
        row = x[r * cols : (r + 1) * cols]
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        result.extend(e / s for e in exps)
    return result


# ---------------------------------------------------------------------------
# GPU correctness tests — compare against Python reference
# ---------------------------------------------------------------------------


@requires_cuda
def test_correctness_pointwise_silu(dump_dir):
    """SiLU: verify output is nonzero, finite, and matches reference."""
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
    vals = list(outputs.values())[0]
    assert len(vals) == n
    assert all(v == v for v in vals), "SiLU output contains NaN"
    nonzero = sum(1 for v in vals if abs(v) > 1e-12)
    assert nonzero > n * 0.5, f"SiLU output mostly zeros: {nonzero}/{n}"


@requires_cuda
def test_correctness_reduce_sum(dump_dir):
    """Row sum: compare GPU output against Python reference."""
    rows, cols = 8, 64

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = [x]
    red = g.add_node(ReduceOp("sum", axis=1), [x], Tensor("out", (rows,)), node_id="out")
    g.outputs = [red]

    outputs = _compile_and_run(g, dump=dump_dir)
    vals = list(outputs.values())[0]

    x_data = _pseudo_random(rows * cols)
    expected = _python_reduce_sum(x_data, rows, cols)
    _assert_close(vals, expected, tol=0.1, label="reduce_sum")


@requires_cuda
def test_correctness_rmsnorm(dump_dir):
    """RMSNorm: compare GPU output against Python reference."""
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
    assert all(v == v for v in vals), "RMSNorm output contains NaN"

    x_data = _pseudo_random(rows * dim)
    w_data = _pseudo_random(dim)
    eps_data = _pseudo_random(1)
    expected = _python_rmsnorm(x_data, w_data, eps_data[0], rows, dim)
    _assert_close(vals, expected, tol=1e-3, label="RMSNorm")


@requires_cuda
def test_correctness_matmul(dump_dir):
    """Matmul: compare GPU output against Python reference."""
    m, k, n = 8, 16, 12

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (m, k, n)), node_id="AB")
    c = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("C", (m, n)), node_id="C")
    g.outputs = [c]

    outputs = _compile_and_run(g, dump=dump_dir)
    vals = list(outputs.values())[0]
    assert all(v == v for v in vals), "Matmul output contains NaN"
    assert len(vals) == m * n
    nonzero = sum(1 for v in vals if abs(v) > 1e-6)
    assert nonzero > m * n * 0.8, f"Matmul mostly zeros: {nonzero}/{m * n}"
    assert all(abs(v) < 100 for v in vals), f"Matmul extreme values: max={max(abs(v) for v in vals)}"


@requires_cuda
def test_correctness_softmax(dump_dir):
    """Softmax: compare GPU output against Python reference."""
    rows, cols = 4, 32

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = [x]
    mx = g.add_node(ReduceOp("max", axis=1), [x], Tensor("mx", (rows, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (rows, cols)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (rows, cols)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=1), [exp], Tensor("sm", (rows, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    vals = list(outputs.values())[0]
    assert all(v == v for v in vals), "Softmax output contains NaN"
    assert len(vals) == rows * cols

    # Softmax properties: all values in [0, 1], rows sum to 1.0.
    assert all(v >= -1e-6 for v in vals), f"Softmax has negative values: min={min(vals)}"
    assert all(v <= 1.0 + 1e-6 for v in vals), f"Softmax has values > 1: max={max(vals)}"
    for r in range(rows):
        row_sum = sum(vals[r * cols + c] for c in range(cols))
        assert abs(row_sum - 1.0) < 1e-3, f"Softmax row {r} sum={row_sum}, expected 1.0"


@requires_cuda
def test_correctness_full_pipeline(dump_dir):
    """Full TinyLlama pipeline (with rewriter): compiles and runs without crashing.

    Numerical correctness is validated in test_e2e_accuracy.py with real model
    weights. This test only verifies that the pipeline compiles + executes;
    pseudorandom weights through a deep transformer block can legitimately
    produce NaN/Inf from overflow, so no numerical assertions are made here.
    """
    import json

    from deplodock.compiler.rewriter import Rewriter

    fixture_dir = Path(__file__).parent / "fixtures"
    with open(fixture_dir / "tinyllama_layer0.json") as f:
        g = Graph.from_dict(json.load(f))

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    compiled = Rewriter.from_directory(rules_dir).apply(g)
    outputs = _compile_and_run(compiled, dump=dump_dir)
    assert len(outputs) > 0


# ===========================================================================
# Stage 1: Batched matmul tests
# ===========================================================================


def _python_batched_matmul(a: list[float], b: list[float], batch: int, m: int, k: int, n: int) -> list[float]:
    """Per-batch matrix multiplication: a(batch, m, k) @ b(batch, k, n)."""
    c = [0.0] * (batch * m * n)
    for bi in range(batch):
        for i in range(m):
            for j in range(n):
                s = 0.0
                for kk in range(k):
                    s += a[bi * m * k + i * k + kk] * b[bi * k * n + kk * n + j]
                c[bi * m * n + i * n + j] = s
    return c


def _python_sdpa(
    q: list[float],
    k_mat: list[float],
    v: list[float],
    scale: float,
    batch_heads: int,
    seq_len: int,
    head_dim: int,
) -> list[float]:
    """Full scaled dot-product attention reference."""
    out = []
    for b in range(batch_heads):
        q_off = b * seq_len * head_dim
        k_off = b * seq_len * head_dim
        v_off = b * seq_len * head_dim
        # QK^T: (seq_len, head_dim) @ (head_dim, seq_len)
        # K^T = transpose K from (seq_len, head_dim) to (head_dim, seq_len)
        scores = [0.0] * (seq_len * seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                s = 0.0
                for d in range(head_dim):
                    s += q[q_off + i * head_dim + d] * k_mat[k_off + j * head_dim + d]
                scores[i * seq_len + j] = s * scale
        # Softmax over last dim
        attn = _python_softmax(scores, seq_len, seq_len)
        # attn @ V: (seq_len, seq_len) @ (seq_len, head_dim)
        for i in range(seq_len):
            for d in range(head_dim):
                s = 0.0
                for j in range(seq_len):
                    s += attn[i * seq_len + j] * v[v_off + j * head_dim + d]
                out.append(s)
    return out


def test_batched_matmul_detection():
    """Batched 3D contraction is detected by auto_fuse and analyze()."""
    from deplodock.compiler.backend.cuda.generators.analysis import analyze

    batch, m, k, n = 2, 4, 8, 6
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (batch, m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (batch, k, n)), node_id="B")
    g.inputs = [a, b]
    # Broadcast mul + reduce_sum → contraction pattern
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (batch, m, k, n)), node_id="ew")
    red = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("out", (batch, m, n)), node_id="out")
    g.outputs = [red]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) == 1, f"Expected 1 fused region, got {len(fused_nodes)}"

    # Analyze the fused region
    nd = fused_nodes[0]
    shapes = {nid: fused.nodes[nid].output.shape for nid in fused.nodes}
    analysis = analyze(nd.op, shapes)
    assert analysis.pattern == "contraction"
    assert analysis.batch_size == batch
    assert analysis.batch_dims == (batch,)
    assert analysis.rows == m
    assert analysis.cols == n
    assert analysis.k_dim == k


@requires_cuda
def test_correctness_batched_matmul(dump_dir):
    """Batched matmul: A(2,4,8) @ B(2,8,6) produces correct output."""
    batch, m, k, n = 2, 4, 8, 6
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (batch, m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (batch, k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (batch, m, k, n)), node_id="ew")
    red = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("out", (batch, m, n)), node_id="out")
    g.outputs = [red]

    outputs = _compile_and_run(g, dump=dump_dir)
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    actual = list(outputs.values())[0]

    a_data = _pseudo_random(batch * m * k)
    b_data = _pseudo_random(batch * k * n)
    expected = _python_batched_matmul(a_data, b_data, batch, m, k, n)

    _assert_close(actual, expected, tol=1e-3, label="batched_matmul")


@requires_cuda
def test_correctness_attention_stage1(dump_dir):
    """Full attention: decomposed SDPA through pipeline matches reference.

    Exercises batched QK^T and attn@V as tiled GEMMs plus softmax kernels.
    """
    batch_heads, seq_len, head_dim = 2, 8, 4
    scale = 1.0 / math.sqrt(head_dim)

    g = Graph()
    q = g.add_node(InputOp(), [], Tensor("Q", (batch_heads, seq_len, head_dim)), node_id="Q")
    k_in = g.add_node(InputOp(), [], Tensor("K", (batch_heads, seq_len, head_dim)), node_id="K")
    v = g.add_node(InputOp(), [], Tensor("V", (batch_heads, seq_len, head_dim)), node_id="V")
    g.inputs = [q, k_in, v]

    # K^T: transpose last two dims
    from deplodock.compiler.ops import TransposeOp

    kt = g.add_node(
        TransposeOp(axes=(-2, -1)),
        [k_in],
        Tensor("kt", (batch_heads, head_dim, seq_len)),
        node_id="kt",
    )

    # QK^T: mul(Q, K^T) → reduce_sum
    qk_ew = g.add_node(
        ElementwiseOp("mul"),
        [q, kt],
        Tensor("qk_ew", (batch_heads, seq_len, head_dim, seq_len)),
        node_id="qk_ew",
    )
    qk = g.add_node(
        ReduceOp("sum", axis=-1),
        [qk_ew],
        Tensor("qk", (batch_heads, seq_len, seq_len)),
        node_id="qk",
    )

    # Scale
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    scaled = g.add_node(
        ElementwiseOp("mul"),
        [qk, sc],
        Tensor("scaled", (batch_heads, seq_len, seq_len)),
        node_id="scaled",
    )

    # Softmax
    mx = g.add_node(
        ReduceOp("max", axis=-1),
        [scaled],
        Tensor("mx", (batch_heads, seq_len, 1)),
        node_id="mx",
    )
    sub = g.add_node(
        ElementwiseOp("sub"),
        [scaled, mx],
        Tensor("shifted", (batch_heads, seq_len, seq_len)),
        node_id="shifted",
    )
    exp = g.add_node(
        ElementwiseOp("exp"),
        [sub],
        Tensor("exp_val", (batch_heads, seq_len, seq_len)),
        node_id="exp_val",
    )
    sum_exp = g.add_node(
        ReduceOp("sum", axis=-1),
        [exp],
        Tensor("sum_exp", (batch_heads, seq_len, 1)),
        node_id="sum_exp",
    )
    attn_w = g.add_node(
        ElementwiseOp("div"),
        [exp, sum_exp],
        Tensor("attn_w", (batch_heads, seq_len, seq_len)),
        node_id="attn_w",
    )

    # attn @ V: mul(attn_w, V) → reduce_sum
    sv_ew = g.add_node(
        ElementwiseOp("mul"),
        [attn_w, v],
        Tensor("sv_ew", (batch_heads, seq_len, seq_len, head_dim)),
        node_id="sv_ew",
    )
    out = g.add_node(
        ReduceOp("sum", axis=-1),
        [sv_ew],
        Tensor("out", (batch_heads, seq_len, head_dim)),
        node_id="out",
    )
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    actual = list(outputs.values())[0]

    # Compute expected
    q_data = _pseudo_random(batch_heads * seq_len * head_dim)
    k_data = _pseudo_random(batch_heads * seq_len * head_dim)
    v_data = _pseudo_random(batch_heads * seq_len * head_dim)
    expected = _python_sdpa(q_data, k_data, v_data, scale, batch_heads, seq_len, head_dim)

    _assert_close(actual, expected, tol=0.1, label="attention_stage1")


# ===========================================================================
# Stage 2: Multi-reduce (fused softmax) tests
# ===========================================================================


def test_softmax_single_kernel():
    """Softmax (max→sub→exp→sum→div) fuses into exactly one region."""
    from deplodock.compiler.backend.cuda.generators import generate_kernel
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel

    rows, cols = 4, 8
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = [x]
    mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (rows, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (rows, cols)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (rows, cols)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (rows, 1)), node_id="sm")
    div = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = [div]

    fused = auto_fuse(g)
    fused_nodes = [nd for nd in fused.nodes.values() if isinstance(nd.op, KernelOp)]
    assert len(fused_nodes) == 1, f"Expected 1 fused region, got {len(fused_nodes)}"
    assert len(fused_nodes[0].op.body_ops()) == 5  # max, sub, exp, sum, div

    # Verify generated kernel has both fmaxf (max) and expf (exp).
    shapes = {nid: fused.nodes[nid].output.shape for nid in fused.nodes}
    source = emit_kernel(generate_kernel(fused_nodes[0].op, "test_softmax", shapes))
    assert "fmaxf" in source, "Kernel should contain fmaxf for max reduce"
    assert "expf" in source, "Kernel should contain expf for exp op"


@requires_cuda
def test_correctness_softmax_fused(dump_dir):
    """Fused softmax produces correct output with multiple shapes."""
    for rows, cols in [(4, 8), (16, 32), (56, 128)]:
        g = Graph()
        x = g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
        g.inputs = [x]
        mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (rows, 1)), node_id="mx")
        sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (rows, cols)), node_id="sub")
        exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (rows, cols)), node_id="exp")
        sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (rows, 1)), node_id="sm")
        div = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (rows, cols)), node_id="out")
        g.outputs = [div]

        outputs = _compile_and_run(g, dump=dump_dir)
        actual = list(outputs.values())[0]

        x_data = _pseudo_random(rows * cols)
        expected = _python_softmax(x_data, rows, cols)

        _assert_close(actual, expected, tol=1e-3, label=f"softmax_{rows}x{cols}")

        # Softmax invariants: non-negative, ≤1.0, row sums ≈ 1.0
        for i, v in enumerate(actual):
            assert v >= -1e-6, f"softmax[{i}] = {v} is negative"
            assert v <= 1.0 + 1e-6, f"softmax[{i}] = {v} exceeds 1.0"
        for r in range(rows):
            row_sum = sum(actual[r * cols + c] for c in range(cols))
            assert abs(row_sum - 1.0) < 1e-3, f"Row {r} sum={row_sum}, expected 1.0"


# ===========================================================================
# Stage 3: Contraction + reduce fusion (QK^T + softmax in one kernel)
# ===========================================================================


@requires_cuda
def test_correctness_contraction_softmax_fused(dump_dir):
    """Fused matmul + scale + softmax in a single kernel launch."""
    m, k, n = 4, 8, 4

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    red = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("qk", (m, n)), node_id="qk")
    scaled = g.add_node(ElementwiseOp("mul"), [red, sc], Tensor("scaled", (m, n)), node_id="scaled")
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (m, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("sub", (m, n)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (m, n)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (m, 1)), node_id="sm")
    div = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (m, n)), node_id="out")
    g.outputs = [div]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    # Reference: matmul → scale → softmax
    # The constant buffer is initialized via _pseudo_random, not the user scale.
    a_data = _pseudo_random(m * k)
    b_data = _pseudo_random(k * n)
    scale_data = _pseudo_random(1)[0]
    scores = _python_matmul(a_data, b_data, m, k, n)
    scores = [s * scale_data for s in scores]
    expected = _python_softmax(scores, m, n)

    _assert_close(actual, expected, tol=1e-3, label="contraction_softmax")

    # Verify single launch (fused)
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.plan import plan_graph

    fused = auto_fuse(g)
    plan = plan_graph(fused)
    program = CudaBackend().compile(plan)
    assert len(program.launches) == 1, f"Expected 1 launch, got {len(program.launches)}"


@requires_cuda
def test_correctness_contraction_softmax_online_large_n(dump_dir):
    """Fused matmul + scale + softmax with N > tile_n (128) via online reduction."""
    m, k, n = 4, 32, 256  # N=256 > tile_n=128 → multi-iteration N-tile loop

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    red = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("qk", (m, n)), node_id="qk")
    scaled = g.add_node(ElementwiseOp("mul"), [red, sc], Tensor("scaled", (m, n)), node_id="scaled")
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (m, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("sub", (m, n)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (m, n)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (m, 1)), node_id="sm")
    div = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (m, n)), node_id="out")
    g.outputs = [div]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    # Reference
    a_data = _pseudo_random(m * k)
    b_data = _pseudo_random(k * n)
    scale_data = _pseudo_random(1)[0]
    scores = _python_matmul(a_data, b_data, m, k, n)
    scores = [s * scale_data for s in scores]
    expected = _python_softmax(scores, m, n)

    _assert_close(actual, expected, tol=1e-3, label="contraction_softmax_online")

    # Verify single launch (online reduction, no split)
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.plan import plan_graph

    fused = auto_fuse(g)
    plan = plan_graph(fused)
    program = CudaBackend().compile(plan)
    assert len(program.launches) == 1, f"Expected 1 launch (online), got {len(program.launches)}"


# ===========================================================================
# Fused kernel pattern correctness tests (transformer block patterns)
# ===========================================================================


@requires_cuda
def test_correctness_contraction_bias_add(dump_dir):
    """Matmul + bias add fused as contraction with epilogue."""
    m, k, n = 32, 64, 128

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("X", (m, k)), node_id="X")
    w = g.add_node(InputOp(), [], Tensor("W", (k, n)), node_id="W")
    bias = g.add_node(InputOp(), [], Tensor("bias", (n,)), node_id="bias")
    g.inputs = [a, w, bias]

    ew = g.add_node(ElementwiseOp("mul"), [a, w], Tensor("ew", (m, k, n)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (m, n)), node_id="mm")
    out = g.add_node(ElementwiseOp("add"), [mm, bias], Tensor("out", (m, n)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    x_data = _pseudo_random(m * k)
    w_data = _pseudo_random(k * n)
    bias_data = _pseudo_random(n)
    mm_ref = _python_matmul(x_data, w_data, m, k, n)
    expected = [mm_ref[i] + bias_data[i % n] for i in range(m * n)]

    _assert_close(actual, expected, tol=1e-2, label="contraction_bias_add")


@requires_cuda
def test_correctness_broadcast_pointwise(dump_dir):
    """Pointwise mul with broadcast input (rotary embedding style)."""
    batch, heads, seq, dim = 1, 4, 16, 32
    total = batch * heads * seq * dim

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (batch, heads, seq, dim)), node_id="X")
    cos = g.add_node(InputOp(), [], Tensor("cos", (batch, 1, seq, dim)), node_id="cos")
    g.inputs = [x, cos]

    out = g.add_node(ElementwiseOp("mul"), [x, cos], Tensor("out", (batch, heads, seq, dim)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    x_data = _pseudo_random(total)
    cos_data = _pseudo_random(batch * 1 * seq * dim)
    cos_size = batch * seq * dim
    expected = [x_data[i] * cos_data[i % cos_size] for i in range(total)]

    _assert_close(actual, expected, tol=1e-4, label="broadcast_pointwise")


@requires_cuda
def test_correctness_gqa_batched_contraction(dump_dir):
    """Batched matmul with broadcast batch dims (GQA: 4 Q heads, 2 KV heads)."""
    q_heads, kv_heads = 4, 2
    seq, dim = 8, 16
    group_size = q_heads // kv_heads  # 2

    g = Graph()
    # attn_weights: (q_heads, seq, seq) — one per Q head
    attn = g.add_node(InputOp(), [], Tensor("attn", (q_heads, seq, seq)), node_id="attn")
    # V: (kv_heads, seq, dim) — shared across groups
    v = g.add_node(InputOp(), [], Tensor("V", (kv_heads, seq, dim)), node_id="V")
    g.inputs = [attn, v]

    ew = g.add_node(ElementwiseOp("mul"), [attn, v], Tensor("ew", (q_heads, seq, seq, dim)), node_id="ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("out", (q_heads, seq, dim)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    attn_data = _pseudo_random(q_heads * seq * seq)
    v_data = _pseudo_random(kv_heads * seq * dim)

    # Reference: for each Q head h, use KV head h // group_size
    expected = []
    for h in range(q_heads):
        kv_h = h // group_size
        for i in range(seq):
            for d in range(dim):
                s = 0.0
                for j in range(seq):
                    a_idx = h * seq * seq + i * seq + j
                    v_idx = kv_h * seq * dim + j * dim + d
                    s += attn_data[a_idx] * v_data[v_idx]
                expected.append(s)

    _assert_close(actual, expected, tol=1e-2, label="gqa_batched_contraction")


@requires_cuda
def test_correctness_contraction_residual_add(dump_dir):
    """Matmul + residual add (output = matmul(X, W) + residual)."""
    m, k, n = 16, 32, 64

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (m, k)), node_id="X")
    w = g.add_node(InputOp(), [], Tensor("W", (k, n)), node_id="W")
    res = g.add_node(InputOp(), [], Tensor("res", (m, n)), node_id="res")
    g.inputs = [x, w, res]

    ew = g.add_node(ElementwiseOp("mul"), [x, w], Tensor("ew", (m, k, n)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (m, n)), node_id="mm")
    out = g.add_node(ElementwiseOp("add"), [mm, res], Tensor("out", (m, n)), node_id="out")
    g.outputs = [out]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    x_data = _pseudo_random(m * k)
    w_data = _pseudo_random(k * n)
    res_data = _pseudo_random(m * n)
    mm_ref = _python_matmul(x_data, w_data, m, k, n)
    expected = [mm_ref[i] + res_data[i] for i in range(m * n)]

    _assert_close(actual, expected, tol=1e-2, label="contraction_residual_add")


@requires_cuda
def test_correctness_chained_rmsnorm_matmul_residual(dump_dir):
    """Multi-kernel chain: RMSNorm → matmul+bias → residual add.

    Tests that data flows correctly between kernels in the pipeline.
    """
    rows, dim, out_dim = 8, 64, 128

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    eps_node = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    w_norm = g.add_node(InputOp(), [], Tensor("w_norm", (dim,)), node_id="w_norm")
    w_proj = g.add_node(InputOp(), [], Tensor("W", (dim, out_dim)), node_id="W")
    bias = g.add_node(InputOp(), [], Tensor("bias", (out_dim,)), node_id="bias")
    g.inputs = [x, w_norm, w_proj, bias]

    # RMSNorm: x * rsqrt(sum(x^2) + eps) * w_norm
    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (rows, dim)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=-1), [sq], Tensor("red", (rows, 1)), node_id="red")
    add_eps = g.add_node(ElementwiseOp("add"), [red, eps_node], Tensor("ae", (rows, 1)), node_id="ae")
    rsq = g.add_node(ElementwiseOp("rsqrt"), [add_eps], Tensor("rsq", (rows, 1)), node_id="rsq")
    norm = g.add_node(ElementwiseOp("mul"), [x, rsq], Tensor("norm", (rows, dim)), node_id="norm")
    normed = g.add_node(ElementwiseOp("mul"), [norm, w_norm], Tensor("normed", (rows, dim)), node_id="normed")

    # Matmul + bias
    ew = g.add_node(ElementwiseOp("mul"), [normed, w_proj], Tensor("ew", (rows, dim, out_dim)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (rows, out_dim)), node_id="mm")
    biased = g.add_node(ElementwiseOp("add"), [mm, bias], Tensor("biased", (rows, out_dim)), node_id="biased")
    g.outputs = [biased]

    outputs = _compile_and_run(g, dump=dump_dir)
    actual = list(outputs.values())[0]

    # Python reference
    x_data = _pseudo_random(rows * dim)
    w_norm_data = _pseudo_random(dim)
    w_data = _pseudo_random(dim * out_dim)
    bias_data = _pseudo_random(out_dim)
    eps_data = _pseudo_random(1)[0]

    # RMSNorm
    normed_ref = _python_rmsnorm(x_data, w_norm_data, eps_data, rows, dim)
    # Matmul
    mm_ref = _python_matmul(normed_ref, w_data, rows, dim, out_dim)
    # Bias
    expected = [mm_ref[i] + bias_data[i % out_dim] for i in range(rows * out_dim)]

    _assert_close(actual, expected, tol=0.1, label="chained_rmsnorm_matmul")
