"""Tests for the end-to-end pipeline: Graph → Rewriter → plan_graph → CudaBackend → GPU."""

from pathlib import Path

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.plan import plan_graph
from deplodock.compiler.rewriter import Rewriter

# ---- helpers ----


def _make_matmul_graph(m, k, n):
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (m, k)), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", (m, k, n)), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", (m, n)), node_id="red")
    g.outputs = [red]
    return g


def _load_rewriter():
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    return Rewriter.from_directory(rules_dir)


# ---- tests ----

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def test_compile_graph_fuses_matmul():
    """compile_graph applies fusion rules and produces FusedRegionOp."""
    g = _make_matmul_graph(4, 3, 2)
    rewriter = _load_rewriter()
    from deplodock.compiler.fusion import auto_fuse

    compiled, traces = compile_graph(g, rewriter)
    compiled = auto_fuse(compiled)

    op_types = {type(n.op).__name__ for n in compiled.nodes.values()}
    assert "FusedRegionOp" in op_types

    assert len(traces) > 0


def test_plan_graph_from_matmul():
    """plan_graph produces a plan with one matmul op from a fused graph."""
    g = _make_matmul_graph(4, 3, 2)
    from deplodock.compiler.fusion import auto_fuse

    compiled, _ = compile_graph(g, _load_rewriter())
    compiled = auto_fuse(compiled)
    plan = plan_graph(compiled)

    fused_ops = [op for op in plan.ops if op.op in ("matmul", "fused_region")]
    assert len(fused_ops) >= 1


def test_matmul_k_dimension_inferred_correctly():
    """Non-square matmul: K must be inferred from input shapes, not output."""
    from deplodock.compiler.backend.cuda.backend import _is_matmul_region
    from deplodock.compiler.fusion import auto_fuse

    # A(32, 2048) @ B(2048, 512) → C(32, 512). K=2048, not 32.
    g = _make_matmul_graph(32, 2048, 512)
    compiled, _ = compile_graph(g, _load_rewriter())
    compiled = auto_fuse(compiled)
    plan = plan_graph(compiled)

    fused_ops = [op for op in plan.ops if op.op == "fused_region"]
    assert len(fused_ops) == 1

    is_matmul, m, n, k = _is_matmul_region(fused_ops[0])
    assert is_matmul
    assert m == 32
    assert n == 512
    assert k == 2048, f"K should be 2048 (shared dim), got {k}"


@requires_cuda
def test_pipeline_end_to_end(dump_dir):
    """Full pipeline: graph → fuse → plan → CudaBackend → GPU."""
    g = _make_matmul_graph(4, 3, 2)
    from deplodock.compiler.fusion import auto_fuse

    dump_dir.dump_input_graph(g)
    compiled, traces = compile_graph(g, _load_rewriter())
    dump_dir.dump_passes(traces)
    compiled = auto_fuse(compiled)
    dump_dir.dump_fused_graph(compiled)
    plan = plan_graph(compiled)
    dump_dir.dump_plan(plan)

    backend = CudaBackend()
    program = backend.compile(plan)
    dump_dir.dump_program(program)
    result = backend.run(program)
    dump_dir.dump_result(result)

    # Output should have values (matmul result).
    assert len(result.outputs) == 1
    output_values = list(result.outputs.values())[0]
    assert len(output_values) == 4 * 2  # M * N


@requires_cuda
def test_pipeline_benchmark(dump_dir):
    """Benchmark through canonical pipeline returns timing."""
    g = _make_matmul_graph(64, 64, 64)
    from deplodock.compiler.fusion import auto_fuse

    dump_dir.dump_input_graph(g)
    compiled, traces = compile_graph(g, _load_rewriter())
    dump_dir.dump_passes(traces)
    compiled = auto_fuse(compiled)
    dump_dir.dump_fused_graph(compiled)
    plan = plan_graph(compiled)
    dump_dir.dump_plan(plan)

    backend = CudaBackend()
    program = backend.compile(plan)
    dump_dir.dump_program(program)
    result = backend.benchmark(program, warmup=2, num_iters=3)
    dump_dir.dump_benchmark(result)

    assert result.time_ms > 0
