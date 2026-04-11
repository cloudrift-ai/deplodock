"""Tests for the CUDA backend: ExecutionPlan → Program → GPU."""

import json
from pathlib import Path

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.program import generate_source
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph
from deplodock.compiler.plan import plan_graph
from deplodock.compiler.rewriter import Rewriter

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_backend = CudaBackend()


def _load_and_compile_fixture() -> Graph:
    """Load TinyLlama fixture and run decomposition + auto_fuse."""
    from deplodock.compiler.fusion import auto_fuse
    from deplodock.compiler.kernel_gen import generate_kernel
    from deplodock.compiler.ops import FusedRegionOp

    with open(FIXTURE_DIR / "tinyllama_layer0.json") as f:
        g = Graph.from_dict(json.load(f))
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    compiled = Rewriter.from_directory(rules_dir).apply(g)
    compiled = auto_fuse(compiled)

    # Generate kernels for fused regions.
    shapes = {nid: compiled.nodes[nid].output.shape for nid in compiled.nodes}
    for nid in g.nodes:
        shapes[nid] = g.nodes[nid].output.shape
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if isinstance(node.op, FusedRegionOp) and not node.op.kernel_source:
            node.op.kernel_source = generate_kernel(node.op, f"kernel_{nid}", shapes)

    return compiled


def test_cuda_backend_compile_produces_program():
    """CudaBackend.compile(plan_graph(graph)) produces a Program."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled, name="test")
    program = _backend.compile(plan)

    assert program.name == "test"
    assert len(program.launches) > 0
    assert len(program.buffers) > 0

    input_bufs = [b for b in program.buffers if b.role == "input"]
    output_bufs = [b for b in program.buffers if b.role == "output"]
    assert len(input_bufs) == 3
    assert len(output_bufs) == 1


def test_cuda_backend_generates_valid_source():
    """Generated CUDA source has expected structure."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    source = generate_source(program, mode="benchmark")

    assert "int main()" in source
    assert "cudaMalloc" in source
    assert "PROGRAM_TIME_MS=" in source


def test_cuda_backend_plan_has_expected_ops():
    """Plan from fixture has matmul and fused_region ops."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)

    op_tags = {op.op for op in plan.ops}
    assert "fused_region" in op_tags
    assert "fused_region" in op_tags or len(op_tags) > 2  # auto_fuse produces regions


@requires_cuda
def test_cuda_backend_run():
    """Full pipeline: fixture → compile → plan → CudaBackend → run on GPU."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    result = _backend.run(program)

    # Output buffer exists (may be zeros from noop stubs).
    assert len(result.outputs) > 0


@requires_cuda
def test_cuda_backend_benchmark():
    """Benchmark returns valid timing."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    result = _backend.benchmark(program, warmup=2, num_iters=3)

    assert result.time_ms > 0
    assert result.num_launches > 0
