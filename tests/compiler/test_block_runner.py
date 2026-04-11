"""Tests for block execution via the layered architecture: graph → plan → CudaBackend → GPU."""

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
    with open(FIXTURE_DIR / "tinyllama_layer0.json") as f:
        g = Graph.from_dict(json.load(f))
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    return Rewriter.from_directory(rules_dir).apply(g)


# ---- Planning + compilation tests (no GPU) ----


def test_plan_and_compile_produces_program():
    """plan_graph → CudaBackend.compile produces a Program."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    prog = _backend.compile(plan)

    assert len(prog.launches) > 0
    assert len(prog.buffers) > 0

    input_bufs = [b for b in prog.buffers if b.role == "input"]
    output_bufs = [b for b in prog.buffers if b.role == "output"]
    assert len(input_bufs) == 3
    assert len(output_bufs) == 1


def test_generate_block_source():
    """Generated .cu source has expected structure."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    prog = _backend.compile(plan)
    source = generate_source(prog, mode="benchmark")

    assert "int main()" in source
    assert "PROGRAM_TIME_MS=" in source


# ---- GPU tests ----


@requires_cuda
def test_block_compiles_and_runs():
    """Full block: graph → plan → compile → run on GPU."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    prog = _backend.compile(plan)
    result = _backend.run(prog)

    assert len(result.outputs) > 0


@requires_cuda
def test_block_benchmark_returns_timing():
    """Benchmark mode returns valid timing."""
    compiled = _load_and_compile_fixture()
    plan = plan_graph(compiled)
    prog = _backend.compile(plan)
    result = _backend.benchmark(prog, warmup=2, num_iters=3)

    assert result.time_ms > 0
    assert result.num_launches > 0
