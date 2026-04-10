"""Tests for matmul lowering to Program abstraction."""

import pytest

from deplodock.compiler.backend.cuda.lower import MatmulConfig, lower_matmul_to_program
from deplodock.compiler.backend.cuda.program import benchmark_program, generate_source, run_program
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _fuse(g):
    """Apply fusion rules to convert Reduce+Elementwise → MatmulOp."""
    from pathlib import Path

    from deplodock.compiler.rewriter import Rewriter

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    return rewriter.apply(g)


def _make_matmul_graph():
    """C = reduce_sum(elementwise_mul(A, B))."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("K", "N")), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", ("M", "K", "N")), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", ("M", "N")), node_id="red")
    g.outputs = [red]
    return g


# ---- Codegen tests ----


def test_lower_matmul_produces_program():
    """lower_matmul_to_program returns a Program with correct structure."""
    g = _make_matmul_graph()
    g = _fuse(g)

    prog = lower_matmul_to_program(g, MatmulConfig(strategy="naive"), dims={"M": 4, "N": 3, "K": 2})

    assert prog.name == "matmul_4x3x2"
    assert len(prog.buffers) == 3
    assert len(prog.launches) == 1

    buf_names = {b.name for b in prog.buffers}
    assert "A" in buf_names
    assert "B" in buf_names
    assert "C" in buf_names

    input_bufs = [b for b in prog.buffers if b.role == "input"]
    output_bufs = [b for b in prog.buffers if b.role == "output"]
    assert len(input_bufs) == 2
    assert len(output_bufs) == 1


def test_lower_matmul_source_valid():
    """Generated source compiles (syntax check)."""
    g = _make_matmul_graph()
    g = _fuse(g)

    prog = lower_matmul_to_program(g, MatmulConfig(strategy="naive"), dims={"M": 4, "N": 3, "K": 2})
    source = generate_source(prog, mode="run")

    assert "int main()" in source
    assert "__global__" in source
    assert "cudaMalloc" in source


# ---- GPU tests ----


@requires_cuda
def test_matmul_program_runs():
    """Matmul Program compiles and runs on GPU."""
    g = _make_matmul_graph()
    g = _fuse(g)

    prog = lower_matmul_to_program(g, MatmulConfig(strategy="naive"), dims={"M": 4, "N": 3, "K": 2})
    result = run_program(prog)

    assert "C" in result.outputs
    assert len(result.outputs["C"]) == 4 * 3


@requires_cuda
def test_matmul_program_benchmark():
    """Matmul Program benchmarks correctly."""
    g = _make_matmul_graph()
    g = _fuse(g)

    prog = lower_matmul_to_program(g, MatmulConfig(strategy="naive"), dims={"M": 64, "N": 64, "K": 64})
    result = benchmark_program(prog, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 1
