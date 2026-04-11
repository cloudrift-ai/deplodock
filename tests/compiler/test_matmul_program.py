"""Tests for matmul through the graph pipeline: Graph → plan_graph → CudaBackend."""

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.program import generate_source
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.plan import plan_graph
from deplodock.compiler.rewriter import Rewriter

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _make_and_compile_matmul(m: int, n: int, k: int) -> Graph:
    """Build a matmul graph with concrete shapes and fuse it."""
    from pathlib import Path

    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (m, k)), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", (m, k, n)), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", (m, n)), node_id="red")
    g.outputs = [red]

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    return Rewriter.from_directory(rules_dir).apply(g)


_backend = CudaBackend()


def test_matmul_plan_graph():
    """plan_graph on a matmul graph produces a plan with one matmul op."""
    compiled = _make_and_compile_matmul(4, 3, 2)
    plan = plan_graph(compiled)

    matmul_ops = [op for op in plan.ops if op.op == "matmul"]
    assert len(matmul_ops) == 1
    assert matmul_ops[0].params["M"] == 4
    assert matmul_ops[0].params["N"] == 3


def test_matmul_source_valid():
    """Generated source compiles (syntax check)."""
    compiled = _make_and_compile_matmul(4, 3, 2)
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    source = generate_source(program, mode="run")

    assert "int main()" in source
    assert "__global__" in source


@requires_cuda
def test_matmul_program_runs():
    """Matmul through plan_graph → CudaBackend runs on GPU."""
    compiled = _make_and_compile_matmul(4, 3, 2)
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    result = _backend.run(program)

    assert len(result.outputs) == 1
    output_values = list(result.outputs.values())[0]
    assert len(output_values) == 4 * 3


@requires_cuda
def test_matmul_program_benchmark():
    """Matmul benchmarks correctly."""
    compiled = _make_and_compile_matmul(64, 64, 64)
    plan = plan_graph(compiled)
    program = _backend.compile(plan)
    result = _backend.benchmark(program, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 1
