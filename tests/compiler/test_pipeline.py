"""End-to-end pipeline tests: Graph → compile_graph → CudaBackend → GPU."""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _pointwise_chain_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8,)), node_id="x")
    g.add_node(op=ElementwiseOp("neg"), inputs=["x"], output=Tensor("n", (8,)), node_id="n")
    g.add_node(op=ElementwiseOp("exp"), inputs=["n"], output=Tensor("y", (8,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _matmul_graph(m: int, k: int, n: int) -> Graph:
    from deplodock.compiler.ops import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (m, n)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def test_compile_graph_fuses_matmul():
    result = compile_graph(_matmul_graph(4, 3, 2))
    assert len(result.kernels) == 1
    # Matmul is lowered as mul + sum (ReduceOp present in body).
    assert any(isinstance(a.op, ReduceOp) for a in result.kernels[0].body)


def test_compile_graph_fuses_chain():
    """neg → exp fuses into one kernel (fan-out 1 chain)."""
    result = compile_graph(_pointwise_chain_graph())
    assert len(result.kernels) == 1


def test_pipeline_to_program():
    result = compile_graph(_matmul_graph(4, 3, 2))
    out_name = result.kernels[-1].outputs[0].buffer_id
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=[out_name]
    )
    assert len(program.launches) >= 1
    roles = {b.name: b.role for b in program.buffers}
    assert roles.get("a") == "input"
    assert roles.get("b") == "input"


def test_compile_result_metadata():
    """compile_graph returns a CompileResult with graph_inputs/outputs."""
    from deplodock.compiler.pipeline import CompileResult

    g = _pointwise_chain_graph()
    result = compile_graph(g)
    assert isinstance(result, CompileResult)
    assert result.graph_inputs == ["x"]
    assert len(result.graph_outputs) == 1
    assert len(result.kernels) == 1


@requires_cuda
def test_pointwise_chain_gpu():
    import math

    g = _pointwise_chain_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    x_data = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0]
    expected = [math.exp(-xi) for xi in x_data]
    result = CudaBackend().run(program, input_data={"x": x_data})
    assert list(result.outputs.values())[0] == pytest.approx(expected, rel=1e-5)


@requires_cuda
def test_matmul_gpu():
    import random

    random.seed(0)
    g = _matmul_graph(3, 4, 5)
    result = compile_graph(g)
    out_name = result.kernels[-1].outputs[0].buffer_id
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=[out_name]
    )
    a_data = [random.random() for _ in range(12)]
    b_data = [random.random() for _ in range(20)]
    expected = []
    for mi in range(3):
        for ni in range(5):
            expected.append(sum(a_data[mi * 4 + k] * b_data[k * 5 + ni] for k in range(4)))
    result = CudaBackend().run(program, input_data={"a": a_data, "b": b_data})
    assert list(result.outputs.values())[0] == pytest.approx(expected, rel=1e-5)
