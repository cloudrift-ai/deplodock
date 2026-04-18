"""End-to-end pipeline tests: Graph → compile_graph → CudaBackend → GPU."""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Update
from deplodock.compiler.ir.tensor import ElementwiseOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.loop import LoopProgram

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
    from deplodock.compiler.ir.frontend import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (m, n)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def test_compile_graph_fuses_matmul():
    from deplodock.compiler.ir.loop import flatten_body

    program = compile_graph(_matmul_graph(4, 3, 2))
    # Matmul decomposes into unsqueeze copies + mul/sum kernel.
    matmul_launches = [L for L in program.launches if any(isinstance(s, Update) for s in flatten_body(L.loop.body))]
    assert len(matmul_launches) == 1


def test_compile_graph_fuses_chain():
    """neg → exp fuses into one kernel (fan-out 1 chain)."""
    program = compile_graph(_pointwise_chain_graph())
    assert len(program.launches) == 1


def test_pipeline_to_program():
    compiled = CudaBackend().compile(_matmul_graph(4, 3, 2))
    assert len(compiled.launches) >= 1
    roles = {b.name: b.role for b in compiled.buffers}
    assert roles.get("a") == "input"
    assert roles.get("b") == "input"


def test_compile_result_metadata():
    """compile_graph returns a LoopProgram with graph_inputs/outputs."""
    g = _pointwise_chain_graph()
    program = compile_graph(g)
    assert isinstance(program, LoopProgram)
    assert program.graph_inputs == ["x"]
    assert len(program.graph_outputs) == 1
    assert len(program.launches) == 1


@requires_cuda
def test_pointwise_chain_gpu():
    import math

    compiled = CudaBackend().compile(_pointwise_chain_graph())
    x_data = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0]
    expected = [math.exp(-xi) for xi in x_data]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    assert list(result.outputs.values())[0] == pytest.approx(expected, rel=1e-5)


@requires_cuda
def test_matmul_gpu():
    import random

    random.seed(0)
    compiled = CudaBackend().compile(_matmul_graph(3, 4, 5))
    a_data = [random.random() for _ in range(12)]
    b_data = [random.random() for _ in range(20)]
    expected = []
    for mi in range(3):
        for ni in range(5):
            expected.append(sum(a_data[mi * 4 + k] * b_data[k * 5 + ni] for k in range(4)))
    result = CudaBackend().run(compiled, input_data={"a": a_data, "b": b_data})
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected, rel=1e-5)
