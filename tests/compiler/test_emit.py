"""Tests for the structural CUDA emitter with the grammar-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs.
"""

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


def _pointwise_add_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("y", (4,)), node_id="y")
    g.add_node(op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]
    return g


def _reduce_sum_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("y", (4,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _matmul_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8, 4)), node_id="m")
    g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=["m"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    g = _pointwise_add_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x", "y"], graph_outputs=["z"])
    assert len(program.launches) == 1
    source = program.launches[0].kernel_source
    assert "blockIdx.x" in source
    assert "x[" in source and "y[" in source


def test_reduce_emits_k_loop():
    g = _reduce_sum_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x"], graph_outputs=["y"])
    source = program.launches[0].kernel_source
    assert "for (int" in source
    assert "+=" in source


def test_contraction_emits_matmul():
    g = _matmul_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["a", "b"], graph_outputs=["o"])
    source = program.launches[0].kernel_source
    assert "for (int k" in source
    assert "acc +=" in source


def test_buffer_roles():
    g = _pointwise_add_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x", "y"], graph_outputs=["z"])
    roles = {b.name: b.role for b in program.buffers}
    assert roles.get("x") == "input"
    assert roles.get("y") == "input"


def test_chained_pointwise_single_kernel():
    """Two fan-out-1 elementwise ops fuse into one kernel."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    program = CudaBackend().compile(kernels, graph_inputs=["x"], graph_outputs=["n"])
    assert len(program.launches) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    g = _pointwise_add_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x", "y"], graph_outputs=["z"])
    result = CudaBackend().run(program, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert result.outputs["z"] == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    g = _reduce_sum_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x"], graph_outputs=["y"])
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert result.outputs["y"] == pytest.approx(expected)


@requires_cuda
def test_matmul_runs_on_gpu():
    g = _matmul_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["a", "b"], graph_outputs=["o"])
    a_data = [float(i) for i in range(32)]
    b_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"a": a_data, "b": b_data})
    expected = []
    for mi in range(4):
        for ni in range(4):
            s = sum(a_data[mi * 8 + k] * b_data[k * 4 + ni] for k in range(8))
            expected.append(s)
    assert result.outputs["o"] == pytest.approx(expected)
