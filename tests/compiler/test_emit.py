"""Tests for the structural CUDA emitter with the pattern-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs. CUDA-specific
by design (source-level assertions on emitted C code); not parameterized
over backends.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runtime import has_cuda_gpu
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp  # noqa: F401

requires_cuda = pytest.mark.skipif(
    not has_cuda_gpu(),
    reason="CUDA not available (need cupy + GPU)",
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
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("y", (4, 1)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _matmul_graph() -> Graph:
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _softmax_graph() -> Graph:
    """A 2-axis softmax graph, reduced on axis=1."""
    from deplodock.compiler.ir.frontend.ir import SoftmaxOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(
        op=SoftmaxOp(axis=-1),
        inputs=["x"],
        output=Tensor("y", (4, 8)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _cuda_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    nodes = _cuda_nodes(compiled)
    assert len(nodes) == 1
    source = nodes[0].op.kernel_source
    assert "blockIdx.x" in source
    assert "x[" in source and "y[" in source


def test_reduce_emits_k_loop():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    source = _cuda_nodes(compiled)[0].op.kernel_source
    assert "for (int" in source
    assert "+=" in source


def test_contraction_emits_matmul():
    compiled = CudaBackend().compile(_matmul_graph())
    source = _cuda_nodes(compiled)[-1].op.kernel_source
    assert "for (int " in source
    assert "+=" in source  # accumulator fold


def test_buffer_roles():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    assert "x" in compiled.inputs
    assert "y" in compiled.inputs
    assert len(compiled.outputs) == 1


def test_softmax_emits_multiple_k_loops():
    """Softmax pattern emits separate K-loops for max, sub+exp+sum, div."""
    compiled = CudaBackend().compile(_softmax_graph())
    sources = [n.op.kernel_source for n in _cuda_nodes(compiled)]
    # Find the softmax-bearing kernel (contains fmaxf for max reduction).
    softmax_src = next((s for s in sources if "fmaxf" in s), None)
    assert softmax_src is not None, f"no kernel with fmaxf found; sources={sources}"
    loop_count = softmax_src.count("for (int")
    assert loop_count >= 2, f"expected >= 2 K-loops, got {loop_count}\n{softmax_src}"
    assert "+=" in softmax_src


def test_softmax_emits_per_element_store():
    """Softmax output is per-element: the final div stores inside a K-loop."""
    compiled = CudaBackend().compile(_softmax_graph())
    out_name = compiled.outputs[0]
    sources = [n.op.kernel_source for n in _cuda_nodes(compiled)]
    assert any(f"{out_name}[" in s for s in sources)


def test_chained_pointwise_single_kernel():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("negative"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    compiled = CudaBackend().compile(g)
    assert len(_cuda_nodes(compiled)) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    result = CudaBackend().run(compiled, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected)


@requires_cuda
def test_softmax_runs_on_gpu():
    import math

    compiled = CudaBackend().compile(_softmax_graph())
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = []
    for row in range(4):
        row_vals = x_data[row * 8 : (row + 1) * 8]
        mx = max(row_vals)
        exps = [math.exp(v - mx) for v in row_vals]
        s = sum(exps)
        expected.extend(e / s for e in exps)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected, rel=1e-3)


@requires_cuda
def test_matmul_runs_on_gpu():
    compiled = CudaBackend().compile(_matmul_graph())
    a_data = [float(i) for i in range(32)]
    b_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"a": a_data, "b": b_data})
    expected = []
    for mi in range(4):
        for ni in range(4):
            s = sum(a_data[mi * 8 + k] * b_data[k * 4 + ni] for k in range(8))
            expected.append(s)
    assert list(result.outputs.values())[0].flatten().tolist() == pytest.approx(expected)
