"""Tests for the structural CUDA emitter with the grammar-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.lower import KernelInfo
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
    from deplodock.compiler.ops import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _softmax_kernel_info():
    """Build a KernelInfo with the softmax SSA pattern directly."""
    from deplodock.compiler.ops import Assign, KernelOp, Port

    kernel = KernelOp(
        inputs=(Port(), Port()),
        body=(
            Assign(name="mx", op=ReduceOp(fn="max", axis=-1), args=("$0",)),
            Assign(name="sub", op=ElementwiseOp("sub"), args=("$0", "mx")),
            Assign(name="ex", op=ElementwiseOp("exp"), args=("sub",)),
            Assign(name="sm", op=ReduceOp(fn="sum", axis=-1), args=("ex",)),
            Assign(name="out", op=ElementwiseOp("div"), args=("ex", "sm")),
        ),
        outputs=(Port(),),
    )
    return KernelInfo(kernel=kernel, input_names=["x", "x"], output_name="y")


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    g = _pointwise_add_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    assert len(program.launches) == 1
    source = program.launches[0].kernel_source
    assert "blockIdx.x" in source
    assert "x[" in source and "y[" in source


def test_reduce_emits_k_loop():
    g = _reduce_sum_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    source = program.launches[0].kernel_source
    assert "for (int" in source
    assert "+=" in source


def test_contraction_emits_matmul():
    g = _matmul_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    source = program.launches[-1].kernel_source
    assert "for (int k" in source
    assert "acc0 +=" in source


def test_buffer_roles():
    g = _pointwise_add_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    roles = {b.name: b.role for b in program.buffers}
    assert roles.get("x") == "input"
    assert roles.get("y") == "input"


def test_softmax_emits_multiple_k_loops():
    """Softmax pattern emits separate K-loops for max, sub+exp+sum, div."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel

    info = _softmax_kernel_info()
    shapes = {"$0": (4, 8), "$1": (4, 8)}
    kdef, arg_order = emit_kernel(info, "k0_softmax", shapes)
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel as emit_src

    source = emit_src(kdef)
    loop_count = source.count("for (int")
    assert loop_count >= 3, f"expected >= 3 K-loops, got {loop_count}\n{source}"
    assert "fmaxf" in source
    assert "+=" in source


def test_softmax_emits_per_element_store():
    """Softmax output is per-element: the final div stores inside a K-loop."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel as emit_src

    info = _softmax_kernel_info()
    shapes = {"$0": (4, 8), "$1": (4, 8)}
    kdef, _ = emit_kernel(info, "k0_softmax", shapes)
    source = emit_src(kdef)
    assert "y[" in source, f"expected output store y[...]\n{source}"


def test_chained_pointwise_single_kernel():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    result = compile_graph(g)
    kernels = result.kernels
    assert len(kernels) == 1
    program = CudaBackend().compile(
        kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    assert len(program.launches) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    g = _pointwise_add_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    result = CudaBackend().run(program, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert list(result.outputs.values())[0] == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    g = _reduce_sum_graph()
    result = compile_graph(g)
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=result.graph_outputs
    )
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert list(result.outputs.values())[0] == pytest.approx(expected)


@requires_cuda
def test_softmax_runs_on_gpu():
    import math

    from deplodock.compiler.backend.cuda.emit import compile_kernels

    info = _softmax_kernel_info()
    program = compile_kernels([info], buf_shapes={"x": (4, 8), "y": (4, 8)}, graph_inputs=["x"], graph_outputs=["y"])
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"x": x_data})
    expected = []
    for row in range(4):
        row_vals = x_data[row * 8 : (row + 1) * 8]
        mx = max(row_vals)
        exps = [math.exp(v - mx) for v in row_vals]
        s = sum(exps)
        expected.extend(e / s for e in exps)
    assert list(result.outputs.values())[0] == pytest.approx(expected, rel=1e-3)


@requires_cuda
def test_matmul_runs_on_gpu():
    g = _matmul_graph()
    result = compile_graph(g)
    out_name = result.kernels[-1].output_name
    program = CudaBackend().compile(
        result.kernels, buf_shapes=result.buf_shapes, graph_inputs=result.graph_inputs, graph_outputs=[out_name]
    )
    a_data = [float(i) for i in range(32)]
    b_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"a": a_data, "b": b_data})
    expected = []
    for mi in range(4):
        for ni in range(4):
            s = sum(a_data[mi * 8 + k] * b_data[k * 4 + ni] for k in range(8))
            expected.append(s)
    assert list(result.outputs.values())[0] == pytest.approx(expected)
