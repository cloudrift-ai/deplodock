"""Tests for the structural CUDA emitter with the grammar-based fusion pipeline.

Exercises source-level assertions and end-to-end GPU runs. CUDA-specific
by design (source-level assertions on emitted C code); not parameterized
over backends.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp  # noqa: F401
from deplodock.compiler.program.loop import LoopBuffer, LoopLaunch, LoopProgram

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
    from deplodock.compiler.ir.frontend import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _softmax_launch() -> LoopLaunch:
    """Build a LoopLaunch with the softmax SSA pattern directly."""
    from deplodock.compiler.ir.expr import Literal, Var
    from deplodock.compiler.ir.loop import Assign, Axis, LocalBuffer, LoopOp, Port, Update, Write

    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    loop = LoopOp(
        axes=axes,
        inputs=(p, p),
        locals=(
            LocalBuffer(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
            LocalBuffer(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
        ),
        body=(
            Update(target="mx", value="$0"),
            Assign(name="sub", op=ElementwiseOp("sub"), args=("$0", "mx")),
            Assign(name="ex", op=ElementwiseOp("exp"), args=("sub",)),
            Update(target="sm", value="ex"),
            Assign(name="out", op=ElementwiseOp("div"), args=("ex", "sm")),
            Write(output=0, index=(Var("a0"), Var("a1")), value="out"),
        ),
    )
    return LoopLaunch(loop=loop, input_names=["x", "x"], output_name="y")


def _softmax_program() -> LoopProgram:
    """Wrap the softmax launch in a standalone LoopProgram for codegen tests."""
    launch = _softmax_launch()
    return LoopProgram(
        name="softmax",
        buffers=[
            LoopBuffer(name="x", shape=(4, 8), role="input"),
            LoopBuffer(name="y", shape=(4, 8), role="output"),
        ],
        launches=[launch],
        graph_inputs=["x"],
        graph_outputs=["y"],
    )


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_emits_correct_source():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    assert len(compiled.launches) == 1
    source = compiled.launches[0].kernel_source
    assert "blockIdx.x" in source
    assert "x[" in source and "y[" in source


def test_reduce_emits_k_loop():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    source = compiled.launches[0].kernel_source
    assert "for (int" in source
    assert "+=" in source


def test_contraction_emits_matmul():
    compiled = CudaBackend().compile(_matmul_graph())
    source = compiled.launches[-1].kernel_source
    assert "for (int k" in source
    assert "acc0 +=" in source


def test_buffer_roles():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    roles = {b.name: b.role for b in compiled.buffers}
    assert roles.get("x") == "input"
    assert roles.get("y") == "input"


def test_softmax_emits_multiple_k_loops():
    """Softmax pattern emits separate K-loops for max, sub+exp+sum, div."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel

    loop_program = _softmax_program()
    launch = loop_program.launches[0]
    kdef, arg_order = emit_kernel(launch, "k0_softmax", loop_program)
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as emit_src

    source = emit_src(kdef)
    loop_count = source.count("for (int")
    assert loop_count >= 3, f"expected >= 3 K-loops, got {loop_count}\n{source}"
    assert "fmaxf" in source
    assert "+=" in source


def test_softmax_emits_per_element_store():
    """Softmax output is per-element: the final div stores inside a K-loop."""
    from deplodock.compiler.backend.cuda.emit import emit_kernel
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as emit_src

    loop_program = _softmax_program()
    launch = loop_program.launches[0]
    kdef, _ = emit_kernel(launch, "k0_softmax", loop_program)
    source = emit_src(kdef)
    assert "y[" in source, f"expected output store y[...]\n{source}"


def test_chained_pointwise_single_kernel():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    compiled = CudaBackend().compile(g)
    assert len(compiled.launches) == 1


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    compiled = CudaBackend().compile(_pointwise_add_graph())
    result = CudaBackend().run(compiled, input_data={"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    assert list(result.outputs.values())[0] == pytest.approx([11, 22, 33, 44])


@requires_cuda
def test_reduce_runs_on_gpu():
    compiled = CudaBackend().compile(_reduce_sum_graph())
    x_data = [float(i) for i in range(32)]
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert list(result.outputs.values())[0] == pytest.approx(expected)


@requires_cuda
def test_softmax_runs_on_gpu():
    """Softmax from a hand-built LoopProgram (no Graph). Uses compile_kernels
    directly — the resulting GpuProgram is what CudaBackend.run expects."""
    import math

    from deplodock.compiler.backend.cuda.emit import compile_kernels

    loop_program = _softmax_program()
    compiled = compile_kernels(loop_program)
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
