"""Tests for the structural recursive-descent CUDA emitter.

Covers source-level assertions (what code gets emitted for each kernel
shape) plus an end-to-end GPU run when nvcc + a CUDA device are
available.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.lower import lower
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp

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
    """Naive mul+sum matmul: (M, K) * (K, N) -> (M, N)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(
        op=ElementwiseOp("mul"),
        inputs=["a", "b"],
        output=Tensor("m", (4, 8, 4)),
        node_id="m",
    )
    g.add_node(
        op=ReduceOp(fn="sum", axis=1),
        inputs=["m"],
        output=Tensor("o", (4, 4)),
        node_id="o",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


# ---------------------------------------------------------------------------
# Source-level structure assertions
# ---------------------------------------------------------------------------


def test_pointwise_source_structure():
    g = _pointwise_add_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["x", "y"], graph_outputs=["z"])
    assert len(program.launches) == 1
    launch = program.launches[0]
    assert launch.kernel_name == "k0_pointwise"
    # 1D grid, 256 threads per block.
    assert launch.block == (256, 1, 1)
    source = launch.kernel_source
    assert "idx = blockIdx.x * blockDim.x + threadIdx.x" in source
    assert "if (idx < 4)" in source
    assert "x[idx]" in source and "y[idx]" in source
    assert "z[idx] =" in source


def test_reduce_source_structure():
    g = _reduce_sum_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["x"], graph_outputs=["y"])
    launch = program.launches[0]
    assert launch.kernel_name == "k0_reduce"
    source = launch.kernel_source
    assert "blockIdx.x" in source  # 1 block per output row
    assert "for (int k0 = 0; k0 < 8;" in source  # K loop over reduced axis
    assert "acc0 +=" in source
    assert "y[row] = acc0" in source


def test_contraction_source_structure():
    g = _matmul_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["a", "b"], graph_outputs=["o"])
    launch = program.launches[0]
    assert launch.kernel_name == "k0_contract"
    # 2D grid over (M, N): grid = (N, M, 1) per the naive layout.
    assert launch.grid == (4, 4, 1)
    source = launch.kernel_source
    assert "for (int k = 0; k < 8;" in source
    assert "acc += a[m * 8 + k] * b[k * 4 + n]" in source
    assert "o[m * 4 + n] = acc" in source


def test_program_buffer_roles():
    g = _pointwise_add_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["x", "y"], graph_outputs=["z"])
    roles = {b.name: b.role for b in program.buffers}
    assert roles["x"] == "input"
    assert roles["y"] == "input"
    assert roles["z"] == "output"


def test_intermediate_buffer_is_scratch():
    """Chain of two pointwise kernels; the intermediate buffer is scratch."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    program = CudaBackend().compile(lower(g), graph_inputs=["x"], graph_outputs=["n"])
    roles = {b.name: b.role for b in program.buffers}
    assert roles["x"] == "input"
    assert roles["e"] == "scratch"
    assert roles["n"] == "output"
    assert len(program.launches) == 2


# ---------------------------------------------------------------------------
# GPU execution (when available)
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_runs_on_gpu():
    g = _pointwise_add_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["x", "y"], graph_outputs=["z"])
    result = CudaBackend().run(
        program,
        input_data={
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
        },
    )
    assert result.outputs["z"] == pytest.approx([11.0, 22.0, 33.0, 44.0])


@requires_cuda
def test_reduce_runs_on_gpu():
    g = _reduce_sum_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["x"], graph_outputs=["y"])
    x_data = [float(i) for i in range(32)]  # shape (4, 8), rows 0-7, 8-15, ...
    result = CudaBackend().run(program, input_data={"x": x_data})
    expected = [sum(x_data[row * 8 : (row + 1) * 8]) for row in range(4)]
    assert result.outputs["y"] == pytest.approx(expected)


@requires_cuda
def test_matmul_runs_on_gpu():
    g = _matmul_graph()
    program = CudaBackend().compile(lower(g), graph_inputs=["a", "b"], graph_outputs=["o"])
    # a: 4x8, b: 8x4. Compute a @ b manually.
    a_data = [float(i) for i in range(32)]
    b_data = [float(i) for i in range(32)]
    result = CudaBackend().run(program, input_data={"a": a_data, "b": b_data})
    expected = []
    for m in range(4):
        for n in range(4):
            s = 0.0
            for k in range(8):
                s += a_data[m * 8 + k] * b_data[k * 4 + n]
            expected.append(s)
    assert result.outputs["o"] == pytest.approx(expected)
