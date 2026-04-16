"""End-to-end pipeline tests: Graph → compile_graph → CudaBackend → GPU.

Exercises the structural pipeline on small synthetic graphs that only use
primitives the naive lowering + codegen fully supports (ElementwiseOp,
ReduceOp, InputOp). Decomposition of higher-level ops (LinearOp,
SdpaOp, MatmulOp, MeanOp) plus layout-op -> IndexMapOp rewriting are
out-of-scope here; TinyLlama-layer-level E2E will land once that
decomposition is ported into the new lowering.
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


def _pointwise_chain_graph() -> Graph:
    """y = exp(-(x)) — exercises two chained pointwise kernels."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8,)), node_id="x")
    g.add_node(op=ElementwiseOp("neg"), inputs=["x"], output=Tensor("n", (8,)), node_id="n")
    g.add_node(op=ElementwiseOp("exp"), inputs=["n"], output=Tensor("y", (8,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _matmul_graph(m: int, k: int, n: int) -> Graph:
    """c = mul_sum(a, b): an (m, k) × (k, n) matmul expressed as mul + sum."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(
        op=ElementwiseOp("mul"),
        inputs=["a", "b"],
        output=Tensor("ab", (m, k, n)),
        node_id="ab",
    )
    g.add_node(
        op=ReduceOp(fn="sum", axis=1),
        inputs=["ab"],
        output=Tensor("c", (m, n)),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


# ---------------------------------------------------------------------------
# compile_graph semantics (no GPU required)
# ---------------------------------------------------------------------------


def test_compile_graph_lowers_matmul_to_single_kernel():
    kernels = compile_graph(_matmul_graph(4, 3, 2))
    assert len(kernels) == 1
    assert kernels[0].contraction is not None
    assert kernels[0].reduce_stages == ()


def test_compile_graph_chains_pointwise_as_separate_kernels():
    kernels = compile_graph(_pointwise_chain_graph())
    # neg and exp become separate singleton pointwise kernels.
    assert len(kernels) == 2
    assert all(k.contraction is None and k.reduce_stages == () for k in kernels)


def test_cuda_backend_compile_from_pipeline():
    """End-to-end: compile_graph -> CudaBackend.compile -> Program."""
    kernels = compile_graph(_matmul_graph(4, 3, 2))
    program = CudaBackend().compile(kernels, graph_inputs=["a", "b"], graph_outputs=["c"])
    assert len(program.launches) == 1
    launch = program.launches[0]
    assert launch.kernel_name.endswith("_contract")
    roles = {b.name: b.role for b in program.buffers}
    assert roles["a"] == "input"
    assert roles["b"] == "input"
    assert roles["c"] == "output"


# ---------------------------------------------------------------------------
# End-to-end on GPU
# ---------------------------------------------------------------------------


@requires_cuda
def test_pointwise_chain_runs_on_gpu():
    g = _pointwise_chain_graph()
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["x"], graph_outputs=["y"])

    import math

    x_data = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0]
    expected = [math.exp(-xi) for xi in x_data]
    result = CudaBackend().run(program, input_data={"x": x_data})
    assert result.outputs["y"] == pytest.approx(expected, rel=1e-5)


@requires_cuda
def test_matmul_runs_on_gpu():
    g = _matmul_graph(3, 4, 5)
    kernels = compile_graph(g)
    program = CudaBackend().compile(kernels, graph_inputs=["a", "b"], graph_outputs=["c"])

    import random

    random.seed(0)
    a_data = [random.random() for _ in range(3 * 4)]
    b_data = [random.random() for _ in range(4 * 5)]
    expected = []
    for m in range(3):
        for n in range(5):
            s = 0.0
            for k in range(4):
                s += a_data[m * 4 + k] * b_data[k * 5 + n]
            expected.append(s)
    result = CudaBackend().run(program, input_data={"a": a_data, "b": b_data})
    assert result.outputs["c"] == pytest.approx(expected, rel=1e-5)
