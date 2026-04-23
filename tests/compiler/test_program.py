"""Tests for codegen + execution on a lowered ``Graph[CudaOp]``."""

import pytest

from deplodock.compiler.backend.cuda.program import benchmark_program, generate_source, run_program
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.graph import Graph, Tensor

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

EW_ADD_SOURCE = """
__global__ void ew_add(const float* A, const float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8) C[i] = A[i] + B[i];
}
"""


def _make_add_graph(n: int = 8) -> Graph:
    """Simple elementwise add: C = A + B."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (n,)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (n,)), node_id="B")
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE,
            kernel_name="ew_add",
            arg_order=("A", "B", "C"),
            grid=((n + 255) // 256, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["A", "B"],
        output=Tensor("C", (n,)),
        node_id="C",
    )
    g.inputs = ["A", "B"]
    g.outputs = ["C"]
    return g


# ---- Codegen tests (no GPU) ----


def test_generate_run_source():
    src = generate_source(_make_add_graph(), mode="run")
    assert "int main()" in src
    assert "ew_add<<<" in src
    assert "cudaMalloc" in src
    assert "OUT C" in src


def test_generate_benchmark_source():
    src = generate_source(_make_add_graph(), mode="benchmark")
    assert "PROGRAM_TIME_MS=" in src
    assert "PROGRAM_LAUNCHES=" in src
    assert "cudaEventCreate" in src


def test_generate_source_bakes_scalar_constants():
    """Pre-baked scalar constants (RMSNorm eps, softmax scale, mask fill, …)
    must be written into the generated .cu as literals."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (8,)), node_id="A")
    g.add_node(op=ConstantOp(name="eps", value=1e-5), inputs=[], output=Tensor("eps", (1,)), node_id="eps")
    g.add_node(op=ConstantOp(name="scale", value=0.125), inputs=[], output=Tensor("scale", (1,)), node_id="scale")
    g.add_node(op=ConstantOp(name="mask_fill", value=-1e9), inputs=[], output=Tensor("mask_fill", (1,)), node_id="mask_fill")
    g.add_node(
        op=CudaOp(
            kernel_source=EW_ADD_SOURCE,
            kernel_name="ew_add",
            arg_order=("A", "eps", "C"),
            grid=(1, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["A", "eps"],
        output=Tensor("C", (8,)),
        node_id="C",
    )
    g.inputs = ["A"]
    g.outputs = ["C"]

    src = generate_source(g, mode="benchmark")

    assert "1e-05" in src
    assert "0.125" in src
    assert "-1000000000.0" in src or "-1e+09" in src

    baked_init_count = src.count("0.01f * ((i * 7")
    assert baked_init_count == 1, f"expected 1 pseudorandom init for A, got {baked_init_count}"


def test_buffers_allocated_by_role():
    src = generate_source(_make_add_graph(), mode="run")
    assert "cudaMemcpy(d_A" in src  # input
    assert "cudaMemcpy(d_B" in src  # input
    assert "cudaMemcpy(h, d_C" in src  # output readback


def test_multi_kernel_program():
    """Graph with 2 CudaOp launches generates both kernel calls."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="X")
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k1(float* X, float* Y) { int i = threadIdx.x; if (i < 4) Y[i] = X[i] * 2; }",
            kernel_name="k1",
            arg_order=("X", "Y"),
            grid=(1, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["X"],
        output=Tensor("Y", (4,)),
        node_id="Y",
    )
    g.add_node(
        op=CudaOp(
            kernel_source="__global__ void k2(float* Y, float* Z) { int i = threadIdx.x; if (i < 4) Z[i] = Y[i] + 1; }",
            kernel_name="k2",
            arg_order=("Y", "Z"),
            grid=(1, 1, 1),
            block=(256, 1, 1),
        ),
        inputs=["Y"],
        output=Tensor("Z", (4,)),
        node_id="Z",
    )
    g.inputs = ["X"]
    g.outputs = ["Y", "Z"]
    src = generate_source(g, mode="run")
    assert "k1<<<" in src
    assert "k2<<<" in src


# ---- GPU tests ----


@requires_cuda
def test_run_program_elementwise_add():
    result = run_program(_make_add_graph(8))
    assert "C" in result.outputs
    assert len(result.outputs["C"]) == 8
    for v in result.outputs["C"]:
        assert v == v  # NaN check


@requires_cuda
def test_benchmark_program_returns_timing():
    result = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5)
    assert result.time_ms > 0
    assert result.num_launches == 1
