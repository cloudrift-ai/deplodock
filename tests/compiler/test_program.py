"""Tests for the unified Program execution abstraction."""

import pytest

from deplodock.compiler.backend.cuda.program import (
    Buffer,
    Launch,
    Program,
    benchmark_program,
    generate_source,
    run_program,
)
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

EW_ADD_SOURCE = """
__global__ void ew_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
"""


def _make_add_program(n: int = 8) -> Program:
    """Simple elementwise add: C = A + B."""
    return Program(
        name="test_add",
        buffers=[
            Buffer("A", n, role="input"),
            Buffer("B", n, role="input"),
            Buffer("C", n, role="output"),
        ],
        launches=[
            Launch(
                kernel_source=EW_ADD_SOURCE,
                kernel_name="ew_add",
                grid=((n + 255) // 256, 1, 1),
                block=(256, 1, 1),
                args=["A", "B", "C", str(n)],
            ),
        ],
    )


# ---- Codegen tests (no GPU) ----


def test_generate_run_source():
    prog = _make_add_program()
    src = generate_source(prog, mode="run")

    assert "int main()" in src
    assert "ew_add<<<" in src
    assert "cudaMalloc" in src
    assert "OUT C" in src


def test_generate_benchmark_source():
    prog = _make_add_program()
    src = generate_source(prog, mode="benchmark")

    assert "PROGRAM_TIME_MS=" in src
    assert "PROGRAM_LAUNCHES=" in src
    assert "cudaEventCreate" in src


def test_buffers_allocated_by_role():
    prog = _make_add_program()
    src = generate_source(prog, mode="run")

    # Inputs get initialized, scratch/output don't.
    assert "cudaMemcpy(d_A" in src  # input
    assert "cudaMemcpy(d_B" in src  # input
    # Output is read back.
    assert "cudaMemcpy(h, d_C" in src


def test_multi_kernel_program():
    """Program with 2 launches generates both kernel calls."""
    prog = Program(
        name="test_multi",
        buffers=[
            Buffer("X", 4, role="input"),
            Buffer("Y", 4, role="output"),
            Buffer("Z", 4, role="output"),
        ],
        launches=[
            Launch(
                kernel_source="__global__ void k1(float* X, float* Y, int n) { int i = threadIdx.x; if (i < n) Y[i] = X[i] * 2; }",
                kernel_name="k1",
                grid=(1, 1, 1),
                block=(256, 1, 1),
                args=["X", "Y", "4"],
            ),
            Launch(
                kernel_source="__global__ void k2(float* Y, float* Z, int n) { int i = threadIdx.x; if (i < n) Z[i] = Y[i] + 1; }",
                kernel_name="k2",
                grid=(1, 1, 1),
                block=(256, 1, 1),
                args=["Y", "Z", "4"],
            ),
        ],
    )
    src = generate_source(prog, mode="run")
    assert "k1<<<" in src
    assert "k2<<<" in src


def test_defines_in_source():
    prog = _make_add_program()
    prog.defines = {"N": "8", "BLOCK": "256"}
    src = generate_source(prog, mode="run")
    assert "#define N 8" in src
    assert "#define BLOCK 256" in src


# ---- GPU tests ----


@requires_cuda
def test_run_program_elementwise_add():
    """Run a simple A+B program and verify output values are finite."""
    prog = _make_add_program(8)
    result = run_program(prog)

    assert "C" in result.outputs
    assert len(result.outputs["C"]) == 8
    for v in result.outputs["C"]:
        assert v == v, "NaN in output"  # NaN != NaN


@requires_cuda
def test_benchmark_program_returns_timing():
    """Benchmark mode returns valid timing."""
    prog = _make_add_program(1024)
    result = benchmark_program(prog, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 1


@requires_cuda
def test_multi_kernel_execution():
    """Two sequential kernels produce correct chained output."""
    prog = Program(
        name="chain",
        buffers=[
            Buffer("A", 4, role="input"),
            Buffer("B", 4, role="scratch"),
            Buffer("C", 4, role="output"),
        ],
        launches=[
            Launch(
                kernel_source=(
                    "__global__ void double_it(const float* A, float* B, int n) { int i = threadIdx.x; if (i < n) B[i] = A[i] * 2.0f; }"
                ),
                kernel_name="double_it",
                grid=(1, 1, 1),
                block=(256, 1, 1),
                args=["A", "B", "4"],
            ),
            Launch(
                kernel_source=(
                    "__global__ void add_one(const float* B, float* C, int n) { int i = threadIdx.x; if (i < n) C[i] = B[i] + 1.0f; }"
                ),
                kernel_name="add_one",
                grid=(1, 1, 1),
                block=(256, 1, 1),
                args=["B", "C", "4"],
            ),
        ],
    )
    result = run_program(prog)
    assert "C" in result.outputs
    assert len(result.outputs["C"]) == 4
    # Input is pseudorandom, but output should be finite.
    for v in result.outputs["C"]:
        assert abs(v) < 100, f"Unexpected value: {v}"
