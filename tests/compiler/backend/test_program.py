"""Tests for cupy dispatch of a lowered ``Graph[CudaOp]``."""

from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp

from ..conftest import requires_cuda

EW_ADD_SOURCE = """
extern "C" __global__ void ew_add(const float* A, const float* B, float* C) {
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


@requires_cuda
def test_run_program_elementwise_add():
    result, _ = run_program(_make_add_graph(8))
    assert "C" in result.outputs
    assert result.outputs["C"].shape == (8,)
    assert all(v == v for v in result.outputs["C"].tolist())  # NaN check


@requires_cuda
def test_benchmark_program_returns_timing():
    result = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5)
    assert result.time_ms > 0
    assert result.num_launches == 1
    assert result.per_launch is not None
    assert len(result.per_launch) == 1


@requires_cuda
def test_iter_once_raises_on_zero_elapsed(monkeypatch):
    """A 0.0ms CUDA event reading means the launch was a degenerate
    no-op (e.g. an all-masked-out grid in a BM=1×BN=128 register tile).
    ``iter_once`` must surface this as a bench_fail RuntimeError so the
    autotune DB never pins a 0µs "win" as the unbeatable best."""
    import cupy as cp
    import pytest

    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock

    monkeypatch.setattr(cp.cuda, "get_elapsed_time", lambda start, stop: 0.0)

    with gpu_lock():
        prog = CompiledProgram.build(_make_add_graph(8))
        with pytest.raises(RuntimeError, match="reported 0.000ms elapsed"):
            prog.iter_once()


@requires_cuda
def test_benchmark_program_propagates_zero_elapsed_as_bench_fail(monkeypatch):
    """End-to-end: a 0.0ms reading inside ``benchmark_program`` must
    bubble up as a RuntimeError, never silently produce a sample of 0.0
    that ``_samples_to_result`` would record as the kernel's median."""
    import cupy as cp
    import pytest

    monkeypatch.setattr(cp.cuda, "get_elapsed_time", lambda start, stop: 0.0)

    with pytest.raises(RuntimeError, match="reported 0.000ms elapsed"):
        benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5)
