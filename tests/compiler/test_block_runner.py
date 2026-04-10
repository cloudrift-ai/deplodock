"""Tests for block execution via the layered architecture: plan → CudaBackend → Program."""

import pytest

from deplodock.compiler.block_planner import BlockConfig, plan_block
from deplodock.compiler.cuda.backend import CudaBackend
from deplodock.compiler.cuda.program import generate_source
from deplodock.compiler.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)

_backend = CudaBackend()


# ---- Planning + compilation tests (no GPU) ----


def test_plan_and_compile_produces_program():
    """plan_block → CudaBackend.compile produces a Program with correct structure."""
    cfg = BlockConfig(batch=1, seq_len=32, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_dim=128)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)

    assert len(prog.launches) == 10
    assert prog.name.startswith("llama_block_")

    input_bufs = [b for b in prog.buffers if b.role == "input"]
    output_bufs = [b for b in prog.buffers if b.role == "output"]
    const_bufs = [b for b in prog.buffers if b.role == "constant"]
    assert len(input_bufs) == 3
    assert len(output_bufs) == 1
    assert len(const_bufs) == 9


def test_generate_block_source():
    """Generated .cu source has expected structure."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)
    source = generate_source(prog, mode="benchmark")

    assert "int main()" in source
    assert "fused_rmsnorm<<<" in source
    assert "triple_matmul<<<" in source
    assert "fused_rope<<<" in source
    assert "attention_qk<<<" in source
    assert "dual_matmul_silu_mul<<<" in source
    assert "PROGRAM_TIME_MS=" in source


# ---- GPU tests ----


@requires_cuda
def test_block_compiles_and_runs():
    """Full block: plan → compile → run on GPU."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)
    result = _backend.run(prog)

    assert "output" in result.outputs
    assert len(result.outputs["output"]) > 0


@requires_cuda
def test_block_output_is_finite():
    """Block output contains finite values."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)
    result = _backend.run(prog)

    for i, v in enumerate(result.outputs["output"]):
        assert v == v, f"NaN at output[{i}]"
        assert abs(v) < 1e6, f"Output[{i}] too large: {v}"


@requires_cuda
def test_block_benchmark_returns_timing():
    """Benchmark mode returns valid timing."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)
    result = _backend.benchmark(prog, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 10


@requires_cuda
def test_block_tinyllama_dims():
    """Block runs with TinyLlama dimensions."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_dim=5632)
    plan = plan_block(cfg)
    prog = _backend.compile(plan)
    result = _backend.benchmark(prog, warmup=2, num_iters=3)

    assert result.time_ms > 0
