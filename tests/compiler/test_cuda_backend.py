"""Tests for the CUDA backend: ExecutionPlan → Program → GPU."""

import pytest

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.backend.cuda.program import generate_source
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.block_planner import BlockConfig, plan_block

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def test_cuda_backend_compile_produces_program():
    """CudaBackend.compile() produces a Program with correct structure."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    backend = CudaBackend()
    program = backend.compile(plan)

    assert program.name.startswith("llama_block_")
    assert len(program.launches) == 10
    assert len(program.buffers) == 22

    # Buffer roles preserved.
    input_bufs = [b for b in program.buffers if b.role == "input"]
    output_bufs = [b for b in program.buffers if b.role == "output"]
    assert len(input_bufs) == 3
    assert len(output_bufs) == 1


def test_cuda_backend_generates_valid_source():
    """Generated CUDA source has all expected kernel launches."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    backend = CudaBackend()
    program = backend.compile(plan)
    source = generate_source(program, mode="benchmark")

    assert "int main()" in source
    assert "fused_rmsnorm<<<" in source
    assert "triple_matmul<<<" in source
    assert "fused_rope<<<" in source
    assert "attention_qk<<<" in source
    assert "attention_softmax<<<" in source
    assert "attention_sv<<<" in source
    assert "matmul_residual_add<<<" in source
    assert "dual_matmul_silu_mul<<<" in source


@requires_cuda
def test_cuda_backend_run():
    """Full pipeline: plan → compile → run on GPU."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    backend = CudaBackend()
    program = backend.compile(plan)
    result = backend.run(program)

    assert "output" in result.outputs
    assert len(result.outputs["output"]) > 0
    for v in result.outputs["output"]:
        assert v == v, "NaN in output"


@requires_cuda
def test_cuda_backend_benchmark():
    """Benchmark returns valid timing."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    plan = plan_block(cfg)

    backend = CudaBackend()
    program = backend.compile(plan)
    result = backend.benchmark(program, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 10


@requires_cuda
def test_cuda_backend_tinyllama_dims():
    """Full pipeline with TinyLlama dimensions."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_dim=5632)
    plan = plan_block(cfg)

    backend = CudaBackend()
    program = backend.compile(plan)
    result = backend.benchmark(program, warmup=2, num_iters=3)

    assert result.time_ms > 0
