"""Tests for block lowering and execution via Program abstraction."""

import pytest

from deplodock.compiler.cuda.block_lower import BlockConfig, lower_block
from deplodock.compiler.cuda.program import benchmark_program, generate_source, run_program
from deplodock.compiler.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


# ---- Lowering tests (no GPU) ----


def test_lower_block_produces_program():
    """lower_block returns a Program with the expected structure."""
    cfg = BlockConfig(batch=1, seq_len=32, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_dim=128)
    prog = lower_block(cfg)

    assert len(prog.launches) == 10  # 7 logical kernels, attention has 3 sub-launches
    assert prog.name.startswith("llama_block_")

    # Verify buffer roles.
    input_bufs = [b for b in prog.buffers if b.role == "input"]
    output_bufs = [b for b in prog.buffers if b.role == "output"]
    const_bufs = [b for b in prog.buffers if b.role == "constant"]
    assert len(input_bufs) == 3  # x, cos, sin
    assert len(output_bufs) == 1  # output
    assert len(const_bufs) == 9  # w_rms1, Wq, Wk, Wv, Wo, w_rms2, Wg, Wu, Wd

    # Verify kernel names in order.
    kernel_names = [launch.kernel_name for launch in prog.launches]
    expected = [
        "fused_rmsnorm",  # 1. RMSNorm
        "triple_matmul",  # 2. Q/K/V projections
        "fused_rope",  # 3. RoPE
        "attention_qk",  # 4a. QK^T + scale
        "attention_softmax",  # 4b. Softmax
        "attention_sv",  # 4c. Scores @ V
        "matmul_residual_add",  # 5a. Wo matmul + residual
        "fused_rmsnorm_2",  # 5b. RMSNorm
        "dual_matmul_silu_mul",  # 6. Gate+Up+SiLU*Mul
        "matmul_residual_add_2",  # 7. Down matmul + residual
    ]
    assert kernel_names == expected


def test_generate_block_source():
    """Generated .cu source has expected structure."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    prog = lower_block(cfg)
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
    """Full block program compiles with nvcc and runs without errors."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    prog = lower_block(cfg)
    result = run_program(prog)

    assert "output" in result.outputs
    assert len(result.outputs["output"]) > 0


@requires_cuda
def test_block_output_is_finite():
    """Block output contains finite values (no NaN/Inf)."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    prog = lower_block(cfg)
    result = run_program(prog)

    assert "output" in result.outputs
    for i, v in enumerate(result.outputs["output"]):
        assert v == v, f"NaN at output[{i}]"  # NaN != NaN
        assert abs(v) < 1e6, f"Output[{i}] too large: {v}"


@requires_cuda
def test_block_benchmark_returns_timing():
    """Benchmark mode returns valid timing."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    prog = lower_block(cfg)
    result = benchmark_program(prog, warmup=2, num_iters=5)

    assert result.time_ms > 0
    assert result.num_launches == 10


@requires_cuda
def test_block_tinyllama_dims():
    """Block runs with TinyLlama dimensions (small seq_len for speed)."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_dim=5632)
    prog = lower_block(cfg)
    result = benchmark_program(prog, warmup=2, num_iters=3)

    assert result.time_ms > 0
