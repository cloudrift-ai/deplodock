"""Tests for the block runner — full transformer block execution on GPU."""

import pytest

from deplodock.compiler.cuda.block_lower import BlockConfig, lower_block
from deplodock.compiler.cuda.block_runner import generate_block_source, run_block
from deplodock.compiler.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


# ---- Lowering tests (no GPU) ----


def test_lower_block_produces_plan():
    """lower_block returns an ExecutionPlan with the expected structure."""
    cfg = BlockConfig(batch=1, seq_len=32, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_dim=128)
    plan = lower_block(cfg)

    assert len(plan.launches) == 10  # 7 logical kernels, attention has 3 sub-launches
    assert len(plan.input_names) == 3  # x, cos, sin
    assert len(plan.output_names) == 1  # output
    assert len(plan.constant_names) == 9  # w_rms1, Wq, Wk, Wv, Wo, w_rms2, Wg, Wu, Wd

    # Verify kernel names in order.
    kernel_names = [launch.kernel_name for launch in plan.launches]
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


def test_generate_block_source_compiles_check():
    """Generated .cu source has expected structure."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, intermediate_dim=32)
    source = generate_block_source(cfg)

    assert "int main()" in source
    assert "fused_rmsnorm<<<" in source
    assert "triple_matmul<<<" in source
    assert "fused_rope<<<" in source
    assert "attention_qk<<<" in source
    assert "attention_softmax<<<" in source
    assert "attention_sv<<<" in source
    assert "matmul_residual_add<<<" in source
    assert "dual_matmul_silu_mul<<<" in source
    assert "BLOCK_TIME_MS=" in source


# ---- GPU tests ----


@requires_cuda
def test_block_compiles_and_runs():
    """Full block program compiles with nvcc and runs without errors."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    result = run_block(cfg)

    assert result.kernel_time_ms is not None, "Kernel timing not parsed"
    assert result.kernel_time_ms > 0
    assert result.output is not None
    assert len(result.output) > 0


@requires_cuda
def test_block_output_is_finite():
    """Block output contains finite values (no NaN/Inf)."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, intermediate_dim=32)
    result = run_block(cfg)

    assert result.output is not None
    for i, v in enumerate(result.output):
        assert not (v != v), f"NaN at output[{i}]"  # NaN != NaN
        assert abs(v) < 1e6, f"Output[{i}] too large: {v}"


@requires_cuda
def test_block_tinyllama_dims():
    """Block runs with TinyLlama dimensions (small seq_len for speed)."""
    cfg = BlockConfig(batch=1, seq_len=4, hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_dim=5632)
    result = run_block(cfg)

    assert result.kernel_time_ms is not None
    assert result.output is not None
    expected_output_size = cfg.batch * cfg.seq_len * cfg.hidden_dim
    assert len(result.output) == expected_output_size
