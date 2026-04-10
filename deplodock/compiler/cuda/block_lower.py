"""Lower a transformer block to a Program.

Takes model config (dimensions) and generates a Program with all
kernel sources, buffer allocations, and launch sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deplodock.compiler.cuda.kernels import (
    dual_matmul_silu_mul_source,
    fused_rmsnorm_source,
    fused_rope_source,
    matmul_residual_add_source,
    naive_attention_qk_source,
    naive_attention_softmax_source,
    naive_attention_sv_source,
    triple_matmul_source,
)
from deplodock.compiler.cuda.program import Buffer, Launch, Program


@dataclass
class BlockConfig:
    """Transformer block dimensions."""

    batch: int = 1
    seq_len: int = 32
    hidden_dim: int = 2048
    num_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_dim: int = 5632
    eps: float = 1e-5


def lower_block(cfg: BlockConfig) -> Program:
    """Generate a Program for a Llama-style transformer block.

    Kernel sequence (10 launches, 7 logical kernels):
      1. FusedRMSNorm(x, w1) → norm1
      2. TripleMatmul(norm1, Wq, Wk, Wv) → Q, K, V
      3. FusedRoPE(Q, K, cos, sin) → Q_rot, K_rot (in-place)
      4a-c. NaiveAttention(Q, K, V, scale) → attn_out (3 sub-launches)
      5a. MatmulResidualAdd(attn_out, Wo, x) → res1
      5b. FusedRMSNorm(res1, w2) → norm2
      6. DualMatmulSiLUMul(norm2, Wg, Wu) → activated
      7. MatmulResidualAdd(activated, Wd, res1) → output
    """
    BS = cfg.batch * cfg.seq_len
    D = cfg.hidden_dim
    NH = cfg.num_heads
    NKV = cfg.num_kv_heads
    HD = cfg.head_dim
    INTER = cfg.intermediate_dim
    S = cfg.seq_len
    scale = 1.0 / math.sqrt(HD)
    batch_heads = cfg.batch * NH

    # --- Buffers ---
    buffers = [
        # Inputs.
        Buffer("x", BS * D, role="input"),
        Buffer("cos", S * (HD // 2), role="input"),
        Buffer("sin", S * (HD // 2), role="input"),
        # Weights.
        Buffer("w_rms1", D, role="constant"),
        Buffer("Wq", D * NH * HD, role="constant"),
        Buffer("Wk", D * NKV * HD, role="constant"),
        Buffer("Wv", D * NKV * HD, role="constant"),
        Buffer("Wo", NH * HD * D, role="constant"),
        Buffer("w_rms2", D, role="constant"),
        Buffer("Wg", D * INTER, role="constant"),
        Buffer("Wu", D * INTER, role="constant"),
        Buffer("Wd", INTER * D, role="constant"),
        # Intermediates.
        Buffer("norm1", BS * D, role="scratch"),
        Buffer("Q", cfg.batch * NH * S * HD, role="scratch"),
        Buffer("K", cfg.batch * NKV * S * HD, role="scratch"),
        Buffer("V", cfg.batch * NKV * S * HD, role="scratch"),
        Buffer("scores", batch_heads * S * S, role="scratch"),
        Buffer("attn_out", BS * D, role="scratch"),
        Buffer("res1", BS * D, role="scratch"),
        Buffer("norm2", BS * D, role="scratch"),
        Buffer("activated", BS * INTER, role="scratch"),
        # Output.
        Buffer("output", BS * D, role="output"),
    ]

    # --- Kernel sources ---
    rmsnorm_src = fused_rmsnorm_source("fused_rmsnorm")
    rmsnorm2_src = fused_rmsnorm_source("fused_rmsnorm_2")
    triple_src = triple_matmul_source("triple_matmul")
    rope_src = fused_rope_source("fused_rope")
    qk_src = naive_attention_qk_source("attention_qk")
    softmax_src = naive_attention_softmax_source("attention_softmax")
    sv_src = naive_attention_sv_source("attention_sv")
    mra_src = matmul_residual_add_source("matmul_residual_add")
    mra2_src = matmul_residual_add_source("matmul_residual_add_2")
    dms_src = dual_matmul_silu_mul_source("dual_matmul_silu_mul")

    Nq = NH * HD
    Nk = NKV * HD
    Nv = NKV * HD
    max_n = max(Nq, Nk, Nv)
    total_rope = cfg.batch * S * NH * (HD // 2) + cfg.batch * S * NKV * (HD // 2)

    # --- Launches ---
    launches = [
        # 1. FusedRMSNorm(x, w_rms1) → norm1
        Launch(
            kernel_source=rmsnorm_src,
            kernel_name="fused_rmsnorm",
            grid=(BS, 1, 1),
            block=(256, 1, 1),
            args=["x", "w_rms1", "norm1", str(BS), str(D), f"{cfg.eps}f"],
        ),
        # 2. TripleMatmul(norm1, Wq, Wk, Wv) → Q, K, V
        Launch(
            kernel_source=triple_src,
            kernel_name="triple_matmul",
            grid=(_cd(max_n, 16), _cd(BS, 16), 3),
            block=(16, 16, 1),
            args=["norm1", "Wq", "Wk", "Wv", "Q", "K", "V", str(BS), str(D), str(Nq), str(Nk), str(Nv)],
        ),
        # 3. FusedRoPE(Q, K, cos, sin)
        Launch(
            kernel_source=rope_src,
            kernel_name="fused_rope",
            grid=(_cd(total_rope, 256), 1, 1),
            block=(256, 1, 1),
            args=["Q", "K", "cos", "sin", str(cfg.batch), str(S), str(NH), str(NKV), str(HD)],
        ),
        # 4a. QK^T + scale
        Launch(
            kernel_source=qk_src,
            kernel_name="attention_qk",
            grid=(_cd(S, 16), _cd(S, 16), batch_heads),
            block=(16, 16, 1),
            args=["Q", "K", "scores", str(batch_heads), str(S), str(HD), f"{scale}f"],
        ),
        # 4b. Row softmax
        Launch(
            kernel_source=softmax_src,
            kernel_name="attention_softmax",
            grid=(batch_heads * S, 1, 1),
            block=(256, 1, 1),
            args=["scores", str(batch_heads), str(S)],
        ),
        # 4c. Scores @ V
        Launch(
            kernel_source=sv_src,
            kernel_name="attention_sv",
            grid=(_cd(HD, 16), _cd(S, 16), batch_heads),
            block=(16, 16, 1),
            args=["scores", "V", "attn_out", str(batch_heads), str(S), str(HD)],
        ),
        # 5a. MatmulResidualAdd(attn_out, Wo, x) → res1
        Launch(
            kernel_source=mra_src,
            kernel_name="matmul_residual_add",
            grid=(_cd(D, 16), _cd(BS, 16), 1),
            block=(16, 16, 1),
            args=["attn_out", "Wo", "x", "res1", str(BS), str(D), str(NH * HD)],
        ),
        # 5b. FusedRMSNorm(res1, w_rms2) → norm2
        Launch(
            kernel_source=rmsnorm2_src,
            kernel_name="fused_rmsnorm_2",
            grid=(BS, 1, 1),
            block=(256, 1, 1),
            args=["res1", "w_rms2", "norm2", str(BS), str(D), f"{cfg.eps}f"],
        ),
        # 6. DualMatmulSiLUMul(norm2, Wg, Wu) → activated
        Launch(
            kernel_source=dms_src,
            kernel_name="dual_matmul_silu_mul",
            grid=(_cd(INTER, 16), _cd(BS, 16), 1),
            block=(16, 16, 1),
            args=["norm2", "Wg", "Wu", "activated", str(BS), str(INTER), str(D)],
        ),
        # 7. MatmulResidualAdd(activated, Wd, res1) → output
        Launch(
            kernel_source=mra2_src,
            kernel_name="matmul_residual_add_2",
            grid=(_cd(D, 16), _cd(BS, 16), 1),
            block=(16, 16, 1),
            args=["activated", "Wd", "res1", "output", str(BS), str(D), str(INTER)],
        ),
    ]

    return Program(
        name=f"llama_block_{D}x{S}",
        buffers=buffers,
        launches=launches,
    )


def _cd(a: int, b: int) -> int:
    """Ceil division."""
    return (a + b - 1) // b
