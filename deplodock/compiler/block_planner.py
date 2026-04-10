"""Backend-agnostic transformer block planning.

Generates an ExecutionPlan describing a Llama-style transformer block:
which operations to run, which buffers they use, and in what order.
Knows nothing about CUDA, grid/block, or kernel source code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deplodock.compiler.plan import BufferSpec, ExecutionPlan, OpKernel


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


def plan_block(cfg: BlockConfig) -> ExecutionPlan:
    """Generate a backend-agnostic execution plan for a Llama-style block.

    10 operations (7 logical kernels, attention is 3 sub-ops):
      1. RMSNorm(x, w1) → norm1
      2. TripleMatmul(norm1, Wq, Wk, Wv) → Q, K, V
      3. RoPE(Q, K, cos, sin) → Q, K (in-place)
      4a-c. Attention(Q, K, V, scale) → attn_out
      5a. MatmulResidualAdd(attn_out, Wo, x) → res1
      5b. RMSNorm(res1, w2) → norm2
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

    buffers = [
        # Inputs.
        BufferSpec("x", (BS, D), role="input"),
        BufferSpec("cos", (S, HD // 2), role="input"),
        BufferSpec("sin", (S, HD // 2), role="input"),
        # Weights.
        BufferSpec("w_rms1", (D,), role="constant"),
        BufferSpec("Wq", (D, NH * HD), role="constant"),
        BufferSpec("Wk", (D, NKV * HD), role="constant"),
        BufferSpec("Wv", (D, NKV * HD), role="constant"),
        BufferSpec("Wo", (NH * HD, D), role="constant"),
        BufferSpec("w_rms2", (D,), role="constant"),
        BufferSpec("Wg", (D, INTER), role="constant"),
        BufferSpec("Wu", (D, INTER), role="constant"),
        BufferSpec("Wd", (INTER, D), role="constant"),
        # Intermediates.
        BufferSpec("norm1", (BS, D)),
        BufferSpec("Q", (cfg.batch, NH, S, HD)),
        BufferSpec("K", (cfg.batch, NKV, S, HD)),
        BufferSpec("V", (cfg.batch, NKV, S, HD)),
        BufferSpec("scores", (batch_heads, S, S)),
        BufferSpec("attn_out", (BS, D)),
        BufferSpec("res1", (BS, D)),
        BufferSpec("norm2", (BS, D)),
        BufferSpec("activated", (BS, INTER)),
        # Output.
        BufferSpec("output", (BS, D), role="output"),
    ]

    ops = [
        # 1. RMSNorm
        OpKernel(
            op="rmsnorm",
            inputs=["x", "w_rms1"],
            outputs=["norm1"],
            params={"rows": BS, "dim": D, "eps": cfg.eps},
        ),
        # 2. TripleMatmul (Q/K/V projections)
        OpKernel(
            op="triple_matmul",
            inputs=["norm1", "Wq", "Wk", "Wv"],
            outputs=["Q", "K", "V"],
            params={"M": BS, "K": D, "Nq": NH * HD, "Nk": NKV * HD, "Nv": NKV * HD},
        ),
        # 3. RoPE
        OpKernel(
            op="rope",
            inputs=["Q", "K", "cos", "sin"],
            outputs=["Q", "K"],
            params={
                "batch": cfg.batch,
                "seq_len": S,
                "q_heads": NH,
                "kv_heads": NKV,
                "head_dim": HD,
            },
        ),
        # 4a. Attention QK^T + scale
        OpKernel(
            op="attention_qk",
            inputs=["Q", "K"],
            outputs=["scores"],
            params={"batch_heads": batch_heads, "seq_len": S, "head_dim": HD, "scale": scale},
        ),
        # 4b. Attention softmax
        OpKernel(
            op="attention_softmax",
            inputs=["scores"],
            outputs=["scores"],
            params={"batch_heads": batch_heads, "seq_len": S},
        ),
        # 4c. Attention scores @ V
        OpKernel(
            op="attention_sv",
            inputs=["scores", "V"],
            outputs=["attn_out"],
            params={"batch_heads": batch_heads, "seq_len": S, "head_dim": HD},
        ),
        # 5a. MatmulResidualAdd(attn_out, Wo, x) → res1
        OpKernel(
            op="matmul_residual_add",
            inputs=["attn_out", "Wo", "x"],
            outputs=["res1"],
            params={"M": BS, "N": D, "K": NH * HD},
        ),
        # 5b. RMSNorm
        OpKernel(
            op="rmsnorm",
            inputs=["res1", "w_rms2"],
            outputs=["norm2"],
            params={"rows": BS, "dim": D, "eps": cfg.eps},
        ),
        # 6. DualMatmulSiLUMul
        OpKernel(
            op="dual_matmul_silu_mul",
            inputs=["norm2", "Wg", "Wu"],
            outputs=["activated"],
            params={"M": BS, "N": INTER, "K": D},
        ),
        # 7. MatmulResidualAdd(activated, Wd, res1) → output
        OpKernel(
            op="matmul_residual_add",
            inputs=["activated", "Wd", "res1"],
            outputs=["output"],
            params={"M": BS, "N": D, "K": INTER},
        ),
    ]

    return ExecutionPlan(
        name=f"llama_block_{D}x{S}",
        buffers=buffers,
        ops=ops,
    )
