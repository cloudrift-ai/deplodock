"""Lower a transformer block to a 7-kernel execution plan.

Takes model config (dimensions) and generates an ExecutionPlan with buffer
allocations and kernel launches. Does not walk the graph generically —
recognizes the Llama-style transformer block structure directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.cuda.ir import BufferAlloc, ExecutionPlan, KernelLaunch


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


def lower_block(cfg: BlockConfig) -> ExecutionPlan:
    """Generate a 7-kernel execution plan for a Llama-style transformer block.

    Kernel sequence:
      1. FusedRMSNorm(x, w1) → norm1
      2. TripleMatmul(norm1, Wq, Wk, Wv) → Q, K, V
      3. FusedRoPE(Q, K, cos, sin) → Q_rot, K_rot (in-place)
      4. NaiveAttention(Q, K, V, scale) → attn_out
         (3 sub-kernels: QK^T+scale, softmax, scores@V)
      5. MatmulResidualAdd(attn_out, Wo, x) → res1
         + FusedRMSNorm(res1, w2) → norm2
      6. DualMatmulSiLUMul(norm2, Wg, Wu) → activated
      7. MatmulResidualAdd(activated, Wd, res1) → output
    """
    BS = cfg.batch * cfg.seq_len  # flattened batch*seq
    D = cfg.hidden_dim
    NH = cfg.num_heads
    NKV = cfg.num_kv_heads
    HD = cfg.head_dim
    INTER = cfg.intermediate_dim
    S = cfg.seq_len

    # --- Buffer allocations ---
    buffers = [
        # Inputs (externally provided).
        BufferAlloc("x", (BS, D)),
        BufferAlloc("cos", (1, S, HD // 2)),
        BufferAlloc("sin", (1, S, HD // 2)),
        # Weights (constants).
        BufferAlloc("w_rms1", (D,)),
        BufferAlloc("Wq", (D, NH * HD)),
        BufferAlloc("Wk", (D, NKV * HD)),
        BufferAlloc("Wv", (D, NKV * HD)),
        BufferAlloc("Wo", (NH * HD, D)),
        BufferAlloc("w_rms2", (D,)),
        BufferAlloc("Wg", (D, INTER)),
        BufferAlloc("Wu", (D, INTER)),
        BufferAlloc("Wd", (INTER, D)),
        # Intermediates.
        BufferAlloc("norm1", (BS, D)),
        BufferAlloc("Q", (cfg.batch, NH, S, HD)),
        BufferAlloc("K", (cfg.batch, NKV, S, HD)),
        BufferAlloc("V", (cfg.batch, NKV, S, HD)),
        BufferAlloc("scores", (cfg.batch * NH, S, S)),  # attention scratch
        BufferAlloc("attn_out", (BS, D)),
        BufferAlloc("res1", (BS, D)),
        BufferAlloc("norm2", (BS, D)),
        BufferAlloc("activated", (BS, INTER)),
        BufferAlloc("output", (BS, D)),
    ]

    input_names = ["x", "cos", "sin"]
    output_names = ["output"]
    constant_names = ["w_rms1", "Wq", "Wk", "Wv", "Wo", "w_rms2", "Wg", "Wu", "Wd"]

    # --- Kernel launches ---
    launches = []

    # 1. FusedRMSNorm(x, w_rms1) → norm1
    launches.append(
        KernelLaunch(
            kernel_name="fused_rmsnorm",
            input_buffers=["x", "w_rms1"],
            output_buffers=["norm1"],
            grid=(BS, 1, 1),
            block=(256, 1, 1),
        )
    )

    # 2. TripleMatmul(norm1, Wq, Wk, Wv) → Q, K, V
    # Grid partitioned by blockIdx.z: 0=Q, 1=K, 2=V.
    # Use max(Nq, Nk, Nv) for grid.x to cover all outputs.
    Nq, Nk, Nv = NH * HD, NKV * HD, NKV * HD
    max_n = max(Nq, Nk, Nv)
    launches.append(
        KernelLaunch(
            kernel_name="triple_matmul",
            input_buffers=["norm1", "Wq", "Wk", "Wv"],
            output_buffers=["Q", "K", "V"],
            grid=(_ceil_div(max_n, 16), _ceil_div(BS, 16), 3),
            block=(16, 16, 1),
        )
    )

    # 3. FusedRoPE(Q, K, cos, sin) → Q_rot, K_rot (in-place).
    total_q = cfg.batch * S * NH * (HD // 2)
    total_k = cfg.batch * S * NKV * (HD // 2)
    total_rope = total_q + total_k
    launches.append(
        KernelLaunch(
            kernel_name="fused_rope",
            input_buffers=["Q", "K", "cos", "sin"],
            output_buffers=["Q", "K"],  # in-place
            grid=(_ceil_div(total_rope, 256), 1, 1),
            block=(256, 1, 1),
        )
    )

    # 4. NaiveAttention: 3 sub-kernels.
    batch_heads = cfg.batch * NH
    # For GQA: expand K/V heads to match Q heads. For simplicity in naive
    # attention, we handle this in the kernel launch by repeating KV heads.
    # This is a TODO for proper GQA support.

    # 4a. QK^T + scale.
    launches.append(
        KernelLaunch(
            kernel_name="attention_qk",
            input_buffers=["Q", "K"],
            output_buffers=["scores"],
            grid=(_ceil_div(S, 16), _ceil_div(S, 16), batch_heads),
            block=(16, 16, 1),
        )
    )

    # 4b. Row softmax.
    launches.append(
        KernelLaunch(
            kernel_name="attention_softmax",
            input_buffers=["scores"],
            output_buffers=["scores"],  # in-place
            grid=(batch_heads * S, 1, 1),
            block=(256, 1, 1),
        )
    )

    # 4c. Scores @ V.
    launches.append(
        KernelLaunch(
            kernel_name="attention_sv",
            input_buffers=["scores", "V"],
            output_buffers=["attn_out"],
            grid=(_ceil_div(HD, 16), _ceil_div(S, 16), batch_heads),
            block=(16, 16, 1),
        )
    )

    # 5a. MatmulResidualAdd(attn_out, Wo, x) → res1
    launches.append(
        KernelLaunch(
            kernel_name="matmul_residual_add",
            input_buffers=["attn_out", "Wo", "x"],
            output_buffers=["res1"],
            grid=(_ceil_div(D, 16), _ceil_div(BS, 16), 1),
            block=(16, 16, 1),
        )
    )

    # 5b. FusedRMSNorm(res1, w_rms2) → norm2
    launches.append(
        KernelLaunch(
            kernel_name="fused_rmsnorm_2",
            input_buffers=["res1", "w_rms2"],
            output_buffers=["norm2"],
            grid=(BS, 1, 1),
            block=(256, 1, 1),
        )
    )

    # 6. DualMatmulSiLUMul(norm2, Wg, Wu) → activated
    launches.append(
        KernelLaunch(
            kernel_name="dual_matmul_silu_mul",
            input_buffers=["norm2", "Wg", "Wu"],
            output_buffers=["activated"],
            grid=(_ceil_div(INTER, 16), _ceil_div(BS, 16), 1),
            block=(16, 16, 1),
        )
    )

    # 7. MatmulResidualAdd(activated, Wd, res1) → output
    launches.append(
        KernelLaunch(
            kernel_name="matmul_residual_add_2",
            input_buffers=["activated", "Wd", "res1"],
            output_buffers=["output"],
            grid=(_ceil_div(D, 16), _ceil_div(BS, 16), 1),
            block=(16, 16, 1),
        )
    )

    return ExecutionPlan(
        kernels=[],  # kernels are source strings, not KernelDef (for fused kernels)
        buffers=buffers,
        launches=launches,
        input_names=input_names,
        output_names=output_names,
        constant_names=constant_names,
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
