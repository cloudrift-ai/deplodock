"""Generate and run a complete .cu program for a transformer block.

Produces a single .cu file with all kernel definitions and a host main()
that allocates buffers, loads weights, launches kernels, and measures timing.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass

from deplodock.compiler.cuda.block_lower import BlockConfig, lower_block
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

logger = logging.getLogger(__name__)


@dataclass
class BlockResult:
    """Result of running a transformer block."""

    output: list[float] | None = None
    kernel_time_ms: float | None = None
    correct: bool | None = None
    max_error: float | None = None


def generate_block_source(cfg: BlockConfig) -> str:
    """Generate a complete .cu program for one transformer block forward pass."""
    plan = lower_block(cfg)

    BS = cfg.batch * cfg.seq_len
    D = cfg.hidden_dim
    NH = cfg.num_heads
    NKV = cfg.num_kv_heads
    HD = cfg.head_dim
    INTER = cfg.intermediate_dim
    S = cfg.seq_len
    scale = 1.0 / math.sqrt(HD)

    # --- Kernel sources ---
    kernels = "\n".join(
        [
            fused_rmsnorm_source("fused_rmsnorm"),
            fused_rmsnorm_source("fused_rmsnorm_2"),
            triple_matmul_source("triple_matmul"),
            fused_rope_source("fused_rope"),
            naive_attention_qk_source("attention_qk"),
            naive_attention_softmax_source("attention_softmax"),
            naive_attention_sv_source("attention_sv"),
            matmul_residual_add_source("matmul_residual_add"),
            matmul_residual_add_source("matmul_residual_add_2"),
            dual_matmul_silu_mul_source("dual_matmul_silu_mul"),
        ]
    )

    # --- Buffer sizes ---
    buf_sizes = {
        "x": BS * D,
        "cos": S * (HD // 2),
        "sin": S * (HD // 2),
        "w_rms1": D,
        "Wq": D * NH * HD,
        "Wk": D * NKV * HD,
        "Wv": D * NKV * HD,
        "Wo": NH * HD * D,
        "w_rms2": D,
        "Wg": D * INTER,
        "Wu": D * INTER,
        "Wd": INTER * D,
        "norm1": BS * D,
        "Q": cfg.batch * NH * S * HD,
        "K": cfg.batch * NKV * S * HD,
        "V": cfg.batch * NKV * S * HD,
        "scores": cfg.batch * NH * S * S,
        "attn_out": BS * D,
        "res1": BS * D,
        "norm2": BS * D,
        "activated": BS * INTER,
        "output": BS * D,
    }

    max_n = max(NH * HD, NKV * HD)
    total_rope = cfg.batch * S * NH * (HD // 2) + cfg.batch * S * NKV * (HD // 2)
    batch_heads = cfg.batch * NH

    # --- Host code ---
    alloc_lines = []
    for name, size in buf_sizes.items():
        alloc_lines.append(f"    float *d_{name};")
        alloc_lines.append(f"    cudaMalloc(&d_{name}, {size} * sizeof(float));")

    alloc_code = "\n".join(alloc_lines)

    # Initialize inputs and weights with small random-ish values.
    init_lines = []
    for name, size in buf_sizes.items():
        if name in plan.input_names or name in plan.constant_names:
            init_lines.append(f"    {{ float* h = (float*)malloc({size} * sizeof(float));")
            init_lines.append(f"      for (int i = 0; i < {size}; i++) h[i] = 0.01f * ((i * 7 + 13) % 101 - 50);")
            init_lines.append(f"      cudaMemcpy(d_{name}, h, {size} * sizeof(float), cudaMemcpyHostToDevice);")
            init_lines.append("      free(h); }")
    init_code = "\n".join(init_lines)

    # Kernel launches.
    def ceil_div(a, b):
        return f"(({a} + {b} - 1) / {b})"

    launches = f"""
    // 1. FusedRMSNorm(x, w_rms1) -> norm1
    fused_rmsnorm<<<{BS}, 256>>>(d_x, d_w_rms1, d_norm1, {BS}, {D}, {cfg.eps}f);

    // 2. TripleMatmul(norm1, Wq, Wk, Wv) -> Q, K, V
    {{
        dim3 tm_block(16, 16);
        dim3 tm_grid({ceil_div(max_n, 16)}, {ceil_div(BS, 16)}, 3);
        triple_matmul<<<tm_grid, tm_block>>>(
            d_norm1, d_Wq, d_Wk, d_Wv, d_Q, d_K, d_V,
            {BS}, {D}, {NH * HD}, {NKV * HD}, {NKV * HD});
    }}

    // 3. FusedRoPE(Q, K, cos, sin)
    fused_rope<<<{ceil_div(total_rope, 256)}, 256>>>(
        d_Q, d_K, d_cos, d_sin,
        {cfg.batch}, {S}, {NH}, {NKV}, {HD});

    // 4a. Attention QK^T + scale
    {{
        dim3 qk_block(16, 16);
        dim3 qk_grid({ceil_div(S, 16)}, {ceil_div(S, 16)}, {batch_heads});
        attention_qk<<<qk_grid, qk_block>>>(d_Q, d_K, d_scores, {batch_heads}, {S}, {HD}, {scale}f);
    }}

    // 4b. Attention softmax
    attention_softmax<<<{batch_heads * S}, 256>>>(d_scores, {batch_heads}, {S});

    // 4c. Attention scores @ V
    {{
        dim3 sv_block(16, 16);
        dim3 sv_grid({ceil_div(HD, 16)}, {ceil_div(S, 16)}, {batch_heads});
        attention_sv<<<sv_grid, sv_block>>>(d_scores, d_V, d_attn_out, {batch_heads}, {S}, {HD});
    }}

    // 5a. MatmulResidualAdd(attn_out, Wo, x) -> res1
    {{
        dim3 mra_block(16, 16);
        dim3 mra_grid({ceil_div(D, 16)}, {ceil_div(BS, 16)});
        matmul_residual_add<<<mra_grid, mra_block>>>(d_attn_out, d_Wo, d_x, d_res1, {BS}, {D}, {NH * HD});
    }}

    // 5b. FusedRMSNorm(res1, w_rms2) -> norm2
    fused_rmsnorm_2<<<{BS}, 256>>>(d_res1, d_w_rms2, d_norm2, {BS}, {D}, {cfg.eps}f);

    // 6. DualMatmulSiLUMul(norm2, Wg, Wu) -> activated
    {{
        dim3 dms_block(16, 16);
        dim3 dms_grid({ceil_div(INTER, 16)}, {ceil_div(BS, 16)});
        dual_matmul_silu_mul<<<dms_grid, dms_block>>>(d_norm2, d_Wg, d_Wu, d_activated, {BS}, {INTER}, {D});
    }}

    // 7. MatmulResidualAdd(activated, Wd, res1) -> output
    {{
        dim3 mra2_block(16, 16);
        dim3 mra2_grid({ceil_div(D, 16)}, {ceil_div(BS, 16)});
        matmul_residual_add_2<<<mra2_grid, mra2_block>>>(d_activated, d_Wd, d_res1, d_output, {BS}, {D}, {INTER});
    }}
"""

    # Read back output.
    output_size = BS * D
    readback = f"""
    cudaDeviceSynchronize();

    float* h_output = (float*)malloc({output_size} * sizeof(float));
    cudaMemcpy(h_output, d_output, {output_size} * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < {output_size}; i++) printf("%.6f\\n", h_output[i]);
    free(h_output);
"""

    # Timing harness.
    timing = f"""
    // Warmup.
    for (int iter = 0; iter < 3; iter++) {{
{_indent(launches, 8)}
    }}
    cudaDeviceSynchronize();

    // Timed runs.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iters = 10;
    cudaEventRecord(start);
    for (int iter = 0; iter < num_iters; iter++) {{
{_indent(launches, 8)}
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    printf("BLOCK_TIME_MS=%.4f\\n", total_ms / num_iters);
    printf("BLOCK_KERNELS=11\\n");
"""

    # Free buffers.
    free_lines = [f"    cudaFree(d_{name});" for name in buf_sizes]
    free_code = "\n".join(free_lines)

    source = f"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// --- Kernel definitions ---
{kernels}

// --- Host program ---
int main() {{
    // Allocate device buffers.
{alloc_code}

    // Initialize inputs and weights.
{init_code}

    // Run and time.
{timing}

    // Read back output.
{readback}

    // Cleanup.
{free_code}

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}}
"""
    return source


def run_block(cfg: BlockConfig) -> BlockResult:
    """Generate, compile, and run a transformer block."""
    source = generate_block_source(cfg)

    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(source)
        cu_path = f.name

    out_path = cu_path.replace(".cu", "")

    logger.info("Compiling block program (%d chars)...", len(source))
    result = subprocess.run(
        ["nvcc", "-O2", "--use_fast_math", cu_path, "-o", out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("nvcc failed:\n%s", result.stderr)
        return BlockResult()

    logger.info("Running block program...")
    result = subprocess.run([out_path], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.error("Block execution failed:\n%s", result.stderr)
        return BlockResult()

    # Parse output.
    lines = result.stdout.strip().split("\n")
    kernel_time_ms = None
    output_values = []

    for line in lines:
        if line.startswith("BLOCK_TIME_MS="):
            kernel_time_ms = float(line.split("=")[1])
        elif line.startswith("BLOCK_KERNELS="):
            pass
        else:
            try:
                output_values.append(float(line))
            except ValueError:
                pass

    return BlockResult(
        output=output_values if output_values else None,
        kernel_time_ms=kernel_time_ms,
    )


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))
