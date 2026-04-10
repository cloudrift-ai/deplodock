"""Fused RoPE kernel: rotary position embeddings for Q and K."""

from __future__ import annotations


def fused_rope_source(name: str = "fused_rope") -> str:
    """Generate CUDA source for fused RoPE kernel.

    Applies rotary position embeddings to Q and K in one kernel.
    Each thread handles one (i, i+half) element pair.

    Layout: Q/K are [batch, seq, heads, head_dim].
    cos/sin are [1, seq, head_dim/2] (broadcast over batch and heads).
    """
    return f"""
__global__ void {name}(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int batch, int seq_len, int q_heads, int kv_heads, int head_dim
) {{
    int half_dim = head_dim / 2;
    int total_q = batch * seq_len * q_heads * half_dim;
    int total_k = batch * seq_len * kv_heads * half_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process Q elements.
    if (idx < total_q) {{
        int tmp = idx;
        int d = tmp % half_dim; tmp /= half_dim;
        int h = tmp % q_heads; tmp /= q_heads;
        int s = tmp % seq_len; tmp /= seq_len;
        int b = tmp;

        int base = ((b * seq_len + s) * q_heads + h) * head_dim;
        float q0 = Q[base + d];
        float q1 = Q[base + d + half_dim];

        int cs_idx = s * half_dim + d;
        float c = cos_cache[cs_idx];
        float sn = sin_cache[cs_idx];

        Q[base + d]            = q0 * c - q1 * sn;
        Q[base + d + half_dim] = q1 * c + q0 * sn;
    }}

    // Process K elements (separate grid region).
    int k_idx = idx - total_q;
    if (idx >= total_q && k_idx < total_k) {{
        int tmp = k_idx;
        int d = tmp % half_dim; tmp /= half_dim;
        int h = tmp % kv_heads; tmp /= kv_heads;
        int s = tmp % seq_len; tmp /= seq_len;
        int b = tmp;

        int base = ((b * seq_len + s) * kv_heads + h) * head_dim;
        float k0 = K[base + d];
        float k1 = K[base + d + half_dim];

        int cs_idx = s * half_dim + d;
        float c = cos_cache[cs_idx];
        float sn = sin_cache[cs_idx];

        K[base + d]            = k0 * c - k1 * sn;
        K[base + d + half_dim] = k1 * c + k0 * sn;
    }}
}}
"""
