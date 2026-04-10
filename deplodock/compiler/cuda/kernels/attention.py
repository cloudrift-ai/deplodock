"""Naive attention kernels: QK^T + scale, row softmax, scores @ V."""

from __future__ import annotations


def naive_attention_qk_source(name: str = "attention_qk") -> str:
    """Generate CUDA source for QK^T + scale.

    scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:]) * scale.
    Each thread computes one element of the scores matrix.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int batch_heads, int seq_len, int head_dim, float scale
) {{
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (bh < batch_heads && i < seq_len && j < seq_len) {{
        const float* q_row = Q + (bh * seq_len + i) * head_dim;
        const float* k_row = K + (bh * seq_len + j) * head_dim;

        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++)
            acc += q_row[d] * k_row[d];

        scores[bh * seq_len * seq_len + i * seq_len + j] = acc * scale;
    }}
}}
"""


def naive_attention_softmax_source(name: str = "attention_softmax") -> str:
    """Generate CUDA source for row-wise softmax.

    One block per row. Shared-memory parallel max, exp, sum, normalize.
    """
    return f"""
__global__ void {name}(
    float* __restrict__ scores,
    int batch_heads, int seq_len
) {{
    int row_idx = blockIdx.x;
    if (row_idx >= batch_heads * seq_len) return;

    float* row = scores + row_idx * seq_len;

    // Phase 1: find row max.
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        local_max = fmaxf(local_max, row[j]);

    // Warp reduce max.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float warp_vals[8];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_vals[warp_id] = local_max;
    __syncthreads();

    float row_max = -1e30f;
    if (threadIdx.x < blockDim.x / warpSize)
        row_max = warp_vals[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // Phase 2: exp(x - max) and sum.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {{
        float v = expf(row[j] - row_max);
        row[j] = v;
        local_sum += v;
    }}

    // Warp reduce sum.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    if (lane == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (threadIdx.x < blockDim.x / warpSize)
        row_sum = warp_vals[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Phase 3: normalize.
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        row[j] *= inv_sum;
}}
"""


def naive_attention_sv_source(name: str = "attention_sv") -> str:
    """Generate CUDA source for scores @ V.

    out[b,h,i,d] = dot(scores[b,h,i,:], V[b,h,:,d]).
    Each thread computes one element of the output.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ out,
    int batch_heads, int seq_len, int head_dim
) {{
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (bh < batch_heads && i < seq_len && d < head_dim) {{
        const float* score_row = scores + (bh * seq_len + i) * seq_len;
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
            acc += score_row[j] * V[(bh * seq_len + j) * head_dim + d];

        out[(bh * seq_len + i) * head_dim + d] = acc;
    }}
}}
"""
