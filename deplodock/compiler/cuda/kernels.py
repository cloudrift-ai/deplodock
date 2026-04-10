"""Fused CUDA kernel templates for transformer block ops.

Each function returns a KernelDef (or CUDA source string for complex kernels
that use features beyond the IR, like warp shuffles).
"""

from __future__ import annotations


def fused_rmsnorm_source(name: str = "fused_rmsnorm") -> str:
    """Generate CUDA source for fused RMSNorm kernel.

    One block per row. Warp-parallel sum of x², warp shuffle for row total,
    rsqrt(mean + eps), write x[i] * rsqrt_val * weight[i].

    Params: x (input), weight, out, rows, dim, eps.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int rows, int dim, float eps
) {{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * dim;
    float* row_out = out + row * dim;

    // Phase 1: compute sum of squares (parallel reduction).
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {{
        float v = row_x[i];
        local_ss += v * v;
    }}

    // Warp-level reduction.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);

    // Cross-warp reduction via shared memory.
    __shared__ float warp_sums[8];  // up to 256 threads = 8 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = local_ss;
    __syncthreads();

    float ss = 0.0f;
    if (threadIdx.x < blockDim.x / warpSize)
        ss = warp_sums[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, offset);

    // Broadcast rsqrt to all threads.
    __shared__ float s_rsqrt;
    if (threadIdx.x == 0)
        s_rsqrt = rsqrtf(ss / (float)dim + eps);
    __syncthreads();

    float scale = s_rsqrt;

    // Phase 2: write normalized + scaled output.
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        row_out[i] = row_x[i] * scale * weight[i];
}}
"""


def fused_silu_mul_source(name: str = "fused_silu_mul") -> str:
    """Generate CUDA source for fused SiLU(gate) * up kernel.

    One thread per element. out[i] = silu(gate[i]) * up[i].
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }}
}}
"""


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


def matmul_residual_add_source(name: str = "matmul_residual_add") -> str:
    """Generate CUDA source for matmul + residual add.

    out[i,j] = dot(A[i,:], B[:,j]) + residual[i,j].
    Naive: one thread per output element.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {{
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[row * K + k] * B[k * N + col];
        out[row * N + col] = acc + residual[row * N + col];
    }}
}}
"""


def triple_matmul_source(name: str = "triple_matmul") -> str:
    """Generate CUDA source for triple matmul (Q/K/V projections).

    Computes Q=A@Wq, K=A@Wk, V=A@Wv in a single kernel launch.
    Grid is partitioned: blockIdx.z selects which output (0=Q, 1=K, 2=V).
    All three share the same input A.
    Naive: one thread per output element.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ A,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ Q,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    int M, int K,
    int Nq, int Nk, int Nv
) {{
    int which = blockIdx.z;  // 0=Q, 1=K, 2=V
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    const float* W;
    float* out;
    int N;

    if (which == 0) {{ W = Wq; out = Q;     N = Nq; }}
    else if (which == 1) {{ W = Wk; out = K_out; N = Nk; }}
    else {{ W = Wv; out = V_out; N = Nv; }}

    if (row < M && col < N) {{
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[row * K + k] * W[k * N + col];
        out[row * N + col] = acc;
    }}
}}
"""


def dual_matmul_silu_mul_source(name: str = "dual_matmul_silu_mul") -> str:
    """Generate CUDA source for dual matmul + SiLU + mul.

    out[i,j] = silu(dot(A[i,:], Wg[:,j])) * dot(A[i,:], Wu[:,j]).
    Gate and up are never written to global memory.
    Naive: one thread per output element.
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ A,
    const float* __restrict__ Wg,
    const float* __restrict__ Wu,
    float* __restrict__ out,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {{
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int k = 0; k < K; k++) {{
            float a = A[row * K + k];
            gate_acc += a * Wg[k * N + col];
            up_acc   += a * Wu[k * N + col];
        }}
        // silu(gate) * up — gate and up stay in registers.
        out[row * N + col] = (gate_acc / (1.0f + expf(-gate_acc))) * up_acc;
    }}
}}
"""


def naive_matmul_source(name: str = "naive_matmul") -> str:
    """Generate CUDA source for a simple matmul. One thread per output element."""
    return f"""
__global__ void {name}(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {{
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }}
}}
"""
