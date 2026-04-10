"""Matmul kernel variants: naive, residual add, triple, dual+SiLU."""

from __future__ import annotations


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
