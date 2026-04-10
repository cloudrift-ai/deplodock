"""Tests for fused CUDA kernel templates."""

import math

import pytest

from deplodock.compiler.backend.cuda.kernels import (
    dual_matmul_silu_mul_source,
    fused_rmsnorm_source,
    fused_rope_source,
    fused_silu_mul_source,
    matmul_residual_add_source,
    naive_attention_qk_source,
    naive_attention_softmax_source,
    naive_attention_sv_source,
    triple_matmul_source,
)
from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


# ---- Codegen tests (no GPU) ----


def test_rmsnorm_source_structure():
    src = fused_rmsnorm_source("test_rms")
    assert "__global__ void test_rms(" in src
    assert "rsqrtf" in src
    assert "__shfl_down_sync" in src
    assert "weight[i]" in src


def test_silu_mul_source_structure():
    src = fused_silu_mul_source("test_silu")
    assert "__global__ void test_silu(" in src
    assert "expf(-g)" in src
    assert "up[i]" in src


def test_rope_source_structure():
    src = fused_rope_source("test_rope")
    assert "__global__ void test_rope(" in src
    assert "cos_cache" in src
    assert "sin_cache" in src
    assert "half_dim" in src


def test_attention_qk_source_structure():
    src = naive_attention_qk_source("test_qk")
    assert "__global__ void test_qk(" in src
    assert "scale" in src
    assert "head_dim" in src


def test_attention_softmax_source_structure():
    src = naive_attention_softmax_source("test_sm")
    assert "__global__ void test_sm(" in src
    assert "expf" in src
    assert "__shfl_down_sync" in src


def test_attention_sv_source_structure():
    src = naive_attention_sv_source("test_sv")
    assert "__global__ void test_sv(" in src
    assert "head_dim" in src


def test_matmul_residual_add_source():
    src = matmul_residual_add_source("test_mra")
    assert "__global__ void test_mra(" in src
    assert "residual[row * N + col]" in src


def test_triple_matmul_source():
    src = triple_matmul_source("test_tm")
    assert "__global__ void test_tm(" in src
    assert "blockIdx.z" in src
    assert "Nq" in src and "Nk" in src and "Nv" in src


def test_dual_matmul_silu_mul_source():
    src = dual_matmul_silu_mul_source("test_dms")
    assert "__global__ void test_dms(" in src
    assert "gate_acc" in src
    assert "up_acc" in src
    assert "expf(-gate_acc)" in src


# ---- GPU correctness tests ----


def _compile_and_run(source: str, host_code: str) -> str:
    """Compile CUDA source with host code and run, return stdout."""
    import subprocess
    import tempfile

    full = source + "\n" + host_code
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(full)
        f.flush()
        out_path = f.name.replace(".cu", "")
        result = subprocess.run(
            ["nvcc", "-O2", f.name, "-o", out_path, "--use_fast_math"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")

    result = subprocess.run([out_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kernel execution failed:\n{result.stderr}")
    return result.stdout


@requires_cuda
def test_rmsnorm_gpu_correctness():
    """FusedRMSNorm kernel matches reference computation."""
    rows, dim = 4, 8
    eps = 1e-5

    # x = [1, 2, ..., 32], weight = [1, 1, ..., 1]
    x_data = [float(i + 1) for i in range(rows * dim)]
    w_data = [1.0] * dim

    # Reference: x * rsqrt(mean(x^2) + eps)
    expected = []
    for r in range(rows):
        row = x_data[r * dim : (r + 1) * dim]
        ss = sum(v * v for v in row) / dim
        scale = 1.0 / math.sqrt(ss + eps)
        expected.extend(v * scale * w for v, w in zip(row, w_data, strict=True))

    host = f"""
#include <cstdio>
#include <cmath>
int main() {{
    float h_x[] = {{{", ".join(f"{v}f" for v in x_data)}}};
    float h_w[] = {{{", ".join(f"{v}f" for v in w_data)}}};
    float h_out[{rows * dim}];

    float *d_x, *d_w, *d_out;
    cudaMalloc(&d_x, {rows * dim} * sizeof(float));
    cudaMalloc(&d_w, {dim} * sizeof(float));
    cudaMalloc(&d_out, {rows * dim} * sizeof(float));
    cudaMemcpy(d_x, h_x, {rows * dim} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, {dim} * sizeof(float), cudaMemcpyHostToDevice);

    fused_rmsnorm<<<{rows}, 256>>>(d_x, d_w, d_out, {rows}, {dim}, {eps}f);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, {rows * dim} * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < {rows * dim}; i++) printf("%.6f\\n", h_out[i]);
    return 0;
}}
"""
    src = fused_rmsnorm_source()
    output = _compile_and_run(src, host)
    values = [float(line) for line in output.strip().split("\n")]

    for i, (got, exp) in enumerate(zip(values, expected, strict=True)):
        assert abs(got - exp) < 1e-3, f"Mismatch at [{i}]: got {got}, expected {exp}"


@requires_cuda
def test_silu_mul_gpu_correctness():
    """FusedSiLUMul kernel matches reference computation."""
    n = 8
    gate = [float(i - 4) for i in range(n)]  # [-4, -3, ..., 3]
    up = [1.0] * n

    expected = [g / (1.0 + math.exp(-g)) * u for g, u in zip(gate, up, strict=True)]

    host = f"""
#include <cstdio>
#include <cmath>
int main() {{
    float h_gate[] = {{{", ".join(f"{v}f" for v in gate)}}};
    float h_up[] = {{{", ".join(f"{v}f" for v in up)}}};
    float h_out[{n}];

    float *d_gate, *d_up, *d_out;
    cudaMalloc(&d_gate, {n} * sizeof(float));
    cudaMalloc(&d_up, {n} * sizeof(float));
    cudaMalloc(&d_out, {n} * sizeof(float));
    cudaMemcpy(d_gate, h_gate, {n} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, {n} * sizeof(float), cudaMemcpyHostToDevice);

    fused_silu_mul<<<1, 256>>>(d_gate, d_up, d_out, {n});
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, {n} * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < {n}; i++) printf("%.6f\\n", h_out[i]);
    return 0;
}}
"""
    src = fused_silu_mul_source()
    output = _compile_and_run(src, host)
    values = [float(line) for line in output.strip().split("\n")]

    for i, (got, exp) in enumerate(zip(values, expected, strict=True)):
        assert abs(got - exp) < 1e-3, f"Mismatch at [{i}]: got {got}, expected {exp}"


@requires_cuda
def test_dual_matmul_silu_mul_gpu_correctness():
    """DualMatmulSiLUMul: silu(A@Wg) * (A@Wu) in one kernel."""
    M, N, K = 2, 3, 4

    # Simple data.
    A = [float(i + 1) for i in range(M * K)]
    Wg = [0.1 * float(i + 1) for i in range(K * N)]
    Wu = [0.05 * float(i + 1) for i in range(K * N)]

    # Reference.
    expected = []
    for i in range(M):
        for j in range(N):
            gate_acc = sum(A[i * K + k] * Wg[k * N + j] for k in range(K))
            up_acc = sum(A[i * K + k] * Wu[k * N + j] for k in range(K))
            silu_gate = gate_acc / (1.0 + math.exp(-gate_acc))
            expected.append(silu_gate * up_acc)

    host = f"""
#include <cstdio>
#include <cmath>
int main() {{
    float h_A[] = {{{", ".join(f"{v}f" for v in A)}}};
    float h_Wg[] = {{{", ".join(f"{v}f" for v in Wg)}}};
    float h_Wu[] = {{{", ".join(f"{v}f" for v in Wu)}}};
    float h_out[{M * N}];

    float *d_A, *d_Wg, *d_Wu, *d_out;
    cudaMalloc(&d_A, {M * K} * sizeof(float));
    cudaMalloc(&d_Wg, {K * N} * sizeof(float));
    cudaMalloc(&d_Wu, {K * N} * sizeof(float));
    cudaMalloc(&d_out, {M * N} * sizeof(float));
    cudaMemcpy(d_A, h_A, {M * K} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wg, h_Wg, {K * N} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wu, h_Wu, {K * N} * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(({N} + 15) / 16, ({M} + 15) / 16);
    dual_matmul_silu_mul<<<grid, block>>>(d_A, d_Wg, d_Wu, d_out, {M}, {N}, {K});
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, {M * N} * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < {M * N}; i++) printf("%.6f\\n", h_out[i]);
    return 0;
}}
"""
    src = dual_matmul_silu_mul_source()
    output = _compile_and_run(src, host)
    values = [float(line) for line in output.strip().split("\n")]

    for i, (got, exp) in enumerate(zip(values, expected, strict=True)):
        assert abs(got - exp) < 1e-3, f"Mismatch at [{i}]: got {got}, expected {exp}"


@requires_cuda
def test_naive_attention_gpu_correctness():
    """NaiveAttention (QK + softmax + SV) matches reference."""
    batch_heads, seq_len, head_dim = 1, 4, 8
    scale = 1.0 / math.sqrt(head_dim)
    total = batch_heads * seq_len * head_dim

    Q = [0.1 * float((i % 17) - 8) for i in range(total)]
    K = [0.1 * float((i % 13) - 6) for i in range(total)]
    V = [0.1 * float((i % 11) - 5) for i in range(total)]

    # Reference: QK^T * scale → softmax → @V.
    scores = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dot = sum(Q[i * head_dim + d] * K[j * head_dim + d] for d in range(head_dim))
            scores[i][j] = dot * scale

    # Softmax per row.
    for i in range(seq_len):
        mx = max(scores[i])
        exps = [math.exp(s - mx) for s in scores[i]]
        s = sum(exps)
        scores[i] = [e / s for e in exps]

    # Scores @ V.
    expected = []
    for i in range(seq_len):
        for d in range(head_dim):
            val = sum(scores[i][j] * V[j * head_dim + d] for j in range(seq_len))
            expected.append(val)

    scores_size = batch_heads * seq_len * seq_len

    host = f"""
#include <cstdio>
#include <cmath>
int main() {{
    float h_Q[] = {{{", ".join(f"{v}f" for v in Q)}}};
    float h_K[] = {{{", ".join(f"{v}f" for v in K)}}};
    float h_V[] = {{{", ".join(f"{v}f" for v in V)}}};
    float h_out[{total}];

    float *d_Q, *d_K, *d_V, *d_scores, *d_out;
    cudaMalloc(&d_Q, {total} * sizeof(float));
    cudaMalloc(&d_K, {total} * sizeof(float));
    cudaMalloc(&d_V, {total} * sizeof(float));
    cudaMalloc(&d_scores, {scores_size} * sizeof(float));
    cudaMalloc(&d_out, {total} * sizeof(float));
    cudaMemcpy(d_Q, h_Q, {total} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, {total} * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, {total} * sizeof(float), cudaMemcpyHostToDevice);

    // QK^T + scale.
    dim3 qk_block(16, 16);
    dim3 qk_grid(({seq_len} + 15) / 16, ({seq_len} + 15) / 16, {batch_heads});
    attention_qk<<<qk_grid, qk_block>>>(d_Q, d_K, d_scores, {batch_heads}, {seq_len}, {head_dim}, {scale}f);

    // Softmax.
    attention_softmax<<<{batch_heads * seq_len}, 256>>>(d_scores, {batch_heads}, {seq_len});

    // Scores @ V.
    dim3 sv_block(16, 16);
    dim3 sv_grid(({head_dim} + 15) / 16, ({seq_len} + 15) / 16, {batch_heads});
    attention_sv<<<sv_grid, sv_block>>>(d_scores, d_V, d_out, {batch_heads}, {seq_len}, {head_dim});

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, {total} * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < {total}; i++) printf("%.6f\\n", h_out[i]);
    return 0;
}}
"""
    src = naive_attention_qk_source() + naive_attention_softmax_source() + naive_attention_sv_source()
    output = _compile_and_run(src, host)
    values = [float(line) for line in output.strip().split("\n")]

    for i, (got, exp) in enumerate(zip(values, expected, strict=True)):
        assert abs(got - exp) < 1e-2, f"Mismatch at [{i}]: got {got}, expected {exp}"
