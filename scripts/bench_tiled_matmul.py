#!/usr/bin/env python3
"""Microbenchmark: naive thread-per-output vs tiled SGEMM on the gate_proj shape.

Shape: Y = X @ W^T  where X:(32, 2048), W:(5632, 2048), Y:(32, 5632).
This is the hot gate_proj kernel (k_n361 / k_n357) from TinyLlama's MLP at seq=32.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    import sys

    print("cupy not installed; pip install -e '.[compile]'")
    sys.exit(1)

M, N, K = 32, 5632, 2048
BM, BN, BK = 32, 64, 32
TM, TN = 2, 8
THREADS = 128

NAIVE_SRC = r"""
extern "C" __global__
__launch_bounds__(256) void naive(const float* __restrict__ W,
                                   const float* __restrict__ X,
                                   float* __restrict__ OUT) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 32 * 5632) {
        int row = tid / 5632;
        int col = tid % 5632;
        float acc = 0.0f;
        for (int k = 0; k < 2048; ++k) {
            acc += X[row * 2048 + k] * W[col * 2048 + k];
        }
        OUT[row * 5632 + col] = acc;
    }
}
"""

TILED_SRC = r"""
#define BM 32
#define BN 64
#define BK 32
#define TM 2
#define TN 8
#define THREADS 128

extern "C" __global__
__launch_bounds__(THREADS) void tiled(const float* __restrict__ W,
                                       const float* __restrict__ X,
                                       float* __restrict__ OUT) {
    __shared__ float A_tile[BM][BK];
    __shared__ float B_tile[BK][BN];

    // Thread layout: 16 thread-rows × 8 thread-cols = 128 threads.
    // Each thread owns a TM(=2) × TN(=8) register tile → 16×2 rows × 8×8 cols = 32×64 = BM×BN.
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int ty = tid / 8;
    const int tx = tid % 8;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < 2048; k0 += BK) {

        #pragma unroll
        for (int i = 0; i < BM*BK/THREADS; ++i) {
            int idx = tid + i * THREADS;
            A_tile[idx / BK][idx % BK] = X[(idx / BK) * 2048 + k0 + (idx % BK)];
        }

        #pragma unroll
        for (int i = 0; i < BK*BN/THREADS; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / BN, c = idx % BN;
            B_tile[r][c] = W[(n0 + c) * 2048 + (k0 + r)];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = A_tile[ty*TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = B_tile[kk][tx*TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = n0 + tx * TN + j;
            OUT[r * 5632 + c] = acc[i][j];
        }
    }
}
"""


def main():
    rng = np.random.default_rng(0)
    X_h = rng.standard_normal((M, K), dtype=np.float32)
    W_h = rng.standard_normal((N, K), dtype=np.float32)
    Y_ref = X_h @ W_h.T  # numpy reference

    X = cp.asarray(X_h)
    W = cp.asarray(W_h)
    Y = cp.zeros((M, N), dtype=cp.float32)

    naive_mod = cp.RawModule(code=NAIVE_SRC, options=("-std=c++17",))
    tiled_mod = cp.RawModule(code=TILED_SRC, options=("-std=c++17",))
    naive_k = naive_mod.get_function("naive")
    tiled_k = tiled_mod.get_function("tiled")

    # Correctness check
    Y.fill(0)
    naive_k(((M * N + 255) // 256,), (256,), (W, X, Y))
    cp.cuda.Stream.null.synchronize()
    naive_out = cp.asnumpy(Y)
    assert np.allclose(naive_out, Y_ref, atol=1e-2, rtol=1e-3), f"naive mismatch: max_diff={np.max(np.abs(naive_out - Y_ref))}"

    Y.fill(0)
    tiled_k((N // BN,), (THREADS,), (W, X, Y))
    cp.cuda.Stream.null.synchronize()
    tiled_out = cp.asnumpy(Y)
    assert np.allclose(tiled_out, Y_ref, atol=1e-2, rtol=1e-3), f"tiled mismatch: max_diff={np.max(np.abs(tiled_out - Y_ref))}"

    print("Correctness: both kernels match numpy reference (atol=1e-2)")

    # Benchmark
    def bench(name, fn, warmup=20, iters=200):
        for _ in range(warmup):
            fn()
        cp.cuda.Stream.null.synchronize()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        ms = cp.cuda.get_elapsed_time(start, end) / iters
        return ms

    naive_ms = bench("naive", lambda: naive_k(((M * N + 255) // 256,), (256,), (W, X, Y)))
    tiled_ms = bench("tiled", lambda: tiled_k((N // BN,), (THREADS,), (W, X, Y)))

    fma = M * N * K
    hbm_naive = 4.0 * (M * N * K * 2)  # 2 reads per FMA, 4 bytes
    hbm_tiled = 4.0 * (M * K * (N / BN) + K * N)  # A reloaded per N-block, B once

    def line(name, ms, hbm):
        gflops = fma / (ms * 1e-3) / 1e9
        bw = hbm / (ms * 1e-3) / 1e9
        print(f"  {name:<8s} {ms * 1000:7.1f} us   {gflops:7.1f} GFLOPs   {bw:6.0f} GB/s HBM-bound")

    print(f"\n{'kernel':<8s} {'time':>8s}   {'FP32 FMA':>12s}   {'HBM load BW':>14s}")
    line("naive", naive_ms, hbm_naive)
    line("tiled", tiled_ms, hbm_tiled)
    print(f"\n  speedup: {naive_ms / tiled_ms:.2f}x")


if __name__ == "__main__":
    main()
