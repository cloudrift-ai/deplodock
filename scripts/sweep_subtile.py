#!/usr/bin/env python3
"""Sweep (TM, TN) sub-tile choices for the block_matmul rule on representative
TinyLlama matmul shapes. Mirrors the kernel structure emitted by
``deplodock/compiler/pipeline/passes/lowering/tile/003_block_matmul.py``:
BM_TG=BN_TG=BK=16, threads = 16x16, each thread owns a TM x TN register tile,
inputs staged via shared memory, sub-tile cells interleaved across threads
(stride BM_TG / BN_TG, not contiguous).

Used to design a smarter ``_pick_subtile`` heuristic — the current rule uses
a fixed ``_MIN_BLOCKS=256`` floor, which forces (2,2) on N=2048 even when
(4,2) would be faster.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

BM_TG = 16
BN_TG = 16
BK = 16
THREADS = BM_TG * BN_TG  # 256

_SUBTILE_CHOICES = ((4, 4), (2, 4), (4, 2), (2, 2), (1, 2), (2, 1), (1, 1))

# (label, M, N, K)
SHAPES = [
    # (label, M, N, K, stage_a) — stage_a=False mirrors k_add_3_reduce where
    # the A operand (attn output) bypasses smem due to a reshape index.
    ("seq128 gate/up        ", 128, 5632, 2048, True),
    ("seq128 down           ", 128, 2048, 5632, True),
    ("seq128 q/o (staged-A) ", 128, 2048, 2048, True),
    ("seq128 add_3 (unstg-A)", 128, 2048, 2048, False),
    ("seq32  gate/up        ", 32, 5632, 2048, True),
    ("seq32  add_3 (unstg-A)", 32, 2048, 2048, False),
]


def make_src(M: int, N: int, K: int, tm: int, tn: int, stage_a: bool = True) -> str:
    bm = BM_TG * tm
    bn = BN_TG * tn
    assert M % bm == 0 and N % bn == 0 and K % BK == 0
    grid_m = M // bm
    grid_n = N // bn
    grid = grid_m * grid_n

    # Build accumulator declarations and the inner FMA block. Sub-tile cells
    # are interleaved across threads at stride BM_TG / BN_TG to match the
    # kernel emitted by 003_block_matmul.py.
    acc_decls = "\n        ".join(f"float acc{i} = 0.0f;" for i in range(tm * tn))
    acc_writes = []
    fmas = []
    if stage_a:
        a_loads = "\n            ".join(f"float ra{i} = X_stage[(ty + {i * BM_TG}) * {BK} + a5];" for i in range(tm))
    else:
        # Unstaged-A: load directly from global, mirroring k_add_3_reduce.
        # Same row across N-cells so the compiler CSEs to TM unique loads/K-step.
        a_loads = "\n            ".join(f"float ra{i} = X[(blk_m * {bm} + ty + {i * BM_TG}) * {K} + (a4 * {BK} + a5)];" for i in range(tm))
    b_loads = "\n            ".join(f"float rb{j} = W_stage[a5 * {bn} + (tx + {j * BN_TG})];" for j in range(tn))
    for i in range(tm):
        for j in range(tn):
            fmas.append(f"acc{i * tn + j} += ra{i} * rb{j};")
    fma_block = "\n            ".join(fmas)
    for i in range(tm):
        for j in range(tn):
            row = f"((blockIdx.x / {grid_n}) * {bm} + ty + {i * BM_TG})"
            col = f"((blockIdx.x % {grid_n}) * {bn} + tx + {j * BN_TG})"
            acc_writes.append(f"OUT[{row} * {N} + {col}] = acc{i * tn + j};")
    acc_writes_block = "\n        ".join(acc_writes)

    x_stage_decl = f"__shared__ float X_stage[{bm} * {BK}];" if stage_a else ""
    x_stage_body = (
        ""
        if not stage_a
        else f"""
        // Stage X: shape ({bm}, {BK}) — {bm * BK // THREADS} elem/thread
        for (int i = threadIdx.x; i < {bm * BK}; i += {THREADS}) {{
            int r = i / {BK};
            int c = i % {BK};
            X_stage[i] = X[(blk_m * {bm} + r) * {K} + (a4 * {BK} + c)];
        }}"""
    )
    return (
        f"""
extern "C" __global__
__launch_bounds__({THREADS}) void k(const float* __restrict__ X,
                                     const float* __restrict__ W,
                                     float* __restrict__ OUT) {{
    int blk_m = blockIdx.x / {grid_n};
    int blk_n = blockIdx.x % {grid_n};
    int ty = threadIdx.x / {BN_TG};
    int tx = threadIdx.x % {BN_TG};
    {acc_decls}
    {x_stage_decl}
    __shared__ float W_stage[{BK} * {bn}];
    for (int a4 = 0; a4 < {K // BK}; a4++) {{
        __syncthreads();{x_stage_body}
        // Stage W: shape ({BK}, {bn}) — W is [N, K] so transposed load
        for (int i = threadIdx.x; i < {BK * bn}; i += {THREADS}) {{
            int r = i / {bn};
            int c = i % {bn};
            W_stage[i] = W[(blk_n * {bn} + c) * {K} + (a4 * {BK} + r)];
        }}
        __syncthreads();
        for (int a5 = 0; a5 < {BK}; a5++) {{
            {a_loads}
            {b_loads}
            {fma_block}
        }}
    }}
    {acc_writes_block}
}}
""",
        grid,
    )


def bench_kernel(src: str, grid: int, X, W, OUT, warmup=20, iters=200) -> float:
    mod = cp.RawModule(code=src, options=("-std=c++17",))
    fn = mod.get_function("k")
    for _ in range(warmup):
        fn((grid,), (THREADS,), (X, W, OUT))
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        fn((grid,), (THREADS,), (X, W, OUT))
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters  # ms


def main():
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"\n{'shape':<24s} {'M':>4s} {'N':>5s} {'K':>5s}   {'(TM,TN)':>8s} {'blocks':>6s} {'us':>7s} {'GFLOPs':>8s}")
    print("-" * 82)
    rng = np.random.default_rng(0)
    for label, M, N, K, stage_a in SHAPES:
        X_h = rng.standard_normal((M, K), dtype=np.float32)
        W_h = rng.standard_normal((N, K), dtype=np.float32)
        X = cp.asarray(X_h)
        W = cp.asarray(W_h)
        Y_ref = cp.asarray(X_h @ W_h.T)
        OUT = cp.zeros((M, N), dtype=cp.float32)

        results = []
        for tm, tn in _SUBTILE_CHOICES:
            bm = BM_TG * tm
            bn = BN_TG * tn
            if M % bm or N % bn or K % BK:
                continue
            src, grid = make_src(M, N, K, tm, tn, stage_a=stage_a)
            try:
                OUT.fill(0)
                ms = bench_kernel(src, grid, X, W, OUT)
            except cp.cuda.compiler.CompileException as e:
                print(f"  ({tm},{tn})  COMPILE ERROR: {e}")
                continue
            # Correctness check on first iter only
            cp.cuda.Stream.null.synchronize()
            max_diff = float(cp.max(cp.abs(OUT - Y_ref)))
            ok = max_diff < 1e-1  # FP32 sloppy tolerance
            gflops = (M * N * K) / (ms * 1e-3) / 1e9
            tag = "" if ok else f" BAD diff={max_diff:.2g}"
            results.append((tm, tn, grid, ms, gflops, tag))

        for tm, tn, grid, ms, gflops, tag in results:
            print(f"{label:<24s} {M:>4d} {N:>5d} {K:>5d}   ({tm},{tn})  {grid:>6d} {ms * 1000:>7.1f} {gflops:>8.1f}{tag}")
        print()


if __name__ == "__main__":
    main()
