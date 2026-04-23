#!/usr/bin/env python3
"""Sweep matmul tile configs for TinyLlama's matmul shapes.

For each (M, N, K) shape in TinyLlama's block, compiles + benches the
tiled SGEMM template with every valid tile config in the search grid and
reports the per-shape winner. Emits a markdown table.

Use:
    ./venv/bin/python scripts/sweep_matmul_tiles.py
    ./venv/bin/python scripts/sweep_matmul_tiles.py --shape 32x2048x2048

Tile config is fed into the detector via ``DEPLODOCK_MATMUL_*`` env vars
(see ``deplodock/compiler/pipeline/passes/loop/matmul/001_detect_matmul.py``).
The emitter's BM==M / BN|N / BK|K divisibility + thread-tile arithmetic
guards prune invalid combos silently.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from deplodock.compiler.backend.cuda.program import benchmark_program
from deplodock.compiler.pipeline import CUDA_PASSES, run_pipeline
from deplodock.compiler.trace.torch import trace_module

# TinyLlama (hidden=2048, ffn=5632, kv_heads=4, head_dim=64) at seq_len=32.
# Pure-matmul shapes; the fused-epilogue shapes (o_proj+residual,
# down_proj+residual, up_proj+silu*gate) use the same matmul geometry so
# per-shape tile picks transfer over.
_DEFAULT_SHAPES = [
    (32, 2048, 2048),  # Q / O projection (2048 out, 2048 hidden)
    (32, 256, 2048),  # K / V projection (256 = 4 heads × 64 head_dim)
    (32, 5632, 2048),  # gate / up projection (ffn expansion)
    (32, 2048, 5632),  # down projection (ffn contraction)
]

# Grid: every axis picks from a modest fan-out. The analyzer + detector
# silently reject any combo that doesn't satisfy (BM==M) / (N%BN==0) /
# (K%BK==0) / (BM/TM * BN/TN == threads), so we just try the product.
_GRID = {
    "tile_n": [32, 64, 128, 256],
    "block_k": [8, 16, 32, 64],
    "thread_m": [1, 2, 4, 8],
    "thread_n": [4, 8, 16],
    "threads": [64, 128, 256],
}


def _valid(tile: dict, m: int, n: int, k: int) -> bool:
    tm, tn = tile["tile_m"], tile["tile_n"]
    tmm, tnn = tile["thread_m"], tile["thread_n"]
    th = tile["threads"]
    if tm != m:
        return False
    if n % tn or k % tile["block_k"]:
        return False
    if tm % tmm or tn % tnn:
        return False
    if (tm // tmm) * (tn // tnn) != th:
        return False
    # Tile load must cover all elements in whole iterations.
    if (tm * tile["block_k"]) % th or (tile["block_k"] * tn) % th:
        return False
    return True


def _bench_tile(m: int, n: int, k: int, tile: dict) -> float | None:
    """Compile + bench a fresh nn.Linear at the given tile config. Return kernel ms, or None on failure."""
    env_backup: dict[str, str | None] = {}
    try:
        for key, val in tile.items():
            env_key = f"DEPLODOCK_MATMUL_{key.upper()}"
            env_backup[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(val)

        module = nn.Linear(k, n, bias=False).eval()
        x = torch.randn(m, k)
        g = trace_module(module, (x,))
        g = run_pipeline(g, CUDA_PASSES)
        inputs = {"input": x.detach().numpy(), "p_weight": module.weight.detach().numpy()}
        br = benchmark_program(g, input_data=inputs, warmup=10, num_iters=50)
        return br.per_launch[0].time_ms
    except Exception:
        return None
    finally:
        for env_key, old in env_backup.items():
            if old is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old


def _default_tile() -> dict:
    return {"tile_m": 32, "tile_n": 64, "block_k": 32, "thread_m": 2, "thread_n": 8, "threads": 128}


def _ref_cublas_ms(m: int, n: int, k: int) -> float:
    """Pure-cuBLAS baseline for the shape via cupy."""
    import cupy as cp

    A = cp.random.randn(m, k, dtype=cp.float32)
    B = cp.random.randn(n, k, dtype=cp.float32)
    for _ in range(10):
        A @ B.T
    cp.cuda.Stream.null.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(50):
        A @ B.T
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / 50


def _scalar_ms(m: int, n: int, k: int) -> float:
    """Current scalar path (no matmul hint) for comparison."""
    env_key = "DEPLODOCK_MATMUL_TILE_M"
    old = os.environ.get(env_key)
    os.environ[env_key] = "999999"  # force detector-reject via M-mismatch
    try:
        module = nn.Linear(k, n, bias=False).eval()
        x = torch.randn(m, k)
        g = trace_module(module, (x,))
        g = run_pipeline(g, CUDA_PASSES)
        inputs = {"input": x.detach().numpy(), "p_weight": module.weight.detach().numpy()}
        br = benchmark_program(g, input_data=inputs, warmup=10, num_iters=50)
        return br.per_launch[0].time_ms
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old


def sweep(m: int, n: int, k: int, top: int = 5) -> list[tuple[dict, float]]:
    results: list[tuple[dict, float]] = []
    keys = list(_GRID.keys())
    vals = [_GRID[k_] for k_ in keys]
    for combo in itertools.product(*vals):
        tile = {"tile_m": m}
        tile.update(dict(zip(keys, combo, strict=True)))
        if not _valid(tile, m, n, k):
            continue
        t_ms = _bench_tile(m, n, k, tile)
        if t_ms is not None:
            results.append((tile, t_ms))
    results.sort(key=lambda x: x[1])
    return results[:top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="MxNxK, e.g. 32x2048x2048 (repeatable)")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if args.shape:
        shapes = []
        for s in args.shape:
            parts = s.split("x")
            if len(parts) != 3:
                print(f"bad --shape: {s!r} (want MxNxK)", file=sys.stderr)
                sys.exit(1)
            shapes.append(tuple(int(p) for p in parts))
    else:
        shapes = list(_DEFAULT_SHAPES)

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"# Matmul tile sweep — top {args.top} per shape\n")
    for m, n, k in shapes:
        print(f"## {m}x{n}x{k}")
        try:
            cu_ms = _ref_cublas_ms(m, n, k)
        except Exception as e:  # cuBLAS may not be loadable — keep going without it
            cu_ms = None
            cu_err = str(e).splitlines()[0]
        sc_ms = _scalar_ms(m, n, k)
        default = _bench_tile(m, n, k, _default_tile())
        default_str = f"{default * 1000:.1f}" if default is not None else "n/a"
        print()
        if cu_ms is not None:
            print(f"- cuBLAS: {cu_ms * 1000:.1f} µs")
        else:
            print(f"- cuBLAS: unavailable ({cu_err})")
        print(f"- scalar: {sc_ms * 1000:.1f} µs")
        print(f"- current default (BN=64,BK=32,TM=2,TN=8,threads=128): {default_str} µs")
        print()
        print("| rank | BN | BK | TM | TN | threads | µs | speedup vs default |")
        print("|---|---|---|---|---|---|---|---|")
        top = sweep(m, n, k, top=args.top)
        for i, (tile, t_ms) in enumerate(top, 1):
            sp = default / t_ms if default else float("nan")
            print(
                f"| {i} | {tile['tile_n']} | {tile['block_k']} | "
                f"{tile['thread_m']} | {tile['thread_n']} | {tile['threads']} "
                f"| {t_ms * 1000:.1f} | {sp:.2f}× |"
            )
        print()


if __name__ == "__main__":
    main()
