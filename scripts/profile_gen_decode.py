#!/usr/bin/env python3
"""Profile a generative decode step at the runner level (no vLLM server): build
`DeplodockGenRunner`, run a T=1 decode step through all layers (pre -> fake attn -> post),
and decompose the time:

  W   = wall per step (kernels + host numpy<->torch I/O + Python/dispatch)
  G   = pure GPU kernel time (sum of each program's CUDA-graph-captured window)
  W-G = host I/O + dispatch + Python overhead

Decides whether the remaining vLLM gap is kernel-bound (-> tune / smaller bucket / M=1 lowering)
or overhead-bound (-> device-resident interleave + whole-step CUDA-graph capture). The reproducer
behind the "Profiling the tuned step" section of `plans/generative-decode-perf-findings.md`.

Usage:
    python scripts/profile_gen_decode.py --bucket 16
    python scripts/profile_gen_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bucket 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="Runner-level generative decode-step profiler")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model id")
    ap.add_argument("--bucket", type=int, default=16, help="Decode bucket M (default: 16, the shipped default)")
    ap.add_argument("--steps", type=int, default=200, help="Timed decode steps (default: 200)")
    ap.add_argument("--warmup", type=int, default=30, help="Warmup decode steps (default: 30)")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError:
        sys.exit("torch + numpy required: pip install -e '.[compile,serving]'")

    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.serving.gen_runner import DeplodockGenRunner

    print(f"building runner (bucket={args.bucket}, reads the tuned prior) — compiles all layers, ~minutes ...")
    runner = DeplodockGenRunner.create(args.model, dtype_str="float16", decode_bucket=args.bucket)
    nL = runner.num_layers
    H = runner._embed_weight.shape[1]
    attn_width = runner.num_heads * runner.head_dim
    npd = runner._np_dtype
    use_decode = runner._pre_decode is not None
    path = "bucket" if use_decode else "SYMBOLIC(fallback)"
    print(f"layers={nL} hidden={H} attn_width={attn_width} decode_path={path}")

    ids = [5]
    attn0 = np.zeros((1, attn_width), dtype=npd)  # vLLM attention faked (timing is value-independent)

    def decode_step():
        h = runner.embed(ids)
        for layer in range(nL):
            runner.forward_layer_pre(layer, h)  # q,k,v discarded (attention is vLLM's)
            h = runner.forward_layer_post(layer, attn0, h)
        return runner.final_norm(h)

    for _ in range(args.warmup):
        decode_step()
    torch.cuda.synchronize()
    wall = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        decode_step()
        wall.append(time.perf_counter() - t0)
    W = statistics.median(wall) * 1e3  # ms/step

    def gpu_ms(prog_wrap, arrays):
        """One program's pure GPU time (CUDA-graph-captured per-replay ms)."""
        p = prog_wrap.program
        feed = dict(zip(prog_wrap.input_names, arrays, strict=True))
        with gpu_lock():
            p.rebind(feed)
            p.run_once()
            p.capture_program_graph()
            p.time_program_window(20)
            return statistics.median(p.time_program_window(50) for _ in range(10))

    G = 0.0
    if use_decode:
        hpad = np.zeros((args.bucket, H), dtype=npd)
        apad = np.zeros((args.bucket, attn_width), dtype=npd)
        for layer in range(nL):
            G += gpu_ms(runner._pre_decode[layer], [hpad])
            G += gpu_ms(runner._post_decode[layer], [apad, hpad])

    print(f"\n=== decode step profile (bucket={args.bucket}) ===")
    print(f"  W   wall/step        : {W:6.2f} ms   ->  {1e3 / W:6.1f} tok/s (runner-only, no vLLM attention)")
    print(f"  G   GPU kernels/step : {G:6.2f} ms   ({100 * G / W:4.1f}% of wall)")
    print(f"  W-G host+dispatch    : {W - G:6.2f} ms   ({100 * (W - G) / W:4.1f}% of wall)")


if __name__ == "__main__":
    main()
