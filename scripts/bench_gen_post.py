#!/usr/bin/env python3
"""Bench the generative decode-bucket `post` subgraph vs cuBLAS (the reproducer behind
`plans/generative-decode-perf-findings.md`).

Times one decoder layer's carved `post` subgraph (o_proj + residual + post-norm + gated MLP)
three ways at decode width:

  - deplodock symbolic (hint 512) run at M=1          -> the "before" (pathological masked tile)
  - deplodock static decode-bucket M=<bucket> (shipped) -> the fix (`DeplodockGenRunner` decode path)
  - torch eager (cuBLAS) at M=<bucket>                -> the baseline

The deplodock rows use the committed CUDA-graph capture path (`capture_program_graph` +
`time_program_window`) — the same pure-GPU, dispatch-free measurement `run --bench` uses; the
programs are compiled by the exact `_compile_split` calls `DeplodockGenRunner.from_model` makes.
The cuBLAS row is the same `post` module in torch eager, `torch.cuda.Event`-timed.

Usage:
    python scripts/bench_gen_post.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python scripts/bench_gen_post.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bucket 8 --layer 0
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _time_deplodock(prog_wrap, arrays, *, warmup, window, windows):
    """Capture the program graph at the current (rebound) shape; return median per-replay ms
    over ``windows`` windows of ``window`` back-to-back replays (CUDA-graph-captured)."""
    from deplodock.compiler.backend.gpu_lock import gpu_lock

    prog = prog_wrap.program
    feed = dict(zip(prog_wrap.input_names, arrays, strict=True))
    with gpu_lock():
        prog.rebind(feed)
        prog.run_once()
        prog.capture_program_graph()
        prog.time_program_window(warmup)  # warmup window
        samples = [prog.time_program_window(window) for _ in range(windows)]
    return statistics.median(samples)


def _time_eager(module, args, *, warmup, window, windows):
    """torch eager (cuBLAS) median per-iter ms, CUDA-event timed after warmup."""
    import torch

    mod = module.to("cuda")
    with torch.no_grad():
        for _ in range(warmup):
            mod(*args)
        torch.cuda.synchronize()
        per_window = []
        for _ in range(windows):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(window):
                mod(*args)
            end.record()
            torch.cuda.synchronize()
            per_window.append(start.elapsed_time(end) / window)
    return statistics.median(per_window)


def main():
    parser = argparse.ArgumentParser(description="Generative decode-bucket post-subgraph benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer index (default: 0)")
    parser.add_argument("--bucket", type=int, default=16, help="Static decode bucket M (default: 16, the shipped default)")
    parser.add_argument("--warmup", type=int, default=50, help="Untimed warmup replays/iters (default: 50)")
    parser.add_argument("--window", type=int, default=200, help="Replays per CUDA-event window (default: 200)")
    parser.add_argument("--windows", type=int, default=30, help="Windows; report median-of-windows (default: 30)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch + transformers + numpy required: pip install -e '.[compile,serving]'")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required for benchmarking")
        sys.exit(1)

    from deplodock.compiler.trace.huggingface import build_attention_split_wrapper
    from deplodock.serving.gen_runner import _compile_split

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float16
    np_dtype = np.dtype("float16")
    bucket = args.bucket

    logger.info("Loading %s (fp16, CPU trace)...", args.model)
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    trunk = getattr(model, "model", model)
    block = trunk.layers[args.layer]
    attn0 = block.self_attn
    head_dim = attn0.head_dim
    num_heads = attn0.q_proj.out_features // head_dim
    hidden = model.config.hidden_size
    attn_width = num_heads * head_dim
    _, post_w = build_attention_split_wrapper(block)

    logger.info("Compiling post (symbolic + static M=%d)...", bucket)
    with torch.device("cpu"):
        post_sym = _compile_split(
            post_w,
            [torch.zeros(8, attn_width, dtype=dtype), torch.zeros(8, hidden, dtype=dtype)],
            ["attn_out", "residual"],
            np_dtype,
        )
        post_bucket = _compile_split(
            post_w,
            [torch.zeros(bucket, attn_width, dtype=dtype), torch.zeros(bucket, hidden, dtype=dtype)],
            None,
            np_dtype,
        )

    a1 = np.random.randn(1, attn_width).astype(np_dtype)
    r1 = np.random.randn(1, hidden).astype(np_dtype)
    ab = np.random.randn(bucket, attn_width).astype(np_dtype)
    rb = np.random.randn(bucket, hidden).astype(np_dtype)
    timing = {"warmup": args.warmup, "window": args.window, "windows": args.windows}

    sym_at_1 = _time_deplodock(post_sym, [a1, r1], **timing)
    bucket_ms = _time_deplodock(post_bucket, [ab, rb], **timing)
    eager = _time_eager(post_w, [torch.from_numpy(ab).cuda(), torch.from_numpy(rb).cuda()], **timing)

    logger.info(
        "\nWARMUP=%d WINDOW=%d WINDOWS=%d seed=%d (median-of-windows, per-replay ms)\n",
        args.warmup,
        args.window,
        args.windows,
        args.seed,
    )
    logger.info("%-44s%10s%12s", "post @ decode", "ms", "vs cuBLAS")
    logger.info("-" * 66)
    logger.info("%-44s%10.3f%11.1fx", "deplodock symbolic (hint 512) @ M=1 (before)", sym_at_1, sym_at_1 / eager)
    logger.info("%-44s%10.3f%11.1fx", f"deplodock static decode-bucket M={bucket} (shipped)", bucket_ms, bucket_ms / eager)
    logger.info("%-44s%10.3f%11.1fx", f"torch eager (cuBLAS) @ M={bucket}", eager, 1.0)


if __name__ == "__main__":
    main()
