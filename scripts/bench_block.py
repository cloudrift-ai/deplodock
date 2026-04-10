#!/usr/bin/env python3
"""Benchmark a transformer block across backends.

Usage:
    python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32
    python scripts/bench_block.py --model Qwen/Qwen2.5-7B --seq-len 2048 --iters 50
    python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32 --backends eager,deplodock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALL_BACKENDS = ["eager", "compile", "deplodock", "flash_attn"]


def main():
    parser = argparse.ArgumentParser(description="Transformer block benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length (default: 32)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"], help="Data type (default: fp32)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations (default: 100)")
    parser.add_argument("--backends", default=",".join(ALL_BACKENDS), help=f"Comma-separated backends (default: {','.join(ALL_BACKENDS)})")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers required: pip install -e '.[compile]'")
        sys.exit(1)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.error("CUDA GPU required for benchmarking")
        sys.exit(1)

    backends = [b.strip() for b in args.backends.split(",")]

    logger.info("Loading %s...", args.model)
    # Load model on CPU first, then move only the target layer to GPU.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    config = model.config
    hidden_size = config.hidden_size

    layers = model.model.layers
    if args.layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", args.layer, len(layers))
        sys.exit(1)

    block = layers[args.layer].to(device)
    # Build rotary embeddings on CPU, then move to device.
    rotary_emb = model.model.rotary_emb
    # Free the rest of the model to save GPU memory.
    del model.model.layers
    del model
    torch.cuda.empty_cache()

    logger.info(
        "%s layer %d | seq_len=%d batch=%d dtype=%s | hidden=%d heads=%d",
        args.model,
        args.layer,
        args.seq_len,
        args.batch,
        args.dtype,
        hidden_size,
        config.num_attention_heads,
    )

    # Build inputs.
    x = torch.randn(args.batch, args.seq_len, hidden_size, dtype=dtype, device=device)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0)
    cos, sin = rotary_emb(x.cpu(), position_ids.cpu())
    pos_emb = (cos.to(device), sin.to(device))

    results: dict[str, float] = {}

    # --- Eager ---
    if "eager" in backends:
        us = _bench_eager(block, x, pos_emb, args.warmup, args.iters)
        results["Eager PyTorch"] = us

    # --- torch.compile ---
    if "compile" in backends:
        us = _bench_compiled(block, x, pos_emb, args.warmup, args.iters)
        if us is not None:
            results["torch.compile"] = us

    # --- Deplodock CUDA pipeline ---
    if "deplodock" in backends:
        us = _bench_deplodock(config, args.seq_len, args.batch)
        if us is not None:
            results["Deplodock (naive attn)"] = us

    # --- FlashAttention (attention only) ---
    if "flash_attn" in backends:
        us = _bench_flash_attention(x, config, dtype, device, args.warmup, args.iters)
        if us is not None:
            results["FlashAttention (attn)"] = us

    # --- Print results ---
    print()
    eager_us = results.get("Eager PyTorch", 0)
    print(f"{'Backend':<32s} {'Latency (us)':>12s} {'vs Eager':>10s}")
    print("-" * 56)
    for name, latency_us in results.items():
        speedup = eager_us / latency_us if latency_us > 0 else 0
        if "attn" in name.lower() and "deplodock" not in name.lower():
            print(f"{name:<32s} {latency_us:>12.0f} {'(attn only)':>10s}")
        else:
            print(f"{name:<32s} {latency_us:>12.0f} {speedup:>10.2f}x")


def _bench_eager(block, x, pos_emb, warmup, iters):
    import torch

    for _ in range(warmup):
        with torch.no_grad():
            block(x, position_embeddings=pos_emb)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            block(x, position_embeddings=pos_emb)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000  # ms → us


def _bench_compiled(block, x, pos_emb, warmup, iters):
    import torch

    try:
        compiled = torch.compile(block, mode="reduce-overhead")
        for _ in range(warmup + 5):
            with torch.no_grad():
                compiled(x, position_embeddings=pos_emb)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning("torch.compile failed: %s", e)
        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            compiled(x, position_embeddings=pos_emb)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000


def _bench_deplodock(config, seq_len, batch):
    from deplodock.compiler.cuda.block_lower import BlockConfig, lower_block
    from deplodock.compiler.cuda.program import benchmark_program

    try:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        intermediate = getattr(config, "intermediate_size", config.hidden_size * 4)

        cfg = BlockConfig(
            batch=batch,
            seq_len=seq_len,
            hidden_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=intermediate,
        )
        program = lower_block(cfg)
        result = benchmark_program(program, warmup=3, num_iters=10)
        return result.time_ms * 1000  # ms → us
    except Exception as e:
        logger.warning("Deplodock pipeline failed: %s", e)
        return None


def _bench_flash_attention(x, config, dtype, device, warmup, iters):
    import torch

    try:
        from flash_attn import flash_attn_func
    except ImportError:
        logger.info("flash-attn not installed, skipping")
        return None

    batch, seq_len, hidden = x.shape
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads

    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    for _ in range(warmup):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        flash_attn_func(q, k, v, causal=True)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000


if __name__ == "__main__":
    main()
