"""Benchmark a single transformer layer across multiple backends."""

import logging
import sys

logger = logging.getLogger(__name__)


def register_bench_layer_command(subparsers):
    parser = subparsers.add_parser("bench-layer", help="Benchmark a transformer layer across backends")
    parser.add_argument("model", help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="Data type (default: fp16)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations (default: 100)")
    parser.set_defaults(func=handle_bench_layer)


def handle_bench_layer(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        logger.error("CUDA GPU required for benchmarking")
        sys.exit(1)

    logger.info("Loading %s...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    layers = model.model.layers
    if args.layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", args.layer, len(layers))
        sys.exit(1)

    block = layers[args.layer].to(device)
    hidden_size = model.config.hidden_size

    logger.info("Benchmarking layer %d, seq_len=%d, batch=%d, dtype=%s", args.layer, args.seq_len, args.batch, args.dtype)

    x = torch.randn(args.batch, args.seq_len, hidden_size, dtype=dtype, device=device)

    results = {}

    # --- Eager PyTorch ---
    results["Eager PyTorch"] = _bench_eager(block, x, args.warmup, args.iters)

    # --- torch.compile ---
    results["torch.compile"] = _bench_compiled(block, x, args.warmup, args.iters)

    # --- FlashAttention (attention only) ---
    fa_result = _bench_flash_attention(x, model.config, dtype, device, args.warmup, args.iters)
    if fa_result is not None:
        results["FlashAttention (attn)"] = fa_result

    # --- Print results ---
    logger.info("")
    logger.info("%-28s %12s %10s", "Backend", "Latency (us)", "vs Eager")
    logger.info("-" * 52)

    eager_us = results.get("Eager PyTorch", 0)
    for name, latency_us in results.items():
        speedup = eager_us / latency_us if latency_us > 0 else 0
        if "attn" in name:
            logger.info("%-28s %12.0f %10s", name, latency_us, "(attn only)")
        else:
            logger.info("%-28s %12.0f %10.2fx", name, latency_us, speedup)


def _bench_eager(block, x, warmup, iters):
    """Benchmark eager PyTorch forward pass."""
    import torch

    # Warmup.
    for _ in range(warmup):
        with torch.no_grad():
            block(x)
    torch.cuda.synchronize()

    # Measure.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        with torch.no_grad():
            block(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000  # convert ms → us


def _bench_compiled(block, x, warmup, iters):
    """Benchmark torch.compile'd forward pass."""
    import torch

    try:
        compiled = torch.compile(block, mode="reduce-overhead")
    except Exception as e:
        logger.warning("torch.compile failed: %s", e)
        return float("inf")

    # Warmup (compile happens here).
    for _ in range(warmup + 5):
        with torch.no_grad():
            compiled(x)
    torch.cuda.synchronize()

    # Measure.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        with torch.no_grad():
            compiled(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000


def _bench_flash_attention(x, config, dtype, device, warmup, iters):
    """Benchmark FlashAttention on synthetic Q/K/V tensors."""
    import torch

    try:
        from flash_attn import flash_attn_func
    except ImportError:
        logger.info("flash-attn not installed, skipping FlashAttention benchmark")
        return None

    batch, seq_len, hidden = x.shape
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads

    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Warmup.
    for _ in range(warmup):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    # Measure.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        flash_attn_func(q, k, v, causal=True)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000
