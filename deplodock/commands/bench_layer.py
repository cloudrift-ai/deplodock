"""Benchmark a single transformer layer across multiple backends."""

import logging
import sys

logger = logging.getLogger(__name__)


def register_bench_layer_command(subparsers):
    parser = subparsers.add_parser("bench-layer", help="Benchmark a transformer layer across backends")
    parser.add_argument("model", help="HuggingFace model ID (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"], help="Data type (default: fp32)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations (default: 100)")
    parser.set_defaults(func=handle_bench_layer)


def handle_bench_layer(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install -e '.[compile]'")
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
    config = model.config
    hidden_size = config.hidden_size

    logger.info(
        "Benchmarking layer %d, seq_len=%d, batch=%d, dtype=%s",
        args.layer,
        args.seq_len,
        args.batch,
        args.dtype,
    )

    # Build example input with position embeddings.
    x = torch.randn(args.batch, args.seq_len, hidden_size, dtype=dtype, device=device)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)
    pos_emb = (cos.to(device), sin.to(device))

    results = {}

    # --- Eager PyTorch ---
    results["Eager PyTorch"] = _bench_eager(block, x, pos_emb, args.warmup, args.iters)

    # --- torch.compile ---
    compile_us = _bench_compiled(block, x, pos_emb, args.warmup, args.iters)
    if compile_us is not None:
        results["torch.compile"] = compile_us

    # --- Deplodock pipeline (our CUDA kernels) ---
    deplodock_us = _bench_deplodock(config, args.seq_len, args.batch, args.warmup, args.iters)
    if deplodock_us is not None:
        results["Deplodock (naive attn)"] = deplodock_us

    # --- FlashAttention (attention only) ---
    fa_result = _bench_flash_attention(x, config, dtype, device, args.warmup, args.iters)
    if fa_result is not None:
        results["FlashAttention (attn)"] = fa_result

    # --- Print results ---
    logger.info("")
    logger.info("%-32s %12s %10s", "Backend", "Latency (us)", "vs Eager")
    logger.info("-" * 56)

    eager_us = results.get("Eager PyTorch", 0)
    for name, latency_us in results.items():
        speedup = eager_us / latency_us if latency_us > 0 else 0
        if "attn" in name.lower() and "deplodock" not in name.lower():
            logger.info("%-32s %12.0f %10s", name, latency_us, "(attn only)")
        else:
            logger.info("%-32s %12.0f %10.2fx", name, latency_us, speedup)


def _bench_eager(block, x, pos_emb, warmup, iters):
    """Benchmark eager PyTorch forward pass."""
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

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000  # ms → us


def _bench_compiled(block, x, pos_emb, warmup, iters):
    """Benchmark torch.compile'd forward pass."""
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

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000


def _bench_deplodock(config, seq_len, batch, warmup, iters):
    """Benchmark our compiled CUDA pipeline."""
    from deplodock.compiler.cuda.block_lower import BlockConfig
    from deplodock.compiler.cuda.block_runner import run_block

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

        result = run_block(cfg)
        if result.kernel_time_ms is not None:
            return result.kernel_time_ms * 1000  # ms → us
        logger.warning("Deplodock pipeline did not return timing")
        return None
    except Exception as e:
        logger.warning("Deplodock pipeline failed: %s", e)
        return None


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

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1000
