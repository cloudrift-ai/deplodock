#!/usr/bin/env python3
"""Benchmark a transformer block across backends.

Usage:
    python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32
    python scripts/bench_block.py --model Qwen/Qwen3-Embedding-0.6B --seq-len 2048 --iters 50
    python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32 --backends eager,emmy
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALL_BACKENDS = ["eager", "compile", "emmy", "flash_attn"]


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
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-launch tensor dumps in the Emmy backend (implies --dump-dir if set).",
    )
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

    # Sanity check: print input/weight/eager-output magnitudes so the accuracy
    # check below is known to be comparing meaningful (non-trivial) values.
    _log_sanity_stats(block, x, pos_emb)

    # Build per-backend closures up-front, then run a single interleaved
    # measurement loop so eager / torch.compile / Emmy all see the
    # same SM clock + cache state. Emmy's ``benchmark(on_iter=...)``
    # drives the loop; the on_iter callback runs each torch closure and
    # records cuda events for it. Per-launch timings come back in
    # ``bench.per_launch`` so the kernel-stats report is on the same
    # measurement window as the comparison.
    torch_fns: dict[str, callable] = {}
    if "eager" in backends:
        torch_fns["Eager PyTorch"] = _make_eager_fn(block, x, pos_emb)
    if "compile" in backends:
        compiled_fn = _make_compiled_fn(block, x, pos_emb, args.warmup)
        if compiled_fn is not None:
            torch_fns["torch.compile"] = compiled_fn

    dd_state = None
    if "emmy" in backends:
        from emmy.compiler.pipeline.dump import CompilerDump

        dump = CompilerDump.resolve(args.dump_dir)
        dd_state = _build_emmy(block, x, rotary_emb, pos_emb, dump=dump, debug=args.debug)

    results: dict[str, float] = {}
    if dd_state is not None:
        results = _bench_interleaved(dd_state, torch_fns, dump, args.warmup, args.iters)
    else:
        # Pure-torch path (no Emmy requested): time each torch
        # closure with its own cuda-event window, sequentially. Only used
        # when ``--backends`` excludes ``emmy`` — we lose interleave
        # fairness but there's no shared driver to interleave through.
        for name, fn in torch_fns.items():
            results[name] = _bench_torch_only(fn, args.warmup, args.iters)

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
        if "attn" in name.lower() and "emmy" not in name.lower():
            print(f"{name:<32s} {latency_us:>12.0f} {'(attn only)':>10s}")
        else:
            print(f"{name:<32s} {latency_us:>12.0f} {speedup:>10.2f}x")


def _log_sanity_stats(block, x, pos_emb):
    """Log magnitudes of inputs, weights, and eager output so the accuracy
    check (against the max_diff threshold) is known to be comparing real
    non-trivial values — not near-zero noise."""
    import torch

    def _stats(t: torch.Tensor) -> str:
        t = t.detach()
        flat = t.flatten().float()
        return f"shape={list(t.shape)} std={flat.std().item():.4f} range=[{flat.min().item():.3f}, {flat.max().item():.3f}]"

    logger.info("Sanity check (inputs / weights / eager output):")
    logger.info("  input x:   %s", _stats(x))
    logger.info("  cos:       %s", _stats(pos_emb[0]))
    logger.info("  sin:       %s", _stats(pos_emb[1]))

    total_params = 0
    sample_names = {"self_attn.q_proj.weight", "mlp.gate_proj.weight", "input_layernorm.weight"}
    for name, p in block.named_parameters():
        total_params += p.numel()
        if name in sample_names:
            logger.info("  weight %-40s %s", name, _stats(p))
    logger.info("  total params: %s", f"{total_params:,}")

    with torch.no_grad():
        eager_out = block(x, position_embeddings=pos_emb)[0]
    flat = eager_out.detach().flatten().float().cpu().numpy()
    n = flat.size
    above_1 = int((abs(flat) > 1.0).sum())
    above_01 = int((abs(flat) > 0.1).sum())
    logger.info(
        "  eager out: shape=%s n=%d std=%.4f range=[%.3f, %.3f]  |v|>1.0: %d/%d (%.1f%%)  |v|>0.1: %d/%d (%.1f%%)",
        list(eager_out.shape),
        n,
        flat.std(),
        flat.min(),
        flat.max(),
        above_1,
        n,
        100 * above_1 / n,
        above_01,
        n,
        100 * above_01 / n,
    )


def _make_eager_fn(block, x, pos_emb):
    import torch

    def _fn():
        with torch.no_grad():
            block(x, position_embeddings=pos_emb)

    return _fn


def _make_compiled_fn(block, x, pos_emb, warmup):
    """Build a ``torch.compile``d closure. Warm it up here so the compile
    + autograd-graph capture cost stays out of the timed window."""
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

    def _fn():
        with torch.no_grad():
            compiled(x, position_embeddings=pos_emb)

    return _fn


def _build_emmy(block, x, rotary_emb, pos_emb, dump=None, debug=False):
    """Trace + compile the block, bind inputs / constants, run once for
    correctness vs eager. Returns ``(backend, compiled_graph)`` ready
    for ``backend.benchmark_async(...)`` — or ``None`` on any failure (a
    warning is logged)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    try:
        graph = trace_module(block.cpu(), (x.cpu(),), kwargs={"position_embeddings": (pos_emb[0].cpu(), pos_emb[1].cpu())})

        if dump:
            dump.dump_input_graph(graph)

        backend = CudaBackend(debug=debug or None, dump=dump)
        compiled = backend.compile(graph)

        import torch

        from emmy.compiler.ir.base import ConstantOp

        # Bind input/constant buffers by walking the compiled graph. Inputs come
        # from tracer-assigned node ids; constants match against the block's
        # named parameters using the trace's ``p_<dotted_name>`` convention.
        input_data: dict[str, list[float]] = {}
        input_ids = set(compiled.inputs)
        bindings = {
            "hidden_states": x.cpu().flatten().tolist(),
            "position_embeddings_0": pos_emb[0].cpu().flatten().tolist(),
            "position_embeddings_1": pos_emb[1].cpu().flatten().tolist(),
        }
        for nid, node in compiled.nodes.items():
            if nid in input_ids and nid in bindings:
                input_data[nid] = bindings[nid]
                continue
            if not isinstance(node.op, ConstantOp):
                continue
            size = 1
            for d in node.output.shape:
                size *= int(d)
            matched = False
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(nid[2:]) and param.numel() == size:
                    input_data[nid] = param.detach().cpu().flatten().tolist()
                    matched = True
                    break
            if not matched and node.op.value is not None:
                input_data[nid] = [node.op.value]

        # Run with actual data.
        run_result = backend.run(compiled, input_data=input_data)
        if dump and backend.last_debug_result is not None:
            dump.dump_per_launch_values(backend.last_debug_result.per_launch)

        # Compute eager reference with same inputs.
        block.cuda()
        with torch.no_grad():
            eager_out = block(x, position_embeddings=pos_emb)[0]
        eager_flat = eager_out.cpu().flatten().tolist()

        for buf_name, arr in run_result.outputs.items():
            values = arr.flatten().tolist() if hasattr(arr, "flatten") else list(arr)
            nonzero = sum(1 for v in values if abs(v) > 1e-12)
            has_nan = any(v != v for v in values)  # NaN != NaN
            if has_nan:
                logger.error("CORRECTNESS FAIL: output %s contains NaN", buf_name)
                return None
            if nonzero == 0:
                logger.error("CORRECTNESS FAIL: output %s is all zeros", buf_name)
                return None
            logger.info(
                "Correctness check: %s has %d/%d nonzero values, range [%.4f, %.4f]",
                buf_name,
                nonzero,
                len(values),
                min(values),
                max(values),
            )

            # Numerical accuracy vs eager PyTorch.
            if len(values) == len(eager_flat):
                max_diff = max(abs(a - e) for a, e in zip(values, eager_flat, strict=True))
                mean_diff = sum(abs(a - e) for a, e in zip(values, eager_flat, strict=True)) / len(values)
                logger.info(
                    "Accuracy vs eager: max_diff=%.6f, mean_diff=%.6f, %s",
                    max_diff,
                    mean_diff,
                    "PASS" if max_diff < 1.0 else "FAIL",
                )

        return backend, compiled
    except Exception as e:
        logger.warning("Emmy pipeline failed: %s", e)
        return None


def _bench_interleaved(dd_state, torch_fns, dump, warmup, iters):
    """Single interleaved measurement loop. Emmy's
    ``backend.benchmark_async(on_iter=...)`` drives the iteration; the
    ``on_iter`` callback runs each torch closure and records its
    cuda events. Per-launch timings come back from the same loop —
    no second non-interleaved pass needed."""
    import torch

    backend, compiled = dd_state
    torch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]] = {name: [] for name in torch_fns}

    def on_iter(batch_size: int) -> None:
        # ``benchmark_program`` passes the per-iter batch size so peer
        # backends run the same number of back-to-back calls per CUDA
        # event window — keeps the comparison apples-to-apples on
        # warm/sustained perf. We divide elapsed_time by batch_size
        # downstream so the reported number stays per-call.
        for name, fn in torch_fns.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(batch_size):
                fn()
            stop.record()
            torch_events[name].append((start, stop, batch_size))

    # capture_graphs=False: this script's torch closures run uncaptured, so the
    # emmy side must too — one loop, one timing semantics (wall, incl. dispatch).
    # ``benchmark_async`` is the only bench entry now; this sync script bridges via asyncio.run.
    bench = asyncio.run(backend.benchmark_async(compiled, warmup=warmup, num_iters=iters, on_iter=on_iter, capture_graphs=False))
    torch.cuda.synchronize()

    results: dict[str, float] = {}
    for name, evt in torch_events.items():
        measured = evt[warmup:]
        if measured:
            per_iter_us = [s.elapsed_time(e) / b * 1000 for s, e, b in measured]
            results[name] = sum(per_iter_us) / len(per_iter_us)
    results["Emmy (naive attn)"] = bench.time_ms * 1000

    # Per-kernel timings: log the top offenders and (if dumping) persist
    # the full per-launch table to ``60_benchmark.json`` so the results
    # dir has the data needed to chase perf regressions post-hoc.
    if bench.per_launch:
        total = sum(lt.time_ms for lt in bench.per_launch)
        logger.info("Top kernels by time (total=%.4f ms over %d launches):", total, bench.num_launches)
        for lt in sorted(bench.per_launch, key=lambda x: x.time_ms, reverse=True)[:10]:
            logger.info("  %3d %-40s %.4f ms (%.1f%%)", lt.idx, lt.kernel_name, lt.time_ms, 100 * lt.time_ms / total)
    if dump:
        dump.dump_benchmark(bench)

    return results


def _bench_torch_only(fn, warmup, iters):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000  # ms → us


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
