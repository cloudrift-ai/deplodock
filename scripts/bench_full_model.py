#!/usr/bin/env python3
"""Benchmark a *whole* HuggingFace CausalLM forward across backends.

Traces the model via ``build_full_model_wrapper`` (static causal mask +
precomputed rotary cos/sin), compiles it to CUDA, binds every weight /
buffer from the live wrapper, checks accuracy against eager, and times
eager / torch.compile / Emmy in one interleaved loop.

Unlike ``scripts/bench_block.py`` (one transformer layer) this runs the
full stack: embedding → N decoder layers → final norm → lm_head. Pass a
tuned ``EMMY_TUNE_DB`` to bench the autotuned kernels.

Usage:
    python scripts/bench_full_model.py --model Qwen/Qwen3-Embedding-0.6B --seq-len 32
    EMMY_TUNE_DB=/tmp/qwen.db python scripts/bench_full_model.py \
        --model Qwen/Qwen3-Embedding-0.6B --seq-len 32 --backends eager,compile,emmy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-model benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3-Embedding-0.6B)")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length (default: 32)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Measurement iterations (default: 50)")
    parser.add_argument("--backends", default="eager,compile,emmy", help="Comma-separated: eager,compile,emmy")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the input token ids (default: 0)")
    parser.add_argument(
        "--bench-timeout",
        type=float,
        default=120.0,
        help="Per-bench GPU-time budget in seconds (default: 120). Bump for slow default-knobs graphs.",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch + transformers + numpy required: pip install -e '.[compile]'")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required for benchmarking")
        sys.exit(1)

    from emmy.commands.run import _bench_interleaved, _build_torch_fns, _print_kernel_stats, _print_table, _resolve_backends
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader import bind_constants_from_module
    from emmy.compiler.pipeline.dump import CompilerDump
    from emmy.compiler.trace.huggingface import build_full_model_wrapper
    from emmy.compiler.trace.torch import trace_module

    dtype = torch.float32
    logger.info("Loading %s (fp32)...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size

    logger.info("Building trace wrapper (seq_len=%d)...", args.seq_len)
    wrapper = build_full_model_wrapper(model, args.seq_len, dtype)
    rng = np.random.default_rng(args.seed)
    input_ids = torch.from_numpy(rng.integers(0, vocab_size, size=(1, args.seq_len), dtype=np.int64))

    logger.info("Tracing...")
    graph = trace_module(wrapper, (input_ids,))

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # ``tune_db="auto"`` resolves EMMY_TUNE_DB → the autotuned kernel
    # variants; falls back to rule defaults if the DB is absent.
    backend = CudaBackend(
        dump=dump, tune_db="auto", bench_run_timeout_s=args.bench_timeout, bench_compile_timeout_s=max(10.0, args.bench_timeout / 4)
    )
    if backend.tune_db is not None and backend.tune_db.exists():
        logger.info("Using tuning DB: %s", backend.tune_db)
    logger.info("Compiling to CUDA...")
    compiled = backend.compile(graph)

    # Bind activations (the single ``input_ids`` input) + every weight /
    # buffer (incl. the wrapper's precomputed cos/sin + causal mask, which
    # aren't in any safetensors checkpoint) from the live traced module.
    input_data = bind_constants_from_module(compiled, wrapper)
    in_ids = list(compiled.inputs)
    assert len(in_ids) == 1, f"expected 1 graph input (input_ids), got {len(in_ids)}"
    nid = in_ids[0]
    input_data[nid] = input_ids.numpy().astype(compiled.nodes[nid].output.dtype.np, copy=False)

    logger.info("Running + checking accuracy vs eager...")
    run_result, _ = backend.run(compiled, input_data=input_data)
    with torch.no_grad():
        eager_out = wrapper(input_ids)
    if isinstance(eager_out, tuple):
        eager_out = eager_out[0]
    _check_accuracy(run_result.outputs, eager_out)

    backends = _resolve_backends(args.backends)
    cuda_wrapper = wrapper.to("cuda")
    cuda_ids = (input_ids.to("cuda"),)
    torch_fns = _build_torch_fns(cuda_wrapper, cuda_ids, {}, args.warmup, backends=backends)
    results, bench = _bench_interleaved(cuda_wrapper, cuda_ids, {}, backend, compiled, args.warmup, args.iters, torch_fns=torch_fns)

    _print_table(results)
    _print_kernel_stats(compiled, bench)


def _check_accuracy(outputs, eager_out) -> None:
    import numpy as np

    eager_flat = eager_out.detach().cpu().float().flatten().numpy()
    for name, arr in outputs.items():
        flat = np.asarray(arr).flatten()
        if not np.isfinite(flat).all():
            logger.error("CORRECTNESS FAIL: output %s has non-finite values", name)
            sys.exit(1)
        if flat.size != eager_flat.size:
            logger.warning("Output %s size %d != eager %d; skipping accuracy", name, flat.size, eager_flat.size)
            continue
        max_diff = float(np.abs(flat - eager_flat).max())
        mean_diff = float(np.abs(flat - eager_flat).mean())
        peak = float(np.abs(eager_flat).max())
        corr = float(np.corrcoef(flat, eager_flat)[0, 1])
        tol = max(1e-2, 0.02 * peak)
        verdict = "PASS" if (max_diff <= tol and corr > 0.999) else "FAIL"
        logger.info(
            "Accuracy vs eager: corr=%.6f max_diff=%.6f mean_diff=%.6f peak=%.3f -> %s",
            corr,
            max_diff,
            mean_diff,
            peak,
            verdict,
        )
        if verdict == "FAIL":
            sys.exit(1)


if __name__ == "__main__":
    main()
