"""Run a compiled graph IR (one-shot) or generate text from a HF model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    parser = subparsers.add_parser("run", help="Run a compiled IR, or generate text from an HF model")
    parser.add_argument(
        "target",
        help="Either a .json IR file (one-shot run) or a HuggingFace model ID (text generation).",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text (required when target is a HF model ID).",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (timed iterations)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-launch tensor dumps (requires --dump-dir). Equivalent to DEPLODOCK_DEBUG=1.",
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Context length for text generation (default: 32).")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Number of tokens to generate (default: 20).")
    parser.set_defaults(func=_handle_run)


def _handle_run(args):
    target = Path(args.target)
    if target.suffix == ".json" and target.exists():
        if args.prompt is not None:
            logger.error("a prompt is only valid with a HF model ID target, not a .json IR file")
            sys.exit(2)
        _handle_run_ir(args)
    else:
        if args.prompt is None:
            logger.error("a prompt is required when running a HF model (e.g. deplodock run MODEL 'hello').")
            sys.exit(2)
        _handle_run_model(args)


def _handle_run_ir(args):
    import json

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.ir.graph import Graph

    dump = CompilerDump.resolve(args.dump_dir)

    ir_path = Path(args.target)
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))

    if dump:
        dump.dump_input_graph(graph)

    backend = CudaBackend(debug=args.debug or None, dump=dump)
    compiled = backend.compile(graph)

    from deplodock.compiler.ir.cuda import CudaOp

    n_kernels = sum(1 for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    logger.info("Compiled %s: %d kernels", ir_path.name, n_kernels)

    if args.benchmark:
        result = backend.benchmark(compiled, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
        if result.per_launch:
            total = sum(lt.time_ms for lt in result.per_launch)
            logger.info("Top kernels by time (sum=%.4f ms):", total)
            for lt in sorted(result.per_launch, key=lambda x: x.time_ms, reverse=True)[:5]:
                logger.info("  %3d %-40s %.4f ms", lt.idx, lt.kernel_name, lt.time_ms)
        if dump:
            dump.dump_benchmark(result)
    else:
        result = backend.run(compiled)
        for buf_name, arr in result.outputs.items():
            flat = arr.flatten()
            logger.info("Output %s: %d elements, first 5: %s", buf_name, flat.size, flat[:5].tolist())
        if dump:
            dump.dump_result(result)
            if backend.last_debug_result is not None:
                dump.dump_per_launch_values(backend.last_debug_result.per_launch)


def _handle_run_model(args):
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("torch, transformers, numpy required: pip install torch transformers")
        sys.exit(1)

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.model_wrapper import build_full_model_wrapper, collect_const_feed
    from deplodock.compiler.torch_trace import trace_module_with_constants

    model_id = args.target
    prompt = args.prompt
    seq_len = args.seq_len
    max_new = args.max_new_tokens
    dtype = torch.float32

    logger.info("Loading %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()

    logger.info("Tracing full model (seq_len=%d)...", seq_len)
    wrapper = build_full_model_wrapper(model, seq_len, dtype)
    input_ids_example = torch.zeros((1, seq_len), dtype=torch.long)
    graph, const_targets = trace_module_with_constants(wrapper, (input_ids_example,))

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    logger.info("Compiling (%d nodes)...", len(graph.nodes))
    backend = CudaBackend(debug=args.debug or None, dump=dump)
    program = backend.compile(graph)

    from deplodock.compiler.ir.cuda import CudaOp

    n_launches = sum(1 for n in program.nodes.values() if isinstance(n.op, CudaOp))
    logger.info("Compiled: %d launches, %d buffers", n_launches, len(program.nodes))

    if len(program.outputs) != 1:
        logger.error("expected exactly one output buffer, got %d", len(program.outputs))
        sys.exit(1)
    output_name = program.outputs[0]
    logger.info("Logits buffer: %s (%s)", output_name, program.nodes[output_name].output.shape)

    # One-time constant feed (parameters + buffers).
    const_feed = collect_const_feed(wrapper, const_targets)

    # Tokenize and right-pad.
    encoded = tokenizer(prompt, return_tensors="np", add_special_tokens=True).input_ids[0]
    if len(encoded) > seq_len:
        logger.error("prompt is %d tokens, longer than --seq-len %d.", len(encoded), seq_len)
        sys.exit(2)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
    ids = np.full((1, seq_len), pad_id, dtype=np.float32)
    ids[0, : len(encoded)] = encoded
    pos = len(encoded) - 1
    logger.info("Prompt tokens: %d, next sampled at position %d", len(encoded), pos + 1)

    eos = tokenizer.eos_token_id
    produced: list[int] = []
    for step in range(max_new):
        feed = dict(const_feed)
        feed["input_ids"] = ids.flatten()
        result = backend.run(program, input_data=feed)
        logits = result.outputs[output_name]  # (1, seq_len, vocab)
        next_tok = int(np.argmax(logits[0, pos]))
        tok_str = tokenizer.decode([next_tok])
        logger.info("step %02d pos=%d token=%d (%r)", step, pos, next_tok, tok_str)
        produced.append(next_tok)
        if eos is not None and next_tok == eos:
            break
        if pos + 1 >= seq_len:
            break
        pos += 1
        ids[0, pos] = next_tok

    text = tokenizer.decode(list(encoded) + produced, skip_special_tokens=False)
    print("---")
    print(text)
