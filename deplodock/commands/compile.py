"""Compile a graph IR through the structural lowering pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph

logger = logging.getLogger(__name__)

# Each --ir stage maps to (passes to run, formatter). "graph" pretty-prints
# the whole Graph (inputs/constants/nodes/outputs). "kernels" renders just
# the post-lowering per-kernel bodies — dispatching on LoopOp / KernelOp /
# CudaOp by type; the syntax itself (``for`` loops vs C-like AST vs CUDA
# source) disambiguates the IR level.
_IR_STAGES = {
    "torch": ([], "graph"),
    "tensor": (["decomposition", "optimization"], "graph"),
    "loop": (["decomposition", "optimization", "lifting", "fusion"], "kernels"),
    "kernel": (["decomposition", "optimization", "lifting", "fusion", "lowering/kernel"], "kernels"),
    "cuda": (["decomposition", "optimization", "lifting", "fusion", "lowering/kernel", "lowering/cuda"], "kernels"),
}

_DEFAULT_PASSES = ["decomposition", "optimization", "lifting", "fusion"]

# Single-letter shortcuts for each pass. Passing a contiguous string of
# these letters to --passes is equivalent to the expanded comma list
# (e.g. 'dolfk' == 'decomposition,optimization,lifting,fusion,lowering/kernel').
_PASS_SHORTCUTS = {
    "d": "decomposition",
    "o": "optimization",
    "l": "lifting",
    "f": "fusion",
    "k": "lowering/kernel",
    "c": "lowering/cuda",
}


def register_compile_command(subparsers):
    parser = subparsers.add_parser("compile", help="Compile a model or IR through structural lowering")
    parser.add_argument("input", nargs="?", help="HuggingFace model ID or .json IR file. Mutually exclusive with --code.")
    parser.add_argument(
        "--code",
        "-c",
        help=(
            "Inline Python expression whose last statement is a call. "
            "The callable may be an nn.Module (e.g. 'nn.RMSNorm(2048)(torch.randn(1,32,2048))') "
            "or a torch function (e.g. 'F.silu(torch.randn(1,32,2048))'). "
            "Traces the expression and compiles it in one step. Mutually exclusive with the positional input."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (when input is a model ID). Omit to compile the whole model.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for full-model tracing (default: 32).",
    )
    parser.add_argument("--output", "-o", help="Output path for compiled IR")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.add_argument(
        "--ir",
        choices=list(_IR_STAGES),
        default=None,
        help="Print the requested IR stage to stdout (or --output) and exit. Skips the normal .compiled.json save.",
    )
    parser.add_argument(
        "--passes",
        default=None,
        help=(
            "Pass list to override the default. Accepts either a comma-separated list "
            "(e.g. 'decomposition,optimization,fusion') or a contiguous string of "
            "single-letter shortcuts: d=decomposition, o=optimization, l=lifting, "
            "f=fusion, k=lowering/kernel, c=lowering/cuda (so 'dolfk' == the first five)."
        ),
    )
    parser.set_defaults(func=handle_compile)


def handle_compile(args):
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        logger.error("either a positional model ID / IR file or --code is required")
        sys.exit(2)

    from deplodock.compiler.pipeline import run_pipeline
    from deplodock.compiler.pipeline.dump import CompilerDump

    passes = _resolve_passes(args)
    graph, base_name = _load_or_trace(args)
    initial_count = len(graph.nodes)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)
    result = run_pipeline(graph, passes, dump=dump)

    if args.ir is not None:
        content = _format_stage(result, args.ir)
        if args.output:
            Path(args.output).write_text(content)
        else:
            sys.stdout.write(content)
            if not content.endswith("\n"):
                sys.stdout.write("\n")
        return

    n_compute = sum(1 for n in result.nodes.values() if not _is_boundary(n.op))
    logger.info("Lowered: %d graph nodes -> %d kernels", initial_count, n_compute)
    output_path = Path(args.output) if args.output else Path(f"{base_name}.fused.txt")
    output_path.write_text(result.pretty_print())
    logger.info("Saved graph: %s", output_path)


def _resolve_passes(args) -> list[str]:
    if args.passes is not None:
        raw = args.passes.strip()
        # Shorthand: no commas AND every character is a known pass letter.
        if "," not in raw and raw and all(c in _PASS_SHORTCUTS for c in raw):
            return [_PASS_SHORTCUTS[c] for c in raw]
        return [p.strip() for p in raw.split(",") if p.strip()]
    if args.ir is not None:
        return _IR_STAGES[args.ir][0]
    return _DEFAULT_PASSES


def _format_stage(graph, stage: str) -> str:
    from deplodock.compiler.pipeline.dump import format_kernels

    formatter = _IR_STAGES[stage][1]
    if formatter == "graph":
        return graph.pretty_print()
    return format_kernels(graph)


def _is_boundary(op) -> bool:
    from deplodock.compiler.ir.base import ConstantOp, InputOp

    return isinstance(op, (InputOp, ConstantOp))


def _load_or_trace(args) -> tuple[Graph, str]:
    if args.code:
        from deplodock.commands.trace import graph_from_code

        return graph_from_code(args.code)

    input_path = Path(args.input)
    if input_path.suffix == ".json" and input_path.exists():
        graph = _load_graph(input_path)
        base_name = input_path.stem
    else:
        graph = _trace_model(args.input, args.layer, args.seq_len)
        safe_name = args.input.replace("/", "-").lower()
        if args.layer is None:
            base_name = f"{safe_name}-full-s{args.seq_len}"
        else:
            base_name = f"{safe_name}-layer{args.layer}"
    return graph, base_name


def _load_graph(path: Path) -> Graph:
    from deplodock.compiler.graph import Graph

    with open(path) as f:
        data = json.load(f)
    return Graph.from_dict(data)


def _trace_model(model_id: str, layer: int | None, seq_len: int) -> Graph:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    from deplodock.compiler.trace.torch import trace_module

    logger.info("Pulling %s...", model_id)
    dtype = torch.float32 if layer is None else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()

    if layer is None:
        from deplodock.compiler.trace.huggingface import build_full_model_wrapper

        logger.info("Tracing full model (seq_len=%d)...", seq_len)
        wrapper = build_full_model_wrapper(model, seq_len, dtype)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        return trace_module(wrapper, (input_ids,))

    layers = model.model.layers
    if layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", layer, len(layers))
        sys.exit(1)

    block = layers[layer]
    logger.info("Tracing layer %d...", layer)

    hidden_size = model.config.hidden_size
    x = torch.randn(1, seq_len, hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    return trace_module(block, (x,), kwargs={"position_embeddings": (cos, sin)})
