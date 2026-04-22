"""Compile a graph IR through the structural lowering pipeline."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph

logger = logging.getLogger(__name__)

_IR_STAGE_FILES = {
    "torch": "00_input_graph.txt",
    "tensor": "10_tensor_ir.txt",
    "loop": "37_loop_kernels.txt",
    "loop-program": "38_loop_program.txt",
    "kernel": "39_kernel_ir.txt",
    "cuda": "40_kernels.cu",
    "cuda-program": "40_program.txt",
}


def register_compile_command(subparsers):
    parser = subparsers.add_parser("compile", help="Compile a model or IR through structural lowering")
    parser.add_argument("input", help="HuggingFace model ID or .json IR file")
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
        choices=list(_IR_STAGE_FILES),
        default=None,
        help="Print the requested IR stage to stdout (or --output) and exit. Skips the normal .compiled.json save.",
    )
    parser.set_defaults(func=handle_compile)


def handle_compile(args):
    if args.ir is not None:
        _handle_compile_inspect(args)
        return

    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.pipeline import compile_graph

    dump = CompilerDump.resolve(args.dump_dir)

    graph, base_name = _load_or_trace(args)

    initial_count = len(graph.nodes)
    if dump:
        dump.dump_input_graph(graph)

    program = compile_graph(graph, dump=dump)

    logger.info("Lowered: %d graph nodes -> %d kernels", initial_count, len(program.launches))

    output_path = args.output or f"{base_name}.compiled.json"
    # Persist as the original graph for now; structural LoopOp serialization
    # lands when the lowering is in place.
    with open(output_path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
    logger.info("Saved: %s", output_path)


def _handle_compile_inspect(args):
    """Run the pipeline to produce --ir STAGE artifact, then print it."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.pipeline import compile_graph

    graph, _ = _load_or_trace(args)
    stage = args.ir
    stage_file = _IR_STAGE_FILES[stage]

    with tempfile.TemporaryDirectory(prefix="deplodock-inspect-") as tmp:
        dump_dir = Path(args.dump_dir) if args.dump_dir else Path(tmp)
        dump = CompilerDump(dir=dump_dir)
        dump.dump_input_graph(graph)

        if stage == "torch":
            pass  # dump_input_graph above already wrote the needed files
        elif stage in ("tensor", "loop", "loop-program"):
            compile_graph(graph, dump=dump)
        else:
            CudaBackend(dump=dump).compile(graph)

        content = (dump.dir / stage_file).read_text()

    if args.output:
        Path(args.output).write_text(content)
    else:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")


def _load_or_trace(args) -> tuple[Graph, str]:
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
    from deplodock.compiler.ir.graph import Graph

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

    from deplodock.compiler.torch_trace import trace_module

    logger.info("Pulling %s...", model_id)
    dtype = torch.float32 if layer is None else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()

    if layer is None:
        from deplodock.compiler.model_wrapper import build_full_model_wrapper

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
