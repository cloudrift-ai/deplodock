"""Compile a graph IR (run assembly passes)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir import Graph

logger = logging.getLogger(__name__)


def register_compile_command(subparsers):
    parser = subparsers.add_parser("compile", help="Compile a model or IR through assembly passes")
    parser.add_argument("input", help="HuggingFace model ID or .json IR file")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (when input is a model ID)")
    parser.add_argument("--output", "-o", help="Output path for compiled IR")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show individual rule applications")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.set_defaults(func=handle_compile)


def handle_compile(args):
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.pipeline import compile_graph
    from deplodock.compiler.rewriter import Rewriter

    dump = CompilerDump.resolve(args.dump_dir)

    input_path = Path(args.input)

    # Detect input type: .json file → load IR, otherwise treat as HF model.
    if input_path.suffix == ".json" and input_path.exists():
        graph = _load_graph(input_path)
        base_name = input_path.stem
    else:
        graph = _trace_model(args.input, args.layer)
        safe_name = args.input.replace("/", "-").lower()
        base_name = f"{safe_name}-layer{args.layer}"

    initial_count = len(graph.nodes)

    if dump:
        dump.dump_input_graph(graph)

    # Load rewriter from rules directory.
    rules_dir = Path(__file__).parent.parent / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)

    # Compile.
    compiled, pass_traces = compile_graph(graph, rewriter)

    if dump:
        dump.dump_passes(pass_traces)

    # Print summary.
    logger.info("Assembly:")
    for pt in pass_traces:
        if pt.rules_applied:
            # Count matches per rule.
            rule_counts = {}
            for ra in pt.rules_applied:
                rule_counts[ra.rule_name] = rule_counts.get(ra.rule_name, 0) + 1
            for rule_name, count in rule_counts.items():
                logger.info("  %s: %d match%s", rule_name, count, "es" if count > 1 else "")
                if args.verbose:
                    for ra in pt.rules_applied:
                        if ra.rule_name == rule_name:
                            logger.info("    at %s: %s", ra.matched_at, ra.bindings)

    logger.info("Result: %d → %d nodes", initial_count, len(compiled.nodes))

    # Op breakdown.
    ops_count = {}
    for n in compiled.nodes.values():
        name = type(n.op).__name__
        ops_count[name] = ops_count.get(name, 0) + 1
    for op_name, count in sorted(ops_count.items()):
        logger.info("  %s: %d", op_name, count)

    # Save compiled IR.
    output_path = args.output or f"{base_name}.compiled.json"
    with open(output_path, "w") as f:
        json.dump(compiled.to_dict(), f, indent=2)
    logger.info("Saved: %s", output_path)


def _load_graph(path: Path) -> Graph:
    """Load a graph from a JSON IR file."""
    from deplodock.compiler.ir import Graph

    with open(path) as f:
        data = json.load(f)
    return Graph.from_dict(data)


def _trace_model(model_id: str, layer: int) -> Graph:
    """Pull + trace a model layer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    from deplodock.compiler.torch_trace import trace_module

    logger.info("Pulling %s...", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    layers = model.model.layers
    if layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", layer, len(layers))
        sys.exit(1)

    block = layers[layer]
    logger.info("Tracing layer %d...", layer)

    hidden_size = model.config.hidden_size
    seq_len = 32
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    return trace_module(block, (x,), kwargs={"position_embeddings": (cos, sin)})
