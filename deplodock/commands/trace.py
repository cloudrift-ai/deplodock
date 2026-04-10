"""Trace a transformer layer to our Graph IR."""

import json
import logging
import sys

logger = logging.getLogger(__name__)


def register_trace_command(subparsers):
    parser = subparsers.add_parser("trace", help="Trace a transformer layer to Graph IR")
    parser.add_argument("model", help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to trace (default: 0)")
    parser.add_argument("--output", "-o", help="Output JSON path (default: auto-generated)")
    parser.set_defaults(func=handle_trace)


def handle_trace(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    from deplodock.compiler.torch_trace import trace_module

    logger.info("Loading %s...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    # Extract the requested layer.
    layers = model.model.layers
    if args.layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", args.layer, len(layers))
        sys.exit(1)

    block = layers[args.layer]
    logger.info("Tracing layer %d...", args.layer)

    # Create example inputs for tracing.
    hidden_size = model.config.hidden_size
    seq_len = 32

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    graph = trace_module(
        block,
        (x,),
        kwargs={"position_embeddings": (cos, sin)},
    )

    # Count ops by type.
    ops_count = {}
    for n in graph.nodes.values():
        name = type(n.op).__name__
        ops_count[name] = ops_count.get(name, 0) + 1

    logger.info("Traced layer %d: %d nodes (%s)", args.layer, len(graph.nodes), ", ".join(f"{v} {k}" for k, v in sorted(ops_count.items())))

    # Save.
    output_path = args.output
    if output_path is None:
        safe_name = args.model.replace("/", "-").lower()
        output_path = f"{safe_name}-layer{args.layer}.json"

    with open(output_path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger.info("Saved: %s", output_path)
