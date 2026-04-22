#!/usr/bin/env python3
"""Diagnostic: trace a full LLM via torch.export + deplodock's tracer.

Prints the aten ops the tracer's ``_handle_call_function`` would either
hit directly or drop into the fallback path, so we can see at a glance
which ops still need explicit entries.

Usage:
    python scripts/trace_full_model_spike.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


_HANDLED_EW = {
    "add",
    "mul",
    "sub",
    "div",
    "neg",
    "exp",
    "rsqrt",
    "reciprocal",
    "silu",
    "relu",
    "tanh",
    "abs",
    "sigmoid",
    "pow",
}
_HANDLED_RED = {"sum", "mean", "amax", "max"}
_HANDLED_OTHER = {
    "linear",
    "mm",
    "matmul",
    "addmm",
    "scaled_dot_product_attention",
    "transpose",
    "t",
    "view",
    "reshape",
    "_unsafe_view",
    "unsqueeze",
    "squeeze",
    "expand",
    "permute",
    "to",
    "contiguous",
    "_assert_tensor_metadata",
    "clone",
    "detach",
    "alias",
    "slice",
    "cat",
    "index_select",
    "gather",
    "embedding",
    # handled as fused fallbacks
    "rms_norm",
    "softmax",
}
_HANDLED = _HANDLED_EW | _HANDLED_RED | _HANDLED_OTHER


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-model trace diagnostic")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from deplodock.compiler.model_wrapper import build_full_model_wrapper

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    logger.info("Loading %s (%s)...", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    logger.info("Wrapping for full-model trace (seq_len=%d)...", args.seq_len)
    wrapper = build_full_model_wrapper(model, args.seq_len, dtype)

    input_ids = torch.zeros((1, args.seq_len), dtype=torch.long)
    logger.info("Exporting via torch.export...")
    exported = torch.export.export(wrapper, (input_ids,), kwargs={})

    counts: Counter[str] = Counter()
    unhandled: Counter[str] = Counter()
    for node in exported.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        op_name = None
        if "aten." in target:
            parts = target.split(".")
            for i, p in enumerate(parts):
                if p == "aten" and i + 1 < len(parts):
                    op_name = parts[i + 1]
                    break
        if op_name is None:
            continue
        counts[op_name] += 1
        if op_name not in _HANDLED:
            unhandled[op_name] += 1

    logger.info("Total ops: %d call_function nodes", sum(counts.values()))
    logger.info("Distinct ops: %d", len(counts))
    logger.info("")
    logger.info("=== Handled ops ===")
    for name, n in sorted(counts.items()):
        if name in _HANDLED:
            logger.info("  %-40s %d", name, n)
    logger.info("")
    logger.info("=== UNHANDLED ops (need tracer entries) ===")
    if not unhandled:
        logger.info("  (none — trace should succeed)")
    else:
        for name, n in sorted(unhandled.items()):
            logger.info("  %-40s %d", name, n)

    return 0 if not unhandled else 1


if __name__ == "__main__":
    sys.exit(main())
