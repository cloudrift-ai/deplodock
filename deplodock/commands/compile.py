"""Compile a graph IR through the structural lowering pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline import (
    CUDA_PASSES,
    KERNEL_PASSES,
    LOOP_PASSES,
    TENSOR_PASSES,
    TILE_PASSES,
)
from deplodock.compiler.pipeline.rule_diff import PASS_SHORTHAND

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph

logger = logging.getLogger(__name__)

# Each --ir stage maps to (passes to run, formatter). "graph" pretty-prints
# the whole Graph (inputs/constants/nodes/outputs). "kernels" renders just
# the post-lowering per-kernel bodies — dispatching on LoopOp / TileOp /
# CudaOp by type; the syntax itself (``for`` loops vs CUDA source)
# disambiguates the IR level.
_IR_STAGES = {
    "torch": ([], "graph"),
    "tensor": (TENSOR_PASSES, "graph"),
    "loop": (LOOP_PASSES, "kernels"),
    "tile": (TILE_PASSES, "kernels"),
    "kernel": (KERNEL_PASSES, "kernels"),
    "cuda": (CUDA_PASSES, "kernels"),
}

_DEFAULT_PASSES = LOOP_PASSES

# Single-letter shortcuts for each pass. Passing a contiguous string of
# these letters to --passes is equivalent to the expanded comma list
# (e.g. 'dolft' expands to the full front-to-tile pipeline). The same
# letters are used as the prefix in ``-vv`` diff markers (e.g.
# ``>>> t:005_blockify_launch``); the canonical mapping lives in
# ``compiler/pipeline/rule_diff.PASS_SHORTHAND`` so the engine can build
# the marker names without depending on the CLI layer.
_PASS_SHORTCUTS = {short: full for full, short in PASS_SHORTHAND.items()}


def resolve_tune_db() -> Path:
    """Resolve the autotune SQLite cache path: ``DEPLODOCK_TUNE_DB``
    env var → ``~/.cache/deplodock/autotune.db``. Same resolution used
    by every CLI command (``compile`` / ``run`` / ``tune``) and by
    :class:`CudaBackend` when constructed with ``tune_db="auto"``.

    Callers should treat the path as advisory — the engine only opens
    it when the file actually exists; otherwise the compile falls back
    to rule defaults (greedy option-0)."""
    override = os.environ.get("DEPLODOCK_TUNE_DB")
    return Path(override) if override else Path.home() / ".cache" / "deplodock" / "autotune.db"


def add_input_args(parser) -> None:
    """Register the model/IR input arguments shared by ``compile`` and ``tune``."""
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
        help=(
            "Sequence length for full-model tracing (default: 32). With ``--dynamic``, "
            "only sizes the example tensors handed to ``torch.export``; the value doesn't "
            "appear in the resulting kernels since torch makes the dim symbolic."
        ),
    )
    parser.add_argument(
        "--dynamic",
        action="append",
        default=None,
        metavar="NAME@INPUT:AXIS",
        help=(
            "Make a tensor dim symbolic. Form: ``NAME@INPUT:AXIS`` — axis ``AXIS`` "
            "of the traced input named ``INPUT`` becomes ``Dim(NAME)``. Repeatable for "
            "multiple dynamic dims. Forwards to ``torch.export(..., dynamic_shapes={...})``, "
            "so torch's SymInt propagation determines which downstream tensors carry the "
            "symbolic dim — no value collisions. The compiled CUDA kernel signature gains "
            "an ``int <NAME>`` runtime arg per dim. Example: ``--dynamic seq_len@x:1``."
        ),
    )
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")

    from deplodock.compiler.target import add_target_arg

    add_target_arg(parser)


def add_diagnostics_args(parser) -> None:
    """Register the ``-v`` / diff-rendering args shared by ``compile`` and ``tune``."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase verbosity. Default (no flag): only the requested IR is printed. "
            "-v: also pass timings and per-rule applied counts. "
            "-vv: also a unified-diff snapshot of every rule application, bracketed by "
            "``>>> <pass>:NNN_rulename`` / ``<<< <pass>:NNN_rulename`` markers (pass shorthands: "
            "d=decomposition, o=optimization, l=lifting, f=fusion, t=tile, k=kernel, c=cuda). "
            "Diffs go to stdout (no ``2>&1`` needed). "
            "Slice one pass: ``... -vv | awk '/^>>> t:/,/^<<< t:/'``. "
            "Slice one rule: ``... -vv | awk '/^>>> t:005/,/^<<< t:005/'``."
        ),
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize the -vv diff body. ``auto`` (default) uses ANSI when stdout is a tty and NO_COLOR is unset.",
    )
    parser.add_argument(
        "--diff-context",
        type=int,
        default=2,
        help="Unified-diff context lines for -vv rule snapshots (default: 2).",
    )
    parser.add_argument(
        "--diff-max-lines",
        type=int,
        default=200,
        help="If a single rule's -vv diff exceeds N lines, fall back to full before/after (default: 200).",
    )


def setup_pipeline_runtime(args) -> None:
    """Apply verbosity, diff-render config, and target overrides from parsed args."""
    from deplodock.compiler.pipeline.rule_diff import RuleRenderConfig, set_config, should_use_color
    from deplodock.compiler.target import apply_target_arg

    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    set_config(
        RuleRenderConfig(
            color=should_use_color(sys.stdout, args.color),
            context=args.diff_context,
            max_lines=args.diff_max_lines,
        )
    )

    apply_target_arg(args)


def register_compile_command(subparsers):
    parser = subparsers.add_parser("compile", help="Compile a model or IR through structural lowering")
    add_input_args(parser)
    parser.add_argument("--output", "-o", help="Output path for compiled IR")
    parser.add_argument(
        "--ir",
        choices=list(_IR_STAGES),
        default="cuda",
        help=(
            "IR stage to print to stdout (or ``--output``) — defaults to "
            "``cuda`` (the final lowered stage). Use a lower stage like "
            "``loop`` / ``tile`` / ``kernel`` to inspect intermediate IRs."
        ),
    )
    parser.add_argument(
        "--passes",
        default=None,
        help=(
            "Pass list to override the default. Accepts either a comma-separated list "
            "(e.g. 'decomposition,optimization,fusion') or a contiguous string of "
            "single-letter shortcuts: d=decomposition, o=optimization, l=lifting, "
            "f=fusion, t=lowering/tile, k=lowering/kernel, c=lowering/cuda."
        ),
    )
    add_diagnostics_args(parser)
    parser.set_defaults(func=handle_compile)


def handle_compile(args):
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        logger.error("either a positional model ID / IR file or --code is required")
        sys.exit(2)

    from deplodock.compiler.pipeline import Pipeline
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search.db import SearchDB

    setup_pipeline_runtime(args)
    passes = resolve_passes(args)
    graph, _ = load_or_trace(args)
    initial_count = len(graph.nodes)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Pick tuned forks from the DB when one is reachable; otherwise the
    # engine falls back to rule defaults (greedy option-0). Compile never
    # errors on a missing DB — that's only a hint, not a requirement.
    # ``DEPLODOCK_TUNE_DB`` env var overrides the default path.
    tune_db_path = resolve_tune_db()
    db = SearchDB(path=tune_db_path) if tune_db_path.exists() else None
    if db is not None:
        logger.info("Using tuning DB: %s", tune_db_path)
    else:
        logger.debug("No tuning DB at %s — using rule defaults", tune_db_path)

    result = Pipeline.build(passes, dump=dump).run(graph, db=db)

    n_compute = sum(1 for n in result.nodes.values() if not _is_boundary(n.op))
    logger.info("Lowered: %d graph nodes -> %d kernels", initial_count, n_compute)
    content = format_stage(result, args.ir)
    if args.output:
        Path(args.output).write_text(content)
        logger.info("Saved %s IR: %s", args.ir, args.output)
    else:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")


def resolve_passes(args) -> list[str]:
    if args.passes is not None:
        raw = args.passes.strip()
        # Shorthand: no commas AND every character is a known pass letter.
        if "," not in raw and raw and all(c in _PASS_SHORTCUTS for c in raw):
            return [_PASS_SHORTCUTS[c] for c in raw]
        return [p.strip() for p in raw.split(",") if p.strip()]
    if args.ir is not None:
        return _IR_STAGES[args.ir][0]
    return _DEFAULT_PASSES


def format_stage(graph, stage: str) -> str:
    from deplodock.compiler.pipeline.dump import format_kernels

    formatter = _IR_STAGES[stage][1]
    if formatter == "graph":
        return graph.pretty_print()
    return format_kernels(graph)


def _is_boundary(op) -> bool:
    from deplodock.compiler.ir.base import ConstantOp, InputOp

    return isinstance(op, (InputOp, ConstantOp))


def load_or_trace(args) -> tuple[Graph, str]:
    dynamic_shapes = _resolve_dynamic_shapes(args)
    if args.code:
        from deplodock.commands.trace import graph_from_code

        return graph_from_code(args.code, dynamic_shapes=dynamic_shapes)

    input_path = Path(args.input)
    if input_path.suffix == ".json" and input_path.exists():
        if dynamic_shapes is not None:
            logger.error("--dynamic is incompatible with loading a pre-traced JSON IR (the trace is already complete)")
            sys.exit(2)
        return _load_graph(input_path), input_path.stem

    graph = _trace_model(args.input, args.layer, args.seq_len, dynamic_shapes=dynamic_shapes)
    safe_name = args.input.replace("/", "-").lower()
    base_name = f"{safe_name}-full-s{args.seq_len}" if args.layer is None else f"{safe_name}-layer{args.layer}"
    return graph, base_name


def _resolve_dynamic_shapes(args) -> dict | None:
    """Parse every ``--dynamic NAME@INPUT:AXIS`` spec into a torch.export
    ``dynamic_shapes`` dict. Returns ``None`` when no specs were passed."""
    specs_raw = getattr(args, "dynamic", None)
    if not specs_raw:
        return None
    from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    try:
        specs = parse_position_specs(specs_raw)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)
    return build_torch_dynamic_shapes(specs)


def _load_graph(path: Path) -> Graph:
    from deplodock.compiler.graph import Graph

    with open(path) as f:
        data = json.load(f)
    return Graph.from_dict(data)


def _trace_model(model_id: str, layer: int | None, seq_len: int, *, dynamic_shapes: dict | None = None) -> Graph:
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
        from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper

        logger.info("Tracing full model (seq_len=%d)...", seq_len)
        if dynamic_shapes:
            # Dynamic mode: wrapper takes (input_ids, attention_mask, position_ids)
            # so the caller (here: the trace step + the eventual launch) can
            # supply per-call mask + position_ids sized to the runtime seq_len.
            wrapper = build_full_model_wrapper(model, seq_len, dtype, dynamic=True)
            input_ids = torch.zeros((1, seq_len), dtype=torch.long)
            attention_mask = build_causal_mask(seq_len, dtype)
            position_ids = torch.arange(seq_len).unsqueeze(0)
            return trace_module(wrapper, (input_ids, attention_mask, position_ids), dynamic_shapes=dynamic_shapes)

        wrapper = build_full_model_wrapper(model, seq_len, dtype)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        return trace_module(wrapper, (input_ids,), dynamic_shapes=dynamic_shapes)

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

    return trace_module(block, (x,), kwargs={"position_embeddings": (cos, sin)}, dynamic_shapes=dynamic_shapes)
