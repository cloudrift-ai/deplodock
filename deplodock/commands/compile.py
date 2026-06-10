"""Compile a graph IR through the structural lowering pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock import config
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
    to rule defaults (single-shot option-0)."""
    return config.tune_db_path()


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
            "Sequence length for full-model tracing (default: 32). With ``--dynamic``, only "
            "sizes the example tensors handed to ``torch.export``; the value doesn't appear "
            "in the resulting kernels since torch makes the dim symbolic. The tuner / planner "
            "instead tile a symbolic axis for ``Dim``'s default hint (DEFAULT_SEQ_HINT=512), "
            "emitting a masked tile (ceil-div grid + boundary guard) correct at any runtime size."
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


def add_golden_arg(parser) -> None:
    """Register ``--golden NAME`` (shared by ``tune`` and ``run``) — a shorthand
    that resolves to ``--code <the named golden config's snippet>``. Pair with
    :func:`resolve_golden_arg` in the handler."""
    parser.add_argument(
        "--golden",
        metavar="NAME",
        help=(
            "Tune / run the named golden config from GOLDEN_CONFIGS (shorthand for --code <its snippet>) — lets you "
            "build the learned prior up one shape at a time and `deplodock eval golden` between runs. An unknown NAME "
            "lists the available names. Mutually exclusive with --code / positional input / --ir."
        ),
    )


def resolve_golden_arg(args) -> None:
    """If ``--golden NAME`` is set, resolve it to ``args.code = <golden snippet>``
    and stash every config recorded under NAME on ``args.golden_configs`` (a list —
    one shape may carry several golden knob sets; ``run --bench`` echoes each under
    its matching kernel). Exits 2 on an unknown name (listing the available names)
    or a conflict with ``--code`` / positional input / ``--ir``."""
    name = getattr(args, "golden", None)
    args.golden_configs = []
    if not name:
        return
    from deplodock.compiler.pipeline.search.data import Dataset

    if args.code or args.input or getattr(args, "ir", None):
        logger.error("--golden is mutually exclusive with --code / positional input / --ir")
        sys.exit(2)
    matches = Dataset.from_golden(name=name).samples
    if not matches:
        from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig

        names = ", ".join(sorted({g.name for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)}))
        logger.error("unknown golden config %r.\nAvailable: %s", name, names)
        sys.exit(2)
    # All configs under one name share the shape, so any snippet is interchangeable.
    args.golden_configs = matches
    args.code = matches[0].snippet
    logger.info("[golden] %s → --code %s (%d recorded config%s)", name, args.code, len(matches), "" if len(matches) == 1 else "s")


def add_diagnostics_args(parser) -> None:
    """Register the ``-v`` / ``-q`` / diff-rendering args shared by ``compile`` and ``tune``."""
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
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
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "Quiet mode: log only errors (suppresses INFO / WARNING chatter). For ``tune`` this also "
            "disables the live progress bar; the final per-op / Σ summary still prints. Mutually exclusive with -v."
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


def add_nvcc_args(parser) -> None:
    """Register ``--nvcc-flags`` (shared by ``compile`` / ``run`` / ``tune``)."""
    parser.add_argument(
        "--nvcc-flags",
        default=None,
        help=(
            "Override the extra nvcc compile flags (space-separated), e.g. "
            '"-Xcicc -O3" or "-Xcicc -O1". Defaults: tune uses "-Xcicc -O1" (fast compile — but latencies are a '
            "RANKING signal, NOT -O3-optimal: reductions/attention can run 1.5-3x slower); compile/run use nvcc's "
            "default -O3. Folded into the cubin + perf cache keys. Equivalent to setting DEPLODOCK_NVCC_FLAGS."
        ),
    )


def apply_nvcc_flags(args, default: str) -> str:
    """Resolve and publish the effective extra nvcc flags. Thin CLI adapter that
    extracts ``--nvcc-flags`` from ``args`` and delegates the override/precedence
    (``--nvcc-flags`` > pre-set env > command ``default``) to
    :func:`deplodock.config.set_nvcc_flags`, so every callsite (CLI, programmatic,
    tests) shares one implementation. Must run before any compile/bench. Returns
    the effective string."""
    return config.set_nvcc_flags(getattr(args, "nvcc_flags", None), default)


def setup_pipeline_runtime(args) -> None:
    """Apply verbosity, diff-render config, and target overrides from parsed args."""
    from deplodock.compiler.pipeline.rule_diff import RuleRenderConfig, set_config, should_use_color
    from deplodock.compiler.target import apply_target_arg

    verbose = getattr(args, "verbose", 0)
    if getattr(args, "quiet", False):
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose == 0:
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
    add_nvcc_args(parser)
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
    apply_nvcc_flags(args, default="")  # compile uses nvcc default -O3 (representative codegen)
    passes = resolve_passes(args)
    graph, _, _ = load_or_trace(args)
    initial_count = len(graph.nodes)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Pick tuned forks from the DB when one is reachable; otherwise the
    # engine falls back to rule defaults (single-shot option-0). Compile never
    # errors on a missing DB — that's only a hint, not a requirement.
    # ``DEPLODOCK_TUNE_DB`` env var overrides the default path.
    tune_db_path = resolve_tune_db()
    db = SearchDB(path=tune_db_path) if tune_db_path.exists() else None
    if db is not None:
        logger.info("Using tuning DB: %s", tune_db_path)
    else:
        logger.debug("No tuning DB at %s — using rule defaults", tune_db_path)

    result = Pipeline.build(passes).run(graph, db=db, dump=dump)

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


def load_or_trace(args) -> tuple[Graph, str, tuple | None]:
    """Return ``(graph, base_name, bundle)`` where ``bundle = (module, args, kwargs)``
    is the runnable torch module + its example inputs (for ``--bench`` real-module
    timing), or ``None`` when no module is available (``--ir`` JSON path)."""
    dynamic_shapes = _resolve_dynamic_shapes(args)
    if args.code:
        from deplodock.commands.trace import graph_from_code

        return graph_from_code(args.code, dynamic_shapes=dynamic_shapes)

    input_path = Path(args.input)
    if input_path.suffix == ".json" and input_path.exists():
        if dynamic_shapes is not None:
            logger.error("--dynamic is incompatible with loading a pre-traced JSON IR (the trace is already complete)")
            sys.exit(2)
        return _load_graph(input_path), input_path.stem, None

    graph, bundle = _trace_model(args.input, args.layer, args.seq_len, dynamic_shapes=dynamic_shapes)
    safe_name = args.input.replace("/", "-").lower()
    base_name = f"{safe_name}-full-s{args.seq_len}" if args.layer is None else f"{safe_name}-layer{args.layer}"
    return graph, base_name, bundle


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


def _trace_model(model_id: str, layer: int | None, seq_len: int, *, dynamic_shapes: dict | None = None) -> tuple[Graph, tuple]:
    """Trace an HF model and return ``(graph, (module, args, kwargs))``. The bundle
    is the runnable torch module + its trace-time example inputs — kept around so
    ``tune --bench`` / ``run --bench`` can time eager / ``torch.compile`` end-to-end
    against the lowered graph."""
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
            args_t = (input_ids, attention_mask, position_ids)
            graph = trace_module(wrapper, args_t, dynamic_shapes=dynamic_shapes)
            return graph, (wrapper, args_t, {})

        wrapper = build_full_model_wrapper(model, seq_len, dtype)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        graph = trace_module(wrapper, (input_ids,), dynamic_shapes=dynamic_shapes)
        return graph, (wrapper, (input_ids,), {})

    decoder = _find_text_decoder(model)
    layers = decoder.layers
    if layer >= len(layers):
        logger.error("Layer %d not found (model has %d layers)", layer, len(layers))
        sys.exit(1)

    block = layers[layer]
    logger.info("Tracing layer %d...", layer)

    hidden_size = decoder.config.hidden_size
    x = torch.randn(1, seq_len, hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    # Some architectures (e.g. Gemma's sliding/global split) key RoPE on the
    # layer's attention type; pass it when the rotary module / layer expose it.
    layer_type = getattr(getattr(block, "self_attn", None), "layer_type", None)
    try:
        cos, sin = decoder.rotary_emb(x, position_ids, layer_type)
    except TypeError:
        cos, sin = decoder.rotary_emb(x, position_ids)

    graph = trace_module(block, (x,), kwargs={"position_embeddings": (cos, sin)}, dynamic_shapes=dynamic_shapes)
    return graph, (block, (x,), {"position_embeddings": (cos, sin)})


def _find_text_decoder(model):
    """Locate the text transformer stack (the module owning the decoder
    ``layers`` ModuleList + its ``rotary_emb``). Handles both the flat
    ``model.model`` layout (Llama / Qwen) and nested multimodal layouts where
    the language model sits under e.g. ``model.model.language_model`` (Gemma's
    unified vision/audio/text models). Returns the deepest matching module."""
    import torch.nn as nn

    best = None
    for _name, mod in model.named_modules():
        if isinstance(getattr(mod, "layers", None), nn.ModuleList) and hasattr(mod, "rotary_emb") and hasattr(mod, "config"):
            best = mod  # deepest wins (named_modules yields parents before children)
    if best is None:
        logger.error("Could not locate a text decoder (a module with `.layers` + `.rotary_emb`) in %s", type(model).__name__)
        sys.exit(1)
    return best
