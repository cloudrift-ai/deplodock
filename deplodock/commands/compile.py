"""Compile a graph IR through the structural lowering pipeline."""

from __future__ import annotations

import json
import logging
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
        "--tune",
        action="store_true",
        help=(
            "Bench every CudaOp produced by the pipeline via CudaBackend, attribute per-kernel "
            "latency (microseconds) to every ancestor along Op.source (CudaOp/KernelOp/TileOp/LoopOp), "
            "and write the rows to the tuning cache. Forces --ir cuda."
        ),
    )
    parser.add_argument(
        "--tune-db",
        default=None,
        help="Path to the tuning SQLite cache. Default: ~/.cache/deplodock/autotune.db when --tune is set.",
    )
    parser.add_argument(
        "--tune-budget",
        type=float,
        default=60.0,
        help="Wall-clock budget for --tune in seconds (default: 60). Set to inf for unbounded.",
    )
    parser.add_argument(
        "--tune-patience",
        type=int,
        default=20,
        help=(
            "Stop after this many consecutive measured variants haven't beaten the current "
            "best latency. Honored only after --tune-min-coverage is reached. Default: 20."
        ),
    )
    parser.add_argument(
        "--tune-min-coverage",
        type=float,
        default=0.3,
        help=(
            "Patience-based early stop is suppressed until this fraction of the autotune "
            "tree is explored. Default: 0.3 (30%%). Set to 1.0 to disable patience."
        ),
    )

    from deplodock.compiler.target import add_target_arg

    add_target_arg(parser)
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
    parser.set_defaults(func=handle_compile)


def handle_compile(args):
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        logger.error("either a positional model ID / IR file or --code is required")
        sys.exit(2)

    # Map -v / -vv to root log level. Default: WARNING (only the requested
    # IR is printed; pass / rule timings emitted at INFO are suppressed).
    # -v → INFO (today's pass + rule timings). -vv → DEBUG (per-rule
    # snapshot emission in `_apply_rules`).
    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    from deplodock.compiler.pipeline import run_pipeline
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.rule_diff import RuleRenderConfig, set_config, should_use_color
    from deplodock.compiler.target import apply_target_arg

    set_config(
        RuleRenderConfig(
            color=should_use_color(sys.stdout, args.color),
            context=args.diff_context,
            max_lines=args.diff_max_lines,
        )
    )

    apply_target_arg(args)
    passes = _resolve_passes(args)
    graph, base_name = _load_or_trace(args)
    initial_count = len(graph.nodes)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    backend = None
    cache = None
    if args.tune:
        if passes != CUDA_PASSES:
            logger.info("--tune forces --ir cuda (overriding %r)", args.ir)
            passes = CUDA_PASSES
            args.ir = "cuda"
        from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
        from deplodock.compiler.cache import TuningCache  # noqa: PLC0415

        backend = CudaBackend()
        db_path = Path(args.tune_db) if args.tune_db else Path.home() / ".cache" / "deplodock" / "autotune.db"
        cache = TuningCache(path=db_path)
        logger.info("Tuning cache: %s", db_path)

    if args.tune:
        import os as _os  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        from deplodock.compiler.pipeline import TuningSearch, run_autotune  # noqa: PLC0415

        search = TuningSearch(
            cache=cache,
            budget_s=args.tune_budget,
            patience=args.tune_patience,
            min_coverage=args.tune_min_coverage,
        )
        t0 = _time.monotonic()
        try:
            candidates = list(run_autotune(graph, passes, search=search, dump=dump, backend=backend))
        except RuntimeError as exc:
            # An autotune-internal raise (bench watchdog couldn't bail
            # in time because the GPU queue is saturated). The CUDA
            # stream is dirty — bypass Python cleanup so cupy's atexit
            # doesn't deadlock on the still-running launch.
            sys.stderr.write(f"\n[tune] aborted: {exc}\n")
            sys.stdout.flush()
            sys.stderr.flush()
            _os._exit(1)
        elapsed = _time.monotonic() - t0
        reason = search.stop_reason or "tree exhausted"
        sys.stderr.write(f"\n[tune] stopped: {reason} after {len(candidates)} variant(s) in {elapsed:.1f}s\n")
        result = _pick_best_candidate(candidates, cache).graph
        _print_tune_summary(candidates, cache)
    else:
        result = run_pipeline(graph, passes, dump=dump)

    n_compute = sum(1 for n in result.nodes.values() if not _is_boundary(n.op))
    logger.info("Lowered: %d graph nodes -> %d kernels", initial_count, n_compute)
    content = _format_stage(result, args.ir)
    if args.output:
        Path(args.output).write_text(content)
        logger.info("Saved %s IR: %s", args.ir, args.output)
    else:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")

    # Under --tune, a bench-timeout abandons its NVRTC worker thread
    # (see ``_benchmark_with_timeout``) which is still holding the CUDA
    # context. Python finalization (cupy memory-pool teardown, CUDA
    # context release) deadlocks against that live worker, so the
    # process hangs after all output has been written. Skip Python's
    # cleanup with ``os._exit`` once we know output is flushed — there
    # is nothing left to do and the daemon thread dies with the process.
    if args.tune:
        import os as _os  # noqa: PLC0415

        sys.stdout.flush()
        sys.stderr.flush()
        _os._exit(0)


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


def _format_knobs(cuda_op) -> str:
    """Render ``CudaOp.knobs`` (forwarded from every Op-rebind along the
    rewrite chain) as a compact ``key=value`` string. Empty dict → ``-``."""
    knobs = getattr(cuda_op, "knobs", None) or {}
    if not knobs:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(knobs.items()))


def _pick_best_candidate(candidates, cache):
    """Pick the autotune candidate whose CudaOps all measured ``ok`` and
    whose summed latency is lowest. Falls back to ``candidates[0]`` when
    nothing has a clean measurement (e.g. every variant timed out)."""
    from deplodock.compiler.cache import op_cache_key  # noqa: PLC0415
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    ctx_key = Context.probe().structural_key()
    best = None
    best_total = float("inf")
    for cand in candidates:
        cuda_nodes = [cand.graph.nodes[nid] for nid in cand.graph.topological_order() if isinstance(cand.graph.nodes[nid].op, CudaOp)]
        if not cuda_nodes:
            continue
        total = 0.0
        all_ok = True
        for node in cuda_nodes:
            key = op_cache_key(node.op)
            row = cache.cuda_perf(ctx_key, key) if key else None
            if row is None or row.status != "ok":
                all_ok = False
                break
            total += row.latency_us
        if all_ok and total < best_total:
            best_total = total
            best = cand
    return best if best is not None else candidates[0]


def _print_tune_summary(candidates, cache) -> None:
    """Print all variants explored by the autotuner, sorted by total
    GPU latency. Each ``Candidate`` is one terminal pipeline run (one
    set of autotune choices); each carries a ``trace`` of ``TraceEntry``
    (rule_name, choice_idx) and a graph with measured ``CudaOp`` nodes.
    """
    from deplodock.compiler.cache import op_cache_key  # noqa: PLC0415
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    ctx_key = Context.probe().structural_key()

    rows: list[tuple[bool, float, str, list[tuple[str, float]]]] = []
    for cand in candidates:
        cuda_nodes = [cand.graph.nodes[nid] for nid in cand.graph.topological_order() if isinstance(cand.graph.nodes[nid].op, CudaOp)]
        per_kernel: list[tuple[str, float]] = []
        total = 0.0
        all_ok = bool(cuda_nodes)
        knob_strs: list[str] = []
        for node in cuda_nodes:
            key = op_cache_key(node.op)
            row = cache.cuda_perf(ctx_key, key) if key else None
            latency = row.latency_us if row else float("nan")
            per_kernel.append((node.op.kernel_name, latency))
            if row is not None and row.status == "ok":
                total += latency
            else:
                all_ok = False
            knob_strs.append(_format_knobs(node.op))
        knobs_str = " | ".join(knob_strs) if knob_strs else "-"
        rows.append((all_ok, total, knobs_str, per_kernel))

    # Sort ok variants by ascending latency first, then any with a
    # bench_fail / unmeasured kernel at the bottom (status tiebreaker
    # before latency so 0.00 placeholders don't masquerade as the winner).
    rows.sort(key=lambda r: (not r[0], r[1]))
    sys.stderr.write(f"\n[tune] explored {len(rows)} variant(s):\n")
    sys.stderr.write(f"{'rank':>4}  {'status':>7}  {'total_us':>10}  knobs per kernel\n")
    for rank, (all_ok, total, knobs_str, _) in enumerate(rows):
        marker = "*" if rank == 0 and all_ok else " "
        status = "ok" if all_ok else "fail"
        sys.stderr.write(f"{rank:>4}{marker} {status:>7}  {total:>10.2f}  {knobs_str}\n")

    ok_rows = [r for r in rows if r[0]]
    if ok_rows:
        _, best_total, best_knobs, best_kernels = ok_rows[0]
        sys.stderr.write(f"\n[tune] winner [{best_knobs}]: {best_total:.2f} us total\n")
        for name, latency in best_kernels:
            sys.stderr.write(f"         {name:<48}  {latency:>10.2f} us\n")


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
