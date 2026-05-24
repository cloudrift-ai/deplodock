"""Autotune CudaOps produced by the lowering pipeline and cache results."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from deplodock.commands.compile import (
    add_diagnostics_args,
    add_input_args,
    format_stage,
    load_or_trace,
    resolve_tune_db,
    setup_pipeline_runtime,
)
from deplodock.compiler.pipeline import CUDA_PASSES, TuningSearch
from deplodock.compiler.pipeline.knob import format_tuning_knobs

logger = logging.getLogger(__name__)


def register_tune_command(subparsers):
    parser = subparsers.add_parser(
        "tune",
        help=(
            "Bench every CudaOp produced by the lowering pipeline, attribute per-kernel "
            "latency to every ancestor along Op.source, and write the rows to the tuning cache."
        ),
    )
    add_input_args(parser)
    parser.add_argument("--output", "-o", help="Output path for the tuned CUDA IR")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Stop after this many consecutive measured variants haven't beaten the current best latency. "
            "Falls back to ``DEPLODOCK_TUNE_PATIENCE`` env var, then to 100."
        ),
    )
    parser.add_argument(
        "--ucb-c",
        type=float,
        default=TuningSearch.DEFAULT_UCB_C,
        help=(
            "UCB1 exploration constant. The canonical value is sqrt(2) ≈ 1.414; larger values "
            f"shift the walk toward exploration. Default: {TuningSearch.DEFAULT_UCB_C:.4f}."
        ),
    )
    parser.add_argument(
        "--fpu",
        type=float,
        default=TuningSearch.DEFAULT_FPU_REDUCTION,
        help=(
            "First-Play Urgency reduction applied to unvisited siblings' UCB value (max(0, parent.Q_norm - fpu)). "
            f"Default {TuningSearch.DEFAULT_FPU_REDUCTION:.2f} lets the search drill deeper into a known-decent "
            "parent path instead of exhausting every sibling before going deeper — essential for hierarchical "
            "Fork trees. Pass a negative value to restore the legacy +∞ breadth-first sweep."
        ),
    )
    add_diagnostics_args(parser)
    parser.set_defaults(func=handle_tune)


def handle_tune(args):
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        logger.error("either a positional model ID / IR file or --code is required")
        sys.exit(2)

    import time

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline import Pipeline
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search import SearchDB

    setup_pipeline_runtime(args)
    graph, _ = load_or_trace(args)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # 10 s wall budget on each variant's bench. Backstops the
    # in-process per-launch/per-iter watchdogs: if a kernel keeps
    # the GPU busy past those, the subprocess worker is SIGKILLed
    # and the dirty stream dies with it — the next bench respawns
    # cleanly. Without this the autotune sweep wedges on the
    # variant after a bench_fail (see ``benchmark_program_isolated``).
    backend = CudaBackend(bench_wall_timeout_s=10.0)
    # ``DEPLODOCK_TUNE_DB`` env overrides the default cache path.
    db_path = resolve_tune_db()
    db = SearchDB(path=db_path)
    logger.info("Tuning DB: %s", db_path)

    patience = args.patience if args.patience is not None else int(os.environ.get("DEPLODOCK_TUNE_PATIENCE", 100))
    # Negative ``--fpu`` opts out of FPU back to the legacy +∞ semantics
    # (sentinel-via-out-of-range value since argparse can't mix float | "off").
    fpu_reduction = None if args.fpu is not None and args.fpu < 0 else args.fpu
    search = TuningSearch(patience=patience, ucb_c=args.ucb_c, fpu_reduction=fpu_reduction)
    t0 = time.monotonic()
    candidates: list = []
    try:
        for cand in Pipeline.build(CUDA_PASSES, dump=dump).tune(graph, search=search, backend=backend, db=db):
            candidates.append(cand)
    except KeyboardInterrupt:
        # Manual abort: cut the sweep, fall through to summary + best-pick.
        search.stop("interrupted (Ctrl-C)")
        sys.stderr.write("\n[tune] interrupted (Ctrl-C) — printing partial results\n")
    except RuntimeError as exc:
        # An autotune-internal raise (bench watchdog couldn't bail
        # in time because the GPU queue is saturated). The CUDA
        # stream is dirty — bypass Python cleanup so cupy's atexit
        # doesn't deadlock on the still-running launch.
        sys.stderr.write(f"\n[tune] aborted: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    elapsed = time.monotonic() - t0
    reason = search.stop_reason or "tree exhausted"
    sys.stderr.write(f"\n[tune] stopped: {reason} after {len(candidates)} variant(s) in {elapsed:.1f}s\n")
    if not candidates:
        sys.stderr.write("[tune] no candidates produced — exiting without output\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    result = _pick_best_candidate(candidates, db).graph
    _print_tune_summary(candidates, db)

    # Only write the winner's CUDA source when ``--output`` is given.
    # Dumping a multi-kB kernel to stdout after a 40s tune is noise —
    # callers that want it can pass ``-o`` (or re-run ``deplodock compile``
    # with the winning knobs to reproduce).
    if args.output:
        Path(args.output).write_text(format_stage(result, "cuda"))
        logger.info("Saved cuda IR: %s", args.output)

    # A bench-timeout abandons its NVRTC worker thread (see
    # ``_benchmark_with_timeout``) which is still holding the CUDA
    # context. Python finalization (cupy memory-pool teardown, CUDA
    # context release) deadlocks against that live worker, so the
    # process hangs after all output has been written. Skip Python's
    # cleanup with ``os._exit`` once we know output is flushed — there
    # is nothing left to do and the daemon thread dies with the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _format_knobs(cuda_op) -> str:
    return format_tuning_knobs(getattr(cuda_op, "knobs", None) or {})


def _pick_best_candidate(candidates, db):
    """Pick the autotune candidate whose CudaOps all measured ``ok`` and
    whose summed latency is lowest. Falls back to ``candidates[0]`` when
    nothing has a clean measurement (e.g. every variant timed out)."""
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.cuda.ir import CudaOp
    from deplodock.compiler.pipeline.search import op_cache_key

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
            row = db.lookup_perf(ctx_key, key, backend="cuda") if key else None
            if row is None or row.status != "ok":
                all_ok = False
                break
            total += row.stats.median
        if all_ok and total < best_total:
            best_total = total
            best = cand
    return best if best is not None else candidates[0]


def _print_tune_summary(candidates, db) -> None:
    """Print all variants explored by the autotuner, sorted by total
    GPU latency. Each ``Candidate`` is one terminal pipeline run (one
    set of autotune choices); per-kernel latencies come from looking up
    each ``CudaOp`` in ``CandidateGraph`` against the measurement
    ``SearchDB``."""
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.cuda.ir import CudaOp
    from deplodock.compiler.pipeline.search import op_cache_key

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
            row = db.lookup_perf(ctx_key, key, backend="cuda") if key else None
            latency = row.stats.median if row else float("nan")
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

    def _emit(rank: int, row: tuple[bool, float, str, list[tuple[str, float]]]) -> None:
        all_ok, total, knobs_str, _ = row
        marker = "*" if rank == 0 and all_ok else " "
        status = "ok" if all_ok else "fail"
        sys.stderr.write(f"{rank:>4}{marker} {status:>7}  {total:>10.2f}  {knobs_str}\n")

    # Collapse the middle of long tables: head 10 + ellipsis + tail 5.
    # 20 is the threshold below which the full table fits on one screen
    # and the elision would hide more than it saves.
    if len(rows) > 20:
        for rank, row in enumerate(rows[:10]):
            _emit(rank, row)
        sys.stderr.write(f"{'...':>4}   {'...':>7}  {'...':>10}  ({len(rows) - 15} variant(s) elided)\n")
        for offset, row in enumerate(rows[-5:]):
            _emit(len(rows) - 5 + offset, row)
    else:
        for rank, row in enumerate(rows):
            _emit(rank, row)
