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
from deplodock.compiler.pipeline import TuningSearch

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
        "--bench-timeout",
        type=float,
        default=20.0,
        help=(
            "Per-variant GPU-time budget (seconds) for the run stage of each bench; exceeding it marks the "
            "variant bench_fail. The default 20s suits single-kernel sweeps but is too tight for whole-model "
            "graphs: the first variant pays a one-time cold-start (hundreds of first-launch kernel loads) that "
            "can exceed 20s even though steady-state is milliseconds, so every variant fails. Bump to ~90 for "
            "full-model tuning. Default: 20."
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
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search import SearchDB
    from deplodock.compiler.pipeline.search.two_level import run_two_level_tune

    setup_pipeline_runtime(args)
    graph, _ = load_or_trace(args)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Wall + per-stage budgets on each variant's bench. The CudaBackend
    # defaults (no wall cap, 2 s compile/run) are sized for single-kernel
    # sweeps; full-model graphs with default greedy knobs (hundreds of
    # mis-tiled kernels in one bench pass) routinely exceed them. The wall
    # cap adds a SIGKILL backstop for real hangs; the bumped run budget
    # lets a slow-but-progressing 394-kernel bench finish rather than
    # getting pinned for cumulative GPU time alone. The inner per-op search
    # benches one kernel at a time, so the cold-start concern is milder, but
    # the final assembled whole-graph bench still wants the headroom.
    backend = CudaBackend(
        bench_wall_timeout_s=max(60.0, args.bench_timeout * 3),
        bench_compile_timeout_s=10.0,
        bench_run_timeout_s=args.bench_timeout,
    )
    # ``DEPLODOCK_TUNE_DB`` env overrides the default cache path.
    db_path = resolve_tune_db()
    db = SearchDB(path=db_path)
    logger.info("Tuning DB: %s", db_path)

    patience = args.patience if args.patience is not None else int(os.environ.get("DEPLODOCK_TUNE_PATIENCE", 100))
    ctx = Context.probe()
    t0 = time.monotonic()
    try:
        result = run_two_level_tune(graph, ctx=ctx, db=db, backend=backend, patience=patience, ucb_c=args.ucb_c, dump=dump)
    except KeyboardInterrupt:
        # Manual abort: per-op bests already landed in the DB as they were
        # measured, so a re-run resumes. Nothing structured to print here.
        sys.stderr.write("\n[tune] interrupted (Ctrl-C) — partial per-op results are persisted in the DB\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except RuntimeError as exc:
        # An autotune-internal raise (bench watchdog couldn't bail in time
        # because the GPU queue is saturated). The CUDA stream is dirty —
        # bypass Python cleanup so cupy's atexit doesn't deadlock on the
        # still-running launch.
        sys.stderr.write(f"\n[tune] aborted: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    elapsed = time.monotonic() - t0
    sys.stderr.write(f"\n[tune] done: {result.n_terminals} fused terminal(s) in {elapsed:.1f}s\n")
    if result.best_reward is None:
        sys.stderr.write("[tune] no kernels tuned — exiting without output\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    _print_two_level_summary(result)

    # Only write the assembled (DB-best) CUDA source when ``--output`` is
    # given. Dumping a multi-kB kernel to stdout after a long tune is noise —
    # callers that want it can pass ``-o`` (or re-run ``deplodock compile``,
    # which replays the same cached forks).
    if args.output and result.assembled is not None:
        Path(args.output).write_text(format_stage(result.assembled, "cuda"))
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


def _print_two_level_summary(result) -> None:
    """Print the per-op bests (the inner separable search), the ``Σ``
    estimate, and the assembled whole-graph latency with the separability
    gap. ``result`` is a :class:`TwoLevelResult`."""
    reward = result.best_reward
    per_op = sorted(reward.per_op, key=lambda r: (r.best_us is None, r.best_us or 0.0), reverse=True)
    sys.stderr.write(f"\n[tune] per-op bests ({len(per_op)} kernel(s), best fused terminal):\n")
    sys.stderr.write(f"{'rank':>4}  {'best_us':>10}  {'state':>8}  kernel\n")
    for rank, r in enumerate(per_op):
        us = f"{r.best_us:.2f}" if r.best_us is not None else "fail"
        state = "tuned" if r.benched else "cached"
        sys.stderr.write(f"{rank:>4}  {us:>10}  {state:>8}  {r.name}\n")

    sys.stderr.write(f"\n[tune] Σ per-op best (estimate):  {reward.total_us:>12.2f} us\n")
    if result.whole_us is not None:
        gap = result.whole_us - reward.total_us
        pct = (gap / reward.total_us * 100.0) if reward.total_us > 0 else 0.0
        sys.stderr.write(f"[tune] assembled whole-graph:     {result.whole_us:>12.2f} us\n")
        sys.stderr.write(f"[tune] separability gap:          {gap:>+12.2f} us  ({pct:+.1f}%)\n")
