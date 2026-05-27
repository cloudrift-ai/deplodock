"""Autotune CudaOps produced by the lowering pipeline and cache results."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from deplodock.commands.compile import (
    add_diagnostics_args,
    add_input_args,
    add_nvcc_args,
    apply_nvcc_flags,
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Before tuning, delete the tuning DB and the cubin/kernel caches for a fresh sweep.",
    )
    add_diagnostics_args(parser)
    add_nvcc_args(parser)
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
    # tune compiles at -Xcicc -O1 by default: it dodges a cicc/LLVM blowup on
    # big unrolled register-tile kernels (up to ~200x faster compile). The
    # trade-off — VERY visible because it changes how you read the numbers:
    nvcc_flags = apply_nvcc_flags(args, default="-Xcicc -O1")
    if "-O1" in nvcc_flags or "-O0" in nvcc_flags:
        sys.stderr.write(
            "\n"
            "  ┌─────────────────────────────────────────────────────────────────────────┐\n"
            f"  │  tune is compiling at cicc {('-O1' if '-O1' in nvcc_flags else '-O0'):<4} — fast, but NOT runtime-optimal.        │\n"
            "  │  Measured latencies are a RANKING signal only; reduction / attention      │\n"
            "  │  kernels can run 1.5-3x slower than -O3. Re-bench the winner with          │\n"
            "  │  `deplodock run --bench` (-O3) for deployable numbers.                     │\n"
            "  └─────────────────────────────────────────────────────────────────────────┘\n\n"
        )

    graph, _ = load_or_trace(args)

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Tight per-variant budgets: tune benches isolated single kernels compiled
    # at -Xcicc -O1 (fast), so 2 s compile / 2 s run is ample and the 6 s wall
    # SIGKILLs any runaway. (nvcc compiles eagerly in ``CompiledProgram.build``,
    # so the compile budget now genuinely bounds compilation — unlike cupy's
    # lazy NVRTC, where only the wall did.)
    _compile_timeout = 2.0
    _run_timeout = 2.0
    backend = CudaBackend(
        bench_compile_timeout_s=_compile_timeout,
        bench_run_timeout_s=_run_timeout,
        bench_wall_timeout_s=_compile_timeout + _run_timeout + 2.0,
    )
    # ``DEPLODOCK_TUNE_DB`` env overrides the default cache path.
    db_path = resolve_tune_db()
    if args.clean:
        _clean_caches(db_path)
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


def _clean_caches(db_path) -> None:
    """``--clean``: nuke the tuning DB (+ WAL/SHM sidecars) and the kernel
    caches (deplodock's cubin cache + cupy's NVRTC cache) for a fresh sweep."""
    import shutil

    from deplodock.compiler.backend.cuda import nvcc

    removed = []
    if db_path is not None:
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{db_path}{suffix}")
            if p.exists():
                p.unlink()
                removed.append(str(p))
    nvcc.clear_cubin_cache()
    removed.append(str(nvcc.cubin_cache_dir()))
    try:
        import cupy as cp

        shutil.rmtree(cp.cuda.compiler.get_cache_dir(), ignore_errors=True)
        removed.append(cp.cuda.compiler.get_cache_dir())
    except Exception:  # noqa: BLE001 — cupy cache clear is best-effort
        pass
    sys.stderr.write(f"[tune] --clean: removed tuning DB + kernel caches ({', '.join(removed)})\n")


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
