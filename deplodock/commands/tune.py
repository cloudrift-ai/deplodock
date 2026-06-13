"""Autotune CudaOps produced by the lowering pipeline and cache results."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from deplodock import config
from deplodock.commands.compile import (
    add_diagnostics_args,
    add_golden_arg,
    add_input_args,
    add_nvcc_args,
    apply_nvcc_flags,
    format_stage,
    load_or_trace,
    resolve_golden_arg,
    resolve_tune_db,
    setup_pipeline_runtime,
)
from deplodock.commands.dataset_args import add_dataset_args, require_source
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
    add_golden_arg(parser)
    # ``--dataset golden`` tunes every golden shape in sequence (the built-in
    # equivalent of looping ``--golden NAME`` over GOLDEN_CONFIGS); ``--kernel``
    # narrows to a name substring. Default None → ordinary single-op tune.
    add_dataset_args(parser, default=None)
    parser.add_argument("--output", "-o", help="Output path for the tuned CUDA IR")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Stop after this many consecutive measured variants haven't beaten the current best latency. "
            "Falls back to ``DEPLODOCK_TUNE_PATIENCE`` env var, then to 50."
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
        "--explore-eps",
        type=float,
        default=None,
        help=(
            "ε-greedy exploration: probability a selection step descends a uniformly random child "
            "instead of the PUCT argmax, perturbing (not replacing) the heuristic order for shapes "
            "where it's known-bad. Falls back to ``DEPLODOCK_TUNE_EPS`` env var, then to 0.0 "
            "(deterministic PUCT) — opt-in; see plans/golden-sweep-report.md."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Before tuning, delete the tuning DB and the cubin/kernel caches for a fresh sweep.",
    )
    parser.add_argument(
        "--bench",
        "-b",
        action="store_true",
        help=(
            "After tuning, re-bench the winner at -O3 (deployable numbers, NOT the -O1 ranking pass): the full "
            "compiled model and each individual kernel (via its provenance .torch.json reproducer) vs eager "
            "PyTorch / torch.compile / Deplodock, then print a comparison table. Writes an HTML per-kernel chart "
            "to <dump-dir>/kernels.html when a dump dir is set. Can take minutes on a large model."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --bench random inputs (default: 0).")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help=(
            "Tune the post-fusion kernels concurrently across this many GPUs (devices 0..N-1): one in-flight "
            "bench per GPU on a single event loop. Default: single-GPU (serial, behaviorally identical). "
            "Bounded by the number of unique kernels; devices must be homogeneous. ``--devices`` overrides."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated GPU ids to tune across (e.g. ``0,1,3``), the explicit form of ``--gpus``. "
            "Devices must be homogeneous (one perf key per tune). Default: single-GPU."
        ),
    )
    parser.add_argument(
        "--bench-backends",
        default="eager,tcompile,deplodock",
        help=(
            "Comma-separated subset of backends to time under --bench: any of ``eager``, ``tcompile`` "
            "(torch.compile), ``deplodock``. Default: all three — tune --bench is the deployable comparison, "
            "so torch.compile's ~0.8s JIT is worth paying. ``deplodock`` is always included."
        ),
    )
    add_diagnostics_args(parser)
    add_nvcc_args(parser)
    parser.set_defaults(func=handle_tune)


def _tune_offline(args):
    """``deplodock tune`` with no op: refit the global learned prior on its
    persisted reservoir dataset and print offline diagnostics — no GPU, no
    benching. Answers "can the prior reach the best configs?" over everything
    tuned so far."""
    from deplodock import config
    from deplodock.compiler.pipeline.search.prior import CatBoostPrior, diagnostics

    prior = CatBoostPrior.load(seed=args.seed)
    if not prior._dataset:
        logger.error("no prior dataset at %s — run `deplodock tune <model>` first", config.prior_path())
        sys.exit(1)
    sys.stderr.write(f"[tune] offline refit on {len(prior._dataset)} rows from {config.prior_path()}\n")
    prior.fit()  # unconditional re-fit on the whole accumulated dataset
    prior.checkpoint()
    sys.stderr.write(diagnostics.report(prior) + "\n")
    sys.stderr.write(diagnostics.golden_prior_eval(prior) + "\n")


def _tune_backend(device_id: int | None = None):
    """The autotune-sweep ``CudaBackend``: benches each variant in a SIGKILL-able
    ``_bench_worker`` **subprocess** (``bench_wall_timeout_s`` set → the isolated
    path in ``benchmark_async``), so a wedged kernel dies
    with the worker and the **parent** CUDA stream stays clean. Tight per-variant
    budgets: tune benches isolated single kernels at -Xcicc -O1 (fast), so 2 s
    compile / 2 s run is ample and the 6 s wall SIGKILLs any runaway. ``device_id``
    pins the async bench worker to a physical GPU (multi-GPU tune)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    return CudaBackend(bench_compile_timeout_s=2.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=6.0, device_id=device_id)


def _resolve_devices(args) -> list[int | None]:
    """Resolve ``--gpus`` / ``--devices`` into a device-id list (``--devices`` wins).
    Default ``[None]`` → a single unpinned slot = today's serial behavior. Two or
    more devices must be homogeneous — the tune keys every perf row on one probed
    ``ctx``, so mixed compute capabilities would corrupt the per-op cache."""
    if args.devices:
        try:
            devices: list[int | None] = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        except ValueError:
            logger.error("--devices must be comma-separated GPU ids, e.g. 0,1,3")
            sys.exit(2)
    elif args.gpus is not None:
        if args.gpus < 1:
            logger.error("--gpus must be >= 1")
            sys.exit(2)
        devices = list(range(args.gpus))
    else:
        return [None]
    if len(devices) <= 1:
        return devices or [None]
    _require_homogeneous_devices(devices)
    return devices


def _require_homogeneous_devices(devices: list[int | None]) -> None:
    try:
        import cupy as cp
    except Exception:  # noqa: BLE001 — no cupy → can't probe; let the bench surface any mismatch
        return
    caps = {}
    for d in devices:
        try:
            props = cp.cuda.runtime.getDeviceProperties(d)
        except Exception as exc:  # noqa: BLE001
            logger.error("--devices: GPU %s not available (%s)", d, exc)
            sys.exit(2)
        caps[d] = (props["major"], props["minor"])
    if len(set(caps.values())) > 1:
        logger.error("--devices must be homogeneous (one perf key per tune); got compute capabilities %s", caps)
        sys.exit(2)


def _tune_one(args, *, backends, db, ctx, dump):
    """Trace ``args.code`` / ``args.input`` and run the two-level tune on that one
    graph; return ``(result, bench_bundle)``. Manages the live progress bar (closed
    in ``finally``) and prints the per-op ``done`` summary. Lets ``KeyboardInterrupt``
    and the saturated-queue ``RuntimeError`` (dirty parent stream) **propagate** so
    the caller decides how to exit — called once per target by ``handle_tune``'s
    loop (one shape or the whole golden set). Benching itself is subprocess-isolated
    (see ``_tune_backend``), so the parent process is safe to reuse shape-to-shape.
    ``backends`` is the device-pinned pool (one per GPU; single-element by default)
    fanning the inner per-kernel search across GPUs."""
    import time

    from deplodock.commands.tune_progress import TuneProgress
    from deplodock.compiler.pipeline.search.two_level import run_two_level_tune

    graph, _, bench_bundle = load_or_trace(args)
    if dump:
        dump.dump_input_graph(graph)
    # Live progress bar — default verbosity on a tty only. Disabled under -v (the
    # [tune] INFO lines show progress instead), -q (errors only), and when stderr is
    # redirected (no \r smearing in piped logs).
    progress = TuneProgress(
        enabled=getattr(args, "verbose", 0) == 0 and not getattr(args, "quiet", False) and sys.stderr.isatty(),
    )
    patience = args.patience if args.patience is not None else config.tune_patience(50)
    explore_eps = args.explore_eps if args.explore_eps is not None else config.tune_eps(0.0)
    t0 = time.monotonic()
    try:
        # The two-level tune is async (per-kernel benches fan across device-pinned
        # workers on one event loop); ``handle_tune`` is the sync CLI boundary, so the
        # whole outer drive runs under one ``asyncio.run`` here.
        result = asyncio.run(
            run_two_level_tune(
                graph,
                ctx=ctx,
                db=db,
                backends=backends,
                patience=patience,
                ucb_c=args.ucb_c,
                explore_eps=explore_eps,
                dump=dump,
                progress=progress,
                prior_seed=args.seed,
            )
        )
    finally:
        progress.close()
    sys.stderr.write(f"\n[tune] done: {result.n_terminals} fused terminal(s) in {time.monotonic() - t0:.1f}s\n")
    for block in result.prior_summaries:  # learned-prior pick-quality sanity stats
        sys.stderr.write(block + "\n")
    return result, bench_bundle


def _exit_flushed(code: int) -> None:
    """Flush stdio and ``os._exit`` — the tune teardown skips Python finalization
    because a bench-timeout can leave a daemon NVRTC worker thread holding the CUDA
    context, which deadlocks cupy's atexit pool teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _tune_targets(args) -> list[tuple[str, str | None, str | None, list[str] | None]]:
    """The ``(label, code, input, dynamic)`` shapes this invocation tunes — the **only**
    place golden and non-golden diverge. ``--dataset golden`` expands to every recorded
    golden shape (deduped by name, ``--kernel SUBSTR`` narrowing); otherwise it's the
    single ``--code`` / positional input / ``--golden NAME`` target. ``dynamic`` is the
    ``--dynamic NAME@INPUT:AXIS`` spec list the target traces with: a dynamic golden's
    own recorded spec, or the CLI flag for an ad-hoc target. ``handle_tune`` then loops
    over this list uniformly, so one shape and the whole golden set share one codepath.
    Exits 2 on a degenerate / conflicting source."""
    if getattr(args, "dataset", None):
        from deplodock.compiler.pipeline.search.data import Dataset

        require_source(args, {"golden"}, "tune --dataset only supports 'golden' (db rows have no shape to tune)")
        if args.code or args.input or getattr(args, "golden", None):
            logger.error("--dataset golden is mutually exclusive with --code / positional input / --golden NAME")
            sys.exit(2)
        if getattr(args, "dynamic", None):
            logger.error("--dynamic is incompatible with --dataset golden (a dynamic golden's spec is part of its config)")
            sys.exit(2)
        # Configs under one name share the shape (and dynamic spec), so any one's
        # snippet is interchangeable.
        by_name: dict[str, tuple[str, list[str] | None]] = {}
        for s in Dataset.from_golden(kernel=args.kernel).samples:
            by_name.setdefault(s.name, (s.snippet, list(s.dynamic) if s.dynamic else None))
        if not by_name:
            logger.error("no golden shapes matched --kernel %r", args.kernel)
            sys.exit(2)
        return [(name, code, None, dyn) for name, (code, dyn) in by_name.items()]

    resolve_golden_arg(args)  # --golden NAME → args.code (+ args.dynamic for a dynamic golden)
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    return [(args.code or args.input, args.code, args.input, getattr(args, "dynamic", None))]


def _bench_dump(args):
    """Per-target dump dir. ``--bench`` reads the ``.torch.json`` provenance
    reproducers from a dump dir; route through a temp dir when no ``--dump-dir`` was
    given (HTML is only written for a real ``--dump-dir`` / ``DEPLODOCK_DUMP_DIR``).
    Returns ``(dump, tmp_dir_or_None)``."""
    from deplodock.compiler.pipeline.dump import CompilerDump

    dump = CompilerDump.resolve(args.dump_dir)
    if args.bench and dump is None:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="deplodock-tune-bench-"))
        return CompilerDump(dir=tmp), tmp
    return dump, None


def handle_tune(args):
    if not getattr(args, "dataset", None) and not args.code and not args.input and not getattr(args, "golden", None):
        # No op to tune → offline mode: refit the learned prior on its persisted
        # dataset and print diagnostics (reachability, calibration, golden coverage).
        _tune_offline(args)
        return

    targets = _tune_targets(args)  # one shape, or the whole golden set — same loop below

    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.search import SearchDB

    setup_pipeline_runtime(args)
    # tune compiles at -Xcicc -O1 by default to dodge a cicc/LLVM blowup on big
    # unrolled register-tile kernels (up to ~200x faster compile). The trade-off:
    # -O1 latencies are a RANKING signal, NOT -O3-optimal — reduction / attention
    # kernels can run 1.5-3x slower. Re-bench the winner at -O3 (``tune --bench``,
    # or ``deplodock run --bench``) for deployable numbers.
    nvcc_flags = apply_nvcc_flags(args, default="-Xcicc -O1")
    if "-O1" in nvcc_flags or "-O0" in nvcc_flags:
        logger.info(
            "tune compiling at cicc %s — latencies are a RANKING signal, not -O3-optimal",
            "-O1" if "-O1" in nvcc_flags else "-O0",
        )

    db_path = resolve_tune_db()  # ``DEPLODOCK_TUNE_DB`` env overrides the default path
    if args.clean:  # one shape or many: a fresh sweep clears once, then accumulates
        _clean_caches(db_path)
    db = SearchDB(path=db_path)
    logger.info("Tuning DB: %s", db_path)
    # One device-pinned bench worker per GPU (subprocess-isolated) + one prior shared
    # across every target — benching can't dirty the parent, so a single long-lived
    # process loops cleanly. ``[None]`` (default) = one unpinned worker = serial.
    devices = _resolve_devices(args)
    backends = [_tune_backend(device_id=d) for d in devices]
    if len(backends) > 1:
        sys.stderr.write(f"[tune] per-kernel parallel across {len(backends)} GPUs: {[d for d in devices]}\n")
    ctx = Context.probe()

    multi = len(targets) > 1
    if multi:
        sys.stderr.write(f"[tune] {len(targets)} shape(s) into {db_path}{' (--clean)' if args.clean else ''}\n")
    done = 0
    for i, (label, code, inp, dyn) in enumerate(targets):
        args.code, args.input, args.dynamic = code, inp, dyn
        if multi:
            sys.stderr.write(f"\n[tune] === {i + 1}/{len(targets)}: {label} → {code} ===\n")
        dump, tmp_dump = _bench_dump(args)
        try:
            result, bench_bundle = _tune_one(args, backends=backends, db=db, ctx=ctx, dump=dump)
        except KeyboardInterrupt:
            # Per-op bests already landed in the DB as they were measured, so a re-run resumes.
            sys.stderr.write(f"\n[tune] interrupted{f' at {label}' if multi else ''} — partial per-op results are in the DB\n")
            _exit_flushed(0)
        except RuntimeError as exc:
            # A NotImplementedError is never the watchdog signal — it's a
            # compiler contract bug (e.g. an unconsumed AtomTile reaching
            # render); re-raise so the traceback isn't swallowed.
            if isinstance(exc, NotImplementedError):
                raise
            # Bench watchdog couldn't bail (GPU queue saturated) → the parent CUDA stream
            # is dirty, so the rest of the sweep can't run reliably here. Abort (the DB has
            # the per-op bests; a re-run resumes). os._exit bypasses the cupy atexit deadlock.
            sys.stderr.write(f"\n[tune] aborted{f' at {label}' if multi else ''}: {exc}\n")
            _exit_flushed(1)

        if result.best_reward is None:
            if not multi:
                sys.stderr.write("[tune] no kernels tuned — exiting without output\n")
        else:
            # Only write the assembled CUDA when ``--output`` is given (a multi-kB dump
            # to stdout after a long tune is noise; ``-o`` or ``compile`` replays it).
            if args.output and result.assembled is not None:
                Path(args.output).write_text(format_stage(result.assembled, "cuda"))
                logger.info("Saved cuda IR: %s", args.output)
            if args.bench and result.assembled is not None:
                _run_bench(args, bench_bundle, result.assembled, dump, html_dir=(dump.dir if dump and tmp_dump is None else None))
        if tmp_dump is not None:
            import shutil

            shutil.rmtree(tmp_dump, ignore_errors=True)
        done += 1

    if multi:
        sys.stderr.write(f"\n[tune] done: {done}/{len(targets)} shape(s)\n")
    _exit_flushed(0)

    _exit_flushed(0)


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
    # The learned-prior checkpoint file (a fresh sweep should start cold).
    for p in (config.prior_path(), config.prior_path().with_suffix(config.prior_path().suffix + ".tmp")):
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


# Parent wall-clock caps for the isolated deployable benches: on overrun the worker is SIGKILLed
# (frees the device, parent stays clean). Generous over real cold-start — a hung kernel is caught
# far sooner by the worker's own 1 s per-launch watchdog, which exits the child promptly. Full-model
# is larger: it reloads the HF module + traces + JITs torch.compile in the child.
_FULL_MODEL_BENCH_WALL_S = 300.0
_PER_KERNEL_BENCH_WALL_S = 120.0


def _run_bench(args, bench_bundle, assembled, dump, *, html_dir) -> None:
    """``tune --bench``: re-bench the tuned winner at -O3 (deployable numbers, NOT the
    -O1 ranking pass) — full model **against the real torch module** (eager /
    torch.compile / Deplodock) and each per-kernel ``.torch.json`` reproducer against
    its torch-ref reconstruction, each in the SIGKILL-able bench worker so a hung kernel
    can't wedge the run. Prints both tables and (when ``html_dir`` is set) writes an HTML
    per-kernel chart. ``bench_bundle = (module, args, kwargs) | None``; when ``None`` (an
    ``--ir`` JSON tune with no module) the full-model bench is skipped and only the
    per-kernel table runs."""
    from deplodock.commands.run import _collect_sym_env, _print_table, _symbolic_bench_note
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from deplodock.compiler.pipeline.search.db import SearchDB

    # Re-bench at -O3 (deployable) unless the user explicitly pinned --nvcc-flags;
    # tune searched at -O1, which is a ranking signal only. The lowering-fork
    # selection that tuning recorded is keyed by op_cache_key (opt-level independent),
    # so the -O1-tuned winners are still picked when re-benching here at -O3.
    bench_flags = args.nvcc_flags if args.nvcc_flags is not None else ""
    os.environ[config.NVCC_FLAGS] = bench_flags
    sys.stderr.write(f"\n[tune] --bench: re-benching at -O3 ({bench_flags or 'nvcc default -O3'}) — deployable numbers\n")

    # Build the bench backend with DEPLODOCK_DUMP_DIR cleared: CudaBackend defaults its
    # dump to CompilerDump.from_env(), whose __post_init__ would rmtree the dump dir —
    # destroying the .torch.json reproducers the assembled tuning run just wrote there
    # (and which the per-kernel bench reads). Clearing it also avoids per-launch dump
    # noise during benching. (No-op when tuning used a temp dump — the env wasn't set.)
    saved_dump_env = os.environ.pop(config.DUMP_DIR, None)
    # ``backend`` here is only the handle to resolve the tune DB (for the per-kernel re-lowering);
    # the benches themselves run in the SIGKILL-able worker (``benchmark_compare_isolated_async``), which
    # builds its own backend. DUMP_DIR is cleared so CudaBackend's default CompilerDump doesn't
    # rmtree the reproducer dir the per-kernel bench reads.
    backend = CudaBackend(tune_db="auto")
    if saved_dump_env is not None:
        os.environ[config.DUMP_DIR] = saved_dump_env
    db = SearchDB(path=backend.tune_db) if (backend.tune_db is not None and backend.tune_db.exists()) else None

    if bench_bundle is not None:
        sys.stderr.write("\n[tune] full-model bench (eager / torch.compile / deplodock):\n")
        # The worker rebuilds the real module from these args via ``load_or_trace`` (no live module
        # crosses the pipe) and runs the comparison in-child — a hung deplodock kernel hangs the
        # child, which the parent SIGKILLs, instead of wedging the run.
        trace_args = {
            "code": args.code,
            "input": args.input,
            "layer": args.layer,
            "seq_len": args.seq_len,
            "dynamic": getattr(args, "dynamic", None),
        }
        try:
            full, _, _, full_captured = asyncio.run(
                benchmark_compare_isolated_async(
                    lowered=assembled,
                    torch_spec=("trace_args", trace_args),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_FULL_MODEL_BENCH_WALL_S,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    nvcc_flags=bench_flags,
                )
            )
            # The worker tiled the torch inputs to the hint for a symbolic graph
            # (``_hint_sized_inputs`` inside ``bench_full_model_real``); label the
            # table so the reader knows the numbers are hint-shaped.
            capture_note = None if full_captured else "(graph-capture fallback: timings include host launch overhead)"
            sym_env = _collect_sym_env([assembled] if assembled is not None else [])
            notes = [n for n in (_symbolic_bench_note(sym_env), capture_note) if n]
            _print_table(full, note="\n".join(notes) if notes else None)
        except RuntimeError as exc:
            # Any worker failure (incl. a SIGKILL on a hung kernel) surfaces as RuntimeError. The
            # parent device stays clean — per-kernel runs in its own worker — so continue.
            sys.stderr.write(f"[tune] full-model bench failed ({exc}); continuing to per-kernel\n")
    else:
        sys.stderr.write("\n[tune] full-model bench skipped (no runnable module — --ir JSON path)\n")

    rows, fallback = _bench_per_kernel(args, dump.dir, db)
    if rows:
        _print_per_kernel_table(rows)
        if fallback:
            print(f"note: timed without CUDA graph capture (host dispatch included): {', '.join(fallback)}")
        if html_dir is not None:
            render_kernel_chart(rows, Path(html_dir) / "kernels.html")


def _bench_per_kernel(args, dump_dir, db):
    """Bench each kernel's ``.torch.json`` provenance reproducer (re-lowered greedily so the tuned
    DB-best forks are picked) vs eager / torch.compile / deplodock at -O3 — each in the SIGKILL-able
    worker (``benchmark_compare_isolated_async``). Re-lowering runs in the parent (CPU; greedy forks read
    the DB); only the GPU bench is isolated, so a hung / failed kernel skips just that reproducer and
    the sweep continues. Returns ``(rows, fallback)`` — ``rows`` is ``[(label, {backend: us})]``,
    ``fallback`` the labels that benched without CUDA graph capture (dispatch-inclusive timings)."""
    import json

    from deplodock.commands.run import _detect_stage, _passes_after_stage
    from deplodock.compiler.backend import torch_ref
    from deplodock.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline import Pipeline

    repros = final_kernel_repros(dump_dir)
    if not repros:
        return [], []
    bench_flags = os.environ.get(config.NVCC_FLAGS, "")
    sys.stderr.write(f"\n[tune] per-kernel bench: {len(repros)} reproducer(s) at -O3\n")
    rows: list[tuple[str, dict]] = []
    fallback: list[str] = []
    records: list[dict] = []  # persisted as 62_kernel_bench.json — the `deplodock compare` input
    for repro in repros:
        label = _short_kernel(repro.name)
        try:
            with open(repro) as f:
                g = Graph.from_dict(json.load(f))
            fe = g.copy() if torch_ref.is_runnable(g) else None
            tail = _passes_after_stage(_detect_stage(g))
            # No dump here — re-creating a CompilerDump on the repro dir would rmtree it.
            lowered = Pipeline.build(tail).run(g, db=db) if tail else g
            results, _, _, captured = asyncio.run(
                benchmark_compare_isolated_async(
                    lowered=lowered,
                    torch_spec=("frontend_graph", fe),
                    bench_backends=args.bench_backends,
                    wall_timeout_s=_PER_KERNEL_BENCH_WALL_S,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                    nvcc_flags=bench_flags,
                )
            )
        except Exception as exc:  # noqa: BLE001 — isolated, so a hung / failed kernel skips just this one
            sys.stderr.write(f"[tune]   {label}: skipped ({exc})\n")
            continue
        rows.append((label, results))
        records.append({"kernel": repro.name.removesuffix(".torch.json"), "label": label, "captured": captured, "backends": results})
        if not captured:
            fallback.append(label)
        dp = results.get("Deplodock")
        sys.stderr.write(f"[tune]   {label}: deplodock={dp:.0f}us\n" if dp is not None else f"[tune]   {label}: (no result)\n")
    if records:
        # Per-kernel -O3 bench results in machine-readable form, beside the table /
        # kernels.html — the per-kernel input `deplodock compare <dumpA> <dumpB>` diffs.
        (Path(dump_dir) / "62_kernel_bench.json").write_text(json.dumps(records, indent=2, default=str))
    return rows, fallback


def final_kernel_repros(dump_dir):
    """The ``.torch.json`` provenance reproducers from the last (CUDA) stage dump."""
    dump_dir = Path(dump_dir)
    kernel_dirs = sorted(dump_dir.glob("*.kernels"))
    return sorted(kernel_dirs[-1].glob("*.torch.json")) if kernel_dirs else []


def _short_kernel(name: str) -> str:
    """Readable kernel label: drop the ``.torch.json`` suffix + trailing structural hash."""
    import re

    return re.sub(r"_[0-9a-f]{6}$", "", name.removesuffix(".torch.json"))


def _fmt_us(us) -> str:
    return f"{us:.0f}" if us is not None else "-"


def _print_per_kernel_table(rows) -> None:
    from deplodock.commands.table import Col, render_table  # noqa: PLC0415

    cols = [Col("Kernel"), Col("eager", "r"), Col("tcompile", "r"), Col("deplodock", "r"), Col("vs eager", "r")]
    data = []
    for label, res in sorted(rows, key=lambda kv: kv[1].get("Deplodock") or 0.0, reverse=True):
        eager = res.get("Eager PyTorch")
        dp = res.get("Deplodock")
        spd = f"{eager / dp:.2f}x" if (eager and dp) else "-"
        data.append([label, _fmt_us(eager), _fmt_us(res.get("torch.compile")), _fmt_us(dp), spd])
    print()
    for line in render_table(cols, data, rule=True):
        print(line)


def render_kernel_chart(rows, out_html) -> None:
    """Render the per-kernel latency comparison as a horizontal bar chart (HTML + a
    best-effort PNG) via :mod:`deplodock.visualize`."""
    from deplodock.visualize import Bar, BarChart, render_bar_chart

    rows = sorted(rows, key=lambda kv: kv[1].get("Deplodock") or 0.0, reverse=True)
    n_vs = sum("Eager PyTorch" in res for _, res in rows)
    chart = BarChart(
        categories=[label for label, _ in rows],
        bars=[
            Bar("Deplodock", [res.get("Deplodock") for _, res in rows], color="#4dabf7"),
            Bar("Eager PyTorch", [res.get("Eager PyTorch") for _, res in rows], color="#999999"),
            Bar("torch.compile", [res.get("torch.compile") for _, res in rows], color="#ffd166"),
        ],
        value_name="latency (µs) — lower is faster",
        title="tune --bench — per-kernel latency (-O3)",
        subtitle=f"{len(rows)} kernels benched from their .torch.json reproducers ({n_vs} torch-comparable, rest deplodock-only).",
        orientation="horizontal",
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_bar_chart(chart, theme="dark", transparent=True))
    sys.stderr.write(f"[tune]   chart → {out_html}\n")
    try:
        from deplodock.visualize import render_image

        png = out_html.with_suffix(".png")
        render_image(out_html.read_text(), png, height=max(300, 40 * len(rows)))
        sys.stderr.write(f"[tune]   png   → {png}\n")
    except Exception as exc:  # noqa: BLE001 — PNG needs the [visualize] extra (playwright)
        sys.stderr.write(f"[tune]   png skipped: {exc}\n")
