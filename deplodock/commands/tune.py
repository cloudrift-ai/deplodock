"""Autotune CudaOps produced by the lowering pipeline and cache results."""

from __future__ import annotations

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


def handle_tune(args):
    resolve_golden_arg(args)  # --golden NAME → --code <snippet>
    if args.code and args.input:
        logger.error("--code and positional input are mutually exclusive")
        sys.exit(2)
    if not args.code and not args.input:
        # No op to tune → offline mode: refit the learned prior on its persisted
        # dataset and print diagnostics (reachability, calibration, golden coverage).
        _tune_offline(args)
        return

    import time

    from deplodock.commands.tune_progress import TuneProgress
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.dump import CompilerDump
    from deplodock.compiler.pipeline.search import SearchDB
    from deplodock.compiler.pipeline.search.two_level import run_two_level_tune

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

    graph, _, bench_bundle = load_or_trace(args)

    # ``--bench`` benches the assembled tuned graph against the **real torch module**
    # end-to-end (eager / torch.compile / deplodock). ``bench_bundle`` carries the
    # runnable module + its trace-time example inputs from ``load_or_trace`` — None
    # only when the input was an ``--ir`` JSON file (no module). With a bundle we
    # use the real module; without, the full-model bench is skipped (only the
    # per-kernel table runs).

    dump = CompilerDump.resolve(args.dump_dir)
    # Per-kernel bench reads the ``.torch.json`` provenance reproducers the assembled
    # run emits into the dump dir. When --bench is set but no dump dir was given, route
    # them through a temp dir (HTML is only written when the user gave a real
    # --dump-dir / DEPLODOCK_DUMP_DIR).
    html_dir = dump.dir if dump is not None else None
    tmp_dump_dir = None
    if args.bench and dump is None:
        import tempfile

        tmp_dump_dir = Path(tempfile.mkdtemp(prefix="deplodock-tune-bench-"))
        dump = CompilerDump(dir=tmp_dump_dir)
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

    # Live progress bar — default verbosity on a tty only. Disabled under -v
    # (the [tune] INFO lines show progress instead) and -q (errors only), and
    # when stderr is redirected (no \r smearing in piped logs).
    progress = TuneProgress(
        enabled=getattr(args, "verbose", 0) == 0 and not getattr(args, "quiet", False) and sys.stderr.isatty(),
    )

    patience = args.patience if args.patience is not None else config.tune_patience(50)
    explore_eps = args.explore_eps if args.explore_eps is not None else config.tune_eps(0.0)
    ctx = Context.probe()
    t0 = time.monotonic()
    try:
        result = run_two_level_tune(
            graph,
            ctx=ctx,
            db=db,
            backend=backend,
            patience=patience,
            ucb_c=args.ucb_c,
            explore_eps=explore_eps,
            dump=dump,
            progress=progress,
            prior_seed=args.seed,
        )
    except KeyboardInterrupt:
        # Manual abort: per-op bests already landed in the DB as they were
        # measured, so a re-run resumes. Nothing structured to print here.
        progress.close()
        sys.stderr.write("\n[tune] interrupted (Ctrl-C) — partial per-op results are persisted in the DB\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except RuntimeError as exc:
        # An autotune-internal raise (bench watchdog couldn't bail in time
        # because the GPU queue is saturated). The CUDA stream is dirty —
        # bypass Python cleanup so cupy's atexit doesn't deadlock on the
        # still-running launch.
        progress.close()
        sys.stderr.write(f"\n[tune] aborted: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    progress.close()

    elapsed = time.monotonic() - t0
    sys.stderr.write(f"\n[tune] done: {result.n_terminals} fused terminal(s) in {elapsed:.1f}s\n")
    for block in result.prior_summaries:  # learned-prior pick-quality sanity stats
        sys.stderr.write(block + "\n")
    if result.best_reward is None:
        sys.stderr.write("[tune] no kernels tuned — exiting without output\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Only write the assembled (DB-best) CUDA source when ``--output`` is
    # given. Dumping a multi-kB kernel to stdout after a long tune is noise —
    # callers that want it can pass ``-o`` (or re-run ``deplodock compile``,
    # which replays the same cached forks).
    if args.output and result.assembled is not None:
        Path(args.output).write_text(format_stage(result.assembled, "cuda"))
        logger.info("Saved cuda IR: %s", args.output)

    if args.bench and result.assembled is not None:
        _run_bench(args, bench_bundle, result.assembled, dump, html_dir=html_dir)
    if tmp_dump_dir is not None:
        import shutil

        shutil.rmtree(tmp_dump_dir, ignore_errors=True)

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


def _run_bench(args, bench_bundle, assembled, dump, *, html_dir) -> None:
    """``tune --bench``: re-bench the tuned winner at -O3 (deployable numbers, NOT the
    -O1 ranking pass) — full model **against the real torch module** (eager /
    torch.compile / Deplodock) and each per-kernel ``.torch.json`` reproducer against
    its torch-ref reconstruction. Prints both tables and (when ``html_dir`` is set)
    writes an HTML per-kernel chart. ``bench_bundle = (module, args, kwargs) | None``;
    when ``None`` (an ``--ir`` JSON tune with no module) the full-model bench is
    skipped and only the per-kernel table runs."""
    from deplodock.commands.run import _print_table, bench_full_model_real
    from deplodock.compiler.backend.cuda.backend import CudaBackend
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
    # Generous per-stage budgets: a whole-model bench compiles 100s of -O3 kernels
    # (the lm-head alone can take ~5-10 s) and the first iteration warms DRAM / L2 /
    # cubin loads for the full weights, so the default 10 s run + 30 s compile are
    # tight. Bump both to 60 s so first-iter cold-start doesn't bench_fail. The
    # bench is in-process (on_iter callback) — bench_wall_timeout_s stays None.
    backend = CudaBackend(tune_db="auto", bench_compile_timeout_s=60.0, bench_run_timeout_s=60.0)
    if saved_dump_env is not None:
        os.environ[config.DUMP_DIR] = saved_dump_env
    db = SearchDB(path=backend.tune_db) if (backend.tune_db is not None and backend.tune_db.exists()) else None

    if bench_bundle is not None:
        module, ex_args, ex_kwargs = bench_bundle
        sys.stderr.write("\n[tune] full-model bench (eager / torch.compile / deplodock):\n")
        try:
            full, _ = bench_full_model_real(
                module,
                ex_args,
                ex_kwargs,
                assembled,
                backend,
                warmup=args.warmup,
                iters=args.iters,
                bench_backends=args.bench_backends,
            )
            _print_table(full)
        except RuntimeError as exc:
            # A whole-graph compile/bench failure (e.g. a slow-compiling kernel) must not
            # cost the per-kernel table — report and fall through.
            sys.stderr.write(f"[tune] full-model bench failed ({exc}); continuing to per-kernel\n")
    else:
        sys.stderr.write("\n[tune] full-model bench skipped (no runnable module — --ir JSON path)\n")

    rows = _bench_per_kernel(args, dump.dir, backend, db)
    if rows:
        _print_per_kernel_table(rows)
        if html_dir is not None:
            render_kernel_chart(rows, Path(html_dir) / "kernels.html")


def _bench_per_kernel(args, dump_dir, backend, db):
    """Bench each kernel's ``.torch.json`` provenance reproducer (re-lowered greedily
    so the tuned DB-best forks are picked) vs eager / torch.compile / deplodock at -O3.
    Returns ``[(label, {backend: us})]``. Per-kernel failures are skipped; a bench
    watchdog ``RuntimeError`` dirties the CUDA context unrecoverably, so it stops the
    per-kernel sweep (we still report what we have)."""
    import json

    from deplodock.commands.run import _detect_stage, _passes_after_stage, bench_lowered_vs_torch
    from deplodock.compiler.backend import torch_ref
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline import Pipeline

    repros = final_kernel_repros(dump_dir)
    if not repros:
        return []
    sys.stderr.write(f"\n[tune] per-kernel bench: {len(repros)} reproducer(s) at -O3\n")
    rows: list[tuple[str, dict]] = []
    for repro in repros:
        label = _short_kernel(repro.name)
        try:
            with open(repro) as f:
                g = Graph.from_dict(json.load(f))
            fe = g.copy() if torch_ref.is_runnable(g) else None
            tail = _passes_after_stage(_detect_stage(g))
            # No dump here — re-creating a CompilerDump on the repro dir would rmtree it.
            lowered = Pipeline.build(tail).run(g, db=db) if tail else g
            results, _, _ = bench_lowered_vs_torch(
                fe,
                lowered,
                backend,
                seed=args.seed,
                do_bench=True,
                warmup=args.warmup,
                iters=args.iters,
                bench_backends=args.bench_backends,
            )
        except RuntimeError as exc:
            sys.stderr.write(f"[tune]   {label}: bench aborted ({exc}); stopping per-kernel sweep\n")
            break
        except Exception as exc:  # noqa: BLE001 — skip a kernel that fails to lower / run
            sys.stderr.write(f"[tune]   {label}: skipped ({exc})\n")
            continue
        rows.append((label, results))
        dp = results.get("Deplodock")
        sys.stderr.write(f"[tune]   {label}: deplodock={dp:.0f}us\n" if dp is not None else f"[tune]   {label}: (no result)\n")
    return rows


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
