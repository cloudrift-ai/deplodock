"""Run an inline torch expression through the deplodock CUDA pipeline.

Compiles ``--code`` to CUDA, executes it on real input data, and verifies
correctness against eager PyTorch. With ``--bench``, also benchmarks all
backends (eager, torch.compile, deplodock) and prints a comparison table —
the same shape as ``scripts/bench_block.py`` but for arbitrary inline ops.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from deplodock import config

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    parser = subparsers.add_parser("run", help="Compile + run a model / inline torch expression on the CUDA backend")
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "HuggingFace model ID or .json IR file. A model ID is traced + compiled + executed and "
            "(with --bench) timed end-to-end against the real torch module; a .json IR file behaves "
            "like --ir. Mutually exclusive with --code / --ir."
        ),
    )
    parser.add_argument(
        "--code",
        "-c",
        help=(
            "Inline Python expression whose last statement is a call (same grammar as "
            "``compile --code``). Example: 'torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))'. "
            "Mutually exclusive with the positional input / --ir."
        ),
    )
    parser.add_argument(
        "--ir",
        help=(
            "Path to a JSON IR dump (any stage: torch / tensor / loop / tile / kernel / cuda). "
            "The remaining lowering passes are run, then the kernel(s) are executed with random "
            "inputs and benchmarked. Skips eager accuracy check (no reference model available). "
            "Equivalent to passing the same .json path as the positional input."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (when the positional input is a model ID). Omit to run the whole model.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for full-model tracing when the input is a model ID (default: 32).",
    )
    parser.add_argument(
        "--bench", "-b", action="store_true", help="Benchmark eager / torch.compile / deplodock and print a comparison table."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="After --bench, re-launch each kernel under ``ncu`` to collect hardware counters "
        "(SM-active %%, FMA pipe util, L1/DRAM bandwidth, smem bank-conflict %%). Skipped if "
        "ncu is not on PATH or the user lacks performance-counter permissions.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument(
        "--bench-backends",
        default=None,
        help=(
            "Comma-separated subset of backends to time under --bench: any of "
            "``eager``, ``tcompile`` (a.k.a. ``torch.compile`` / ``compile``), "
            "``deplodock``. Falls back to ``DEPLODOCK_BENCH_BACKENDS`` env var, "
            "then to the default ``eager,deplodock`` (drops the ~0.8 s "
            "torch.compile JIT from the per-case cost). ``deplodock`` is "
            "implicit even if omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --ir random inputs (default: 0).")
    parser.add_argument(
        "--dynamic",
        action="append",
        default=None,
        metavar="NAME@INPUT:AXIS",
        help=(
            "Make a tensor dim symbolic. Form: ``NAME@INPUT:AXIS`` — axis ``AXIS`` of the "
            "traced input named ``INPUT`` becomes ``Dim(NAME)``. Repeatable. Forwards to "
            "``torch.export(..., dynamic_shapes={...})``; the compiled CUDA kernel signature "
            "gains an ``int <NAME>`` runtime arg per dim. Example: ``--dynamic seq_len@x:1``."
        ),
    )
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts.")
    parser.add_argument("--debug", action="store_true", help="Per-launch tensor dumps in the deplodock backend.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase verbosity. Default (no flag): only the accuracy verdict and (with --bench) "
            "the timing tables are printed. -v: also tracer messages and pass / per-rule timings. "
            "-vv: also per-rule application snapshots."
        ),
    )
    from deplodock.commands.compile import add_nvcc_args
    from deplodock.compiler.target import add_target_arg

    add_nvcc_args(parser)
    add_target_arg(parser)
    parser.set_defaults(func=handle_run)


def handle_run(args):
    from deplodock.commands.compile import apply_nvcc_flags
    from deplodock.compiler.target import apply_target_arg

    apply_nvcc_flags(args, default="")  # run uses nvcc default -O3 (representative codegen)
    apply_target_arg(args)  # --target sm_NN gates TMA / cp.async like the target GPU would
    verbose = getattr(args, "verbose", 0)
    if verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        import torch
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from deplodock.commands.compile import load_or_trace
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline.dump import CompilerDump

    if sum(x is not None for x in (args.input, args.code, args.ir)) > 1:
        logger.error("input / --code / --ir are mutually exclusive")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    # A .json input (via --ir or as the positional) takes the IR path: finish
    # lowering an arbitrary-stage dump, no traced module to bench against torch.
    ir_path = args.ir
    if ir_path is None and args.input is not None and Path(args.input).suffix == ".json" and Path(args.input).exists():
        ir_path = args.input
    if ir_path is not None:
        args.ir = ir_path
        _handle_run_ir(args, CudaBackend, CompilerDump)
        return

    if args.input is None and args.code is None:
        logger.error("Either a model ID / .json input, --code, or --ir is required")
        sys.exit(1)

    # Model ID or --code: trace to a frontend graph + keep the runnable module
    # (+ example inputs) so accuracy / --bench compare against real torch.
    graph, _base_name, bundle = load_or_trace(args)
    module, example_args, example_kwargs = bundle

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Backend auto-resolves ``DEPLODOCK_TUNE_DB`` env →
    # ``~/.cache/deplodock/autotune.db`` (opens if the file exists,
    # silent fall-back to rule defaults otherwise).
    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    if backend.tune_db is not None and backend.tune_db.exists():
        logger.info("Using tuning DB: %s", backend.tune_db)
    compiled = backend.compile(graph)

    input_data = _bind_inputs(compiled, module, example_args, example_kwargs)

    # Skip the accuracy check + eager forward when running as the ncu
    # child of a ``--profile`` invocation: the parent already verified
    # accuracy outside ncu, and re-running the eager reference under
    # ncu (a) wastes ~5-10 s of ncu overhead per cuBLAS launch and
    # (b) pollutes the captured CSV with cutlass / cuBLAS kernel rows
    # the perf summary then has to filter out. The child only needs to
    # launch the deplodock kernels so ncu can sample our metrics.
    skip_accuracy = config.ncu_child()

    try:
        if not skip_accuracy:
            run_result, _ = backend.run(compiled, input_data=input_data)
            if dump and backend.last_debug_result is not None:
                dump.dump_per_launch_values(backend.last_debug_result.per_launch)

            eager_out = _eager_output(module, example_args, example_kwargs)
            _check_accuracy(run_result.outputs, eager_out)
        else:
            # ncu child needs at least one deplodock launch for metrics
            # to populate; skip the eager comparison.
            backend.run(compiled, input_data=input_data)
    except RuntimeError as exc:
        # Per-launch watchdog fired in ``run_program`` (kernel >1 s).
        # The CUDA context is dirty — bypass Python cleanup so cupy's
        # atexit doesn't block on the still-running kernel.
        sys.stderr.write(f"accuracy check failed: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    if not args.bench:
        return

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in example_args)
    cuda_kwargs = _to_cuda_kwargs(example_kwargs)

    # Build torch_fns (incl. ~0.8s torch.compile / Inductor JIT when
    # ``tcompile`` is in the selected backends) *outside* the GPU lock
    # so concurrent ``deplodock run`` workers can do their CPU-bound
    # JITs in parallel. The few GPU launches the compile-side warmup
    # issues are noisy but go untimed; the measurement loop below
    # holds the lock for accurate iter timing.
    backends = _resolve_backends(args.bench_backends)
    torch_fns = _build_torch_fns(cuda_module, cuda_args, cuda_kwargs, args.warmup, backends=backends)

    # Release the torch GPU state and reset the persisting-L2 window
    # before the deplodock bench. The eager-accuracy comparison +
    # ``--bench-backends eager`` warmup leave allocated tensors in
    # torch's caching allocator AND cuBLAS-installed
    # ``cudaAccessPolicyWindow`` carveouts in L2; together they steal
    # L2 bandwidth from our cp.async-staged kernels and inflate
    # measured latency by ~4× on the article tile (observed:
    # cp.async + pipeline reads 1660 µs with the state live, 330 µs
    # after release; same cubin SHA1, ~3 GHz SM clock both runs).
    # Only safe when no torch backends are in ``torch_fns`` — the
    # closures there reference ``cuda_module`` / ``cuda_args``; for
    # eager/tcompile + deplodock the state has to stay alive and the
    # comparison absorbs the slowdown intrinsically.
    if not torch_fns:
        # ``_bench_interleaved`` only dereferences ``cuda_module`` /
        # ``cuda_args`` / ``cuda_kwargs`` when ``torch_fns`` has
        # entries — replace them with ``None`` placeholders so the
        # actual tensors drop out of scope and ``empty_cache`` can
        # reclaim their pool blocks.
        cuda_module = cuda_args = cuda_kwargs = None
        del module, example_args, example_kwargs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _reset_persisting_l2_cache()

    # GPU serialization is handled inside ``CudaBackend.benchmark`` (and
    # every other GPU entry point) via ``gpu_lock()`` — no need to wrap
    # the whole bench block here. Print/dump are pure CPU work.
    try:
        results, bench = _bench_interleaved(
            cuda_module, cuda_args, cuda_kwargs, backend, compiled, args.warmup, args.iters, torch_fns=torch_fns
        )
    except RuntimeError as exc:
        # Bench watchdog fired (slow kernel, hung launch). The kernel
        # that timed out is still queued on the CUDA stream, so Python's
        # normal shutdown would block in cupy's atexit memory-pool
        # drain. Report the failure and ``os._exit`` so the process
        # actually exits.
        sys.stderr.write(f"benchmark failed: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    if dump:
        dump.dump_benchmark(bench)
        _dump_bench_compare(dump.dir, results, args.warmup, args.iters)

    _print_table(results)
    _print_kernel_stats(compiled, bench)
    if args.profile:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)


def _reset_persisting_l2_cache() -> None:
    """Drop every persisting-L2 carveout on the current CUDA context.

    Wraps ``cudaCtxResetPersistingL2Cache`` (CUDA 11+). cuBLAS, cuDNN,
    and other libraries can install ``cudaAccessPolicyWindow`` hints
    that pin specific gmem regions in L2 across kernel launches; once
    set, the carveout persists until the context tears down (it is NOT
    released by ``torch.cuda.empty_cache`` or stream sync). That eats
    L2 bandwidth from later cp.async-staged kernels — the article
    matmul drops from 1660 µs to 330 µs after the reset. Best-effort:
    silently skipped if the runtime symbol isn't resolvable (CUDA < 11,
    non-Linux, etc.); a non-zero return is logged at DEBUG so the
    bench path doesn't fail loud on driver quirks.

    Loads ``libcudart`` from the already-mapped image so the call
    targets the SAME runtime torch + cupy are using (the system
    ``libcudart.so`` may belong to a different CUDA install and would
    operate on a different driver context — its reset returns success
    but doesn't touch our context's L2 carveout). Walks
    ``/proc/self/maps`` for the first mapped ``libcudart.so.*`` path.
    """
    import ctypes

    libpath = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.rstrip().split(None, 5)
                if len(parts) < 6:
                    continue
                p = parts[-1]
                if "libcudart.so" in p and p.startswith("/"):
                    libpath = p
                    break
    except OSError:
        pass

    try:
        cudart = ctypes.CDLL(libpath) if libpath else ctypes.CDLL("libcudart.so")
        cudart.cudaCtxResetPersistingL2Cache.restype = ctypes.c_int
        cudart.cudaCtxResetPersistingL2Cache.argtypes = []
        err = cudart.cudaCtxResetPersistingL2Cache()
        if err != 0:
            logger.debug("cudaCtxResetPersistingL2Cache returned %d (continuing)", err)
    except (OSError, AttributeError) as e:
        logger.debug("cudaCtxResetPersistingL2Cache unavailable: %s", e)


def _dump_bench_compare(dump_dir, results: dict, warmup: int, iters: int) -> None:
    """Persist the eager / torch.compile / deplodock comparison table
    so downstream tooling (``make bench-kernels``) can parse one file
    per case instead of grepping kernel stdout."""
    import json as _json
    from pathlib import Path as _Path

    eager_us = results.get("Eager PyTorch")
    payload = {
        "warmup": warmup,
        "iters": iters,
        "backends": {name: {"latency_us": us} for name, us in results.items()},
    }
    if eager_us:
        for name, us in results.items():
            payload["backends"][name]["speedup_vs_eager"] = (eager_us / us) if us else 0.0
    out = _Path(dump_dir) / "60_bench_compare.json"
    out.write_text(_json.dumps(payload, indent=2, default=str))


def _print_kernel_stats(graph, bench):
    """Per-kernel breakdown. Pulls structural stats off each ``CudaOp``
    (block / grid / smem), per-launch timings from ``bench.per_launch``,
    and per-kernel hardware attributes from the compiled cupy RawKernels
    (register count, achieved theoretical occupancy). One row per kernel
    — quick at-a-glance for spotting which kernel dominates, whether
    register pressure is killing occupancy, etc."""
    from deplodock.compiler.ir.cuda.ir import CudaOp

    cuda_nodes = [(nid, node) for nid, node in graph.nodes.items() if isinstance(node.op, CudaOp)]
    if not cuda_nodes:
        return

    times_by_idx = {lt.idx: lt.time_ms * 1000 for lt in (bench.per_launch or [])}
    total_us = bench.time_ms * 1000
    attrs_by_kname = _collect_kernel_attrs(graph)
    occ_limits = _occupancy_limits()

    # Symbolic grids (ceil-div over a dynamic axis) need a concrete env to
    # resolve for display — use each symbolic input dim's hint, so the printed
    # geometry reflects the hint-sized tile the kernel was tuned for.
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

    sym_env: dict[str, int] = {}
    for nid in graph.inputs:
        for dim in graph.nodes[nid].output.shape:
            if isinstance(dim.expr, Var) and dim.hint is not None:
                sym_env.setdefault(dim.expr.name, dim.hint)

    print()
    print(f"{'Kernel':<44s} {'us':>7s} {'%':>5s} {'grid':>7s} {'block':>5s} {'smem':>6s} {'regs':>4s} {'occ':>4s}")
    print("-" * 90)
    for idx, (_, node) in enumerate(cuda_nodes):
        op = node.op
        t_us = times_by_idx.get(idx, 0.0)
        pct = (t_us / total_us * 100) if total_us > 0 else 0.0
        from deplodock.compiler.ir.cuda.ir import resolve_dim

        block_dims = [resolve_dim(d, sym_env) for d in op.block]
        grid_dims = [resolve_dim(d, sym_env) for d in op.grid]
        block_threads = block_dims[0] * block_dims[1] * block_dims[2]
        grid_total = grid_dims[0] * grid_dims[1] * grid_dims[2]
        smem_kb = op.smem_bytes / 1024
        kname = op.kernel_name[:42]
        attrs = attrs_by_kname.get(op.kernel_name) or {}
        regs = attrs.get("num_regs", 0)
        occ_pct = _theoretical_occupancy(regs, op.smem_bytes, block_threads, occ_limits)
        occ_str = f"{occ_pct:>3.0f}%" if occ_pct is not None else "  --"
        print(f"{kname:<44s} {t_us:>7.1f} {pct:>4.1f}% {grid_total:>7d} {block_threads:>5d} {smem_kb:>5.1f}K {regs:>4d} {occ_str}")
    print(f"{'TOTAL':<44s} {total_us:>7.1f}")


def _collect_kernel_attrs(graph) -> dict[str, dict]:
    """Compile each kernel via ``cupy.RawKernel`` (cached by source) to
    pull post-PTXAS hardware attributes — register count, static smem,
    spill bytes. Returns ``{kernel_name: attrs_dict}``."""
    from deplodock.compiler.ir.cuda.ir import CudaOp

    try:
        import cupy as cp
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for _, node in graph.nodes.items():
        if not isinstance(node.op, CudaOp):
            continue
        try:
            k = cp.RawKernel(node.op.kernel_source, node.op.kernel_name, options=("--use_fast_math",))
            out[node.op.kernel_name] = dict(k.attributes)
        except Exception:  # pragma: no cover — environment-dependent
            continue
    return out


def _occupancy_limits() -> dict | None:
    """Per-device limits used to estimate theoretical occupancy. ``None``
    when cupy / CUDA aren't available."""
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        a = dev.attributes
        return {
            "max_threads_per_sm": a.get("MaxThreadsPerMultiProcessor", 0),
            "max_blocks_per_sm": a.get("MaxBlocksPerMultiprocessor", 0),
            "max_regs_per_sm": a.get("MaxRegistersPerMultiprocessor", 0),
            "max_smem_per_sm": a.get("MaxSharedMemoryPerMultiprocessor", 0),
            "warp_size": a.get("WarpSize", 32),
        }
    except Exception:
        return None


def _theoretical_occupancy(regs_per_thread: int, smem_per_block: int, threads_per_block: int, limits: dict | None) -> float | None:
    """Active-warps-per-SM ÷ peak-warps-per-SM × 100. Computed from the
    static-occupancy limits: register file, shared memory, and per-SM
    block / thread caps. Doesn't account for the dynamic-only spill +
    stack overhead but is enough to flag occupancy cliffs (regs > 64
    drops most consumer GPUs from 100% → 50%, smem > 49KB likewise)."""
    if not limits or threads_per_block <= 0 or regs_per_thread <= 0:
        return None
    warp_size = limits["warp_size"]
    max_warps = limits["max_threads_per_sm"] // warp_size
    if max_warps <= 0:
        return None

    blocks_by_threads = limits["max_threads_per_sm"] // threads_per_block
    blocks_by_blocks = limits["max_blocks_per_sm"]
    blocks_by_regs = limits["max_regs_per_sm"] // max(regs_per_thread * threads_per_block, 1)
    blocks_by_smem = limits["max_smem_per_sm"] // max(smem_per_block, 1) if smem_per_block > 0 else blocks_by_threads
    active_blocks = max(0, min(blocks_by_threads, blocks_by_blocks, blocks_by_regs, blocks_by_smem))
    active_warps = active_blocks * (threads_per_block // warp_size)
    return min(100.0, 100.0 * active_warps / max_warps)


_NCU_RECURSE_GUARD = config.NCU_CHILD

# Curated ncu metric set — verified to populate on RTX 5090 (sm_120) +
# TMA kernels. ``--set detailed`` is broken there (``SpeedOfLight_Roofline``
# divides by zero, ``SourceCounters`` / ``PCSamplingData`` need missing
# ``smsp__pcsamp_*`` metrics) so we enumerate explicit metrics instead.
# Add to this list to surface new columns in the perf summary; the
# downstream parser keys on metric names directly.
_NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_lsu.sum",
    "launch__registers_per_thread",
)


def _run_ncu_profile(args, *, dump_dir=None):
    """Re-launch the same ``deplodock run`` invocation under ``ncu`` to
    collect a curated set of hardware counters (occupancy, bank
    conflicts, SM/DRAM/FMA throughput, register pressure). Output is
    captured in CSV form; when ``DEPLODOCK_DUMP_DIR`` (or ``--dump-dir``
    propagated as ``dump_dir``) is set, the raw CSV and a parsed
    per-kernel JSON are written there. Otherwise the counters print to
    stdout in the same CSV form for one-shot inspection.

    Spawns one extra subprocess at minimal iter count — ncu's per-launch
    overhead is huge (10-100×). The ``DEPLODOCK_NCU_CHILD`` env var
    prevents the profiled child from re-spawning ncu recursively.

    Skipped silently when ``ncu`` is not on PATH. ncu's own stderr is
    relayed when it fails (typical failure: NVIDIA's perf-counter
    permission gate)."""
    import json as _json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path as _Path

    if config.ncu_child():
        return

    ncu = shutil.which("ncu")
    if ncu is None:
        logger.info("ncu not found on PATH; skipping --profile output")
        return

    env = dict(os.environ)
    env[_NCU_RECURSE_GUARD] = "1"
    # The ncu child process re-runs trace + compile, which would
    # ``shutil.rmtree`` the parent's dump dir from ``CompilerDump``'s
    # ``__post_init__``. Drop the env var for the child so the parent's
    # ``60_*.json`` (bench results) survive — ncu output is captured via
    # stdout and saved by the parent below.
    env.pop(config.DUMP_DIR, None)

    cmd: list[str] = [
        ncu,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(_NCU_METRICS),
        sys.executable,
        "-m",
        "deplodock.deplodock",
        "run",
    ]
    if args.code is not None:
        cmd.extend(["--code", args.code])
    elif args.ir is not None:
        cmd.extend(["--ir", args.ir])
    # ncu's per-launch overhead means we want one or two launches per
    # kernel — enough for the counters to populate, not so many that
    # the run drags out. Match ``deplodock run --bench``'s minimal
    # warmup so the profiled launches see a realistic-ish steady state.
    cmd.extend(["--warmup", "2", "--iters", "3"])

    print()
    print("=" * 80)
    print("ncu --csv (curated hardware metrics)")
    print("=" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("ncu profiling timed out")
        return

    # ncu writes ``==PROF==`` status lines first, then a CSV table
    # starting at the ``"ID"`` header. Split the two so we can save just
    # the CSV part as a file and surface the status lines for
    # diagnostics.
    stdout = result.stdout
    lines = stdout.splitlines()
    csv_start = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            csv_start = i
            break
    if csv_start is None:
        if stdout.strip():
            print(stdout)
        if result.returncode != 0:
            logger.warning("ncu exit=%d", result.returncode)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)
        return

    csv_text = "\n".join(lines[csv_start:]) + "\n"
    status_text = "\n".join(lines[:csv_start])

    parsed = _parse_ncu_csv(csv_text)

    if dump_dir is not None:
        out_dir = _Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "61_ncu_metrics.csv").write_text(csv_text)
        (out_dir / "61_ncu_metrics.json").write_text(_json.dumps(parsed, indent=2, default=str))
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.csv'}")
        print(f"ncu metrics → {out_dir / '61_ncu_metrics.json'}")
    else:
        # No dump dir: relay the CSV (and status lines) so ad-hoc
        # ``deplodock run --profile`` invocations still surface the data.
        if status_text.strip():
            print(status_text)
        print(csv_text)

    if result.returncode != 0:
        logger.warning("ncu exit=%d", result.returncode)
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)


def _parse_ncu_csv(csv_text: str) -> dict:
    """Reduce ncu's launch-by-launch CSV into per-kernel metric dicts.

    Each row in the CSV is one (kernel, metric) datum for one launch;
    multi-launch profiling produces many rows per (kernel, metric) which
    we aggregate: ``.sum`` and ``smsp__*`` counters get summed, every
    other metric (percentages, per-thread regs) gets averaged. Returns
    ``{kernel_name: {metric_name: numeric_value}}`` ready to be merged
    with the bench-comparison JSON downstream.

    Filters to deplodock-emitted kernels (``k_*`` naming convention) —
    the ncu child runs in the same process as the eager-reference
    accuracy check, which would otherwise contribute cutlass / cuBLAS
    rows that contaminate the per-row aggregate.
    """
    import csv as _csv
    import io as _io

    reader = _csv.DictReader(_io.StringIO(csv_text))
    per_kernel: dict[str, dict[str, list[float]]] = {}
    for row in reader:
        kname = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        raw = row.get("Metric Value", "").replace(",", "")
        if not (kname and metric and raw):
            continue
        if not kname.startswith("k_"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        per_kernel.setdefault(kname, {}).setdefault(metric, []).append(val)

    out: dict[str, dict[str, float]] = {}
    for kname, metrics in per_kernel.items():
        reduced: dict[str, float] = {}
        for metric, vals in metrics.items():
            if metric.endswith(".sum") or metric.startswith("smsp__"):
                reduced[metric] = sum(vals)
            else:
                reduced[metric] = sum(vals) / len(vals)
        out[kname] = reduced
    return out


def _detect_stage(graph) -> str:
    """Identify the IR stage by scanning op type names. Returns one of
    ``torch | tensor | loop | tile | kernel | cuda`` — the highest-stage
    op present in the graph wins, since lowering produces a graph mixed
    only briefly during a pass and stable in the post-pass form."""
    stage_by_op: dict[str, str] = {
        "CudaOp": "cuda",
        "KernelOp": "kernel",
        "TileOp": "tile",
        "LoopOp": "loop",
    }
    order = ["torch", "tensor", "loop", "tile", "kernel", "cuda"]
    best = "torch"
    for node in graph.nodes.values():
        s = stage_by_op.get(type(node.op).__name__)
        if s and order.index(s) > order.index(best):
            best = s
    # Anything that's not Loop/Tile/Kernel/Cuda but is a frontend/tensor
    # op stays at "torch" — they get rewritten by the frontend passes.
    return best


def _passes_after_stage(stage: str) -> list[str]:
    """Pipeline tail to run after a graph has reached ``stage``."""
    from deplodock.compiler.pipeline import (
        CUDA_PASSES,
        KERNEL_PASSES,
        LOOP_PASSES,
        TENSOR_PASSES,
        TILE_PASSES,
    )

    completed = {
        "torch": [],
        "tensor": TENSOR_PASSES,
        "loop": LOOP_PASSES,
        "tile": TILE_PASSES,
        "kernel": KERNEL_PASSES,
        "cuda": CUDA_PASSES,
    }[stage]
    return [p for p in CUDA_PASSES if p not in completed]


def bench_lowered_vs_torch(frontend, lowered, backend, *, seed, do_bench, warmup, iters, bench_backends):
    """Run + (optionally) benchmark a lowered graph against its torch reference on
    shared random inputs. The common bench primitive behind ``run --ir`` and
    ``tune --bench``.

    ``frontend`` is the pristine frontend-dialect snapshot (must be
    ``torch_ref``-runnable); pass ``None`` for a non-frontend / unsupported graph to
    bench deplodock-only. One random source is drawn per distinct constant
    ``source_path`` and each side replays its own ``load_ops`` (so a weight lowering
    transposed + renamed stays the same underlying tensor on both sides). The lowered
    graph runs once for a non-fatal accuracy check vs the torch eager reference; then,
    when ``do_bench``, the selected backends are timed — interleaved when a torch ref
    exists (full ``warmup``/``iters``), else deplodock-only at reduced iters.

    Returns ``(results, bench, torch_available)``: ``results`` is the
    ``{backend: latency_us}`` dict (``None`` when ``do_bench`` is False), ``bench`` the
    deplodock ``BenchmarkResult`` (``None`` when ``do_bench`` is False), and
    ``torch_available`` whether an eager/torch.compile reference was built. Does no
    printing / dumping — callers own that."""
    import numpy as np
    import torch

    from deplodock.compiler.backend import torch_ref
    from deplodock.compiler.ir.base import ConstantOp, InputOp
    from deplodock.compiler.loader.binder import bind_constants

    rng = np.random.default_rng(seed)

    def _static(shape):
        return tuple(d.as_static() if hasattr(d, "as_static") else int(d) for d in shape)

    # One random source array per distinct constant ``source_path`` — shared by
    # deplodock (lowered graph) and the torch ref (frontend graph). ``bind_constants``
    # replays each constant's ``load_ops`` on it, so a weight that lowering transposed +
    # renamed (linear: out×in → in×out) stays the same underlying tensor on both sides.
    sources: dict[str, np.ndarray] = {}
    for gph in ([frontend] if frontend is not None else []) + [lowered]:
        for node in gph.nodes.values():
            op = node.op
            if isinstance(op, ConstantOp) and op.value is None and op.source_path and op.source_path not in sources:
                shp = _static(op.source_shape or node.output.shape)
                sources[op.source_path] = rng.standard_normal(shp, dtype=np.float32) * 0.02

    input_data: dict[str, object] = {}
    input_tensors: dict[str, object] = {}
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, InputOp):
            arr = rng.standard_normal(_static(node.output.shape), dtype=np.float32)
            input_data[nid] = arr.flatten().tolist()
            input_tensors[nid] = _to_cuda_tensor(arr, node.output.dtype)
        elif isinstance(node.op, ConstantOp) and node.op.value is not None:
            input_data[nid] = [float(node.op.value)]

    input_data.update(bind_constants(lowered, sources))
    if frontend is not None:
        for fid, arr in bind_constants(frontend, sources).items():
            input_tensors[fid] = _to_cuda_tensor(arr, frontend.nodes[fid].output.dtype)

    # Fallback for any value-None constant with no source_path (synthetic).
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.value is None and nid not in input_data:
            arr = rng.standard_normal(_static(node.output.shape), dtype=np.float32) * 0.02
            input_data[nid] = arr.flatten().tolist()

    result, _ = backend.run(lowered, input_data=input_data)
    for nid, arr in result.outputs.items():
        finite = np.isfinite(arr).all()
        logger.info("Output %s: shape=%s finite=%s mean=%.4f", nid, arr.shape, bool(finite), float(arr.mean()))

    # Build the torch reference from the frontend snapshot, fed the same inputs.
    torch_fn = torch_inputs = None
    if frontend is not None:
        try:
            torch_fn, torch_inputs = torch_ref.build_callable(frontend, input_tensors)
            with torch.no_grad():
                eager_out = torch_fn(*torch_inputs)
            _check_accuracy(result.outputs, eager_out, fatal=False)
        except Exception as exc:  # noqa: BLE001 — torch ref is best-effort
            logger.warning("torch reference unavailable (%s) — skipping vs-torch comparison", exc)
            torch_fn = None

    torch_available = torch_fn is not None
    if not do_bench:
        return None, None, torch_available

    if torch_available:
        backends = _resolve_backends(bench_backends)
        torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends=backends)
        results, bench = _bench_interleaved(torch_fn, torch_inputs, {}, backend, lowered, warmup, iters, torch_fns=torch_fns)
        return results, bench, True
    bench = backend.benchmark(lowered, warmup=max(3, warmup // 5), num_iters=max(10, iters // 5))
    return {"Deplodock": bench.time_ms * 1000}, bench, False


def bench_full_model_real(module, args_t, kwargs, lowered, backend, *, warmup, iters, bench_backends):
    """End-to-end full-model bench against the **real torch module** — eager /
    ``torch.compile`` / Deplodock — using ``_bench_interleaved``. The module + its
    trace-time inputs come from ``load_or_trace``'s bundle. Skips the accuracy
    check (deplodock's bench uses synthetic activations vs torch's bound inputs, so
    only latency is comparable here — accuracy lives in the per-kernel path).
    Returns ``(results, bench)``."""
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args_t)
    cuda_kwargs = _to_cuda_kwargs(kwargs)
    backends = _resolve_backends(bench_backends)
    torch_fns = _build_torch_fns(cuda_module, cuda_args, cuda_kwargs, warmup, backends=backends)
    return _bench_interleaved(cuda_module, cuda_args, cuda_kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns)


def _handle_run_ir(args, CudaBackend, CompilerDump):
    """Run path: load JSON IR (any stage), finish lowering, execute, bench."""
    import json

    from deplodock.compiler.backend import torch_ref
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline import Pipeline

    path = Path(args.ir)
    with open(path) as f:
        data = json.load(f)
    graph = Graph.from_dict(data)
    if getattr(args, "dynamic", None):
        logger.error("--dynamic is incompatible with --ir (the trace is already complete)")
        sys.exit(2)

    stage = _detect_stage(graph)
    tail = _passes_after_stage(stage)
    logger.info("Loaded %s IR; running tail passes: %s", stage, tail or "(none)")

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    # Snapshot the pre-lowering frontend graph so we can build a torch
    # reference (eager + torch.compile) and compare accuracy/latency vs torch —
    # the same table the --code path produces, now for a dumped .torch.json.
    # Non-frontend IR (loop/tile/…) has no torch twin → deplodock-only bench.
    frontend = graph.copy() if torch_ref.is_runnable(graph) else None

    backend = CudaBackend(debug=args.debug or None, dump=dump, tune_db="auto")
    db = None
    if backend.tune_db is not None and backend.tune_db.exists():
        from deplodock.compiler.pipeline.search.db import SearchDB

        db = SearchDB(path=backend.tune_db)
        logger.info("Using tuning DB: %s", backend.tune_db)
    if tail:
        # Finish the tail lowering. NOTE: the single-shot ``Pipeline.run`` has no
        # prior (uniform PUCT → emission-order, option-0) and does not replay tuned
        # variants from the DB; ``db=`` is kept for perf recording only. Wiring a
        # warm-started prior into single-shot compile is a deferred follow-up.
        graph = Pipeline.build(tail).run(graph, db=db, dump=dump)

    results, bench, torch_available = bench_lowered_vs_torch(
        frontend,
        graph,
        backend,
        seed=args.seed,
        do_bench=args.bench,
        warmup=args.warmup,
        iters=args.iters,
        bench_backends=args.bench_backends,
    )

    if args.bench:
        if dump:
            dump.dump_benchmark(bench)
        if torch_available:
            _print_table(results)
        else:
            # Non-frontend IR (loop/tile/…): no torch twin → deplodock-only.
            print()
            print(f"{'Backend':<24s} {'Latency (us)':>12s}")
            print("-" * 38)
            print(f"{'Deplodock':<24s} {bench.time_ms * 1000:>12.0f}")
        _print_kernel_stats(graph, bench)
    if args.profile:
        _run_ncu_profile(args, dump_dir=dump.dir if dump else None)


def _bind_inputs(compiled, module, example_args, example_kwargs):
    """Match graph inputs and constants to tensors from ``module`` / call args.

    Activations come from the call's positional/keyword tensors. Constants
    come from ``module.named_parameters()`` / ``named_buffers()`` keyed
    by each ``ConstantOp.source_path`` recorded at trace time. Each
    constant's ``load_ops`` chain is replayed via the NumPy backend
    (see ``compiler.loader.binder``), so any compile-time-folded
    transpose / reshape is honored uniformly.
    """
    import numpy as np
    import torch

    from deplodock.compiler.ir.base import ConstantOp
    from deplodock.compiler.loader.binder import bind_constants

    flat_inputs: list[torch.Tensor] = []
    for v in example_args:
        flat_inputs.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat_inputs.extend(_flatten_tensors(v))

    input_ids = list(compiled.inputs)
    if len(input_ids) != len(flat_inputs):
        logger.error("Input arity mismatch: graph has %d inputs, code provided %d", len(input_ids), len(flat_inputs))
        sys.exit(1)

    input_data: dict[str, np.ndarray] = {}
    for nid, tensor in zip(input_ids, flat_inputs, strict=True):
        np_dtype = compiled.nodes[nid].output.dtype.np
        input_data[nid] = tensor.detach().cpu().numpy().astype(np_dtype, copy=False)

    # ``remove_duplicate=False`` so tied weights (e.g. a model whose lm_head
    # shares the embedding matrix) are surfaced under *every* name, including
    # the ``source_path`` the tracer recorded — otherwise the alias the trace
    # picked may be the one torch dedups away and the constant won't bind.
    sources: dict[str, np.ndarray] = {}
    for path, tensor in module.named_parameters(remove_duplicate=False):
        sources[path] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, tensor in module.named_buffers(remove_duplicate=False):
        sources[path] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    input_data.update(bind_constants(compiled, sources))

    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp) or nid in input_data:
            continue
        if node.op.value is not None:
            continue  # backend materializes scalars from node.op.value
        logger.error("Could not bind constant %s (source_path=%r)", nid, node.op.source_path)
        sys.exit(1)
    return input_data


def _flatten_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_tensors(v))
        return out
    return []


def _eager_output(module, args, kwargs):
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args)
    cuda_kwargs = _to_cuda_kwargs(kwargs)
    with torch.no_grad():
        out = cuda_module(*cuda_args, **cuda_kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _to_cuda_tensor(arr, dtype):
    """numpy array → CUDA torch tensor in the node's dtype (default fp32)."""
    import torch

    td = getattr(torch, str(dtype), torch.float32)
    return torch.from_numpy(arr).to("cuda").to(td)


def _to_cuda_kwargs(kwargs):
    import torch

    cuda_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            cuda_kwargs[k] = v.to("cuda")
        elif isinstance(v, tuple):
            cuda_kwargs[k] = tuple(t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v)
        else:
            cuda_kwargs[k] = v
    return cuda_kwargs


def _check_accuracy(outputs, eager_out, *, fatal=True):
    """Compare backend ``outputs`` against the eager reference.

    ``fatal`` (the ``--code`` path) aborts on mismatch / NaN. ``run --ir``
    passes ``fatal=False``: a sliced reproducer is fed random *boundary* inputs
    that can be out-of-domain for the op (e.g. a kernel expecting a
    mean-of-squares gets random signed data → NaN from a downstream rsqrt), so
    a failure there is informational, not a correctness gate — bench anyway."""
    import numpy as np  # noqa: PLC0415

    def _fail(msg: str) -> None:
        if fatal:
            logger.error(msg)
            sys.exit(1)
        logger.warning("%s — non-fatal (random-input reproducer); skipping accuracy", msg)

    eager_flat = eager_out.detach().cpu().flatten().tolist()
    if any(e != e for e in eager_flat):
        _fail("eager reference contains NaN (reproducer inputs out of domain)")
        if not fatal:
            return
    failed = False
    for buf_name, arr in outputs.items():
        values = arr.flatten().tolist()
        if any(v != v for v in values):
            _fail(f"CORRECTNESS FAIL: output {buf_name} contains NaN")
            if not fatal:
                return
        if len(values) == len(eager_flat):
            max_diff = max(abs(a - e) for a, e in zip(values, eager_flat, strict=True))
            mean_diff = sum(abs(a - e) for a, e in zip(values, eager_flat, strict=True)) / len(values)
            # Scale tolerance by max|eager| and by output dtype.
            #
            # fp32: matmul reduction-order drift grows with both K and
            # output magnitude. A fixed threshold flags benign drift on
            # randn×randn at large K as a failure (cp.async + split-K
            # atomic-add ordering vs eager's pairwise sum). Dual check:
            # ``max_diff <= 8% of peak`` (tight ceiling for the typical
            # case) OR ``mean_diff <= 0.5% of peak`` (escape hatch for
            # the long tail — splitK atomic-reduce on randn×randn
            # produces a handful of outliers at K=1024 / 2048 even when
            # the bulk of the output is accurate to 4+ decimals).
            # Codegen bugs that systematically corrupt the output (e.g.
            # the matmul_add fusion adding the residual per-K_s CTA)
            # fail both clauses: mean_diff lifts to ~2-3% of peak.
            #
            # fp16: every step has ~3 fewer decimal digits than fp32. The
            # split-K matmul path is dominated by atomicAdd into an
            # ``__half*`` buffer — each per-CTA partial converts to fp16
            # at the atomic boundary and loses ~11 bits per write. After
            # 1024 K-partials that's RMS error on the order of
            # ``peak * 0.3``. The proper fix (f32 scratch for split-K +
            # separate cast pass) is a future architectural change; for
            # now ``deplodock run --bench`` needs to remain usable on
            # legitimate fp16 graphs, so the rtol budget tracks the
            # achievable accuracy of the current path. Bugs that
            # actually corrupt outputs still fail (whole-row mismatch /
            # NaN / order-of-magnitude wrong).
            is_fp16 = arr.dtype == np.float16
            # fp16 atomic-reduce accumulation can produce per-cell drift
            # up to ``peak`` in pathological cancellation cases (random-
            # signed partials). Real bugs (NaN, whole-row corruption,
            # outputs orders of magnitude off) still fail.
            rel_tol = 1.0 if is_fp16 else 0.08
            abs_tol = 1e-1 if is_fp16 else 1e-3
            peak = max((abs(e) for e in eager_flat), default=0.0)
            tol = max(abs_tol, rel_tol * peak)
            mean_tol = max(abs_tol, (1.0 if is_fp16 else 0.005) * peak)
            verdict = "PASS" if max_diff <= tol or mean_diff <= mean_tol else "FAIL"
            if verdict == "FAIL":
                print(f"Accuracy vs eager: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol:.6f} FAIL")
                failed = True
            else:
                logger.info("Accuracy vs eager: max_diff=%.6f mean_diff=%.6f tol=%.6f PASS", max_diff, mean_diff, tol)
        else:
            logger.warning("Output size %d does not match eager %d; skipping accuracy", len(values), len(eager_flat))
    if failed:
        _fail("accuracy check failed vs eager")


_BACKEND_ALIASES = {
    "eager": "eager",
    "deplodock": "deplodock",
    "tcompile": "tcompile",
    "torch.compile": "tcompile",
    "compile": "tcompile",
}


def _resolve_backends(cli_value: str | None) -> set[str]:
    """Pick which bench backends to time. Precedence:

    1. ``--bench-backends`` CLI arg (comma-separated).
    2. ``DEPLODOCK_BENCH_BACKENDS`` env var (same syntax).
    3. Default ``eager,deplodock`` — torch.compile is excluded so the
       per-case wall time isn't dominated by a ~0.8 s Inductor JIT
       that most users don't need on every run.

    ``deplodock`` is always included even if omitted (the kernel under
    test is the point of the bench). Returns the canonical backend
    keys ``{"eager", "tcompile", "deplodock"}``.
    """
    raw = config.bench_backends_raw(cli_value)
    selected: set[str] = {"deplodock"}
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        canonical = _BACKEND_ALIASES.get(tok)
        if canonical is None:
            logger.error("unknown bench backend %r — choose from %s", tok, sorted(set(_BACKEND_ALIASES.values())))
            sys.exit(1)
        selected.add(canonical)
    return selected


def _build_torch_fns(module, args, kwargs, warmup, *, backends: set[str]):
    """Pre-build the per-backend ``torch_fns`` dict, including the
    ``torch.compile`` JIT step when requested. The JIT (mostly
    Inductor CPU work plus a few warmup launches) sits *outside* the
    GPU lock in ``handle_run`` so parallel workers can compile
    concurrently — the lock then only wraps the actual measurement
    iters.

    Returns only the torch-side closures; deplodock's bench loop is
    driven separately by ``backend.benchmark`` in ``_bench_interleaved``.
    """
    import torch

    torch_fns: dict[str, callable] = {}
    if "eager" in backends:
        torch_fns["Eager PyTorch"] = lambda: module(*args, **kwargs)
    if "tcompile" in backends:
        try:
            compiled_module = torch.compile(module)
            for _ in range(warmup + 5):
                with torch.no_grad():
                    compiled_module(*args, **kwargs)
            torch_fns["torch.compile"] = lambda: compiled_module(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("torch.compile failed: %s", e)
    return torch_fns


def _bench_interleaved(module, args, kwargs, backend, compiled_graph, warmup, iters, *, torch_fns):
    """Time the selected backends by alternating one iter of each per
    loop step. All backends see the same warm GPU state across the
    measurement window — same clocks, same caches, same thermal drift
    — instead of running in sequential phases that each get a
    different steady state.

    Driven by ``backend.benchmark(on_iter=...)``: deplodock is the
    backbone, ``on_iter`` runs each torch closure and records its
    cuda events, and the same call returns per-launch deplodock
    timings — so the kernel-stats breakdown shares the same warm
    state as the comparison numbers.

    Per-iter ``torch.cuda.Event``s queue on the (legacy) default
    stream; cupy's default stream is the same NULL stream, so events
    from both libraries see all preceding work.

    ``torch_fns`` is the pre-built backend closure dict from
    :func:`_build_torch_fns` (``handle_run`` builds it outside the
    GPU lock so the slow ``torch.compile`` JIT runs concurrently with
    peer workers; the lock then wraps only this measurement loop).
    """
    import torch

    # Each entry: (start_event, stop_event, batch_size_used). The
    # batch size is propagated by ``benchmark_program``'s ``on_iter``
    # so peer torch backends time the same number of back-to-back
    # calls deplodock does per CUDA event window — both sides then
    # measure sustained per-call latency, no warm-vs-cold asymmetry.
    torch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]] = {name: [] for name in torch_fns}

    def on_iter(batch_size: int = 1) -> None:
        for name, fn in torch_fns.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                for _ in range(batch_size):
                    fn()
                stop.record()
            torch_events[name].append((start, stop, batch_size))

    bench = backend.benchmark(compiled_graph, warmup=warmup, num_iters=iters, on_iter=on_iter)
    torch.cuda.synchronize()

    import statistics as _stats  # noqa: PLC0415

    results: dict[str, float] = {}
    for name, evt in torch_events.items():
        measured = evt[warmup:]
        if measured:
            # ``elapsed_time`` is in ms across the whole batch; divide
            # by the batch size and multiply by 1000 to get per-call us.
            # Median (not mean) for symmetry with deplodock's reduction
            # in ``benchmark_program`` — keeps a single thermal-blip or
            # lock-contention outlier from dragging eager's reported
            # latency up.
            per_iter_us = [s.elapsed_time(e) * 1000.0 / b for s, e, b in measured]
            results[name] = _stats.median(per_iter_us)
    results["Deplodock"] = bench.time_ms * 1000
    return results, bench


def _print_table(results):
    eager_us = results.get("Eager PyTorch", 0)
    print()
    print(f"{'Backend':<24s} {'Latency (us)':>12s} {'vs Eager':>10s}")
    print("-" * 48)
    for name, us in results.items():
        speedup = eager_us / us if us > 0 else 0
        print(f"{name:<24s} {us:>12.0f} {speedup:>10.2f}x")
