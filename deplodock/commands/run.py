"""Run an inline torch expression through the deplodock CUDA pipeline.

Compiles ``--code`` to CUDA, executes it on real input data, and verifies
correctness against eager PyTorch. With ``--bench``, also benchmarks all
backends (eager, torch.compile, deplodock) and prints a comparison table —
the same shape as ``scripts/bench_block.py`` but for arbitrary inline ops.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    parser = subparsers.add_parser("run", help="Compile + run an inline torch expression on the CUDA backend")
    parser.add_argument(
        "--code",
        "-c",
        help=(
            "Inline Python expression whose last statement is a call (same grammar as "
            "``compile --code``). Example: 'torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))'. "
            "Mutually exclusive with --ir."
        ),
    )
    parser.add_argument(
        "--ir",
        help=(
            "Path to a JSON IR dump (any stage: torch / tensor / loop / tile / kernel / cuda). "
            "The remaining lowering passes are run, then the kernel(s) are executed with random "
            "inputs and benchmarked. Skips eager accuracy check (no reference model available)."
        ),
    )
    parser.add_argument(
        "--bench", "-b", action="store_true", help="Benchmark eager / torch.compile / deplodock and print a comparison table."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="After --bench, re-launch each kernel under ``ncu`` to collect hardware counters "
        "(SM-active %, FMA pipe util, L1/DRAM bandwidth, smem bank-conflict %). Skipped if "
        "ncu is not on PATH or the user lacks performance-counter permissions.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --ir random inputs (default: 0).")
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
    parser.set_defaults(func=handle_run)


def handle_run(args):
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

    from deplodock.commands.trace import trace_inline_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline.dump import CompilerDump

    if args.ir is not None and args.code is not None:
        logger.error("--ir and --code are mutually exclusive")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    if args.ir is not None:
        _handle_run_ir(args, CudaBackend, CompilerDump)
        return

    if args.code is None:
        logger.error("Either --code or --ir is required")
        sys.exit(1)

    info = trace_inline_code(args.code)
    graph = info["graph"]
    module = info["module"]
    example_args = info["args"]
    example_kwargs = info["kwargs"]

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    backend = CudaBackend(debug=args.debug or None, dump=dump)
    compiled = backend.compile(graph)

    input_data = _bind_inputs(compiled, module, example_args, example_kwargs)

    run_result = backend.run(compiled, input_data=input_data)
    if dump and backend.last_debug_result is not None:
        dump.dump_per_launch_values(backend.last_debug_result.per_launch)

    eager_out = _eager_output(module, example_args, example_kwargs)
    _check_accuracy(run_result.outputs, eager_out)

    if not args.bench:
        return

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in example_args)
    cuda_kwargs = _to_cuda_kwargs(example_kwargs)

    results = _bench_interleaved(cuda_module, cuda_args, cuda_kwargs, backend, compiled, args.warmup, args.iters)

    # Per-kernel breakdown still uses the dedicated deplodock-only path so we
    # get per-launch CUDA event timings.
    bench = backend.benchmark(compiled, warmup=max(3, args.warmup // 5), num_iters=max(10, args.iters // 5))
    if dump:
        dump.dump_benchmark(bench)

    _print_table(results)
    _print_kernel_stats(compiled, bench)
    if args.profile:
        _run_ncu_profile(args)


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

    print()
    print(f"{'Kernel':<44s} {'us':>7s} {'%':>5s} {'grid':>7s} {'block':>5s} {'smem':>6s} {'regs':>4s} {'occ':>4s}")
    print("-" * 90)
    for idx, (_, node) in enumerate(cuda_nodes):
        op = node.op
        t_us = times_by_idx.get(idx, 0.0)
        pct = (t_us / total_us * 100) if total_us > 0 else 0.0
        block_threads = op.block[0] * op.block[1] * op.block[2]
        grid_total = op.grid[0] * op.grid[1] * op.grid[2]
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


_NCU_RECURSE_GUARD = "DEPLODOCK_NCU_CHILD"


def _run_ncu_profile(args):
    """Re-launch the same ``deplodock run`` invocation under ``ncu`` and
    pass the detailed-page text output through verbatim. ncu's own
    per-section breakdown (Speed-of-Light, Occupancy, Memory Workload,
    Compute Workload, Scheduler Stats, Warp State, Source Counters,
    plus per-kernel optimization hints) is far richer than any single
    metric set we'd curate, so we just relay it.

    Spawns one extra subprocess at minimal iter count — ncu's per-launch
    overhead is huge (10-100×). The ``DEPLODOCK_NCU_CHILD`` env var
    prevents the profiled child from re-spawning ncu recursively.

    Skipped silently when ``ncu`` is not on PATH. ncu's own stderr is
    relayed when it fails (typical failure: NVIDIA's perf-counter
    permission gate)."""
    import os
    import shutil
    import subprocess
    import sys

    if os.environ.get(_NCU_RECURSE_GUARD):
        return

    ncu = shutil.which("ncu")
    if ncu is None:
        logger.info("ncu not found on PATH; skipping --profile output")
        return

    env = dict(os.environ)
    env[_NCU_RECURSE_GUARD] = "1"

    # Explicit section list — `--set detailed` pulls in two sections that
    # break on TMA kernels in ncu 2025.3.1: `SpeedOfLight_Roofline`
    # (`float division by zero`) and `SourceCounters` /
    # `PCSamplingData` (missing `smsp__pcsamp_*` metrics). Either one
    # raises and poisons every subsequent section to NaN. Enumerating
    # the sections that work sidesteps both.
    sections = (
        "SpeedOfLight",
        "ComputeWorkloadAnalysis",
        "MemoryWorkloadAnalysis",
        "LaunchStats",
        "Occupancy",
    )
    cmd: list[str] = [
        ncu,
        "--target-processes",
        "all",
        "--page",
        "details",
    ]
    for s in sections:
        cmd.extend(["--section", s])
    cmd.extend(
        [
            sys.executable,
            "-m",
            "deplodock.deplodock",
            "run",
        ]
    )
    if args.code is not None:
        cmd.extend(["--code", args.code])
    elif args.ir is not None:
        cmd.extend(["--ir", args.ir])
    # Minimal iters — ncu's overhead means we only want one launch per kernel.
    cmd.extend(["--warmup", "1", "--iters", "1"])

    print()
    print("=" * 80)
    print("ncu --set detailed (per-kernel hardware analysis)")
    print("=" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("ncu profiling timed out")
        return

    # ncu writes its analysis to stdout. Relay it directly. If ncu failed,
    # show stderr too so the user sees the diagnostic.
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        logger.warning("ncu exit=%d", result.returncode)
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)


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


def _handle_run_ir(args, CudaBackend, CompilerDump):
    """Run path: load JSON IR (any stage), finish lowering, execute, bench."""
    import json
    from pathlib import Path

    import numpy as np

    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.base import ConstantOp, InputOp
    from deplodock.compiler.pipeline import run_pipeline

    path = Path(args.ir)
    with open(path) as f:
        data = json.load(f)
    graph = Graph.from_dict(data)

    stage = _detect_stage(graph)
    tail = _passes_after_stage(stage)
    logger.info("Loaded %s IR; running tail passes: %s", stage, tail or "(none)")

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    if tail:
        graph = run_pipeline(graph, tail, dump=dump)

    rng = np.random.default_rng(args.seed)
    input_data: dict[str, list[float]] = {}
    for nid, node in graph.nodes.items():
        if isinstance(node.op, InputOp):
            shape = tuple(int(d) for d in node.output.shape)
            input_data[nid] = rng.standard_normal(shape, dtype=np.float32).flatten().tolist()
        elif isinstance(node.op, ConstantOp):
            if node.op.value is not None:
                input_data[nid] = [float(node.op.value)]
            else:
                shape = tuple(int(d) for d in node.output.shape)
                input_data[nid] = (rng.standard_normal(shape, dtype=np.float32) * 0.02).flatten().tolist()

    backend = CudaBackend(debug=args.debug or None, dump=dump)
    result = backend.run(graph, input_data=input_data)
    for nid, arr in result.outputs.items():
        finite = np.isfinite(arr).all()
        logger.info("Output %s: shape=%s finite=%s mean=%.4f", nid, arr.shape, bool(finite), float(arr.mean()))

    if args.bench:
        bench = backend.benchmark(graph, warmup=max(3, args.warmup // 5), num_iters=max(10, args.iters // 5))
        print()
        print(f"{'Backend':<24s} {'Latency (us)':>12s}")
        print("-" * 38)
        print(f"{'Deplodock':<24s} {bench.time_ms * 1000:>12.0f}")
        if dump:
            dump.dump_benchmark(bench)
        _print_kernel_stats(graph, bench)
    if args.profile:
        _run_ncu_profile(args)


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
        input_data[nid] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    sources: dict[str, np.ndarray] = {}
    for path, tensor in module.named_parameters():
        sources[path] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, tensor in module.named_buffers():
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


def _check_accuracy(outputs, eager_out):
    eager_flat = eager_out.detach().cpu().flatten().tolist()
    failed = False
    for buf_name, arr in outputs.items():
        values = arr.flatten().tolist()
        if any(v != v for v in values):
            logger.error("CORRECTNESS FAIL: output %s contains NaN", buf_name)
            sys.exit(1)
        if len(values) == len(eager_flat):
            max_diff = max(abs(a - e) for a, e in zip(values, eager_flat, strict=True))
            mean_diff = sum(abs(a - e) for a, e in zip(values, eager_flat, strict=True)) / len(values)
            # Scale tolerance by max|eager|: fp32 matmul reduction-order drift
            # grows with both K and output magnitude. A fixed threshold flags
            # benign drift on randn×randn at large K as a failure (cp.async +
            # split-K atomic-add ordering vs eager's pairwise sum). 5% of peak
            # eager magnitude is loose enough for legit drift, tight enough
            # that whole-row corruption / NaNs still fail.
            peak = max((abs(e) for e in eager_flat), default=0.0)
            tol = max(1e-3, 0.05 * peak)
            verdict = "PASS" if max_diff < tol else "FAIL"
            if verdict == "FAIL":
                print(f"Accuracy vs eager: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol:.6f} FAIL")
                failed = True
            else:
                logger.info("Accuracy vs eager: max_diff=%.6f mean_diff=%.6f tol=%.6f PASS", max_diff, mean_diff, tol)
        else:
            logger.warning("Output size %d does not match eager %d; skipping accuracy", len(values), len(eager_flat))
    if failed:
        sys.exit(1)


def _bench_interleaved(module, args, kwargs, backend, compiled_graph, warmup, iters):
    """Time eager / torch.compile / deplodock by alternating one iter of
    each per loop step. All three see the same warm GPU state across the
    measurement window — same clocks, same caches, same thermal drift —
    instead of running in three sequential phases that each get a
    different steady state.

    Per-iter ``torch.cuda.Event``s queue on the (legacy) default stream;
    cupy's default stream is the same NULL stream, so events from both
    libraries see all preceding work. We sync once at the end and sum
    per-iter elapsed times per backend.
    """
    import torch

    eager_fn = lambda: module(*args, **kwargs)  # noqa: E731

    compile_fn = None
    try:
        compiled_module = torch.compile(module)
        for _ in range(warmup + 5):
            with torch.no_grad():
                compiled_module(*args, **kwargs)
        compile_fn = lambda: compiled_module(*args, **kwargs)  # noqa: E731
    except Exception as e:  # noqa: BLE001
        logger.warning("torch.compile failed: %s", e)

    deplodock_run = backend.make_runner(compiled_graph)

    for _ in range(warmup):
        with torch.no_grad():
            eager_fn()
        deplodock_run()
    torch.cuda.synchronize()

    e_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e_stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if compile_fn else []
    c_stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if compile_fn else []
    d_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        with torch.no_grad():
            e_starts[i].record()
            eager_fn()
            e_stops[i].record()
            if compile_fn is not None:
                c_starts[i].record()
                compile_fn()
                c_stops[i].record()
        d_starts[i].record()
        deplodock_run()
        d_stops[i].record()
    torch.cuda.synchronize()

    eager_us = sum(e_starts[i].elapsed_time(e_stops[i]) for i in range(iters)) / iters * 1000
    dd_us = sum(d_starts[i].elapsed_time(d_stops[i]) for i in range(iters)) / iters * 1000

    results: dict[str, float] = {"Eager PyTorch": eager_us}
    if compile_fn is not None:
        compile_us = sum(c_starts[i].elapsed_time(c_stops[i]) for i in range(iters)) / iters * 1000
        results["torch.compile"] = compile_us
    results["Deplodock"] = dd_us
    return results


def _print_table(results):
    eager_us = results.get("Eager PyTorch", 0)
    print()
    print(f"{'Backend':<24s} {'Latency (us)':>12s} {'vs Eager':>10s}")
    print("-" * 48)
    for name, us in results.items():
        speedup = eager_us / us if us > 0 else 0
        print(f"{name:<24s} {us:>12.0f} {speedup:>10.2f}x")
