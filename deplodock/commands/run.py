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
    parser.add_argument("--bench", action="store_true", help="Benchmark eager / torch.compile / deplodock and print a comparison table.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for --bench (default: 10).")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for --bench (default: 100).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --ir random inputs (default: 0).")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts.")
    parser.add_argument("--debug", action="store_true", help="Per-launch tensor dumps in the deplodock backend.")
    parser.set_defaults(func=handle_run)


def handle_run(args):
    try:
        import torch
    except ImportError:
        logger.error("torch is required: pip install torch")
        sys.exit(1)

    from deplodock.commands.trace import trace_inline_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline.dump import CompilerDump

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required")
        sys.exit(1)

    if args.ir is not None:
        if args.code is not None:
            logger.error("--ir and --code are mutually exclusive")
            sys.exit(1)
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
    const_targets = info["const_targets"]

    dump = CompilerDump.resolve(args.dump_dir)
    if dump:
        dump.dump_input_graph(graph)

    backend = CudaBackend(debug=args.debug or None, dump=dump)
    compiled = backend.compile(graph)

    input_data = _bind_inputs(compiled, module, example_args, example_kwargs, const_targets)

    run_result = backend.run(compiled, input_data=input_data)
    if dump and backend.last_debug_result is not None:
        dump.dump_per_launch_values(backend.last_debug_result.per_launch)

    eager_out = _eager_output(module, example_args, example_kwargs)
    _check_accuracy(run_result.outputs, eager_out)

    if not args.bench:
        return

    results: dict[str, float] = {}
    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in example_args)
    cuda_kwargs = _to_cuda_kwargs(example_kwargs)

    results["Eager PyTorch"] = _bench_eager(cuda_module, cuda_args, cuda_kwargs, args.warmup, args.iters)
    compiled_us = _bench_torch_compile(cuda_module, cuda_args, cuda_kwargs, args.warmup, args.iters)
    if compiled_us is not None:
        results["torch.compile"] = compiled_us

    bench = backend.benchmark(compiled, warmup=max(3, args.warmup // 5), num_iters=max(10, args.iters // 5))
    results["Deplodock"] = bench.time_ms * 1000
    if dump:
        dump.dump_benchmark(bench)

    _print_table(results)
    _print_kernel_stats(compiled, bench)


def _print_kernel_stats(graph, bench):
    """Per-kernel breakdown. Pulls structural stats off each ``CudaOp``
    (block / grid / smem) and per-launch timings from ``bench.per_launch``.
    Prints one row per kernel — quick at-a-glance for spotting which kernel
    dominates and whether smem / launch geometry looks right."""
    from deplodock.compiler.ir.cuda.ir import CudaOp

    cuda_nodes = [(nid, node) for nid, node in graph.nodes.items() if isinstance(node.op, CudaOp)]
    if not cuda_nodes:
        return

    times_by_idx = {lt.idx: lt.time_ms * 1000 for lt in (bench.per_launch or [])}
    total_us = bench.time_ms * 1000

    print()
    print(f"{'Kernel':<48s} {'us':>8s} {'%':>5s} {'grid':>8s} {'block':>6s} {'smem':>6s} {'thr':>5s}")
    print("-" * 95)
    for idx, (_, node) in enumerate(cuda_nodes):
        op = node.op
        t_us = times_by_idx.get(idx, 0.0)
        pct = (t_us / total_us * 100) if total_us > 0 else 0.0
        block_threads = op.block[0] * op.block[1] * op.block[2]
        grid_total = op.grid[0] * op.grid[1] * op.grid[2]
        smem_kb = op.smem_bytes / 1024
        kname = op.kernel_name[:46]
        print(f"{kname:<48s} {t_us:>8.1f} {pct:>4.1f}% {grid_total:>8d} {block_threads:>6d} {smem_kb:>5.1f}K {block_threads:>5d}")
    print(f"{'TOTAL':<48s} {total_us:>8.1f}")


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

    if not args.bench:
        return

    bench = backend.benchmark(graph, warmup=max(3, args.warmup // 5), num_iters=max(10, args.iters // 5))
    print()
    print(f"{'Backend':<24s} {'Latency (us)':>12s}")
    print("-" * 38)
    print(f"{'Deplodock':<24s} {bench.time_ms * 1000:>12.0f}")
    if dump:
        dump.dump_benchmark(bench)
    _print_kernel_stats(graph, bench)


def _bind_inputs(compiled, module, example_args, example_kwargs, const_targets):
    """Match graph inputs and constants to tensors from ``module`` / call args."""
    import torch

    from deplodock.compiler.ir.base import ConstantOp

    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    flat_inputs: list[torch.Tensor] = []
    for v in example_args:
        flat_inputs.extend(_flatten_tensors(v))
    for v in example_kwargs.values():
        flat_inputs.extend(_flatten_tensors(v))

    input_ids = list(compiled.inputs)
    if len(input_ids) != len(flat_inputs):
        logger.error("Input arity mismatch: graph has %d inputs, code provided %d", len(input_ids), len(flat_inputs))
        sys.exit(1)

    input_data: dict[str, list[float]] = {}
    for nid, tensor in zip(input_ids, flat_inputs, strict=True):
        input_data[nid] = tensor.detach().cpu().flatten().tolist()

    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp):
            continue
        target = const_targets.get(node.op.name)
        tensor = None
        if target is not None:
            tensor = params.get(target)
            if tensor is None:
                tensor = buffers.get(target)
        if tensor is None and node.op.value is not None:
            input_data[nid] = [float(node.op.value)]
            continue
        if tensor is None:
            logger.error("Could not bind constant %s (target=%r)", nid, target)
            sys.exit(1)
        input_data[nid] = tensor.detach().cpu().flatten().tolist()
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
            verdict = "PASS" if max_diff < 1.0 else "FAIL"
            logger.info("Accuracy vs eager: max_diff=%.6f mean_diff=%.6f %s", max_diff, mean_diff, verdict)
            if verdict == "FAIL":
                failed = True
        else:
            logger.warning("Output size %d does not match eager %d; skipping accuracy", len(values), len(eager_flat))
    if failed:
        sys.exit(1)


def _bench_eager(module, args, kwargs, warmup, iters):
    import torch

    for _ in range(warmup):
        with torch.no_grad():
            module(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            module(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000


def _bench_torch_compile(module, args, kwargs, warmup, iters):
    import torch

    try:
        compiled = torch.compile(module)
        for _ in range(warmup + 5):
            with torch.no_grad():
                compiled(*args, **kwargs)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        logger.warning("torch.compile failed: %s", e)
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            compiled(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000


def _print_table(results):
    eager_us = results.get("Eager PyTorch", 0)
    print()
    print(f"{'Backend':<24s} {'Latency (us)':>12s} {'vs Eager':>10s}")
    print("-" * 48)
    for name, us in results.items():
        speedup = eager_us / us if us > 0 else 0
        print(f"{name:<24s} {us:>12.0f} {speedup:>10.2f}x")
