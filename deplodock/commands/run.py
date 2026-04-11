"""Run a compiled graph IR through the full pipeline."""

import logging

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    parser = subparsers.add_parser("run", help="Run a compiled graph IR file")
    parser.add_argument("ir_file", help="Path to a .json Graph IR file")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (timed iterations)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--dump-dir", default=None, help="Directory to dump intermediate compilation artifacts")
    parser.set_defaults(func=_handle_run)


def _handle_run(args):
    import json
    from pathlib import Path

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.ir import Graph
    from deplodock.compiler.plan import plan_graph
    from deplodock.compiler.rewriter import PassTrace, Rewriter

    dump = CompilerDump.resolve(args.dump_dir)

    ir_path = Path(args.ir_file)
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))

    if dump:
        dump.dump_input_graph(graph)

    from deplodock.compiler.fusion import auto_fuse

    # Apply decomposition + matmul recognition.
    rules_dir = Path(__file__).parent.parent / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    pass_traces: list[PassTrace] = []
    compiled = rewriter.apply(graph, pass_traces=pass_traces)

    if dump:
        dump.dump_passes(pass_traces)

    # Auto-fuse remaining ops into regions.
    compiled = auto_fuse(compiled)

    if dump:
        dump.dump_fused_graph(compiled)

    # Plan from graph.
    plan = plan_graph(compiled, name=ir_path.stem)
    backend = CudaBackend()
    program = backend.compile(plan)

    if dump:
        dump.dump_plan(plan)
        dump.dump_program(program)
        from deplodock.compiler.backend.cuda.program import generate_source

        mode = "benchmark" if args.benchmark else "run"
        dump.dump_source(generate_source(program, mode=mode))

    op_counts: dict[str, int] = {}
    for op in plan.ops:
        op_counts[op.op] = op_counts.get(op.op, 0) + 1
    logger.info(
        "Graph %s: %d ops (%s)",
        ir_path.name,
        len(plan.ops),
        ", ".join(f"{v} {k}" for k, v in sorted(op_counts.items())),
    )

    if args.benchmark:
        result = backend.benchmark(program, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
        if dump:
            dump.dump_benchmark(result)
    else:
        result = backend.run(program)
        for buf_name, values in result.outputs.items():
            logger.info("Output %s: %d elements, first 5: %s", buf_name, len(values), values[:5])
        if dump:
            dump.dump_result(result)
