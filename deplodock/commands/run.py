"""Run a compiled graph IR through the structural compiler + CUDA backend."""

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
    from deplodock.compiler.ir.graph import Graph

    dump = CompilerDump.resolve(args.dump_dir)

    ir_path = Path(args.ir_file)
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))

    if dump:
        dump.dump_input_graph(graph)

    backend = CudaBackend()
    compiled = backend.compile(graph)

    if dump:
        dump.dump_program(compiled)

    logger.info("Compiled %s: %d kernels", ir_path.name, len(compiled.launches))

    if args.benchmark:
        result = backend.benchmark(compiled, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
        if dump:
            dump.dump_benchmark(result)
    else:
        result = backend.run(compiled)
        for buf_name, arr in result.outputs.items():
            flat = arr.flatten()
            logger.info("Output %s: %d elements, first 5: %s", buf_name, flat.size, flat[:5].tolist())
        if dump:
            dump.dump_result(result)
