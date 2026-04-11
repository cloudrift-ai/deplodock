"""Run a GPU program: matmul or transformer block."""

import logging

logger = logging.getLogger(__name__)


def register_run_command(subparsers):
    parser = subparsers.add_parser("run", help="Run a GPU program (matmul or transformer block)")
    sub = parser.add_subparsers(dest="program_type", required=True)

    # deplodock run matmul --size 1024
    matmul_parser = sub.add_parser("matmul", help="Run a matmul kernel")
    matmul_parser.add_argument("--size", required=True, help="Matrix size (e.g., 1024 or 4096x2048x1024)")
    matmul_parser.add_argument("--strategy", default="naive", help="Kernel strategy (default: naive)")
    matmul_parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (timed iterations)")
    matmul_parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    matmul_parser.set_defaults(func=_handle_matmul)

    # deplodock run graph <ir_file>
    graph_parser = sub.add_parser("graph", help="Run a compiled graph IR file")
    graph_parser.add_argument("ir_file", help="Path to compiled .json Graph IR file")
    graph_parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode")
    graph_parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    graph_parser.set_defaults(func=_handle_graph)


def _parse_size(size_str: str) -> dict[str, int]:
    if "x" in size_str:
        parts = size_str.split("x")
        m, n = int(parts[0]), int(parts[1])
        k = int(parts[2]) if len(parts) > 2 else m
        return {"M": m, "N": n, "K": k}
    n = int(size_str)
    return {"M": n, "N": n, "K": n}


def _handle_matmul(args):
    from deplodock.compiler.backend.cuda.lower import MatmulConfig, lower_matmul_to_program
    from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
    from deplodock.compiler.rewriter import Rewriter

    dims = _parse_size(args.size)

    # Build matmul graph.
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("K", "N")), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", ("M", "K", "N")), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", ("M", "N")), node_id="red")
    g.outputs = [red]

    # Fuse.
    from pathlib import Path

    rules_dir = Path(__file__).parent.parent / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    g = rewriter.apply(g)

    config = MatmulConfig(strategy=args.strategy)
    program = lower_matmul_to_program(g, config, dims)

    logger.info("Matmul %dx%dx%d, strategy=%s", dims["M"], dims["N"], dims["K"], args.strategy)

    if args.benchmark:
        result = benchmark_program(program, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
    else:
        result = run_program(program)
        output = result.outputs.get("C", [])
        logger.info("Output: %d elements, first 5: %s", len(output), output[:5])


def _handle_graph(args):
    import json
    from pathlib import Path

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir import Graph
    from deplodock.compiler.plan import plan_graph
    from deplodock.compiler.rewriter import Rewriter

    ir_path = Path(args.ir_file)
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))

    # Apply decomposition + fusion.
    rules_dir = Path(__file__).parent.parent / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    compiled = rewriter.apply(graph)

    # Plan from graph.
    plan = plan_graph(compiled, name=ir_path.stem)
    backend = CudaBackend()
    program = backend.compile(plan)

    op_counts = {}
    for op in plan.ops:
        op_counts[op.op] = op_counts.get(op.op, 0) + 1
    logger.info("Graph %s: %d ops (%s)", ir_path.name, len(plan.ops), ", ".join(f"{v} {k}" for k, v in sorted(op_counts.items())))

    if args.benchmark:
        result = backend.benchmark(program, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
    else:
        result = backend.run(program)
        for buf_name, values in result.outputs.items():
            logger.info("Output %s: %d elements, first 5: %s", buf_name, len(values), values[:5])
