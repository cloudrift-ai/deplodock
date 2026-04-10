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

    # deplodock run block --hidden-dim 2048 --num-heads 32 --seq-len 32
    block_parser = sub.add_parser("block", help="Run a transformer block")
    block_parser.add_argument("--hidden-dim", type=int, required=True, help="Hidden dimension")
    block_parser.add_argument("--num-heads", type=int, required=True, help="Number of attention heads")
    block_parser.add_argument("--num-kv-heads", type=int, default=None, help="Number of KV heads (default: num-heads)")
    block_parser.add_argument("--head-dim", type=int, default=None, help="Head dimension (default: hidden-dim / num-heads)")
    block_parser.add_argument("--intermediate-dim", type=int, default=None, help="FFN intermediate dim (default: hidden-dim * 4)")
    block_parser.add_argument("--seq-len", type=int, default=32, help="Sequence length (default: 32)")
    block_parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    block_parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode")
    block_parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    block_parser.set_defaults(func=_handle_block)


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


def _handle_block(args):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.block_planner import BlockConfig, plan_block

    num_kv_heads = args.num_kv_heads or args.num_heads
    head_dim = args.head_dim or (args.hidden_dim // args.num_heads)
    intermediate_dim = args.intermediate_dim or (args.hidden_dim * 4)

    cfg = BlockConfig(
        batch=args.batch,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
    )

    plan = plan_block(cfg)
    backend = CudaBackend()
    program = backend.compile(plan)

    logger.info(
        "Block: hidden=%d heads=%d kv_heads=%d seq=%d batch=%d",
        cfg.hidden_dim,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.seq_len,
        cfg.batch,
    )

    if args.benchmark:
        result = backend.benchmark(program, num_iters=args.iters)
        logger.info("Time: %.3f ms (%d launches)", result.time_ms, result.num_launches)
    else:
        result = backend.run(program)
        output = result.outputs.get("output", [])
        logger.info("Output: %d elements, first 5: %s", len(output), output[:5])
