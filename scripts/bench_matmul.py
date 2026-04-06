#!/usr/bin/env python3
"""Run matmul benchmarks with configurable strategies.

Usage:
    python scripts/bench_matmul.py --strategy hybrid_smem_f4 --bk 32
    python scripts/bench_matmul.py --strategy flat_scalar --block-n 128
    python scripts/bench_matmul.py --strategy adaptive
    python scripts/bench_matmul.py --sizes 256,512,1024  # specific sizes only
    python scripts/bench_matmul.py --extended  # include non-rectangular/odd sizes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deplodock.compiler.benchmark import (
    MATRIX_SIZES,
    MATRIX_SIZES_EXTENDED,
    run_adaptive_benchmark_suite,
    run_benchmark_suite,
)
from deplodock.compiler.cuda.lower import MatmulConfig
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import FusedReduceElementwiseOp, InputOp
from deplodock.compiler.rewriter import Rewriter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "matmul_overnight"


def make_matmul_graph() -> Graph:
    """Create a simple matmul graph: C = sum(A * B, axis=k)."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
    b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
    c = g.add_node(FusedReduceElementwiseOp("sum", "mul", 1), [a, b], Tensor("C", ("M", "N")))
    g.inputs = [a, b]
    g.outputs = [c]
    return g


def parse_sizes(sizes_str: str) -> list[dict[str, int]]:
    """Parse comma-separated sizes like '256,512,1024' into square matrix dicts."""
    sizes = []
    for s in sizes_str.split(","):
        s = s.strip()
        if "x" in s:
            parts = s.split("x")
            m, n = int(parts[0]), int(parts[1])
            k = int(parts[2]) if len(parts) > 2 else m
            sizes.append({"M": m, "N": n, "K": k})
        else:
            n = int(s)
            sizes.append({"M": n, "N": n, "K": n})
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Matmul benchmark runner")
    parser.add_argument("--strategy", default="hybrid_smem_f4", help="Kernel strategy")
    parser.add_argument("--bk", type=int, default=32, help="Block K dimension")
    parser.add_argument("--block-m", type=int, default=8, help="Block M (threads_y)")
    parser.add_argument("--block-n", type=int, default=32, help="Block N (threads_x)")
    parser.add_argument("--thread-m", type=int, default=1, help="Rows per thread (thread_m)")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes")
    parser.add_argument("--extended", action="store_true", help="Include non-standard sizes")
    parser.add_argument("--description", type=str, default="", help="Description for trace")
    parser.add_argument("--save", action="store_true", help="Save trace to results dir")
    parser.add_argument("--assume-aligned", action="store_true", help="Skip bounds checks (for pow2 sizes)")
    parser.add_argument("--k-splits", type=int, default=1, help="K-dimension splitting (for TMA)")
    parser.add_argument("--batch", type=int, default=1, help="Batch count for batched GEMM")
    parser.add_argument(
        "--cublas-math",
        default="default",
        choices=["default", "pedantic"],
        help="cuBLAS math mode: default (tensor cores) or pedantic (pure FP32)",
    )
    args = parser.parse_args()

    graph = make_matmul_graph()
    # Graph is already fused (FusedReduceElementwiseOp), so no rewrite passes needed.
    rewriter = Rewriter()

    # Determine sizes.
    if args.sizes:
        sizes = parse_sizes(args.sizes)
    elif args.extended:
        sizes = MATRIX_SIZES_EXTENDED
    else:
        sizes = MATRIX_SIZES

    output_dir = RESULTS_DIR if args.save else None

    if args.strategy == "adaptive":
        # Best config per size from empirical tuning on RTX 5090 sm_120.
        def _h(tm, bk, bm=4, bn=32):
            return MatmulConfig(
                strategy="hybrid_smem_f4",
                block_m=bm,
                block_n=bn,
                thread_m=tm,
                block_k=bk,
                coarsen_rows=tm,
                coarsen_cols=4,
                assume_aligned=args.assume_aligned,
            )

        # Best FP32-accurate config per size (CUDA 13.2, sm_120, RTX 5090):
        # TMA double-buffer with K-splitting and size-adaptive thread_m.
        def tma(bk=32, tm=8, ks=1):
            return MatmulConfig(strategy="tma_db", block_k=bk, thread_m=tm, k_splits=ks)

        strategy_map = [
            (256, tma(bk=32, tm=8, ks=4)),  # ~95% — K-splits for grid parallelism
            (512, tma(bk=32, tm=8, ks=4)),  # ~96%
            (1024, tma(bk=32, tm=8, ks=1)),  # 101% — beats cuBLAS
            (2048, tma(bk=32, tm=26, ks=1)),  # 105% — large tile hides latency
            (4096, tma(bk=32, tm=20, ks=1)),  # 99% — sweet spot tile size
            (8192, tma(bk=32, tm=28, ks=1)),  # 96%
            (99999, tma(bk=32, tm=28)),  # 90% at 16K
        ]
        suite = run_adaptive_benchmark_suite(
            graph,
            rewriter,
            strategy_map,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or "Adaptive benchmark",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
        )
    else:
        # Determine coarsening from strategy.
        coarsen_rows, coarsen_cols = 1, 1
        if args.strategy == "coarsened_f4":
            coarsen_cols = 4
        elif args.strategy in ("coarsened_2r4c", "hybrid_smem_f4"):
            thread_m = args.thread_m if args.thread_m > 1 else 2
            coarsen_rows, coarsen_cols = thread_m, 4

        config = MatmulConfig(
            strategy=args.strategy,
            block_k=args.bk,
            block_m=args.block_m,
            block_n=args.block_n,
            thread_m=args.thread_m,
            coarsen_rows=coarsen_rows,
            coarsen_cols=coarsen_cols,
            assume_aligned=args.assume_aligned,
            k_splits=args.k_splits,
            batch_count=args.batch,
        )
        suite = run_benchmark_suite(
            graph,
            rewriter,
            config,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or f"{args.strategy} BK={args.bk}",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
        )

    print()
    print(suite.summary_table())
    if suite.error:
        print(f"\nERROR: {suite.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
