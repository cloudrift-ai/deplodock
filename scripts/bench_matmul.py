#!/usr/bin/env python3
"""Run a single matmul benchmark point.

Usage:
    python scripts/bench_matmul.py --size 4096 --batch 1 --strategy adaptive
    python scripts/bench_matmul.py --size 4096x2048x1024 --batch 4
    python scripts/bench_matmul.py --size 8192 --output-dir /tmp/results
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deplodock.compiler.benchmark import (
    run_adaptive_benchmark_suite,
    run_benchmark_suite,
)
from deplodock.compiler.cuda.lower import MatmulConfig
from deplodock.compiler.cuda.tuning import default_matmul_strategy_map
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import FusedReduceElementwiseOp, InputOp
from deplodock.compiler.rewriter import Rewriter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def collect_system_info() -> dict[str, str]:
    """Gather GPU/driver/CUDA/cuBLAS info for the report."""
    info: dict[str, str] = {"host": platform.node(), "os": platform.platform()}

    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
    if smi:
        first = smi.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 4:
            info["gpu"], info["driver"], info["vram"], info["compute_cap"] = parts

    nvcc = _run(["nvcc", "--version"])
    m = re.search(r"release (\S+),\s*V(\S+)", nvcc)
    if m:
        info["cuda"] = m.group(2)
    nvcc_banner = " ".join(line.strip() for line in nvcc.splitlines() if line.strip())
    if nvcc_banner:
        info["nvcc"] = nvcc_banner
    info["nvcc_flags"] = "-O3 --fmad=true -arch=<auto> -lcuda -lcublas -lcurand"

    import os

    env_git = os.environ.get("DEPLODOCK_GIT_REV")
    if env_git:
        info["git"] = env_git
    else:
        git_dir = Path(__file__).resolve().parent.parent
        sha = _run(["git", "-C", str(git_dir), "rev-parse", "--short", "HEAD"])
        branch = _run(["git", "-C", str(git_dir), "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = _run(["git", "-C", str(git_dir), "status", "--porcelain"])
        if sha:
            info["git"] = f"{branch}@{sha}{' (dirty)' if dirty else ''}"

    try:
        cublas = ctypes.CDLL("libcublas.so")
        major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        cublas.cublasGetProperty(0, ctypes.byref(major))
        cublas.cublasGetProperty(1, ctypes.byref(minor))
        cublas.cublasGetProperty(2, ctypes.byref(patch))
        info["cublas"] = f"{major.value}.{minor.value}.{patch.value}"
    except (OSError, AttributeError):
        info["cublas"] = "unknown"

    return info


def make_matmul_graph() -> Graph:
    """Create a simple matmul graph: C = sum(A * B, axis=k)."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
    b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
    c = g.add_node(FusedReduceElementwiseOp("sum", "mul", 1), [a, b], Tensor("C", ("M", "N")))
    g.inputs = [a, b]
    g.outputs = [c]
    return g


def parse_size(size_str: str) -> dict[str, int]:
    """Parse a single size like '4096' or '4096x2048x1024' into a matrix dimension dict."""
    if "x" in size_str:
        parts = size_str.split("x")
        m, n = int(parts[0]), int(parts[1])
        k = int(parts[2]) if len(parts) > 2 else m
        return {"M": m, "N": n, "K": k}
    n = int(size_str)
    return {"M": n, "N": n, "K": n}


def main():
    parser = argparse.ArgumentParser(description="Single-point matmul benchmark runner")
    parser.add_argument("--strategy", default="adaptive", help="Kernel strategy (use 'adaptive' for size-dependent tuning)")
    parser.add_argument("--size", type=str, required=True, help="Matrix size (e.g. '4096' or '4096x2048x1024')")
    parser.add_argument("--batch", type=int, default=1, help="Batch count for batched GEMM")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--bk", type=int, default=32, help="Block K dimension (non-adaptive only)")
    parser.add_argument("--threads-y", type=int, default=8, help="Threads per block, Y dim (non-adaptive only)")
    parser.add_argument("--threads-x", type=int, default=32, help="Threads per block, X dim (non-adaptive only)")
    parser.add_argument("--thread-m", type=int, default=1, help="Rows per thread (non-adaptive only)")
    parser.add_argument("--k-splits", type=int, default=1, help="K-dimension splitting (non-adaptive only)")
    parser.add_argument("--assume-aligned", action="store_true", help="Skip bounds checks")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for saved JSON trace")
    parser.add_argument("--description", type=str, default="", help="Description for trace")
    parser.add_argument(
        "--cublas-math",
        default="default",
        choices=["default", "pedantic"],
        help="cuBLAS math mode: default (tensor cores) or pedantic (pure FP32)",
    )
    args = parser.parse_args()

    graph = make_matmul_graph()
    rewriter = Rewriter()
    sizes = [parse_size(args.size)]
    output_dir = Path(args.output_dir) if args.output_dir else None
    system_info = collect_system_info()

    batch = args.batch

    if args.strategy == "adaptive":
        strategy_map, profile_name = default_matmul_strategy_map()
        logger.info("Using matmul tuning profile: %s", profile_name)
        if batch > 1:
            strategy_map = [(t, dataclasses.replace(c, batch_count=batch, k_splits=1)) for t, c in strategy_map]
        suite = run_adaptive_benchmark_suite(
            graph,
            rewriter,
            strategy_map,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or f"Adaptive benchmark (size={args.size}, batch={batch})",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
            system_info=system_info,
        )
    else:
        config = MatmulConfig(
            strategy=args.strategy,
            block_k=args.bk,
            threads_y=args.threads_y,
            threads_x=args.threads_x,
            thread_m=args.thread_m,
            assume_aligned=args.assume_aligned,
            k_splits=args.k_splits,
            batch_count=batch,
        )
        suite = run_benchmark_suite(
            graph,
            rewriter,
            config,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or f"{args.strategy} BK={args.bk} batch={batch}",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
            system_info=system_info,
        )

    print()
    print(suite.summary_table())

    if suite.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
