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
import json
import logging
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deplodock.compiler.backend.cuda.codegen import emit_kernel
from deplodock.compiler.backend.cuda.generators import lower_matmul
from deplodock.compiler.backend.cuda.runner import MatmulBenchmarkResult, run_benchmark
from deplodock.compiler.backend.cuda.tuning import default_matmul_strategy_map
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix sizes
# ---------------------------------------------------------------------------

MATRIX_SIZES = [
    {"M": 256, "N": 256, "K": 256},
    {"M": 512, "N": 512, "K": 512},
    {"M": 1024, "N": 1024, "K": 1024},
    {"M": 2048, "N": 2048, "K": 2048},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 8192, "N": 8192, "K": 8192},
    {"M": 16384, "N": 16384, "K": 16384},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Type alias for matmul hints (keys without cuda.matmul. prefix).
MatmulHints = dict[str, Any]


def _set_matmul_hints(graph: Graph, hints: MatmulHints) -> None:
    """Set cuda.matmul.* hints on a graph from a flat dict."""
    for k, v in hints.items():
        graph.hints.set(f"cuda.matmul.{k}", v)


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSuite:
    """Results from a full benchmark run across matrix sizes."""

    timestamp: str
    strategy: str
    config: dict
    results: list[MatmulBenchmarkResult] = field(default_factory=list)
    generated_code: str = ""
    error: str | None = None
    description: str = ""
    system_info: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "config": self.config,
            "description": self.description,
            "generated_code": self.generated_code,
            "results": [asdict(r) for r in self.results],
            "system_info": self.system_info,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        """Format results as a readable table."""
        lines = [
            f"Strategy: {self.strategy} | {self.description}",
            f"{'Size':>12} | {'Kernel ms':>10} | {'cuBLAS ms':>10} | {'GFLOPS':>8} | {'Eff %':>7}",
            "-" * 65,
        ]
        for r in self.results:
            dims = r.dimensions or {}
            size = f"{dims.get('M', '?')}x{dims.get('N', '?')}"
            cublas = f"{r.cublas_time_ms:.3f}" if r.cublas_time_ms else "N/A"
            eff = f"{r.efficiency_pct:.1f}" if r.efficiency_pct else "N/A"
            lines.append(f"{size:>12} | {r.kernel_time_ms:>10.3f} | {cublas:>10} | {r.gflops:>8.1f} | {eff:>7}")
        return "\n".join(lines)


def run_benchmark_suite(
    graph: Graph,
    hints: MatmulHints,
    sizes: list[dict[str, int]] | None = None,
    output_dir: Path | None = None,
    description: str = "",
    num_iterations: int = 10,
    cublas_math_mode: str = "default",
    system_info: dict[str, str] | None = None,
) -> BenchmarkSuite:
    """Benchmark a matmul config across multiple matrix sizes."""
    sizes = sizes or MATRIX_SIZES
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    suite = BenchmarkSuite(
        timestamp=timestamp,
        strategy=hints.get("strategy", "naive"),
        config=dict(hints),
        description=description,
        system_info=system_info or {},
    )

    try:
        g = graph.copy()
        _set_matmul_hints(g, hints)
        kernel = lower_matmul(g)
        source = emit_kernel(kernel)
        suite.generated_code = source

        logger.info("Generated kernel for strategy=%s", hints.get("strategy", "naive"))

        for dim_args in sizes:
            m, n = dim_args.get("M", 0), dim_args.get("N", 0)
            k_splits = hints.get("k_splits", 1)
            batch_count = hints.get("batch_count", 1)
            if k_splits > 1:
                dim_args = {**dim_args, "k_splits": k_splits}
            if batch_count > 1:
                dim_args = {**dim_args, "batch": batch_count}
            logger.info("Benchmarking %dx%d ...", m, n)
            try:
                result = run_benchmark(
                    kernel=kernel,
                    kernel_source=source,
                    dim_args=dim_args,
                    num_iterations=num_iterations,
                    coarsen_cols=hints.get("coarsen_cols", 1),
                    coarsen_rows=hints.get("coarsen_rows", 1),
                    cublas_math_mode=cublas_math_mode,
                )
                suite.results.append(result)
                logger.info(
                    "  %dx%d: %.3f ms (cuBLAS: %.3f ms, eff: %.1f%%)",
                    m,
                    n,
                    result.kernel_time_ms,
                    result.cublas_time_ms or 0.0,
                    result.efficiency_pct or 0.0,
                )
            except Exception as e:
                logger.error("  %dx%d failed: %s", m, n, e)
                suite.error = f"Benchmark at {m}x{n} failed: {e}"
                break

    except Exception as e:
        logger.error("Suite failed: %s", e)
        suite.error = str(e)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp}_{hints.get('strategy', 'naive')}.json"
        trace_path = output_dir / filename
        trace_path.write_text(suite.to_json())
        logger.info("Trace saved to %s", trace_path)

    return suite


def run_adaptive_benchmark_suite(
    graph: Graph,
    strategy_map: list[tuple[int, MatmulHints]],
    sizes: list[dict[str, int]] | None = None,
    output_dir: Path | None = None,
    description: str = "",
    num_iterations: int = 10,
    cublas_math_mode: str = "default",
    system_info: dict[str, str] | None = None,
) -> BenchmarkSuite:
    """Benchmark with size-adaptive strategy selection."""
    sizes = sizes or MATRIX_SIZES
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    suite = BenchmarkSuite(
        timestamp=timestamp,
        strategy="adaptive",
        config={"strategy_map": [(t, dict(h)) for t, h in strategy_map]},
        description=description,
        system_info=system_info or {},
    )

    try:
        kernels: dict[int, tuple] = {}
        for _threshold, hints in strategy_map:
            key = id(hints)
            g = graph.copy()
            _set_matmul_hints(g, hints)
            kernel = lower_matmul(g)
            source = emit_kernel(kernel)
            kernels[key] = (kernel, source, hints)

        for dim_args in sizes:
            m = dim_args.get("M", 0)
            n = dim_args.get("N", 0)
            size = max(m, n)

            max_tuned_size = strategy_map[-1][0]
            selected = strategy_map[-1][1]
            for threshold, hints in strategy_map:
                if size <= threshold:
                    selected = hints
                    break
            else:
                logger.warning(
                    "Size %d exceeds the largest tuned bucket (%d) — falling back to "
                    "the last entry (TM=%d, BK=%d, ks=%d). Numbers may be suboptimal.",
                    size,
                    max_tuned_size,
                    selected.get("thread_m", 1),
                    selected.get("block_k", 16),
                    selected.get("k_splits", 1),
                )

            kernel, source, hints = kernels[id(selected)]
            suite.generated_code = source

            run_dim_args = dim_args
            k_splits = hints.get("k_splits", 1)
            batch_count = hints.get("batch_count", 1)
            if k_splits > 1:
                run_dim_args = {**dim_args, "k_splits": k_splits}
            if batch_count > 1:
                run_dim_args = {**run_dim_args, "batch": batch_count}
            logger.info("Benchmarking %dx%d with %s ...", m, n, hints.get("strategy", "?"))
            try:
                result = run_benchmark(
                    kernel=kernel,
                    kernel_source=source,
                    dim_args=run_dim_args,
                    num_iterations=num_iterations,
                    coarsen_cols=hints.get("coarsen_cols", 1),
                    coarsen_rows=hints.get("coarsen_rows", 1),
                    cublas_math_mode=cublas_math_mode,
                )
                suite.results.append(result)
                logger.info(
                    "  %dx%d [%s]: %.3f ms (cuBLAS: %.3f ms, eff: %.1f%%)",
                    m,
                    n,
                    hints.get("strategy", "?"),
                    result.kernel_time_ms,
                    result.cublas_time_ms or 0.0,
                    result.efficiency_pct or 0.0,
                )
            except Exception as e:
                logger.error("  %dx%d failed: %s", m, n, e)
                suite.error = f"Benchmark at {m}x{n} failed: {e}"
                break

    except Exception as e:
        logger.error("Adaptive suite failed: %s", e)
        suite.error = str(e)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp}_adaptive.json"
        trace_path = output_dir / filename
        trace_path.write_text(suite.to_json())
        logger.info("Trace saved to %s", trace_path)

    return suite


# ---------------------------------------------------------------------------
# System info collection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def make_matmul_graph() -> Graph:
    """Create a primitive matmul graph: C = Reduce{sum}(Elementwise{mul}(A, B))."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
    b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("AB", ("M", "K", "N")))
    c = g.add_node(ReduceOp(fn="sum", axis=1), [ew], Tensor("C", ("M", "N")))
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument("--dump-dir", type=str, default=None, help="Directory to dump intermediate compilation artifacts")
    parser.add_argument("--description", type=str, default="", help="Description for trace")
    parser.add_argument(
        "--cublas-math",
        default="default",
        choices=["default", "pedantic"],
        help="cuBLAS math mode: default (tensor cores) or pedantic (pure FP32)",
    )
    args = parser.parse_args()

    graph = make_matmul_graph()
    sizes = [parse_size(args.size)]
    output_dir = Path(args.output_dir) if args.output_dir else None
    system_info = collect_system_info()

    batch = args.batch

    if args.strategy == "adaptive":
        strategy_map, profile_name = default_matmul_strategy_map()
        logger.info("Using matmul tuning profile: %s", profile_name)
        if batch > 1:
            strategy_map = [(t, {**h, "batch_count": batch, "k_splits": 1}) for t, h in strategy_map]
        suite = run_adaptive_benchmark_suite(
            graph,
            strategy_map,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or f"Adaptive benchmark (size={args.size}, batch={batch})",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
            system_info=system_info,
        )
    else:
        hints: MatmulHints = {
            "strategy": args.strategy,
            "block_k": args.bk,
            "threads_y": args.threads_y,
            "threads_x": args.threads_x,
            "thread_m": args.thread_m,
            "assume_aligned": args.assume_aligned,
            "k_splits": args.k_splits,
            "batch_count": batch,
        }
        suite = run_benchmark_suite(
            graph,
            hints,
            sizes=sizes,
            output_dir=output_dir,
            description=args.description or f"{args.strategy} BK={args.bk} batch={batch}",
            num_iterations=args.iterations,
            cublas_math_mode=args.cublas_math,
            system_info=system_info,
        )

    from deplodock.compiler.dump import CompilerDump

    dump = CompilerDump.resolve(args.dump_dir)
    if dump and suite.generated_code:
        dump.dump_source(suite.generated_code)

    print()
    print(suite.summary_table())

    if suite.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
