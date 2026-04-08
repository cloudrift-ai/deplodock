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
    MATRIX_SIZES,
    MATRIX_SIZES_EXTENDED,
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "matmul_overnight"


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
    # Full nvcc banner — version, build date, target. Useful for repro when
    # cuBLAS minor versions drift between driver updates.
    nvcc_banner = " ".join(line.strip() for line in nvcc.splitlines() if line.strip())
    if nvcc_banner:
        info["nvcc"] = nvcc_banner
    # Compile flags actually used by the runner. Hardcoded mirror of
    # runner.run_benchmark — keep in sync if that function changes.
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


def _config_for_size(suite, m: int) -> dict | None:
    """Pick the strategy_map entry whose threshold covers M (adaptive runs only)."""
    cfg = suite.config or {}
    smap = cfg.get("strategy_map")
    if not smap:
        return cfg if isinstance(cfg, dict) and "strategy" in cfg else None
    for threshold, entry in smap:
        if m <= threshold:
            return entry
    return smap[-1][1]


def _fmt(v: float | None, spec: str) -> str:
    return format(v, spec) if v else "—"


def _suite_section(suite, batch: int) -> list[str]:
    lines: list[str] = []
    title = f"Batch = {batch}" if batch > 1 else "Single GEMM (batch=1)"
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"_{suite.description}_")
    lines.append("")
    lines.append("| Size | TM | BK | K-splits | Kernel ms | Kernel var % | cuBLAS ms | cuBLAS var % | Eff vs cuBLAS | TFLOPS | cuBLAS TFLOPS | Clock MHz | Temp °C |")
    lines.append("|------|----|----|----------|----------:|-------------:|----------:|-------------:|--------------:|-------:|--------------:|----------:|--------:|")
    for r in suite.results:
        dims = r.dimensions or {}
        m, n = dims.get("M", 0), dims.get("N", 0)
        size = f"{m}x{n}" + (f"x{batch}" if batch > 1 else "")
        cfg = _config_for_size(suite, m) or {}
        tm = cfg.get("thread_m", "—")
        bk = cfg.get("block_k", "—")
        ks = cfg.get("k_splits", "—")
        eff = f"{r.efficiency_pct:.1f}%" if r.efficiency_pct else "—"
        kvar = (
            f"{(r.kernel_max_ms - r.kernel_min_ms) / r.kernel_time_ms * 100:.1f}%"
            if r.kernel_time_ms and r.kernel_min_ms is not None and r.kernel_max_ms is not None
            else "—"
        )
        cvar = (
            f"{(r.cublas_max_ms - r.cublas_min_ms) / r.cublas_time_ms * 100:.1f}%"
            if r.cublas_time_ms and r.cublas_min_ms is not None and r.cublas_max_ms is not None
            else "—"
        )
        def _range(pre, post):
            if pre is None and post is None:
                return "—"
            if pre is None:
                return str(post)
            if post is None:
                return str(pre)
            return f"{pre}" if pre == post else f"{pre}→{post}"

        clk = _range(r.sm_clock_mhz_pre, r.sm_clock_mhz_post)
        tmp = _range(r.gpu_temp_c_pre, r.gpu_temp_c_post)
        lines.append(
            f"| {size} | {tm} | {bk} | {ks} | "
            f"{r.kernel_time_ms:.3f} | {kvar} | "
            f"{_fmt(r.cublas_time_ms, '.3f')} | {cvar} | {eff} | "
            f"{_fmt(r.gflops and r.gflops / 1000, '.1f')} | {_fmt(r.cublas_gflops and r.cublas_gflops / 1000, '.1f')} | "
            f"{clk} | {tmp} |"
        )
    lines.append("")
    if suite.error:
        lines.append(f"**ERROR:** {suite.error}")
        lines.append("")
    return lines


def render_markdown_report(runs: list[tuple[int, "BenchmarkSuite"]], sysinfo: dict[str, str], args) -> str:
    lines: list[str] = []
    first = runs[0][1]
    lines.append("# Matmul Benchmark Report")
    lines.append("")
    lines.append(f"_Generated: {first.timestamp}_")
    lines.append("")

    lines.append("## System")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k in ("host", "os", "gpu", "compute_cap", "vram", "driver", "cuda", "nvcc", "nvcc_flags", "cublas", "git"):
        if k in sysinfo:
            lines.append(f"| {k} | {sysinfo[k]} |")
    lines.append("")

    lines.append("## Run")
    lines.append("")
    lines.append(f"- Strategy: `{first.strategy}`")
    lines.append(f"- Iterations per measurement: {args.iterations}")
    lines.append(f"- cuBLAS math mode: `{args.cublas_math}`")
    lines.append(f"- Batches swept: {', '.join(str(b) for b, _ in runs)}")
    if first.strategy != "adaptive":
        lines.append(f"- Config: BK={args.bk}, block={args.block_m}x{args.block_n}, thread_m={args.thread_m}, k_splits={args.k_splits}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    for batch, suite in runs:
        lines.extend(_suite_section(suite, batch))

    last_kernel = next((s.cuda_kernel for _, s in reversed(runs) if s.cuda_kernel), "")
    if last_kernel:
        lines.append("## Kernel")
        lines.append("")
        lines.append("Last CUDA source compiled during this run (representative; per-size kernels differ in `thread_m` / `k_splits` per the strategy map above):")
        lines.append("")
        lines.append("```cuda")
        lines.append(last_kernel.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


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
    parser.add_argument("--output-dir", type=str, default=None, help="Override directory for saved trace JSON (implies --save)")
    parser.add_argument("--assume-aligned", action="store_true", help="Skip bounds checks (for pow2 sizes)")
    parser.add_argument("--k-splits", type=int, default=1, help="K-dimension splitting (for TMA)")
    parser.add_argument("--batch", type=int, default=1, help="Batch count for batched GEMM")
    parser.add_argument("--batches", type=str, default=None, help="Comma-separated batch counts to sweep (e.g. '1,4'). Overrides --batch.")
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

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.save:
        output_dir = RESULTS_DIR
    else:
        output_dir = None

    if args.batches:
        batch_counts = [int(b) for b in args.batches.split(",") if b.strip()]
    else:
        batch_counts = [args.batch]

    runs: list[tuple[int, "BenchmarkSuite"]] = []
    any_error = False

    for batch in batch_counts:
        logger.info("=== Running benchmark with batch=%d ===", batch)
        if args.strategy == "adaptive":
            strategy_map, profile_name = default_matmul_strategy_map()
            logger.info("Using matmul tuning profile: %s", profile_name)
            if batch > 1:
                # batch>1 already saturates grid.z; k_splits would collide on blockIdx.z.
                strategy_map = [
                    (t, dataclasses.replace(c, batch_count=batch, k_splits=1)) for t, c in strategy_map
                ]
            suite = run_adaptive_benchmark_suite(
                graph,
                rewriter,
                strategy_map,
                sizes=sizes,
                output_dir=output_dir,
                description=args.description or f"Adaptive benchmark (batch={batch})",
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
            )

        print()
        print(suite.summary_table())
        runs.append((batch, suite))
        if suite.error:
            any_error = True

    if output_dir is not None:
        sysinfo = collect_system_info()
        report = render_markdown_report(runs, sysinfo, args)
        ts = runs[0][1].timestamp
        report_path = Path(output_dir) / f"report_{ts}_{runs[0][1].strategy}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        logger.info("Markdown report written to %s", report_path)

    if any_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
