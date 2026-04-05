"""Benchmark harness for matmul kernel optimization."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deplodock.compiler.cuda.codegen import emit_kernel
from deplodock.compiler.cuda.lower import MatmulConfig, lower_graph
from deplodock.compiler.cuda.runner import BenchmarkResult, run_benchmark
from deplodock.compiler.ir import Graph
from deplodock.compiler.rewriter import Rewriter

logger = logging.getLogger(__name__)

MATRIX_SIZES = [
    {"M": 256, "N": 256, "K": 256},
    {"M": 512, "N": 512, "K": 512},
    {"M": 1024, "N": 1024, "K": 1024},
    {"M": 2048, "N": 2048, "K": 2048},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 8192, "N": 8192, "K": 8192},
    {"M": 16384, "N": 16384, "K": 16384},
]

MATRIX_SIZES_EXTENDED = MATRIX_SIZES + [
    # Non-rectangular
    {"M": 1024, "N": 512, "K": 1024},
    {"M": 2048, "N": 768, "K": 2048},
    {"M": 512, "N": 2048, "K": 512},
    # Non-power-of-2
    {"M": 1000, "N": 1000, "K": 1000},
    {"M": 1500, "N": 1500, "K": 1500},
    {"M": 3000, "N": 3000, "K": 3000},
    # Odd sizes (stress boundary handling)
    {"M": 1001, "N": 1001, "K": 1001},
    {"M": 1023, "N": 1023, "K": 1023},
]


@dataclass
class BenchmarkSuite:
    """Results from a full benchmark run across matrix sizes."""

    timestamp: str
    strategy: str
    config: dict
    results: list[BenchmarkResult] = field(default_factory=list)
    cuda_kernel: str = ""
    error: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "config": self.config,
            "description": self.description,
            "cuda_kernel": self.cuda_kernel,
            "results": [asdict(r) for r in self.results],
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
    rewriter: Rewriter,
    config: MatmulConfig,
    sizes: list[dict[str, int]] | None = None,
    output_dir: Path | None = None,
    description: str = "",
    num_iterations: int = 10,
    cublas_math_mode: str = "default",
) -> BenchmarkSuite:
    """Benchmark a matmul config across multiple matrix sizes.

    Applies rewriter passes, lowers with config, benchmarks each size.
    Saves JSON trace if output_dir is provided.
    """
    sizes = sizes or MATRIX_SIZES
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    suite = BenchmarkSuite(
        timestamp=timestamp,
        strategy=config.strategy,
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else {},
        description=description,
    )

    try:
        # Apply rewriter passes.
        optimized = rewriter.apply(graph.copy())

        # Lower to CUDA IR.
        kernel = lower_graph(optimized, config=config)
        source = emit_kernel(kernel)
        suite.cuda_kernel = source

        logger.info("Generated kernel for strategy=%s", config.strategy)
        logger.info("Kernel source:\n%s", source)

        # Determine coarsening factor from config.
        coarsen_cols = config.coarsen_cols
        coarsen_rows = config.coarsen_rows

        # Benchmark each size.
        for dim_args in sizes:
            m, n = dim_args.get("M", 0), dim_args.get("N", 0)
            logger.info("Benchmarking %dx%d ...", m, n)
            try:
                result = run_benchmark(
                    kernel=kernel,
                    kernel_source=source,
                    dim_args=dim_args,
                    num_iterations=num_iterations,
                    coarsen_cols=coarsen_cols,
                    coarsen_rows=coarsen_rows,
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

    # Save trace.
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp}_{config.strategy}.json"
        trace_path = output_dir / filename
        trace_path.write_text(suite.to_json())
        logger.info("Trace saved to %s", trace_path)

    return suite


def run_adaptive_benchmark_suite(
    graph: Graph,
    rewriter: Rewriter,
    strategy_map: list[tuple[int, MatmulConfig]],
    sizes: list[dict[str, int]] | None = None,
    output_dir: Path | None = None,
    description: str = "",
    num_iterations: int = 10,
    cublas_math_mode: str = "default",
) -> BenchmarkSuite:
    """Benchmark with size-adaptive strategy selection.

    strategy_map: list of (threshold, config) sorted ascending by threshold.
    For each size, picks the first config whose threshold >= max(M, N).
    Lowers a separate kernel per config, benchmarks each size with its matching kernel.
    """
    sizes = sizes or MATRIX_SIZES
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    suite = BenchmarkSuite(
        timestamp=timestamp,
        strategy="adaptive",
        config={"strategy_map": [(t, asdict(c)) for t, c in strategy_map]},
        description=description,
    )

    try:
        optimized = rewriter.apply(graph.copy())

        # Pre-lower all unique configs.
        kernels: dict[int, tuple] = {}
        for _threshold, cfg in strategy_map:
            key = id(cfg)
            kernel = lower_graph(optimized, config=cfg)
            source = emit_kernel(kernel)
            kernels[key] = (kernel, source, cfg)

        for dim_args in sizes:
            m = dim_args.get("M", 0)
            n = dim_args.get("N", 0)
            size = max(m, n)

            # Select config for this size.
            selected = strategy_map[-1][1]
            for threshold, cfg in strategy_map:
                if size <= threshold:
                    selected = cfg
                    break

            kernel, source, cfg = kernels[id(selected)]
            suite.cuda_kernel = source  # Last kernel used.

            logger.info("Benchmarking %dx%d with %s ...", m, n, selected.strategy)
            try:
                result = run_benchmark(
                    kernel=kernel,
                    kernel_source=source,
                    dim_args=dim_args,
                    num_iterations=num_iterations,
                    coarsen_cols=cfg.coarsen_cols,
                    coarsen_rows=cfg.coarsen_rows,
                    cublas_math_mode=cublas_math_mode,
                )
                suite.results.append(result)
                logger.info(
                    "  %dx%d [%s]: %.3f ms (cuBLAS: %.3f ms, eff: %.1f%%)",
                    m,
                    n,
                    selected.strategy,
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
