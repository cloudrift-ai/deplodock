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
        coarsen_cols = 1
        coarsen_rows = 1
        if config.strategy == "coarsened_f4":
            coarsen_cols = 4
        elif config.strategy == "coarsened_2r4c":
            coarsen_cols = 4
            coarsen_rows = 2

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
