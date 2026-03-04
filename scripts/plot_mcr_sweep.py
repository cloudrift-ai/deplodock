#!/usr/bin/env python3
"""Plot benchmark metrics from an MCR (max concurrent requests) sweep experiment."""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deplodock.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)


def load_results(results_dir):
    """Load all benchmark JSON files from a results directory."""
    results = []
    for path in sorted(results_dir.glob("*_benchmark.json")):
        with open(path) as f:
            data = json.load(f)
        # Some files may contain a list (e.g. tasks.json parsed wrong); skip those
        if isinstance(data, list):
            continue
        mcr = data["recipe"]["benchmark"]["max_concurrency"]
        metrics = data["metrics"]
        results.append({"mcr": mcr, **metrics})
    results.sort(key=lambda r: r["mcr"])
    return results


def plot_mcr_sweep(results, engine, model, gpu, output_path):
    """Create a single plot with throughput, median TTFT, and median TPOT vs MCR."""
    mcr_values = [r["mcr"] for r in results]
    output_throughput = [r["output_token_throughput"] for r in results]
    median_ttft = [r["median_ttft_ms"] for r in results]
    median_tpot = [r["median_tpot_ms"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"{model} on {gpu} ({engine})\nOptimal Max Concurrent Requests Sweep",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    marker_opts = dict(markersize=6, linewidth=2)
    color_throughput = "#2563eb"
    color_ttft = "#dc2626"
    color_tpot = "#f59e0b"

    # Left y-axis: throughput
    ax1.set_xlabel("Max Concurrent Requests")
    ax1.set_ylabel("Output Token Throughput (tok/s)", color=color_throughput)
    ax1.plot(mcr_values, output_throughput, "o-", color=color_throughput, label="Throughput (tok/s)", **marker_opts)
    ax1.tick_params(axis="y", labelcolor=color_throughput)
    ax1.set_xticks(mcr_values)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # Annotate peak throughput
    peak_idx = output_throughput.index(max(output_throughput))
    ax1.annotate(
        f"{output_throughput[peak_idx]:,.0f}",
        (mcr_values[peak_idx], output_throughput[peak_idx]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=color_throughput,
    )

    # Right y-axis: latency (TTFT and TPOT)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency (ms)")
    ax2.plot(mcr_values, median_ttft, "s--", color=color_ttft, label="Median TTFT (ms)", **marker_opts)
    ax2.plot(mcr_values, median_tpot, "^:", color=color_tpot, label="Median TPOT (ms)", **marker_opts)
    ax2.tick_params(axis="y")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", output_path)


def main():
    setup_cli_logging()
    parser = argparse.ArgumentParser(description="Plot MCR sweep benchmark results.")
    parser.add_argument("results_dir", type=Path, help="Path to experiment results directory")
    parser.add_argument(
        "--engine",
        default=None,
        help="Engine name for plot title (default: auto-detect from results)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for plot title (default: auto-detect from results)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU name for plot title (default: auto-detect from results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <results_dir>/mcr_sweep.png)",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        logger.error("Error: %s is not a directory", args.results_dir)
        sys.exit(1)

    results = load_results(args.results_dir)
    if not results:
        logger.error("Error: no benchmark JSON files found in %s", args.results_dir)
        sys.exit(1)

    # Auto-detect model and GPU from first result file if not specified
    first_json = next(args.results_dir.glob("*_benchmark.json"))
    with open(first_json) as f:
        first_data = json.load(f)
    if isinstance(first_data, list):
        first_data = first_data[0]

    model = args.model or first_data["recipe"]["model"]["huggingface"].split("/")[-1]
    gpu = args.gpu or first_data["task"].get("gpu_name", "GPU")
    if args.engine:
        engine = args.engine
    elif first_data["recipe"]["engine"]["llm"].get("sglang"):
        engine = "SGLang"
    else:
        engine = "vLLM"
    output_path = args.output or args.results_dir / "mcr_sweep.png"

    plot_mcr_sweep(results, engine, model, gpu, output_path)


if __name__ == "__main__":
    main()
