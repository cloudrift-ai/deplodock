"""Benchmark result file parsing."""

import re
from pathlib import Path


def parse_benchmark_result(result_file: Path) -> tuple[float, dict]:
    """Parse vLLM benchmark result file and extract total token throughput."""
    with open(result_file) as f:
        content = f.read()

    total_match = re.search(r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)", content)
    if not total_match:
        return None, {}

    total_throughput = float(total_match.group(1))

    metrics = {}

    req_match = re.search(r"Request throughput \(req/s\):\s+([\d.]+)", content)
    if req_match:
        metrics["request_throughput"] = float(req_match.group(1))

    output_match = re.search(r"Output token throughput \(tok/s\):\s+([\d.]+)", content)
    if output_match:
        metrics["output_throughput"] = float(output_match.group(1))

    ttft_match = re.search(r"Mean TTFT \(ms\):\s+([\d.]+)", content)
    if ttft_match:
        metrics["mean_ttft_ms"] = float(ttft_match.group(1))

    tpot_match = re.search(r"Mean TPOT \(ms\):\s+([\d.]+)", content)
    if tpot_match:
        metrics["mean_tpot_ms"] = float(tpot_match.group(1))

    return total_throughput, metrics
