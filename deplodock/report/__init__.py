"""Report library: parse results, collect tasks, generate Excel reports."""

from deplodock.report.parser import parse_benchmark_result
from deplodock.report.pricing import get_gpu_price
from deplodock.report.collector import load_config, collect_tasks_from_manifests
from deplodock.report.generator import generate_report

__all__ = [
    "parse_benchmark_result",
    "get_gpu_price",
    "load_config",
    "collect_tasks_from_manifests",
    "generate_report",
]
