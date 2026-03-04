"""Tests for scripts/plot_mcr_sweep.py — load_results() function."""

import json
import sys
from pathlib import Path

import pytest


# Add scripts/ to sys.path so we can import the module directly
@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    yield
    sys.path.remove(scripts_dir)


def _write_benchmark_json(directory, filename, mcr, output_throughput, mean_ttft, median_ttft, p99_ttft, mean_tpot, median_tpot, p99_tpot):
    """Write a minimal benchmark JSON file."""
    data = {
        "task": {"gpu_name": "NVIDIA GeForce RTX 5090", "gpu_short": "rtx5090", "gpu_count": 1},
        "recipe": {
            "model": {"huggingface": "org/test-model"},
            "benchmark": {"max_concurrency": mcr},
        },
        "metrics": {
            "output_token_throughput": output_throughput,
            "mean_ttft_ms": mean_ttft,
            "median_ttft_ms": median_ttft,
            "p99_ttft_ms": p99_ttft,
            "mean_tpot_ms": mean_tpot,
            "median_tpot_ms": median_tpot,
            "p99_tpot_ms": p99_tpot,
        },
    }
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


# --- load_results ---


def test_load_results(tmp_path):
    """load_results returns sorted results with correct metric values."""
    from plot_mcr_sweep import load_results

    # Write in non-sorted order to verify sorting
    _write_benchmark_json(
        tmp_path,
        "mc16_benchmark.json",
        mcr=16,
        output_throughput=1200,
        mean_ttft=100,
        median_ttft=90,
        p99_ttft=200,
        mean_tpot=10,
        median_tpot=9,
        p99_tpot=15,
    )
    _write_benchmark_json(
        tmp_path,
        "mc8_benchmark.json",
        mcr=8,
        output_throughput=900,
        mean_ttft=50,
        median_ttft=45,
        p99_ttft=100,
        mean_tpot=8,
        median_tpot=7,
        p99_tpot=12,
    )
    _write_benchmark_json(
        tmp_path,
        "mc32_benchmark.json",
        mcr=32,
        output_throughput=1400,
        mean_ttft=300,
        median_ttft=250,
        p99_ttft=600,
        mean_tpot=15,
        median_tpot=14,
        p99_tpot=25,
    )

    results = load_results(tmp_path)

    assert len(results) == 3
    # Sorted by MCR
    assert [r["mcr"] for r in results] == [8, 16, 32]
    # Check metric values from first entry (mcr=8)
    assert results[0]["output_token_throughput"] == 900
    assert results[0]["mean_ttft_ms"] == 50
    assert results[0]["mean_tpot_ms"] == 8


def test_load_results_empty_dir(tmp_path):
    """Empty directory returns empty list."""
    from plot_mcr_sweep import load_results

    results = load_results(tmp_path)
    assert results == []


def test_load_results_skips_non_benchmark(tmp_path):
    """Non-benchmark files (like tasks.json) are ignored."""
    from plot_mcr_sweep import load_results

    # Write a tasks.json (should be ignored — doesn't match *_benchmark.json)
    (tmp_path / "tasks.json").write_text(json.dumps([{"task": "foo"}]))
    # Write a real benchmark file
    _write_benchmark_json(
        tmp_path,
        "mc8_benchmark.json",
        mcr=8,
        output_throughput=900,
        mean_ttft=50,
        median_ttft=45,
        p99_ttft=100,
        mean_tpot=8,
        median_tpot=7,
        p99_tpot=12,
    )

    results = load_results(tmp_path)
    assert len(results) == 1
    assert results[0]["mcr"] == 8
