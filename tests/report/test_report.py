"""Tests for manifest-based report generation."""

import json

from deplodock.commands.report import _collect_tasks_from_manifests, parse_benchmark_result


SAMPLE_BENCHMARK = """\
============ Serving Benchmark Result ============
Request throughput (req/s):                     12.50
Output token throughput (tok/s):                3000.00
Total token throughput (tok/s):                 5000.00
Mean TTFT (ms):                                 150.00
Mean TPOT (ms):                                 25.00
"""


def _write_run(results_dir, run_name, tasks, benchmark_content=SAMPLE_BENCHMARK):
    """Helper to create a run directory with manifest and result files."""
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True)

    manifest = {
        "timestamp": "2026-02-23T14:30:00",
        "code_hash": "abc123",
        "recipes": list({t["recipe"] for t in tasks}),
        "tasks": tasks,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    for task in tasks:
        result_path = run_dir / task["result_file"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(benchmark_content)

    return run_dir


def test_collect_tasks_from_manifests(tmp_path):
    tasks = [
        {
            "recipe": "MyModel",
            "variant": "RTX5090",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_short": "rtx5090",
            "gpu_count": 1,
            "model_name": "org/MyModel",
            "result_file": "MyModel/RTX5090_vllm_benchmark.txt",
            "status": "completed",
        },
    ]
    _write_run(tmp_path, "2026-02-23_14-30-00_abc12345", tasks)

    collected = list(_collect_tasks_from_manifests(tmp_path))
    assert len(collected) == 1
    meta, path = collected[0]
    assert meta["gpu_short"] == "rtx5090"
    assert path.exists()


def test_collect_skips_failed_tasks(tmp_path):
    tasks = [
        {
            "recipe": "MyModel",
            "variant": "RTX5090",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_short": "rtx5090",
            "gpu_count": 1,
            "model_name": "org/MyModel",
            "result_file": "MyModel/RTX5090_vllm_benchmark.txt",
            "status": "failed",
        },
    ]
    _write_run(tmp_path, "2026-02-23_14-30-00_abc12345", tasks)

    collected = list(_collect_tasks_from_manifests(tmp_path))
    assert len(collected) == 0


def test_collect_multiple_runs(tmp_path):
    tasks1 = [
        {
            "recipe": "A", "variant": "V1",
            "gpu_name": "G", "gpu_short": "g", "gpu_count": 1,
            "model_name": "m/A", "result_file": "A/V1_vllm_benchmark.txt",
            "status": "completed",
        },
    ]
    tasks2 = [
        {
            "recipe": "B", "variant": "V2",
            "gpu_name": "G", "gpu_short": "g", "gpu_count": 2,
            "model_name": "m/B", "result_file": "B/V2_vllm_benchmark.txt",
            "status": "completed",
        },
    ]
    _write_run(tmp_path, "run1", tasks1)
    _write_run(tmp_path, "run2", tasks2)

    collected = list(_collect_tasks_from_manifests(tmp_path))
    assert len(collected) == 2


def test_parse_benchmark_result(tmp_path):
    result_file = tmp_path / "bench.txt"
    result_file.write_text(SAMPLE_BENCHMARK)

    throughput, metrics = parse_benchmark_result(result_file)
    assert throughput == 5000.0
    assert metrics["request_throughput"] == 12.5
    assert metrics["output_throughput"] == 3000.0
    assert metrics["mean_ttft_ms"] == 150.0
    assert metrics["mean_tpot_ms"] == 25.0
