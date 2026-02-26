"""Tests for structured JSON benchmark results: parsers and composition."""

from pathlib import Path

from deplodock.benchmark.results import (
    BenchmarkMetrics,
    SystemInfo,
    compose_json_result,
    parse_benchmark_metrics,
    parse_system_info,
)
from deplodock.planner import BenchmarkTask
from deplodock.recipe.types import Recipe

# ── Sample benchmark output (from real vLLM bench serve) ──────────

BENCHMARK_OUTPUT_FULL = """\
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  365.50
Total input tokens:                      320000
Total generated tokens:                  320000
Request throughput (req/s):              0.22
Output token throughput (tok/s):         875.52
Peak output token throughput (tok/s):    1096.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          1751.04
---------------Time to First Token----------------
Mean TTFT (ms):                          771.60
Median TTFT (ms):                        845.10
P99 TTFT (ms):                           1492.58
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.95
Median TPOT (ms):                        8.88
P99 TPOT (ms):                           9.67
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.95
Median ITL (ms):                         8.75
P99 ITL (ms):                            10.60
==================================================
"""

BENCHMARK_OUTPUT_WITH_E2EL = BENCHMARK_OUTPUT_FULL.replace(
    "==================================================\n",
    """\
--------------End to End Latency------------------
Mean E2EL (ms):                          4570.00
Median E2EL (ms):                        4600.12
P99 E2EL (ms):                           5200.50
==================================================
""",
)

SYSTEM_INFO_RAW = """\
=== HOSTNAME ===
riftvm

=== OS ===
PRETTY_NAME="Ubuntu 24.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"

=== KERNEL ===
6.8.0-51-generic

=== CPU INFORMATION ===
Architecture:                         x86_64
Model name:                           AMD EPYC 7702 64-Core Processor
CPU(s):                               7

=== CPU COUNT ===
7

=== MEMORY ===
               total        used        free      shared  buff/cache   available
Mem:            49Gi       884Mi        47Gi       3.0Mi       791Mi        48Gi
Swap:             0B          0B          0B

=== GPU INFORMATION ===
NVIDIA GeForce RTX 5090, 32607 MiB, 580.65.06, P0, 42, 2 %

=== GPU DETAILS ===
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------------------------------------------------------+

=== DOCKER VERSION ===
Docker version 28.5.1, build e180ab8
"""


# ── parse_benchmark_metrics ───────────────────────────────────────


def test_parse_benchmark_metrics():
    m = parse_benchmark_metrics(BENCHMARK_OUTPUT_FULL)
    assert isinstance(m, BenchmarkMetrics)
    assert m.successful_requests == 80
    assert m.failed_requests == 0
    assert m.max_request_concurrency == 8
    assert m.benchmark_duration_s == 365.50
    assert m.total_input_tokens == 320000
    assert m.total_generated_tokens == 320000
    assert m.request_throughput == 0.22
    assert m.output_token_throughput == 875.52
    assert m.peak_output_token_throughput == 1096.00
    assert m.peak_concurrent_requests == 16.00
    assert m.total_token_throughput == 1751.04
    assert m.mean_ttft_ms == 771.60
    assert m.median_ttft_ms == 845.10
    assert m.p99_ttft_ms == 1492.58
    assert m.mean_tpot_ms == 8.95
    assert m.median_tpot_ms == 8.88
    assert m.p99_tpot_ms == 9.67
    assert m.mean_itl_ms == 8.95
    assert m.median_itl_ms == 8.75
    assert m.p99_itl_ms == 10.60
    # No E2EL in this output
    assert m.mean_e2el_ms is None
    assert m.median_e2el_ms is None
    assert m.p99_e2el_ms is None


def test_parse_benchmark_metrics_with_e2el():
    m = parse_benchmark_metrics(BENCHMARK_OUTPUT_WITH_E2EL)
    assert m.mean_e2el_ms == 4570.00
    assert m.median_e2el_ms == 4600.12
    assert m.p99_e2el_ms == 5200.50
    # Other fields still parsed
    assert m.successful_requests == 80


def test_parse_benchmark_metrics_empty():
    m = parse_benchmark_metrics("garbage text with no metrics")
    assert isinstance(m, BenchmarkMetrics)
    # All fields should be None
    for field_name in BenchmarkMetrics.__dataclass_fields__:
        assert getattr(m, field_name) is None


# ── parse_system_info ─────────────────────────────────────────────


def test_parse_system_info():
    s = parse_system_info(SYSTEM_INFO_RAW)
    assert isinstance(s, SystemInfo)
    assert s.hostname == "riftvm"
    assert s.os == "Ubuntu 24.04.1 LTS"
    assert s.kernel == "6.8.0-51-generic"
    assert s.cpu_model == "AMD EPYC 7702 64-Core Processor"
    assert s.cpu_arch == "x86_64"
    assert s.cpu_count == 7
    assert s.memory_total_gib == 49.0
    assert s.gpu_name == "NVIDIA GeForce RTX 5090"
    assert s.gpu_memory_mib == 32607
    assert s.gpu_driver == "580.65.06"
    assert s.cuda_version == "13.0"
    assert s.gpu_count == 1
    assert s.docker_version == "28.5.1"


def test_parse_system_info_empty():
    s = parse_system_info("")
    assert isinstance(s, SystemInfo)
    for field_name in SystemInfo.__dataclass_fields__:
        assert getattr(s, field_name) is None


# ── compose_json_result ───────────────────────────────────────────


def _make_task(tmp_path: Path) -> BenchmarkTask:
    """Build a minimal BenchmarkTask for testing."""
    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {
                "llm": {
                    "context_length": 8192,
                    "vllm": {"image": "vllm/vllm-openai:latest"},
                }
            },
            "benchmark": {"max_concurrency": 8, "num_prompts": 80},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )
    return BenchmarkTask(
        recipe_dir="experiments/TestModel/test_experiment",
        variant="rtx5090_c8",
        recipe=recipe,
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_count=1,
        run_dir=tmp_path,
    )


def test_compose_json_result(tmp_path):
    task = _make_task(tmp_path)
    result = compose_json_result(
        task,
        benchmark_output=BENCHMARK_OUTPUT_FULL,
        compose_content="services:\n  vllm_0:\n    image: vllm/vllm-openai:latest",
        bench_command="vllm bench serve --model test-org/test-model",
        system_info_raw=SYSTEM_INFO_RAW,
    )

    assert set(result.keys()) == {"task", "recipe", "metrics", "system", "compose", "bench_command"}

    # task section
    assert result["task"]["recipe_dir"] == "experiments/TestModel/test_experiment"
    assert result["task"]["variant"] == "rtx5090_c8"
    assert result["task"]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result["task"]["gpu_short"] == "rtx5090"
    assert result["task"]["gpu_count"] == 1

    # recipe section — dict from asdict(Recipe)
    assert result["recipe"]["model"]["huggingface"] == "test-org/test-model"

    # metrics section
    assert result["metrics"]["successful_requests"] == 80
    assert result["metrics"]["output_token_throughput"] == 875.52

    # system section
    assert result["system"]["hostname"] == "riftvm"
    assert result["system"]["gpu_name"] == "NVIDIA GeForce RTX 5090"

    # compose and bench_command are strings
    assert isinstance(result["compose"], str)
    assert isinstance(result["bench_command"], str)


# ── json_result_path ──────────────────────────────────────────────


def test_json_result_path(tmp_path):
    task = _make_task(tmp_path)
    assert task.json_result_path().suffix == ".json"
    assert task.json_result_path().stem == task.result_path().stem
    assert task.json_result_path().parent == task.result_path().parent
