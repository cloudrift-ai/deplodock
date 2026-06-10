"""Tests for benchmark result-file composition (compose_result Timing section)."""

from pathlib import Path

from deplodock.benchmark.workload import compose_result
from deplodock.planner import BenchmarkTask
from deplodock.planner.variant import Variant
from deplodock.recipe.types import Recipe


def _make_task(tmp_path: Path) -> BenchmarkTask:
    recipe = Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {"llm": {"context_length": 8192, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
            "benchmark": {"max_concurrency": 8, "num_prompts": 80},
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
        }
    )
    variant = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1})
    return BenchmarkTask(
        recipe_dir="experiments/TestModel/test_experiment",
        variant=variant,
        recipe=recipe,
        run_dir=tmp_path,
    )


def test_compose_result_includes_timing_section(tmp_path):
    task = _make_task(tmp_path)
    timing = {"image_pull": 95.3, "model_load_and_warmup": 73.1, "total": 168.4}
    out = compose_result(
        task,
        benchmark_output="============ Serving Benchmark Result ============\n",
        compose_content="services: {}",
        bench_command="vllm bench serve",
        system_info="",
        timing=timing,
    )
    assert "============ Timing ============" in out
    assert "image_pull" in out
    assert "total" in out


def test_compose_result_omits_timing_section_when_absent(tmp_path):
    task = _make_task(tmp_path)
    out = compose_result(
        task,
        benchmark_output="result\n",
        compose_content="services: {}",
        bench_command="vllm bench serve",
        system_info="",
    )
    assert "Timing" not in out
