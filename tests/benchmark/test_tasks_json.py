"""Tests for BenchmarkTask.write_tasks_json() / read_tasks_json() round-trip."""

from unittest.mock import MagicMock

from deplodock.planner import BenchmarkTask
from deplodock.planner.variant import Variant


def _make_task(variant, gpu_name, gpu_count, model_name, recipe_dir="/recipes/TestModel", run_dir=None):
    """Helper to build a BenchmarkTask with a minimal mock recipe."""
    recipe = MagicMock()
    recipe.model_name = model_name
    recipe.engine.llm.engine_name = "vllm"
    recipe.deploy.gpu = gpu_name
    recipe.deploy.gpu_count = gpu_count
    task = BenchmarkTask(
        recipe_dir=recipe_dir,
        variant=variant,
        recipe=recipe,
    )
    if run_dir is not None:
        task.run_dir = run_dir
    return task


def test_tasks_json_round_trip(tmp_path):
    variant = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "benchmark.max_concurrency": 8,
            "engine.llm.max_concurrent_requests": 8,
        }
    )
    task = _make_task(
        variant=variant,
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_count=1,
        model_name="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
        run_dir=tmp_path,
    )
    BenchmarkTask.write_tasks_json(tmp_path, [task])

    result = BenchmarkTask.read_tasks_json(tmp_path)
    assert len(result) == 1
    assert result[0]["variant"] == str(variant)
    assert result[0]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result[0]["task_id"] == f"TestModel/{variant}"


def test_tasks_json_multiple(tmp_path):
    v1 = Variant(params={"deploy.gpu": "G", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "G", "deploy.gpu_count": 2})
    t1 = _make_task(variant=v1, gpu_name="G", gpu_count=1, model_name="m", run_dir=tmp_path)
    t2 = _make_task(variant=v2, gpu_name="G", gpu_count=2, model_name="m", run_dir=tmp_path)
    BenchmarkTask.write_tasks_json(tmp_path, [t1, t2])

    result = BenchmarkTask.read_tasks_json(tmp_path)
    assert len(result) == 2
    assert result[0]["variant"] == str(v1)
    assert result[1]["gpu_count"] == 2
