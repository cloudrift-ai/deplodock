"""Tests for BenchmarkTask.write_tasks_json() / read_tasks_json() round-trip."""

from unittest.mock import MagicMock

from deplodock.planner import BenchmarkTask


def _make_task(variant, gpu_name, gpu_count, model_name, recipe_dir="/recipes/TestModel", run_dir=None):
    """Helper to build a BenchmarkTask with a minimal mock recipe."""
    recipe = MagicMock()
    recipe.model_name = model_name
    recipe.engine.llm.engine_name = "vllm"
    task = BenchmarkTask(
        recipe_dir=recipe_dir,
        variant=variant,
        recipe=recipe,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
    )
    if run_dir is not None:
        task.run_dir = run_dir
    return task


def test_tasks_json_round_trip(tmp_path):
    task = _make_task(
        variant="rtx5090_c8_mcr8",
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_count=1,
        model_name="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
        run_dir=tmp_path,
    )
    BenchmarkTask.write_tasks_json(tmp_path, [task])

    result = BenchmarkTask.read_tasks_json(tmp_path)
    assert len(result) == 1
    assert result[0]["variant"] == "rtx5090_c8_mcr8"
    assert result[0]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result[0]["task_id"] == "TestModel/rtx5090_c8_mcr8"


def test_tasks_json_multiple(tmp_path):
    t1 = _make_task(variant="V1", gpu_name="G", gpu_count=1, model_name="m", run_dir=tmp_path)
    t2 = _make_task(variant="V2", gpu_name="G", gpu_count=2, model_name="m", run_dir=tmp_path)
    BenchmarkTask.write_tasks_json(tmp_path, [t1, t2])

    result = BenchmarkTask.read_tasks_json(tmp_path)
    assert len(result) == 2
    assert result[0]["variant"] == "V1"
    assert result[1]["gpu_count"] == 2
