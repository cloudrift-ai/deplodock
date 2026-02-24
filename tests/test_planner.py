"""Unit tests for the planner module."""

from deplodock.planner import BenchmarkTask, ExecutionGroup
from deplodock.planner.group_by_model_and_gpu import GroupByModelAndGpuPlanner


def _make_task(model="org/model-a", gpu="NVIDIA GeForce RTX 5090", gpu_count=1, recipe_dir="/r"):
    """Helper to build a BenchmarkTask with minimal config."""
    return BenchmarkTask(
        recipe_dir=recipe_dir,
        variant="V1",
        recipe_config={"model": {"name": model}},
        gpu_name=gpu,
        gpu_count=gpu_count,
    )


# ── BenchmarkTask ─────────────────────────────────────────────────


def test_task_model_name():
    task = _make_task(model="org/my-model")
    assert task.model_name == "org/my-model"


def test_task_result_filename():
    task = _make_task(model="org/my-model", gpu="NVIDIA GeForce RTX 5090", gpu_count=1)
    assert task.result_filename == "rtx5090_1x_org_my-model_vllm_benchmark.txt"


def test_task_result_filename_multi_gpu():
    task = _make_task(model="org/my-model", gpu="NVIDIA H200 141GB", gpu_count=8)
    assert task.result_filename == "h200_8x_org_my-model_vllm_benchmark.txt"


# ── GroupByModelAndGpuPlanner ─────────────────────────────────────


def test_same_model_same_gpu_one_group():
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 1
    assert len(groups[0].tasks) == 2


def test_different_models_same_gpu_separate_groups():
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/model-a", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/model-b", gpu="GPU_A", gpu_count=1),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 2


def test_same_model_different_gpu_separate_groups():
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_B", gpu_count=1),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 2


def test_max_gpu_count_per_group():
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 1
    assert groups[0].gpu_count == 4


def test_tasks_sorted_descending_by_gpu_count():
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
    ]
    groups = planner.plan(tasks)
    counts = [t.gpu_count for t in groups[0].tasks]
    assert counts == [4, 2, 1]


def test_cross_recipe_grouping():
    """Tasks from different recipe dirs but same model+GPU are grouped."""
    planner = GroupByModelAndGpuPlanner()
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1, recipe_dir="/recipe1"),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2, recipe_dir="/recipe2"),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 1
    assert len(groups[0].tasks) == 2
