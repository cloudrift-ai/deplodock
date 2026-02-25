"""Unit tests for the planner module."""

from pathlib import Path

from deplodock.planner import BenchmarkTask
from deplodock.planner.group_by_model_and_gpu import GroupByModelAndGpuPlanner
from deplodock.recipe import ModelConfig, Recipe


def _make_task(model="org/model-a", gpu="NVIDIA GeForce RTX 5090", gpu_count=1, recipe_dir="/r", variant="rtx5090"):
    """Helper to build a BenchmarkTask with minimal config."""
    return BenchmarkTask(
        recipe_dir=recipe_dir,
        variant=variant,
        recipe=Recipe(model=ModelConfig(huggingface=model)),
        gpu_name=gpu,
        gpu_count=gpu_count,
    )


# ── BenchmarkTask ─────────────────────────────────────────────────


def test_task_model_name():
    task = _make_task(model="org/my-model")
    assert task.model_name == "org/my-model"


def test_task_recipe_name():
    task = _make_task(recipe_dir="/recipes/Qwen3-Coder-30B")
    assert task.recipe_name == "Qwen3-Coder-30B"


def test_task_recipe_name_trailing_slash():
    task = _make_task(recipe_dir="/recipes/Qwen3-Coder-30B/")
    # os.path.basename strips trailing slash correctly
    assert task.recipe_name == "" or task.recipe_name == "Qwen3-Coder-30B"


def test_task_result_path():
    task_obj = BenchmarkTask(
        recipe_dir="/recipes/MyModel",
        variant="rtx5090",
        recipe=Recipe(model=ModelConfig(huggingface="org/my-model")),
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_count=1,
        run_dir=Path("/run/123"),
    )
    result = task_obj.result_path()
    assert result == Path("/run/123/rtx5090_vllm_benchmark.txt")


def test_task_result_path_with_matrix_label():
    task_obj = BenchmarkTask(
        recipe_dir="/recipes/MyModel",
        variant="rtx5090_c128",
        recipe=Recipe(model=ModelConfig(huggingface="org/my-model")),
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_count=1,
        run_dir=Path("/run/123"),
    )
    result = task_obj.result_path()
    assert result == Path("/run/123/rtx5090_c128_vllm_benchmark.txt")


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


# ── GPU concurrency splitting ────────────────────────────────────


def test_gpu_concurrency_default_no_splitting():
    """gpu_concurrency=1 (default) produces the same result as before."""
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=1)
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 1
    assert len(groups[0].tasks) == 3


def test_gpu_concurrency_splits_into_subgroups():
    """gpu_concurrency=2 with 4 tasks produces 2 groups of 2."""
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=2)
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=8),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 2
    assert len(groups[0].tasks) == 2
    assert len(groups[1].tasks) == 2


def test_gpu_concurrency_capped_at_task_count():
    """gpu_concurrency > len(tasks) is capped at len(tasks) groups."""
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=10)
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 3
    for g in groups:
        assert len(g.tasks) == 1


def test_gpu_concurrency_single_task():
    """Single task always produces 1 group regardless of concurrency."""
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=4)
    tasks = [_make_task(model="org/m", gpu="GPU_A", gpu_count=2)]
    groups = planner.plan(tasks)
    assert len(groups) == 1
    assert len(groups[0].tasks) == 1
    assert groups[0].gpu_count == 2


def test_gpu_concurrency_subgroup_gpu_count():
    """Each sub-group has gpu_count = max of its respective tasks."""
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=2)
    # After sorting desc: [8, 4, 2, 1]
    # Round-robin: group0=[8, 2], group1=[4, 1]
    tasks = [
        _make_task(model="org/m", gpu="GPU_A", gpu_count=1),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=2),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=4),
        _make_task(model="org/m", gpu="GPU_A", gpu_count=8),
    ]
    groups = planner.plan(tasks)
    assert len(groups) == 2
    assert groups[0].gpu_count == 8
    assert groups[1].gpu_count == 4
