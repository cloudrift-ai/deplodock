"""Unit tests for scale-out strategies."""

import pytest

from deplodock.deploy.scale_out import DataParallelismScaleOutStrategy, ReplicaParallelismScaleOutStrategy
from deplodock.recipe import Recipe

# ── helpers ────────────────────────────────────────────────────────


def _make_recipe(tp=1, pp=1, dp=1, gpu_count=1) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "test-org/test-model"},
            "engine": {
                "llm": {
                    "tensor_parallel_size": tp,
                    "pipeline_parallel_size": pp,
                    "data_parallel_size": dp,
                    "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
                }
            },
            "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": gpu_count},
        }
    )


# ── DataParallelismScaleOutStrategy ───────────────────────────────


class TestDataParallelismScaleOut:
    def test_basic(self):
        """1 GPU recipe on 8 GPUs -> dp=8."""
        recipe = _make_recipe(tp=1, pp=1, dp=1, gpu_count=1)
        result = DataParallelismScaleOutStrategy().apply(recipe, 8)
        assert result.engine.llm.data_parallel_size == 8
        assert result.deploy.gpu_count == 8

    def test_with_tp(self):
        """tp=2 recipe on 8 GPUs -> dp=4."""
        recipe = _make_recipe(tp=2, pp=1, dp=1, gpu_count=2)
        result = DataParallelismScaleOutStrategy().apply(recipe, 8)
        assert result.engine.llm.data_parallel_size == 4
        assert result.deploy.gpu_count == 8

    def test_no_change(self):
        """Already matching GPU count -> dp=1."""
        recipe = _make_recipe(tp=1, pp=1, dp=1, gpu_count=1)
        result = DataParallelismScaleOutStrategy().apply(recipe, 1)
        assert result.engine.llm.data_parallel_size == 1
        assert result.deploy.gpu_count == 1

    def test_insufficient_gpus(self):
        """Fewer GPUs than tp*pp -> ValueError."""
        recipe = _make_recipe(tp=4, pp=1, dp=1, gpu_count=4)
        with pytest.raises(ValueError, match="requires at least 4"):
            DataParallelismScaleOutStrategy().apply(recipe, 2)

    def test_immutable(self):
        """Input recipe is not mutated."""
        recipe = _make_recipe(tp=1, pp=1, dp=1, gpu_count=1)
        DataParallelismScaleOutStrategy().apply(recipe, 8)
        assert recipe.engine.llm.data_parallel_size == 1
        assert recipe.deploy.gpu_count == 1

    def test_with_tp_and_pp(self):
        """tp=2, pp=2 recipe on 8 GPUs -> dp=2."""
        recipe = _make_recipe(tp=2, pp=2, dp=1, gpu_count=4)
        result = DataParallelismScaleOutStrategy().apply(recipe, 8)
        assert result.engine.llm.data_parallel_size == 2
        assert result.deploy.gpu_count == 8


# ── ReplicaParallelismScaleOutStrategy ────────────────────────────


class TestReplicaParallelismScaleOut:
    def test_basic(self):
        """1 GPU recipe on 8 GPUs -> gpu_count=8, dp stays 1."""
        recipe = _make_recipe(tp=1, pp=1, dp=1, gpu_count=1)
        result = ReplicaParallelismScaleOutStrategy().apply(recipe, 8)
        assert result.deploy.gpu_count == 8
        assert result.engine.llm.data_parallel_size == 1

    def test_with_tp(self):
        """tp=2 on 8 GPUs -> gpu_count=8, dp stays 1 (4 replicas via calculate_num_instances)."""
        recipe = _make_recipe(tp=2, pp=1, dp=1, gpu_count=2)
        result = ReplicaParallelismScaleOutStrategy().apply(recipe, 8)
        assert result.deploy.gpu_count == 8
        assert result.engine.llm.data_parallel_size == 1

        from deplodock.deploy import calculate_num_instances

        assert calculate_num_instances(result) == 4

    def test_insufficient_gpus(self):
        """Fewer GPUs than gpus_per_instance -> ValueError."""
        recipe = _make_recipe(tp=4, pp=1, dp=1, gpu_count=4)
        with pytest.raises(ValueError, match="requires at least 4"):
            ReplicaParallelismScaleOutStrategy().apply(recipe, 2)

    def test_immutable(self):
        """Input recipe is not mutated."""
        recipe = _make_recipe(tp=1, pp=1, dp=1, gpu_count=1)
        ReplicaParallelismScaleOutStrategy().apply(recipe, 8)
        assert recipe.deploy.gpu_count == 1
