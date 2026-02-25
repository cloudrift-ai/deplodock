"""Unit tests for recipe loading and deep merge."""

import pytest
import yaml

from deplodock.recipe import Recipe, deep_merge, load_recipe, validate_extra_args

# ── deep_merge ──────────────────────────────────────────────────────


def test_deep_merge_scalars_override_wins():
    base = {"a": 1, "b": 2}
    override = {"b": 3}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 3}


def test_deep_merge_nested_dicts():
    base = {"engine": {"llm": {"tp": 1, "extra": "--foo"}}}
    override = {"engine": {"llm": {"extra": "--bar"}}}
    result = deep_merge(base, override)
    assert result == {"engine": {"llm": {"tp": 1, "extra": "--bar"}}}


def test_deep_merge_adds_new_keys():
    base = {"a": 1}
    override = {"b": 2}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 2}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    deep_merge(base, override)
    assert base == {"a": {"b": 1}}


# ── load_recipe ─────────────────────────────────────────────────────


def test_load_recipe_returns_recipe(tmp_recipe_dir):
    recipe = load_recipe(tmp_recipe_dir)
    assert isinstance(recipe, Recipe)


def test_load_recipe_defaults(tmp_recipe_dir):
    recipe = load_recipe(tmp_recipe_dir)
    assert recipe.model.huggingface == "test-org/test-model"
    assert recipe.engine.llm.tensor_parallel_size == 1
    assert recipe.engine.llm.context_length == 8192
    assert recipe.engine.llm.extra_args == ""


def test_load_recipe_strips_matrices(tmp_recipe_dir):
    """load_recipe returns base config without matrix overrides."""
    recipe = load_recipe(tmp_recipe_dir)
    assert recipe.engine.llm.tensor_parallel_size == 1
    assert recipe.engine.llm.context_length == 8192


def test_load_recipe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recipe(str(tmp_path / "does_not_exist"))


def test_load_recipe_no_deploy_gpu(tmp_recipe_dir):
    """Base recipe has no deploy.gpu (it comes from matrices)."""
    recipe = load_recipe(tmp_recipe_dir)
    assert recipe.deploy.gpu is None


# ── benchmark section ──────────────────────────────────────────────


def test_load_recipe_benchmark_defaults(tmp_recipe_dir):
    recipe = load_recipe(tmp_recipe_dir)
    assert recipe.benchmark.max_concurrency == 128
    assert recipe.benchmark.num_prompts == 256
    assert recipe.benchmark.random_input_len == 4000
    assert recipe.benchmark.random_output_len == 4000


# ── validate_extra_args ────────────────────────────────────────────


def test_validate_extra_args_clean():
    validate_extra_args("--kv-cache-dtype fp8 --enable-expert-parallel")


def test_validate_extra_args_empty():
    validate_extra_args("")


def test_validate_extra_args_banned_flag_space_separated():
    with pytest.raises(ValueError, match="--max-model-len"):
        validate_extra_args("--kv-cache-dtype fp8 --max-model-len 8192")


def test_validate_extra_args_banned_flag_equals_style():
    with pytest.raises(ValueError, match="--gpu-memory-utilization"):
        validate_extra_args("--gpu-memory-utilization=0.95")


def test_validate_extra_args_multiple_banned():
    with pytest.raises(ValueError, match="--max-model-len.*--max-num-seqs|--max-num-seqs.*--max-model-len"):
        validate_extra_args("--max-model-len 8192 --max-num-seqs 256")


def test_load_recipe_rejects_banned_extra_args(tmp_path):
    """load_recipe() raises when extra_args contains banned flags."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {
                    "image": "vllm/vllm-openai:latest",
                    "extra_args": "--max-model-len 8192",
                },
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    with pytest.raises(ValueError, match="--max-model-len"):
        load_recipe(str(tmp_path))
