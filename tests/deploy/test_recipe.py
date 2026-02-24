"""Unit tests for recipe loading and deep merge."""

import pytest

from deplodock.deploy import deep_merge, load_recipe, validate_extra_args

# ── deep_merge ──────────────────────────────────────────────────────


def test_deep_merge_scalars_override_wins():
    base = {"a": 1, "b": 2}
    override = {"b": 3}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 3}


def test_deep_merge_nested_dicts():
    base = {"backend": {"vllm": {"tp": 1, "extra": "--foo"}}}
    override = {"backend": {"vllm": {"extra": "--bar"}}}
    result = deep_merge(base, override)
    assert result == {"backend": {"vllm": {"tp": 1, "extra": "--bar"}}}


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


def test_load_recipe_defaults(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir)
    assert config["model"]["name"] == "test-org/test-model"
    assert config["backend"]["vllm"]["tensor_parallel_size"] == 1
    assert config["backend"]["vllm"]["context_length"] == 8192
    assert "extra_args" not in config["backend"]["vllm"]
    assert "variants" not in config


def test_load_recipe_empty_variant(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="RTX5090")
    assert config["backend"]["vllm"]["tensor_parallel_size"] == 1
    assert config["backend"]["vllm"]["context_length"] == 8192


def test_load_recipe_variant_deep_merge(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="8xH200")
    assert config["backend"]["vllm"]["tensor_parallel_size"] == 8
    assert config["backend"]["vllm"]["context_length"] == 16384
    assert config["backend"]["vllm"]["extra_args"] == "--kv-cache-dtype fp8"
    # Preserved from base
    assert config["backend"]["vllm"]["image"] == "vllm/vllm-openai:latest"
    assert config["backend"]["vllm"]["gpu_memory_utilization"] == 0.9


def test_load_recipe_unknown_variant_raises(tmp_recipe_dir):
    with pytest.raises(ValueError, match="Unknown variant 'nonexistent'"):
        load_recipe(tmp_recipe_dir, variant="nonexistent")


def test_load_recipe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recipe(str(tmp_path / "does_not_exist"))


def test_load_recipe_variants_stripped(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="RTX5090")
    assert "variants" not in config


def test_load_recipe_variants_stripped_no_variant(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir)
    assert "variants" not in config


# ── benchmark section ──────────────────────────────────────────────


def test_load_recipe_benchmark_defaults(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir)
    assert config["benchmark"]["max_concurrency"] == 128
    assert config["benchmark"]["num_prompts"] == 256
    assert config["benchmark"]["random_input_len"] == 4000
    assert config["benchmark"]["random_output_len"] == 4000


def test_load_recipe_benchmark_preserved_with_variant(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="RTX5090")
    assert config["benchmark"]["max_concurrency"] == 128
    assert config["benchmark"]["random_input_len"] == 4000


def test_load_recipe_benchmark_variant_override(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="8xH200")
    assert config["benchmark"]["max_concurrency"] == 128  # from base
    assert config["benchmark"]["random_input_len"] == 8000  # overridden
    assert config["benchmark"]["random_output_len"] == 8000  # overridden
    assert config["benchmark"]["num_prompts"] == 256  # from base


def test_load_recipe_benchmark_no_override_variant(tmp_recipe_dir):
    config = load_recipe(tmp_recipe_dir, variant="4xH100")
    assert config["benchmark"]["random_input_len"] == 4000  # base preserved
    assert config["benchmark"]["random_output_len"] == 4000  # base preserved


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
    import yaml

    recipe = {
        "model": {"name": "test-org/test-model"},
        "backend": {
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 1,
                "extra_args": "--max-model-len 8192",
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    with pytest.raises(ValueError, match="--max-model-len"):
        load_recipe(str(tmp_path))
