"""Unit tests for recipe loading and deep merge."""

import os
import pytest
import yaml

from deplodock.commands.deploy import load_recipe, deep_merge


class TestDeepMerge:
    def test_scalars_override_wins(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_dicts(self):
        base = {"backend": {"vllm": {"tp": 1, "extra": "--foo"}}}
        override = {"backend": {"vllm": {"extra": "--bar"}}}
        result = deep_merge(base, override)
        assert result == {"backend": {"vllm": {"tp": 1, "extra": "--bar"}}}

    def test_override_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        assert base == {"a": {"b": 1}}


class TestLoadRecipe:
    def test_load_defaults(self, tmp_recipe_dir):
        config = load_recipe(tmp_recipe_dir)
        assert config["model"]["name"] == "test-org/test-model"
        assert config["backend"]["vllm"]["tensor_parallel_size"] == 1
        assert config["backend"]["vllm"]["extra_args"] == "--max-model-len 8192"
        assert "variants" not in config

    def test_load_empty_variant(self, tmp_recipe_dir):
        config = load_recipe(tmp_recipe_dir, variant="RTX5090")
        assert config["backend"]["vllm"]["tensor_parallel_size"] == 1
        assert config["backend"]["vllm"]["extra_args"] == "--max-model-len 8192"

    def test_load_variant_deep_merge(self, tmp_recipe_dir):
        config = load_recipe(tmp_recipe_dir, variant="8xH200")
        assert config["backend"]["vllm"]["tensor_parallel_size"] == 8
        assert config["backend"]["vllm"]["extra_args"] == "--max-model-len 16384 --kv-cache-dtype fp8"
        # Preserved from base
        assert config["backend"]["vllm"]["image"] == "vllm/vllm-openai:latest"
        assert config["backend"]["vllm"]["gpu_memory_utilization"] == 0.9

    def test_unknown_variant_raises(self, tmp_recipe_dir):
        with pytest.raises(ValueError, match="Unknown variant 'nonexistent'"):
            load_recipe(tmp_recipe_dir, variant="nonexistent")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_recipe(str(tmp_path / "does_not_exist"))

    def test_variants_stripped(self, tmp_recipe_dir):
        config = load_recipe(tmp_recipe_dir, variant="RTX5090")
        assert "variants" not in config

    def test_variants_stripped_no_variant(self, tmp_recipe_dir):
        config = load_recipe(tmp_recipe_dir)
        assert "variants" not in config
