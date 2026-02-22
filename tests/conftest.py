"""Shared fixtures for deploy tests."""

import os
import pytest
import yaml


@pytest.fixture
def tmp_recipe_dir(tmp_path):
    """Create a temp directory with a sample recipe.yaml."""
    recipe = {
        "model": {"name": "test-org/test-model"},
        "backend": {
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "extra_args": "--max-model-len 8192",
            }
        },
        "variants": {
            "RTX5090": {},
            "8xH200": {
                "backend": {
                    "vllm": {
                        "tensor_parallel_size": 8,
                        "extra_args": "--max-model-len 16384 --kv-cache-dtype fp8",
                    }
                }
            },
            "4xH100": {
                "backend": {
                    "vllm": {
                        "tensor_parallel_size": 4,
                        "extra_args": "--max-model-len 8192 --kv-cache-dtype fp8",
                    }
                }
            },
        },
    }

    recipe_path = tmp_path / "recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    return str(tmp_path)


@pytest.fixture
def sample_config():
    """Return a resolved config dict for testing compose generation."""
    return {
        "model": {"name": "test-org/test-model"},
        "backend": {
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "extra_args": "--max-model-len 8192",
            }
        },
    }


@pytest.fixture
def sample_config_multi():
    """Return a resolved config dict for multi-instance testing."""
    return {
        "model": {"name": "test-org/test-model"},
        "backend": {
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "extra_args": "--max-model-len 16384",
            }
        },
        "_num_instances": 2,
    }
