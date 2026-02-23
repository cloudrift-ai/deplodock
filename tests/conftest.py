"""Shared pytest fixtures for all test modules."""

import os
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")


@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def recipes_dir():
    """Absolute path to the recipes/ directory."""
    return RECIPES_DIR


@pytest.fixture(scope="session")
def run_cli(project_root):
    """Return a callable that invokes the deplodock CLI as a subprocess."""

    def _run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "deplodock.deplodock", *args],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return result.returncode, result.stdout, result.stderr

    return _run


@pytest.fixture
def make_bench_config(recipes_dir):
    """Return a factory that writes a temporary bench config.yaml."""

    def _make(tmp_dir, servers=None):
        if servers is None:
            servers = [
                {
                    "name": "test_server",
                    "ssh_key": "~/.ssh/id_ed25519",
                    "recipes": [
                        {
                            "recipe": os.path.join(
                                recipes_dir,
                                "Qwen3-Coder-30B-A3B-Instruct-AWQ",
                            ),
                            "variant": "RTX5090",
                        }
                    ],
                }
            ]

        config = {
            "benchmark": {
                "local_results_dir": os.path.join(str(tmp_dir), "results"),
                "model_dir": "/hf_models",
            },
            "servers": servers,
        }
        config_path = os.path.join(str(tmp_dir), "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    return _make


# ── Unit-test fixtures ──────────────────────────────────────────────


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
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "variants": {
            "RTX5090": {
                "gpu": "NVIDIA GeForce RTX 5090",
                "gpu_count": 1,
            },
            "8xH200": {
                "gpu": "NVIDIA H200 141GB",
                "gpu_count": 8,
                "backend": {
                    "vllm": {
                        "tensor_parallel_size": 8,
                        "extra_args": "--max-model-len 16384 --kv-cache-dtype fp8",
                    }
                },
                "benchmark": {
                    "random_input_len": 8000,
                    "random_output_len": 8000,
                },
            },
            "4xH100": {
                "gpu": "NVIDIA H100 80GB",
                "gpu_count": 4,
                "backend": {
                    "vllm": {
                        "tensor_parallel_size": 4,
                        "extra_args": "--max-model-len 8192 --kv-cache-dtype fp8",
                    }
                },
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
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
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
