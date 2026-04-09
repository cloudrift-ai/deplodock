"""Unit tests for recipe loading and deep merge."""

import pytest
import yaml

from deplodock.recipe import Recipe, deep_merge, load_recipe, resolve_for_hardware, validate_docker_options, validate_extra_args

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


# ── resolve_for_hardware ──────────────────────────────────────────


def test_resolve_for_hardware_name_only(tmp_recipe_dir):
    """Without gpu_count, resolves first name match (H200 entry)."""
    recipe = resolve_for_hardware(tmp_recipe_dir, "NVIDIA H200 141GB")
    assert recipe.deploy.gpu == "NVIDIA H200 141GB"
    assert recipe.deploy.gpu_count == 8
    assert recipe.engine.llm.tensor_parallel_size == 8
    assert recipe.engine.llm.context_length == 16384
    assert recipe.engine.llm.extra_args == "--kv-cache-dtype fp8"


def test_resolve_for_hardware_exact_count_match(tmp_recipe_dir):
    """Exact name + count match selects the right entry."""
    recipe = resolve_for_hardware(tmp_recipe_dir, "NVIDIA H100 80GB", 4)
    assert recipe.deploy.gpu == "NVIDIA H100 80GB"
    assert recipe.deploy.gpu_count == 4
    assert recipe.engine.llm.tensor_parallel_size == 4


def test_resolve_for_hardware_divisible_count(tmp_recipe_dir):
    """gpu_count=8 with entry count=4 (8 % 4 == 0) selects divisible match."""
    # tmp_recipe_dir has H100 entry with gpu_count=4
    recipe = resolve_for_hardware(tmp_recipe_dir, "NVIDIA H100 80GB", 8)
    assert recipe.deploy.gpu == "NVIDIA H100 80GB"
    assert recipe.deploy.gpu_count == 4
    assert recipe.engine.llm.tensor_parallel_size == 4


def test_resolve_for_hardware_divisible_picks_largest(tmp_path):
    """When multiple entries divide evenly, picks the largest entry count."""
    recipe_data = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 1,
            },
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 2,
                "engine.llm.tensor_parallel_size": 2,
            },
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 4,
                "engine.llm.tensor_parallel_size": 4,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe_data, f)

    # gpu_count=8 divides by 1, 2, and 4 — should pick 4 (largest)
    recipe = resolve_for_hardware(str(tmp_path), "NVIDIA H100 80GB", 8)
    assert recipe.deploy.gpu_count == 4
    assert recipe.engine.llm.tensor_parallel_size == 4


def test_resolve_for_hardware_count_no_match(tmp_recipe_dir):
    """Raises ValueError when gpu_count doesn't match any entry."""
    # RTX 5090 entry has gpu_count=1, and 3 % 1 == 0 so it would match.
    # H100 has gpu_count=4, and 3 % 4 != 0.
    with pytest.raises(ValueError, match="Available counts"):
        resolve_for_hardware(tmp_recipe_dir, "NVIDIA H100 80GB", 3)


def test_resolve_for_hardware_no_match(tmp_recipe_dir):
    """Raises ValueError for unknown GPU."""
    with pytest.raises(ValueError, match="No matrix entry matches GPU"):
        resolve_for_hardware(tmp_recipe_dir, "NVIDIA A100 40GB")


def test_resolve_for_hardware_no_matrices(tmp_path):
    """Returns base recipe when no matrices section exists."""
    recipe_data = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 2,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe_data, f)

    recipe = resolve_for_hardware(str(tmp_path), "Any GPU")
    assert recipe.engine.llm.tensor_parallel_size == 2
    assert recipe.deploy.gpu is None


def test_resolve_for_hardware_skips_sweeps(tmp_path):
    """Skips matrix entries that contain list values (sweeps)."""
    recipe_data = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
            }
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "benchmark.max_concurrency": [1, 2, 4],
            },
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe_data, f)

    recipe = resolve_for_hardware(str(tmp_path), "NVIDIA GeForce RTX 5090")
    assert recipe.deploy.gpu == "NVIDIA GeForce RTX 5090"
    assert recipe.deploy.gpu_count == 1


# ── validate_extra_args ────────────────────────────────────────────


def test_load_recipe_rejects_banned_extra_args(tmp_path):
    """load_recipe() raises when extra_args contains banned flags."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                    "extra_args": "--max-model-len 8192",
                },
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    with pytest.raises(ValueError, match="--max-model-len"):
        load_recipe(str(tmp_path))


# ── command recipes ────────────────────────────────────────────────


def test_load_recipe_command_only(tmp_path):
    recipe = {
        "command": {
            "run": "nvidia-smi > $task_dir/result.csv",
            "result_files": ["result.csv"],
        },
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)
    r = load_recipe(str(tmp_path))
    assert r.kind == "command"
    assert r.command.run == "nvidia-smi > $task_dir/result.csv"


def test_load_recipe_command_and_engine_mutually_exclusive(tmp_path):
    recipe = {
        "command": {"run": "echo hi"},
        "engine": {"llm": {"vllm": {}}},
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)
    with pytest.raises(ValueError, match="exactly one"):
        load_recipe(str(tmp_path))


def test_load_recipe_command_skips_extra_args_validation(tmp_path):
    recipe = {
        "command": {"run": "echo hi"},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090"},
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)
    r = load_recipe(str(tmp_path))
    assert r.kind == "command"


# ── deploy.driver_version / deploy.cuda_version ─────────────────────


def test_load_recipe_parses_driver_and_cuda_version(tmp_path):
    recipe = {
        "model": {"huggingface": "x/y"},
        "engine": {"llm": {"vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
        "deploy": {
            "gpu": "NVIDIA H100 80GB",
            "driver_version": "550",
            "cuda_version": "12.4",
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)
    r = load_recipe(str(tmp_path))
    assert r.deploy.driver_version == "550"
    assert r.deploy.cuda_version == "12.4"


def test_matrix_expands_driver_version(tmp_path):
    recipe = {
        "model": {"huggingface": "x/y"},
        "engine": {"llm": {"vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
        "matrices": [
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 1,
                "deploy.driver_version": "560",
                "deploy.cuda_version": "12.6",
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)
    r = resolve_for_hardware(str(tmp_path), "NVIDIA H100 80GB", 1)
    assert r.deploy.driver_version == "560"
    assert r.deploy.cuda_version == "12.6"


# ── validate_docker_options ───────────────────────────────────────


def test_validate_docker_options_accepts_valid_keys():
    validate_docker_options({"security_opt": ["seccomp=unconfined"], "cap_add": ["SYS_PTRACE"]})


def test_validate_docker_options_accepts_empty():
    validate_docker_options({})


def test_validate_docker_options_rejects_managed_keys():
    with pytest.raises(ValueError, match="image"):
        validate_docker_options({"image": "custom:latest"})


def test_validate_docker_options_rejects_multiple_managed_keys():
    with pytest.raises(ValueError, match="volumes"):
        validate_docker_options({"volumes": ["/a:/b"], "ports": ["8080:8080"]})


def test_load_recipe_rejects_conflicting_docker_options(tmp_path):
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
                "docker_options": {"image": "override:latest"},
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    with pytest.raises(ValueError, match="image"):
        load_recipe(str(tmp_path))
