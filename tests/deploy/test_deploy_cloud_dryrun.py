"""Dry-run end-to-end tests for the deploy cloud command."""

import os

import yaml

# ── deploy cloud dry-run ─────────────────────────────────────────


def test_deploy_cloud_dry_run(run_cli, tmp_path):
    """Cloud deploy resolves matrix entry from --gpu and --gpu-count."""
    recipe = {
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
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_deploy_cloud_dry_run_deploy_steps(run_cli, tmp_path):
    recipe = {
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
                "deploy.gpu_count": 1,
            },
        ],
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        str(tmp_path),
        "--gpu",
        "NVIDIA GeForce RTX 5090",
        "--gpu-count",
        "1",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # VM provisioning step
    assert "Creating CloudRift instance" in stdout
    # Deploy steps
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout
    assert "dry-run (not deployed)" in stdout


def test_deploy_cloud_missing_gpu_flag_fails(run_cli, recipes_dir):
    """Without --gpu and --gpu-count, cloud deploy fails (required args)."""
    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--dry-run",
    )
    assert rc != 0
    assert "gpu" in stderr.lower()


# ── CLI help ─────────────────────────────────────────────────────


def test_deploy_cloud_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "cloud", "--help")
    assert rc == 0
    assert "--recipe" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--name" in stdout


def test_deploy_help_includes_cloud(run_cli):
    rc, stdout, _ = run_cli("deploy", "--help")
    assert rc == 0
    assert "cloud" in stdout
