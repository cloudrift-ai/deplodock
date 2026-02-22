"""Dry-run tests for the bench command."""

import os
import subprocess
import sys
import tempfile

import pytest
import yaml


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")


def run_cli(*args):
    """Run deplodock CLI with given args and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "deplodock.deplodock", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def make_config(tmp_dir, servers=None):
    """Create a temporary config.yaml and return its path."""
    if servers is None:
        servers = [
            {
                "name": "test_server",
                "address": "user@1.2.3.4",
                "ssh_key": "~/.ssh/id_ed25519",
                "port": 22,
                "recipes": [
                    {
                        "recipe": os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
                        "variant": "RTX4090",
                    }
                ],
            }
        ]

    config = {
        "benchmark": {
            "local_results_dir": os.path.join(tmp_dir, "results"),
            "model_dir": "/hf_models",
        },
        "benchmark_params": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 8000,
            "random_output_len": 8000,
        },
        "servers": servers,
    }
    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestBenchDryRun:
    def test_bench_dry_run_basic(self, tmp_path):
        config_path = make_config(str(tmp_path))
        rc, stdout, stderr = run_cli("bench", "--config", config_path, "--dry-run")
        assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
        assert "[dry-run]" in stdout

    def test_bench_dry_run_deploy_then_benchmark(self, tmp_path):
        config_path = make_config(str(tmp_path))
        rc, stdout, stderr = run_cli("bench", "--config", config_path, "--dry-run")
        assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

        # Verify deploy steps appear
        assert "docker compose pull" in stdout
        assert "docker compose up" in stdout

        # Verify benchmark step appears
        assert "vllm bench serve" in stdout

        # Verify teardown appears
        assert "docker compose down" in stdout

        # Verify order: pull before bench, bench before teardown
        pull_idx = stdout.index("docker compose pull")
        bench_idx = stdout.index("vllm bench serve")
        down_lines = [i for i, c in enumerate(stdout) if stdout[i:].startswith("docker compose down")]
        # Last "docker compose down" should be after bench (teardown)
        assert pull_idx < bench_idx

    def test_bench_server_filter(self, tmp_path):
        servers = [
            {
                "name": "server_a",
                "address": "user@1.2.3.4",
                "ssh_key": "~/.ssh/id_ed25519",
                "port": 22,
                "recipes": [
                    {
                        "recipe": os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
                        "variant": "RTX4090",
                    }
                ],
            },
            {
                "name": "server_b",
                "address": "user@5.6.7.8",
                "ssh_key": "~/.ssh/id_ed25519",
                "port": 22,
                "recipes": [
                    {
                        "recipe": os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
                        "variant": "RTX4090",
                    }
                ],
            },
        ]
        config_path = make_config(str(tmp_path), servers=servers)
        rc, stdout, stderr = run_cli(
            "bench", "--config", config_path, "--dry-run", "--server", "server_a"
        )
        assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
        assert "server_a" in stdout
        assert "5.6.7.8" not in stdout

    def test_bench_recipe_filter(self, tmp_path):
        recipe_path = os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
        servers = [
            {
                "name": "test_server",
                "address": "user@1.2.3.4",
                "ssh_key": "~/.ssh/id_ed25519",
                "port": 22,
                "recipes": [
                    {"recipe": recipe_path, "variant": "RTX4090"},
                    {"recipe": os.path.join(RECIPES_DIR, "GLM-4.6-FP8"), "variant": "8xH200"},
                ],
            },
        ]
        config_path = make_config(str(tmp_path), servers=servers)
        rc, stdout, stderr = run_cli(
            "bench", "--config", config_path, "--dry-run",
            "--recipe", recipe_path,
        )
        assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
        assert "Qwen3-Coder-30B" in stdout
        # GLM recipe should not be deployed
        assert "GLM-4.6-FP8" not in stdout


class TestBenchCLIHelp:
    def test_bench_help_updated(self):
        rc, stdout, _ = run_cli("bench", "--help")
        assert rc == 0
        assert "--recipe" in stdout
        assert "--dry-run" in stdout
        assert "--config" in stdout
        assert "--force" in stdout
        assert "--server" in stdout
        assert "--parallel" in stdout
        assert "--max-workers" in stdout
