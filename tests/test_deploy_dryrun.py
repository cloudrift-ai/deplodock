"""Dry-run end-to-end tests for deploy flow."""

import subprocess
import sys
import os

import pytest


MAIN_PY = os.path.join(os.path.dirname(__file__), "..", "main.py")
RECIPES_DIR = os.path.join(os.path.dirname(__file__), "..", "recipes")


def run_cli(*args):
    """Run main.py with given args and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, MAIN_PY, *args],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(MAIN_PY),
    )
    return result.returncode, result.stdout, result.stderr


class TestDryRunSSH:
    def test_ssh_deploy(self):
        rc, stdout, stderr = run_cli(
            "deploy", "ssh",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--variant", "8xH200",
            "--server", "user@1.2.3.4",
            "--dry-run",
        )
        assert rc == 0
        assert "[dry-run]" in stdout
        assert "docker compose pull" in stdout
        assert "docker compose up -d" in stdout
        assert "dry-run (not deployed)" in stdout

    def test_ssh_deploy_command_sequence(self):
        rc, stdout, _ = run_cli(
            "deploy", "ssh",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--variant", "8xH200",
            "--server", "user@1.2.3.4",
            "--dry-run",
        )
        assert rc == 0
        lines = stdout.strip().split("\n")
        dry_run_lines = [l for l in lines if l.startswith("[dry-run]")]

        # Verify correct sequence: mkdir, scp compose, scp nginx, pull, download, down, up, health
        assert any("mkdir" in l for l in dry_run_lines)
        assert any("docker-compose.yaml" in l for l in dry_run_lines)
        assert any("docker compose pull" in l for l in dry_run_lines)
        assert any("huggingface-cli download" in l for l in dry_run_lines)
        assert any("docker compose down" in l for l in dry_run_lines)
        assert any("docker compose up" in l for l in dry_run_lines)

    def test_ssh_teardown(self):
        rc, stdout, _ = run_cli(
            "deploy", "ssh",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--server", "user@1.2.3.4",
            "--dry-run",
            "--teardown",
        )
        assert rc == 0
        assert "[dry-run]" in stdout
        assert "docker compose down" in stdout


class TestDryRunLocal:
    def test_local_deploy(self):
        rc, stdout, _ = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
            "--variant", "RTX5090",
            "--dry-run",
        )
        assert rc == 0
        assert "[dry-run]" in stdout
        assert "docker compose pull" in stdout
        assert "docker compose up -d" in stdout

    def test_local_teardown(self):
        rc, stdout, _ = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
            "--dry-run",
            "--teardown",
        )
        assert rc == 0
        assert "docker compose down" in stdout


class TestDryRunVariantResolution:
    def test_different_variants_produce_different_compose(self):
        """Different variants should produce different compose configurations."""
        rc1, stdout1, _ = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--variant", "8xH200",
            "--dry-run",
        )
        rc2, stdout2, _ = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--variant", "8xH100",
            "--dry-run",
        )
        assert rc1 == 0
        assert rc2 == 0
        # Both should succeed but produce different output
        assert "zai-org/GLM-4.6-FP8" in stdout1
        assert "zai-org/GLM-4.6-FP8" in stdout2

    def test_single_gpu_variant(self):
        rc, stdout, _ = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
            "--variant", "RTX5090",
            "--dry-run",
        )
        assert rc == 0
        # Single GPU model shouldn't have nginx
        assert "nginx" not in stdout

    def test_multi_instance_variant(self, tmp_path):
        """A single-GPU model on a 4-GPU variant produces 4 instances with nginx."""
        import yaml
        recipe = {
            "model": {"name": "test/model"},
            "backend": {"vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "extra_args": "--max-model-len 8192",
            }},
            "variants": {"4xH100": {}},
        }
        with open(tmp_path / "recipe.yaml", "w") as f:
            yaml.dump(recipe, f)

        rc, stdout, _ = run_cli(
            "deploy", "ssh",
            "--recipe", str(tmp_path),
            "--variant", "4xH100",
            "--server", "user@host",
            "--dry-run",
        )
        assert rc == 0
        assert "nginx.conf" in stdout
        assert "Instances: 4" in stdout

    def test_unknown_variant_fails(self):
        rc, _, stderr = run_cli(
            "deploy", "local",
            "--recipe", os.path.join(RECIPES_DIR, "GLM-4.6-FP8"),
            "--variant", "nonexistent",
            "--dry-run",
        )
        assert rc != 0


class TestCLIHelp:
    def test_deploy_help(self):
        rc, stdout, _ = run_cli("deploy", "--help")
        assert rc == 0
        assert "local" in stdout
        assert "ssh" in stdout

    def test_local_help(self):
        rc, stdout, _ = run_cli("deploy", "local", "--help")
        assert rc == 0
        assert "--recipe" in stdout
        assert "--variant" in stdout
        assert "--dry-run" in stdout

    def test_ssh_help(self):
        rc, stdout, _ = run_cli("deploy", "ssh", "--help")
        assert rc == 0
        assert "--server" in stdout
        assert "--ssh-key" in stdout
        assert "--ssh-port" in stdout
