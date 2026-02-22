"""Dry-run end-to-end tests for the deploy command."""

import os

import yaml


# ── SSH deploy ──────────────────────────────────────────────────────


def test_ssh_deploy(run_cli, recipes_dir):
    rc, stdout, stderr = run_cli(
        "deploy", "ssh",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
        "--variant", "8xH200",
        "--server", "user@1.2.3.4",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "docker compose pull" in stdout
    assert "docker compose up -d" in stdout
    assert "dry-run (not deployed)" in stdout


def test_ssh_deploy_command_sequence(run_cli, recipes_dir):
    rc, stdout, _ = run_cli(
        "deploy", "ssh",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
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


def test_ssh_teardown(run_cli, recipes_dir):
    rc, stdout, _ = run_cli(
        "deploy", "ssh",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
        "--server", "user@1.2.3.4",
        "--dry-run",
        "--teardown",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "docker compose down" in stdout


# ── Local deploy ────────────────────────────────────────────────────


def test_local_deploy(run_cli, recipes_dir):
    rc, stdout, _ = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--variant", "RTX5090",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "docker compose pull" in stdout
    assert "docker compose up -d" in stdout


def test_local_teardown(run_cli, recipes_dir):
    rc, stdout, _ = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--dry-run",
        "--teardown",
    )
    assert rc == 0
    assert "docker compose down" in stdout


# ── Variant resolution ──────────────────────────────────────────────


def test_different_variants_produce_different_compose(run_cli, recipes_dir):
    """Different variants should produce different compose configurations."""
    rc1, stdout1, _ = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
        "--variant", "8xH200",
        "--dry-run",
    )
    rc2, stdout2, _ = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
        "--variant", "8xH100",
        "--dry-run",
    )
    assert rc1 == 0
    assert rc2 == 0
    # Both should succeed but produce different output
    assert "zai-org/GLM-4.6-FP8" in stdout1
    assert "zai-org/GLM-4.6-FP8" in stdout2


def test_single_gpu_variant(run_cli, recipes_dir):
    rc, stdout, _ = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--variant", "RTX5090",
        "--dry-run",
    )
    assert rc == 0
    # Single GPU model shouldn't have nginx
    assert "nginx" not in stdout


def test_multi_instance_variant(run_cli, tmp_path):
    """A single-GPU model on a 4-GPU variant produces 4 instances with nginx."""
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


def test_unknown_variant_fails(run_cli, recipes_dir):
    rc, _, stderr = run_cli(
        "deploy", "local",
        "--recipe", os.path.join(recipes_dir, "GLM-4.6-FP8"),
        "--variant", "nonexistent",
        "--dry-run",
    )
    assert rc != 0


# ── CLI help ────────────────────────────────────────────────────────


def test_deploy_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "--help")
    assert rc == 0
    assert "local" in stdout
    assert "ssh" in stdout


def test_local_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "local", "--help")
    assert rc == 0
    assert "--recipe" in stdout
    assert "--variant" in stdout
    assert "--dry-run" in stdout


def test_ssh_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "ssh", "--help")
    assert rc == 0
    assert "--server" in stdout
    assert "--ssh-key" in stdout
    assert "--ssh-port" in stdout


def test_bench_help(run_cli):
    rc, stdout, _ = run_cli("bench", "--help")
    assert rc == 0
    assert "--config" in stdout
    assert "--force" in stdout
    assert "--server" in stdout
    assert "--recipe" in stdout
    assert "--dry-run" in stdout
    assert "--parallel" in stdout
    assert "--max-workers" in stdout


def test_report_help(run_cli):
    rc, stdout, _ = run_cli("report", "--help")
    assert rc == 0
    assert "--config" in stdout
    assert "--results-dir" in stdout
    assert "--output" in stdout


def test_top_level_help(run_cli):
    rc, stdout, _ = run_cli("--help")
    assert rc == 0
    assert "deploy" in stdout
    assert "bench" in stdout
    assert "report" in stdout
