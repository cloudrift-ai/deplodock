"""Dry-run end-to-end tests for the deploy cloud command."""

import os

# ── deploy cloud dry-run ─────────────────────────────────────────


def test_deploy_cloud_dry_run(run_cli, recipes_dir):
    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--variant",
        "RTX5090",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_deploy_cloud_dry_run_deploy_steps(run_cli, recipes_dir):
    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--variant",
        "RTX5090",
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # VM provisioning step
    assert "Creating CloudRift instance" in stdout
    # Deploy steps
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout
    assert "dry-run (not deployed)" in stdout


def test_deploy_cloud_no_variant_fails(run_cli, recipes_dir):
    """Without a variant, no gpu field -> error."""
    rc, stdout, stderr = run_cli(
        "deploy",
        "cloud",
        "--recipe",
        os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        "--dry-run",
    )
    assert rc != 0
    assert "gpu" in stderr.lower() or "gpu" in stdout.lower()


# ── CLI help ─────────────────────────────────────────────────────


def test_deploy_cloud_help(run_cli):
    rc, stdout, _ = run_cli("deploy", "cloud", "--help")
    assert rc == 0
    assert "--recipe" in stdout
    assert "--variant" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--name" in stdout


def test_deploy_help_includes_cloud(run_cli):
    rc, stdout, _ = run_cli("deploy", "--help")
    assert rc == 0
    assert "cloud" in stdout
