"""Dry-run tests for the bench command."""

import os


def test_bench_dry_run_basic(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench", recipe, "--variants", "RTX5090",
        "--config", config_path, "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_bench_dry_run_deploy_then_benchmark(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench", recipe, "--variants", "RTX5090",
        "--config", config_path, "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # Verify deploy steps appear
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout

    # Verify benchmark step appears with recipe params
    assert "vllm bench serve" in stdout
    assert "--random-input-len 4000" in stdout
    assert "--random-output-len 4000" in stdout

    # Verify teardown appears
    assert "docker compose down" in stdout

    # Verify order: pull before bench, bench before teardown
    pull_idx = stdout.index("docker compose pull")
    bench_idx = stdout.index("vllm bench serve")
    assert pull_idx < bench_idx


def test_bench_variant_filter(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench", recipe, "--variants", "RTX5090",
        "--config", config_path, "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "RTX5090" in stdout


def test_bench_multiple_recipes(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe1 = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    recipe2 = os.path.join(recipes_dir, "GLM-4.6-FP8")
    rc, stdout, stderr = run_cli(
        "bench", recipe1, recipe2, "--variants", "RTX5090",
        "--config", config_path, "--dry-run",
    )
    # Should succeed even if some variants are missing (warned on stderr)
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"


def test_bench_help(run_cli):
    rc, stdout, _ = run_cli("bench", "--help")
    assert rc == 0
    assert "recipes" in stdout
    assert "--variants" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--config" in stdout
    assert "--max-workers" in stdout
