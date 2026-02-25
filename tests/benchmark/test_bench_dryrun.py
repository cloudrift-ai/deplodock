"""Dry-run tests for the bench command."""

import os
from pathlib import Path


def test_bench_dry_run_basic(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_bench_dry_run_deploy_then_benchmark(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # Verify deploy steps appear
    assert "docker compose pull" in stdout
    assert "docker compose up" in stdout

    # Verify benchmark step appears with recipe params
    assert "bench serve" in stdout
    assert "--random-input-len 4000" in stdout
    assert "--random-output-len 4000" in stdout

    # Verify teardown appears
    assert "docker compose down" in stdout

    # Verify order: pull before bench, bench before teardown
    pull_idx = stdout.index("docker compose pull")
    bench_idx = stdout.index("bench serve")
    assert pull_idx < bench_idx


def test_bench_multiple_recipes(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe1 = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    recipe2 = os.path.join(recipes_dir, "GLM-4.6-FP8")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe1,
        recipe2,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"


def test_bench_no_teardown_dry_run(run_cli, make_bench_config, recipes_dir, tmp_path):
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
        "--no-teardown",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # With --no-teardown, per-task teardown should be skipped
    assert "bench serve" in stdout
    assert "Tearing down..." not in stdout
    assert "Skipping VM deletion (--no-teardown)" in stdout


def test_bench_results_in_recipe_dir(run_cli, make_bench_config, recipes_dir, tmp_path):
    """Results are stored directly in the recipe directory."""
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # Run directory should be inside {recipe_dir}/
    expected_parent = str(Path(recipe).resolve())
    assert expected_parent in stdout


def test_bench_experiment_dry_run(run_cli, make_bench_config, project_root, tmp_path):
    """Experiment recipe runs successfully in dry-run mode."""
    config_path = make_bench_config(tmp_path)
    experiment = os.path.join(
        project_root,
        "experiments",
        "Qwen3-Coder-30B-A3B-Instruct-AWQ",
        "optimal_mcr_rtx5090",
    )
    rc, stdout, stderr = run_cli(
        "bench",
        experiment,
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # Should have multiple benchmark tasks from the sweep
    assert stdout.count("bench serve") >= 2
    # Results should go directly into the experiment dir
    expected_parent = str(Path(experiment).resolve())
    assert expected_parent in stdout


def test_bench_help(run_cli):
    rc, stdout, _ = run_cli("bench", "--help")
    assert rc == 0
    assert "recipes" in stdout
    assert "--ssh-key" in stdout
    assert "--dry-run" in stdout
    assert "--config" in stdout
    assert "--max-workers" in stdout
    assert "--no-teardown" in stdout


def test_teardown_help(run_cli):
    rc, stdout, _ = run_cli("teardown", "--help")
    assert rc == 0
    assert "run_dir" in stdout
    assert "--ssh-key" in stdout
