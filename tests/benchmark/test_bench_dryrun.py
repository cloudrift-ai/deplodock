"""Dry-run tests for the bench command."""

import glob
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


def test_bench_group_log_captures_provisioning(run_cli, make_bench_config, tmp_path):
    """Per-group log files should contain provisioning and deploy logs."""
    import shutil

    import yaml

    # Create a minimal recipe in tmp_path so we don't pollute the repo
    recipe_dir = tmp_path / "TestRecipe"
    recipe_dir.mkdir()
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {"image": "vllm/vllm-openai:latest"},
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "matrices": [
            {"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1},
        ],
    }
    (recipe_dir / "recipe.yaml").write_text(yaml.dump(recipe))

    config_path = make_bench_config(tmp_path)
    rc, stdout, stderr = run_cli(
        "bench",
        str(recipe_dir),
        "--config",
        config_path,
        "--dry-run",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"

    # Find the per-group log file (benchmark_rtx5090_x_1.log)
    group_logs = glob.glob(str(recipe_dir / "*/benchmark_rtx5090_x_1.log"))
    assert group_logs, f"No per-group log file found under {recipe_dir}"

    log = Path(group_logs[0]).read_text()

    # Group logger messages (rtx5090_x_1.*)
    assert "Starting group:" in log, f"Group start missing.\nLog:\n{log}"
    assert "Deploying model..." in log, f"Deploy start missing.\nLog:\n{log}"
    assert "Running benchmark..." in log, f"Benchmark start missing.\nLog:\n{log}"
    assert "Tearing down..." in log, f"Teardown missing.\nLog:\n{log}"

    # Cloud provisioning (deplodock.provisioning.cloudrift)
    assert "Creating CloudRift instance" in log, f"CloudRift provisioning missing.\nLog:\n{log}"

    # Remote provisioning (deplodock.provisioning.remote)
    assert "install docker" in log, f"Remote provisioning missing.\nLog:\n{log}"
    assert "install nvidia-container-toolkit" in log, f"NVIDIA toolkit provisioning missing.\nLog:\n{log}"

    # Deploy orchestration (deplodock.deploy.orchestrate)
    assert "Pulling images" in log, f"Image pull missing.\nLog:\n{log}"
    assert "Downloading model" in log, f"Model download missing.\nLog:\n{log}"
    assert "Cleaning up old containers" in log, f"Container cleanup missing.\nLog:\n{log}"
    assert "Starting services" in log, f"Service start missing.\nLog:\n{log}"
    assert "Waiting for health check" in log, f"Health check missing.\nLog:\n{log}"
    assert "Teardown complete." in log, f"Teardown complete missing.\nLog:\n{log}"

    # SSH transport (deplodock.provisioning.ssh_transport)
    assert "docker compose pull" in log, f"SSH docker compose pull missing.\nLog:\n{log}"
    assert "docker compose up" in log, f"SSH docker compose up missing.\nLog:\n{log}"

    # Clean up run dirs created in tmp_path
    for d in recipe_dir.iterdir():
        if d.is_dir() and d.name != "recipe.yaml":
            shutil.rmtree(d)


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
