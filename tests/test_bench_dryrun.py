"""Dry-run tests for the bench command."""

import os


def test_bench_dry_run_basic(run_cli, make_bench_config, tmp_path):
    config_path = make_bench_config(tmp_path)
    rc, stdout, stderr = run_cli("bench", "--config", config_path, "--dry-run")
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "[dry-run]" in stdout


def test_bench_dry_run_deploy_then_benchmark(run_cli, make_bench_config, tmp_path):
    config_path = make_bench_config(tmp_path)
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
    assert pull_idx < bench_idx


def test_bench_server_filter(run_cli, make_bench_config, recipes_dir, tmp_path):
    servers = [
        {
            "name": "server_a",
            "address": "user@1.2.3.4",
            "ssh_key": "~/.ssh/id_ed25519",
            "port": 22,
            "recipes": [
                {
                    "recipe": os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
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
                    "recipe": os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"),
                    "variant": "RTX4090",
                }
            ],
        },
    ]
    config_path = make_bench_config(tmp_path, servers=servers)
    rc, stdout, stderr = run_cli(
        "bench", "--config", config_path, "--dry-run", "--server", "server_a"
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "server_a" in stdout
    assert "5.6.7.8" not in stdout


def test_bench_recipe_filter(run_cli, make_bench_config, recipes_dir, tmp_path):
    recipe_path = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    servers = [
        {
            "name": "test_server",
            "address": "user@1.2.3.4",
            "ssh_key": "~/.ssh/id_ed25519",
            "port": 22,
            "recipes": [
                {"recipe": recipe_path, "variant": "RTX4090"},
                {"recipe": os.path.join(recipes_dir, "GLM-4.6-FP8"), "variant": "8xH200"},
            ],
        },
    ]
    config_path = make_bench_config(tmp_path, servers=servers)
    rc, stdout, stderr = run_cli(
        "bench", "--config", config_path, "--dry-run",
        "--recipe", recipe_path,
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "Qwen3-Coder-30B" in stdout
    # GLM recipe should not be deployed
    assert "GLM-4.6-FP8" not in stdout


def test_bench_help(run_cli):
    rc, stdout, _ = run_cli("bench", "--help")
    assert rc == 0
    assert "--recipe" in stdout
    assert "--dry-run" in stdout
    assert "--config" in stdout
    assert "--force" in stdout
    assert "--server" in stdout
    assert "--parallel" in stdout
    assert "--max-workers" in stdout
