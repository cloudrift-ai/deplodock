"""Tests for fixed-host mode (--local / --ssh) of `deplodock bench`."""

import os

import pytest

from deplodock.benchmark.fixed_hosts import (
    AllocatedHost,
    parse_ssh_target,
    resolve_fixed_hosts,
    validate_hosts_cover_groups,
)
from deplodock.planner import ExecutionGroup
from deplodock.provisioning.types import VMConnectionInfo


def test_parse_ssh_target_default_port():
    assert parse_ssh_target("alice@example.com") == ("alice", "example.com", 22)


def test_parse_ssh_target_explicit_port():
    assert parse_ssh_target("bob@10.0.0.1:2222") == ("bob", "10.0.0.1", 2222)


@pytest.mark.parametrize("bad", ["nouser", "@host", "user@", "user@host:abc"])
def test_parse_ssh_target_invalid(bad):
    with pytest.raises(ValueError):
        parse_ssh_target(bad)


def _host(gpu_name, count, addr="x@h"):
    return AllocatedHost(
        conn=VMConnectionInfo(host=addr.split("@")[1], username=addr.split("@")[0]),
        gpu_name=gpu_name,
        gpu_count=count,
    )


def test_satisfies_matches_gpu_and_count():
    h = _host("NVIDIA H100 80GB", 4)
    assert h.satisfies("NVIDIA H100 80GB", 4)
    assert h.satisfies("NVIDIA H100 80GB", 2)
    assert not h.satisfies("NVIDIA H100 80GB", 8)
    assert not h.satisfies("NVIDIA B200", 2)


def test_validate_hosts_cover_groups_ok():
    hosts = [_host("NVIDIA H100 80GB", 8)]
    groups = [
        ExecutionGroup(gpu_name="NVIDIA H100 80GB", gpu_count=1),
        ExecutionGroup(gpu_name="NVIDIA H100 80GB", gpu_count=8),
    ]
    validate_hosts_cover_groups(hosts, groups)


def test_validate_hosts_cover_groups_missing_gpu():
    hosts = [_host("NVIDIA H100 80GB", 2)]
    groups = [ExecutionGroup(gpu_name="NVIDIA H100 80GB", gpu_count=8)]
    with pytest.raises(RuntimeError, match="No supplied host can satisfy"):
        validate_hosts_cover_groups(hosts, groups)


async def test_resolve_fixed_hosts_dry_run_skips_detection():
    hosts = await resolve_fixed_hosts(
        use_local=True,
        ssh_targets=["alice@example.com", "bob@10.0.0.1:2200"],
        ssh_key="/dev/null",
        dry_run=True,
    )
    assert len(hosts) == 3
    assert hosts[0].conn.host == "127.0.0.1"
    assert hosts[0].conn.username == os.environ.get("USER", "deploy")
    assert hosts[1].conn.host == "example.com"
    assert hosts[1].conn.username == "alice"
    assert hosts[1].conn.ssh_port == 22
    assert hosts[2].conn.ssh_port == 2200
    # All have unknown GPU in dry-run
    assert all(h.gpu_name is None for h in hosts)


def test_bench_fixed_host_dry_run_cli(run_cli, make_bench_config, recipes_dir, tmp_path):
    """End-to-end CLI: --ssh skips provisioning and runs against the given host."""
    config_path = make_bench_config(tmp_path)
    recipe = os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ")
    rc, stdout, stderr = run_cli(
        "bench",
        recipe,
        "--config",
        config_path,
        "--dry-run",
        "--ssh",
        "fakeuser@fake.host:2222",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    assert "Fixed-host mode" in stdout
    assert "Using pre-allocated host: fakeuser@fake.host" in stdout
    # No cloud provisioning happened
    assert "Creating CloudRift instance" not in stdout
    assert "VM provisioned" not in stdout
    # No instances.json prompt
    assert "Instance info saved" not in stdout
