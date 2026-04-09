"""Tests for the Host abstraction and driver/CUDA provisioning."""

import asyncio

import click
import pytest

from deplodock.provisioning.host import LocalHost, RemoteHost
from deplodock.provisioning.remote import _ensure_nvidia_versions, _matches


def test_matches_prefix():
    assert _matches("595.58.03", "595")
    assert _matches("550.127.05", "550")
    assert not _matches("595.58.03", "550")
    assert _matches("12.4.1", "12.4")
    assert not _matches("12.5.0", "12.4")
    assert not _matches(None, "550")


def test_local_host_sudo_raises():
    host = LocalHost()
    with pytest.raises(click.ClickException, match="Refusing to run privileged"):
        asyncio.run(host.run("apt-get update", sudo=True))


def test_local_host_non_sudo_runs():
    host = LocalHost()
    rc, out = asyncio.run(host.run("echo hello", capture=True))
    assert rc == 0
    assert out == "hello"


def test_local_host_dry_run():
    host = LocalHost(dry_run=True)
    # dry-run still refuses sudo
    with pytest.raises(click.ClickException):
        asyncio.run(host.run("apt-get update", sudo=True))
    # non-sudo dry-run is a no-op
    rc, out = asyncio.run(host.run("echo hello"))
    assert rc == 0


def test_remote_host_dry_run_logs(caplog):
    host = RemoteHost("user@host", None, 22, dry_run=True)
    with caplog.at_level("INFO"):
        rc, _ = asyncio.run(host.run("apt-get update", sudo=True))
    assert rc == 0
    assert any("sudo apt-get update" in r.message for r in caplog.records)


def test_remote_host_build_args_includes_sudo():
    host = RemoteHost("user@host", "/tmp/key", 2222, dry_run=True)
    args = host._build_args("ls")
    assert "ssh" in args[0]
    assert args[-1] == "ls"
    assert "user@host" in args
    assert "-p" in args and "2222" in args
    assert "-i" in args and "/tmp/key" in args


def test_ensure_nvidia_skip_when_matching(caplog):
    """If installed driver matches requested, no sudo is invoked."""

    class FakeHost(LocalHost):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[str, bool]] = []

        async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
            self.calls.append((cmd, sudo))
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            if sudo:
                raise AssertionError(f"sudo unexpectedly invoked: {cmd}")
            return 0, ""

    host = FakeHost()
    with caplog.at_level("INFO"):
        installed = asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))
    assert installed is False
    assert any("already matches" in r.message for r in caplog.records)


def test_ensure_nvidia_install_on_mismatch_raises_locally():
    """LocalHost rejects the sudo install command when driver mismatches."""

    class FakeHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            return await super().run(cmd, sudo=sudo, capture=capture, timeout=timeout)

    host = FakeHost()
    with pytest.raises(click.ClickException, match="cuda"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="550", cuda_version=None))
