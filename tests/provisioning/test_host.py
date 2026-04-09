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


def test_ensure_nvidia_install_failure_raises():
    """Regression: when `apt-get install cuda-drivers-...` exits non-zero
    (e.g. unmet dependencies on a CloudRift base image), the harness must
    raise loudly rather than silently proceeding to a half-installed state.

    Before the fix, `_ensure_nvidia_versions` ignored the exit code and
    returned `installed=True`, the caller rebooted, and the bench produced
    empty result tables an hour later with no indication why.
    """
    from deplodock.provisioning.remote import _ensure_nvidia_versions

    class FailingAptHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            self_cmd = cmd
            if "nvidia-smi" in self_cmd:
                return 0, "550.54.15"  # mismatched driver triggers install path
            if "test -d /usr/local/cuda" in self_cmd:
                return 1, ""  # cuda toolkit not present
            if "cuda-keyring" in self_cmd or "apt-get update" in self_cmd:
                return 0, ""
            if "dpkg --configure" in self_cmd or "fix-broken" in self_cmd:
                return 0, ""
            if "apt-get install" in self_cmd and "nvidia-open" in self_cmd:
                return (
                    100,
                    "E: Unmet dependencies. Try 'apt --fix-broken install' with no packages",
                )
            return 0, ""

    host = FailingAptHost()
    with pytest.raises(RuntimeError, match="nvidia-open.*failed"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))


def test_ensure_nvidia_cuda_install_silent_failure_raises():
    """Regression: apt-get install of cuda-toolkit-X-Y can return rc=0 yet not
    actually create /usr/local/cuda-X.Y (kept-back packages, etc.). The post-
    install dir check must catch this and raise."""
    from deplodock.provisioning.remote import _ensure_nvidia_versions

    class SilentlyFailingHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            if "test -d /usr/local/cuda" in cmd:
                return 1, ""  # never present, even after install
            if "cuda-keyring" in cmd or "apt-get update" in cmd:
                return 0, ""
            if "dpkg --configure" in cmd or "fix-broken" in cmd:
                return 0, ""
            if "apt-get install" in cmd:
                return 0, ""  # apt lies and exits 0
            return 0, ""

    host = SilentlyFailingHost()
    with pytest.raises(RuntimeError, match="reported success but"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version="13.2"))


def test_ensure_nvidia_purges_old_packages_first():
    """Regression: when installing a new driver, the harness must purge any
    pre-existing nvidia/libnvidia packages first. CloudRift base images ship
    nvidia-driver-510 from the Ubuntu archive; without a purge step, the
    cuda repo's libnvidia-* (= 595.58.03-1ubuntu1) cannot unpack over the
    older files and the install fails."""
    from deplodock.provisioning.remote import _ensure_nvidia_versions

    class TrackingHost(LocalHost):
        def __init__(self):
            super().__init__()
            self.commands: list[str] = []

        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            self.commands.append(cmd)
            if "nvidia-smi" in cmd:
                return 0, "510.47.03"  # old version triggers install
            if "test -d /usr/local/cuda" in cmd:
                return 0, ""  # cuda toolkit not requested in this test
            if "cuda-keyring" in cmd or "apt-get update" in cmd:
                return 0, ""
            if "dpkg --configure" in cmd or "fix-broken" in cmd:
                return 0, ""
            if "apt-get purge" in cmd:
                return 0, ""
            if "apt-get install" in cmd and "nvidia-open" in cmd:
                return 0, ""
            return 0, ""

    host = TrackingHost()
    asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))
    # Find the install command and the purge command, ensure purge came first.
    purge_idx = next((i for i, c in enumerate(host.commands) if "apt-get purge" in c and "nvidia-*" in c), -1)
    install_idx = next((i for i, c in enumerate(host.commands) if "apt-get install" in c and "nvidia-open" in c), -1)
    assert purge_idx >= 0, f"purge step never invoked. commands: {host.commands}"
    assert install_idx >= 0, f"install step never invoked. commands: {host.commands}"
    assert purge_idx < install_idx, f"purge must run before install (purge at {purge_idx}, install at {install_idx})"
