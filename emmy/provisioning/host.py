"""Host abstraction for provisioning steps.

A Host knows how to run a shell command somewhere — either locally or on a
remote machine over SSH. Privileged commands go through ``run(..., sudo=True)``.
``LocalHost`` refuses sudo so that ``deplodock deploy local`` cannot silently
modify the developer's machine; instead it raises a clear error naming the
offending command. Provisioning code stays generic: it only ever calls
``host.run`` and never branches on local vs remote.
"""

from __future__ import annotations

import asyncio
import logging
import shlex

import click

from deplodock.provisioning.ssh_transport import ssh_base_args

logger = logging.getLogger(__name__)


class Host:
    """Abstract host. Subclasses implement ``run``."""

    name: str
    is_local: bool

    async def run(
        self,
        cmd: str,
        *,
        sudo: bool = False,
        capture: bool = False,
        timeout: int = 600,
    ) -> tuple[int, str]:
        raise NotImplementedError


class LocalHost(Host):
    """Run commands on the local machine. Refuses sudo."""

    name = "local"
    is_local = True

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    async def run(
        self,
        cmd: str,
        *,
        sudo: bool = False,
        capture: bool = False,
        timeout: int = 600,
    ) -> tuple[int, str]:
        if sudo:
            raise click.ClickException(
                f"Refusing to run privileged command on local host: {cmd!r}. "
                "Provisioning steps that require root (driver/CUDA install, "
                "reboot) are only supported for remote deploys (ssh, cloud)."
            )
        if self.dry_run:
            logger.info(f"[dry-run] local: {cmd}")
            return 0, ""
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                cmd,
                stdout=asyncio.subprocess.PIPE if capture else None,
                stderr=asyncio.subprocess.PIPE if capture else None,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout_bytes.decode().strip() if capture and stdout_bytes else ""
            if proc.returncode != 0 and capture and stderr_bytes:
                logger.debug(f"local stderr: {stderr_bytes.decode().strip()}")
            return proc.returncode, out
        except TimeoutError:
            logger.error(f"Local command timed out after {timeout}s: {cmd}")
            proc.kill()
            await proc.wait()
            return 1, ""


class RemoteHost(Host):
    """Run commands on a remote server over SSH."""

    is_local = False

    def __init__(
        self,
        server: str,
        ssh_key: str | None,
        ssh_port: int | None,
        dry_run: bool = False,
    ):
        self.server = server
        self.ssh_key = ssh_key
        self.ssh_port = ssh_port
        self.dry_run = dry_run
        self.name = server

    def _build_args(self, cmd: str, connect_timeout: int | None = None) -> list[str]:
        args = ssh_base_args(self.server, self.ssh_key, self.ssh_port)
        if connect_timeout is not None:
            # Insert ConnectTimeout option right after `ssh`.
            args = args[:1] + ["-o", f"ConnectTimeout={connect_timeout}"] + args[1:]
        args.append(cmd)
        return args

    async def run(
        self,
        cmd: str,
        *,
        sudo: bool = False,
        capture: bool = False,
        timeout: int = 600,
        connect_timeout: int | None = None,
    ) -> tuple[int, str]:
        full = f"sudo bash -c {shlex.quote(cmd)}" if sudo else cmd
        if self.dry_run:
            prefix = "sudo " if sudo else ""
            logger.info(f"[dry-run] ssh {self.server}: {prefix}{cmd}")
            return 0, ""
        args = self._build_args(full, connect_timeout=connect_timeout)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode != 0 and stderr_bytes:
                logger.debug(f"SSH stderr ({self.server}): {stderr_bytes.decode().strip()}")
            out = stdout_bytes.decode().strip() if capture and stdout_bytes else ""
            return proc.returncode, out
        except TimeoutError:
            logger.error(f"SSH command timed out after {timeout}s: {cmd}")
            proc.kill()
            await proc.wait()
            return 1, ""
