"""SSH transport: run commands and write files on remote servers via SSH/SCP."""

import asyncio
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

REMOTE_DEPLOY_DIR = "~/deploy"


def ssh_base_args(server, ssh_key, ssh_port):
    """Build base SSH arguments."""
    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=5",
    ]
    if ssh_key:
        args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        args += ["-p", str(ssh_port)]
    args.append(server)
    return args


def make_run_cmd(server, ssh_key, ssh_port, dry_run=False):
    """Create a run_cmd callable for SSH execution."""

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        # Use sg to run docker commands under the docker group
        if command.strip().startswith("docker"):
            escaped = command.replace('"', '\\"')
            full_cmd = f'sg docker -c "cd {REMOTE_DEPLOY_DIR} && {escaped}"'
        else:
            full_cmd = f"cd {REMOTE_DEPLOY_DIR} && {command}"
        if dry_run:
            logger.info(f"[dry-run] ssh {server}: {full_cmd}")
            return 0, "", ""

        ssh_args = ssh_base_args(server, ssh_key, ssh_port)
        ssh_args.append(full_cmd)

        try:
            use_pipe = not stream or log_output
            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE if use_pipe else None,
                stderr=asyncio.subprocess.PIPE if use_pipe else None,
            )

            if log_output:
                stdout_lines, stderr_lines = [], []

                async def _read_stream(pipe, lines, level):
                    async for raw_line in pipe:
                        line = raw_line.decode().rstrip("\n")
                        logger.log(level, line)
                        lines.append(line)

                await asyncio.wait_for(
                    asyncio.gather(
                        _read_stream(proc.stdout, stdout_lines, logging.INFO),
                        _read_stream(proc.stderr, stderr_lines, logging.ERROR),
                        proc.wait(),
                    ),
                    timeout=timeout,
                )
                return proc.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                stdout = "" if stream else (stdout_bytes.decode() if stdout_bytes else "")
                stderr = "" if stream else (stderr_bytes.decode() if stderr_bytes else "")
                return proc.returncode, stdout, stderr
        except TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {command}")
            proc.kill()
            await proc.wait()
            return 1, "", ""
        except Exception as e:
            logger.error(f"Error running SSH command: {e}")
            return 1, "", ""

    return run_cmd


async def scp_file(local_path, server, ssh_key, ssh_port, remote_path, timeout=300):
    """Copy a file to the remote server via SCP."""
    scp_args = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=5",
    ]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        scp_args += ["-P", str(ssh_port)]
    scp_args += [local_path, f"{server}:{remote_path}"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *scp_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        return proc.returncode, stderr
    except TimeoutError:
        logger.error(f"SCP timed out after {timeout}s: {local_path} -> {server}:{remote_path}")
        proc.kill()
        await proc.wait()
        return 1, "timeout"


def make_write_file(server, ssh_key, ssh_port, dry_run=False):
    """Create a write_file callable that SCPs files to the remote server."""

    async def write_file(path, content):
        remote_path = f"{REMOTE_DEPLOY_DIR}/{path}"
        if dry_run:
            logger.info(f"[dry-run] scp {path} -> {server}:{remote_path}")
            return

        # Write to a temp file locally, then SCP
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{path}", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            rc, stderr = await scp_file(tmp_path, server, ssh_key, ssh_port, remote_path)
            if rc != 0:
                logger.error(f"Failed to SCP {path} to {server}:{remote_path}: {stderr}")
        finally:
            os.unlink(tmp_path)

    return write_file
