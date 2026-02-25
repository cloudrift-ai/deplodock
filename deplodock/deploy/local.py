"""Local transport: run commands and write files locally."""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def make_run_cmd(deploy_dir, dry_run=False):
    """Create a run_cmd callable for local execution."""

    async def run_cmd(command, stream=True, timeout=600):
        if dry_run:
            logger.info(f"[dry-run] {command}")
            return 0, "", ""

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=deploy_dir,
                stdout=None if stream else asyncio.subprocess.PIPE,
                stderr=None if stream else asyncio.subprocess.PIPE,
            )
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
            logger.error(f"Error running command: {e}")
            return 1, "", ""

    return run_cmd


def make_write_file(deploy_dir, dry_run=False):
    """Create a write_file callable for local file writes."""

    async def write_file(path, content):
        full_path = os.path.join(deploy_dir, path)
        if dry_run:
            logger.info(f"[dry-run] write {full_path}")
            return
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    return write_file
