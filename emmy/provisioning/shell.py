"""Shell command execution helper."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def run_shell_cmd(command, dry_run=False, timeout=600):
    """Run a shell command and return (returncode, stdout, stderr).

    Args:
        command: list of command arguments
        dry_run: if True, print the command instead of executing
        timeout: maximum seconds to wait for the command

    Returns:
        (returncode, stdout, stderr) tuple
    """
    if dry_run:
        logger.info(f"[dry-run] {' '.join(command)}")
        return 0, "", ""

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        return proc.returncode, stdout, stderr
    except TimeoutError:
        logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
        proc.kill()
        await proc.wait()
        return 1, "", ""
    except FileNotFoundError:
        logger.error(f"Error: '{command[0]}' not found. Is it installed and on PATH?")
        return 1, "", f"'{command[0]}' not found"
