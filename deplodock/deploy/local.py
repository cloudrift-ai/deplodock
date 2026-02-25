"""Local transport: run commands and write files locally."""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def make_run_cmd(deploy_dir, dry_run=False):
    """Create a run_cmd callable for local execution."""

    async def run_cmd(command, stream=True, timeout=600, log_output=False):
        if dry_run:
            logger.info(f"[dry-run] {command}")
            return 0, "", ""

        try:
            use_pipe = not stream or log_output
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=deploy_dir,
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
