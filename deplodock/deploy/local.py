"""Local transport: run commands and write files locally."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def make_run_cmd(deploy_dir, dry_run=False):
    """Create a run_cmd callable for local execution."""

    def run_cmd(command, stream=True):
        if dry_run:
            logger.info(f"[dry-run] {command}")
            return 0, "", ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=deploy_dir,
                capture_output=not stream,
                text=True,
                stdout=None if stream else subprocess.PIPE,
                stderr=None if stream else subprocess.PIPE,
            )
            stdout = "" if stream else (result.stdout or "")
            stderr = "" if stream else (result.stderr or "")
            return result.returncode, stdout, stderr
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return 1, "", ""

    return run_cmd


def make_write_file(deploy_dir, dry_run=False):
    """Create a write_file callable for local file writes."""

    def write_file(path, content):
        full_path = os.path.join(deploy_dir, path)
        if dry_run:
            logger.info(f"[dry-run] write {full_path}")
            return
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    return write_file
