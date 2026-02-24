"""Shell command execution helper."""

import subprocess
import sys


def run_shell_cmd(command, dry_run=False):
    """Run a shell command and return (returncode, stdout, stderr).

    Args:
        command: list of command arguments
        dry_run: if True, print the command instead of executing

    Returns:
        (returncode, stdout, stderr) tuple
    """
    if dry_run:
        print(f"[dry-run] {' '.join(command)}")
        return 0, "", ""

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        print(f"Error: '{command[0]}' not found. Is it installed and on PATH?", file=sys.stderr)
        return 1, "", f"'{command[0]}' not found"
