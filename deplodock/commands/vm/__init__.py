"""VM lifecycle management: start/stop cloud GPU instances."""

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


def register_vm_command(subparsers):
    """Register the 'vm' command with start/stop action subparsers."""
    from deplodock.commands.vm.gcp_flex_start import register_start_target, register_stop_target

    vm_parser = subparsers.add_parser("vm", help="Manage cloud VM instances")

    action_subparsers = vm_parser.add_subparsers(dest="action", required=True)

    # start action
    start_parser = action_subparsers.add_parser("start", help="Start a VM instance")
    start_subparsers = start_parser.add_subparsers(dest="provider", required=True)
    register_start_target(start_subparsers)

    # stop action
    stop_parser = action_subparsers.add_parser("stop", help="Stop a VM instance")
    stop_subparsers = stop_parser.add_subparsers(dest="provider", required=True)
    register_stop_target(stop_subparsers)
