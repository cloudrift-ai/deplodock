"""VM lifecycle management: create/delete cloud GPU instances."""

import subprocess
import sys
import time


def wait_for_ssh(host, username, ssh_port, ssh_key_path, timeout=120, interval=5):
    """Poll SSH connectivity until success or timeout.

    Uses plain ssh (not gcloud) for provider-agnostic SSH readiness check.

    Returns:
        True if SSH connected, False on timeout.
    """
    address = f"{username}@{host}" if username else host
    elapsed = 0
    while elapsed < timeout:
        rc = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             "-i", ssh_key_path, "-p", str(ssh_port),
             address, "true"],
            capture_output=True,
        ).returncode
        if rc == 0:
            return True
        time.sleep(interval)
        elapsed += interval

    print(f"Timeout after {timeout}s waiting for SSH connectivity to {address}:{ssh_port}", file=sys.stderr)
    return False


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
    """Register the 'vm' command with create/delete action subparsers."""
    from deplodock.commands.vm.gcp_flex_start import register_create_target, register_delete_target
    from deplodock.commands.vm.cloudrift import (
        register_create_target as register_cloudrift_create,
        register_delete_target as register_cloudrift_delete,
    )

    vm_parser = subparsers.add_parser("vm", help="Manage cloud VM instances")

    action_subparsers = vm_parser.add_subparsers(dest="action", required=True)

    # create action
    create_parser = action_subparsers.add_parser("create", help="Create a VM instance")
    create_subparsers = create_parser.add_subparsers(dest="provider", required=True)
    register_create_target(create_subparsers)
    register_cloudrift_create(create_subparsers)

    # delete action
    delete_parser = action_subparsers.add_parser("delete", help="Delete a VM instance")
    delete_subparsers = delete_parser.add_subparsers(dest="provider", required=True)
    register_delete_target(delete_subparsers)
    register_cloudrift_delete(delete_subparsers)
