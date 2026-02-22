"""GCP flex-start provider: start/stop GPU VMs using gcloud compute."""

import sys
import time

from deplodock.commands.vm import run_shell_cmd


# ── Command builders ───────────────────────────────────────────────


def _gcloud_start_cmd(instance, zone):
    """Build gcloud command to start an instance."""
    return ["gcloud", "compute", "instances", "start", instance, "--zone", zone]


def _gcloud_stop_cmd(instance, zone):
    """Build gcloud command to stop an instance."""
    return ["gcloud", "compute", "instances", "stop", instance, "--zone", zone]


def _gcloud_status_cmd(instance, zone):
    """Build gcloud command to get instance status."""
    return [
        "gcloud", "compute", "instances", "describe", instance,
        "--zone", zone, "--format", "value(status)",
    ]


def _gcloud_external_ip_cmd(instance, zone):
    """Build gcloud command to get external IP."""
    return [
        "gcloud", "compute", "instances", "describe", instance,
        "--zone", zone, "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)",
    ]


def _gcloud_ssh_check_cmd(instance, zone):
    """Build gcloud command to check SSH connectivity."""
    return [
        "gcloud", "compute", "ssh", instance,
        "--zone", zone, "--command", "true",
        "--ssh-flag=-o", "--ssh-flag=ConnectTimeout=5",
        "--ssh-flag=-o", "--ssh-flag=StrictHostKeyChecking=no",
    ]


# ── Core logic ─────────────────────────────────────────────────────


def wait_for_status(instance, zone, target_status, timeout, interval=10, dry_run=False):
    """Poll instance status until it matches target_status or timeout.

    Returns:
        True if target status reached, False on timeout.
    """
    if dry_run:
        cmd = _gcloud_status_cmd(instance, zone)
        print(f"[dry-run] Poll every {interval}s (up to {timeout}s): {' '.join(cmd)} -> {target_status}")
        return True

    elapsed = 0
    while elapsed < timeout:
        rc, stdout, _ = run_shell_cmd(_gcloud_status_cmd(instance, zone))
        status = stdout.strip()
        if rc == 0 and status == target_status:
            return True
        time.sleep(interval)
        elapsed += interval

    print(f"Timeout after {timeout}s waiting for status '{target_status}' (last: '{status}')", file=sys.stderr)
    return False


def wait_for_ssh(instance, zone, timeout=300, interval=10, dry_run=False):
    """Poll SSH connectivity until success or timeout.

    Returns:
        True if SSH connected, False on timeout.
    """
    if dry_run:
        cmd = _gcloud_ssh_check_cmd(instance, zone)
        print(f"[dry-run] Poll SSH every {interval}s (up to {timeout}s): {' '.join(cmd)}")
        return True

    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = run_shell_cmd(_gcloud_ssh_check_cmd(instance, zone))
        if rc == 0:
            return True
        time.sleep(interval)
        elapsed += interval

    print(f"Timeout after {timeout}s waiting for SSH connectivity", file=sys.stderr)
    return False


def start_instance(instance, zone, timeout=14400, wait_ssh=False, wait_ssh_timeout=300, dry_run=False):
    """Start a GCP instance and optionally wait for SSH.

    Steps:
        1. Issue gcloud compute instances start
        2. Wait for RUNNING status (up to timeout)
        3. Print external IP
        4. Optionally wait for SSH connectivity
    """
    print(f"Starting instance '{instance}' in zone '{zone}'...")

    rc, stdout, stderr = run_shell_cmd(_gcloud_start_cmd(instance, zone), dry_run=dry_run)
    if rc != 0:
        print(f"Failed to start instance: {stderr.strip()}", file=sys.stderr)
        return False

    print(f"Waiting for instance to reach RUNNING status (timeout: {timeout}s)...")
    if not wait_for_status(instance, zone, "RUNNING", timeout, dry_run=dry_run):
        return False
    print("Instance is RUNNING.")

    # Get and print external IP
    rc, stdout, _ = run_shell_cmd(_gcloud_external_ip_cmd(instance, zone), dry_run=dry_run)
    if rc == 0:
        ip = stdout.strip()
        if ip:
            print(f"External IP: {ip}")
        elif not dry_run:
            print("Warning: No external IP found.")

    if wait_ssh:
        print(f"Waiting for SSH connectivity (timeout: {wait_ssh_timeout}s)...")
        if not wait_for_ssh(instance, zone, timeout=wait_ssh_timeout, dry_run=dry_run):
            return False
        print("SSH is ready.")

    return True


def stop_instance(instance, zone, timeout=300, dry_run=False):
    """Stop a GCP instance and wait for TERMINATED status.

    Steps:
        1. Issue gcloud compute instances stop
        2. Wait for TERMINATED status (up to timeout)
    """
    print(f"Stopping instance '{instance}' in zone '{zone}'...")

    rc, stdout, stderr = run_shell_cmd(_gcloud_stop_cmd(instance, zone), dry_run=dry_run)
    if rc != 0:
        print(f"Failed to stop instance: {stderr.strip()}", file=sys.stderr)
        return False

    print(f"Waiting for instance to reach TERMINATED status (timeout: {timeout}s)...")
    if not wait_for_status(instance, zone, "TERMINATED", timeout, dry_run=dry_run):
        return False
    print("Instance is TERMINATED.")

    return True


# ── CLI handlers ───────────────────────────────────────────────────


def handle_start(args):
    """CLI handler for 'vm start gcp-flex-start'."""
    success = start_instance(
        instance=args.instance,
        zone=args.zone,
        timeout=args.timeout,
        wait_ssh=args.wait_ssh,
        wait_ssh_timeout=args.wait_ssh_timeout,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


def handle_stop(args):
    """CLI handler for 'vm stop gcp-flex-start'."""
    success = stop_instance(
        instance=args.instance,
        zone=args.zone,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


# ── Registration ───────────────────────────────────────────────────


def _add_common_args(parser):
    """Add arguments shared by start and stop."""
    parser.add_argument("--instance", required=True, help="GCP instance name")
    parser.add_argument("--zone", required=True, help="GCP zone (e.g. us-central1-a)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")


def register_start_target(subparsers):
    """Register the gcp-flex-start provider under 'vm start'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Start a GCP flex-start GPU VM")
    _add_common_args(parser)
    parser.set_defaults(timeout=14400)
    parser.add_argument("--wait-ssh", action="store_true", help="Wait for SSH connectivity after start")
    parser.add_argument("--wait-ssh-timeout", type=int, default=300, help="SSH wait timeout in seconds (default: 300)")
    parser.set_defaults(func=handle_start)


def register_stop_target(subparsers):
    """Register the gcp-flex-start provider under 'vm stop'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Stop a GCP flex-start GPU VM")
    _add_common_args(parser)
    parser.set_defaults(timeout=300)
    parser.set_defaults(func=handle_stop)
