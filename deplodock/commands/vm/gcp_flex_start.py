"""GCP flex-start provider: create/delete GPU VMs using gcloud compute."""

import shlex
import sys
import time

from deplodock.commands.vm import run_shell_cmd


# ── Command builders ───────────────────────────────────────────────


def _gcloud_create_cmd(
    instance,
    zone,
    machine_type,
    max_run_duration="7d",
    request_valid_for_duration="2h",
    termination_action="DELETE",
    image_family="debian-12",
    image_project="debian-cloud",
    extra_gcloud_args=None,
):
    """Build gcloud command to create a flex-start instance."""
    cmd = [
        "gcloud", "compute", "instances", "create", instance,
        "--zone", zone,
        "--machine-type", machine_type,
        "--provisioning-model=FLEX_START",
        "--maintenance-policy=TERMINATE",
        "--reservation-affinity=none",
        "--max-run-duration", max_run_duration,
        f"--instance-termination-action={termination_action}",
        "--image-family", image_family,
        "--image-project", image_project,
    ]
    if request_valid_for_duration:
        cmd.extend(["--request-valid-for-duration", request_valid_for_duration])
    if extra_gcloud_args:
        cmd.extend(shlex.split(extra_gcloud_args))
    return cmd


def _gcloud_delete_cmd(instance, zone):
    """Build gcloud command to delete an instance."""
    return ["gcloud", "compute", "instances", "delete", instance, "--zone", zone, "--quiet"]


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


def _gcloud_ssh_check_cmd(instance, zone, ssh_gateway=None):
    """Build gcloud command to check SSH connectivity."""
    cmd = [
        "gcloud", "compute", "ssh", instance,
        "--zone", zone, "--command", "true",
        "--ssh-flag=-o", "--ssh-flag=ConnectTimeout=5",
        "--ssh-flag=-o", "--ssh-flag=StrictHostKeyChecking=no",
    ]
    if ssh_gateway:
        cmd.extend(["--ssh-flag=-o", f"--ssh-flag=ProxyJump={ssh_gateway}"])
    return cmd


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


def wait_for_ssh(instance, zone, timeout=300, interval=10, ssh_gateway=None, dry_run=False):
    """Poll SSH connectivity until success or timeout.

    Returns:
        True if SSH connected, False on timeout.
    """
    if dry_run:
        cmd = _gcloud_ssh_check_cmd(instance, zone, ssh_gateway=ssh_gateway)
        print(f"[dry-run] Poll SSH every {interval}s (up to {timeout}s): {' '.join(cmd)}")
        return True

    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = run_shell_cmd(_gcloud_ssh_check_cmd(instance, zone, ssh_gateway=ssh_gateway))
        if rc == 0:
            return True
        time.sleep(interval)
        elapsed += interval

    print(f"Timeout after {timeout}s waiting for SSH connectivity", file=sys.stderr)
    return False


def create_instance(
    instance,
    zone,
    machine_type,
    max_run_duration="7d",
    request_valid_for_duration="2h",
    termination_action="DELETE",
    image_family="debian-12",
    image_project="debian-cloud",
    extra_gcloud_args=None,
    timeout=14400,
    wait_ssh=False,
    wait_ssh_timeout=300,
    ssh_gateway=None,
    dry_run=False,
):
    """Create a GCP flex-start instance and optionally wait for SSH.

    Steps:
        1. Issue gcloud compute instances create with flex-start flags
        2. Wait for RUNNING status (up to timeout)
        3. Print external IP
        4. Optionally wait for SSH connectivity
    """
    print(f"Creating flex-start instance '{instance}' in zone '{zone}'...")

    cmd = _gcloud_create_cmd(
        instance, zone, machine_type,
        max_run_duration=max_run_duration,
        request_valid_for_duration=request_valid_for_duration,
        termination_action=termination_action,
        image_family=image_family,
        image_project=image_project,
        extra_gcloud_args=extra_gcloud_args,
    )
    rc, stdout, stderr = run_shell_cmd(cmd, dry_run=dry_run)
    if rc != 0:
        print(f"Failed to create instance: {stderr.strip()}", file=sys.stderr)
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
        if not wait_for_ssh(instance, zone, timeout=wait_ssh_timeout, ssh_gateway=ssh_gateway, dry_run=dry_run):
            return False
        print("SSH is ready.")

    return True


def delete_instance(instance, zone, dry_run=False):
    """Delete a GCP instance.

    Uses gcloud compute instances delete --quiet (blocks until complete).
    """
    print(f"Deleting instance '{instance}' in zone '{zone}'...")

    rc, stdout, stderr = run_shell_cmd(_gcloud_delete_cmd(instance, zone), dry_run=dry_run)
    if rc != 0:
        print(f"Failed to delete instance: {stderr.strip()}", file=sys.stderr)
        return False

    print("Instance deleted.")
    return True


# ── CLI handlers ───────────────────────────────────────────────────


def handle_create(args):
    """CLI handler for 'vm create gcp-flex-start'."""
    success = create_instance(
        instance=args.instance,
        zone=args.zone,
        machine_type=args.machine_type,
        max_run_duration=args.max_run_duration,
        request_valid_for_duration=args.request_valid_for_duration,
        termination_action=args.termination_action,
        image_family=args.image_family,
        image_project=args.image_project,
        extra_gcloud_args=args.gcloud_args,
        timeout=args.timeout,
        wait_ssh=args.wait_ssh,
        wait_ssh_timeout=args.wait_ssh_timeout,
        ssh_gateway=args.ssh_gateway,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


def handle_delete(args):
    """CLI handler for 'vm delete gcp-flex-start'."""
    success = delete_instance(
        instance=args.instance,
        zone=args.zone,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


# ── Registration ───────────────────────────────────────────────────


def register_create_target(subparsers):
    """Register the gcp-flex-start provider under 'vm create'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Create a GCP flex-start GPU VM")
    parser.add_argument("--instance", required=True, help="GCP instance name")
    parser.add_argument("--zone", required=True, help="GCP zone (e.g. us-central1-a)")
    parser.add_argument("--machine-type", required=True, help="Machine type (e.g. a2-highgpu-1g)")
    parser.add_argument("--max-run-duration", default="7d", help="Max VM run time (default: 7d)")
    parser.add_argument("--request-valid-for-duration", default="2h", help="How long to wait for capacity (default: 2h)")
    parser.add_argument("--termination-action", default="DELETE", choices=["STOP", "DELETE"], help="Action when max-run-duration expires (default: DELETE)")
    parser.add_argument("--image-family", default="debian-12", help="Boot disk image family (default: debian-12)")
    parser.add_argument("--image-project", default="debian-cloud", help="Boot disk image project (default: debian-cloud)")
    parser.add_argument("--gcloud-args", default=None, help="Extra args passed to gcloud compute instances create")
    parser.add_argument("--timeout", type=int, default=14400, help="How long to poll for RUNNING status in seconds (default: 14400)")
    parser.add_argument("--wait-ssh", action="store_true", help="Wait for SSH connectivity after VM is RUNNING")
    parser.add_argument("--wait-ssh-timeout", type=int, default=300, help="SSH wait timeout in seconds (default: 300)")
    parser.add_argument("--ssh-gateway", default=None, help="SSH gateway host for ProxyJump (e.g. gcp-ssh-gateway)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_create)


def register_delete_target(subparsers):
    """Register the gcp-flex-start provider under 'vm delete'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Delete a GCP flex-start GPU VM")
    parser.add_argument("--instance", required=True, help="GCP instance name")
    parser.add_argument("--zone", required=True, help="GCP zone (e.g. us-central1-a)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_delete)
