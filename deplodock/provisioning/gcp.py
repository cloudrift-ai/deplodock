"""GCP provider: create/delete GPU VMs using gcloud compute."""

import shlex
import sys
import time

from deplodock.provisioning.shell import run_shell_cmd
from deplodock.provisioning.types import VMConnectionInfo

# ── Command builders ───────────────────────────────────────────────


def _gcloud_create_cmd(
    instance,
    zone,
    machine_type,
    provisioning_model="FLEX_START",
    max_run_duration="7d",
    request_valid_for_duration="2h",
    termination_action="DELETE",
    image_family="debian-12",
    image_project="debian-cloud",
    extra_gcloud_args=None,
):
    """Build gcloud command to create a GPU instance.

    Args:
        provisioning_model: FLEX_START, SPOT, or STANDARD.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance,
        "--zone",
        zone,
        "--machine-type",
        machine_type,
        f"--provisioning-model={provisioning_model}",
        "--maintenance-policy=TERMINATE",
        "--reservation-affinity=none",
        "--image-family",
        image_family,
        "--image-project",
        image_project,
    ]
    # Duration and termination flags are only valid for FLEX_START
    if provisioning_model == "FLEX_START":
        cmd.extend(["--max-run-duration", max_run_duration])
        cmd.append(f"--instance-termination-action={termination_action}")
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
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance,
        "--zone",
        zone,
        "--format",
        "value(status)",
    ]


def _gcloud_external_ip_cmd(instance, zone):
    """Build gcloud command to get external IP."""
    return [
        "gcloud",
        "compute",
        "instances",
        "describe",
        instance,
        "--zone",
        zone,
        "--format",
        "value(networkInterfaces[0].accessConfigs[0].natIP)",
    ]


def _gcloud_ssh_check_cmd(instance, zone, ssh_gateway=None):
    """Build gcloud command to check SSH connectivity."""
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        instance,
        "--zone",
        zone,
        "--command",
        "true",
        "--ssh-flag=-o",
        "--ssh-flag=ConnectTimeout=5",
        "--ssh-flag=-o",
        "--ssh-flag=StrictHostKeyChecking=no",
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
    provisioning_model="FLEX_START",
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
    """Create a GCP GPU instance and optionally wait for SSH.

    Steps:
        1. Issue gcloud compute instances create
        2. Wait for RUNNING status (up to timeout)
        3. Get external IP
        4. Optionally wait for SSH connectivity

    Args:
        provisioning_model: FLEX_START, SPOT, or STANDARD.

    Returns:
        VMConnectionInfo on success, None on failure.
        In dry-run mode, returns a VMConnectionInfo with placeholder values.
    """
    print(f"Creating instance '{instance}' in zone '{zone}' (provisioning: {provisioning_model})...")

    cmd = _gcloud_create_cmd(
        instance,
        zone,
        machine_type,
        provisioning_model=provisioning_model,
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
        return None

    print(f"Waiting for instance to reach RUNNING status (timeout: {timeout}s)...")
    if not wait_for_status(instance, zone, "RUNNING", timeout, dry_run=dry_run):
        return None
    print("Instance is RUNNING.")

    # Get external IP
    rc, stdout, _ = run_shell_cmd(_gcloud_external_ip_cmd(instance, zone), dry_run=dry_run)
    external_ip = stdout.strip() if rc == 0 else ""

    if not dry_run:
        if not external_ip:
            print("Warning: No external IP found.")
        else:
            print(f"External IP: {external_ip}")

    if wait_ssh:
        print(f"Waiting for SSH connectivity (timeout: {wait_ssh_timeout}s)...")
        if not wait_for_ssh(instance, zone, timeout=wait_ssh_timeout, ssh_gateway=ssh_gateway, dry_run=dry_run):
            return None
        print("SSH is ready.")

    return VMConnectionInfo(
        host=external_ip or "dry-run-gcp-host",
        username="",
        ssh_port=22,
        delete_info=("gcp", instance, zone),
    )


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
