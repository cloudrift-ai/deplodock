"""GCP provider: create/delete GPU VMs using gcloud compute."""

import asyncio
import logging
import shlex

from emmy.provisioning.errors import CapacityExhausted, TerminalProvisionError
from emmy.provisioning.shell import run_shell_cmd
from emmy.provisioning.types import VMConnectionInfo

logger = logging.getLogger(__name__)

# Substrings in gcloud stderr that map to CapacityExhausted (try next candidate).
# These are the messages GCP returns when a zone/machine type is out of capacity
# or when the project's quota is the bottleneck — both are zone/type-specific
# and may succeed on the next candidate.
_CAPACITY_STDERR_MARKERS = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
    "QUOTA_EXCEEDED",
    "STOCKOUT",
    "RESOURCE_EXHAUSTED",
)


def _classify_create_failure(stderr: str) -> Exception:
    """Map a gcloud-create stderr blob to the right exception class."""
    if any(marker in stderr for marker in _CAPACITY_STDERR_MARKERS):
        return CapacityExhausted(f"gcloud create reported capacity error: {stderr.strip()}")
    return TerminalProvisionError(f"gcloud create failed: {stderr.strip()}")


def _duration_to_seconds(duration: str) -> int:
    """Convert a GCP duration string (e.g. '2h', '30m', '1800s') to seconds."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    suffix = duration[-1]
    if suffix in units:
        return int(duration[:-1]) * units[suffix]
    return int(duration)


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
        # Return immediately; we poll for RUNNING status ourselves.
        # Without --async, gcloud waits up to 1800s internally which is
        # shorter than typical FLEX_START provisioning windows.
        cmd.append("--async")
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


async def wait_for_status(instance, zone, target_status, timeout, interval=10, dry_run=False):
    """Poll instance status until it matches target_status or timeout.

    Returns:
        True if target status reached, False on timeout.
    """
    if dry_run:
        cmd = _gcloud_status_cmd(instance, zone)
        logger.info(f"[dry-run] Poll every {interval}s (up to {timeout}s): {' '.join(cmd)} -> {target_status}")
        return True

    elapsed = 0
    while elapsed < timeout:
        rc, stdout, _ = await run_shell_cmd(_gcloud_status_cmd(instance, zone))
        status = stdout.strip()
        if rc == 0 and status == target_status:
            return True
        await asyncio.sleep(interval)
        elapsed += interval

    logger.error(f"Timeout after {timeout}s waiting for status '{target_status}' (last: '{status}')")
    return False


async def wait_for_ssh(instance, zone, timeout=300, interval=10, ssh_gateway=None, dry_run=False):
    """Poll SSH connectivity until success or timeout.

    Returns:
        True if SSH connected, False on timeout.
    """
    if dry_run:
        cmd = _gcloud_ssh_check_cmd(instance, zone, ssh_gateway=ssh_gateway)
        logger.info(f"[dry-run] Poll SSH every {interval}s (up to {timeout}s): {' '.join(cmd)}")
        return True

    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = await run_shell_cmd(_gcloud_ssh_check_cmd(instance, zone, ssh_gateway=ssh_gateway))
        if rc == 0:
            return True
        await asyncio.sleep(interval)
        elapsed += interval

    logger.error(f"Timeout after {timeout}s waiting for SSH connectivity")
    return False


async def create_instance(
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
    logger.info(f"Creating instance '{instance}' in zone '{zone}' (provisioning: {provisioning_model})...")

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
    rc, stdout, stderr = await run_shell_cmd(cmd, dry_run=dry_run, timeout=timeout)
    if rc != 0:
        logger.error(f"Failed to create instance: {stderr.strip()}")
        # Before raising, nothing has been provisioned — no orphan to clean up.
        raise _classify_create_failure(stderr)

    # gcloud create succeeded; from here on, any failure leaks the VM unless we
    # explicitly delete it. Mirror CloudRift's orphan-termination pattern so
    # callers (orchestrator or direct) can rely on a uniform cleanup invariant.
    try:
        logger.info(f"Waiting for instance to reach RUNNING status (timeout: {timeout}s)...")
        if not await wait_for_status(instance, zone, "RUNNING", timeout, dry_run=dry_run):
            raise CapacityExhausted(f"GCP instance '{instance}' did not reach RUNNING within {timeout}s in zone {zone}")
        logger.info("Instance is RUNNING.")

        rc, stdout, _ = await run_shell_cmd(_gcloud_external_ip_cmd(instance, zone), dry_run=dry_run)
        external_ip = stdout.strip() if rc == 0 else ""

        if not dry_run:
            if not external_ip:
                logger.warning("Warning: No external IP found.")
            else:
                logger.info(f"External IP: {external_ip}")

        if wait_ssh:
            logger.info(f"Waiting for SSH connectivity (timeout: {wait_ssh_timeout}s)...")
            if not await wait_for_ssh(instance, zone, timeout=wait_ssh_timeout, ssh_gateway=ssh_gateway, dry_run=dry_run):
                raise RuntimeError(f"SSH never came up on GCP instance '{instance}' within {wait_ssh_timeout}s")
            logger.info("SSH is ready.")

        return VMConnectionInfo(
            host=external_ip or "dry-run-gcp-host",
            username="",
            ssh_port=22,
            delete_info=("gcp", instance, zone),
        )
    except Exception:
        logger.warning(f"Terminating orphan GCP instance '{instance}' in zone '{zone}' after exception.")
        try:
            await run_shell_cmd(_gcloud_delete_cmd(instance, zone), dry_run=dry_run)
        except Exception as cleanup_exc:
            logger.error(f"Failed to delete orphan GCP instance '{instance}': {cleanup_exc}")
        raise


async def delete_instance(instance, zone, dry_run=False):
    """Delete a GCP instance.

    Uses gcloud compute instances delete --quiet (blocks until complete).
    """
    logger.info(f"Deleting instance '{instance}' in zone '{zone}'...")

    rc, stdout, stderr = await run_shell_cmd(_gcloud_delete_cmd(instance, zone), dry_run=dry_run)
    if rc != 0:
        logger.error(f"Failed to delete instance: {stderr.strip()}")
        return False

    logger.info("Instance deleted.")
    return True
