"""CloudRift provider: create/delete GPU VMs via the CloudRift REST API."""

import asyncio
import json
import logging
import os

import httpx

from deplodock.provisioning.errors import CapacityExhausted, TerminalProvisionError
from deplodock.provisioning.ssh import wait_for_ssh
from deplodock.provisioning.types import VMConnectionInfo

logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.environ.get("CLOUDRIFT_API_URL", "https://api.cloudrift.ai")
DEFAULT_IMAGE_URL_NVIDIA = (
    "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpu-580-129-20251015-183936.img"
)
DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY = (
    "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpup-580-129-20260430-084759.img"
)
DEFAULT_IMAGE_URL_AMD = "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-rocm-64-20260220-025112.img"
DEFAULT_CLOUDINIT_URL = "https://storage.googleapis.com/cloudrift-vm-disks/cloudinit/ubuntu-base.cloudinit"
API_VERSION = "2026-02-10"


def select_image_url(instance_type):
    """Pick the right OS image for a CloudRift instance type.

    AMD Instinct instance types start with ``mi`` (e.g. ``mi350x-15-250-1000-gv.1``)
    and need the ROCm image. Pre-Turing NVIDIA GPUs (V100, P100) lack a GPU System
    Processor and can't load the open kernel modules baked into the standard image,
    so they need the proprietary-driver image. Everything else gets the default
    open-driver NVIDIA image. Mismatches leave the GPU unusable because the kernel
    module for the wrong vendor — or wrong driver flavor — isn't on disk.
    """
    if instance_type.startswith("mi"):
        return DEFAULT_IMAGE_URL_AMD
    if instance_type.startswith(("v100", "p100")):
        return DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY
    return DEFAULT_IMAGE_URL_NVIDIA


# ── API helpers ───────────────────────────────────────────────────


async def _api_request(method, path, data, api_key, api_url=DEFAULT_API_URL, dry_run=False):
    """Make an authenticated CloudRift API request.

    Wraps *data* in the versioned envelope ``{"version": ..., "data": ...}``.

    Returns:
        Parsed JSON response ``data`` dict, or ``None`` in dry-run mode.
    """
    url = f"{api_url}{path}"
    payload = {"version": API_VERSION, "data": data}

    if dry_run:
        logger.info(f"[dry-run] {method} {url}")
        logger.info(f"[dry-run] payload: {json.dumps(payload, indent=2)}")
        return None

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", resp.json())


async def _rent_instance(
    api_key,
    instance_type,
    ssh_public_keys,
    image_url,
    cloudinit_url=DEFAULT_CLOUDINIT_URL,
    ports=None,
    api_url=DEFAULT_API_URL,
    dry_run=False,
    billing_exempt=False,
    network=None,
):
    """Rent a new CloudRift VM instance.

    POST /api/v1/instances/rent

    Args:
        ssh_public_keys: list of public key strings (e.g. ["ssh-ed25519 AAAA..."])
    """
    vm_config = {
        "ssh_key": {"PublicKeys": ssh_public_keys},
        "image_url": image_url,
        "cloudinit_url": cloudinit_url,
    }
    if ports:
        vm_config["ports"] = [str(p) for p in ports]
    data = {
        "selector": {
            "ByInstanceTypeAndLocation": {
                "instance_type": instance_type,
            },
        },
        "config": {
            "VirtualMachine": vm_config,
        },
        "with_public_ip": True,
    }
    if billing_exempt:
        data["billing_exempt"] = True
    if network is not None:
        data["network"] = network
    return await _api_request("POST", "/api/v1/instances/rent", data, api_key, api_url, dry_run)


async def _terminate_instance(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Terminate a CloudRift instance.

    POST /api/v1/instances/terminate with ById selector.
    """
    data = {"selector": {"ById": [instance_id]}}
    return await _api_request("POST", "/api/v1/instances/terminate", data, api_key, api_url, dry_run)


async def _get_instance_info(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Get info for a single instance by ID.

    POST /api/v1/instances/list with ById selector.
    Returns the instance dict or None.
    """
    data = {"selector": {"ById": [instance_id]}}
    result = await _api_request("POST", "/api/v1/instances/list", data, api_key, api_url, dry_run)
    if result is None:  # dry-run
        return None
    instances = result.get("instances", [])
    return instances[0] if instances else None


async def _list_ssh_keys(api_key, api_url=DEFAULT_API_URL, dry_run=False):
    """List registered SSH keys.

    POST /api/v1/ssh-keys/list
    """
    return await _api_request("POST", "/api/v1/ssh-keys/list", {}, api_key, api_url, dry_run)


async def _add_ssh_key(api_key, name, public_key, api_url=DEFAULT_API_URL, dry_run=False):
    """Register a new SSH key.

    POST /api/v1/ssh-keys/add
    """
    data = {"name": name, "public_key": public_key}
    return await _api_request("POST", "/api/v1/ssh-keys/add", data, api_key, api_url, dry_run)


# ── Core logic ─────────────────────────────────────────────────────


async def _ensure_ssh_key(api_key, ssh_key_path, api_url=DEFAULT_API_URL, dry_run=False):
    """Ensure the SSH public key is registered on CloudRift.

    Reads the public key file, checks existing keys by content match,
    and registers if not found.

    Returns:
        The SSH key ID (str), or "dry-run-key-id" in dry-run mode.
    """
    ssh_key_path = os.path.expanduser(ssh_key_path)
    if dry_run and not os.path.exists(ssh_key_path):
        return "dry-run-key-id"
    with open(ssh_key_path) as f:
        public_key = f.read().strip()

    result = await _list_ssh_keys(api_key, api_url, dry_run)
    if result is not None:
        for key in result.get("keys", []):
            if key.get("public_key", "").strip() == public_key:
                logger.info(f"SSH key already registered (id={key['id']}).")
                return key["id"]

    # Register new key
    key_name = os.path.basename(ssh_key_path)
    logger.info(f"Registering SSH key '{key_name}' on CloudRift...")
    add_result = await _add_ssh_key(api_key, key_name, public_key, api_url, dry_run)
    if dry_run:
        return "dry-run-key-id"
    key_id = add_result["ssh_key"]["id"]
    logger.info(f"SSH key registered (id={key_id}).")
    return key_id


async def wait_for_status(
    api_key, instance_id, target_status, timeout, api_url=DEFAULT_API_URL, interval=10, dry_run=False, fail_statuses=None
):
    """Poll instance status until it matches *target_status* or timeout.

    Args:
        fail_statuses: optional set of status strings that trigger immediate failure.

    Returns:
        The instance dict if target status reached, None on timeout or fail status.
    """
    if dry_run:
        logger.info(f"[dry-run] Poll every {interval}s (up to {timeout}s) for status '{target_status}'")
        return {"status": target_status}

    fail_statuses = fail_statuses or set()
    elapsed = 0
    status = None
    last_info = None
    while elapsed < timeout:
        try:
            info = await _get_instance_info(api_key, instance_id, api_url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                logger.warning(f"Transient {exc.response.status_code} from CloudRift while polling {instance_id}; retrying.")
                await asyncio.sleep(interval)
                elapsed += interval
                continue
            raise
        except httpx.RequestError as exc:
            logger.warning(f"Network error polling CloudRift for {instance_id}: {exc}; retrying.")
            await asyncio.sleep(interval)
            elapsed += interval
            continue
        if info is None:
            logger.warning(f"Warning: instance {instance_id} not found.")
            await asyncio.sleep(interval)
            elapsed += interval
            continue
        last_info = info
        status = info.get("status")
        if status == target_status and _instance_fully_ready(info):
            return info
        if status in fail_statuses:
            logger.error(f"Instance {instance_id} reached fail status '{status}'")
            return None
        await asyncio.sleep(interval)
        elapsed += interval

    if last_info is not None and last_info.get("status") == target_status:
        vms = last_info.get("virtual_machines") or []
        vm_ready = vms[0].get("ready") if vms else None
        logger.error(
            f"Timeout after {timeout}s: instance {instance_id} reached '{target_status}' but never became ready "
            f"(host_address={last_info.get('host_address')!r}, port_mappings={last_info.get('port_mappings')!r}, "
            f"vm_ready={vm_ready!r})"
        )
    else:
        logger.error(f"Timeout after {timeout}s waiting for status '{target_status}' (last: '{status}')")
    return None


def _instance_fully_ready(info):
    """Return True if the instance has finished networking and (for VMs) is ready.

    CloudRift flips ``status`` to ``Active`` before ``host_address`` / ``port_mappings``
    are populated and before the VM's own ``ready`` flag is set. Acting on Active
    alone gives a connection target with ``host=None`` and an unset port table.

    ``port_mappings`` is ``None`` until allocation completes, and ``[]`` once
    allocation finishes for instances that expose ports directly on the host IP
    (no NAT forwarding). Both ``[]`` and a populated list count as ready.
    """
    if not info.get("host_address") or info.get("port_mappings") is None:
        return False
    vms = info.get("virtual_machines") or []
    if vms and not vms[0].get("ready"):
        return False
    return True


def _extract_connection_info(instance, delete_info=()):
    """Extract connection info from an instance dict into a VMConnectionInfo.

    VMs provide login credentials in virtual_machines[].login_info.
    Port mappings are [internal_port, external_port] tuples.
    """
    host = instance.get("host_address", "")
    port_mappings = instance.get("port_mappings", [])

    # Extract login credentials from VM info
    username = "user"
    vms = instance.get("virtual_machines", [])
    if vms:
        login_info = vms[0].get("login_info", {})
        creds = login_info.get("UsernameAndPassword", {})
        username = creds.get("username", username)

    # Find SSH port mapping: each mapping is [internal_port, external_port]
    ssh_port = 22
    for mapping in port_mappings:
        if mapping[0] == 22:
            ssh_port = mapping[1]
            break

    return VMConnectionInfo(
        host=host,
        username=username,
        ssh_port=ssh_port,
        port_mappings=[(m[0], m[1]) for m in port_mappings],
        delete_info=delete_info,
    )


def _log_connection_info(instance):
    """Log SSH connection info based on instance networking.

    VMs provide login credentials in virtual_machines[].login_info.
    Port mappings are [internal_port, external_port] tuples.
    """
    host = instance.get("host_address")
    port_mappings = instance.get("port_mappings", [])

    # Extract login credentials from VM info
    username = "user"
    vms = instance.get("virtual_machines", [])
    if vms:
        login_info = vms[0].get("login_info", {})
        creds = login_info.get("UsernameAndPassword", {})
        username = creds.get("username", username)

    # Find SSH port mapping: each mapping is [internal_port, external_port]
    ssh_ext_port = None
    for mapping in port_mappings:
        if mapping[0] == 22:
            ssh_ext_port = mapping[1]
            break

    if ssh_ext_port and host:
        logger.info(f"Host:     {host}")
        logger.info(f"User:     {username}")
        logger.info(f"Connect:  ssh -p {ssh_ext_port} {username}@{host}")
        for internal, external in port_mappings:
            logger.info(f"  Port {internal} -> {host}:{external}")
    elif host:
        logger.info(f"Host:     {host}")
        logger.info(f"User:     {username}")
        logger.info(f"Connect:  ssh {username}@{host}")
    else:
        logger.warning("Warning: no host address found in instance info.")


async def create_instance(
    api_key,
    instance_type,
    ssh_key_path,
    image_url=None,
    ports=None,
    timeout=600,
    api_url=DEFAULT_API_URL,
    dry_run=False,
    fail_statuses=None,
    wait_ssh=False,
    ssh_private_key_path=None,
    billing_exempt=False,
    network=None,
):
    """Create a CloudRift VM instance.

    Args:
        image_url: VM image URL. If ``None``, auto-picks ROCm for ``mi*`` instance
            types and NVIDIA otherwise via :func:`select_image_url`.
        ssh_key_path: path to the SSH **public** key file.
        fail_statuses: optional set of statuses that trigger immediate failure
            (e.g. {"Inactive"}).
        wait_ssh: if True, wait for SSH connectivity after Active status.
        ssh_private_key_path: path to the SSH private key (needed for wait_ssh).

    Returns:
        VMConnectionInfo on success, None on failure.
        In dry-run mode, returns a VMConnectionInfo with placeholder values.
    """
    if image_url is None:
        image_url = select_image_url(instance_type)
        logger.info(f"Creating CloudRift instance (type={instance_type}, auto-selected image={image_url})...")
    else:
        logger.info(f"Creating CloudRift instance (type={instance_type}, image={image_url})...")

    ssh_key_path = os.path.expanduser(ssh_key_path)
    if dry_run and not os.path.exists(ssh_key_path):
        public_key = "dry-run-placeholder"
    else:
        with open(ssh_key_path) as f:
            public_key = f.read().strip()

    try:
        result = await _rent_instance(
            api_key,
            instance_type,
            [public_key],
            image_url=image_url,
            ports=ports,
            api_url=api_url,
            dry_run=dry_run,
            billing_exempt=billing_exempt,
            network=network,
        )
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        body = exc.response.text
        if code in (429, 503):
            raise CapacityExhausted(f"CloudRift rent returned {code} for {instance_type}") from exc
        # 400 "Instance X not found" means this rift-server doesn't carry this instance type
        # (per-datacenter availability). Treat as capacity so the orchestrator advances candidates.
        if code == 400 and "not found" in body.lower():
            raise CapacityExhausted(f"CloudRift rent returned 400 for {instance_type}: {body}") from exc
        if 400 <= code < 500:
            raise TerminalProvisionError(f"CloudRift rent returned {code}: {body}") from exc
        raise
    if dry_run:
        logger.info("[dry-run] Would wait for Active status, then print connection info.")
        return VMConnectionInfo(
            host="dry-run-host",
            username="riftuser",
            ssh_port=22222,
            delete_info=("cloudrift", "dry-run-id"),
        )

    instance_ids = result.get("instance_ids", [])
    if not instance_ids:
        # Rent succeeded HTTP-wise but allocated nothing — treat as no-capacity.
        raise CapacityExhausted(f"CloudRift rent returned no instance ID for {instance_type}")
    instance_id = instance_ids[0]
    logger.info(f"Instance rented (id={instance_id}). Waiting for Active status (timeout: {timeout}s)...")

    try:
        info = await wait_for_status(api_key, instance_id, "Active", timeout, api_url, fail_statuses=fail_statuses)
    except Exception:
        logger.warning(f"Terminating orphaned instance {instance_id} after exception during wait_for_status.")
        try:
            await _terminate_instance(api_key, instance_id, api_url)
        except Exception as exc:
            logger.error(f"Failed to terminate orphaned instance {instance_id}: {exc}")
        raise
    if info is None:
        logger.warning(f"Terminating orphaned instance {instance_id} after wait_for_status failure.")
        try:
            await _terminate_instance(api_key, instance_id, api_url)
        except Exception as exc:
            logger.error(f"Failed to terminate orphaned instance {instance_id}: {exc}")
        # wait_for_status returns None for fail-status (e.g. Inactive) and for timeout.
        # Either way, this candidate has effectively no usable capacity — advance.
        raise CapacityExhausted(
            f"CloudRift instance {instance_id} ({instance_type}) never reached Active (fail-status or timeout after {timeout}s)"
        )

    logger.info("Instance is Active.")
    logger.info(f"Instance details: {json.dumps(info, indent=2)}")
    conn = _extract_connection_info(info, delete_info=("cloudrift", instance_id))
    _log_connection_info(info)

    if wait_ssh and ssh_private_key_path:
        logger.info("Waiting for SSH connectivity...")
        await wait_for_ssh(conn.host, conn.username, conn.ssh_port, ssh_private_key_path)

    return conn


async def delete_instance(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Terminate a CloudRift instance."""
    logger.info(f"Terminating CloudRift instance '{instance_id}'...")

    await _terminate_instance(api_key, instance_id, api_url, dry_run)

    if not dry_run:
        logger.info("Instance terminated.")
    return True
