"""CloudRift provider: create/delete GPU VMs via the CloudRift REST API."""

import json
import os
import time

import requests

from deplodock.provisioning.types import VMConnectionInfo
from deplodock.provisioning.ssh import wait_for_ssh


DEFAULT_API_URL = "https://api.cloudrift.ai"
DEFAULT_IMAGE_URL = "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpu-580-129-20251015-183936.img"
DEFAULT_CLOUDINIT_URL = "https://storage.googleapis.com/cloudrift-vm-disks/cloudinit/ubuntu-base.cloudinit"
API_VERSION = "~upcoming"


# ── API helpers ───────────────────────────────────────────────────


def _api_request(method, path, data, api_key, api_url=DEFAULT_API_URL, dry_run=False):
    """Make an authenticated CloudRift API request.

    Wraps *data* in the versioned envelope ``{"version": ..., "data": ...}``.

    Returns:
        Parsed JSON response ``data`` dict, or ``None`` in dry-run mode.
    """
    url = f"{api_url}{path}"
    payload = {"version": API_VERSION, "data": data}

    if dry_run:
        print(f"[dry-run] {method} {url}")
        print(f"[dry-run] payload: {json.dumps(payload, indent=2)}")
        return None

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    resp = requests.request(method, url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", resp.json())


def _rent_instance(api_key, instance_type, ssh_public_keys,
                   image_url=DEFAULT_IMAGE_URL, cloudinit_url=DEFAULT_CLOUDINIT_URL,
                   ports=None, api_url=DEFAULT_API_URL, dry_run=False):
    """Rent a new CloudRift VM instance.

    POST /api/v1/instances/rent

    Args:
        ssh_public_keys: list of public key strings (e.g. ["ssh-ed25519 AAAA..."])
    """
    data = {
        "selector": {
            "ByInstanceTypeAndLocation": {
                "instance_type": instance_type,
            },
        },
        "config": {
            "VirtualMachine": {
                "ssh_key": {"PublicKeys": ssh_public_keys},
                "image_url": image_url,
                "cloudinit_url": cloudinit_url,
            },
        },
        "with_public_ip": True,
    }
    if ports:
        data["ports"] = [str(p) for p in ports]
    return _api_request("POST", "/api/v1/instances/rent", data, api_key, api_url, dry_run)


def _terminate_instance(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Terminate a CloudRift instance.

    POST /api/v1/instances/terminate with ById selector.
    """
    data = {"selector": {"ById": [instance_id]}}
    return _api_request("POST", "/api/v1/instances/terminate", data, api_key, api_url, dry_run)


def _get_instance_info(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Get info for a single instance by ID.

    POST /api/v1/instances/list with ById selector.
    Returns the instance dict or None.
    """
    data = {"selector": {"ById": [instance_id]}}
    result = _api_request("POST", "/api/v1/instances/list", data, api_key, api_url, dry_run)
    if result is None:  # dry-run
        return None
    instances = result.get("instances", [])
    return instances[0] if instances else None


def _list_ssh_keys(api_key, api_url=DEFAULT_API_URL, dry_run=False):
    """List registered SSH keys.

    POST /api/v1/ssh-keys/list
    """
    return _api_request("POST", "/api/v1/ssh-keys/list", {}, api_key, api_url, dry_run)


def _add_ssh_key(api_key, name, public_key, api_url=DEFAULT_API_URL, dry_run=False):
    """Register a new SSH key.

    POST /api/v1/ssh-keys/add
    """
    data = {"name": name, "public_key": public_key}
    return _api_request("POST", "/api/v1/ssh-keys/add", data, api_key, api_url, dry_run)


# ── Core logic ─────────────────────────────────────────────────────


def _ensure_ssh_key(api_key, ssh_key_path, api_url=DEFAULT_API_URL, dry_run=False):
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

    result = _list_ssh_keys(api_key, api_url, dry_run)
    if result is not None:
        for key in result.get("keys", []):
            if key.get("public_key", "").strip() == public_key:
                print(f"SSH key already registered (id={key['id']}).")
                return key["id"]

    # Register new key
    key_name = os.path.basename(ssh_key_path)
    print(f"Registering SSH key '{key_name}' on CloudRift...")
    add_result = _add_ssh_key(api_key, key_name, public_key, api_url, dry_run)
    if dry_run:
        return "dry-run-key-id"
    key_id = add_result["ssh_key"]["id"]
    print(f"SSH key registered (id={key_id}).")
    return key_id


def wait_for_status(api_key, instance_id, target_status, timeout,
                    api_url=DEFAULT_API_URL, interval=10, dry_run=False,
                    fail_statuses=None):
    """Poll instance status until it matches *target_status* or timeout.

    Args:
        fail_statuses: optional set of status strings that trigger immediate failure.

    Returns:
        The instance dict if target status reached, None on timeout or fail status.
    """
    if dry_run:
        print(f"[dry-run] Poll every {interval}s (up to {timeout}s) for status '{target_status}'")
        return {"status": target_status}

    fail_statuses = fail_statuses or set()
    elapsed = 0
    status = None
    while elapsed < timeout:
        info = _get_instance_info(api_key, instance_id, api_url)
        if info is None:
            print(f"Warning: instance {instance_id} not found.", file=__import__("sys").stderr)
            time.sleep(interval)
            elapsed += interval
            continue
        status = info.get("status")
        if status == target_status:
            return info
        if status in fail_statuses:
            print(f"Instance {instance_id} reached fail status '{status}'", file=__import__("sys").stderr)
            return None
        time.sleep(interval)
        elapsed += interval

    print(f"Timeout after {timeout}s waiting for status '{target_status}' (last: '{status}')", file=__import__("sys").stderr)
    return None


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


def _print_connection_info(instance):
    """Print SSH connection info based on instance networking.

    VMs provide login credentials in virtual_machines[].login_info.
    Port mappings are [internal_port, external_port] tuples.
    """
    host = instance.get("host_address")
    port_mappings = instance.get("port_mappings", [])

    # Extract login credentials from VM info
    username = "user"
    password = None
    vms = instance.get("virtual_machines", [])
    if vms:
        login_info = vms[0].get("login_info", {})
        creds = login_info.get("UsernameAndPassword", {})
        username = creds.get("username", username)
        password = creds.get("password")

    # Find SSH port mapping: each mapping is [internal_port, external_port]
    ssh_ext_port = None
    for mapping in port_mappings:
        if mapping[0] == 22:
            ssh_ext_port = mapping[1]
            break

    if ssh_ext_port and host:
        print(f"Host:     {host}")
        print(f"User:     {username}")
        if password:
            print(f"Password: {password}")
        print(f"Connect:  ssh -p {ssh_ext_port} {username}@{host}")
        for internal, external in port_mappings:
            print(f"  Port {internal} -> {host}:{external}")
    elif host:
        print(f"Host:     {host}")
        print(f"User:     {username}")
        if password:
            print(f"Password: {password}")
        print(f"Connect:  ssh {username}@{host}")
    else:
        print("Warning: no host address found in instance info.")


def create_instance(api_key, instance_type, ssh_key_path, image_url=DEFAULT_IMAGE_URL,
                    ports=None, timeout=600, api_url=DEFAULT_API_URL, dry_run=False,
                    fail_statuses=None, wait_ssh=False, ssh_private_key_path=None):
    """Create a CloudRift VM instance.

    Args:
        ssh_key_path: path to the SSH **public** key file.
        fail_statuses: optional set of statuses that trigger immediate failure
            (e.g. {"Inactive"}).
        wait_ssh: if True, wait for SSH connectivity after Active status.
        ssh_private_key_path: path to the SSH private key (needed for wait_ssh).

    Returns:
        VMConnectionInfo on success, None on failure.
        In dry-run mode, returns a VMConnectionInfo with placeholder values.
    """
    print(f"Creating CloudRift instance (type={instance_type})...")

    ssh_key_path = os.path.expanduser(ssh_key_path)
    if dry_run and not os.path.exists(ssh_key_path):
        public_key = "dry-run-placeholder"
    else:
        with open(ssh_key_path) as f:
            public_key = f.read().strip()

    result = _rent_instance(api_key, instance_type, [public_key], image_url=image_url,
                            ports=ports, api_url=api_url, dry_run=dry_run)
    if dry_run:
        print("[dry-run] Would wait for Active status, then print connection info.")
        return VMConnectionInfo(
            host="dry-run-host", username="riftuser", ssh_port=22222,
            delete_info=("cloudrift", "dry-run-id"),
        )

    instance_ids = result.get("instance_ids", [])
    if not instance_ids:
        print("Error: no instance ID returned from rent API.", file=__import__("sys").stderr)
        return None
    instance_id = instance_ids[0]
    print(f"Instance rented (id={instance_id}). Waiting for Active status (timeout: {timeout}s)...")

    info = wait_for_status(api_key, instance_id, "Active", timeout, api_url,
                           fail_statuses=fail_statuses)
    if info is None:
        return None

    print("Instance is Active.")
    conn = _extract_connection_info(info, delete_info=("cloudrift", instance_id))
    _print_connection_info(info)

    if wait_ssh and ssh_private_key_path:
        print(f"Waiting for SSH connectivity...")
        wait_for_ssh(conn.host, conn.username, conn.ssh_port, ssh_private_key_path)

    return conn


def delete_instance(api_key, instance_id, api_url=DEFAULT_API_URL, dry_run=False):
    """Terminate a CloudRift instance."""
    print(f"Terminating CloudRift instance '{instance_id}'...")

    _terminate_instance(api_key, instance_id, api_url, dry_run)

    if not dry_run:
        print("Instance terminated.")
    return True
