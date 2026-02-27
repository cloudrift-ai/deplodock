#!/usr/bin/env python3
"""Stress-test CloudRift VM SSH provisioning.

Allocates RTX 5090 VMs one at a time, tests SSH connectivity with verbose
logging, then terminates. Repeats N times to surface intermittent failures.

Usage:
    ./venv/bin/python scripts/test_cloudrift_ssh.py [--iterations 20] [--ssh-key ~/.ssh/id_ed25519]

Requires:
    CLOUDRIFT_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time

import httpx

# ── Config ────────────────────────────────────────────────────────

API_URL = "https://api.cloudrift.ai"
API_VERSION = "~upcoming"
INSTANCE_TYPE = "rtx59-7-50-400-ec.1"
IMAGE_URL = "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpu-580-129-20251015-183936.img"
CLOUDINIT_URL = "https://storage.googleapis.com/cloudrift-vm-disks/cloudinit/ubuntu-base.cloudinit"
PORTS = ["22", "8000", "8080"]
ACTIVE_TIMEOUT = 300
SSH_TIMEOUT = 120
SSH_POLL_INTERVAL = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_ssh")


# ── API helpers ───────────────────────────────────────────────────


async def api_request(method, path, data, api_key):
    url = f"{API_URL}{path}"
    payload = {"version": API_VERSION, "data": data}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", resp.json())


async def rent_instance(api_key, public_key):
    data = {
        "selector": {"ByInstanceTypeAndLocation": {"instance_type": INSTANCE_TYPE}},
        "config": {
            "VirtualMachine": {
                "ssh_key": {"PublicKeys": [public_key]},
                "image_url": IMAGE_URL,
                "cloudinit_url": CLOUDINIT_URL,
            },
        },
        "with_public_ip": True,
        "ports": PORTS,
    }
    return await api_request("POST", "/api/v1/instances/rent", data, api_key)


async def get_instance(api_key, instance_id):
    data = {"selector": {"ById": [instance_id]}}
    result = await api_request("POST", "/api/v1/instances/list", data, api_key)
    instances = result.get("instances", [])
    return instances[0] if instances else None


async def terminate_instance(api_key, instance_id):
    data = {"selector": {"ById": [instance_id]}}
    return await api_request("POST", "/api/v1/instances/terminate", data, api_key)


async def wait_for_active(api_key, instance_id):
    elapsed = 0
    while elapsed < ACTIVE_TIMEOUT:
        info = await get_instance(api_key, instance_id)
        if info is None:
            log.warning(f"  Instance {instance_id} not found")
        else:
            status = info.get("status")
            if status == "Active":
                return info
            log.info(f"  Status: {status} ({elapsed}s)")
        await asyncio.sleep(10)
        elapsed += 10
    return None


def extract_connection(info):
    host = info.get("host_address", "")
    username = "user"
    vms = info.get("virtual_machines", [])
    if vms:
        login_info = vms[0].get("login_info", {})
        creds = login_info.get("UsernameAndPassword", {})
        username = creds.get("username", username)
    ssh_port = 22
    for mapping in info.get("port_mappings", []):
        if mapping[0] == 22:
            ssh_port = mapping[1]
            break
    return host, username, ssh_port


# ── SSH testing ───────────────────────────────────────────────────


def ssh_verbose(host, username, ssh_port, ssh_key, command="true"):
    """Run an SSH command with full verbose output. Returns (returncode, stdout, stderr)."""
    args = [
        "ssh",
        "-vvv",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-i",
        ssh_key,
    ]
    if ssh_port != 22:
        args += ["-p", str(ssh_port)]
    args += [f"{username}@{host}", command]

    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


async def wait_and_test_ssh(host, username, ssh_port, ssh_key):
    """Poll SSH with verbose logging on every attempt."""
    elapsed = 0
    attempt = 0
    while elapsed < SSH_TIMEOUT:
        attempt += 1
        log.info(f"  SSH attempt {attempt} ({elapsed}s elapsed)")

        rc, stdout, stderr = ssh_verbose(host, username, ssh_port, ssh_key)

        if rc == 0:
            log.info(f"  SSH SUCCESS on attempt {attempt}")
            # Now do a second check to see if it's stable
            await asyncio.sleep(2)
            rc2, stdout2, stderr2 = ssh_verbose(host, username, ssh_port, ssh_key, "echo hello")
            if rc2 == 0:
                log.info("  SSH stable (second check passed)")
                return True, attempt, None
            else:
                log.warning("  SSH UNSTABLE: first attempt passed but second failed!")
                log.warning(f"  Second attempt stderr:\n{stderr2}")
                return False, attempt, f"unstable: {stderr2}"
        else:
            # Extract key lines from verbose output
            error_lines = []
            for line in stderr.splitlines():
                line_lower = line.lower()
                if any(
                    kw in line_lower
                    for kw in [
                        "permission denied",
                        "connection refused",
                        "connection reset",
                        "no route",
                        "timed out",
                        "authentications that can continue",
                        "next authentication method",
                        "offering public key",
                        "server accepts key",
                        "authentication succeeded",
                        "send packet: type 50",  # userauth request
                        "receive packet: type 5",  # userauth response
                        "host key ",
                        "identity file",
                    ]
                ):
                    error_lines.append(line.strip())

            if error_lines:
                log.info(f"  SSH failed (rc={rc}), key debug lines:")
                for el in error_lines:
                    log.info(f"    {el}")
            else:
                log.info(f"  SSH failed (rc={rc}), last 5 stderr lines:")
                for el in stderr.strip().splitlines()[-5:]:
                    log.info(f"    {el.strip()}")

        await asyncio.sleep(SSH_POLL_INTERVAL)
        elapsed += SSH_POLL_INTERVAL

    # Final verbose dump on timeout
    log.error(f"  SSH TIMEOUT after {SSH_TIMEOUT}s")
    rc, stdout, stderr = ssh_verbose(host, username, ssh_port, ssh_key)
    log.error(f"  Final attempt stderr:\n{stderr}")
    return False, attempt, stderr


# ── Main loop ─────────────────────────────────────────────────────


async def run_one(api_key, public_key, ssh_key, iteration):
    """Allocate one VM, test SSH, terminate. Returns result dict."""
    log.info(f"{'=' * 60}")
    log.info(f"Iteration {iteration}")
    log.info(f"{'=' * 60}")

    instance_id = None
    result = {
        "iteration": iteration,
        "instance_id": None,
        "host": None,
        "ssh_port": None,
        "rent_ok": False,
        "active_ok": False,
        "ssh_ok": False,
        "ssh_attempts": 0,
        "error": None,
        "duration_s": 0,
    }

    t0 = time.monotonic()
    try:
        # Rent
        log.info("  Renting instance...")
        rent_result = await rent_instance(api_key, public_key)
        instance_ids = rent_result.get("instance_ids", [])
        if not instance_ids:
            result["error"] = "no instance_id returned"
            log.error(f"  {result['error']}")
            return result
        instance_id = instance_ids[0]
        result["instance_id"] = instance_id
        result["rent_ok"] = True
        log.info(f"  Instance rented: {instance_id}")

        # Wait for Active
        log.info("  Waiting for Active...")
        info = await wait_for_active(api_key, instance_id)
        if info is None:
            result["error"] = "timeout waiting for Active"
            log.error(f"  {result['error']}")
            return result
        result["active_ok"] = True

        host, username, ssh_port = extract_connection(info)
        result["host"] = host
        result["ssh_port"] = ssh_port
        log.info(f"  Active: {username}@{host}:{ssh_port}")
        log.info(f"  Instance info: {json.dumps(info, indent=2, default=str)}")

        # Test SSH
        ssh_ok, attempts, error = await wait_and_test_ssh(host, username, ssh_port, ssh_key)
        result["ssh_ok"] = ssh_ok
        result["ssh_attempts"] = attempts
        if error:
            result["error"] = error[:500]

    except Exception as e:
        result["error"] = str(e)[:500]
        log.exception(f"  Exception: {e}")
    finally:
        result["duration_s"] = round(time.monotonic() - t0, 1)
        # Always terminate
        if instance_id:
            log.info(f"  Terminating {instance_id}...")
            try:
                await terminate_instance(api_key, instance_id)
                log.info("  Terminated.")
            except Exception as e:
                log.error(f"  Failed to terminate: {e}")

    status = "PASS" if result["ssh_ok"] else "FAIL"
    log.info(f"  Result: {status} (duration={result['duration_s']}s, attempts={result['ssh_attempts']})")
    return result


async def main():
    parser = argparse.ArgumentParser(description="Stress-test CloudRift SSH provisioning")
    parser.add_argument("--iterations", "-n", type=int, default=20)
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    args = parser.parse_args()

    api_key = os.environ.get("CLOUDRIFT_API_KEY")
    if not api_key:
        print("ERROR: CLOUDRIFT_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    ssh_key = os.path.expanduser(args.ssh_key)
    pub_key_path = f"{ssh_key}.pub"

    if not os.path.exists(ssh_key):
        print(f"ERROR: SSH private key not found: {ssh_key}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(pub_key_path):
        print(f"ERROR: SSH public key not found: {pub_key_path}", file=sys.stderr)
        sys.exit(1)

    with open(pub_key_path) as f:
        public_key = f.read().strip()

    log.info(f"SSH key: {ssh_key}")
    log.info(f"Public key: {public_key[:50]}...")
    log.info(f"Instance type: {INSTANCE_TYPE}")
    log.info(f"Iterations: {args.iterations}")
    log.info("")

    results = []
    for i in range(1, args.iterations + 1):
        r = await run_one(api_key, public_key, ssh_key, i)
        results.append(r)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    total = len(results)
    passed = sum(1 for r in results if r["ssh_ok"])
    failed = sum(1 for r in results if not r["ssh_ok"])
    log.info(f"Total: {total}  Passed: {passed}  Failed: {failed}  ({100 * passed / total:.0f}% success)")
    log.info("")

    if failed:
        log.info("Failed iterations:")
        for r in results:
            if not r["ssh_ok"]:
                log.info(f"  #{r['iteration']}: host={r['host']}:{r['ssh_port']} instance={r['instance_id']} error={r['error']}")

    # Dump full results as JSON
    results_path = "cloudrift_ssh_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
