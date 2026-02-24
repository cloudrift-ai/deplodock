"""Provider-agnostic SSH readiness polling."""

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
