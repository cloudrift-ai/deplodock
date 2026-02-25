"""Provider-agnostic SSH readiness polling."""

import logging
import time

from deplodock.provisioning.ssh_transport import ssh_base_args

logger = logging.getLogger(__name__)


def wait_for_ssh(host, username, ssh_port, ssh_key_path, timeout=120, interval=5):
    """Poll SSH connectivity until success or timeout.

    Uses plain ssh (not gcloud) for provider-agnostic SSH readiness check.

    Returns:
        True if SSH connected, False on timeout.
    """
    import subprocess

    address = f"{username}@{host}" if username else host
    elapsed = 0
    while elapsed < timeout:
        args = ssh_base_args(address, ssh_key_path, ssh_port)
        # Add ConnectTimeout for fast failure during polling
        args.insert(-1, "-o")
        args.insert(-1, "ConnectTimeout=5")
        args.append("true")
        rc = subprocess.run(args, capture_output=True).returncode
        if rc == 0:
            return True
        time.sleep(interval)
        elapsed += interval

    logger.error(f"Timeout after {timeout}s waiting for SSH connectivity to {address}:{ssh_port}")
    return False
