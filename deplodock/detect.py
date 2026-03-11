"""GPU detection via PCI sysfs device IDs."""

import asyncio
import logging

from deplodock.provisioning.ssh_transport import ssh_base_args

logger = logging.getLogger(__name__)

NVIDIA_VENDOR = "0x10de"
AMD_VENDOR = "0x1002"

# PCI device ID (hex, no prefix) -> GPU name
# Source: PCI ID database (vendor 10de for NVIDIA, 1002 for AMD)
GPU_PCI_DEVICE_IDS: dict[str, str] = {
    # NVIDIA GeForce RTX 4090
    "2684": "NVIDIA GeForce RTX 4090",
    # NVIDIA GeForce RTX 5090
    "2b85": "NVIDIA GeForce RTX 5090",
    # NVIDIA RTX PRO 6000 Blackwell Workstation Edition
    "2ba0": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    # NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
    "2ba2": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    # NVIDIA RTX PRO 6000 Blackwell Server Edition
    "2ba4": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    # NVIDIA L40S
    "26b9": "NVIDIA L40S",
    # NVIDIA H100 80GB (SXM/PCIe)
    "2330": "NVIDIA H100 80GB",
    "2331": "NVIDIA H100 80GB",
    # NVIDIA H200 141GB
    "2335": "NVIDIA H200 141GB",
    # NVIDIA B200
    "2900": "NVIDIA B200",
    "2901": "NVIDIA B200",
    # NVIDIA A100 40GB (PCIe)
    "20f1": "NVIDIA A100 40GB",
    # NVIDIA A100 80GB (SXM/PCIe)
    "20b2": "NVIDIA A100 80GB",
    "20b5": "NVIDIA A100 80GB",
    # AMD Instinct MI350X
    "75b0": "AMD Instinct MI350X",
}

_SYSFS_SCAN_CMD = (
    "for d in /sys/bus/pci/devices/*/; do "
    'v=$(cat "$d/vendor" 2>/dev/null); '
    'p=$(cat "$d/device" 2>/dev/null); '
    '[ -n "$v" ] && echo "$v $p"; '
    "done"
)


def _parse_sysfs_output(output: str) -> tuple[str, int]:
    """Parse sysfs vendor/device lines and return (gpu_name, count)."""
    gpu_vendors = {NVIDIA_VENDOR, AMD_VENDOR}
    found: dict[str, int] = {}

    for line in output.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        vendor, device = parts
        if vendor not in gpu_vendors:
            continue
        device_id = device.replace("0x", "")
        gpu_name = GPU_PCI_DEVICE_IDS.get(device_id)
        if gpu_name is not None:
            found[gpu_name] = found.get(gpu_name, 0) + 1

    if not found:
        raise RuntimeError("No supported GPUs detected via PCI sysfs")

    if len(found) > 1:
        names = ", ".join(sorted(found.keys()))
        raise RuntimeError(f"Mixed GPU types detected: {names}. All GPUs must be the same type.")

    gpu_name = next(iter(found))
    count = found[gpu_name]
    return gpu_name, count


def detect_local_gpus() -> tuple[str, int]:
    """Detect local GPUs by scanning PCI sysfs. Returns (gpu_name, count)."""
    import subprocess

    result = subprocess.run(
        ["bash", "-c", _SYSFS_SCAN_CMD],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to scan PCI devices: {result.stderr}")

    return _parse_sysfs_output(result.stdout)


async def detect_remote_gpus(server: str, ssh_key: str, ssh_port: int) -> tuple[str, int]:
    """Detect GPUs on a remote server via SSH. Returns (gpu_name, count)."""
    args = ssh_base_args(server, ssh_key, ssh_port)
    args.append(_SYSFS_SCAN_CMD)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        raise RuntimeError(f"Failed to scan PCI devices on {server}: {stderr}")

    return _parse_sysfs_output(stdout_bytes.decode())
