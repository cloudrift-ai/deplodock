"""Collect system information from remote servers."""

import logging

logger = logging.getLogger(__name__)

SYSTEM_INFO_CMD = r"""
echo "=== HOSTNAME ==="
hostname

echo ""
echo "=== OS ==="
cat /etc/os-release 2>/dev/null || echo "N/A"

echo ""
echo "=== KERNEL ==="
uname -r

echo ""
echo "=== CPU INFORMATION ==="
lscpu 2>/dev/null || echo "N/A"

echo ""
echo "=== CPU COUNT ==="
nproc 2>/dev/null || echo "N/A"

echo ""
echo "=== MEMORY ==="
free -h 2>/dev/null || echo "N/A"

echo ""
echo "=== GPU INFORMATION ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,pstate,temperature.gpu,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "N/A"

echo ""
echo "=== GPU DETAILS ==="
nvidia-smi 2>/dev/null || echo "N/A"

echo ""
echo "=== DISK USAGE ==="
df -h 2>/dev/null || echo "N/A"

echo ""
echo "=== BLOCK DEVICES ==="
lsblk 2>/dev/null || echo "N/A"

echo ""
echo "=== UPTIME ==="
uptime 2>/dev/null || echo "N/A"

echo ""
echo "=== DOCKER VERSION ==="
docker --version 2>/dev/null || echo "N/A"

echo ""
echo "=== DOCKER INFO ==="
docker info --format \
  '{{.ServerVersion}} | OS: {{.OperatingSystem}} | Containers: {{.Containers}} | Images: {{.Images}}' \
  2>/dev/null || echo "N/A"
"""


async def collect_system_info(run_cmd) -> str:
    """Collect system information from a remote server via SSH.

    Returns the output wrapped in section delimiters, or empty string on failure.
    """
    rc, output, _ = await run_cmd(SYSTEM_INFO_CMD, stream=False, timeout=120)
    if rc != 0:
        logger.warning("Failed to collect system info")
        return ""
    return output
