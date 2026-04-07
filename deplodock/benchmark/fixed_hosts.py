"""Fixed-host mode for `deplodock bench`.

Resolves a user-provided pool of pre-allocated hosts (`--local`, `--ssh`)
into AllocatedHost records with detected GPU type/count, and validates
that every planned execution group can run on at least one of them.
"""

import logging
import os
from dataclasses import dataclass

from deplodock.detect import detect_local_gpus, detect_remote_gpus
from deplodock.provisioning.ssh_target import parse_ssh_target
from deplodock.provisioning.types import VMConnectionInfo

__all__ = [
    "AllocatedHost",
    "parse_ssh_target",
    "resolve_fixed_hosts",
    "validate_hosts_cover_groups",
]

logger = logging.getLogger(__name__)


@dataclass
class AllocatedHost:
    """A pre-allocated host the user supplied via --local / --ssh."""

    conn: VMConnectionInfo
    gpu_name: str | None
    gpu_count: int

    def satisfies(self, required_gpu: str, required_count: int) -> bool:
        if self.gpu_name is None:
            return False
        return self.gpu_name == required_gpu and self.gpu_count >= required_count


async def resolve_fixed_hosts(
    use_local: bool,
    ssh_targets: list[str],
    ssh_key: str,
    dry_run: bool = False,
) -> list[AllocatedHost]:
    """Build AllocatedHost records from --local / --ssh CLI args.

    In dry-run mode, GPU detection is skipped and hosts get gpu_name=None /
    gpu_count=0; the dispatcher then routes groups round-robin without
    validating GPU compatibility.
    """
    hosts: list[AllocatedHost] = []

    if use_local:
        username = os.environ.get("USER", "deploy")
        conn = VMConnectionInfo(host="127.0.0.1", username=username, ssh_port=22)
        if dry_run:
            hosts.append(AllocatedHost(conn=conn, gpu_name=None, gpu_count=0))
        else:
            try:
                gpu_name, gpu_count = detect_local_gpus()
            except Exception as e:
                raise RuntimeError(f"Failed to detect local GPUs for --local: {e}") from e
            logger.info(f"--local: detected {gpu_name} x{gpu_count}")
            hosts.append(AllocatedHost(conn=conn, gpu_name=gpu_name, gpu_count=gpu_count))

    for target in ssh_targets:
        user, host, port = parse_ssh_target(target)
        conn = VMConnectionInfo(host=host, username=user, ssh_port=port)
        if dry_run:
            hosts.append(AllocatedHost(conn=conn, gpu_name=None, gpu_count=0))
            continue
        try:
            gpu_name, gpu_count = await detect_remote_gpus(conn.address, ssh_key, port)
        except Exception as e:
            raise RuntimeError(f"Failed to detect GPUs on {target}: {e}") from e
        logger.info(f"--ssh {target}: detected {gpu_name} x{gpu_count}")
        hosts.append(AllocatedHost(conn=conn, gpu_name=gpu_name, gpu_count=gpu_count))

    return hosts


def validate_hosts_cover_groups(hosts: list[AllocatedHost], groups) -> None:
    """Ensure every group has at least one compatible host.

    Raises RuntimeError listing the unsatisfied groups otherwise.
    """
    unsatisfied = []
    for g in groups:
        if not any(h.satisfies(g.gpu_name, g.gpu_count) for h in hosts):
            unsatisfied.append(f"{g.label} ({g.gpu_name} x{g.gpu_count})")
    if unsatisfied:
        host_summary = ", ".join(f"{h.conn.address} [{h.gpu_name} x{h.gpu_count}]" for h in hosts)
        raise RuntimeError(
            "No supplied host can satisfy the following execution group(s): "
            + "; ".join(unsatisfied)
            + f". Available hosts: {host_summary}"
        )
