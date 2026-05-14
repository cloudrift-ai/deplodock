"""GPU-based `vm create` handler.

Provisions a cloud VM by GPU name rather than by exact instance type.
Goes through :func:`deplodock.provisioning.cloud.provision_cloud_vm` so it
shares the same candidate iteration, retry, fallback, and orphan-cleanup
behavior as ``deploy cloud`` and ``bench``.

Unlike the provider-specific ``vm create cloudrift`` / ``vm create gcp``
subcommands (which take an exact ``--instance-type`` / ``--machine-type``
and do a single-shot create), this handler enumerates candidates from the
hardware table and tries them in preference order.
"""

import asyncio
import logging
import os
import sys

from deplodock.benchmark.config import load_config
from deplodock.provisioning.cloud import provision_cloud_vm
from deplodock.provisioning.errors import CapacityExhausted, TerminalProvisionError

logger = logging.getLogger(__name__)


def handle_create(args):
    """CLI handler for 'vm create gpu'."""
    asyncio.run(_handle_create(args))


async def _handle_create(args):
    ssh_key = os.path.expanduser(args.ssh_key)

    providers_config = None
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        providers_config = config.get("providers")
    if args.billing_exempt or args.network:
        providers_config = providers_config or {}
        # `cloudrift:` in config.yaml can parse to None when it has only commented children.
        if providers_config.get("cloudrift") is None:
            providers_config["cloudrift"] = {}
        if args.billing_exempt:
            providers_config["cloudrift"]["billing_exempt"] = True
        if args.network:
            providers_config["cloudrift"]["network"] = args.network

    try:
        conn = await provision_cloud_vm(
            gpu_name=args.gpu,
            gpu_count=args.gpu_count,
            ssh_key=ssh_key,
            providers_config=providers_config,
            server_name=args.name,
            dry_run=args.dry_run,
            provider=args.provider,
        )
    except (CapacityExhausted, TerminalProvisionError) as exc:
        logger.error(f"{exc}")
        sys.exit(1)

    if conn is None:
        logger.error("VM provisioning failed: all candidates exhausted.")
        sys.exit(1)

    logger.info(f"VM ready at {conn.address}:{conn.ssh_port}")


def register_create_target(subparsers):
    """Register the GPU-based provisioning target under 'vm create'."""
    parser = subparsers.add_parser(
        "gpu",
        help="Create a VM by GPU name (with cross-candidate fallback)",
    )
    parser.add_argument("--gpu", required=True, help="GPU name from hardware table (e.g. 'NVIDIA H200 141GB')")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--name", default=None, help="Server name prefix used in the VM hostname")
    parser.add_argument(
        "--provider",
        choices=["cloudrift", "gcp"],
        default=None,
        help="Restrict candidates to one provider (default: hardware-table preference order)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml for provider-specific defaults (default: config.yaml)",
    )
    parser.add_argument("--billing-exempt", action="store_true", help="Skip billing for CloudRift (admin-only)")
    parser.add_argument(
        "--network",
        default=None,
        help="CloudRift network name (must exist in target datacenter; default: provider picks a public network)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.set_defaults(func=handle_create)
