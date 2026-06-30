"""Teardown command: clean up VMs left running by --no-teardown."""

import asyncio
import json
import logging
import sys
from pathlib import Path

from deplodock.deploy.orchestrate import run_teardown
from deplodock.provisioning.cloud import delete_cloud_vm
from deplodock.provisioning.ssh_transport import make_run_cmd

logger = logging.getLogger(__name__)


def handle_teardown(args):
    """Handle the teardown command."""
    asyncio.run(_handle_teardown(args))


async def _handle_teardown(args):
    run_dir = Path(args.run_dir)
    instances_path = run_dir / "instances.json"
    ssh_key = args.ssh_key

    if not instances_path.exists():
        logger.error(f"No instances.json found in {run_dir}")
        sys.exit(1)

    instances = json.loads(instances_path.read_text())
    if not instances:
        logger.info("instances.json is empty — nothing to tear down.")
        return

    logger.info(f"Tearing down {len(instances)} instance(s) from {run_dir}")
    logger.info("")

    errors = []
    for inst in instances:
        label = inst.get("group_label", "unknown")
        address = inst.get("address", "")
        ssh_port = inst.get("ssh_port", 22)
        provider = inst.get("provider")
        instance_id = inst.get("instance_id")

        logger.info(f"[{label}] {address} ({provider}: {instance_id})")

        # Docker compose down
        if address:
            logger.info(f"  Stopping containers on {address}...")
            run_cmd = make_run_cmd(address, ssh_key, ssh_port)
            await run_teardown(run_cmd)

        # Delete VM
        if provider and instance_id:
            logger.info(f"  Deleting VM ({provider}: {instance_id})...")
            try:
                if provider == "gcp":
                    zone = inst.get("zone")
                    if not zone:
                        logger.error(f"  WARNING: missing zone for GCP instance {instance_id}")
                        errors.append(label)
                        continue
                    delete_info = (provider, instance_id, zone)
                else:
                    delete_info = (provider, instance_id)
                await delete_cloud_vm(delete_info)
                logger.info("  VM deleted.")
            except Exception as e:
                logger.error(f"  ERROR deleting VM: {e}")
                errors.append(label)
                continue

    if errors:
        logger.info(f"\nFailed to clean up {len(errors)} instance(s): {', '.join(errors)}")
        sys.exit(1)
    else:
        instances_path.unlink()
        logger.info(f"\nAll instances cleaned up. Removed {instances_path}")


def register_teardown_command(subparsers):
    """Register the teardown subcommand."""
    parser = subparsers.add_parser(
        "teardown",
        help="Tear down VMs left running by 'bench --no-teardown'",
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing instances.json",
    )
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="SSH private key path (default: ~/.ssh/id_ed25519)",
    )
    parser.set_defaults(func=handle_teardown)
