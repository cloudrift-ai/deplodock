"""Teardown command: clean up VMs left running by --no-teardown."""

import json
import sys
from pathlib import Path

from deplodock.deploy.orchestrate import run_teardown
from deplodock.provisioning.cloud import delete_cloud_vm
from deplodock.provisioning.ssh_transport import make_run_cmd


def handle_teardown(args):
    """Handle the teardown command."""
    run_dir = Path(args.run_dir)
    instances_path = run_dir / "instances.json"
    ssh_key = args.ssh_key

    if not instances_path.exists():
        print(f"No instances.json found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    instances = json.loads(instances_path.read_text())
    if not instances:
        print("instances.json is empty — nothing to tear down.")
        return

    print(f"Tearing down {len(instances)} instance(s) from {run_dir}")
    print()

    errors = []
    for inst in instances:
        label = inst.get("group_label", "unknown")
        address = inst.get("address", "")
        ssh_port = inst.get("ssh_port", 22)
        provider = inst.get("provider")
        instance_id = inst.get("instance_id")

        print(f"[{label}] {address} ({provider}: {instance_id})")

        # Docker compose down
        if address:
            print(f"  Stopping containers on {address}...")
            run_cmd = make_run_cmd(address, ssh_key, ssh_port)
            run_teardown(run_cmd)

        # Delete VM
        if provider and instance_id:
            print(f"  Deleting VM ({provider}: {instance_id})...")
            try:
                if provider == "gcp":
                    zone = inst.get("zone")
                    if not zone:
                        print(f"  WARNING: missing zone for GCP instance {instance_id}", file=sys.stderr)
                        errors.append(label)
                        continue
                    delete_info = (provider, instance_id, zone)
                else:
                    delete_info = (provider, instance_id)
                delete_cloud_vm(delete_info)
                print("  VM deleted.")
            except Exception as e:
                print(f"  ERROR deleting VM: {e}", file=sys.stderr)
                errors.append(label)
                continue

    if errors:
        print(f"\nFailed to clean up {len(errors)} instance(s): {', '.join(errors)}")
        sys.exit(1)
    else:
        instances_path.unlink()
        print(f"\nAll instances cleaned up. Removed {instances_path}")


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
