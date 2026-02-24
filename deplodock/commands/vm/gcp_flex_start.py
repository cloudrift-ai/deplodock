"""GCP flex-start provider CLI handlers."""

import sys

from deplodock.provisioning.gcp_flex_start import (
    # Re-export business logic for backward compatibility
    create_instance,
    delete_instance,
)

# ── CLI handlers ───────────────────────────────────────────────────


def handle_create(args):
    """CLI handler for 'vm create gcp-flex-start'."""
    conn = create_instance(
        instance=args.instance,
        zone=args.zone,
        machine_type=args.machine_type,
        provisioning_model=args.provisioning_model,
        max_run_duration=args.max_run_duration,
        request_valid_for_duration=args.request_valid_for_duration,
        termination_action=args.termination_action,
        image_family=args.image_family,
        image_project=args.image_project,
        extra_gcloud_args=args.gcloud_args,
        timeout=args.timeout,
        wait_ssh=args.wait_ssh,
        wait_ssh_timeout=args.wait_ssh_timeout,
        ssh_gateway=args.ssh_gateway,
        dry_run=args.dry_run,
    )
    if conn is None:
        sys.exit(1)


def handle_delete(args):
    """CLI handler for 'vm delete gcp-flex-start'."""
    success = delete_instance(
        instance=args.instance,
        zone=args.zone,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


# ── Registration ───────────────────────────────────────────────────


def register_create_target(subparsers):
    """Register the gcp-flex-start provider under 'vm create'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Create a GCP GPU VM")
    parser.add_argument("--instance", required=True, help="GCP instance name")
    parser.add_argument("--zone", required=True, help="GCP zone (e.g. us-central1-a)")
    parser.add_argument("--machine-type", required=True, help="Machine type (e.g. a2-highgpu-1g)")
    parser.add_argument(
        "--provisioning-model",
        default="FLEX_START",
        choices=["FLEX_START", "SPOT", "STANDARD"],
        help="Provisioning model (default: FLEX_START)",
    )
    parser.add_argument("--max-run-duration", default="7d", help="Max VM run time (default: 7d)")
    parser.add_argument("--request-valid-for-duration", default="2h", help="How long to wait for capacity (default: 2h)")
    parser.add_argument(
        "--termination-action",
        default="DELETE",
        choices=["STOP", "DELETE"],
        help="Action when max-run-duration expires (default: DELETE)",
    )
    parser.add_argument("--image-family", default="debian-12", help="Boot disk image family (default: debian-12)")
    parser.add_argument("--image-project", default="debian-cloud", help="Boot disk image project (default: debian-cloud)")
    parser.add_argument("--gcloud-args", default=None, help="Extra args passed to gcloud compute instances create")
    parser.add_argument("--timeout", type=int, default=14400, help="How long to poll for RUNNING status in seconds (default: 14400)")
    parser.add_argument("--wait-ssh", action="store_true", help="Wait for SSH connectivity after VM is RUNNING")
    parser.add_argument("--wait-ssh-timeout", type=int, default=300, help="SSH wait timeout in seconds (default: 300)")
    parser.add_argument("--ssh-gateway", default=None, help="SSH gateway host for ProxyJump (e.g. gcp-ssh-gateway)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_create)


def register_delete_target(subparsers):
    """Register the gcp-flex-start provider under 'vm delete'."""
    parser = subparsers.add_parser("gcp-flex-start", help="Delete a GCP flex-start GPU VM")
    parser.add_argument("--instance", required=True, help="GCP instance name")
    parser.add_argument("--zone", required=True, help="GCP zone (e.g. us-central1-a)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_delete)
