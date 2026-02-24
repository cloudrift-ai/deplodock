"""Cloud deploy target CLI handler."""

import os
import sys

from deplodock.deploy import (
    DeployParams,
    deploy as deploy_entry,
    load_recipe,
)
from deplodock.provisioning.cloud import (
    resolve_vm_spec,
    provision_cloud_vm,
    delete_cloud_vm,
)
from deplodock.provisioning.remote import provision_remote


# ── CLI handler ────────────────────────────────────────────────────


def handle_cloud(args):
    """CLI handler for 'deploy cloud'."""
    config = load_recipe(args.recipe, variant=args.variant)

    gpu_name = config.get('gpu')
    gpu_count = config.get('gpu_count', 1)
    if not gpu_name:
        print("Error: recipe must have a 'gpu' field (use a variant with GPU info).", file=sys.stderr)
        sys.exit(1)

    ssh_key = os.path.expanduser(args.ssh_key)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    conn = provision_cloud_vm(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        ssh_key=ssh_key,
        server_name=args.name,
        dry_run=args.dry_run,
    )
    if conn is None:
        print("Error: VM provisioning failed.", file=sys.stderr)
        sys.exit(1)

    params = DeployParams(
        server=conn.address,
        ssh_key=ssh_key,
        ssh_port=conn.ssh_port,
        recipe_config=config,
        model_dir=args.model_dir,
        hf_token=hf_token,
        dry_run=args.dry_run,
    )
    provision_remote(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)

    if not deploy_entry(params):
        sys.exit(1)


def register_cloud_target(subparsers):
    """Register the cloud deploy target."""
    parser = subparsers.add_parser("cloud", help="Provision a cloud VM and deploy via SSH")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--variant", default=None, help="Hardware variant (e.g. RTX5090)")
    parser.add_argument("--name", default="cloud-deploy", help="VM name prefix (default: cloud-deploy)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/hf_models", help="Model cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_cloud)
