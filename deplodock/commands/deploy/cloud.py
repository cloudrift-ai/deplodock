"""Cloud deploy target CLI handler."""

import logging
import os
import sys

from deplodock.deploy import DeployParams
from deplodock.deploy import (
    deploy as deploy_entry,
)
from deplodock.provisioning.cloud import (
    provision_cloud_vm,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.recipe import load_recipe

logger = logging.getLogger(__name__)

# ── CLI handler ────────────────────────────────────────────────────


def handle_cloud(args):
    """CLI handler for 'deploy cloud'."""
    recipe = load_recipe(args.recipe)

    if not recipe.deploy.gpu:
        logger.error("Error: recipe must have a 'deploy.gpu' field for cloud provisioning.")
        sys.exit(1)

    ssh_key = os.path.expanduser(args.ssh_key)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    conn = provision_cloud_vm(
        gpu_name=recipe.deploy.gpu,
        gpu_count=recipe.deploy.gpu_count,
        ssh_key=ssh_key,
        server_name=args.name,
        dry_run=args.dry_run,
    )
    if conn is None:
        logger.error("Error: VM provisioning failed.")
        sys.exit(1)

    params = DeployParams(
        server=conn.address,
        ssh_key=ssh_key,
        ssh_port=conn.ssh_port,
        recipe=recipe,
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
    parser.add_argument("--name", default="cloud-deploy", help="VM name prefix (default: cloud-deploy)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/hf_models", help="Model cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_cloud)
