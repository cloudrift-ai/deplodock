"""SSH deploy target CLI handler."""

import os
import sys

from deplodock.deploy import (
    DeployParams,
    deploy as deploy_entry,
    load_recipe,
    teardown as teardown_entry,
)
from deplodock.deploy.ssh import make_run_cmd, make_write_file, scp_file, ssh_base_args, REMOTE_DEPLOY_DIR
from deplodock.provisioning.remote import provision_remote


def handle_ssh(args):
    """Handle the SSH deploy target."""
    config = load_recipe(args.recipe, variant=args.variant)
    params = DeployParams(
        server=args.server,
        ssh_key=args.ssh_key,
        ssh_port=args.ssh_port,
        recipe_config=config,
        model_dir=args.model_dir,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN", ""),
        dry_run=args.dry_run,
    )
    provision_remote(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)

    if args.teardown:
        return teardown_entry(params)

    if not deploy_entry(params):
        sys.exit(1)


def register_ssh_target(subparsers):
    """Register the SSH deploy target."""
    parser = subparsers.add_parser("ssh", help="Deploy via SSH to a remote server")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--variant", default=None, help="Hardware variant (e.g. 8xH200)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--server", required=True, help="SSH address (user@host)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.set_defaults(func=handle_ssh)
