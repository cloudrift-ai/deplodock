"""SSH deploy target CLI handler."""

import asyncio
import os
import sys

from deplodock.deploy import DeployParams
from deplodock.deploy import (
    deploy as deploy_entry,
)
from deplodock.deploy import (
    teardown as teardown_entry,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.recipe import load_recipe


def handle_ssh(args):
    """Handle the SSH deploy target."""
    asyncio.run(_handle_ssh(args))


async def _handle_ssh(args):
    recipe = load_recipe(args.recipe)
    params = DeployParams(
        server=args.server,
        ssh_key=args.ssh_key,
        ssh_port=args.ssh_port,
        recipe=recipe,
        model_dir=args.model_dir,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN", ""),
        dry_run=args.dry_run,
    )
    await provision_remote(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)

    if args.teardown:
        return await teardown_entry(params)

    if not await deploy_entry(params):
        sys.exit(1)


def register_ssh_target(subparsers):
    """Register the SSH deploy target."""
    parser = subparsers.add_parser("ssh", help="Deploy via SSH to a remote server")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--server", required=True, help="SSH address (user@host)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.set_defaults(func=handle_ssh)
