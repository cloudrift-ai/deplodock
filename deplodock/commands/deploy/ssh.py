"""SSH deploy target CLI handler."""

import asyncio
import logging
import os
import sys

from deplodock.deploy import DEFAULT_STRATEGY, STRATEGIES, DeployParams
from deplodock.deploy import (
    deploy as deploy_entry,
)
from deplodock.deploy import (
    teardown as teardown_entry,
)
from deplodock.detect import detect_remote_gpus
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.ssh_target import parse_ssh_target
from deplodock.recipe import resolve_for_hardware

logger = logging.getLogger(__name__)


def handle_ssh(args):
    """Handle the SSH deploy target."""
    asyncio.run(_handle_ssh(args))


async def _handle_ssh(args):
    if args.ssh:
        if args.server is not None or args.ssh_port is not None:
            logger.error("--ssh cannot be combined with --server / --ssh-port")
            sys.exit(2)
        user, host, port = parse_ssh_target(args.ssh)
        server = f"{user}@{host}"
    elif args.server:
        logger.warning("--server is deprecated; use --ssh USER@HOST[:PORT] instead. This flag will be removed in a future release.")
        if args.ssh_port is not None:
            logger.warning("--ssh-port is deprecated; encode the port in --ssh USER@HOST:PORT.")
        server = args.server
        port = args.ssh_port if args.ssh_port is not None else 22
    else:
        logger.error("--ssh USER@HOST[:PORT] is required")
        sys.exit(2)

    # GPU detection (overridable via CLI flags)
    if not args.gpu or not args.gpu_count:
        detected_name, detected_count = await detect_remote_gpus(server, args.ssh_key, port)
    gpu_name = args.gpu or detected_name
    gpu_count = args.gpu_count or detected_count
    logger.info(f"GPU: {gpu_count}x {gpu_name}")

    # Matrix resolution
    recipe = resolve_for_hardware(args.recipe, gpu_name, gpu_count)

    # Scale-out
    strategy_cls = STRATEGIES[args.scale_out_strategy]
    recipe = strategy_cls().apply(recipe, gpu_count)

    params = DeployParams(
        server=server,
        ssh_key=args.ssh_key,
        ssh_port=port,
        recipe=recipe,
        model_dir=args.model_dir,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN", ""),
        dry_run=args.dry_run,
    )
    skip_nvidia = recipe.deploy.gpu is not None and recipe.deploy.gpu.startswith("AMD")
    await provision_remote(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run, skip_nvidia=skip_nvidia)

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
    parser.add_argument(
        "--ssh",
        default=None,
        metavar="USER@HOST[:PORT]",
        help="SSH target (e.g. user@host or user@host:2222). Default port: 22",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH key path")
    # Deprecated — kept for backwards compatibility. Prefer --ssh USER@HOST[:PORT].
    parser.add_argument("--server", default=None, help="[DEPRECATED] SSH address (user@host); use --ssh instead")
    parser.add_argument("--ssh-port", type=int, default=None, help="[DEPRECATED] SSH port; encode it in --ssh USER@HOST:PORT")
    parser.add_argument("--gpu", default=None, help="Override GPU name (skips detection)")
    parser.add_argument("--gpu-count", type=int, default=None, help="Override GPU count (skips count detection)")
    parser.add_argument(
        "--scale-out-strategy",
        choices=list(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Scale-out strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.set_defaults(func=handle_ssh)
