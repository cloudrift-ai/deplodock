"""CloudRift provider CLI handlers."""

import asyncio
import logging
import os
import sys

from deplodock.provisioning.cloudrift import (
    DEFAULT_API_URL,
    DEFAULT_IMAGE_URL,
    create_instance,
    delete_instance,
)

logger = logging.getLogger(__name__)


def _resolve_api_key(args_api_key):
    """Return the API key from the CLI flag or CLOUDRIFT_API_KEY env var.

    Raises SystemExit if neither is set.
    """
    api_key = args_api_key or os.environ.get("CLOUDRIFT_API_KEY")
    if not api_key:
        logger.error("Error: CloudRift API key required. Use --api-key or set CLOUDRIFT_API_KEY.")
        sys.exit(1)
    return api_key


# ── CLI handlers ───────────────────────────────────────────────────


def handle_create(args):
    """CLI handler for 'vm create cloudrift'."""
    asyncio.run(_handle_create(args))


async def _handle_create(args):
    api_key = _resolve_api_key(args.api_key)
    ports = [int(p) for p in args.ports.split(",")] if args.ports else None
    conn = await create_instance(
        api_key=api_key,
        instance_type=args.instance_type,
        ssh_key_path=args.ssh_key,
        image_url=args.image_url,
        ports=ports,
        timeout=args.timeout,
        api_url=args.api_url,
        dry_run=args.dry_run,
    )
    if conn is None:
        sys.exit(1)


def handle_delete(args):
    """CLI handler for 'vm delete cloudrift'."""
    asyncio.run(_handle_delete(args))


async def _handle_delete(args):
    api_key = _resolve_api_key(args.api_key)
    success = await delete_instance(
        api_key=api_key,
        instance_id=args.instance_id,
        api_url=args.api_url,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


# ── Registration ───────────────────────────────────────────────────


def register_create_target(subparsers):
    """Register the cloudrift provider under 'vm create'."""
    parser = subparsers.add_parser("cloudrift", help="Create a CloudRift GPU VM")
    parser.add_argument("--instance-type", required=True, help="Instance type (e.g. rtx4090.1)")
    parser.add_argument("--ssh-key", required=True, help="Path to SSH public key file")
    parser.add_argument("--api-key", default=None, help="CloudRift API key (fallback: CLOUDRIFT_API_KEY env var)")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="VM image URL (default: Ubuntu 24.04)")
    parser.add_argument("--ports", default="22,8000", help="Comma-separated ports to open (default: 22,8000)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for Active status (default: 600)")
    parser.add_argument("--dry-run", action="store_true", help="Print requests without executing")
    parser.set_defaults(func=handle_create)


def register_delete_target(subparsers):
    """Register the cloudrift provider under 'vm delete'."""
    parser = subparsers.add_parser("cloudrift", help="Delete a CloudRift GPU VM")
    parser.add_argument("--instance-id", required=True, help="CloudRift instance ID")
    parser.add_argument("--api-key", default=None, help="CloudRift API key (fallback: CLOUDRIFT_API_KEY env var)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--dry-run", action="store_true", help="Print requests without executing")
    parser.set_defaults(func=handle_delete)
