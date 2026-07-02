"""Cloud deploy target CLI handler."""

import asyncio
import logging
import os
import sys

from emmy.deploy import DeployParams
from emmy.deploy import (
    deploy as deploy_entry,
)
from emmy.provisioning.cloud import (
    provision_cloud_vm,
    read_public_key_files,
)
from emmy.provisioning.host import RemoteHost
from emmy.provisioning.remote import provision_remote
from emmy.recipe import resolve_for_hardware
from emmy.redact import register_secret
from emmy.timing import PHASE_REMOTE_PROVISION, PHASE_VM_PROVISION, PhaseTimer

logger = logging.getLogger(__name__)

# ── CLI handler ────────────────────────────────────────────────────


def handle_cloud(args):
    """CLI handler for 'deploy cloud'."""
    asyncio.run(_handle_cloud(args))


async def _handle_cloud(args):
    recipe = resolve_for_hardware(args.recipe, args.gpu, args.gpu_count)
    logger.info(f"GPU: {recipe.deploy.gpu_count}x {recipe.deploy.gpu}")

    ssh_key = os.path.expanduser(args.ssh_key)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    register_secret(hf_token)

    try:
        extra_authorized_keys = read_public_key_files(args.authorized_key)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    providers_config = None
    if args.billing_exempt or args.network:
        providers_config = {"cloudrift": {}}
        if args.billing_exempt:
            providers_config["cloudrift"]["billing_exempt"] = True
        if args.network:
            providers_config["cloudrift"]["network"] = args.network

    timer = PhaseTimer()
    async with timer.ameasure(PHASE_VM_PROVISION):
        conn = await provision_cloud_vm(
            gpu_name=recipe.deploy.gpu,
            gpu_count=recipe.deploy.gpu_count,
            ssh_key=ssh_key,
            providers_config=providers_config,
            server_name=args.name,
            dry_run=args.dry_run,
            provider=args.provider,
            extra_authorized_keys=extra_authorized_keys,
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
        port_mappings=conn.port_mappings,
    )
    host = RemoteHost(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    async with timer.ameasure(PHASE_REMOTE_PROVISION):
        await provision_remote(
            host,
            driver_version=recipe.deploy.driver_version,
            cuda_version=recipe.deploy.cuda_version,
        )

    if not await deploy_entry(params, timer=timer):
        sys.exit(1)

    logger.info("\nTiming:")
    for line in timer.format_table().splitlines():
        logger.info(line)


def register_cloud_target(subparsers):
    """Register the cloud deploy target."""
    parser = subparsers.add_parser("cloud", help="Provision a cloud VM and deploy via SSH")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--name", default="cloud-deploy", help="VM name prefix (default: cloud-deploy)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument(
        "--authorized-key",
        action="append",
        default=None,
        metavar="PATH",
        help="Extra SSH public key file to install in the VM's authorized_keys (repeatable)",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/hf_models", help="Model cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--billing-exempt", action="store_true", help="Skip billing for CloudRift (admin-only)")
    parser.add_argument(
        "--network",
        default=None,
        help="CloudRift network name (must exist in target datacenter; default: provider picks a public network)",
    )
    parser.add_argument("--gpu", required=True, help="GPU name (selects matching matrix entry)")
    parser.add_argument("--gpu-count", type=int, required=True, help="GPU count (selects matching matrix entry)")
    parser.add_argument(
        "--provider",
        choices=["gcp", "cloudrift"],
        default=None,
        help="Force cloud provider (default: first listed for the GPU in the hardware table)",
    )
    parser.set_defaults(func=handle_cloud)
