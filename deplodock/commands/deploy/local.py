"""Local deploy target CLI handler."""

import asyncio
import logging
import os
import sys

from deplodock.deploy import DEFAULT_STRATEGY, STRATEGIES, run_deploy, run_teardown
from deplodock.deploy.local import make_run_cmd, make_write_file
from deplodock.detect import detect_local_gpus
from deplodock.provisioning.host import LocalHost
from deplodock.provisioning.remote import provision_remote
from deplodock.recipe import resolve_for_hardware

logger = logging.getLogger(__name__)


def handle_local(args):
    """Handle the local deploy target."""
    asyncio.run(_handle_local(args))


async def _handle_local(args):
    recipe_dir = args.recipe
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    model_dir = args.model_dir
    dry_run = args.dry_run
    teardown = args.teardown

    # Deploy directory is the recipe directory itself
    deploy_dir = os.path.abspath(recipe_dir)

    run_cmd = make_run_cmd(deploy_dir, dry_run=dry_run)
    write_file = make_write_file(deploy_dir, dry_run=dry_run)

    if teardown:
        return await run_teardown(run_cmd)

    # GPU detection (overridable via CLI flags)
    if not args.gpu or not args.gpu_count:
        detected_name, detected_count = detect_local_gpus()
    gpu_name = args.gpu or detected_name
    gpu_count = args.gpu_count or detected_count
    logger.info(f"GPU: {gpu_count}x {gpu_name}")

    # Matrix resolution
    recipe = resolve_for_hardware(recipe_dir, gpu_name, gpu_count)

    # Scale-out
    strategy_cls = STRATEGIES[args.scale_out_strategy]
    recipe = strategy_cls().apply(recipe, gpu_count)

    # Driver/CUDA provisioning. On a LocalHost any sudo step raises a
    # ClickException, so this is a no-op when the installed versions already
    # match and an error otherwise.
    if recipe.deploy.driver_version or recipe.deploy.cuda_version:
        local_host = LocalHost(dry_run=dry_run)
        await provision_remote(
            local_host,
            skip_nvidia=False,
            driver_version=recipe.deploy.driver_version,
            cuda_version=recipe.deploy.cuda_version,
        )

    success = await run_deploy(
        run_cmd=run_cmd,
        write_file=write_file,
        recipe=recipe,
        model_dir=model_dir,
        hf_token=hf_token,
        host="localhost",
        dry_run=dry_run,
    )

    if not success:
        sys.exit(1)


def register_local_target(subparsers):
    """Register the local deploy target."""
    parser = subparsers.add_parser("local", help="Deploy locally via docker compose")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--gpu", default=None, help="Override GPU name (skips detection)")
    parser.add_argument("--gpu-count", type=int, default=None, help="Override GPU count (skips count detection)")
    parser.add_argument(
        "--scale-out-strategy",
        choices=list(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Scale-out strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.set_defaults(func=handle_local)
