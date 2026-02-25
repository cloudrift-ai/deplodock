"""Local deploy target CLI handler."""

import os
import sys

from deplodock.deploy import run_deploy, run_teardown
from deplodock.deploy.local import make_run_cmd, make_write_file
from deplodock.recipe import load_recipe


def handle_local(args):
    """Handle the local deploy target."""
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
        return run_teardown(run_cmd)

    recipe = load_recipe(recipe_dir)

    success = run_deploy(
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
    parser.set_defaults(func=handle_local)
