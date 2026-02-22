"""Local deploy target: runs docker compose directly via subprocess."""

import os
import subprocess
import sys

from deplodock.commands.deploy import load_recipe, run_deploy, run_teardown


def _make_run_cmd(deploy_dir, dry_run=False):
    """Create a run_cmd callable for local execution."""

    def run_cmd(command, stream=True):
        if dry_run:
            print(f"[dry-run] {command}")
            return 0, ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=deploy_dir,
                capture_output=not stream,
                text=True,
                stdout=None if stream else subprocess.PIPE,
                stderr=None if stream else subprocess.PIPE,
            )
            stdout = "" if stream else (result.stdout or "")
            return result.returncode, stdout
        except Exception as e:
            print(f"Error running command: {e}", file=sys.stderr)
            return 1, ""

    return run_cmd


def _make_write_file(deploy_dir, dry_run=False):
    """Create a write_file callable for local file writes."""

    def write_file(path, content):
        full_path = os.path.join(deploy_dir, path)
        if dry_run:
            print(f"[dry-run] write {full_path}")
            return
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    return write_file


def handle_local(args):
    """Handle the local deploy target."""
    recipe_dir = args.recipe
    variant = args.variant
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    model_dir = args.model_dir
    dry_run = args.dry_run
    teardown = args.teardown

    # Deploy directory is the recipe directory itself
    deploy_dir = os.path.abspath(recipe_dir)

    run_cmd = _make_run_cmd(deploy_dir, dry_run=dry_run)
    write_file = _make_write_file(deploy_dir, dry_run=dry_run)

    if teardown:
        return run_teardown(run_cmd)

    config = load_recipe(recipe_dir, variant=variant)

    success = run_deploy(
        run_cmd=run_cmd,
        write_file=write_file,
        config=config,
        model_dir=model_dir,
        hf_token=hf_token,
        host="localhost",
        variant=variant,
        dry_run=dry_run,
    )

    if not success:
        sys.exit(1)


def register_local_target(subparsers):
    """Register the local deploy target."""
    parser = subparsers.add_parser("local", help="Deploy locally via docker compose")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--variant", default=None, help="Hardware variant (e.g. 8xH200)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_local)
