"""Bench command: deploy + benchmark + teardown on cloud VMs.

Accepts recipe directories as positional args. A Planner groups tasks into
ExecutionGroups that share VMs; groups run in parallel via asyncio.
"""

import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from deplodock.benchmark import (
    _expand_path,
    _run_groups,
    _task_meta,
    add_file_handler,
    compute_code_hash,
    create_run_dir,
    enumerate_tasks,
    load_config,
    setup_logging,
    validate_config,
    write_manifest,
)
from deplodock.planner.group_by_model_and_gpu import GroupByModelAndGpuPlanner


def handle_bench(args):
    """Handle the bench command."""
    setup_logging()
    root_logger = logging.getLogger()

    config = load_config(args.config)
    validate_config(config)

    ssh_key = _expand_path(args.ssh_key)
    dry_run = args.dry_run
    no_teardown = args.no_teardown

    # Enumerate tasks from recipe dirs
    tasks = enumerate_tasks(args.recipes)
    if not tasks:
        root_logger.error("Error: No benchmark tasks found.")
        sys.exit(1)

    # Create run directory
    if args.output_dir:
        local_results_dir = _expand_path(args.output_dir)
    elif len(args.recipes) == 1:
        local_results_dir = str(Path(args.recipes[0]).resolve())
    else:
        local_results_dir = _expand_path(config["benchmark"]["local_results_dir"])
    run_dir = create_run_dir(local_results_dir)

    # Attach file handler now that run_dir exists
    log_file_path = add_file_handler(run_dir)
    root_logger.info(f"Logging to: {log_file_path}")
    root_logger.info(f"Run directory: {run_dir}")
    root_logger.info("")

    # Copy recipe files into run directory
    seen_recipes = set()
    for task in tasks:
        rname = task.recipe_name
        if rname not in seen_recipes:
            seen_recipes.add(rname)
            recipe_subdir = run_dir / rname
            recipe_subdir.mkdir(parents=True, exist_ok=True)
            src = Path(task.recipe_dir) / "recipe.yaml"
            if src.exists():
                shutil.copy2(str(src), str(recipe_subdir / "recipe.yaml"))

    # Plan execution groups
    planner = GroupByModelAndGpuPlanner()
    groups = planner.plan(tasks)

    root_logger.info(f"Running {len(tasks)} benchmark task(s) in {len(groups)} execution group(s)")
    root_logger.info(f"Parallel mode (max workers: {args.max_workers or len(groups)})")
    root_logger.info("")

    # Run groups
    raw_results = asyncio.run(_run_groups(groups, config, ssh_key, run_dir, dry_run, args.max_workers, no_teardown))

    # Flatten results, handling exceptions
    all_task_meta = []
    all_instance_infos = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            root_logger.error(f"Group {i} generated an exception: {r}")
            for task in groups[i].tasks:
                all_task_meta.append(_task_meta(task, run_dir, status="failed"))
        else:
            task_meta_list, instance_info = r
            all_task_meta.extend(task_meta_list)
            if instance_info is not None:
                all_instance_infos.append(instance_info)

    # Write instances.json if any VMs were kept alive
    if all_instance_infos:
        instances_path = run_dir / "instances.json"
        instances_path.write_text(json.dumps(all_instance_infos, indent=2))
        root_logger.info(f"Instance info saved to: {instances_path}")
        root_logger.info("Run 'deplodock teardown <run_dir>' to clean up.")

    # Write manifest
    code_hash = compute_code_hash()
    timestamp = datetime.now().isoformat(timespec="seconds")
    write_manifest(run_dir, timestamp, code_hash, sorted(seen_recipes), all_task_meta)
    root_logger.info(f"Manifest written to: {run_dir / 'manifest.json'}")

    # Print summary
    root_logger.info("")
    root_logger.info("SUMMARY")

    successful = [t for t in all_task_meta if t["status"] == "completed"]
    failed = [t for t in all_task_meta if t["status"] != "completed"]

    root_logger.info(f"Successful: {len(successful)}/{len(all_task_meta)}")
    if successful:
        for t in successful:
            root_logger.info(f"   - {t['result_file']}")

    if failed:
        root_logger.info("")
        root_logger.info(f"Failed: {len(failed)}/{len(all_task_meta)}")
        for t in failed:
            root_logger.info(f"   - {t['result_file']}")

    root_logger.info("")
    root_logger.info("All done!")
    root_logger.info(f"Full logs saved to: {log_file_path}")


def register_bench_command(subparsers):
    """Register the bench subcommand."""
    parser = subparsers.add_parser(
        "bench",
        help="Run LLM benchmarks on cloud VMs",
    )
    parser.add_argument(
        "recipes",
        nargs="+",
        help="Recipe directories to benchmark",
    )
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="SSH private key path (default: ~/.ssh/id_ed25519)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: {recipe_dir} for single recipe, config local_results_dir for multiple)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel execution groups (default: number of groups)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-teardown",
        action="store_true",
        help="Skip teardown and VM deletion after benchmarks (save instance info for later cleanup)",
    )
    parser.set_defaults(func=handle_bench)
