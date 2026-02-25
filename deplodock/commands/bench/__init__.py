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

    # Create per-recipe run directories
    recipe_run_dirs = {}
    for recipe_dir in args.recipes:
        resolved = str(Path(recipe_dir).resolve())
        if resolved not in recipe_run_dirs:
            recipe_run_dirs[resolved] = create_run_dir(resolved)

    # Assign run_dir to each task and copy recipe files
    for task in tasks:
        resolved = str(Path(task.recipe_dir).resolve())
        task.run_dir = recipe_run_dirs[resolved]
        src = Path(task.recipe_dir) / "recipe.yaml"
        dest = task.run_dir / "recipe.yaml"
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))

    # Attach file handlers for each run directory
    log_file_paths = []
    for run_dir in recipe_run_dirs.values():
        log_file_paths.append(add_file_handler(run_dir))

    for run_dir in recipe_run_dirs.values():
        root_logger.info(f"Run directory: {run_dir}")
    root_logger.info("")

    # Plan execution groups
    planner = GroupByModelAndGpuPlanner()
    groups = planner.plan(tasks)

    root_logger.info(f"Running {len(tasks)} benchmark task(s) in {len(groups)} execution group(s)")
    root_logger.info(f"Parallel mode (max workers: {args.max_workers or len(groups)})")
    root_logger.info("")

    # Run groups
    raw_results = asyncio.run(_run_groups(groups, config, ssh_key, dry_run, args.max_workers, no_teardown))

    # Flatten results, handling exceptions
    all_task_meta = []
    all_instance_infos = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            root_logger.error(f"Group {i} generated an exception: {r}")
            for task in groups[i].tasks:
                all_task_meta.append(_task_meta(task, status="failed"))
        else:
            task_meta_list, instance_info = r
            all_task_meta.extend(task_meta_list)
            if instance_info is not None:
                all_instance_infos.append(instance_info)

    # Write per-recipe manifests and instances.json
    code_hash = compute_code_hash()
    timestamp = datetime.now().isoformat(timespec="seconds")

    for resolved, run_dir in recipe_run_dirs.items():
        recipe_name = Path(resolved).name
        recipe_tasks = [t for t in all_task_meta if t["recipe"] == recipe_name]
        write_manifest(run_dir, timestamp, code_hash, [recipe_name], recipe_tasks)
        root_logger.info(f"Manifest written to: {run_dir / 'manifest.json'}")

        if all_instance_infos:
            instances_path = run_dir / "instances.json"
            instances_path.write_text(json.dumps(all_instance_infos, indent=2))
            root_logger.info(f"Instance info saved to: {instances_path}")
            root_logger.info("Run 'deplodock teardown <run_dir>' to clean up.")

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
    for p in log_file_paths:
        root_logger.info(f"Full logs saved to: {p}")


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
