"""Bench command: deploy + benchmark + teardown on cloud VMs.

Accepts recipe directories as positional args. A Planner groups tasks into
ExecutionGroups that share VMs; groups run in parallel via asyncio.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from deplodock.benchmark import (
    _expand_path,
    _run_groups,
    add_file_handler,
    enumerate_tasks,
    load_config,
    setup_logging,
    validate_config,
)
from deplodock.benchmark.execution import _run_groups_on_hosts
from deplodock.benchmark.fixed_hosts import (
    resolve_fixed_hosts,
    validate_hosts_cover_groups,
)
from deplodock.planner import BenchmarkTask
from deplodock.planner.group_by_model_and_gpu import GroupByModelAndGpuPlanner


def handle_bench(args):
    """Handle the bench command."""
    setup_logging()
    root_logger = logging.getLogger()

    config = load_config(args.config)
    validate_config(config)

    if args.billing_exempt:
        config.setdefault("providers", {}).setdefault("cloudrift", {})["billing_exempt"] = True

    ssh_key = _expand_path(args.ssh_key)
    dry_run = args.dry_run
    no_teardown = args.no_teardown
    use_local = getattr(args, "local", False)
    ssh_targets = list(getattr(args, "ssh", None) or [])
    fixed_host_mode = use_local or bool(ssh_targets)

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
            recipe_run_dirs[resolved] = BenchmarkTask.create_run_dir(resolved)

    # Assign run_dir to each task and copy recipe files
    for task in tasks:
        resolved = str(Path(task.recipe_dir).resolve())
        task.setup_run_dir(recipe_run_dirs[resolved])

    # Write tasks.json per run_dir
    for resolved, run_dir in recipe_run_dirs.items():
        run_tasks = [t for t in tasks if str(Path(t.recipe_dir).resolve()) == resolved]
        BenchmarkTask.write_tasks_json(run_dir, run_tasks)

    # Attach file handlers for each run directory
    log_file_paths = []
    for run_dir in recipe_run_dirs.values():
        log_file_paths.append(add_file_handler(run_dir))

    for run_dir in recipe_run_dirs.values():
        root_logger.info(f"Run directory: {run_dir}")
    root_logger.info("")

    # Plan execution groups
    gpu_concurrency = args.gpu_concurrency
    planner = GroupByModelAndGpuPlanner(gpu_concurrency=gpu_concurrency)
    groups = planner.plan(tasks)

    root_logger.info(f"Running {len(tasks)} benchmark task(s) in {len(groups)} execution group(s)")
    if gpu_concurrency > 1:
        root_logger.info(f"GPU concurrency: {gpu_concurrency} (groups split across multiple VMs)")
    root_logger.info(f"Parallel mode (max workers: {args.max_workers or len(groups)})")
    root_logger.info("")

    # Set up commit callback if requested
    on_task_done = None
    if args.commit_results:
        from deplodock.commands.bench.committer import GitCommitter

        root_logger.info("Commit mode enabled: results will be committed after each task")
        on_task_done = GitCommitter(asyncio.Lock())

    # Run groups
    if fixed_host_mode:
        try:
            allocated = asyncio.run(resolve_fixed_hosts(use_local, ssh_targets, ssh_key, dry_run))
        except Exception as e:
            root_logger.error(f"Failed to resolve fixed hosts: {e}")
            sys.exit(1)

        if not dry_run:
            try:
                validate_hosts_cover_groups(allocated, groups)
            except Exception as e:
                root_logger.error(str(e))
                sys.exit(1)

        root_logger.info(f"Fixed-host mode: {len(allocated)} host(s), running {len(groups)} group(s)")
        raw_results = asyncio.run(_run_groups_on_hosts(groups, allocated, config, ssh_key, dry_run, on_task_done))
    else:
        raw_results = asyncio.run(_run_groups(groups, config, ssh_key, dry_run, args.max_workers, no_teardown, on_task_done))

    # Flatten results, handling exceptions
    all_results: list[tuple[BenchmarkTask, bool]] = []
    all_instance_infos = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            root_logger.error(f"Group {i} generated an exception: {r}")
            all_results.extend((t, False) for t in groups[i].tasks)
        else:
            task_results, instance_info = r
            all_results.extend(task_results)
            if instance_info is not None:
                all_instance_infos.append(instance_info)

    # Write instances.json for --no-teardown (cloud-provisioned VMs only)
    if all_instance_infos and not fixed_host_mode:
        for run_dir in recipe_run_dirs.values():
            instances_path = run_dir / "instances.json"
            instances_path.write_text(json.dumps(all_instance_infos, indent=2))
            root_logger.info(f"Instance info saved to: {instances_path}")
            root_logger.info("Run 'deplodock teardown <run_dir>' to clean up.")

    # Print summary
    root_logger.info("")
    root_logger.info("SUMMARY")

    successful = [t for t, ok in all_results if ok]
    failed = [t for t, ok in all_results if not ok]

    root_logger.info(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        for t in successful:
            root_logger.info(f"   - {t.task_id}")

    if failed:
        root_logger.info("")
        root_logger.info(f"Failed: {len(failed)}/{len(all_results)}")
        for t in failed:
            root_logger.info(f"   - {t.task_id}")

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
        "--gpu-concurrency",
        type=int,
        default=1,
        help="Split each (model, GPU) group across up to N VMs (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run benchmarks on the local machine (skips cloud provisioning; uses ssh to 127.0.0.1)",
    )
    parser.add_argument(
        "--ssh",
        action="append",
        default=None,
        metavar="USER@HOST[:PORT]",
        help="Pre-allocated SSH host to run benchmarks on (repeatable). Skips cloud provisioning.",
    )
    parser.add_argument(
        "--no-teardown",
        action="store_true",
        help="Skip teardown and VM deletion after benchmarks (save instance info for later cleanup)",
    )
    parser.add_argument(
        "--commit-results",
        action="store_true",
        help="Git commit and push result files after each task completes",
    )
    parser.add_argument(
        "--billing-exempt",
        action="store_true",
        help="Skip billing for CloudRift instances (admin-only)",
    )
    parser.set_defaults(func=handle_bench)
