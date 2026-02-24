"""Bench command: deploy + benchmark + teardown on cloud VMs.

Accepts recipe directories as positional args. A Planner groups tasks into
ExecutionGroups that share VMs; groups run in parallel via asyncio.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from deplodock.commands.deploy import (
    DeployParams,
    calculate_num_instances,
    deploy as deploy_entry,
    load_recipe,
    teardown as teardown_entry,
)
from deplodock.commands.deploy.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
)
from deplodock.commands.deploy.ssh import make_run_cmd, provision_remote
from deplodock.hardware import gpu_short_name
from deplodock.planner import BenchmarkTask, ExecutionGroup
from deplodock.planner.group_by_model_and_gpu import GroupByModelAndGpuPlanner


# Global log file path
LOG_FILE = None


def setup_logging() -> str:
    """Setup logging with timestamped log file and console output.

    Returns:
        Path to the log file
    """
    global LOG_FILE

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = log_dir / f"benchmark_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)

    class CustomConsoleFormatter(logging.Formatter):
        def format(self, record):
            if "." in record.name:
                server, model = record.name.split(".", 1)
                record.name = f"{server}] [{model}"
            return super().format(record)

    console_formatter = CustomConsoleFormatter("[%(name)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return str(LOG_FILE)


def _get_group_logger(group: ExecutionGroup, model_name: Optional[str] = None) -> logging.Logger:
    """Get a logger for an execution group."""
    short = gpu_short_name(group.gpu_name)
    group_label = f"{short}_x_{group.gpu_count}"
    if model_name:
        short_model = model_name.split("/")[-1] if "/" in model_name else model_name
        return logging.getLogger(f"{group_label}.{short_model}")
    return logging.getLogger(group_label)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def validate_config(config: dict) -> None:
    """Validate that required configuration fields are present."""
    if "benchmark" not in config:
        print("Error: Missing 'benchmark' section in config.")
        sys.exit(1)

    required_benchmark_fields = ["local_results_dir"]
    for field in required_benchmark_fields:
        if field not in config["benchmark"]:
            print(f"Error: Missing '{field}' in 'benchmark' section.")
            sys.exit(1)


def _expand_path(path: str) -> str:
    """Expand user home directory and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))


def extract_benchmark_results(output: str) -> str:
    """Extract benchmark results section from vllm bench serve output."""
    marker = "============ Serving Benchmark Result ============"
    idx = output.find(marker)
    if idx == -1:
        return output
    return output[idx:]


def _parse_max_model_len(extra_args: str) -> Optional[int]:
    """Extract --max-model-len value from extra_args string."""
    parts = extra_args.split()
    for i, part in enumerate(parts):
        if part == "--max-model-len" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def run_benchmark_workload(run_cmd, recipe_config, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str)
    """
    benchmark_params = recipe_config.get("benchmark", {})
    max_concurrency = benchmark_params.get("max_concurrency", 128)
    num_prompts = benchmark_params.get("num_prompts", 256)
    random_input_len = benchmark_params.get("random_input_len", 8000)
    random_output_len = benchmark_params.get("random_output_len", 8000)

    # Warn if input + output lengths risk exceeding max-model-len
    extra_args = recipe_config.get("backend", {}).get("vllm", {}).get("extra_args", "")
    max_model_len = _parse_max_model_len(extra_args)
    if max_model_len is not None and random_input_len + random_output_len >= max_model_len:
        logging.getLogger().warning(
            f"benchmark random_input_len ({random_input_len}) + "
            f"random_output_len ({random_output_len}) = "
            f"{random_input_len + random_output_len} >= "
            f"max-model-len ({max_model_len})"
        )

    model_name = recipe_config["model"]["name"]
    image = recipe_config["backend"]["vllm"].get("image", "vllm/vllm-openai:latest")

    num_instances = calculate_num_instances(recipe_config)
    port = 8080 if num_instances > 1 else 8000

    bench_cmd = (
        f"docker run --rm --network host --entrypoint bash {image} -c '"
        f"vllm bench serve "
        f"--model {model_name} "
        f"--base-url http://localhost:{port} "
        f"--max-concurrency {max_concurrency} "
        f"--num-prompts {num_prompts} "
        f"--random-input-len {random_input_len} "
        f"--random-output-len {random_output_len}"
        f"'"
    )

    rc, output = run_cmd(bench_cmd, stream=False)
    return rc == 0, output


# ── Task enumeration ──────────────────────────────────────────────


def enumerate_tasks(recipe_dirs, variants_filter=None):
    """Build BenchmarkTask list from recipe dirs and optional variant filter.

    For each recipe dir, reads the raw variants from recipe.yaml. If
    variants_filter is given, only matching variants are kept (with a
    warning for missing ones). Each selected variant is loaded and
    turned into a BenchmarkTask.
    """
    tasks = []
    for recipe_dir in recipe_dirs:
        recipe_path = os.path.join(recipe_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            print(f"Warning: No recipe.yaml in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        available_variants = list((raw.get("variants") or {}).keys())
        if not available_variants:
            print(f"Warning: No variants in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        if variants_filter is not None:
            selected = []
            for v in variants_filter:
                if v in available_variants:
                    selected.append(v)
                else:
                    print(
                        f"Warning: variant '{v}' not in {recipe_dir} "
                        f"(available: {', '.join(available_variants)}), skipping.",
                        file=sys.stderr,
                    )
            variants_to_run = selected
        else:
            variants_to_run = available_variants

        for variant in variants_to_run:
            config = load_recipe(recipe_dir, variant=variant)
            gpu_name = config.get("gpu")
            gpu_count = config.get("gpu_count", 1)
            if gpu_name is None:
                print(
                    f"Warning: variant '{variant}' in {recipe_dir} missing 'gpu', skipping.",
                    file=sys.stderr,
                )
                continue

            tasks.append(BenchmarkTask(
                recipe_dir=recipe_dir,
                variant=variant,
                recipe_config=config,
                gpu_name=gpu_name,
                gpu_count=gpu_count,
            ))

    return tasks


# ── Execution ─────────────────────────────────────────────────────


def run_execution_group(group: ExecutionGroup, config: dict, ssh_key: str,
                        force: bool = False, dry_run: bool = False) -> List[tuple]:
    """Run all benchmark tasks for one execution group.

    Provisions a VM with group.gpu_count GPUs, then runs each task.
    Tasks with fewer GPUs than the group get _gpu_device_ids set.
    """
    results = []
    model_dir = config["benchmark"].get("model_dir", "/hf_models")
    hf_token = os.environ.get("HF_TOKEN", "")
    local_results_dir = Path(_expand_path(config["benchmark"]["local_results_dir"]))
    providers_config = config.get("providers", {})

    short = gpu_short_name(group.gpu_name)
    group_label = f"{short}_x_{group.gpu_count}"
    logger = _get_group_logger(group)
    logger.info(f"Starting group: {group.gpu_name} x{group.gpu_count} ({len(group.tasks)} tasks)")

    conn = None
    try:
        conn = provision_cloud_vm(
            group.gpu_name, group.gpu_count, ssh_key, providers_config,
            server_name=group_label, dry_run=dry_run, logger=logger,
        )
        if conn is None:
            logger.error("VM provisioning failed")
            for task in group.tasks:
                results.append((task.result_filename, task.model_name, False))
            return results

        logger.info(f"VM provisioned: {conn.address}:{conn.ssh_port}")
        provision_remote(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)

        for task in group.tasks:
            recipe_config = task.recipe_config
            model_name = task.model_name
            result_filename = task.result_filename
            result_path = local_results_dir / result_filename

            task_logger = _get_group_logger(group, model_name)
            task_logger.info(f"Recipe: {task.recipe_dir} (variant: {task.variant})")

            # Check if result already exists
            if not force and result_path.exists():
                task_logger.info(f"Result already exists: {result_path}, skipping (use --force to re-run)")
                results.append((result_filename, model_name, True))
                continue

            # Set GPU device IDs if task needs fewer GPUs than group
            if task.gpu_count < group.gpu_count:
                recipe_config = dict(recipe_config)
                recipe_config["_gpu_device_ids"] = list(range(task.gpu_count))

            params = DeployParams(
                server=conn.address,
                ssh_key=ssh_key,
                ssh_port=conn.ssh_port,
                recipe_config=recipe_config,
                model_dir=model_dir,
                hf_token=hf_token,
                dry_run=dry_run,
            )
            task_logger.info("Deploying model...")
            success = deploy_entry(params)

            if not success:
                task_logger.error("Deploy failed, skipping benchmark")
                results.append((result_filename, model_name, False))
                continue

            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
            bench_success, output = run_benchmark_workload(
                run_cmd, recipe_config, dry_run=dry_run,
            )

            if bench_success or dry_run:
                if not dry_run:
                    benchmark_results = extract_benchmark_results(output)
                    local_results_dir.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(benchmark_results)
                    task_logger.info(f"Results saved to: {result_path}")
                else:
                    task_logger.info(f"[dry-run] Would save results to: {result_path}")
                results.append((result_filename, model_name, True))
            else:
                task_logger.error("Benchmark failed")
                results.append((result_filename, model_name, False))

            task_logger.info("Tearing down...")
            teardown_entry(params)

    finally:
        if conn is not None and conn.delete_info:
            logger.info("Deleting VM...")
            try:
                delete_cloud_vm(conn.delete_info, dry_run)
                logger.info("VM deleted.")
            except Exception as e:
                logger.error(f"Failed to delete VM: {e}")

    logger.info(f"Completed group: {group_label}")
    return results


# ── Async orchestration ───────────────────────────────────────────


async def _run_groups(groups, config, ssh_key, force, dry_run, max_workers):
    """Run execution groups concurrently with a semaphore."""
    sem = asyncio.Semaphore(max_workers or len(groups))

    async def _run_with_semaphore(group):
        async with sem:
            return await asyncio.to_thread(
                run_execution_group, group, config, ssh_key, force, dry_run,
            )

    results = await asyncio.gather(
        *(_run_with_semaphore(g) for g in groups),
        return_exceptions=True,
    )
    return results


def handle_bench(args):
    """Handle the bench command."""
    log_file_path = setup_logging()
    root_logger = logging.getLogger()
    root_logger.info(f"Logging to: {log_file_path}")
    root_logger.info("")

    config = load_config(args.config)
    validate_config(config)

    ssh_key = _expand_path(args.ssh_key)
    dry_run = args.dry_run

    # Enumerate tasks from recipe dirs
    tasks = enumerate_tasks(args.recipes, variants_filter=args.variants)
    if not tasks:
        root_logger.error("Error: No benchmark tasks found.")
        sys.exit(1)

    # Plan execution groups
    planner = GroupByModelAndGpuPlanner()
    groups = planner.plan(tasks)

    root_logger.info(
        f"Running {len(tasks)} benchmark task(s) in {len(groups)} execution group(s)"
    )
    root_logger.info(
        f"Parallel mode (max workers: {args.max_workers or len(groups)})"
    )
    root_logger.info("")

    # Run groups
    raw_results = asyncio.run(
        _run_groups(groups, config, ssh_key, args.force, dry_run, args.max_workers)
    )

    # Flatten results, handling exceptions
    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            root_logger.error(f"Group {i} generated an exception: {r}")
            for task in groups[i].tasks:
                results.append((task.result_filename, task.model_name, False))
        else:
            results.extend(r)

    # Print summary
    root_logger.info("SUMMARY")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    root_logger.info(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        for name, model, _ in successful:
            root_logger.info(f"   - {name}")

    if failed:
        root_logger.info("")
        root_logger.info(f"Failed: {len(failed)}/{len(results)}")
        for name, model, _ in failed:
            root_logger.info(f"   - {name}")

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
        "--variants",
        nargs="+",
        default=None,
        help="Variant names to run (default: all variants in each recipe)",
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
        "--force",
        action="store_true",
        help="Force benchmark even if results already exist",
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
    parser.set_defaults(func=handle_bench)
