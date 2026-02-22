"""Bench command: run LLM benchmarks on remote servers via SSH.

Uses the deploy infrastructure (recipe loading, SSH deploy/teardown) instead of
cloning a repo on the remote server.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from deplodock.commands.deploy import (
    calculate_num_instances,
    load_recipe,
    run_deploy,
    run_teardown,
)
from deplodock.commands.deploy.ssh import (
    REMOTE_DEPLOY_DIR,
    make_run_cmd,
    make_write_file,
    provision_remote,
)


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

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)

    class CustomConsoleFormatter(logging.Formatter):
        def format(self, record):
            if '.' in record.name:
                server, model = record.name.split('.', 1)
                record.name = f"{server}] [{model}"
            return super().format(record)

    console_formatter = CustomConsoleFormatter('[%(name)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return str(LOG_FILE)


def get_server_logger(server_name: str, model_name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger for a specific server and model."""
    if model_name:
        short_model = model_name.split('/')[-1] if '/' in model_name else model_name
        logger_name = f"{server_name}.{short_model}"
    else:
        logger_name = server_name
    return logging.getLogger(logger_name)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
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
    if 'benchmark' not in config:
        print("Error: Missing 'benchmark' section in config.")
        sys.exit(1)

    if 'servers' not in config or not config['servers']:
        print("Error: Missing 'servers' section or empty servers list in config.")
        sys.exit(1)

    required_benchmark_fields = ['local_results_dir']
    for field in required_benchmark_fields:
        if field not in config['benchmark']:
            print(f"Error: Missing '{field}' in 'benchmark' section.")
            sys.exit(1)

    required_server_fields = ['name', 'address', 'ssh_key', 'recipes']
    for idx, server in enumerate(config['servers']):
        for field in required_server_fields:
            if field not in server:
                print(f"Error: Missing '{field}' in server entry {idx} ({server.get('name', 'unnamed')}).")
                sys.exit(1)

        if not server['recipes'] or not isinstance(server['recipes'], list):
            print(f"Error: Server '{server['name']}' must have a non-empty 'recipes' list.")
            sys.exit(1)


def expand_path(path: str) -> str:
    """Expand user home directory and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))


def extract_benchmark_results(output: str) -> str:
    """Extract benchmark results section from vllm bench serve output."""
    marker = "============ Serving Benchmark Result ============"
    idx = output.find(marker)
    if idx == -1:
        return output
    return output[idx:]


def run_benchmark_workload(run_cmd, config, recipe_config, variant, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str)
    """
    benchmark_params = config.get('benchmark_params', {})
    max_concurrency = benchmark_params.get('max_concurrency', 128)
    num_prompts = benchmark_params.get('num_prompts', 256)
    random_input_len = benchmark_params.get('random_input_len', 8000)
    random_output_len = benchmark_params.get('random_output_len', 8000)

    model_name = recipe_config['model']['name']
    image = recipe_config['backend']['vllm'].get('image', 'vllm/vllm-openai:latest')

    num_instances = calculate_num_instances(recipe_config, variant)
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


def run_server_benchmarks(server: dict, recipe_entries: List[dict], config: dict,
                          force: bool = False, dry_run: bool = False) -> List[tuple]:
    """Run all benchmarks for a single server sequentially."""
    results = []
    server_name = server['name']
    address = server['address']
    ssh_key = expand_path(server['ssh_key'])
    ssh_port = server.get('port', 22)
    model_dir = config['benchmark'].get('model_dir', '/hf_models')
    hf_token = os.environ.get('HF_TOKEN', '')
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))

    logger = get_server_logger(server_name)
    logger.info(f"Starting benchmarks for server: {server_name}")

    # Provision the remote server
    provision_remote(address, ssh_key, ssh_port, dry_run=dry_run)

    for entry in recipe_entries:
        recipe_path = entry['recipe']
        variant = entry.get('variant')

        # Load recipe
        try:
            recipe_config = load_recipe(recipe_path, variant=variant)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load recipe {recipe_path}: {e}")
            results.append((server_name, recipe_path, False))
            continue

        model_name = recipe_config['model']['name']
        model_name_safe = model_name.replace('/', '_')
        result_filename = f"{server_name}_{model_name_safe}_vllm_benchmark.txt"
        result_path = local_results_dir / result_filename

        logger = get_server_logger(server_name, model_name)
        logger.info(f"Recipe: {recipe_path} (variant: {variant})")

        # Check if result already exists
        if not force and result_path.exists():
            logger.info(f"Result already exists: {result_path}, skipping (use --force to re-run)")
            results.append((server_name, model_name, True))
            continue

        # Create run_cmd / write_file for this server
        run_cmd = make_run_cmd(address, ssh_key, ssh_port, dry_run=dry_run)
        write_file = make_write_file(address, ssh_key, ssh_port, dry_run=dry_run)

        # Extract host from server address
        host = address.split("@")[-1] if "@" in address else address

        # Deploy model
        logger.info("Deploying model...")
        success = run_deploy(
            run_cmd=run_cmd,
            write_file=write_file,
            config=recipe_config,
            model_dir=model_dir,
            hf_token=hf_token,
            host=host,
            variant=variant,
            dry_run=dry_run,
        )

        if not success:
            logger.error("Deploy failed, skipping benchmark")
            results.append((server_name, model_name, False))
            continue

        # Run benchmark
        logger.info("Running benchmark...")
        bench_success, output = run_benchmark_workload(
            run_cmd, config, recipe_config, variant, dry_run=dry_run,
        )

        if bench_success or dry_run:
            if not dry_run:
                # Extract and save results
                benchmark_results = extract_benchmark_results(output)
                local_results_dir.mkdir(parents=True, exist_ok=True)
                result_path.write_text(benchmark_results)
                logger.info(f"Results saved to: {result_path}")
            else:
                logger.info(f"[dry-run] Would save results to: {result_path}")
            results.append((server_name, model_name, True))
        else:
            logger.error("Benchmark failed")
            results.append((server_name, model_name, False))

        # Teardown
        logger.info("Tearing down...")
        run_teardown(run_cmd)

    logger.info(f"Completed benchmarks for server: {server_name}")
    return results


def handle_bench(args):
    """Handle the bench command."""
    log_file_path = setup_logging()
    root_logger = logging.getLogger()
    root_logger.info(f"Logging to: {log_file_path}")
    root_logger.info("")

    config = load_config(args.config)
    validate_config(config)

    servers = config['servers']
    dry_run = args.dry_run

    if args.server:
        servers = [s for s in servers if s['name'] == args.server]
        if not servers:
            root_logger.error(f"Error: Server '{args.server}' not found in config.")
            sys.exit(1)

    # Build list of (server, recipe_entries) tuples
    server_tasks = []
    total_combinations = 0

    for server in servers:
        recipes = server['recipes']

        if args.recipe:
            recipes = [r for r in recipes if r['recipe'] == args.recipe]
            if not recipes:
                root_logger.warning(
                    f"Warning: Recipe '{args.recipe}' not found in server '{server['name']}'. "
                    "Skipping this server."
                )
                continue

        server_tasks.append((server, recipes))
        total_combinations += len(recipes)

    if not server_tasks:
        root_logger.error("Error: No server-recipe combinations to benchmark.")
        sys.exit(1)

    root_logger.info(
        f"Running benchmarks for {total_combinations} server-recipe combination(s) "
        f"across {len(server_tasks)} server(s)"
    )
    if args.parallel:
        root_logger.info(
            f"Parallel mode: Servers will run in parallel "
            f"(max workers: {args.max_workers or len(server_tasks)})"
        )
    else:
        root_logger.info("Sequential mode: Servers will run one at a time")
    root_logger.info("")

    results = []

    if args.parallel:
        max_workers = args.max_workers or len(server_tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_server = {
                executor.submit(
                    run_server_benchmarks, server, recipes, config,
                    args.force, dry_run,
                ): server['name']
                for server, recipes in server_tasks
            }

            for future in as_completed(future_to_server):
                server_name = future_to_server[future]
                try:
                    server_results = future.result()
                    results.extend(server_results)
                except Exception as exc:
                    root_logger.error(f"Server {server_name} generated an exception: {exc}")
                    for server, recipes in server_tasks:
                        if server['name'] == server_name:
                            for entry in recipes:
                                results.append((server_name, entry['recipe'], False))
    else:
        for server, recipes in server_tasks:
            server_results = run_server_benchmarks(
                server, recipes, config, args.force, dry_run,
            )
            results.extend(server_results)

    # Print summary
    root_logger.info("SUMMARY")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    root_logger.info(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        for server_name, model, _ in successful:
            root_logger.info(f"   - {server_name} x {model}")

    if failed:
        root_logger.info("")
        root_logger.info(f"Failed: {len(failed)}/{len(results)}")
        for server_name, model, _ in failed:
            root_logger.info(f"   - {server_name} x {model}")

    root_logger.info("")
    root_logger.info("All done!")
    root_logger.info(f"Full logs saved to: {log_file_path}")


def register_bench_command(subparsers):
    """Register the bench subcommand."""
    parser = subparsers.add_parser(
        "bench",
        help="Run LLM benchmarks on remote servers via SSH",
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force benchmark even if results already exist',
    )
    parser.add_argument(
        '--server',
        help='Run benchmarks only for a specific server (by name)',
    )
    parser.add_argument(
        '--recipe',
        help='Run benchmarks only for a specific recipe path',
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run benchmarks on different servers in parallel',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel server benchmarks (default: number of servers)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing',
    )
    parser.set_defaults(func=handle_bench)
