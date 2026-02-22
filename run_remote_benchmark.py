#!/usr/bin/env python3
"""
Automated LLM Benchmark Runner

This script reads configuration from config.yaml and runs benchmarks on remote servers via SSH.
It supports multiple servers and models, running all combinations.
It will skip benchmarks if the results file already exists locally.

Benchmark scripts follow the naming convention: <step_index>_<benchmark_name>.sh
Result files follow the naming convention: <benchmark_name>.txt
"""

import argparse
import os
import sys
import subprocess
import time
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


# Global log file path
LOG_FILE = None


def setup_logging() -> str:
    """Setup logging with timestamped log file and console output.

    Returns:
        Path to the log file
    """
    global LOG_FILE

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = log_dir / f"benchmark_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # File handler - detailed format with timestamps
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler - custom format to split server.model into [server] [model]
    console_handler = logging.StreamHandler(sys.stdout)

    class CustomConsoleFormatter(logging.Formatter):
        def format(self, record):
            # Split logger name into server and model parts
            if '.' in record.name:
                server, model = record.name.split('.', 1)
                record.name = f"{server}] [{model}"
            return super().format(record)

    console_formatter = CustomConsoleFormatter('[%(name)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return str(LOG_FILE)


def get_server_logger(server_name: str, model_name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger for a specific server and model.

    Args:
        server_name: Name of the server (e.g., 'rtx5090_x_1')
        model_name: Name of the model (e.g., 'kosbu/Llama-3.3-70B-Instruct-AWQ')

    Returns:
        Logger instance for this server/model combination
    """
    if model_name:
        # Extract just the model name without organization prefix for brevity
        short_model = model_name.split('/')[-1] if '/' in model_name else model_name
        logger_name = f"{server_name}.{short_model}"
    else:
        logger_name = server_name
    return logging.getLogger(logger_name)


def run_ssh_command_with_logging(
    server_name: str,
    ssh_cmd: List[str],
    logger: logging.Logger
) -> int:
    """Run SSH command and log output line-by-line with server prefix.

    Args:
        server_name: Name of the server
        ssh_cmd: SSH command as list of arguments
        logger: Logger instance for this server

    Returns:
        Return code from the SSH command
    """
    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output line by line
    for line in iter(process.stdout.readline, ''):
        if line:
            # Log to file with full context, print to console with server prefix
            logger.info(line.rstrip())

    returncode = process.wait()
    return returncode


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}")
        sys.exit(1)


def validate_config(config: dict) -> None:
    """Validate that required configuration fields are present."""
    # Check for required sections
    if 'benchmark' not in config:
        print("❌ Error: Missing 'benchmark' section in config.")
        sys.exit(1)

    if 'servers' not in config or not config['servers']:
        print("❌ Error: Missing 'servers' section or empty servers list in config.")
        sys.exit(1)

    # Check benchmark fields
    required_benchmark_fields = ['local_results_dir']
    for field in required_benchmark_fields:
        if field not in config['benchmark']:
            print(f"❌ Error: Missing '{field}' in 'benchmark' section.")
            sys.exit(1)

    # Validate server entries
    required_server_fields = ['name', 'address', 'ssh_key', 'models']
    for idx, server in enumerate(config['servers']):
        for field in required_server_fields:
            if field not in server:
                print(f"❌ Error: Missing '{field}' in server entry {idx} ({server.get('name', 'unnamed')}).")
                sys.exit(1)

        # Validate models list
        if not server['models'] or not isinstance(server['models'], list):
            print(f"❌ Error: Server '{server['name']}' must have a non-empty 'models' list.")
            sys.exit(1)


def expand_path(path: str) -> str:
    """Expand user home directory and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))


def check_result_file_exists(server: dict, model_name: str, config: dict, result_file: str) -> bool:
    """Check if a specific result file already exists locally."""
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    model_name_safe = model_name.replace('/', '_')
    server_name = server['name']

    prefixed_results_file = f"{server_name}_{model_name_safe}_{result_file}"
    local_results_path = local_results_dir / prefixed_results_file

    return local_results_path.exists()


def run_ssh_command(server: dict, command: str, capture_output: bool = False) -> Optional[str]:
    """Execute a command on the remote server via SSH."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        command
    ]

    try:
        if capture_output:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            subprocess.run(ssh_cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ SSH command failed: {e}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def setup_remote_repo(server: dict, logger: logging.Logger) -> bool:
    """Clone or update the server-benchmark repository on remote server. Returns True if successful."""
    server_name = server['name']
    logger.info("📦 Setting up repository on remote server...")

    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    # Clean checkout: remove old repo and clone fresh to ensure latest code
    setup_cmd = (
        'rm -rf server-benchmark && '
        'git clone https://github.com/cloudrift-ai/server-benchmark.git'
    )

    try:
        ssh_cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(port),
            '-o', 'StrictHostKeyChecking=no',
            address,
            setup_cmd
        ]
        returncode = run_ssh_command_with_logging(server_name, ssh_cmd, logger)
        if returncode != 0:
            logger.error(f"❌ Failed to setup repository (exit code: {returncode})")
            return False
        logger.info("✅ Repository ready on remote server")
    except Exception as e:
        logger.error(f"❌ Failed to setup repository: {e}")
        return False

    # Run setup to install dependencies
    logger.info("🔧 Installing dependencies on remote server...")
    setup_script_cmd = 'cd server-benchmark && make setup'

    try:
        ssh_cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(port),
            '-o', 'StrictHostKeyChecking=no',
            address,
            setup_script_cmd
        ]
        returncode = run_ssh_command_with_logging(server_name, ssh_cmd, logger)
        if returncode != 0:
            logger.error(f"❌ Failed to install dependencies (exit code: {returncode})")
            return False
        logger.info("✅ Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def run_benchmark_step(server: dict, model_config: dict, config: dict, step_name: str, script_name: str,
                       result_file: str, force: bool = False, logger: logging.Logger = None) -> bool:
    """Run a single benchmark step if result doesn't exist, then download the result.

    Args:
        server: Server configuration dict
        model_config: Model configuration dict with 'name' and optional 'tensor_parallel_size'
        config: Full config dict
        step_name: Display name (e.g., "System Info", "HF Download")
        script_name: Shell script to run (e.g., "run_system_info.sh")
        result_file: Result file to download (e.g., "system_info.txt")
        force: Force re-run even if result exists
        logger: Logger instance for this server

    Returns:
        True if successful (either skipped or downloaded), False on failure
    """
    server_name = server['name']
    model_name = model_config['name']

    # Use provided logger or get one for this server
    if logger is None:
        logger = get_server_logger(server_name, model_name)

    # Check if result already exists
    if not force and check_result_file_exists(server, model_name, config, result_file):
        logger.info(f"⏭️  {step_name} result already exists, skipping...")
        return True

    # Run the benchmark step
    logger.info(f"🚀 Running {step_name}...")

    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    # Get benchmark parameters from config (required)
    if 'benchmark_params' not in config:
        raise ValueError("benchmark_params must be defined in config.yaml")

    benchmark_params = config['benchmark_params']
    required_params = ['max_concurrency', 'num_prompts', 'random_input_len', 'random_output_len']
    for param in required_params:
        if param not in benchmark_params:
            raise ValueError(f"benchmark_params.{param} must be defined in config.yaml")

    max_concurrency = benchmark_params['max_concurrency']
    num_prompts = benchmark_params['num_prompts']
    random_input_len = benchmark_params['random_input_len']
    random_output_len = benchmark_params['random_output_len']

    # Get parallelism parameters from model config (default to 1 if not specified)
    tensor_parallel_size = model_config.get('tensor_parallel_size', 1)
    pipeline_parallel_size = model_config.get('pipeline_parallel_size', 1)
    num_instances = model_config.get('num_instances', 1)
    extra_args = model_config.get('extra_args', '')

    # Get HuggingFace cache directory from config (default to /hf_models for backwards compatibility)
    hf_cache_dir = config['benchmark'].get('huggingface_cache_dir', '/hf_models')

    env_vars = (
        f'IMAGE_NAME="vllm/vllm-openai:latest" '
        f'CONTAINER_NAME="vllm_benchmark_container" '
        f'MODEL_NAME="{model_name}" '
        f'TENSOR_PARALLEL_SIZE={tensor_parallel_size} '
        f'PIPELINE_PARALLEL_SIZE={pipeline_parallel_size} '
        f'NUM_INSTANCES={num_instances} '
        f'VLLM_EXTRA_ARGS="{extra_args}" '
        f'MAX_CONCURRENCY={max_concurrency} '
        f'NUM_PROMPTS={num_prompts} '
        f'RANDOM_INPUT_LEN={random_input_len} '
        f'RANDOM_OUTPUT_LEN={random_output_len} '
        f'BENCHMARK_RESULTS_FILE="vllm_benchmark.txt" '
        f'HF_DIRECTORY="{hf_cache_dir}"'
    )

    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        f'cd server-benchmark && {env_vars} ./benchmarks/{script_name}'
    ]

    try:
        returncode = run_ssh_command_with_logging(server_name, ssh_cmd, logger)
        if returncode != 0:
            logger.error(f"❌ Failed to run {step_name} (exit code: {returncode})")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to run {step_name}: {e}")
        return False

    # Download the result file
    logger.info(f"📥 Downloading {step_name} results...")
    success = download_single_file(server, model_name, config, result_file, result_file)

    if success:
        logger.info(f"✅ {step_name} completed")
    else:
        logger.warning(f"⚠️  {step_name} completed but download failed")

    return success


def extract_benchmark_name(script_name: str) -> str:
    """Extract benchmark name from script name.

    Example: '1_system_info.sh' -> 'system_info'
    """
    match = re.match(r'\d+_(.+)\.sh$', script_name)
    if match:
        return match.group(1)
    return script_name


def get_benchmark_display_name(benchmark_name: str) -> str:
    """Convert benchmark_name to a display name.

    Example: 'system_info' -> 'System Info'
    """
    return benchmark_name.replace('_', ' ').title()


def get_benchmark_result_file(benchmark_name: str) -> str:
    """Get the result file name for a benchmark.

    Example: 'system_info' -> 'system_info.txt'
    """
    return f"{benchmark_name}.txt"


def discover_benchmarks(server: dict, logger: logging.Logger = None) -> List[str]:
    """Discover benchmark scripts from the remote server's benchmarks directory.

    Returns a list of script names (e.g., ['1_system_info.sh', '2_hf_download.sh']).
    Scripts must follow naming convention: <step_index>_<benchmark_name>.sh
    Result files follow convention: <benchmark_name>.txt
    """
    server_name = server['name']
    if logger is None:
        logger = get_server_logger(server_name)  # No model context for discovery

    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    # List benchmark scripts matching pattern: <digit>_*.sh
    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        'ls server-benchmark/benchmarks/[0-9]*_*.sh 2>/dev/null | sort'
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        script_paths = result.stdout.strip().split('\n')

        if not script_paths or script_paths == ['']:
            logger.warning("⚠️  Warning: No benchmark scripts found matching pattern [0-9]*_*.sh")
            return []

        script_names = []
        for script_path in script_paths:
            # Extract filename from path
            script_name = script_path.split('/')[-1]

            # Validate: <step_index>_<benchmark_name>.sh
            if re.match(r'\d+_.+\.sh$', script_name):
                script_names.append(script_name)

        return script_names
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Warning: Failed to discover benchmarks: {e}")
        return []


def run_benchmark(server: dict, model_config: dict, config: dict, force: bool = False, logger: logging.Logger = None) -> bool:
    """Run all benchmark steps sequentially. Returns True if successful, False otherwise."""
    model_name = model_config['name']
    server_name = server['name']

    # Use provided logger or get one for this server
    if logger is None:
        logger = get_server_logger(server_name, model_name)

    logger.info(f"🚀 Starting benchmarks for model: {model_name}")
    logger.info(f"📡 Connecting to: {server['address']} ({server_name})")

    # Discover benchmark scripts from remote server
    script_names = discover_benchmarks(server, logger)

    if not script_names:
        logger.error("❌ No benchmark steps found. Ensure scripts follow naming convention: <step_index>_<benchmark_name>.sh")
        return False

    logger.info(f"📋 Found {len(script_names)} benchmark step(s):")
    for script_name in script_names:
        benchmark_name = extract_benchmark_name(script_name)
        display_name = get_benchmark_display_name(benchmark_name)
        result_file = get_benchmark_result_file(benchmark_name)
        logger.info(f"   • {display_name} → {result_file}")

    all_success = True
    for script_name in script_names:
        benchmark_name = extract_benchmark_name(script_name)
        display_name = get_benchmark_display_name(benchmark_name)
        result_file = get_benchmark_result_file(benchmark_name)

        success = run_benchmark_step(server, model_config, config, display_name, script_name, result_file, force, logger)
        if not success:
            logger.warning(f"⚠️  Warning: {display_name} failed")
            all_success = False

    if all_success:
        logger.info("🎉 All benchmark steps completed successfully!")
    else:
        logger.warning("⚠️  Some benchmark steps failed")

    return all_success


def download_single_file(server: dict, model_name: str, config: dict, remote_file: str, local_prefix: str) -> bool:
    """Download a single result file from remote server."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)
    model_name_safe = model_name.replace('/', '_')
    server_name = server['name']

    # Create local results directory
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    local_results_dir.mkdir(parents=True, exist_ok=True)

    # Build local and remote paths
    local_file = f"{server_name}_{model_name_safe}_{local_prefix}"
    local_path = local_results_dir / local_file
    remote_path = f"server-benchmark/{remote_file}"

    print(f"📥 Downloading {local_prefix} to: {local_path}")

    scp_cmd = [
        'scp',
        '-i', ssh_key,
        '-P', str(port),
        '-o', 'StrictHostKeyChecking=no',
        f'{address}:{remote_path}',
        str(local_path)
    ]

    try:
        subprocess.run(scp_cmd, check=True)

        # Check if downloaded file is empty
        if local_path.stat().st_size == 0:
            print(f"❌ Error: Downloaded {local_prefix} is empty (benchmark likely failed)")
            # Remove the empty file
            local_path.unlink()
            return False

        print(f"✅ {local_prefix} saved")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to download {local_prefix}: {e}")
        return False


def normalize_model_config(model) -> dict:
    """Normalize model configuration to dict format.

    Args:
        model: Either a string (model name) or dict (model config)

    Returns:
        Dict with 'name' and optional 'tensor_parallel_size'
    """
    if isinstance(model, str):
        return {'name': model, 'tensor_parallel_size': 1}
    elif isinstance(model, dict):
        if 'name' not in model:
            raise ValueError(f"Model config must have 'name' field: {model}")
        # Set default tensor_parallel_size if not specified
        if 'tensor_parallel_size' not in model:
            model['tensor_parallel_size'] = 1
        return model
    else:
        raise ValueError(f"Model must be string or dict, got: {type(model)}")


def run_benchmark_combination(server: dict, model_config: dict, config: dict, force: bool = False, logger: logging.Logger = None) -> tuple:
    """Run a single server-model combination benchmark. Returns (server_name, model_name, success)."""
    server_name = server['name']
    model_name = model_config['name']

    # Use provided logger or get one for this server
    if logger is None:
        logger = get_server_logger(server_name, model_name)

    # Run benchmark (with per-step skip logic)
    success = run_benchmark(server, model_config, config, force, logger)

    return (server_name, model_name, success)


def run_server_benchmarks(server: dict, models: List, config: dict, force: bool = False) -> List[tuple]:
    """Run all benchmarks for a single server sequentially. Returns list of (server_name, model_name, success)."""
    results = []
    server_name = server['name']

    # Create logger for this server (no model context yet)
    logger = get_server_logger(server_name)

    logger.info(f"Starting benchmarks for server: {server_name}")

    # Setup repository with logging
    if not setup_remote_repo(server, logger):
        logger.error(f"Failed to setup repository on {server_name}, skipping all benchmarks for this server")
        for model in models:
            model_config = normalize_model_config(model)
            results.append((server_name, model_config['name'], False))
        return results

    for model in models:
        model_config = normalize_model_config(model)
        model_name = model_config['name']

        logger.info(f"Server: {server_name} | Model: {model_name}")

        # Don't pass logger - let run_benchmark_combination create model-specific logger
        result = run_benchmark_combination(server, model_config, config, force)
        results.append(result)

    logger.info(f"Completed benchmarks for server: {server_name}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run LLM benchmarks on remote servers via SSH for multiple server-model combinations'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force benchmark even if results already exist'
    )
    parser.add_argument(
        '--server',
        help='Run benchmarks only for a specific server (by name)'
    )
    parser.add_argument(
        '--model',
        help='Run benchmarks only for a specific model'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run benchmarks on different servers in parallel (models on same server run sequentially)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel server benchmarks (default: number of servers)'
    )

    args = parser.parse_args()

    # Setup logging
    log_file_path = setup_logging()
    root_logger = logging.getLogger()
    root_logger.info(f"📝 Logging to: {log_file_path}")
    root_logger.info("")

    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)

    # Filter servers based on command-line argument
    servers = config['servers']

    if args.server:
        servers = [s for s in servers if s['name'] == args.server]
        if not servers:
            root_logger.error(f"❌ Error: Server '{args.server}' not found in config.")
            sys.exit(1)

    # Build list of (server, models) tuples for parallel execution
    server_tasks = []
    total_combinations = 0

    for server in servers:
        models = server['models']

        # Filter models if specified
        if args.model:
            # Handle both string and dict model formats
            filtered_models = []
            for m in models:
                model_name = m if isinstance(m, str) else m.get('name')
                if model_name == args.model:
                    filtered_models.append(m)

            if not filtered_models:
                root_logger.warning(f"⚠️  Warning: Model '{args.model}' not found in server '{server['name']}'. Skipping this server.")
                continue

            models = filtered_models

        server_tasks.append((server, models))
        total_combinations += len(models)

    if not server_tasks:
        root_logger.error("❌ Error: No server-model combinations to benchmark.")
        sys.exit(1)

    root_logger.info(f"📊 Running benchmarks for {total_combinations} server-model combination(s) across {len(server_tasks)} server(s)")
    if args.parallel:
        root_logger.info(f"🚀 Parallel mode: Servers will run in parallel (max workers: {args.max_workers or len(server_tasks)})")
    else:
        root_logger.info("⏳ Sequential mode: Servers will run one at a time")
    root_logger.info("")

    # Track results
    results = []

    # Run benchmarks - either in parallel or sequentially
    if args.parallel:
        # Parallel execution: each server runs in its own thread
        max_workers = args.max_workers or len(server_tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all server benchmark tasks
            future_to_server = {
                executor.submit(run_server_benchmarks, server, models, config, args.force): server['name']
                for server, models in server_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_server):
                server_name = future_to_server[future]
                try:
                    server_results = future.result()
                    results.extend(server_results)
                except Exception as exc:
                    root_logger.error(f"❌ Server {server_name} generated an exception: {exc}")
                    # Add failed results for all models on this server
                    for server, models in server_tasks:
                        if server['name'] == server_name:
                            for model in models:
                                results.append((server_name, model, False))
    else:
        # Sequential execution
        for server, models in server_tasks:
            server_results = run_server_benchmarks(server, models, config, args.force)
            results.extend(server_results)

    # Print summary
    root_logger.info("📋 SUMMARY")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    root_logger.info(f"✅ Successful: {len(successful)}/{len(results)}")
    if successful:
        for server_name, model, _ in successful:
            root_logger.info(f"   - {server_name} × {model}")

    if failed:
        root_logger.info("")
        root_logger.info(f"❌ Failed: {len(failed)}/{len(results)}")
        for server_name, model, _ in failed:
            root_logger.info(f"   - {server_name} × {model}")

    root_logger.info("")
    root_logger.info("🎉 All done!")
    root_logger.info(f"📝 Full logs saved to: {log_file_path}")


if __name__ == '__main__':
    main()
