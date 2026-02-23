"""Bench command: run LLM benchmarks on remote servers via SSH.

Uses the deploy infrastructure (recipe loading, SSH deploy/teardown) instead of
cloning a repo on the remote server.
"""

import json
import logging
import os
import subprocess
import sys
import time
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
from deplodock.hardware import GPU_INSTANCE_TYPES, resolve_instance_type
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

    required_server_fields = ['name', 'ssh_key', 'recipes']
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


def run_benchmark_workload(run_cmd, config, recipe_config, dry_run=False):
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


def _resolve_vm_spec(recipe_entries, server_name):
    """Resolve GPU type and count from recipe entries for VM provisioning.

    All recipes in a server entry must target the same GPU. Uses the max
    gpu_count across all entries.

    Returns:
        (gpu_name, gpu_count, loaded_configs) where loaded_configs is a list
        of (entry, recipe_config) tuples.
    """
    gpu_name = None
    max_gpu_count = 0
    loaded = []

    for entry in recipe_entries:
        recipe_path = entry['recipe']
        variant = entry.get('variant')
        recipe_config = load_recipe(recipe_path, variant=variant)

        entry_gpu = recipe_config.get('gpu')
        entry_gpu_count = recipe_config.get('gpu_count', 1)

        if entry_gpu is None:
            raise ValueError(
                f"Recipe '{recipe_path}' variant '{variant}' is missing 'gpu' field"
            )

        if gpu_name is None:
            gpu_name = entry_gpu
        elif entry_gpu != gpu_name:
            raise ValueError(
                f"Server '{server_name}': mixed GPUs ({gpu_name} vs {entry_gpu}). "
                "All recipes in a server entry must target the same GPU."
            )

        max_gpu_count = max(max_gpu_count, entry_gpu_count)
        loaded.append((entry, recipe_config))

    return gpu_name, max_gpu_count, loaded


def _provision_vm(provider, instance_type, ssh_key, server_name, providers_config, dry_run):
    """Create a VM and return (address, ssh_port, vm_delete_info).

    vm_delete_info is whatever is needed to delete the VM later.
    """
    from deplodock.commands.vm import cloudrift as cr_provider
    from deplodock.commands.vm import gcp_flex_start as gcp_provider

    if provider == "cloudrift":
        api_key = os.environ.get("CLOUDRIFT_API_KEY")
        if not api_key and not dry_run:
            raise RuntimeError("CLOUDRIFT_API_KEY env var required for CloudRift provisioning")

        pub_key_path = f"{ssh_key}.pub"
        logger = logging.getLogger("cloudrift")

        if dry_run:
            logger.info(f"[dry-run] create instance type={instance_type} ssh_key={pub_key_path}")
            return "riftuser@dry-run-host", 22222, ("cloudrift", "dry-run-id")

        # Read public key
        api_key = api_key or ""
        with open(pub_key_path) as f:
            public_key = f.read().strip()
        logger.info(f"SSH public key loaded from {pub_key_path}")

        # Rent instance
        rent_payload = {
            "selector": {"ByInstanceTypeAndLocation": {"instance_type": instance_type}},
            "config": {"VirtualMachine": {
                "ssh_key": {"PublicKeys": [public_key]},
                "image_url": cr_provider.DEFAULT_IMAGE_URL,
                "cloudinit_url": cr_provider.DEFAULT_CLOUDINIT_URL,
            }},
            "with_public_ip": True,
            "ports": ["22", "8000", "8080"],
        }
        logger.info(f"Rent request: {json.dumps(rent_payload, indent=2)}")
        result = cr_provider._rent_instance(
            api_key, instance_type, [public_key], ports=[22, 8000, 8080],
        )
        logger.info(f"Rent response: {json.dumps(result, indent=2)}")

        instance_ids = result.get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("CloudRift: no instance ID returned from rent API")
        instance_id = instance_ids[0]
        logger.info(f"Instance rented (id={instance_id}). Waiting for Active status...")

        # Poll with Inactive detection and per-iteration logging
        timeout = 600
        interval = 10
        elapsed = 0
        info = None
        while elapsed < timeout:
            info = cr_provider._get_instance_info(api_key, instance_id)
            if info is None:
                logger.warning(f"Instance {instance_id} not found (elapsed={elapsed}s)")
            else:
                status = info.get("status")
                node_mode = info.get("node_mode")
                logger.info(f"Poll: status={status} node_mode={node_mode} (elapsed={elapsed}s)")
                if status == "Active":
                    break
                if status == "Inactive":
                    logger.error(f"Instance went Inactive — provisioning failed")
                    logger.error(f"Full instance info: {json.dumps(info, indent=2)}")
                    raise RuntimeError(
                        f"CloudRift instance {instance_id} went Inactive (provisioning failed)"
                    )
            time.sleep(interval)
            elapsed += interval
        else:
            raise RuntimeError(
                f"CloudRift instance {instance_id} did not reach Active within {timeout}s"
            )

        logger.info("Instance is Active")

        # Extract SSH connection info
        host = info.get("host_address")
        port_mappings = info.get("port_mappings", [])
        vms = info.get("virtual_machines", [])
        username = "user"
        if vms:
            login_info = vms[0].get("login_info", {})
            creds = login_info.get("UsernameAndPassword", {})
            username = creds.get("username", username)

        ssh_ext_port = 22
        for mapping in port_mappings:
            if mapping[0] == 22:
                ssh_ext_port = mapping[1]
                break

        address = f"{username}@{host}"
        logger.info(f"VM ready: {address} port {ssh_ext_port}")
        logger.info(f"Port mappings: {port_mappings}")

        # Wait for SSH to become reachable
        ssh_timeout = 120
        ssh_interval = 5
        ssh_elapsed = 0
        logger.info(f"Waiting for SSH to become reachable (up to {ssh_timeout}s)...")
        while ssh_elapsed < ssh_timeout:
            rc = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=5",
                 "-i", ssh_key, "-p", str(ssh_ext_port),
                 address, "true"],
                capture_output=True,
            ).returncode
            if rc == 0:
                logger.info(f"SSH reachable after {ssh_elapsed}s")
                break
            time.sleep(ssh_interval)
            ssh_elapsed += ssh_interval
        else:
            logger.warning(f"SSH not reachable after {ssh_timeout}s, proceeding anyway")

        return address, ssh_ext_port, ("cloudrift", instance_id)

    elif provider == "gcp":
        gcp_config = (providers_config or {}).get("gcp", {})
        zone = gcp_config.get("zone", "us-central1-b")
        logger = logging.getLogger("gcp")

        instance_name = f"bench-{server_name}"

        if dry_run:
            logger.info(f"[dry-run] create instance={instance_name} zone={zone} type={instance_type}")
            return "user@dry-run-gcp-host", 22, ("gcp", instance_name, zone)

        success = gcp_provider.create_instance(
            instance=instance_name,
            zone=zone,
            machine_type=instance_type,
            wait_ssh=True,
        )
        if not success:
            raise RuntimeError(f"GCP: failed to create instance {instance_name}")

        # Get external IP
        from deplodock.commands.vm import run_shell_cmd
        rc, stdout, _ = run_shell_cmd(gcp_provider._gcloud_external_ip_cmd(instance_name, zone))
        if rc != 0 or not stdout.strip():
            raise RuntimeError(f"GCP: failed to get external IP for {instance_name}")

        external_ip = stdout.strip()
        address = external_ip
        logger.info(f"VM ready: {address} port 22")
        return address, 22, ("gcp", instance_name, zone)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _delete_vm(vm_delete_info, dry_run):
    """Delete a VM using the info returned by _provision_vm."""
    from deplodock.commands.vm import cloudrift as cr_provider
    from deplodock.commands.vm import gcp_flex_start as gcp_provider

    provider = vm_delete_info[0]

    if provider == "cloudrift":
        instance_id = vm_delete_info[1]
        if dry_run:
            print(f"[dry-run] cloudrift: terminate instance {instance_id}")
            return
        api_key = os.environ.get("CLOUDRIFT_API_KEY", "")
        cr_provider.delete_instance(api_key, instance_id)

    elif provider == "gcp":
        instance_name = vm_delete_info[1]
        zone = vm_delete_info[2]
        gcp_provider.delete_instance(instance_name, zone, dry_run=dry_run)


def run_server_benchmarks(server: dict, recipe_entries: List[dict], config: dict,
                          force: bool = False, dry_run: bool = False) -> List[tuple]:
    """Run all benchmarks for a single server.

    Auto-provisions a VM based on the GPU requirements in the recipes,
    runs benchmarks, and deletes the VM afterwards.
    """
    results = []
    server_name = server['name']
    ssh_key = expand_path(server['ssh_key'])
    model_dir = config['benchmark'].get('model_dir', '/hf_models')
    hf_token = os.environ.get('HF_TOKEN', '')
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    providers_config = config.get('providers', {})

    logger = get_server_logger(server_name)
    logger.info(f"Starting benchmarks for server: {server_name}")

    # Resolve GPU requirements from recipes
    try:
        gpu_name, gpu_count, loaded_configs = _resolve_vm_spec(recipe_entries, server_name)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to resolve VM spec: {e}")
        for entry in recipe_entries:
            results.append((server_name, entry['recipe'], False))
        return results

    # Look up provider and instance type
    gpu_entries = GPU_INSTANCE_TYPES.get(gpu_name)
    if not gpu_entries:
        logger.error(f"Unknown GPU '{gpu_name}' — not in hardware table")
        for entry in recipe_entries:
            results.append((server_name, entry['recipe'], False))
        return results

    provider, base_type = gpu_entries[0]
    instance_type = resolve_instance_type(provider, base_type, gpu_count)
    logger.info(f"GPU: {gpu_name} x{gpu_count} -> {provider} {instance_type}")

    # Provision VM
    vm_delete_info = None
    try:
        address, ssh_port, vm_delete_info = _provision_vm(
            provider, instance_type, ssh_key, server_name, providers_config, dry_run,
        )
        logger.info(f"VM provisioned: {address}:{ssh_port}")

        # Provision the remote server (install Docker, NVIDIA toolkit, etc.)
        provision_remote(address, ssh_key, ssh_port, dry_run=dry_run)

        for entry, recipe_config in loaded_configs:
            recipe_path = entry['recipe']
            variant = entry.get('variant')

            model_name = recipe_config['model']['name']
            model_name_safe = model_name.replace('/', '_')
            result_filename = f"{server_name}_{model_name_safe}_vllm_benchmark.txt"
            result_path = local_results_dir / result_filename

            recipe_logger = get_server_logger(server_name, model_name)
            recipe_logger.info(f"Recipe: {recipe_path} (variant: {variant})")

            # Check if result already exists
            if not force and result_path.exists():
                recipe_logger.info(f"Result already exists: {result_path}, skipping (use --force to re-run)")
                results.append((server_name, model_name, True))
                continue

            # Create run_cmd / write_file for this server
            run_cmd = make_run_cmd(address, ssh_key, ssh_port, dry_run=dry_run)
            write_file = make_write_file(address, ssh_key, ssh_port, dry_run=dry_run)

            # Extract host from server address
            host = address.split("@")[-1] if "@" in address else address

            # Deploy model
            recipe_logger.info("Deploying model...")
            success = run_deploy(
                run_cmd=run_cmd,
                write_file=write_file,
                config=recipe_config,
                model_dir=model_dir,
                hf_token=hf_token,
                host=host,
                dry_run=dry_run,
            )

            if not success:
                recipe_logger.error("Deploy failed, skipping benchmark")
                results.append((server_name, model_name, False))
                continue

            # Run benchmark
            recipe_logger.info("Running benchmark...")
            bench_success, output = run_benchmark_workload(
                run_cmd, config, recipe_config, dry_run=dry_run,
            )

            if bench_success or dry_run:
                if not dry_run:
                    benchmark_results = extract_benchmark_results(output)
                    local_results_dir.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(benchmark_results)
                    recipe_logger.info(f"Results saved to: {result_path}")
                else:
                    recipe_logger.info(f"[dry-run] Would save results to: {result_path}")
                results.append((server_name, model_name, True))
            else:
                recipe_logger.error("Benchmark failed")
                results.append((server_name, model_name, False))

            # Teardown containers
            recipe_logger.info("Tearing down...")
            run_teardown(run_cmd)

    finally:
        # Always delete the VM
        if vm_delete_info is not None:
            logger.info("Deleting VM...")
            try:
                _delete_vm(vm_delete_info, dry_run)
                logger.info("VM deleted.")
            except Exception as e:
                logger.error(f"Failed to delete VM: {e}")

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
