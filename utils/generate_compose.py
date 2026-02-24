#!/usr/bin/env python3
"""
Generate docker-compose.yml for multi-instance vLLM deployment with nginx load balancer.
"""

import argparse
import subprocess
import sys


def get_total_gpu_count() -> tuple[int, bool]:
    """Get the total number of GPUs available on the system.

    Returns:
        tuple: (gpu_count, has_mig)
            - gpu_count: Number of usable GPU devices (MIG instances if present, else physical GPUs)
            - has_mig: True if MIG devices are detected
    """
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")

        # Count MIG devices (lines that start with whitespace and contain "MIG")
        mig_devices = [line for line in lines if line.startswith(" ") and "MIG" in line]

        # Count physical GPUs (lines that start with "GPU")
        physical_gpus = [line for line in lines if line.startswith("GPU")]

        # If MIG devices exist, use MIG count; otherwise use physical GPU count
        if mig_devices:
            return len(mig_devices), True
        else:
            return len(physical_gpus), False

    except (subprocess.CalledProcessError, FileNotFoundError):
        # If nvidia-smi is not available, return 0
        return 0, False


def generate_vllm_service(
    instance_id: int,
    gpu_list: str,
    port: int,
    container_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    model_path: str,
    model_name: str,
    hf_directory: str,
    hf_token: str,
    extra_args: str = "",
    num_instances: int = 1,
) -> str:
    """Generate a single vLLM service definition."""
    # Build command with extra args if provided
    extra_args_str = f"\n      {extra_args}" if extra_args.strip() else ""

    # Determine GPU configuration
    # If gpu_list is "all", use count: all
    # Otherwise, use device_ids with specific IDs
    if gpu_list.lower() == "all":
        gpu_config = """count: all"""
    else:
        # Format GPU list for YAML: ['0', '1'] instead of ['0,1']
        gpu_ids_yaml = ", ".join(f"'{gpu}'" for gpu in gpu_list.split(","))
        gpu_config = f"""device_ids: [{gpu_ids_yaml}]"""

    return f"""
  vllm_{instance_id}:
    image: vllm/vllm-openai:latest
    container_name: {container_name}_{instance_id}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              {gpu_config}
              capabilities: [gpu]
    volumes:
      - {hf_directory}:{hf_directory}
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}
      - HF_HOME={hf_directory}
    ports:
      - "{port}:8000"
    shm_size: '16gb'
    ipc: host
    command: >
      --trust-remote-code
      --gpu-memory-utilization=0.9
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size {tensor_parallel_size}
      --pipeline-parallel-size {pipeline_parallel_size}
      --model {model_path}
      --served-model-name {model_name}{extra_args_str}
    healthcheck:
      test: ["CMD", "bash", "-c",
        "curl -f http://localhost:8000/health && curl -f http://localhost:8000/v1/models | grep -q 'object.*list'"]
      interval: 10s
      timeout: 10s
      retries: 180
      start_period: 600s
"""


def generate_nginx_service(num_instances: int, nginx_conf_path: str) -> str:
    """Generate nginx load balancer service definition."""
    depends_on = "\n".join([f"      - vllm_{i}" for i in range(num_instances)])

    return f"""
  nginx:
    image: nginx:alpine
    container_name: nginx_lb
    ports:
      - "8080:8080"
    volumes:
      - {nginx_conf_path}:/etc/nginx/nginx.conf:ro
    depends_on:
{depends_on}
"""


def generate_benchmark_service(hf_directory: str, hf_token: str) -> str:
    """Generate benchmark client service definition."""
    return f"""
  benchmark:
    image: vllm/vllm-openai:latest
    container_name: vllm_benchmark_client
    volumes:
      - {hf_directory}:{hf_directory}
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}
      - HF_HOME={hf_directory}
    entrypoint: ["/bin/bash", "-c"]
    command: ["sleep infinity"]
    profiles:
      - tools
"""


def generate_nginx_conf(num_instances: int) -> str:
    """Generate nginx configuration file content."""
    upstream_servers = "\n".join([f"        server vllm_{i}:8000;" for i in range(num_instances)])

    return f"""# Use all available CPU cores
worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    upstream vllm_backend {{
        # Use least connections for better load distribution
        least_conn;
{upstream_servers}
    }}

    server {{
        listen 8080;

        location / {{
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Increase timeouts for long-running LLM requests
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;

            # Disable buffering for streaming responses
            proxy_buffering off;
        }}
    }}
}}
"""


def calculate_gpu_list(instance_id: int, gpus_per_instance: int) -> str:
    """Calculate comma-separated GPU list for an instance."""
    start = instance_id * gpus_per_instance
    return ",".join(str(g) for g in range(start, start + gpus_per_instance))


def generate_compose_file(
    num_instances: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    container_name: str,
    model_path: str,
    model_name: str,
    hf_directory: str,
    hf_token: str,
    nginx_conf_path: str = None,
    extra_args: str = "",
) -> str:
    """Generate complete docker-compose.yml content.

    Automatically uses 'count: all' instead of specific device IDs when:
    - Single instance (num_instances == 1)
    - GPUs per instance equals total system GPUs
    This is useful for MIG devices and simplifies configuration.
    """

    compose_content = "services:"

    # Calculate total GPUs per instance (tensor_parallel * pipeline_parallel)
    gpus_per_instance = tensor_parallel_size * pipeline_parallel_size

    # Get total GPU count from system (properly handles MIG devices)
    total_gpus, has_mig = get_total_gpu_count()

    # Automatically determine if we should use "all" GPUs:
    # Use "all" if single instance and requesting all system GPUs
    # This works correctly with MIG since we count MIG devices as the total
    use_all_gpus = num_instances == 1 and total_gpus > 0 and gpus_per_instance == total_gpus

    # Add vLLM service instances
    for i in range(num_instances):
        # Use "all" to get all GPUs, or calculate specific device IDs
        gpu_list = "all" if use_all_gpus else calculate_gpu_list(i, gpus_per_instance)
        port = 8000 + i

        compose_content += generate_vllm_service(
            instance_id=i,
            gpu_list=gpu_list,
            port=port,
            container_name=container_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            model_path=model_path,
            model_name=model_name,
            hf_directory=hf_directory,
            hf_token=hf_token,
            extra_args=extra_args,
            num_instances=num_instances,
        )

    # Add nginx load balancer if multiple instances
    if num_instances > 1 and nginx_conf_path:
        compose_content += generate_nginx_service(num_instances, nginx_conf_path)

    # Add benchmark client service
    compose_content += generate_benchmark_service(hf_directory, hf_token)

    return compose_content


def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml for vLLM multi-instance deployment")
    parser.add_argument("--num-instances", type=int, required=True, help="Number of vLLM instances")
    parser.add_argument("--tensor-parallel-size", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size (default: 1)")
    parser.add_argument("--container-name", required=True, help="Base container name")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--hf-directory", required=True, help="HuggingFace cache directory")
    parser.add_argument("--hf-token", default="", help="HuggingFace token")
    parser.add_argument("--extra-args", default="", help='Extra vLLM arguments (e.g., "--enable-expert-parallel --swap-space 16")')
    parser.add_argument("--nginx-conf", default=None, help="Path to nginx config file (for multi-instance)")
    parser.add_argument("--output", "-o", required=True, help="Output docker-compose.yml file path")
    parser.add_argument("--nginx-conf-output", default=None, help="Output nginx.conf file path (for multi-instance)")

    args = parser.parse_args()

    # Generate nginx config if needed
    if args.num_instances > 1:
        if not args.nginx_conf_output:
            print("Error: --nginx-conf-output required for multi-instance setup", file=sys.stderr)
            sys.exit(1)

        nginx_conf_content = generate_nginx_conf(args.num_instances)
        with open(args.nginx_conf_output, "w") as f:
            f.write(nginx_conf_content)

        # Convert to absolute path for Docker Compose
        import os

        nginx_conf_path = os.path.abspath(args.nginx_conf_output)
    else:
        nginx_conf_path = None

    # Generate docker-compose.yml
    compose_content = generate_compose_file(
        num_instances=args.num_instances,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        container_name=args.container_name,
        model_path=args.model_path,
        model_name=args.model_name,
        hf_directory=args.hf_directory,
        hf_token=args.hf_token,
        nginx_conf_path=nginx_conf_path,
        extra_args=args.extra_args,
    )

    with open(args.output, "w") as f:
        f.write(compose_content)

    print(f"Generated docker-compose.yml: {args.output}")
    if nginx_conf_path:
        print(f"Generated nginx.conf: {nginx_conf_path}")


if __name__ == "__main__":
    main()
