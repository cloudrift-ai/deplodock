"""Shared deploy logic: recipe loading, compose generation, deploy orchestration."""

import os
import re
import sys
import time

import yaml


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_recipe(recipe_dir, variant=None):
    """Load recipe.yaml from recipe_dir, optionally deep-merging a variant."""
    recipe_path = os.path.join(recipe_dir, "recipe.yaml")
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(recipe_path) as f:
        config = yaml.safe_load(f)

    variants = config.pop("variants", {})

    if variant is not None:
        if variant not in variants:
            available = ", ".join(sorted(variants.keys())) if variants else "none"
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {available}"
            )
        config = deep_merge(config, variants[variant])

    return config


def _parse_variant_gpus(variant):
    """Parse GPU count from variant name like '8xH200' -> 8, 'RTX5090' -> 1."""
    if variant is None:
        return None
    match = re.match(r"^(\d+)x", variant)
    if match:
        return int(match.group(1))
    return 1


def _calculate_num_instances(config, variant):
    """Calculate number of instances from config and variant GPU count."""
    vllm = config["backend"]["vllm"]
    tp = vllm.get("tensor_parallel_size", 1)
    pp = vllm.get("pipeline_parallel_size", 1)
    gpus_per_instance = tp * pp

    variant_gpus = _parse_variant_gpus(variant)
    if variant_gpus is None:
        return 1

    return max(1, variant_gpus // gpus_per_instance)


def generate_compose(config, model_dir, hf_token):
    """Build docker-compose.yaml string from resolved config.

    Single instance: 1 vllm service, count: all GPUs, port 8000.
    Multi-instance: N vllm services with device IDs + nginx on 8080.
    """
    vllm = config["backend"]["vllm"]
    model_name = config["model"]["name"]
    image = vllm.get("image", "vllm/vllm-openai:latest")
    tp = vllm.get("tensor_parallel_size", 1)
    pp = vllm.get("pipeline_parallel_size", 1)
    gpu_mem = vllm.get("gpu_memory_utilization", 0.9)
    extra_args = vllm.get("extra_args", "")
    gpus_per_instance = tp * pp

    # Determine number of instances from config metadata
    num_instances = config.get("_num_instances", 1)

    services = "services:\n"

    for i in range(num_instances):
        if num_instances == 1:
            gpu_config = "count: all"
            port = 8000
        else:
            start = i * gpus_per_instance
            gpu_ids = [str(g) for g in range(start, start + gpus_per_instance)]
            gpu_ids_yaml = ", ".join(f"'{g}'" for g in gpu_ids)
            gpu_config = f"device_ids: [{gpu_ids_yaml}]"
            port = 8000 + i

        extra_args_line = f"\n      {extra_args}" if extra_args.strip() else ""

        services += f"""
  vllm_{i}:
    image: {image}
    container_name: vllm_{i}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              {gpu_config}
              capabilities: [gpu]
    volumes:
      - {model_dir}:{model_dir}
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}
      - HF_HOME={model_dir}
    ports:
      - "{port}:8000"
    shm_size: '16gb'
    ipc: host
    command: >
      --trust-remote-code
      --gpu-memory-utilization={gpu_mem}
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size {tp}
      --pipeline-parallel-size {pp}
      --model {model_name}
      --served-model-name {model_name}{extra_args_line}
    healthcheck:
      test: ["CMD", "bash", "-c", "curl -f http://localhost:8000/health"]
      interval: 10s
      timeout: 10s
      retries: 180
      start_period: 600s
"""

    if num_instances > 1:
        depends = "\n".join(f"      vllm_{i}:\n        condition: service_healthy" for i in range(num_instances))
        services += f"""
  nginx:
    image: nginx:alpine
    container_name: nginx_lb
    ports:
      - "8080:8080"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
{depends}
"""

    return services


def generate_nginx_conf(num_instances):
    """Generate nginx config with least_conn upstream."""
    upstream_servers = "\n".join(
        f"        server vllm_{i}:8000;" for i in range(num_instances)
    )

    return f"""worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    upstream vllm_backend {{
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

            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;

            proxy_buffering off;
        }}
    }}
}}
"""


def run_deploy(run_cmd, write_file, config, model_dir, hf_token, host, variant=None, dry_run=False):
    """Shared deploy orchestration.

    Args:
        run_cmd: callable(command, stream=True) -> (returncode, stdout)
        write_file: callable(path, content) -> None - writes file to target
        config: resolved recipe config dict
        model_dir: model cache directory path
        hf_token: HuggingFace token
        host: hostname/IP for endpoint display
        variant: variant name for instance calculation
        dry_run: if True, skip sleep in health polling
    """
    num_instances = _calculate_num_instances(config, variant)
    config["_num_instances"] = num_instances

    model_name = config["model"]["name"]
    image = config["backend"]["vllm"].get("image", "vllm/vllm-openai:latest")

    # Generate and write compose file
    compose_content = generate_compose(config, model_dir, hf_token)
    write_file("docker-compose.yaml", compose_content)

    # Generate and write nginx config if multi-instance
    if num_instances > 1:
        nginx_content = generate_nginx_conf(num_instances)
        write_file("nginx.conf", nginx_content)

    port = 8080 if num_instances > 1 else 8000

    # Step 1: Pull images
    print("Pulling images...")
    rc, _ = run_cmd("docker compose pull")
    if rc != 0:
        print("Failed to pull images", file=sys.stderr)
        return False

    # Step 2: Download model via huggingface-cli in container
    print(f"Downloading model {model_name}...")
    dl_cmd = (
        f"docker run --rm"
        f" -e HUGGING_FACE_HUB_TOKEN={hf_token}"
        f" -e HF_HOME={model_dir}"
        f" -v {model_dir}:{model_dir}"
        f" --entrypoint bash"
        f" {image}"
        f" -c 'pip install huggingface_hub[cli,hf_transfer] && HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {model_name}'"
    )
    rc, _ = run_cmd(dl_cmd)
    if rc != 0:
        print("Failed to download model", file=sys.stderr)
        return False

    # Step 3: Clean up old containers
    print("Cleaning up old containers...")
    run_cmd("docker compose down")

    # Step 4: Start services
    print("Starting services...")
    rc, _ = run_cmd("docker compose up -d --wait --wait-timeout 1800")
    if rc != 0:
        print("Failed to start services", file=sys.stderr)
        return False

    # Step 5: Poll health
    print("Waiting for health check...")
    health_url = f"http://localhost:{port}/health"
    timeout = 1800  # 30 minutes
    interval = 10
    elapsed = 0
    while elapsed < timeout:
        rc, _ = run_cmd(f"curl -sf {health_url}", stream=False)
        if rc == 0:
            break
        if dry_run:
            break
        time.sleep(interval)
        elapsed += interval
    else:
        print(f"Health check timed out after {timeout}s", file=sys.stderr)
        return False

    # Step 6: Print endpoint info
    status = "dry-run (not deployed)" if dry_run else "deployed"
    print(f"\nEndpoint: http://{host}:{port}/v1")
    print(f"Model: {model_name}")
    print(f"Instances: {num_instances}")
    print(f"Status: {status}")

    # Step 7: Smoke test inference
    if not dry_run:
        print("\nRunning smoke test...")
        smoke_cmd = (
            f"curl -sf http://localhost:{port}/v1/chat/completions"
            f" -H 'Content-Type: application/json'"
            f''' -d '{{"model":"{model_name}","messages":[{{"role":"user","content":"Say hello"}}],"max_tokens":16}}' '''
        )
        rc, _ = run_cmd(smoke_cmd, stream=False)
        if rc == 0:
            print("Smoke test passed.")
        else:
            print("WARNING: Smoke test failed. The endpoint may not be ready yet.", file=sys.stderr)

    # Print curl example
    print(f"\nExample curl:")
    print(
        f"  curl http://{host}:{port}/v1/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\n"
        f'      "model": "{model_name}",\n'
        f'      "messages": [{{"role": "user", "content": "Hello"}}],\n'
        f'      "max_tokens": 64\n'
        f"    }}'"
    )

    return True


def run_teardown(run_cmd):
    """Tear down: docker compose down."""
    print("Tearing down...")
    rc, _ = run_cmd("docker compose down")
    if rc == 0:
        print("Teardown complete.")
    else:
        print("Teardown failed.", file=sys.stderr)
    return rc == 0
