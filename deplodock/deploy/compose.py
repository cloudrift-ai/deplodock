"""Docker Compose and nginx config generation."""


def calculate_num_instances(config):
    """Calculate number of instances from config gpu_count and parallelism."""
    vllm = config["backend"]["vllm"]
    tp = vllm.get("tensor_parallel_size", 1)
    pp = vllm.get("pipeline_parallel_size", 1)
    gpus_per_instance = tp * pp

    gpu_count = config.get("gpu_count")
    if gpu_count is None:
        return 1

    return max(1, gpu_count // gpus_per_instance)


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

    # Optional: restrict GPU visibility to specific device IDs
    device_ids_override = config.get("_gpu_device_ids")

    for i in range(num_instances):
        if num_instances == 1:
            if device_ids_override is not None:
                gpu_ids_yaml = ", ".join(f"'{g}'" for g in device_ids_override)
                gpu_config = f"device_ids: [{gpu_ids_yaml}]"
            else:
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
