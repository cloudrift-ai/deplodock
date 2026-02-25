"""Docker Compose and nginx config generation."""

from deplodock.recipe.engines import build_engine_args
from deplodock.recipe.types import Recipe


def calculate_num_instances(recipe: Recipe) -> int:
    """Calculate number of instances from recipe deploy.gpu_count and parallelism."""
    gpus_per_instance = recipe.engine.llm.gpus_per_instance

    if recipe.deploy.gpu_count is None or recipe.deploy.gpu is None:
        return 1

    return max(1, recipe.deploy.gpu_count // gpus_per_instance)


def generate_compose(recipe: Recipe, model_dir, hf_token, num_instances=1, gpu_device_ids=None):
    """Build docker-compose.yaml string from resolved Recipe.

    Single instance: 1 vllm service, count: all GPUs, port 8000.
    Multi-instance: N vllm services with device IDs + nginx on 8080.
    """
    llm = recipe.engine.llm
    model_name = recipe.model_name
    image = llm.image
    engine = llm.engine_name
    entrypoint = llm.entrypoint
    gpus_per_instance = llm.gpus_per_instance

    engine_args = build_engine_args(llm, model_name)
    command_str = "\n      ".join(engine_args)

    services = "services:\n"

    for i in range(num_instances):
        if num_instances == 1:
            if gpu_device_ids is not None:
                gpu_ids_yaml = ", ".join(f"'{g}'" for g in gpu_device_ids)
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

        entrypoint_line = f"\n    entrypoint: {entrypoint}" if entrypoint else ""
        services += f"""
  {engine}_{i}:
    image: {image}
    container_name: {engine}_{i}{entrypoint_line}
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
      {command_str}
    healthcheck:
      test: ["CMD", "bash", "-c", "curl -f http://localhost:8000/health"]
      interval: 10s
      timeout: 10s
      retries: 180
      start_period: 600s
"""

    if num_instances > 1:
        depends = "\n".join(f"      {engine}_{i}:\n        condition: service_healthy" for i in range(num_instances))
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


def generate_nginx_conf(num_instances, engine="vllm"):
    """Generate nginx config with least_conn upstream."""
    upstream_servers = "\n".join(f"        server {engine}_{i}:8000;" for i in range(num_instances))

    return f"""worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    upstream llm_backend {{
        least_conn;
{upstream_servers}
    }}

    server {{
        listen 8080;

        location / {{
            proxy_pass http://llm_backend;
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
