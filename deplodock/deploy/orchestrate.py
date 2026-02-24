"""Deploy orchestration: run_deploy, run_teardown, deploy, teardown."""

import sys
import time

from deplodock.deploy.compose import (
    calculate_num_instances,
    generate_compose,
    generate_nginx_conf,
)
from deplodock.deploy.params import DeployParams
from deplodock.provisioning.ssh_transport import make_run_cmd, make_write_file
from deplodock.recipe.types import Recipe


def run_deploy(run_cmd, write_file, recipe: Recipe, model_dir, hf_token, host, dry_run=False, gpu_device_ids=None):
    """Shared deploy orchestration.

    Args:
        run_cmd: callable(command, stream=True) -> (returncode, stdout, stderr)
        write_file: callable(path, content) -> None - writes file to target
        recipe: resolved Recipe dataclass
        model_dir: model cache directory path
        hf_token: HuggingFace token
        host: hostname/IP for endpoint display
        dry_run: if True, skip sleep in health polling
        gpu_device_ids: optional list of GPU device IDs to restrict visibility
    """
    num_instances = calculate_num_instances(recipe)

    model_name = recipe.model_name
    image = recipe.engine.llm.image

    # Generate and write compose file
    compose_content = generate_compose(recipe, model_dir, hf_token, num_instances=num_instances, gpu_device_ids=gpu_device_ids)
    write_file("docker-compose.yaml", compose_content)

    # Generate and write nginx config if multi-instance
    if num_instances > 1:
        nginx_content = generate_nginx_conf(num_instances, engine=recipe.engine.llm.engine_name)
        write_file("nginx.conf", nginx_content)

    port = 8080 if num_instances > 1 else 8000

    # Step 1: Pull images
    print("Pulling images...")
    rc, _, _ = run_cmd("docker compose pull")
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
    rc, _, _ = run_cmd(dl_cmd)
    if rc != 0:
        print("Failed to download model", file=sys.stderr)
        return False

    # Step 3: Clean up old containers
    print("Cleaning up old containers...")
    run_cmd("docker compose down")

    # Step 4: Start services
    print("Starting services...")
    rc, _, _ = run_cmd("docker compose up -d --wait --wait-timeout 1800")
    if rc != 0:
        print("Failed to start services", file=sys.stderr)
        print("Container logs:", file=sys.stderr)
        run_cmd("docker compose logs --tail=100")
        return False

    # Step 5: Poll health
    print("Waiting for health check...")
    health_url = f"http://localhost:{port}/health"
    timeout = 1800  # 30 minutes
    interval = 10
    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = run_cmd(f"curl -sf {health_url}", stream=False)
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
        rc, _, _ = run_cmd(smoke_cmd, stream=False)
        if rc == 0:
            print("Smoke test passed.")
        else:
            print("WARNING: Smoke test failed. The endpoint may not be ready yet.", file=sys.stderr)

    # Print curl example
    print("\nExample curl:")
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
    rc, _, _ = run_cmd("docker compose down")
    if rc == 0:
        print("Teardown complete.")
    else:
        print("Teardown failed.", file=sys.stderr)
    return rc == 0


def deploy(params: DeployParams) -> bool:
    """Deploy a recipe to a server via SSH. Single entry point."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    write_file = make_write_file(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    host = params.server.split("@")[-1] if "@" in params.server else params.server
    return run_deploy(
        run_cmd,
        write_file,
        params.recipe,
        params.model_dir,
        params.hf_token,
        host,
        params.dry_run,
        gpu_device_ids=params.gpu_device_ids,
    )


def teardown(params: DeployParams) -> bool:
    """Teardown containers on a server."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    return run_teardown(run_cmd)
