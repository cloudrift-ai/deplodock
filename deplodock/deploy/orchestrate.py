"""Deploy orchestration: run_deploy, run_teardown, deploy, teardown."""

import asyncio
import json
import logging

from deplodock.deploy.compose import (
    calculate_num_instances,
    generate_compose,
    generate_nginx_conf,
)
from deplodock.deploy.params import DeployParams
from deplodock.provisioning.ssh_transport import make_run_cmd, make_write_file
from deplodock.recipe.types import Recipe

logger = logging.getLogger(__name__)


async def run_deploy(run_cmd, write_file, recipe: Recipe, model_dir, hf_token, host, dry_run=False, gpu_device_ids=None):
    """Shared deploy orchestration.

    Args:
        run_cmd: async callable(command, stream=True, timeout=600) -> (returncode, stdout, stderr)
        write_file: async callable(path, content) -> None - writes file to target
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
    await write_file("docker-compose.yaml", compose_content)

    # Generate and write nginx config if multi-instance
    if num_instances > 1:
        nginx_content = generate_nginx_conf(num_instances, engine=recipe.engine.llm.engine_name)
        await write_file("nginx.conf", nginx_content)

    port = 8080 if num_instances > 1 else 8000

    # Step 1: Pull images
    logger.info("Pulling images...")
    rc, _, _ = await run_cmd("docker compose pull", timeout=1800, log_output=True)
    if rc != 0:
        logger.error("Failed to pull images")
        return False

    # Step 2: Download model via huggingface-cli in container
    logger.info(f"Downloading model {model_name}...")
    dl_cmd = (
        f"docker run --rm"
        f" -e HUGGING_FACE_HUB_TOKEN={hf_token}"
        f" -e HF_HOME={model_dir}"
        f" -v {model_dir}:{model_dir}"
        f" --entrypoint bash"
        f" {image}"
        f" -c 'pip install huggingface_hub[cli,hf_transfer] && HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {model_name}'"
    )
    rc, _, _ = await run_cmd(dl_cmd, timeout=3600, log_output=True)
    if rc != 0:
        logger.error("Failed to download model")
        return False

    # Step 3: Clean up old containers
    logger.info("Cleaning up old containers...")
    await run_cmd("docker compose down", timeout=300, log_output=True)

    # Step 4: Start services
    logger.info("Starting services...")
    rc, _, _ = await run_cmd("docker compose up -d --wait --wait-timeout 1800", timeout=1800, log_output=True)
    if rc != 0:
        logger.error("Failed to start services")
        logger.error("Container logs:")
        await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
        return False

    # Step 5: Poll health
    logger.info("Waiting for health check...")
    health_url = f"http://localhost:{port}/health"
    timeout = 1800  # 30 minutes
    interval = 10
    elapsed = 0
    while elapsed < timeout:
        rc, _, _ = await run_cmd(f"curl -sf {health_url}", stream=False, timeout=30, log_output=True)
        if rc == 0:
            break
        if dry_run:
            break
        await asyncio.sleep(interval)
        elapsed += interval
    else:
        logger.error(f"Health check timed out after {timeout}s")
        return False

    # Step 6: Print endpoint info
    status = "dry-run (not deployed)" if dry_run else "deployed"
    logger.info(f"\nEndpoint: http://{host}:{port}/v1")
    logger.info(f"Model: {model_name}")
    logger.info(f"Instances: {num_instances}")
    logger.info(f"Status: {status}")

    # Step 7: Smoke test inference (retry — first request may be slow due to warmup)
    # Asks a trivial factual question and checks the answer to detect broken models
    # (e.g. wrong quantization producing garbage output).
    if not dry_run:
        logger.info("\nRunning smoke test...")
        prompt = "What is 2+2? Answer with just the number."
        smoke_cmd = (
            f"curl -s http://localhost:{port}/v1/chat/completions"
            f" -H 'Content-Type: application/json'"
            f''' -d '{{"model":"{model_name}","messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":16}}' '''
        )
        smoke_timeout = 600
        smoke_interval = 10
        deadline = asyncio.get_event_loop().time() + smoke_timeout
        while asyncio.get_event_loop().time() < deadline:
            rc, stdout, _ = await run_cmd(smoke_cmd, stream=False, timeout=180)
            if rc != 0 or not stdout.strip():
                # Server not ready yet, keep retrying
                await asyncio.sleep(smoke_interval)
                continue
            try:
                body = json.loads(stdout)
                answer = body["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                # Malformed response, server may still be starting
                await asyncio.sleep(smoke_interval)
                continue
            if "4" in answer:
                logger.info("Smoke test passed.")
                break
            # Model returned a valid response but wrong answer — broken model
            logger.error(f"Smoke test failed: model returned wrong answer: {answer!r}")
            logger.error("Container logs:")
            await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
            return False
        else:
            logger.error(f"Smoke test timed out after {smoke_timeout}s. The endpoint is not ready.")
            logger.error("Container logs:")
            await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
            return False

    # Print curl example
    logger.info("\nExample curl:")
    logger.info(
        f"  curl http://{host}:{port}/v1/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\n"
        f'      "model": "{model_name}",\n'
        f'      "messages": [{{"role": "user", "content": "Hello"}}],\n'
        f'      "max_tokens": 64\n'
        f"    }}'"
    )

    return True


async def run_teardown(run_cmd):
    """Tear down: docker compose down."""
    logger.info("Tearing down...")
    rc, _, _ = await run_cmd("docker compose down", timeout=300, log_output=True)
    if rc == 0:
        logger.info("Teardown complete.")
    else:
        logger.error("Teardown failed.")
    return rc == 0


async def deploy(params: DeployParams) -> bool:
    """Deploy a recipe to a server via SSH. Single entry point."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    write_file = make_write_file(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    host = params.server.split("@")[-1] if "@" in params.server else params.server
    return await run_deploy(
        run_cmd,
        write_file,
        params.recipe,
        params.model_dir,
        params.hf_token,
        host,
        params.dry_run,
        gpu_device_ids=params.gpu_device_ids,
    )


async def teardown(params: DeployParams) -> bool:
    """Teardown containers on a server."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    return await run_teardown(run_cmd)
