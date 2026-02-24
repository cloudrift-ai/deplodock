"""Benchmark workload execution."""

import logging

from deplodock.deploy.compose import calculate_num_instances


def extract_benchmark_results(output: str) -> str:
    """Extract benchmark results section from vllm bench serve output."""
    marker = "============ Serving Benchmark Result ============"
    idx = output.find(marker)
    if idx == -1:
        return output
    return output[idx:]


def _parse_max_model_len(extra_args: str) -> int | None:
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

    rc, output, _ = run_cmd(bench_cmd, stream=False)
    return rc == 0, output
