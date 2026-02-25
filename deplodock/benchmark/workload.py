"""Benchmark workload execution."""

import logging

from deplodock.deploy.compose import calculate_num_instances
from deplodock.recipe.types import Recipe


def extract_benchmark_results(output: str) -> str:
    """Extract benchmark results section from vllm bench serve output."""
    marker = "============ Serving Benchmark Result ============"
    idx = output.find(marker)
    if idx == -1:
        return output
    return output[idx:]


def run_benchmark_workload(run_cmd, recipe: Recipe, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str)
    """
    bench = recipe.benchmark
    max_concurrency = bench.max_concurrency
    num_prompts = bench.num_prompts
    random_input_len = bench.random_input_len
    random_output_len = bench.random_output_len

    # Warn if input + output lengths risk exceeding context_length
    context_length = recipe.engine.llm.context_length
    if context_length is not None and random_input_len + random_output_len >= context_length:
        logging.getLogger().warning(
            f"benchmark random_input_len ({random_input_len}) + "
            f"random_output_len ({random_output_len}) = "
            f"{random_input_len + random_output_len} >= "
            f"context_length ({context_length})"
        )

    model_name = recipe.model_name
    # vllm bench serve is an HTTP client that works against any OpenAI-compatible
    # endpoint, so we always use the vLLM image for benchmarking.
    image = recipe.engine.llm.image
    if recipe.engine.llm.engine_name != "vllm":
        image = "vllm/vllm-openai:latest"

    num_instances = calculate_num_instances(recipe)
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
