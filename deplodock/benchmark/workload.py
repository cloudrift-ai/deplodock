"""Benchmark workload execution."""

import logging
from dataclasses import asdict

import yaml

from deplodock.deploy.compose import calculate_num_instances
from deplodock.planner import BenchmarkTask
from deplodock.recipe.types import Recipe
from deplodock.redact import redact_secrets

SECTION_DELIMITER = "=" * 50


def _section(title: str, content: str) -> str:
    """Wrap content in a section with title delimiters."""
    return f"============ {title} ============\n{content}\n{SECTION_DELIMITER}"


def extract_benchmark_results(output: str) -> str:
    """Extract benchmark results section from vllm bench serve output."""
    marker = "============ Serving Benchmark Result ============"
    idx = output.find(marker)
    if idx == -1:
        return output
    return output[idx:]


def build_bench_command(recipe: Recipe) -> str:
    """Build the vllm bench serve command string (without docker wrapper).

    Returns the bench command as a human-readable multi-line string.
    """
    bench = recipe.benchmark
    model_name = recipe.model_name
    num_instances = calculate_num_instances(recipe)
    port = 8080 if num_instances > 1 else 8000

    return (
        f"vllm bench serve\n"
        f"    --model {model_name}\n"
        f"    --max-concurrency {bench.max_concurrency}\n"
        f"    --num-prompts {bench.num_prompts}\n"
        f"    --random-input-len {bench.random_input_len}\n"
        f"    --random-output-len {bench.random_output_len}\n"
        f"    --base-url http://localhost:{port}"
    )


def format_task_yaml(task: BenchmarkTask) -> str:
    """Serialize BenchmarkTask metadata to a YAML block.

    Includes recipe_dir, variant, gpu_name, gpu_count, and the full recipe.
    Excludes run_dir (ephemeral).
    """
    data = {
        "recipe_dir": task.recipe_dir,
        "variant": str(task.variant),
        "gpu_name": task.gpu_name,
        "gpu_count": task.gpu_count,
        "recipe": asdict(task.recipe),
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=False).rstrip()


def compose_result(
    task: BenchmarkTask,
    benchmark_output: str,
    compose_content: str,
    bench_command: str,
    system_info: str,
) -> str:
    """Assemble the full result file from all sections."""
    sections = [
        _section("Benchmark Task", format_task_yaml(task)),
        benchmark_output.rstrip(),
        _section("Docker Compose Configuration", redact_secrets(compose_content.rstrip())),
        _section("Benchmark Command", bench_command),
    ]
    if system_info:
        sections.append(_section("System Information", system_info.rstrip()))
    return "\n\n".join(sections) + "\n"


async def run_benchmark_workload(run_cmd, recipe: Recipe, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str, bench_command: str)
    """
    bench = recipe.benchmark

    # Warn if input + output lengths risk exceeding context_length
    context_length = recipe.engine.llm.context_length
    if context_length is not None and bench.random_input_len + bench.random_output_len >= context_length:
        logging.getLogger().warning(
            f"benchmark random_input_len ({bench.random_input_len}) + "
            f"random_output_len ({bench.random_output_len}) = "
            f"{bench.random_input_len + bench.random_output_len} >= "
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
        f"--max-concurrency {bench.max_concurrency} "
        f"--num-prompts {bench.num_prompts} "
        f"--random-input-len {bench.random_input_len} "
        f"--random-output-len {bench.random_output_len}"
        f"'"
    )

    bench_command_str = build_bench_command(recipe)
    rc, output, _ = await run_cmd(bench_cmd, stream=False, timeout=10800)
    return rc == 0, output, bench_command_str
