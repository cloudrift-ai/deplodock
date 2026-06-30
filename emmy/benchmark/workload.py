"""Benchmark workload execution."""

import logging
from dataclasses import asdict

import yaml

from emmy.deploy.compose import calculate_num_instances
from emmy.planner import BenchmarkTask
from emmy.recipe.types import Recipe, VllmConfig
from emmy.redact import redact_secrets
from emmy.timing import format_timing

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


def _bench_args(recipe: Recipe) -> list[str]:
    """The vllm bench serve argument list shared by the display string and the
    docker invocation. Embedding recipes target /v1/embeddings via the
    openai-embeddings backend and have no output length (nothing is generated)."""
    bench = recipe.benchmark
    num_instances = calculate_num_instances(recipe)
    port = 8080 if num_instances > 1 else 8000
    args = [
        f"--model {recipe.model_name}",
        "--trust-remote-code",
    ]
    if recipe.is_embedding:
        args += ["--backend openai-embeddings", "--endpoint /v1/embeddings"]
    args += [
        f"--max-concurrency {bench.max_concurrency}",
        f"--num-prompts {bench.num_prompts}",
        f"--random-input-len {bench.random_input_len}",
    ]
    if not recipe.is_embedding:
        args.append(f"--random-output-len {bench.random_output_len}")
    args.append(f"--base-url http://localhost:{port}")
    return args


def build_bench_command(recipe: Recipe) -> str:
    """Build the vllm bench serve command string (without docker wrapper).

    Returns the bench command as a human-readable multi-line string.
    """
    return "vllm bench serve\n" + "\n".join(f"    {a}" for a in _bench_args(recipe))


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
    timing: dict[str, float] | None = None,
) -> str:
    """Assemble the full result file from all sections."""
    sections = [
        _section("Benchmark Task", format_task_yaml(task)),
        benchmark_output.rstrip(),
        _section("Docker Compose Configuration", redact_secrets(compose_content.rstrip())),
        _section("Benchmark Command", bench_command),
    ]
    if timing:
        sections.append(_section("Timing", format_timing(timing)))
    if system_info:
        sections.append(_section("System Information", system_info.rstrip()))
    return "\n\n".join(sections) + "\n"


async def run_benchmark_workload(run_cmd, recipe: Recipe, dry_run=False):
    """Run vllm bench serve on the remote server and return output.

    Returns:
        (success: bool, output: str, stderr: str, bench_command: str)
    """
    bench = recipe.benchmark

    # Warn if input + output lengths risk exceeding context_length (embedding
    # workloads generate nothing — only the input length counts there).
    context_length = recipe.engine.llm.context_length
    request_len = bench.random_input_len + (0 if recipe.is_embedding else bench.random_output_len)
    if context_length is not None and request_len >= context_length:
        logging.getLogger().warning(f"benchmark request length ({request_len}) >= context_length ({context_length})")

    # vllm bench serve is an HTTP client that works against any OpenAI-compatible
    # endpoint, so we always use the vLLM image for benchmarking.
    image = recipe.engine.llm.image
    if recipe.engine.llm.engine_name != "vllm":
        image = VllmConfig().image

    # The ROCm vLLM image crashes on import without GPU devices, even for
    # the pure-HTTP benchmark client.  Pass device flags so it can start.
    is_amd = recipe.deploy.gpu is not None and recipe.deploy.gpu.startswith("AMD")
    device_flags = " --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri" if is_amd else ""

    bench_cmd = (
        f"docker run --rm --network host{device_flags} --entrypoint bash {image} -c 'vllm bench serve {' '.join(_bench_args(recipe))}'"
    )

    bench_command_str = build_bench_command(recipe)
    rc, output, stderr = await run_cmd(bench_cmd, stream=False, timeout=10800)
    return rc == 0, output, stderr, bench_command_str
