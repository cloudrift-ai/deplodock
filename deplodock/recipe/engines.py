"""Engine flag mapping and CLI argument building."""

from deplodock.recipe.types import LLMConfig

# Maps (recipe field name → CLI flag) for each engine.
# Used to generate command-line args and to derive the banned-flags set.
VLLM_FLAG_MAP = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "context_length": "--max-model-len",
    "max_concurrent_requests": "--max-num-seqs",
}

SGLANG_FLAG_MAP = {
    "tensor_parallel_size": "--tp",
    "pipeline_parallel_size": "--dp",
    "gpu_memory_utilization": "--mem-fraction-static",
    "context_length": "--context-length",
    "max_concurrent_requests": "--max-running-requests",
}

# Flags that must never appear in extra_args (they are emitted from named fields
# or hardcoded by generate_compose).
_HARDCODED_FLAGS = {
    "--trust-remote-code",
    "--host",
    "--port",
    "--model",
    "--model-path",
    "--served-model-name",
}


def banned_extra_arg_flags(engine: str = "vllm") -> set[str]:
    """Return the set of CLI flags that must not appear in extra_args."""
    flag_map = VLLM_FLAG_MAP if engine == "vllm" else SGLANG_FLAG_MAP
    return set(flag_map.values()) | _HARDCODED_FLAGS


def build_engine_args(llm: LLMConfig, model_name: str) -> list[str]:
    """Build the full CLI argument list for the active engine.

    Each element in the returned list is a complete flag-value pair (e.g.
    "--tensor-parallel-size 1") suitable for joining with newlines in a
    docker-compose command block.
    """
    flag_map = VLLM_FLAG_MAP if llm.engine_name == "vllm" else SGLANG_FLAG_MAP
    args = [
        "--trust-remote-code",
        f"--gpu-memory-utilization={llm.gpu_memory_utilization}"
        if llm.engine_name == "vllm"
        else f"--mem-fraction-static {llm.gpu_memory_utilization}",
        "--host 0.0.0.0",
        "--port 8000",
        f"{flag_map['tensor_parallel_size']} {llm.tensor_parallel_size}",
        f"{flag_map['pipeline_parallel_size']} {llm.pipeline_parallel_size}",
        f"--model-path {model_name}" if llm.engine_name == "sglang" else f"--model {model_name}",
        f"--served-model-name {model_name}",
    ]

    if llm.context_length is not None:
        args.append(f"{flag_map['context_length']} {llm.context_length}")

    if llm.max_concurrent_requests is not None:
        args.append(f"{flag_map['max_concurrent_requests']} {llm.max_concurrent_requests}")

    if llm.extra_args.strip():
        args.append(llm.extra_args)

    return args
