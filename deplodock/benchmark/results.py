"""Structured JSON benchmark results: dataclasses and parsers."""

import re
from dataclasses import asdict, dataclass

from deplodock.hardware import gpu_short_name
from deplodock.redact import redact_secrets


@dataclass
class BenchmarkMetrics:
    """Parsed metrics from vLLM bench serve output."""

    successful_requests: int | None = None
    failed_requests: int | None = None
    max_request_concurrency: int | None = None
    benchmark_duration_s: float | None = None
    total_input_tokens: int | None = None
    total_generated_tokens: int | None = None
    request_throughput: float | None = None
    output_token_throughput: float | None = None
    peak_output_token_throughput: float | None = None
    peak_concurrent_requests: float | None = None
    total_token_throughput: float | None = None
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    median_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    median_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    mean_e2el_ms: float | None = None
    median_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None


@dataclass
class SystemInfo:
    """Parsed system information from remote server."""

    hostname: str | None = None
    os: str | None = None
    kernel: str | None = None
    cpu_model: str | None = None
    cpu_count: int | None = None
    cpu_arch: str | None = None
    memory_total_gib: float | None = None
    gpu_name: str | None = None
    gpu_memory_mib: int | None = None
    gpu_driver: str | None = None
    cuda_version: str | None = None
    gpu_count: int | None = None
    docker_version: str | None = None


# (label in bench output, dataclass field name, type constructor)
_METRIC_FIELDS = [
    ("Successful requests", "successful_requests", int),
    ("Failed requests", "failed_requests", int),
    ("Maximum request concurrency", "max_request_concurrency", int),
    ("Benchmark duration (s)", "benchmark_duration_s", float),
    ("Total input tokens", "total_input_tokens", int),
    ("Total generated tokens", "total_generated_tokens", int),
    ("Request throughput (req/s)", "request_throughput", float),
    ("Output token throughput (tok/s)", "output_token_throughput", float),
    ("Peak output token throughput (tok/s)", "peak_output_token_throughput", float),
    ("Peak concurrent requests", "peak_concurrent_requests", float),
    ("Total token throughput (tok/s)", "total_token_throughput", float),
    ("Mean TTFT (ms)", "mean_ttft_ms", float),
    ("Median TTFT (ms)", "median_ttft_ms", float),
    ("P99 TTFT (ms)", "p99_ttft_ms", float),
    ("Mean TPOT (ms)", "mean_tpot_ms", float),
    ("Median TPOT (ms)", "median_tpot_ms", float),
    ("P99 TPOT (ms)", "p99_tpot_ms", float),
    ("Mean ITL (ms)", "mean_itl_ms", float),
    ("Median ITL (ms)", "median_itl_ms", float),
    ("P99 ITL (ms)", "p99_itl_ms", float),
    ("Mean E2EL (ms)", "mean_e2el_ms", float),
    ("Median E2EL (ms)", "median_e2el_ms", float),
    ("P99 E2EL (ms)", "p99_e2el_ms", float),
]


def parse_benchmark_metrics(output: str) -> BenchmarkMetrics:
    """Parse vLLM bench serve output into BenchmarkMetrics."""
    parsed = {}
    for label, field_name, typ in _METRIC_FIELDS:
        m = re.search(rf"{re.escape(label)}:\s+([\d.]+)", output)
        if m:
            try:
                parsed[field_name] = typ(m.group(1))
            except (ValueError, TypeError):
                pass
    return BenchmarkMetrics(**parsed)


def _get_section(raw_text: str, section_name: str) -> str:
    """Extract content between === SECTION === markers."""
    pattern = rf"=== {re.escape(section_name)} ===\n(.*?)(?=\n=== |\Z)"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_memory_total(mem_section: str) -> float | None:
    """Parse total memory from `free -h` output to GiB."""
    # Match the Mem: line, e.g. "Mem:  49Gi  ..."
    m = re.search(r"Mem:\s+([\d.]+)\s*([A-Za-z]+)", mem_section)
    if not m:
        return None
    value = float(m.group(1))
    suffix = m.group(2).lower()
    multipliers = {"gi": 1.0, "g": 1.0 / 1.073741824, "mi": 1.0 / 1024, "ti": 1024.0}
    return round(value * multipliers.get(suffix, 1.0), 2)


def parse_system_info(raw_text: str) -> SystemInfo:
    """Parse system info collected via collect_system_info() into SystemInfo."""
    if not raw_text:
        return SystemInfo()

    fields: dict = {}

    # HOSTNAME
    hostname = _get_section(raw_text, "HOSTNAME")
    if hostname:
        fields["hostname"] = hostname

    # OS
    os_section = _get_section(raw_text, "OS")
    m = re.search(r'PRETTY_NAME="(.+?)"', os_section)
    if m:
        fields["os"] = m.group(1)

    # KERNEL
    kernel = _get_section(raw_text, "KERNEL")
    if kernel:
        fields["kernel"] = kernel

    # CPU INFORMATION
    cpu_section = _get_section(raw_text, "CPU INFORMATION")
    m = re.search(r"Model name:\s+(.+)", cpu_section)
    if m:
        fields["cpu_model"] = m.group(1).strip()
    m = re.search(r"Architecture:\s+(\w+)", cpu_section)
    if m:
        fields["cpu_arch"] = m.group(1)

    # CPU COUNT
    cpu_count = _get_section(raw_text, "CPU COUNT")
    if cpu_count:
        try:
            fields["cpu_count"] = int(cpu_count)
        except ValueError:
            pass

    # MEMORY
    mem_section = _get_section(raw_text, "MEMORY")
    mem_total = _parse_memory_total(mem_section)
    if mem_total is not None:
        fields["memory_total_gib"] = mem_total

    # GPU INFORMATION — CSV: name, memory_mib, driver, pstate, temp, util
    gpu_section = _get_section(raw_text, "GPU INFORMATION")
    if gpu_section and gpu_section != "N/A":
        gpu_lines = [line.strip() for line in gpu_section.strip().splitlines() if line.strip()]
        fields["gpu_count"] = len(gpu_lines)
        if gpu_lines:
            parts = [p.strip() for p in gpu_lines[0].split(",")]
            if len(parts) >= 1:
                fields["gpu_name"] = parts[0]
            if len(parts) >= 2:
                try:
                    fields["gpu_memory_mib"] = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if len(parts) >= 3:
                fields["gpu_driver"] = parts[2]

    # GPU DETAILS — CUDA version
    gpu_details = _get_section(raw_text, "GPU DETAILS")
    m = re.search(r"CUDA Version:\s+([\d.]+)", gpu_details)
    if m:
        fields["cuda_version"] = m.group(1)

    # DOCKER VERSION
    docker_section = _get_section(raw_text, "DOCKER VERSION")
    m = re.search(r"Docker version ([\d.]+)", docker_section)
    if m:
        fields["docker_version"] = m.group(1)

    return SystemInfo(**fields)


def compose_json_result(
    task,
    benchmark_output: str,
    compose_content: str,
    bench_command: str,
    system_info_raw: str,
) -> dict:
    """Assemble the structured JSON result dict from all benchmark data."""
    return {
        "task": {
            "recipe_dir": task.recipe_dir,
            "variant": task.variant,
            "gpu_name": task.gpu_name,
            "gpu_short": gpu_short_name(task.gpu_name),
            "gpu_count": task.gpu_count,
        },
        "recipe": asdict(task.recipe),
        "metrics": asdict(parse_benchmark_metrics(benchmark_output)),
        "system": asdict(parse_system_info(system_info_raw)),
        "compose": redact_secrets(compose_content),
        "bench_command": bench_command,
    }
