"""Benchmark library: tracking, config, logging, workload, tasks, execution."""

from deplodock.benchmark.bench_logging import _get_group_logger, add_file_handler, setup_logging
from deplodock.benchmark.config import _expand_path, load_config, validate_config
from deplodock.benchmark.execution import _run_groups, run_execution_group
from deplodock.benchmark.system_info import collect_system_info
from deplodock.benchmark.tasks import _task_meta, enumerate_tasks
from deplodock.benchmark.tracking import (
    compute_code_hash,
    create_run_dir,
    read_manifest,
    write_manifest,
)
from deplodock.benchmark.workload import (
    build_bench_command,
    compose_result,
    extract_benchmark_results,
    format_task_yaml,
    run_benchmark_workload,
)

__all__ = [
    "compute_code_hash",
    "create_run_dir",
    "write_manifest",
    "read_manifest",
    "load_config",
    "validate_config",
    "_expand_path",
    "setup_logging",
    "add_file_handler",
    "_get_group_logger",
    "collect_system_info",
    "build_bench_command",
    "compose_result",
    "extract_benchmark_results",
    "format_task_yaml",
    "run_benchmark_workload",
    "enumerate_tasks",
    "_task_meta",
    "run_execution_group",
    "_run_groups",
]
