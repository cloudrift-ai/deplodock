"""Benchmark logging setup."""

import logging
import sys
from pathlib import Path

from deplodock.hardware import gpu_short_name
from deplodock.planner import ExecutionGroup


def setup_logging():
    """Setup logging with console output only.

    Call add_file_handler() after the run directory is created to attach
    a file handler that writes directly into the run directory.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)

    class CustomConsoleFormatter(logging.Formatter):
        def format(self, record):
            if "." in record.name:
                server, model = record.name.split(".", 1)
                record.name = f"{server}] [{model}"
            return super().format(record)

    console_formatter = CustomConsoleFormatter("[%(name)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def add_file_handler(run_dir: Path) -> str:
    """Add a file handler that writes to {run_dir}/benchmark.log.

    Returns:
        Path to the log file.
    """
    log_file = run_dir / "benchmark.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    return str(log_file)


def _get_group_logger(group: ExecutionGroup, model_name: str | None = None) -> logging.Logger:
    """Get a logger for an execution group."""
    short = gpu_short_name(group.gpu_name)
    group_label = f"{short}_x_{group.gpu_count}"
    if model_name:
        short_model = model_name.split("/")[-1] if "/" in model_name else model_name
        return logging.getLogger(f"{group_label}.{short_model}")
    return logging.getLogger(group_label)
