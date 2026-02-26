"""Benchmark logging setup."""

import contextvars
import logging
import sys
from pathlib import Path

from deplodock.hardware import gpu_short_name
from deplodock.planner import ExecutionGroup
from deplodock.redact import SecretRedactingFilter

active_run_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar("active_run_dir", default=None)


class _RunDirFilter(logging.Filter):
    """Only pass records that match this handler's run_dir.

    Root-level messages (where active_run_dir is None) go to all handlers.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def filter(self, record):
        current = active_run_dir.get()
        if current is None:
            return True
        return current == self.run_dir


class _BenchConsoleFormatter(logging.Formatter):
    """Console formatter for bench output.

    - ``deplodock.deploy.orchestrate`` → ``[orchestrate]``
    - ``rtx5090_x_1.ModelName`` → ``[rtx5090_x_1] [ModelName]``
    - ``root`` → no prefix (plain message)
    """

    def format(self, record):
        if record.name.startswith("deplodock."):
            # Library logger: show last segment only
            record.name = record.name.rsplit(".", 1)[-1]
        elif "." in record.name:
            # Bench group logger: split into [server] [model]
            server, model = record.name.split(".", 1)
            record.name = f"{server}] [{model}"
        return super().format(record)


def setup_logging():
    """Setup logging with console output only.

    Call add_file_handler() after the run directory is created to attach
    a file handler that writes directly into the run directory.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = _BenchConsoleFormatter("[%(name)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    root_logger.addFilter(SecretRedactingFilter())


def add_file_handler(run_dir: Path) -> str:
    """Add a file handler that writes to {run_dir}/benchmark.log.

    A RunDirFilter is attached so that only messages for this run_dir
    (or root-level messages) are written to the file.

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
    file_handler.addFilter(_RunDirFilter(run_dir))
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
