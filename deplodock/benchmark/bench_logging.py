"""Benchmark logging setup."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from deplodock.hardware import gpu_short_name
from deplodock.planner import ExecutionGroup

# Global log file path
LOG_FILE = None


def setup_logging() -> str:
    """Setup logging with timestamped log file and console output.

    Returns:
        Path to the log file
    """
    global LOG_FILE

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = log_dir / f"benchmark_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

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

    return str(LOG_FILE)


def _get_group_logger(group: ExecutionGroup, model_name: str | None = None) -> logging.Logger:
    """Get a logger for an execution group."""
    short = gpu_short_name(group.gpu_name)
    group_label = f"{short}_x_{group.gpu_count}"
    if model_name:
        short_model = model_name.split("/")[-1] if "/" in model_name else model_name
        return logging.getLogger(f"{group_label}.{short_model}")
    return logging.getLogger(group_label)
