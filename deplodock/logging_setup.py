"""CLI logging setup: simple %(message)s format for standalone commands."""

import logging
import sys


def setup_cli_logging():
    """Configure root logger with plain message format for CLI commands.

    Produces output identical to print(). The bench command's setup_logging()
    overrides this with a prefixed format.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
