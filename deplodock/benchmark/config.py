"""Benchmark configuration loading and validation."""

import os
import sys

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def validate_config(config: dict) -> None:
    """Validate that required configuration fields are present."""
    if "benchmark" not in config:
        print("Error: Missing 'benchmark' section in config.")
        sys.exit(1)

    required_benchmark_fields = ["local_results_dir"]
    for field in required_benchmark_fields:
        if field not in config["benchmark"]:
            print(f"Error: Missing '{field}' in 'benchmark' section.")
            sys.exit(1)


def _expand_path(path: str) -> str:
    """Expand user home directory and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))
