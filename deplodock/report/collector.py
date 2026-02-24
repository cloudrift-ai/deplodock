"""Collect benchmark tasks from manifest-based run directories."""

import json
from pathlib import Path

import yaml


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def collect_tasks_from_manifests(results_path: Path):
    """Scan results_dir for run subdirectories containing manifest.json.

    Yields (task_meta, result_file_path) tuples for each completed task.
    """
    for run_dir in sorted(results_path.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for task in manifest.get("tasks", []):
            if task.get("status") != "completed":
                continue
            result_file = run_dir / task["result_file"]
            if result_file.exists():
                yield task, result_file
