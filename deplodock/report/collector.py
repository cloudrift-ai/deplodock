"""Collect benchmark tasks from run directories."""

from pathlib import Path

import yaml

from deplodock.benchmark.tracking import parse_task_from_result, read_tasks_json


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file) as f:
        return yaml.safe_load(f)


def collect_tasks_from_results(results_path: Path):
    """Scan results_dir for run subdirectories with tasks.json or result files.

    If tasks.json exists, uses it for metadata and checks which result files
    are present. Otherwise, globs for *_benchmark.txt and parses metadata
    from the 'Benchmark Task' YAML section.

    Yields (task_meta, result_file_path) tuples for each completed task.
    """
    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue

        tasks_json_path = run_dir / "tasks.json"
        if tasks_json_path.exists():
            for task in read_tasks_json(run_dir):
                result_file = run_dir / task["result_file"]
                if result_file.exists():
                    yield task, result_file
        else:
            for result_file in sorted(run_dir.glob("*_benchmark.txt")):
                meta = parse_task_from_result(result_file)
                if meta is not None:
                    meta["result_file"] = result_file.name
                    yield meta, result_file
