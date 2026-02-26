"""Benchmark run tracking: code hash, run directories, and task metadata."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import yaml


def compute_code_hash() -> str:
    """SHA256 hash of all .py files under deplodock/, sorted by relative path.

    Concatenates '{relative_path}\n{content}\n' per file.
    Returns full hex digest.
    """
    pkg_dir = Path(__file__).parent.parent
    hasher = hashlib.sha256()
    for py_file in sorted(pkg_dir.rglob("*.py")):
        rel = py_file.relative_to(pkg_dir)
        content = py_file.read_text(encoding="utf-8")
        hasher.update(f"{rel}\n{content}\n".encode())
    return hasher.hexdigest()


def create_run_dir(base_dir: str) -> Path:
    """Create a timestamped run directory: {base_dir}/{YYYY-MM-DD_HH-MM-SS}_{hash[:8]}/.

    Returns the created directory path.
    """
    code_hash = compute_code_hash()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{ts}_{code_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_tasks_json(run_dir, tasks_meta: list[dict]):
    """Write tasks.json to run_dir. Called once at run start.

    Args:
        run_dir: Path to the run directory.
        tasks_meta: List of task identity dicts (no status field).
    """
    tasks_path = Path(run_dir) / "tasks.json"
    tasks_path.write_text(json.dumps(tasks_meta, indent=2) + "\n")


def read_tasks_json(run_dir) -> list[dict]:
    """Read and return parsed tasks.json from run_dir."""
    tasks_path = Path(run_dir) / "tasks.json"
    return json.loads(tasks_path.read_text())


_TASK_SECTION_RE = re.compile(
    r"={3,}\s*Benchmark Task\s*={3,}\s*\n(.*?)\n={3,}",
    re.DOTALL,
)


def parse_task_from_result(result_file) -> dict | None:
    """Parse the 'Benchmark Task' YAML section from a result .txt file.

    Returns a dict with variant, gpu_name, gpu_count, model_name,
    or None if the section is not found.
    """
    content = Path(result_file).read_text()
    m = _TASK_SECTION_RE.search(content)
    if not m:
        return None
    data = yaml.safe_load(m.group(1))
    return {
        "variant": data["variant"],
        "gpu_name": data["gpu_name"],
        "gpu_count": data["gpu_count"],
        "model_name": data["recipe"]["model"]["huggingface"],
    }
