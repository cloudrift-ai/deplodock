"""Benchmark run tracking: code hash, run directories, and manifests."""

import hashlib
import json
from datetime import datetime
from pathlib import Path


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
        hasher.update(f"{rel}\n{content}\n".encode("utf-8"))
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


def write_manifest(run_dir, timestamp, code_hash, recipes, tasks_metadata):
    """Write manifest.json to run_dir.

    Args:
        run_dir: Path to the run directory.
        timestamp: ISO-format timestamp string.
        code_hash: Full SHA256 hex digest.
        recipes: List of recipe names.
        tasks_metadata: List of task metadata dicts.
    """
    manifest = {
        "timestamp": timestamp,
        "code_hash": code_hash,
        "recipes": recipes,
        "tasks": tasks_metadata,
    }
    manifest_path = Path(run_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def read_manifest(run_dir) -> dict:
    """Read and return parsed manifest.json from run_dir."""
    manifest_path = Path(run_dir) / "manifest.json"
    return json.loads(manifest_path.read_text())
