"""Planner: group benchmark tasks into execution groups for VM allocation."""

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from deplodock.hardware import gpu_short_name
from deplodock.recipe.types import Recipe


@dataclass
class BenchmarkTask:
    """One recipe+variant combination to benchmark."""

    recipe_dir: str
    variant: str
    recipe: Recipe
    gpu_name: str
    gpu_count: int
    run_dir: Path | None = None

    @property
    def task_id(self) -> str:
        """Unique task identifier: {recipe_name}/{variant}."""
        return f"{self.recipe_name}/{self.variant}"

    @property
    def model_name(self) -> str:
        return self.recipe.model_name

    @property
    def recipe_name(self) -> str:
        """Basename of the recipe directory (e.g. 'Qwen3-Coder-30B-A3B-Instruct-AWQ')."""
        return os.path.basename(self.recipe_dir)

    def result_path(self) -> Path:
        """Full result path: run_dir / {variant}_{engine}_benchmark.txt."""
        engine = self.recipe.engine.llm.engine_name
        return self.run_dir / f"{self.variant}_{engine}_benchmark.txt"

    def json_result_path(self) -> Path:
        """Full result path: run_dir / {variant}_{engine}_benchmark.json."""
        engine = self.recipe.engine.llm.engine_name
        return self.run_dir / f"{self.variant}_{engine}_benchmark.json"

    def to_dict(self) -> dict:
        """Build a task dict for tasks.json."""
        return {
            "task_id": self.task_id,
            "recipe_dir": self.recipe_dir,
            "variant": self.variant,
            "recipe_name": self.recipe_name,
            "gpu_name": self.gpu_name,
            "gpu_short": gpu_short_name(self.gpu_name),
            "gpu_count": self.gpu_count,
            "model_name": self.model_name,
            "result_file": str(self.result_path().relative_to(self.run_dir)),
            "json_result_file": str(self.json_result_path().relative_to(self.run_dir)),
        }

    def setup_run_dir(self, run_dir: Path) -> None:
        """Assign run_dir and copy recipe.yaml into it."""
        self.run_dir = run_dir
        src = Path(self.recipe_dir) / "recipe.yaml"
        dest = run_dir / "recipe.yaml"
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))

    @staticmethod
    def compute_code_hash() -> str:
        """SHA256 hash of all .py files under deplodock/, sorted by relative path."""
        pkg_dir = Path(__file__).parent.parent
        hasher = hashlib.sha256()
        for py_file in sorted(pkg_dir.rglob("*.py")):
            rel = py_file.relative_to(pkg_dir)
            content = py_file.read_text(encoding="utf-8")
            hasher.update(f"{rel}\n{content}\n".encode())
        return hasher.hexdigest()

    @staticmethod
    def create_run_dir(base_dir: str) -> Path:
        """Create a timestamped run directory: {base_dir}/{YYYY-MM-DD_HH-MM-SS}_{hash[:8]}/."""
        code_hash = BenchmarkTask.compute_code_hash()
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(base_dir) / f"{ts}_{code_hash[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def write_tasks_json(run_dir, tasks: "list[BenchmarkTask]") -> None:
        """Write tasks.json to run_dir from a list of BenchmarkTask objects."""
        tasks_path = Path(run_dir) / "tasks.json"
        tasks_path.write_text(json.dumps([t.to_dict() for t in tasks], indent=2) + "\n")

    @staticmethod
    def read_tasks_json(run_dir) -> list[dict]:
        """Read and return parsed tasks.json from run_dir."""
        tasks_path = Path(run_dir) / "tasks.json"
        return json.loads(tasks_path.read_text())


@dataclass
class ExecutionGroup:
    """Group of tasks sharing one VM."""

    gpu_name: str
    gpu_count: int
    tasks: list[BenchmarkTask] = field(default_factory=list)
    index: int | None = None

    @property
    def gpu_short(self) -> str:
        """Short GPU name (e.g. 'rtx5090')."""
        return gpu_short_name(self.gpu_name)

    @property
    def label(self) -> str:
        """Unique group label (e.g. 'rtx5090_x_8' or 'rtx5090_x_8_r01')."""
        base = f"{self.gpu_short}_x_{self.gpu_count}"
        if self.index is not None:
            return f"{base}_r{self.index:02d}"
        return base


class BenchmarkPlanner(ABC):
    """Abstract base for grouping benchmark tasks into execution groups."""

    @abstractmethod
    def plan(self, tasks: list[BenchmarkTask]) -> list[ExecutionGroup]:
        """Group benchmark tasks into execution groups."""
        ...
