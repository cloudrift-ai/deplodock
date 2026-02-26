"""Planner: group benchmark tasks into execution groups for VM allocation."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class ExecutionGroup:
    """Group of tasks sharing one VM."""

    gpu_name: str
    gpu_count: int
    tasks: list[BenchmarkTask] = field(default_factory=list)


class BenchmarkPlanner(ABC):
    """Abstract base for grouping benchmark tasks into execution groups."""

    @abstractmethod
    def plan(self, tasks: list[BenchmarkTask]) -> list[ExecutionGroup]:
        """Group benchmark tasks into execution groups."""
        ...
