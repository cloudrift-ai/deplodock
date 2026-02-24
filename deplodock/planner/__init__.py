"""Planner: group benchmark tasks into execution groups for VM allocation."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BenchmarkTask:
    """One recipe+variant combination to benchmark."""

    recipe_dir: str
    variant: str
    recipe_config: dict
    gpu_name: str
    gpu_count: int

    @property
    def model_name(self) -> str:
        return self.recipe_config["model"]["name"]

    @property
    def recipe_name(self) -> str:
        """Basename of the recipe directory (e.g. 'Qwen3-Coder-30B-A3B-Instruct-AWQ')."""
        return os.path.basename(self.recipe_dir)

    def result_path(self, run_dir) -> Path:
        """Full result path: run_dir / recipe_name / {variant}_vllm_benchmark.txt."""
        return Path(run_dir) / self.recipe_name / f"{self.variant}_vllm_benchmark.txt"


@dataclass
class ExecutionGroup:
    """Group of tasks sharing one VM."""

    gpu_name: str
    gpu_count: int
    tasks: List[BenchmarkTask] = field(default_factory=list)


class BenchmarkPlanner(ABC):
    """Abstract base for grouping benchmark tasks into execution groups."""

    @abstractmethod
    def plan(self, tasks: List[BenchmarkTask]) -> List[ExecutionGroup]:
        """Group benchmark tasks into execution groups."""
        ...
