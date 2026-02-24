"""Planner: group benchmark tasks into execution groups for VM allocation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from deplodock.hardware import gpu_short_name


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
    def result_filename(self) -> str:
        """Result filename: {gpu_short}_{gpu_count}x_{model_safe}_vllm_benchmark.txt"""
        short = gpu_short_name(self.gpu_name)
        model_safe = self.model_name.replace("/", "_")
        return f"{short}_{self.gpu_count}x_{model_safe}_vllm_benchmark.txt"


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
