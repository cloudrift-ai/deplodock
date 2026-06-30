"""Scale-out strategies for adjusting recipes to detected GPU counts."""

import copy
from abc import ABC, abstractmethod

from deplodock.recipe.types import Recipe


class ScaleOutStrategy(ABC):
    """Base class for scale-out strategies."""

    @abstractmethod
    def apply(self, recipe: Recipe, detected_gpu_count: int) -> Recipe:
        """Return a new Recipe adjusted for the detected GPU count.

        Must not mutate the input recipe.
        Raises ValueError if detected_gpu_count < recipe's minimum requirement.
        """


class DataParallelismScaleOutStrategy(ScaleOutStrategy):
    """Scale out by increasing engine-level data parallelism within a single container."""

    def apply(self, recipe: Recipe, detected_gpu_count: int) -> Recipe:
        llm = recipe.engine.llm
        gpus_per_replica = llm.tensor_parallel_size * llm.pipeline_parallel_size
        if detected_gpu_count < gpus_per_replica:
            raise ValueError(
                f"Detected {detected_gpu_count} GPU(s), but recipe requires at least "
                f"{gpus_per_replica} (tp={llm.tensor_parallel_size} * pp={llm.pipeline_parallel_size})"
            )
        new_dp = detected_gpu_count // gpus_per_replica
        result = copy.deepcopy(recipe)
        result.engine.llm.data_parallel_size = new_dp
        result.deploy.gpu_count = detected_gpu_count
        return result


class ReplicaParallelismScaleOutStrategy(ScaleOutStrategy):
    """Scale out by running multiple independent engine containers with nginx load balancing."""

    def apply(self, recipe: Recipe, detected_gpu_count: int) -> Recipe:
        llm = recipe.engine.llm
        gpus_per_replica = llm.gpus_per_instance
        if detected_gpu_count < gpus_per_replica:
            raise ValueError(
                f"Detected {detected_gpu_count} GPU(s), but recipe requires at least "
                f"{gpus_per_replica} per replica "
                f"(tp={llm.tensor_parallel_size} * pp={llm.pipeline_parallel_size} * dp={llm.data_parallel_size})"
            )
        result = copy.deepcopy(recipe)
        result.deploy.gpu_count = detected_gpu_count
        return result


STRATEGIES: dict[str, type[ScaleOutStrategy]] = {
    "data-parallelism": DataParallelismScaleOutStrategy,
    "replica-parallelism": ReplicaParallelismScaleOutStrategy,
}

DEFAULT_STRATEGY = "data-parallelism"
