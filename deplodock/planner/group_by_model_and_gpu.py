"""GroupByModelAndGpuPlanner: group tasks by (model_name, gpu_name).

Same LLM model on same GPU type shares a VM so model weights are cached
across different GPU-count benchmarks. Different models on the same GPU
type get separate VMs.
"""

from deplodock.planner import BenchmarkPlanner, ExecutionGroup


class GroupByModelAndGpuPlanner(BenchmarkPlanner):
    """Group tasks by (model_name, gpu_name) tuple."""

    def plan(self, tasks):
        groups = {}
        for task in tasks:
            key = (task.model_name, task.gpu_name)
            groups.setdefault(key, []).append(task)

        result = []
        for (_model, gpu), group_tasks in groups.items():
            group_tasks.sort(key=lambda t: t.gpu_count, reverse=True)
            max_count = max(t.gpu_count for t in group_tasks)
            result.append(
                ExecutionGroup(
                    gpu_name=gpu,
                    gpu_count=max_count,
                    tasks=group_tasks,
                )
            )
        return result
