"""GroupByModelAndGpuPlanner: group tasks by (model_name, gpu_name).

Same LLM model on same GPU type shares a VM so model weights are cached
across different GPU-count benchmarks. Different models on the same GPU
type get separate VMs.

With gpu_concurrency > 1, each (model, gpu) group is split into up to N
sub-groups via round-robin, each provisioning its own VM. This trades
weight-cache reuse for wall-clock time.
"""

from deplodock.planner import BenchmarkPlanner, ExecutionGroup


class GroupByModelAndGpuPlanner(BenchmarkPlanner):
    """Group tasks by (model_name, gpu_name) tuple."""

    def __init__(self, gpu_concurrency: int = 1):
        self.gpu_concurrency = max(1, gpu_concurrency)

    def plan(self, tasks):
        groups = {}
        for task in tasks:
            key = (task.model_name, task.gpu_name)
            groups.setdefault(key, []).append(task)

        result = []
        for (_model, gpu), group_tasks in groups.items():
            group_tasks.sort(key=lambda t: t.gpu_count, reverse=True)

            n_splits = min(self.gpu_concurrency, len(group_tasks))
            if n_splits <= 1:
                max_count = max(t.gpu_count for t in group_tasks)
                result.append(
                    ExecutionGroup(
                        gpu_name=gpu,
                        gpu_count=max_count,
                        tasks=group_tasks,
                    )
                )
            else:
                # Round-robin into sub-groups
                sub_groups: list[list] = [[] for _ in range(n_splits)]
                for i, task in enumerate(group_tasks):
                    sub_groups[i % n_splits].append(task)

                for idx, sub in enumerate(sub_groups):
                    if sub:
                        max_count = max(t.gpu_count for t in sub)
                        result.append(
                            ExecutionGroup(
                                gpu_name=gpu,
                                gpu_count=max_count,
                                tasks=sub,
                                index=idx + 1,
                            )
                        )
        return result
