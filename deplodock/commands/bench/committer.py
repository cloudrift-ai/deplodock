"""Git committer for incremental benchmark result commits."""

import asyncio
import logging

from deplodock.planner import BenchmarkTask
from deplodock.provisioning.shell import run_shell_cmd

logger = logging.getLogger(__name__)


class GitCommitter:
    """Async callback that commits result files after each task completes.

    Uses an asyncio.Lock to serialize git operations across concurrent
    execution groups. All failures log warnings, never raise.
    """

    def __init__(self, lock: asyncio.Lock):
        self._lock = lock

    async def __call__(self, task: BenchmarkTask, task_meta: dict) -> None:
        async with self._lock:
            status = task_meta.get("status", "unknown")
            variant = task_meta.get("variant", "unknown")
            model = task_meta.get("model_name", "unknown")

            # Only commit if there's a result file to add
            result_path = task.result_path()
            tasks_json_path = task.run_dir / "tasks.json"

            files_to_add = [str(tasks_json_path)]
            if result_path.exists():
                files_to_add.append(str(result_path))

            # Also add the log file if it exists
            log_path = task.run_dir / "benchmark.log"
            if log_path.exists():
                files_to_add.append(str(log_path))

            # git add --force (results may be gitignored)
            rc, _, stderr = await run_shell_cmd(["git", "add", "--force", *files_to_add])
            if rc != 0:
                logger.warning(f"git add failed: {stderr}")
                return

            # Check if there are staged changes
            rc, _, _ = await run_shell_cmd(["git", "diff", "--cached", "--quiet"])
            if rc == 0:
                logger.info(f"No changes to commit for {variant}")
                return

            # Commit
            message = f"bench: {status} {variant} ({model})"
            rc, _, stderr = await run_shell_cmd(["git", "commit", "-m", message])
            if rc != 0:
                logger.warning(f"git commit failed: {stderr}")
                return

            logger.info(f"Committed: {message}")

            # Push (warn on failure, never raise)
            rc, _, stderr = await run_shell_cmd(["git", "push"], timeout=60)
            if rc != 0:
                logger.warning(f"git push failed: {stderr}")
            else:
                logger.info("Pushed results to remote")
