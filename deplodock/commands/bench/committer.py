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

    async def __call__(self, task: BenchmarkTask, success: bool) -> None:
        async with self._lock:
            status = "pass" if success else "fail"

            # git add --force the entire run_dir (results may be gitignored)
            rc, _, stderr = await run_shell_cmd(["git", "add", "--force", str(task.run_dir)])
            if rc != 0:
                logger.warning(f"git add failed: {stderr}")
                return

            # Check if there are staged changes
            rc, _, _ = await run_shell_cmd(["git", "diff", "--cached", "--quiet"])
            if rc == 0:
                logger.info(f"No changes to commit for {task.task_id}")
                return

            # Commit
            message = f"bench: {status} {task.task_id}"
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
