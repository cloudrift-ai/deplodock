"""Benchmark execution: run execution groups on cloud VMs."""

import asyncio
import os
from pathlib import Path

from deplodock.benchmark.bench_logging import _get_group_logger
from deplodock.benchmark.tasks import _task_meta
from deplodock.benchmark.workload import extract_benchmark_results, run_benchmark_workload
from deplodock.deploy import (
    DeployParams,
)
from deplodock.deploy import (
    deploy as deploy_entry,
)
from deplodock.deploy import (
    teardown as teardown_entry,
)
from deplodock.hardware import gpu_short_name
from deplodock.planner import ExecutionGroup
from deplodock.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.ssh_transport import make_run_cmd


def run_execution_group(group: ExecutionGroup, config: dict, ssh_key: str, run_dir: Path, dry_run: bool = False) -> list[dict]:
    """Run all benchmark tasks for one execution group.

    Provisions a VM with group.gpu_count GPUs, then runs each task.
    Tasks with fewer GPUs than the group get gpu_device_ids set.

    Returns a list of task metadata dicts for manifest assembly.
    """
    task_results = []
    model_dir = config["benchmark"].get("model_dir", "/hf_models")
    hf_token = os.environ.get("HF_TOKEN", "")
    providers_config = config.get("providers", {})

    short = gpu_short_name(group.gpu_name)
    group_label = f"{short}_x_{group.gpu_count}"
    logger = _get_group_logger(group)
    logger.info(f"Starting group: {group.gpu_name} x{group.gpu_count} ({len(group.tasks)} tasks)")

    conn = None
    try:
        conn = provision_cloud_vm(
            group.gpu_name,
            group.gpu_count,
            ssh_key,
            providers_config,
            server_name=group_label,
            dry_run=dry_run,
            logger=logger,
        )
        if conn is None:
            logger.error("VM provisioning failed")
            for task in group.tasks:
                task_results.append(_task_meta(task, run_dir, status="failed"))
            return task_results

        logger.info(f"VM provisioned: {conn.address}:{conn.ssh_port}")
        provision_remote(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)

        for task in group.tasks:
            recipe = task.recipe
            model_name = task.model_name
            result_path = task.result_path(run_dir)

            # Ensure recipe subdir exists
            result_path.parent.mkdir(parents=True, exist_ok=True)

            task_logger = _get_group_logger(group, model_name)
            task_logger.info(f"Recipe: {task.recipe_dir} (variant: {task.variant})")

            # Set GPU device IDs if task needs fewer GPUs than group
            gpu_device_ids = list(range(task.gpu_count)) if task.gpu_count < group.gpu_count else None

            params = DeployParams(
                server=conn.address,
                ssh_key=ssh_key,
                ssh_port=conn.ssh_port,
                recipe=recipe,
                model_dir=model_dir,
                hf_token=hf_token,
                dry_run=dry_run,
                gpu_device_ids=gpu_device_ids,
            )
            task_logger.info("Deploying model...")
            success = deploy_entry(params)

            if not success:
                task_logger.error("Deploy failed, skipping benchmark")
                task_results.append(_task_meta(task, run_dir, status="failed"))
                continue

            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
            bench_success, output = run_benchmark_workload(
                run_cmd,
                recipe,
                dry_run=dry_run,
            )

            if bench_success or dry_run:
                if not dry_run:
                    benchmark_results = extract_benchmark_results(output)
                    result_path.write_text(benchmark_results)
                    task_logger.info(f"Results saved to: {result_path}")
                else:
                    task_logger.info(f"[dry-run] Would save results to: {result_path}")
                task_results.append(_task_meta(task, run_dir, status="completed"))
            else:
                task_logger.error("Benchmark failed")
                task_results.append(_task_meta(task, run_dir, status="failed"))

            task_logger.info("Tearing down...")
            teardown_entry(params)

    finally:
        if conn is not None and conn.delete_info:
            logger.info("Deleting VM...")
            try:
                delete_cloud_vm(conn.delete_info, dry_run)
                logger.info("VM deleted.")
            except Exception as e:
                logger.error(f"Failed to delete VM: {e}")

    logger.info(f"Completed group: {group_label}")
    return task_results


async def _run_groups(groups, config, ssh_key, run_dir, dry_run, max_workers):
    """Run execution groups concurrently with a semaphore."""
    sem = asyncio.Semaphore(max_workers or len(groups))

    async def _run_with_semaphore(group):
        async with sem:
            return await asyncio.to_thread(
                run_execution_group,
                group,
                config,
                ssh_key,
                run_dir,
                dry_run,
            )

    results = await asyncio.gather(
        *(_run_with_semaphore(g) for g in groups),
        return_exceptions=True,
    )
    return results
