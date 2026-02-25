"""Benchmark execution: run execution groups on cloud VMs."""

import asyncio
import os

from deplodock.benchmark.bench_logging import _get_group_logger, active_run_dir
from deplodock.benchmark.system_info import collect_system_info
from deplodock.benchmark.tasks import _task_meta
from deplodock.benchmark.workload import compose_result, extract_benchmark_results, run_benchmark_workload
from deplodock.deploy import (
    DeployParams,
)
from deplodock.deploy import (
    deploy as deploy_entry,
)
from deplodock.deploy import (
    teardown as teardown_entry,
)
from deplodock.deploy.compose import generate_compose
from deplodock.hardware import gpu_short_name
from deplodock.planner import ExecutionGroup
from deplodock.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.ssh_transport import make_run_cmd


def _build_instance_info(group: ExecutionGroup, group_label: str, conn) -> dict:
    """Build instance info dict from a VM connection for instances.json."""
    info = {
        "group_label": group_label,
        "gpu_name": group.gpu_name,
        "gpu_count": group.gpu_count,
        "address": conn.address,
        "ssh_port": conn.ssh_port,
    }

    if conn.delete_info:
        provider = conn.delete_info[0]
        info["provider"] = provider
        info["instance_id"] = conn.delete_info[1]
        if provider == "gcp" and len(conn.delete_info) > 2:
            info["zone"] = conn.delete_info[2]

    return info


def run_execution_group(
    group: ExecutionGroup,
    config: dict,
    ssh_key: str,
    dry_run: bool = False,
    no_teardown: bool = False,
) -> tuple[list[dict], dict | None]:
    """Run all benchmark tasks for one execution group.

    Provisions a VM with group.gpu_count GPUs, then runs each task.
    Tasks with fewer GPUs than the group get gpu_device_ids set.
    Each task's run_dir (set before calling) determines where results go.

    Returns:
        A tuple of (task_metadata_list, instance_info). instance_info is
        non-None only when no_teardown is set and the VM was kept alive.
    """
    task_results = []
    instance_info = None
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
                task_results.append(_task_meta(task, status="failed"))
            return task_results, None

        logger.info(f"VM provisioned: {conn.address}:{conn.ssh_port}")
        provision_remote(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)

        # Collect system info once per execution group
        sysinfo_run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
        system_info = collect_system_info(sysinfo_run_cmd)

        for task in group.tasks:
            active_run_dir.set(task.run_dir)
            recipe = task.recipe
            model_name = task.model_name
            result_path = task.result_path()

            # Ensure run_dir exists
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
                task_results.append(_task_meta(task, status="failed"))
                continue

            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
            bench_success, output, bench_command = run_benchmark_workload(
                run_cmd,
                recipe,
                dry_run=dry_run,
            )

            if bench_success or dry_run:
                if not dry_run:
                    benchmark_output = extract_benchmark_results(output)
                    compose_content = generate_compose(recipe, model_dir, hf_token, gpu_device_ids=gpu_device_ids)
                    full_result = compose_result(task, benchmark_output, compose_content, bench_command, system_info)
                    result_path.write_text(full_result)
                    task_logger.info(f"Results saved to: {result_path}")
                else:
                    task_logger.info(f"[dry-run] Would save results to: {result_path}")
                task_results.append(_task_meta(task, status="completed"))
            else:
                task_logger.error("Benchmark failed")
                task_results.append(_task_meta(task, status="failed"))

            if not no_teardown:
                task_logger.info("Tearing down...")
                teardown_entry(params)

    finally:
        active_run_dir.set(None)
        if conn is not None and conn.delete_info:
            if no_teardown:
                instance_info = _build_instance_info(group, group_label, conn)
                logger.info(f"Skipping VM deletion (--no-teardown): {conn.address}")
            else:
                logger.info("Deleting VM...")
                try:
                    delete_cloud_vm(conn.delete_info, dry_run)
                    logger.info("VM deleted.")
                except Exception as e:
                    logger.error(f"Failed to delete VM: {e}")

    logger.info(f"Completed group: {group_label}")
    return task_results, instance_info


async def _run_groups(groups, config, ssh_key, dry_run, max_workers, no_teardown=False):
    """Run execution groups concurrently with a semaphore."""
    sem = asyncio.Semaphore(max_workers or len(groups))

    async def _run_with_semaphore(group):
        async with sem:
            return await asyncio.to_thread(
                run_execution_group,
                group,
                config,
                ssh_key,
                dry_run,
                no_teardown,
            )

    results = await asyncio.gather(
        *(_run_with_semaphore(g) for g in groups),
        return_exceptions=True,
    )
    return results
