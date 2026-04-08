"""Benchmark execution: run execution groups on cloud VMs."""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable

from deplodock.benchmark.bench_logging import _get_group_logger, active_run_dir, add_group_file_handler
from deplodock.benchmark.results import compose_json_result
from deplodock.benchmark.system_info import collect_system_info
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
from deplodock.planner import BenchmarkTask, ExecutionGroup
from deplodock.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.ssh_transport import make_run_cmd

OnTaskDone = Callable[[BenchmarkTask, bool], Awaitable[None]]


async def _invoke_callback(
    on_task_done: OnTaskDone | None,
    task: BenchmarkTask,
    success: bool,
    logger: logging.Logger,
):
    """Invoke the on_task_done callback, catching and logging any errors."""
    if on_task_done is None:
        return
    try:
        await on_task_done(task, success)
    except Exception as e:
        logger.warning(f"on_task_done callback failed: {e}")


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


async def run_execution_group(
    group: ExecutionGroup,
    config: dict,
    ssh_key: str,
    dry_run: bool = False,
    no_teardown: bool = False,
    on_task_done: OnTaskDone | None = None,
    server: str | None = None,
    ssh_port: int = 22,
    skip_deploy: bool = False,
) -> tuple[list[tuple[BenchmarkTask, bool]], dict | None]:
    """Run all benchmark tasks for one execution group.

    Provisions a VM with group.gpu_count GPUs, then runs each task.
    Tasks with fewer GPUs than the group get gpu_device_ids set.
    Each task's run_dir (set before calling) determines where results go.

    When ``server`` is provided, VM provisioning and remote setup are
    skipped — the given address is used directly.  When ``skip_deploy``
    is also set, model deployment and per-task teardown are skipped too.

    Args:
        on_task_done: Optional async callback invoked after each task completes
            (success or failure). Receives (task, success). Exceptions are
            caught and logged, never propagated.
        server: SSH address of an existing server (user@host).  When set,
            cloud VM provisioning and remote provisioning are skipped.
        ssh_port: SSH port for the server (default 22).  Only used when
            ``server`` is set.
        skip_deploy: Skip model deployment and teardown.  Only used when
            ``server`` is set.

    Returns:
        A tuple of (task_results, instance_info). task_results is a list of
        (BenchmarkTask, bool) pairs. instance_info is non-None only when
        no_teardown is set and the VM was kept alive.
    """
    task_results: list[tuple[BenchmarkTask, bool]] = []
    instance_info = None
    model_dir = config["benchmark"].get("model_dir", "/hf_models")
    hf_token = os.environ.get("HF_TOKEN", "")
    providers_config = config.get("providers", {})

    group_label = group.label
    logger = _get_group_logger(group)

    # Attach per-group log file if we have a run_dir from the first task
    group_handler = None
    if group.tasks:
        run_dir = group.tasks[0].run_dir
        if run_dir is not None:
            group_handler = add_group_file_handler(run_dir, group_label)

    logger.info(f"Starting group: {group.gpu_name} x{group.gpu_count} ({len(group.tasks)} tasks)")

    # Set active_run_dir early so provisioning logs are captured by the group handler
    if group.tasks and group.tasks[0].run_dir is not None:
        active_run_dir.set(group.tasks[0].run_dir)

    conn = None
    try:
        # Resolve server connection — either provision a cloud VM or use the provided server
        if server:
            address = server
            actual_ssh_port = ssh_port
            port_mappings: list[tuple[int, int]] = []
            logger.info(f"Using existing server: {address}:{actual_ssh_port}")
        else:
            conn = await provision_cloud_vm(
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
                    task_results.append((task, False))
                    await _invoke_callback(on_task_done, task, False, logger)
                return task_results, None

            address = conn.address
            actual_ssh_port = conn.ssh_port
            port_mappings = conn.port_mappings
            instance_id_str = f" (instance_id={conn.delete_info[1]})" if conn.delete_info else ""
            logger.info(f"VM provisioned: {address}:{actual_ssh_port}{instance_id_str}")
            await provision_remote(address, ssh_key, actual_ssh_port, dry_run=dry_run)

        # Collect system info once per execution group
        sysinfo_run_cmd = make_run_cmd(address, ssh_key, actual_ssh_port, dry_run=dry_run)
        system_info = await collect_system_info(sysinfo_run_cmd)

        for task in group.tasks:
            active_run_dir.set(task.run_dir)
            recipe = task.recipe
            model_name = task.model_name
            result_path = task.result_path()

            # Ensure run_dir exists
            result_path.parent.mkdir(parents=True, exist_ok=True)

            task_logger = _get_group_logger(group, model_name)
            task_logger.info(f"Recipe: {task.recipe_dir} (variant: {task.variant})")

            # Always set explicit GPU device IDs so the container only sees
            # the GPUs the task needs — the provisioned VM may have more GPUs
            # than requested (e.g. B200 only available as 8-GPU instances).
            gpu_device_ids = list(range(task.gpu_count))

            if not skip_deploy:
                params = DeployParams(
                    server=address,
                    ssh_key=ssh_key,
                    ssh_port=actual_ssh_port,
                    recipe=recipe,
                    model_dir=model_dir,
                    hf_token=hf_token,
                    dry_run=dry_run,
                    gpu_device_ids=gpu_device_ids,
                    port_mappings=port_mappings,
                )
                task_logger.info("Deploying model...")
                success = await deploy_entry(params)

                if not success:
                    task_logger.error("Deploy failed, skipping benchmark")
                    task_results.append((task, False))
                    await _invoke_callback(on_task_done, task, False, task_logger)
                    continue

            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(address, ssh_key, actual_ssh_port, dry_run=dry_run)
            bench_success, output, stderr, bench_command = await run_benchmark_workload(
                run_cmd,
                recipe,
                dry_run=dry_run,
            )

            if bench_success or dry_run:
                if not dry_run:
                    benchmark_output = extract_benchmark_results(output)
                    if skip_deploy:
                        compose_content = "N/A (server mode)"
                    else:
                        compose_content = generate_compose(recipe, model_dir, hf_token, gpu_device_ids=gpu_device_ids)
                    full_result = compose_result(task, benchmark_output, compose_content, bench_command, system_info)
                    result_path.write_text(full_result)
                    json_data = compose_json_result(task, benchmark_output, compose_content, bench_command, system_info)
                    task.json_result_path().write_text(json.dumps(json_data, indent=2) + "\n")
                    task_logger.info(f"Results saved to: {result_path}")
                else:
                    task_logger.info(f"[dry-run] Would save results to: {result_path}")
                task_results.append((task, True))
                await _invoke_callback(on_task_done, task, True, task_logger)
            else:
                task_logger.error("Benchmark failed")
                if output:
                    task_logger.error(output)
                if stderr:
                    task_logger.error(stderr)
                task_results.append((task, False))
                await _invoke_callback(on_task_done, task, False, task_logger)

            if not skip_deploy and not no_teardown:
                task_logger.info("Tearing down...")
                await teardown_entry(params)

    finally:
        active_run_dir.set(None)
        if group_handler is not None:
            logging.getLogger().removeHandler(group_handler)
            group_handler.close()
        if conn is not None and conn.delete_info:
            if no_teardown:
                instance_info = _build_instance_info(group, group_label, conn)
                logger.info(f"Skipping VM deletion (--no-teardown): {conn.address}")
            else:
                logger.info("Deleting VM...")
                try:
                    await delete_cloud_vm(conn.delete_info, dry_run)
                    logger.info("VM deleted.")
                except Exception as e:
                    logger.error(f"Failed to delete VM: {e}")

    logger.info(f"Completed group: {group_label}")
    return task_results, instance_info


async def _run_groups(
    groups,
    config,
    ssh_key,
    dry_run,
    max_workers,
    no_teardown=False,
    on_task_done: OnTaskDone | None = None,
    server: str | None = None,
    ssh_port: int = 22,
    skip_deploy: bool = False,
):
    """Run execution groups concurrently with a semaphore."""
    sem = asyncio.Semaphore(max_workers or len(groups))

    async def _run_with_semaphore(group):
        async with sem:
            return await run_execution_group(
                group,
                config,
                ssh_key,
                dry_run,
                no_teardown,
                on_task_done,
                server=server,
                ssh_port=ssh_port,
                skip_deploy=skip_deploy,
            )

    results = await asyncio.gather(
        *(_run_with_semaphore(g) for g in groups),
        return_exceptions=True,
    )
    return results
