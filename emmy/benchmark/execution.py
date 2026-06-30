"""Benchmark execution: run execution groups on cloud VMs."""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from emmy.benchmark.bench_logging import _get_group_logger, active_run_dir, add_group_file_handler
from emmy.benchmark.command_workload import run_command_workload
from emmy.benchmark.results import compose_json_result
from emmy.benchmark.system_info import collect_system_info
from emmy.benchmark.workload import compose_result, extract_benchmark_results, run_benchmark_workload
from emmy.deploy import (
    DeployParams,
)
from emmy.deploy import (
    deploy as deploy_entry,
)
from emmy.deploy import (
    teardown as teardown_entry,
)
from emmy.deploy.compose import generate_compose
from emmy.planner import BenchmarkTask, ExecutionGroup
from emmy.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
)
from emmy.provisioning.host import RemoteHost
from emmy.provisioning.remote import provision_remote
from emmy.provisioning.ssh_transport import REMOTE_DEPLOY_DIR, make_run_cmd
from emmy.provisioning.staging import stage_to_remote
from emmy.redact import register_secret
from emmy.timing import (
    PHASE_BENCHMARK,
    PHASE_COMMAND,
    PHASE_REMOTE_PROVISION,
    PHASE_TEARDOWN,
    PHASE_VM_PROVISION,
    PhaseTimer,
)

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
    preallocated_conn=None,
    provider: str | None = None,
) -> tuple[list[tuple[BenchmarkTask, bool, dict]], dict | None]:
    """Run all benchmark tasks for one execution group.

    Provisions a VM with group.gpu_count GPUs, then runs each task.
    Tasks with fewer GPUs than the group get gpu_device_ids set.
    Each task's run_dir (set before calling) determines where results go.

    Args:
        on_task_done: Optional async callback invoked after each task completes
            (success or failure). Receives (task, success). Exceptions are
            caught and logged, never propagated.
        preallocated_conn: Optional VMConnectionInfo for a pre-existing host.
            When provided, cloud provisioning and VM deletion are skipped.
            `provision_remote` (Docker, NVIDIA Container Toolkit) still runs;
            each step is idempotent, so already-provisioned hosts are a no-op.

    Returns:
        A tuple of (task_results, instance_info). task_results is a list of
        (BenchmarkTask, bool, timing) triples, where timing is a phase->seconds dict
        (``{}`` for early-failure paths). instance_info is non-None only when
        no_teardown is set and the VM was kept alive.
    """
    task_results: list[tuple[BenchmarkTask, bool, dict]] = []
    instance_info = None
    model_dir = config["benchmark"].get("model_dir", "/hf_models")
    hf_token = os.environ.get("HF_TOKEN", "")
    register_secret(hf_token)
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
    # Per-group provisioning durations (shared across the group's tasks). Seeded into
    # each task's timer so every task's result reflects what it cost to stand up its
    # host. vm_provision is omitted for pre-allocated/fixed/local hosts (no VM created).
    group_timer = PhaseTimer()
    try:
        if preallocated_conn is not None:
            conn = preallocated_conn
            logger.info(f"Using pre-allocated host: {conn.address}:{conn.ssh_port}")
        else:
            async with group_timer.ameasure(PHASE_VM_PROVISION):
                conn = await provision_cloud_vm(
                    group.gpu_name,
                    group.gpu_count,
                    ssh_key,
                    providers_config,
                    server_name=group_label,
                    dry_run=dry_run,
                    logger=logger,
                    provider=provider,
                )
            if conn is None:
                logger.error("VM provisioning failed")
                for task in group.tasks:
                    task_results.append((task, False, {}))
                    await _invoke_callback(on_task_done, task, False, logger)
                return task_results, None

            instance_id_str = f" (instance_id={conn.delete_info[1]})" if conn.delete_info else ""
            logger.info(f"VM provisioned: {conn.address}:{conn.ssh_port}{instance_id_str}")

        first_recipe = group.tasks[0].recipe if group.tasks else None
        host = RemoteHost(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
        async with group_timer.ameasure(PHASE_REMOTE_PROVISION):
            await provision_remote(
                host,
                driver_version=first_recipe.deploy.driver_version if first_recipe else None,
                cuda_version=first_recipe.deploy.cuda_version if first_recipe else None,
            )

        # Collect system info once per execution group
        sysinfo_run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
        system_info = await collect_system_info(sysinfo_run_cmd)

        # Stage repo files once per group for command recipes that request it.
        repo_dir_remote: str | None = None
        stage_paths_union: list[str] = []
        for t in group.tasks:
            if t.recipe.kind == "command" and t.recipe.command and t.recipe.command.stage:
                for p in t.recipe.command.stage:
                    if p not in stage_paths_union:
                        stage_paths_union.append(p)
        if stage_paths_union:
            repo_dir_remote = f"{REMOTE_DEPLOY_DIR}/{group_label}/repo"
            try:
                await stage_to_remote(
                    Path.cwd(),
                    stage_paths_union,
                    conn.address,
                    ssh_key,
                    conn.ssh_port,
                    repo_dir_remote,
                    dry_run=dry_run,
                )
            except Exception as e:
                logger.error(f"Staging failed: {e}")
                for task in group.tasks:
                    task_results.append((task, False, {}))
                    await _invoke_callback(on_task_done, task, False, logger)
                return task_results, None

        for task in group.tasks:
            active_run_dir.set(task.run_dir)
            recipe = task.recipe
            model_name = task.model_name
            result_path = task.result_path()

            # Ensure run_dir exists
            result_path.parent.mkdir(parents=True, exist_ok=True)

            task_logger = _get_group_logger(group, model_name)
            task_logger.info(f"Recipe: {task.recipe_dir} (variant: {task.variant})")

            # Per-task timer, seeded (silently) with the group's provisioning durations.
            task_timer = PhaseTimer()
            for phase_name, seconds in group_timer.phases.items():
                task_timer.record(phase_name, seconds, log=False)

            # Always set explicit GPU device IDs so the container only sees
            # the GPUs the task needs — the provisioned VM may have more GPUs
            # than requested (e.g. B200 only available as 8-GPU instances).
            gpu_device_ids = list(range(task.gpu_count))

            # ── Command-recipe dispatch ───────────────────────────────
            if recipe.kind == "command":
                run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
                # Per-run suffix so concurrent invocations don't collide and stale
                # outputs from previous runs aren't pulled back. We reuse the
                # local run_dir name (a timestamp + short hash) so remote and
                # local artifacts are easy to correlate when debugging.
                run_suffix = task.run_dir.name if task.run_dir is not None else "default"
                task_dir_remote = f"{REMOTE_DEPLOY_DIR}/{group_label}/{task.variant}/{run_suffix}"
                try:
                    async with task_timer.ameasure(PHASE_COMMAND):
                        cmd_success, _info = await run_command_workload(
                            task,
                            run_cmd,
                            repo_dir=repo_dir_remote,
                            task_dir=task_dir_remote,
                            gpu_device_ids=gpu_device_ids,
                            server=conn.address,
                            ssh_key=ssh_key,
                            ssh_port=conn.ssh_port,
                            dry_run=dry_run,
                        )
                except Exception as e:
                    task_logger.error(f"Command workload error: {e}")
                    cmd_success = False
                task_results.append((task, cmd_success, task_timer.as_dict()))
                await _invoke_callback(on_task_done, task, cmd_success, task_logger)
                continue

            params = DeployParams(
                server=conn.address,
                ssh_key=ssh_key,
                ssh_port=conn.ssh_port,
                recipe=recipe,
                model_dir=model_dir,
                hf_token=hf_token,
                dry_run=dry_run,
                gpu_device_ids=gpu_device_ids,
                port_mappings=conn.port_mappings,
            )
            task_logger.info("Deploying model...")
            success = await deploy_entry(params, timer=task_timer)

            if not success:
                task_logger.error("Deploy failed, skipping benchmark")
                task_results.append((task, False, task_timer.as_dict()))
                await _invoke_callback(on_task_done, task, False, task_logger)
                continue

            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
            async with task_timer.ameasure(PHASE_BENCHMARK):
                bench_success, output, stderr, bench_command = await run_benchmark_workload(
                    run_cmd,
                    recipe,
                    dry_run=dry_run,
                )

            # Tear down before persisting results so the teardown duration is captured
            # in the stored timing. The benchmark output is already in memory.
            if not no_teardown:
                task_logger.info("Tearing down...")
                async with task_timer.ameasure(PHASE_TEARDOWN):
                    await teardown_entry(params)

            timing = task_timer.as_dict()
            if bench_success or dry_run:
                if not dry_run:
                    benchmark_output = extract_benchmark_results(output)
                    compose_content = generate_compose(recipe, model_dir, hf_token, gpu_device_ids=gpu_device_ids)
                    full_result = compose_result(task, benchmark_output, compose_content, bench_command, system_info, timing=timing)
                    result_path.write_text(full_result)
                    json_data = compose_json_result(task, benchmark_output, compose_content, bench_command, system_info, timing=timing)
                    task.json_result_path().write_text(json.dumps(json_data, indent=2) + "\n")
                    task_logger.info(f"Results saved to: {result_path}")
                else:
                    task_logger.info(f"[dry-run] Would save results to: {result_path}")
                task_results.append((task, True, timing))
                await _invoke_callback(on_task_done, task, True, task_logger)
            else:
                task_logger.error("Benchmark failed")
                if output:
                    task_logger.error(output)
                if stderr:
                    task_logger.error(stderr)
                task_results.append((task, False, timing))
                await _invoke_callback(on_task_done, task, False, task_logger)

    finally:
        active_run_dir.set(None)
        if group_handler is not None:
            logging.getLogger().removeHandler(group_handler)
            group_handler.close()
        if preallocated_conn is not None:
            logger.info(f"Leaving pre-allocated host in place: {conn.address}")
        elif conn is not None and conn.delete_info:
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


async def _run_groups_on_hosts(
    groups,
    hosts: list,
    config,
    ssh_key,
    dry_run,
    on_task_done: OnTaskDone | None = None,
    provider: str | None = None,
):
    """Run execution groups on a fixed pool of pre-allocated hosts.

    Each group is dispatched to any host whose detected GPU matches the
    group's (gpu_name, gpu_count) requirement. Hosts run at most one group
    at a time; groups are queued until a compatible host is free.
    """
    # Per-host lock so each host serializes its groups.
    locks: dict[int, asyncio.Lock] = {id(h): asyncio.Lock() for h in hosts}
    # Global lock to make host selection atomic.
    select_lock = asyncio.Lock()
    in_use: set[int] = set()

    async def _acquire_host(group):
        while True:
            async with select_lock:
                for h in hosts:
                    if id(h) in in_use:
                        continue
                    if dry_run or h.satisfies(group.gpu_name, group.gpu_count):
                        in_use.add(id(h))
                        return h
            await asyncio.sleep(0.05)

    async def _run_one(group):
        host = await _acquire_host(group)
        try:
            async with locks[id(host)]:
                return await run_execution_group(
                    group,
                    config,
                    ssh_key,
                    dry_run,
                    no_teardown=True,  # never delete fixed hosts
                    on_task_done=on_task_done,
                    preallocated_conn=host.conn,
                    provider=provider,
                )
        finally:
            in_use.discard(id(host))

    return await asyncio.gather(*(_run_one(g) for g in groups), return_exceptions=True)


async def _run_groups(
    groups,
    config,
    ssh_key,
    dry_run,
    max_workers,
    no_teardown=False,
    on_task_done: OnTaskDone | None = None,
    provider: str | None = None,
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
                provider=provider,
            )

    results = await asyncio.gather(
        *(_run_with_semaphore(g) for g in groups),
        return_exceptions=True,
    )
    return results
