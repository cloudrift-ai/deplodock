# Commands Architecture

## Layered Design

```
commands/bench ──► benchmark (config, tasks, execution)
commands/bench ──► deploy (DeployParams, deploy/teardown)
commands/bench ──► provisioning (cloud VM lifecycle)
commands/deploy ─► deploy (DeployParams, deploy/teardown)
commands/deploy ─► provisioning (remote setup, cloud VMs)
commands/vm ────► provisioning (create/delete instances)
```

**Dependency rule:** `commands/` is the CLI-only layer. All reusable business logic lives in top-level library packages:
- `deplodock/recipe/` — recipe loading, dataclass types (`Recipe`, `LLMConfig`, etc.), engine flag mapping
- `deplodock/deploy/` — compose generation, deploy orchestration
- `deplodock/provisioning/` — VM types, SSH polling, shell helpers, cloud providers
- `deplodock/logging_setup.py` — CLI logging setup (`setup_cli_logging()`)
- `deplodock/benchmark/` — tracking, config, logging, workload, tasks, execution
- `deplodock/report/` — parsing, pricing, collection, report generation

## Layers

### `deplodock/recipe/` — Recipe Library

Recipe loading, configuration dataclasses, and engine flag mapping.

**Modules:**
- `types.py` — `Recipe`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`, `SglangConfig`, `BenchmarkConfig` dataclasses
- `recipe.py` — `deep_merge()`, `load_recipe()`, `validate_extra_args()`
- `engines.py` — `VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`, `banned_extra_arg_flags()`, `build_engine_args()`

`load_recipe()` returns a `Recipe` dataclass. All consumers use attribute access (e.g. `recipe.engine.llm.tensor_parallel_size`).

### `deplodock/deploy/` — Deploy Library

The central orchestration layer. Provides a single entry point for deploying recipes to servers.

**Modules:**
- `params.py` — `DeployParams` dataclass (holds `recipe: Recipe`, `gpu_device_ids`, etc.)
- `compose.py` — `calculate_num_instances()`, `generate_compose()`, `generate_nginx_conf()`
- `orchestrate.py` — `run_deploy()`, `run_teardown()`, `deploy()`, `teardown()`
- `ssh.py` — `ssh_base_args()`, `make_run_cmd()`, `scp_file()`, `make_write_file()`
- `local.py` — `make_run_cmd()`, `make_write_file()` (local subprocess transport)

**GPU visibility:** `generate_compose()` accepts a `gpu_device_ids` parameter to restrict GPU visibility via `device_ids: [...]` instead of `count: all`. Used by bench when a task needs fewer GPUs than the VM has.

### `deplodock/provisioning/` — Provisioning Library

VM lifecycle management and cloud provisioning.

**Modules:**
- `types.py` — `VMConnectionInfo` dataclass
- `ssh.py` — `wait_for_ssh()` (provider-agnostic SSH polling)
- `shell.py` — `run_shell_cmd()`
- `remote.py` — `provision_remote()` (install Docker, NVIDIA toolkit)
- `cloud.py` — `resolve_vm_spec()`, `provision_cloud_vm()`, `delete_cloud_vm()`
- `cloudrift.py` — CloudRift REST API provider
- `gcp.py` — GCP gcloud provider

### `deplodock/benchmark/` — Benchmark Library

Benchmark tracking, configuration, task enumeration, and execution.

**Modules:**
- `tracking.py` — `compute_code_hash()`, `create_run_dir()`, `write_manifest()`, `read_manifest()`
- `config.py` — `load_config()`, `validate_config()`, `_expand_path()`
- `bench_logging.py` — `setup_logging()`, `add_file_handler()`, `_get_group_logger()`, `active_run_dir` context var, `_RunDirFilter`, `_BenchConsoleFormatter`
- `workload.py` — `extract_benchmark_results()`, `run_benchmark_workload()`
- `tasks.py` — `enumerate_tasks()`, `_task_meta()`
- `execution.py` — `run_execution_group()`, `_run_groups()`

### `deplodock/report/` — Report Library

Excel report generation from benchmark results.

**Modules:**
- `parser.py` — `parse_benchmark_result()`
- `pricing.py` — `get_gpu_price()`
- `collector.py` — `load_config()`, `collect_tasks_from_manifests()`
- `generator.py` — `generate_report()`

### `planner/` — Planner Layer

Groups benchmark tasks into execution groups for VM allocation.

**Abstract interface (`planner/__init__.py`):**
- `BenchmarkTask` — one recipe+variant combination (recipe_dir, variant, recipe, gpu_name, gpu_count)
- `ExecutionGroup` — group of tasks sharing one VM (gpu_name, gpu_count, tasks)
- `BenchmarkPlanner` — ABC with `plan(tasks) -> list[ExecutionGroup]`

**Implementations:**
- `planner/group_by_model_and_gpu.py` — `GroupByModelAndGpuPlanner`: groups by (model_name, gpu_name) tuple. Same model on same GPU shares a VM to reuse cached weights. Max gpu_count determines VM size; tasks sorted descending.

### `commands/` — CLI Layer (thin handlers only)

Each command module contains only argparse registration and `handle_*` functions that delegate to library packages. CLI handlers use `asyncio.run()` to bridge sync argparse entry points into async internals:

```python
def handle_foo(args):
    asyncio.run(_handle_foo(args))

async def _handle_foo(args):
    await ...
```

**Command modules:**
- `commands/bench/` — `handle_bench()`, `register_bench_command()`
- `commands/deploy/ssh.py` — `handle_ssh()`, `register_ssh_target()`
- `commands/deploy/local.py` — `handle_local()`, `register_local_target()`
- `commands/deploy/cloud.py` — `handle_cloud()`, `register_cloud_target()`
- `commands/report/` — `handle_report()`, `register_report_command()`
- `commands/teardown.py` — `handle_teardown()`, `register_teardown_command()`
- `commands/vm/` — `register_vm_command()`, CLI handlers for each provider

## Data Flow

```
Recipe dirs (positional args)
    |
    v
enumerate_tasks() -> list[BenchmarkTask]
    |
    v
Create per-recipe run directories:
    +-- for each recipe_dir: create_run_dir(recipe_dir)
    +-- assign task.run_dir per task
    |
    v
GroupByModelAndGpuPlanner.plan() -> list[ExecutionGroup]
    |
    v
asyncio.gather(*groups)  -- each group runs as async task:
    |
    v
provision_cloud_vm() -> VMConnectionInfo
    |
    v
For each task in group:
    +-- set gpu_device_ids if task.gpu_count < group.gpu_count
    +-- deploy(DeployParams) -> compose up
    +-- run_benchmark_workload()
    +-- save results
    +-- teardown() (skipped with --no-teardown)
    |
    v
delete_cloud_vm(conn.delete_info) (skipped with --no-teardown; writes instances.json)
```

## CLI Command Tree

```
deplodock
+-- deploy
|   +-- local    -- deploy locally via docker compose
|   +-- ssh      -- deploy to remote server via SSH
|   +-- cloud    -- provision cloud VM + deploy via SSH
+-- bench        -- deploy + benchmark + teardown on cloud VMs
+-- teardown     -- clean up VMs left by bench --no-teardown
+-- report       -- generate Excel reports from benchmark results
+-- vm
    +-- create
    |   +-- gcp
    |   +-- cloudrift
    +-- delete
        +-- gcp
        +-- cloudrift
```

## Adding a New VM Provider

1. Create `provisioning/<provider>.py` with `create_instance()` -> `VMConnectionInfo | None` and `delete_instance()`
2. Add CLI handlers in `commands/vm/<provider>.py`
3. Register CLI in `commands/vm/__init__.py`
4. Add entries to `hardware.py` GPU_INSTANCE_TYPES table
5. Add provider dispatch in `provisioning/cloud.py` `provision_cloud_vm()` and `delete_cloud_vm()`
