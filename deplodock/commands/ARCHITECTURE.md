# Commands Architecture

## Layered Design

```
bench ──────► planner (task grouping)       # group tasks into VMs
bench ──────► deploy(DeployParams)          # single entry point
bench ──────► deploy/cloud (VM lifecycle)   # provision/delete
deploy CLI ─► deploy(DeployParams)          # same entry point
deploy/cloud ► vm providers                 # clean interface
```

**Dependency rule:** `bench` never imports from `vm` directly — it goes through `deploy/cloud`.

## Layers

### `planner/` — Planner Layer

Groups benchmark tasks into execution groups for VM allocation.

**Abstract interface (`planner/__init__.py`):**
- `BenchmarkTask` — one recipe+variant combination (recipe_dir, variant, recipe_config, gpu_name, gpu_count)
- `ExecutionGroup` — group of tasks sharing one VM (gpu_name, gpu_count, tasks)
- `BenchmarkPlanner` — ABC with `plan(tasks) -> list[ExecutionGroup]`

**Implementations:**
- `planner/group_by_model_and_gpu.py` — `GroupByModelAndGpuPlanner`: groups by (model_name, gpu_name) tuple. Same model on same GPU shares a VM to reuse cached weights. Max gpu_count determines VM size; tasks sorted descending.

### `deploy/` — Deploy Layer

The central orchestration layer. Provides a single entry point for deploying recipes to servers.

**Public API (`deploy/__init__.py`):**
- `DeployParams` — dataclass with all parameters for a deployment
- `deploy(params: DeployParams) -> bool` — deploy a recipe via SSH
- `teardown(params: DeployParams) -> bool` — teardown containers
- `load_recipe(recipe_dir, variant)` — load and resolve a recipe config
- `run_deploy(run_cmd, write_file, config, ...)` — low-level orchestration (used by `deploy()` and `local`)
- `run_teardown(run_cmd)` — low-level teardown

**GPU visibility:** `generate_compose()` supports `_gpu_device_ids` in the config dict to restrict GPU visibility via `device_ids: [...]` instead of `count: all`. Used by bench when a task needs fewer GPUs than the VM has.

**Targets:**
- `deploy/ssh.py` — SSH target, uses `deploy()` / `teardown()`
- `deploy/local.py` — local target, uses `run_deploy()` / `run_teardown()` directly
- `deploy/cloud.py` — cloud target, provisions VM then uses `deploy()`

### `deploy/cloud.py` — Cloud Bridge

Bridge between deploy and vm layers. Handles VM lifecycle for both `deploy cloud` CLI and `bench`.

**Public API:**
- `resolve_vm_spec(recipe_entries, server_name)` — resolve GPU requirements from recipes
- `provision_cloud_vm(gpu_name, gpu_count, ssh_key, ...)` → `VMConnectionInfo | None`
- `delete_cloud_vm(delete_info, dry_run)` — delete a cloud VM

### `vm/` — VM Provider Layer

Low-level VM lifecycle management. Each provider implements `create_instance()` and `delete_instance()`.

**Shared types (`vm/types.py`):**
- `VMConnectionInfo` — structured return from `create_instance()` with host, username, ssh_port, port_mappings, delete_info

**Shared helpers (`vm/__init__.py`):**
- `wait_for_ssh(host, username, ssh_port, ssh_key_path, ...)` — provider-agnostic SSH readiness check
- `run_shell_cmd(command, dry_run)` — shell command runner

**Providers:**
- `vm/cloudrift.py` — CloudRift REST API
- `vm/gcp_flex_start.py` — GCP flex-start via gcloud CLI

### `bench/` — Benchmark Layer

Orchestrates deploy + benchmark + teardown on cloud VMs. Accepts recipe directories as positional args. Uses the Planner to group tasks into ExecutionGroups, then runs groups in parallel via `asyncio`.

**Data flow:**
1. `enumerate_tasks(recipe_dirs, variants_filter)` → list of `BenchmarkTask`
2. `GroupByModelAndGpuPlanner().plan(tasks)` → list of `ExecutionGroup`
3. `asyncio.run(_run_groups(...))` → concurrent execution via `asyncio.to_thread()`
4. Each group: provision VM → for each task: deploy → benchmark → teardown → delete VM

## Data Flow

```
Recipe dirs (positional args)
    │
    ▼
enumerate_tasks() ─► list[BenchmarkTask]
    │
    ▼
GroupByModelAndGpuPlanner.plan() ─► list[ExecutionGroup]
    │
    ▼
asyncio.gather(*groups)  ── each group runs in a thread:
    │
    ▼
provision_cloud_vm() ─► VMConnectionInfo
    │
    ▼
For each task in group:
    ├── set _gpu_device_ids if task.gpu_count < group.gpu_count
    ├── deploy(DeployParams) ─► compose up
    ├── run_benchmark_workload()
    ├── save results
    └── teardown()
    │
    ▼
delete_cloud_vm(conn.delete_info)
```

## CLI Command Tree

```
deplodock
├── deploy
│   ├── local    — deploy locally via docker compose
│   ├── ssh      — deploy to remote server via SSH
│   └── cloud    — provision cloud VM + deploy via SSH
├── bench        — deploy + benchmark + teardown on cloud VMs
├── report       — generate Excel reports from benchmark results
└── vm
    ├── create
    │   ├── gcp-flex-start
    │   └── cloudrift
    └── delete
        ├── gcp-flex-start
        └── cloudrift
```

## Adding a New VM Provider

1. Create `vm/<provider>.py` with `create_instance()` → `VMConnectionInfo | None` and `delete_instance()`
2. Register CLI handlers in `vm/__init__.py`
3. Add entries to `hardware.py` GPU_INSTANCE_TYPES table
4. Add provider dispatch in `deploy/cloud.py` `provision_cloud_vm()` and `delete_cloud_vm()`
