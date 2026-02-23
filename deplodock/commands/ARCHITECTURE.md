# Commands Architecture

## Layered Design

```
bench ──────► deploy(DeployParams)          # single entry point
bench ──────► deploy/cloud (VM lifecycle)   # provision/delete
deploy CLI ─► deploy(DeployParams)          # same entry point
deploy/cloud ► vm providers                 # clean interface
```

**Dependency rule:** `bench` never imports from `vm` directly — it goes through `deploy/cloud`.

## Layers

### `deploy/` — Deploy Layer

The central orchestration layer. Provides a single entry point for deploying recipes to servers.

**Public API (`deploy/__init__.py`):**
- `DeployParams` — dataclass with all parameters for a deployment
- `deploy(params: DeployParams) -> bool` — deploy a recipe via SSH
- `teardown(params: DeployParams) -> bool` — teardown containers
- `load_recipe(recipe_dir, variant)` — load and resolve a recipe config
- `run_deploy(run_cmd, write_file, config, ...)` — low-level orchestration (used by `deploy()` and `local`)
- `run_teardown(run_cmd)` — low-level teardown

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

Orchestrates deploy + benchmark + teardown across servers. Uses `deploy()` and `deploy/cloud` — never reaches into vm providers directly.

## Data Flow

```
Recipe entries
    │
    ▼
resolve_vm_spec() ─► (gpu_name, gpu_count, loaded_configs)
    │
    ▼
provision_cloud_vm() ─► VMConnectionInfo
    │                      ├── host, username, ssh_port
    │                      └── delete_info (for cleanup)
    ▼
DeployParams(server=conn.address, ssh_key=..., recipe_config=...)
    │
    ▼
deploy(params) ─► run_deploy() ─► compose up + health check
    │
    ▼
teardown(params) ─► run_teardown() ─► compose down
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
├── bench        — deploy + benchmark + teardown on remote servers
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
