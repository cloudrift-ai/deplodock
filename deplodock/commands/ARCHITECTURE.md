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
- `deplodock/config.py` — the single owner of `os.environ` for all `DEPLODOCK_*` config vars. Typed getters
  (`tune_db_path`, `nvcc_flags`, `debug_enabled`, `dump_dir`, `tune_patience`, `bench_backends_raw`, `cubin_cache_dir`,
  …) read the env live; `set_nvcc_flags(cli_value, default)` holds the `--nvcc-flags` > env > command-default precedence
  that used to live in this CLI layer, so every callsite (CLI, programmatic, tests) shares it. The thin
  `compile.apply_nvcc_flags` / `compile.resolve_tune_db` wrappers just adapt argparse to it. (Provider/secret vars stay
  with `redact.py`; the dynamic `DEPLODOCK_<KNOB>` namespace stays with `compiler/pipeline/knob.py`, which borrows
  `config.knob_var` / `config.knob_raw`.)
- `deplodock/redact.py` — `redact_secrets()`, `SecretRedactingFilter`, `install_redaction()` (attach the filter to a handler — must be a handler, not a logger, so child-logger records that propagate up are still redacted), `register_secret()` (call after resolving any secret from a CLI flag — `--hf-token`, `--api-key` — or env var so its value is added to the redaction set)
- `deplodock/benchmark/` — config, logging, workload, tasks, execution, structured JSON results

## Layers

### `deplodock/recipe/` — Recipe Library

Recipe loading, configuration dataclasses, and engine flag mapping.

**Modules:**
- `types.py` — `Recipe`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`, `SglangConfig`, `BenchmarkConfig` dataclasses
- `recipe.py` — `deep_merge()`, `load_recipe()`, `resolve_for_hardware()`, `validate_extra_args()`
- `engines.py` — `VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`, `banned_extra_arg_flags()`, `build_engine_args()`

`load_recipe()` returns a `Recipe` dataclass. All consumers use attribute access (e.g. `recipe.engine.llm.tensor_parallel_size`).

### `deplodock/deploy/` — Deploy Library

The central orchestration layer. Provides a single entry point for deploying recipes to servers.

**Modules:**
- `params.py` — `DeployParams` dataclass (holds `recipe: Recipe`, `gpu_device_ids`, etc.)
- `compose.py` — `calculate_num_instances()`, `generate_compose()`, `generate_nginx_conf()`
- `orchestrate.py` — `run_deploy()`, `run_teardown()`, `deploy()`, `teardown()`. `run_deploy()` / `deploy()` accept an
  optional `timer: PhaseTimer` that records per-step durations (see [Timing metrics](#timing-metrics))
- `log_phases.py` — `parse_engine_load_phases()` (best-effort `weights_load` / `cuda_graph_capture` from container logs)
- `ssh.py` — `ssh_base_args()`, `make_run_cmd()`, `scp_file()`, `make_write_file()`
- `scale_out.py` — `ScaleOutStrategy` ABC, `DataParallelismScaleOutStrategy`, `ReplicaParallelismScaleOutStrategy`, `STRATEGIES`, `DEFAULT_STRATEGY`
- `local.py` — `make_run_cmd()`, `make_write_file()` (local subprocess transport)

**GPU visibility:** `generate_compose()` accepts a `gpu_device_ids` parameter to restrict GPU visibility via `device_ids: [...]` instead of `count: all`. Used by bench when a task needs fewer GPUs than the VM has.

### `deplodock/provisioning/` — Provisioning Library

VM lifecycle management and cloud provisioning.

**Modules:**
- `types.py` — `VMConnectionInfo` dataclass
- `ssh.py` — `wait_for_ssh()` (provider-agnostic SSH polling)
- `shell.py` — `run_shell_cmd()`
- `host.py` — `Host`, `LocalHost`, `RemoteHost` (sudo-gated command runner — `LocalHost.run(sudo=True)` raises so local deploys can't modify the dev box)
- `remote.py` — `provision_remote()` (install Docker, optional NVIDIA driver/CUDA, NVIDIA container toolkit; reboots and waits for the host on driver/CUDA install)
- `cloud.py` — `resolve_vm_spec()`, `provision_cloud_vm()`, `delete_cloud_vm()`
- `cloudrift.py` — CloudRift REST API provider
- `gcp.py` — GCP gcloud provider

### `deplodock/benchmark/` — Benchmark Library

Benchmark configuration, task enumeration, and execution.

**Modules:**
- `config.py` — `load_config()`, `validate_config()`, `_expand_path()`
- `bench_logging.py` — `setup_logging()`, `add_file_handler()`, `add_group_file_handler()`, `_get_group_logger()`, `active_run_dir` context var, `_RunDirFilter`, `_GroupNameFilter`, `_BenchConsoleFormatter`
- `workload.py` — `extract_benchmark_results()`, `run_benchmark_workload()`
- `tasks.py` — `enumerate_tasks()`
- `execution.py` — `run_execution_group()`, `_run_groups()`, `OnTaskDone` callback type. Times provisioning (per
  group) + deploy/bench/teardown (per task); task results are `(task, ok, timing)` triples
- `results.py` — `BenchmarkMetrics`, `SystemInfo` dataclasses, `parse_benchmark_metrics()`, `parse_system_info()`,
  `compose_json_result()` (optional `timing` arg → `"timing"` key)
- `workload.py` — `compose_result()` (optional `timing` arg → `=== Timing ===` section), `run_benchmark_workload()`

### `planner/` — Planner Layer

Groups benchmark tasks into execution groups for VM allocation.

**Abstract interface (`planner/__init__.py`):**
- `BenchmarkTask` — one recipe+variant combination (recipe_dir, variant, recipe, gpu_name, gpu_count); includes `task_id` property, `to_dict()`, `setup_run_dir()`, and static methods for run directory management (`compute_code_hash()`, `create_run_dir()`, `write_tasks_json()`, `read_tasks_json()`)
- `ExecutionGroup` — group of tasks sharing one VM (gpu_name, gpu_count, tasks, index); `gpu_short` property returns short GPU name, `label` property returns unique label (e.g. `rtx5090_x_8` or `rtx5090_x_8_r01` when index is set)
- `BenchmarkPlanner` — ABC with `plan(tasks) -> list[ExecutionGroup]`

**Implementations:**
- `planner/group_by_model_and_gpu.py` — `GroupByModelAndGpuPlanner(gpu_concurrency=1)`: groups by (model_name, gpu_name) tuple. Same model on same GPU shares a VM to reuse cached weights. Max gpu_count determines VM size; tasks sorted descending. With `gpu_concurrency > 1`, each group is split into up to N sub-groups via round-robin, each provisioning its own VM (trades weight-cache reuse for wall-clock time).

### `commands/` — CLI Layer (thin handlers only)

Each command module contains only argparse registration and `handle_*` functions that delegate to library packages. CLI handlers use `asyncio.run()` to bridge sync argparse entry points into async internals:

```python
def handle_foo(args):
    asyncio.run(_handle_foo(args))

async def _handle_foo(args):
    await ...
```

**Command modules:**
- `commands/bench/` — `handle_bench()`, `register_bench_command()`, `GitCommitter` (incremental result commits)
- `commands/deploy/ssh.py` — `handle_ssh()`, `register_ssh_target()` — auto-detects remote GPU via SSH, resolves matrix, applies scale-out strategy
- `commands/deploy/local.py` — `handle_local()`, `register_local_target()` — auto-detects local GPU via PCI sysfs, resolves matrix, applies scale-out strategy
- `commands/deploy/cloud.py` — `handle_cloud()`, `register_cloud_target()` — uses recipe's `deploy.gpu` for matrix resolution
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
    +-- write tasks.json per run_dir
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
    +-- on_task_done callback (--commit-results: git add + commit + push)
    +-- teardown() (skipped with --no-teardown)
    |
    v
delete_cloud_vm(conn.delete_info) (skipped with --no-teardown; writes instances.json)
```

## Timing metrics

`deplodock/timing.py` provides `PhaseTimer` — an ordered collector of `phase -> seconds` durations, threaded by
mutation through the async deploy/bench chain (so `run_deploy()` keeps its `bool` return). Each phase is wrapped in
`async with timer.ameasure(name)` (sync `measure()` also exists); the elapsed is recorded even if the body raises,
and a `[timing] <name>: 12.3s` line is logged. Phase-name constants live in `timing.py`.

**Measured phases:** provisioning `vm_provision`, `remote_provision`; deploy `image_pull`, `model_download`,
`model_load_and_warmup` (the `compose up --wait` window — covers weight load into GPU + CUDA graph capture + warmup),
`smoke_test`; plus `benchmark`, `teardown`, and `command` (command recipes). After `model_load_and_warmup`,
`orchestrate.py` scrapes `docker compose logs` and runs `log_phases.parse_engine_load_phases()` +
`log_phases.decompose_model_load()` to break that window into a **non-overlapping** set of sub-phases that sums to the
parent: `startup` (container + CUDA init + imports) / `weights_load` / `torch_compile` (engine kernel / torch.compile
time) / `engine_warmup` (profile + KV cache + warmup, derived from vLLM's `init engine … took X s` line) /
`cuda_graph_capture`. When the engine-init line isn't present (older vLLM / SGLang) the unattributed time collapses into
a single `other` remainder. All of these are a breakdown of `model_load_and_warmup`, so they are **excluded from
`total`** (which would otherwise double-count). Near-zero phases
(`container_cleanup`, health poll, `system_info`) are intentionally not timed, so the phases don't fully sum to raw
wall-clock.

**Attribution:** provisioning runs once per `ExecutionGroup` (shared VM) but is seeded into each task's timer, so every
task's result reflects what it cost to stand up its host. `vm_provision` is omitted for fixed/local hosts (no VM
created). `timing["benchmark"]` is wall-clock (incl. the docker bench-client startup), distinct from
`metrics.benchmark_duration_s` (the server-measured window).

**Output:** `bench` persists timing into each task's `.json` (`"timing"` key) and `.txt` (`=== Timing ===` section) and
prints a per-task `TIMING` table in the end-of-run summary (`commands/bench/__init__.py::_format_timing_table`).
Standalone `deploy local/ssh/cloud` are display-only (no results dir) — they log the `PhaseTimer.format_table()`
breakdown at the end.

### Fixed-host mode (`--local` / `--ssh`)

When the user supplies pre-allocated hosts via `--local` and/or `--ssh user@host[:port]`,
`bench` skips cloud provisioning entirely. `benchmark/fixed_hosts.py` resolves each host
into an `AllocatedHost(conn, gpu_name, gpu_count)` (GPU detected via PCI sysfs through the
existing `detect_local_gpus()` / `detect_remote_gpus()` helpers), then validates that every
planned `ExecutionGroup` can run on at least one supplied host. The dispatcher
`_run_groups_on_hosts()` routes each group to a compatible idle host (locking per-host so
each runs at most one group at a time) and calls `run_execution_group(...,
preallocated_conn=host.conn)` — which skips both `provision_cloud_vm()` and
`delete_cloud_vm()`. `provision_remote()` (Docker, NVIDIA Container Toolkit, optional
driver/CUDA pinning) still runs and is idempotent, so already-provisioned hosts are a
fast no-op while bare VMs (e.g. straight from `vm create`) get set up on first use.

## CLI Command Tree

```
deplodock
+-- deploy
|   +-- local    -- deploy locally via docker compose
|   +-- ssh      -- deploy to remote server via SSH
|   +-- cloud    -- provision cloud VM + deploy via SSH
+-- bench        -- deploy + benchmark + teardown on cloud VMs
+-- teardown     -- clean up VMs left by bench --no-teardown
+-- vm
    +-- create
    |   +-- gcp
    |   +-- cloudrift
    +-- delete
        +-- gcp
        +-- cloudrift
```

## CLI Reference

### `deplodock deploy local`

Runs `docker compose` directly on the current machine. Auto-detects the local GPU via PCI sysfs and selects the matching `matrices` entry from the recipe.

```bash
deplodock deploy local --recipe <path> [--dry-run] [--teardown]
deplodock deploy local --recipe <path> --gpu "..." --gpu-count N    # override detection
```

### `deplodock deploy ssh`

Deploys to a remote server via SSH + SCP. Auto-detects the remote GPU and resolves the matrix the same way as `deploy local`. The remote host is responsible for having Docker + NVIDIA toolkit installed (or supplying `deploy.driver_version` / `deploy.cuda_version` in the recipe — see Recipe ARCHITECTURE).

```bash
deplodock deploy ssh --recipe <path> --ssh user@host[:port] [--ssh-key ~/.ssh/id_ed25519] [--dry-run] [--teardown]
```

### `deplodock deploy cloud`

Provisions a cloud VM and deploys via SSH. Requires `--gpu` and `--gpu-count` to select the matching matrix entry from the recipe (no auto-detection — there is no host yet). When a GPU is offered by more than one provider, the first provider in `hardware.py`'s `GPU_INSTANCE_TYPES` table is chosen by default; pass `--provider {gcp,cloudrift}` to override.

```bash
deplodock deploy cloud --recipe <path> --gpu "NVIDIA H200 141GB" --gpu-count 8 [--provider gcp] [--name prefix]
```

### Hardware-Aware Deploy (Local / SSH)

Both `deploy local` and `deploy ssh` auto-detect the target GPU by scanning PCI sysfs device IDs (locally or over SSH) and select the matching `matrices` entry. If more GPUs are available than the recipe's base configuration needs, a scale-out strategy is applied (`--scale-out-strategy {data-parallelism,replica-parallelism}`, default `data-parallelism`).

### `deplodock bench`

Loads each recipe, provisions cloud VMs, deploys the model, runs `vllm bench serve`, captures results, and tears down. Recipes sharing the same model and GPU type are grouped onto the same VM (see `GroupByModelAndGpuPlanner`).

```bash
deplodock bench recipes/*                                    # All recipes
deplodock bench experiments/.../optimal_mcr_rtx5090          # An experiment
deplodock bench recipes/* --filter "deploy.gpu=*5090*"       # Subset (fnmatch glob, AND across multiple --filter)
deplodock bench recipes/* --gpu-concurrency 4                # Split each (model, GPU) group across up to N VMs
deplodock bench recipes/* --no-teardown                      # Skip teardown; writes instances.json for later cleanup
deplodock bench recipes/* --local                            # Run on this machine via ssh to 127.0.0.1
deplodock bench recipes/* --ssh user@host1 --ssh user@host2  # Pre-allocated host pool (no provisioning, no teardown)
```

Results are stored in `{recipe_dir}/{timestamp}_{hash}/` — each recipe directory holds its own run directories alongside `recipe.yaml`.

**`--local` note:** runs the workload over SSH to `127.0.0.1` (same code path as remote hosts). Requires a running SSH server on localhost and that `--ssh-key` (default `~/.ssh/id_ed25519`) is in `~/.ssh/authorized_keys`. Quick check: `ssh -i ~/.ssh/id_ed25519 $USER@127.0.0.1 echo ok`.

**Fixed-host mode:** when `--local` and/or `--ssh` are supplied, `bench` detects each host's GPU and verifies that every planned execution group can run on at least one host (matching `deploy.gpu` and sufficient `deploy.gpu_count`). Unsatisfied groups abort the run before any work starts. Fixed hosts are never deleted at the end of the run.

### `deplodock teardown`

Cleans up VMs left running by `bench --no-teardown`. Reads `instances.json` from the run directory.

```bash
deplodock teardown <run_dir> [--ssh-key ~/.ssh/id_ed25519]
```

### `deplodock vm create / delete`

Manages cloud GPU VM lifecycles directly. Instances are ephemeral — `delete` removes them entirely. Run `deplodock vm create {gpu,gcp,cloudrift} --help` for full flag lists.

There are two `vm create` modes:

* **`gpu`** (recommended) — name a GPU from the hardware table; the orchestrator picks the provider and instance type, retries transient failures, and falls back to alternative candidates on capacity errors. Same code path as `deploy cloud` and `bench`.
* **`gcp` / `cloudrift`** — single-shot manual create. You pass the exact `--machine-type` / `--instance-type`. No retries, no fallback. Useful for debugging an exact instance shape or for instance types not yet in the hardware table.

```bash
# GPU-based (uses orchestrator: retries, candidate fallback, orphan cleanup)
deplodock vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 2 \
  --ssh-key ~/.ssh/id_ed25519 --provider cloudrift

# Manual single-shot
deplodock vm create gcp --instance my-vm --zone us-central1-a --machine-type a2-highgpu-1g
deplodock vm delete gcp --instance my-vm --zone us-central1-a

deplodock vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
deplodock vm delete cloudrift --instance-id <id>
```

CloudRift attach to a specific network with `--network <name>` (on `vm create cloudrift`, `vm create gpu`, `deploy cloud`, and `bench`). The name must exist in the target datacenter; omit to let CloudRift pick a public network.

**Extra authorized keys.** `--ssh-key` is the *private* key deplodock connects with; its `.pub` is always installed in
the VM's `authorized_keys`. To grant additional people access, pass `--authorized-key PATH` (repeatable, on `vm create
gpu` and `deploy cloud`) — each points to one public key file. The authorized set becomes `[ssh-key's .pub] + [every
--authorized-key]` (CloudRift via the rent payload's `PublicKeys` list; GCP via newline-joined `ssh-keys` metadata).
Missing or empty key files fail fast before any VM is provisioned.

```bash
deplodock vm create gpu --gpu "NVIDIA H200 141GB" --gpu-count 2 --provider cloudrift \
  --authorized-key ~/.ssh/alice.pub --authorized-key ~/.ssh/bob.pub
```

#### Allocation strategy (shared by `deploy cloud`, `bench`, `vm create gpu`)

All three commands go through `provision_cloud_vm()` in `deplodock/provisioning/cloud.py`. It enumerates *candidates* from `hardware.GPU_INSTANCE_TYPES` (preference-ordered) and, for GCP, fans out across the zones listed in `GPU_GCP_ZONES`. For each candidate it makes up to `SAME_CANDIDATE_RETRIES` attempts on transient failures, then advances. Providers signal "no capacity, try next" by raising `CapacityExhausted`; non-retryable errors raise `TerminalProvisionError` and abort. Fallback never silently crosses provider boundaries — `--provider` (or the first hardware-table entry) bounds the search.

Capacity-class signals recognized today: CloudRift HTTP 503/429 on rent, CloudRift `Inactive` terminal status / readiness timeout, GCP `ZONE_RESOURCE_POOL_EXHAUSTED` / `QUOTA_EXCEEDED` / `STOCKOUT` in `gcloud` stderr, and GCP `RUNNING`-status timeout. Both providers terminate VMs they created but couldn't bring to readiness, so orchestrator fallback does not leak orphan instances.

GCP project is inferred from `gcloud` config. CloudRift reads `CLOUDRIFT_API_KEY` and `CLOUDRIFT_API_URL` from the environment by default. **H200 on CloudRift** is only available on on-prem clusters — set `CLOUDRIFT_API_URL` to the on-prem endpoint (the public `api.cloudrift.ai` does not offer H200).

## Experiments

Experiments are self-contained parameter sweeps in `experiments/{model}/{name}/`. Each directory contains a `recipe.yaml` and stores its results alongside it:

```
experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090/
  recipe.yaml
  2026-02-24_19-13-50_abc12345/
    tasks.json
    recipe.yaml
    rtx5090x1_mcr8_c8_vllm_benchmark.txt
    ...
```

```bash
deplodock bench experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090
```

## CI Benchmark Workflow

External developers can submit experiments via pull requests. A maintainer triggers benchmarks by commenting `/run-experiment` on the PR. CI runs benchmarks on cloud GPUs and commits results back to the PR branch.

```
/run-experiment                                                       # Auto-detect: all experiments changed in the PR
/run-experiment experiments/MyModel/my_experiment                      # Explicit
/run-experiment experiments/MyModel/my_experiment --gpu-concurrency 2  # Split groups across 2 VMs each
```

Only users with **write** or **admin** access can trigger benchmarks. For fork PRs, "Allow edits from maintainers" must be checked for results to be pushed back to the fork branch (otherwise results are downloadable as workflow artifacts).

## Adding a New VM Provider

1. Create `provisioning/<provider>.py` with `create_instance()` -> `VMConnectionInfo | None` and `delete_instance()`. The function must raise `CapacityExhausted` on no-capacity errors and `TerminalProvisionError` on auth/malformed-request errors, and terminate any VM it created but couldn't bring to readiness before re-raising (see `deplodock/provisioning/errors.py`).
2. Add CLI handlers in `commands/vm/<provider>.py` (the single-shot manual subcommand).
3. Register CLI in `commands/vm/__init__.py`.
4. Add entries to `hardware.py` `GPU_INSTANCE_TYPES` table. If the provider has zone-affinity, add `GPU_<provider>_ZONES` and teach `iter_candidates` in `provisioning/candidates.py` to fan out across them.
5. Add provider dispatch in `provisioning/cloud.py` (`_provision_candidate`) and `delete_cloud_vm()`.
