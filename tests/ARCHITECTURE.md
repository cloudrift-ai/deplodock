# Test Architecture

## Overview

All tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` in `pyproject.toml`) and live in the `tests/` directory, organized into subdirectories that mirror the `deplodock/` source tree. Tests are designed to run without GPU hardware, Docker, or network access — every external interaction is avoided via dry-run mode or by testing pure functions directly.

## Directory Structure

```
tests/
├── conftest.py              # shared fixtures
├── test_detect.py               # deplodock.detect (GPU detection via PCI sysfs)
├── test_hardware.py         # deplodock.hardware (top-level module)
├── test_redact.py           # deplodock.redact (secret redaction)
├── benchmark/
│   ├── test_bench_dryrun.py # bench CLI dry-run
│   ├── test_code_hash.py    # BenchmarkTask.compute_code_hash()
│   ├── test_tasks_json.py   # BenchmarkTask.write_tasks_json(), read_tasks_json()
│   ├── test_run_dir.py      # BenchmarkTask.create_run_dir()
│   ├── test_results.py      # parse_benchmark_metrics(), parse_system_info(), compose_json_result()
│   └── test_command_workload.py # build_substitution_map(), render_command()
├── recipe/
│   ├── test_types.py        # Recipe.from_dict(), LLMConfig properties, dataclass defaults
│   └── test_engines.py      # build_engine_args(), banned_extra_arg_flags()
├── deploy/
│   ├── test_compose.py      # generate_compose(), generate_nginx_conf()
│   ├── test_deploy_cloud_dryrun.py  # deploy cloud CLI dry-run
│   ├── test_deploy_dryrun.py        # deploy ssh/local CLI dry-run
│   ├── test_recipe.py       # load_recipe(), deep_merge(), validate_extra_args(), resolve_for_hardware()
│   └── test_scale_out.py    # DataParallelismScaleOutStrategy, ReplicaParallelismScaleOutStrategy
├── planner/
│   ├── test_planner.py      # BenchmarkTask, GroupByModelAndGpuPlanner
│   └── test_variant.py      # Variant class, _abbreviate()
├── provisioning/
│   ├── test_cloud.py        # resolve_vm_spec(), delete_cloud_vm(), VMConnectionInfo
│   ├── test_cloudrift.py    # CloudRift API helpers
│   ├── test_gcp.py             # GCP command builders
│   ├── test_staging.py      # enumerate_staged_files(), build_stage_tar()
│   └── test_vm_dryrun.py    # vm create/delete CLI dry-run
├── compiler/
│   ├── fixtures/               # pre-computed traces (tinyllama_layer0.json)
│   ├── test_ir.py              # Graph, Node, Tensor — add/remove/replace/topo/copy
│   ├── test_hints.py           # Hints get/set/merge/serialize + integration
│   ├── test_matcher.py         # Pattern matching engine
│   ├── test_rewriter.py        # Rewrite engine (SiLU decomposition)
│   ├── rules/
│   │   ├── test_decompose_rules.py    # Decomposition rules (structural + correctness)
│   │   ├── test_optimization_rules.py # Optimization rules (broadcast indexmap)
│   │   ├── test_fusion_rules.py       # Fusion pass (lift-then-merge, structural)
│   │   └── test_merge_core.py         # σ solver and merge_loop_ops (unit)
│   ├── test_fusion.py          # auto_fuse — softmax, RMSNorm, SiLU, matmul, etc.
│   ├── test_plan.py            # plan_graph — ExecutionPlan from Graph
│   ├── test_pipeline.py        # Full compile pipeline: graph → GPU
│   ├── test_real_trace.py      # TinyLlama fixture validation
│   ├── test_torch_trace.py     # PyTorch tracer smoke tests
│   ├── test_torch_trace_ops.py # PyTorch tracer op handlers and helpers
│   ├── test_backend_ir.py      # Backend IR AST nodes + codegen emission
│   ├── test_kernel_gen.py      # Kernel generation from FusedRegionOps
│   ├── test_loop_ir.py         # LoopIR dataclasses, pretty-print, structure, round-trip
│   ├── test_cuda.py            # CUDA codegen, lowering, GPU correctness
│   ├── test_cuda_backend.py    # CudaBackend compile/run/benchmark
│   ├── test_program.py         # Program source gen + GPU execution
│   ├── test_tuning.py          # GPU tuning profile dispatch
│   ├── test_llama_block.py     # Full Llama block through compiler
│   └── test_torch_ops.py   # Op.forward() + numpy backend (no GPU needed)
├── scripts/
│   └── test_plot_mcr_sweep.py  # load_results() from scripts/plot_mcr_sweep.py
```

## Test Layers

### Unit Tests

Test individual functions in isolation with synthetic inputs.

| File | Covers |
|------|--------|
| `recipe/test_types.py` | `Recipe.from_dict()`, `LLMConfig` properties (`engine_name`, `gpus_per_instance`, `image`, `extra_args`, `extra_env`, `docker_options`), dataclass defaults |
| `recipe/test_engines.py` | `build_engine_args()`, `banned_extra_arg_flags()` — engine flag mapping, CLI argument building for vLLM and SGLang |
| `deploy/test_recipe.py` | `deplodock.recipe.load_recipe()`, `deep_merge()`, `validate_extra_args()`, `validate_docker_options()`, `resolve_for_hardware()` — recipe loading, variant resolution, YAML parsing, extra_args validation, docker_options validation, hardware-aware matrix resolution |
| `deploy/test_scale_out.py` | `DataParallelismScaleOutStrategy`, `ReplicaParallelismScaleOutStrategy` — scale-out strategy application, GPU count validation, immutability |
| `deploy/test_compose.py` | `deplodock.deploy.generate_compose()`, `generate_nginx_conf()` — Docker Compose and nginx config generation, `gpu_device_ids` support, `docker_options` rendering |
| `provisioning/test_cloud.py` | `deplodock.provisioning.cloud.resolve_vm_spec()`, `delete_cloud_vm()`, `_provision_once()`, `VMConnectionInfo` — cloud provisioning unit tests |
| `planner/test_planner.py` | `BenchmarkTask`, `GroupByModelAndGpuPlanner` — task properties (`recipe_name`, `result_path`, `gpu_name`, `gpu_count`, `gpu_short`), grouping logic, sorting |
| `planner/test_variant.py` | `Variant` — `__str__`, `gpu_short`, `gpu_count`, `__eq__`, `__hash__`, `_abbreviate()` |
| `test_detect.py` | `_parse_sysfs_output()`, `detect_local_gpus()`, `detect_remote_gpus()` — PCI sysfs GPU detection, mixed GPU errors, mock SSH |
| `test_hardware.py` | `resolve_instance_type()`, `gpu_short_name()`, `GPU_INSTANCE_TYPES` — hardware lookup tables |
| `test_redact.py` | `deplodock.redact.redact_secrets()`, `SecretRedactingFilter` — value-based secret redaction for text and log records |
| `benchmark/test_code_hash.py` | `BenchmarkTask.compute_code_hash()` — determinism, hex format |
| `benchmark/test_run_dir.py` | `BenchmarkTask.create_run_dir()` — directory creation, naming format |
| `benchmark/test_tasks_json.py` | `BenchmarkTask.write_tasks_json()`, `read_tasks_json()` — tasks.json round-trip |
| `benchmark/test_results.py` | `parse_benchmark_metrics()`, `parse_system_info()`, `compose_json_result()` — structured JSON result parsing and composition |
| `provisioning/test_cloudrift.py` | `deplodock.provisioning.cloudrift._api_request()`, `_rent_instance()`, etc. — CloudRift API helpers |
| `provisioning/test_gcp.py` | `deplodock.provisioning.gcp._gcloud_*_cmd()` — GCP command builders |
| `scripts/test_plot_mcr_sweep.py` | `load_results()` — benchmark JSON loading and sorting from `scripts/plot_mcr_sweep.py` |

Unit tests use **fixtures from `conftest.py`** (`tmp_recipe_dir`, `sample_config`, `sample_config_multi`) to supply pre-built recipe directories and config dicts.

### CLI Dry-Run Tests

Test the full CLI pipeline end-to-end by invoking `deplodock` as a subprocess with `--dry-run`. This exercises argument parsing, config loading, recipe resolution, and the deploy/bench orchestration — stopping just before any real side effects (SSH, Docker, file writes).

| File | Covers |
|------|--------|
| `deploy/test_deploy_dryrun.py` | `deploy ssh`, `deploy local` — dry-run output, command sequence, variant resolution, teardown, CLI help |
| `deploy/test_deploy_cloud_dryrun.py` | `deploy cloud` — dry-run output, deploy steps, error handling, CLI help |
| `benchmark/test_bench_dryrun.py` | `bench` — dry-run output, deploy->benchmark->teardown sequence, variant filtering, `--no-teardown` flag, per-recipe result directories, experiment recipe dry-run, CLI help; `teardown` — CLI help |
| `provisioning/test_vm_dryrun.py` | `vm create/delete gcp`, `vm create/delete cloudrift` — dry-run output, argparse validation, CLI help |

CLI tests use the **`run_cli` fixture** (a subprocess wrapper) and **`make_bench_config`** (a factory for temporary `config.yaml` files). Both are defined in `conftest.py`.

## Shared Fixtures (`conftest.py`)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `project_root` | session | Absolute path to repo root |
| `recipes_dir` | session | Absolute path to `recipes/` |
| `run_cli` | session | Callable that invokes `python -m deplodock.deplodock` as a subprocess |
| `make_bench_config` | function | Factory that writes a temp `config.yaml` for bench tests (benchmark section only) |
| `tmp_recipe_dir` | function | Temp directory with a sample `recipe.yaml` for unit tests |
| `sample_config` | function | Single-instance vLLM config dict for compose tests |
| `sample_config_sglang` | function | Single-instance SGLang config dict for compose tests |
| `sample_config_multi` | function | Multi-instance config dict for compose tests |

## Conventions

- **Async tests** — tests for async functions are plain `async def` (no decorator needed; `asyncio_mode = "auto"` handles it). Mock async callables with `AsyncMock`.
- **No mocking** — dry-run mode is the primary strategy for testing command orchestration without side effects.
- **Real recipes** — CLI dry-run tests use recipes from the `recipes/` directory to catch config drift.
- **Temp recipes** — unit tests and multi-instance edge cases create throwaway recipes via `tmp_path`.
- **Plain functions** — no test classes; tests are grouped by file and separated with comment headers.
- **Assertions on stdout** — dry-run tests verify that the correct commands and messages appear in the expected order.
- **Mirror source layout** — test directories match `deplodock/` subdirectories (e.g. `tests/deploy/` ↔ `deplodock/deploy/`).

## Running

```bash
pytest tests/ -v                       # all tests
pytest tests/deploy/test_recipe.py -v  # single file
pytest tests/planner/ -v               # single directory
```
