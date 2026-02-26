# Test Architecture

## Overview

All tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"` in `pyproject.toml`) and live in the `tests/` directory, organized into subdirectories that mirror the `deplodock/` source tree. Tests are designed to run without GPU hardware, Docker, or network access — every external interaction is avoided via dry-run mode or by testing pure functions directly.

## Directory Structure

```
tests/
├── conftest.py              # shared fixtures
├── test_hardware.py         # deplodock.hardware (top-level module)
├── benchmark/
│   ├── test_bench_dryrun.py # bench CLI dry-run
│   ├── test_code_hash.py    # compute_code_hash()
│   ├── test_tasks_json.py   # write_tasks_json(), read_tasks_json(), parse_task_from_result()
│   └── test_run_dir.py      # create_run_dir()
├── recipe/
│   ├── test_types.py        # Recipe.from_dict(), LLMConfig properties, dataclass defaults
│   └── test_engines.py      # build_engine_args(), banned_extra_arg_flags()
├── deploy/
│   ├── test_compose.py      # generate_compose(), generate_nginx_conf()
│   ├── test_deploy_cloud_dryrun.py  # deploy cloud CLI dry-run
│   ├── test_deploy_dryrun.py        # deploy ssh/local CLI dry-run
│   └── test_recipe.py       # load_recipe(), deep_merge(), validate_extra_args()
├── planner/
│   └── test_planner.py      # BenchmarkTask, GroupByModelAndGpuPlanner
├── provisioning/
│   ├── test_cloud.py        # resolve_vm_spec(), delete_cloud_vm(), VMConnectionInfo
│   ├── test_cloudrift.py    # CloudRift API helpers
│   ├── test_gcp.py             # GCP command builders
│   └── test_vm_dryrun.py    # vm create/delete CLI dry-run
└── report/
    └── test_report.py       # collect_tasks_from_results(), parse_benchmark_result()
```

## Test Layers

### Unit Tests

Test individual functions in isolation with synthetic inputs.

| File | Covers |
|------|--------|
| `recipe/test_types.py` | `Recipe.from_dict()`, `LLMConfig` properties (`engine_name`, `gpus_per_instance`, `image`, `extra_args`), dataclass defaults |
| `recipe/test_engines.py` | `build_engine_args()`, `banned_extra_arg_flags()` — engine flag mapping, CLI argument building for vLLM and SGLang |
| `deploy/test_recipe.py` | `deplodock.recipe.load_recipe()`, `deep_merge()`, `validate_extra_args()` — recipe loading, variant resolution, YAML parsing, extra_args validation |
| `deploy/test_compose.py` | `deplodock.deploy.generate_compose()`, `generate_nginx_conf()` — Docker Compose and nginx config generation, `gpu_device_ids` support |
| `provisioning/test_cloud.py` | `deplodock.provisioning.cloud.resolve_vm_spec()`, `delete_cloud_vm()`, `VMConnectionInfo` — cloud provisioning unit tests |
| `planner/test_planner.py` | `BenchmarkTask`, `GroupByModelAndGpuPlanner` — task properties (`recipe_name`, `result_path`), grouping logic, sorting |
| `test_hardware.py` | `resolve_instance_type()`, `gpu_short_name()`, `GPU_INSTANCE_TYPES` — hardware lookup tables |
| `benchmark/test_code_hash.py` | `deplodock.benchmark.compute_code_hash()` — determinism, hex format |
| `benchmark/test_run_dir.py` | `deplodock.benchmark.create_run_dir()` — directory creation, naming format |
| `benchmark/test_tasks_json.py` | `deplodock.benchmark.write_tasks_json()`, `read_tasks_json()`, `parse_task_from_result()` — tasks.json round-trip and result file parsing |
| `report/test_report.py` | `deplodock.report.collect_tasks_from_results()`, `parse_benchmark_result()` — tasks.json-based report data collection |
| `provisioning/test_cloudrift.py` | `deplodock.provisioning.cloudrift._api_request()`, `_rent_instance()`, etc. — CloudRift API helpers |
| `provisioning/test_gcp.py` | `deplodock.provisioning.gcp._gcloud_*_cmd()` — GCP command builders |

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
