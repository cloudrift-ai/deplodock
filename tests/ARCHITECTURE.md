# Test Architecture

## Overview

All tests use **pytest** and live in the `tests/` directory. Tests are designed to run without GPU hardware, Docker, or network access — every external interaction is avoided via dry-run mode or by testing pure functions directly.

## Test Layers

### Unit Tests

Test individual functions in isolation with synthetic inputs.

| File | Covers |
|------|--------|
| `test_recipe.py` | `deplodock.deploy.load_recipe()`, `deep_merge()` — recipe loading, variant resolution, YAML parsing |
| `test_compose.py` | `deplodock.deploy.generate_compose()`, `generate_nginx_conf()` — Docker Compose and nginx config generation, `_gpu_device_ids` support |
| `test_deploy_cloud.py` | `deplodock.provisioning.cloud.resolve_vm_spec()`, `delete_cloud_vm()`, `VMConnectionInfo` — cloud provisioning unit tests |
| `test_planner.py` | `BenchmarkTask`, `GroupByModelAndGpuPlanner` — task properties (`recipe_name`, `result_path`), grouping logic, sorting |
| `test_hardware.py` | `resolve_instance_type()`, `gpu_short_name()`, `GPU_INSTANCE_TYPES` — hardware lookup tables |
| `benchmark/test_code_hash.py` | `deplodock.benchmark.compute_code_hash()` — determinism, hex format |
| `benchmark/test_run_dir.py` | `deplodock.benchmark.create_run_dir()` — directory creation, naming format |
| `benchmark/test_manifest.py` | `deplodock.benchmark.write_manifest()`, `read_manifest()` — round-trip serialization |
| `report/test_report.py` | `deplodock.report.collect_tasks_from_manifests()`, `parse_benchmark_result()` — manifest-based report data collection |
| `test_vm_cloudrift.py` | `deplodock.provisioning.cloudrift._api_request()`, `_rent_instance()`, etc. — CloudRift API helpers |
| `test_vm_gcp_flex_start.py` | `deplodock.provisioning.gcp_flex_start._gcloud_*_cmd()` — GCP command builders |

Unit tests use **fixtures from `conftest.py`** (`tmp_recipe_dir`, `sample_config`, `sample_config_multi`) to supply pre-built recipe directories and config dicts.

### CLI Dry-Run Tests

Test the full CLI pipeline end-to-end by invoking `deplodock` as a subprocess with `--dry-run`. This exercises argument parsing, config loading, recipe resolution, and the deploy/bench orchestration — stopping just before any real side effects (SSH, Docker, file writes).

| File | Covers |
|------|--------|
| `test_deploy_dryrun.py` | `deploy ssh`, `deploy local` — dry-run output, command sequence, variant resolution, teardown, CLI help |
| `test_deploy_cloud_dryrun.py` | `deploy cloud` — dry-run output, deploy steps, error handling, CLI help |
| `test_bench_dryrun.py` | `bench` — dry-run output, deploy->benchmark->teardown sequence, variant filtering, CLI help |
| `test_vm_dryrun.py` | `vm create/delete gcp-flex-start`, `vm create/delete cloudrift` — dry-run output, argparse validation, CLI help |

CLI tests use the **`run_cli` fixture** (a subprocess wrapper) and **`make_bench_config`** (a factory for temporary `config.yaml` files). Both are defined in `conftest.py`.

## Shared Fixtures (`conftest.py`)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `project_root` | session | Absolute path to repo root |
| `recipes_dir` | session | Absolute path to `recipes/` |
| `run_cli` | session | Callable that invokes `python -m deplodock.deplodock` as a subprocess |
| `make_bench_config` | function | Factory that writes a temp `config.yaml` for bench tests (benchmark section only) |
| `tmp_recipe_dir` | function | Temp directory with a sample `recipe.yaml` for unit tests |
| `sample_config` | function | Single-instance config dict for compose tests |
| `sample_config_multi` | function | Multi-instance config dict for compose tests |

## Conventions

- **No mocking** — dry-run mode is the primary strategy for testing command orchestration without side effects.
- **Real recipes** — CLI dry-run tests use recipes from the `recipes/` directory to catch config drift.
- **Temp recipes** — unit tests and multi-instance edge cases create throwaway recipes via `tmp_path`.
- **Plain functions** — no test classes; tests are grouped by file and separated with comment headers.
- **Assertions on stdout** — dry-run tests verify that the correct commands and messages appear in the expected order.

## Running

```bash
pytest tests/ -v          # all tests
pytest tests/test_recipe.py -v   # single file
```
