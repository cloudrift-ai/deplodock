# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deplodock is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

The `README.md` is intentionally short — example-driven, no narrative. For details, consult the ARCHITECTURE.md files:

- **CLI usage** (deploy local/ssh/cloud, bench, teardown, vm, hardware-aware deploy, fixed-host mode, experiments, CI workflow) → [`deplodock/commands/ARCHITECTURE.md`](deplodock/commands/ARCHITECTURE.md)
- **Recipe format** (matrices/cross/zip combinators, variant filtering, deep merge, named fields, extra_args validation, command recipes, aggregate, docker_options, driver/cuda pinning, SGLang) → [`deplodock/recipe/ARCHITECTURE.md`](deplodock/recipe/ARCHITECTURE.md)
- **Compiler** (Graph IR dialects, passes, backends) → [`deplodock/compiler/ARCHITECTURE.md`](deplodock/compiler/ARCHITECTURE.md) and child docs

When the user asks about a CLI flag, recipe field, or matrix combinator, read the relevant ARCHITECTURE.md before answering — they hold details that are no longer in the README.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `DEPLODOCK_DUMP_DIR` environment variable (optional) — when set, all compiler stages dump intermediate artifacts (graphs, CUDA kernels, execution plans) to this directory for debugging

## Running Tests

```bash
make test
```

Or for a specific test file:

```bash
./venv/bin/pytest tests/test_recipe.py -v
```

When running a large subset (e.g. `tests/compiler/`), pass the same xdist flags `make test` uses to parallelize:

```bash
./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup
```

`-n auto` spawns one worker per core; `--dist=loadgroup` keeps tests sharing an `xdist_group` (e.g. CUDA context) on the
same worker.

## CLI Commands

- `deplodock deploy local ...` — deploy locally via docker compose
- `deplodock deploy ssh ...` — deploy to remote server via SSH
- `deplodock deploy cloud ...` — provision a cloud VM and deploy via SSH
- `deplodock bench recipes/* ...` — deploy + benchmark + teardown on cloud VMs (recipe dirs as positional args)
- `deplodock bench recipes/* --filter "KEY=PATTERN"` — run only variants matching the filter (fnmatch glob, repeatable, AND logic)
- `deplodock bench experiments/...` — run an experiment (results stored in the experiment dir)
- `deplodock teardown <run_dir>` — clean up VMs left running by `bench --no-teardown`
- `deplodock vm create gcp ...` — create a GCP GPU VM
- `deplodock vm create cloudrift ...` — create a CloudRift GPU VM
- `deplodock vm delete gcp ...` — delete a GCP GPU VM
- `deplodock vm delete cloudrift ...` — delete a CloudRift GPU VM
- `deplodock pull <model>` — download a HuggingFace model to local cache
- `deplodock trace <model> [--layer N] [--seq-len N]` — trace a transformer layer (or the whole model if `--layer` is omitted) to Graph IR (JSON). Whole-model tracing patches HF's dynamic causal-mask construction via `trace.huggingface.build_full_model_wrapper`.
- `deplodock trace --code "EXPR"` — trace an inline `nn.Module` expression (last stmt must be a call, e.g. `"torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))"`)
- `deplodock compile <model_or_ir> [--layer N] [--seq-len N] [--dump-dir DIR]` — run `decomposition → optimization → fusion` and save the fused `Graph[LoopOp]` (auto-pulls + traces if given a model ID; omit `--layer` for whole-model)
- `deplodock compile --code "EXPR" [--ir STAGE]` — trace + compile an inline `nn.Module` expression in one step (same grammar as `trace --code`; last stmt must be a call)
- `deplodock compile <ir_file> --ir {torch|tensor|loop|kernel|cuda}` — print the requested IR stage to stdout. `loop` renders fused `LoopOp` bodies (post decomposition+optimization+fusion); `kernel` renders the per-kernel AST (post LoopOp→KernelOp lowering); `cuda` renders the per-kernel CUDA source (post KernelOp→CudaOp lowering).
- `deplodock run --code "EXPR" [--bench] [--warmup N] [--iters N]` — compile + execute an inline `nn.Module`/torch expression on the CUDA backend, check accuracy vs eager, and (with `--bench`) print a latency table comparing eager PyTorch / `torch.compile` / Deplodock. Same `--code` grammar as `compile --code`.
- `deplodock inspect <ir_file>` — display graph IR summary (op counts, inputs, outputs)
- Quick test model (ungated, Llama arch): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- GPU benchmark model (ungated, 7B): `Qwen/Qwen2.5-7B`
- Block benchmark script: `python scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 32`

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips `perf`-marked tests; see `tests/perf/ARCHITECTURE.md`)
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`deplodock bench recipes/*`)
- `make bench-kernels` — run per-kernel perf comparison vs PyTorch (`tests/perf/`, requires CUDA)
- `make clean` — remove venv and generated files

## Documentation Conventions

- Target ~120 characters for `ARCHITECTURE.md`, `README.md`, and other docs (ASCII diagrams, tables, prose). Wider is fine if a table or diagram reads better that way — some overflow is acceptable. Python code stays under 140 chars (Ruff-enforced).

## Contribution Instructions

IMPORTANT: You MUST follow ALL of these steps for EVERY code change. Do NOT skip any step.

### Writing code

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`) — NEVER commit directly to `main`
2. Write code following guidelines in `STYLE.md`, `README.md` and `ARCHITECTURE.md` files in respective folders
3. Add tests if reasonable (in `tests/` following `tests/ARCHITECTURE.md` guidelines)

### Before committing (MANDATORY — do NOT skip these)

You MUST complete ALL of the following checks before every commit. These are not optional:

4. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
5. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md` and compare
6. **Update `CLAUDE.md`** if general instructions are no longer accurate — READ this file and compare
7. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant `ARCHITECTURE.md` and compare
8. **Run tests**: `make test` — fix any failures before proceeding
9. **Run linter**: `make lint` — if it fails, run `make format` and re-check

### Submitting

10. Push and open a PR
