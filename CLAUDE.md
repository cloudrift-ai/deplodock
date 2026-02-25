# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deplodock is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

See `README.md` for full project structure, recipe format, and CLI usage.

## Prerequisites

- Python 3.12+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads

## Running Tests

```bash
make test
```

Or for a specific test file:

```bash
./venv/bin/pytest tests/test_recipe.py -v
```

## CLI Commands

- `deplodock deploy local ...` — deploy locally via docker compose
- `deplodock deploy ssh ...` — deploy to remote server via SSH
- `deplodock deploy cloud ...` — provision a cloud VM and deploy via SSH
- `deplodock bench recipes/* ...` — deploy + benchmark + teardown on cloud VMs (recipe dirs as positional args)
- `deplodock teardown <run_dir>` — clean up VMs left running by `bench --no-teardown`
- `deplodock report ...` — generate Excel reports from benchmark results
- `deplodock vm create gcp ...` — create a GCP GPU VM
- `deplodock vm create cloudrift ...` — create a CloudRift GPU VM
- `deplodock vm delete gcp ...` — delete a GCP GPU VM
- `deplodock vm delete cloudrift ...` — delete a CloudRift GPU VM

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`deplodock bench recipes/*`)
- `make report` — generate Excel report (`deplodock report`)
- `make clean` — remove venv and generated files

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
