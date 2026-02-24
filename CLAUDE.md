# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Server Benchmark is a Python tool for deploying and benchmarking LLM inference on GPU servers using vLLM. It provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

See `README.md` for full project structure, recipe format, and CLI usage.

## Prerequisites

- Python 3.13+ with `venv`
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads

## Running Tests

```bash
pytest tests/ -v
```

Or for a specific test file:

```bash
pytest tests/test_recipe.py -v
```

## CLI Commands

- `deplodock deploy local ...` — deploy locally via docker compose
- `deplodock deploy ssh ...` — deploy to remote server via SSH
- `deplodock deploy cloud ...` — provision a cloud VM and deploy via SSH
- `deplodock bench recipes/* ...` — deploy + benchmark + teardown on cloud VMs (recipe dirs as positional args)
- `deplodock report ...` — generate Excel reports from benchmark results
- `deplodock vm create gcp-flex-start ...` — create a GCP flex-start GPU VM
- `deplodock vm create cloudrift ...` — create a CloudRift GPU VM
- `deplodock vm delete gcp-flex-start ...` — delete a GCP flex-start GPU VM
- `deplodock vm delete cloudrift ...` — delete a CloudRift GPU VM

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make lint` — run `ruff check` and `ruff format --check`
- `make format` — auto-format code and fix lint violations
- `make bench` — run benchmarks (`deplodock bench recipes/*`)
- `make report` — generate Excel report (`deplodock report`)
- `make clean` — remove venv and generated files

## Contribution Instructions

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`)
2. Write code following guidelines here, in `STYLE.md`, `README.md` and `ARCHITECTURE.md` files in respective folders
3. Add tests if reasonable (in `tests/` following `tests/ARCHITECTURE.md` guidelines)
4. Check if style changes were introduced and update `STYLE.md` if necessary
5. Check if project setup, structure or usage patterns have changes and update `README.md` if necessary
6. Check if general instructions in `CLAUDE.md` are still accurate and update if necessary
7. Check for `ARCHITECTURE.md` files in directories that were updated and update them if necessary
8. Run tests: `pytest tests/ -v`
9. Run linter and format check: `make lint`
10. Auto-fix formatting: `make format`
11. Push and open a PR
