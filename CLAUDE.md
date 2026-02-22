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
- `deplodock bench ...` — run benchmarks on remote servers
- `deplodock report ...` — generate Excel reports from benchmark results

## Key Make Targets

- `make setup` — create venv and install dependencies
- `make bench` — run benchmarks in parallel (`deplodock bench --parallel`)
- `make report` — generate Excel report (`deplodock report`)
- `make clean` — remove venv and generated files

## Contribution Instructions

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`)
2. Write code following guidelines here, in `STYLE.md` and in `README.md`
3. Add tests if reasonable (in `tests/`)
4. Update `STYLE.md`, `README.md` and `CLAUDE.md` if necessary
5. Run tests: `pytest tests/ -v`
6. Push and open a PR
