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

## Key Make Targets

- `make setup` — create venv and install dependencies
- `make bench` — run benchmarks in parallel across configured servers
- `make report` — generate Excel report from benchmark results
- `make clean` — remove venv and generated files

## PR Procedure

1. Write code following `STYLE.md`
2. Add tests if reasonable (in `tests/`)
3. Run tests: `pytest tests/ -v`
4. Push and open a PR
