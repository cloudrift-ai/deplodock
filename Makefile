.PHONY: help setup clean bench bench-force test-compose lint format

help:
	@echo "Server Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Install system dependencies, create venv, and install Python packages"
	@echo "  lint           - Run linter and format checks"
	@echo "  format         - Auto-format code and fix lint violations"
	@echo "  bench          - Run benchmarks in parallel"
	@echo "  bench-force    - Run benchmarks in parallel (force re-run, skip cached results)"
	@echo "  clean          - Remove virtual environment and generated files"
	@echo "  test-compose   - Test docker-compose generation with sample config"

setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3.12 -m venv venv; \
		echo "Installing Python dependencies..."; \
		./venv/bin/pip install -e ".[dev]"; \
		echo "✅ Setup complete!"; \
	fi

lint: setup
	./venv/bin/ruff check
	./venv/bin/ruff format --check

format: setup
	./venv/bin/ruff format
	./venv/bin/ruff check --fix

test: setup
	./venv/bin/pytest tests/ -v

bench: setup
	@echo "Running benchmarks..."
	./venv/bin/deplodock bench recipes/*

bench-force: setup
	@echo "Running benchmarks (force mode)..."
	./venv/bin/deplodock bench recipes/* --force

clean:
	@echo "Removing virtual environment and generated files..."
	rm -rf venv/
	rm -f docker-compose.*.yml nginx.*.conf
	rm -rf __pycache__/ utils/__pycache__/
	@echo "✅ Clean complete!"

test-compose:
	@if [ ! -d "venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Testing docker-compose generation..."
	./venv/bin/python utils/generate_compose.py \
		--num-instances 1 \
		--tensor-parallel-size 4 \
		--container-name test \
		--model-path /test/model \
		--model-name test-model \
		--hf-directory /hf \
		--hf-token test \
		--extra-args "--enable-expert-parallel --swap-space 16" \
		--output /tmp/test-compose.yml
	@echo "✅ Generated: /tmp/test-compose.yml"
	@echo ""
	@cat /tmp/test-compose.yml
