.PHONY: help setup clean bench bench-force bench-kernels bench-kernels-tune test-compose lint format

help:
	@echo "Server Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Install system dependencies, create venv, and install Python packages"
	@echo "  lint           - Run linter and format checks"
	@echo "  format         - Auto-format code and fix lint violations"
	@echo "  bench          - Run benchmarks in parallel"
	@echo "  bench-force    - Run benchmarks in parallel (force re-run, skip cached results)"
	@echo "  bench-kernels  - Run per-kernel perf comparison vs PyTorch (tests/perf/, requires CUDA)"
	@echo "  wheel          - Build the emmy wheel into dist/"
	@echo "  vllm-emmy-image - Build the vLLM + emmy serving image (docker/vllm-emmy)"
	@echo "  vllm-emmy-push  - Push the serving image to Docker Hub (cloudriftai/)"
	@echo "  clean          - Remove virtual environment and generated files"
	@echo "  test-compose   - Test docker-compose generation with sample config"

setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3.12 -m venv venv --prompt "emmy"; \
		echo "Installing Python dependencies..."; \
		./venv/bin/pip install -e ".[dev]"; \
	fi

setup-ci:
	python3.12 -m venv venv --prompt "emmy"
	./venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
	./venv/bin/pip install -e ".[compile,test]"

lint: setup
	./venv/bin/ruff check
	./venv/bin/ruff format --check

format: setup
	./venv/bin/ruff format
	./venv/bin/ruff check --fix

# Compile CUDA kernels at -Xcicc -O1: ~3x faster suite (dodges the cicc/LLVM unroll
# blowup on big register-tile kernels). This is the CORRECTNESS lane — -O1 changes
# runtime perf, not numerics, and the deployable perf tests (tests/perf, -m perf) run
# at -O3 via `make bench-kernels`. Override with EMMY_NVCC_FLAGS= to test at -O3.
test: setup
	EMMY_NVCC_FLAGS="-Xcicc -O1" ./venv/bin/pytest tests/ -v -n auto --dist=loadgroup

bench-kernels-clean: setup
	@rm -f /tmp/emmy-gpu.lock
	./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

bench-kernels-tuned: setup
	@rm -f /tmp/emmy-gpu.lock
	@test -f ~/.cache/emmy/tune-kernels.db || (echo "The kernel tuning DB not foud; run make tune-kernels"; exit 1)
	EMMY_TUNE_DB=~/.cache/emmy/tune-kernels.db ./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

tune-kernels: setup
	@rm -f /tmp/emmy-gpu.lock
	@rm -f ~/.cache/emmy/tune-kernels.db
	EMMY_TUNE=1 EMMY_TUNE_DB=~/.cache/emmy/tune-kernels.db ./venv/bin/pytest tests/perf/ -m perf -n 4 --dist=loadgroup -v -p no:randomly --no-header

# --- vLLM + emmy serving image (emmy/serving, docker/vllm-emmy) ---
VLLM_VERSION ?= v0.22.1
VLLM_EMMY_TAG ?= cloudriftai/vllm-emmy:$(patsubst v%,%,$(VLLM_VERSION))-$(shell git rev-parse --short HEAD)

wheel: setup
	./venv/bin/pip install --quiet build
	rm -rf dist && ./venv/bin/python -m build --wheel -o dist/ .

vllm-emmy-image: wheel
	docker build -f docker/vllm-emmy/Dockerfile --build-arg VLLM_VERSION=$(VLLM_VERSION) \
		-t $(VLLM_EMMY_TAG) .

vllm-emmy-push: vllm-emmy-image
	docker push $(VLLM_EMMY_TAG)

bench: setup
	@echo "Running benchmarks..."
	./venv/bin/emmy bench recipes/*

bench-force: setup
	@echo "Running benchmarks (force mode)..."
	./venv/bin/emmy bench recipes/* --force

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
