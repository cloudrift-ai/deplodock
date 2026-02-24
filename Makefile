.PHONY: help setup clean bench bench-force logs clean-logs test-compose report report-nov2025 lint format

help:
	@echo "Server Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Install system dependencies, create venv, and install Python packages"
	@echo "  lint           - Run linter and format checks"
	@echo "  format         - Auto-format code and fix lint violations"
	@echo "  bench          - Run benchmarks in parallel"
	@echo "  bench-force    - Run benchmarks in parallel (force re-run, skip cached results)"
	@echo "  report         - Generate Excel report from benchmark results"
	@echo "  report-nov2025 - Generate Excel report from Nov 2025 results (pro6000_l40s_h100_h200_11_2025)"
	@echo "  logs           - Show the latest benchmark log"
	@echo "  clean-logs     - Remove all log files"
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

bench: setup
	@echo "Running benchmarks..."
	./venv/bin/deplodock bench recipes/*

bench-force: setup
	@echo "Running benchmarks (force mode)..."
	./venv/bin/deplodock bench recipes/* --force

report: setup
	@echo "Generating Excel report from benchmark results..."
	./venv/bin/deplodock report
	@echo "Report generated: benchmark_report.xlsx"

report-nov2025: setup
	@echo "Generating Excel report from Nov 2025 benchmark results..."
	./venv/bin/deplodock report \
		--results-dir results/pro6000_l40s_h100_h200_11_2025 \
		--output results/pro6000_l40s_h100_h200_11_2025/benchmark_report.xlsx
	./venv/bin/deplodock report \
		--results-dir results/pro6000_l40s_h100_h200_11_2025/single-query \
		--output results/pro6000_l40s_h100_h200_11_2025/single-query/benchmark_report.xlsx
	./venv/bin/deplodock report \
		--results-dir results/pro6000_l40s_h100_h200_11_2025/single-gpu \
		--output results/pro6000_l40s_h100_h200_11_2025/single-gpu/benchmark_report.xlsx

report-jan2026: setup
	@echo "Generating Excel report from Jan 2026 benchmark results..."
	./venv/bin/deplodock report \
		--results-dir results/pro6000_h100_h200_b200_01_2026/summary \
		--output results/pro6000_h100_h200_b200_01_2026/benchmark_report.xlsx



logs:
	@if [ ! -d "logs" ]; then \
		echo "❌ No logs directory found."; \
		exit 1; \
	fi
	@LATEST_LOG=$$(ls -t logs/benchmark_*.log 2>/dev/null | head -1); \
	if [ -z "$$LATEST_LOG" ]; then \
		echo "❌ No log files found."; \
		exit 1; \
	fi; \
	echo "📝 Showing: $$LATEST_LOG"; \
	echo ""; \
	cat "$$LATEST_LOG"

clean-logs:
	@echo "Removing log files..."
	rm -rf logs/
	@echo "✅ Logs cleaned!"

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
