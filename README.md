# Server Benchmark

Tools for deploying and benchmarking LLM inference on GPU servers.

## Quick Start

### Install

```bash
git clone https://github.com/cloudrift-ai/server-benchmark.git
cd server-benchmark
make setup
```

### Deploy a Model

```bash
python main.py deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --variant 8xH200 \
  --server user@host
```

### Deploy Locally

```bash
python main.py deploy local \
  --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --variant RTX5090
```

### Teardown

```bash
python main.py deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --server user@host \
  --teardown
```

### Dry Run

Preview commands without executing:

```bash
python main.py deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --variant 8xH200 \
  --server user@host \
  --dry-run
```

## Recipes

Recipes are declarative YAML configs in `recipes/<model>/recipe.yaml`. Each recipe defines a model, backend settings, and hardware variants.

### Format

```yaml
model:
  name: "org/model-name"

backend:
  vllm:
    image: "vllm/vllm-openai:latest"
    tensor_parallel_size: 8
    pipeline_parallel_size: 1
    gpu_memory_utilization: 0.9
    extra_args: "--max-num-seqs 512 --max-model-len 16384"

variants:
  8xH200: {}                    # Uses defaults
  8xH100:
    backend:
      vllm:
        extra_args: "--max-model-len 8192"  # Override for this hardware
```

### Variant Naming

Variants use hardware-descriptive names for per-instance GPU setup:
- `H200` — 1 GPU
- `8xH200` — 8 GPUs (tensor parallel)
- `4xH100` — 4 GPUs

Format: `[<gpus>x]<GPU_TYPE>`. Multi-instance deployment is automatic when variant GPUs exceed per-instance needs.

### Available Recipes

| Recipe | Model | GPU Config |
|--------|-------|-----------|
| `GLM-4.6-FP8` | zai-org/GLM-4.6-FP8 | TP=8, dense FP8 |
| `GLM-4.6` | zai-org/GLM-4.6 | TP=8, dense FP16 |
| `GLM-4.5-Air-AWQ-4bit` | cpatonn/GLM-4.5-Air-AWQ-4bit | TP=1, MoE |
| `Qwen3-Coder-480B-A35B-Instruct-AWQ` | QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ | TP=4, large MoE |
| `Qwen3-Coder-30B-A3B-Instruct-AWQ` | QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ | TP=1, small MoE |
| `Meta-Llama-3.3-70B-Instruct-AWQ-INT4` | ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 | PP=2, dense |

## Deploy Targets

### Local

Runs docker compose directly on the current machine.

```bash
python main.py deploy local --recipe <path> [--variant <name>] [--dry-run]
```

### SSH

Deploys to a remote server via SSH + SCP.

```bash
python main.py deploy ssh --recipe <path> --server user@host [--variant <name>] [--dry-run]
```

### Common Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--recipe` | Yes | - | Path to recipe directory |
| `--variant` | No | - | Hardware variant (e.g. 8xH200) |
| `--hf-token` | No | `$HF_TOKEN` | HuggingFace token |
| `--model-dir` | No | `/mnt/models` | Model cache dir |
| `--teardown` | No | false | Stop containers instead of deploying |
| `--dry-run` | No | false | Print commands without executing |

### SSH-only Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--server` | Yes | - | SSH address (user@host) |
| `--ssh-key` | No | `~/.ssh/id_ed25519` | SSH key path |
| `--ssh-port` | No | 22 | SSH port |

## Running Tests

```bash
pytest tests/ -v
```

## Benchmarking

### Run Benchmarks

```bash
deplodock bench --parallel            # Run benchmarks in parallel
deplodock bench --server my_server    # Run for specific server
deplodock bench --model org/model     # Run for specific model
deplodock bench --parallel --force    # Force re-run, skip cached results
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--force` | false | Force re-run even if results exist |
| `--server` | - | Run only for a specific server |
| `--model` | - | Run only for a specific model |
| `--parallel` | false | Run servers in parallel |
| `--max-workers` | num servers | Max parallel server benchmarks |

### Generate Reports

```bash
deplodock report                                      # Default: results/ -> results/benchmark_report.xlsx
deplodock report --results-dir results/custom --output results/custom/report.xlsx
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--results-dir` | `results` | Directory containing benchmark results |
| `--output` | `results/benchmark_report.xlsx` | Output Excel file path |

## Project Structure

```
deplodock/
├── deplodock/
│   ├── deplodock.py                 # CLI entrypoint
│   └── commands/
│       ├── deploy/
│       │   ├── __init__.py          # Shared: recipe, compose, orchestration
│       │   ├── local.py             # Local target
│       │   └── ssh.py               # SSH target
│       ├── bench/
│       │   └── __init__.py          # Benchmark runner
│       └── report/
│           └── __init__.py          # Report generator
├── recipes/                         # Model deploy recipes
│   ├── GLM-4.6-FP8/
│   ├── GLM-4.6/
│   ├── GLM-4.5-Air-AWQ-4bit/
│   ├── Qwen3-Coder-480B-A35B-Instruct-AWQ/
│   ├── Qwen3-Coder-30B-A3B-Instruct-AWQ/
│   └── Meta-Llama-3.3-70B-Instruct-AWQ-INT4/
├── tests/                           # pytest tests
├── benchmarks/                      # Benchmark scripts
├── utils/                           # Utilities
├── config.yaml                      # Benchmark configuration
└── Makefile                         # Build automation
```
