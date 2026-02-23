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

benchmark:
  max_concurrency: 128
  num_prompts: 256
  random_input_len: 8000
  random_output_len: 8000

variants:
  8xH200: {}                    # Uses defaults
  8xH100:
    backend:
      vllm:
        extra_args: "--max-model-len 8192"  # Override for this hardware
    benchmark:                              # Per-variant benchmark override
      max_concurrency: 64
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

### Cloud

Provisions a cloud VM based on recipe GPU requirements, then deploys via SSH.

```bash
deplodock deploy cloud --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ --variant RTX5090 --dry-run
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

## VM Management

The `vm` command manages cloud GPU VM lifecycles. Currently supports GCP flex-start, which provisions new GPU capacity using `--provisioning-model=FLEX_START` (can take hours). Instances are ephemeral — `delete` removes them entirely.

### Create a VM

```bash
deplodock vm create gcp-flex-start --instance my-gpu-vm --zone us-central1-a --machine-type a2-highgpu-1g
deplodock vm create gcp-flex-start --instance my-gpu-vm --zone us-central1-a --machine-type e2-micro --wait-ssh
deplodock vm create gcp-flex-start --instance my-gpu-vm --zone us-central1-a --machine-type e2-micro --gcloud-args "--no-service-account --no-scopes" --dry-run
```

### Delete a VM

```bash
deplodock vm delete gcp-flex-start --instance my-gpu-vm --zone us-central1-a
deplodock vm delete gcp-flex-start --instance my-gpu-vm --zone us-central1-a --dry-run
```

### Create Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--instance` | (required) | GCP instance name |
| `--zone` | (required) | GCP zone (e.g. us-central1-a) |
| `--machine-type` | (required) | Machine type (e.g. a2-highgpu-1g) |
| `--max-run-duration` | `7d` | Max VM run time (10m–7d) |
| `--request-valid-for-duration` | `2h` | How long to wait for capacity |
| `--termination-action` | `DELETE` | Action when max-run-duration expires (`STOP` or `DELETE`) |
| `--image-family` | `debian-12` | Boot disk image family |
| `--image-project` | `debian-cloud` | Boot disk image project |
| `--gcloud-args` | - | Extra args passed to `gcloud compute instances create` |
| `--timeout` | `14400` | How long to poll for RUNNING status (seconds) |
| `--wait-ssh` | false | Wait for SSH after VM is RUNNING |
| `--wait-ssh-timeout` | `300` | SSH wait timeout in seconds |
| `--dry-run` | false | Print commands without executing |

### Delete Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--instance` | (required) | GCP instance name |
| `--zone` | (required) | GCP zone (e.g. us-central1-a) |
| `--dry-run` | false | Print commands without executing |

GCP project is inferred from `gcloud` config (no `--project` flag needed).

## Running Tests

```bash
pytest tests/ -v
```

## Benchmarking

The `bench` command reuses the deploy infrastructure — it loads a recipe, deploys the model via SSH, runs `vllm bench serve`, captures results, and tears down. No repo cloning needed.

### Configuration

`config.yaml` references recipe paths instead of inline model definitions:

```yaml
benchmark:
  local_results_dir: "results"
  model_dir: "/hf_models"

servers:
  - name: "rtx4090_x_1"
    ssh_key: "~/.ssh/id_ed25519"
    recipes:
      - recipe: "recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ"
        variant: "RTX4090"
```

Benchmark parameters (`max_concurrency`, `num_prompts`, `random_input_len`, `random_output_len`) are defined per-recipe in the recipe's `benchmark` section, not in `config.yaml`.

### Run Benchmarks

```bash
deplodock bench --dry-run                                         # Preview commands
deplodock bench --parallel                                        # Run benchmarks in parallel
deplodock bench --server my_server                                # Run for specific server
deplodock bench --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ  # Run for specific recipe
deplodock bench --parallel --force                                # Force re-run, skip cached results
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--force` | false | Force re-run even if results exist |
| `--server` | - | Run only for a specific server |
| `--recipe` | - | Run only for a specific recipe path |
| `--parallel` | false | Run servers in parallel |
| `--max-workers` | num servers | Max parallel server benchmarks |
| `--dry-run` | false | Print commands without executing |

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
│       │   ├── __init__.py          # Shared: recipe, compose, orchestration, DeployParams
│       │   ├── local.py             # Local target
│       │   ├── ssh.py               # SSH target
│       │   └── cloud.py             # Cloud target (provision VM + deploy)
│       ├── bench/
│       │   └── __init__.py          # Benchmark runner
│       ├── report/
│       │   └── __init__.py          # Report generator
│       └── vm/
│           ├── __init__.py          # VM command registration, shared helpers
│           ├── types.py             # VMConnectionInfo dataclass
│           ├── cloudrift.py         # CloudRift API provider
│           └── gcp_flex_start.py    # GCP flex-start provider
├── recipes/                         # Model deploy recipes
│   ├── GLM-4.6-FP8/
│   ├── GLM-4.6/
│   ├── GLM-4.5-Air-AWQ-4bit/
│   ├── Qwen3-Coder-480B-A35B-Instruct-AWQ/
│   ├── Qwen3-Coder-30B-A3B-Instruct-AWQ/
│   └── Meta-Llama-3.3-70B-Instruct-AWQ-INT4/
├── tests/                           # pytest tests (see [tests/ARCHITECTURE.md](tests/ARCHITECTURE.md))
├── config.yaml                      # Benchmark configuration
└── Makefile                         # Build automation
```
