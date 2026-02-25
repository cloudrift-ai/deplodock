# Deplodock

Tools for deploying and benchmarking LLM inference on GPU servers. Supports **vLLM** and **SGLang** engines.

## Project Structure

- [deplodock/](deplodock/) — Python package
  - [deplodock.py](deplodock/deplodock.py) — CLI entrypoint
  - [hardware.py](deplodock/hardware.py) — GPU specs and instance type mapping
  - [commands/](deplodock/commands/) — CLI layer (thin argparse handlers, see [ARCHITECTURE.md](deplodock/commands/ARCHITECTURE.md))
    - [deploy/](deplodock/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](deplodock/commands/bench/) — `bench` command
    - [teardown.py](deplodock/commands/teardown.py) — `teardown` command
    - [report/](deplodock/commands/report/) — `report` command
    - [vm/](deplodock/commands/vm/) — `vm create/delete` commands (GCP, CloudRift)
  - [recipe/](deplodock/recipe/) — Recipe loading, dataclass types, engine flag mapping (see [ARCHITECTURE.md](deplodock/recipe/ARCHITECTURE.md))
  - [deploy/](deplodock/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](deplodock/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](deplodock/benchmark/) — Benchmark tracking, config, task enumeration, execution
  - [planner/](deplodock/planner/) — Groups benchmark tasks into execution groups for VM allocation
  - [report/](deplodock/report/) — Excel report generation from benchmark results
- [recipes/](recipes/) — Model deploy recipes (YAML configs per model)
- [experiments/](experiments/) — Experiment parameter sweeps (self-contained recipe + results)
- [docs/](docs/) — Technical notes and engine-specific guides
  - [sglang-awq-moe.md](docs/sglang-awq-moe.md) — SGLang quantization for AWQ MoE models
- [tests/](tests/) — pytest tests (see [ARCHITECTURE.md](tests/ARCHITECTURE.md))
- [utils/](utils/) — Standalone utility scripts
- [config.yaml](config.yaml) — Benchmark configuration
- [Makefile](Makefile) — Build automation
- [pyproject.toml](pyproject.toml) — Package metadata and tool config

## Quick Start

### Install

```bash
git clone https://github.com/cloudrift-ai/deplodock.git
cd deplodock
make setup
```

### Deploy a Model

```bash
deplodock deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --server user@host
```

### Deploy Locally

```bash
deplodock deploy local \
  --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ
```

### Teardown

```bash
deplodock deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --server user@host \
  --teardown
```

### Dry Run

Preview commands without executing:

```bash
deplodock deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --server user@host \
  --dry-run
```

## Recipes

Recipes are declarative YAML configs in `recipes/<model>/recipe.yaml`. Each recipe defines a model, engine settings, and a `matrices` section for benchmark configurations.

### Format

```yaml
model:
  huggingface: "org/model-name"

engine:
  llm:
    tensor_parallel_size: 8
    pipeline_parallel_size: 1
    gpu_memory_utilization: 0.9
    context_length: 16384
    max_concurrent_requests: 512
    vllm:
      image: "vllm/vllm-openai:latest"
      extra_args: "--kv-cache-dtype fp8"    # Flags not covered by named fields

benchmark:
  max_concurrency: 128
  num_prompts: 256
  random_input_len: 8000
  random_output_len: 8000

matrices:
  # Simple single-point entry
  - deploy.gpu: "NVIDIA H200 141GB"
    deploy.gpu_count: 8

  # Override engine and benchmark settings
  - deploy.gpu: "NVIDIA H100 80GB"
    deploy.gpu_count: 8
    engine.llm.max_concurrent_requests: 256
    benchmark.max_concurrency: 64

  # Concurrency sweep (8 runs from one entry)
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    benchmark.max_concurrency: [1, 2, 4, 8, 16, 32, 64, 128]

  # Correlated engine+bench sweep (3 zip runs)
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    engine.llm.max_concurrent_requests: [128, 256, 512]
    benchmark.max_concurrency: [128, 256, 512]
```

Matrix entries use **dot-notation** for all parameter paths. Scalars are broadcast; lists are zipped (all lists in one entry must have the same length). `deploy.gpu` is required in each entry.

Engine-agnostic fields (`tensor_parallel_size`, `context_length`, etc.) live at `engine.llm`. Engine-specific fields (`image`, `extra_args`) nest under `engine.llm.vllm` or `engine.llm.sglang`.

### SGLang Matrix Entry Example

To benchmark with SGLang alongside vLLM, add a matrix entry with `engine.llm.sglang.*` overrides:

```yaml
matrices:
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
    engine.llm.sglang.image: "lmsysorg/sglang:latest"
    engine.llm.sglang.extra_args: "--quantization moe_wna16"  # Required for AWQ MoE models
```

For AWQ-quantized MoE models on SGLang, `--quantization moe_wna16` is required in `extra_args`. See [docs/sglang-awq-moe.md](docs/sglang-awq-moe.md) for details.

### Optional Deploy Section

For `deploy cloud` (which needs GPU info without matrix expansion), add a `deploy` section to the base recipe:

```yaml
deploy:
  gpu: "NVIDIA GeForce RTX 5090"
  gpu_count: 1
```

### Named Fields → CLI Flags

| Recipe YAML key           | vLLM CLI flag              | SGLang CLI flag          |
|---------------------------|----------------------------|--------------------------|
| `tensor_parallel_size`    | `--tensor-parallel-size`   | `--tp`                   |
| `pipeline_parallel_size`  | `--pipeline-parallel-size` | `--dp`                   |
| `gpu_memory_utilization`  | `--gpu-memory-utilization` | `--mem-fraction-static`  |
| `context_length`          | `--max-model-len`          | `--context-length`       |
| `max_concurrent_requests` | `--max-num-seqs`           | `--max-running-requests` |

These flags must **not** appear in `extra_args` — `load_recipe()` validates this and raises an error on duplicates.

## Experiments

Experiments are self-contained parameter sweeps that live in `experiments/`. Each experiment directory contains a `recipe.yaml` and stores its results alongside it. The directory structure follows `experiments/{model_name}/{experiment_name}/`.

### Example: Optimal max_concurrent_requests on RTX 5090

```bash
deplodock bench experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090
```

Results are saved directly in the experiment directory:
```
experiments/Qwen3-Coder-30B-A3B-Instruct-AWQ/optimal_mcr_rtx5090/
  recipe.yaml
  2026-02-24_19-13-50_abc12345/
    manifest.json
    optimal_mcr_rtx5090/
      recipe.yaml
      RTX5090_mcr8_c8_vllm_benchmark.txt
      RTX5090_mcr12_c12_vllm_benchmark.txt
      ...
```

## Deploy Targets

### Local

Runs docker compose directly on the current machine.

```bash
deplodock deploy local --recipe <path> [--dry-run]
```

### SSH

Deploys to a remote server via SSH + SCP.

```bash
deplodock deploy ssh --recipe <path> --server user@host [--dry-run]
```

### Cloud

Provisions a cloud VM based on recipe GPU requirements (from the `deploy` section), then deploys via SSH.

```bash
deplodock deploy cloud --recipe <path> [--name <vm-name>] [--dry-run]
```

### Common Flags

| Flag          | Required  | Default       | Description                          |
|---------------|-----------|---------------|--------------------------------------|
| `--recipe`    | Yes       | -             | Path to recipe directory             |
| `--hf-token`  | No        | `$HF_TOKEN`   | HuggingFace token                    |
| `--model-dir` | No        | `/mnt/models` | Model cache dir                      |
| `--teardown`  | No        | false         | Stop containers instead of deploying |
| `--dry-run`   | No        | false         | Print commands without executing     |

### SSH-only Flags

| Flag         | Required  | Default             | Description             |
|--------------|-----------|---------------------|-------------------------|
| `--server`   | Yes       | -                   | SSH address (user@host) |
| `--ssh-key`  | No        | `~/.ssh/id_ed25519` | SSH key path            |
| `--ssh-port` | No        | 22                  | SSH port                |

### Cloud-only Flags

| Flag        | Required  | Default             | Description          |
|-------------|-----------|---------------------|----------------------|
| `--name`    | No        | `cloud-deploy`      | VM name prefix       |
| `--ssh-key` | No        | `~/.ssh/id_ed25519` | SSH private key path |

## VM Management

The `vm` command manages cloud GPU VM lifecycles. Supports GCP and CloudRift providers. Instances are ephemeral — `delete` removes them entirely.

### GCP

```bash
deplodock vm create gcp --instance my-gpu-vm --zone us-central1-a --machine-type a2-highgpu-1g
deplodock vm create gcp --instance my-gpu-vm --zone us-central1-a --machine-type e2-micro --wait-ssh
deplodock vm create gcp --instance my-gpu-vm --zone us-central1-a --machine-type e2-micro --gcloud-args "--no-service-account --no-scopes" --dry-run
deplodock vm delete gcp --instance my-gpu-vm --zone us-central1-a
```

#### GCP Create Flags

| Flag                           | Default        | Description                                               |
|--------------------------------|----------------|-----------------------------------------------------------|
| `--instance`                   | (required)     | GCP instance name                                         |
| `--zone`                       | (required)     | GCP zone (e.g. us-central1-a)                             |
| `--machine-type`               | (required)     | Machine type (e.g. a2-highgpu-1g)                         |
| `--provisioning-model`         | `FLEX_START`   | Provisioning model (`FLEX_START`, `SPOT`, or `STANDARD`)  |
| `--max-run-duration`           | `7d`           | Max VM run time (10m–7d)                                  |
| `--request-valid-for-duration` | `2h`           | How long to wait for capacity                             |
| `--termination-action`         | `DELETE`       | Action when max-run-duration expires (`STOP` or `DELETE`) |
| `--image-family`               | `debian-12`    | Boot disk image family                                    |
| `--image-project`              | `debian-cloud` | Boot disk image project                                   |
| `--gcloud-args`                | -              | Extra args passed to `gcloud compute instances create`    |
| `--timeout`                    | `14400`        | How long to poll for RUNNING status (seconds)             |
| `--wait-ssh`                   | false          | Wait for SSH after VM is RUNNING                          |
| `--wait-ssh-timeout`           | `300`          | SSH wait timeout in seconds                               |
| `--ssh-gateway`                | -              | SSH gateway host for ProxyJump (e.g. gcp-ssh-gateway)     |
| `--dry-run`                    | false          | Print commands without executing                          |

#### GCP Delete Flags

| Flag         | Default    | Description                      |
|--------------|------------|----------------------------------|
| `--instance` | (required) | GCP instance name                |
| `--zone`     | (required) | GCP zone (e.g. us-central1-a)    |
| `--dry-run`  | false      | Print commands without executing |

GCP project is inferred from `gcloud` config (no `--project` flag needed).

### CloudRift

```bash
deplodock vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
deplodock vm delete cloudrift --instance-id <id>
```

#### CloudRift Create Flags

| Flag              | Default              | Description                       |
|-------------------|----------------------|-----------------------------------|
| `--instance-type` | (required)           | Instance type (e.g. rtx4090.1)    |
| `--ssh-key`       | (required)           | Path to SSH public key file       |
| `--api-key`       | `$CLOUDRIFT_API_KEY` | CloudRift API key                 |
| `--image-url`     | Ubuntu 24.04         | VM image URL                      |
| `--ports`         | `22,8000`            | Comma-separated ports to open     |
| `--timeout`       | `600`                | Seconds to wait for Active status |
| `--dry-run`       | false                | Print requests without executing  |

#### CloudRift Delete Flags

| Flag            | Default              | Description                      |
|-----------------|----------------------|----------------------------------|
| `--instance-id` | (required)           | CloudRift instance ID            |
| `--api-key`     | `$CLOUDRIFT_API_KEY` | CloudRift API key                |
| `--dry-run`     | false                | Print requests without executing |

## Benchmarking

The `bench` command accepts recipe directories as positional arguments. It loads each recipe, provisions cloud VMs, deploys the model, runs `vllm bench serve`, captures results, and tears down. Recipes sharing the same model and GPU type are grouped onto the same VM.

### Configuration

`config.yaml` defines global benchmark settings and provider config:

```yaml
benchmark:
  local_results_dir: "results/intermediate"
  model_dir: "/hf_models"

pricing:
  h100: 1.91
  h200: 2.06

providers:
  gcp:
    zone: "us-central1-b"
    provisioning_model: "SPOT"
    image_family: "common-cu128-ubuntu-2204-nvidia-570"
    image_project: "deeplearning-platform-release"
```

Benchmark parameters (`max_concurrency`, `num_prompts`, `random_input_len`, `random_output_len`) are defined per-recipe in the recipe's `benchmark` section, not in `config.yaml`.

### Run Benchmarks

```bash
deplodock bench recipes/*                                    # Run all recipes (results in each recipe dir)
deplodock bench experiments/.../optimal_mcr_rtx5090          # Run an experiment
deplodock bench recipes/* --max-workers 2                    # Limit parallel execution groups
deplodock bench recipes/* --dry-run                          # Preview commands
```

| Flag            | Default             | Description                            |
|-----------------|---------------------|----------------------------------------|
| `recipes`       | (required)          | Recipe directories (positional args)   |
| `--ssh-key`     | `~/.ssh/id_ed25519` | SSH private key path                   |
| `--config`      | `config.yaml`       | Path to configuration file             |
| `--max-workers` | num groups          | Max parallel execution groups          |
| `--dry-run`     | false               | Print commands without executing       |
| `--no-teardown` | false               | Skip teardown and VM deletion (saves `instances.json` for later cleanup) |

Results are always stored in `{recipe_dir}/{timestamp}_{hash}/` — each recipe directory holds its own run directories alongside `recipe.yaml`.

### Teardown

Clean up VMs left running by `bench --no-teardown`:

```bash
deplodock teardown results/intermediate/2026-02-24_12-00-00_abc12345
deplodock teardown results/intermediate/2026-02-24_12-00-00_abc12345 --ssh-key ~/.ssh/id_ed25519
```

| Flag       | Default             | Description                       |
|------------|---------------------|-----------------------------------|
| `run_dir`  | (required)          | Run directory with `instances.json` (positional arg) |
| `--ssh-key`| `~/.ssh/id_ed25519` | SSH private key path              |

### Generate Reports

```bash
deplodock report                                      # Default: results/ -> results/benchmark_report.xlsx
deplodock report --results-dir results/custom --output results/custom/report.xlsx
```

| Flag            | Default                         | Description                            |
|-----------------|---------------------------------|----------------------------------------|
| `--config`      | `config.yaml`                   | Path to configuration file             |
| `--results-dir` | `results`                       | Directory containing benchmark results |
| `--output`      | `results/benchmark_report.xlsx` | Output Excel file path                 |

## Running Tests

```bash
make test
```

## Linting & Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `pyproject.toml`.

```bash
make lint      # check for lint errors and formatting issues
make format    # auto-fix formatting and lint violations
```
