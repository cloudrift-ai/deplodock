<p align="center">
  <img src="logo.png" alt="DeploDock" width="300">
</p>

<p align="center">
  <a href="https://pypi.org/project/deplodock/"><img src="https://img.shields.io/pypi/v/deplodock" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/deplodock/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/deplodock/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

Benchmark and deploy optimized LLM models on GPU servers with **vLLM** or **SGLang**. Chose from a list of optimized recipes for popular models or create your own with custom configurations. Run benchmarks across different GPU types and configurations, track results, and share experiments with the community.

## Project Structure

- [deplodock/](deplodock/) — Python package
  - [deplodock.py](deplodock/deplodock.py) — CLI entrypoint
  - [logging_setup.py](deplodock/logging_setup.py) — CLI logging configuration
  - [hardware.py](deplodock/hardware.py) — GPU specs and instance type mapping
  - [detect.py](deplodock/detect.py) — GPU detection via PCI sysfs (local and remote)
  - [commands/](deplodock/commands/) — CLI layer (thin argparse handlers, see [ARCHITECTURE.md](deplodock/commands/ARCHITECTURE.md))
    - [deploy/](deplodock/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](deplodock/commands/bench/) — `bench` command
    - [teardown.py](deplodock/commands/teardown.py) — `teardown` command
    - [vm/](deplodock/commands/vm/) — `vm create/delete` commands (GCP, CloudRift)
  - [recipe/](deplodock/recipe/) — Recipe loading, dataclass types, engine flag mapping (see [ARCHITECTURE.md](deplodock/recipe/ARCHITECTURE.md))
  - [deploy/](deplodock/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](deplodock/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](deplodock/benchmark/) — Benchmark tracking, config, task enumeration, execution
  - [planner/](deplodock/planner/) — Groups benchmark tasks into execution groups for VM allocation
- [recipes/](recipes/) — Model deploy recipes (YAML configs per model)
- [experiments/](experiments/) — Experiment parameter sweeps (self-contained recipe + results)
- [docker/](docker/) — Custom Docker images (e.g., vLLM ROCm for MI350X)
- [docs/](docs/) — Technical notes and engine-specific guides
  - [sglang-awq-moe.md](docs/sglang-awq-moe.md) — SGLang quantization for AWQ MoE models
- [tests/](tests/) — pytest tests (see [ARCHITECTURE.md](tests/ARCHITECTURE.md))
- [scripts/](scripts/) — Analysis and visualization scripts
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
  --ssh user@host
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
  --ssh user@host \
  --teardown
```

### Dry Run

Preview commands without executing:

```bash
deplodock deploy ssh \
  --recipe recipes/GLM-4.6-FP8 \
  --ssh user@host \
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
      image: "vllm/vllm-openai:v0.17.0"
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

`deploy.driver_version` and `deploy.cuda_version` (optional) request a specific NVIDIA driver / CUDA toolkit on the target host. If the installed version already matches (prefix-match — `"550"` matches `550.127.05`), provisioning is a no-op. On a mismatch, a remote (`ssh`/`cloud`) deploy installs the requested version, reboots the host, and waits for SSH to come back. Local deploys refuse to run privileged commands and will error out instead — these fields are intended for remote machines only.

Engine-agnostic fields (`tensor_parallel_size`, `context_length`, etc.) live at `engine.llm`. Engine-specific fields (`image`, `extra_args`) nest under `engine.llm.vllm` or `engine.llm.sglang`.

### SGLang Matrix Entry Example

To benchmark with SGLang alongside vLLM, add a matrix entry with `engine.llm.sglang.*` overrides:

```yaml
matrices:
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
    engine.llm.sglang.image: "lmsysorg/sglang:v0.5.9"
```

### Named Fields → CLI Flags

| Recipe YAML key           | vLLM CLI flag              | SGLang CLI flag          |
|---------------------------|----------------------------|--------------------------|
| `tensor_parallel_size`    | `--tensor-parallel-size`   | `--tp`                   |
| `pipeline_parallel_size`  | `--pipeline-parallel-size` | `--pp`                   |
| `data_parallel_size`      | `--data-parallel-size`     | `--dp`                   |
| `gpu_memory_utilization`  | `--gpu-memory-utilization` | `--mem-fraction-static`  |
| `context_length`          | `--max-model-len`          | `--context-length`       |
| `max_concurrent_requests` | `--max-num-seqs`           | `--max-running-requests` |

These flags must **not** appear in `extra_args` — `load_recipe()` validates this and raises an error on duplicates.

### Command Recipes (Generic Workload)

A recipe may declare a `command` block instead of `engine.llm` to run an arbitrary tool on the provisioned VM (e.g. a microbenchmark, a profiling sweep, or `nvidia-smi`). The harness expands the matrix, renders the command template per variant, runs it on the VM, and pulls back result files.

```yaml
command:
  stage: ["scripts"]              # repo paths to ship to the VM; empty = no staging
  run: |
    nvidia-smi --query-gpu=name,memory.used --format=csv > $task_dir/result.csv
    echo "marker,$marker" >> $task_dir/result.csv
  result_files:                    # filenames or shell globs (expanded on the remote)
    - result.csv
    - "*.log"
  timeout: 60

matrices:
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1
    marker: [a, b, c]
```

The `run` template uses `string.Template` `$var` syntax. Substitution variables are the variant params (flattened to leaf names — `deploy.gpu` → `gpu`, `marker` → `marker`) plus harness-injected `$task_dir`, `$gpu_device_ids`, and `$repo_dir` (when staging is configured). `command` and `engine.llm` are mutually exclusive.

Staging uses `git ls-files --cached --others --exclude-standard <paths>` so unversioned edits ride along without a commit, while gitignored files are excluded. Each pulled result file lands in the run directory as `{variant}_{basename}`.

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
    tasks.json
    recipe.yaml
    RTX5090_mcr8_c8_vllm_benchmark.txt
    RTX5090_mcr12_c12_vllm_benchmark.txt
    ...
```

## CI Benchmark Workflow

External developers can submit experiments via pull requests. A maintainer triggers benchmarks by commenting `/run-experiment` on the PR.

### How It Works

1. **Submit a PR** with an experiment definition in `experiments/{model}/{experiment}/recipe.yaml`
2. **A maintainer reviews** and comments `/run-experiment` on the PR
3. **CI runs benchmarks** on cloud GPUs, commits results back to the PR branch
4. **Review results** in the PR comment summary and committed files

### Trigger Modes

```
/run-experiment                                                        # Auto-detect: benchmarks all experiments changed in the PR
/run-experiment experiments/MyModel/my_experiment                       # Explicit: benchmark specific experiment(s)
/run-experiment experiments/MyModel/my_experiment --gpu-concurrency 2   # Split groups across 2 VMs each
```

Only users with **write** or **admin** access to the repository can trigger benchmarks.

### Fork PRs

For the workflow to push results back to a fork's branch, the PR must have **"Allow edits from maintainers"** checked (this is the GitHub default). If unchecked, results are still available as downloadable workflow artifacts.

## Deploy Targets

### Local

Runs docker compose directly on the current machine.

```bash
deplodock deploy local --recipe <path> [--dry-run]
```

### SSH

Deploys to a remote server via SSH + SCP.

```bash
deplodock deploy ssh --recipe <path> --ssh user@host[:port] [--dry-run]
```

### Cloud

Provisions a cloud VM and deploys via SSH. Requires `--gpu` and `--gpu-count` to select the matching matrix entry from the recipe.

```bash
deplodock deploy cloud --recipe <path> --gpu "NVIDIA GeForce RTX 5090" --gpu-count 1 [--dry-run]
```

### Hardware-Aware Deploy

When deploying locally or via SSH, deplodock auto-detects the target GPU by scanning PCI sysfs device IDs and selects the matching `matrices` entry from the recipe. If more GPUs are available than the recipe's base configuration needs, a scale-out strategy is applied.

### Common Flags

| Flag                    | Required  | Default             | Description                                                  |
|-------------------------|-----------|---------------------|--------------------------------------------------------------|
| `--recipe`              | Yes       | -                   | Path to recipe directory                                     |
| `--hf-token`            | No        | `$HF_TOKEN`         | HuggingFace token                                            |
| `--model-dir`           | No        | `/mnt/models`       | Model cache dir                                              |
| `--teardown`            | No        | false               | Stop containers instead of deploying                         |
| `--dry-run`             | No        | false               | Print commands without executing                             |
| `--gpu`                 | No        | auto-detect         | Override GPU name (skips detection)                          |
| `--gpu-count`           | No        | auto-detect         | Override GPU count (skips count detection)                   |
| `--scale-out-strategy`  | No        | `data-parallelism`  | Scale-out: `data-parallelism` or `replica-parallelism`       |

### SSH-only Flags

| Flag         | Required  | Default             | Description                                          |
|--------------|-----------|---------------------|------------------------------------------------------|
| `--ssh`      | Yes       | -                   | SSH target `USER@HOST[:PORT]` (default port 22)      |
| `--ssh-key`  | No        | `~/.ssh/id_ed25519` | SSH key path                                         |

The same `--ssh USER@HOST[:PORT]` syntax is used by `deplodock bench --ssh ...`.

### Cloud-only Flags

| Flag          | Required  | Default             | Description                            |
|---------------|-----------|---------------------|----------------------------------------|
| `--gpu`       | Yes       | -                   | GPU name (selects matching matrix entry)|
| `--gpu-count` | Yes       | -                   | GPU count (selects matching matrix entry)|
| `--name`      | No        | `cloud-deploy`      | VM name prefix                         |
| `--ssh-key`   | No        | `~/.ssh/id_ed25519` | SSH private key path                   |

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

### Run Benchmarks

```bash
deplodock bench recipes/*                                    # Run all recipes (results in each recipe dir)
deplodock bench experiments/.../optimal_mcr_rtx5090          # Run an experiment
deplodock bench recipes/* --gpu-concurrency 4                # Number of VMs per GPU type to spin up
deplodock bench recipes/* --dry-run                          # Preview commands
deplodock bench recipes/* --local                            # Run on the local machine
deplodock bench recipes/* --ssh user@host1 --ssh user@host2  # Run on a fixed pool of pre-allocated hosts
```

| Flag                 | Default             | Description                                                              |
|----------------------|---------------------|--------------------------------------------------------------------------|
| `recipes`            | (required)          | Recipe directories (positional args)                                     |
| `--ssh-key`          | `~/.ssh/id_ed25519` | SSH private key path                                                     |
| `--config`           | `config.yaml`       | Path to configuration file                                               |
| `--max-workers`      | num groups          | Max parallel execution groups                                            |
| `--gpu-concurrency`  | 1                   | Split each (model, GPU) group across up to N VMs                         |
| `--dry-run`          | false               | Print commands without executing                                         |
| `--no-teardown`      | false               | Skip teardown and VM deletion (saves `instances.json` for later cleanup) |
| `--local`            | false               | Run on the local machine via ssh to 127.0.0.1 (skips cloud provisioning) |
| `--ssh USER@HOST[:PORT]` | none            | Pre-allocated SSH host (repeatable). Skips cloud provisioning            |

When `--local` and/or `--ssh` are supplied, deplodock detects each host's GPU via PCI sysfs and verifies that every planned execution group can run on at least one of the supplied hosts (matching `deploy.gpu` and sufficient `deploy.gpu_count`). If any group is unsatisfied, the run aborts before any work starts. Fixed hosts are assumed to be already provisioned (docker, NVIDIA toolkit, etc.) and are never deleted at the end of the run.

> **Note:** `--local` runs the workload over SSH to `127.0.0.1` (the same code path used for remote hosts). This requires a running SSH server on localhost and that your `--ssh-key` (default `~/.ssh/id_ed25519`) is listed in `~/.ssh/authorized_keys`. Quick check: `ssh -i ~/.ssh/id_ed25519 $USER@127.0.0.1 echo ok`.

Results are always stored in `{recipe_dir}/{timestamp}_{hash}/` — each recipe directory holds its own run directories alongside `recipe.yaml`.

### Teardown

Clean up VMs left running by `bench --no-teardown`:

```bash
deplodock teardown results/intermediate/2026-02-24_12-00-00_abc12345
deplodock teardown results/intermediate/2026-02-24_12-00-00_abc12345 --ssh-key ~/.ssh/id_ed25519
```

| Flag        | Default              | Description                                          |
|-------------|----------------------|------------------------------------------------------|
| `run_dir`   | (required)           | Run directory with `instances.json` (positional arg) |
| `--ssh-key` | `~/.ssh/id_ed25519`  | SSH private key path                                 |

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
