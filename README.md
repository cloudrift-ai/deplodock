<p align="center">
  <img src="logo.png" alt="DeploDock" width="300">
</p>

<p align="center">
  <a href="https://pypi.org/project/deplodock/"><img src="https://img.shields.io/pypi/v/deplodock" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/deplodock/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/deplodock/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

**Compile → Benchmark → Deploy** any LLM on any GPU. vLLM, SGLang, or create your own specialized deployment using a hackable compiler.

## Install

```bash
git clone https://github.com/cloudrift-ai/deplodock.git
cd deplodock && make setup
```

## Compile

A hackable PyTorch → Graph IR → CUDA compiler. Trace any `nn.Module`, fuse it into one kernel, run it, and inspect the emitted CUDA. See the blog post: [*A Principled ML Compiler Stack in 5,000 Lines of Python*](https://www.cloudrift.ai/blog/building-gpu-compiler-from-scratch-1).

```bash
# Compile a single layer
deplodock compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))"
# Benchmark, profile and optimize kernels locally
deplodock run --bench --profile -c "torch.nn.Softmax(dim=-1)(torch.randn(1, 28, 2048, 2048))"
# Compile full model from HuggingFace (will download weights)
deplodock compile Qwen/Qwen3-Embedding-0.6B
```

Layer-norm-style reduction (two reductions, broadcast subtract, elementwise chain) fused into two kernels:

```bash
deplodock compile -c "
class LN(torch.nn.Module):
    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) * torch.rsqrt(v + 1e-6)
LN()(torch.randn(64, 2048))"
```

Principled compilation stack with six IR stages, each printable on demand via `--ir <stage>`:

1. **Torch IR** — captures the FX graph as a 1:1 mirror of PyTorch's op set (`rmsnorm`, `linear`, `softmax`, ...)
2. **Tensor IR** — decomposes every Torch op into three primitives: `Elementwise`, `Reduction`, and `IndexMap`
3. **Loop IR** — lifts each primitive to a `LoopOp` and fuses
4. **Tile IR** — schedules kernels onto GPU
5. **Kernel IR** — materializes the schedule into framework-agnostic hardware primitives
6. **CUDA** — optimized CUDA code ready for `nvcc`


**Readable Schedule**: `deplodock compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir tile`
```
kernel k_rms_norm_reduce  inputs: rms_norm_mean_count, rms_norm_eps, x, p_weight  outputs: rms_norm
    in0 = load rms_norm_mean_count[0]
    in1 = load rms_norm_eps[0]
    Tile(axes=(a0:256=THREAD, a1:32=BLOCK)):
        x_smem = Stage(x, origin=(0, a1, 0), slab=(a2:2048@2)) async
        p_weight_smem = Stage(p_weight, origin=(0), slab=(a3:2048@0)) async
        StridedLoop(a2 = a0; < 2048; += 256):  # reduce
            in2 = load x_smem[a2]
            v0 = multiply(in2, in2)
            acc0 <- add(acc0, v0)
        v1 = divide(acc0, in0)
        v2 = add(v1, in1)
        v3 = rsqrt(v2)
        StridedLoop(a3 = a0; < 2048; += 256):  # free
            in3 = load x_smem[a3]
            in4 = load p_weight_smem[a3]
            v4 = multiply(in3, v3)
            v5 = multiply(v4, in4)
            rms_norm[0, a1, a3] = v5
```

**Optimized CUDA kernel**: `deplodock compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir cuda`

```c
extern "C" __global__
__launch_bounds__(256) void k_rms_norm_reduce(const float* x, const float* p_weight, float* rms_norm) {
    float in0 = 2048.0f;
    float in1 = 1e-06f;
    {
        int a1 = blockIdx.x;
        int a0 = threadIdx.x;
        float acc0 = 0.0f;
        __syncthreads();
        __shared__ float x_smem[2048];
        for (int x_smem_flat = a0; x_smem_flat < 2048; x_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&x_smem[x_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&x[a1 * 2048 + x_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        __shared__ float p_weight_smem[2048];
        for (int p_weight_smem_flat = a0; p_weight_smem_flat < 2048; p_weight_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&p_weight_smem[p_weight_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&p_weight[p_weight_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        for (int a2 = a0; a2 < 2048; a2 += 256) {
            float in2 = x_smem[a2];
            float v0 = in2 * in2;
            acc0 += v0;
        }
        __shared__ float acc0_smem[256];
        acc0_smem[a0] = acc0;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (a0 < s) {
                acc0_smem[a0] = acc0_smem[a0] + acc0_smem[a0 + s];
            }
            __syncthreads();
        }
        __syncthreads();
        float acc0_b = acc0_smem[0];
        float v1 = acc0_b / in0;
        float v2 = v1 + in1;
        float v3 = rsqrtf(v2);
        for (int a3 = a0; a3 < 2048; a3 += 256) {
            float in3 = x_smem[a3];
            float in4 = p_weight_smem[a3];
            float v4 = in3 * v3;
            float v5 = v4 * in4;
            rms_norm[a1 * 2048 + a3] = v5;
        }
    }
}
```

## Benchmark

```bash
deplodock bench recipes/*                                    # All recipes
deplodock bench experiments/.../optimal_mcr_rtx5090          # An experiment
deplodock bench recipes/* --filter "deploy.gpu=*5090*"       # Subset
deplodock bench recipes/* --gpu-concurrency 4                # Parallel VMs per GPU
deplodock bench recipes/* --local                            # On this machine
deplodock bench recipes/* --ssh user@host1 --ssh user@host2  # Pre-allocated hosts
```

External contributors: open a PR with an experiment under `experiments/{model}/{name}/`, then a maintainer triggers a cloud run by commenting `/run-experiment` on the PR.

## Deploy

```bash
# Remote server via SSH
deplodock deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host

# Local Docker Compose
deplodock deploy local --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ

# Cloud (auto-provisions a VM)
deplodock deploy cloud --recipe recipes/GLM-4.6-FP8 --gpu "NVIDIA H200 141GB" --gpu-count 8

# Teardown / preview
deplodock deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host --teardown
deplodock deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host --dry-run
```

## Serve (compiled embeddings via vLLM)

```bash
# vLLM's OpenAI shell (/v1/embeddings, tokenizer, scheduler, pooler) over deplodock-compiled kernels
pip install -e ".[compile,serving]"
deplodock serve Qwen/Qwen3-Embedding-0.6B                  # extra flags pass through to vllm serve

curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"Hello"}'

# One-shot benchmark (vllm bench serve against the started server), and the raw-vLLM baseline
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32 --stock
```

See [`deplodock/serving/ARCHITECTURE.md`](deplodock/serving/ARCHITECTURE.md); embedding recipes
(`recipes/Qwen3-Embedding-*`) A/B this against stock vLLM via `deplodock bench`.

## Recipe

```yaml
model:
  huggingface: "org/model-name"

engine:
  llm:
    tensor_parallel_size: 8
    gpu_memory_utilization: 0.9
    context_length: 16384
    max_concurrent_requests: 512
    vllm:
      image: "vllm/vllm-openai:v0.17.0"
      extra_args: "--kv-cache-dtype fp8"

benchmark:
  max_concurrency: 128
  num_prompts: 256
  random_input_len: 8000
  random_output_len: 8000

# Cross-product: 3 GPUs × 2 concurrency configs = 6 variants
matrices:
  cross:
    deploy.gpu_count: 1
    deploy.gpu:
      - "NVIDIA GeForce RTX 5090"
      - "NVIDIA H100 80GB"
      - "NVIDIA H200 141GB"
    zip:
      engine.llm.max_concurrent_requests: [128, 512]
      benchmark.max_concurrency: [128, 512]
```

Generic workload (run any tool on the VM, pull back result files):

```yaml
command:
  stage: ["scripts"]
  run: |
    nvidia-smi --query-gpu=name,memory.used --format=csv > $task_dir/result.csv
  result_files: ["result.csv"]
  timeout: 60

matrices:
  deploy.gpu: "NVIDIA GeForce RTX 5090"
  deploy.gpu_count: 1
```

## Virtual Machine Management

```bash
# GCP
deplodock vm create gcp --instance my-vm --zone us-central1-a --machine-type a2-highgpu-1g
deplodock vm delete gcp --instance my-vm --zone us-central1-a

# CloudRift
deplodock vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
deplodock vm delete cloudrift --instance-id <id>
```

## Development

```bash
make test      # run pytest
make lint      # ruff check + format check
make format    # auto-fix
```

## Project Structure

- [deplodock/](deplodock/) — Python package
  - [deplodock.py](deplodock/deplodock.py) — CLI entrypoint
  - [logging_setup.py](deplodock/logging_setup.py) — CLI logging configuration
  - [hardware.py](deplodock/hardware.py) — GPU specs and instance type mapping
  - [detect.py](deplodock/detect.py) — GPU detection via PCI sysfs (local and remote)
  - [redact.py](deplodock/redact.py) — Secret redaction for logs and dumps
  - [commands/](deplodock/commands/) — CLI layer (thin argparse handlers, see [ARCHITECTURE.md](deplodock/commands/ARCHITECTURE.md))
    - [deploy/](deplodock/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](deplodock/commands/bench/) — `bench` command
    - [vm/](deplodock/commands/vm/) — `vm create/delete` commands (GCP, CloudRift)
    - [teardown.py](deplodock/commands/teardown.py) — `teardown` command
    - [pull.py](deplodock/commands/pull.py) — `pull` command (download HF model)
    - [trace.py](deplodock/commands/trace.py) — `trace` command (PyTorch → Graph IR)
    - [compile.py](deplodock/commands/compile.py) — `compile` command (decomposition → optimization → fusion → kernel/CUDA lowering)
    - [run.py](deplodock/commands/run.py) — `run` command (compile + execute on CUDA backend, optional benchmarks)
    - [inspect_graph.py](deplodock/commands/inspect_graph.py) — `inspect` command (graph summary)
  - [compiler/](deplodock/compiler/) — PyTorch → Graph IR → CUDA compiler (see [ARCHITECTURE.md](deplodock/compiler/ARCHITECTURE.md))
    - [graph.py](deplodock/compiler/graph.py) — `Graph`, `Node`, `Tensor`, `Hints` container
    - [ir/](deplodock/compiler/ir/) — per-dialect op definitions (torch / tensor / loop / kernel / cuda) (see [ARCHITECTURE.md](deplodock/compiler/ir/ARCHITECTURE.md))
    - [trace/](deplodock/compiler/trace/) — PyTorch/HuggingFace → Graph IR capture (see [ARCHITECTURE.md](deplodock/compiler/trace/ARCHITECTURE.md))
    - [pipeline/](deplodock/compiler/pipeline/) — rewrite engine + passes + dump hooks (see [ARCHITECTURE.md](deplodock/compiler/pipeline/ARCHITECTURE.md))
    - [rules/](deplodock/compiler/rules/) — rewrite rules (decomposition, optimization, fusion, lowering)
    - [program/](deplodock/compiler/program/) — kernel program assembly (LoopOp → KernelOp → CudaOp)
    - [cuda/](deplodock/compiler/cuda/) — CUDA source rendering and runtime helpers
    - [backend/](deplodock/compiler/backend/) — numpy / loop / CUDA execution (see [ARCHITECTURE.md](deplodock/compiler/backend/ARCHITECTURE.md))
      - [cuda/](deplodock/compiler/backend/cuda/) — CUDA backend internals (see [ARCHITECTURE.md](deplodock/compiler/backend/cuda/ARCHITECTURE.md))
    - [tuning.py](deplodock/compiler/tuning.py) — autotuning utilities
  - [recipe/](deplodock/recipe/) — Recipe loading, dataclass types, engine flag mapping (see [ARCHITECTURE.md](deplodock/recipe/ARCHITECTURE.md))
  - [serving/](deplodock/serving/) — vLLM out-of-tree embedding plugin (see [ARCHITECTURE.md](deplodock/serving/ARCHITECTURE.md))
  - [deploy/](deplodock/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](deplodock/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](deplodock/benchmark/) — Benchmark tracking, config, task enumeration, execution
  - [planner/](deplodock/planner/) — Groups benchmark tasks into execution groups for VM allocation
- [recipes/](recipes/) — Model deploy recipes (YAML configs per model)
- [docker/](docker/) — Custom image builds ([vllm-deplodock](docker/vllm-deplodock/) — vLLM + the deplodock plugin)
- [experiments/](experiments/) — Experiment parameter sweeps (self-contained recipe + results)
- [kernels/](kernels/) — Standalone CUDA kernel sources
- [docs/](docs/) — Technical notes and engine-specific guides
  - [sglang-awq-moe.md](docs/sglang-awq-moe.md) — SGLang quantization for AWQ MoE models
- [tests/](tests/) — pytest tests (see [ARCHITECTURE.md](tests/ARCHITECTURE.md))
  - [compiler/passes/](tests/compiler/passes/) — compiler pass tests (see [ARCHITECTURE.md](tests/compiler/passes/ARCHITECTURE.md))
- [scripts/](scripts/) — Analysis and visualization scripts
- [utils/](utils/) — Standalone utility scripts
- [config.yaml](config.yaml) — Benchmark configuration
- [Makefile](Makefile) — Build automation
- [pyproject.toml](pyproject.toml) — Package metadata and tool config

## Contributing

1. Branch from `main` (e.g. `feature/my-change`).
2. Follow [STYLE.md](STYLE.md) and per-directory `ARCHITECTURE.md` files.
3. Add tests in `tests/` (see [tests/ARCHITECTURE.md](tests/ARCHITECTURE.md)).
4. `make test && make lint` (use `make format` to auto-fix).
5. Open a PR against `main`.

## License

Licensed under the [Apache License 2.0](LICENSE).
