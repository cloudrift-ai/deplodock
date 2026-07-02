<p align="center">
  <a href="https://pypi.org/project/emmy-llm/"><img src="https://img.shields.io/pypi/v/emmy-llm" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

**Compile → Benchmark → Deploy** any LLM on any GPU. vLLM, SGLang, or create your own specialized deployment using a hackable compiler.

## Install

```bash
git clone https://github.com/cloudrift-ai/emmy.git
cd emmy && make setup
```

## Compile

A hackable PyTorch → Graph IR → CUDA compiler. Trace any `nn.Module`, fuse it into one kernel, run it, and inspect the emitted CUDA. See the blog post: [*A Principled ML Compiler Stack in 5,000 Lines of Python*](https://www.cloudrift.ai/blog/building-gpu-compiler-from-scratch-1).

```bash
# Compile a single layer
emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))"
# Benchmark, profile and optimize kernels locally
emmy run --bench --profile -c "torch.nn.Softmax(dim=-1)(torch.randn(1, 28, 2048, 2048))"
# Compile full model from HuggingFace (will download weights)
emmy compile Qwen/Qwen3-Embedding-0.6B
```

Layer-norm-style reduction (two reductions, broadcast subtract, elementwise chain) fused into two kernels:

```bash
emmy compile -c "
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


**Readable Schedule**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir tile`
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

**Optimized CUDA kernel**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir cuda`

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
emmy bench recipes/*                                    # All recipes
emmy bench experiments/.../optimal_mcr_rtx5090          # An experiment
emmy bench recipes/* --filter "deploy.gpu=*5090*"       # Subset
emmy bench recipes/* --gpu-concurrency 4                # Parallel VMs per GPU
emmy bench recipes/* --local                            # On this machine
emmy bench recipes/* --ssh user@host1 --ssh user@host2  # Pre-allocated hosts
```

External contributors: open a PR with an experiment under `experiments/{model}/{name}/`, then a maintainer triggers a cloud run by commenting `/run-experiment` on the PR.

## Deploy

```bash
# Remote server via SSH
emmy deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host

# Local Docker Compose
emmy deploy local --recipe recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ

# Cloud (auto-provisions a VM)
emmy deploy cloud --recipe recipes/GLM-4.6-FP8 --gpu "NVIDIA H200 141GB" --gpu-count 8

# Teardown / preview
emmy deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host --teardown
emmy deploy ssh --recipe recipes/GLM-4.6-FP8 --ssh user@host --dry-run
```

## Serve (compiled embeddings via vLLM)

```bash
# vLLM's OpenAI shell (/v1/embeddings, tokenizer, scheduler, pooler) over emmy-compiled kernels
pip install -e ".[compile,serving]"
emmy serve Qwen/Qwen3-Embedding-0.6B                  # extra flags pass through to vllm serve

curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"Hello"}'

# One-shot benchmark (vllm bench serve against the started server), and the raw-vLLM baseline
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32 --stock
```

See [`emmy/serving/ARCHITECTURE.md`](emmy/serving/ARCHITECTURE.md); embedding recipes
(`recipes/Qwen3-Embedding-*`) A/B this against stock vLLM via `emmy bench`.

## Generate (chat) — experimental

```bash
# Standalone generation oracle (no vLLM) — re-runs the whole prefix each step, O(S²); the
# token-for-token reference (matches HF eager greedy, e.g. on TinyLlama-1.1B-Chat).
emmy generate TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --max-new-tokens 10

# Serve a chat model through emmy-compiled per-layer kernels (vLLM owns the OpenAI API /
# sampler / scheduler / paged KV-cache / lm_head; emmy owns embed + the trunk).
emmy serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --generate
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Hi"}]}'
```

**Status:** correctness complete for decoder-only **Llama / Qwen3** (full-causal, fp16, TP=1). Perf is **not yet
hardened** — host-sync interleave at the per-layer seam, and `serve` compiles 2× n_layers programs (startup- and
memory-heavy → small models for now). See [`emmy/serving/ARCHITECTURE.md`](emmy/serving/ARCHITECTURE.md) for
the design.

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
emmy vm create gcp --instance my-vm --zone us-central1-a --machine-type a2-highgpu-1g
emmy vm delete gcp --instance my-vm --zone us-central1-a

# CloudRift
emmy vm create cloudrift --instance-type rtx4090.1 --ssh-key ~/.ssh/id_ed25519.pub
emmy vm delete cloudrift --instance-id <id>
```

## Development

```bash
make test      # run pytest
make lint      # ruff check + format check
make format    # auto-fix
```

## Project Structure

- [emmy/](emmy/) — Python package
  - [emmy.py](emmy/emmy.py) — CLI entrypoint
  - [logging_setup.py](emmy/logging_setup.py) — CLI logging configuration
  - [hardware.py](emmy/hardware.py) — GPU specs and instance type mapping
  - [detect.py](emmy/detect.py) — GPU detection via PCI sysfs (local and remote)
  - [redact.py](emmy/redact.py) — Secret redaction for logs and dumps
  - [commands/](emmy/commands/) — CLI layer (thin argparse handlers, see [ARCHITECTURE.md](emmy/commands/ARCHITECTURE.md))
    - [deploy/](emmy/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](emmy/commands/bench/) — `bench` command
    - [vm/](emmy/commands/vm/) — `vm create/delete` commands (GCP, CloudRift)
    - [teardown.py](emmy/commands/teardown.py) — `teardown` command
    - [pull.py](emmy/commands/pull.py) — `pull` command (download HF model)
    - [trace.py](emmy/commands/trace.py) — `trace` command (PyTorch → Graph IR)
    - [compile.py](emmy/commands/compile.py) — `compile` command (decomposition → optimization → fusion → kernel/CUDA lowering)
    - [run.py](emmy/commands/run.py) — `run` command (compile + execute on CUDA backend, optional benchmarks)
    - [inspect_graph.py](emmy/commands/inspect_graph.py) — `inspect` command (graph summary)
  - [compiler/](emmy/compiler/) — PyTorch → Graph IR → CUDA compiler (see [ARCHITECTURE.md](emmy/compiler/ARCHITECTURE.md))
    - [graph.py](emmy/compiler/graph.py) — `Graph`, `Node`, `Tensor`, `Hints` container
    - [ir/](emmy/compiler/ir/) — per-dialect op definitions (torch / tensor / loop / kernel / cuda) (see [ARCHITECTURE.md](emmy/compiler/ir/ARCHITECTURE.md))
    - [trace/](emmy/compiler/trace/) — PyTorch/HuggingFace → Graph IR capture (see [ARCHITECTURE.md](emmy/compiler/trace/ARCHITECTURE.md))
    - [pipeline/](emmy/compiler/pipeline/) — rewrite engine + passes + dump hooks (see [ARCHITECTURE.md](emmy/compiler/pipeline/ARCHITECTURE.md))
    - [rules/](emmy/compiler/rules/) — rewrite rules (decomposition, optimization, fusion, lowering)
    - [program/](emmy/compiler/program/) — kernel program assembly (LoopOp → KernelOp → CudaOp)
    - [cuda/](emmy/compiler/cuda/) — CUDA source rendering and runtime helpers
    - [backend/](emmy/compiler/backend/) — numpy / loop / CUDA execution (see [ARCHITECTURE.md](emmy/compiler/backend/ARCHITECTURE.md))
      - [cuda/](emmy/compiler/backend/cuda/) — CUDA backend internals (see [ARCHITECTURE.md](emmy/compiler/backend/cuda/ARCHITECTURE.md))
    - [tuning.py](emmy/compiler/tuning.py) — autotuning utilities
  - [recipe/](emmy/recipe/) — Recipe loading, dataclass types, engine flag mapping (see [ARCHITECTURE.md](emmy/recipe/ARCHITECTURE.md))
  - [serving/](emmy/serving/) — vLLM out-of-tree embedding plugin (see [ARCHITECTURE.md](emmy/serving/ARCHITECTURE.md))
  - [deploy/](emmy/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](emmy/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](emmy/benchmark/) — Benchmark tracking, config, task enumeration, execution
  - [planner/](emmy/planner/) — Groups benchmark tasks into execution groups for VM allocation
- [recipes/](recipes/) — Model deploy recipes (YAML configs per model)
- [docker/](docker/) — Custom image builds ([vllm-emmy](docker/vllm-emmy/) — vLLM + the emmy plugin)
- [experiments/](experiments/) — Experiment parameter sweeps (self-contained recipe + results)
- [kernels/](kernels/) — Standalone CUDA kernel sources
- [docs/](docs/) — Docusaurus user-docs site (getting started, benchmarking, custom configurations, deployment)
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
