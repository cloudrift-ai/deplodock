# Recipe Architecture

## Overview

The `recipe` package owns all recipe-related logic: YAML loading, matrix expansion for benchmark parameter sweeps, typed configuration dataclasses, engine flag mapping, and extra_args validation.

## Modules

- `types.py` — dataclasses: `Recipe`, `DeployConfig`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`, `SglangConfig`, `BenchmarkConfig`
- `recipe.py` — `deep_merge()`, `load_recipe()`, `validate_extra_args()`, `_load_raw_config()`, `_validate_and_build()`
- `matrix.py` — `expand_matrix_entry()`, `dot_to_nested()`, `matrix_label()`, `build_override()`, `PARAM_ABBREVIATIONS`
- `engines.py` — `VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`, `banned_extra_arg_flags()`, `build_engine_args()`

## Key Design Decisions

### Matrix Expansion for Benchmark Sweeps

Recipes use `matrices` — a list of entries that define benchmark configurations with broadcast + zip semantics:

```yaml
matrices:
  # Simple single-point entry (all scalars)
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    deploy.gpu_count: 1

  # Concurrency sweep (8 runs from one entry)
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    benchmark.max_concurrency: [1, 2, 4, 8, 16, 32, 64, 128]

  # Correlated sweep (3 zip runs)
  - deploy.gpu: "NVIDIA GeForce RTX 5090"
    engine.llm.max_concurrent_requests: [128, 256, 512]
    benchmark.max_concurrency: [128, 256, 512]
```

Rules:
- **Scalars** are broadcast to all runs in the entry
- **Lists** are zipped together (all lists in one entry must have the same length)
- **`deploy.gpu`** is required in each matrix entry
- **Dot-notation** is used for all parameter paths (`deploy.gpu`, `engine.llm.max_concurrent_requests`, etc.)

Each combination is converted to a nested override dict via `build_override()` and deep-merged with the base config.

### Deep Merge

Override dicts are applied to base configs via recursive deep merge (`deep_merge()`). Nested dicts are merged key-by-key; scalars in the override replace the base. This allows matrix entries to selectively override any field at any depth:

```yaml
# base
engine:
  llm:
    tensor_parallel_size: 8
    context_length: 16384
    vllm:
      extra_args: "--kv-cache-dtype fp8"

# matrix entry overrides only what changes
matrices:
  - deploy.gpu: "NVIDIA H100 80GB"
    engine.llm.max_concurrent_requests: 256
```

The merged result keeps `tensor_parallel_size: 8`, `context_length: 16384`, and the vllm block from the base, while adding `max_concurrent_requests: 256` from the matrix entry.

### Auto-Generated Run Identifiers

Each matrix combination gets an auto-generated identifier: `{gpu_short_name}` + `_{matrix_label}` (if any list params).

Matrix labels use abbreviations: `max_concurrency` → `c`, `num_prompts` → `n`, `random_input_len` → `in`, `random_output_len` → `out`, `max_concurrent_requests` → `mcr`, `context_length` → `ctx`. Unknown keys use the last path segment.

Examples: `rtx5090_c1_vllm_benchmark.txt`, `rtx5090_c128_vllm_benchmark.txt`, `rtx5090_vllm_benchmark.txt` (single-point entry).

### DeployConfig

GPU provisioning info is encapsulated in `DeployConfig` (nested under `Recipe.deploy`):

```python
@dataclass
class DeployConfig:
    gpu: str | None = None
    gpu_count: int = 1
```

Matrix entries use `deploy.gpu` and `deploy.gpu_count` via dot-notation override. The `deploy` section is optional in the base recipe — it's only needed when `deploy cloud` requires GPU info directly (without matrix expansion).

### First-Class Named Parameters

Engine-agnostic serving parameters are promoted to first-class named fields on `LLMConfig` rather than being buried in `extra_args` strings:

| Field | vLLM flag | SGLang flag |
|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | `--tp` |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | `--dp` |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` |
| `context_length` | `--max-model-len` | `--context-length` |
| `max_concurrent_requests` | `--max-num-seqs` | `--max-running-requests` |

This design provides:

1. **Type safety** — numeric values are validated at parse time, not when Docker fails.
2. **Engine portability** — the same recipe field maps to different CLI flags per engine via `VLLM_FLAG_MAP` / `SGLANG_FLAG_MAP` in `engines.py`.
3. **Computed properties** — `LLMConfig.gpus_per_instance` derives from `tensor_parallel_size * pipeline_parallel_size` without parsing strings.
4. **Deep merge support** — named fields participate in matrix merging naturally. An `extra_args` string cannot be partially overridden.

### Extra Args Ban Enforcement

Users must not duplicate named fields in `extra_args`. The `validate_extra_args()` function enforces this by:

1. Building a banned set from the active engine's flag map (`VLLM_FLAG_MAP` or `SGLANG_FLAG_MAP`) plus hardcoded flags (`--trust-remote-code`, `--host`, `--port`, `--model`, `--model-path`, `--served-model-name`).
2. Tokenizing the `extra_args` string and checking each token (handling both `--flag value` and `--flag=value` forms).
3. Raising `ValueError` listing all offending flags if any are found.

This validation runs inside `_validate_and_build()` before returning the `Recipe`, so invalid configs fail fast at load time rather than at Docker runtime.

`extra_args` is the escape hatch for engine-specific flags that don't have a named field (e.g. `--kv-cache-dtype fp8`, `--enable-expert-parallel`). It is passed through verbatim to `build_engine_args()`.

### Engine-Specific Model Flag

`build_engine_args()` emits the model path using the flag expected by each engine:
- vLLM: `--model {name}`
- SGLang: `--model-path {name}`

Both `--model` and `--model-path` are in the hardcoded banned set, so they cannot appear in `extra_args` regardless of which engine is active.

### Engine-Specific Nesting

Engine-specific config (`image`, `extra_args`) nests under `engine.llm.vllm` or `engine.llm.sglang`, while engine-agnostic config lives at `engine.llm`. `LLMConfig.engine_name` is determined by which sub-config is present (SGLang takes priority if both exist). The `image` and `extra_args` properties delegate to the active engine's config.

### SGLang Quantization for AWQ MoE Models

SGLang does not automatically detect AWQ quantization for MoE architectures. For AWQ-quantized MoE models, `--quantization moe_wna16` must be passed via `extra_args`. See [/docs/sglang-awq-moe.md](/docs/sglang-awq-moe.md) for full details and tested configurations.

## Data Flow

```
recipe.yaml
    |
    v
_load_raw_config(recipe_dir) -> raw dict
    |
    +-- load_recipe(): strips matrices, calls _validate_and_build()
    |       -> base Recipe (for deploy commands)
    |
    +-- enumerate_tasks(): reads matrices, expands each entry:
            |-- expand_matrix_entry() -> list of combinations
            |-- build_override() -> nested override dict
            |-- deep_merge(base, override) -> merged config
            |-- _validate_and_build() -> Recipe per combination
            v
        list[BenchmarkTask] (for bench command)

Recipe dataclass
    |
    v
build_engine_args(recipe.engine.llm, model_name) -> ["--flag value", ...]
    |
    v
generate_compose() -> docker-compose.yaml string
```
