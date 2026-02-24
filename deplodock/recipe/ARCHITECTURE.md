# Recipe Architecture

## Overview

The `recipe` package owns all recipe-related logic: YAML loading, variant resolution via deep merge, typed configuration dataclasses, engine flag mapping, and extra_args validation.

## Modules

- `types.py` — dataclasses: `Recipe`, `ModelConfig`, `EngineConfig`, `LLMConfig`, `VllmConfig`, `SglangConfig`, `BenchmarkConfig`
- `recipe.py` — `deep_merge()`, `load_recipe()`, `validate_extra_args()`
- `engines.py` — `VLLM_FLAG_MAP`, `SGLANG_FLAG_MAP`, `banned_extra_arg_flags()`, `build_engine_args()`

## Key Design Decisions

### Deep Merge for Variant Resolution

Variants override base recipe values via recursive deep merge (`deep_merge()`). Nested dicts are merged key-by-key; scalars in the variant override the base. This allows variants to selectively override any field at any depth without repeating the full config:

```yaml
# base
engine:
  llm:
    tensor_parallel_size: 8
    context_length: 16384
    vllm:
      extra_args: "--kv-cache-dtype fp8"

# variant overrides only what changes
variants:
  8xH100:
    engine:
      llm:
        max_concurrent_requests: 256
```

The merged result keeps `tensor_parallel_size: 8`, `context_length: 16384`, and the vllm block from the base, while adding `max_concurrent_requests: 256` from the variant.

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
4. **Deep merge support** — named fields participate in variant merging naturally. An `extra_args` string cannot be partially overridden.

### Extra Args Ban Enforcement

Users must not duplicate named fields in `extra_args`. The `validate_extra_args()` function enforces this by:

1. Building a banned set from the active engine's flag map (`VLLM_FLAG_MAP` or `SGLANG_FLAG_MAP`) plus hardcoded flags (`--trust-remote-code`, `--host`, `--port`, `--model`, `--served-model-name`).
2. Tokenizing the `extra_args` string and checking each token (handling both `--flag value` and `--flag=value` forms).
3. Raising `ValueError` listing all offending flags if any are found.

This validation runs inside `load_recipe()` before returning the `Recipe`, so invalid configs fail fast at load time rather than at Docker runtime.

`extra_args` is the escape hatch for engine-specific flags that don't have a named field (e.g. `--kv-cache-dtype fp8`, `--enable-expert-parallel`). It is passed through verbatim to `build_engine_args()`.

### Engine-Specific Nesting

Engine-specific config (`image`, `extra_args`) nests under `engine.llm.vllm` or `engine.llm.sglang`, while engine-agnostic config lives at `engine.llm`. `LLMConfig.engine_name` is determined by which sub-config is present (SGLang takes priority if both exist). The `image` and `extra_args` properties delegate to the active engine's config.

## Data Flow

```
recipe.yaml
    |
    v
load_recipe(recipe_dir, variant)
    |-- yaml.safe_load()
    |-- pop variants, deep_merge if variant specified
    |-- validate_extra_args()
    |-- Recipe.from_dict()
    v
Recipe dataclass
    |
    v
build_engine_args(recipe.engine.llm, model_name) -> ["--flag value", ...]
    |
    v
generate_compose() -> docker-compose.yaml string
```
