# SGLang + AWQ MoE Models

## Problem

SGLang does not automatically detect AWQ quantization for Mixture of Experts (MoE) models. When loading an AWQ-quantized MoE model (e.g. `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`), SGLang loads all weights at full precision (bf16/fp16), consuming ~30 GB instead of the expected ~16 GB. This causes OOM on GPUs with 32 GB or less.

vLLM handles this correctly without any extra flags.

## Symptoms

Container exits shortly after startup with logs like:

```
Load weight end. dtype=torch.bfloat16, avail mem=0.41 GB, mem usage=30.28 GB.
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

The key indicator is `mem usage=30.28 GB` — the model weights are not quantized despite being an AWQ checkpoint.

## Quantization Flags That Do NOT Work for MoE

| Flag                                        | Result                                                                    |
|---------------------------------------------|---------------------------------------------------------------------------|
| `--quantization awq`                        | `ValueError: torch.bfloat16 is not supported for quantization method awq` |
| `--quantization awq_marlin`                 | Loads at full bf16 precision, OOM                                         |
| `--quantization awq_marlin --dtype float16` | Loads at full fp16 precision, OOM                                         |

These methods handle dense (non-MoE) model quantization but do not apply to expert weights in MoE architectures.

## Solution

Use `--quantization moe_wna16`:

```yaml
engine:
  llm:
    sglang:
      image: "lmsysorg/sglang:latest"
      extra_args: "--quantization moe_wna16"
```

This correctly quantizes the MoE expert weights:

```
Load weight end. dtype=torch.bfloat16, avail mem=14.90 GB, mem usage=15.79 GB.
```

With `moe_wna16`, the model uses ~15.8 GB (properly quantized) vs ~30.3 GB (unquantized), leaving enough memory for KV cache and CUDA graphs.

## When to Use `moe_wna16`

Use `--quantization moe_wna16` in `extra_args` when **all** of the following apply:

1. Engine is **SGLang** (vLLM handles AWQ MoE automatically)
2. Model uses **AWQ quantization** (check the model name or `config.json`)
3. Model has a **Mixture of Experts** architecture (e.g. Qwen3MoE, Mixtral)

For non-MoE AWQ models on SGLang, `--quantization awq` or `--quantization awq_marlin` should work fine.

## SGLang Server Arguments Reference

Full list of SGLang quantization methods and server arguments:
https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md

Notable MoE-related quantization options:
- `moe_wna16` — WnA16 quantization for MoE expert weights (works with AWQ checkpoints)
- `quark_int4fp8_moe` — INT4/FP8 mixed quantization for MoE

## Tested Configuration

- Model: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` (Qwen3MoE, 30B params, 3B active)
- GPU: NVIDIA GeForce RTX 5090 (32 GB)
- Image: `lmsysorg/sglang:latest`
- Result: Server starts, serves inference, ~15.8 GB model memory usage
