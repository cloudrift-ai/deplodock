---
sidebar_position: 1
title: Custom Configurations
description: Create custom deployment configurations for your LLM models.
---

# Custom Configurations

If the built-in recipes don't cover your model or GPU setup, you can define your own configuration.

A custom configuration lets you control:

- **Inference backend** — vLLM or SGLang
- **Quantization** — FP16, BF16, INT8, AWQ, GPTQ
- **Tensor parallelism** — across multiple GPUs
- **Batch size and max sequence length**
- **Model source** — HuggingFace Hub or a private registry

## Example

```yaml
model: mistralai/Mixtral-8x7B-Instruct-v0.1
backend: vllm
quantization: awq
tensor_parallel_size: 2
max_model_len: 32768
```
