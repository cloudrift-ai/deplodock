# Trace Architecture

Frontend capture: PyTorch/HuggingFace model → `Graph` populated with
Layer-1 frontend ops.

## Modules

### `torch.py` — FX → Graph IR

`trace_module(module, args, kwargs=None) → Graph` runs
`torch.export.export()` to get an FX graph, then walks each node and
emits the matching frontend op (`LinearOp`, `MatmulOp`, `SdpaOp`,
`ElementwiseOp`, `ReduceOp`, …). `trace_module_with_constants` returns
the graph plus a `placeholder_name → attr_path` dict for resolving
weights/buffers back to their module attributes.

Per-op handlers map aten names (`aten.add.Tensor`,
`aten.linear.default`, …) to dialect op constructors; input shapes are
pulled from the FX meta and fed into the op's `infer_output_shape` to
stamp the output tensor.

### `huggingface.py` — trace-friendly wrapper

HuggingFace `CausalLM` models build their causal attention mask
dynamically at forward time (`arange` → `cumsum` → `triu` → `eq` …),
which pollutes the traced FX graph with dozens of mask-construction
ops. One helper cleans this up:

- `build_full_model_wrapper(model, seq_len, dtype) → nn.Module` wraps
  the HF model in a module exposing `forward(input_ids) → logits`.
  Precomputes a `(1, 1, seq_len, seq_len)` causal mask and
  `position_ids` as buffers and monkey-patches HF's internal mask
  builders (`_update_causal_mask` / `_prepare_4d_causal_attention_mask`)
  to return the precomputed mask verbatim.

## Entry points

- Whole-model trace: `trace_module(build_full_model_wrapper(model, …), (input_ids,))`.
- Single-layer trace: `trace_module(model.model.layers[N], (x,), kwargs={…})`.
- Inline expression: `graph_from_code("torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))")` (used by `deplodock compile --code` and `deplodock trace --code`).

## Rule

Frontend capture is **upstream of decomposition** — `trace/` emits
`ir/frontend/` ops only, never primitives. Decomposition rules
(`pipeline/passes/frontend/decomposition/`) rewrite frontend ops into tensor-IR
primitives; `trace/` is unaware of that rewrite.
