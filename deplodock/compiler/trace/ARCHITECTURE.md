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

`scaled_dot_product_attention` captures an explicit `attn_mask` tensor (when
present) as a 4th `SdpaOp` input — HF passes its precomputed `(1, 1, S, S)`
causal mask this way (an additive `0` / `-inf` bias) rather than via the
`is_causal` flag. The decomposition (`frontend/decomposition/010_sdpa.py`)
broadcasts that mask to the scores shape and adds it before the softmax.
Dropping it silently turns masked attention into full bidirectional
attention — invisible to uniform input but wrong on any varying sequence.

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

The wrapper also **replaces the rotary embedding in both modes** — HF's in-graph rotary silently breaks under
`torch.export`: its `inv_freq` buffer is `persistent=False` and doesn't survive export with its real value, so the
traced cos/sin constant-fold to `cos=1, sin=0` and RoPE degenerates to identity. Static mode precomputes cos/sin at
the trace seq_len (`_PassThroughRotary`); dynamic mode precomputes out to `DYNAMIC_DIM_MAX + 1` positions and slices
to the runtime seq_len in-graph (`_SlicedRotary` — the slice end is a SymInt; the `+1` exists because export guards a
symbolic slice end strictly below the sliced extent). Beware that an accuracy check with `input_ids = zeros` cannot
catch a degenerate RoPE or wrong attention scores — identical value rows make the attention output independent of the
attention weights; `tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager`
checks with non-zero ids for exactly that reason. The model passed in is **not** restricted to `CausalLM` — wrapping
an `AutoModel` trunk yields hidden states instead of logits (the serving plugin's embedding path, `deplodock/serving`).

- `build_layer_wrapper(block, rotary_emb, hidden_size, dtype, layer_type=None) → nn.Module` is the per-layer dynamic
  analogue: `forward(x)` slices precomputed cos/sin buffers (out to `DYNAMIC_DIM_MAX + 1`, same `+1` guard) by
  `x.shape[1]` in-graph and calls `block(x, position_embeddings=(cos, sin))`. The static per-layer trace instead passes
  concrete `(cos, sin)` kwargs — those specialise rotary to the trace seq_len, which is exactly what dynamic mode must
  avoid. Forward arg is named `x`, so the CLI spec is `--dynamic seq_len@x:1`
  (`tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager`).

`SliceOp` nodes record `dim`/`start` as **op fields** at trace time (`torch.py`'s slice handler reads the raw FX
args): the legacy constant-input convention can't represent a `None` start (`x[:, :s]`) or a SymInt end —
`_resolve_inputs` drops both, leaving the surviving constants positionally ambiguous. Pre-field IR dumps still
decompose via the constant-input fallback.

### `quantized.py` — W4A16 (compressed-tensors AWQ-INT4) pre-trace substitution

A compressed-tensors checkpoint registers a forward *decompress pre-hook* that unpacks the int4 weights on first
forward; under `torch.export` that hook fires mid-trace and injects a data-dependent slice (`unpack_from_int32 →
int(shape[1])`) export can't specialize. `apply_quant_substitution(model)` walks the loaded model **before** export
(the established `huggingface.py` submodule-swap pattern) and replaces each targeted quantized linear with a
`DequantLinear` whose `forward` is a single opaque custom op `deplodock::dequant_linear` (registered via
`torch.library.custom_op` + `register_fake`). This keeps the packed int32 / fp16-scale / int32-zp tensors as graph
**constants** (passed as explicit tensor args, never closed over), prevents the decompress hook from firing, and emits
one `DequantLinearOp` node the `045_dequant_linear` decomposition expands. The tracer maps the custom op by name in
`_op_name` / `_handle_dequant_linear` (a dedicated arg branch peels the tensor inputs from the scalar `QuantScheme`,
which becomes op metadata — never graph `ConstantOp`s). `parse_quant_config` handles the transformers wrapper
(`CompressedTensorsConfig → .quantization_config.config_groups`) and skips `ignore` modules (`lm_head`). The module is
deliberately written **without** `from __future__ import annotations` so `torch.library` can infer the op schema from
real annotations. No-op for non-quantized models. Wired into `commands/compile.py::_trace_model` (so `run` / `tune`
reach it). All format-specific logic (config parse + the torch dequant the eager impl runs) lives here; everything
below `DequantLinearOp` is format-agnostic. See `plans/w4a16-quantization-support.md`.

## Entry points

- Whole-model trace: `trace_module(build_full_model_wrapper(model, …), (input_ids,))`.
- Single-layer trace: `trace_module(model.model.layers[N], (x,), kwargs={…})` (static); with `--dynamic`,
  `trace_module(build_layer_wrapper(block, …), (x,), dynamic_shapes={"x": {1: Dim("seq_len")}})`.
- Inline expression: `graph_from_code("torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))")` (used by `deplodock compile --code` and `deplodock trace --code`).

## Rule

Frontend capture is **upstream of decomposition** — `trace/` emits
`ir/frontend/` ops only, never primitives. Decomposition rules
(`pipeline/passes/frontend/decomposition/`) rewrite frontend ops into tensor-IR
primitives; `trace/` is unaware of that rewrite.
