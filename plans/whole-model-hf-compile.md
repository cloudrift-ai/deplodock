# Whole-Model HF Compile with `torch.export` Dynamic Traces

Scope: turn the current "trace works, compile fails" state into "trace + compile + run end-to-end" on TinyLlama
/ Qwen single-layer prefill with `--dynamic seq_len@…`. The dynamic-shapes infrastructure
(`plans/dynamic-shapes.md` M0–M6) is done; what's missing is op coverage + a small number of trace-side
adjustments. None of this work touches the dynamic-shape machinery itself — it slots cleanly on top.

## Motivation

`deplodock compile <hf_model> --dynamic seq_len@input_ids:1 …` runs `torch.export` to completion and the
FX→IR walk yields a clean graph: 134 nodes for Qwen2.5-7B (1 layer), every internal tensor carrying
`Dim('seq_len')` where it should, only three `i64` nodes (the index tensors). The op set is unsurprising:

```
GatherOp:1, LinearOp:8, MatmulOp implicit via Linear, ElementwiseOp:27, MeanOp:3, IndexMapOp:16,
ReshapeOp:8, TransposeOp:4, UnsqueezeOp:4, SliceOp:4, CatOp:2, SdpaOp:1, ConstantOp:53, InputOp:3
```

Every op type already has a lifting / lowering rule. The compiler dies further downstream on three concrete
gaps that this plan closes.

## Validation slice (final state)

```bash
deplodock run TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dynamic seq_len@input_ids:1 \
    --dynamic seq_len@attention_mask:2 \
    --dynamic seq_len@attention_mask:3 \
    --dynamic seq_len@position_ids:1
```

Compiles once, runs prefill for two distinct token counts (e.g. 8 and 32) against a single cached
kernel set, asserts logits within fp32 tolerance of `transformers` eager forward.

## Current gaps (audit, in pipeline order)

1. **`i32` / `i64` CUDA name rendering.** `compiler/dtype.py` registers `i32`/`i64` (so the trace doesn't
   crash on dtype lookup) but `backend/cuda/dtype.py:_CUDA_NAME` doesn't map them to `int` / `int64_t`
   → kernel-signature renderer raises `KeyError`. Single-line fix; covers `input_ids` /
   `position_ids` showing up as kernel params.

2. **Embedding lookup (`GatherOp` over `(vocab_size, hidden)` with `i64` indices).** Lifted by
   `040_lift_gather.py`; we have a CUDA path for `GatherOp` (`test_dynamic_shapes.py` has no coverage
   though). Two concrete worries:
   - Index tensor is `i64`; the existing tile/kernel passes assume f32-ish bandwidth math
     (`BYTES_PER_ELEM = 4` in `tile/ir.py`). For an i64 index of shape `(1, S)` it under-counts by 2x
     but doesn't change behavior.
   - The gather output is `(B, S, hidden=3584)`, hidden being well past warp-size — needs to thread
     through partition_loops cleanly with symbolic S on the M axis.

3. **`MeanOp` over a static dim with a symbolic free axis.** The trace produces three `MeanOp` nodes
   for RMSNorm's `mean(x*x, dim=-1)`. Decomposition rule `090_mean.py` rewrites this to `sum / N`
   where `N = x.shape[-1]` (static here). Should work post-M5 (symbolic reduce supported); verify.

4. **`SliceOp` + `CatOp` from RoPE.** RoPE does `cat([-x[..., d/2:], x[..., :d/2]], -1)` and similar
   half-rotations. Each adds a `SliceOp` + `CatOp`. After M0, both ops carry `tuple[int | str, ...]`
   shapes and accept `Dim('seq_len')` in the head_dim-adjacent positions. Verify the slice/cat
   decomposition produces a sensible IndexMapOp for the symbolic-S case.

5. **`UnsqueezeOp` on RoPE cos/sin.** Trace produces `(1, seq_len, 1, head_dim)` from
   `(1, seq_len, head_dim)`. M0's `UnsqueezeOp` already handles symbolic.

6. **`lm_head` matmul over `vocab_size`.** Final op:
   `(1, S, hidden) @ (hidden, vocab=152064)` → `(1, S, vocab)`. Matmul-on-symbolic-M works (M5).
   The kernel is fine but the launch produces `vocab/BN * S` CTAs — for vocab=150k that's a lot;
   may want a thread-block-per-vocab-tile partition. Perf concern, not correctness.

## Milestones (single branch, milestone commits per `[[feedback_single_branch_milestones]]`)

| M  | Slice | Validation |
|----|-------|------------|
| M0 | Add `i32`/`i64` → `int` / `int64_t` to `backend/cuda/dtype.py:_CUDA_NAME`; smoke a kernel whose signature carries an `int64_t*` param | `compile --code "torch.nn.Embedding(256,128)(torch.zeros(1,8,dtype=torch.long))"` renders without `KeyError` |
| M1 | End-to-end run of standalone embedding lookup with dynamic seq_len (`(B, S) i64 → (B, S, H) f32`) | New test `test_cuda_embedding_dynamic_seq_len`; accuracy match vs `torch.nn.Embedding` |
| M2 | RoPE in isolation: `cos/sin × q/k` rotation with symbolic seq_len. Standalone module wrapping the half-rotation + multiply | New test; accuracy vs `torch.nn.functional` cos/sin rotation |
| M3 | Single-layer TinyLlama block trace with the wrapper switched to dynamic mode AND cos/sin passed in as args (extend `_trace_model`'s `--layer N` path) | `compile <model> --layer 0 --dynamic seq_len@…` produces a kernel list (every node → CudaOp) |
| M4 | `run` of M3 — execute single-layer prefill at two seq_lens; compare to torch eager | `test_cuda_tinyllama_layer_dynamic`: rtol≤1e-4 vs eager |
| M5 | Whole-model TinyLlama prefill (1 layer, random weights) via the existing whole-model wrapper | `test_cuda_tinyllama_whole_model_dynamic_prefill`: rtol≤1e-4 vs eager, two seq_lens, single compiled artifact |
| M6 | Whole-model Qwen2.5-7B prefill (1 layer, random weights). Stress for the bigger hidden (3584) / vocab (152k) / num_heads (28 / 4) shapes | `test_cuda_qwen_whole_model_dynamic_prefill`: rtol≤1e-3 vs eager |

M0–M2 are independent unit pieces and can land in any order. M3 is the load-bearing integration
milestone — once a single block compiles + runs dynamically, M4/M5/M6 are mostly scale-up.

## Surfaces that need work

1. **`backend/cuda/dtype.py`** — `_CUDA_NAME` needs `I32 → "int"`, `I64 → "int64_t"`. Likely also wants
   `cupy_dtype` extensions for int types so allocation works for input buffers.

2. **`commands/compile.py:_trace_model`** — `--layer N` path eagerly computes cos/sin from the concrete
   seq_len. For M3, switch to a wrapper class whose forward takes `(x, cos, sin)` (or
   `(x, position_embeddings)` tuple matching HF block's signature) so the user marks all three
   `--dynamic seq_len@…`. Choose between:
   - (a) A small wrapper class similar to `build_full_model_wrapper(dynamic=True)`.
   - (b) Have the wrapper compute cos/sin inside via `rotary_emb(x, position_ids)` and rely on
     torch.export to capture the rotation — same trick the whole-model wrapper uses for the mask.

   (a) is cleaner; (b) ties us to HF's `rotary_emb` API stability.

3. **`backend/cuda/_tma.py` / TMA descriptor** — already bails on any symbolic shape (M2+M3).
   Should be fine.

4. **Tile planner cooperative-K + i64 index path** — for the embedding lookup, the GatherOp body has
   a `Load` against the index tensor (i64). The CUDA renderer needs to emit
   `int64_t in_idx = ...` declarations and have `Var.render` not assume f32. Likely already works
   since `buffer_dtypes` carries per-buffer dtype tokens (`render_kernelop` populates it); verify
   for the int case.

5. **Test fixture** — the existing `_compile_and_run_block` in `tests/compiler/test_block_accuracy.py`
   already uses `from_config` (random weights, no checkpoint download). Reuse that pattern for the
   new dynamic-shape tests so we don't pull large weights.

## Open decisions

- **`i64` vs `i32` for `position_ids`.** torch uses i64 by default but i32 fits all realistic seq_lens
  and would let us share code with the static-shape path. v1: support both as buffer dtypes; pick
  i64 when the trace emits i64, no implicit downcast.

- **Cos/sin: input or computed?** Per M3 decision above. Default: input (option (a)).

- **`lm_head` perf.** Vocab=152k matmul will be a big kernel. Out of scope to optimize; v1 just runs it.

- **Decode vs prefill.** Decode adds KV cache management (running-shape past_kv concat). Strictly
  prefill in v1; decode is a separate plan ([[plan_kv_cache_dynamic]] when written).

## Explicitly out of scope (v1)

- Autoregressive generation loop (multi-token decode with KV cache).
- Sampling / beam search / top-k.
- Quantized weights (int8 / int4 / fp4 / nvfp4).
- Multi-GPU / tensor-parallel / pipeline-parallel.
- LoRA adapters.
- Models with attention shapes deplodock doesn't yet handle (Mistral sliding-window, MoE routing).
- Performance-tuning the resulting kernels (autotune sweep over the dynamic variants). v1 measures
  correctness; perf comes after the validation slice is green.

## Relationship to existing plans

- [[plans/dynamic-shapes]] — done. Provides the `Dim` type, the position-based `--dynamic` CLI, the
  symbolic CUDA codegen / launch path. This plan consumes that and adds op + dtype coverage.
- [[project_full_model_compile]] — memory note from the pre-dynamic-shapes era. Captures the
  static-shape SDPA fusion bug that's since been worked around; will need updating once M5 lands
  to reflect that whole-model dynamic compile is the production flow.
