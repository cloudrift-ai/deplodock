# Generative (chat) inference for deplodock

> Status: **design / scoping doc. Not yet implemented.** Companion to `plans/w4a16-quantization-support.md`. Goal: serve
> a decoder-only chat model (Qwen3 / Llama-3 / the AWQ chat model) through deplodock-compiled kernels, with **vLLM
> owning the API / sampler / scheduler / KV-cache / chat template** first; a deplodock-standalone server later.
>
> Detail is front-loaded on **Phases 0–4 (to a first working vLLM chat)**; standalone serving + perf are a future-work
> coda.

## Context

`deplodock serve` today serves **embeddings only** (`deplodock/serving/`): a causal trunk → hidden states, vLLM's
pooler does last-token pooling. There is **no generation / decode / KV-cache / sampling code anywhere**. The compiler
decomposes attention **fully and internally** (`passes/frontend/decomposition/010_sdpa.py`: QK^T → scale → mask →
softmax → PV) over a full-sequence `[batch, heads, seq, dim]` layout, causal mask `(1,1,S,S)`, with RoPE cos/sin **baked
as graph constants** at trace time (`trace/huggingface.py` `_SlicedRotary` / `_PassThroughRotary`). This is correct for
*whole-sequence prefill* but structurally incompatible with how vLLM drives generation.

**The hard constraint (verified against vLLM 0.22.1 source).** vLLM's V1 generative runner
(`vllm/v1/worker/gpu_model_runner.py`) feeds the model **only the newly-scheduled tokens** per step, as a flattened
`[num_tokens]` continuous-batched layout, passing absolute `positions`; it **never re-feeds prior context**. KV state +
cross-token attention live in vLLM's **paged-attention backend** — a generative model only supplies q/k/v to a vLLM
`Attention` layer (`self.attn(q,k,v)`), and vLLM owns the cache, the attention math, sampling, chat template, streaming,
and the OpenAI API. Consequences:

- **"Recompute the whole sequence each step" is impossible inside vLLM** — the runner never gives back prior tokens.
- **Mamba-style self-managed state is invalid** — `IsAttentionFree` / `MambaSpec` assume a *constant-size* state; a KV
  cache grows with sequence length, so deplodock cannot present its cache as Mamba "state".
- Therefore, to ride vLLM's generation runner at all, deplodock **must** support **incremental decode** and let vLLM's
  `Attention` layer own the KV cache. "vLLM-first" does **not** defer the hard work — it requires it.

**The enabling corollary.** In a decoder layer, *everything except the SDPA core is per-token* — input norm, qkv
projection (+ q/k norm), RoPE, o-projection, MLP, post-norms are all pointwise or matmuls over the hidden axis. So once
SDPA is carved out, deplodock's compiled per-layer compute is **layout-invariant** to the flattened `[num_tokens, H]`
layout, and the only cross-token coupling (attention) becomes vLLM's paged backend. This is the cut Option A makes.

## Decision: Option A — carve SDPA out, vLLM does paged attention

Compile the per-layer **everything-but-attention** compute over the flattened per-token layout, emitting `q/k/v` and
consuming vLLM's attention output; vLLM's `Attention` layer does paged-KV attention. Validated directly by vLLM's own
`Qwen3Attention.forward` — every line except `self.attn(q,k,v)` is per-token. Two refinements locked in:

- **Carve via submodule substitution**, not post-trace graph surgery — the established pre-export pattern
  (`_PassThroughRotary`, and the W4A16 `DequantLinear` swap). A thin substitute `self_attn` runs `qkv_proj` (+ q/k norm)
  and **returns q,k,v directly** (no SDPA, no o_proj); a second subgraph runs o_proj + MLP + post-norm from an attn-out
  input. SDPA never enters the trace by construction.
- **vLLM applies RoPE in the first cut (A2).** deplodock emits **un-rotated** q/k; `DeplodockGenModel.forward` calls
  vLLM's `self.rotary_emb(positions, q, k)` between the pre-attention kernel and `self.attn`. This deletes the
  runtime-arbitrary-positions trace work from the critical path. **deplodock owning RoPE with runtime positions (A1) is
  a later optimization**, not a prerequisite.

Rejected alternatives are *forced, not chosen*: Mamba-state (KV grows ≠ constant state), full-recompute-in-vLLM (runner
never re-feeds), one monolithic deplodock kernel that calls vLLM attention mid-graph (deplodock compiles static
dataflow — it cannot yield to a Python `self.attn` call mid-kernel; the interleave must happen at the Python `forward`
level, per layer).

## Phases (correctness-first; detail front-loaded on 0–4)

### Phase 0 — Standalone naive `deplodock generate` (the correctness ORACLE)

First, even though the end goal is vLLM: it needs **zero new compiler work**, produces real chat output now, and is the
token-for-token correctness reference every later phase verifies against (and is reused by the eventual standalone
server). It uses the full-recompute strategy (re-run the whole growing prefix each step) — valid *standalone* (deplodock
controls the loop), intentionally O(S²), an oracle not a product. Key simplification: the recompute loop always runs at
positions `0..S-1`, so the existing `_SlicedRotary` is exactly right — **no RoPE-runtime-positions work needed here**
(that is purely a Phase-6 / incremental concern).

- **Reuses (no compiler change):** `commands/compile.py::_trace_model` whole-model dynamic path, but traced from
  `AutoModelForCausalLM` so `lm_head` / `logits` are in-graph — confirmed: `build_full_model_wrapper(dynamic=True).forward`
  returns `out[0]` = logits `[1, S, vocab]`. Plus `CompiledProgram` build + `set_sym_values` + `run` + `outputs`, and
  the dtype-preserving binding (incl. the W4A16 packed-int path) so the **AWQ chat model** generates too.
- **Create:** `deplodock/commands/generate.py` (the host generate loop: embed → run dynamic program at S → slice
  `logits[0,-1,:]` → sample → append → repeat to EOS / max), `deplodock/serving/sampling.py` (greedy / temperature /
  top-k / top-p — pure numpy/torch, no vLLM), a chat-template helper (delegate to `transformers.apply_chat_template`).
- **Verify:** greedy decode matches HF `model.generate(do_sample=False)` token-for-token on an fp16 model and the AWQ
  model (GPU+network gated, manual); fixed-seed sampling sanity; a hermetic 1-layer random-weight CausalLM test of the
  loop wiring (last-token slice, append, position increment) vs an eager reference for a few steps.

### Phase 1 — Per-layer "everything-but-attention" subgraph (the deplodock enabler; no vLLM)

The core compiler enabler and riskiest trace work. From one HF decoder layer, produce compiled subgraphs:
**pre-attention** `(hidden[num_tokens,H]) → (q,k,v)` and **post-attention** `(attn_out[num_tokens,Hv]) → layer_out`,
with `SdpaOp` excised.

- **Create:** `build_attention_split_wrapper(block, ...)` in `trace/huggingface.py`, sibling to `build_layer_wrapper`.
  It **substitutes `block.self_attn`** with a module that runs `qkv_proj` (+ q/k norm), returns **un-rotated** q,k,v
  (A2), and never calls SDPA / o_proj; a second wrapper runs o_proj + MLP + post-norm from an attn-out input. (Fallback
  only if substitution can't express the post-half: post-trace cut of the `SdpaOp` node in frontend IR — promote its
  q/k/v inputs to graph outputs, its output to a graph input.)
- **Flattened layout:** trace the subgraph with a 2-D `[num_tokens, H]` activation, `num_tokens` symbolic (reuse
  `parse_position_specs` / `build_torch_dynamic_shapes`, spec `seq_len@x:0`). All pre/post compute is pointwise or a
  matmul over H, so collapsing `[1,seq,H] → [seq,H]` is layout-invariant — the property that makes Option A work; the
  q/k/v reshape-into-heads is the danger spot to test.
- **Modify:** `commands/compile.py::_trace_model` — `--attention-split` debug branch beside the `--layer` branch.
- **Verify (hermetic, no vLLM):** 1-layer Qwen3 (GQA), dynamic `num_tokens`. Run pre-subgraph → q/k/v → external torch
  SDPA → post-subgraph; compare layer output to eager `block(x)`. Assert flattened-vs-`[1,seq,H]` equivalence.

### Phase 2 — `DeplodockGenRunner` + multi-layer host stitch (still no vLLM)

Consolidate Phase 1 into a per-layer program and prove a whole-model Python stitch interleaving a **reference torch
SDPA** between deplodock per-layer kernels across all layers — the dress rehearsal for the vLLM forward without vLLM's
runner, isolating the interleave / host-sync mechanics.

- **Create:** `deplodock/serving/gen_runner.py` — `DeplodockGenRunner` (sibling to `DeplodockForwardRunner`): traces
  every layer's split subgraphs, compiles, binds weights (`binder.bind_constants`), exposes `embed`,
  `forward_layer_pre(L, hidden, positions) → (q,k,v)`, `forward_layer_post(L, attn_out) → hidden`, `final_norm`. Reuses
  the captured-graph replay machinery verbatim (`set_sym_values` / `upload_prefix_device` / `capture_program_graph` /
  `replay_program_graph` / `output_prefix_device`) and the dlpack zero-copy device I/O.
- **Decision settled here:** one compiled program **per layer** (simplest; layers are structurally identical so
  capture/replay caches well) vs one all-layers program with attn boundaries (heavier surgery). Start per-layer.
- **Verify:** full forward of a small CausalLM, deplodock per-layer kernels + reference torch SDPA between, matches
  eager logits.

### Phase 3 — `DeplodockGenModel` vLLM plugin (the integration)

Wire Phase 2's runner into a vLLM generative model.

- **Create:** `deplodock/serving/vllm_model_gen.py` — `DeplodockGenModel(nn.Module, …text-gen interfaces)`. **NOT**
  `IsAttentionFree`: it constructs real vLLM `Attention` layers (one per decoder layer, via `vllm.attention.Attention`
  with the model's `num_heads` / `num_kv_heads` / `head_dim` / scale / kv-dtype) so vLLM allocates a KV-cache spec and
  runs paged attention; those layers carry no weights. All weight-bearing compute stays in the deplodock per-layer
  programs (`DeplodockGenRunner`), so `load_weights` returns ~empty (same trick as `DeplodockEmbedModel`; lm_head / embed
  that vLLM owns *are* loaded — get the split-ownership `load_weights` set exactly right or vLLM rejects the model).
  - `forward(input_ids, positions, …)` over flattened `[num_tokens]`: `hidden = runner.embed(ids)`; per layer:
    `q,k,v = runner.forward_layer_pre(L, hidden, positions)`; `q,k = self.rotary_emb[L](positions, q, k)` (A2);
    `attn_out = self.attn[L](q,k,v)` (vLLM paged attention; pulls `attn_metadata` from the forward-context global);
    `hidden = runner.forward_layer_post(L, attn_out)`. Final norm in runner. Return `hidden[num_tokens, H]`.
  - `compute_logits(hidden)`: `self.logits_processor(self.lm_head, hidden)` — vLLM owns lm_head + vocab-parallel (a
    single matmul not worth compiling on the first cut). Plus `embed_input_ids`.
- **Register:** add `ModelRegistry.register_model("DeplodockGenModel", "deplodock.serving.vllm_model_gen:DeplodockGenModel")`
  in `serving/__init__.py::register` (the `vllm.general_plugins` entry point already exists). Select via
  `--hf-overrides '{"architectures":["DeplodockGenModel"]}'`.
- **Modify:** `commands/serve.py` — generative branch (`--runner generate` not `pooling`, gen arch override, keep
  `--enforce-eager`), reusing the flag-split / forward plumbing.
- **Dropped (embedding-specific):** `IsAttentionFree`, `DispatchPooler`, `packed.split_spans`, `attn_type="encoder_only"`,
  the `_mask` / `_causal_mask_np` causal-mask machinery (vLLM's paged attention owns masking now).
- **Verify:** `vllm serve <model> --runner generate --hf-overrides …DeplodockGenModel…`, then a `/v1/chat/completions`
  greedy request matches the **Phase 0 oracle** token-for-token (the reason Phase 0 came first). GQA explicitly checked
  (Qwen3). GPU-gated `tests/serving/test_vllm_plugin_gen_gpu.py` mirroring `test_vllm_plugin_gpu.py`.

### Phase 4 — Prefill + decode hardening

Make it robust across vLLM's two regimes: **prefill** (`num_tokens` = prompt length, large) and **decode**
(`num_tokens` = concurrently-decoding sequences, small / 1). One dynamic per-layer program serves both.

- **Captured-graph reuse:** decode runs at small, recurring `num_tokens`; confirm capture is keyed on `num_tokens` and
  steady-state decode hits the replay path (one launch / layer / step), not recapture. If decode width thrashes
  (sequences finishing), pad to a few bucket widths (mirrors the embedding runner's per-S capture).
- **GQA / head alignment** at the deplodock↔`Attention` seam (deplodock emits the right kv-head count; vLLM's
  `Attention` must be constructed to match or it misreads the layout).
- **Verify:** long-prompt prefill + multi-step decode under concurrency; logits drift over a 100-token generation vs the
  oracle; decode is one replay / layer / step.

### Future work (coda — out of this doc's detailed scope)

- **Phase 5 — tuning / decode-shaped goldens:** the per-layer kernels are new op identities → tuned via the existing
  `tune` path; decode `[small, H]` skinny matmuls likely need decode-shaped goldens (`tune-golden`). Bench vs stock vLLM
  (`vllm bench serve`).
- **Phase 6 — A1: fold RoPE into deplodock** with runtime positions (cos/sin table baked, index gathered by a
  `positions[num_tokens]` runtime input via `IndexMapOp`) — drops the vLLM rope hop. Optional; A2 stays the fallback.
- **Phase 7 — standalone serving:** deplodock's own KV cache + an **incremental-attention kernel** (new q `[1,Hq]` vs
  cached K/V `[S,Hkv]` — the inverse of today's full-sequence SDPA, the one deep new compiler capability) + a minimal
  OpenAI server reusing Phase 0's generate loop + `sampling.py`. Deliberately last: it duplicates what vLLM gives free.

## Critical files

**Create:** `commands/generate.py` (Phase 0 loop; reused by Phase 7), `serving/sampling.py` (sampler + chat template; no
vLLM), `serving/gen_runner.py` (`DeplodockGenRunner`), `serving/vllm_model_gen.py` (`DeplodockGenModel`).

**Modify (reuse heavily):** `compiler/trace/huggingface.py` (new `build_attention_split_wrapper`; A1 rope later),
`commands/compile.py::_trace_model` (`--attention-split` branch), `serving/__init__.py` (register gen model),
`commands/serve.py` (generative branch). `passes/frontend/decomposition/010_sdpa.py` + `ir/frontend/ir.py` only if the
Phase-1 surgery fallback is used (the substitution path leaves them untouched).

**Reuse unchanged (load-bearing machinery):** `backend/cuda/program.py` (`CompiledProgram` build / `set_sym_values` /
capture/replay / `*_prefix_device`), `loader/binder.py` (`bind_constants`, the W4A16 dtype-preserving helpers),
`trace/dynamic.py` (`parse_position_specs` / `build_torch_dynamic_shapes` / `DYNAMIC_DIM_MAX`), `trace/torch.py`
(`trace_module`), and the entire embedding runner as the structural template (`serving/runner.py`,
`serving/vllm_model.py`).

## Top risks

1. **Per-layer interleave host-sync (headline).** N round-trips/step: deplodock kernel → Python → vLLM `self.attn` →
   Python → deplodock kernel. The embedding runner proves dlpack zero-copy keeps tensors on-GPU across the boundary and
   captured replay collapses each deplodock side to one launch — but the interleave itself can't be one captured graph.
   Keep everything on one stream (`cp.cuda.Stream.from_external(torch.cuda.current_stream())`); measure decode-step
   latency early (Phase 4). Fallback: fewer / larger compiled chunks.
2. **Flattened-token compile correctness** — layout-invariance holds only because pre/post compute is pointwise +
   matmul-over-H; the q/k/v reshape-into-heads is the danger spot. De-risk with the Phase-1 equivalence test before any
   vLLM work.
3. **Decode-shaped skinny matmuls** (`num_tokens` tiny) — far from the goldens' prefill-ish shapes; verify *correct* at
   `num_tokens=1` first, tune in Phase 5.
4. **Captured-graph reuse across decode steps** — must be keyed on `num_tokens` and replay (not recapture) in steady
   state; bucket widths if thrash.
5. **GQA / head-count alignment** at the deplodock↔`Attention` seam — cross-check at construction; test on Qwen3.
6. **lm_head / `load_weights` split ownership** — vLLM owns lm_head / embed (loaded via `load_weights`), deplodock owns
   the trunk (not loaded); the returned set must satisfy vLLM's strict all-params-initialized check exactly.

## End-to-end verification

1. Phase 0: `deplodock generate <fp16 model>` greedy == HF `model.generate(do_sample=False)` token-for-token; same on
   the AWQ chat model. Hermetic loop-wiring unit test.
2. Phase 1: hermetic per-layer split (pre → torch SDPA → post) == eager `block(x)`; flattened-vs-`[1,seq,H]` equivalence.
3. Phase 2: whole-model host stitch (deplodock layers + ref SDPA) == eager logits.
4. Phase 3: `vllm serve … DeplodockGenModel` → `/v1/chat/completions` greedy == Phase 0 oracle, token-for-token; GQA ok.
5. Phase 4: 100-token generation logits drift vs oracle; decode = one replay / layer / step under concurrency.
6. `make test && make lint` throughout.
