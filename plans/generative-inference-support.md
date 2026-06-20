# Generative (chat) inference for deplodock

> Status: **design / scoping doc. Not yet implemented.** Goal: serve a decoder-only chat model (Qwen3 / Llama-3)
> through deplodock-compiled kernels, with **vLLM owning the API / sampler / scheduler / KV-cache / chat template**
> first; a deplodock-standalone server later. **Independent of the W4A16 quantization work on its own branch** — this
> plan runs the **unquantized fp16 path** (fp16 is a dtype, not quantization: the embedding serving runner already binds
> + runs fp16 end-to-end, `serving/runner.py`); quantized (AWQ) generation rides on the quantization branch separately,
> once both land. (Qwen3 / Llama-3 ship bf16, and numpy has no native bf16, so the constant path runs them as **fp16** —
> a small accuracy delta; true bf16 is separate plumbing, still not quantization.)
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

- **Carve via explicit per-architecture pre/post wrappers** that read the block's own submodules (not post-trace graph
  surgery; substituting `self_attn` won't work — HF `self_attn.forward` returns `(attn_output, weights)`, not q/k/v).
  The **pre** wrapper runs `input_layernorm` → HF's separate `q_proj`/`k_proj`/`v_proj` (+ per-head `q_norm`/`k_norm` on
  Qwen3) → **returns q,k,v** (no SDPA, no o_proj); the **post** wrapper runs `o_proj` + residual + post-norm + MLP +
  residual from an `(attn_out, residual)` input. SDPA never enters the trace by construction. (Note: the fused
  `qkv_proj` in vLLM's `Qwen3Attention.forward` above is a vLLM detail — the HF module deplodock traces has three
  separate projections.)
- **vLLM applies RoPE in the first cut (A2).** deplodock emits **un-rotated** q/k; `DeplodockGenModel.forward` calls
  vLLM's `self.rotary_emb(positions, q, k)` between the pre-attention kernel and `self.attn`. This deletes the
  runtime-arbitrary-positions trace work from the critical path. **deplodock owning RoPE with runtime positions (A1) is
  a later optimization**, not a prerequisite.

**Scope (Phases 0–4): `tensor_parallel_size=1` only.** The runner loads the full checkpoint per process
(`serving/runner.py`) with no projection sharding; vLLM's paged `Attention` and vocab-parallel `lm_head` assume
rank-local shards, which the deplodock trunk does not produce. TP>1 (sharded q/k/v + o_proj, local head counts,
row-parallel o_proj/MLP reductions, vocab-parallel lm_head, the matching collectives) is **future work** — Phases 0–4
assume a single rank (fp16, so the first cut covers models that fit one GPU at fp16).

## Alternatives considered

The end-state the user asked for is *vLLM first, deplodock-standalone later* — so the two genuinely-viable options (A
and B) are not competitors but **phases of one trajectory**; the other three are forced out by the constraints above.

- **Option A — carve SDPA out, vLLM owns paged attention (CHOSEN; this whole doc).** deplodock compiles each layer's
  per-token everything-but-attention; vLLM's `Attention` layer owns the KV cache and attention math. *Why first:* it
  reuses vLLM's entire serving stack for free — OpenAI API, sampler, scheduler, paged KV cache, continuous batching,
  chat template, streaming — and needs **no new deplodock cache or incremental-attention kernel**, so a working chat
  model lands soonest and the per-layer kernels (where deplodock's value is) get de-risked and tuned under real load.
  *Cost:* the per-layer host-sync interleave (Top risk #1) — N deplodock↔vLLM round-trips per step, and the interleave
  itself can't collapse into one captured graph.

- **Option B — deplodock standalone: own KV cache + incremental-attention kernel (the eventual goal; Phase 7).** No
  vLLM: deplodock manages its own KV cache and a new **incremental-attention kernel** (new q `[1,Hq]` vs cached K/V
  `[S,Hkv]` — the inverse of today's full-sequence SDPA, the one deep new compiler capability), with Phase 0's loop +
  `sampling.py` behind a minimal OpenAI server. *Why it wins long-term:* no Python interleave — a whole decode step can
  be **one captured graph**, the low-latency story deplodock is built for — and no vLLM coupling. *Why deferred:*
  it reimplements everything vLLM gives free (cache, scheduler, continuous batching, API, sampler) AND needs the hardest
  new kernel; doing it first would gate the first chat output on the largest pile of new code. A proves the kernels
  correct so B becomes "swap the host loop", not "build it all at once".

- **Option C — full-sequence recompute each step (O(S²)).** Re-run the whole growing prefix every token. *Adopted only
  as the Phase 0 standalone oracle* — no new attention/cache compiler work (it reuses the existing whole-model
  path), a correctness reference for every later phase. Rejected as a product: quadratic, and structurally **impossible
  inside vLLM** (the V1 runner never re-feeds prior context).

- **Option D — Mamba-style self-managed state (`IsAttentionFree` / `MambaSpec`).** Forced out: `MambaSpec` assumes a
  *constant-size* recurrent state, but a KV cache grows with sequence length — deplodock cannot present its cache as
  Mamba "state". (This is the embedding plugin's path; it does not generalize to generation.)

- **Option E — one monolithic deplodock kernel that calls vLLM attention mid-graph.** Forced out: deplodock compiles a
  static dataflow graph; it cannot yield to a Python `self.attn(q,k,v)` call partway through a kernel. The deplodock↔
  attention interleave must live at the Python `forward` level, per layer — which is exactly what Option A does.

**Sub-options inside A** (each resolved in the phases below, not re-litigated here): *who applies RoPE* — vLLM (**A2**,
chosen for the first cut) vs deplodock with runtime positions (**A1**, Phase 6); *how to carve SDPA* — explicit
per-architecture pre/post wrappers reading the block's submodules (chosen) vs post-trace frontend-IR graph surgery
(fallback); *program granularity* — two programs (pre + post) per layer (chosen) vs one all-layers program with
attention boundaries (Phase 2).

## Phases (correctness-first; detail front-loaded on 0–4)

### Phase 0 — Standalone naive `deplodock generate` (the correctness ORACLE)

First, even though the end goal is vLLM: it needs **no new SDPA-carve / attention-split compiler work**, produces real
chat output now, and is the correctness reference every later phase verifies against (and is reused by the eventual
standalone server). It uses the full-recompute strategy (re-run the whole growing prefix each step) — valid *standalone*
(deplodock controls the loop), intentionally O(S²), an oracle not a product. Runs the **unquantized fp16 path** — the
whole-model CLI path currently forces fp32 (`_trace_model` for `layer is None`; the runnable binder /
`bind_constants_from_module` cast to float32), so the one fp16 change is a `dtype` param threaded through those two
helpers (dtype plumbing, **no dependency on the W4A16 branch** — the serving runner already binds fp16). Key
simplification: the recompute loop always runs at positions `0..S-1`, so the existing `_SlicedRotary` is exactly
right — **no RoPE-runtime-positions work needed here** (that is purely a Phase-6 / incremental concern).

- **Prerequisite spike (de-risk first):** the whole-model dynamic path is **trace-tested but not yet proven to lower and
  execute end-to-end** — `tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_traces` asserts only
  the trace, and its docstring flags whole-model CUDA lowering (int64 embedding-gather kernels, lm_head matmul) as
  uncovered. Phase 0 starts with a one-shot `compile`+`run` of a small CausalLM whole-model **fp16** graph to surface
  and fill any lowering gaps *before* the loop is built. "Reuse, no compiler change" holds only if this spike is clean —
  budget for embedding-gather / lm_head lowering work if not.
- **Reuses:** `commands/compile.py::_trace_model` whole-model dynamic path, traced from `AutoModelForCausalLM` so
  `lm_head` / `logits` are in-graph — `build_full_model_wrapper(dynamic=True).forward` returns `out[0]` = logits
  `[1, S, vocab]` (Phase 0's generation wrapper slices the final position before `lm_head` → `[1, vocab]`; see Create).
  Plus `CompiledProgram` build + `set_sym_values` + `run` + `outputs` and the constant binding (the
  `dtype`-parameterized fp16 path).
- **Create:** `deplodock/commands/generate.py` (the host generate loop: feed `input_ids[0:S]` → run dynamic program →
  read last-token `logits[1, vocab]` → sample → append id → repeat to EOS / max; the graph embeds ids internally — the
  loop never handles embeddings). A **generation wrapper** (variant of `build_full_model_wrapper`) slices the final
  hidden state and applies `lm_head` to **that position only**, returning `[1, vocab]` — not the whole-prefix
  `[1, S, vocab]` (`CompiledProgram.outputs` copies the full buffer to host, and lm_head over every prefix position is
  O(S·vocab) wasted/step). Plus `deplodock/serving/sampling.py` (greedy / temperature / top-k / top-p — pure
  numpy/torch, no vLLM), a chat-template helper (delegate to `transformers.apply_chat_template`).
- **Verify:** greedy decode vs HF `model.generate(do_sample=False)` **run in fp16** (apples-to-apples with deplodock's
  fp16 path — oracle matches the serving dtype, so the test isolates the carve/interleave, not a dtype gap) — gated
  **primarily on per-step logits within tolerance + top-1 agreement when the margin is sufficient**,
  with token-for-token kept as a short-decode smoke test (a single fp-driven argmax flip compounds, so it is not the
  primary gate — see Top risk #7). Fixed-seed sampling sanity; a hermetic 1-layer random-weight CausalLM test of the
  loop wiring (last-token slice, append, position increment) vs an eager reference for a few steps.

### Phase 1 — Per-layer "everything-but-attention" subgraph (the deplodock enabler; no vLLM)

The core compiler enabler and riskiest trace work. From one HF decoder layer, produce compiled subgraphs:
**pre-attention** `(hidden[num_tokens,H]) → (q,k,v)` and **post-attention**
`(attn_out[num_tokens,Hv], residual[num_tokens,H]) → layer_out`, with `SdpaOp` excised. The `residual` is the
**layer-input `hidden`** (the pre-norm residual the decoder block adds back after o_proj) — the host already holds it,
so `pre` need not return it; `post` takes it as a second input and computes `residual + o_proj(attn_out)` then the
MLP sub-block (with its own internal second residual).

- **Create:** `build_attention_split_wrapper(block, ...)` in `trace/huggingface.py`, sibling to `build_layer_wrapper`.
  Two **explicit per-architecture wrappers** that read the block's own submodules (NOT a `self_attn` substitution — HF
  `self_attn.forward` returns `(attn_output, attn_weights)`, not q/k/v, and the block adds `residual + attn_output`
  after it, so swapping `self_attn` can't yield a clean pre-graph). For **Qwen3**: the **pre** wrapper runs
  `input_layernorm` → separate `q_proj` / `k_proj` / `v_proj` (HF Qwen3 has three; the fused `qkv_proj` is a vLLM
  detail) → reshape-into-heads → per-head `q_norm` / `k_norm`, returns **un-rotated** q,k,v (A2) — no SDPA, no o_proj.
  The **post** wrapper runs `o_proj` → `residual +` → `post_attention_layernorm` → `mlp` → `residual2 +` from an
  `(attn_out, residual)` pair (residual = the layer-input `hidden`). An **architecture adapter** handles Llama next (no
  `q_norm`/`k_norm`; otherwise identical). (Fallback only if the explicit wrappers prove unworkable: post-trace cut of
  the `SdpaOp` node in frontend IR — promote its q/k/v inputs to graph outputs, its output to a graph input.)
- **Seam q/k/v ABI (define exactly):** the pre wrapper emits **2-D** `q[T, Hq·D]`, `k[T, Hkv·D]`, `v[T, Hkv·D]` — heads
  folded into the last dim, tokens leading — exactly what vLLM's `Attention.forward(q,k,v)` consumes (**assert** these
  shapes at the seam). HF Qwen3's `.view(B,S,-1,D).transpose(1,2)` assumes a `[batch, seq, hidden]` input; on the
  flattened `[T, H]` layout the reshape is `.view(T, n_heads, D)` with **no batch/seq transpose**. The `[1, n_heads, T,
  D]` layout torch SDPA wants is built **only inside the oracle** (Phase 1/2 reference), never in the deplodock kernels
  or at the vLLM seam.
- **Flattened layout:** trace the subgraph with a 2-D `[num_tokens, H]` activation, `num_tokens` symbolic (reuse
  `parse_position_specs` / `build_torch_dynamic_shapes`, spec `seq_len@x:0`). All pre/post compute is pointwise or a
  matmul over H, so collapsing `[1,seq,H] → [seq,H]` is layout-invariant — the property that makes Option A work; the
  q/k/v reshape-into-heads is the danger spot to test.
- **Modify:** `commands/compile.py::_trace_model` — `--attention-split` debug branch beside the `--layer` branch.
- **Verify (hermetic, no vLLM):** 1-layer Qwen3 (GQA), dynamic `num_tokens`. The reference path must **reconstruct
  RoPE** (pre emits un-rotated q/k under A2): pre-subgraph → q/k/v → **apply rotary to q,k at `positions`** → external
  causal GQA torch SDPA → post-subgraph (fed `attn_out` **and** the saved layer-input `hidden` as residual); compare
  layer output to eager `block(x)`. Assert flattened-vs-`[1,seq,H]` equivalence.

### Phase 2 — `DeplodockGenRunner` + multi-layer host stitch (still no vLLM)

Consolidate Phase 1 into a per-layer program and prove a whole-model Python stitch interleaving a **reference torch
SDPA** between deplodock per-layer kernels across all layers — the dress rehearsal for the vLLM forward without vLLM's
runner, isolating the interleave / host-sync mechanics.

- **Create:** `deplodock/serving/gen_runner.py` — `DeplodockGenRunner` (sibling to `DeplodockForwardRunner`): traces
  every layer's split subgraphs, compiles, binds weights (`binder.bind_constants`), exposes `embed`,
  `forward_layer_pre(L, hidden, positions) → (q,k,v)`, `forward_layer_post(L, attn_out, residual) → hidden` (residual =
  the `hidden` fed to `forward_layer_pre`), `final_norm`. Reuses the captured-graph replay machinery verbatim
  (`set_sym_values` / `upload_prefix_device` / `capture_program_graph` / `replay_program_graph` /
  `output_prefix_device`) and the dlpack zero-copy device I/O.
- **Decision settled here:** **two compiled programs per layer** (`pre` + `post`, each its own captured graph / replay /
  buffer set — separate because vLLM attention runs between them, so a decode step is **two replays / layer**, not one)
  at **per-layer** granularity (simplest; layers are structurally identical so capture/replay caches well) vs one
  all-layers program with attn boundaries (heavier surgery). Start per-layer.
- **Memory budget — an early spike that GATES this design:** each `CompiledProgram` owns persistent input/output buffers
  **and** a pinned scratch slab, plus its own 64-entry LRU CUDA-graph cache (`backend/cuda/program.py`). With **2 ×
  n_layers** capacity-sized programs the MLP-intermediate workspaces can add several GB, and the graph caches can reach
  thousands of entries globally. Run the spike first; if over budget, the lightest lever is a **graph-cache cap**
  — but `_graph_cache_max` is a **per-`CompiledProgram` field** (default 64, `program.py`), so a *global* budget across
  the 2×n_layers programs needs the **runner to set each program's cap** (e.g. budget / program-count; small allocation
  logic, not pure config) — plus capping capacity widths. **Shared scratch across programs is NOT free** — it needs
  backend work (externally-supplied storage, per-program slab views, pointers stable across CUDA-graph capture), which
  moves `program.py` out of *reuse-unchanged* (add to Critical Files if pursued). If neither suffices the spike can
  **reject two-programs-per-layer** in favor of fewer/larger chunks.
- **Verify:** full forward of a small CausalLM — deplodock per-layer `pre` kernels → **reconstruct RoPE on q/k at
  `positions`** → reference causal GQA torch SDPA → `post` kernels, across all layers — matches eager logits. (The
  reference stitch applies the same RoPE the vLLM forward will, since `pre` emits un-rotated q/k under A2.)

### Phase 3 — `DeplodockGenModel` vLLM plugin (the integration)

Wire Phase 2's runner into a vLLM generative model.

- **Create:** `deplodock/serving/vllm_model_gen.py` — `DeplodockGenModel(nn.Module, …text-gen interfaces)`. **NOT**
  `IsAttentionFree`: it constructs real vLLM `Attention` layers (one per decoder layer, via `vllm.attention.Attention`
  with the model's `num_heads` / `num_kv_heads` / `head_dim` / scale / kv-dtype) so vLLM allocates a KV-cache spec and
  runs paged attention; those layers carry no weights. **Each `Attention` needs a unique `prefix`** — vLLM keys
  `static_forward_context` / cache-spec discovery by it and rejects duplicates (the default empty prefix collides on the
  second layer), so derive `f"{prefix}.layers.{i}.self_attn.attn"` from the model's own `prefix`, and pass the
  version-pinned config args (`cache_config` / `quant_config` off `vllm_config`). Test that vLLM discovers one KV-cache
  spec per layer. **Dtype coherence (force fp16 across the seam):** vLLM owns `Attention` / the KV cache / `lm_head`
  and defaults to `--dtype auto` → **bf16** for a bf16 checkpoint, but the deplodock trunk emits **fp16** q/k/v + hidden
  — a mismatch at every seam. The generative `serve` branch **forces `--dtype float16`** (so vLLM's `Attention`,
  KV-cache dtype, and `lm_head` all init fp16) and rejects an incompatible `--dtype` override. **RoPE is NOT in
  `Attention`:** a bare vLLM `Attention` does paged attention only (stock `Qwen3Attention` builds RoPE separately via
  `get_rope`, then calls it before `self.attn`), so `DeplodockGenModel` constructs its own RoPE via
  `get_rope(head_dim, rotary_dim, max_position, rope_parameters=…)` from the HF config (theta + any rope-scaling) — one
  shared module across layers (RoPE is position-only). Test Qwen3 + Llama-3 RoPE vs stock vLLM. **Split ownership
  (settled):** the **deplodock runner owns the
  embedding + the whole trunk** (bound into its constants); **vLLM owns only `lm_head`** (loaded via `load_weights`). So
  `load_weights` claims just `lm_head.*` and returns ~empty for the trunk (same trick as `DeplodockEmbedModel`) — get
  this set exactly right or vLLM rejects the model on its strict all-params-initialized check. **Tied embeddings
  (config-dependent):** when `tie_word_embeddings=True` (e.g. small Qwen3) the checkpoint may store **only**
  `embed_tokens.weight` (no `lm_head.weight`), so `load_weights` must load the head from that embedding alias; when
  untied (e.g. Llama-3-8B) it loads `lm_head.*` normally. Either way the runner separately binds the embedding for the
  trunk, so the tied case **duplicates** the matrix (one extra copy, simplest); sharing is a later optimization.
  - `forward(input_ids, positions, …)` over flattened `[num_tokens]`: `hidden = runner.embed(ids)`; per layer:
    `residual = hidden`; `q,k,v = runner.forward_layer_pre(L, hidden, positions)`;
    `q,k = self.rotary_emb(positions, q, k)` (A2, the shared RoPE module above); `attn_out = self.attn[L](q,k,v)` (pulls
    `attn_metadata` from the forward-context global); `hidden = runner.forward_layer_post(L, attn_out, residual)`. Two
    deplodock replays per layer (`pre`, `post`) bracket the vLLM attention call. Final norm in runner. Return
    `hidden[num_tokens, H]`.
  - `compute_logits(hidden)`: `self.logits_processor(self.lm_head, hidden)` — vLLM owns lm_head + vocab-parallel (a
    single matmul not worth compiling on the first cut). Any vLLM `get_input_embeddings` / `embed_input_ids` hook
    **delegates to `runner.embed`** (the runner owns embedding).
- **Register:** add `ModelRegistry.register_model("DeplodockGenModel", "deplodock.serving.vllm_model_gen:DeplodockGenModel")`
  in `serving/__init__.py::register` (the `vllm.general_plugins` entry point already exists). Select via
  `--hf-overrides '{"architectures":["DeplodockGenModel"]}'`.
- **Modify:** `commands/serve.py` — generative branch (`--runner generate` not `pooling`, gen arch override, **force
  `--dtype float16`** for coherence + reject incompatible `--dtype`, keep `--enforce-eager`), reusing the flag-split
  / forward plumbing.
- **Dropped (embedding-specific):** `IsAttentionFree`, `DispatchPooler`, `packed.split_spans`, `attn_type="encoder_only"`,
  the `_mask` / `_causal_mask_np` causal-mask machinery (vLLM's paged attention owns masking now).
- **Verify (two tiers — HTTP can't expose full logits):** (1) **endpoint smoke** — `vllm serve <model> --runner
  generate --hf-overrides …DeplodockGenModel…`, then a `/v1/chat/completions` greedy request matches the **Phase 0
  oracle**'s generated **tokens** (and top-k `logprobs` where returned); (2) **in-process numerical** — the
  `self.attn(q,k,v)` needs initialized KV caches **and** vLLM's forward-context attention metadata — it can't just call
  `DeplodockGenModel.forward` in a vacuum: run an **in-process V1 engine** (`vllm.LLM(...)`) with a **logits-capture
  hook** around `compute_logits` (or build an explicit forward-context + dummy-cache harness) and gate **per-step full
  logits within tolerance + top-1 agreement** vs the oracle (the HTTP path can't return `[vocab]` per step). GQA
  explicitly checked (Qwen3). GPU-gated `tests/serving/test_vllm_plugin_gen_gpu.py` mirroring `test_vllm_plugin_gpu.py`.

### Phase 4 — Prefill + decode hardening

Make it robust across vLLM's two regimes: **prefill** (`num_tokens` = prompt length, large) and **decode**
(`num_tokens` = concurrently-decoding sequences, small / 1). The same two dynamic per-layer programs (`pre` + `post`)
serve both.

- **Bound the flattened width `T` (capacity sizing):** `T` = `num_tokens` is the **sum of newly-scheduled tokens across
  all requests** in a step (continuous batching), **not** one request's length — so `--max-model-len` does NOT bound it.
  The cap is vLLM's `scheduler_config.max_num_batched_tokens`; assert it `≤ DYNAMIC_DIM_MAX` (4096, `trace/dynamic.py`)
  and **size the programs' capacity buffers from `max_num_batched_tokens`**, not max-model-len. `commands/serve.py`'s
  generative branch sets `--max-num-batched-tokens` accordingly (and rejects a larger value).
- **Captured-graph reuse:** decode runs at small, recurring `num_tokens`; confirm capture is keyed on `num_tokens` and
  steady-state decode hits the replay path (**two replays / layer / step** — `pre` and `post`, bracketing vLLM
  attention), not recapture. **Do NOT pad q/k/v to bucket widths the way the embedding runner does** — vLLM's
  `attn_metadata` is built for *exactly* the scheduled token count, and padded rows have no paged-attention cache slots,
  so they'd desync the seam. With `--enforce-eager` vLLM presents exact widths, so capture **per exact `num_tokens`**
  (steady-state decode widths recur, so the cache still hits); only pad if coordinated with vLLM's own metadata/padding.
- **GQA / head alignment** at the deplodock↔`Attention` seam (deplodock emits the right kv-head count; vLLM's
  `Attention` must be constructed to match or it misreads the layout).
- **Verify:** long-prompt prefill + multi-step decode under concurrency; **per-step logits within tolerance** vs the
  oracle over a 100-token generation (token-for-token a smoke check only — argmax flips compound); decode is **two
  replays / layer / step** (`pre` + `post`).

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

**Create:** `commands/generate.py` (Phase 0 loop; reused by Phase 7 — exposes `register_generate_command`, wired into
`deplodock.py`'s imports + `main()` like every other command, with CLI-parsing tests), `serving/sampling.py` (sampler +
chat template; no vLLM), `serving/gen_runner.py` (`DeplodockGenRunner`), `serving/vllm_model_gen.py`
(`DeplodockGenModel`).

**Modify (reuse heavily):** `compiler/trace/huggingface.py` (new `build_attention_split_wrapper` + a Phase-0
generation wrapper that slices the final position before `lm_head`; A1 rope later),
`commands/compile.py::_trace_model` (`--attention-split` branch; thread a `dtype` param here + through the runnable
binder so the whole-model path binds fp16), `serving/__init__.py` (register gen model), `deplodock.py` (import + call
`register_generate_command` in `main()`), `commands/serve.py` (generative branch).
`passes/frontend/decomposition/010_sdpa.py` + `ir/frontend/ir.py` only if the
Phase-1 surgery fallback is used (the explicit-wrapper path leaves them untouched).

**Reuse unchanged (load-bearing machinery):** `backend/cuda/program.py` (`CompiledProgram` build / `set_sym_values` /
capture/replay / `*_prefix_device`; unchanged **unless** the Phase-2 memory spike forces shared cross-program scratch —
that is backend work, see Phase 2), `loader/binder.py` (`bind_constants`, fp16 constant binding),
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
   matmul-over-H; the q/k/v reshape-into-heads is the danger spot (HF's `.view(B,S,-1,D).transpose(1,2)` assumes
   `[batch,seq,hidden]` — on `[T,H]` it must be `.view(T,n_heads,D)` with no transpose; the seam emits 2-D `q[T,Hq·D]` /
   `k,v[T,Hkv·D]`, asserted against vLLM's `Attention` input). De-risk with the Phase-1 equivalence test before any vLLM
   work.
3. **Decode-shaped skinny matmuls** (`num_tokens` tiny) — far from the goldens' prefill-ish shapes; verify *correct* at
   `num_tokens=1` first, tune in Phase 5.
4. **Captured-graph reuse across decode steps** — must be keyed on `num_tokens` and replay (not recapture) in steady
   state. Can't pad to bucket widths like the embedding runner (vLLM's `attn_metadata` expects the exact scheduled
   count — padded rows have no cache slots); capture per exact width (`--enforce-eager` presents exact widths).
5. **GQA / head-count alignment** at the deplodock↔`Attention` seam — cross-check at construction; test on Qwen3.
6. **lm_head / `load_weights` split ownership** — vLLM owns **only lm_head** (loaded via `load_weights`); the deplodock
   runner owns embed + trunk (not loaded); the returned set must satisfy vLLM's strict all-params-initialized check
   exactly. `tie_word_embeddings` is config-dependent: when tied, the checkpoint may carry only `embed_tokens.weight`,
   so `load_weights` loads the head from that alias (trunk duplicates the matrix); when untied, load `lm_head.*`.
7. **Oracle methodology — token-for-token is brittle.** A single fp-driven argmax flip diverges every later token, so
   greedy token equality reads as catastrophic when logits are actually within tolerance. Gate **primarily on per-step
   logits within tolerance + top-1 agreement when the margin is sufficient** (plus a KV-backed-logits vs eager-recompute
   check); keep token-for-token as a short-decode smoke test only. Compare against HF run at the **same dtype as
   deplodock (fp16)**, so the test measures the carve/interleave, not an fp16↔fp32 gap.
8. **Whole-model lowering unproven (dtype-independent)** — only the *trace* is tested today
   (`test_qwen_whole_model_dynamic_traces`); whole-model CUDA execution (int64 embedding-gather, lm_head matmul) is
   uncovered. Phase 0's fp16 compile-and-run spike de-risks this before the generate loop is built; fp16 vs fp32 does
   not change the gap.
9. **Program-count memory blow-up** — 2 × n_layers persistent `CompiledProgram`s, each with capacity-sized buffers + a
   pinned scratch slab + a 64-entry graph cache, can cost several GB of workspace and thousands of cached graphs.
   Memory-budget spike early (Phase 2); share scratch across layer programs and/or cap the global graph cache.
10. **Seam dtype coherence** — the deplodock trunk runs **fp16** but vLLM defaults to `--dtype auto` → **bf16** for a
    bf16 checkpoint, so vLLM's `Attention` / KV cache / `lm_head` would mismatch the fp16 q/k/v + hidden at every seam.
    The generative `serve` branch forces `--dtype float16` and rejects incompatible overrides.
11. **RoPE not provided by `Attention`** — a bare vLLM `Attention` is paged-attention only; `DeplodockGenModel` must
    build its own `get_rope(...)` module (theta + rope-scaling from the HF config) and apply it before `self.attn` (A2).
    Test Qwen3 + Llama-3 against stock vLLM.

## End-to-end verification

1. Phase 0: `deplodock generate <model>` (fp16 path) greedy vs HF `model.generate(do_sample=False)` **run in fp16** —
   per-step logits within tolerance + top-1 agreement (token-for-token a short-decode smoke test). Hermetic loop-wiring
   unit test. Plus the whole-model fp16 compile-and-run spike up front.
2. Phase 1: hermetic per-layer split (pre → **rotary(q,k)** → torch GQA SDPA → post, post fed `attn_out` + residual) ==
   eager `block(x)`; flattened-vs-`[1,seq,H]` equivalence.
3. Phase 2: whole-model host stitch (deplodock layers + **rotary(q,k)** + ref SDPA) == eager logits.
4. Phase 3: `/v1/chat/completions` greedy matches the Phase 0 oracle's **tokens** (HTTP smoke); a separate
   **in-process** engine test with a `compute_logits` hook gates **per-step full logits within tolerance** (HTTP can't
   return `[vocab]`); GQA ok.
5. Phase 4: 100-token generation per-step logits within tolerance vs oracle; decode = **two replays / layer / step**
   (`pre` + `post`) under concurrency.
6. `make test && make lint` throughout.
