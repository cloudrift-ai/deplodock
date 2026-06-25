# Generative (chat) inference for deplodock

> Status: **Phases 0–4 implemented** (merged; this doc is the design record). A chat model runs through
> deplodock-compiled kernels: `deplodock generate` is **token-for-token identical to HF eager** on TinyLlama-1.1B, and
> `deplodock serve --generate` (the vLLM plugin) is **decode-validated** in-process. **Correctness is complete; perf
> hardening remains** (see *Remaining work*). Goal: serve a decoder-only chat model (Qwen3 / Llama-3) through
> deplodock-compiled kernels, with **vLLM owning the API / sampler / scheduler / KV-cache / chat template** first; a
> deplodock-standalone server later. **Independent of the W4A16 quantization work on its own branch** — this runs the
> **unquantized fp16 path** (fp16 is a dtype, not quantization; quantized AWQ generation rides on the quantization
> branch separately). Qwen3 / Llama-3 ship bf16, and numpy has no native bf16, so the constant path runs them as
> **fp16** — a small accuracy delta.
>
> This doc keeps the **design rationale, status, and remaining work**; the step-by-step mechanics live in the code +
> `deplodock/serving/ARCHITECTURE.md`, and the decode-perf analysis in
> [`generative-decode-perf-findings.md`](generative-decode-perf-findings.md).

## Implementation status

**Phases 0–4 implemented and verified:**

- **Phase 0 ✅** — `deplodock generate <model>` standalone oracle (`commands/generate.py`, `serving/sampling.py`).
  Token-for-token identical to HF eager greedy on **TinyLlama-1.1B-Chat** (real weights).
- **Phase 1 ✅** — `build_attention_split_wrapper` carve (`trace/huggingface.py`). Eager-equivalent to `block(x)` for
  Qwen3 (GQA + q/k norm) and Llama; dynamic-compiled and run at multiple token counts.
- **Phase 2 ✅** — `DeplodockGenRunner` (`serving/gen_runner.py`), two per-layer programs. Multi-layer host stitch
  (deplodock kernels + reference SDPA) == eager logits.
- **Phase 3 ✅** — `DeplodockGenModel` vLLM plugin (`serving/vllm_model_gen.py`) + `serve --generate`. In-process vLLM
  engine, greedy decode token-for-token vs HF eager on a tiny Llama.
- **Phase 4 ✅** — flattened-width bound enforced (`serve` caps `--max-num-batched-tokens` at `DYNAMIC_DIM_MAX`).
  **Decode-bucket landed**: decode was ~95% one symbolic kernel whose hint-512 M-tile is pathological at decode M=1
  (~66× off cuBLAS); the runner compiles a **static small-M (bucket) program** per layer used when `num_tokens ≤ bucket`
  → ~11× decode speedup measured (4.5 → ~50 tok/s on TinyLlama). The device zero-copy interleave (~1%) and
  captured-graph replay (~0%) were **ruled out by measurement** — see
  [`generative-decode-perf-findings.md`](generative-decode-perf-findings.md).

**Known limits:** full-causal attention only (sliding-window / per-layer-sliding / dual-chunk configs are **rejected**,
not miscomputed), fp16, TP=1; `serve` compiles up to **4 × n_layers** programs (symbolic + decode-bucket twins), so
startup + memory scale with depth (small models for now). The in-graph `slice_last_logits` lm_head optimization stays
**off** (cold M=1-matmul lowering gap, tracked by an xfail tripwire); `generate` uses full logits + a host slice.

## Context

`deplodock serve` started as **embeddings only** (`deplodock/serving/`): a causal trunk → hidden states, vLLM's pooler
does last-token pooling. The compiler decomposes attention **fully and internally**
(`passes/frontend/decomposition/010_sdpa.py`: QK^T → scale → mask → softmax → PV) over a full-sequence
`[batch, heads, seq, dim]` layout, causal mask `(1,1,S,S)`, with RoPE cos/sin **baked as graph constants** at trace time
(`trace/huggingface.py` `_SlicedRotary` / `_PassThroughRotary`). Correct for *whole-sequence prefill* but structurally
incompatible with how vLLM drives generation.

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
  `qkv_proj` in vLLM's `Qwen3Attention.forward` is a vLLM detail — the HF module deplodock traces has three separate
  projections.) The seam ABI is **2-D**: `q[T, Hq·D]`, `k/v[T, Hkv·D]` (HF's `.view(B,S,-1,D).transpose(1,2)` assumes
  `[batch,seq,hidden]`; on `[T,H]` the reshape is `.view(T, n_heads, D)` with no transpose — the `[1,H,T,D]` SDPA layout
  exists only inside the reference oracle, never in the kernels or at the vLLM seam).
- **vLLM applies RoPE in the first cut (A2).** deplodock emits **un-rotated** q/k; `DeplodockGenModel.forward` calls a
  `get_rope`-built module between the pre-attention kernel and `self.attn`. deplodock owning RoPE with runtime positions
  (A1) is a later optimization, not a prerequisite.

**Scope (Phases 0–4): `tensor_parallel_size=1` only.** The runner loads the full checkpoint per process with no
projection sharding; vLLM's paged `Attention` and vocab-parallel `lm_head` assume rank-local shards, which the deplodock
trunk does not produce. TP>1 (sharded q/k/v + o_proj, local head counts, row-parallel reductions, vocab-parallel
lm_head, collectives) is future work.

## Alternatives considered

The end-state asked for is *vLLM first, deplodock-standalone later* — so the two genuinely-viable options (A and B) are
not competitors but **phases of one trajectory**; the other three are forced out by the constraints above.

- **Option A — carve SDPA out, vLLM owns paged attention (CHOSEN).** *Why first:* reuses vLLM's entire serving stack for
  free — OpenAI API, sampler, scheduler, paged KV cache, continuous batching, chat template, streaming — and needs **no
  new deplodock cache or incremental-attention kernel**, so a working chat model lands soonest and the per-layer kernels
  get de-risked under real load. *Cost:* the per-layer host-sync interleave (N deplodock↔vLLM round-trips per step,
  can't collapse into one captured graph).
- **Option B — deplodock standalone: own KV cache + incremental-attention kernel (the eventual goal; Phase 7).** No
  vLLM: deplodock manages its own KV cache and a new incremental-attention kernel (new q `[1,Hq]` vs cached K/V
  `[S,Hkv]` — the inverse of today's full-sequence SDPA). *Why it wins long-term:* no Python interleave — a whole decode
  step can be **one captured graph**. *Why deferred:* it reimplements everything vLLM gives free AND needs the hardest
  new kernel. A proves the kernels correct so B becomes "swap the host loop", not "build it all at once".
- **Option C — full-sequence recompute each step (O(S²)).** *Adopted only as the Phase 0 standalone oracle* — no new
  attention/cache compiler work, a correctness reference. Rejected as a product: quadratic, and **impossible inside
  vLLM** (the V1 runner never re-feeds prior context).
- **Option D — Mamba-style self-managed state (`IsAttentionFree` / `MambaSpec`).** Forced out: `MambaSpec` assumes a
  *constant-size* recurrent state, but a KV cache grows with sequence length. (The embedding plugin's path; it does not
  generalize to generation.)
- **Option E — one monolithic deplodock kernel that calls vLLM attention mid-graph.** Forced out: deplodock compiles a
  static dataflow graph; it cannot yield to a Python `self.attn(q,k,v)` call partway through a kernel. The interleave
  must live at the Python `forward` level, per layer — which is what Option A does.

**Sub-options inside A** (each resolved in the code): *who applies RoPE* — vLLM (**A2**, chosen) vs deplodock with
runtime positions (**A1**, Phase 6); *how to carve SDPA* — explicit per-architecture pre/post wrappers (chosen) vs
post-trace frontend-IR graph surgery (fallback); *program granularity* — two programs (pre + post) per layer (chosen)
vs one all-layers program with attention boundaries.

## Remaining work

- **Decode perf (near-term):** multi-bucket / per-`Dim` hints for high-concurrency decode (the single bucket covers
  single-stream / small batch); shared weights + scratch to cut the up-to-4×-n_layers program memory; the M=1
  demoted-matmul cold-lowering fix (also unblocks the `slice_last_logits` lm_head optimization). Details +
  measurements in [`generative-decode-perf-findings.md`](generative-decode-perf-findings.md).
- **Phase 5 — tuning / decode-shaped goldens:** the per-layer kernels are new op identities → tuned via the `tune` path;
  decode `[small, H]` skinny matmuls need decode-shaped goldens (`tune-golden`). Bench vs stock vLLM (`vllm bench
  serve`).
- **Phase 6 — A1: fold RoPE into deplodock** with runtime positions (cos/sin baked, gathered by a
  `positions[num_tokens]` runtime input via `IndexMapOp`) — drops the vLLM rope hop. Optional; A2 stays the fallback.
- **Phase 7 — standalone serving:** deplodock's own KV cache + an incremental-attention kernel (new q `[1,Hq]` vs cached
  K/V `[S,Hkv]`) + a minimal OpenAI server reusing Phase 0's loop + `sampling.py`. Deliberately last: it duplicates what
  vLLM gives free.
- **Tensor parallelism (TP > 1)** — sharded projections, local head counts, row-parallel reductions, vocab-parallel
  lm_head, the matching collectives.

## Risks / things to watch

**Open:**

- **Per-layer interleave host-sync** — N deplodock↔vLLM round-trips per step; the interleave itself can't be one
  captured graph. *Measured ~1.3% of the decode step on TinyLlama* — far smaller than feared; the decode bottleneck was
  kernel codegen (the decode-bucket fix), not the interleave.
- **Decode-shaped skinny matmuls** — the decode-bucket covers single-stream / small-batch decode; high-concurrency
  decode (large `num_tokens`) and proper kernel tuning remain.
- **Program-count memory** — up to 4 × n_layers persistent programs (symbolic + decode twins, re-binding weights), each
  with capacity buffers + a pinned scratch slab + a 64-entry graph cache → several GB. Shared weights/scratch + a global
  graph-cache cap are the levers.

**Addressed during implementation** (recorded here as the things that *did* bite): flattened-token correctness (the
q/k/v reshape — proven by the Phase-1 equivalence test); whole-model fp16 lowering (Phase-0 spike); GQA / head alignment
(tested on Qwen3); `load_weights` split ownership + tied embeddings; oracle methodology (gate on per-step logits within
tolerance, not brittle token equality); seam dtype coherence (`serve --generate` forces `--dtype float16`); RoPE not
provided by a bare `Attention` (build `get_rope` with each arch's default theta); the out-of-the-box
`max_num_batched_tokens` cap.
