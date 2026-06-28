# deplodock.serving — vLLM out-of-tree embedding plugin

Serve an embedding model (Qwen3-Embedding family) with vLLM's serving shell — OpenAI `/v1/embeddings`, tokenizer,
scheduler, pooling — while the transformer trunk runs on deplodock-compiled CUDA kernels. The point is a clean A/B
inside one serving stack: stock vLLM kernels vs deplodock kernels, same API, same batching, same pooler.

```
vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --enforce-eager \
  --max-model-len 4096 --hf-overrides '{"architectures":["DeplodockEmbedModel"]}'
```

`deplodock serve` (`commands/serve.py`) wraps that boilerplate: `deplodock serve <model> [vllm flags...]`, with
`--stock` for the raw-vLLM baseline at the same max-model-len, and `--bench` for a one-shot start → `/health` →
`vllm bench serve --backend openai-embeddings` → results → shutdown cycle.

Requires the `serving` extra (`pip install -e ".[compile,serving]"` + cupy). vLLM discovers the plugin through the
`vllm.general_plugins` entry point (`deplodock.serving:register` in pyproject.toml), which registers
`DeplodockEmbedModel` by lazy string path; `--hf-overrides` swaps the served repo's `architectures` to it, so the
checkpoint, tokenizer, and sentence-transformers pooling config still come from the original HF repo.

## Module map

- `__init__.py` — `register()`, the entry-point hook. Never imports vllm/torch at module level.
- `vllm_model.py` — `DeplodockEmbedModel` (the only module importing vllm). An `nn.Module` with **no parameters**:
  `is_pooling_model = True`, `IsAttentionFree` (no vLLM `Attention` layers → V1 builds an empty KV-cache spec),
  `attn_type = "encoder_only"` (vLLM disables chunked prefill → every request reaches `forward` whole),
  `pooler = DispatchPooler.for_embedding(...)` (last-token pooling + L2 normalize + matryoshka — identical to stock
  Qwen3-Embedding serving), stub `embed_input_ids`, no-op `load_weights` (the runner loads the checkpoint itself; not
  consuming the iterator skips reading the safetensors). `forward` keeps everything **on-device**: it clamps + casts
  the packed ids to int64 on the GPU, slices each span straight into the runner (torch tensors, no `.cpu()`/numpy), and
  `torch.cat`s the torch results back. The only host touch is a small `positions.cpu()` to find span boundaries for
  `split_spans` (a `(num_tokens,)` int vector). With `batch_cap > 1` it hands all spans to
  `runner.forward_hidden_states_batched` (one padded batched forward) instead of looping per span.
- `runner.py` — `DeplodockForwardRunner`. At engine start: load the `AutoModel` **trunk** (hidden states out — no
  lm_head), `build_full_model_wrapper(dynamic=True)`, trace with the canonical 4-spec dynamic seq_len
  (`seq_len@input_ids:1`, `@attention_mask:2`, `@attention_mask:3`, `@position_ids:1`), compile through `CudaBackend`
  (greedy fork picks from the global prior — benefits from any prior `deplodock tune`), bind weights as graph
  constants (`named_parameters` + `named_buffers`, `remove_duplicate=False`, in the traced dtype), and build ONE
  `CompiledProgram` over a buffer set sized at **`max_seq_len`** (`--max-model-len`). Per
  sequence (`forward_hidden_states`) it takes a **1-D int torch CUDA tensor** and returns an `(S, hidden)` **torch CUDA
  tensor** — no host round-trip. It enters a cupy external stream bound to torch's current stream
  (`cp.cuda.Stream.from_external`), then: bridge the
  torch ids to cupy (`cp.from_dlpack`, zero-copy), size the launch grids to S (`set_sym_values`), copy ids /
  **device-built** causal mask / position_ids into the buffers' contiguous **prefix** device-to-device
  (`upload_prefix_device`), **capture-or-reuse** the whole-program CUDA graph for this S, **replay** it — one host launch
  instead of the ~hundreds the uncaptured loop issues — and wrap the output buffer's real-S prefix back as a torch
  tensor (`output_prefix_device` + `torch.from_dlpack`, cloned because the shared buffer is reused next request). The
  causal mask is built once per S as a cupy array (the device twin of `_causal_mask_np`) and reused. Captured graphs are
  cached per seq_len (bounded LRU);
  each is captured at its EXACT S so every kernel runs at its exact grid — no oversized-grid masking (a single
  capacity-baked graph for all S is **not** viable: several symbolic-M kernels do illegal reads at an oversized grid,
  the swizzle decode + staged loads among them). See `compiler/backend/cuda/ARCHITECTURE.md`
  → repeated execution + captured replay. Trunk compute dtype follows vLLM's `--dtype` (`mc.dtype`, mapped in
  `vllm_model._trunk_dtype_str`): `float32`→fp32, `float16`→fp16, anything else (e.g. `bfloat16`/`auto`) downcasts to
  fp16 with a warn — the runner's numpy weight carrier can't represent bf16, and only fp16/fp32 trunks are supported.
  When `DEPLODOCK_SERVING_STATIC=1` (`config.serving_static`), `create` instead traces a **fully-static**
  `(max_num_seqs, max_seq_len)` graph (no dynamic_shapes — static extents for both batch and seq) and
  `forward_hidden_states_batched` runs each step as one padded batched forward — see "Static mode" below.
- `packed.py` — `split_spans(positions, max_seq_len)`: vLLM V1 hands pooling models one packed `(num_tokens,)` tensor
  with per-request 0-based positions; spans split at `positions == 0`. Hardened for `_dummy_run`'s garbage profiling
  batches (index 0 always opens a span; overlong spans are chopped).
- `sampling.py` — **no vLLM, no CUDA**. Pure-numpy token sampling (`Sampler`: greedy / temperature / top-k / top-p) +
  `apply_chat_template` (delegates to the HF tokenizer). Used by the standalone **generation oracle**
  (`commands/generate.py`) — `deplodock generate`'s host loop re-runs the whole fp16 prefix each step on the CUDA
  backend and samples with this. The generative *vLLM plugin* (`DeplodockGenModel`) builds on this oracle.
- `gen_runner.py` — `DeplodockGenRunner` (Phase 2; sibling to `DeplodockForwardRunner`). Carves SDPA out of every
  decoder layer (`build_attention_split_wrapper`), compiles **two dynamic-`num_tokens` programs per layer** (`pre` +
  `post`) over the flattened `[num_tokens, H]` layout, and exposes `embed` / `forward_layer_pre(L,…)→(q,k,v)`
  (un-rotated 2-D seam) / `forward_layer_post(L, attn_out, residual)→hidden` / `final_norm`. The caller stitches between
  `pre` and `post` (a reference torch SDPA in the Phase-2 host stitch; vLLM paged `Attention` in Phase 3). **I/O:**
  prefill / `num_tokens > bucket` use the host numpy `rebind` path; the **decode hot path** (`num_tokens ≤ bucket`)
  is **device-resident** (Phase A — `run_device` / `embed_device` / `forward_layer_*_device` / `final_norm_device`:
  captured-replay over the static program with torch↔cupy DLPack zero-copy, no host hop; reuses the embedding
  runner's pattern). This removed the ~40% host/dispatch overhead and ~2×'d served decode. **Decode bucket:** it
  also compiles a **static M=`decode_bucket` (default 16)** `pre`/`post` twin per layer and uses it when
  `num_tokens ≤ bucket` (pad → run → slice the real rows) — the symbolic hint-512 M-tile is ~66× too slow at decode
  M=1; falls back to symbolic above the bucket or if a static compile fails. So
  up to 4 capacity programs/layer — a real memory-budget risk.
- `vllm_model_gen.py` — `DeplodockGenModel` (Phase 3; the generative vLLM model class). **NOT** `IsAttentionFree`: it
  builds real vLLM `Attention` layers (one per decoder layer, unique `prefix` → vLLM allocates a KV-cache spec and runs
  paged attention) + a shared `get_rope` module (a bare `Attention` does no RoPE) + `ParallelLMHead` + `LogitsProcessor`.
  The trunk compute (embed + per-layer pre/post + final norm) is the `DeplodockGenRunner`; vLLM owns only `lm_head`
  (`load_weights` claims `lm_head.weight`, or the tied embed alias). `forward` brackets each `self.attn[L](q,k,v)` with
  two deplodock replays (pre/post), applying RoPE in between (A2). `forward` branches on `num_tokens`: the decode hot
  path (`≤ bucket`) runs `_forward_device` (q/k/v + attn_out stay CUDA tensors through RoPE + attention, no host
  hop); prefill keeps the numpy path. Select via `--runner generate` +
  `--hf-overrides '{"architectures":["DeplodockGenModel"]}'` + `--dtype float16` (the `serve --generate` branch forces
  this for seam coherence). Registered in `__init__.py`. Whole-step CUDA-graph capture (drop `--enforce-eager`) is
  future work.

## Static mode (`DEPLODOCK_SERVING_STATIC=1`) — static extents for both batch and seq

The default path is symbolic-seq, **one sequence per forward** (`batch_cap = 1`). Setting `DEPLODOCK_SERVING_STATIC=1`
switches the runner to a **fully-static `(N, max_seq_len)` program** (static extents for both batch and seq) so a whole
scheduler step runs as one batched forward. The batch `N` is **vLLM's own `max_num_seqs`**
(`vllm_config.scheduler_config.max_num_seqs`), read at init — so the batch is sized by the standard `--max-num-seqs`
flag, not a separate deplodock knob (the toggle is boolean; the size comes from what vLLM hands us). Mind the default
`max_num_seqs=256`: pair the opt-in with a sane `--max-num-seqs` or the static `(256, max_seq_len)` program will be huge.

**Measured (RTX 5090, Qwen3-Embedding-0.6B, uniform 512 tokens, concurrency 32):** this is currently a throughput
**regression**, not a win — the static `(32, 512)` forward takes ~1.1 s (26.6 req/s) vs the batch-1 path's 64 req/s and
stock vLLM's ~232 req/s. Making the extents static is *not* the lever: the batched program does B× the existing,
non-flash, O(B·H·S²)-materialized attention,
and the static-shape kernels are cold greedy picks (untuned, unlike the batch-1 symbolic kernels which carried prior
tuning). The throughput win needs efficient attention (flash/varlen, follow-up #1) and/or an autotune of the static
batched shape — batching alone, over today's kernels, loses. Why static, not symbolic: the dynamic-seq
(masked-tile) kernels **miscompute batch>1** (the batch axis isn't threaded through the masked attention/reduction
codegen — empirically every batch row is wrong), whereas a fully-static trace bakes correct batch strides. So the
batched program fixes both axes: every step is **padded to `(N, max_seq_len)`** — short sequences pad on the right,
short batches pad with dummy rows (`runner.forward_hidden_states_batched`). Causal masking makes this safe: a token
attends only to earlier positions, so a row's real prefix is independent of its right-padding, and dummy rows are never
read out. The mask + position_ids are request-independent (full causal at `max_seq_len`, arange positions) so they are
fed once at build and never updated — only ids change per step. **Limitation:** correct only for the static shape, so
it pads every sequence to `max_seq_len` (zero waste when all requests are that length — the benchmark case — but
wasteful for mixed/short lengths); set `--max-model-len` to the actual workload length. Not a general production path;
the general fix is follow-up #1.

## Execution model (v2: captured graphs) and its known costs

Each sequence runs **individually** through the compiled dynamic-seq_len program (batch axis is compile-time fixed at
1), as a captured whole-program CUDA graph (one host launch) replayed over a single capacity-sized buffer set, with the
request's torch inputs bridged to the buffers' prefix device-to-device (dlpack, no host hop) and the output handed back
as a torch view of the output buffer. One captured graph is cached per
distinct seq_len (bounded LRU); a new length pays one capture (~one forward) on first sight, then replays. The captured
graph removes the per-launch dispatch overhead and the ~hundreds of host calls the uncaptured loop made — the
precondition for fast low-concurrency serving. (Measured A/B is ~flat today: the uncaptured `run_once` loop already
queues launches async, so the Python dispatch overlaps GPU execution and stays hidden while the symbolic kernels are
slow — 0.6B at S=32 ≈ 32 ms GPU. The win materializes once those kernels are fast enough that dispatch stops hiding;
the captured path is in place for that.) Low-concurrency latency is representative; high-concurrency throughput
structurally trails stock vLLM's packed-batch prefill — that gap measures the integration, not kernel quality.
Recorded follow-ups, in impact order:

1. **Packed-varlen attention** (cu_seqlens-aware SDPA tiles) — run vLLM's whole packed batch in one launch at *mixed*
   lengths; the general form of the throughput fix. At concurrency 1 (no batching) deplodock is ~1.5× stock; the rest of
   the concurrency-32 gap is batching. The static batched mode above is the fixed-length stand-in; the remaining work is
   (a) making the masked-tile kernels batch-correct so a symbolic-seq batched program works, then (b) cu_seqlens varlen
   so one launch handles mixed lengths without padding to `max_seq_len`.
2. **dlpack zero-copy I/O** — **done**: `forward_hidden_states` takes/returns torch CUDA tensors, bridged to the cupy
   buffers via `cp.from_dlpack` / `torch.from_dlpack` on torch's stream — no GPU↔host round-trip (`upload_prefix_device`
   / `output_prefix_device`). The only residual host touch is `positions.cpu()` for span boundaries.
3. **Device-side causal mask** — host build + upload **removed**: the `(1,1,S,S)` mask is now built once per S as a cupy
   array on the GPU (`runner._mask`) and copied into the prefix device-to-device. Still open: an in-kernel `j <= i`
   predicate would drop the mask input + its per-request D2D copy entirely.
4. **Single capacity-baked graph** — would collapse the per-S cache to one graph, but needs every symbolic-M kernel to
   be correct at an oversized (capacity) grid; today several aren't (swizzle decode + staged loads read OOB). Future
   work.

## Serving constraints

- `--max-model-len` ≤ `DYNAMIC_DIM_MAX` (4096, `compiler/trace/dynamic.py`) — the runner raises at startup otherwise.
  Qwen3-Embedding natively supports 32k; raising the cap means re-examining the `torch.export.Dim` bounds and the
  rotary buffer (`_SlicedRotary` precomputes `DYNAMIC_DIM_MAX + 1` positions).
- `--enforce-eager`: vLLM never torch.compiles/cudagraph-captures an undecorated OOT class, but enforce-eager makes
  the whole engine eager so the runner's own kernel launches can't race a capture.
- Startup compiles the whole model (~1–2 min for 0.6B warm-cubin-cache; first boot pays nvcc). `DEPLODOCK_CUBIN_CACHE`
  persistence across container restarts is what keeps reboots fast.
- The shared buffer set is allocated at `max_seq_len` (`--max-model-len`); every accepted request (S ≤ `max_seq_len`)
  uses the captured-graph path. The S²-attention scratch dominates that allocation (0.6B at 4096 ≈ 15 GB), so lower
  `--max-model-len` for bigger models / smaller cards.
- vLLM's memory profiler only sees torch allocations; the runner's cupy-held weights/activations are invisible to it.
  Leave `--gpu-memory-utilization` headroom accordingly (the attention-free model needs no KV cache, so vLLM's own
  budget is tiny).

## Testing

- `tests/serving/test_packed.py` — pure span-split logic, runs everywhere.
- `tests/serving/test_vllm_plugin_gpu.py` — `perf`-marked (deselected by default), needs CUDA + vllm: in-process
  `vllm.LLM(runner="pooling", hf_overrides=...)` on Qwen3-Embedding-0.6B, `.embed()` cosine vs the HF eager reference.
  The three texts have different token counts, so it exercises the per-seq_len captured-graph cache end to end.
- `tests/compiler/ir/test_dynamic_shapes.py` — the captured-replay primitives directly (RMSNorm + a 1-layer Qwen3
  trunk through `set_sym_values` + `upload_prefix` + `capture_program_graph` + `replay_program_graph` +
  `outputs(sym_values)`); run under `compute-sanitizer` in dev to confirm zero illegal accesses. Plus
  `test_capture_replay_device_io_matches_eager` — the zero-copy device path (`upload_prefix_device` + cupy-in,
  `output_prefix_device` + `torch.from_dlpack`-out) matches eager, the primitive behind the runner's torch I/O.
- `tests/serving/test_runner_batched_gpu.py` — `perf`-marked: a 1-layer static `(batch, S)` trunk wrapped in a runner;
  `forward_hidden_states_batched` runs several different-length sequences in one padded batched forward and matches
  eager per row (the causal-independence-under-padding gate for `DEPLODOCK_SERVING_STATIC`).
- `scripts/compare_embeddings.py` — the accuracy gate against a *server*: embeds a fixed text set through two
  OpenAI-compatible endpoints (deplodock-backed and stock) and asserts pairwise cosine > 0.99.
