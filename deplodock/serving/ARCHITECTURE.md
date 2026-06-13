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
  consuming the iterator skips reading the safetensors).
- `runner.py` — `DeplodockForwardRunner`. At engine start: load the `AutoModel` **trunk** (hidden states out — no
  lm_head), `build_full_model_wrapper(dynamic=True)`, trace with the canonical 4-spec dynamic seq_len
  (`seq_len@input_ids:1`, `@attention_mask:2`, `@attention_mask:3`, `@position_ids:1`), compile through `CudaBackend`
  (greedy fork picks from the global prior — benefits from any prior `deplodock tune`), bind weights as graph
  constants (`named_parameters` + `named_buffers`, `remove_duplicate=False`, in the traced dtype), and build ONE
  `CompiledProgram` over a **capacity-sized** buffer set (`min(max_seq_len, DEPLODOCK_SERVING_CAPTURE_CAP)`). Per
  sequence (`forward_hidden_states`): size the launch grids to S (`set_sym_values`), upload the request's ids / causal
  mask / position_ids into the buffers' contiguous **prefix** (`upload_prefix`), **capture-or-reuse** the whole-program
  CUDA graph for this S, and **replay** it — one host launch instead of the ~hundreds the uncaptured loop issues; then
  `outputs({"seq_len": S})` slices each output to its real shape. Captured graphs are cached per seq_len (bounded LRU);
  each is captured at its EXACT S so every kernel runs at its exact grid — no oversized-grid masking (a single
  capacity-baked graph for all S is **not** viable: several symbolic-M kernels do illegal reads at an oversized grid,
  the swizzle decode + staged loads among them). `DEPLODOCK_SERVING_NO_GRAPHS=1` keeps the uncaptured `rebind` →
  `run_once` path; requests above the capture capacity fall back to it too. See `compiler/backend/cuda/ARCHITECTURE.md`
  → repeated execution + captured replay. Trace dtype: `DEPLODOCK_SERVING_DTYPE` (default `float16`; `float32` is the
  accuracy escape hatch — 2× weight memory).
- `packed.py` — `split_spans(positions, max_seq_len)`: vLLM V1 hands pooling models one packed `(num_tokens,)` tensor
  with per-request 0-based positions; spans split at `positions == 0`. Hardened for `_dummy_run`'s garbage profiling
  batches (index 0 always opens a span; overlong spans are chopped).

## Execution model (v2: captured graphs) and its known costs

Each sequence runs **individually** through the compiled dynamic-seq_len program (batch axis is compile-time fixed at
1), as a captured whole-program CUDA graph (one host launch) replayed over a single capacity-sized buffer set, with the
request's inputs uploaded into the buffers' prefix and the output sliced back out. One captured graph is cached per
distinct seq_len (bounded LRU); a new length pays one capture (~one forward) on first sight, then replays. The captured
graph removes the per-launch dispatch overhead and the ~hundreds of host calls the uncaptured loop made — the
precondition for fast low-concurrency serving. (Measured A/B is ~flat today: the uncaptured `run_once` loop already
queues launches async, so the Python dispatch overlaps GPU execution and stays hidden while the symbolic kernels are
slow — 0.6B at S=32 ≈ 32 ms GPU. The win materializes once those kernels are fast enough that dispatch stops hiding;
the captured path is in place for that.) Low-concurrency latency is representative; high-concurrency throughput
structurally trails stock vLLM's packed-batch prefill — that gap measures the integration, not kernel quality.
Recorded follow-ups, in impact order:

1. **Packed-varlen attention** (cu_seqlens-aware SDPA tiles) — run vLLM's whole packed batch in one launch; the only
   item that makes the throughput regime apples-to-apples. Matmuls/norms/pointwise are already token-wise.
2. **dlpack zero-copy I/O** — cupy ↔ torch without the host round-trip (`cp.from_dlpack` / `torch.from_dlpack`).
3. **Device-side causal mask** — the `(1,1,S,S)` host mask upload (~32 MB fp16 at S=4096) per request; a tiny kernel
   (or in-kernel `j <= i` predicate) removes the mask input + its upload entirely.
4. **Single capacity-baked graph** — would collapse the per-S cache to one graph, but needs every symbolic-M kernel to
   be correct at an oversized (capacity) grid; today several aren't (swizzle decode + staged loads read OOB). Tracked
   in `plans/serving-dynamic-shape-cuda-graphs.md`.

## Serving constraints

- `--max-model-len` ≤ `DYNAMIC_DIM_MAX` (4096, `compiler/trace/dynamic.py`) — the runner raises at startup otherwise.
  Qwen3-Embedding natively supports 32k; raising the cap means re-examining the `torch.export.Dim` bounds and the
  rotary buffer (`_SlicedRotary` precomputes `DYNAMIC_DIM_MAX + 1` positions).
- `--enforce-eager`: vLLM never torch.compiles/cudagraph-captures an undecorated OOT class, but enforce-eager makes
  the whole engine eager so the runner's own kernel launches can't race a capture.
- Startup compiles the whole model (~1–2 min for 0.6B warm-cubin-cache; first boot pays nvcc). `DEPLODOCK_CUBIN_CACHE`
  persistence across container restarts is what keeps reboots fast.
- `DEPLODOCK_SERVING_CAPTURE_CAP` caps the seq_len the shared buffer set is allocated at (default `max_seq_len`). The
  S²-attention scratch dominates that allocation (0.6B at 4096 ≈ 15 GB), so lower the cap for bigger models / smaller
  cards — requests above it fall back to the uncaptured rebind path (correct, just slower).
  `DEPLODOCK_SERVING_NO_GRAPHS=1` forces that uncaptured path for every request (debug / A-B).
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
  `outputs(sym_values)`); run under `compute-sanitizer` in dev to confirm zero illegal accesses.
- `scripts/compare_embeddings.py` — the accuracy gate against a *server*: embeds a fixed text set through two
  OpenAI-compatible endpoints (deplodock-backed and stock) and asserts pairwise cosine > 0.99.
