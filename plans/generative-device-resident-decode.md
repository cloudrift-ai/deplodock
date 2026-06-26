# Device-resident generative decode + whole-step capture — scoping

> Scoping doc for closing the remaining ~3× gap to native vLLM on generative (chat) serving, after the decode-bucket
> (~11×) and kernel-tune (~2×) wins. Grounded in the profiling in
> [`generative-decode-perf-findings.md`](generative-decode-perf-findings.md): the remaining gap is **overhead-bound**,
> not kernel-bound. Status: **design only — not started.**

## Why (the measurement that motivates this)

Tuned single-stream decode is 85.5 tok/s vs vLLM's 273 (~3.2×). The runner-level profile of one tuned decode step
(`scripts/profile_gen_decode.py`, bucket-16):

| decode step (runner-level) | ms | share |
|---|---|---|
| W — wall | 6.79 | 100% |
| G — GPU kernels | 4.06 | 60% |
| W−G — host numpy↔torch + Python/dispatch | 2.72 | **40%** |

Two removable costs, both from the host-numpy seam:

1. **The 40% host/dispatch directly** — per token the plugin does **66 H2D + 22 D2H copies** (q/k/v up, attn_out down)
   plus numpy conversions and per-launch dispatch, across 22 layers.
2. **The `--enforce-eager` penalty indirectly** — the host I/O forces the plugin to run `--enforce-eager`, which
   disables vLLM's *whole-step* CUDA graph. That is the ~5 ms gap between the 6.79 ms runner step and the 11.7 ms
   **served** step (vLLM's attention + framework running eager). Device-residency is the precondition for removing it.

Note: this supersedes the pre-tune conclusion (device zero-copy "~1.3%, not worth it") recorded in the older plan — the
proportions flipped once the kernels were tuned.

## Current seam (what changes)

`deplodock/serving/vllm_model_gen.py::DeplodockGenModel.forward` (lines 120–135), per layer:

```
ids → runner.embed(numpy) → hidden_np
  for each layer:
    q_np,k_np,v_np = runner.forward_layer_pre(layer, hidden_np)      # numpy out  (implicit D2H)
    q,k,v = torch.from_numpy(...).to(device)                          # 3× H2D + numpy convert
    q,k = rotary_emb(positions, q, k)                                 # vLLM RoPE (device)
    attn_out = attn[layer](q, k, v)                                   # vLLM paged attention (device)
    hidden_np = runner.forward_layer_post(layer, attn_out.cpu().numpy(), residual_np)   # 1× D2H + numpy
  hidden_np = runner.final_norm(hidden_np)
```

`deplodock/serving/gen_runner.py::_Program.run` (lines 37–48) is the host path: `rebind(numpy)` → `run_once()` →
`outputs()` (`.get()` → numpy). `forward_layer_pre/post` (lines 201–222) take/return numpy and pad→run→slice on the
host.

## What already exists (reuse, don't rebuild)

The **embedding** serving path is already fully device-resident through a vLLM plugin — it is the working template:

- `deplodock/serving/runner.py::DeplodockForwardRunner.forward_hidden_states` (lines 204–237): enters
  `cp.cuda.Stream.from_external(torch.cuda.current_stream())`, then `set_sym_values` → `upload_prefix_device` →
  `capture_program_graph` → `replay_program_graph` → `output_prefix_device`, bridging torch↔cupy with
  `cp.from_dlpack` / `torch.from_dlpack` (no host copy).
- `CompiledProgram` already exposes every device API we need (`deplodock/compiler/backend/cuda/program.py`):
  `set_sym_values` (797), `upload_prefix_device` (951), `capture_program_graph` (870, per-`sym_values` LRU cache),
  `replay_program_graph` (925), `output_prefix_device` (1089). Capture runs on a side stream; replay on the current
  stream; graphs bake grids/pointers per `sym_values`.

**Key simplification for the decode path:** the decode-bucket programs are **static M=16** → `sym_values` is empty, so
`capture_program_graph` caches exactly **one graph per program** (key `()`), no per-seq_len cache. And since
`pre`/`post` are per-token-independent, the M=16 buffer's stale padding rows are harmless (we read only the first `T`),
so device-resident "pad" is just "upload `T` rows into the prefix" — no zero-fill needed.

## Design

### Phase A — device-resident seam (captures the 40%, lower risk)

Change the runner seam ABI from numpy to **torch CUDA tensors**, and run each `pre`/`post` decode program via the
device path under the external-stream context.

- `gen_runner.py`: add a device-resident `_Program.run_device(arrays_cupy) -> list[cupy]` mirroring the embedding
  runner (set_sym_values trivial for static, upload_prefix_device, capture_program_graph once, replay_program_graph,
  output_prefix_device). `forward_layer_pre/post`, `embed`, `final_norm` take/return torch CUDA tensors;
  `cp.from_dlpack` / `torch.from_dlpack` bridge at the boundary. Keep the numpy path for the standalone
  `deplodock generate` oracle / tests.
- `vllm_model_gen.py::forward`: drop every `.cpu().numpy()` / `torch.from_numpy().to(device)`; pass q/k/v and attn_out
  as device tensors straight through RoPE + `attn[layer]`. Wrap the per-token forward in the external-stream context so
  deplodock replays and vLLM's attention enqueue in order on `torch.cuda.current_stream()`.
- `final_norm` already a torch module → run it on CUDA (drop the numpy hop).

Expected: removes the 2.72 ms host/dispatch from the runner step (6.79 → ~4.3 ms) and, captured-replayed, trims
per-program dispatch. Served single-stream target ~**100–120 tok/s** (gap ~2.3–2.7×). `--enforce-eager` stays.

### Phase B — drop `--enforce-eager` / whole-step capture (the big lever, higher risk)

Once device-resident, let vLLM's CUDA-graph machinery capture the whole `model.forward` (deplodock kernels + vLLM
attention) so the ~5 ms eager framework cost disappears.

- **Pre-capture, then replay-only.** vLLM stream-captures `model.forward`; deplodock must NOT do its own
  begin/end-capture inside that (nested capture conflict). So capture each per-layer program's graph **once at warmup**
  (outside vLLM's capture), and have the steady-state forward issue **only** `replay_program_graph()` (a `graph.launch`,
  which is capturable as a child node).
- **Drop the forced `--enforce-eager`** in `commands/serve.py::build_serve_cmd` for the generative branch (make it
  opt-out), and align the decode-bucket M with vLLM's `cudagraph_capture_sizes` so the captured batch sizes hit the
  bucketed programs (fall back to eager/symbolic otherwise).
- Verify replay-inside-capture composes (CUDA allows launching a pre-instantiated graph during stream capture; confirm
  cupy's `graph.launch` + the external-stream binding don't trip vLLM's capture).

Expected: removes the eager framework penalty; served single-stream target ~**180–220 tok/s**, approaching vLLM.

## Risks / open questions

- **Stream/capture composition (Phase B):** the central unknown — does a cupy pre-instantiated-graph `launch` survive
  inside vLLM's stream capture, with buffers' baked pointers stable across vLLM's capture-size variants? Prototype this
  in isolation before committing to Phase B.
- **vLLM capture sizes vs decode bucket:** vLLM captures specific decode batch sizes; the static M=16 bucket must align
  or cleanly fall back. May motivate multi-bucket (8/32/…) keyed to vLLM's capture set.
- **Memory:** capturing per-layer graphs (already up to 4 programs/layer) over capacity buffers — confirm the footprint
  on top of vLLM's allocator (Top-risk #9 already bit us; we capped `--max-num-batched-tokens` to fit 16 GB).
- **Correctness:** the device path must stay token-for-token vs the numpy oracle (`deplodock generate`) — the existing
  GPU plugin tests are the gate.
- **bf16 seam:** unchanged (fp16 forced); not in scope.

## Phasing & success criteria

1. **Phase A** → verify: served single-stream ≥ ~100 tok/s (from 85.5), runner-step host/dispatch share < 10% (from
   40%) via `scripts/profile_gen_decode.py`, and `test_vllm_plugin_gen_gpu.py` still token-for-token.
   **Do this first** — most of the measurable win, reuses proven machinery, no vLLM-capture risk.
2. **Phase B** → verify: served single-stream ≥ ~180 tok/s with `--enforce-eager` dropped, plugin tests green. Gate on a
   standalone replay-inside-vLLM-capture prototype passing first.

Phase A is self-contained and shippable on its own; Phase B is a separate PR gated on the capture prototype.

## Out of scope

Standalone serving with deplodock's own KV cache + incremental-attention kernel (the eventual Phase 7 in
`generative-inference-support.md`) — that removes the vLLM interleave entirely and is a much larger effort. This doc
stays within the vLLM-plugin design.
