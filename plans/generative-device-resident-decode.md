# Device-resident generative decode + whole-step capture — scoping

> Scoping doc for closing the remaining ~3× gap to native vLLM on generative (chat) serving, after the decode-bucket
> (~11×) and kernel-tune (~2×) wins. Grounded in the runner-level profiling below (`scripts/profile_gen_decode.py`):
> the remaining gap is **overhead-bound**, not kernel-bound. Status: **Phase A implemented + measured (served decode
> 85.5 → 178.7 tok/s, ~2×, gap to vLLM
> ~3.2× → ~1.5×); Phase B not started.** ⚠️ The 178.7 was measured **pre-merge**; a later `main` tile-lowering refactor
> regressed these decode kernels ~2.4× (current main: ~91.8 re-tuned) — see **Post-merge re-bench** below. Phase A's
> device-residency win is intact; the regression is in the kernels `main` now emits.

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

`emmy/serving/vllm_model_gen.py::EmmyGenModel.forward` (lines 120–135), per layer:

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

`emmy/serving/gen_runner.py::_Program.run` (lines 37–48) is the host path: `rebind(numpy)` → `run_once()` →
`outputs()` (`.get()` → numpy). `forward_layer_pre/post` (lines 201–222) take/return numpy and pad→run→slice on the
host.

## What already exists (reuse, don't rebuild)

The **embedding** serving path is already fully device-resident through a vLLM plugin — it is the working template:

- `emmy/serving/runner.py::EmmyForwardRunner.forward_hidden_states` (lines 204–237): enters
  `cp.cuda.Stream.from_external(torch.cuda.current_stream())`, then `set_sym_values` → `upload_prefix_device` →
  `capture_program_graph` → `replay_program_graph` → `output_prefix_device`, bridging torch↔cupy with
  `cp.from_dlpack` / `torch.from_dlpack` (no host copy).
- `CompiledProgram` already exposes every device API we need (`emmy/compiler/backend/cuda/program.py`):
  `set_sym_values` (797), `upload_prefix_device` (951), `capture_program_graph` (870, per-`sym_values` LRU cache),
  `replay_program_graph` (925), `output_prefix_device` (1089). Capture runs on a side stream; replay on the current
  stream; graphs bake grids/pointers per `sym_values`.

**Key simplification for the decode path:** the decode-bucket programs are **static M=16** → `sym_values` is empty, so
`capture_program_graph` caches exactly **one graph per program** (key `()`), no per-seq_len cache. And since
`pre`/`post` are per-token-independent, the M=16 buffer's stale padding rows are harmless (we read only the first `T`),
so device-resident "pad" is just "upload `T` rows into the prefix" — no zero-fill needed.

## Design

### Phase A — device-resident seam (captures the 40%, lower risk)

Scope: make the **decode hot path** (`T ≤ decode_bucket`, where the profiled 40% lives) device-resident. Prefill /
`T > bucket` **keeps the existing host `rebind` path** — it's one-shot per request, not the hot loop, and the symbolic
program already serves it correctly. This sidesteps the per-seq-len capture + capacity-buffer work for now.

Change the decode seam ABI from numpy to **torch CUDA tensors**, running each static M=`bucket` `pre`/`post` program via
the device path under the external-stream context. Concrete work (the items below are *not* free ports — Codex review
flagged each):

- **`_Program.run_device(arrays) -> list[torch]`** (new, gen_runner.py): mirrors the embedding runner — under
  `cp.cuda.Stream.from_external(torch.cuda.current_stream())`, `upload_prefix_device` the `T` real rows into the static
  M=bucket buffer prefix, `capture_program_graph` (cached once, empty `sym_values`), `replay_program_graph`,
  `output_prefix_device`. `cp.from_dlpack` / `torch.from_dlpack` bridge. **Slice each output to `T`** before returning
  (the static program returns M=bucket rows; `output_prefix_device` can't infer `T` — preserve the host path's `q[:t]`).
- **Device `embed`** (gen_runner.py:181,195 is a CPU numpy lookup): hold `_embed_weight` as a CUDA torch tensor and
  gather on device (`embed_weight[input_ids]`), matching `vllm_model_gen.py::embed_input_ids`. Returns `[T, H]` CUDA.
- **Device `final_norm`** (gen_runner.py:101,224 is a CPU-built torch module): move the norm module to CUDA in
  `from_model`/serving init (`.to(device)`), run it on the CUDA hidden tensor. Keep the CPU module for the oracle.
- **`vllm_model_gen.py::forward`**: branch on `T = input_ids.shape[0]`. `T ≤ bucket` → device path end-to-end (no
  `.cpu().numpy()`): q/k/v and attn_out stay CUDA tensors through RoPE + `attn[layer]`, wrapped in the external-stream
  context so emmy replays and vLLM's attention enqueue in order. `T > bucket` → the existing host path, unchanged.
- Keep all numpy methods for the `emmy generate` oracle / CPU tests.

Expected: removes the 2.72 ms host/dispatch from the decode step (6.79 → ~4.3 ms runner) and, captured-replayed, trims
per-program dispatch. Served single-stream target ~**100–120 tok/s** (gap ~2.3–2.7×). `--enforce-eager` stays.

**Result (implemented).** Served single-stream **85.5 → 178.7 tok/s (~2.09×)**, concurrency 1007 → 2367 tok/s (~2.35×),
TTFT 29 → 17 ms — gap to native vLLM **~3.2× → ~1.5×**. Beat the projection because the **captured-replay** also removed
the per-program launch dispatch (each program's ~10 kernel launches → one `graph.launch`, ~44 program-calls/step), which
the `run_once`-based profile lumped into "dispatch" but didn't eliminate. Bit-identical to the host path (`max|Δ|=0`),
and `test_vllm_plugin_gen_gpu.py` still token-for-token. Bug caught in review: `final_norm`'s `.to("cuda")` was
in-place on the **shared** norm module (broke the host path's CPU tensor) — fixed with a deep-copied device norm.
Notably this win lands **under `--enforce-eager`**, so Phase A already captured much of the dispatch headroom previously
attributed to Phase B — re-estimate B's remaining payoff (the ~5 ms vLLM-eager-framework slice) before committing to it.

### Post-merge re-bench (main's tile-lowering refactor regressed the decode kernels)

After merging `main` (the flash/monoid tile-lowering dissolution, #272–#279), a re-bench showed **two separate effects**
— Phase A's device-residency is unaffected, but the kernels `main` now emits are slower:

| runner decode step (bucket=16) | pre-merge (Phase A) | post-merge, stale prior | post-merge, **re-tuned** |
|---|---|---|---|
| G — GPU kernels | 4.06 ms | 37.69 ms | **9.86 ms** |
| W − G — host+dispatch (Phase A's lever) | 2.72 ms | 3.56 ms | 3.03 ms |
| served single-stream | 178.7 tok/s | 25.5 | **91.8 tok/s** |

1. **Stale prior (recoverable, ~74% of the drop).** The refactor renamed/restructured the kernels
   (`k_linear_mean_reduce` → `k_linear_mean_loop_reduce`), so the tuned prior's keys stopped matching → cold picks
   (G 4.06 → 37.69 ms). **Re-tuning recovers it** (`tune <post/pre>_m16.json --bench --clean` → G 9.86 ms, 25.5 → 91.8
   tok/s). The prior is per-user cache, so any deployment must re-tune after a compiler change.
2. **Genuine lowering regression (residual, ~2.4×).** Even after a clean re-tune, G is **9.86 vs 4.06 ms** — same
   shapes, same device-path code, only the compiler changed. So `main`'s refactor emits ~2.4× slower decode matmuls.
   Host/dispatch is unchanged, so **Phase A's contribution stands** (it still removes the ~3 ms host overhead); the
   regression is entirely in the kernels. Known issue — the tile-lowering owner is working on it.

### Phase B — drop `--enforce-eager` / whole-step capture (the big lever, higher risk)

Once device-resident, let vLLM's CUDA-graph machinery capture the whole `model.forward` (emmy kernels + vLLM
attention) so the ~5 ms eager framework cost disappears.

- **Pre-capture, then replay-only.** vLLM stream-captures `model.forward`; emmy must NOT do its own
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
- **Correctness:** the device path must stay token-for-token vs the numpy oracle (`emmy generate`) — the existing
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

Standalone serving with emmy's own KV cache + incremental-attention kernel (the eventual standalone-serving phase)
— that removes the vLLM interleave entirely and is a much larger effort. This doc stays within the vLLM-plugin design.
