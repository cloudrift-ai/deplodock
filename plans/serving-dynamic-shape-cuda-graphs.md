# Dynamic-shape CUDA graphs for the serving path

## Status: shipped as a per-seq_len captured-graph cache (not capacity-baking)

The original design below (one capacity-baked graph for all seq_lens, device-resident symbolic args) was **abandoned
during implementation** because its load-bearing premise — "oversized grids are guard-safe for every symbolic lowering
variant" — is empirically false. The Phase-2 multi-S replay test (the gate the plan itself named for Risk #1) caught it:
at a capacity-baked grid with a runtime `seq_len < capacity`, multiple symbolic-M kernel classes do illegal global
reads. Confirmed on the 1-layer Qwen3-Embedding-0.6B trunk under `compute-sanitizer`:

- **CTA-swizzle exact-cover kernels** (`k_linear_sdpa_reduce`, grid `(seq_len, 128)`): the swizzle decode reconstructs
  `num_m = seq_len` (runtime) while the grid is baked at capacity, so the N-index `a1 = (bid % gsz) / gsize_m` divides
  by a shrunk `gsize_m` and explodes → OOB read of the weight.
- **ceil-div masked-M staged-load kernels** (`k_linear_reduce`, grid `(2, ceil_div(seq_len,32), 64)`): after disabling
  swizzle, a vectorized (16-byte) staged global load over-reads at the oversized grid even with the output guard.

Making *every* symbolic-M kernel correct at an oversized grid (capacity-consistent swizzle decode + guards on all staged
loads, audited across all classes) is open-ended compiler work across several lowering passes. **User decision (this
session): ship the per-seq_len captured-graph cache instead** — it reaches the same real goal (captured serving, exact
per-S compute, no padding, one shared buffer set) with zero compiler changes and is `compute-sanitizer`-clean.

## What shipped

**One captured graph per distinct seq_len, over a single capacity-sized buffer set.** Each graph is captured at its
EXACT S, so every kernel runs at its exact grid — the oversized-grid problem never arises. Buffers are allocated once at
capacity and each request's inputs upload into their contiguous prefix (prefix occupancy: a logically `(1,S,…)` tensor
is the first `S·…` elements of the `(1,S_cap,…)` allocation; verified for ids/position_ids and, since the attention
kernels stride the mask by the per-graph `seq_len`, for the `(1,1,S,S)` causal mask too). A repeated length replays with
no re-capture; a new length pays one capture (~one forward). Per-request host calls drop from ~hundreds to single
digits.

`CompiledProgram` (`backend/cuda/program.py`) gained:
- `set_sym_values(values)` — set the host symbolic values that resolve launch grids + by-value `seq_len`, WITHOUT
  re-allocating buffers (errors if the value exceeds capacity).
- `capture_program_graph()` — now captures at the current `self.sym_values` and **caches by the resolved symbolic
  tuple** (`_graph_cache`, bounded LRU `_graph_cache_max=64`); the static-shape bench path keeps using the single `()`
  key, unchanged. `rebind` clears the cache on re-allocation.
- `replay_program_graph()` — one `self._e2e_graph.launch()` on the default stream.
- `upload_prefix(input_data)` — H2D each host array into its capacity buffer's contiguous prefix (no realloc → captured
  pointers stay valid).
- `outputs(sym_values=None)` — with the override, slice each output to its real-S prefix before `.get()`.

Serving runner (`serving/runner.py`):
- `create()` builds the program over a capacity-sized buffer set
  (`capacity = min(max_seq_len, DEPLODOCK_SERVING_CAPTURE_CAP)`).
- `forward_hidden_states(token_ids)`: `set_sym_values({"seq_len": S})` → `upload_prefix({ids, host causal mask,
  position_ids})` → `capture_program_graph()` (cached) → `replay_program_graph()` → `outputs({"seq_len": S})[name][0]`.
  Falls back to the uncaptured `rebind` + `run_once` path when `DEPLODOCK_SERVING_NO_GRAPHS=1` or `S > capacity`.

Config (`config.py`): `DEPLODOCK_SERVING_CAPTURE_CAP` (buffer-allocation cap, default `max_seq_len`) and
`DEPLODOCK_SERVING_NO_GRAPHS` (kill switch). No codegen flag, no device-resident symbolic args, no compiler-pass changes
(the Phase-1 device-sym-args codegen was prototyped, proven unnecessary under per-S capture, and reverted).

Tests: `tests/compiler/ir/test_dynamic_shapes.py` — RMSNorm at capacity 64 and a 1-layer Qwen3 trunk, each served at
several seq_lens through the cache (capture → replay → slice) vs torch eager, asserting one cached graph per distinct S;
`compute-sanitizer`-clean. The existing `tests/serving/test_vllm_plugin_gpu.py` (3 texts of differing lengths) exercises
the cache end-to-end through vLLM.

Docs: `compiler/backend/cuda/ARCHITECTURE.md` (captured-graph-replay section), `serving/ARCHITECTURE.md` (execution
model v2 + capacity/cap env + follow-ups), `CLAUDE.md` env-var list.

## Costs / follow-ups

1. **Host causal-mask upload** per request (`(1,1,S,S)`, ~32 MB fp16 at S=4096) — a device-side mask-gen kernel (or an
   in-kernel `j <= i` predicate) removes the mask input + upload. Already listed in `serving/ARCHITECTURE.md`.
2. **Capacity memory** — the S² attention scratch at the default cap; `DEPLODOCK_SERVING_CAPTURE_CAP` lowers it, with
   graceful fallback above the cap.
3. **Many distinct lengths** — N graphs (cheap: launch nodes only, buffers shared) + a re-capture per new length, LRU-
   bounded. Worst case (all-unique lengths) degrades to ~the uncaptured cost on each first sight.
4. **Single capacity-baked graph** — the original ambition; needs every symbolic-M kernel oversized-grid-safe (see the
   `compute-sanitizer` findings above). Deferred.

---

## Original design (NOT shipped — kept for context)

The text below is the pre-implementation plan for the single capacity-baked graph + device-resident symbolic args. It is
retained only to record the rejected approach and why; see "Status" above for what actually landed.

**User decision: no bucketing/padding — true dynamic shapes.** One captured graph must serve any seq_len:

- **Grids bake at a capacity size** (`ceil-div(S_cap)`), masked-tile guards deactivate excess blocks. *(False premise —
  see Status: several symbolic-M kernels are not oversized-grid-safe.)*
- **`seq_len` moves to device memory** (`const int* sym_vals`), so the captured graph's content changes per request.
  *(Prototyped + reverted: with a per-S graph each kernel bakes its own by-value `seq_len`, so device-resident args add
  nothing.)*
- **Prefix-occupancy** for shared capacity buffers, with the 2-D causal mask generated on-device. *(Prefix occupancy
  kept; on-device mask-gen deferred to follow-up #1 — the host upload is correct and simpler for the first cut.)*
- TMA is rejected on symbolic-dim graphs (`lowering/tile/050_use_tma.py:152-157`), so it was never a concern.
