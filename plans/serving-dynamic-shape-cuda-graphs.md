# Dynamic-shape CUDA graphs for the serving path

## Context

Serving (deplodock/serving) executes the dynamic-seq_len compiled whole model per request via `rebind()` +
`run_once()` — a 337-launch Python loop plus numpy host I/O. Measured this session at S=32: 13.4 ms GPU (the
symbolic-kernel quality issue, tracked separately) + ~0.5 ms dispatch + ~0.7 ms host I/O. Once the symbolic kernels
are fixed (the static graph runs 1.98 ms), the uncaptured loop becomes a ~15 ms wall (337 × ~30 µs dispatch can no
longer hide behind slow kernels) — CUDA graphs are a precondition for fast serving.

**User decision: no bucketing/padding — true dynamic shapes.** One captured graph must serve any seq_len. The design
that achieves this (validated against the codebase this session):

- **Grids bake at a capacity size** (`ceil-div(S_cap)`), and the existing masked-tile guards (`if (coord < seq_len)`)
  deactivate excess blocks at runtime. Guard audit says oversized grids are safe for every symbolic lowering variant:
  masked tiles guard symbolic output axes (`lowering/tile/010_partition_loops.py:983-997`), hoisted staged loads are
  Cond-wrapped (`021_hoist_staged_loads_above_mask.py`), split-K atomics are row-disjoint with zero-fill
  (`lowering/cuda/010_lower_kernelop.py:57-79`), and no address math uses gridDim. Garbage in rows ≥ real S stays
  within capacity allocations and is sliced off.
- **`seq_len` moves from a by-value `int` kernel arg (frozen by capture) to device memory**: kernels take
  `const int* sym_vals` and load `const int seq_len = sym_vals[i];` in a prologue. The pointer is baked into the
  graph; the *content* changes per request. Serial loop bounds, masked guards, and S-dependent address math
  (`row * (seq_len*128)`) all use the loaded value, so one graph computes the exact real-S result — numerics identical
  to today's uncaptured path, no padding compute.
- **Prefix-occupancy** (verified: inner strides are static, only the symbolic axis multiplies the runtime value): a
  logically `(1,S,16,128)` tensor occupies the first `S*16*128` elements of a capacity allocation, so all replays
  share ONE capacity-sized buffer set. Caveat found in review: this holds for buffers *written at the current S*, and
  for 1-D-symbolic inputs (`input_ids`, `position_ids` — `arange(S_cap)` prefix ≡ `arange(S)`), but NOT for the 2-D
  causal mask (layout `i*S + j` depends on S) — the mask is therefore **generated on-device inside the captured
  graph** by a small helper kernel, which also deletes its 32 MB-at-4096 host upload.
- TMA is a non-issue: the lowering rejects TMA on any graph with symbolic dims (`lowering/tile/050_use_tma.py:151`).

Per-request hot path becomes: write real S into the device sym buffer (tiny H2D) → upload ids prefix → `graph.launch()`
→ prefix D2H of hidden states. Single-digit host calls instead of ~350.

## Phase 1 — codegen: device-resident symbolic args (opt-in)

**Flag carrier.** A compile-mode flag `sym_args_device` reaching the CUDA lowering. Preferred: a field on `Context`
(`compiler/context.py:91`), threaded from `CudaBackend` (check where `Pipeline.run` builds the Context; if the backend
can't inject Context fields cleanly, fall back to a non-tunable env knob via `deplodock/config.py` — it must NOT enter
the tune fork space). Kernel source changes under the flag, so cubin/perf cache keys split naturally; tune/run keep
the by-value path, zero behavior change with the flag off.

**Lowering** (`pipeline/passes/lowering/cuda/010_lower_kernelop.py`, `_launch_geometry` at :83):
- `CudaOp` (`ir/cuda/ir.py`) gains `sym_args_device: bool = False` (and keeps `runtime_args` name order).
- Global index per symbolic name = position in `sorted(all symbolic axis names of the graph)` (one name — `seq_len` —
  in serving, but keep general). Stamp the per-kernel `runtime_args` with their global indices (e.g. a parallel
  `runtime_arg_idx: tuple[int, ...]` field).

**Render** (`ir/kernel/render.py:311`): under the flag, replace `int <name>` params with one `const int* sym_vals`
param + prologue `const int <name> = sym_vals[<global idx>];` per runtime arg.

**Launch packing** (`backend/cuda/program.py::_launch` :344-348): when `launch.sym_args_device`, append the program's
shared device sym array (stable pointer) instead of by-value ints. `CompiledProgram.build` allocates
`self._sym_dev: cp.ndarray int32` sized to the global name list and writes the build-time `sym_values`; new method
`set_sym_values(dict)` updates its contents (tiny H2D, never re-allocates). The uncaptured paths (`run_once`,
`iter_once`) work unchanged in this mode — they just read the device value the caller set.

Tests (CUDA): compile the small dynamic elementwise/RMSNorm graphs from `tests/compiler/ir/test_dynamic_shapes.py`
with the flag — assert `const int* sym_vals` in the kernel source, run at multiple S via `set_sym_values` +
`run_once`, match numpy; flag-off graphs byte-identical to today.

## Phase 2 — CompiledProgram: capture/replay at capacity

(`backend/cuda/program.py`, beside the existing `capture_program_graph` — reuse its side-stream + drain-on-error +
throwaway-instantiation pattern.)

- `capture_program_graph(sym_values=None, pre_launch=None)`: extend the existing method — optional `sym_values`
  override resolves grids at capacity for the capture (host-side `resolve_dim`, verified deterministic, no host
  branching inside the window); optional `pre_launch(stream)` callable captured FIRST (the runner's mask-gen kernel).
  Reject (clear error) if any `buf.resolve_shape(sym_values)` exceeds the allocated array sizes.
- `replay_program_graph()`: `self._e2e_graph.launch()` (the bench-only `time_program_window` keeps its own path).
- `upload_prefix(input_data)`: H2D into each buffer's contiguous prefix (`arr.ravel()[:n].set(host)`) — no realloc, no
  graph invalidation; error if larger than capacity.
- `outputs(sym_values=None)`: with the override, slice each output to `buf.resolve_shape(sym_values)` prefix
  (flat-slice then reshape) before `.get()`.
- `rebind` already drops `_e2e_graph` on realloc — that is the capacity-growth path (grow → lazy re-capture); keep.
- Note in docstrings: zero-fill of `zero_outputs` is captured as a full-capacity memset per replay — correct (split-K
  atomics need it), mildly wasteful, acceptable.

Tests (CUDA): build the flag-compiled RMSNorm graph at capacity 64; capture once; for S ∈ {5, 12, 33, 64}:
`set_sym_values` + `upload_prefix` + `replay_program_graph` + `outputs({"seq_len": S})` vs numpy. Then a 1-layer
random-weight Qwen3 trunk (mirror `test_qwen_whole_model_dynamic_compiles_and_matches_eager`) through the same
captured path — validates the attention/mask/oversized-grid story end to end with non-zero ids.

## Phase 3 — serving runner integration

(`deplodock/serving/runner.py`)

- `create()` compiles with `sym_args_device=True` and builds the program at **capacity** =
  `min(max_seq_len, DEPLODOCK_SERVING_CAPTURE_CAP)` (new env via `config.py`, default = max_seq_len). Memory note: the
  S² attention-score scratch dominates (0.6B at 4096 ≈ 15 GB — fits the 32 GB 5090; document lowering the cap for
  bigger models/smaller cards). Requests above capacity (only possible when the cap was lowered) fall back to the
  existing `rebind` + `run_once` path — which drops the graph; re-capture lazily on the next in-cap request.
- **Device mask-gen kernel**: a small cupy RawKernel owned by the runner (`mask[i*S + j] = (j > i) ? -inf : 0` for
  `i,j < S`, grid at capacity ceil-div, guarded, dtype-matched) — captured via `pre_launch` so every replay rebuilds
  the mask at the current S. `position_ids` uploads once at capacity (`arange` prefix property); the per-request host
  traffic is ids in (≤ 32 KB) and hidden-states prefix out.
- `forward_hidden_states`: `set_sym_values({"seq_len": S})` → `upload_prefix({ids})` → `replay_program_graph()` →
  `outputs({"seq_len": S})[0]`. Kill switch `DEPLODOCK_SERVING_NO_GRAPHS=1` keeps the current rebind+run_once path for
  debugging (also exercised by the fallback test).
- `vllm_model.py` unchanged.

## Phase 4 — validation, docs

- GPU integration test (perf-marked, beside `tests/serving/test_vllm_plugin_gpu.py`): in-process `vllm.LLM` embed with
  texts of widely differing lengths, cosine > 0.99 vs HF eager — exercises mixed seq_lens through one captured graph.
- Live A/B on this box (RTX 5090): `deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --max-concurrency 1` before vs
  after — expect ~14.6 → ~13.7 ms now (graphs remove dispatch+I/O; GPU time unchanged until the symbolic-kernel fix),
  and re-run the direct probe (serving path vs captured window) to confirm the shim is gone (<0.2 ms gap).
- Accuracy gate: `scripts/compare_embeddings.py` vs stock vLLM unchanged (> 0.99).
- Docs: `compiler/backend/cuda/ARCHITECTURE.md` (device-sym-args mode + capture-at-capacity semantics),
  `serving/ARCHITECTURE.md` (execution model rewrite: graph replay, mask-gen kernel, capacity/cap env, updated
  follow-ups list), `CLAUDE.md` env-var list (`DEPLODOCK_SERVING_CAPTURE_CAP`, `DEPLODOCK_SERVING_NO_GRAPHS`),
  `tests/ARCHITECTURE.md`.
- `make test`, `make lint`; commit on the existing `feature/vllm-embedding-plugin` branch (or a follow-up branch off
  it if the PR should stay reviewable — decide at commit time based on PR review state).

## Risks

1. **Guard coverage** — the audit says every symbolic lowering variant is oversized-grid-safe, but the multi-S
   replay tests (Phase 2/4) are the real gate; any unguarded kernel shows up as a numeric mismatch at small S.
2. **Capacity memory** — S² scratch at the default cap; documented env to lower it, with graceful fallback.
3. **First-request capture latency** — one capture + instantiation (~one forward) on the first request; acceptable,
   noted in docs (could pre-capture at startup later).
4. **Pointer-load overhead** — one extra global load per kernel; unmeasurable against µs-scale kernels.
5. **GPU time unchanged for now** — 13.4 ms of degenerate symbolic kernels remains; this work removes the shim and
   unblocks the kernel fix from being masked by dispatch. Tracked separately.
