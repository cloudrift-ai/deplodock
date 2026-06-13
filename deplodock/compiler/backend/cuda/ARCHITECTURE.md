# CUDA Backend

CUDA-specific dispatch. Shared backend contract lives in
`backend/ARCHITECTURE.md`. The lowering chain that produces the
`Graph[CudaOp]` this backend consumes lives in `pipeline/passes/lowering/`
(see `pipeline/ARCHITECTURE.md`).

## Modules

```
cuda/
├── backend.py   # CudaBackend(Backend) — drives lowering + delegates dispatch
├── nvcc.py      # offline `nvcc --cubin` compile (+ content-addressed cubin cache) → RawModule load
└── program.py   # Graph[CudaOp] → kernel dispatch (via nvcc.py) + per-kernel event timing
```

## Compile

`CudaBackend.compile(graph)` runs
`run_pipeline(graph, [..., "lowering/kernel", "lowering/cuda"], dump=…)`,
producing a `Graph[CudaOp]` where every compute node carries a rendered
`__global__` source plus its launch geometry (grid / block / smem /
arg_order).

## Dispatch (`program.py`)

`_compile(graph) → _Compiled` walks the lowered graph:

- Classifies each node as `input` / `constant` / `output` / `scratch`
  from `graph.inputs` / `ConstantOp` membership / `graph.outputs`.
- Compiles each unique `kernel_name` via `nvcc.load_function` (`nvcc.py`):
  offline `nvcc --cubin` into a content-addressed disk cache, then a cupy
  `RawModule` load. This avoids the driver PTX→SASS JIT cupy's NVRTC path
  pays on a cold compile — ~3× faster on the complex tile-search kernels that
  dominate autotune, and the compile step is GPU-free so the cubin cache can be
  warmed by a parallel pool (planned). Falls back to `cupy.RawKernel` (NVRTC)
  when `nvcc` is absent or a compile fails (`DEPLODOCK_NO_NVCC=1` forces the
  fallback). Kernels are emitted with `extern "C" __global__` so neither
  toolchain name-mangles them (and the cubin symbol loads by `kernel_name`).
  Compile vs load is split (`compile_to_cubin` / `load_function`) so the
  GPU-free compile can run off-process; the loaded `Function` is launch- and
  smem-attr-compatible with `RawKernel`.
- **Opt level** comes from `DEPLODOCK_NVCC_FLAGS` (`nvcc.effective_flags`, which
  delegates to `config.nvcc_flags()` — `deplodock/config.py` is the single owner
  of `os.environ` for `DEPLODOCK_*` vars, incl. `DEPLODOCK_NO_NVCC` /
  `DEPLODOCK_CUBIN_CACHE`). The CLI sets the flags via `config.set_nvcc_flags`
  (override logic, no longer in the command layer) — `tune` → `-Xcicc -O1`,
  `compile`/`run` → nvcc default -O3, `--nvcc-flags` overrides. -O1 dodges a cicc/LLVM front-end blowup on big
  unrolled register-tile kernels (cicc, not ptxas, is the cost: a tall-thin
  register tile unrolls into a ~5K-instruction basic block → up to 21 s → 0.1 s
  at -O1) but is **NOT runtime-optimal** (reductions/attention ~1.5–3× slower),
  so it's a tune-time *ranking* knob only. The flags are folded into both the
  cubin cache key and `Context.structural_key` (the `perf` context key), so
  -O1-tuned and -O3 measurements never collide. The bench-worker subprocess
  inherits the env, so its compiles use the same flags.
- Builds a static launch plan: per launch, a tuple of
  `(kernel, arg_names, grid, block, smem_bytes, zero_outputs)`.

`run_program(graph, input_data) → RunResult`:

1. Allocate a `cupy.ndarray` for every buffer. Inputs + optional
   constant overrides come from `input_data`; scalar `ConstantOp`s
   materialize as single-element arrays; scratch buffers zero-init.
   Inputs without supplied data get a deterministic pseudo-random fill
   (useful for standalone compile-and-benchmark scripts).
2. Launch each kernel in topological order; `zero_outputs` fills run
   before the launch.
3. Copy `graph.outputs` buffers back to numpy.

**Repeated execution (`CompiledProgram.rebind` / `run_once`).** One built program can serve request after request —
the serving path (the vLLM plugin runs one compiled dynamic-seq_len program per sequence). `rebind(input_data)`
re-binds fresh inputs on the existing program: supplied buffers re-upload (in place when the resolved shape is
unchanged, re-allocated otherwise), un-supplied buffers whose shape carries a symbolic dim (seq_len-sized
scratch/outputs) re-materialize under the same fill policy as `build` (shared `_materialize`), and static-shaped
un-supplied buffers — the weights — keep their device arrays untouched. Any re-allocation rebuilds TMA descriptors
(they embed device pointers + shapes) and drops captured CUDA graphs (they bake old pointers). `run_once()` launches
every kernel in program order with none of `iter_once`'s per-launch event record/sync/watchdog — bench semantics stay
in `iter_once`; the caller's `outputs()` `.get()` synchronizes. Both expect the caller to hold `gpu_lock()`. See
`tests/compiler/test_program_rebind.py`.

**Captured-graph replay over a capacity buffer set (`set_sym_values` / `upload_prefix` / `capture_program_graph` /
`replay_program_graph` / `outputs(sym_values)`).** The serving fast path: instead of `rebind` re-sizing buffers and
`run_once` issuing ~hundreds of host launches per request, build the program once at a **capacity** seq_len, then per
request (1) `set_sym_values({"seq_len": S})` sizes the launch grids + by-value seq_len to the real S without
re-allocating (errors if S exceeds capacity), (2) `upload_prefix(input_data)` H2D's each input into the contiguous
prefix of its capacity buffer (a logically `(1, S, …)` tensor occupies the first `S·…` elements), (3)
`capture_program_graph()` captures the whole program at the current S into ONE CUDA graph — **cached per seq_len**
(bounded LRU `_graph_cache`), so a repeated length replays with no re-capture — and (4) `replay_program_graph()` is one
host launch; `outputs({"seq_len": S})` slices each capacity buffer to its real-S prefix. Each graph is captured at its
EXACT S, so every kernel runs at its exact grid: no oversized-grid masking is needed (and a single capacity-baked graph
for ALL S is not viable — several symbolic-M kernels read OOB at an oversized grid: the CTA-swizzle decode reconstructs
`num_m` from the runtime seq_len, and ceil-div staged loads over-read; see
`plans/serving-dynamic-shape-cuda-graphs.md`). `rebind` clears `_graph_cache` on re-allocation. Validate multi-S
correctness under `compute-sanitizer` (`tests/compiler/ir/test_dynamic_shapes.py`).

`benchmark_program(graph, input_data, warmup, num_iters)` adds a
warmup loop + timed loop wrapped in `cupy.cuda.Event` pairs — one
global pair for `BenchmarkResult.time_ms`, one pair per launch index
(averaged over iters) for `BenchmarkResult.per_launch`. Each launch is awaited via the polling
`_wait_for_event` (`_KERNEL_TIMEOUT_MS`, 1 s) rather than a blocking `synchronize()`, which would
hang forever on a non-terminating kernel; on overrun it raises **`HungKernelError`** (a
`RuntimeError` subclass, so callers' `except RuntimeError → bench_fail` still catch it). This is the
in-process timing core; both the autotune bench and the deployable comparison run it **inside the
worker** (below), so a hung kernel hangs the child, not the parent.

`benchmark_program` captures each launch position's batch into a CUDA graph **by default**
(`capture_graphs=True` → `CompiledProgram.capture_launch_graphs`, right after batch-size calibration),
and `iter_once` replays `graph_i.launch()` — one host call per event window — instead of the Python
launch loop. CUDA events measure *stream elapsed* time, which only equals GPU time when the stream is
saturated; for sub-10 µs kernels the per-launch cupy dispatch starves the stream and the events time
the starvation. The graph replay keeps the stream dense, so the same event windows become pure-GPU
measurements. Capture happens on a temporary side stream (stream capture is illegal on the legacy NULL
stream); replay targets the current (NULL) stream, so cupy/torch event interleaving is unchanged, and
so is the `_wait_for_event` watchdog (a hung kernel inside a graph still never completes its stop
event). Warmup iters always run uncaptured, so the zero-elapsed degenerate-launch guard and the
watchdog probe real launches before any graph exists. Re-fires of the calibration branch (warmup
extension) re-capture when batch sizes change. A capture failure (`GraphCaptureError`, raised by
`capture_launch_graphs` after draining the capture state) is caught in place: the bench warns,
continues uncaptured, and reports it via `BenchmarkResult.captured` — comparison callers
(`bench_lowered_vs_torch` / `bench_full_model_real`) pair that flag with their torch-side capture and
re-run all-or-nothing so one table never mixes semantics. The tune sweep persists the flag per `perf`
row (`SearchDB.record_perf`): captured measurements supersede wall-semantics ones for the same key
regardless of median, never the reverse — old rows stay usable (replay, prior training) and upgrade
in place as re-tunes measure them captured. See `tests/compiler/backend/test_graph_capture.py`.

**Whole-program (e2e) windows.** The per-launch windows each replay a *single* kernel back-to-back, so
their sum is not an end-to-end time: it misses cross-kernel cache effects and inter-kernel gaps, and on
multi-kernel programs individual per-launch numbers can mis-attribute wildly (two identical-work gemms in
the Qwen3 layer-0 assembly measured 5.2 µs vs 0.8 µs solo; NCU shows them equal — finding 6 of
`plans/qwen3-embedding-layer0-tune-findings.md`). For any **multi-launch** program `benchmark_program`
therefore also captures **one** CUDA graph holding every launch in program order
(`CompiledProgram.capture_program_graph`) and, once per measured iter, times one event window around
`replays` back-to-back whole-program replays (`time_program_window`, replays calibrated to
`_BATCH_TARGET_MS`) — the same semantics the captured torch closures get in the interleaved bench, so the
backend table compares like-for-like. Reported as `BenchmarkResult.e2e_ms`/`e2e_min_ms`; `run --bench`'s
comparison table prefers `e2e_min_ms` for the Deplodock row and the kernel table prints a
`whole-program (e2e)` footer beside the per-launch `TOTAL`. Automatic — no flag: a single-launch program's
solo window already IS the program time (the autotune sweep's usual single-node slice — fields stay `None`,
nothing is measured twice), and multi-launch programs get it whenever capture holds (a program-graph capture
failure warns and skips, never fatal; uncaptured benches skip it too). The sweep still *ranks* variants on
the per-op sum (`time_ms`) by design — per-op results key structurally and transfer across graphs, which an
e2e scalar can't — so for its multi-launch slices (split-K fixups) the e2e fields are measured-but-unread
(~1 ms/iter); pricing those variants by slice-e2e instead is a possible future tune-semantics change.

**One worker, two jobs.** `_bench_worker.py`'s `_run_job` dispatches on `torch_spec`: `None` is the
deplodock-only autotune bench (`benchmark_program`); otherwise it's the deployable eager /
torch.compile / deplodock comparison — `("trace_args", {code/input/layer/seq_len/dynamic})` →
`load_or_trace` rebuilds the real module (HF id or `--code` expr) → `bench_full_model_real` (for a
symbolic graph the torch closures run on hint-**tiled** example inputs — `commands/run._hint_sized_inputs`
grows every symbolic input axis to its `Dim` hint by repeating the trace values, the same size the
deplodock side resolves to when benching without inputs, so the full-model table compares one shape; the
printed table carries a `benched at seq_len=… (symbolic hint)` note);
`("frontend_graph", Graph|None)` → `bench_lowered_vs_torch`. Rebuilding the torch side **in the
child** (not pickling a live module) is what lets the interleaved comparison — which couldn't cross a
subprocess boundary before — run isolated. So `tune --bench` (`commands/tune.py` `_run_bench` /
`_bench_per_kernel`) and any deployable bench go through `benchmark_compare_isolated_async`: a hung kernel
hangs the child, the parent SIGKILLs it at `wall_timeout_s`, the device is freed, and the per-kernel
sweep **continues** to the next reproducer (no device-poisoning wedge, no skip). The whole bench surface
is **async-only** — the parent transport is the single **`_AsyncBenchWorker.run_job`** (the old sync
`_BenchWorker` and the sync `benchmark_program_isolated` / `benchmark_compare_isolated` bridges are gone).
`benchmark_compare_isolated_async` awaits a one-shot instance (`_run_job_oneshot`); the autotune sweep
awaits a persistent per-GPU instance directly via `benchmark_program_isolated_async`. Synchronous CLI
entry points (`handle_run`, `_handle_run_ir`, `_run_bench`) bridge with `asyncio.run`. See
`tests/compiler/backend/test_bench_worker_compare.py` (compare-in-worker + SIGKILL recovery),
`test_hung_kernel_watchdog.py` (watchdog raises promptly), and
`tests/compiler/cli/test_tune_bench_hung_kernel.py` (the `_run_bench` control flow).

**One async transport — `_AsyncBenchWorker`.** It drives the `_bench_worker.py` subprocess protocol (`<8-byte LE
length><pickle>`, both directions) over `asyncio` streams, so one event loop can keep N device-pinned workers benching
concurrently (`tune --gpus`, see `pipeline/ARCHITECTURE.md` → *Per-kernel GPU parallelism*). Two entry shapes:

- **Autotune sweep** awaits `benchmark_program_isolated_async(graph, worker=…)`. `CudaBackend(device_id=i)` lazily owns
  one **persistent** worker (reused across configs — pay the ~0.2 s Python spawn once) and exposes `benchmark_async`,
  the single benchmarking entry point: the isolated-worker path when `bench_wall_timeout_s` is set and no `on_iter`,
  else the in-process `benchmark_program` path (interactive `run --bench` interleaving). The device pin is a **per-worker spawn-env overlay** —
  `CUDA_VISIBLE_DEVICES=<id>` (so the child's logical device 0 *is* that GPU; every argumentless `cp.cuda.Device()`
  resolves correctly) plus, when a base `DEPLODOCK_GPU_LOCK` is set, a per-device `…-<id>` lock path so workers on
  different GPUs take distinct `FileLock`s instead of serialising. The overlay rides the child only — the parent's
  `os.environ` is never mutated (all slots share one event-loop thread).
- **Deployable `--bench`** awaits `benchmark_compare_isolated_async`, which uses `_run_job_oneshot` (spawn → run →
  `aclose`). The worker's streams bind to the loop, so it can't persist across `asyncio.run` calls — a per-call spawn,
  negligible against a minutes-long deployable bench. Sync callers (`_run_bench` / `_bench_per_kernel`) `asyncio.run` it.

The wall-clock cap is `asyncio.wait_for`; on overrun the child is SIGKILLed and the next bench respawns it on a clean
device. Because the persistent worker is reused across configs, an illegal / misaligned access is a hazard: that error
is **sticky** — it corrupts the CUDA context so every later call returns the same status until the process dies, which
would cascade identical false `bench_fail`s across all subsequent configs. So after any failure the worker probes its
context (`_context_dirty` — a cheap `deviceSynchronize`) and *exits* if it's poisoned; `run_job` respawns a clean
context on the next request (the dead-proc check before `await self._spawn()`). Benign failures (NVRTC compile errors,
cleaned-up OOM) leave the context healthy and keep the worker alive, so they pay no respawn cost. A stale-worker race
on the send (a `BrokenPipeError`/`ConnectionResetError` from `stdin.drain` after the worker's dirty-context exit)
triggers one respawn + resend before surfacing as `bench worker died during request send`. Error paths `await aclose()`
(SIGKILL + reap) so the subprocess transport is cleaned before the loop closes. See
`tests/compiler/backend/test_bench_worker_recovery.py` and `test_async_bench_worker.py`.

`iter_once` (the per-iter sample loop) rejects a `cp.cuda.get_elapsed_time`
reading of `<= 0.0` as `bench_fail` instead of accepting it as a 0µs sample.
CUDA event timing has sub-µs resolution and any real launch consumes at least
one device cycle — a 0.0 reading means a degenerate kernel (BM=1×BN=128 with
the M tile fully masked, a kernel fused into a no-op, an event-pair quirk)
that would otherwise lock in as the autotune DB's unbeatable best. Phase B
of the same work; the existing worker → parent → `record_perf(bench_fail)`
path carries the failure unchanged.

`run_program_debug(...)` snapshots every non-input buffer after each
launch — consumed by `--dump-dir` runs.

## Invariants

- `CudaOp.arg_order` embeds the original node id as the output buffer
  name. The lowering rules therefore mutate node ops **in place**
  instead of splicing a fresh node — see `pipeline/ARCHITECTURE.md`
  under "Rule module convention".
- The CUDA backend imports from `ir/` and `pipeline/` but never into
  them. A ROCm/SYCL/Metal backend replaces `program.py` only.
