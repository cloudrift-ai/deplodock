# Parallel-compile pool for the autotuner

## Context

`deplodock tune` is **compile-bound, not GPU-bound**. Profiling a whole-model tune this session (py-spy) put **2808 of
2890 samples in `cupy.cuda.compiler.compile`** — the driver PTX→SASS JIT — while the main process sat idle in
`_BenchWorker._read_with_deadline` waiting on the bench worker. Kernel *execution* is microseconds; main-process
lowering (building the `CudaOp` source) is ~2%. So ~98% of wall time is spent turning kernel source into machine code,
and that work is currently serial.

Why serial: the inner per-op search (`two_level.inner_reward`) tunes each post-fusion kernel in its own single-node
slice via `Pipeline.build(LOWERING_PASSES).tune(...)`. `Pipeline.tune` (`pipeline/pipeline.py`) yields one terminal
`Graph[CudaOp]` at a time and each `_bench_terminal` **blocks** on a single `backend.benchmark(...)` call that routes
through the reused singleton `_BenchWorker` subprocess (`backend/cuda/program.py`). The MCTS (`search/policy/mcts.py`)
pops one leaf at a time, so exactly one variant is in flight at any moment: lower → compile → bench → backprop, then the
next. Each inner variant is a single-kernel slice with a **unique** source, so cupy's on-disk cubin cache misses within
a sweep — every variant pays a full cold compile.

Measured parallel-compile scaling on an idle 32-core box (microbenchmarks this session):

- cupy `RawKernel.compile()` in a process pool caps at **~1.8×** (driver JIT / toolchain serialization), ~1.19 s per
  kernel.
- `nvcc --cubin` in independent processes caps at **~3×**, **and** is **~3.5× faster per kernel single-threaded**
  (~0.34 s vs cupy's ~1.19 s) because it uses offline `ptxas` with no driver JIT.
- A warm-cache cubin **reload is ~24 ms** — ~25× cheaper than a cold compile.

**Step (A) is already done.** Compilation now flows through `nvcc.compile_to_cubin` (GPU-free, content-addressed disk
cache, atomic publish) + a cupy `RawModule` load — see `backend/cuda/nvcc.py` and `program.py::_compile`. It was
validated correct across the full suite (1313 passed) with zero NVRTC fallbacks. `compile_to_cubin(source, name, *,
arch) -> Path` runs only the `nvcc` subprocess (no CUDA context), explicitly designed to be driven from a worker pool;
`load_function(...)` is the GPU half. `DEPLODOCK_NO_NVCC=1` forces the old cupy/NVRTC path.

**This plan is step (B): a parallel-compile pool** that warms the cubin cache ahead of the (still serial, GPU-exclusive)
bench worker, so the bench worker only ever pays the ~24 ms warm reload instead of a cold compile.

### Honest framing of the win

The compile-parallelism ceiling on this box is **~2–3×** (toolchain serialization is real and we cannot beat it). So the
pool's **incremental** win *over (A) alone* is ~2–3×. Combined with (A) (~3.5× per-kernel + ~24 ms warm load), the
projected total over the *original cupy-serial* baseline is order-10×, but the **pool itself contributes only the ~2–3×
parallel factor** — the rest is already banked by (A). This plan should not be sold as a 10× win on its own.

A genuinely orthogonal — and possibly higher-leverage — lever is **reducing variant count**: better MCTS priors,
tighter patience, or pruning provably-dominated tiles. Fewer compiles beats faster compiles, and the two compose. We
mention it here as a complementary direction; it is out of scope for this plan but worth tracking.

## Design

**Decouple compile (parallel, GPU-free) from measure (serial, GPU-exclusive).** Timing N kernels concurrently on one GPU
corrupts every latency (shared clocks / caches / thermal state — the exact reason `gpu_lock()` exists), so measurement
stays serialized. Only compilation parallelizes. The pool is **purely additive**: it pre-warms cubins ahead of the bench
cursor; the existing bench worker is otherwise unchanged and still hits the cache.

### Two process pools

- **`compile_pool`** — large, sized at ~`os.cpu_count()`. Each task runs `nvcc.compile_to_cubin(source, name,
  arch=...)` to warm the on-disk cubin cache for one kernel. **GPU-free**: the workers run only the `nvcc` subprocess
  and never import cupy or create a CUDA context. Note this explicitly removes the per-worker-VRAM concern that a
  GPU-touching pool would have — there is no device state to pin, so we can run as many compile workers as we have
  cores. `compile_to_cubin` is already idempotent + atomic (`os.replace`), so concurrent compilers (and a racing bench
  loader) never observe a half-written cubin; two workers handed the same source dedupe on the content-addressed path.
- **`bench_pool`** — **one worker per GPU** (`n_gpus`, = 1 today), each pinned via `CUDA_VISIBLE_DEVICES`, doing
  exclusive timing. This is essentially today's `_BenchWorker`, generalized from a singleton to a per-GPU set. It keeps
  its wall-timeout SIGKILL hang protection (`bench_wall_timeout_s`) and respawn-on-EOF behavior unchanged. Because the
  cubin is already on disk by the time bench runs, the worker's `_compile` is a ~24 ms `RawModule` load — no NVRTC, no
  JIT.

`compile_pool` and `bench_pool` are both `concurrent.futures.ProcessPoolExecutor`s; the orchestrator drives them with
`asyncio.wrap_future` over the futures they return.

### Single-threaded asyncio orchestrator

A **single-threaded asyncio event loop** in the main process owns the MCTS `SearchTree` and the `SearchDB`. All tree and
DB mutations happen in the event loop's completion handlers, which the loop runs one at a time — so **no mutex is
needed**. There is exactly one thread of control touching `SearchTree.record_terminal`, `pop`, the `live` counters, and
`SearchDB.record_perf`; concurrency lives entirely in the worker subprocesses, which exchange only plain values (source
strings in, cubin paths / `BenchmarkResult`s out). This is the key simplification: parallelism without shared-state
locking.

### Virtual loss (parallel MCTS)

The driver hands out up to `max_inflight` leaves before any reward returns. Standard single-tree MCTS assumes one
rollout completes before the next selection; with several in flight, the bare UCB descent (`pop`) would re-select the
same promising leaf repeatedly because nothing has updated its `visits` / `best_reward` yet. The fix is **virtual
loss**, the standard parallel-MCTS technique — required here **even though the orchestrator is single-threaded**,
precisely because selection runs ahead of backprop.

Today's `mcts.py` already has the structural hook we need: `SearchNode.live` counts un-popped frontier leaves, and `pop`
filters to `descendable = [c for c in node.children if c.live > 0]`, returning `None` when the whole frontier under root
is drained. Virtual loss extends this from "is there anything left to pop" to "bias *away from* subtrees that already
have rollouts pending":

- Add a per-node `in_flight: int` counter (sibling to `visits` / `live`).
- `apply_virtual_loss(leaf)` — on select, walk `leaf`→root incrementing `in_flight` on every node on the path.
- `clear_virtual_loss(leaf)` — on completion, walk the same path decrementing.
- Fold `in_flight` into UCB selection in `_ucb_key` so an in-flight subtree looks both **more-visited** (larger
  denominator → smaller exploration bonus) and **pessimistically-rewarded** (its `Q_norm` is dragged toward 0). The
  cleanest formulation that matches the existing max-Q normalized UCB1: treat each in-flight rollout as a virtual visit
  with reward 0, i.e. selection uses `effective_visits = visits + in_flight` in the bonus denominator and a
  virtual-loss-discounted `Q_norm` so a node with all rollouts pending is deprioritized but not excluded.
- `pop`/`select` must **skip a subtree whose entire live frontier is in-flight** and return `None` (→ the driver
  `await`s a completion) rather than spin. Concretely: the `descendable` filter stays `c.live > 0`, but when descent
  reaches a leaf that is itself already in-flight with no un-dispatched sibling, selection returns `None`. The driver
  treats `None` as "nothing new is selectable right now" and blocks on `asyncio.wait`.

Backprop is unchanged in spirit (`record_terminal` max-propagates the real reward up `last_popped`→root) — but the
parallel driver must pass the **specific leaf** to clear/backprop rather than relying on the `last_popped` field, since
`last_popped` is meaningful only for strictly-serial pops. We add an explicit `tree.backprop(leaf, reward)` and
`tree.select() -> SearchNode | None` (returning the node, not just its candidate) so the driver can carry the leaf
identity through the async round-trip. The existing serial `pop()` / `record_terminal()` stay for `GreedySearch` and the
single-threaded callers; the parallel path uses the new leaf-explicit methods.

### Per-variant pipeline

For each selected leaf, the round-trip is:

1. **select** the leaf (`tree.select()`), **apply virtual loss** along its root→leaf path.
2. **lower** to `Graph[CudaOp]` in the main process (cheap — this is the ~2% lowering cost; resolve the `LazyCandidate`,
   run `LOWERING_PASSES` to the terminal). Extract each kernel's `(source, name, arch)`.
3. **compile** — submit each kernel's source to `compile_pool` (`nvcc.compile_to_cubin`); `await` all cubins for this
   variant. Parallel, GPU-free, warms the cache.
4. **bench** — submit the lowered graph to `bench_pool` (load + time); serial per GPU, now a cache hit. Returns
   `BenchmarkResult`.
5. **completion handler** (back on the event loop): build `PerfStats`, `db.record(...)`, `tree.backprop(leaf,
   reward(stats))`, `tree.clear_virtual_loss(leaf)`, and call `search.observe(stats, status)` so patience / stop logic
   advances exactly as today.

### Async driver skeleton

```python
pending = set()
while (not stop()) or pending:
    while len(pending) < max_inflight and (leaf := tree.select()) is not None:
        tree.apply_virtual_loss(leaf)
        pending.add(asyncio.ensure_future(run_variant(leaf)))   # lower → compile_pool → bench_pool
    if not pending:
        break
    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    for t in done:
        leaf, stats, status = t.result()
        db.record(stats)
        search.observe(stats, status)
        tree.backprop(leaf, reward(stats))
        tree.clear_virtual_loss(leaf)
```

`run_variant(leaf)` is the per-variant coroutine implementing steps 2–4; it returns `(leaf, stats, status)`. `stop()`
reads `TuningSearch.stop_reason` (patience). `max_inflight ≈ ncores`: the single bench worker drains fast (µs of
execution + ~24 ms load) so it is **not** the bottleneck until compile parallelism saturates — at which point `pending`
naturally stays full of compile-bound coroutines and the bench worker keeps up.

### Where this lives

The inner per-op search is the hot loop, so the async driver replaces the serial `for _ in
Pipeline.build(LOWERING_PASSES).tune(sub, ...)` in `two_level.inner_reward`. Rather than fork `Pipeline.tune` wholesale,
add a parallel sibling — `Pipeline.tune_async(graph, *, search, compile_pool, bench_pool, ...)` (an async generator or a
coroutine returning the best) — that reuses `Pipeline.search`'s expansion/resolve machinery but drives selection through
the new leaf-explicit tree API and dispatches compile/bench to the pools. The existing synchronous `Pipeline.tune` stays
for `GreedySearch`, `run`, and every non-tune caller. `inner_reward` chooses the async path when pools are supplied,
falling back to the serial path when they are not (and when `nvcc` is absent — see Risks).

### Multi-GPU generalization

`bench_pool` is sized `n_gpus`, one worker per device, each spawned with `CUDA_VISIBLE_DEVICES=<i>` so it pins device 0
within its own namespace. The driver dispatches a bench to *any* free bench worker; since `compile_to_cubin`'s `arch` is
the **target** arch (already threaded via `device_arch` / the `--target sm_NN` flow), a cubin compiled once is loadable
on any same-arch device — homogeneous multi-GPU boxes share one cubin cache for free. The selection/backprop logic is
unchanged: more bench workers just means a higher *measurement* throughput ceiling, which only matters once compile
parallelism stops being the bottleneck. Today `n_gpus = 1`, so `bench_pool` has one worker and this collapses to the
current behavior plus the compile pool.

## Milestones (single branch, commit after `make test`)

- **M1 — Virtual loss in `mcts.py`.** Add `SearchNode.in_flight`; `SearchTree.apply_virtual_loss(leaf)` /
  `clear_virtual_loss(leaf)` (path walks); leaf-explicit `tree.select() -> SearchNode | None` and `tree.backprop(leaf,
  reward)`; fold `in_flight` into `_ucb_key`; make selection return `None` when the entire live frontier is in-flight.
  Keep the serial `pop()` / `record_terminal()` intact for existing callers. Pure in-memory, GPU-free — fully
  unit-testable. Commit.
- **M2 — Compile pool module.** A small module (`backend/cuda/compile_pool.py`) wrapping a `ProcessPoolExecutor` whose
  worker entry point is a thin shim over `nvcc.compile_to_cubin` (top-level function so it pickles). Exposes
  `submit(source, name, arch) -> Future[Path]` and a context-manager lifecycle. No GPU, no cupy import in the worker.
  Unit test: submit several distinct sources, assert each cubin lands in the cache and a re-submit of an identical
  source is a no-op (path hit). Commit.
- **M3 — Async driver.** `Pipeline.tune_async` (or an equivalent driver helper) implementing the skeleton above:
  select+virtual-loss → lower → `compile_pool` → `bench_pool` → completion (record + backprop + clear). Wire
  `two_level.inner_reward` to use it when pools are supplied; keep the serial path as fallback. Spin up / tear down the
  pools once per `run_two_level_tune` (so a whole-model tune shares them across every inner per-op search). Commit.
- **M4 — Multi-GPU bench pool.** Generalize the singleton `_BenchWorker` into a `bench_pool` of `n_gpus` workers, each
  `CUDA_VISIBLE_DEVICES`-pinned; the driver dispatches each bench to any free worker. Default `n_gpus = 1` (probe via
  cupy) so single-GPU behavior is unchanged. Commit.
- **M5 — Docs.** Update `backend/cuda/ARCHITECTURE.md` (compile/dispatch section: the two-pool split, GPU-free compile,
  one-bench-worker-per-GPU) and `pipeline/ARCHITECTURE.md` (autotune/two-level section: parallel compile, virtual loss,
  the single-thread async orchestrator, the honest ~2–3× compile ceiling). Mention the variant-count lever as a
  complementary direction. Update `CLAUDE.md` if a new env knob (e.g. `DEPLODOCK_COMPILE_WORKERS`) is introduced.
  Commit.

## Files to modify

- `deplodock/compiler/pipeline/search/policy/mcts.py` — `SearchNode.in_flight`; `apply_virtual_loss` /
  `clear_virtual_loss`; leaf-explicit `select` / `backprop`; `_ucb_key` virtual-loss term; all-in-flight → `None`.
- `deplodock/compiler/backend/cuda/compile_pool.py` — **new**: `ProcessPoolExecutor` over `nvcc.compile_to_cubin`,
  GPU-free worker shim, lifecycle.
- `deplodock/compiler/pipeline/pipeline.py` — `Pipeline.tune_async` (parallel driver reusing `search` machinery via the
  leaf-explicit tree API + the two pools); serial `tune` untouched.
- `deplodock/compiler/pipeline/search/two_level.py` — `inner_reward` drives the async path when pools are supplied;
  `run_two_level_tune` owns pool lifecycle.
- `deplodock/compiler/backend/cuda/program.py` — generalize the `_BenchWorker` singleton into a per-GPU `bench_pool`
  (M4); keep `compile_to_cubin` warm-cache assumption (loads stay cache hits).
- `deplodock/commands/tune.py` — surface a `--compile-workers` flag (default `os.cpu_count()`), thread `n_gpus` /
  `nvcc`-absent fallback.
- Docs: `backend/cuda/ARCHITECTURE.md`, `pipeline/ARCHITECTURE.md`, `CLAUDE.md` (new env knob, if any).

## Tests

- **Virtual-loss unit tests** (`tests/compiler/pipeline/search/`, no GPU):
  - `apply_virtual_loss` / `clear_virtual_loss` round-trip: `in_flight` increments along the whole root→leaf path and
    returns to 0 after clear; nested apply/clear of two disjoint leaves compose.
  - **UCB-skip**: with virtual loss applied to the only attractive subtree, the next `select()` either descends a
    different subtree or returns `None` when every live frontier leaf is in-flight (it must not re-select an in-flight
    leaf).
  - **Backprop correctness under interleaving**: dispatch K leaves (apply virtual loss to each), then complete them in a
    permuted order, asserting final `visits` / `best_reward` at the root match what a strictly-serial sequence of the
    same rewards would produce (max-Q is order-independent, so this is a clean invariant).
  - **Determinism-as-sets**: the *set* of leaves enumerated by the parallel driver over a deterministic stub equals the
    serial enumeration's set (order may differ — assert on sets / multisets, never on sequence).
- **Async-driver test with a stub backend** (`asyncio` + a fake `compile_pool` / `bench_pool`):
  - Assert **≤ pool-size in-flight**: instrument `run_variant` to record the peak `len(pending)` and assert it never
    exceeds `max_inflight`.
  - Assert **correct backprop**: every dispatched leaf gets exactly one `backprop` + one `clear_virtual_loss`, and the
    root's final `best_reward` equals the max stub reward.
  - Use a deterministic counting backend (mirror `_CountingBackend` in `test_two_level.py`) so per-variant rewards are
    stable and the assertions are exact.
- **Compile-pool test** (M2): distinct sources each produce a cubin; identical source re-submit is a cache hit (no
  second `nvcc` invocation — assert via the content-addressed path existing before the second submit). Skippable when
  `nvcc` is absent.
- **Keep existing search tests green**: `tests/compiler/pipeline/search/test_thunk_forks.py`,
  `test_greedy_db_lookup.py`, `test_two_level.py`, `tests/compiler/test_tune_accuracy.py` must pass unchanged — the
  serial `tune` / `pop` / `record_terminal` API is preserved, and `inner_reward`'s serial fallback path is exercised
  whenever pools are not supplied (as in those tests).

## Verification (end-to-end)

- `./venv/bin/pytest tests/compiler/pipeline/search/ tests/compiler/test_tune_accuracy.py -p no:randomly` green.
- **Speedup vs serial baseline**: time a real inner-heavy tune (the Qwen layer / `sdpa.s512`) with the pool vs the
  serial baseline forced two ways: (a) `DEPLODOCK_NO_NVCC=1` (original cupy-serial, the true baseline the order-10× is
  measured against), and (b) nvcc-on but pool disabled (`--compile-workers 1`, isolating the pool's ~2–3× from (A)'s
  ~3.5×). Expect the pool's incremental factor in the ~2–3× range over (b), consistent with the compile-parallelism
  ceiling.
- **Results must match the serial tuner's bests**: the per-op best latencies and the chosen forks the parallel tuner
  records to the DB must match the serial tuner's (modulo bench noise) — the parallel path only reorders *compilation*,
  not *measurement*, so the winning variant per op must be identical. Diff the `perf` / `lowering` rows of a
  pool-tuned DB against a serial-tuned DB on the same graph; the bests should agree. The final assembled whole-graph
  bench (the separability check) must also agree across the two paths.
- Poll long runs (don't block silently): watch `perf` row count + GPU util; with the pool, GPU util should *rise* (the
  bench worker stops idling on compiles) and the compile-bound cores should saturate.
- `make test` + `make lint` before each milestone commit.

## Risks / assumptions

- **Compile-parallelism ceiling (~2–3×).** Toolchain serialization is real; the pool will not scale linearly with
  cores. The plan is honest about this — the headline order-10× is (A)+pool over the *original* baseline, of which the
  pool contributes only the ~2–3× factor. If the measured pool factor comes in below ~2×, the variant-count lever is the
  fallback, not more compile workers.
- **Drain-on-stop.** When patience fires mid-sweep there may be `max_inflight` rollouts still in flight. The driver must
  drain `pending` (the `or pending` in the loop condition) and backprop their results before returning — otherwise the
  DB misses measurements that were already paid for, and a completing-but-discarded variant could have been the new
  best. Decision to lock in: results that complete after `stop()` are still recorded (they're free — the compile/bench
  already happened) but do not re-open the search.
- **Nondeterminism.** Completion order is nondeterministic (compile times vary), so the *sequence* of backprops differs
  run-to-run. Max-Q backprop is order-independent for the final `best_reward`, but the *exploration path* (which leaves
  get selected) can differ because virtual loss biases selection based on what is currently in flight. Tests assert on
  sets, not sequences; the end-to-end check asserts the *bests* match, not the exploration trace. The patience count
  ("terminals since last new best") is also slightly path-dependent — acceptable, but worth noting it can change the
  exact stop point vs the serial run.
- **Bench-worker pinning / single-GPU serialization.** Measurement must stay strictly serial per GPU; if a future
  refactor lets two benches overlap on one device, latencies silently corrupt (the whole reason `gpu_lock()` exists).
  The `bench_pool`-of-one-per-GPU invariant is load-bearing. The wall-timeout SIGKILL + respawn behavior must survive
  the singleton→pool generalization unchanged.
- **Fallback when `nvcc` is absent.** `compile_to_cubin` raises `RuntimeError("nvcc unavailable")` with no toolkit, and
  `DEPLODOCK_NO_NVCC=1` forces it off. In that case there is **nothing for the compile pool to do** — the cubin cache
  can't be warmed and bench would re-NVRTC anyway. The async driver must detect this (pools are `None`) and fall back to
  the existing serial `Pipeline.tune` path so correctness never depends on the pool. This is also the fallback for any
  per-kernel `nvcc` compile error (today's `load_function` already NVRTC-falls-back per kernel; the pool must not turn a
  single bad kernel into a sweep-killing exception — a failed `compile_to_cubin` future just means that variant's bench
  pays a cold NVRTC compile, exactly as today).
- **Process-pool overhead.** Pickling a `Graph[CudaOp]` to the bench worker already happens (the singleton does it);
  the compile pool only needs the **source string + name + arch** per kernel, which is cheap to pickle. Spawning
  `os.cpu_count()` workers has a one-time cost (~Python startup × N); amortized over a whole-model tune's thousands of
  compiles it's negligible, but the pools should be created **once** in `run_two_level_tune`, not per inner per-op
  search.
- **Assumption: cubin cache is the handoff.** The whole design rests on `compile_pool` warming the *same*
  content-addressed cache (`DEPLODOCK_CUBIN_CACHE`) the bench worker's `load_function` reads. Both must agree on
  `_cache_key(source, name, arch)` and on `arch` (target, not just live device). This is true today — `compile_to_cubin`
  is the single keyer — but any divergence (e.g. a compile worker resolving a different `arch` than the bench worker)
  silently turns every load back into a cold compile, erasing the win. The driver computes `arch` once and passes the
  same value to both pools.
