# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `005_contract` — construct the contraction node (before materialize)

A tensor-core contraction's high-level node is built one pass **before** the materializer. `005_contract` matches a
warp-tier `SemiringKernel` (a `Semiring` whose schedule carries a `WarpTile`) and emits a single high-level
`MmaContraction` (`ir/kernel/ir.py`) — **thin**, doing only the op-tree-dependent part: capture the m/n/k axes, read the
atomize binding (resolved in `020_schedule`), resolve the projection epilogue (`_store.with_store`). Every other tier
(scalar / reduce / register-tile) is left untouched here (`RuleSkipped`) and binds to threads in `010_materialize`.

Homing the construction here keeps the contraction a first-class node that exists *before* thread-binding. The
`MmaContraction` IS `structural_key`-ed as an intermediate `KernelOp` (it carries the kernel-stmt protocol + a `_rewrite`
handler), so the final `KernelOp` / `CudaOp` keys stay byte-identical — the perf cache / prior transfer untouched.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` dispatches on the kernel kind / its schedule (the article's "schedule separate from combine" thesis —
the op tree + `ir/tile/ops.lower` are shared across kinds; only the partition changes):

- **Scalar tier** — one thread per output cell (`lower(op)` + an output-store glue).
- **Reduce tier** (`_reduce`, a `MonoidKernel` whose `ReducePlan` cooperates / register-folds) — the reduce axis is
  partitioned `coop` ways across the CTA's threads and `reg` ways across per-thread accumulators (ILP), then a REG-tree
  fold, the cross-thread combine (`_combine`), and the projection.
- **Register-tile tier** (`_reg_tile`, a `SemiringKernel` whose `TILE` plan tiles the output) — each thread owns a
  `reg_m × reg_n` block of cells, the reduce body replicated per cell with its operand loads deduped (the
  arithmetic-intensity reuse).
The materializer takes one of the thread-binding tiers above for a `TileOp` root, OR — for a `KernelOp(MmaContraction)`
root that `005_contract` produced — **expands the contraction**: `_warp_factor.factorize_mma` turns the high-level node
into the `Tile` of `RegFragment` / `LdmatrixLoad` / `MmaSyncPtx` / `RegStore` (the four-way GRID/WARP/REGISTER/ATOM
split, the operand staging decision, the per-cell epilogue). The pass's `PATTERN` matches `(TileOp, KernelOp)` so the
single rule covers both; at this point in the kernel pass the only `KernelOp`s are the contraction nodes.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The exact tensor-core atom factorization (`_warp_factor.py`)

The warp tier is the one place the lowering used to carry hundreds of lines of atom geometry. That geometry is factored
out: `005_contract` emits an `MmaContraction` (`ir/kernel/ir.py`) carrying everything the expansion needs (the operand
`Load`s + roles, the accumulator, the resolved epilogue `Body`, the `WarpTile`, the `Stage`, the m/n/k axes, the output
buffer), and `010_materialize` expands it via `_warp_factor.factorize_mma`. All atom geometry — `_axis_base`, the staging
eligibility (`_can_stage_warp[_tma]`), the staged K-loops, the `RegStore` epilogue — lives in `_warp_factor.py`, the
new-atom seam. (The store-glue helpers `with_store` / `has_write`, shared by the constructor and the thread-binding
tiers, live in `_store.py`.)

## Operand staging (`_stage.py`) — the `STAGE` codec

When the schedule carries a `Stage` (the `STAGE` codec `d<depth>/sync|cp|tma[/ring][/p<reg_depth>]`), the warp tier
stages its A/B operands through a shared-memory slab instead of reading gmem-direct. `_stage.py` assembles the surviving
Kernel-IR transport leaves — it does **not** resurrect the demolished `StageBundle` / `StagePolicy` orchestration.

**The two-level pipeline.** Staging has two independent buffering levels, each with its own depth (`depth` and
`reg_depth` are both buffer depths on `Stage`; `WarpTile.bk` is the slab K-*granularity*, kept distinct):

- **`depth` — the gmem→smem ring** (cp.async only today; TMA stays single-buffer). A `depth`-slot slab; a prologue
  primes the first `depth-1` K-chunks, then each step prefetches the chunk `depth-1` ahead into a free slot while the
  mma consumes the current one (`CpAsyncWait(group=depth-1)` keeps the prefetches in flight). The tail prefetch is
  clamped to the last chunk — an in-bounds re-read into a slot that's never consumed — so the commit/wait stays uniform
  across all CTA threads (the barrier-under-mask invariant). `depth` is clamped to the 48 KB smem cap (a deeper pin
  falls back to a shallower ring). `depth=1` is the plain single-buffer fill→wait→drain.
- **`reg_depth` (`/p<n>`) — the smem→register double-buffer.** `_staged_inner_atom_loop` unrolls the inner atom-K loop
  into a software pipeline that ldmatrixes the next atom-K step into an alternate fragment slot (`_a{i}_s{slot}`)
  `reg_depth-1` steps ahead, breaking the per-step WAR hazard on the operand fragments. `reg_depth` is capped at the
  atom-K step count (`bk`).

Both compose (`d3/cp/p2`) and are pure perf transforms — bit-identical to the single-buffer / gmem-direct baseline.

**The `CtaTile` seam.** The fill is written against a small `CtaTile` (the CTA tile-base coords + an independent linear
intra-CTA thread id + the thread count), NOT a materializer's internal warp/register geometry — so one fill helper drives
any tier. `factorize_mma` builds the seam from its decoded block / warp / lane axis vars (never a raw `threadIdx.x`,
whose `free_vars()` would leak into param collection).

**Two producers, one drain.** `cp_async_fill` (the `sync` / `cp.async` thread-stripe) and `tma_fill`
(`cp.async.bulk.tensor` — one thread issues `MbarrierArriveExpectTx` + the operand box `TmaLoad`s onto a single mbarrier;
every thread waits the parity) share the slab + the staged `LdmatrixLoad(staged=True)` drain (`_staged_inner_atom_loop`),
plain row-major NONE-swizzle slabs. The TMA path declares a `TmaDescriptor` per operand (encoded host-side off the bound
array's device pointer at launch — `backend/cuda/_tma.py`); its slabs are 128 B-aligned. A masked / symbolic **M** rides
the fill: cp.async clamp-reads the overhang row (`% M`), TMA zero-fills the box past the descriptor's runtime globalDim —
either way the `RegStore` `m_guard` discards the masked-row stores. A symbolic / non-divisible **K** zero-fills the tail;
a masked **N** or transposed-B stays gmem-direct.

**Shared-row staging (`_reduce`).** The fused norm→linear prologue is a `MonoidKernel`: an input row folded by the
cooperative reduce AND re-read per output column of a contraction tail (a free-axis `Loop` over an inner reduce).
`_reduce` stages that one row into a single `__shared__` slab (cooperatively filled) and rewrites both readers to it. The
trigger is narrow (`_has_contraction_tail`) so a plain softmax sum or a bare reduction is untouched.

Staging only ever *adds* a faster lowering — an ineligible kernel silently falls back to gmem-direct.

## Kernel-IR peepholes

`030_stamp_types` / `040_demote_to_write_dtype` resolve element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `110_drop_redundant_syncs` collapses the defensive `Sync`s the staging
templates emit (body-level only — the slab `Smem` decls flag `smem_seen`, so a load-bearing prologue `Sync` after a
`Cond(MbarrierInit)` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).
