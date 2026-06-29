# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `005_contract` — construct the contraction node (before materialize)

A `Semiring` contraction's high-level node is built one pass **before** the materializer. The node is one of two arms of
the `Contraction` family (`ir/kernel/ir.py`), keyed by its **atom** (`ir/tile/atom.py`) — a tensor-core mma cell
(`AtomKind`, `lanes == 32`) or a scalar fma cell (`ScalarAtom`, `lanes == 1`):

- **mma arm** (`MmaContraction`) — a warp-tier `SemiringKernel` (its schedule carries a `WarpTile`): **thin**, the
  op-tree-dependent part only — capture the m/n/k axes, read the atomize binding (resolved in `020_schedule`), resolve
  the projection epilogue (`_store.with_store`).
- **scalar arm** (`ScalarContraction`) — a register-tiled `SemiringKernel` (its `TILE` plan tiles the output): lower the
  per-cell body (`lower(op)` + output-store glue) and capture it with the tiled axes + register / parallel widths.

A non-tiled contraction (per-cell fallback) and the cooperative reduce tier are left untouched here (`RuleSkipped`) and
bind to threads in `010_materialize`. Homing the construction here keeps the contraction a first-class node that exists
*before* thread-binding. Both arms ARE `structural_key`-ed as an intermediate `KernelOp` (the kernel-stmt protocol + a
`_rewrite` handler). The mma arm's final `KernelOp` / `CudaOp` keys stay byte-identical to the old single-pass
materialize; the scalar arm is restructured onto the shared tiling layer (explicit unit axes — its `op_cache_key`
shifts, so the scalar perf cache / prior re-warms).

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` dispatches on the kernel kind / its schedule (the article's "schedule separate from combine" thesis —
the op tree + `ir/tile/ops.lower` are shared across kinds; only the partition changes):

- **Scalar tier** — one thread per output cell (`lower(op)` + an output-store glue).
- **Reduce tier** (`_reduce`, a `MonoidKernel` whose `ReducePlan` cooperates / register-folds) — the reduce axis is
  partitioned `coop` ways across the CTA's threads and `reg` ways across per-thread accumulators (ILP), then a REG-tree
  fold, the cross-thread combine (`_combine`), and the projection.

The materializer takes one of the thread-binding tiers above for a `TileOp` root, OR — for a `KernelOp(Contraction)`
root that `005_contract` produced — **expands the contraction** through the one atom-generic `_factor.factorize` over the
shared tiling layer (below); the atom selects the leaf `Unit` (mma / scalar). The pass's `PATTERN` matches `(TileOp,
KernelOp)` so the single rule covers both; at this point in the kernel pass the only `KernelOp`s are the contraction
nodes.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The atom factorization — one factorizer, two atoms (`_factor.py` / `_tiling.py`)

`_factor.factorize` is the **single** contraction factorizer — there is no per-atom variant. It expands any `Contraction`
by tiling a **leaf atom** four ways through the unit-generic layer in `_tiling.py`:
`grid_tile(unit_tile(register_tile(atomize(...))))` — **GRID** block / **UNIT** / **REGISTER** / **ATOM**. Both arms run
that *same* pipeline; the atom only selects the leaf `Unit`, which exposes a common tiling-geometry interface
(`atom_m`/`reg_m`/`units_m`/`m_uvar`/`tile_m`/`lanes`/…) the factorizer reads:

- **mma** (`_warp_factor.AtomUnit`) — atom `(16, 8, 16)`, `lanes == 32`. The UNIT is a **warp**; the leaf emits
  `RegFragment` / `LdmatrixLoad` / `MmaSyncPtx` / `RegStore`, owns the K-loop + operand staging (gmem-direct / cp.async /
  TMA), and decodes the atom-lane offset at render. All mma geometry lives in `_warp_factor.py`, the new-atom seam.
- **scalar** (`_scalar_factor.ScalarUnit`) — atom `(1, 1, 1)`, `lanes == 1`. The UNIT is a **single thread** (so there is
  no `_lane` axis); the leaf comes from the lowered per-cell body (split into a `pre` region / the reduce `Loop` / a
  projection `tail`), replicated per register cell with its operand loads deduped (the arithmetic-intensity reuse).

The **unit** is the atom's parallel thread footprint (`atom.lanes`) — so the tensor-core warp tile and the scalar
parallel thread-tile are the *same* level, differing only in `lanes`; `block_threads = units · lanes`. `grid_tile` also
carries any leading (batch) grid axes and supports a 1-D (m-absent) output. (The store-glue helpers `with_store` /
`has_write`, shared by the constructor and the thread-binding tiers, live in `_store.py`.)

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
any tier. `AtomUnit` builds the seam from its decoded block / warp / lane axis vars (never a raw `threadIdx.x`,
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
