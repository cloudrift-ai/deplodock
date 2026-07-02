# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` is a thin wrapper: after the split-survivor assert it makes **one** call to
`_factor.factorize(tile, root)`, the entry to the recursive emitter. `factorize` builds the ambient `Ctx` and dispatches
`tile.op` through `_factorize`, which peels the projecting `Map`s and binds the leaf via the ONE root binder
(`_factor._bind`) — a single pipeline whose form is read off the node's SCHEDULE (which axes are tiled), never a kernel
kind, sealed through the one `grid_tile` finalizer (the article's "schedule separate from combine" thesis — the op tree
+ `ir/tile/ops.lower` are shared across kinds; only the partition changes). Its arms are points of one
`(output-tiling) × (reduce-folding)` space:

- **OUTPUT-tiled** (a `Contraction` — warp / register tile) — the high-level `Contraction` Stmt
  (`ir/tile/ir.py`) was already built **recognize-side** at fork-emit
  (`lowering/tile/_schedule._contraction_node`, resolving the operand→role binding via `_atomize.semiring_binding`), so
  `_bind` only **synthesizes its bare grid-`Write`** (needs `root.output`, so it can't ride the node) and
  **expands** it through the shared tiling layer (below); the leaf type selects the codegen
  (mma / scalar). An unbindable contraction (a non-`Load` operand) keeps the `Map` form and falls through to the
  degenerate arm here. (This build was a separate `005_contract` pass, then folded into materialize, and now lives
  recognize-side so the node's `tile` / `bind` exist before scheduling — seam #1.)
- **REDUCE-tiled** (`_tile_reduce_axis`, a `PLANAR` / `TWISTED` reduce — or a non-output-tiled `CONTRACTION` — whose
  `ReducePlan` cooperates / register-folds) — the reduce axis is tiled instead: `coop` lanes across the CTA's threads
  (its unit level) and `reg` ILP chains across per-thread accumulators (its register level), then a REG-tree fold, the
  cross-thread combine (`emit_combine`), and the projection. It reads the reduce straight off the `Reduction` node (no
  `lower`-then-refind) and builds its per-cell body via the recursion (`_emit`, below); the output stays one cell per
  thread (the 1×1 `atomize`, the grid riding `lead_axes` untiled).
- **Degenerate** — nothing tiled: one thread per output cell (`_emit(op)` + an output-store glue).

### The recursive node walk (`_emit`) — one hierarchical emitter

Two recursions cooperate. The **root** recursion `_factorize(op, ctx, tail, out_val)` binds a node to the grid: a `Map`
with a `source` recurses (projection → `tail`), the leaf binds via the one `_bind` pipeline. The **body**
recursion `_emit(op, ctx) -> Frag` builds the per-cell loop-IR — over the `Map` / `Reduction` / `Contraction` tree,
through **`source` AND `partial`** — threading a `Ctx` **down** (the ambient cell environment: the grid axes, operand
`inputs`, `stage`, output buffer) and returning a `Frag` **up** (the per-cell `body` this node contributes, the produced
`Handle` wire, and the reduce `carrier` when it folds one). The reduce binder drives `_emit` off the `Reduction` node to
build its per-cell reduce loop, so a **nested** `Contraction` (flash's Q@K / P@V) is reached AS A NODE. This is the
tile-IR-rebuild mandate's *one hierarchical emitter, no divergent codegen path*: `_emit(node).body` is byte-identical to
`ir/tile/ops.lower(node)` for a scalar-nested (block=1) node today. `Handle` carries `name` + `residence` (a scalar
register value); the **tensor-core seam** is the `Contraction` case in `_emit` — an output-warp-tiled contraction (an mma
`TilePlan`) emits through the register-tile pipeline + the accumulator→operand fragment recast there, where the rebuild
extends `Handle` with the mma fragment descriptor `(mma_role, shape, dtype)` and `_emit`'s `Ctx` grows the warp binding +
the inbound `wires` (flash's score fragment feeding P@V's A operand).

The `Contraction` node is **one flat** Stmt — binding-driven for both atoms, with **no per-atom subclass** — that cleanly
splits the **algebra params** (what to contract: the m/n output `axes` + the `k_axis`, the leading batch `lead_axes`, the
B operand `Load` + the A `a_operand` — a gmem `Load` **or** a computed register-resident `Body` (flash PV's `P = exp(S −
M)`, produced from an in-register score, not a gmem address) — the fold accumulator `acc`, and the projection `epilogue`)
from the **schedule** (one `tile: TilePlan` field carrying the leaf `atom` — a tensor-core `AtomKind` / the scalar
`ScalarAtom`, `ir/atom.py` — plus the unit/register widths + K-chunk). The per-CTA geometry (the `(m, n)` `Side` pair —
tile width / mask / block+unit var names — plus `block_threads`) is **derived** on the node from `tile` × `axes`
(`@property`). Keeping the schedule a single swappable
field is what lets the same operand/`acc` params be tiled by a *different* `TilePlan` — the seam the flash inner QK/PV
reuse needs.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The one factorizer — the single binder + reduce-axis tiling (`_factor.py`), atom strategies (`_atom.py`)

`_factor.factorize(tile, root)` is the **entry** every `TileOp` root lowers through: it builds the ambient `Ctx` and
dispatches `tile.op` into the recursion `_factorize(op, ctx, tail, out_val)`. `_factorize` walks the node tree — a `Map`
with a `source` **recurses** (its projection `body` walked, via `_emit_body`, into the `tail`), and the leaf binds to
the grid via the **ONE** root binder, `_bind` — a single pipeline that reads WHICH AXES the schedule tiles off the node
and seals through the one `grid_tile` finalizer. A tiled `Contraction` tiles its OUTPUT `(m, n)` axes (register / warp
cells; the reduce K serial per cell); a cooperating `Reduction` tiles its REDUCE axis instead (`_tile_reduce_axis` —
BLOCK `coop` lanes at the unit level, REG `reg` ILP chains at the register level, the carrier merge closing the fold),
its per-cell reduce loop built via `_emit` off the node; anything else tiles nothing and folds serially one thread per
output cell (the degenerate `_emit(op)` + `with_store`) — there is **no** separate "scalar tier" branch, and no
per-kind emitter: which axis is tiled is schedule data, not a kernel identity. The projection sink and the store value
(`out_val`, the root node's produced `Handle`) are threaded down the recursion, so `with_store` is node-agnostic. The
recursion, the binder, the reduce-axis tiling, and the shared-row staging apply live in `_factor.py`. **There is no
kind-specific path — no flash / attention special case.** Flash is the two-`Contraction` `TWISTED` reduce tree, so its
Q@K / P@V contractions and its streaming reduce factorize through this one recursion (scalar block=1 today). A
tensor-core flash tier is a matter of the contractions carrying an mma `TilePlan` (a schedule field on the node) and
routing through the `_emit` `Contraction` warp seam like any other mma matmul — **never** a bespoke emitter, which
would be a divergent codegen path the mandate forbids.

**The contraction factorization — two atoms.** `_bind`'s output-tiled arm is atom-generic — there is no per-atom
variant, and **no per-atom geometry object**. It expands any `Contraction` by tiling a **leaf atom** four ways through
the tiling layer (now inlined in `_factor.py`):
`grid_tile(unit_tile(register_tile(atomize(...))))` — **GRID** block / **UNIT** / **REGISTER** / **ATOM**. The tiling
geometry (the `(m, n)` `Side` pair — `tile` / `mask` / `block` / `unit` per axis — plus `block_threads` / `lanes`) is
**derived on the `Contraction` node itself** (`@property`, from the `tile` schedule × the output axes); the two sides
thread through the tiling levels + the codegen callables as one `(m, n)` pair. `factorize` reads it straight off `c`
and hands
`grid_tile` the codegen in two halves: `_atom.reduce_codegen` — the reusable, **sink-agnostic** `(state_decls,
reduce_region)` (operand fragments + the K-loop) — and a per-cell **sink** `store`. The
default is the matmul `_atom.store_sink`; `factorize(c, store=…)` swaps it. The K-loop itself is **one driver** on the
strategy base (`_AtomOps.reduce`), deciding nothing: the **scheduler-resolved** `Stage` picks its form — gmem-direct
(`None`) through the shared `_contract_kloop` `read → ⊗ → fold` spine, or staged through the shared `_staged`
fill→drain skeleton — and the atom contributes only leaves, never a loop. Per-atom diff:

- **mma** (`_MmaOps`) — atom `(16, 8, 16)`, `lanes == 32`. The UNIT is a **warp**; its leaves emit `RegFragment` /
  `LdmatrixLoad` / `MmaSyncPtx` / `RegStore` and decode the atom-lane offset at render.
- **scalar** (`_ScalarOps`) — atom `(1, 1, 1)`, `lanes == 1`. The UNIT is a **single thread** (so there is no `_lane`
  axis); its leaves are plain `Load`s + an fma cell, the projection `tail` replicated per register cell with its
  operand loads deduped (the arithmetic-intensity reuse).

Each atom is a strategy class in **`_atom.py`** supplying `state` / `store` plus the descriptor reads the shared
`reduce` consumes — `gmem_leaves` (the four gmem-direct leaf constructors), `staged_drain` (the slab-reading leaf),
`slab_elem` (the slab element dtype) — with `_atom_ops` the dispatch + `reduce_codegen` / `store_sink` the seam
`_factor` calls: the new-atom seam. Staging eligibility + sizing are **not** an atom method: they resolved
scheduler-side into the stamped `Stage` (see Operand staging below).

The **unit** is the atom's parallel thread footprint (`atom.lanes`) — so the tensor-core warp tile and the scalar
parallel thread-tile are the *same* level, differing only in `lanes`; `block_threads = units · lanes`. `grid_tile` also
carries any leading (batch) grid axes and supports a 1-D (m-absent) output. (The store-glue helpers `with_store` /
`has_write`, shared by the constructor and the thread-binding tiers, live in `_factor.py`.)

## Operand staging — the warp-tier smem pipeline (`STAGE` codec → `Stage`)

The warp (mma) tier stages its reused gmem operands through an smem slab, driven off the node's `STAGE` codec →
`schedule.Stage`. Every staged path runs **one** K-loop skeleton, `staged_kloop` in **`_stage.py`**
(`fill → commit → wait → drain → Sync`, `depth` the sole buffering knob — `depth == 1` is the single-buffer degenerate,
`depth >= 2` a gmem→smem prefetch ring), behind a `Transport` strategy: `CpAsyncTransport` (fill → commit → wait-group)
and `TmaTransport` (an `arrive.expect_tx` + box copy gated by a **per-slot mbarrier array**, so `depth` is a free knob
for TMA too). The two producers — structurally different primitives — sit behind one `fill`/`commit`/`wait` seam, and
**one atom-agnostic driver** (`_atom._staged`) builds the operand pair + the transport for either atom; the atom
supplies only the slab drain leaf via `_AtomOps.staged_drain` (the shared inner `ldmatrix` drain
`_staged_inner_atom_loop`, or the scalar `_scalar_drain`). The staging **decision** does not live here at all: the
`Stage` on the `TileOp` arrives **already resolved** by the scheduler (`_schedule._resolve_warp_stage` /
`_resolve_scalar_stage` — transport eligibility, the slab K-chunk `bk_elems`, the depth clamps — or `None`,
gmem-direct), and `state` (which slots the operand fragments) and the shared `reduce` (which emits the loop) apply it
verbatim. The `Stage` spells two buffering levels:
`d<depth>` is the gmem→smem ring (cp.async commit group / TMA mbarrier-phased prefetch over the K-slab loop),
`p<reg_depth>` is the smem→register double-buffer (the `ldmatrix` ping-pong over the inner atom-K steps). Staging is a
**pure perf transform** — an ineligible kernel (transposed-B, masked N, symbolic / non-divisible K, or a computed-A
flash operand) silently falls back to gmem-direct, and a staged kernel is
**bit-identical** to its gmem-direct baseline. Unpinned, the schedule fork enumerates the resolver-gated stage grid
(`search/space.stage_moves`) alongside the tile / reduce moves; a `EMMY_STAGE` pin stays authoritative.

**The fused edge — the mma tier's `sync` transport.** A demoted-cone matmul (`f(x, …) @ w`) takes the warp tier
under a warp `TILE` pin: `_schedule._demoted_warp_option` nodifies the PLANAR ⊗-fold to a computed-A `Contraction`
(the same `a_operand = Body` flash P@V rides) and stamps a `sync` `Stage`; `_staged` then builds a `SyncTransport`
whose A fill is the producer CONE evaluated per slab cell (compute-fill) and whose B fill is a plain copy — the same
`fill`/`commit`/`wait` seam, single-buffer, one CTA barrier, feeding the unchanged `ldmatrix` drain. Pin-driven,
exact-cover geometry (static M/N/K divisible; masks are a follow-on). A **reduce-bearing (MONOID) cone** — the fused
norm→linear edge (`_schedule._prologue_warp_option`) — fuses too: the A cone carries its k-invariant prefix (the
per-row statistic reduce `Loop` + scalar epilogue), split off at the K seam by `_sync_operands`
(`_split_stat_prologue`) and run ONCE per tile row as the transport prologue (`_stage.sync_stat_fill` — threads
stripe the tile's rows, each folds its row's statistic into a stat smem row, one barrier); the per-cell compute-fill
reads the bridged values back from the stat rows. The prologue is a one-shot `SyncTransport.prologue`, ahead of the
staged K-loop.

The **scalar** contraction tier stages too, under the same `STAGE` codec, through the **same** `_staged` driver — the
scheduler's `_resolve_scalar_stage` sizes the slab (the depth-aware fit-to-smem K-chunk `bk_elems`, not a codec field;
the depth steps down when no chunk fits) and its `staged_drain` is the plain-`Load` inner loop (`_scalar_drain`,
reading the ring slot via the same slot-row seam as the mma drain). `depth >= 2` is the scalar gmem→smem prefetch
ring — the identical `staged_kloop` cp.async / TMA-mbarrier phases the warp tier runs; only `p<n>` (the
smem→register double-buffer) stays warp-only (an `ldmatrix` transform). The nested outer-slab / inner-drain
accumulator lifetime is handled by seeding the per-cell accumulators once in `_ScalarOps.state` (outside the outer
loop) and marking the inner drain `Loop(seed=False)` so it folds without re-declaring. Masked M / N is supported (TMA
zero-fills / cp.async clamps; the drain indexes the slab by LOCAL tile coords). Unstaged is byte-identical
gmem-direct.

**Split-K composes with staging.** `_splitk_option` resolves a `STAGE` spec against the SLICED inner `Contraction`
(the `kslice` extent + the `ksplit`-offset operand indices) and `030_split` threads the resolved `Stage` onto its
partial `TileOp`s, so the partial kernel's K-loop stages its slice through the same pipeline (the TMA box origin is
the operand's own index evaluated at the tile base — an offset operand lands the box at absolute coordinates).

## The fragment realizer (`_twist.py`) — a TWISTED carrier at warp-fragment residence

A `TWISTED` streaming reduce whose contractions carry mma `TilePlan`s (stamped by
`_schedule._twisted_warp_option` — tensor-core flash) realizes at FRAGMENT residence: `_bind`'s reduce arm keys on the
structural warp-tile read (`_twist.warp_source`) and `realize_warp_twist` produces the `(state, fold, close)` triple
the one pipeline seals, the kernel warp-collective through the same `lanes` parameter `grid_tile` already takes. This
is the placement-keyed fold's **fragment row**: where the scalar tier folds in-thread, the fragment tier's per-block
fold is a `FragmentRowReduce` `__shfl` butterfly over the C-fragment lanes. The fold MOVE itself is never re-decided
per site: `ReduceStage.combine` (`ir/schedule.py`) is the ONE placement-keyed selector — within-warp → `SHFL`,
within-block → `SHFL`+`SMEM` tree, cross-CTA → `ATOMIC`/`KERNEL` — and every emitter consumes its output
(`emit_combine` at scalar residence, this realizer at fragment residence, `030_split` as the graph rewrite); only the
residence-specific realization differs. Everything realizes from structure — the
head contraction's `ldmatrix`/`mma.sync` off its node geometry (`_frag_contraction`); the score prologue stmt-by-stmt
(`Assign` → `FragmentApply`, a coordinate `Select` → `FragmentMask` with the keep-predicate negated, loop-invariant
constant `Load`s hoisted); the streaming merge REGENERATED from the carrier's channel spec (pivot → rowmax + running
stats + α-rescale; denom → rowsum; the expect channel's ⊗ `lift` IS the P@V node, its register-resident A operand fed
through the `flash_pv_smem` C→A smem handoff); the projection tail as an in-place `FragmentApply` + the `RegStore`
close. Symbolic seq masks at the fragment (`FragmentMask(col ≥ seq)`) with the gmem reads clamped; causal composes as
another mask. No kernel-identity dispatch anywhere — an unrealizable tree is rejected at schedule time (the additive
`(m, kv)` score bias, a non-exp family), never here.

**Shared-row staging (`_tile_reduce_axis`) — the reduce tier's `sync` transport.** The fused norm→linear prologue is a
cooperative reduce: an input row folded by the cooperative reduce AND re-read per output column of a contraction tail (a
free-axis `Loop` over an inner reduce). Like the contraction tiers, it is **`Stage`-driven**: the scheduler
(`_schedule._row_stage`, narrow so a plain softmax sum or a bare reduction is untouched) detects that one row and stamps
a depth-1 `sync` `Stage` whose `smem` names it — a derived schedule field, never a knob. `_tile_reduce_axis` only *applies*
it: the row is filled cooperatively via `_stage.sync_row_fill` — the **same `_stage.py` fill module** the warp tier's
cp.async / TMA fills live in, indexed off the same linear-tid / thread-count seam — and both readers are rewritten to
the slab (`_restage_loads`). So every staging decision rides a `Stage` on the schedule and every transport (`sync` 1-D
row · cp.async / TMA 2-D slab) lowers through one module. A contraction operand `Stage` never sets `smem`, which is how
the two apply paths stay distinct on a coop-K contraction.

## Kernel-IR peepholes

`030_stamp_types` / `040_demote_to_write_dtype` resolve element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `110_drop_redundant_syncs` collapses the defensive `Sync`s the
cooperative / shared-row templates emit (body-level only — a slab `Smem` decl flags `smem_seen`, so a load-bearing
prologue `Sync` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).
