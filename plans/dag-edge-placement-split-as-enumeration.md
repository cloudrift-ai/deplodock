# Edge placement: make split a block-DAG enumeration move (cross-kernel edges)

Fold the structural split (`005_split_demoted`) into the block-DAG enumeration as a generic **edge-placement** move,
instead of a bespoke pre-enumeration pass that does graph surgery on `LoopOp`s. The thesis of
[`tile-ir-block-dag.md`](tile-ir-block-dag.md) is "every scheduling choice is a `Schedule` annotation over an invariant
algorithm, applied by one deterministic `assemble`." Edge placement — *where the boundary between two compute blocks
lives* — is the one annotation still hiding in a hand-written `LoopOp` pass. This plan moves it into the Schedule.

## The idea in one table

Every DAG edge (a producer's output read by a consumer) has a **placement**. That placement is a search dimension, the
same way staging already is:

| placement                     | mechanism                                   | status today                            |
|-------------------------------|---------------------------------------------|-----------------------------------------|
| **inline / register** (fused) | edge stays in the consumer's `compute` body | the degenerate `FM=FN=1` fused matmul   |
| **smem** (staged)             | `Schedule.staged[edge]` slab                | `stage` (R1) — landed                   |
| **gmem** (cut)                | global buffer; grid barrier (two-launch *or* cooperative `grid.sync`) | `split_demoted` — bespoke `LoopOp` pass, two-launch only |

Once edge placement is one move family, **split and prologue-fusion are the same move in opposite directions** — where
you place the edge boundary plus how you materialize it. The tensor-core "demotion" question
([`tile-ir-block-dag.md` → "atomize"], and the split-vs-fuse tradeoff) stops being a special pass and becomes the search
benching two placements of one edge.

## Why now / why this is the right R7 shape

`tile-ir-block-dag.md` R7 calls for "`005_split_demoted` reborn." The current working tree has the **legacy pass restored
verbatim** into `lowering/tile/split/` (untracked, operates on `LoopOp`, docstring still references the deleted
`010_partition_loops` / `020_stage_inputs`). Adapting that legacy pass and *then* re-expressing it as enumeration is
double work. This plan is what "reborn" should mean.

Two concrete gains over the legacy `LoopOp`-surgery approach, beyond cleanliness:

- **Co-optimization across the cut.** The legacy pass emits separate fragment `LoopOp`s that re-enter enumeration blind
  to each other. A single-`TileGraph` cross-kernel-edge model lets the outer MCTS co-decide the producer's and
  consumer's tiling/staging/atom in one search.
- **Legality from derived views.** The cut's "is this worth offering" predicate (a consumer matmul stuck degenerate
  because an operand is a computed cone) is readable off `Block.atom` + `Block.reads` — not re-derived by `LoopOp`
  pattern-matching.

## What the IR already gives us (not a new concept)

The block-DAG already models multi-kernel DAGs; this plan *finishes* that, it does not invent it:

- `TileGraph.blocks` is a tuple; `Edge`s are **derived** from cross-block buffer def-use; `Schedule.launch: dict[str,int]`
  maps block → launch group ("one group = one kernel"); `assemble` partitions by group into `KernelOp`s with cross-group
  intermediates becoming graph tensors.
- `partition_reduce(K, GRID|SERIAL)` **already cuts** — R3's atomic-free split-K produces "two launch groups in one
  `TileGraph`, joined by a partial intermediate `Buffer` whose edge is derived." Split-K is the existing proof that a
  kernel-set-changing cut-as-move works; this plan generalizes its machinery.

The single-block restriction in `assemble_block` (it `raise`s on `len(blocks) != 1`) is the coexistence stopgap to lift,
not a design constraint.

## What dissolves vs. what relocates

`split_demoted` does four things. They generalize unevenly — the honest scope is "thin cut/fission move + derived offer
predicate," **not** "no split logic":

1. **Cut mechanism** — reassign `Schedule.launch` so an edge crosses kernels + materialize the intermediate buffer.
   **Dissolves cleanly**; it is exactly `partition_reduce`'s launch-group machinery. Unify them.
2. **Block fission** — pull an *inline* cone (rotary on Q/K, softmax-normalized P) out of the consumer's `compute` into
   its own producer block, rewriting the consumer to read the materialized intermediate. **A body move** (analogous to
   `partition_reduce` inserting a combine block), mechanizable as a generic `extract_block` — but real rewrite logic, not
   a no-op cut.
3. **Which edges to cut** (the offer set) — a fully general "any edge may cross kernels" fork is *legal* but explodes the
   outer tree with useless cuts. **Re-expressed, not eliminated**: the bespoke `LoopOp` cone-classification becomes a
   **derived-projection predicate** (`consumer.atom is None` / stuck at `FM=FN=1` because operand X is a computed cone).
   Cleaner, but the decision stays.
4. **Intermediate-layout choreography** ("K second-to-last for an N-reading cone so the consumer keeps the canonical B
   layout") — **generalizes to a clean rule but is real codegen**: materialize the edge buffer in the layout the
   consumer's `atom` / `AccessMap` requires.

So the deliverable is: **split becomes a thin cut + fission move over the DAG, its legality a derived-view query, its
profitability the search's job** — sharing `partition_reduce`'s launch-group machinery, framed as edge placement so
prologue-fusion falls out of the same mechanism.

## Design

### New / changed moves

- **`place_edge(edge, INLINE | SMEM | GMEM)`** — the unifying annotation. `SMEM` is today's `stage`; `GMEM` is the cut;
  `INLINE` is the default (no annotation). Legality:
  - `SMEM` — derived reuse across a parallel axis (already implemented in `enumeration/_stage.py`).
  - `GMEM` (cut) — the edge is materializable (its producer sub-body is extractable; the intermediate has a well-formed
    `AccessMap`); **offered** only when the derived demotion predicate fires (see below), to keep the outer fork tight.
    `GMEM` says *where the buffer lives*, not *how many kernels* — the kernel count is the barrier mechanism below.
  - `INLINE` — always legal for a fusible producer (the body already holds it).
- **`extract_block(cone)`** — the body move that fission realizes a `GMEM`/cut placement with: lift an inline
  sub-computation into a new `Block`, rewrite the consumer `Load` to the new intermediate `Buffer`, choose the
  intermediate's layout from the consumer's `atom`/`AccessMap`. Value-identical cones share one producer (CSE — a
  canonicalization, not a heuristic). Multi-accum K loops (gated-MLP gate+up) extract each accumulator into its own clean
  GEMM producer with the consumer rebuilt as the pointwise combine.

### Offer predicate — tier-monotonic cuts (the general gate)

Every existing cut is the **same** trigger: inlining a computed operand forces its consumer matmul below the warp tier,
and materializing the operand restores it (rotary `QK^T`, SDPA `P@V`, `o_proj` attn-out, gated-MLP down-proj — the
multi-accum extraction and the contiguizing copy are *fission shapes* once a cut is decided, not separate triggers). So
the offer rule generalizes to a **monotonicity check on a derived tier lattice**, not a matmul-specific pattern:

> Each `Block` has a derived **best achievable tier** — `tier(block)` — a function of its body + carrier + atom
> eligibility (`MAP` < `scalar-reduce` < `coop-reduce` < `warp-MMA` < …). A fused body is pinned to the *meet* of its
> blocks' tiers. **Offer a cut on edge `P→C` iff `tier(C | edge materialized) > tier(C | edge inline)`** — materializing
> the edge *strictly raises* the consumer's (or producer's) maximal tier.

Mechanically this is the existing `eligible_atoms` check run twice: operand inline (computed cone → no / degenerate atom)
vs. operand as a clean materialized `Load` (real atom). Offer iff they differ. One derived query, no per-op cone
classification — and any *future* tier a fused operand can demote (TMA, cluster, flash) is covered by the same test.

**Why it is tight — exactly today's cuts, no more:**

- **Necessary by construction.** The strict inequality fires only where fusion is *provably lossy*. A pointwise→pointwise
  edge has `tier(C)=MAP=tier(fused)` → no drop → no offer; a clean GEMM with plain-load operands is already max-tier
  inline → no offer. The rule cannot manufacture a spurious cut.
- **The discriminating case it gets right for free:** `RMSNorm → linear` (the fused blocked prologue we do **not** cut).
  The rmsnorm scale is a per-row pointwise factor that rides the load into smem *without* disturbing the `ldmatrix`
  layout → atom eligibility preserved → no tier drop → **no cut offered.** A *layout-changing or reduction* cone (rotary,
  softmax-norm) cannot preserve the atom layout inline → tier drop → cut offered. The line between "fusible pointwise
  prologue" and "demoting cone" falls straight out of `eligible_atoms`, not a hand-list.
- **Bounded offer count.** ≤1 offer per demoted-operand edge, ≤2 operands per matmul, value-identical cones dedup. The
  outer fork grows as O(#demoted operands), not O(#edges) — no blow-up.

Profitability stays the search's job: the predicate is a **necessary condition** (a tier gain is *available*), not a
verdict — materializing still pays a gmem round-trip that can exceed the gain, so the outer MCTS benches cut-vs-fused.
Tight offer set + measured decision.

The one thing to get right is the **precision of `tier` / `eligible_atoms` about fused-operand atom eligibility** —
pessimistic → over-offer, optimistic → under-offer. That is *one* derived function to make accurate (the same one R4's
tensorize already depends on), not N heuristics. Split-K / parallelism recovery stays a **separate** move
(`partition_reduce`, gated on grid underfill — a different lattice); folding it in here would offer *more* than we do
today.

### Cut-decision module — `enumeration/_cut.py`

All cut-offer logic lives in one helper module so it can be extended / tuned without touching the pass or `assemble`,
mirroring how `enumeration/_stage.py` owns the staging offer set and `_atom.py` owns atom eligibility:

- `tier(block: Block, *, edge: Edge | None = None, materialized: bool) -> Tier` — the derived best-achievable tier of a
  block, optionally with one in-edge forced inline vs. materialized. The single source of truth for "what would fusion
  cost here," built on `_atom.eligible_atoms` + the carrier. The `Tier` lattice (an `IntEnum` or ordered enum) is defined
  here so new tiers slot in by extending the lattice, not by editing call sites.
- `cut_offers(graph: TileGraph) -> list[CutOffer]` — the ranked offer set: for each derived `Edge`, evaluate the
  monotonicity predicate and emit a `CutOffer(edge, expected_tier_gain, fission_shape)` when it raises a tier. Ranking by
  tier gain (then a stable key) is the only ordering the search sees.
- `CutOffer` / `Tier` dataclasses — the typed offer the `lowering/tile/split` fork consumes; the pass itself is a thin
  shell (match the seed, call `cut_offers`, fork on the result, stamp `SPLIT_CONE`), holding **no** decision logic.

This keeps the "which cuts" policy in one auditable place: tightening the predicate, adding a tier, or (later) admitting a
non-tier trigger is a localized edit to `_cut.py` with its own unit tests, never a change to the fork pass, the fission
body move, or `assemble`. `eval`-style introspection (e.g. "what cuts does `_cut` offer for this graph and why") can call
`cut_offers` directly, the same way `eval knobs` introspects the knob schema.

### Sync scope vs. barrier mechanism (one kernel or two)

A `GMEM` edge does **not** imply two kernel launches. The kernel count is a *separate* decision, and it is set by the
edge's **dependency span across the grid**, which is **derived** — not chosen:

- Compare the producer-`Write` and consumer-`Read` `AccessMap`s under `Schedule.binding`'s GRID axes.
- **CTA-local** (same GRID projection for every element — consumer tile X reads only what the same CTA produced): no
  grid barrier; the edge can stay in **one kernel**. gmem here is only a register/smem spill, so `SMEM`/`INLINE` usually
  dominates — a CTA-local `GMEM` edge is rare.
- **Grid-crossing** (different GRID projection — any layout remap / transpose / cross-CTA reduction): needs a **grid-wide
  barrier**. This is where the demoted-matmul cuts land **by construction** — the cut exists to re-lay-out the
  intermediate into the consumer's canonical layout, i.e. a different CTA partition.

For a grid-crossing edge the barrier mechanism is itself a (transport-like) choice, **not** forced to two launches:

| mechanism                                | realization                                                                             | requires                                                                  | trade                                                                   |
|------------------------------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **two launches** (default)               | `Schedule.launch` separate group; the kernel boundary is the grid barrier               | nothing — always legal                                                    | launch overhead + cold gmem round-trip; byte-identical to today's split |
| **one cooperative kernel + `grid.sync`** | same `launch` group, a derived grid-barrier `Stmt` between producer and consumer phases | `cudaLaunchCooperativeKernel` + grid fits **resident** (occupancy-capped) | saves launch overhead, exploits L2 residency; caps grid size            |

This mirrors the machinery that already exists for `SMEM` edges (a `Transport` SYNC/CPASYNC/TMA + a `cohort`/barrier):
**every materialized edge has a sync scope derived from its dependency span, and a barrier mechanism that is a transport
choice.** The full lattice: thread-local → none; cross-thread-in-CTA → `__syncthreads` (+ the `SMEM` transport);
cross-CTA-in-grid → grid barrier via **{ two-launch | cooperative `grid.sync` }**.

**v1 scope:** grid-crossing cuts always take **two launches** (always legal, byte-identical to the goldens the perf DB is
tuned on). The cooperative-`grid.sync` mechanism is a **later** enumeration field, gated on cooperative-launch support +
resident-grid feasibility and benched against the two-launch baseline — it is the same sm_90+/persistent-kernel/stream-K
frontier `tile-ir-block-dag.md`'s backlog already reserves a persistent-loop flag for. The IR must therefore model the
barrier mechanism as its **own** field (not bake "GMEM ⇒ separate `launch` group"), even though v1 only ever picks
two-launch.

### Outer vs. inner MCTS

Unchanged from `tile-ir-block-dag.md`: a `GMEM` cut is **kernel-set-changing**, so it branches the **outer** tree (one
terminal per kernel set) — exactly where `005_split_demoted` already branches. `SMEM`/`INLINE` are inner per-kernel
moves. So this does not alter the two-level topology; it moves the cut's *implementation* from a `LoopOp` pre-pass to a
block-DAG outer fork. Both branches stamp the decision into `op.knobs` (`SPLIT_CONE: True/False`, the considered-vs-
declined idiom) so the perf DB / learned prior key on it and `policy/greedy._pick_structural` deploys the cut only when
the trained prior prices the kernel-set Σ cheaper.

### Assemble

General N-block / multi-launch `assemble`: partition `blocks` by `Schedule.launch`, topo-sort within and across groups by
the derived edge DAG, emit one `KernelOp` per group, materialize each cross-group edge as a graph-node intermediate
tensor (its `Buffer` shape/dtype derived from the producer `Write`). This is the undeclared deliverable
`tile-ir-block-dag.md` flagged ("multi-block `assemble` gates R2/R3") taken to its general form.

## Prerequisites & sequencing

- **[BLOCKER] General multi-block / multi-launch `assemble`.** R3's atomic-free split-K introduced *a* multi-launch path;
  confirm it generalizes to arbitrary N blocks + arbitrary derived edges before stacking cuts on it. If `assemble_block`
  still `raise`s on the general >1-block case, that is the first task — not the cut move.
- **Edge-placement unification of `stage`.** Re-key `Schedule.staged` (or add `Schedule.placement: dict[Edge, Placement]`)
  so `stage` and `split` are two values of one field. Keep `staged` as a compatibility view if cheaper.
- **Then** the cut + fission moves + the derived offer predicate, replacing `lowering/tile/split/`.

Order: (1) general multi-block `assemble` + a determinism/byte-identical guard; (2) `place_edge` field + fold `stage`
into it; (3) `extract_block` fission body move; (4) `enumeration/_cut.py` (the `tier` lattice + `cut_offers` predicate)
with its own unit tests, then the thin `GMEM` cut fork that consumes it; (5) delete the legacy `lowering/tile/split/`
restore; (6) de-quarantine the R7 structural tests.

## Gate

- **Goldens hold.** The legacy split produces specific kernel sets the per-GPU goldens are tuned on. The generalized cut
  must **reproduce-or-beat** every golden's kernel set, benched (`deplodock tune --dataset golden` / `run --bench`),
  before it replaces the pass. A kernel-set change that is not a measured win is a regression.
- **Accuracy-vs-torch** for every newly-cut graph (rotary `QK^T`, SDPA `P@V` split-consumer, `o_proj` attn-out, gated-MLP
  multi-accum), static and symbolic.
- **Outer-search parity.** The two-level MCTS enumerates the same kernel-set terminals the legacy pass did
  (`test_two_level.py::test_outer_branches_on_structural_fork`,
  `test_structural_push.py::test_split_demoted_fork_pushes_structural` de-quarantined and green).
- **Determinism** of the multi-block `assemble` (same `TileGraph` → byte-identical `KernelOp` set), per the RF invariant
  guard discipline.

## Risks & non-goals

- **Offer-set blowup.** A "cut any edge" fork would explode the outer tree. The tier-monotonicity predicate
  (`_cut.cut_offers`) is what keeps it tight — it offers only on a provable tier gain, so O(#demoted operands) not
  O(#edges). It is the relocated form of the split logic, not removable; its tightness is exactly the precision of
  `_cut.tier` / `eligible_atoms`, which therefore needs its own golden test. Log any cut the predicate declines.
- **Layout choreography is real work.** Item 4 above (materialize-in-consumer-layout) is the part most resistant to
  "just a cut." Budget for it; it is derivable from the consumer `atom` but it is codegen, not an annotation.
- **Not the perf frontier.** A `GMEM` cut still pays a gmem round-trip of the intermediate. The strictly-better placement
  for many cases is `INLINE` *with* a non-degenerate fused-tensor-core prologue (no round-trip) — which this refactor
  makes *reachable as a search option* but does **not** itself implement. Growing the fused-prologue-into-warp-tile
  codegen (so `INLINE` is non-degenerate on computed operands) is the separate, harder follow-up; flash (R6) is the proof
  it is buildable. For attention specifically, the on-chip fused path (flash) dominates the cut at long sequence; the cut
  remains the non-flash fallback.
- **Do after R6 closes, as the R7 deliverable.** Not a separate big-bang refactor ahead of the tiers — it rides R7 and
  replaces the restored-legacy scaffold.

## Relationship to existing docs

- [`tile-ir-block-dag.md`](tile-ir-block-dag.md) — the parent refactor. This plan realizes its `Schedule.launch` /
  multi-block-`assemble` / "`005_split_demoted` reborn" line items as a single edge-placement move family. The R7 bullet
  there should point here.
- [`sdpa-n-axis-detection.md`](sdpa-n-axis-detection.md) / [`atomic-free-streamk.md`](atomic-free-streamk.md) — the
  demoted-matmul cases (rotary `QK^T`, SDPA `P@V`, split-K) the cut move must cover.
- The (deleted) perf backlog item "vectorization / warp-shape / fused-prologue as Schedule fields" — the `INLINE`
  non-degenerate path is the same frontier; track it separately.
