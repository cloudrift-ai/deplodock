# TileOp as a clean Schedule render: high-value boundary fixes

`assemble(TileGraph + Schedule) -> TileOp` is the right shape for the final vision: `TileOp` is a readable tower IR, and
delegating mechanical lowering to the kernel IR (`lowering/kernel/`) is a sound division of labor. Going straight to
`KernelOp` from assemble buys little. **This plan does not change that boundary — it makes it clean.**

The problem is not that `TileOp` exists; it is that the assemble→`TileOp` boundary is **leaky in two directions**, which
lets the brittleness the block-DAG refactor ([`tile-ir-block-dag.md`](tile-ir-block-dag.md)) eliminated *upstream*
quietly re-accrete *downstream*:

- **Forks leak down.** A benched scheduling choice (vectorization width) is made by a tree pass on `TileOp`, not by the
  search on the `Schedule`.
- **Schedule facts re-implicitize.** assemble renders explicit `Schedule` fields (`cohort`, `distance`, `ring_depth`,
  `role`) **back into implicit tree shape** (`StageBundle` nesting, peeled pipeline, `WarpSpecialize` `Cond`). Any
  downstream pass that needs one of those facts must pattern-match the tree to recover it — the exact "schedule lives in
  the tree shape" brittleness, relocated from between tile passes to between kernel passes.

There is also a **frontier ceiling**: `TileOp` is a single binding tree, so it cannot render a single-kernel *multi-phase*
structure (persistent / cooperative `grid.sync`, the edge-placement barrier mechanism, two-nest warp-spec). That bounds
the schedule space silently.

## The discipline this restores

> Every `lowering/kernel/` pass is **mechanical**: it lowers the tree, it never *makes* or *re-derives* a scheduling
> decision. Every benched choice lives in the `Schedule` (decided by the search, above assemble); every schedule fact a
> kernel pass needs is **stamped explicitly** on the `TileOp` node, never recovered from tree adjacency.

`TileOp` stays the render; the `Schedule` stays the single source of truth *through* the render.

## Fix 1 — Stamp resolved schedule facts on `TileOp` nodes (no re-derivation) [highest value]

assemble already *knows* every schedule fact when it builds the tower; it currently throws the explicit form away and
keeps only the tree shape. Instead, carry the resolved facts as **explicit node attributes**:

- `StageBundle` ← `cohort` id + barrier id (which loads share one `__syncthreads` / mbarrier).
- `SerialTile` (pipelined) ← `ring_depth`, ring slot, `distance` per axis (so the peel is *labeled*, not inferred).
- `WarpSpecialize` ← `role`, `reg_budget`, producer/consumer partition (already mostly explicit — finish it).
- staged `Load` ← its `Transport` + slab id.

Then any kernel pass that needs a fact **reads the attribute**; none pattern-matches the tree. This keeps "the schedule
is explicit" alive across the assemble boundary — the cheapest, most broadly protective fix, and it touches only
`assembly/_assemble.py` (stamp) + the few kernel passes that today re-derive (read).

**Why first:** it is the structural firewall. Until facts are stamped, Fix 2's "mechanical kernel passes" rule is
unenforceable (a pass *must* pattern-match to do its job), and every new kernel pass risks re-introducing tree surgery.

**Gate:** a lint — `lowering/kernel/` imports nothing from `enumeration/`'s offer/knob/move modules; a pass may read a
stamped `TileOp` attribute but not call a classifier. Plus a test: assemble stamps every fact its kernel passes consume
(no kernel pass reads tree *shape* for a schedule decision).

**Landed (the firewall — Gate):** `tests/architecture/test_layering.py` now carries the three guards
(`test_lowering_kernel_does_not_import_enumeration` — no kernel pass imports `tile/enumeration/` or `tile/split/`;
`test_lowering_kernel_calls_no_schedule_classifier` — no kernel pass calls an offer / classifier / cut fn, allow-listing
the mechanical `classify_fragment_epilogue`; `test_materialized_tile_flavors_stamp_schedule_facts` — the facts kernel
passes consume are explicit fields: `StageBundle.{policy, buffer_count, phase, pipeline_depth}`, `SerialTile.kind`,
`WarpSpecialize.{ring_depth, n_producer_threads, consumer_thread_axes}`, `AtomTile.atom`). The **stamp** half was
already in place — the materialized typed flavors carry each consumed schedule fact as a typed attribute, and
every kernel pass already reads those attributes (verified clean: no kernel pass re-derives a decision from tree shape).
The remaining `Schedule` fields the bullet list names (`cohort` / `distance` / `reg_budget`, and `WarpSpecialize`) have
**no live producer** in the current pipeline — no enumeration pass sets them and no assemble path constructs a
`WarpSpecialize` (its emitting pass was deleted in the block-DAG refactor; warp-spec transport is future work). Stamping
those onto materialized nodes now would be speculative, so it rides the work that revives each producer. The firewall is
the durable deliverable: it makes Fix 2's mechanical-pass rule enforceable and blocks a new kernel pass from
re-accreting the leak.

## Fix 2 — Pull leaked forks above assemble (the real content of G3/G4) [high value]

The audit: which `lowering/kernel/` passes make a **benched choice** vs. apply a deterministic rule?

- **`050_vectorize_loads` / `080_vectorize_stores`** — vectorization width (v4/v2/v1) is a genuine fork (it shifts
  register pressure + predication; `070_pad`'s own gate already reads the realized `ld.shared.v4` width). → add
  `Schedule.vector_width` (an enumeration fork, env-pinnable `DEPLODOCK_VEC`), and demote the pass to **apply** the
  resolved width.
- **`060_permute_lane_accesses`, `095_interleave_loads`, `110_drop_redundant_syncs`** — audit each: a *deterministic
  peephole* (one output per input — stays a mechanical kernel pass) or a *ranked choice* (→ a `Schedule` field). Default
  expectation: these are mechanical; confirm, don't assume.

The principle is the fork test from `tile-ir-block-dag.md`: *does it offer ≥2 ranked candidates the search must bench?*
If yes, it belongs in the `Schedule` above assemble. This is what "G3/G4" actually means here — **not** "emit
`KernelOp`," but "stop making schedule decisions below assemble." The variant identity (`vector_width`) then keys the
perf DB / prior like any other knob.

**Gate:** every surviving `lowering/kernel/` pass passes the fork test as *mechanical* (one output per input), proven by
a determinism test per pass; the moved forks (`vector_width`) recover the same kernels they produced as a tree pass
(byte-identical), then become search-tunable.

## Fix 3 — Make the multi-phase ceiling explicit, not silent [frontier; guard now, build later]

`TileOp`'s single binding tree cannot render a single-kernel multi-phase schedule (persistent / cooperative `grid.sync`;
the edge-placement plan's cooperative barrier mechanism; a two-nest producer/consumer warp-spec). `Graph[TileOp]` covers
multi-*launch*; it does not cover multi-*phase within one kernel*.

Do **not** build the multi-phase node now. Do make the ceiling a **chosen, guarded** boundary:

- assemble **raises a specific error** when a `Schedule` requires within-kernel multi-phase (e.g. a `grid.sync` barrier
  mechanism, or `role` partitions that need disjoint nests), naming the unsupported field — never a silent mis-render.
- Reserve the shape: document `TileOp` as the **Ampere-class single-tower render**, and note the multi-phase node (or a
  drop to a richer form) as the sm_90+/persistent-kernel extension. The enumeration passes that *could* emit such a
  Schedule (the cooperative barrier mechanism) must not offer it until the render exists — the guard makes that a hard
  error if they do.

**Gate:** a test that the persistent/`grid.sync` Schedule path is *rejected with its named field*, so the search space
the IR actually supports is explicit and can't silently bound a tune.

## Smaller: variant identity should be the Schedule, `TileOp` its deterministic render

The perf DB / prior key on the materialized `TileOp`, not on `canonical(bodies+edges)+Schedule`. That makes `TileOp`'s
shape a *second* source of truth for identity. The RF determinism guard (same `TileGraph` → byte-identical `TileOp`)
already makes the render a function of the Schedule, so this is mostly closed — but the keying should derive from
`canonical(bodies+edges)+Schedule` directly (the `TileOp` is then provably its 1:1 render), so a future `TileOp`
shape-tweak can't silently re-key the prior. Fold into Fix 1's stamping work (the stamped Schedule facts *are* the key).

## Prioritization & sequencing

1. **Fix 1** (stamp + lint/test) — the firewall; unblocks Fix 2's discipline. Self-contained, low risk, no behavior
   change (additive attributes + a lint). **Landed** — the three layering guards; the stamp half was already in place via
   the typed flavors (see the Fix 1 "Landed" note above).
2. **Fix 2** (vectorization fork first, then the kernel-pass audit) — the one known leaked fork, then prove the rest
   mechanical. Behavior-neutral under byte-identical, then opens `vector_width` to tuning.
3. **Fix 3** (guard the multi-phase ceiling) — cheap guard now; the node itself rides the sm_90/persistent work whenever
   it lands (it is the same frontier the edge-placement plan's cooperative-`grid.sync` barrier mechanism needs).

## Non-goals

- **Not** killing `TileOp` or emitting `KernelOp` from assemble — the render + kernel-IR delegation is the intended final
  shape; these fixes preserve it.
- **Not** building the multi-phase / persistent-kernel node — only making its absence a hard, named boundary.
- **Not** moving *mechanical* lowering up — vectorization *application*, lane permutation, sync elision stay in the
  kernel IR; only the *fork* (which width) moves to the `Schedule`.

## Relationship to existing docs

- [`tile-ir-block-dag.md`](tile-ir-block-dag.md) — Fix 2 is the honest content of its deferred **G3/G4** ("the
  `→ KernelOp` switch + deterministic post-passes"): the goal is a clean fork/mechanical split at the assemble boundary,
  not a literal `KernelOp` return type. Fix 1 extends its RF "schedule is explicit / derived-view discipline" *through*
  assemble. The flagged backlog item "vectorization width as a Schedule field" is Fix 2's first target.
- [`dag-edge-placement-split-as-enumeration.md`](dag-edge-placement-split-as-enumeration.md) — Fix 3's multi-phase node is
  the render that plan's **cooperative `grid.sync` barrier mechanism** needs; until it exists, the barrier-mechanism field
  stays two-launch-only (its v1 scope), enforced by Fix 3's guard.
