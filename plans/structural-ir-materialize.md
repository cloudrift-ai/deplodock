# Scheduled structural tile IR — Map / Reduction / Contraction, fork-tree construction, walk-to-emit

**Branch:** `refactoring/tile-ir-rebuild` · **Status:** in progress — migration steps **1–4 landed**
(each e2e-green + parity-proven), steps **5–6 remain**. Consolidates and supersedes the earlier
`schedule-enumeration-tree` note (the fork tree is now *how* recognize builds this IR).

**Progress (see Migration section for the per-step contract):**

- ✅ **Step 1** — `Contraction` lifted to `ir/tile/structural.py` (mechanical move).
- ✅ **Step 2** — `Reduction` node introduced; recognize builds it for `PLANAR`/`TWISTED`; `_reduce`
  expands it via `ops.lower`. Byte-identical `op_cache_key` parity proven.
- ✅ **Step 3** — `Map` moved into `ir/tile/structural.py` and given an optional `source: Reduction
  | Contraction | None` (`project ∘ reduce`); a projected reduce is `Map(body=sweep, source=Reduction)`,
  a bare reduce a root `Reduction`. `ops.lower`/`reduce_loop`/`axis_role` recurse through `Map.source`.
- ✅ **Step 4** — `005_contract` deleted; the `Contraction` build folded into `010_materialize`
  (`_build_contraction`) — the plan's "lazy resolve in the expander" alternative. `TileOp.op` stays a
  `Map` this step, so cache keys + `030_split` are untouched.
- ⏳ **Step 5** — recognize emits the fork tree of structural leaves; **dissolve `TileSchedule`** (move
  `reduce`→`Reduction`, `place`→`Map`, `tier`/`bind`→`Contraction`). The big one — touches `fork.py`,
  `search/`, the knob system. Byte-identical parity over the kernel fixture is the bar.
- ⏳ **Step 6** — compose `Reduction ⊃ Contraction` (flash nesting + split-K warp tier + multi-axis
  planar); new e2e `test_mma_splitk_finalize` / `sum` over `(H, W)`.

Note: this section records where the migration actually stands; the design below is the target. When
steps 5–6 land, delete this plan (per the Contribution Instructions).

## Goal

Make the **stored tile IR a tree of scheduled structural nodes** — `Map` / `Reduction` /
`Contraction` — each holding its own **scheduling parameters** (grid `Placement` / `ReducePlan` /
`TilePlan`). `010_recognize` recognizes the algebra and then emits a **fork tree** of scheduling
choices whose every **leaf is a complete structural-IR tree** with those params baked onto the nodes
(ranked lazily by the learned `Prior`). `005_contract` is **deleted** — `Contraction` is constructed
at recognize-time, lifted up to tile IR. `010_materialize` collapses from a transformation pass to a
**walk + expand**: dispatch on node type, read the schedule off the node, synthesize the renderable
kernel-IR the CUDA backend already walks via `Stmt.render`.

The pivot from the just-landed *annotated-loop* form (`Map(body=[annotated Loop])` + a separate
`TileSchedule`): the reduce/contraction structure **and its schedule** become typed nodes, so the
schedule is no longer a monolithic sidecar struct and materialize no longer re-recognizes anything.

## The structural-IR vocabulary (3 nodes, lifted to `ir/tile/`)

Each node follows the **`Contraction` discipline already proven in `ir/kernel/ir.py`**: a small set
of *param fields* + *one schedule slice* + **derived geometry as `@property`** (so `structural_key`
digests only the compact fields and the `--ir` dumps stay readable). Composition is by **nesting**
(a node's `source` / operand is another node) — the recursion the retired `Map`/`Monoid`/`Semiring`
op-tree had, but now every node is the **post-schedule** form, not an algebra tag.

- **`Map`** — the pointwise lift / projection wrapper (+ the kernel's grid schedule).
  - params: `body: Body` (per-cell pointwise stmts + the output `Write`); `source: Reduction |
    Contraction | None` (the reduction it projects over — `project ∘ reduce`; `None` = pure pointwise).
  - schedule: `place: Placement` (free → grid binding; later: free-axis register tile).
- **`Reduction`** — a `PLANAR` / `TWISTED` reduce (the scheduled successor of the old `Monoid`).
  - params: `carrier: Carrier`, `axis: Axis`, `source`/`body` (the per-element **partial** it folds —
    a buffer `Load` for a plain reduce, a nested `Contraction` for flash's score).
  - schedule: `reduce: ReducePlan` (serial / `b` coop / `r` reg / `g` split). The cross-partition
    **Combine is derived**, not a node — the expander emits `emit_combine(carrier, plan)` from the
    carrier's `combine_states` + the level→fold mechanism (so `Combine` folds *into* `Reduction`).
- **`Contraction`** — the tiled `⊗∘⊕` (the scheduled successor of the old `Semiring`; the existing
  node, moved `ir/kernel` → `ir/tile`).
  - params: `axes (m,n)`, `k_axis`, `a_load`, `b_load`, `acc`, `epilogue`, `lead_axes`.
  - schedule: `tile: TilePlan` (scalar register tile OR warp mma) + `bind: AtomBinding` (resolved at
    construction — `_atomize.semiring_binding` moves recognize-side). **Staging deferred** (see below).

### How kernels compose (nesting = the structure)

| Kernel | Structural tree |
| --- | --- |
| pointwise | `Map(body, source=None, place)` |
| sum / max | `Reduction(carrier=id, axis, source=Load, reduce)` (bare; grid `Write` is glue) |
| softmax / RMSNorm | `Map(body=[normalize, Write], source=Reduction(carrier=exp/id, axis, …), place)` |
| matmul | `Contraction(a,b,acc,epi, tile)` (bare) or `Map(source=Contraction, …)` for a fused epilogue |
| flash | `Map(body=[O/l, Write], source=Reduction(carrier=exp(m,l,O), axis=kv, source=Contraction(QK)))` |

Flash makes the payoff concrete: warp-flash later is just *the inner `Contraction` gets a warp
`TilePlan`* and *the outer `Reduction` gets a split-KV `ReducePlan`* — no new code path.

### Staging (deferred — do NOT build now)

`Stage` is **not** a node and not in scope this refactor. When it returns it is a **field**, either a
per-operand stage entry on `Contraction`'s operands (the expressive target — A/B stage
independently) or a single `stage` param on `Contraction` (the trivial lift of today's
`TileSchedule.stage`). Until then `factorize` keeps its gmem-direct path and the existing `STAGE`
pin still rides the legacy schedule field.

## recognize → a fork tree whose leaves are structural IR

`010_recognize` keeps step 1–3 (flash / online-softmax / the structural skeleton via `ops`), then —
replacing today's flat `_schedule.schedule()` candidate list — emits a **lazy fork tree** of
scheduling choices, **each axis offering permitted moves keyed on its `AxisRole`**, producted with
incremental legality pruning. Every **leaf** resolves to the structural-IR tree above with the chosen
params on the nodes. The consume side is REUSED unchanged: `fork.py::build_fork_tree` (knob-rows →
lazy tree) + `search/` (`LazyCandidate`, the greedy/MCTS policies ranking the frontier with the
learned `Prior`; cold fallback = emission order, conservative-pick-first).

Permitted-move catalog (emits the **same codec strings** the knobs already parse, so `ReducePlan` /
`TilePlan` / the prior featurizer / the perf DB are untouched — the move is just the *generator*):

| Axis role | Permitted moves (→ which node param) |
| --- | --- |
| `FREE` (output) | grid-map → `Map.place` (later: scalar tile) |
| `PLANAR` / `TWISTED` | serial · `b<n>` · `r<n>` · `g<n>[a\|k]` → `Reduction.reduce` |
| `CONTRACTION` (m,n)+K | scalar `n../f..` · warp `a:<atom>/w../f../k<bk>` → `Contraction.tile`; `g<n>` split → a `Reduction` wrapping the `Contraction` (Phase-4 compose) |

Legality guards (lift today's piecemeal checks into per-move guards, evaluated **incrementally** so
illegal subtrees never generate): scalar `block_threads ≤ 1024`; warp K-step `atom_k·bk` divides a
**static** K; no cross-CTA split of a symbolic reduce; env pins win via `Knob.narrow`. Blowup control:
incremental legality (dominant), the lazy tree + a beam/budget cap, conservative emission order.

## `TileSchedule` dissolves into per-node params

The monolithic `TileSchedule(place, block, reduce, tier, stage, workers, bind)` is decomposed:
`reduce` → `Reduction`, `tier`/`bind` → `Contraction`, `place` → `Map` (root), `stage`/`workers` →
deferred. `Kernel(op, schedule)` collapses — a `TileOp` holds just the structural-IR root. (Decision
below: whether a thin root `place` survives on `TileOp` or moves entirely onto `Map`.)

## materialize = walk + expand (one expander per node)

`010_materialize` becomes a dispatch-on-node walk; each expander reads its schedule slice off the node
and synthesizes the renderable kernel-IR (the reduce/contraction loop is **synthesized, never
stored**, exactly as `Contraction` does today):

- `Map` → bind the grid (`Tile`), emit `body`, recurse into `source`.
- `Reduction` → the current `_reduce` logic as the expander: serial fold, or the `StridedLoop` +
  `reg`-replication + REG tree (`carrier.as_state_merge`) + cross-thread `emit_combine`; recurse into
  the partial `source`.
- `Contraction` → the existing `_factor.factorize` (mma fragment soup / scalar register tile).

`005_contract` is **deleted** (its `Contraction` build + `semiring_binding` move into recognize).
`030_split` is **kept** — a `Reduction`/`Contraction` whose `ReducePlan` carries a `GRID` stage still
triggers the cross-CTA graph rewrite (partial + finalize), now reading off the structural node. The
backend's `render_body` walk is unchanged — "generate" is the existing `Stmt.render` recursion.

## Reconciliation with the retired Monoid/Semiring work

Not a revert: `Carrier` / `Twist` / `StateMerge` / the carrier-generic combine **survive** (a
`Reduction` holds a `Carrier`). The difference from the old `Monoid`/`Semiring` nodes is that
`Reduction`/`Contraction` are the **scheduled** form (they carry `ReducePlan`/`TilePlan` and are the
fork-tree leaves), so a node type now legitimately encodes a *realization choice*, not an algebra tag
to keep in sync — recognition still reads the *algebra* structurally off the loop body before
constructing the typed scheduled node. `ops.axis_role` / `reduce_loop` / `contraction_loop` become
construction helpers (loop body → the structural node); `lower(op)` is subsumed by the expanders.

## Migration (each step e2e-green; recovery contract governs)

1. **Lift `Contraction` to `ir/tile/`** (move the class; `005` still builds it). Mechanical, no
   behavior change.
2. **Introduce `Reduction`** as a tile-IR node (carrier + axis + partial + `ReducePlan`); recognize
   builds it for `PLANAR`/`TWISTED` instead of the annotated `Loop`; `_reduce` becomes its expander.
   Parity-test the emitted candidates against today.
3. **Introduce `Map`** as the scheduled pointwise/projection node (`place` on it); recognize wraps
   reductions/contractions as `Map(source=…)`; materialize's scalar/grid path reads `Map.place`.
4. **Move `Contraction` construction into recognize, delete `005_contract`**; materialize's
   `KernelOp(Contraction)` arm becomes the tile-IR `Contraction` expander.
5. **recognize emits the fork tree** of structural leaves (the move catalog + legality); dissolve the
   flat candidate list and `TileSchedule`. Byte-identical parity over the kernel fixture.
6. **Compose `Reduction ⊃ Contraction`** — split-K on the warp tier (Phase 4) and multi-axis planar
   (Phase 5) fall out as nesting; new e2e `test_mma_splitk_finalize` / `sum` over `(H,W)`.

## Verification

- `tests/compiler/e2e/` green at **every** step; `tests/xfail_registry.py` only shrinks; `make lint`.
- Parity tests at steps 2 & 5 (emitted leaves / candidate `knobs` byte-identical to the prior form).
- A structural-coverage test: the fork-tree leaf set over a matmul fixture equals the hand-computed
  legal product (so a missing/extra move is caught).

## Open decisions (resolve before step 1)

- **Root `place`** — does `TileOp` keep a thin root `Placement`, or does `place` live entirely on the
  root `Map` (and a bare `Reduction`/`Contraction` root carry its own)? Lean: on the root node.
- **`bind` timing** — resolve `AtomBinding` at construction (recognize) for both tiers, or keep a
  lazy resolve in the `Contraction` expander? Lean: at construction (the unbindable-atom error
  surfaces at fork build, as `_warp_option` does today).
- **Staging shape when it returns** — per-operand stage entries vs. one `Contraction.stage` param
  (deferred; do not build now).
