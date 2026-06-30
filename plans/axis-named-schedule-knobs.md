# Dissolve `TileSchedule` — structural nodes, fork-tree construction, axis-named knobs (all in `knob.py`)

**Branch:** `refactoring/tile-ir-rebuild` · **Status:** in progress. The **single** plan for the
remaining tile-IR-rebuild work (it replaced the earlier `structural-ir-materialize` and
`schedule-enumeration-tree` notes; that plan's steps 1–4 have landed — see below). Four coupled moves,
one parity bar:

1. **Dissolve `TileSchedule`** — the schedule slices move onto the structural nodes
   (`Reduction.reduce` ✅, `Contraction.tile`/`bind`, node `stage`) + a thin root holder
   (`place`/`workers`). `Kernel(op, schedule)` collapses; a `TileOp` holds the structural-IR root.
2. **Fork-tree construction** — `010_recognize` emits a lazy fork tree whose **leaves are
   fully-scheduled structural trees** (replacing the flat `_schedule.schedule()` candidate list), via
   the move catalog keyed on `AxisRole`. `materialize` becomes a walk + expand.
3. **Axis-named knobs** — the schedule codecs key `@<axis>` (the node's reduce/contraction axis) so a
   multi-node kernel (flash) can address each node; bare names stay first-class for single-node kernels.
4. **All knobs in `knob.py`** — every `Knob` definition (the native moveset *and* the schedule codecs
   `REDUCE`/`TILE`/`STAGE`/`WSPEC`, today defined in `_schedule.py`) is consolidated into `knob.py` so
   the whole knob surface is visible in one file.

## What has already landed (each e2e-green + `op_cache_key`-parity-proven)

The structural-node vocabulary exists and recognize builds it; the **scheduling/fork/knob side** remains.

- ✅ `Contraction` lifted to `ir/tile/structural.py`.
- ✅ `Reduction` node (carrier + reduce `axis` + `partial`); recognize builds it for `PLANAR`/`TWISTED`.
- ✅ `Map` moved into `ir/tile/structural.py` with `source: Reduction | Contraction | None`
  (`project ∘ reduce`): bare reduce = root `Reduction`, softmax/RMSNorm = `Map(body=sweep,
  source=Reduction)`, pointwise = `Map(source=None)`. `ops.lower`/`reduce_loop`/`axis_role` recurse
  through `Map.source`.
- ✅ `005_contract` deleted — the `Contraction` build folded into `010_materialize`
  (`_build_contraction`, the lazy-resolve-in-the-expander cut). **`Contraction` is built at materialize
  today, not recognize** — the load-bearing seam #1 below.
- ✅ The `reduce` partition moved off `TileSchedule` onto the `Reduction` node, read via
  `ops.reduce_plan` (falls back to `schedule.reduce` for the two not-yet-node forms: a non-tiled
  contraction's split-K, and flash's legacy loop-in-body `Map`).

## Structural nodes + field homes (the dissolution, seams resolved)

`TileSchedule(place, block, reduce, tier, stage, workers, bind)` decomposes into **node-local** slices
and a thin **root-global** holder. The codec *classes* (`TilePlan`/`ReducePlan`/`Stage`/`WarpSpec`) +
their `parse`/`spell` are **kept verbatim** — only their home (and knob key) changes.

| Field | Home | Notes |
| --- | --- | --- |
| `reduce` (`ReducePlan`) | `Reduction.reduce` ✅ | read via `ops.reduce_plan` |
| `tier` (`TilePlan`) | `Contraction.tile` | needs the node built recognize-side (seam #1) |
| `bind` (`AtomBinding`) | `Contraction.bind` | resolved at construction (seam #1) |
| `stage` (`Stage`) | the scheduling node | pin-only this cut (materialization still reserved) |
| `place` (`Placement`) | **root-global** (`TileOp.place`) | CTA-wide — one grid per kernel |
| `workers` (`WarpSpec`) | **root-global** (`TileOp.workers`) | CTA-wide — one warp split per kernel |
| `block` | fold into `place` | strided-coop rows; revisit with the planar set |

**Seam #1 — `tier`/`bind` → `Contraction` requires the node built recognize-side.** Step 4 folded the
build into `010_materialize`, so the node doesn't exist until materialize and its `tier`/`bind` can't
ride it earlier. Resolution: **move the build to the fork-emit (recognize) side** — the fork-tree leaf
for a tiled contraction *is* a `Contraction` node with `tile`+`bind` resolved
(`_atomize.semiring_binding` at fork build, the unbindable-atom error surfacing there as `_warp_option`
does today). `_build_contraction` retires; materialize only `factorize`s. The synthesized grid-`Write`
for a bare contraction stays a materialize concern (needs `root.output`).

**Seam #2 — `place`/`workers` home once `Kernel` collapses.** `Kernel(op, schedule)` collapses; `TileOp`
holds the structural root **plus** thin `place: Placement` and `workers: WarpSpec | None` (the only
`TileSchedule` survivors — both CTA-global). Per-node slices ride the nodes.
`tests/compiler/e2e/test_matmul_coverage.py` reads `tile_op.kernel.schedule.workers`/`.stage` today —
repoint those accessors to `tile_op.workers` / the node's `stage`; assert the same values.

**Seam #3 — knob renaming shifts `op_cache_key`.** `op_cache_key` digests `sorted(knobs.items())`, so
`@<axis>` suffixes change the tile-op cache key for every scheduled kernel. **Intended** — the DB/search
re-keys onto the axis-named identity (the transfer handle), not a regression. The parity bar is the
**emitted kernels** + the **featurizer vector** (both byte-identical); the cache-key shift is expected.

How kernels compose (nesting = the structure; every node carries its schedule slice):

| Kernel | Structural tree |
| --- | --- |
| pointwise | `Map(source=None)` + root `place` |
| sum / max | `Reduction(carrier=id, axis, partial=Load, reduce)` (bare; grid `Write` is glue) |
| softmax / RMSNorm | `Map(body=[normalize, Write], source=Reduction(carrier=exp/id, axis, reduce))` |
| matmul | `Contraction(a,b,acc,epi, tile, bind)` (bare) or `Map(source=Contraction)` for a fused epilogue |
| flash | `Map(body=[O/l, Write], source=Reduction(exp(m,l,O), axis=sk, source=Contraction(QK, tile@d)))` |

Flash is the payoff: warp-flash is just *the inner `Contraction` gets a warp `TilePlan`* and *the outer
`Reduction` gets a split-KV `ReducePlan`* — no new code path, the nesting carries it.

## The decision (settled)

Schedule knobs gain an **axis-named family** — `TILE@<axis>`, `STAGE@<axis>`, and the reduce partition
keyed by its axis — addressing each schedule-bearing node by **the reduce/contraction axis it schedules**.
The **bare form stays first-class**: `TILE` / `REDUCE` / `STAGE` with no `@<axis>` resolve to the *unique
suitable axis* for that family (see "Bare names" below), so the common single-node kernel — and every
existing pin / recipe / golden — keeps working unchanged. The suffix is required only to disambiguate a
kernel with two eligible nodes (flash). We evaluated three options:

1. **Flat positional** — one key per family, an ordered list bound to nodes by a canonical walk. Rejected:
   "next relevant axis" still has to compute a stable order, and the only retune-invariant order is the
   axis itself — so positional is axis-naming with the stable part discarded. No transfer handle for the
   prior; needs a frozen QK-before-PV tiebreak.
2. **Hierarchical / nested** — a knob value tree mirroring the structural tree. Deferred: breaks the
   flat-dict invariant `fork.py` / the DB / the prior lean on ("the row IS the variant identity"). Revisit
   only when a param is scoped *under another param*, not just under a node — which we don't have yet.
3. **Axis-named** ✅ — key by `Reduction.axis` / `Contraction.k_axis` (both confirmed present, and `Axis`
   carries a stable `name`). Chosen.

Why axis-named wins, concretely on flash (QK contracts over `d`; the online reduce and PV both ride the
streaming key axis `sk`):

- **Self-disambiguating.** QK and PV have different k-axes (`d` vs `sk`), so the two contractions key
  apart with no artificial tiebreak — the open disambiguator problem dissolves.
- **A transfer handle for the prior.** `TILE@d` (a head-dim contraction) keys the same learned signal
  across every model with that axis; positional `TILE[0]` means a different beast in a 1-contraction GEMM
  vs 2-contraction flash, so the prior can't pool.
- **It surfaces the fusion seam.** `TILE@sk` (PV) and `REDUCE@sk` (online reduce) share `sk` — *that
  shared axis is why flash fuses PV into the streaming reduce*. Family+axis is the full key
  (`TILE@sk` ≠ `REDUCE@sk`, no collision), so the relationship is visible in the knob set, not hidden.
- **Least-surprising.** The native moveset already addresses this way — `SPLIT@<axis>`, `REDUCE@<axis>`
  (see `analytic.py:215` `REDUCE@<k>.cta`), `ATOM@<axis>`, `PLACE@<element>` — and `knob_sort_key` /
  `_FAMILY_ORDER` already parse `FAMILY@element`. This is **unifying the schedule codec knobs onto the
  existing native convention**, not new syntax.

## Global vs node-local — the cut that keeps it honest

The schema partitions into two tiers. Not everything decentralizes:

| Knob                                | Keying                                                  | Lives on       |
|-------------------------------------|---------------------------------------------------------|----------------|
| reduce partition (`REDUCE` codec)   | `@<axis>` (the `Reduction.axis` / `Contraction.k_axis`) | the node       |
| `TILE` (output fragment / mma tile) | `@<k_axis>` (identifies the `Contraction`)              | the node       |
| `STAGE` (operand pipeline)          | `@<axis>` (one per reduce loop)                         | the node       |
| `bind` (`AtomBinding`)              | `@<k_axis>`                                             | the node       |
| `PLACE` (free → grid)               | **root-global** (one grid per CTA)                      | the root `Map` |
| `WSPEC` (warp split)                | **root-global** (one warp partition per CTA)            | the root       |

`PLACE`/`WSPEC` are CTA-wide — one grid, one warp split for the whole kernel — so they stay **singleton
root knobs**, never `@<axis>`. `TILE` keys on the contraction's single `k_axis` (enough to *identify* the
node; the output m/n axes come from the node at materialize/featurize time, and `PLACE` keeps its own
free-axis keying).

## Bare names — `@<axis>` is optional when one axis is suitable

A bare `TILE` / `REDUCE` / `STAGE` (no suffix) **resolves to the unique axis eligible for that family**, so
the suffix is sugar a single-node kernel never needs:

- **Eligibility is by family + role.** `TILE` → the sole `CONTRACTION` `k_axis`; `REDUCE` → the sole reduce
  axis (`Reduction.axis` / a `Contraction.k_axis` carrying a partition); `STAGE` → the sole reduce loop. The
  resolver enumerates the kernel's eligible axes for the family and binds the bare key to the one.
- **Zero eligible** → the knob doesn't apply (drop / OFF — a pointwise `Map` has no `TILE`).
- **Exactly one** → bare and `@<that-axis>` are **the same decision**; they must parse, featurize, key, and
  round-trip identically.
- **Two or more** (flash) → bare is **ambiguous**: a hand-written pin raises a loud error naming the
  candidate axes (`TILE is ambiguous: use TILE@d or TILE@sk`); enumeration never emits a bare key, so the
  ambiguous case only arises from a pin.

Canonical form is **`@<axis>` on the stored / enumerated / DB-keyed dict** (the resolver runs at the input
boundary — env pins via `Knob.raw`/`narrow`, the `--ab` / `DEPLODOCK_KNOBS` spec, recipe knobs — and the
featurizer's bare-read fallback); the display layer (`tuning_knob_items`) **may collapse `@<axis>` back to
bare when the kernel has a single eligible axis**, so one-node tables read exactly as today. This is what
makes the migration **non-breaking for pins**: `DEPLODOCK_TILE=…` / `DEPLODOCK_REDUCE=…` and every recorded
golden keep resolving on the single-node kernels that are ~all of them.

One resolver, shared: `resolve_axis(family, bare_or_suffixed, eligible_axes) -> "FAMILY@<axis>"`. Pins, the
featurizer, and storage canonicalization all route through it so bare and suffixed behave identically and the
ambiguity error fires in exactly one place.

## Before / after (real codec strings — every value below is an actual `.spell()` output)

**GEMM (one `Contraction`) — bare names suffice** (one eligible axis each), so the dict is unchanged from
today; the resolver canonicalizes to `@d` internally, display collapses back to bare:

```python
{ 'TILE':  'a:mma_m16n8k16_f16/w2x2/f2x2/k2', 'REDUCE': 'g4k/b8', 'STAGE': 'd3/cp/p2',   # bare = @d here
  'PLACE': '…', 'WSPEC': '' }
```

**Flash (two contractions + a twisted reduce) — the case the flat schema can't express** (one `TILE` key
can't hold both tiles). Axis-named:

```python
{ 'TILE@d':    'a:mma_m16n8k16_f16/w4x1/f2x2/k2', 'STAGE@d':  'd2/cp',   # QK, k-axis d
  'REDUCE@sk': 'b8',                                                     # online reduce, axis sk
  'TILE@sk':   'a:mma_m16n8k16_f16/w2x2/f4x1',     'STAGE@sk': 'd2/cp',   # PV, k-axis sk
  'PLACE': '…', 'WSPEC': 'p2:q8' }                                       # root-global
```

**Softmax (`Map(body=sweep, source=Reduction[PLANAR])`)** — only the reduce node is schedule-rich; the
`Map` projection carries none, so there's nothing to OFF-pad (see below):

```python
{ 'REDUCE@row': 'b8', 'STAGE@row': '', 'PLACE': '…' }
```

## Fork-tree construction — `010_recognize` emits structural leaves

`010_recognize` keeps recognition (flash / online-softmax / the structural skeleton via `ops`), then —
**replacing today's flat `_schedule.schedule()` candidate list** — emits a **lazy fork tree** of
scheduling choices: each axis offers the permitted moves for its `AxisRole`, producted with incremental
legality pruning. Every **leaf resolves to a complete structural-IR tree** with the chosen params baked
onto the nodes (a `Reduction` with its `reduce`, a `Contraction` with its `tile`/`bind`, the root with
its `place`/`workers`). The **consume side is reused unchanged**: `fork.py::build_fork_tree` (knob-rows
→ lazy tree) + `search/` (`LazyCandidate`, the greedy/MCTS policies ranking the frontier with the
learned `Prior`; cold fallback = emission order, conservative-pick-first).

**Permitted-move catalog** — each move emits the **same codec string** the knobs already parse (the
`parse`/`spell` grammar, the prior featurizer, and the perf DB grammar are untouched; the move is just
the *generator*) under the **axis-named key** for the node it schedules:

| Axis role | Permitted moves → node param → knob key |
| --- | --- |
| `FREE` (output) | grid-map → root `place` → `PLACE` (later: free-axis scalar tile) |
| `PLANAR` / `TWISTED` | serial · `b<n>` · `r<n>` · `g<n>[a\|k]` → `Reduction.reduce` → `REDUCE@<axis>` |
| `CONTRACTION` (m,n)+K | scalar `n../f..` · warp `a:<atom>/w../f../k<bk>` → `Contraction.tile`+`bind` → `TILE@<k_axis>`; `g<n>` split → a `Reduction` wrapping the `Contraction` (the composition step) |
| any reduce/contraction loop | `d<depth>/sync\|cp\|tma` → node `stage` → `STAGE@<axis>` (pin-only) |
| root | the warp split → root `workers` → `WSPEC` (pin-only) |

**Legality guards** (lift today's piecemeal checks into per-move guards, evaluated **incrementally** so
illegal subtrees never generate): scalar `block_threads ≤ 1024`; warp K-step `atom_k·bk` divides a
**static** K; no cross-CTA split of a symbolic reduce; the flash QK/PV **warp-budget coupling** (below);
env pins win via `Knob.narrow`. Blowup control: incremental legality (dominant), the lazy tree + a
beam/budget cap, conservative emission order (option-0 = the conservative pick so a cold greedy compile
is stable). This *is* the move catalog the native moveset already follows — the schedule families just
join it (see "all knobs in `knob.py`" below), so `build_fork_tree` consumes one uniform move stream.

## materialize = walk + expand (one expander per node)

Once the leaf is the fully-scheduled structural tree, `010_materialize` is a dispatch-on-node walk; each
expander reads its schedule slice off the node and synthesizes the renderable kernel-IR (the
reduce/contraction loop is **synthesized, never stored**):

- `Map` → bind the grid (`Tile`) from the root `place`, emit `body`, recurse into `source`.
- `Reduction` → the current `_reduce` logic as the expander (serial fold, or `StridedLoop` +
  `reg`-replication + the REG tree via `carrier.as_state_merge` + cross-thread `emit_combine`), reading
  `reduce` off the node; recurse into the partial `source`.
- `Contraction` → `_factor.factorize` (mma fragment soup / scalar register tile), reading `tile`/`bind`
  off the node. `_build_contraction` retires (seam #1).

`030_split` is **kept** — a `Reduction`/`Contraction` whose `ReducePlan` carries a `GRID` stage still
triggers the cross-CTA graph rewrite (partial + finalize), reading the plan off the node via
`ops.reduce_plan` (✅ already). The backend's `render_body` walk is unchanged.

## What changes, file by file

### `knob.py` — consolidate **all** `Knob` definitions here

Today the schedule codec knobs `REDUCE` / `TILE` / `STAGE` / `WSPEC` are defined in
`lowering/tile/_schedule.py` (module-level `Knob(...)` constants picked up by `knob._walk_modules`),
while the native-moveset knobs live elsewhere. **Move the schedule `Knob` definitions into `knob.py`** so
every knob — native moves and schedule codecs alike — is declared in one file and the whole tunable
surface is visible at a glance. `_schedule.py` imports them from `knob.py` instead of declaring them; the
`_walk_modules` discovery still finds them (now via `knob.py`). No behavior change — same `Knob.name` /
`off` / `help`, same `DEPLODOCK_<KNOB>` env binding (`knob.py` already owns that namespace). Do this as
the first migration step (it is mechanical and unblocks the axis-named keying living next to the codecs).

### `knob.py` — the featurizer goes per-node, then composes

The blocker today is that `knob_features` reads each schedule knob **once** as a singleton — `knob.py:559`
`tile_spec = knobs.get("TILE")`, and likewise `_reduce_decomp` reads `knobs.get("REDUCE")`,
`_stage_features` reads `knobs.get("STAGE")`. With N nodes those reads must become a **loop over the
`@<axis>` keys**:

- Group the dict by axis suffix → one sub-schedule per node, **resolving any bare key to its axis first**
  (via the shared `resolve_axis`) so a bare `TILE` and a `TILE@d` land in the same node slice. Each node
  featurizes through the *existing* `_geom_feats` / `_free_slots` / `_warp_tile_features` /
  `_stage_features` machinery unchanged (they take a `knobs` slice; give them the node's slice).
- Compose to the model's input. Two sub-options, pick per the prior decision below:
  - **Pool** — sum/max the per-node `D_*` blocks into today's fixed-width vector. Smallest change; keeps a
    single prediction; loses per-node attribution and most transfer.
  - **Per-node predict** — emit a per-node feature block keyed by axis; the prior predicts *that node's*
    sub-schedule and the kernel cost composes (bottleneck / Σ). This is the form that earns the migration
    (a standalone GEMM's learned `TILE@d` transfers to flash's inner QK).
- `tile_signature` / `_stage_sig` (the golden-match + variant identity) become per-axis tuples.
- `S_*` structural features (`S_ext_free_prod`, `S_ext_reduce_prod`, `S_masked_*`) become **per-node**,
  addressed `S_ext_reduce_prod@<axis>` — so the per-node featurizer has the node's own extents/masking.
  `020_stamp_structural_features` stamps them addressed.

### `knob.py` — rendering / ordering

`KNOB_ORDER = ("TILE","REDUCE","STAGE","WSPEC")` (exact names) is replaced by the schedule families joining
`_FAMILY_ORDER`; `knob_sort_key` already strips `FAMILY@element` and sorts by element, so the tuning-table
columns group per family then per axis with no new code. `tuning_knob_items` is unaffected (it iterates the
dict; the keys just carry suffixes).

### `fork.py` — `Level`s mirror the structural tree

`build_fork_tree` is reused unchanged in *shape*; the `Level` list stops being arbitrary knob groupings and
becomes: root-global level(s) outermost (`PLACE`, `WSPEC`), then one bundle per node in pre-order, keyed by
the node's axis. Leaves still carry the complete flat row (now with `@<axis>` keys) — the DB/prior keying is
untouched.

### `search/` + prior

`LazyCandidate` / the greedy + MCTS policies are unchanged (they consume `fork.knobs` flat). The prior
changes only if we take **per-node predict** (a per-node feature block + a compose step in the scorer);
**pool** leaves the prior signature identical. The analytic prior's `D_splitk` (= `REDUCE@<k>.cta`) already
speaks the axis-keyed split count, so its split-K gate needs only the per-axis read.

### `schedule.py` — `Kernel` / `TileSchedule` dissolve

The field homes are the "Structural nodes + field homes" table above: `reduce`→`Reduction` (✅),
`tier`/`bind`→`Contraction`, `stage`→the node, `place`/`workers`→thin `TileOp` root fields.
`Kernel(op, schedule)` collapses to the structural-IR root + the two root fields. The codec *classes*
(`TilePlan` / `ReducePlan` / `Stage` / `WarpSpec`) and their `parse`/`spell` are **kept verbatim** — only
the dict key they ride under gains the `@<axis>` suffix. This plan changes *addressing* + *home*, not the
codec grammar.

## The `REDUCE` reconciliation — the sharp edge

The native moveset already emits `REDUCE@<axis>` (the contraction-tower / split count, `REDUCE@<k>.cta`),
**and** the schedule reduce partition is the bare `REDUCE` codec. Making the schedule partition `REDUCE@<axis>`
makes these two **the same key** for the same axis. That is convergence, not accident — but they must be
**unified into one axis-keyed reduce family**, not two families colliding. Resolve before coding: confirm the
native `REDUCE@<axis>` and the schedule `REDUCE` codec are two spellings of one decision (the per-axis
partition) and collapse them; if any native `REDUCE@` carries a *different* decision, give one family a
distinct token. `TILE` / `STAGE` / `WSPEC` have no native `@`-family today, so they collide with nothing.

## Coupling — a legality rule, not an addressing one

QK and PV share the CTA's warps, so `TILE@d` and `TILE@sk` carry a **joint legality constraint**
(`block_threads` must agree) no matter how they're spelled. Axis-naming doesn't solve this and isn't meant
to. Enumerate the coupled tuple at **one** `Level` (multi-knob `knob_names`, already supported) or derive
PV's tile from QK's (master→slave). The check lives in the `020`/recognize fork legality guards regardless.

## OFF-default simplification (a bonus)

`apply_off_defaults` exists so a flat kernel carries explicit OFF values for tier-foreign knobs (`WM` on a
scalar row) — letting the prior read "decided: unused" vs "not yet decided." Per node, **the node kind
declares its knob set**: a `Contraction` has `TILE`/`STAGE`/`bind`, a `Reduction` has `REDUCE`/`STAGE`, a
pure `Map` has none. Absence becomes structurally meaningful, so much of the tier-foreign OFF-padding can
retire (keep only the within-node scalar-vs-warp OFF, e.g. `TILE@d=""` = "this contraction is per-cell").

## Migration (each step e2e-green; the recovery contract governs)

`reduce`-on-node has landed. The remaining order interleaves the structural moves, the fork-tree emit,
and the knob schema:

1. **Consolidate knobs in `knob.py`.** Move the `REDUCE`/`TILE`/`STAGE`/`WSPEC` `Knob` definitions from
   `_schedule.py` into `knob.py`; `_schedule.py` imports them. Mechanical, no behavior change — emitted
   kernels + features byte-identical.
2. **Build `Contraction` recognize-side** (seam #1). Move the `_build_contraction` body to the fork-emit
   side; a tiled-contraction leaf is a `Contraction` node with `tile`+`bind`; materialize only
   `factorize`s. Emitted-kernel parity over the matmul / fused-epilogue / warp-mma fixture.
3. **Collapse `Kernel`/`TileSchedule`** (seam #2). `TileOp` holds the structural root + thin
   `place`/`workers`; `tier`/`bind`/`stage` ride the nodes. Repoint the `test_matmul_coverage`
   accessors. Emitted kernels byte-identical.
4. **The resolver + addressing shim, behavior-preserving.** Land `resolve_axis(family, key, eligible_axes)`
   (bare → unique eligible axis; ambiguous → loud error; zero → drop) and route env pins / `--ab` /
   `DEPLODOCK_KNOBS` / the featurizer's bare-read through it. One schedule node per kernel (pre-flash) ⇒
   bare and `@<axis>` resolve identically. Byte-identical featurizer output; **every existing bare pin
   keeps resolving** (`op_cache_key` shifts per seam #3).
5. **Per-node featurizer loop.** Convert the singleton `knobs.get("TILE"/"REDUCE"/"STAGE")` reads into the
   group-by-axis loop; per-node `S_*` stamping. Still one node per kernel, so **pool == today** — parity
   bar is byte-identical features.
6. **Reconcile `REDUCE@`.** Unify the native moveset `REDUCE@<axis>` with the schedule reduce partition
   (the sharp edge above). Parity over a split-K matmul fixture.
7. **Fork `Level`s per node + emit structural leaves.** Restructure `build_fork_tree`'s levels to
   root-global-then-per-node; the recognize emit produces the move catalog → structural leaves.
   Structural-coverage test: the leaf set over a matmul fixture equals the hand-computed legal product.
8. **Compose `Reduction ⊃ Contraction` (flash + split-K) — the real multi-node test.** Convert flash to
   `Map(source=Reduction(source=Contraction(QK)))` (retiring the `reduce_plan` legacy-`Map` fallback);
   build split-K (`g<n>` wraps the `Contraction` in a `Reduction`). Add the coupling legality `Level`.
   New e2e `test_mma_splitk_finalize` / `sum` over `(H, W)`, and a `{TILE@d, TILE@sk, REDUCE@sk}`
   parse/spell + DB-key round-trip asserting the two tiles key apart.
9. **Per-node prior (optional, gated).** If we commit to per-node predict, add the per-node feature block +
   compose step; otherwise stop at pool. Decide via an A/B of prior rank on a flash fixture.

## Verification

- `tests/compiler/e2e/` green at **every** step; `tests/xfail_registry.py` only shrinks; `make lint`.
- **Emitted-kernel parity** at steps 1–3 (knob consolidation + the structural moves don't change codegen).
- **Featurizer parity** at steps 4–5 (suffixed knobs + per-node features byte-identical to the singleton
  form on one-node kernels) — the schema is invisible until a kernel actually has two nodes.
- **Structural-coverage test** (step 7): the fork-tree leaf set over a matmul fixture equals the
  hand-computed legal product, so a missing/extra `@<axis>` move is caught.
- **Multi-tile round-trip** (step 8): a flash kernel's `{TILE@d, TILE@sk, REDUCE@sk}` parse/spell + DB-key
  round-trip, and the coupling guard rejects a `block_threads`-mismatched pair.
- **Bare-name resolution** (step 4): a bare `TILE` pin on a single-contraction kernel resolves to and keys
  identically to `TILE@d` (featurize + DB-key parity); a bare `TILE` pin on a flash kernel raises the
  ambiguity error naming `TILE@d` / `TILE@sk`; a bare `TILE` on a pointwise `Map` is dropped (no eligible
  axis).

## Open decisions

- **Pool vs per-node predict** — pool keeps the prior unchanged and is enough to *unblock* the schema; per-node
  predict is the transfer payoff but adds a compose step. Lean: ship pool with the schema (steps 4–8), gate
  per-node predict (step 9) behind the A/B.
- **`REDUCE@` unification** — are the native moveset `REDUCE@<axis>` and the schedule `REDUCE` codec one
  decision (collapse) or two (distinct tokens)? Resolve before step 6. Lean: one decision, collapse.
- **`TILE` key = `k_axis`?** — keying the output tile by the contraction's single `k_axis` identifies the node
  cleanly; the alternative (key by the output m/n pair) is more literal but multi-axis. Lean: `k_axis`.
- **Coupling shape** — joint `Level` vs master→slave tile derivation for the flash QK/PV warp budget. Lean:
  joint `Level` (no derived-state to keep in sync; `knob_names` already takes a tuple).
- **Canonical stored form** — store always-suffixed `@<axis>` (resolver runs at the input boundary; display
  collapses to bare when one axis) or store bare when unambiguous (suffix only on multi-node)? Lean:
  always-suffixed storage + display collapse — one canonical DB/prior key, bare is purely an input/display
  affordance.
