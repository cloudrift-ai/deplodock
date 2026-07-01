# Recursive `factorize` with a wiring context

**Status: the pure refactor (Changes A + B, Steps 1 + 2) is LANDED, bit-identical, full suite green. Step 3 (tensor-core
flash — the payoff) is the remaining work; its seam is in place.** The recursive-emitter direction of the tile-IR
refactor (the `tile-ir-rebuild.md` mandate: *one hierarchical emitter, no divergent codegen paths*). It replaced the flat
dispatch (`_factorize_contraction` vs `_factorize_reduce` + `lower(op)`-then-refind) with a recursion over the node tree
`Map(Reduction(Contraction))`, threading a **context** down and returning a **wiring handle** up. The payoff is
tensor-core flash (register-tiled QK/PV inside the streaming reduce) falling out of the same emitter.

## Progress (landed)

- **Change A** — flash's Q@K moved off `Reduction.source` onto the single walked `partial` edge (QK at the head, PV
  embedded); `ir._flatten_nodes` generalized to a node-walk over `Contraction` / `Reduction` / `Map`. `source` now serves
  only the split-K composition. Bit-identical.
- **Change B** — a coop-K / ILP contraction (and the `030_split` residual) is nodified to a `Reduction` node carrying the
  K partition, via the shared `ops.nodify_reduce`; the `TileOp.reduce` residual field and `reduce_plan`'s fallback are
  deleted. Bit-identical.
- **Step 1** — `_factorize_reduce` reads its reduce loop / carrier / axis straight off the `Reduction` node (no
  `lower`-then-refind); unconditional after B. Bit-identical.
- **Step 2** — `factorize` drives a recursive walk `_emit(op, ctx) -> Frag` over the `Map` / `Reduction` / `Contraction`
  tree (through `source` AND `partial`), threading a `Ctx` down and returning a `Frag` (per-cell body + `Handle` wire +
  carrier) up. The two root binders (`_factorize_contraction`, the reduce partitioner) consume it; the reduce
  partitioner builds its per-cell loop / tail via `_emit`, so flash's nested Q@K / P@V are reached AS NODES.
  `_emit(node).body == ops.lower(node)` for every scalar node. Bit-identical.

## Step 3 (remaining) — the tensor-core seam

The seam is the **`Contraction` case in `_factor._emit`**: an output-warp-tiled `Contraction` (an mma `TilePlan`) must
emit through the register-tile pipeline + the fragment recast there, instead of `op.lower()` (scalar). The remaining
pieces (all interlocked — none lands in isolation) are Change C + P2 below, plus the fragment-level online-softmax
codegen (the rowmax / rowsum butterflies on the score C-fragment) and the per-node warp tiles + schedule fork that give
Q@K / P@V their mma `TilePlan`s. The executable spec is `test_fused_tensorcore_flash_reference_matches_torch`'s
hand-written FA-2 kernel; the `test_generated_tensorcore_flash_*` cases stay xfailed until the kernel reproduces it
through the one emitter.

Five adversarial agents stress-tested the load-bearing claims (**Verification findings** below). The direction survives,
but the **naive form is wrong in three structural ways**: the binding is NOT a single ambient thing threaded down (each
`Contraction` owns its `(m,n)` tile; `kv` is an output axis of QK and a reduce axis of PV); the wiring `Handle` must be a
**fragment descriptor** (role + shape + dtype) paired with an explicit **accumulator→operand recast node**, not a bare
name; and even the "safe" step-1 reframe must branch (a coop-K contraction rides a flat `Map` with no `Reduction` node).
The scope is also narrower than claimed: the recursion is **intra-CTA only** — cross-CTA split-K stays a graph rewrite in
`030_split`. **Do not build past the revised step 1 until R1/R2 are designed out.** See the revised model + migration at
the end.

## The unifying model

Every kernel = **tile the OUTPUT axes across threads** (grid → warp → register) + **fold the REDUCE axes within each
cell** (serial | ILP-reg | coop-thread | cross-CTA) + **project**. The node tree tags which axes are which:

- `Contraction`: `k_axis` = a reduce axis; `axes (m, n)` = output axes; produces an accumulator.
- `Reduction`: `axis` = a reduce axis; wraps a `source`; produces the carrier state.
- `Map`: no axis; projects its `source`'s value (the sweep / epilogue).

So `Map(Reduction(Contraction))` = `project ∘ fold-over-kv ∘ reduce-over-d`. The recursion walks the tree; **the binding
is shared** across levels because they all tile the *same* CTA threads over the *same* output cells. The two current
tiers are just two points in `(output-tiling) × (reduce-folding)` space:

- contraction tier: output tiled (m×n reg/warp), reduce (K) serial per cell, **no** combine.
- reduce tier: output block=1, reduce partitioned cooperatively, **combine**.

## The three pieces

### A. Down (arguments) — `Ctx`, the ambient cell environment

```python
@dataclass(frozen=True)
class Ctx:
    bind:  Binding            # WHO am I: grid axes + warp/unit axes + reg cell counts + lane + block_threads
    cell:  Offset             # WHERE in the tile: the per-output-axis AxisOffset (== today's Offset)
    wires: dict[str, Handle]  # inbound tensors the parent feeds this child (flash: P -> PV's A operand)
    inputs: dict; stage: Stage | None; out_buf: str
    scope: str                # SSA-name suffix unique to this node's path (no cross-node collisions)
```

`bind` / `cell` are today's `Tiling` / `Offset`, but established once for the WHOLE kernel and threaded down unchanged,
so every node emits reads/writes at the same cell coordinate. `wires` is the key addition: when a parent produces a
tensor a child consumes (flash's `P` feeds the PV contraction's A-operand), it is handed down as a `Handle`, not
re-lowered.

### B. Accumulated (context state) — `Emit`

Collected on the way through, emitted once by the root binder:

```python
@dataclass
class Emit:
    decls:  list[Stmt]        # hoisted state: accumulator RegFragments, smem slabs, mbar — outside all loops
    axes:   tuple[Axis, ...]  # the kernel's thread signature (unit/lane/coop axes) accumulated across nodes
    smem:   NameRegistry      # slab/name bookkeeping so nested nodes don't collide
```

This is what `Tiling` already accumulates (axes, block_threads), generalized across the node tree, plus the hoisted
`decls` that `state_decls` produces today.

### C. Up (return) — `Frag`, the wiring handle + body

```python
@dataclass(frozen=True)
class Frag:
    out:     Handle           # the produced tensor — the wire the parent connects to
    body:    list[Stmt]       # the per-cell loop-IR this node contributes (K-loop / reduce loop / sweep)
    carrier: Carrier | None   # set iff this node folds a reduce needing cross-partition combine
```

The **`Handle`** — "a tensor that needs wiring up" — is the load-bearing type:

```python
@dataclass(frozen=True)
class Handle:
    names:     tuple[str, ...]     # the SSA / fragment names holding the value ("_c{i}{j}", "acc", ...)
    residence: Residence           # REG_FRAG | SMEM | GMEM — HOW a consumer reads it
    layout:    Layout              # which output axes it spans + fragment shape (for mma operand wiring)
```

`residence` + `layout` are what let an upper layer connect stuff: a consumer knows whether to `ldmatrix` a register
fragment, `Load` from an smem slab, or read gmem — and whether the fragment layout is mma-compatible.

## Per-node behavior

**`factorize(Contraction c, ctx) -> Frag`** (a leaf w.r.t. node recursion)
- If its A-operand is a *computed* body (flash `P`), pull the `Handle` from `ctx.wires["a"]`; else it is an operand
  `Load`.
- Register-tile the K-loop into `ctx.bind`'s cells (this is today's `reduce_codegen`: `state` -> `Emit.decls`,
  `reduce_region` -> `Frag.body`).
- Return `Frag(out=Handle(acc fragments, REG_FRAG, (m, n) layout), carrier=additive)`.

**`factorize(Reduction r, ctx) -> Frag`**
- Recurse `factorize(r.source, ctx)` -> the per-step input `Handle`.
- Emit the fold over `r.axis` around it (serial, or `StridedLoop` with coop/reg partition per `r.reduce`), seeding the
  carrier state.
- If `r.reduce.coop > 1` or `reg > 1`, hand `carrier` up so the root emits `combine_tail`; the surviving carrier state is
  the `out` Handle.
- Return `Frag(out=Handle(carrier state), carrier=r.carrier if cooperative else None)`.

**`factorize(Map m, ctx) -> Frag`**
- Recurse `factorize(m.source, ctx)` -> value `Handle` (or `None` for a pure pointwise Map).
- Append the projection `m.body`, reading the child `Handle` (this is today's swappable **`store` sink**).
- Return `Frag(out=Handle(projected value))`; the root writes it via `with_store`.

## Who decides the binding (the crux)

The schedule slices already live on the nodes (`Contraction.tile`, `Reduction.reduce`). The output tiling is owned by the
node that produces the kernel output — the root `Contraction` for a matmul, or the carrier / PV tile for flash. A small
pre-pass reads the composite binding from the tree (like `reduce_plan` already reaches through `Map.source`), builds
`Ctx.bind` once, and the recursion threads it down. A nested `Contraction`'s `tile` must be **consistent** with the
ambient warp layout — that consistency is the one real correctness obligation, checked when `bind` is built (today it
holds trivially because nested = scalar block=1).

## How it maps onto today's code (reorg, not rewrite)

| Recursion piece                   | Already is                                                          |
|-----------------------------------|---------------------------------------------------------------------|
| `Frag` for a `Contraction`        | `reduce_codegen` -> `(state_decls, reduce_region)`                  |
| `factorize(Map)` (projection)     | the swappable **`store` sink** (`store_sink` / flash's custom sink) |
| `Frag.carrier` -> combine         | `combine_tail` (just extracted)                                     |
| `Ctx.bind` / `Ctx.cell`           | `Tiling` / `Offset`                                                 |
| the root binder consuming `Frag`s | `grid_tile` (+ the reduce partitioner)                              |

The recursion **deletes the flatten-then-refind smell**: no `lower(op)` + `next(... Loop and carrier)` — the `Reduction`
node *is* the carrier, so `factorize(Reduction)` reads it directly.

## Migration

The initial-sketch migration is **superseded** by the findings — see **The op-tree changes** + **Revised migration
(ordered)** at the end of this doc. The op-tree nodification (Changes A + B) lands FIRST; only then the recursion reframe
and root; tensor-core (Change C + P2) last.

## Verification findings (5 adversarial agents, against the real code)

### R1 — binding threadability → **BROKEN for the tensor-core goal** (holds only in today's scalar `block=1`)

The "one output binding, owned by a node, threaded down unchanged" model collapses for tensor-core flash. Concrete axes:
QK `Contraction.axes = (m, kv)`, contracts `dd`, produces the score `[m, kv]` (`_flash.py:312-319`); the streaming
`Reduction.axis = kv` (TWISTED); PV `Contraction.axes = (m, d)` with `k_axis = pj` a **singleton** dim-1 (the kv sum is
done by the enclosing `Reduction`, not PV's own axis) (`_flash.py:249-256`); kernel output `(batch…, m, d)`
(`_flash.py:199`). So the three levels have **disjoint-except-`m` output-axis tuples**, and `kv` is an OUTPUT axis of QK
but the REDUCE axis of the streaming reduce / PV. The `Side`/`mn` tiler binds each `Contraction` to exactly two output
axes (`ir/tile/ir.py:294`) — QK's `(m,kv)` and PV's `(m,d)` are different pairs, each tiled independently. There is no
single output tile all three emit into. Two further corrections: (a) today the output binding is **not owned by a node**
— it rides `tile.place` (the free-axis→grid scheduler, `_factor.py:340`); the "PV tile owns it" line is wrong. (b) It
"works" today only because flash is scalar `block=1` (one thread per `(m,d)` cell, score never materialized, recomputed
per output element). **Revision: bindings are per-node; `kv` swaps output↔reduce role between QK and PV; a shared ambient
`bind` is not the model.**

### R2 — `Handle` sufficiency for mma wiring → **BROKEN as stated** (feed is a *recast*, not a wire; `Handle` under-specified)

No IR path lets a register fragment be an mma operand: mma A/B `RegFragment`s are **only** filled by `LdmatrixLoad` from a
buffer (`ir/kernel/ir.py` `MmaSyncPtx`/`LdmatrixLoad`), and `_MmaOps.reduce` **asserts** `not c.a_computed`
("register-resident A operand … is a scalar-tier-only capability", `_atom.py:594-596`); `_mma_stage_plan` forces gmem for
a computed A (`_atom.py:357`). The P→PV feed IS hardware-real with no smem round-trip — 2× m16n8 **f32** accumulators pack
into 1× m16k16 **f16** A-operand (the FlashAttention-2 "convert fragment" trick) — but it carries three obligations:
f32→f16 **downconvert**, C→A **role change**, and **2:1 kv packing**. `Handle(names, residence, layout)` with
`layout = axes + shape` omits **dtype** and **mma role/kind** — exactly what the recast keys on. Redeeming nuance: once
`(role, shape, dtype)` are fixed, the per-lane element map is canonical (PTX ISA), so `Handle` needs the coarse triple,
not a full register table. **Revision: `Handle.layout` → a fragment descriptor `(mma_role, m/n/k shape, dtype, residence)`;
add a first-class accumulator→operand recast node; lift the `a_computed` mma restriction.**

### R3 — node coverage → **PARTIAL** (3 shapes escape the Map.body / Map.source / Reduction.source edges)

The 3 arms cover every shape whose reduce/contraction is the root or on those three edges (bare `Contraction`;
`Map(None)` pointwise; `Map(Reduction)` softmax; bare `Reduction`; flash's QK via `source`; `lead_axes`; the degenerate
`coop==reg==1` arm). Three MISSES:
- **A — `Contraction` in `Reduction.partial` (flash PV).** `_split_pv` appends PV as a **stmt inside `partial`**
  (`_flash.py:262`, `:353`), reached today only by `_flatten_nodes` inside `Reduction.loop` (`ir.py:69-80,142-144`) — no
  arm dispatches to it. The tensor-core payoff *requires* reaching it. Needs a `partial`-walk in `factorize(Reduction)`
  or re-homing PV onto a node edge.
- **B — coop-K / ILP non-tiled contraction on a flat `Map(source=None)`.** Its partition rides `TileOp.reduce` (schedule
  field, "not yet a node", `ops.py:46-48`); `_factorize_reduce` finds the loop by flattening. `factorize(Map(None))` in
  the design just "appends the projection" — it does not partition a body-embedded reduce loop. No arm sees it.
- **C — shared-row staging is a cross-node rewrite.** `_restage_loads` rewrites loads across `rloop.body`, `pre`, AND
  `tail_src` with one shared smem slab (`_factor.py:394-396`) — i.e. it reaches across what would be the `Reduction` body
  *and* the `Map` projection. Independent `Frag`s can't express a parent reaching into a child's emitted body. Needs a
  shared-staging channel in `Ctx`/`Emit`.

### R4 — step-1 bit-identity → **BROKEN as unconditional, PARTIAL in practice**

For the four **nodified** shapes (bare `Reduction`, `Map(Reduction)` softmax, flash, fused RMSNorm→linear) node-read is
bit-identical: `Reduction.lower()` is a single loop, so the carrier loop is always `stmts[0]`, `pre = []`, and
`tail == Map.body`. But the coop-K flat-`Map` contraction (MISS B; `DEPLODOCK_REDUCE=b4/r4`, tested by
`test_matmul_reduce_partition`) reaches the same partitioned arm with **no `Reduction` node** — `reduce_plan` returns the
partition off `tile.reduce` and `lower(op)`+refind finds the `CONTRACTION` K-loop that is not a `Reduction`. **Revision:
step 1 must branch — node-read for nodified reduces, retain flatten-then-refind for the flat-`Map` contraction.**

### R5 — split-K / cross-CTA → **PARTIAL** (invariant holds; the "same recursion at a different Fold" framing over-reaches)

`030_split` (in the earlier `lowering/tile` phase) strips the GRID stage; the materializer asserts `not needs_split`
(`010_materialize.py:42`) and only ever reads `plan.coop` (BLOCK) / `plan.reg` (REG), never `plan.cta`. So the recursion
is **intra-CTA only**; `Fold.ATOMIC` is emitted exclusively by `030_split`. What *is* unified: the split-K **partial**
kernel lowers through the same `factorize` (a bare grid-partitioned `Contraction`). The **combine** (atomicAdd / finalize
kernel) is not. Also: the partition is only readable off a `Reduction` node once contractions are nodified; a `Map`-riding
contraction carries it on `TileOp.reduce` (the `ops.py:53` fallback) — same root cause as MISS B / R4. **Revision: strike
"split-K = same recursion at a different Fold"; the recursion consumes the partition via `reduce_plan`, not off a node.**

## Revised model (what survives)

The direction is sound and matches the mandate (one hierarchical emitter), but the shape changes:

1. **Intra-CTA only.** Cross-CTA (`030_split`, `Fold.ATOMIC/GRID`) stays a graph rewrite. The recursion consumes
   `plan.coop`/`plan.reg` via `reduce_plan`, unchanged.
2. **Per-node binding, not one ambient `bind`.** Each `Contraction` owns its `(m,n)` output tile (its `TilePlan`). `Ctx`
   carries the **shared** grid/batch axes + the inbound **wiring handles**, but a node tiles its own two output axes.
   The consequence of R1: `kv` being an output axis of QK and a reduce axis of PV is legal *because* they are different
   nodes with different tiles — the recursion must not assume one tile spans both.
3. **`Handle` = fragment descriptor + explicit recast.** `Handle(names, residence, fragment=(mma_role, shape, dtype))`.
   A node boundary emits a **relayout/recast**: identity/copy for scalar, the C→A downconvert+2:1-pack for mma
   (a new node). This is the FMHA "convert fragment", first-class.
4. **Traversal reaches `partial`, not just `source`.** `factorize(Reduction)` must walk `partial` for embedded
   `Contraction` nodes (flash PV), plus a shared-staging channel in `Ctx`/`Emit` for the cross-node norm→linear slab.
5. **Step 1 branches** on whether a `Reduction` node exists.

## Prerequisites the findings surfaced (blockers for the tensor-core payoff, step 3)

- **P1 (R1):** reconcile that `kv` lives at two tree levels — QK's output axis `(m,kv)` vs PV's singleton `k_axis=pj`
  while the real kv sum is the enclosing `Reduction.axis`. Tensor-core PV needs kv as its mma-K; the tree must express
  that without a bespoke flash path.
- **P2 (R2):** `Handle` → fragment descriptor; add the accumulator→operand recast node; lift `assert not c.a_computed`
  in the mma tier.
- **P3 (R3-A):** re-home flash PV onto a node edge (or add a `partial`-walk) so `factorize` reaches it structurally.
- **P4 (R3-B / R4 / R5):** nodify the coop-K flat-`Map` contraction (kill the `TileOp.reduce` residual) so the partition
  reads off a node uniformly.
- **P5 (R3-C):** a cross-node shared-staging channel in `Ctx`/`Emit` for the fused norm→linear smem row.

## The op-tree changes — the INITIAL steps (behavior-preserving nodification)

The recursion is total only over a tree where **every reduce / contraction is a node on a walked edge**
(`Map.source` / `Map.body` / `Reduction.source` / `Reduction.partial`). Two shapes violate that today, and both are
fixable **without new codegen** — they re-home / nodify existing structure so the SAME loop-IR is emitted. These land
FIRST, before any recursion refactor: they make the tree hierarchical so steps 1–2 have nothing special to branch on.

### Change A — make `Reduction.partial` node-bearing; re-home flash PV (fixes P3 / MISS A)

- **Today:** `Reduction.source` is a node (spliced ahead by `Reduction.loop`), but `Reduction.partial` is flat loop-IR
  where a `Contraction` (flash PV) hides as a stmt, reached ONLY by `_flatten_nodes` inside `.loop`
  (`ir/tile/ir.py:69-80,142-144`; PV appended at `_flash.py:262,353`).
- **Change:** treat `partial` as a `Body` that may carry `Contraction` / `Map` nodes, and have the materializer **walk**
  it — generalize `_flatten_nodes` (flatten-to-loop-IR) into the node-walk the recursion uses, so PV is reached AS A NODE.
  Flash's per-step pipeline then reads as uniform nodes on one edge:
  ```
  Map(body=[O/l projection],
      source=Reduction(TWISTED, axis=kv_stream, carrier=flash_combine,
        partial=Body([ Contraction(QK → S[m, kv_tile]),          # produce the score tile
                       Map(softmax twist: m/l update, P = exp(S − m)),  # carrier-coupled
                       Contraction(PV → O_upd[m, d]) ])))          # contract the score tile
  ```
- **Behavior-preserving:** the walk yields the same scalar loop nest `_flatten_nodes` produces today; PV just becomes
  reachable as a node instead of pre-flattened. QK may stay `source` or move to `partial`'s head — the point is *partial
  is walked*, so the source/partial asymmetry stops mattering.
- **Touches:** `ir/tile/ir.py` (`Reduction._flatten_nodes` / `.loop`), `_flash.py` (`_split_pv` placement), the
  materializer's node-walk. **Acceptance:** `tests/compiler/e2e/test_attention_coverage.py` bit-identical.

### Change B — nodify the coop-K contraction (fixes P4 / R4 / the R5 residual)

- **Today:** a non-output-tiled coop/ILP-K matmul rides a flat `Map(source=None)` holding an annotated `CONTRACTION`
  loop, with the partition on the schedule field `TileOp.reduce` — **no node** (`_schedule.py:405-408`; `ops.py:52-53`
  fallback). This is what forces `_factorize_reduce`'s `lower(op)`-then-refind and would force step 1's branch (R4).
- **Change:** build it recognize/schedule-side as `Reduction(axis=k, source=Contraction(...))` — its K-loop already
  carries the degenerate additive carrier — so the partition rides the node and `reduce_plan` reads it off the node
  uniformly (the `tile.reduce` fallback can then be deleted).
- **Behavior-preserving:** same lowering; only the partition's *home* moves from a root field onto the node.
- **Touches:** `_schedule.py` (the coop-K keep-as-`Map` branch), `ops.py` (drop the `reduce_plan` fallback once nothing
  rides `tile.reduce`), `_factor.py` (the `role is CONTRACTION and not tier.is_tiled` arm). **Acceptance:**
  `test_matmul_reduce_partition` (`DEPLODOCK_REDUCE=b4/r4/r2/b4`) bit-identical.

## Revised migration (ordered)

1. **Change A + Change B (op-tree nodification, above)** — behavior-preserving. After these, EVERY reduce/contraction is
   a node on a walked edge; there are no schedule-field-only partitions and no codegen hidden in flattened stmts. This is
   the precondition that makes the recursion total, and it removes the branch step 1 would otherwise need.
2. **Step 1 (recursion reframe, bit-identical):** define `Frag` / `Handle` / `Ctx` (intra-CTA); make `_factorize_reduce`
   read the reduce off the `Reduction` **node** — now unconditional (B removed the flat-`Map` case). Kills the
   flatten-then-refind smell.
3. **Step 2 (recursive root):** `factorize` dispatches `Map` / `Reduction` / `Contraction` recursively, walking `source`
   AND `partial`; `grid_tile` / the reduce partitioner are the two root binders consuming `Frag`s. Still scalar-nested,
   bit-identical. Needs P5 (a shared-staging channel in `Ctx`/`Emit`) for the fused norm→linear row.
4. **Step 3 (feature — tensor-core flash):** Change C (split `kv` into `kv_tile × kv_stream` so PV's `k_axis = kv_tile`
   is a real mma-K — the P1 shape fix) + P2 (`Handle` → fragment descriptor, the accumulator→operand recast node, lift
   `assert not c.a_computed`). Per-node warp tiles let QK/PV run register-tiled. Behind a flag, accuracy tests vs the
   scalar baseline.

Changes A + B and steps 1–2 are the pure refactor (behavior-preserving, tests green each step). Step 3 is the payoff and
carries the real correctness work (the fragment recast + the `kv` role-swap).

## Deferred op-tree change (tensor-core only)

**Change C — give PV a real `kv-tile` K axis (P1).** PV's `k_axis = pj` is a **singleton** today (scalar streaming does
a rank-1 update per kv-element; the kv-sum is the enclosing `Reduction`). Tensor-core PV must contract a *tile* of kv as
its mma-K: split `kv` on the schedule into `(kv_tile × kv_stream)` — PV's `k_axis = kv_tile`, `Reduction.axis =
kv_stream`. A tiling decision on the flash node, built recognize/schedule-side. This is the one genuine *shape* change
(not just nodification), so it rides with the tensor-core feature (step 3), not the initial nodification.
