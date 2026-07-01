# Tile IR rebuild

Status: **the structural-node skeleton, the single-`factorize` emitter, and the mma tier have landed.** The tile IR is
now a tree of **structural nodes** (`Map` / `Reduction` / `Contraction`, all in `ir/tile/ir.py`) carried on a `TileOp`
with root-global schedule fields (`place` / `reduce` / `tier` / `stage` / `workers`) plus per-node schedule slices
(`Contraction.tile`, `Reduction.reduce`). `010_recognize` is the sole Loop-IR → Tile-IR boundary; `020_schedule` maps
the free axes onto the grid and stamps the partition / tile / (pin-only) warp fragment; `030_split` consumes a cross-CTA
`GRID` stage as a graph rewrite; `lowering/kernel/010_materialize` calls **one** emitter, `_factor.factorize`, per
kernel. Recovered: every elementwise / reduction / online-softmax / RMSNorm / scalar + register-tile + **mma** matmul /
scalar-flash / whole-block kernel, the cooperative (BLOCK) + register-ILP (REG) reduce tiers, the dynamic-grid tier, the
cross-CTA (GRID) split tier, and the gmem-direct `mma.sync` **warp / tensor-core matmul** (canonical + transposed-B,
f16/f32 out, fused epilogues, static + dynamic-grid, pin-driven mma split-K via the structural `Reduction ⊃ Contraction`
fork). All over **static AND symbolic** axes. **Phase 0 (purge the consolidation's residue) has landed** — the
`Schedule` alias, the `_with_reduce` no-op, and the dead docstring framing are gone; the audit corrected one false
premise (the `Map.out` / `reduce_loop` reads of the flat-contraction `Map` are load-bearing, not legacy — see Phase 0
below). **Remaining, in restore order — each phase followed by a purge of the code it obsoletes: (1) finish mma
(tensor-core flash + symbolic-K masked edges); (2) operand staging (smem + `ldmatrix`); (3) pipelining (cp.async ring /
register double-buffer / TMA); (4) warp specialization (`WarpSpec`).** Branch `refactoring/tile-ir-rebuild`.

## The mandate — purity, not accretion

This rebuild is a demolition, not a migration. Its output is a codebase that reads as if the old tiers never existed —
**not** one stratified with compatibility shims, "not-yet-migrated" fallbacks, and reserved-for-later fields.

The rules are absolute:

- **Zero back-compat aliases, zero re-exports "during the transition."** A renamed symbol is renamed at every call
  site in the same change; the old name does not survive one commit past its last use.
- **Zero legacy fallback branches.** Every `if it's the old form / else the new form` is a lie about the IR's
  invariants. If recognition guarantees every reduce is a `Reduction` node, then a body-scan fallback that handles a
  bare annotated `Loop` is dead code *asserting the opposite* — delete it and put an `assert isinstance(...)` in its
  place. A fallback that "can't happen" either can't (delete it) or can (a recognizer bug the fallback is hiding — fix
  the recognizer). Never both.
- **Zero divergent codegen paths.** Everything flows through the one `factorize` emitter over `Reduction` /
  `Contraction` / `Map` (see the invariant below). Two code paths that produce the same kind of kernel are one bug and
  one dead path.
- **The recovery contract is the license to delete without mercy.** `tests/compiler/e2e/` is black-box and pins
  numerics + generated source. Anything a green e2e suite does not require is, by definition, dead — remove it and let
  the suite catch you if you were wrong. No feature is kept "just in case."
- **A phase is not done when the capability works. It is done when the code it replaced no longer exists and every doc
  describes the world as it is.** The purge sub-step is not optional cleanup deferred to "later"; it ships in the same
  phase, and the phase's PR is incomplete without it.

Boldness here is not recklessness — it is *decisiveness backed by the contract*. Delete first; the e2e suite is the
proof.

## Governing invariant — one hierarchical emitter, no divergent codegen paths

Everything lowers through the **same hierarchical structure of `Reduction` / `Contraction` / `Map`**. One entry point —
`010_materialize.rewrite` calls `_factor.factorize(tile, root)` once per kernel — and one flattener, `ops.lower(op)`,
which unfolds any node tree (including nesting: `Map(source=Reduction(source=Contraction))`) back to the same loop nest.
`factorize` reads the node kind + role (`ops.axis_role`) + reduce plan (`ops.reduce_plan`) off `tile.op` and routes to a
tier that differs **only in the schedule's partition/tiling**, never in the algebra:

- a `Contraction` → `_factorize_contraction` (the four-level `atomize → register_tile → unit_tile → grid_tile`
  pipeline; both atoms — tensor-core `AtomKind` and scalar `ScalarAtom` — share it, dispatching only at `reduce_codegen`
  and the `store` sink);
- a cooperative / ILP `PLANAR` / `TWISTED` reduce (or a non-output-tiled `CONTRACTION`) → `_factorize_reduce`
  (carrier-generic: a contraction is the degenerate carrier of its additive fold);
- anything else (pointwise `Map`, trivial-plan reduction) → the inline scalar tier (`lower(op)` + `with_store`).

**The bar for every remaining phase: it lands as new data/config on a node** (a `TilePlan` atom, a `Stage`, a
`WarpSpec`) **that the one emitter already structurally supports — not a bespoke path.** Tensor-core flash is the inner
`Contraction` gaining an mma `TilePlan` + a flash `store` sink; operand staging is a `Stage` step spliced into
`reduce_codegen` symmetrically for both atoms; warp specialization is a `WarpSpec` whose roles each bottom out in a
`Map` / `Reduction` / `Contraction` sub-schedule. **If a phase needs its own emitter, the node model is wrong — fix
the model, do not fork the codegen.**

## The recovery contract

`tests/compiler/e2e/` is the **only** thing the rebuild must satisfy. Every file there is black-box: it builds a graph,
runs it through `CudaBackend`, and compares the output to a numpy/torch reference (a handful also assert on generated
CUDA source — part of the contract too). None assert on tile-IR Python objects, so they survive any internal redesign.
The rebuild is "done" when the whole `e2e/` suite is green with an **empty** xfail registry.

## Integration-test xfail mechanism

A single registry drives expected failures during the rebuild — no scattered `@pytest.mark.xfail` decorators.

- **Registry:** `tests/xfail_registry.py` exports `XFAIL: dict[str, str]` mapping a **test node-id substring** to a
  one-line reason. `"test_foo.py"` xfails a whole file; `"test_foo.py::test_bar"` xfails one case. The
  `pytest_collection_modifyitems` hook in `tests/conftest.py` marks every collected item whose `nodeid` contains a
  registered substring with `xfail(strict=False)`.
- **Recovery semantics:** `strict=False` means a test that starts passing reports as **XPASS**, not a failure — the
  signal a capability came back. Delete its entry the moment it XPASSes; a stale xfail entry is itself dead scaffolding.
  **An empty `XFAIL` means the rebuild is fully recovered** — at which point the registry, the conftest hook, and the
  guarded `try/except ModuleNotFoundError` tile imports are all torn out too (see the Phase 4 purge).
- **A file whose module-level import of a tile symbol breaks** fails at collection — its import was guarded so it
  still collects. `TILE_ENTANGLED_FILES` is empty now.

**Deleting unit tests while rebuilding:** a tile-IR **unit** test (imports `deplodock.compiler.ir.tile` or
`…passes.lowering.tile` and asserts on those objects) → **delete it**, don't port it; the new internals get new unit
tests. An **integration/accuracy** test (anything in `tests/compiler/e2e/`) → **never delete or weaken** it; xfail it if
the in-progress rebuild breaks it, and it flips back to a hard requirement when the capability returns.

## What has landed

- **The structural-node skeleton** — the `Monoid` / `Semiring` node wrappers were **retired**; the tile IR is now `Map`
  (lift / projection, optional `source`), `Reduction` (a `PLANAR` / `TWISTED` reduce splitting its `Carrier` algebra
  from its `axis` + `partial`, the fold `Loop` synthesized on demand), and `Contraction` (a matmul before atom
  factorization), all in `ir/tile/ir.py` alongside `ir/stmt/algebra` (where the live `Carrier` / `State` / `Twist`
  algebra lives — those are carrier components, not node kinds). The algebra is read **structurally** off the annotated
  reduce loop / node, never a stored kind tag; `ops.lower` flattens any tree back to the loop nest. Flash is
  `Map(source=Reduction(TWISTED, source=Contraction(QK)))`.
- **One `factorize` emitter (factorize-consolidation Part I)** — `010_materialize` is a thin wrapper;
  `_factor.factorize` is the single node-kind dispatcher (scalar / pointwise + coop-ILP reduce + tiled contraction). The
  old three-tier `010_materialize` (`_reduce` + inline scalar fallback) is gone. `reduce_codegen` (operand fragments +
  the K-loop, sink-agnostic) and the pluggable `store` sink are the shared seams; `emit_combine` /
  `carrier.as_state_merge` fold any carrier (degenerate `id` or twisted `exp`) identically.
- **The scalar + cooperative + register-tile + dynamic-grid tiers** — pointwise, per-cell reduce (plain + online
  softmax via the twist), scalar matmul, scalar flash, the BLOCK coop + REG ILP reduce partitions (combine + REG-tree
  fold derived from the plan), the scalar register-tile (`TILE` codec), and the symbolic launch grid. Static AND
  symbolic.
- **The cross-CTA (GRID) split tier** — `030_split`: `g<n>[a|k]` codec, `atomicAdd` one-kernel OR deferred
  `__partial`-workspace + combine kernel; additive `sum` / split-K AND twisted flash split-KV, carrier-generic.
- **The mma / tensor-core matmul tier** — the gmem-direct `mma.sync` `Contraction` (`_mma_state` / `_mma_reduce` /
  `_mma_store`, the `WARP` codec): canonical + transposed-B, f16/f32 out, fused epilogues, static + dynamic-grid.
  **mma split-K** via the structural fork (pin-driven; auto-fork enumeration is a Phase-1 follow-up).

## The current architecture — where the schedule lives

The combine (the ⊕ algebra + reduce/contract structure) lives in the op tree; the schedule lives on the `TileOp` and its
nodes. **Root-global** `TileOp` fields: `place` (`Placement`), `reduce` (`ReducePlan` for a not-yet-nodified reduce),
`tier` (`TilePlan` for a non-tiled contraction), `stage` (`Stage` — materialization currently dropped), `workers`
(`WarpSpec` — not yet built). **Per-node** slices: `Contraction.tile` (a `TilePlan` carrying the leaf `atom` +
unit/register widths + K-chunk), `Reduction.reduce` (the partition). There is no per-kind schedule *type*: the role is
read structurally, the materializer reads the schedule straight off the node. `ops.reduce_plan` reads the partition off
a `Reduction` node **or falls back to `TileOp.reduce`** for the not-yet-nodified coop/ILP-K contraction — a Phase-1
purge target.

### The knob schema — shared orthogonal codes

A schedule is **spelled** by orthogonal codes, one per tunable sub-component; a kernel's config is the union that
applies. One vocabulary, learned-feature generalization across kinds (the featurizer reads one code kind-agnostically);
**not** cross-kind measured-evidence sharing (`op_cache_key` silos `perf` rows by kind).

| code | sub-component | where | grammar (coarse→fine) | status |
|---|---|---|---|---|
| `REDUCE` | `ReducePlan` (reduce-axis partition) | `Reduction`, non-tiled `Contraction` | `g<n>[a\|k]` / `b<n>` / `r<n>` · empty = serial | **built** |
| `TILE` | scalar output tile (`TilePlan` — par + reg) | `Contraction` (scalar atom) | `n<N>[xm<M>]` par · `f<fn>[xf<fm>]` reg · empty = per-cell | **built** |
| `WARP` | mma fragment (`TilePlan` w/ tensor-core atom) | `Contraction` (mma atom) | `a:<atom>` · `w<WM>xw<WN>` · `f<FM>xf<FN>` · `k<bk>` | **built** (gmem-direct; pin-only) |
| `STAGE` | `Stage` (operand pipeline) | `Reduction`, `Contraction` | `d<depth>` ring · `sync\|cp\|tma` · `[ring]` · `[p<reg_depth>]` · empty = gmem-direct | **materialization dropped — Phase 2/3** |

**Delimiter hierarchy** (so codes survive the `DEPLODOCK_KNOBS` / `run --ab` parser): **`,` is reserved** as the
knob-list separator and MUST NOT appear inside a code value. Within a value: `/` separates fields, `x` pairs dims, `:`
introduces a name (`a:<atom>`), `;` lists. Sub-field order is fixed **m-then-n**. Interpretation is per-node-kind
(`REDUCE b32` partitions *the* reduce axis — a `Reduction` reads its reduce axis, a `Contraction` reads K); fragment is
implicit in which output code is present (`TILE` ⇒ scalar atom, `WARP` ⇒ mma atom, never both).

## Phase 0 — Purge the consolidation's residue (do first; blocks nothing)

The landed consolidation left shims and lies behind. Delete them **before** building on top, so the new phases stand on
clean ground. All e2e-covered; each deletion is proven by the suite staying green.

- **Kill `Schedule = Placement`** (`ir/tile/ir.py`) — the "kept re-exported during the transition" alias. Rewrite every
  `Schedule` reference to `Placement`, delete the alias and its two-line apology.
- **The `Map.out` annotated-`Loop` branch and the `ops.reduce_loop` body-scan are BOTH load-bearing — not legacy.**
  The plan's premise ("recognition nodifies every reduce/contraction") is **false for contractions**, verified against
  the e2e suite. Recognition emits every **reduce** as a `Reduction` node (bare) or `Map(source=Reduction)` (projected),
  but a **contraction** rides a flat `Map(body=(annotated CONTRACTION loop, …))`: a *tiled* / warp / split-K contraction
  is nodified to a `Contraction` by `_schedule`, but a **scalar per-cell contraction keeps the flat `Map` all the way
  through materialize** (deleting the `Map.out` `carrier.out` arm broke every scalar-matmul e2e test — `_store`'s
  store-glue reads `op.out` off that loop's carrier). Likewise `reduce_loop`'s body-scan reads the K loop off the
  pre-nodification flat `Map` (`_contraction_node` / `_check_warp_static_k` / `axis_role`), and `030_split`'s
  finalize/partial kernels carry sliced reduce loops. The node can't be built at recognize time (its `tile` is a
  fork-chosen `TilePlan`; its operand binding needs the mapped `place.grid`). **Both branches stay** and their
  docstrings now describe the flat-contraction `Map` as a real current form. Fully retiring the body-scan would need a
  recognizer refactor (nodify the scalar contraction too, with a deferred `tile`) — a real follow-up, not Phase-0
  residue.
- **Delete `_schedule._with_reduce`'s "returns the op unchanged for a legacy non-`Reduction` op" no-op path** and the
  sibling legacy-pin branches (`_schedule.py` ~158/164/218) — same principle: if the stamp target always exists, the
  no-op arm is dead.
- **Reconcile the docstrings to reality.** Strike every "not-yet-migrated" / "legacy form" / "residual fallback" /
  "reserved during the transition" phrase from `ir/tile/ir.py`, `ir/tile/ops.py`, `_atomize.py` (its `Semiring` /
  `Monoid` op-tree reference), and `_schedule.py`. A docstring that describes a fallback you just deleted is a document
  inconsistency — the audit is not done until grep for those phrases over `ir/tile/` + `passes/lowering/` is empty.
- **Resolve the STAGE "built vs dropped" contradiction now.** The knob table says built, `_factor.py`'s docstring says
  dropped, `kernel/ARCHITECTURE.md` says "reserved", and `xfail_registry._STAGE` says materialization is gone. Pick one
  truth — *materialization dropped, codec stamps* — and make all four say it verbatim (the table above already does).

## The remaining phases — build, then purge

Ordered as requested. Each phase adds config the `factorize` path already structurally supports (no divergent path),
**then immediately deletes what it obsoletes.** Named tests flip XPASS → delete the xfail entry.

### Phase 1 — finish mma (tensor-core flash + symbolic-K masked edges)

**Build.** Two mma capabilities remain, both reusing the existing `Contraction` mma codegen through the node hierarchy.

#### Tensor-core flash — RE-PLANNED: two `Contraction` nodes over a blocked kv (architecture first)

The original plan said "give the flash tree's **inner** `Contraction` an mma `TilePlan`," as if flash already had the
right shape. It does not, and the gap is architectural, not a wiring oversight. The current flash op tree
(`_flash._flash_op`) is:

```
Map(body=[O_i / l_i],
    source=Reduction(role=TWISTED, axis=kv,
        carrier=flash_combine(m_i, l_i, O_i, score, v),   # (max, denom, EXPECT=O) twist
        source=Contraction(QK: S = Σ_dd Q·K),             # the ONLY contraction node
        partial=[scale/mask S, load V, carrier.dissolve()]))
```

Only **QK** is a `Contraction`. **PV is dissolved into the twisted carrier's expectation channel** — `O_i` folds
`O_i·α + p·v` **per single kv element**, so the "P@V" is a rank-1 FMA whose reduce axis *is* the streaming `kv` axis, not
a separate contraction. And `d` (the value dim) is a **grid axis** — one output column per thread, the score recomputed
redundantly for every `d`. mma needs the opposite: `d` inside a PV output **tile**, and `P@V` as a real tiled
contraction. So the scalar tree cannot be "given an mma TilePlan"; it must be **restructured**.

**The clean architecture the user asked for — both QK and PV are `Contraction` nodes, over a *blocked* kv.** Split the
streaming axis into `(kv_block, j)`; stream over `kv_block`, contract within the block:

```
Map(body=[O_i / l_i],
    source=Reduction(role=TWISTED, axis=kv_block,
        carrier=<(m_i, l_i) softmax stats + the O_i rescale>,   # O is NO LONGER a carrier expectation channel
        source=Contraction(QK: S[m, j] = Σ_dd Q·K),             # reduce dd → score tile  (b_trans)
        partial=[<softmax on S → P[m, j], α, rowsum>,
                 Contraction(PV: Oblk[m, d] = Σ_j P·V),         # reduce j → output tile
                 <carrier merge: O_i = O_i·α + Oblk>]))
```

Now QK reduces `dd` and PV reduces the intra-block `j` — **different axes** from the streaming `kv_block`, so neither
duplicates the streaming reduce. The `expect(v)` channel's `lift` (already carried on `Channel` for exactly this — see
its docstring, "a future fragment realizer can lower ⊗ to a contraction (mma)") is **realized as the PV `Contraction`**:
its per-element `p·v` term becomes the PV output `Oblk`, so the carrier keeps its `(m_i, l_i)` stats **and** the O-fold
`O_i = O_i·α + Oblk` (α still generated internally), but the ⊗ is now a tiled contraction node instead of a scalar FMA.
Both contractions factorize through the **same** `_factorize_contraction`; the tier is chosen by each node's `TilePlan`,
never a divergent path.

**Consolidation steps (architecture first, each kept green by the non-xfailed *scalar* flash e2e):**

1. ✅ **Structural seam — a reduce `partial` can carry a nested `Contraction`.** `Reduction.loop` now flattens a
   `Contraction` (a `Stmt`) sitting in its `partial` to its own loop nest in place (`_flatten_nodes`), the same
   recursion the `source` splice does. Backward-compatible (a no-op for a plain partial → every existing reduce lowers
   byte-identically); pinned by `test_reduce_partial_flattens_a_nested_pv_contraction`. This is the QK-on-`source` +
   PV-in-`partial` capability the two-`Contraction` tree rests on. (Step 3's mma tier will factorize the nested PV
   instead of flattening it — `factorize` recursion is a step-3 concern.)
2. ✅ **Rebuilt `_flash._flash_op` to the blocked two-`Contraction` tree (block=1).** `_split_pv`
   (`_flash.py`) rewrites the exp carrier's dissolved `merge` so the expectation fold `O_i = O_i·α + v·P` is redirected
   through a real **PV `Contraction`** `O_i__pv = Σ_j P·V` whose **A operand is the register-resident `P`** (a `copy`
   rebind of the exp weight the carrier already computes — extracted structurally off the `O_i` Accum's `multiply(v, P)`
   value) and whose B is the value `Load`. `O_i` **stays a carrier channel** (so its seed + cross-tier machinery are
   untouched — the online-softmax ordering coupling is resolved by splicing the PV *inside* the generated merge, right
   before the O-fold, where `M`/`α`/`P` are already computed), only the value it folds is redirected. Both Q@K (on
   `source`) and P@V (in `partial`) are now `Contraction` nodes; `Reduction.loop` flattens both. At `block = 1` the `j`
   reduce is a singleton — the scalar streaming degenerate, **numerically identical** to the old inline `v·P` (proven:
   `test_scalar_flash_matches_torch` / `_dynamic_` / `_kv_tile_` / `_flash_chain_causal_and_gqa_` all green). Structure
   pinned by `test_flash_op_is_a_two_contraction_tree`. Chose **option (b)** but realized it *without* carrier-generator
   surgery: keeping `O_i` a carrier channel (not removing it) is what preserves the seed/cross-tier machinery while still
   de-fusing the ⊗ into a contraction.
3. **Layer mma** — QK and PV get mma `TilePlan`s (`kv_block` = the mma tile); `_factorize_contraction` tiles both; a
   flash `store` **sink** (the existing `factorize(store=…)` seam) bridges the QK C-fragment into the softmax twist and
   feeds the resulting `P` fragment straight into PV as an operand **without a gmem round-trip** (the one genuinely new
   primitive); the twisted `(m_i, l_i)` carrier + `O_i` rescale run over in-register fragments.

**Decision (chosen): option (b)** — unify on the blocked form, scalar = its `block=1` degenerate, prove parity against
the non-xfailed scalar flash e2e.

**THE CRUX — PV's A operand is register-resident, not a gmem `Load` (confirmed against the generated merge).** The exp
merge computes `P = exp(s − M)` (`m_i__t5`) and folds `O_i = O_i·α + v·P`. Making PV a real `Contraction` means its **A
operand is `P`** — the softmax probabilities, which live **in registers** (they are computed from the QK score
fragment), not in gmem. The `Contraction` node used to bake in **gmem `Load` operands** everywhere.

**✅ Register-resident A operand landed (this step).** `Contraction.a_load` is now `a_operand: Load | Body` — a gmem
`Load` **or** a computed `Body` producing `P` (its last def is the operand value). New node accessors: `a_body` (the
producing stmts — a singleton `(Load,)` for gmem, the body's stmts for computed), `a_computed`, `a_name`. The gmem path
is **byte-identical** (`a_body` for a `Load` is `(a_load,)`, so `contraction_loop`/`_synth_reduce` splice the same
single-stmt operand body). The **scalar tier** handles a computed A **for free**: the register-tile replication treats
`P = exp(S)` as ordinary K-loop body, so `for j: s=S[m,j]; p=exp(s); v=V[j,d]; oblk__v=v·p; oblk += oblk__v` factorizes
with no gmem A address. The **mma tier** (`_mma_reduce`) asserts `not a_computed` — the fragment-feed is step 3. Proven
by `test_contraction_computed_a_*` (standalone P@V: `O[m,d] = Σ_j exp(S[m,j])·V[j,d]`) in
`tests/compiler/ir/tile/test_structural_reduction.py`. `_atomize.bind_contraction` is unchanged — a computed-A
contraction is **constructed directly** (flash / tests), never recognized (recognition rejects pre-scaled operands), so
the binding path stays gmem-`Load`-only.

✅ **`_flash._flash_op` rebuilt on the register-resident PV `Contraction`** (block=1; see consolidation step 2 above).
**Next concrete step (step 3): mma-tile the two contractions.** Block the `kv` axis so `j` = the mma tile width (not a
singleton); the QK `Contraction` gets an mma `TilePlan` producing a score fragment; the softmax twist runs over the
in-register score; the PV `Contraction`'s A operand is the resulting `P` **fragment** (the register-resident-A mma path
`_mma_reduce` currently asserts against — this is the genuinely new `ldmatrix`-from-fragment primitive) fed straight into
`mma.sync` without a gmem round-trip. Gate: `test_generated_tensorcore_flash_*` / `test_warp_chain_*` flip XPASS while
the scalar-flash e2e stays green.

Flips `test_generated_tensorcore_flash_*`, `test_warp_chain_*`, `test_attention_split_gpu.py`,
`test_attention_coverage.py::test_cooperative_flash_matches_torch`. Scalar-parity risk is real (step 2 changes the
scalar tree); the non-xfailed scalar flash cases are the gate — do not proceed to mma until they are green.

- **Symbolic-K masked mma edges** ✅ **landed.** The transposed-B symbolic-K guard is gone; two gmem-direct
  zero-fill helpers (`dpl_mma_load_b_gmem_trans_kzero` / `…_trans_nclamp_kzero`, the (n,k)-swapped mirror of the
  canonical-B ones) zero the masked-K tail, and the `LdmatrixLoad` renderer dispatches them off `b_trans`. Proven by
  `test_transposed_b_symbolic_k_zero_fills` (structure) + `test_masked_symbolic_accuracy[symbolic_k_trans-*]`
  (accuracy at straddling K = 16/31/130/512/700).

**Purge.**

- ✅ **Deleted the `raise LoweringError("warp tier: transposed-B symbolic-K mma not supported…")`** — landed with the
  masked path above.
- **Rip out the "future work" scaffolding in `_flash.py` and `_atomize.py`** — the scalar-`TilePlan()` note, the E2
  gating comment ("requires warp-flash to first attach that inner geometry"), and the redundant scalar-flash prose. The
  scalar `ScalarAtom` config stays (it is a real config of the one path), but every comment framing tensor-core flash as
  unbuilt goes.
- **Land the mma split-K auto-fork and delete the "pin-only" hedge.** `_schedule.schedule()` must emit unpinned `g<w>`
  candidates (`_splitk_specs` + occupancy gate); then strike "split-K stays pin-only" from the docs and flip the
  structural-fork search tests (`test_structural_push.py`, `test_two_level.py`, `test_resolve.py`,
  `test_diagnostics.py`). Sequence after the golden sweep re-validates (`tile_signature` parity).

### Phase 2 — operand staging (smem + `ldmatrix`)

**Build.** Restore **single-buffer** staging: a `sync`-transport smem slab filled cooperatively (a `__syncthreads` fill,
no prefetch), operands read `ldmatrix`-from-smem. It splices into `reduce_codegen` as a stage step **symmetric across
both atoms** (mma: `ldmatrix` from smem; scalar: from the slab), driven off the `Stage` on the node (`STAGE=d1/sync`).
The stage is a wrapper on the K-loop's operand loads, **not** a second contraction tier. Flips
`test_matmul_coverage.py::test_staged_matches_gmem_direct_bit_for_bit`, `::test_masked_symbolic_m_structure`,
`test_bank_conflicts.py`.

**Purge — the big "no divergent path" win.**

- **Subsume the fused-prologue shared-row staging into the general mechanism and delete the bespoke helpers.** Today the
  *only* surviving staging is `_factor.py`'s `_shared_row_buf` / `_restage_loads` / `_shared_row_fill` /
  `_has_contraction_tail` / `_shared_row` — a special-cased RMSNorm→linear prologue that stages one smem row by hand.
  Once general `Stage` staging exists, that prologue is just a `Stage` on the reduce's input. Re-express it as such and
  **delete all five helpers.** Two staging mechanisms for one concept is exactly the divergence this rebuild exists to
  kill.
- **Flip `TileOp.stage` from "dropped/reserved" to live** and strike "reserved" / "materialization dropped" from the
  knob table, `_factor.py`, `kernel/ARCHITECTURE.md`, and `xfail_registry._STAGE` (delete `_STAGE` itself as its tests
  XPASS).

### Phase 3 — pipelining (cp.async ring / register double-buffer / TMA)

**Build.** Layer the depth/async variants on Phase 2's single-buffer stage, **all as fields on the same `Stage`** — no
new path:

- **cp.async ring** (`STAGE.depth>1`, `sm_80`) — prefetch the next K-chunk's fill over the current mma, a `depth`-slot
  ring.
- **smem→register double-buffer** (`STAGE.reg_depth` / `/p<n>`) — ping-pong `ldmatrix` over the inner atom-K steps
  (`STAGE.depth` = gmem→smem ring; `STAGE.reg_depth` = smem→register; `WarpTile.bk` = slab K-granularity, not a depth).
- **TMA** (`sm_90`, `cp.async.bulk.tensor` + mbarrier) — descriptor-driven bulk transfer; single-buffer then the ring.

Flips `test_cp_async_deep_ring_matches_gmem_direct_bit_for_bit`,
`test_register_double_buffer_matches_single_buffer_bit_for_bit`, `test_bf16_operands_stage_via_cp_async`,
`test_pinned_transport_and_shape_fire`, `test_knob_pinning.py::test_article_tma_sgemm_reproduction`.

**Purge.**

- **Collapse single-buffer into depth=1 of the ring — do not keep two emitters.** If Phase 2 produced a standalone
  single-buffer code path, `depth=1` must now be the ring with one slot; delete the standalone path. The transport
  (`sync` / `cp` / `tma`) is a parameter, not a branch of duplicated fill logic.
- **The kernel-IR transport vocabulary survived the demolition** (`cp.async`, TMA, mbarrier, the ring codec) — if any of
  it is now unreachable after wiring, delete the unreachable renderers rather than leaving them as museum pieces. Grep
  the kernel-IR for transport nodes with no producer.

### Phase 4 — warp specialization (`WarpSpec`)

**Build.** Heterogeneous warps: the CTA partitions into producer / mma / reducer roles wired by shared smem rings.
`WarpSpec` **delegates** — each `WarpRole` carries a sub-schedule that bottoms out in a uniform `Map` / `Reduction` /
`Contraction`, so warp-spec composes the *same three nodes*, never a fourth. The uniform `Stage` splits: the gmem→smem
*fill* becomes the shared `Channel`; each consumer's *local* register double-buffer stays on `role.stage`. `WarpSpec`
lives only at the top CTA level (`TileOp.workers`); roles do not nest.

```python
@dataclass(frozen=True)
class Channel:                       # a shared smem ring — the producer/consumer seam
    name: str
    depth: int
    transport: str = "cp.async"      # cp.async | tma

@dataclass(frozen=True)
class WarpRole:                      # one warp group's job; its sub-schedule node NAMES the role
    stage_node: object               # the Map / Reduction / Contraction this role runs
    warps: int
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    stage: Stage | None = None       # this role's LOCAL smem→register double-buffer

@dataclass(frozen=True)
class WarpSpec:
    place: Placement                 # the CTA-tile grid
    channels: tuple[Channel, ...] = ()
    roles: tuple[WarpRole, ...] = () # Σ role.warps = the CTA warp count
```

`WarpSpec` reuses the role vocabulary (`REDUCE` reducer / `WARP` mma / `STAGE` producer) and adds **role-namespacing**
(`mma:WARP=…` + the `CHANNEL` ring) — new grammar registered so `knob_features` / `apply_off_defaults` /
`tuning_knob_items` handle role-prefixed keys. Spelling: `CHANNEL=K:d3/cp;V:d3/cp mma:WARP=…/k2 reducer:REDUCE=b2
producer:STAGE=d3/cp`.

**Purge — tear down the rebuild scaffolding itself.**

- **`Channel` must not reimplement `Stage`'s transport.** If `Channel.depth` / `.transport` duplicate the ring logic
  Phase 3 built on `Stage`, factor the shared ring into one place and have `Channel` hold the *shared* variant only. One
  transport implementation, period.
- **Delete `TileOp.workers`'s "not yet built" language** and any `WarpSpec` placeholder / `None`-only handling once it
  is live.
- **When `XFAIL` reaches empty, demolish the recovery apparatus.** Delete `tests/xfail_registry.py`, the
  `pytest_collection_modifyitems` hook in `tests/conftest.py`, `TILE_ENTANGLED_FILES`, and unwrap every
  `try/except ModuleNotFoundError` guard around a tile import (the imports are unconditional now). **Then delete this
  plan** — a rebuild plan for a completed rebuild is the last shim.

## Sequencing & verification

```
Phase 0  purge residue    Schedule alias · Map.out/reduce_loop fallbacks · _with_reduce no-op · STAGE doc contradiction
   ▼
Phase 1  finish mma        tensor-core flash (Contraction mma atom + flash store sink) · symbolic-K masked edges
   │  purge → delete transposed-B-symbolic-K guard · flash "future work" scaffolding · split-K "pin-only" hedge
   ▼
Phase 2  staging           single-buffer smem + ldmatrix, symmetric across atoms (Stage on the node)
   │  purge → SUBSUME + delete the _shared_row_* prologue helpers · flip TileOp.stage live, delete _STAGE
   ▼
Phase 3  pipelining        cp.async ring · smem→reg double-buffer · TMA — fields on the same Stage
   │  purge → collapse single-buffer into depth=1 · delete unreachable transport renderers
   ▼
Phase 4  warp spec         WarpSpec roles bottoming out in Map / Reduction / Contraction
      purge → unify Channel/Stage transport · demolish xfail registry + conftest hook + guarded imports · delete this plan
```

- **Per phase:** delete the flipped xfail entries; `./venv/bin/pytest tests/compiler/e2e/ -p no:randomly -n auto
  --dist=loadgroup`; `make lint`. Staging / pipelining tests assert **bit-identity vs gmem-direct** — a stage must not
  change numerics. Guard the learned-prior featurization with a `tile_signature` invariance check (a phase adds a code
  but must not re-key existing kernels).
- **Purge gate (per phase):** grep proves the deletion is total — no surviving reference to the removed symbol, no
  docstring describing the removed fallback, no dead branch. A phase PR that adds capability without its purge is
  rejected.
- **Whole rebuild:** `make test` green with `XFAIL` empty and the registry deleted; no golden regression
  (`make bench-kernels` spot-check on a reduction + a matmul + a flash kernel).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — the nodes + `TileOp` fields; Phase 0 alias/fallback purge.
- `deplodock/compiler/ir/tile/ops.py` — `lower` / `axis_role` / `reduce_loop` / `reduce_plan`; Phase 0 fallback purge.
- `deplodock/compiler/pipeline/passes/lowering/kernel/_factor.py` — the one emitter; Phase 1 flash sink, Phase 2/3 stage
  splice, Phase 2 `_shared_row_*` deletion.
- `deplodock/compiler/pipeline/passes/lowering/kernel/_tiling.py` — the four tiling levels + the `grid_tile` splice.
- `deplodock/compiler/pipeline/passes/lowering/tile/_flash.py`, `.../_atomize.py` — Phase 1 inner-contraction mma
  geometry + scaffolding purge.
- `deplodock/compiler/pipeline/passes/lowering/tile/_schedule.py` — schedule forks; Phase 0 legacy-branch purge, Phase 1
  split-K auto-fork.
- `deplodock/compiler/ir/schedule` — `Stage` (Phase 2/3), `WarpSpec` / `Channel` / `WarpRole` (Phase 4).
- `tests/xfail_registry.py`, `tests/conftest.py` — the recovery ledger + hook; demolished in the Phase 4 purge.
</content>
