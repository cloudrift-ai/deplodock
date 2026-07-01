# Consolidate all kernel lowering into one generalized `factorize`

## Context

Kernel materialization (`lowering/kernel/010_materialize.py:rewrite`, ~287-327) dispatches into **three tiers**:
`factorize()` for output-tiled matmul (`_factor.py` — the `atomize → register_tile → unit_tile → grid_tile` pipeline
in `_tiling.py`), `_reduce()` for cooperative/ILP reductions/softmax/flash/split-K (`010_materialize.py:188-284`, with
`emit_combine` from `_combine.py`), and a scalar fallback (`lower()` + `_with_store`). The goal is **one emitter**:
every structural node (`Contraction`, `Reduction`, `Map`) lowers through a single generalized `factorize`, and the
`_reduce` and scalar tiers are retired.

This duplicates real machinery across the tiers: scalar per-cell emission exists in both `_factor._scalar_reduce` and
the materialize fallback; register-ILP replication/fold exists in both `_factor.register_tile` and `_reduce`; the
tiling levels in `_tiling.py` are re-implemented inline by `_reduce`. Collapsing to one path removes that redundancy
and makes later capability work (mma split-K, tensor-core flash) a *configuration* of one emitter rather than a new
bespoke tier.

## Governing principle: dissolve duplicates first, then extend

The work splits into two parts with different bars:

- **Part I — Dissolve duplicates (behavior-preserving refactor).** Route all three tiers through one factorize-based
  emitter that reproduces today's output. **No new capability**: after Part I, split-K is still scalar and flash is
  still scalar — exactly as today, just one code path. Success is a *parity* bar (IR-identical where structure allows,
  numeric parity + byte-equivalent combine structure elsewhere), with the whole suite green.
- **Part II — Extend on the unified base.** Only after one path exists: mma split-K, then tensor-core (warp) flash.
  Each becomes "enable a tiling/atom/combine combination the unified emitter already structurally supports."

Part II must not precede Part I. (mma split-K is *technically* independent of the dedup — its partial is already a
bare `Contraction` that factorizes — but per this principle it lands on the clean single-path base.)

## Thesis correction (what the mechanism actually requires — verified against code)

The tempting framing "the tiers differ only in nesting order; add a reduce term to `OffsetFn.base()`" is wrong about
where the work lands:

1. **The reduce axis is not an `OffsetFn` concern today.** `OffsetFn.levels` is keyed only by `"m"`/`"n"`
   (`_tiling.py:39,79-83`); `base()` (`:43-53`) computes only output coordinates. The K axis rides as a *sequential*
   loop inside the codegen callable (`_mma_reduce` `StridedLoop(axis=k_axis…)` `_factor.py:157`; `_scalar_reduce`
   `Loop(axis=k_axis)` `:302-310`).
2. **Output vs reduce use different decomposition kinds, not just order.** Output axes are **blocked-contiguous**
   (`base()` = `block·tile + unit·(reg·atom) + r·atom`), each unit/register owning an independent sub-tile → no
   combine. The reduce axis in `_reduce` is **strided-interleaved** (`lane + r·coop + serial·(coop·reg)`,
   `StridedLoop(start=lane, step=coop·reg)`, `010_materialize.py:210-211,247`) producing partials of one output →
   combine required. A reduce level needs a genuinely different coordinate formula.
3. **factorize is `Contraction`-typed.** It reads `c.m_axis/n_axis/a_load/b_load/atom/reg_m/units_m/tile_m/mask_m/…`
   (`structural.py:180-264`). A `Reduction` has `carrier/axis/partial/role/reduce/source` — none of those. Routing a
   plain reduce through factorize needs a node-kind dispatch + a "monoid" reduce-role codegen tier.

**What is already the right seam (load-bearing):** the `store` sink is pluggable (`_factor.py:339-357`,
`grid_tile(store=…)` `_tiling.py:126,151`); `emit_combine` is already decoupled from `_reduce`
(`emit_combine(carrier, t, n_threads, *, warp_size=32, segmented=False)` `_combine.py:32`, reading only carrier fields
+ the coop lane var name/width `010_materialize.py:262`); the semantic loop (`ops.contraction_loop`) and the tiling
levels are shared. So the combine and store relocate without structural surgery; the new work is the role-keyed
partition record + reduce codegen.

---

## Part I — Dissolve duplicates (behavior-preserving)

### D1 — Fold scalar + pointwise into a degenerate factorize  [lowest risk; IR-identical target]

Make `factorize` accept `Map(source=None)` (pure pointwise, `structural.py:312-346`) and the residual scalar case:
`atomize(1,1) → register_tile(1,1) → grid_tile(store=with_store-sink)` with an empty `reduce_region` and the
pointwise body (from `lower(op)`, `ops.py:65-75`) spliced as `top_decls`. This introduces factorize's **node-kind
dispatch** (the skeleton Part I builds on) and kills the duplicated scalar emission.

- **Files:** `_factor.py` (`Map(source=None)` dispatch), `010_materialize.py:322-327` (scalar tier now delegates).
- **Must tolerate:** empty `kstmts` in the `grid_tile` splice (`_tiling.py:151`); `with_store`/`has_write` already
  no-op when the body carries its `Write` (`_store.py:16-35`); `lanes==1` emits no `_lane` axis (`_tiling.py:145`).
- **Parity bar:** emitted `Tile` structurally identical to today's scalar tier (both wrap
  `Tile(axes=place.grid, body=…)`). Verify by diffing dumped kernels on a pointwise/elementwise model + the e2e
  pointwise coverage.
- **Risk (low):** multi-`Write` pointwise bodies and the `op.out` glue (`_store.py:27-35`) — preserve the `has_write`
  recursion.

### D2 — Fold `_reduce` (PLANAR + TWISTED) into a reduce-role codegen  [the crux; numeric + combine-structure parity]

Generalize the tiling layer with a **role-keyed axis-partition record** and add a **reduce codegen pair** analogous
to the mma/scalar pairs, so cooperative/ILP reductions — sum/max/mean, RMSNorm, softmax, and the coop-KV **TWISTED**
flash reduce — lower through factorize. The twisted carrier is *data*, not a separate path (`emit_combine` /
`carrier.as_state_merge` already handle the exp-family branch, `_combine.py:51-53`, `algebra.py:171`), so PLANAR and
TWISTED coop-KV flash both fold in here.

- **`_tiling.py`:** tag `OffsetFn.levels` entries with a role (`"out"` vs `"reduce"`; output entries keep `base()`
  unchanged); add `OffsetFn.reduce_coord(...)` for the strided formula + `StridedLoop` step (lift
  `010_materialize.py:210-211,247`); add a `reduce_tile(t, coop, reg, lane_var, masked)` level that records the
  reduce partition and, when `coop>1`, appends the coop `lane` axis to `offset.axes` and **multiplies**
  `block_threads` by `coop` (composing with the atom `_lane` axis at `:145-146` — they are orthogonal Tile axes).
- **`_factor.py`:** `_monoid_state`/`_monoid_reduce` (take carrier + partial body, emit the strided `StridedLoop` with
  `reg` replicated copies + masked tail, then the REG-tree fold via `carrier.as_state_merge`), and
  `reduce_store_sink` (full-row sweep distributed across lanes, or scalar guarded to `lane==0` — port
  `010_materialize.py:267-279`; `coop==1` degenerates to `_with_store`). Relocate `_mask_streamed`, `_replicate`,
  and the `_shared_row_*` fused-prologue helpers (`010_materialize.py:71-185`) here intact.
- **`grid_tile` splice:** call `emit_combine(carrier, t=lane_var, n_threads=coop)` when `coop>1`, between
  `reduce_region` and `stores`. `emit_combine`'s signature is **unchanged**; `segmented` defaults False (preserving
  current behavior).
- **Parity bar (strong):** `tests/compiler/e2e/test_reduce_coverage.py::_assert_combine_structure` (~:166) pins the
  exact emitted structure — `__shfl_xor_sync` presence, the `_smem[` + `for (int s =` cross-warp tree, register-fold
  replication, and `__launch_bounds__(N)` block size. Target byte-equivalent combine structure + numeric parity via
  `test_cooperative_combine_accuracy` (:198), `test_symbolic_cooperative_softmax_sweep` (:232),
  `test_attention_combine_accuracy` (:257).
- **Risks:** (a) masked-tail identity — the clamp-to-identity + `%extent` wrap must survive the move exactly (a
  `sum(x·x)` prologue needs additive-0, not multiply-1; silent-wrong-answer if botched); (b) `block_threads` must
  *multiply* the coop and atom lane axes, not overwrite (wrong `__launch_bounds__` caught by the structure asserts);
  (c) the SSA `protected` set (`010_materialize.py:241-243`) must move intact or reg pressure/occupancy shifts.

### D3 — Unify the dispatch, delete `_reduce` + the scalar tier

Rewrite `rewrite()` to a single `factorize(op)` call that reads node kind + roles/carriers/plan off the node and
picks atom + combine internally; the three-tier conditionals (`010_materialize.py:301-327`) collapse.

- **Delete:** `_reduce` (`:188-284`) and its now-relocated helpers; the scalar tier block (`:322-327`); the
  `coop_eligible`/`plan` conditional; the `emit_combine`/`_with_store` direct imports (fold into factorize).
- **Grep-confirmed consumers:** `_reduce` has one caller (self, `:320`); `emit_combine` one call site (`:262` → moves
  to `grid_tile`); `reduce_plan` consumers (`030_split.py:151`, `_schedule.py`, `ops.py:42`) read the *node* and are
  dispatch-agnostic — unchanged. Update the `_schedule.py`/ARCHITECTURE.md text references to `_reduce`.
- **Sharpest risk — learned-prior featurization:** keep dispatch/featurization keyed on `reduce_plan(tile)` /
  `axis_role` exactly as today (the featurizer reads the node, not the tier code), and guard with a `tile_signature`
  invariance test (same guard the split-K plan uses). `assert not needs_split` (`:294`) still holds (030_split strips
  GRID first).

**End of Part I:** one `factorize` emitter; matmul (output-tiled), reductions, softmax/RMSNorm, coop-KV flash, scalar
split-K, and pointwise all emit identical structure/numerics. No capability added.

---

## Part II — Extend on the unified base

### E1 — mma split-K  →  detailed in `plans/splitk-structural-fork.md`

Represent split-K structurally as `Reduction(axis=ksplit, source=Contraction(k_axis=kslice))`, introduced by a new
**split-K schedule fork** (`g<w>[a|k]` codec), lowered so the partial is a bare `Contraction` (→ factorize/**mma**),
reusing `030_split`'s cross-CTA finalize; then retire the residual scalar-only `_tile_option` split-K path. See that
plan for the fork catalog, axis factoring (`ksplit` = partition count / reduce axis; `kslice` = per-partition chunk /
contraction axis), 030_split consumption, retirement checklist, and tests.

### E2 — tensor-core (warp) flash  [hard-gated on deferred warp-flash inner-geometry wiring]

The `Map(source=Reduction(TWISTED, source=Contraction(QK)))` with the QK/PV matmuls on tensor cores. Reuses D2's
reduce-role combine + E1's Contraction-in-factorize (the inner `Contraction` factorizes to mma via a flash `store`
sink). **Genuinely new:** the score `S` fragment layout as the *next* matmul's operand without a gmem round-trip, and
the twisted carrier operating on those in-register fragments. **Gating:** `_flash.py:269-276` builds the score
`Contraction` with a bare scalar `TilePlan()`; `_atomize.py:20-26` documents that wiring the recursive
`bind_contraction` on the inner loop "requires warp-flash to first attach that inner geometry" — it does not exist
yet. The tensor-core-flash tests are xfail (`xfail_registry.py:114-128`), so this is parkable and never blocks Part I
or E1.

*(A cross-warp fragment-combine — summing partial mma fragments in shared memory — is the enabling primitive shared by
a warp-split-K variant and E2; build it here if/when E2 is scheduled. Out of scope until then.)*

---

## Sequencing & dependency graph

```
Part I  (dissolve duplicates — behavior-preserving)
  D1 fold scalar+pointwise ─┐   (lowest risk, IR-identical; establishes node-kind dispatch)
  D2 fold _reduce (P+T) ────┤   (the crux: role-keyed tiling + reduce codegen + combine splice)
                            ▼
  D3 unify dispatch + delete _reduce/scalar tier   (requires D1 + D2 green)
        │  END STATE: one factorize path, zero new capability
        ▼
Part II (extend)
  E1 mma split-K            (independent of D-internals, but sequenced after Part I per principle)
  E2 tensor-core flash      (reuses D2 + E1; HARD-GATED on warp-flash inner geometry — parkable, tests xfail)
```

- **Order within Part I:** D1 → D2 → D3. D1 first because it is the lowest-risk, closest-to-byte-identical dedup and
  it builds the node-kind dispatch D2/D3 need. D2 is the crux. D3 only deletes once D1+D2 route through factorize.
- **Independently shippable behind the still-present tiers:** D1, then D2 (each merges alone; the old tiers remain
  until D3).
- **Part II after Part I.** E1 detailed separately; E2 hard-gated and parkable.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/kernel/_tiling.py` — role-keyed `OffsetFn`, `reduce_tile`, `grid_tile`
  combine splice (D2).
- `deplodock/compiler/pipeline/passes/lowering/kernel/_factor.py` — node-kind dispatch (D1), `_monoid_*` reduce
  codegen + `reduce_store_sink` + relocated helpers (D2), single-emitter entry (D3).
- `deplodock/compiler/pipeline/passes/lowering/kernel/010_materialize.py` — tier delegation (D1), dispatch collapse +
  deletions (D3).
- `deplodock/compiler/pipeline/passes/lowering/kernel/_combine.py` — no change (validated decoupled); call site moves.
- `deplodock/compiler/pipeline/passes/lowering/tile/_flash.py`, `.../tile/_atomize.py` — E2 only.
- Tests: `tests/compiler/e2e/test_reduce_coverage.py` (the combine-structure parity gate), `tests/compiler/e2e/`
  pointwise + attention coverage, `tests/compiler/ir/tile/test_structural_reduction.py`.

## Verification

- **D1:** diff dumped kernels for a pointwise/elementwise model before/after (structurally identical); e2e pointwise
  coverage green.
- **D2:** `./venv/bin/pytest tests/compiler/e2e/test_reduce_coverage.py -v` — combine-structure asserts +
  cooperative/symbolic/attention accuracy all green (byte-equivalent structure + numeric parity).
- **D3:** full `make test`; a `tile_signature` invariance test proving the learned-prior featurization is unchanged;
  `make lint`.
- **Whole of Part I:** `make test` green with `_reduce` and the scalar tier deleted; no golden regression
  (`make bench-kernels` spot-check on a reduction + a matmul kernel).
- **Part II:** per `plans/splitk-structural-fork.md` (E1); E2 tests remain xfail until warp-flash lands.
</content>
