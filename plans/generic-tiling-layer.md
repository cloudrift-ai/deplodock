# Generic tiling layer — decompose factorize into composable stages; unify warp + scalar register tiling

## Context

`015_factorize` expanded a warp `MmaContraction` via the **monolithic** `_warp_factor.factorize_mma`, which did the
whole four-way GRID/WARP/REGISTER/ATOM split at once (`_axis_base` computed the offset combined). The scalar register
tier (`_reg_tile` in `010_materialize`) did its own, separate register tiling (`_replicate_cells` + `_dedup_loads`). The
goal: make construction the natural composable nesting (atom is the leaf, tiling wraps it) — `atomize → register_tile →
warp_tile → grid_tile` over a `Unit` leaf (an mma **Atom** or a **scalar** cell) — and share the register-tiling code
between the two tiers.

This is a kernel-IR construction concern: the *scheduling decision* is consolidated in `020_schedule` (the `WarpTile` /
`TilePlan` codecs), and `op_cache_key` digests `lower(op.op)` (not the schedule), so staging the *construction* into
composable kernel-IR stages is cache-free — no op-tree rewrite, only how the already-decided schedule is built out.

## What landed (Phase 0 + Phase A) — on `feature/tile-ir-atomize-pass`

- **Phase 0:** atomize inlined into `020_schedule` (resolves the operand→role `AtomBinding` in `_warp_option`, so an
  unbindable atom — a non-`Load` / computed-cone operand — is rejected at fork construction alongside
  `_check_warp_static_k`); the empty Monoid-side unification (`MonoidAtom` / `ReduceBinding`) dropped. Byte-identical.
- **Phase A:** the warp tier is built on the generic layer:
  - `lowering/kernel/_tiling.py` — `atomize → register_tile → warp_tile → grid_tile` over a `Unit`, with `OffsetFn`
    building the per-cell coordinate incrementally (reproduces `_axis_base`), `Operand` (axis-dependence metadata), and
    the body-splice + `Tile` finalize.
  - `factorize_mma` is now `grid_tile(warp_tile(register_tile(atomize(AtomUnit))))`. `AtomUnit` (in `_warp_factor.py`)
    owns the K-loop + operand staging (gmem-direct / cp.async / TMA); the layer owns the offset, the `Tile` axes, the
    mask/guard plumbing, and the splice. The monolithic assembly is gone.
  - **Byte-identical CUDA** (warp matmuls diffed vs baseline), full suite green, lint clean.

## Phase B finding — the deep scalar unification does NOT cleanly exist

The original plan was to fold `_reg_tile` through the same layer (a `ScalarUnit` + a `TileContraction` node `015`
expands), sharing `register_tile` and deleting the scalar helpers. Implementing it surfaced that the warp and scalar
**replication** are different algorithms, not one algorithm over two leaves:

| | Warp (`AtomUnit`) | Scalar (`_reg_tile`) |
| --- | --- | --- |
| Leaf | C `RegFragment` + `LdmatrixLoad` + `MmaSyncPtx` | scalar `Accum` cell |
| Operand load site | *inside* the K-loop, per K-chunk (+ cp.async/TMA staging) | the reduce body, replicated, then `_dedup_loads` |
| Operand dedup | structural (A-per-i / B-per-j separate loops) | syntactic (`_dedup_loads` collapses identical loads) |
| Grid axes | `(m_b, n_b, m_w, n_w, _lane)` — renamed, drops batch, adds warp + lane levels | the **full** grid (leading batch dims kept) + in-place shrink of the last two |

The grid-axis row is the blocker: a single `grid_tile` can't produce both shapes (warp drops batch + adds warp/lane;
scalar keeps the full grid + shrinks in place). And the replication bodies share no code — `AtomUnit`'s fragment loops
vs `_replicate_cells`'s whole-body-replicate-then-dedup are distinct algorithms.

**What genuinely shares** is `OffsetFn` (the offset arithmetic) + the composable-stage **structure**. The replication and
the finalize (axes) are irreducibly tier-specific — the "moderate reuse, the reduce-region stays tier-owned" caveat,
extended to the grid level.

**Forcing the deep unification** would require rewriting the scalar tier onto a fragment-style structural dedup, which
(a) regresses scalar perf — loses `_dedup_loads`'s arithmetic-intensity reuse unless the generic dedup (Phase C below) is
also built — and (b) risks the just-stabilized warp path. Bad trade.

**Decision: bank Phase A.** It captured the real reuse (composable stages + the offset layer, warp byte-identical). The
scalar tier stays as-is. Two non-recommended alternatives if revisited: *shallow routing* (route `_reg_tile` through
`atomize → register_tile → scalar_tile` reusing its own `_replicate_cells` internally — byte-identical, but cosmetic:
the replication still isn't shared), or the *deep* rewrite (only if Phase C lands first).

## Future direction (not scheduled)

- **Phase C — generic operand dedup.** Make `register_tile` emit each `Operand` once per distinct coordinate of its
  `axes` set (the metadata already exists on `Operand`), replacing BOTH the warp structural A-per-i/B-per-j loops and the
  scalar syntactic `_dedup_loads`. Only then can the scalar tier reuse the warp replication without a perf regression.
  This touches the byte-identical warp path, so gate it behind a real need (a third tier, or a second atom family).
- **Warp-flash (`# TODO(warp-flash)` on `MonoidSchedule.WarpTile`).** Flash is a `Monoid` over a nested `Semiring`. The
  `bind_contraction` helper (`lowering/tile/_atomize.py`) is node-addressable so atomize can recurse into the inner
  QK^T / PV contractions once `_flash` keeps them as structural `Semiring` nodes with inner geometry (today it
  `lower()`-s the score straight to loop-IR). The generic tiling layer + `AtomUnit` are the construction substrate that
  reuse would build on.

## Critical files

- `compiler/pipeline/passes/lowering/kernel/_tiling.py` — the generic layer (`Unit` / `Operand` / `OffsetFn` /
  `atomize` / `register_tile` / `warp_tile` / `grid_tile`).
- `compiler/pipeline/passes/lowering/kernel/_warp_factor.py` — `AtomUnit` + `factorize_mma` on the layer.
- `compiler/pipeline/passes/lowering/kernel/010_materialize.py` — `_reg_tile` (the scalar tier, left as-is) +
  `MmaContraction` emission.
- `compiler/pipeline/passes/lowering/kernel/015_factorize.py` — expands `MmaContraction` via `factorize_mma`.
- `compiler/pipeline/passes/lowering/tile/_atomize.py` — the atomize binding helpers (`bind_contraction` — the
  warp-flash recursion seam), called from `020_schedule`.
