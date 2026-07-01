# Split-K matmul: structural `Reduction ⊃ Contraction` via a schedule fork

## Status — ✅ core landed (pin-driven); auto-fork (F.6) deferred

mma split-K works for the first time via the structural fork. Done: axis factoring (`_schedule._factor_k`),
`_schedule._splitk_option` building `Reduction(role=CONTRACTION, axis=ksplit, source=Contraction(k_axis=kslice))`,
the `schedule()` CONTRACTION arm routing a pinned `g<w>[a|k]` through it (`_splitk_pin`), and
`030_split._split_contraction` making the partial a **bare Contraction** (→ `factorize` → mma / scalar, no
`_slice_loop`) with the deferred-kernel + atomic finalizes. `test_mma_splitk_finalize` drives the fork (deferred =
mma, no `atomicAdd`, accurate; the `[deferred]` xfail removed); structural build/lower + `_factor_k` unit tests added.

Deviations from the plan below:

- **`_semiring_reduce_spec` NOT deleted (D.1) — split its two jobs.** Coop/ILP-K on a non-output-tiled contraction
  (`REDUCE=b<n>`/`r<n>`) is a *live, tested* feature (`test_matmul_reduce_partition`), so it stays on the residual
  `reduce` path via the renamed `_coop_reduce_spec` (b/r only). Only the split-K `g<w>` was retired from the residual
  and routed through the structural `_splitk_option`. Consequently the `ops.reduce_plan` residual fallback (D.3) is
  **kept** (coop/ILP-K needs it).
- **mma + atomic (`g<w>a`) is refused**, not supported: an mma C-fragment can't `atomicAdd` (`RegStore` has no atomic).
  Deferred (`g<w>k`) is the mma split-K path; scalar-tier atomic split-K is unaffected (covered by
  `test_reduce_coverage`'s cross-CTA matrix).
- **F.6 (auto-fork enumeration) deferred.** Split-K stays **pin-only** — `schedule()` does not yet emit unpinned
  `g<w>` candidates (`_splitk_specs` + the occupancy gate + golden-sweep re-validation). The structural base is in
  place; enabling auto-fork is a follow-up gated on the golden sweep.

## Context

Split-K partitions a matmul's contraction axis `k` across CTAs: each block contracts a slice of `k`, then the
partial output tiles are summed. Today deplodock has *scalar-only* split-K riding a residual path, and **mma split-K
does not actually work**. This plan makes mma split-K work for the first time by representing it structurally and
introducing it through a schedule fork, then retires the residual path.

> **Position in the consolidation:** this is **E1 (Part II)** of `plans/factorize-consolidation.md` — the first
> capability *extension* that lands on the unified single-`factorize` path. Per that plan's governing principle
> ("dissolve duplicates first, then extend"), it is sequenced **after** Part I (the behavior-preserving dedup of the
> `factorize` / `_reduce` / scalar tiers). It is technically independent of the dedup internals (the split-K partial
> is already a bare `Contraction` that factorizes), but lands on the clean base by intent, not necessity.

**Why now.** This is the deferred "step 7 (fork-catalog)" work called out in the tile-IR rebuild: a plain split-K
matmul has only one reduce (the contraction's own `k` fold), so the naive `Reduction(axis=k, source=Contraction(k))`
is a double-reduce `for k:[for k:]`. The correct form factors `k` into two axes (below); those axes don't exist until
something splits them, and doing that split at *build* time (a fork) rather than *rewrite* time is what unlocks an
mma partial. It also lands the "generalized factorize" consolidation: the split-K partial becomes a bare
`Contraction`, so it lowers through the same `factorize()` path as every other output-tiled matmul.

### `ksplit` vs `kslice`

Split factor `w` (number of partitions) factors the original axis: `K = ksplit × kslice`.

- **`ksplit`** — outer axis, extent `w`. The *partition index*; becomes the outer **`Reduction`'s reduce axis**
  (parallelized across CTAs, summed in the finalize). Additive combine.
- **`kslice`** — inner axis, extent `K/w`. The per-partition chunk; stays as the **`Contraction`'s `k_axis`** (the
  sequential/mma mul-add each CTA runs).

```
for ksplit in [0, w):          # Reduction sums across CTAs
    for kslice in [0, K/w):    # Contraction folds within one CTA
        k = ksplit*(K/w) + kslice
        acc += A[..,k] * B[k,..]
```

Two distinct names (not reusing `k`) is what avoids the double-reduce: the `Contraction` folds `kslice` into a
partial, the `Reduction` sums partials across `ksplit`; every original `k` element is visited once.

## Decisions (locked)

- **Combine: cross-CTA, reuse `030_split.py`** — partials → global workspace → existing atomic (`g<w>a`) or
  deferred-kernel (`g<w>k`) finalize. No new combine machinery.
- **Migrate and retire** the residual `_tile_option` split-K path (scalar-only `g<n>`-on-`Map`); one split-K path
  at the end.
- **Static-K only** first (matches the current `030_split` guard); symbolic-K is out of scope.

## Ground truth (verified against the code — corrects the original framing)

1. `030_split.py` is a **tile** pass (`deplodock/compiler/pipeline/passes/lowering/tile/030_split.py`), runs before
   the kernel pass; materialize dispatch is `010_materialize.py:301` (`isinstance(tile.op, Contraction)` → `factorize`).
2. The split-K codec is **`g<n>[a|k]`** (`ir/tile/schedule.py` `_REDUCE_SCHEMA`), `a|k` = atomic|kernel finalize.
   Goldens use it bare, e.g. `rtxpro6000_sm120.yaml` `REDUCE: 'g2a'`. The `s2/c2a` spelling and `DEPLODOCK_SPLIT` /
   `DEPLODOCK_ATOM` env vars are **dead** (no consumer; `parse("s2/c2a")` throws, swallowed to `""`).
3. `test_mma_splitk_finalize` is **stale/vacuous**: the atomic arm passes as a plain scalar single-CTA matmul; the
   deferred arm is xfail. It does not exercise mma or split-K today.
4. What works today: **scalar-tile** split-K via the residual `Map` (`_tile_option` keeps the `Map` and stamps
   `reduce=ReducePlan.parse("g2a")` on the `TileOp`; `030_split` reads it). The partial op is always a scalar `Map`,
   **never a bare `Contraction`**, so it never factorizes to mma.
5. mma-tile split-K genuinely does not form: `_warp_option` takes no `reduce_spec`, so the grid split is dropped and
   `030_split` `RuleSkipped`s.
6. Split-K is **pin/golden-only** today: `_reduce_specs` never emits `g<n>` candidates, so the autotuner never
   explores it. Adding the fork is net-new.

## Plan

### A. The split-K fork — `_schedule.py`

In the `CONTRACTION` arm of `schedule()`, take an **orthogonal outer product** of the existing tile options with
`{no-split} ∪ {split width w, finalize f}`.

- **`_splitk_specs(tile, place) -> list[str]`** (mirror `_reduce_specs`): emit `g<w>[a|k]` codec candidates via
  `REDUCE.narrow(...)`. Option-0 is `""` (no split) so cold greedy compile is unchanged. Widths `w ∈ {2,4,8,16}`
  gated to `w ≤ k_extent // (atom_k·bk)` (each slice keeps ≥1 inner K-step) and `extent.is_static and extent % w == 0`
  (reuse the `030_split` static/divisibility guard). Offer split-K only when the output grid is small enough that the
  un-split CTA count underfills the SMs (reuse `_pick_coop`'s occupancy/wave gate). Offer both `a` and `k` finalize
  (carrier is additive → both legal); a pin is authoritative through `REDUCE.narrow`.
- **`_splitk_option(...)`**: when `split_spec` is non-empty, factor the axes (B), build the inner `Contraction` with
  the **existing** `_contraction_node(...)` (same node a non-split matmul builds — no new codegen) but with
  `k_axis = kslice` and operand loads reindexed to `ksplit·(K/w) + kslice`, then wrap:
  `Reduction(carrier=<additive>, axis=ksplit, source=<Contraction>, reduce=ReducePlan.parse(split_spec))`.
- **Additive carrier — do not hand-roll.** Build it exactly as plain-sum reduces and `contraction_loop` do:
  `Accum(name=acc, value=f"{acc}__v", op=ElementwiseImpl("add"), dtype=<acc dtype F32>).as_carrier()`
  (`ir/stmt/leaves.py`). This guarantees identity `0.0`, 1 component — the identity `030_split._carrier_identities`
  and the atomic/deferred finalize already read off the carrier, so the finalize needs **zero** changes.
- **Knob keying (load-bearing for golden/prior parity).** Stamp `TILE@<k_orig>` and `REDUCE@<k_orig>` on the
  *original* contraction k-axis name (as `_tile_option` does today), **not** `@ksplit`/`@kslice`. This keeps the kernel
  single-eligible-axis so `_COLLAPSE_FAMILIES` bare-collapses to match the bare golden YAML and keeps the prior
  featurizer (`_reduce_decomp → plan.cta`) invariant. `ksplit`/`kslice` stay internal to the loop nest.

### B. Axis factoring — `_factor_k(k_axis, w)`

No standalone axis-split helper exists; extract one from the σ/extent-shrink pattern already in
`030_split._slice_loop`. Returns `(ksplit, kslice, sigma)`: `ksplit = Axis(name=f"{k}_ks", extent=Dim(w))`,
`kslice = replace(k_axis, extent=Dim(K//w))`, and `sigma = Sigma({k: ksplit·(K//w) + kslice})` applied to the
Contraction's `a_load`/`b_load` so their K index reconstructs the original `k`. Both extents static by A's guard.

### C. Lowering — generalized factorize + `030_split` consumption

Dispatch of `tile.op = Reduction(axis=ksplit, source=Contraction(k_axis=kslice))` (verified):
`axis_role` recurses to `CONTRACTION`; `reduce_plan(tile)` reads the GRID stage off the `Reduction` node (no residual
fallback needed); `030_split.rewrite` fires on `needs_split`.

Add a structural branch to `030_split.rewrite`:

- Partial op = the **bare `Contraction`** (`op.source`), epilogue retargeted to `ws[ksplit, *cell]` (deferred) or an
  `atomicAdd` `Write` (atomic), grid prefixed with `ksplit`. Because it's a `Contraction`, materialize `:301` →
  `factorize` → **mma**. **No `_slice_loop`** (axis already factored; operands already offset).
- **Preserve the warp tile** onto the partial — remove the current `not tier.is_warp` early-out
  (`030_split.py:184-186`) that drops the warp tier, so `factorize` tiles the partial.
- **Finalize unchanged** — deferred kernel seeds via `_carrier_identities`, folds over `ksplit` with
  `carrier.as_state_merge`; atomic `atomicAdd`s the additive state. Both already consume the additive carrier.
- Keep the existing residual `_slice_loop` branch **only** for a genuine non-contraction split (plain `sum` split-K,
  `Reduction(source=None)` + GRID stage); branch on `isinstance(op.source, Contraction)`.

Materialize needs no partial-side change. Confirm the `assert not needs_split` guard (`010_materialize.py:294`) still
holds (030_split strips the split first).

### D. Retire the residual path — checklist

1. Delete `_semiring_reduce_spec` (`_schedule.py`) — it only smuggled a `g/b/r` pin onto the residual.
2. `_tile_option`: drop the `reduce_spec` param and the `reduce=ReducePlan.parse(reduce_spec)` residual stamp; remove
   the tiled-`Map` split-K carve-out.
3. `ops.reduce_plan`: remove the `tile.reduce` residual fallback for contractions once nothing stamps it (assert unused
   for CONTRACTION; grep `reduce=` on `TileOp` constructions to confirm no other producer).
4. `030_split`: delete the residual lowered-loop-nest slicing path for contractions (kept only for plain-sum split).
5. Verify no other `needs_split` / `.cta` / `.finalize` consumer breaks (all read through `reduce_plan`, which now
   resolves to the node).
6. Rewrite `test_mma_splitk_finalize` to drive the fork (`DEPLODOCK_TILE=a:mma_m16n8k16_f16/w2x2/f2x2/k2` +
   `DEPLODOCK_REDUCE=g2a`/`g2k`), keep the `mma.sync…` present / `atomicAdd` absent assertions, and remove the
   `[deferred]` entry from `tests/xfail_registry.py` (it XPASSes once the workspace retarget lands).

### E. Tests

- **Structural unit** (`tests/compiler/ir/tile/test_structural_reduction.py`): build `Reduction(axis=ksplit(2),
  source=Contraction(k_axis=kslice(256)))` with the `Accum(...).as_carrier()` carrier; assert `axis_role` is
  `CONTRACTION`, `reduce_plan.cta == 2`, and `lower(op)` is a **single** `for ksplit:[<kslice contraction loop>,
  <partial>]` — two nested loops with **distinct** axis names, not `for k:[for k:]`.
- **Fork/knob** (`tests/compiler/pipeline/test_knob.py`): assert the CONTRACTION arm now enumerates a `g<n>`
  candidate; assert `REDUCE=g2a` stays bare on the single-contraction kernel and `tile_signature` is invariant between
  the residual golden spelling and the structural fork (golden-match guard).
- **e2e**: `test_mma_splitk_finalize` per D.6; optionally a scalar structural case (`TILE=n16x8/f2x4`, `REDUCE=g2a`)
  proving the scalar path still works through the structural form.

### F. Sequencing (each green before the next)

1. **B** — `_factor_k` + σ remap; unit-test the axis objects/indices. (No pipeline wiring.)
2. **E.1** — structural build+lower / no-double-reduce test. Pure IR, no GPU.
3. **A** (pin-only) — verify `TILE_PASSES` on the mma pin now yields `Reduction(source=Contraction)` with
   `reduce.cta == 2` (flips from "split dropped" to "present").
4. **C** — `030_split` structural branch → mma partial + reused finalize. Emitted src has `mma.sync…`, deferred has no
   `atomicAdd`. This is the xfail→XPASS moment.
5. **D** — retire residual; delete xfail entry; run full matmul/knob/search/eval suite + `make lint`.
6. Enable auto-fork candidates so unpinned `tune` explores split-K; re-run the golden sweep, confirm no golden
   regresses (`tile_signature` parity).

## Risks

- **Double-reduce on axis collision** — `ksplit`/`kslice` must have distinct names (asserted in E.1).
- **Golden/featurizer parity (sharpest)** — keying both families on the original k-axis name keeps the kernel
  single-node so bare-collapse and `_reduce_decomp` stay invariant vs the residual/golden spelling. Guarded by E.2.
- **Additive-carrier dtype/identity** — must be F32 mma-`c`, identity `0.0`, 1 component; derive strictly from
  `Accum(op="add").as_carrier()`, never hand-author.
- **Atomic-vs-deferred selection** moves from residual to the node's `g<w>[a|k]`; both resolve through
  `reduce_plan.finalize`, so the `030_split` finalize branch is unchanged — verify the atomic `_projection_distributes`
  guard still sees the bare matmul epilogue.
- **Symbolic-K** — static-only this task (`_splitk_specs` guards `is_static`).
- **Prior featurization** — unchanged (no `D_splitk` feature; split-K featurizes via `plan.cta`) provided
  single-axis keying holds.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/_schedule.py` — the fork (A), axis factoring (B), residual
  retirement (D).
- `deplodock/compiler/pipeline/passes/lowering/tile/030_split.py` — structural partial → mma, reused finalize (C).
- `deplodock/compiler/ir/tile/structural.py`, `ir/tile/ops.py` — `Reduction`/`Contraction` build + dispatch.
- `deplodock/compiler/ir/stmt/leaves.py` (`as_carrier`), `ir/stmt/algebra.py` (`as_state_merge`) — carrier reuse.
- `tests/compiler/e2e/test_matmul_coverage.py`, `tests/compiler/ir/tile/test_structural_reduction.py`,
  `tests/compiler/pipeline/test_knob.py`, `tests/xfail_registry.py` — tests.

## Verification

- `./venv/bin/pytest tests/compiler/ir/tile/test_structural_reduction.py -v` — structural build/lower.
- `./venv/bin/pytest tests/compiler/pipeline/test_knob.py -v` — fork enumeration + golden/signature parity.
- `./venv/bin/pytest tests/compiler/e2e/test_matmul_coverage.py -k splitk -v` — mma split-K accuracy, both finalize
  modes (deferred XPASSes).
- `make test` then `make lint`.
- Golden sweep re-run to confirm no golden regresses.
</content>
</invoke>
