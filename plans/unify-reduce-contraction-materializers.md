# Unify `_reduce` and `_factor` into one carrier-generic reduce-tiling materializer

**Branch:** `refactoring/tile-ir-rebuild` · **Status:** foundations landed (steps 1–3); the
observable merge (steps 4–5) is blocked on a scheduling gap.

## Goal

One carrier-generic reduce-tiling codegen where the carrier (`Monoid` | `Semiring`) supplies the
inner reduce body + fold, and the tiling machinery (register-fold ILP, cooperative combine,
projection — eventually the output register tile + mma) is shared. The unifying fact: **a `Semiring`
contraction is a `Monoid` reduction whose inner body is `a⊗b`.**

Today two materializers tile **orthogonal** axes, which is why this is multi-step:

- `_factor.factorize` (`lowering/kernel/_factor.py`) — tiles the **output** axes (`reg_m×reg_n`
  independent cells) + mma. Serial K per thread/warp. Consumes `SemiringSchedule.tier` (a `TilePlan`).
- `_reduce` (`lowering/kernel/010_materialize.py`) — partitions the **reduce** axis (`coop` split +
  ILP `reg` fold + cross-thread combine + projection). Consumes `MonoidSchedule.reduce` (a `ReducePlan`).

A full GEMM wants **both** (output tile AND reduce partition); today they don't compose.

## The two schedules (nearly identical — `ir/tile/schedule.py`)

Shared core (5 fields): `place`, `block`, `reduce: ReducePlan`, `stage`, `workers`. Differences:

- `MonoidSchedule.warp_tile: TilePlan | None` — reserved flash inner-QK/PV tile (`TODO(warp-flash)`).
- `SemiringSchedule.tier: TilePlan | None` — the output tile (scalar reg sub-tile OR mma, discriminated
  by the `TilePlan.atom`); plus `bind: AtomBinding | None` (operand→role A/B, `b_trans`).

**Both already carry `reduce: ReducePlan`** — so a cooperative contraction is representable on the
schedule. The gap is purely the materializer (below).

## Work completed this session (commits, oldest→newest)

Codegen consolidation (the contraction factorizer):
- `4a179584` consolidate scalar+mma codegen into one `_factor.codegen` dispatched off the atom;
  delete `_warp_factor.py` / `_scalar_factor.py`.
- `b12e3a0a` drop unused `Contraction.output`; make `b_trans` a derived property.
- `e55b0e02` merge `SemiringSchedule.tile`/`warp_tile` → one mutually-exclusive `tier`.
- `5df1159d` merge `WarpTile`+`TilePlan` → one atom-discriminated `TilePlan` (atom defaults to scalar;
  `is_warp = isinstance(atom, AtomKind)`; `units`/`regs` tuples, tier-native order + `units_m/...` accessors).
- `49886a5f` tuplify `Contraction.axes`/`units`/`regs` to mirror `TilePlan`.
- `49bee371` separate `Contraction` **schedule** (one `tile: TilePlan` field) from **algebra params**
  (axes / k_axis / a_load / b_load / acc / epilogue).
- `5fa2db62` lift the contraction store to a swappable **sink**: `factorize(c, store=None)`;
  `reduce_codegen` (reusable, sink-agnostic `(state_decls, reduce_region)`) + `store_sink` (default matmul).

Flash + batched-mma validation:
- `cf9dd8f9` pin a **batched** transposed-B `Q@Kᵀ` mma test (`test_mma_batched_qk_matches_torch`) — it
  surfaced a real bug (wrong on warp, correct on scalar).
- `7049c6e9` **fix** it: the lift's free-axis order can diverge from the output layout, so the binding
  put `m_axis` on the *contiguous* output axis and `n_axis` on the *row* — inverting the mma store's
  coalescing invariant (`float2` per-lane store needs `n_axis` = the contiguous output dim). Fix:
  `_order_free_by_output` in `010_recognize` orders the lifted free/grid axes to match the output
  Write's index order (no-op for 2D / bare contractions).
- `fcb6e847` validate batched **canonical-B** `P@V` mma (`test_mma_batched_pv_matches_torch`) — passes.
- `4ee62d7d` flash's inner `Q·K` score is now a first-class `Semiring` (the score `Map`'s `source`),
  lowering through `ops._lower_semiring` (the shared contraction lowering); accuracy-preserving.

Unification foundations (the `_reduce`/`_factor` merge):
- **Step 1** `2f1bb41e` `Semiring.as_monoid()` — the carrier bridge ("Semiring = Monoid with a ⊗ lift";
  state = the additive fold, single partial = the operand ⊗ product, `id`-twist). Verified byte-identical:
  `lower(semi) == lower(semi.as_monoid())`. Mirrors `Accum.as_monoid()`. Purely additive (not yet wired).
- **Step 2** `997e53fe` `_factor._synth_reduce` now builds the scalar reduce loop via `lower(Semiring)`
  (the carrier-generic `_lower_semiring`) — so **both** materializers source the contraction reduce body
  from one place (`ops.lower`). Byte-identical.
- **Step 3** `e01c0cba` extract `_geom.copy_cell(body, sigma, suffix, protected)` — the **one**
  replicate-and-rename mechanic shared by `_factor` (register tile, per output cell `__c{i}_{j}`:
  `_scalar_cells`/`_scalar_store`) and `_reduce` (ILP reg fold, per accumulator `__r{r}`: `_replicate`).

All steps verified green across e2e (matmul / attention / reduce / ops / accuracy / block / fused /
pipeline / rebind), passes, ir, golden, codec, trace. `-O1` nvcc lane.

## What's in place for the merge

1. **Carrier bridge** — `Semiring.as_monoid()` (a contraction can become a `Monoid` carrier).
2. **Shared reduce-body generation** — both source the reduce loop from `ops.lower`.
3. **Shared reduce replication** — both use `_geom.copy_cell`.
4. **Reusable contraction codegen** — `_factor.reduce_codegen` (state+reduce, sink-agnostic) +
   swappable `store=` sink; output-tile geometry derived on `Contraction` from `tile × axes`.

## Blocker (found by probing)

Pinning `REDUCE=b4`/`r4` on a matmul is **silently ignored** — same 256-thread kernel, no cooperative
combine. The schedule *can* hold the reduce plan; the materializer doesn't apply it:

- `005_contract` / `factorize` read `sched.tier` but **never `sched.reduce`** → K-loop is always serial.
- `_reduce` (which consumes `reduce.coop`/`reduce.reg`) is gated on `MonoidKernel`, so a `SemiringKernel`
  never reaches it.

So a `SemiringSchedule.reduce` cooperative/split-K partition has **no effect** today.

## Next steps

### Step 4 — make `_reduce` carrier-generic + route a contraction to it (the first *observable* milestone)
- `_reduce`: at carrier extraction, `if isinstance(carrier, Semiring): carrier = carrier.as_monoid()`
  (uses step 1; the rest already uses the `Monoid` API: `.axis`/`.state`/`.as_state_merge`/`emit_combine`).
- Dispatch: in `010_materialize.rewrite`, route a `SemiringKernel` whose `schedule.reduce` cooperates
  (`coop>1` or `reg>1`) to `_reduce` (today the gate is `isinstance(kernel, MonoidKernel)`).
- **Scheduling prerequisite** (the real gate): the scheduler must actually *produce* a cooperative
  contraction. Investigate why `REDUCE` is dropped for a `Semiring` in `020_schedule` (does `_tile_option`
  thread `reduce_spec` for contractions? is the knob enumerated for Semirings?). Start with a *non-output-
  tiled* contraction + a `coop`/`reg`-K reduce (avoids composing with the output tile) so `_reduce` does
  the reduce partition with no output tile.
- **Validate:** a split-K / coop-K matmul (`REDUCE=b<n>`/`r<n>`, no `TILE`) vs numpy. The xfailed
  `test_mma_splitk_finalize[deferred]` is the related target.

### Step 5 — compose the two tilings in one carrier-dispatched materializer
- Output register tile (`_factor`) + reduce partition (`_reduce`) become two stages of one path.
  Contraction = `Semiring` carrier (mma/scalar sink); reduction = `Monoid` carrier. The `reduce` and
  `tier` fields on the schedule are consumed by the *same* materializer.

### Parallel track — warp-flash (separate from the `_reduce` merge, but related)
Flash's QK is now a live `Semiring` (commit `4ee62d7d`). Remaining to get tensor-core flash:
- Route the live QK `Semiring` through `_factor.reduce_codegen` (mma) *inside* the flash kv-loop, with a
  flash sink (score fragments → rowmax/exp/rescale). Batched QKᵀ/P@V mma reuse is already validated
  (`7049c6e9` / `fcb6e847`).
- PV is the entangled half: `P = exp(score − m)` is a *computed* operand (not a plain `Load`), folded by
  the twist — `bind_contraction` requires plain-Load operands, so PV needs the twist threaded through the
  accumulator. Do QK first.
- The `test_generated_tensorcore_flash_*` cases in `tests/xfail_registry.py` (reason `_R`) are the targets.

## Key invariants / gotchas (don't relearn these)

- **mma store coalescing:** `n_axis` MUST be the contiguous (innermost) output axis — the `float2`
  per-lane fragment store coalesces along it. The lift/binding must respect the output layout, not the
  loop/naming order (`_order_free_by_output`). Scalar is immune (real strides everywhere).
- `Semiring.as_monoid()` and `_lower_semiring` are byte-identical (operands order, `fold.value` name,
  `id`-twist). Keep them so.
- `Contraction` stores `tile: TilePlan` (the whole schedule) + algebra params; `units_m`/`reg_m`/`atom`/
  `tile_m` are properties delegating to `tile`. `TilePlan` stores `units`/`regs` tier-native; the
  `_m`/`_n` accessors normalize via `is_warp`.
- Test lane: `make test` compiles at `-Xcicc -O1` (correctness, ~3× faster nvcc); perf tests skipped.
  This GPU is RTX 5090 = **sm_120**, so the warp mma tier is functional locally.
