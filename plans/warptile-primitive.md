# WarpTile primitive

## Context

`deplodock/compiler/ir/tile/ir.py` carries one `ParallelTile` subclass per binding tier: `GridTile` (blockIdx),
`ThreadTile` (threadIdx, one coord = one thread), `RegisterTile` (per-thread register cell, consumed before kernel
render). The warp tier — one coord = one warp (32 lanes acting collectively) — is **missing**. Two in-flight
consumers want it:

1. **MMA fragment factorization** (`plans/mma-fragment-factorization.md`). WMMA / `mma.sync` has 32 threads jointly
   own one 16×16 cell. A `WarpTile` is the natural binding tier to encode "this axis indexes warps"; the materializer
   reads it to emit one `MmaSync` per cell.
2. **Warp-specialized TMA pipelining** (`tile/085_warp_specialize.py`). Today the pass *extends* the inner
   `ThreadTile` axis by `N_producer_threads / inner_extent` extra slots and σ-shifts the consumer subtree so its
   references see the original range — an axis-arithmetic workaround for the missing warp coord. A `WarpTile` carrying
   a role axis would replace the extension+shift with `WarpTile(axes=(role,), body=Cond(role < N_producer, prod, cons))`.

This plan ships the primitive and its render/launch-bounds wiring only. **No consumer is modified here** — each
follow-up consumer (MMA's `Role.ATOM` + `AtomTile`, WS's refactor) lands in its own plan and can be sequenced
independently.

### Why a separate primitive, not a `ThreadTile` flag

`ThreadTile.render` (`ir/tile/ir.py:528–558`) computes per-CTA threads = `prod(axes.extent)`. A WMMA tower with warp
axes on `ThreadTile` would launch 32× too few threads. The fix is either (a) a `ThreadTile.is_warp_indexed: bool` flag
that triggers a ×32 in two render sites under an `if`, or (b) a new flavor. Option (a) smuggles non-obvious behaviour
into a widely-used type; option (b) keeps the invariant "flavor *type* encodes the binding decision" that the rest of
the tile-IR follows. Option (b) — `WarpTile` — is the choice here.

### Out of scope

- The MMA / WMMA codegen path. Lands separately via `plans/mma-fragment-factorization.md`.
- Refactoring `085_warp_specialize.py` to use `WarpTile`. Lands in a follow-up after this primitive is stable.
- A `WgroupTile` (128-thread warp groups, Hopper sm_90+ wgmma) — slots in beside `WarpTile` the same way.

## Design decisions

1. **`WarpTile(ParallelTile)` lives beside `RegisterTile` in `ir/tile/ir.py`.** Same shape as the other
   `ParallelTile` subclasses (`axes: tuple[Axis, ...]`, `body: Body`). One warp owns one coord tuple; the 32 lanes
   inside execute the body collectively.

2. **`WarpTile` renders to a warp-id decode + an unconditional `lane` decl.** Cooperative form (inside `GridTile`):
   `int warp_id = threadIdx.x / 32;` followed by row-major decode into the warp axes, then `int lane = threadIdx.x &
   31;` unconditionally (the body presumes lane is available — that's the whole reason a warp coord exists). Mirrors
   the `_render_grid_axis_decode` pattern `ThreadTile` uses for its thread axes. Standalone (no `GridTile` wrapper) is
   not supported in v1 — pointwise kernels use `ThreadTile`, and a top-level standalone `WarpTile` has no use case
   yet. Raise `NotImplementedError` from that render branch with a clear message.

3. **`WarpTile` and `ThreadTile` are mutually exclusive inside one `TileOp.body`.** Both bind `threadIdx`; mixing them
   re-binds the same coord at two scopes. `TileOp.__post_init__` (`ir/tile/ir.py:1109–1117`) currently enforces "at
   most one outer `GridTile`/`ThreadTile`"; extend to include `WarpTile`. The MMA case is `GridTile > WarpTile > …`;
   scalar is `GridTile > ThreadTile > …`; pointwise is standalone `ThreadTile`. Never `ThreadTile > WarpTile` or
   `WarpTile > ThreadTile` nested.

4. **`_launch_bounds_for` reads `prod(warp_extents) × 32` from a `WarpTile`.** Sits beside the existing `ThreadTile`
   branch in `ir/kernel/render.py:286–307`. The render-target `_BLOCK_SIZE = 256` fallback stays for the
   no-`GridTile` standalone case.

5. **`_build_linear_tid` gains a `WarpTile` arm.** Today (`100_materialize_tile.py:470–489`) it returns a row-major
   `Var`-flatten of `ThreadTile.axes`. For `WarpTile`, the equivalent is a flatten of the warp axes for the warp_id
   piece; lane is implicit. Callers that need a single linear thread id keep using `threadIdx.x` directly (it's a
   builtin).

6. **`Role.WARP` is added to the planner-internal `Role` enum in `010_partition_loops.py`.** No planner branch emits
   it in this plan — `_layer_kind_for(Role.WARP) → "warp"` and `_wrap_tower`'s grouping gain a `"warp"` case so
   downstream plans can flip a tier without revisiting the wrap-tower mechanics. Goldens stay byte-identical because
   no `TileParams` requests `Role.WARP` yet.

7. **Helpers generalize from `ThreadTile` to "the inner `ParallelTile`."** `tile/_helpers.py::single_tile` and
   `thread_tile_of` (lines 52–110) hardcode `GridTile`/`ThreadTile`. Extend to recognise `WarpTile` everywhere they
   recognise `ThreadTile` — either by listing it in the `isinstance` tuple or by switching to `ParallelTile`. The
   `thread_tile_of` name becomes a misnomer; rename to `parallel_tile_of` and re-export the old name as an alias to
   avoid breaking in-flight imports.

## M1 — Add `WarpTile` to Tile IR (no emitter)

**Why.** Land the primitive in isolation. No upstream pass emits it; no downstream pass sees it. Every test stays
green.

**Change.**

- `deplodock/compiler/ir/tile/ir.py`:
  - New `@dataclass(frozen=True) class WarpTile(ParallelTile)` beside `RegisterTile`. `_pretty_label` returns
    `"warp"`. `render(ctx)` implements the cooperative form per Design decision 2; the standalone branch raises
    `NotImplementedError("WarpTile outside GridTile not supported in v1")`.
  - Export `WarpTile` from `__all__`.
  - `TileOp.__post_init__`: extend the `n_tiles` count to include `WarpTile` (it counts outer
    `(GridTile, ThreadTile)` today; make it `(GridTile, ThreadTile, WarpTile)` and validate not both `ThreadTile` and
    `WarpTile` appear).
- `deplodock/compiler/ir/stmt/blocks.py`: `_body_uses_lane_warp` keeps its meaning for `ThreadTile`. `WarpTile` always
  emits `lane`; no helper-detection branch needed for the warp-tile side.

**Files.**

- `deplodock/compiler/ir/tile/ir.py` (~60 lines: class + render + `__all__` + validation)

**Verification.** `make test` — no change. New unit test `tests/compiler/ir/test_warp_tile.py`: construct a
`WarpTile(axes=(Axis("m_w", 2), Axis("n_w", 4)), body=Body([…]))` inside a `GridTile` inside a `TileOp`; assert
pretty-print contains a `└ warp` bracket; assert `TileOp.__post_init__` rejects a sibling `ThreadTile`.

## M2 — `_launch_bounds_for` + `_build_linear_tid` + materializer arms

**Why.** Without these, a `WarpTile`-bearing kernel would launch the wrong thread count and emit nonsense tid
decode. Test independently using hand-built IR — no upstream emitter needed.

**Change.**

- `deplodock/compiler/ir/kernel/render.py::_launch_bounds_for`: add a `WarpTile` branch that returns
  `prod(child.axes.extent) * 32`. Mirrors today's `ThreadTile` branch.
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py`:
  - `_materialize_top` (`:95–121`): add `WarpTile` arm. Walks `GridTile.body` for a `WarpTile` child and calls a new
    `_materialize_warp` parallel to `_materialize`.
  - `_materialize_warp` (new, parallel to `_materialize` `:129–436`): same structure but `tid_expr` is a warp-id
    flatten over `WarpTile.axes`; `n_threads = prod(extents) * 32`. The cooperative `Accum` combine path passes
    `n_threads` and the warp's `tid_var` to `emit_combine` — verify `_combine.single_thread_var` handles a
    `WarpTile`-derived var the same way (today it walks `thread_axes`; for the warp case it'd walk warp axes — the
    semantics are "the one thread that owns the accumulator broadcast slot," still index 0 of whatever the local
    binding-tier coord is).
  - `_build_linear_tid` (`:470–489`): no change — callers that need it stay on the thread-axes flatten. Add a sibling
    `_build_warp_id_expr` for `WarpTile`'s warp_id flatten.
- `deplodock/compiler/ir/tile/ir.py::TileOp._launch_geometry` (`:1120–1138`): extend so the second tuple becomes
  whichever inner `ParallelTile` axes are present (`ThreadTile.axes` or `WarpTile.axes`). Document that the **kind**
  of inner tile matters to callers — return a 3-tuple `(block_axes, inner_axes, inner_kind: Literal["thread",
  "warp"])`, or expose two named accessors `_thread_axes()` / `_warp_axes()`. Pick the lighter touch — likely the
  accessor split is cleaner than a tuple length change since existing callers pattern-match on a 2-tuple.

**Files.**

- `deplodock/compiler/ir/kernel/render.py` (~12 lines)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` (~80 lines for `_materialize_warp` +
  dispatch; most of the body is copy-from-`_materialize` with the tid_expr swap)
- `deplodock/compiler/ir/tile/ir.py` (~20 lines: launch-geometry accessor split)

**Verification.** New test `tests/compiler/ir/test_warp_tile_render.py`: hand-build a minimal kernel —
`GridTile(axes=(M_b,), body=WarpTile(axes=(M_w,), body=[Write(C, …, value=1.0)]))` with `M_b=4, M_w=2` — feed
through `render_kernelop`. Assert the rendered CUDA contains `__launch_bounds__(64)` (2 warps × 32), `int warp_id =
threadIdx.x / 32;`, `int lane = threadIdx.x & 31;`, and the M_w decode. Compile via the project's NVRTC fixture and
assert no compile error. No existing golden moves because nothing emits `WarpTile` yet.

## M3 — Planner `Role.WARP` hookup (still no emitter)

**Why.** Wire the primitive into the planner's tower-builder so downstream plans (MMA, WS-refactor) can flip a tier
to `Role.WARP` without touching `_wrap_tower`'s mechanics.

**Change.** In `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py`:

- `class Role` (`:99–112`): add `WARP = "warp"`.
- `_layer_kind_for` (`:567–574`): `Role.WARP → "warp"`.
- `_wrap_tower` grouping (`:526–563`): add a `"warp"` case beside `"grid" / "thread" / "register"` that wraps in
  `WarpTile(axes, body)`. WARP groups consecutive same-kind axes the same way the other parallel kinds do.
- No call site emits `Role.WARP` in this plan.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (~12 lines)

**Verification.** Unit test: construct a synthetic `layers = [(Axis("m_w", 2), Role.WARP), (Axis("n_b", 4),
Role.BLOCK)]` and assert `_wrap_tower(layers, [<stub>])` returns a `(GridTile(n_b), WarpTile(m_w), <stub>)` tower.
`make test` byte-clean — no planner branch reaches the new path yet.

## M4 — Helper + downstream-pass audit

**Why.** Every tile / kernel pass that introspects flavor type explicitly must recognise `WarpTile`. Missing one is a
silent passthrough or a crash when a consumer first emits a `WarpTile`.

**Change.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_helpers.py`:
  - `single_tile` (`:93–110`): match `(GridTile, ThreadTile, WarpTile)`.
  - Rename `thread_tile_of` → `parallel_tile_of`; recognise `WarpTile` everywhere `ThreadTile` is recognised. Keep
    `thread_tile_of` as a deprecated alias that calls `parallel_tile_of` (one release of overlap; remove after the
    MMA + WS consumers ship). Same for `replace_thread_tile_body` → `replace_parallel_tile_body`.
- Audit pass — read every rule under `pipeline/passes/lowering/tile/` and `pipeline/passes/lowering/kernel/`:
  - Any `isinstance(s, ThreadTile)` that should be "the inner parallel tile" becomes `isinstance(s, (ThreadTile,
    WarpTile))` or `isinstance(s, ParallelTile)`.
  - Any `isinstance(s, GridTile)` / `isinstance(s, RegisterTile)` stays as-is — those are tier-specific.
- `deplodock/compiler/pipeline/passes/lowering/tile/085_warp_specialize.py::_find_thread_tile` (`:111+`): not in
  this plan's scope to refactor, but verify the pass doesn't misclassify a downstream-emitted `WarpTile` as a target.
  Today no pass emits `WarpTile`, so a defensive `isinstance(s, ThreadTile)` (not `WarpTile`) keeps WS scoped to the
  scalar-tower input it expects.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/_helpers.py` (~20 lines: rename + alias + `WarpTile` recognition)
- Audit edits scattered across passes (~5–30 lines total, depends on what the audit finds)

**Verification.** `grep -rn "isinstance.*ThreadTile" deplodock/compiler/pipeline/` produces a finite list; each is
either consciously thread-only or has a `WarpTile` arm added. `make test` byte-clean. Re-run the M1/M2 hand-built
`WarpTile` kernel through the *full* pipeline (not just render) and confirm no pass crashes on the new flavor.

## M5 — Documentation

**Change.**

- `deplodock/compiler/ir/tile/ir.py` module docstring: add `WarpTile` to the flavor inventory.
- `deplodock/compiler/ir/ARCHITECTURE.md` (~line 54): add `WarpTile` to the Tile-IR flavor list with a one-line role
  description ("warp-parallel tile; one coord = one warp, 32 lanes execute the body collectively").
- `deplodock/compiler/pipeline/ARCHITECTURE.md`: if the partition-planner section describes the binding tiers
  (`BLOCK · THREAD · REGISTER`), add a forward-pointer that future plans (MMA, WS-refactor) will introduce a `WARP`
  tier via this primitive.

**Files.**

- `deplodock/compiler/ir/tile/ir.py` (~5 lines)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~3 lines)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` (~3 lines)

**Verification.** `make lint` — markdown wrap at ~120 chars (CLAUDE.md convention).

---

## Failure modes to watch

- **Launch-bounds mismatch.** The M2 `_launch_bounds_for` `WarpTile` branch must return `prod(extents) × 32`. A
  missing ×32 launches one thread per warp coord (1/32 of what's needed) → kernel runs but data races silently. The
  M2 NVRTC-compile test catches the symbol-level miscount; the silent-data-race version only surfaces in M2-followup
  consumer tests. **Assert the integer value explicitly** in the M2 unit test, not just "compiles."
- **`ThreadTile` / `WarpTile` mixed in one body.** `TileOp.__post_init__` (M1) rejects this. If a consumer ever
  *constructs* a TileOp with both — typically by composing two rewrites — the constructor catches it. Tests should
  exercise the rejection path.
- **`thread_tile_of` callers missed by the audit.** Renaming + alias buys one release of safety. The deprecated alias
  should `logging.warning(stacklevel=2)` so the remaining callers surface in test logs.
- **`_combine.single_thread_var` semantics under WarpTile.** Today it walks `ThreadTile.axes` to find "the thread
  that owns the broadcast slot." For `WarpTile`, the analogous concept is "the warp that owns the broadcast slot,"
  and within that warp, lane 0. M2's `_materialize_warp` must pass the right var; verify by reading
  `single_thread_var`'s callers and (if it's not already abstract enough) generalise to take an "inner-binding-tier
  axes" tuple.

## Follow-up consumers (out of scope)

- **MMA fragment factorization** (`plans/mma-fragment-factorization.md`). M1/M3 of that plan emits `Role.WARP` axes
  in the matmul tower for `ATOM_KIND != "scalar"`; with this plan landed, no `_wrap_tower` work is needed there.
- **`085_warp_specialize.py` refactor.** The current pass extends `ThreadTile.axes` by `N_producer_threads /
  inner_extent` slots and σ-shifts the consumer subtree. After this primitive lands, the pass can be reframed: wrap
  the existing inner `ThreadTile` (or `WarpTile`, when MMA is also active) in an outer `WarpTile(axes=(role_axis,),
  body=Cond(role_axis < N_producer, prod, cons))`. Benefits the pass already pays for the hard way:
  - The σ-shift on the consumer subtree drops (warp_id < N_producer is structural, not arithmetic).
  - The consumer-side `AsyncWait` no longer needs `barrier_id=1, barrier_count=n_consumer_threads` metadata to coax
    the materializer's trailing `Sync` into a named `bar.sync` — the structural `WarpTile`-region marker tells the
    materializer to pick the right barrier without per-`AsyncWait` annotation.
  - The pretty-print becomes self-documenting (`WarpTile(role) > Cond(role<P, prod, cons)` vs today's
    "extend-then-shift" idiom).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — `WarpTile` class; `TileOp.__post_init__` validation; `_launch_geometry`
  accessor.
- `deplodock/compiler/ir/kernel/render.py` — `_launch_bounds_for` `WarpTile` branch.
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — `_materialize_warp` +
  `_materialize_top` dispatch + `_build_warp_id_expr`.
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — `Role.WARP`, `_layer_kind_for`,
  `_wrap_tower` grouping.
- `deplodock/compiler/pipeline/passes/lowering/tile/_helpers.py` — `single_tile`, `parallel_tile_of` (rename),
  `replace_parallel_tile_body` (rename).
- `deplodock/compiler/ir/ARCHITECTURE.md`, `deplodock/compiler/pipeline/ARCHITECTURE.md` — docs.
