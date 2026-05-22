# Planner Emits Tiles Directly

## Context

After the tile-flavor refactor (merged into `feature/partition-planner`), the IR pipeline still has a two-step
dance for binding decisions:

1. `000_partition_planner` picks the partition, builds a Loop tower, and stamps a `Role` enum on each Loop
   (`Role.BLOCK`, `Role.THREAD`, `Role.REGISTER`, `Role.SERIAL_OUTER`, `Role.STAGE_INNER`,
   `Role.COOPERATIVE_STRIDE`, `Role.SPLITK_BLOCK`, `Role.PIPELINE`). Output is **LoopOp**.

2. `001_launch_geometry` reads the role tags, strips the outer Loop chain into `Tile.axes`, runs coordination
   (atomic-Write for splitk lifts, Combine emission for cooperative lifts), then converts every remaining
   `Loop(role=R)` into the matching tile flavor (`SerialTile(kind=…)` / `RegisterTile` / `StridedTile`). Output
   is **TileOp**.

The planner already commits to each axis's binding at step 1. Re-deriving that binding from the Role tag in
step 2 is the same decision encoded twice. We already hit one regression from the impedance mismatch:
`is_matmul_reduce` in `_helpers.py` had to accept both `Loop` and `SerialTile` because the planner walks Loop
IR while downstream passes walk Tile IR.

Goal: collapse the dance. The planner constructs typed tile flavors directly. `Role` enum is deleted.
`001_launch_geometry` shrinks to coordination only.

**Why now**: refactor context is fresh; tile flavors are settled; the planner's tower construction sits in one
function (`_wrap_tower`) that's the natural injection point.

## Target Shape

### Planner output

```
LoopOp(body=<raw>)
  │
  ▼  000_partition_planner
  │
TileOp(body=(
    GridTile(axes=(M_b, N_b, …), splitk_axes=(K_s,) if splitk else (),
      body=Body((
        ThreadTile(axes=(M_t, N_t, …), cooperative_axes=(K_c,) if coop else (),
          body=Body((
            RegisterTile(axes=(M_r, N_r), body=Body((
              SerialTile(K_o, kind="serial_outer", body=Body((
                SerialTile(K_i, kind="stage_inner", body=Body((
                  <leaf compute: Load / Assign / Accum / Write>,
                ))),
              ))),
            ))),
          )),
        ),
      )),
    ),
))
```

Pre-register-tile shape (when planner doesn't tile registers): no `RegisterTile`, the K-inner loop sits directly
under `ThreadTile.body`. Pre-K-split shape (single-CTA reduce): no `SerialTile(serial_outer)`, just one
`SerialTile(stage_inner)`.

### `001_launch_geometry` post-refactor

Renamed to `001_coordination.py` (or kept as-is — the name still fits). Scope:

- **Splitk atomic-Write rewrite.** Walk every `GridTile` whose `splitk_axes` is non-empty; descend to Writes,
  rewrite values that aren't axis-dependent to atomic-add (today's `_rewrite_for_atomic_lift` logic).
- **Cooperative Combine emission.** Walk every `ThreadTile` whose `cooperative_axes` is non-empty; find the
  reduce subtree(s) below it; emit `Combine` siblings; guard scalar Writes with `Cond(coop_axis == 0)`
  (today's `_rewrite_for_cooperative_lift` logic).
- **Kernel naming.** Stays here — derives from the LoopOp's shape via `_kernel_name_for`. Or move into the
  planner since the planner already has the input LoopOp; either works.

Everything else in today's `001_launch_geometry` deletes: `_strip_outer_free_chain`, `_lift_output_loops`,
`_convert_stmt`, `_convert_tile_to_flavors`, `_bind_for_role`.

### `Role` enum

Deleted from `ir/axis.py`. Two replacement strategies:

- **A**: planner-private label class. A simple `enum.Enum` inside `000_partition_planner.py` that distinguishes
  layer kinds during construction. Never reaches IR.
- **B**: pass axis-binding decisions as constructor arguments to `_wrap_tower` directly (e.g. `_wrap_tower([
  ("BLOCK", N_b), ("THREAD", N_t), ("REGISTER", N_r), ("SERIAL_OUTER", K_o)], body)`). No enum at all.

Pick **A** to keep `_wrap_tower`'s call sites readable. The label enum becomes an implementation detail.

## Strategy

Same shape as the tile-flavor refactor: **fork off `feature/partition-planner`, nuke-and-rebuild the planner +
`001_launch_geometry` in one big-bang commit, then fix tests incrementally with one commit per test or tight
group.**

The surface is narrower than the tile-flavor refactor — most of the Sigma σ-rewrite, kernel-shape detection,
variant enumeration, and pruning machinery stays unchanged. The rewrite is concentrated in:

- `_wrap_tower` (the tower constructor) — switch from `Loop(axis, role=Role.X, body=…)` to typed-flavor
  constructors.
- `001_launch_geometry` — gut everything except coordination.
- `is_matmul_reduce` in `_helpers.py` — drop the legacy `Loop`/`StridedLoop` cases (no longer needed; the
  planner emits tile flavors throughout).
- `_helpers.py::single_tile` — unchanged (still finds GridTile/ThreadTile).
- Downstream passes (`002_stage_inputs`, `006a`, etc.) — unchanged. They only saw tile flavors anyway.

## Sub-commit Sequence

Same template as the tile-flavor refactor: four logical sub-commits + per-phase test fixes.

### Sub-commit 1 — planner emits typed tile flavors

- `000_partition_planner.py`:
  - Add a planner-private `_AxisRole` enum (or reuse `axis.Role` for now and delete it later — see SC2).
  - Rewrite `_wrap_tower(layers, body)` to construct tile flavors. `layers` is a list of `(axis,
    layer_kind)` tuples where `layer_kind` is `"BLOCK"` / `"THREAD"` / `"REGISTER"` / `"SERIAL_OUTER"` /
    `"STAGE_INNER"` / `"COOPERATIVE_STRIDE"` / `"SPLITK_BLOCK"` / `"PIPELINE"`.
  - Group consecutive `BLOCK` / `SPLITK_BLOCK` axes into one `GridTile` (with `splitk_axes` set from the
    SPLITK_BLOCK subset).
  - Group consecutive `THREAD` / `COOPERATIVE_STRIDE` axes into one `ThreadTile` (with `cooperative_axes`
    set from the COOPERATIVE_STRIDE subset).
  - Group consecutive `REGISTER` axes into one `RegisterTile`.
  - `SERIAL_OUTER` / `STAGE_INNER` / `PIPELINE` axes each become a `SerialTile(kind=…)`.
  - Wrap the final tower in `TileOp` (instead of `LoopOp`).
- Pattern stays `Pattern("root", LoopOp)` since the planner's input is still Loop IR.
- The planner's `is_matmul_reduce` check still walks Loop IR — keep that, since the input is still LoopOp.
- Variant enumeration unchanged: each combinatorial knob choice still produces one tower.

**Smoke check**: `deplodock compile --code "torch.matmul(...)" --ir tile` produces a `TileOp` directly from
the planner. `001_launch_geometry` runs next and (in its current still-converting form) should be a no-op for
the tile flavors it sees. Tests downstream of the planner break because launch_geometry double-handles
something, but `--ir tile` should look right.

### Sub-commit 2 — gut `001_launch_geometry` to coordination only

- Delete `_strip_outer_free_chain`, `_lift_output_loops`, `_convert_stmt`, `_convert_tile_to_flavors`,
  `_bind_for_role`, `_SERIAL_KIND_FOR_ROLE`.
- Rewrite `rewrite(root)`:
  - If `GridTile.splitk_axes` non-empty → run `_rewrite_for_atomic_lift` (now walks `SerialTile` /
    `StridedTile` / `RegisterTile` for atomic-eligible Writes).
  - If `ThreadTile.cooperative_axes` non-empty → run `_rewrite_for_cooperative_lift` (now walks tile
    flavors for reduce subtrees).
- Pattern changes to `Pattern("root", TileOp)` (input is now TileOp from the planner).
- Kernel naming stays here OR moves into the planner — pick one. Recommended: move into planner so 001 has
  no naming concern.

If kernel naming moves to planner: SC1 includes `_kernel_name_for` in the planner output.

**Smoke check**: `deplodock compile --code "torch.matmul(...)" --ir cuda` matches pre-refactor output for
representative kernels (matmul, softmax cooperative, RMSNorm, pointwise, SDPA chain).

### Sub-commit 3 — delete `Role` enum + `Loop.role` / `StridedLoop.role` field

- `ir/axis.py`: delete `Role` enum.
- `ir/stmt/blocks.py`: drop the `role: Role | None` field from `Loop` and `StridedLoop`. Drop `_source_suffix`
  references that touch it (the field's only consumer was launch_geometry).
- `ir/stmt/passes.py::rewrite`: drop the `role=s.role` kwarg from the Loop / StridedLoop rebuild.
- `ir/loop/normalize.py`: any pass that filtered by role (e.g. role-tagged Loops stop the canonical-axis-order
  sort) drops the role check.
- Find every `s.role`, `Role.X`, `from deplodock.compiler.ir.axis import Role`. Delete or rewrite.
- Tests: `test_role_field_default_is_none`, `test_with_bodies_preserves_role`,
  `test_role_excluded_from_structural_key` — delete (role is gone).

This is mechanical but touches ~10–15 files. Saved for SC3 so SC1+SC2 land cleanly first.

### Sub-commit 4 — drop the Loop / StridedLoop branch from `is_matmul_reduce`

- `_helpers.py::is_matmul_reduce`: the planner still sees Loop IR (its input is LoopOp), so the Loop /
  StridedLoop case in `is_matmul_reduce` stays — the function is called *inside* the planner's matmul
  detection. **No change needed.** Keep this sub-commit for any small follow-on cleanups discovered along
  the way; otherwise skip.

If nothing's left to clean up, drop SC4 and proceed straight to phase fixes.

## Test-Fix Phases

After the core replace, follow the same phase template as the tile-flavor refactor:

### Phase 0 — collection clean

`pytest --collect-only tests/compiler/` empty. Most likely cause of breakage: tests that import `Role` from
`ir.axis`. Bulk-rename / drop.

### Phase 1 — `tests/compiler/passes/test_partition_planner_rules.py`

The three role-roundtrip tests (`test_role_field_default_is_none`, `test_with_bodies_preserves_role`,
`test_role_excluded_from_structural_key`) get **deleted** — the field no longer exists. The three
`test_launch_geometry_*` tests already navigate tile flavors; verify they still pass without changes (they
should — the planner now produces what `_run_only_launch_geometry` returned before, modulo coordination).

### Phase 2 — `tests/compiler/passes/test_launch_geometry_rules.py`

Tests already inspect `ThreadTile` / `SerialTile` after the tile-flavor refactor. Verify they still pass:
the planner's output is now what launch_geometry's output used to be. Coordination tests (atomic Write, Combine
emission) still exercise 001's residual logic.

### Phase 3 — per-pass tests

`test_register_tile_rules.py`, `test_reduction_rules.py`, `test_matmul_rules.py`, etc. Mostly unaffected
(they test rule firing, which keys on stmt type — still the same flavors).

### Phase 4 — E2E accuracy

Full sweep. Should be transparent — the IR shape at the planner→stage-inputs boundary is identical to today's
launch_geometry output.

### Phase 5 — diagnostics / dump

`test_bank_conflicts.py` still navigates `binding.tile` (ThreadTile) and `binding.block_axes` (GridTile) —
unchanged.

### Phase 6 — lint, format, full `make test`

Final gate.

## Branching Protocol

Fork from `feature/partition-planner` (current branch). Merge back to `feature/partition-planner` with
`--no-ff` to preserve the sub-commit history.

```bash
git checkout feature/partition-planner
git checkout -b feature/planner-emits-tiles

# ... SC1 + SC2 + SC3 + phases ...

git checkout feature/partition-planner
git merge --no-ff feature/planner-emits-tiles
# autotune DB is already invalidated from the tile-flavor refactor — no extra cleanup
```

## Critical Files

- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_planner.py` — `_wrap_tower` rewrite +
  build typed tile tree.
- `deplodock/compiler/pipeline/passes/lowering/tile/001_launch_geometry.py` — gut to coordination only.
- `deplodock/compiler/ir/axis.py` — delete `Role` enum.
- `deplodock/compiler/ir/stmt/blocks.py` — drop `role` field from `Loop` / `StridedLoop`.
- `deplodock/compiler/ir/stmt/passes.py` — update `rewrite` for Loop / StridedLoop without `role`.
- `deplodock/compiler/ir/loop/normalize.py` — drop role-aware checks in canonicalization passes.
- `deplodock/compiler/graph.py` — drop `Role` from `_stmt_eval_scope`.

## Risks and Sharp Edges

1. **`canonicalize_free_axis_order` checks `Loop.role`** to terminate the chain-sort at role-tagged Loops. After
   the refactor, every role-tagged Loop becomes a typed tile flavor — and that pass operates on Loop IR
   (LoopOp body), which the planner consumes. So the planner sees vanilla `Loop` objects (no role) coming in.
   The chain-sort termination still works — it terminates at reduce Loops, which is the only condition that
   was non-role-based.

   **Verify**: trace through `canonicalize_free_axis_order` after dropping role; confirm it terminates
   correctly on the planner's input shapes.

2. **`Body.structural_key` excludes `role`** from the canonical pretty form today. After the field is deleted,
   structural_key is unaffected. But every existing autotune-DB key generated when `role` existed becomes
   stale — already invalidated by the tile-flavor refactor's `rm ~/.cache/deplodock/autotune.db`, no extra
   work.

3. **Direct planner→TileOp output skips one layer of normalization.** Today the planner emits LoopOp, which
   runs through `LoopOp.__post_init__` (calls `normalize_body`). After the refactor, the planner emits TileOp,
   which runs through `TileOp.__post_init__` (calls `normalize_body` with `hoist=False`). Verify the
   normalization passes work correctly on tile-flavor bodies — `rename_ssa_sequential` was already fixed in
   the tile-flavor refactor (recognizes ParallelTile / SerialTileBase axes).

4. **Coordination ordering.** Today coordination runs on still-Loop-IR (before tile conversion). After the
   refactor, coordination runs on tile IR. The `_compute_axis_dep_set` walker keys on `Accum` presence which
   is type-agnostic; `_rewrite_for_atomic_lift` descends `Loop`/`StridedLoop`/`Cond` which becomes
   `SerialTile`/`StridedTile`/`RegisterTile`/`Cond`. Mechanical rename in 001.

5. **`Pattern("root", LoopOp)` → `Pattern("root", TileOp)`** in `001_launch_geometry` means it now fires after
   the planner has already produced a TileOp. The pipeline order is unchanged (planner runs first), but the
   pattern matcher needs the input to be a TileOp not LoopOp. Verify the pipeline declaration in
   `pipeline/__init__.py` doesn't require LoopOp → LoopOp transitions before 001.

## Verification Plan

End-to-end:

1. `pytest tests/compiler/ -x` is green.
2. `pytest tests/ -x` is green (full repo).
3. `make lint` is clean.
4. `make test` is clean.
5. `deplodock compile --code "torch.matmul(torch.randn(256,256), torch.randn(256,256))" --ir tile` produces
   the same shape as before (modulo `Loop` having no `role` field in repr).
6. `deplodock compile --code "torch.softmax(torch.randn(64,512), dim=-1)" --ir cuda` produces
   byte-identical CUDA source as pre-refactor (with `~/.cache/deplodock/autotune.db` already cleared from the
   tile-flavor refactor).
7. Merge `feature/planner-emits-tiles` → `feature/partition-planner` with `--no-ff` to preserve sub-commit
   history.

## Scope Estimate

- SC1 (planner emits tiles): ~150–200 LOC change in `000_partition_planner.py`.
- SC2 (gut 001): delete ~250 LOC, keep ~150 LOC in `001_launch_geometry.py`.
- SC3 (delete Role): ~10 files touched, mostly single-line edits or import removals.
- Test fixes: ~5–10 commits, mostly deletions (role round-trip tests) and re-asserts.

Total: probably 400–500 LOC net delete (the refactor removes machinery), maybe 100 LOC added in the planner's
new `_wrap_tower`.
