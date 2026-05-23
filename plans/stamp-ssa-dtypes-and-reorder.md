# Stamp SSA Dtypes and Reorder Kernel-IR Passes

Supersedes [`defer-register-tile-unroll.md`](defer-register-tile-unroll.md). The previous plan moved 006a/007a
to the materialize boundary while keeping them `TileOp`-pattern passes. This plan goes further: it adds
first-class `dtype` fields to `Load` / `Assign` / `Write` / `Source`, lifts dtype resolution into a single
`001_stamp_types` pass, **consolidates every dtype-aware pass into `lowering/kernel/` before
`materialize_tile`**, and turns materialize into a purely mechanical lowering. The previous plan is kept
on disk as reference; if this one stalls, fall back to it.

## Context

Today dtype information is scattered across five lookup paths:

- `kernel/001_materialize_tile.rewrite` (line 128-134) builds `buf_cuda: dict[bid → cuda_type]` from
  `match.graph.nodes[bid].output.dtype` and threads it through `_emit_stage`.
- `kernel/002_demote_to_write_dtype` reads `kop.inputs` / `kop.outputs` / `kop.smem_buffers`.
- `kernel/003_vectorize_loads._buf_dtype` (line 71-83) walks the same union.
- `tile/007a_permute_register_tile.rewrite` (line 94-98) builds its own `buf_nbytes` from `match.graph.nodes`.
- `RenderCtx.ssa_dtypes` (`ir/stmt/base.py:88-92`) infers per-SSA dtypes at *render time*.

`tile/014_pad_smem` papers over the missing Stage dtype with `BYTES_PER_ELEM = 4`, which over-counts fp16 by
2× ([[project_tile_ir_fp32_only]]).

The architectural fix: **dtype as a first-class IR property, stamped once before any analytical pass runs**.
Add `dtype: DataType | None` to `Load`, `Assign`, `Write` (`value_dtype`), and `Source`. Add a single
`001_stamp_types` pass that walks the body once. Every downstream analytical pass — demote, register-tile
unroll, vectorize_loads, permute, pack_fp16, vectorize_stores — reads the IR. With dtypes explicit, those
passes no longer need Stage's structural side-channels and can all run as `TileOp`-pattern passes co-located
in `lowering/kernel/` before `materialize_tile`. Materialize becomes purely mechanical.

### Why `lowering/kernel/` (not a new folder, not split `lowering/tile/`)

The new passes pattern-match `TileOp` because they run before materialize. But conceptually they're
Kernel-IR-level concerns (dtype assignment, register-tile unrolling, vector ops, packing) — not Tile-IR
structural rewrites (Stage formation, async/TMA promotion, double-buffering). Folder placement should track
*conceptual concern*, not pattern type.

Options considered:

- **Split `tile/` into `tile/structural/` + `tile/scalar/`.** Cleanest split but introduces a new folder for
  unclear benefit; the boundary is fuzzy for borderline passes (014_pad_smem makes a structural decision
  driven by a scalar concern).
- **Introduce a new IR layer between Tile and Kernel.** Cleanest in principle (Scalar IR = TileOp body
  post-stamp, pre-materialize) but the benefits are unclear vs the infrastructure cost.
- **Put the new passes in `lowering/kernel/` before `materialize_tile`** (this plan). They still pattern-
  match `TileOp` (materialize hasn't run yet), but they live in `kernel/` because the *concern* is
  Kernel-IR-level. Materialize moves to the last position in `kernel/`. Lowest infra churn, conceptually
  clean.

The third option is what this plan does. If we later find ourselves wanting per-IR-level separation, we can
revisit by either splitting `kernel/` or introducing a new IR layer.

### Proposed pass order

`lowering/tile/` shrinks (006a and 007a moved out):

```
000_partition_planner            — existing
001_coordination                 — existing
002_stage_inputs                 — existing
007b_hoist_invariant_compute     — existing
010_double_buffer                — existing
011_tma_copy                     — existing
013_async_copy                   — existing
014_pad_smem                     — existing; reads source.dtype after Step 1 (fixes BYTES_PER_ELEM bug)
015_lower_pipelined_async_stage  — existing
016_mark_unroll                  — existing
```

`lowering/kernel/` grows and reorders:

```
000_place_inits                  — existing, unchanged
001_stamp_types                  — NEW
002_demote_to_write_dtype        — existing, now reads stamped dtypes
003_register_tile                — MOVED from tile/006a_register_tile_planned, renamed
004_vectorize_loads              — existing (renumbered from 003), now reads Load.dtype
005_permute_register_tile        — MOVED from tile/007a_permute_register_tile, renamed
006_pack_fp16_register_tile      — existing (renumbered from 004)
007_vectorize_stores             — existing (renumbered from 005)
008_materialize_tile             — MOVED from kernel/001 to the end; now purely mechanical
```

All passes 000-007 carry `PATTERN = [Pattern("root", TileOp)]`. Only 008_materialize_tile produces a
`KernelOp`. The folder name reflects conceptual concern, not pattern.

Justification for each new position:

| Pass | Why this slot |
|------|---------------|
| 000_place_inits | Existing; runs first (places Init at correct scope, transparent w.r.t. RegisterTile). |
| 001_stamp_types | First dtype-aware pass; runs once, output is fixed for all downstream. |
| 002_demote_to_write_dtype | Pre-unroll = FM·FN× fewer Assigns to walk; demote's linear-cost analysis benefits. |
| 003_register_tile | After demote so demoted dtypes propagate into register-cell replicas automatically. |
| 004_vectorize_loads | Needs unrolled consecutive scalar Loads; reads Load.dtype to pick vector_type. |
| 005_permute_register_tile | Operates on widened Loads (compact); reads vector width and dtype directly. |
| 006_pack_fp16_register_tile | After unroll (scalar Init/Accum pairs exist) + after demote (f16 stamping final). |
| 007_vectorize_stores | After all Write rewrites; final pass before materialize. |
| 008_materialize_tile | Last; purely mechanical Tile → Kernel lowering with no dtype work. |

### Design decisions

1. **`None` default on dtype fields.** Existing tests construct Load/Assign/Write/Source by hand without
   dtype; `None` keeps them compiling. `RenderCtx.ssa_dtypes` inference stays as the fallback. Once
   `001_stamp_types` is wired in, production-path bodies have non-`None` dtypes throughout; partial-pipeline
   tests still work via fallback. Long term the fallback can be deleted and `dtype: DataType` made
   non-optional — out of scope here.

2. **Stamping rules.**
   - `Source(name, buf, ...)` → `dtype = graph.nodes[buf].output.dtype`.
   - `Load(input=B)` → `dtype = buf_dtype(B)`. `B` is either a graph buffer id
     (`graph.nodes[B].output.dtype`) or a Stage source name (resolve via the enclosing Stage's
     `sources[i].dtype`, which step (a) just stamped).
   - `Assign(op, args)` → `dtype = promote(args' dtypes)`. Lift the promote logic from `RenderCtx.ssa_dtypes`
     inference into a public `dtype_promote(op_name, arg_dtypes) → DataType` helper.
   - `Write(output, value)` → `value_dtype = ssa_dtypes[value]`. The target buffer's dtype is resolved
     separately at render time.
   - `Accum` / `Init` / `Pack` / `Unpack` — already typed; re-stamp idempotently.
   - `Stage` body Loads referencing the Stage's own sources resolve to `source.dtype`; Loads referencing
     graph inputs (outside Stage scope) resolve to `graph.nodes[bid].output.dtype`.
   - `Combine` / `WarpShuffle` / `TreeHalve` — picked up by materialize from `pending_reduce[stmt.name].dtype`
     today; lift into stamp_types so it's set on the IR before materialize runs.

3. **Dtype propagation through rewrites.** Each pass that introduces or rewrites Stmts propagates dtype:
   - `003_register_tile`'s σ-replication of `Load(dtype=F16)` produces `Load(dtype=F16, name=…_0)` etc.
   - `004_vectorize_loads` merging N typed Loads at dtype `T` produces one widened Load at dtype `T`.
   - `005_permute_register_tile`'s index rewrite doesn't touch dtypes.
   - `002_demote_to_write_dtype` stamps the new (narrower) dtype on demoted Assigns (already does this).
   - `006_pack_fp16_register_tile` stamps F16x2 on packed Init/Accum (already does this).

4. **Materialize simplification.** After this plan:
   - Drop `buf_cuda` parameter from `_emit_stage` / `_emit_loop` / `_emit_compute_stage`.
   - Smem decls read `Smem(name=src.name, ..., dtype=src.dtype)` from the stamped Source.
   - Combine emission reads `Accum.dtype` directly (no more `pending_reduce` dtype lookup in materialize —
     stamp_types handled it).

5. **Permute's lane-axis cleanup, bundled.** Step-2b from the superseded plan (`tt.axes[-1]` + `knobs["FN"]`
   instead of `_infer_lane_stride` + extent-sort) applies here too. Folded into Step 8.

6. **`tile/014_pad_smem` fp16 fix.** Stamped `Source.dtype` enables 014 to read per-source byte count
   instead of `BYTES_PER_ELEM = 4`. Closes [[project_tile_ir_fp32_only]] as a side benefit. Folded into
   Step 11.

7. **Workflow — single feature branch, WIP-then-fix.** Per [[feedback_single_branch_milestones]]: one
   branch off `feature/partition-planner`. Don't try to make each step land green — the IR changes break
   too many tests for that to be practical. Phases:
   - **Phase 1**: Implement all code changes (Steps 1-11) as a single WIP push. Most tests will be red.
   - **Phase 2**: Iteratively fix tests, committing per fix-group. Each commit makes some red tests green
     without making any green tests red. Aim for ~10-20 commits in this phase.
   - **Phase 3**: `make test` + `make lint` both green.
   - **Phase 4**: Merge back to `feature/partition-planner`.

   No tune-kernels resweep, no perf-golden re-bless beyond what tests demand. Just tests. Perf validation
   is a separate follow-up if needed.

---

## Workflow

**Branch setup.**

```bash
git checkout feature/partition-planner
git pull --ff-only origin feature/partition-planner    # only if remote is current
git checkout -b feature/stamp-ssa-dtypes
```

**Phase 1 — implement all code (Steps 1-11).** Most tests will fail at the end of this phase. That's
expected. Don't try to fix tests as you go.

**Phase 2 — iterative test fixes.** Run `make test` (or focused subsets). For each red test:

- If it's a fixture issue (test constructs IR without dtype): update the fixture, or rely on
  `001_stamp_types` running in the test setup.
- If it's a golden mismatch (rendered CUDA differs): inspect the diff. If the new output is correct, rebless
  the golden. If it's a real regression, fix the code.
- If it's a logic break (test asserts on a side-channel that's gone): rewrite the test to query the IR.

Suggested commit groups:
- `stamp_types tests + dtype field plumbing`
- `demote: switch to stamped dtypes (rebless any goldens that shift)`
- `vectorize_loads: switch to stamped dtypes`
- `register_tile: move tests to new path`
- `permute_register_tile: move tests; fold in lane-axis cleanup`
- `pack_fp16 + vectorize_stores: pattern switch to TileOp`
- `materialize_tile: drop buf_cuda; update emit tests`
- `014_pad_smem: switch to source.dtype.nbytes`
- `top-level tests (end-to-end compile / run)`

**Phase 3 — green gate.** `make test` and `make lint` both clean. Spot-check a few `deplodock compile`
invocations to confirm the rendered CUDA still looks right.

**Phase 4 — merge back.**

```bash
git checkout feature/partition-planner
git merge --no-ff feature/stamp-ssa-dtypes
```

---

## Step 1 — Add `dtype` fields to Load / Assign / Write / Source

**Why.** Make dtype representable on the IR. `None` default keeps existing constructors compatible.

**Change.**

- `deplodock/compiler/ir/stmt/ir.py`: add `dtype: DataType | None = None` to `Load`, `Assign`; add
  `value_dtype: DataType | None = None` to `Write`.
- `deplodock/compiler/ir/tile/ir.py`: add `dtype: DataType | None = None` to `Source`.
- `deplodock/compiler/ir/stmt/passes.py`: update `rewrite` / `simplify` dispatches for the three classes to
  carry dtype through Sigma/rename. `dtype=s.dtype` in the constructor calls.
- `deplodock/compiler/ir/stmt/blocks.py`: add `_resolve_dtype(stmt, ctx)` helper. Prefer `stmt.dtype` if set,
  fall back to `ctx.ssa_dtypes` inference (existing behavior). Patch the three render call sites.

**Files.**

- `deplodock/compiler/ir/stmt/ir.py`
- `deplodock/compiler/ir/tile/ir.py`
- `deplodock/compiler/ir/stmt/passes.py`
- `deplodock/compiler/ir/stmt/blocks.py`

## Step 2 — Lift `dtype_promote` out of the renderer

**Why.** `001_stamp_types` needs the promote logic that `RenderCtx.ssa_dtypes` inference currently uses inline.

**Change.**

- Find the promote-and-infer block in `RenderCtx.ssa_dtypes` population (wherever the body walk lives).
  Extract as `dtype_promote(op_name: str, arg_dtypes: list[DataType]) → DataType`.
- Call sites: stamp_types (new); RenderCtx.ssa_dtypes inference (existing fallback path).

**Files.**

- `deplodock/compiler/ir/stmt/base.py` (or `ir/expr.py` — whichever currently holds the inference logic).

## Step 3 — Add `001_stamp_types` pass

**Why.** Single pass that walks the body once and populates every Load.dtype / Assign.dtype /
Write.value_dtype / Source.dtype.

**Change.**

- New file: `deplodock/compiler/pipeline/passes/lowering/kernel/001_stamp_types.py`.
  `PATTERN = [Pattern("root", TileOp)]`.
- Walk the body with a running `ssa_dtypes: dict[str, DataType]` accumulator:
  - `Stage`: stamp each Source.dtype from `match.graph.nodes[source.buf].output.dtype`. Descend into
    Stage.body; Loads inside resolve against the enclosing Stage's sources.
  - `Load(input=B)`: if `B` matches a Stage source name in scope, dtype = that source.dtype; else dtype =
    `graph.nodes[B].output.dtype`. Stamp; register in `ssa_dtypes[s.name]`.
  - `Assign(op, args)`: dtype = `dtype_promote(op.name, [ssa_dtypes[a] for a in args])`. Stamp; register.
  - `Write(output, value)`: `value_dtype = ssa_dtypes[value]`. Stamp.
  - `Init` / `Accum` / `Pack` / `Unpack` — already typed; register in `ssa_dtypes`.
  - `Loop` / `StridedLoop` / `Cond` / `SerialTile` / `StridedTile` / `RegisterTile` / `ThreadTile` /
    `GridTile`: descend into nested bodies.
- Idempotent: if every Stmt is already stamped, `RuleSkipped`.

**Files.**

- New `deplodock/compiler/pipeline/passes/lowering/kernel/001_stamp_types.py` (~120 lines).
- Update `kernel/__init__.py` (if explicit registration).

## Step 4 — Switch `demote_to_write_dtype` to read stamped dtypes

**Why.** With Source.dtype + Load.dtype stamped, demote can read dtypes from IR instead of `kop.inputs` /
`kop.outputs`. Demote also moves from `KernelOp` pattern to `TileOp` pattern (since it now runs before
materialize).

**Change.**

- Change `PATTERN` from `KernelOp` to `TileOp` in `kernel/002_demote_to_write_dtype.py`.
- Replace `kop.inputs.get(name)` / `kop.outputs.get(name)` / `kop.smem_buffers.get(name)` with reads off the
  stamped IR:
  - Output buffer dtype for a Write: look up via the enclosing TileOp's `inputs` / `outputs` Tensor maps
    (TileOp has these — same shape as KernelOp).
  - Per-SSA dtype for the f16 carrier seed: `Load.dtype.name == "f16"`.
  - `_seed_fp16_carriers` simplifies to "Loads whose stamped dtype is f16".
- `_TARGET.has_native_op` usage stays.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/002_demote_to_write_dtype.py` (~30 lines touched).

## Step 5 — Move register_tile_planned into kernel/

**Why.** `tile/006a_register_tile_planned.py` → `kernel/003_register_tile.py`. The pass operates on TileOp
body (still pre-materialize) but its concern (replicating per register cell, SSA-name suffixing) is a
Kernel-IR-level scalar lowering.

**Change.**

- `git mv tile/006a_register_tile_planned.py kernel/003_register_tile.py`.
- No code change to the rewrite logic — pattern stays `TileOp`.
- Update both directories' `__init__.py`s.

**Files.**

- File move.
- Two `__init__.py` updates.

## Step 6 — Switch `vectorize_loads` to read stamped dtypes; renumber

**Why.** With Load.dtype stamped, vectorize reads source dtype off the Load directly. Pattern stays `TileOp`
(no change there — wait, it was KernelOp; now it becomes TileOp since it runs before materialize).

**Change.**

- Change `PATTERN` from `KernelOp` to `TileOp` in `kernel/003_vectorize_loads.py`. After Step 5's renumber
  this file lives at `kernel/004_vectorize_loads.py`.
- Delete `_buf_dtype` helper.
- `_try_vec_load`: `src_dt = loads[0].dtype.name` (assumes stamped). Fall back to `RuleSkipped` if any Load
  is unstamped (defensive).
- Tile-IR Loads can reference Stage source names (smem-local); vectorize doesn't need to distinguish.
  Consecutive Loads against the same `input` name with anchor+1, anchor+2 indices vectorize the same way.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/004_vectorize_loads.py` (renamed from `003_`, ~15
  lines touched).

## Step 7 — Move permute_register_tile into kernel/; fold in lane-axis cleanup

**Why.** `tile/007a_permute_register_tile.py` → `kernel/005_permute_register_tile.py`. Also fold in the
Step-2b cleanup from the superseded plan.

**Change.**

- `git mv tile/007a_permute_register_tile.py kernel/005_permute_register_tile.py`.
- Inside the file:
  - Replace `_infer_lane_stride` + candidates loop with `lane = tt.axes[-1]; F = int(root.op.knobs["FN"])`.
  - Replace `_vec_elems_for_lane`'s `buf_nbytes` lookup with a walk that reads `load.dtype.nbytes` from
    affected Loads. Drop the `buf_nbytes` parameter entirely.
  - Delete `_infer_lane_stride` (~30 lines).
  - `_chunk_expr` handles widened-Load anchors as well as scalar — no change.

**Files.**

- File move + ~40 lines deleted, ~10 added.

## Step 8 — Renumber `pack_fp16` and `vectorize_stores`; switch PATTERN to TileOp

**Why.** They run before materialize now, so they pattern-match TileOp. Renumber to fit the new sequence.

**Change.**

- `kernel/004_pack_fp16_register_tile.py` → `kernel/006_pack_fp16_register_tile.py`. Change `PATTERN` to
  `TileOp`. No other body changes — pack_fp16's walk already handles all the tile-flavor wrappers.
- `kernel/005_vectorize_stores.py` → `kernel/007_vectorize_stores.py`. Change `PATTERN` to `TileOp`. Audit
  for any KernelOp-specific lookups (probably reads target-buffer dtype like demote does); switch to the
  TileOp-side lookup.

**Files.**

- Two renames + ~10 lines edited per file.

## Step 9 — Simplify and move materialize_tile to the end

**Why.** With Source.dtype stamped and Combine dtypes set by 001_stamp_types, materialize stops doing dtype
work. Move to last position so kernel/ pass order reads top-to-bottom.

**Change.**

- `kernel/001_materialize_tile.py` → `kernel/008_materialize_tile.py`.
- Drop the `buf_cuda: dict[str, str]` construction in `rewrite` (lines 128-134).
- Drop the `buf_cuda` parameter from `_emit_stage` / `_emit_loop` / `_emit_compute_stage` and the closures
  inside `_materialize`.
- `Smem(name=src.name, extents=..., dtype=src.dtype)` reads from the stamped Source. Drop the
  `cuda_name(node.output.dtype)` conversion at the materialize boundary — that's now a render-time concern.
- The `pending_reduce` dtype-lookup at line 474 (`accum_dtype = accum.dtype or F32`) stays — Accum carries
  dtype after stamp_types.

**Files.**

- `kernel/008_materialize_tile.py` (renamed from `001_`, ~50 lines simplified).
- `kernel/__init__.py`.

## Step 10 — Drop now-dead Tile-IR helpers

**Why.** `_helpers.py` in `tile/` has functions used only by 006a/007a (`loads_reading`, `single_tile`,
`thread_tile_of`, `replace_thread_tile_body`). Audit and delete any now-unused helpers.

**Change.**

- Read `lowering/tile/_helpers.py` after Steps 5 and 7 land. Cross-check uses against the moved files'
  imports. Delete any function that has no remaining caller.
- Audit `permute_register_tile`'s `_vec_elems_for_lane` for further simplification post-Step-7 dtype
  switch.

**Files.**

- `lowering/tile/_helpers.py` (~30 lines deleted, estimated).

## Step 11 — Fix `014_pad_smem`'s `BYTES_PER_ELEM = 4`

**Why.** With `Source.dtype` stamped, 014 reads the per-source byte count instead of guessing 4. Closes
[[project_tile_ir_fp32_only]] latent bug. Side benefit of Source.dtype existing.

**Change.**

- `tile/014_pad_smem.py`: find every reference to `BYTES_PER_ELEM` or `4`-hardcoded slab math. Replace with
  `source.dtype.nbytes` reads.
- Audit `tile/010_double_buffer.py` for similar assumptions; touch if needed.

**Files.**

- `tile/014_pad_smem.py` (~20 lines touched).
- Possibly `tile/010_double_buffer.py`.

## Step 12 — Update ARCHITECTURE.md across affected directories

**Change.**

- `lowering/tile/ARCHITECTURE.md`: remove 006a / 007a entries; note register-tile passes moved to
  `kernel/`. Mention 014_pad_smem now reads stamped Source dtypes.
- `lowering/kernel/ARCHITECTURE.md`: rewrite for the new 9-pass sequence. Document the "stamp → demote →
  unroll → vectorize → permute → pack → vectorize_stores → materialize" ordering and rationale. Note that
  passes 000-007 are `TileOp`-patterned even though they live in `kernel/`.
- `compiler/ir/stmt/ARCHITECTURE.md` (or equivalent): document Load/Assign/Write/Source carrying optional
  `dtype`, populated by `001_stamp_types`.
- `compiler/ARCHITECTURE.md` (top-level): bump the pass-order summary.

**Files.**

- Four `ARCHITECTURE.md` updates.

---

## Out of scope (filed for follow-up plans)

- **Delete the render-time inference fallback.** Once `001_stamp_types` is mandatory in every production
  pipeline, `RenderCtx.ssa_dtypes` inference can go. Hold until enough downstream call sites are confirmed
  to run stamp_types first.
- **Make `dtype` non-optional on Load / Assign / Write / Source.** Requires the inference fallback to be
  gone *and* every IR-construction site (tests, tracer, decomposition) to pass dtype explicitly. Substantial
  test-fixture churn.
- **Split materialize into smaller passes.** Today materialize still handles Stage → Smem + cooperative
  load + mbar/TMA prologue + Combine → TreeHalve. Splitting into mechanical sub-passes is a follow-up that
  this plan doesn't touch — but is much easier *after* this plan because materialize no longer makes
  analytical decisions.
- **Perf validation (tune-kernels resweep, perf golden re-bless).** Deferred. This plan is correctness-only;
  if a perf regression slips through, it surfaces in the next tune sweep.
- **New IR layer between Tile and Kernel.** Could give the pre-materialize scalar-lowering passes their own
  pattern type instead of `TileOp`. Benefits unclear; revisit if `kernel/` becomes unwieldy.
