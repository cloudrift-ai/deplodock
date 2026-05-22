# Finish Stage Wrap-Body Refactor — Restore Promotion Passes

## Context

The stage-wrap-body refactor changed `Stage` from a sibling stmt declaring smem + a cooperative-load body into
a block-structured Stmt that *wraps* its consumer body. The IR-level work has landed:

- `tile/ir.py` defines `Stage` / `BufferedStage` / `AsyncBufferedStage` / `TmaBufferedStage` /
  `ComputeStage` in the wrap-body shape with `sources: tuple[Source, ...]` and per-source `cache_dims`.
- `002_stage_inputs.py:358` produces wrap-body sync `Stage` already.
- `001_materialize_tile.py:803-898` (`_emit_stage`) handles each subclass — sync cooperative `Load+Write`,
  `BufferedStage` with phase-prepended smem index, `AsyncBufferedStage` via `CpAsyncCopy`, and the higher-
  level `emit_tma_stage` closure (`001_materialize_tile.py:284-407`) emits the
  `MbarrierArriveExpectTx + tma_load + MbarrierWait` triplet for `TmaBufferedStage`.
- `AsyncWait` is documented as a deprecated stub (`ARCHITECTURE.md:241`); its semantics fold into
  `pipeline_depth` on async-bearing Stage subclasses.

**Seven** tile-lowering passes are currently stubbed (always `RuleSkipped`):

| Pass | Bucket | Pre-refactor job | Post-refactor job |
|---|---|---|---|
| `007b_hoist_invariant_compute.py` | 7 | hoist invariant compute cone; emit multi-source Stage (inline-fuse) or sibling ComputeStage | same shapes, but Stage is wrap-body and ComputeStage carries a separate `compute` body |
| `010_double_buffer.py` | 11 | rewrite `Stage` → `BufferedStage(buffer_count=2, phase=K_o%2)` | subclass swap; stamp `buffer_count`/`phase` |
| `013_async_copy.py` | 11 | promote `BufferedStage` → `AsyncBufferedStage`, append `AsyncWait(keep=0)` | subclass swap; no `AsyncWait` (implicit) |
| `014_pad_smem.py` | 11 | per-stage bank-conflict pad via `Stage.pad` | same eligibility, applied to wrap-body `Source.pad` (already on Source per `ir.py:243`) |
| `011_tma_copy.py` | 12 | single-source `BufferedStage` → `TmaBufferedStage`, trailing `AsyncWait(phase=…)` | subclass swap; mbarrier wait emitted by materializer **(materializer needs rewrite — see M5)** |
| `012_split_inner_for_swizzle.py` | 12 | split inner cache dim on `TmaBufferedStage` for swizzle | same logic over `Source.cache_dims` |
| `015_pipeline_k_outer.py` | 10 | sigma-expand prologue/main/epilogue with explicit `AsyncWait` siblings | consume `AsyncBufferedStage(pipeline_depth>1)`, emit expanded subtree (waits implicit) — rename `015_lower_pipelined_async_stage.py` |

In addition, the **materializer has un-rewritten old-shape code paths** (because TmaBufferedStage never reaches
it today — 011 is stubbed): `emit_tma_stage` (lines 284-431, ~120 LOC reading `stage.name` / `.addressing` /
`.axes` / `.origin` / `.buf` / `.alloc_extents` / `.buffer_count` / `.phase`) and `_partition_tma_groups`
(line 607, reads `stage.name` / `.buffer_count`). These crash on the new shape and must be rewritten before
M5 / M6 can land. Counted as part of M5.

Net behavior today: every kernel that previously got hoist / double-buffer / cp.async / TMA / pipelining
promotion comes out as plain sync `Stage` with leading + trailing `__syncthreads`. Performance is regressed
on those kernels — the regression is the cost being paid for the structural cleanup, and this plan repays it.

### Scope

- **In scope.** Restore all seven stubbed passes (007b hoist + 010 / 011 / 012 / 013 / 014 / 015)
  against the wrap-body shape. Rewrite the materializer's old-shape TMA path (`emit_tma_stage` +
  `_partition_tma_groups`) for per-Source iteration. Delete Tile-IR `AsyncWait` Stmt + its
  materializer dispatch. Add unit tests per restored pass. Add an end-to-end perf-restoration
  snapshot test. Update each pass's docstring and `ARCHITECTURE.md`.
- **Out of scope.** Any IR shape change beyond what's already landed (other than possibly a
  `compute_chain` field on Source if the M5b spike picks that direction). Any new optimization that
  didn't exist before the refactor. Source-axis annotation in CacheDim
  (`plans/stage-source-axis.md` — Axis.source_axis is in place; semantic queries on CacheDim are
  the follow-up). MMA-fragment factorization (`plans/mma-fragment-factorization.md`).

### Risk note up front

The single load-bearing concern is **byte-identical CUDA output** vs the pre-refactor baseline. The
classes / materializer / `002_stage_inputs.py` paths have changed; the *promoted* IR shapes are different
even when the emitted CUDA is meant to be equivalent. M8's snapshot test is the durable gate: every
restored kernel's CUDA source must match the pre-refactor recorded baseline (or, where the refactor
intentionally improves emission e.g. dropping a redundant `AsyncWait`, the diff must be limited to that
known-safe class).

A secondary risk is **dependency ordering across the promotion chain**. 013 expects to see `BufferedStage`
(010's output); 011 expects `BufferedStage` or `AsyncBufferedStage`; 014 needs to see slab extents in the
buffered form; 015 needs `pipeline_depth>1` to be a settable knob on the async/TMA forms. Land in the
dependency order so each milestone has a working predecessor to consume.

## Design decisions

1. **Single feature branch, milestone commits after `make test`.** Per user preference for multi-step
   refactors (memory: `feedback_single_branch_milestones.md`). Branch
   `feature/finish-stage-wrap-body-refactor` or similar; each milestone's commit message references the
   bucket and the pass.

2. **Dependency order: 010 → 013 → 014 → 011 (+ materializer TMA rewrite) → 012 → 007b → 015 →
   AsyncWait deletion.** 014 sits after 013 since bank-pad cares about whether transport is sync or async
   (the cp.async path has 16-byte alignment constraints that affect pad choice). 011 includes rewriting
   `emit_tma_stage` + `_partition_tma_groups` in the materializer (M5 has materializer churn, not zero).
   007b can land in parallel with 010–012 (it's pre-staging), but is placed here so its consumer side
   (multi-source Stage / sibling ComputeStage) gets exercised by the promotion chain. 015 last because
   it consumes `AsyncBufferedStage` / `TmaBufferedStage` with `pipeline_depth>1`. AsyncWait emission
   deletion is the final cleanup once nothing emits it.

3. **Each promotion pass is a structural rewrite, not a body rewrite.** Subclass swap + arg stamping, no
   σ-substitution, no body restructure. The materializer carries the per-subclass emission logic; the
   tile-level pass just decides "this Stage is eligible for promotion to subclass X."

4. **Eligibility predicates port over verbatim from pre-refactor.** What changes is where they read the
   inputs (`stage.sources[i].cache_dims` instead of `stage.axes + addressing.dims`; `stage.body` is the
   consumer scope, not the producer Load+Write). The decisions themselves don't change.

5. **`AsyncWait` Stmt stays as a deprecated stub until M7 lands.** It's still imported by passes that
   transitively pull it in via `tile.ir`'s `__all__`. The deletion in M8 removes the class entirely along
   with the deprecated import. This avoids a churning import-list across every intermediate milestone.

6. **`015_pipeline_k_outer.py` is renamed `015_lower_pipelined_async_stage.py` as part of M7.** Reflects
   the new contract — it lowers a `pipeline_depth>1` annotation, not an arbitrary K-outer loop. The
   rename happens in the same commit as the rewrite so the file's history reads cleanly.

7. **Unit tests per pass; integration test at M8.** Each promotion-restoration milestone (M2–M7) ships
   with a focused unit test asserting (a) eligibility regression vs pre-refactor cases and (b) the new
   wrap-body output shape. M8 adds an end-to-end snapshot test across the kernel corpus.

8. **Perf restoration validated post-M8 via `make bench-kernels-tuned`.** The autotune DB picks the same
   variants as the pre-refactor baseline on the canonical workload (TinyLlama, Qwen 7B layers, SDPA,
   RMSNorm, softmax). Any kernel where the picked variant differs gets explicit attention — either the
   restoration has a bug, or the new shape genuinely changed which variant the priority key surfaces.

9. **Pre-refactor CUDA snapshots are the baseline.** Take from a `git checkout c63e842e` — the last
   source-changing commit on `feature/partition-planner` before the stage-wrap-body refactor began
   (`stage_inputs: move pre-006a, add REGISTER axes to cache roles`). The Phase A commit `0228ce91`'s
   parent is `3e98af60` (plan doc only), whose parent is `c63e842e` — confirmed via
   `git log --oneline 0228ce91^^`. Commit snapshots to `tests/perf/snapshots/wrap_body_refactor/` for M8;
   delete after merge.

## M1 — Audit + snapshot baseline + branch setup

**Why.** Establish the comparison baseline before any restoration work. Without snapshots, M8 can't gate
on byte-identical output and the refactor regresses silently into a "looks like it works" state.

**Change.**

- Check out `c63e842e` (the pre-refactor tip on `feature/partition-planner` — see Design Decision 9).
  Use a temporary worktree: `git worktree add /tmp/wrap-body-snapshot c63e842e`.
- Generate CUDA dumps for the kernel corpus via `DEPLODOCK_DUMP_DIR`:
  - TinyLlama layer (`deplodock compile TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
  - Qwen 7B layer (`deplodock compile Qwen/Qwen2.5-7B`)
  - Synthetic matmul (M=N=K=4096 fp32, fp16)
  - Synthetic SDPA (head_dim=64, seq_q=512)
  - Synthetic RMSNorm, softmax
- Copy `08_lowering_cuda.cu` outputs to `tests/perf/snapshots/wrap_body_refactor/`.
- Return to the working branch.
- Confirm the corpus reproduces post-refactor with the same kernels (just regressed perf), so the
  snapshot comparison surface is well-defined.

**Files.**

- `tests/perf/snapshots/wrap_body_refactor/*.cu` (new directory)
- No source changes.

**Verification.** Snapshots committed. `git status` clean on source.

## M2 — Restore `010_double_buffer.py` (Bucket 11)

**Why.** Foundation pass — every async / TMA / pipelined kernel depends on a `BufferedStage` having been
emitted earlier in the chain. Without M2, M3–M7 have nothing to consume.

**Change.**

- In `010_double_buffer.py`, replace the stub `rewrite()` with the post-refactor implementation. Walk the
  TileOp body for a `Loop(SERIAL_OUTER)` whose body contains a wrap-body `Stage` (not `BufferedStage` or
  any subclass) where the consumer body has a `STAGE_INNER` reduce. Eligibility predicate (~30 lines)
  ports pre-refactor logic: smem fits 2× allocation, no inter-iter SSA reads in the consumer body that
  conflict with slot rotation, no `ComputeStage` inside the wrap (these have their own buffering knob).
- For each eligible Stage, return `BufferedStage(sources=stage.sources, body=stage.body,
  buffer_count=2, phase=Var(K_o.name) % Literal(2))`. Stamp the new fields on the existing wrap-body
  shape — sources, cache_dims, and body all pass through unchanged.
- **Verify `_assert_stage_body_shape` doesn't reject the new shape.** This helper in
  `001_materialize_tile.py:786` was written for the old shape (asserts `stage.body` is a single
  `Load + Write` producer chain). In the new shape `stage.body` is the consumer subtree. Either delete
  the helper (its preconditions don't apply) or rewrite the assertion to reflect wrap-body invariants
  (one cooperative-load StridedLoop per Source synthesized at materialize time). Do this in M2 since
  M2 is the first milestone whose output a downstream pass actually consumes.
- Update the module docstring: drop "stubbed during stage-wrap-body refactor" prefix; describe the
  rewrite as "promote wrap-body Stage to BufferedStage with double-buffered ring."

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/010_double_buffer.py` (~80 lines net; replaces
  the 26-line stub)

**Verification.**

- Unit test `tests/compiler/passes/test_double_buffer_wrap_body.py`: input a TileOp with a wrap-body
  `Stage` inside a `Loop(SERIAL_OUTER)`; assert post-rewrite the Stage is a `BufferedStage` with
  `buffer_count=2`, `phase=K_o%2`, identical sources/cache_dims/body.
- Eligibility regression: cases that pre-refactor 010 rejected (smem-budget overflow, ComputeStage in
  body, no SERIAL_OUTER scope) still produce `RuleSkipped`.
- `make test` clean.

## M3 — Restore `013_async_copy.py` (Bucket 11)

**Why.** Async transport is the prerequisite for both TMA (M5) and pipelining (M7). cp.async on its own
provides ~10–30% gain over double-buffered sync on transformer-prefill GEMMs.

**Change.**

- In `013_async_copy.py`, replace the stub. Walk the TileOp body for `BufferedStage` (output of M2)
  inside a `Loop(SERIAL_OUTER)`. Eligibility (~25 lines): `ctx.arch >= 80`; every Source's inner cache
  dim is contiguous and aligned for `cp.async.ca` size (4 / 8 / 16 B); no fused producer body (cp.async
  doesn't fuse with elementwise compute).
- Promote eligible `BufferedStage` to `AsyncBufferedStage(pipeline_depth=1, ...)`. The
  `pipeline_depth=1` is sync-style wait; M7 bumps it when pipelining fires. All other fields pass
  through.
- Materializer already handles `AsyncBufferedStage` in `_emit_stage` (post-bucket-11 rewrite, line 803+).
  The cp.async branch at line ~870 emits `CpAsyncCopy` per `Source`; the trailing `CpAsyncCommit` is
  emitted at the end of `_emit_stage`. The implicit wait at the wrap boundary is `CpAsyncWait(0) + Sync`
  emitted by the consumer-side flow. No materializer change in M3.
- Update module docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/013_async_copy.py` (~70 lines)

**Verification.**

- Unit test `tests/compiler/passes/test_async_copy_wrap_body.py`: input a TileOp with `BufferedStage`
  inside `SERIAL_OUTER`; assert post-rewrite is `AsyncBufferedStage` with `pipeline_depth=1`. Confirm
  materializer emits expected `CpAsyncCopy + CpAsyncCommit + CpAsyncWait(0) + Sync` sequence in the
  rendered CUDA.
- Eligibility regression: pre-refactor reject cases (`arch < 80`, misaligned cache dim, fp16 sub-byte
  size) still `RuleSkipped`.
- `make test` clean.

## M4 — Restore `014_pad_smem.py` (Bucket 11)

**Why.** Bank-conflict padding directly affects smem read bandwidth in the consumer body. Without it,
power-of-2 cache dims trigger 32-way bank conflicts on `LDS.128` reads — observed >25 M conflicts on
TinyLlama matmuls per `007a_permute_register_tile.py:25`.

**Change.**

- In `014_pad_smem.py`, replace the stub. Walk the TileOp body for `BufferedStage` /
  `AsyncBufferedStage` (NOT `TmaBufferedStage` — TMA has its own swizzle path; pad incompatible — the
  `TmaBufferedStage.__post_init__` at `tile/ir.py:472-476` asserts every source's pad is empty).
  For each eligible stage, walk its `sources` and decide per-source pad: detect stride-aliased layouts
  in the cache extents, pad the second-innermost cache dim by `+1` to break 32-way aliasing.
  Pre-refactor logic, ported to read from `Source.cache_dims` instead of `Stage.axes`.
- `pad` already lives on `Source` (`tile/ir.py:243`) — no field migration needed. Set per-source via
  `Source.with_pad(pad)` helper (`tile/ir.py:272`).
- Update module docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/014_pad_smem.py` (~90 lines)

**Verification.**

- Unit test `tests/compiler/passes/test_pad_smem_wrap_body.py`: input an `AsyncBufferedStage` with
  power-of-2 cache dims that pre-refactor would trigger padding; assert post-rewrite the appropriate
  Source carries `pad=(0, 1)` (or whatever the canonical fix is).
- TMA stages: assert pad NOT applied (already-existing `tile/ir.py:525` assertion catches this if
  violated).
- Bank-conflict snapshot for one matmul: count `ld.shared` conflicts in `compiler/diagnostics/
  bank_conflicts.py` output before vs after pad; assert reduction.

## M5 — Restore `011_tma_copy.py` + materializer TMA path (Bucket 12)

**Why.** TMA provides ~2× bandwidth on sm_90+ vs cp.async for large slabs by issuing one box-copy
instruction from one elected thread per source. Without M5, sm_90 kernels stay on cp.async.

**M5 is the biggest single milestone** because it includes both a pass-side rewrite AND a materializer
rewrite. Bucket 11's `_emit_stage` rewrite handled Stage / BufferedStage / AsyncBufferedStage, but the
TMA path (`emit_tma_stage` at `001_materialize_tile.py:284-431` and `_partition_tma_groups` at
`001_materialize_tile.py:607`) was left in its pre-refactor form because no TmaBufferedStage exists
in the IR pipeline today (011 is stubbed). Re-enabling 011 requires fixing the materializer first.

**Change.**

- **Materializer rewrite (do first):** `emit_tma_stage` reads 27 old-shape attributes (`stage.name`,
  `stage.addressing`, `stage.axes`, `stage.origin`, `stage.buf`, `stage.alloc_extents`,
  `stage.buffer_count`, `stage.phase`). Rewrite to iterate `stage.sources` exactly as `_emit_stage`
  does post-bucket-11. For TMA's single-source eligibility, the loop iterates once; the per-source
  `Source.buf` becomes the TMA descriptor's `src_buf`, `Source.origin` becomes the box-copy `coords`,
  and `Source.cache_dims` drives the box-extents collapse. The split-tail path for swizzle (lines
  304-368) reads `Source.cache_dims[-2:]` instead of `stage.axes[-2:]`. `_partition_tma_groups` reads
  the per-source smem name (`source.name`) for issuer-tid allocation; today's per-stage `stage.name`
  becomes a per-source iteration.
- **Pass rewrite:** in `011_tma_copy.py`, replace the stub. Walk for `AsyncBufferedStage` (output of M3)
  with exactly one Source whose addressing is `AffineAddressing` and whose inner source dim is
  contiguous + 16 B aligned. Eligibility (~30 lines): `ctx.arch >= 90`, single source, AffineAddressing
  only, inner alignment.
- Promote eligible `AsyncBufferedStage` to `TmaBufferedStage(pipeline_depth=1, swizzle=NONE, ...)`.
  All Sources/cache_dims/body pass through.
- Multi-source `AsyncBufferedStage` (the matmul A+B case) stays on cp.async — TMA's elected-thread
  model with two sources requires either two separate `TmaBufferedStage` instances (each with one
  source) or a fused-arrive-tx primitive. Defer; flag in failure-modes.
- Update module docstring on 011 and `emit_tma_stage`'s docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/011_tma_copy.py` (~80 lines)
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` (~150 lines:
  `emit_tma_stage` + `_partition_tma_groups` rewritten for per-Source)

**Verification.**

- Unit test `tests/compiler/passes/test_tma_copy_wrap_body.py`: input a single-source
  `AsyncBufferedStage` on sm_90; assert post-rewrite is `TmaBufferedStage` with correct
  swizzle/phase. Confirm materializer emits `MbarrierArriveExpectTx + tma_load + MbarrierWait` in
  the rendered CUDA (use `DEPLODOCK_TARGET=sm_90` for the compile).
- Eligibility regression: multi-source stages stay `AsyncBufferedStage`; misaligned cache dim
  rejected; `arch < 90` rejected.
- Add materializer unit test: construct a `TmaBufferedStage` directly (single Source), call
  `emit_tma_stage` via the materialize harness, assert the rendered subtree has exactly one TMA box
  copy per source with the right descriptor name and coords.

## M5b — Restore `007b_hoist_invariant_compute.py` (Bucket 7)

**Why.** Hoists invariant compute cones out of K-inner reduce bodies — the fused-MLP optimization that
turns silu·gate·matmul from three sequential reduces into one combined producer. Without it,
fused-MLP kernels stay on the un-hoisted shape, losing ~15–25% on the affected reduces. The
`FUSED_PIPELINE` knob (False = inline-fuse, True = sibling ComputeStage) is an autotune fork.

Can land in parallel with M2–M4 (doesn't depend on Buffered/Async promotion), but placed here so the
multi-source Stage / ComputeStage outputs are exercised by the M2–M5 promotion chain immediately
after.

**Change.**

- In `007b_hoist_invariant_compute.py`, replace the stub. Walk for groups of same-cache-axes Stages
  (in the new shape: same `Source.cache_dims` tuple structure across two or more Sources of a Stage,
  or across two sibling Stages whose Sources match). The pre-refactor cone-walk logic ports over but
  the input shape changed: walk descends into `Stage.body` to find the K_inner reduce loop;
  cone-candidate detection groups by per-Source `cache_dims` instead of per-Stage `axes`.
- **Inline-fuse output (`FUSED_PIPELINE=False`):** emit a new wrap-body Stage with multiple Sources
  covering all the fused gmem operands; the cone Assigns move into a producer-side mode (TBD — the
  old shape ran them inline in the cooperative-load body; the new shape may need a `compute_chain`
  field on Source, or a `ComputeStage` variant). **Open design decision** — write a 50-line spike
  before committing to either shape.
- **Hoist-compute output (`FUSED_PIPELINE=True`):** insert a sibling `ComputeStage` wrapping a
  reduced consumer body, with `compute` field holding the cone Assigns. The producing transport
  Stages stay as separate wrap-body Stages — `010` / `011` / `013` can still promote them in M2–M5.
- Update module docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/007b_hoist_invariant_compute.py` (~300 lines —
  this is the largest non-materializer rewrite in the plan)
- Possibly `deplodock/compiler/ir/tile/ir.py` (~20 lines) if the inline-fuse path needs a
  per-Source `compute_chain` field; defer that decision to the spike.

**Verification.**

- Unit test `tests/compiler/passes/test_hoist_invariant_compute_wrap_body.py`: input a TileOp with a
  silu·gate·matmul shape (two same-cache-axes producer Stages + one outer matmul Stage + reduce
  body); assert both `FUSED_PIPELINE` polarities emit correctly. Inline-fuse: single Stage with
  multi-Source; hoist-compute: ComputeStage sibling carrying the cone.
- End-to-end accuracy on a Qwen / TinyLlama MLP block: assert post-hoist kernel matches eager
  output within fp16 tolerance.
- Cone-rejection cases (lone Stage, non-invariant cone) still `RuleSkipped`.

## M6 — Restore `012_split_inner_for_swizzle.py` (Bucket 12)

**Why.** TMA's 128 B swizzle modes (`B32` / `B64` / `B128`) eliminate bank conflicts on smem reads in
the consumer body when the inner cache dim is split to match the swizzle's tile geometry. Without M6,
TMA kernels get the descriptor-based TMA box-copy speedup but lose ~20-40% to bank conflicts on the
consumer reads.

**Change.**

- In `012_split_inner_for_swizzle.py`, replace the stub. Walk for `TmaBufferedStage` (output of M5).
  For each, inspect the inner cache dim's extent — if it's > swizzle granularity (`B128 = 128 B`), split
  the inner dim into outer (swizzle-tile-count) + inner (swizzle-tile-elem-count). Update the Source's
  `cache_dims` tuple accordingly.
- The split rewrite operates on `Source.cache_dims` directly: `(outer_dim, inner_dim, ..., the_inner) →
  (outer_dim, inner_dim, ..., the_inner_outer, the_inner_inner)`. The body's Loads referencing the
  inner dim Var get σ-rewritten through `Var(inner) → Var(inner_outer) * swizzle_elem + Var(inner_inner)`.
- Stamp `TmaBufferedStage.swizzle` with the picked mode (`B32` / `B64` / `B128`).
- Update module docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/012_split_inner_for_swizzle.py` (~100 lines)

**Verification.**

- Unit test `tests/compiler/passes/test_swizzle_wrap_body.py`: input a `TmaBufferedStage` with inner
  cache extent 256 at fp32 (= 1024 B); assert post-rewrite the inner is split into (8, 32) with
  swizzle `B128`. Confirm consumer body Loads have σ-rewritten indices.
- Bank-conflict snapshot: count `ld.shared` conflicts on a TMA TinyLlama matmul before vs after M6.

## M7 — Restore `015_pipeline_k_outer.py` as `015_lower_pipelined_async_stage.py` (Bucket 10)

**Why.** Software pipelining provides another ~30-80% on top of cp.async / TMA by overlapping iter N+1
issue with iter N compute. The biggest single perf restoration step.

**Change.**

- Rename file: `015_pipeline_k_outer.py` → `015_lower_pipelined_async_stage.py`.
- New `rewrite()` consumes `AsyncBufferedStage` or `TmaBufferedStage` whose `pipeline_depth > 1` (set
  by this pass itself when it picks an eligible stage; or set by a future autotune knob). Eligibility
  (~40 lines): K_o extent ≥ 2; the wrapped consumer body has no inter-iter SSA reads.
- Bump `pipeline_depth` (e.g. from 1 to 2 for depth-2 pipelining, matching pre-refactor behavior) and
  `buffer_count` to match.
- Emit the prologue / steady-state / epilogue subtree as siblings inside the `Loop(SERIAL_OUTER)`. The
  expansion logic ports from the pre-refactor 015's `_pipeline()` function. Apply σ_first/next/last to
  the stage's `Source` origins (which are properties — the σ rewrites the surrounding scope's
  contribution via outer Loop axis Vars). The wrapped consumer body comes along for the ride.
- Critically: **no Tile-IR `AsyncWait` Stmts emitted**. The wait is implicit at the lowered Stage's
  wrap boundary; the materializer's per-subclass dispatch handles it.
- The materializer currently has an `emit_async_wait` closure (line 263) that lowers Tile-IR
  `AsyncWait` Stmts to Kernel-IR `CpAsyncWait` / `MbarrierWait`. Since M7 stops emitting Tile-IR
  AsyncWait, the closure's *callers* (the `isinstance(stmt, AsyncWait)` dispatch branches at
  `_materialize`'s top loop and inside `_emit_loop`) become unreachable. Delete them in M7. The
  Kernel-IR `MbarrierWait(...)` / `CpAsyncWait(...)` constructors (lines 310, 313) **stay** — those
  are Kernel-IR primitives emitted by the new pipelined-stage lowering itself, not by an
  AsyncWait→Kernel-IR translation.
- Update module docstring.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/015_lower_pipelined_async_stage.py` (renamed;
  ~180 lines)
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` (~20 lines: drop the
  `AsyncWait` Tile-IR Stmt handler if present)

**Verification.**

- Unit test `tests/compiler/passes/test_pipelined_async_stage.py`: input an `AsyncBufferedStage`
  inside `SERIAL_OUTER`; assert post-rewrite the body has `[prologue_stage, main_loop, epilogue]`
  shape with no `AsyncWait` Stmts anywhere.
- Same for `TmaBufferedStage` path.
- Pre-refactor regression: kernels that pre-refactor 015 rejected (single-stage, K_o<2, mixed
  sync/async) still `RuleSkipped`.

## M8 — Delete `AsyncWait` + perf restoration gate

**Why.** Final cleanup once nothing emits `AsyncWait` (M7's last emission removal closed the loop).
Then validate that the restored chain produces byte-identical CUDA against the M1 snapshots.

**Change.**

- Delete `AsyncWait` class from `tile/ir.py`. Drop the registration in `tile/passes.py`. Drop the
  deprecated stub entry from `ARCHITECTURE.md:241`. Drop the import from `tile/ir.py`'s `__all__`.
- Snapshot-equality test `tests/compiler/test_wrap_body_refactor_restoration.py`: parametrize over the
  M1 kernel corpus; compile each with the post-M7 pipeline; assert byte-equal vs the snapshotted
  CUDA. Normalize variable-name suffix patterns if they leak (ideally none do).
- Where the new shape intentionally improves emission (e.g. fewer `__syncthreads` from absorbed
  `AsyncWait`), document the expected diff in the test as a known-safe whitelist; assert the diff is
  limited to that class.
- Run `make bench-kernels-tuned` post-M8; record perf numbers in the PR description vs the M1
  pre-refactor baseline. Acceptable: each kernel's best-variant latency within ±5% of pre-refactor.
- Optionally run a TinyLlama / Qwen 7B end-to-end inference test (`deplodock run --code ...`) to
  validate accuracy.

**Files.**

- `deplodock/compiler/ir/tile/ir.py` (~30 lines deletion)
- `deplodock/compiler/ir/tile/passes.py` (~5 lines deletion)
- `deplodock/compiler/ir/ARCHITECTURE.md` (~5 lines deletion of deprecated entry)
- `tests/compiler/test_wrap_body_refactor_restoration.py` (~80 lines new)
- M1 snapshots can be deleted after the test passes (they were validation scaffolding).

**Verification.**

- `make test` clean.
- Snapshot equality passes for every kernel in the corpus.
- `make bench-kernels-tuned` shows perf within ±5% of pre-refactor baseline. Kernels outside that
  range get a written explanation (e.g. picked variant differs, but the new variant is faster) before
  merge.

---

## Failure modes to watch

- **TMA multi-source promotion gap** (M5). Multi-source `AsyncBufferedStage` (matmul A+B) stays on
  cp.async because TMA's single-thread elected-issue + arrive-tx model doesn't trivially generalize
  to two sources. Pre-refactor 011 had the same limitation. Future work: emit two `TmaBufferedStage`
  instances inside one Loop scope (siblings) — but that requires the sm_90 mbarrier-coordination
  shape that doesn't exist yet. Mitigation: document the gap; perf on matmul keeps cp.async path until
  multi-stage TMA work lands.
- **Inline-fuse output shape unresolved** (M5b). Pre-refactor `007b` with `FUSED_PIPELINE=False`
  emitted a Stage whose cooperative-load body interleaved gmem Loads + cone Assigns + final Write.
  The new shape has no producer body — Sources synthesize one at materialize time. The fused-producer
  needs to live somewhere; candidates are (a) a `compute_chain: tuple[Assign, ...]` field on Source
  applied between Load and Write in `_emit_stage`, or (b) a new `FusedStage` subclass that carries
  an explicit producer-side body. Mitigation: 50-line spike in M5b before committing.
- **Materializer TMA path is 27 old-API reads** (M5). The plan's pre-update assumption "no
  materializer change in M5" was wrong — `emit_tma_stage` and `_partition_tma_groups` reference
  `stage.name` / `.addressing` / `.axes` / `.origin` / `.buf` / `.alloc_extents` / `.buffer_count` /
  `.phase` across 27 sites. Each needs per-Source replacement. Mitigation: write a TmaBufferedStage
  unit-test directly against the materializer FIRST (before 011's pass rewrite), get it green, then
  enable 011 — guarantees 011 doesn't crash on first activation.
- **σ-substitution mismatch across the rename** (M7). Pre-refactor 015 applied σ_first / σ_next /
  σ_last to a specific Loop-tree shape (the flat `[Stage..., AsyncWait, reduce]` siblings). The new
  shape is `[AsyncBufferedStage(body=[reduce])]` — σ application targets are different. Apply σ to the
  Source origins (which are properties walking outer scope — the K_o Var rewrite propagates through)
  and to the wrapped body. Verify on a small matmul that σ_next correctly produces "issue chunk k_o+1"
  with the source origin advanced by one chunk.
- **Pipeline-depth > 2 vs phase math** (M7). M7's first cut may target `pipeline_depth=2` only (lookahead 1).
  Deeper pipelines (4-stage ring with depth=4) need explicit phase tracking via `(k_o / depth) % 2`
  semantics — pre-refactor 015 already had this at lines 186-195. Port the math; gate `depth>2` paths
  behind a knob.
- **Snapshot diff from variable-name suffix drift** (M8). The materializer's variable naming may
  introduce new suffix patterns that don't affect semantics. Mitigation: M8's test normalizes via a
  regex that strips suffix digits before comparison, OR snapshots are regenerated under controlled
  conditions and the diff is exact.
- **AsyncWait import-chain breakage** (M8). Deletion of `AsyncWait` may surface stale imports outside
  the tile-lowering chain (tests, graph serialization, etc). Mitigation: grep the repo for `AsyncWait`
  before M8's deletion commit; clean up each site in the same commit.

## Future extensions (out of scope for this plan)

- **Source-axis annotation** (`plans/stage-source-axis.md`). Adds `source_axis: Axis` to `CacheDim` for
  semantic queries. Slots in cleanly after this plan since promotion passes already query `cache_dims`.
- **Pipelined compound unification.** `plans/pipeline-refactor.md` was absorbed into the
  stage-wrap-body refactor and deleted; the `Pipelined` primitive it proposed became
  `AsyncBufferedStage(pipeline_depth>1)` directly. After M7, the prologue/main/epilogue expansion
  lives in 015. A future refactor could still extract a separate `Pipelined(stages, body, depth)`
  Tile-IR primitive that both 015 (temporal) and a hypothetical 016 (warpspec) lower into — but only
  if warpspec genuinely needs a different lowering input shape than the current AsyncBufferedStage.
- **MMA fragment factorization** (`plans/mma-fragment-factorization.md`). Builds on the wrap-body Stage
  + source-axis machinery. Independent track; doesn't require this plan to land but benefits from it.
- **Multi-source TMA**. The M5 gap (TMA single-source only) gets revisited when sm_90 mbarrier
  coordination for multi-source issue lands as a separate primitive.

## Test additions summary

- `tests/compiler/passes/test_double_buffer_wrap_body.py` — M2 promotion + eligibility regression.
- `tests/compiler/passes/test_async_copy_wrap_body.py` — M3 promotion + cp.async eligibility regression.
- `tests/compiler/passes/test_pad_smem_wrap_body.py` — M4 pad detection + TMA-skip assertion.
- `tests/compiler/passes/test_tma_copy_wrap_body.py` — M5 single-source TMA promotion + arch-gate
  regression + materializer per-Source `emit_tma_stage` rendering.
- `tests/compiler/passes/test_hoist_invariant_compute_wrap_body.py` — M5b hoist cone detection +
  both `FUSED_PIPELINE` polarities.
- `tests/compiler/passes/test_swizzle_wrap_body.py` — M6 inner-dim split + swizzle picking.
- `tests/compiler/passes/test_pipelined_async_stage.py` — M7 pipelining expansion (cp.async + TMA
  paths).
- `tests/compiler/test_wrap_body_refactor_restoration.py` — M8 byte-identical CUDA snapshot equality
  across the kernel corpus.
- ~~Existing `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py`~~ — **deleted** in commit
  `89cfc456` (its old-shape assertions don't apply). Write fresh tests for the new shape in
  `test_pipelined_async_stage.py` (M7).
- ~~Existing `tests/compiler/passes/test_hoist_invariant_compute.py`~~ — **deleted** in commit
  `89cfc456`. Fresh tests in `test_hoist_invariant_compute_wrap_body.py` (M5b).
- `make bench-kernels-tuned` post-M8 — perf-restoration acceptance gate (±5% of pre-refactor baseline).

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/010_double_buffer.py` — M2
- `deplodock/compiler/pipeline/passes/lowering/tile/013_async_copy.py` — M3
- `deplodock/compiler/pipeline/passes/lowering/tile/014_pad_smem.py` — M4
- `deplodock/compiler/pipeline/passes/lowering/tile/011_tma_copy.py` — M5
- `deplodock/compiler/pipeline/passes/lowering/tile/007b_hoist_invariant_compute.py` — M5b
- `deplodock/compiler/pipeline/passes/lowering/tile/012_split_inner_for_swizzle.py` — M6
- `deplodock/compiler/pipeline/passes/lowering/tile/015_lower_pipelined_async_stage.py` — M7
  (renamed from `015_pipeline_k_outer.py`)
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` — M2 (delete /
  rewrite `_assert_stage_body_shape`), M5 (`emit_tma_stage` + `_partition_tma_groups`), M7
  (delete Tile-IR `AsyncWait` dispatch branches)
- `deplodock/compiler/ir/tile/ir.py` — `AsyncWait` deletion (M8)
- `deplodock/compiler/ir/tile/passes.py` — `AsyncWait` dispatch deletion (M8)
- `deplodock/compiler/ir/ARCHITECTURE.md` — deprecated `AsyncWait` entry removal (M8)
- `tests/perf/snapshots/wrap_body_refactor/` — M1 baseline snapshots (deletable after M8 passes)
