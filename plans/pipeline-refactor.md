# Pipeline Refactor — Declarative `Pipelined` Compound for K-outer Pipelining

## Context

Today's `015_pipeline_k_outer.py` transforms a sync-style async K-outer loop into a software-pipelined
prologue / steady-state / epilogue triple. The expansion happens entirely inside the pass: σ-substitutions
(`sigma_first`, `sigma_next`, `sigma_last`) are computed and applied; the reduce body is duplicated across
the steady-state and epilogue; the TMA-vs-cp.async split branches at lines 204–220; the per-slot phase/slot
math sits at lines 186–195. The output is a flat list of three sibling subtrees inside the same Tile body —
no Tile-IR node remembers that "this is a pipelined K-outer."

This refactor inverts that: 015 stops expanding and starts *annotating*. It wraps the eligible K_o body in a
new declarative compound Stmt — `Pipelined(stages, body, depth)` — that captures the pipelining intent
without committing to a lowering. A new sibling pass `015b_lower_pipelined_temporal` performs today's
expansion. The kernel output is byte-identical.

The motivation is twofold:

- **Eliminate duplication in 015.** Today the reduce body (`others`) is written into the main loop once and
  into the epilogue again with `sigma_last` applied; the stage decls are written for the prologue, the
  steady-state, and (via `keep`/`phase`/`slot` math) for the consumer side. Pulling the structural intent
  into one IR node lets the lowering compute these once.
- **Make warp specialization a sibling, not a parallel implementation.** A future `016_lower_pipelined_warpspec`
  consumes the same `Pipelined` annotation and emits a `WarpSpec(producer, consumer)` form instead of the
  temporal expansion. Two lowering paths share one declarative input. Without this refactor, 016 would
  duplicate 015's eligibility check, slot/phase math, and stage handling.

### Why now

The `Pipelined` compound is small (~80 lines of IR plumbing) and its only consumer today is 015. Doing the
refactor before 016 starts means 016 lands as a sibling lowering pass (~50 lines) rather than a parallel
implementation of 015's expansion logic. It also means today's 015 code — which has small but real friction
(double-substituted reduce body, has_tma branching, scattered σ math) — gets cleaned up as a load-bearing
side-effect of the refactor, not as a separate "cleanup" PR.

### Scope guard

This plan **only** refactors the pipelining-pass surface. It does not touch:

- The eligibility predicate (`_eligible` in `015_pipeline_k_outer.py:96-153`) — that logic moves verbatim
  into the new annotation pass.
- `013_async_copy` / `011_tma_copy` / `010_double_buffer` / `014_pad_smem` — they continue to produce
  today's flat `[Stage..., AsyncWait, reduce]` shape inside the K_o loop. The annotation pass consumes
  that and rewrites to `Pipelined`.
- `016_mark_unroll` — must still see today's prologue/main/epilogue shape post-temporal-lowering. The
  refactor guarantees byte-identical output, so mark_unroll is unaffected.
- Warp specialization itself — out of scope; the refactor merely *enables* it as a future plan.

### Risk note up front

The load-bearing risk is **byte-identical output**: M4 must demonstrate that every kernel produced by the
refactored pipeline matches a pre-refactor snapshot of the CUDA source under `DEPLODOCK_DUMP_DIR`. If
anything diverges — even a renamed temporary — that's a bug in the refactor, not a re-bless candidate.

A secondary risk is the **walker-protocol surface** for `Pipelined`. It's a block-structured stmt with two
body groups (`stages` and `body`); generic walkers (`Body.iter`, `Body.map`, σ-rewrite dispatch) need to
descend into both. Skipping a walker hookup means a downstream pass silently fails to rewrite something
inside `Pipelined` — observable only at materialize time as a stale axis reference or unrewritten σ var.

## Design decisions

1. **Name: `Pipelined`, not `Pipeline` or `PipelinedLoop`.** `Pipeline` collides with the pass-orchestrator
   class (`deplodock.compiler.pipeline.Pipeline`). `PipelinedLoop` would make it a Loop variant at
   `ir.stmt.blocks`, requiring axis/role/is_reduce plumbing it doesn't need. `Pipelined` is a body-level
   compound Stmt that sits as the *sole child* of a regular `Loop(SERIAL_OUTER, K_o)`; the K_o axis stays a
   normal Loop so existing role tagging, σ rewrites, and downstream-pass pattern matches work unchanged.

2. **Lives in `tile/ir.py`, not `stmt/blocks.py`.** `Pipelined` references `Stage` (specifically
   `AsyncBufferedStage` / `TmaBufferedStage`) in its `stages` field — it's a Tile-IR-only concept. `Loop`
   and `StridedLoop` are dialect-neutral and live at `stmt/blocks.py`; `Pipelined` does not.

3. **Two body groups, not one.** `Pipelined.stages` is a tuple of `Stage` Stmts (producer side);
   `Pipelined.body` is a `Body` (consumer side — typically a single K_i reduce loop plus sibling stmts
   like `ComputeStage` reads). Keeping them separate makes the lowering pass's job obvious: substitute
   `K_o → K_o+1` over `stages` for the steady-state issue; substitute `K_o → K_o` over `body` for the
   steady-state compute. A single mixed body would force the lowering to re-classify each child.

4. **`depth` is a structural field, not derived.** Today's 015 reads `buffer_count` off the first stage
   (`015_pipeline_k_outer.py:180`). The new `Pipelined.depth` carries it explicitly so the lowering
   doesn't have to inspect children, and so future passes can introduce `Pipelined` with depth > buffer
   count (e.g. multi-stage pipelines where compute lags by more than one issue).

5. **Pattern match on `Loop(SERIAL_OUTER, body=[Pipelined])` for temporal lowering.** The new pass's
   PATTERN is `Loop` with role `SERIAL_OUTER`; its `_eligible` checks that the loop's body is exactly one
   `Pipelined` stmt. This keeps Loop / Pipelined separable: a `Loop(SERIAL_OUTER)` whose body has not yet
   been annotated (because 015a was disabled or skipped) falls through to materialize unchanged.

6. **Annotation pass replaces, doesn't shadow, 015.** Old `015_pipeline_k_outer.py` is renamed to
   `015a_annotate_pipelined.py`; the temporal expansion logic moves to `015b_lower_pipelined_temporal.py`.
   File rename rather than a parallel pass keeps the eligibility predicate co-located with its only
   consumer (the annotation pass) and the temporal-expansion logic with its only consumer (the lowering
   pass). 016_mark_unroll keeps its slot.

7. **Walker integration via `nested()` / `with_bodies()`.** `Pipelined.nested()` returns
   `(Body(stages), body)` — two `Body` groups in a defined order. `with_bodies()` accepts the same
   2-tuple and rebuilds. Every generic body walker
   (`Body.iter`, `Body.map`, σ-rewrite dispatch via `tile/passes.py`) goes through these and needs no
   `isinstance(Pipelined)` branch.

8. **Structural key excludes `depth`.** `Pipelined` participates in `Body.structural_key` via its child
   bodies' keys plus its own type tag; `depth` is a tuning-time choice (already reflected in `Stage.buffer_count`),
   not a structural fingerprint. Excluding it keeps autotune cache hits stable across depth variations
   when the rest of the kernel is identical.

9. **The temporal-lowering pass owns σ_first / σ_next / σ_last.** These three σ maps live in
   `015b_lower_pipelined_temporal.py` and are derived from `Pipelined.body`'s enclosing Loop. The old 015
   computed them on stages directly; the new lowering applies them to `Pipelined.stages` and
   `Pipelined.body` separately, which is structurally simpler — `stages` get σ_first for prologue,
   σ_next for steady-state; `body` gets identity for steady-state, σ_last for epilogue.

10. **Pass numbering: `015a` before `015b` before `016`.** Tile-IR lowering order is annotate-then-lower-then-unroll.
    The annotate pass runs at the position 015 occupies today; the lowering pass runs immediately after;
    mark_unroll continues to run after the temporal lowering. 016_mark_unroll's `_eligible` already
    pattern-matches today's prologue/main/epilogue shape — byte-identical output (M4) means no change there.

## M1 — Define `Pipelined` Tile-IR Stmt

**Why.** Establish the compound stmt with full walker integration before anything emits it. This is a pure
IR addition: no passes generate `Pipelined` after M1, so no callers can observe it.

**Change.**

- `deplodock/compiler/ir/tile/ir.py`: add `Pipelined(Stmt)` dataclass with fields
  `stages: tuple[Stage, ...]`, `body: Body`, `depth: int`. Implement:
  - `__post_init__`: coerce `body` via `Body.coerce`; assert `depth >= 2`; assert every stage is an
    `AsyncBufferedStage` or `TmaBufferedStage`; assert all stages share `buffer_count == depth`.
  - `deps()`: union of every child stage's deps and every body stmt's deps (default Body aggregation).
  - `external_reads()`: union of every stage's `external_reads()`.
  - `local_decls()`: union of every stage's `local_decls()` (the staged buffer names — same as today).
  - `nested()`: return `(Body(self.stages), self.body)`.
  - `with_bodies(bodies)`: assert `len(bodies) == 2`; return `replace(self, stages=tuple(bodies[0]),
    body=bodies[1])`.
  - `exprs()`: empty — child stages and body carry the Exprs.
  - `pretty(indent)`: emit `Pipelined(depth=N) stages:` then stages indented, then `body:` then body
    indented.
- `deplodock/compiler/ir/tile/passes.py`: register `Pipelined` with the shared `rewrite` / `simplify`
  dispatch. `rewrite` walks both `stages` and `body` through `rewrite(child, rename, sigma, axis_fn)`.
  `simplify` walks both through `simplify(child, ctx)`. Pattern matches today's `Stage` handler at lines
  23–35 — same template, two body groups.
- Re-export `Pipelined` from `tile/ir.py`'s `__all__`.

**Files.**

- `deplodock/compiler/ir/tile/ir.py` (~80 lines: class + protocol methods + `__all__` entry)
- `deplodock/compiler/ir/tile/passes.py` (~25 lines: two singledispatch registrations)

**Verification.** Unit tests in `tests/compiler/ir/test_tile_pipelined.py`:

- Construct a minimal `Pipelined(stages=[AsyncBufferedStage(...)], body=Body([Loop(...)]), depth=2)`;
  assert `nested()` returns a 2-tuple of Body, `local_decls()` matches the stage name, `deps()` is empty
  for a self-contained body, `pretty()` indents stages and body under the wrapper.
- `Body.iter` over a body containing `Pipelined(...)` yields every nested stmt at the right depth.
- `Body.map(lambda s: s, Pipelined-containing body)` returns a body equal to the input (idempotent map
  through the new walker hooks).
- σ-rewrite a `Pipelined` with `Var(K_o)` in one stage's origin via `s.rewrite(_id, Sigma({K_o:
  Literal(7)}))`; assert the resulting stage's origin has `7` substituted, and the body is also walked.

## M2 — Annotation pass: rename 015 → `015a_annotate_pipelined.py`

**Why.** Replace today's σ-expansion implementation with a much smaller wrap-in-Pipelined transform. The
eligibility predicate is preserved verbatim; everything below `_eligible`'s return site is replaced.

**Change.**

- Rename `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py` →
  `015a_annotate_pipelined.py`. Keep the module docstring up to "Trigger conditions" but rewrite the
  "Output shape" section to describe the `Pipelined` annotation.
- Keep `_eligible(loop, invariant_names)` verbatim — same predicate, same role checks, same buffer/sync
  guards.
- Delete `_pipeline()` (the σ-expansion logic). Replace its single call site in `_process` with:
  ```
  stages = tuple(s for s in s.body if isinstance(s, (AsyncBufferedStage, TmaBufferedStage)))
  others = tuple(s for s in s.body if not isinstance(s, (AsyncBufferedStage, TmaBufferedStage, AsyncWait)))
  depth = stages[0].buffer_count
  new_body = (Pipelined(stages=stages, body=Body(others), depth=depth),)
  replacement = (Loop(axis=s.axis, role=s.role, body=new_body, unroll=s.unroll),)
  ```
- Drop the imports the deleted logic referenced (`Sigma`, `Var`, `Literal`, `Expr`).
- Update `PATTERN` if needed (it's still `Pattern("root", TileOp)`).

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/015a_annotate_pipelined.py` (~90 lines after renames
  and deletions — net delete of ~140 lines from the original 232)

**Verification.** Two-pronged:

- Unit test `tests/compiler/passes/test_annotate_pipelined.py`: input a TileOp matching today's 015
  eligibility (cooperative async-staged K_o loop with two `AsyncBufferedStage`s and a reduce); assert the
  output Loop body is exactly `(Pipelined(...),)`, that `Pipelined.depth == 2`, that the stages and reduce
  are correctly split.
- Eligibility regression: any TileOp the old 015 rejected (mixed sync/async stages, single stage, K_o
  extent < 2, cross-iter SSA dep) must still produce `RuleSkipped` from 015a. Re-use the test cases from
  `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py`.

## M3 — Lowering pass: `015b_lower_pipelined_temporal.py`

**Why.** Perform today's prologue / steady-state / epilogue expansion, consuming `Pipelined` instead of
the flat `[stages, AsyncWait, reduce]` shape.

**Change.**

- New file `deplodock/compiler/pipeline/passes/lowering/tile/015b_lower_pipelined_temporal.py`.
- `PATTERN = [Pattern("root", TileOp)]`.
- `rewrite(root)`:
  - Find the single Tile in `root.op.body`.
  - Walk the Tile body for `Loop(SERIAL_OUTER) with body == (Pipelined(...),)`; for each match, emit
    prologue + main loop + epilogue.
  - If no matches, `RuleSkipped`.
- Expansion logic (factored out into `_expand_pipelined(loop, pipelined) -> tuple[Stmt, ...]`):
  - `K_o = loop.axis`; `n_chunks = K_o.extent`; `depth = pipelined.depth`.
  - Build `sigma_first = Sigma({K_o: Literal(0)})`, `sigma_next = Sigma({K_o: Var(K_o) + Literal(1)})`,
    `sigma_last = Sigma({K_o: Literal(n_chunks - 1)})`.
  - Prologue: `[s.rewrite(_id, sigma_first) for s in pipelined.stages]`.
  - Steady-state body, branched on TMA presence:
    - cp.async path: `[*body_stages, AsyncWait(keep=len(stages)), *pipelined.body]`.
    - TMA path: `[AsyncWait(keep=len(stages), phase=body_phase, slot=body_slot), *pipelined.body,
      *body_stages]` (WAIT-then-PREFETCH ordering — same constraint as today's 015 at lines 204–220).
  - Compute `body_phase = (Var(K_o) / depth) % 2`, `body_slot = Var(K_o) % depth`,
    `epi_phase = Literal((n_chunks - 1) // depth) % 2`, `epi_slot = Literal((n_chunks - 1) % depth)` —
    these are exactly today's 015 expressions at lines 186–195, now living in one place.
  - Wrap steady-state in `Loop(K_o', n_chunks - 1, role=SERIAL_OUTER, unroll=loop.unroll)`.
  - Epilogue: `AsyncWait(keep=0, phase=epi_phase, slot=epi_slot)` followed by
    `[s.rewrite(_id, sigma_last) for s in pipelined.body]`.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/015b_lower_pipelined_temporal.py` (~170 lines:
  pattern + expansion + the σ/phase/slot math relocated from old 015)

**Verification.**

- Golden IR: for every test case in `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py` (and any
  matmul/sdpa goldens that exercise pipelining), the output of 015a → 015b must be structurally equal to
  the output of pre-refactor 015. Compare via `Body.structural_key` on the post-015b TileOp body vs a
  snapshotted pre-refactor body.
- Unit tests in `tests/compiler/passes/test_lower_pipelined_temporal.py`: input a TileOp containing
  `Loop(SERIAL_OUTER, body=(Pipelined(...),))`; assert output is `[prologue stages..., main loop with
  K_o-1 iters, AsyncWait(keep=0), epilogue body]`.

## M4 — Byte-identical kernel output verification

**Why.** The refactor's contract is that nothing user-visible changes. Validate that contract directly by
comparing rendered CUDA against pre-refactor snapshots.

**Change.** No code beyond test additions.

- Snapshot fixture: before M1, dump `DEPLODOCK_DUMP_DIR` outputs for the kernel corpus —
  TinyLlama layer, Qwen 7B layer, RMSNorm, softmax, SDPA, plus the smaller synthetic matmul cases under
  `tests/compiler/`. Commit the snapshots (the `08_lowering_cuda.cu` files) to `tests/perf/snapshots/` or
  an equivalent location.
- New test `tests/compiler/test_pipeline_refactor_snapshot.py`: parametrize over the snapshotted kernels;
  compile each with the post-refactor pipeline; assert byte-equal CUDA output (after normalizing
  variable-name suffixes if any leak through — but ideally none do).
- After M4 passes, snapshots can be deleted; they were validation scaffolding, not durable test fixtures.

**Verification.** All snapshots match. If any kernel diverges, the refactor has a bug — investigate and
fix; do NOT bless the new output.

## M5 — Documentation

**Why.** Update Tile-IR architecture docs to describe `Pipelined` as a first-class structural primitive,
and update the new passes' docstrings.

**Change.**

- `deplodock/compiler/ir/tile/ARCHITECTURE.md` (if exists; else `deplodock/compiler/ir/ARCHITECTURE.md`):
  add `Pipelined` to the Tile-IR Stmt table; describe the two-body-group structure and its lowering
  paths.
- `deplodock/compiler/pipeline/ARCHITECTURE.md`: update the tile-lowering chain table to reflect the
  015a/015b split; describe `Pipelined` as the structural primitive linking them.
- `015a_annotate_pipelined.py` module docstring: explain the annotation contract — eligibility unchanged
  from old 015, output is `Loop(SERIAL_OUTER, body=(Pipelined(...),))`.
- `015b_lower_pipelined_temporal.py` module docstring: describe the prologue/steady/epilogue expansion;
  cite the (now-shared) phase/slot math; note the warpspec sibling as a future extension.

**Files.**

- `deplodock/compiler/ir/tile/ARCHITECTURE.md` or `deplodock/compiler/ir/ARCHITECTURE.md` (~15 lines)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` (~10 lines)
- The two new pass files' docstrings (covered in M2 / M3)

**Verification.** `make lint` clean (markdown wrapping at ~120 chars per CLAUDE.md). Spot-read the
changed sections.

---

## Failure modes to watch

- **Walker miss on `Pipelined.stages`.** If a downstream pass uses `Body.iter` to find every Stage in a
  TileOp and the walker doesn't descend into `Pipelined.nested()[0]`, the pass silently fails. Mitigation:
  M1's walker unit test exercises `Body.iter` over a Pipelined-containing body and counts stages.
- **σ-rewrite skips the body group.** If `tile/passes.py`'s rewrite registration only walks `stages` and
  forgets `body` (or vice versa), σ substitution becomes partial. Mitigation: M1's rewrite test
  substitutes a Var that appears in both groups and asserts both are rewritten.
- **`Pipelined.depth` drifts from stage `buffer_count`.** `__post_init__` asserts equality at construction
  time; a downstream pass that rewrites a stage's `buffer_count` without updating the wrapping
  `Pipelined.depth` would break the invariant. Mitigation: the assertion fires at the next rewrite
  through `tile/passes.py` (which calls the dataclass constructor); add a regression test if observed.
- **Pattern match on `Loop(SERIAL_OUTER, body=(Pipelined,))` skips a valid loop.** 015b's PATTERN matches
  the Loop role; the `_eligible` check tests `len(body) == 1 and isinstance(body[0], Pipelined)`. If 015a
  ever emits a Pipelined alongside sibling stmts in the same Loop body, 015b's eligibility silently
  rejects it. Mitigation: 015a's invariant is "Pipelined replaces the entire Loop body" — add an
  assertion to that effect in 015a after the wrap.
- **Snapshot diff from σ-application order.** Today's 015 applies σ_first/next/last in a specific order;
  if 015b's expansion applies them in a different order (e.g. σ_next before computing body_phase), the
  resulting CUDA may differ in variable-name suffix patterns even when semantically equivalent.
  Mitigation: M4's byte-identical check catches this; align order with old 015 if it fires.
- **016_mark_unroll runs after 015b and may pattern-match the old prologue/main/epilogue shape
  differently.** Today mark_unroll fires on the steady-state Loop produced by 015. If 015b produces a
  structurally-equivalent Loop (same axis name, same role, same body shape), mark_unroll's pattern
  still fires. Mitigation: M4's byte-identical CUDA proves this end-to-end.

## Future extensions (out of scope for this plan)

This refactor establishes `Pipelined` as the structural primitive. Future plans extend it without
redesigning the surface:

- **Warp specialization (`016_lower_pipelined_warpspec.py`).** Sibling lowering pass: matches
  `Loop(SERIAL_OUTER, body=(Pipelined,))` plus a `WARPSPEC=1` knob on the TileOp; emits a new
  `WarpSpec(producer_body, consumer_body)` Stmt and mbarrier-based sync (new `MbarrierInit`,
  `MbarrierWait`, `MbarrierArrive` Kernel-IR Stmts). Reuses today's phase/slot math via shared helpers in
  `_helpers.py`. Mutually exclusive with `015b_lower_pipelined_temporal` — exactly one fires per
  Pipelined-bearing TileOp, keyed by the knob. See discussion in conversation history; this is the
  primary motivator for the refactor.
- **Asymmetric pipeline depth.** `Pipelined.depth > max(stage.buffer_count)` to enable multi-stage
  pipelines where compute lags by more than one issue (Hopper/Blackwell wgmma patterns). Requires
  loosening the M1 invariant; lowering passes already read `depth` from the wrapper.
- **Ping-pong consumers.** `WarpSpec` variant with two consumer groups taking turns reading the ring
  slots. Encoded as a `Pipelined` with a `consumer_groups` parameter; only the warpspec lowering reads
  it (temporal lowering treats it as 1).
- **Producer-side fusion.** A `Pipelined` where some `stages` are `ComputeStage`s reading from earlier
  `AsyncBufferedStage`s in the same group — fused producer that does load + elementwise compute before
  signaling consumers. Stages are already polymorphic on `Stage` subclass, so the IR primitive doesn't
  change; only the lowering grows a `ComputeStage` branch.

## Test additions summary

- `tests/compiler/ir/test_tile_pipelined.py` — `Pipelined` construction, `nested()`/`with_bodies()`,
  `pretty()`, σ-rewrite walks both body groups, `Body.iter` descent (M1).
- `tests/compiler/passes/test_annotate_pipelined.py` — 015a wrap-in-Pipelined transform; eligibility
  regression vs old 015 (M2).
- `tests/compiler/passes/test_lower_pipelined_temporal.py` — 015b expansion to prologue/main/epilogue;
  cp.async and TMA paths (M3).
- `tests/compiler/test_pipeline_refactor_snapshot.py` — byte-identical CUDA output for snapshotted
  kernels (M4; deletable after merge).
- `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py` — existing; verify still passes against
  015a + 015b chain (M2 / M3).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — new `Pipelined` dataclass (M1)
- `deplodock/compiler/ir/tile/passes.py` — `Pipelined` rewrite/simplify dispatch (M1)
- `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py` → renamed to
  `015a_annotate_pipelined.py` (M2)
- `deplodock/compiler/pipeline/passes/lowering/tile/015b_lower_pipelined_temporal.py` — new file (M3)
- `deplodock/compiler/ir/tile/ARCHITECTURE.md` or `deplodock/compiler/ir/ARCHITECTURE.md` — doc (M5)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` — doc (M5)
- `tests/compiler/passes/test_pipeline_k_outer_sync_stage.py` — existing tests must still pass against
  the new pass chain
