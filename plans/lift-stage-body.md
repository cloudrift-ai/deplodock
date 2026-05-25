# Lift Body Out of Stage — Flatten the Wrap-Body Chain

**Status:** Exploratory. This plan inverts a refactor that just landed (see `plans/finish-stage-wrap-body-refactor.md`).
The audit below documents what flattening would buy and what it would cost. Read both before committing.

## Context

`Stage` and its subclasses (`BufferedStage`, `AsyncBufferedStage`, `TmaBufferedStage`, `ComputeStage`) currently wrap
their consumer subtree via `Stage.body: Body`. Multiple Stages stacked in a K_o body form a chain where each outer
Stage wraps the next:

```
SerialTile(K_o)
└── Stage[w_smem]
    body:
    └── Stage[y_smem]
        body:
        └── Stage[x_smem]
            body:
            └── ComputeStage[fused]
                compute: <producer template>
                body:
                └── SerialTile(K_i, reduce) <consumer>
```

The pretty printer renders each Stage's decl at the same indent as its body, so the chain visually looks like
siblings — but in the IR `w_smem` is the only direct child of K_o.body.

**Proposed flat shape:** lift `Stage.body` out. Stages become sibling stmts followed by the consumer body:

```
SerialTile(K_o)
├── Stage[w_smem]              # producer only
├── Stage[y_smem]
├── Stage[x_smem]
├── ComputeStage[fused]        # producer + compute; no consumer body
└── SerialTile(K_i, reduce)    # consumer, now at K_o body level
```

`Stage.body` disappears. `ComputeStage.compute` stays — it's the producer template (cooperative load that fills the
slab), not a consumer wrap. `_flatten_wrap_stages` in `100_materialize_tile.py:85-120` already does this conversion
at the kernel-IR boundary — the proposal is to push that flat shape up into the Tile IR itself.

## Goals

1. Sibling Stages render honestly as siblings (no "looks flat, is actually a chain" confusion in pretty output).
2. Remove `_flatten_wrap_stages` from the materialize boundary — it becomes a no-op.
3. Uniform iteration: passes iterate K_o.body and see all Stages directly without descending wrap chains.
4. Decouple Stage *insertion order* from Stage *promotion eligibility* — today's "outermost wraps everything else"
   accident (002's admit order picks which Stage gets cp.async) goes away.

## Audit findings — what wrap-body is buying today

From an audit of every `Stage.body` reader in `deplodock/compiler/`:

### Things that would need reconstruction in a flat shape

1. **Scope-transparency for Init placement** (`020_place_inits.py:181-230`). Recently extended in commit `791c5a59`
   to walk nested wrap-body Stage chains: when `stmt` is a `Stage`, `_accums_under_reduces_only` walks `stmt.body`
   to collect Accums that belong at the enclosing scope. `_is_reduce_recursive` synthesizes a probe-loop with
   `.body = s.body` so the recursive reduce-check treats the consumer subtree as if it were a loop body. Without
   `Stage.body`, this becomes a sibling-window walk: collect Accums from stmts *after* the last Stage in the scope,
   up to the next reduce boundary. Doable but loses the structural anchoring.

2. **Phase-rewriting in 040_use_ring_buffers** (`040_use_ring_buffers.py:94-112`). Today: `body.map(_make_phase_load_
   rewriter(staged_names, phase))` rewrites every `Load` in the consumer subtree to prepend the ring phase. In a
   flat shape, this becomes: collect the stmts between the promoted Stage and the next non-staged stmt at K_o.body
   level, treat that window as the "consumer scope," and apply the rewriter. The window detection is the new logic.

3. **Cone detection in 030_hoist_invariant_compute** (`030_hoist_invariant_compute.py:131-189`). Uses
   `_find_unique_stage_inner_reduce(stage.body)` to locate the K_inner reduce inside a multi-source Stage's
   consumer. Cone candidates are grouped by per-Source cache_dims, then Assign-dep walks happen inside the body.
   Flat shape needs explicit sibling-window scope tracking; cone-source grouping needs a way to know which sibling
   Stages contribute to which downstream reduce.

4. **Pipelined-stage expansion in 015** (`080_pipeline_stages.py:116-200`). Today unpacks
   `*stage.body` directly into prologue/main/epilogue siblings after σ-rewriting. Flat shape: identify the sibling
   window after the AsyncBufferedStage (up to and including the K_inner reduce), apply σ_first / σ_next / σ_last
   to *that window*, splice the rewritten copies back in. The expansion logic is the same; the window discovery
   is new.

5. **Stage classification in 020_stage_inputs** (`020_stage_inputs.py:121-192`). Today walks
   `SerialTile(stage_inner)` bodies inside the consumer subtree to collect Loads and classify by cache axes. Flat
   shape: walks K_o.body for `stage_inner` reduces directly (simpler — they're now siblings, not nested).

### Costs that are smaller than they look

- **`_flatten_wrap_stages`** (`100_materialize_tile.py:85-120`) becomes a no-op — net deletion, not a cost.
- **`Stage.pretty`** simplifies — no special-case logic to render wrapped consumer at same indent.
- **`ComputeStage` doesn't need restructuring** — its `compute` field is the producer template, semantically
  unrelated to `body`. Only `body` goes away.

### Misconceptions about wrap-body benefits

- **"Outermost-only double-buffer is a wrap-body accident."** **False.** It's also semantically correct. Inner
  Stages don't iterate with K_o — they read from sibling-stage smem or load M/N-indexed gmem that doesn't change
  per K_o iter. Ring-buffering them would waste smem with no latency benefit. So in flat shape, 010 still needs
  to gate on "this Stage's loads depend on K_o" — same predicate, just expressed differently. The candidate-set
  size grows (every Stage at K_o.body level instead of just the outermost), but the *promoted* set should be
  identical for matmul-shape kernels.
- **"Pretty rendering needs flat IR to render flatly."** False — the pretty printer could be changed independently
  to walk wrap-body chains and render siblings. The deceptive rendering today is a pretty-printer choice, not an
  IR constraint.
- **"Pass simplification across the board."** Mixed. 002 / 015 simplify. 020_place_inits / 010 / 007b get more
  complex (need sibling-window logic instead of structural traversal). Net LOC is approximately a wash.

## Scope

- **In scope.** Remove `Stage.body` (and `BufferedStage.body`, etc.) from `tile/ir.py`. Update every pass that
  reads it. Replace scope-transparency mechanisms with sibling-window walkers. Delete `_flatten_wrap_stages`.
  Update `Stage.pretty` to render trivially (decl line only). Update `ARCHITECTURE.md`. Snapshot-equal CUDA output.
- **Out of scope.** `ComputeStage.compute` stays untouched (it's the producer template, not a consumer wrap).
  No new optimization opportunities; this is a pure structural refactor with byte-identical CUDA as the gate.
- **Decision deferred.** Whether `Source.compute_chain` (from `finish-stage-wrap-body-refactor.md` M5b) lands
  before or after this refactor — they're independent but both touch the producer-side shape. Sketch the
  interaction in M1.

## Risk note up front

This refactor inverts work the team just landed. Before starting, verify:

1. **`plans/finish-stage-wrap-body-refactor.md` is fully complete.** M5b (030_hoist_invariant_compute) was the
   last open milestone per the doc; the recent commit history (`791c5a59`, `50eae636`) suggests it's done.
   Reversing in the middle of the wrap-body restoration would be much worse than reversing after it lands.
2. **The user genuinely wants flat.** This plan should be approved against the alternative of "fix the pretty
   printer to render chains as chains, keep wrap-body." That's a 50-line change vs. this plan's ~600-LOC
   refactor. Most of the "honest rendering" benefit can be had without an IR change.
3. **The "outermost double-buffer" complaint isn't actually a complaint.** If 010's current behavior is
   semantically correct (it is — see findings), then "more uniform candidacy" buys nothing.

If those checks pass and flat is still the right call, the work is bounded and tractable. If they don't, this
plan should not land.

## Design decisions

1. **Single feature branch, milestone commits after `make test`.** Per `feedback_single_branch_milestones.md`.
   Branch `feature/lift-stage-body` or similar.

2. **Sibling-window helper as shared infrastructure.** Multiple passes need "find the consumer scope after this
   Stage at K_o.body level." Implement `consumer_window(body: Body, stage_idx: int) -> tuple[int, int]` once in
   `tile/passes.py` (or a new `tile/scope.py`); reuse from 010, 007b, 015, 020_place_inits. Single
   implementation = single source of truth for sibling-window semantics.

3. **`ComputeStage.compute` unchanged.** The producer template stays as a separate Body. Only `body` is lifted.
   Pretty printer for `ComputeStage` renders `compute` as the synthesized cooperative for-nest (today's behavior);
   the consumer disappears from its rendering (now lives at the same scope as the ComputeStage).

4. **`Stage.body` deletion is the last IR change.** Land sibling-window infrastructure (M2) and update each pass
   to *consume both shapes* (M3–M7) before flipping the IR to flat-only (M8). This avoids a single mega-commit
   that touches every pass at once.

5. **Snapshot-equal CUDA as the acceptance gate.** Same gate as the wrap-body restoration plan. Take baseline
   snapshots at M1 from the current `main`/feature tip; M9 asserts byte-equality on the kernel corpus
   (TinyLlama / Qwen 7B layers / synthetic matmul / SDPA / RMSNorm / softmax).

6. **No perf change expected.** This is a pure IR refactor — same passes, same eligibility, same emission.
   `make bench-kernels-tuned` is sanity, not gate.

## M1 — Spike: sibling-window semantics + snapshot baseline

**Why.** The single biggest unknown is whether sibling-window logic in 020_place_inits, 010, 007b, 015 can
faithfully reproduce wrap-body's scope transparency. Spike it on the simplest pass (010) before committing to
the full migration.

**Change.**

- Implement `consumer_window` prototype in a scratch file; use it in a parallel-but-not-committed version of
  010 against a flat input shape (construct one manually in a test fixture).
- Validate the window-walk produces the same Load-rewrite set as `body.map(...)` on the equivalent wrap-body
  shape.
- Snapshot baseline CUDA: regenerate `tests/perf/snapshots/lift_stage_body/*.cu` for the kernel corpus from the
  current branch tip (post-wrap-body-restoration). These are the byte-equality targets for M9.
- Decide `Source.compute_chain` interaction: if `finish-stage-wrap-body-refactor.md` M5b's spike picked the
  per-Source `compute_chain` shape, document how it composes with flat Stages (likely: each Stage carries its
  own compute_chain on Source; the consumer sees the slab post-chain regardless of wrap shape).

**Files.**

- `tests/perf/snapshots/lift_stage_body/*.cu` (new)
- Scratch spike under `plans/spikes/lift-stage-body/` — committed for review, not for merge.

**Verification.** Spike's sibling-window 010 produces identical CUDA on a matmul as the wrap-body 010.

## M2 — Sibling-window helper in `tile/scope.py`

**Why.** Shared infrastructure for M3–M7. Land once, reuse everywhere.

**Change.**

- New file `deplodock/compiler/ir/tile/scope.py` (or extend `tile/passes.py`).
- `consumer_window(body: Body, stage_idx: int) -> tuple[int, int]`: given a Body and a Stage's index, return
  `(start, end)` indices of the stmts that constitute that Stage's consumer scope. Definition: stmts strictly
  after this Stage, strictly before the next Stage (or end of body), up to and including the first
  reduce-loop sibling.
- `consumer_scope(body: Body, stage: Stage) -> Body`: convenience returning a `Body` view of the window.
- `enclosing_stages(body: Body, stmt_idx: int) -> tuple[Stage, ...]`: inverse — for a given stmt index, which
  Stages at this scope's level have it in their consumer window. Used by passes that walk inside-out.
- Unit tests for each helper covering: single Stage + reduce; multiple Stages + reduce; nested SerialTile
  with reduce inside; empty consumer (Stage at end of body).

**Files.**

- `deplodock/compiler/ir/tile/scope.py` (~120 lines)
- `tests/compiler/ir/test_tile_scope.py` (~150 lines)

**Verification.** Unit tests cover the window-walk shapes seen in 010 / 007b / 015 / 020_place_inits.

## M3 — Update 020_place_inits.py to use sibling-window

**Why.** Most subtle consumer of `Stage.body` (scope transparency for Accum collection). Validating the helper
works here before touching the promotion passes catches sibling-window bugs early.

**Change.**

- Replace `_accums_under_reduces_only`'s `stmt.body` walk with: detect Stage in iteration; use
  `consumer_window` to find the consumer scope; recurse into that window's stmts. The Accum collection logic
  is otherwise unchanged.
- Replace `_is_reduce_recursive`'s `_Probe()` shim with: detect Stage; check if any stmt in `consumer_window`
  is a reduce loop. Direct walk replaces the probe-loop hack.
- This pass must work with **both wrap-body and flat shapes** during the migration window. Add a `_stage_body`
  helper: returns `stage.body` if `Stage` still has the field, else returns `consumer_window(parent_body,
  stage_idx)`. Once M8 lands, the wrap-body branch deletes.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/kernel/020_place_inits.py` (~40 lines net)

**Verification.**

- Existing tests for 020_place_inits pass unchanged (it still operates on wrap-body input).
- New test in dual-shape mode: construct a flat-shape input manually, verify same Init placement decisions
  as wrap-body equivalent.

## M4 — Update 040_use_ring_buffers.py to use sibling-window

**Why.** Phase-rewriting is the most surgical body-walker; getting it right validates the helper for the
promotion-chain passes.

**Change.**

- Replace `body.map(_make_phase_load_rewriter(...))` with: compute `consumer_window`, build a new Body from
  window stmts with the phase rewriter applied, splice back into K_o.body.
- The eligibility predicate stays the same. The window discovery is the only new logic.
- Dual-shape mode: same `_stage_body` shim approach as M3.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py` (~50 lines net)

**Verification.**

- `tests/compiler/passes/test_double_buffer_wrap_body.py` passes unchanged.
- Dual-shape test: flat-shape input → same `BufferedStage` output with phase-rewritten Loads in the sibling
  consumer scope.

## M5 — Update 030_hoist_invariant_compute.py to use sibling-window

**Why.** Largest body-walker. The cone-source grouping logic is the trickiest sibling-window port.

**Change.**

- Replace `_find_unique_stage_inner_reduce(stage.body)` with `consumer_window` + reduce search.
- Cone-source grouping: today walks per-stage body to find Loads sharing cache_dims. Flat version walks
  the consumer window for each Stage in a sibling group, groups by source cache_dims across the windows.
  The grouping logic is unchanged; the input is "Loads in window W" instead of "Loads in stage S's body".
- Dual-shape mode via `_stage_body` shim.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/030_hoist_invariant_compute.py` (~60 lines net)

**Verification.**

- `tests/compiler/passes/test_hoist_invariant_compute_wrap_body.py` passes unchanged.
- Dual-shape test for both `FUSED_PIPELINE` polarities.

## M6 — Update 080_pipeline_stages.py to use sibling-window

**Why.** Pipelined expansion unpacks `*stage.body` into prologue/main/epilogue siblings — the most direct
"give me the consumer stmts so I can σ-rewrite them" use case.

**Change.**

- Replace `*stage.body` unpacks with `*consumer_window_stmts`.
- σ_first / σ_next / σ_last applied to the window contents; spliced back at the original Stage position
  (prologue) and at the K_o.body level (main / epilogue).
- Dual-shape mode via `_stage_body` shim.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/080_pipeline_stages.py` (~50 lines net)

**Verification.**

- `tests/compiler/passes/test_pipelined_async_stage.py` passes unchanged.
- Dual-shape test for both `AsyncBufferedStage` and `TmaBufferedStage` paths.

## M7 — Update 020_stage_inputs.py + 070_pad_smem.py

**Why.** 002 today produces wrap-body chains; it needs to produce flat siblings instead. 014's only direct
`Stage.body` interaction is via `op.body.iter()` for collection — minor update.

**Change.**

- 002: when emitting Stages for the chosen candidate buffers, lay them out as siblings before the consumer
  body instead of nesting each one inside the previous. The flat output is what the rest of the chain expects.
  This is the *constructive* flip — every other pass has been reading wrap-body or sibling-window
  transparently; this pass starts producing flat.
- After M7, downstream passes still see wrap-body shape from old fixtures and produce wrap-body for tests
  that construct inputs manually. The pipeline as a whole runs flat end-to-end.
- 014: `op.body.iter()` covers nested wrap-body bodies too via `Body.iter` (verify); flat shape works
  unchanged.

**Files.**

- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` (~80 lines net)
- `deplodock/compiler/pipeline/passes/lowering/tile/070_pad_smem.py` (verify; ~0 lines)

**Verification.**

- Full `make test` clean. The flat shape now flows through the entire promotion chain.
- CUDA snapshot diff vs M1 baseline: limited to whitespace / variable suffix drift if any (ideally none).

## M8 — Remove `Stage.body` from IR (the flip)

**Why.** With every reader updated to use sibling-window helpers (with `_stage_body` shim), the wrap-body
branch becomes dead. Delete it.

**Change.**

- `tile/ir.py`: remove `body: Body` from `Stage` dataclass. `Stage.__post_init__` no longer coerces body.
  `nested()` returns `()` (was `(self.body,)`). `with_bodies(())` returns self (no-op). `BufferedStage` /
  `AsyncBufferedStage` / `TmaBufferedStage` inherit the change automatically. `ComputeStage` still has
  `compute: Body` and `nested()` returns `(self.compute,)` only.
- `Stage.pretty` simplifies to: emit per-source decl lines + optional prefix/suffix. No body rendering.
  `ComputeStage.pretty` keeps the cooperative for-nest rendering of `compute`; drops the consumer rendering.
- Delete `_flatten_wrap_stages` from `100_materialize_tile.py:85-120`. Replace its callers with direct
  iteration over `body`.
- Delete `_stage_body` shims from 020_place_inits / 010 / 007b / 015. They now always call `consumer_window`.
- Update `ARCHITECTURE.md` files for `tile/` and `pipeline/` to describe the flat shape.

**Files.**

- `deplodock/compiler/ir/tile/ir.py` (~80 lines deletion + simplification)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` (~40 lines deletion)
- `deplodock/compiler/pipeline/passes/lowering/kernel/020_place_inits.py` (~15 lines deletion)
- `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py` (~10 lines deletion)
- `deplodock/compiler/pipeline/passes/lowering/tile/030_hoist_invariant_compute.py` (~10 lines deletion)
- `deplodock/compiler/pipeline/passes/lowering/tile/080_pipeline_stages.py` (~10 lines deletion)
- `deplodock/compiler/ir/ARCHITECTURE.md` (update Stage section)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` (update wrap-body description)

**Verification.**

- `make test` clean.
- `make lint` clean.
- Snapshot equality test (`tests/compiler/test_lift_stage_body_snapshots.py`) asserts byte-equal CUDA vs M1
  baseline for the kernel corpus.
- Test fixtures that construct wrap-body shapes manually are updated to construct flat shapes.

## M9 — Snapshot-equal gate + perf sanity

**Why.** Acceptance gate. Refactor must produce byte-identical CUDA to pre-refactor baseline.

**Change.**

- Snapshot-equality test parameterized over the M1 kernel corpus.
- Run `make bench-kernels-tuned`; record numbers vs M1 baseline in PR description. Acceptable: within ±2%
  (this is a pure IR refactor; any larger delta is a bug).
- Optionally run TinyLlama / Qwen 7B end-to-end inference test to validate accuracy.

**Files.**

- `tests/compiler/test_lift_stage_body_snapshots.py` (~80 lines new)
- M1 snapshots deletable after the test passes.

**Verification.**

- Snapshot test passes for every kernel.
- Bench within ±2% of pre-refactor.

---

## Failure modes to watch

- **Sibling-window semantics drift.** The single piece of new logic. Risk: `consumer_window` returns a
  slightly different scope than `Stage.body` did, causing one pass to phase-rewrite a stmt that wrap-body
  didn't, or miss one that wrap-body included. Mitigation: M1 spike must validate window output stmt-for-stmt
  against `body.map(...)` on equivalent inputs. M2's unit tests cover every shape M3–M7 will encounter.

- **Pass ordering becomes load-bearing.** Wrap-body's scope was structural; flat's scope is positional. If a
  pass reorders sibling stmts (none should today, but a future one might), the window-walk gives wrong answers.
  Mitigation: document the invariant in `tile/scope.py`'s module docstring; add a `_validate_stage_grouping`
  assertion that runs in debug mode after each promotion pass.

- **`ComputeStage` rendering loses information.** Today's `ComputeStage.pretty` shows the cooperative
  for-nest *and* the consumer body (with shared indent). After M8, the consumer is no longer
  ComputeStage's responsibility — it renders at K_o.body level. Pretty output looks materially different.
  Mitigation: this is a feature (honest rendering), but flag it for reviewer awareness. Update any
  documentation/examples that show ComputeStage rendering.

- **Init placement loses an invariant.** 020_place_inits's wrap-body extension (commit `791c5a59`) was added
  specifically because flat shape needs explicit scope tracking. Reversing means trusting that the
  sibling-window walker reconstructs scope correctly. If place_inits ever needs to look across multiple
  K_o.body scopes (e.g. for outer-Init placement), the lack of structural anchoring may surface as a bug.
  Mitigation: porting test from `tests/compiler/passes/test_place_inits.py` covers known cases; new test for
  the multi-stage chain shape that 791c5a59 was added for.

- **Snapshot drift from suffix patterns.** Variable name suffixes generated by materialize may differ
  trivially if iteration order over body changes. Mitigation: M9 test normalizes via regex if needed; ideally
  the iteration order is unchanged.

- **User changes mind mid-refactor.** This plan inverts a recent landed refactor. If wrap-body proves valuable
  for an unanticipated future use case, mid-refactor reversal would be worse than not starting. Mitigation:
  this plan ships as one branch; the M8 commit is the point-of-no-return. M1–M7 are all dual-shape and safely
  abandonable up to that point.

## Future extensions (out of scope)

- **Pretty-printer fix without IR change.** Alternative path: keep wrap-body in the IR, change `Stage.pretty`
  to walk chains and render with explicit nesting indent (so wrap-body looks wrapped, not flat-with-illusion).
  ~50 LOC. Captures the "honest rendering" benefit without the structural cost. Should be evaluated against
  this plan before committing.

- **More uniform double-buffer eligibility.** Today 010 picks the outermost Stage. With flat siblings + a
  semantic predicate ("Loads in this Stage depend on K_o"), 010 could buffer any K_o-dependent Stage,
  potentially enabling new variants (e.g. buffer A but not B if smem is tight). Out of scope for this plan
  but enabled by it.

- **Reorder Stages for better smem reuse.** With flat siblings, a downstream pass could reorder Stages within
  K_o.body to minimize peak smem live-set. Wrap-body chains constrain reordering; flat doesn't. Worth
  exploring after this refactor.

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — `Stage.body` deletion (M8)
- `deplodock/compiler/ir/tile/scope.py` — new sibling-window helpers (M2)
- `deplodock/compiler/pipeline/passes/lowering/kernel/020_place_inits.py` — sibling-window port (M3)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — delete
  `_flatten_wrap_stages` (M8)
- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` — construct flat output (M7)
- `deplodock/compiler/pipeline/passes/lowering/tile/030_hoist_invariant_compute.py` — sibling-window
  port (M5)
- `deplodock/compiler/pipeline/passes/lowering/tile/040_use_ring_buffers.py` — sibling-window port (M4)
- `deplodock/compiler/pipeline/passes/lowering/tile/070_pad_smem.py` — verify only (M7)
- `deplodock/compiler/pipeline/passes/lowering/tile/080_pipeline_stages.py` — sibling-window
  port (M6)
- `deplodock/compiler/ir/ARCHITECTURE.md`, `deplodock/compiler/pipeline/ARCHITECTURE.md` — flat shape docs
  (M8)
- `tests/perf/snapshots/lift_stage_body/` — M1 baseline (deletable after M9)
- `tests/compiler/test_lift_stage_body_snapshots.py` — M9 acceptance gate
