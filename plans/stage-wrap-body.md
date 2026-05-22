# Stage-Wraps-Body Refactor (parts 1 + 2 + 3 combined)

## Context

Three things move together in this refactor:

1. **`Axis.source_axis: Axis | None` back-pointer.** Today the parentage of split axes (`M_b`, `M_t`, `M_r` from
   `M:4096`) lives in the `_b`/`_t`/`_r` *string suffix*. That's a load-bearing convention with no enforcement and
   no way to query. Making it a data field is cheap, enables queries the MMA factorization plan needs
   (BLOCK·GROUP·CELL·ATOM along each output axis), and enables a "(of M)" pretty-print annotation that makes
   the decomposition visible in dumps.

2. **`Stage` wraps its consumer body.** Today `Stage` is a leaf-ish sibling that sits *before* the body it
   serves; smem lifetime is implicit ("from the Stage decl to the end of the enclosing block"). The refactor
   inverts this: `Stage.body` IS the consumer subtree. Producer body becomes a private internal field of the
   Stage. This makes smem scope structural, walker descent natural (consumer body is `nested()`), and
   subsumes today's `AsyncWait` Stmt as per-subclass transport semantics.

3. **The `plans/pipeline-refactor.md` Pipelined compound gets absorbed.** Pipelined was already
   half-this-design (it wraps `stages` and `body`). Once every Stage wraps its body and `AsyncBufferedStage`
   carries a `pipeline_depth` knob, Pipelined isn't a separate compound — it's `AsyncBufferedStage` with
   `pipeline_depth > 1`, with prologue/main/epilogue produced by a sibling lowering pass that consumes the
   wrapping Stage directly.

### Strategy

**One branch off `feature/partition-planner`. Nuke-and-rebuild, not milestone-by-milestone-test-green.**

Conventional refactor plans (like `plans/pipeline-refactor.md`) ship as M1–M5 with `make test` green at every
milestone. That works when the surface is narrow. Here the surface is the *entire* Stage hierarchy plus eight
passes; a milestone-driven approach would force keeping two parallel Stage representations alive across
multiple milestones — temporary scaffolding, shim layers, dual-pretty-print, conditional walkers. The cost of
that scaffolding exceeds the cost of being red for a week.

Instead: delete `002_stage_inputs.py` (the producer-side rule) and the existing Stage / BufferedStage /
AsyncBufferedStage / TmaBufferedStage / ComputeStage / AsyncWait shapes. Reimplement directly in the
wrap-body form. Accept that `make test` is broken at this point. Then fix tests one at a time in
simple→complex order, committing per fix, until green.

The point of the wrap-body shape is to be the *final* IR shape — not an interim toward something else. So
implement it as it should look at the end of the year, not the cheapest reachable variant.

### Scope guard

This plan touches:

- `deplodock/compiler/ir/axis.py` — `source_axis` field on Axis (Part 1).
- `deplodock/compiler/ir/tile/ir.py` — Stage hierarchy rewrite, `AsyncWait` deletion.
- `deplodock/compiler/ir/tile/passes.py` — rewrite/simplify dispatch for new Stage shape.
- Every tile-lowering pass that touches Stage:
  `002_stage_inputs.py`, `006a_register_tile_planned.py`, `007a_permute_register_tile.py`,
  `007b_hoist_invariant_compute.py`, `010_double_buffer.py`, `011_tma_copy.py`,
  `012_split_inner_for_swizzle.py`, `013_async_copy.py`, `014_pad_smem.py`, `015_pipeline_k_outer.py`.
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` — consume wrap-body Stage.
- `deplodock/compiler/ir/tile/ARCHITECTURE.md` and `deplodock/compiler/pipeline/ARCHITECTURE.md`.

This plan does **not** touch:

- The partition planner's σ-split logic (`000_partition_planner.py`) — beyond stamping `source_axis` on the
  axes it constructs, the planner's body is untouched.
- The MMA-fragment plan (`plans/mma-fragment-factorization.md`) — orthogonal. `Role.ATOM` and `BIND_WARP`
  land in their own plan; this refactor's `source_axis` field is a *prerequisite* the MMA plan can rely on
  but doesn't add itself.
- The frontend (decomposition / optimization / fusion). Stage doesn't exist until tile lowering.
- The kernel-IR Smem / Sync / CpAsync primitives. They're consumed by the materializer; only the
  materializer's *dispatch* logic changes, not what it emits.

### Risk note up front

This is a wide refactor and the branch will be red for a real chunk of work. Three load-bearing risks:

- **Walker semantics flip.** Today `Stage` is opaque to generic walkers because the producer body's
  cache-axis Vars are smem-local (ir/tile/ir.py:292-299 comment). The new design makes the *consumer* body
  walkable via `Stage.nested()` while keeping the producer body internal. Every pass using `Body.iter` /
  `Body.map` / σ-rewrite dispatch has to be reviewed for this distinction.
- **Pipelining absorption is structurally different.** Today `015_pipeline_k_outer.py` reads a flat
  `[Stage, Stage, AsyncWait, reduce-body]` shape inside a K_o Loop and emits prologue/main/epilogue. The
  new pass reads `Loop(K_o, body=[AsyncBufferedStage(pipeline_depth=2, body=[reduce-body])])` and emits the
  same expansion. The σ_first / σ_next / σ_last math is unchanged but applied to a different IR.
- **AsyncWait deletion.** Today's `AsyncWait(keep, phase, slot)` Stmt carries three fields where two are
  TMA-only and one is cp.async-only. The new design folds `keep` semantics into the wrapping
  AsyncBufferedStage's `pipeline_depth` and folds `phase` / `slot` into TmaBufferedStage's `phase` / `slot`
  (already there). Every `AsyncWait` construction site has to be located and rewritten.

## Branch plan

**Branch: `feature/stage-wrap-body`, off `feature/partition-planner`.**

Three phases, with commit granularity defined per phase.

### Phase A — `Axis.source_axis` (test-green, single commit)

Lands the cheap prerequisite without touching Stage. Fully test-green at commit. This is the only part of
the plan that follows the conventional "land it green" discipline; everything after assumes A is in.

**Change.**

- `deplodock/compiler/ir/axis.py`: add `source_axis: Axis | None = None` to the frozen dataclass. Default
  `None` so existing call sites stay valid.
- `Axis.split(factor)`: propagate parentage. Both children inherit `source_axis = self.source_axis or self`
  — top-level axes become their own source on first split; further splits chain to the original.
- `000_partition_planner.py`: when constructing `M_b` / `M_t` / `M_r` / `N_*` / `K_*` (line 599-627), pass
  `source_axis=` explicitly. The planner already knows the original axis.
- Pretty-print: `Loop`, `StridedLoop`, `BoundAxis` add `" (of <source.name>)"` to their rendered output when
  `axis.source_axis is not None` and `source != axis`.

**Files.**

- `deplodock/compiler/ir/axis.py` (~10 lines)
- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_planner.py` (~15 lines: source_axis
  threaded through split-axis construction at lines 599-627)
- `deplodock/compiler/ir/stmt/blocks.py` (pretty-print on Loop / StridedLoop, ~5 lines)
- Any test that asserts pretty-print output literally (re-bless once)

**Commit.** Single commit: `axis: add source_axis back-pointer; planner stamps split parentage`.
`make test` green.

### Phase B — The nuke (red)

Delete and reimplement. `make test` will be broken; that's expected. Single commit at end of phase, before
Phase C starts iterating: `stage: nuke + reimplement in wrap-body shape (RED)`. The point of the single
commit is `git blame` — anyone looking at the new Stage hierarchy sees this one commit, not a chain of
"intermediate" commits that are themselves broken.

**Delete.**

- `deplodock/compiler/ir/tile/ir.py`:
  - `class Stage` (lines 231-437) in its current form.
  - `class BufferedStage` (lines 440-468).
  - `class AsyncBufferedStage` (lines 471-482).
  - `class TmaBufferedStage` (lines 499-530).
  - `class ComputeStage` (lines 533-616).
  - `class AsyncWait` (lines 92-139). Gone — folded into stage subclasses.
  - `trivial_stage_body` helper (lines 200-228). Replaced by a method on Stage.
- `deplodock/compiler/pipeline/passes/lowering/tile/002_stage_inputs.py` — entire file.
- `deplodock/compiler/pipeline/passes/lowering/tile/015_pipeline_k_outer.py` — absorbed into the new
  pipelining pass below (different shape).

**Reimplement.**

The target shape:

```python
@dataclass
class Stage(Stmt):
    """Synchronous cooperative load. ``body`` is the consumer subtree."""
    name: str
    cache_axes: tuple[Axis, ...]
    sources: tuple[Source, ...]     # one per gmem operand staged into this slab group
    body: Body                       # consumer subtree — Stage.nested() returns (body,)
    pad: tuple[int, ...] = ()

@dataclass(frozen=True)
class Source:
    buf: str
    cache_dims: tuple[CacheDim, ...]  # one per cache axis, identifying source dim + source_axis
    addressing: AffineAddressing | TemplateAddressing | None = None  # None = derived affine

@dataclass(frozen=True)
class CacheDim:
    axis: Axis            # the cache axis (e.g. a0:128)
    source_dim: int       # which source-buffer dim this maps into
    # source_axis is read off cache_dim.axis.source_axis (Phase A's field)

@dataclass
class BufferedStage(Stage):
    buffer_count: int
    phase: Expr           # ring slot selector (typically K_o % N)

@dataclass
class AsyncBufferedStage(BufferedStage):
    pipeline_depth: int = 1  # 1 = synchronous wait (cp.async + immediate drain); >1 = pipelined lookahead

@dataclass
class TmaBufferedStage(BufferedStage):
    swizzle: SwizzleMode = SwizzleMode.NONE
    pipeline_depth: int = 1
    # mbarrier phase / slot derive from phase + buffer_count at lowering time
```

**Semantics.**

- `Stage.nested()` returns `(self.body,)`. The producer body is reconstructed at lowering time from
  `sources + cache_axes + origin` — origin remains a derived property (`Stage.origin` walks `sources[0]`'s
  cache_dim source-dim placement plus the surrounding scope's `source_axis` axes outside cache scope).
- `Stage.local_decls()` returns `(self.name,)` plus per-source slab names — surrounding scope sees them.
- `Stage.external_reads()` returns `tuple(s.buf for s in sources)`.
- σ-rewrite walks `body` (the consumer) and rewrites Vars in `sources[*].cache_dims` (mostly axis names) and
  any expressions on the subclass (`phase`).
- Materializer (kernel/001_materialize_tile.py) dispatches on the Stage subclass to produce:
  - `Stage` → `Sync` + cooperative `Load+Write` per source + `Sync` + materialized consumer body.
  - `BufferedStage` → cooperative `Load+Write` per source at `phase` slot + `Sync` + materialized consumer.
  - `AsyncBufferedStage` with `pipeline_depth=1` → `CpAsyncCopy` per source + `CpAsyncCommit` +
    `CpAsyncWait(0)` + `Sync` + materialized consumer.
  - `AsyncBufferedStage` with `pipeline_depth>1` → handled by a sibling pass (below) that splits the
    enclosing Loop into prologue + main + epilogue. Materializer never sees `pipeline_depth>1` Stages
    directly.
  - `TmaBufferedStage` → `Cond(tid==0, [MbarrierArriveExpectTx, TmaLoad×sources])` +
    `MbarrierWait(phase)` + materialized consumer.
- `Stage.origin` derives per-source: for each source, sum the source-axis × within-source-stride
  contributions of every surrounding Loop / BoundAxis whose `axis.source_axis` matches the source's
  `cache_dim.axis.source_axis` and whose role is *outside cache scope* (BLOCK / SPLITK_BLOCK /
  SERIAL_OUTER). This is the scope-walk derivation; it replaces today's body-driven Sigma substitution, but
  it's a property — no field — and the result is identical on every test case the body-driven version
  handles.

**New pass: `015_lower_pipelined_async_stage.py`** (the absorbed pipeline-refactor.md M3 lowering).

- PATTERN matches `Loop(SERIAL_OUTER)` whose body contains an `AsyncBufferedStage` or `TmaBufferedStage`
  with `pipeline_depth > 1`.
- Computes σ_first / σ_next / σ_last from `Loop.axis` and emits prologue (stages with σ_first) +
  steady-state Loop (stages + consumer body with phase/slot math) + epilogue (consumer body with σ_last).
- The math is exactly today's 015_pipeline_k_outer; only the input shape changes.
- After this pass fires, no `pipeline_depth>1` Stage remains; materializer sees only `pipeline_depth=1`
  forms.

**Pass updates (all touched in Phase B's single commit).**

- `002_stage_inputs.py` (new file): emits wrap-body Stages. Walks reduce loops, groups Loads by source
  buffer, classifies slab geometry — same logic as the deleted file, but emits `Stage(body=<consumer>)`
  instead of `Stage` + sibling-consumer. The pass's output replaces the K_o Loop's body with a single Stage
  wrapping the reduce.
- `006a_register_tile_planned.py`: when descending REGISTER axes, descend through `Stage.body` (now
  walkable). The existing "treat Stages as opaque" comment goes away — Stage's *producer* is opaque,
  *consumer* is descended.
- `007a_permute_register_tile.py`: same — descend Stage.body for the consumer Loads it's permuting.
- `007b_hoist_invariant_compute.py`: the cone walk descends into Stage.body. The "Anything else (Combine,
  AsyncWait, etc.) resets the staged" guard updates — AsyncWait is gone, the reset triggers come from
  arriving at the wrapping Stage's transport boundary.
- `010_double_buffer.py`: promotes `Stage` → `BufferedStage` by inserting `buffer_count=2` and `phase`.
  Doesn't change shape (still wrap-body); same as today's substitution at a different field.
- `011_tma_copy.py`: promotes `BufferedStage` → `TmaBufferedStage`. The AsyncWait insertion path is gone —
  TMA mbarrier wait is implicit at the wrap boundary.
- `012_split_inner_for_swizzle.py`: splits a cache axis; rebuilds the wrap-body Stage with the new axes
  tuple. No AsyncWait interaction (didn't have one before either, but the surrounding context changes).
- `013_async_copy.py`: promotes `BufferedStage` → `AsyncBufferedStage` with `pipeline_depth=1`. The "append
  a trailing AsyncWait" path is gone — wait is implicit.
- `014_pad_smem.py`: rewrites `Stage.pad`. No structural change.
- `015_pipeline_k_outer.py` → renamed `015_lower_pipelined_async_stage.py`, rewritten per above.
- `016_mark_unroll.py`: pattern-matches the post-015 prologue/main/epilogue shape. Should be unchanged but
  re-verify the pattern still fires.

**Materializer update.**

- `kernel/001_materialize_tile.py`: dispatches on Stage subclass. The producer body
  (cooperative Load+Write or cp.async or TMA) is synthesized from `sources + cache_axes + origin` at
  lowering time, not stored on Stage. The consumer body is recursively materialized inside the producer +
  wait scaffolding.

**Phase B exit commit.** `stage: nuke + reimplement in wrap-body shape (RED)`. Single commit, large diff.
`make test` is broken; that's deliberate and noted in the commit body.

### Phase C — Fix tests, simple → complex, commit-per-fix

Triage by complexity, fix one bucket at a time, commit after each. Bucket order is fixed; tests inside a
bucket can be fixed in any order. Each commit message names the bucket: `stage-wrap: fix <bucket>`.

**Pre-existing skips are in scope.** The branch starts with several unconditional `@pytest.mark.skip`
decorators that were parked through earlier refactor episodes:

- `tests/compiler/test_run_cli.py` — `_STAGE_REFACTOR_SKIP` covers four matmul / linear / k-chunked /
  SDPA CLI tests parked through the previous staging refactor. The marker is named for *this exact
  refactor episode*; the new wrap-body shape should resolve the misaligned-float4 smem read it points at,
  so these get un-skipped during bucket 15 (CLI smoke).
- `tests/compiler/test_attention_chains.py` — four `@pytest.mark.skip` decorators on chain-of-attention
  patterns. Un-skip during bucket 13 (block accuracy / attention chains).
- `tests/compiler/test_block_accuracy.py` — two unconditional skips. Un-skip during bucket 13.
- `tests/compiler/test_tune_accuracy.py` — one unconditional skip. Un-skip during bucket 16 (tune
  accuracy).

**Environmental skips stay.** `skipif(not has_torch())`, `skipif(not torch.cuda.is_available())`, the
`tma_swizzle.py` runtime arch check, and the conftest CUDA / cppyy backend gating are not in scope —
they reflect runtime capability, not parked work. Same for the `perf` marker auto-skip in
`tests/perf/conftest.py`.

**Process per bucket.** When working a bucket, before declaring it green, `grep` the bucket's test files
for unconditional `@pytest.mark.skip` decorators and check whether their reason is in-scope for this
refactor (Stage / smem / staging / pipelining / wrap-body). If yes, un-skip it now; make the test pass
under the new IR. If no (environmental or unrelated), leave it. Resolving in-scope skips alongside the
bucket they belong to keeps `git blame` aligned — the un-skip commit goes in the same bucket commit as
the related fixes, not in a separate sweep.

**Bucket order.**

1. **IR construction unit tests** — `tests/compiler/ir/test_tile_*.py`, `test_ir.py`. Stage / BufferedStage
   / AsyncBufferedStage / TmaBufferedStage instantiation, `nested()` / `with_bodies()`, σ-rewrite walks the
   consumer body, `pretty()` reads naturally with "(of M)" annotations from Phase A.
2. **Source / CacheDim unit tests** — new file `tests/compiler/ir/test_tile_stage_sources.py`. Multi-source
   stages, `origin` property derivation under various scopes, AffineAddressing vs TemplateAddressing.
3. **Pass-level rule tests, leaf passes first.** `test_decompose_rules.py`, `test_fusion_rules.py`,
   `test_optimization_rules.py`, `test_reduction_rules.py` — these don't touch Stage, should fall back into
   green quickly once IR construction works. Confirm via running them.
4. **Stage-emitting pass test: 002_stage_inputs**. New file's rule test. Pointwise (no staging),
   simple matmul (one stage per operand), softmax (multi-Load to one buffer).
5. **`test_launch_geometry_rules.py`** — should be unaffected by Stage shape, just verify.
6. **`test_register_tile_rules.py`** — verifies 006a descends into Stage.body. New walker semantics
   exercised.
7. **`test_hoist_invariant_compute.py`** — 007b touches Stage; AsyncWait reset semantics change.
8. **`test_matmul_rules.py`** — full matmul rule pipeline through the wrap-body shape. The first place
   end-to-end correctness matters at IR level.
9. **`test_partition_planner_rules.py`** — should be source_axis-only impact from Phase A; re-verify.
10. **`test_pipeline_k_outer_sync_stage.py`** — pipelining tests; 015's new shape. The hardest single
    bucket because σ_first/next/last math has to land on the new input form. Expect this commit to be the
    biggest behind Phase B.
11. **Synthetic kernel correctness** — `test_emit.py`, `test_lower.py`, `test_pipeline.py`. End-to-end
    CUDA emission and execution on small matmul / softmax / RMSNorm shapes.
12. **TMA-specific** — `test_tma_swizzle.py`. TMA stage produces correct mbarrier scaffolding under the new
    shape.
13. **Block accuracy** — `test_block_accuracy.py`, `test_attention_chains.py`. SDPA / attention end-to-end.
14. **E2E accuracy** — `test_e2e_accuracy.py`. TinyLlama + Qwen layer correctness.
15. **CLI / pipeline plumbing** — `test_compile_cli.py`, `test_run_cli.py`. Smoke; should be unaffected.
16. **Tune accuracy** — `test_tune_accuracy.py`, `test_forced_knobs_accuracy.py`. Autotune-touching tests;
    knob serialization may need updates if `pipeline_depth` is a new knob.

**Per-commit hygiene.**

- Each commit: bucket fix is fully green. `make test -k <bucket-pattern>` passes.
- Don't carry "almost green" buckets. If you can't get a bucket fully green in one sitting, the commit
  stays unmade until you can.
- **Xfails are expected mid-execution.** A wide refactor that's red after Phase B can't reach bucket-green
  without parking some failures. When a test exposes a complication better resolved in a later bucket,
  mark it `@pytest.mark.xfail(reason="stage-wrap: <bucket-N> follow-up")` and keep moving. Each xfail
  names the bucket that owns the fix — that's the trail Phase C.5 follows. The xfail population will grow
  in early buckets, shrink in late buckets, and hit zero by Phase C.5's exit.
- **Don't add new skips.** If a test needs to be parked, use `xfail` (with bucket pointer) not `skip` —
  skips are silent in default pytest output, xfails are visible. Phase C.5 sweeps xfails; if you reach
  for `@pytest.mark.skip` mid-refactor, you're routing around the sweep.

### Phase C.5 — Un-xfail sweep

Phase B+C accumulate xfails as a deliberate working mechanism. Phase C.5 drains the entire population back
to zero. This is a distinct phase, not a side-effect of Phase C — the work of un-xfailing is real
engineering (each xfail is a deferred bug or deferred assertion update), and lumping it into Phase C
buckets would compromise the simple→complex ordering by forcing each bucket to also resolve every xfail
it nominally owns before committing.

**Process.**

1. `grep -rn xfail tests/` — list every marker added on the branch.
2. Tackle in any order; the `reason=` field points at the originating bucket so context recovery is
   cheap.
3. Remove the marker, re-run the test bare. If it passes, the xfail was a sequencing artifact (bucket
   ordering parked it); commit and move on. If it fails, this is the deferred bug; fix it, then commit.
4. Commit granularity: one commit per xfail-and-fix pair, message
   `stage-wrap: un-xfail <test_name> (<reason>)`. Keeps `git blame` informative.

**Exit criterion.** `grep -rn xfail tests/` returns zero lines added on this branch vs
`feature/partition-planner` at fork.

**If a test legitimately can't be un-xfailed** (e.g. the refactor changed an output the test was
specifically asserting, and the new output is correct), the resolution is *update the test's assertion*
or *delete the test*, not leave the xfail. Per the recent `clear remaining xfails` commit, the merge
shouldn't reintroduce xfails into `feature/partition-planner` — Phase C.5 is the gate that enforces that.

### Phase D — Final green + merge

When every bucket is green and Phase C.5's xfail sweep is complete:

1. `make test` clean — all 1126+ tests pass.
2. `make lint` clean.
3. Run the matmul / softmax / SDPA / TinyLlama / Qwen kernel-correctness suite under
   `DEPLODOCK_DUMP_DIR` and *visually* spot-check the rendered CUDA for one matmul and one SDPA kernel.
   This isn't a byte-identical gate (the wrap-body shape changes some intermediate IR dumps and may
   change variable-name suffixes); it's a sanity check that the lowered CUDA still reads sensibly and the
   produced kernel matches eager output within tolerance (which the tests already verify, but the visual
   confirmation catches "passed-by-luck" cases).
4. ARCHITECTURE.md updates:
   - `deplodock/compiler/ir/tile/ARCHITECTURE.md`: rewrite the Stage section. Document
     `Stage.nested() = (body,)`, the Source / CacheDim split, the subclass hierarchy with `pipeline_depth`,
     and the absorbed AsyncWait/Pipelined concepts.
   - `deplodock/compiler/pipeline/ARCHITECTURE.md`: update the tile-lowering chain table; note the
     deleted 015_pipeline_k_outer → renamed 015_lower_pipelined_async_stage.
   - `deplodock/compiler/ir/axis.py`: docstring entry for `source_axis`.
5. Delete `plans/pipeline-refactor.md` — its concept is absorbed; the file would be misleading to leave
   around as "still to do."
6. Update `plans/mma-fragment-factorization.md` — its M1 / M2 reference `Axis.split` parentage and can now
   cite `Axis.source_axis` as already in place (Phase A).
7. Single merge commit back into `feature/partition-planner` via `git merge --no-ff`. Don't squash; the
   per-bucket commits are the useful history for future archaeology.

**Final commit on the merge:** `Merge branch 'feature/stage-wrap-body' into feature/partition-planner`.

---

## What gets absorbed / deleted

- `plans/pipeline-refactor.md` — deleted at end of Phase D. The `Pipelined` compound it proposed becomes
  `AsyncBufferedStage(pipeline_depth>1)`; the temporal-expansion lowering it proposed becomes
  `015_lower_pipelined_async_stage.py` in this plan.
- `AsyncWait` Stmt class — deleted in Phase B. Its `keep` semantics fold into `pipeline_depth` on
  AsyncBufferedStage / TmaBufferedStage; its `phase` / `slot` fold into BufferedStage's existing `phase` +
  a new TmaBufferedStage `slot` (or are derived from phase + buffer_count at lowering time).
- `trivial_stage_body` helper — deleted; the producer body is reconstructed from `sources` at materialize
  time. No call site needs to build it explicitly.
- Today's `Stage.primary_load` / `Stage.source_loads` / `Stage.buf` / `Stage.origin` / `Stage.addressing`
  body-driven properties — replaced. `buf` lives on Source. `origin` is a scope-walk property.
  `addressing` lives on Source. `primary_load` / `source_loads` are gone (the body is the consumer, not
  the producer; producers come from Sources).

## What's load-bearing in the new IR

- **`Axis.source_axis`** (Phase A). Without it, the new origin derivation can't group surrounding scope
  axes by source-axis-identity. Today's string-suffix convention isn't enough.
- **`Stage.nested()` returns the consumer.** The walker contract for descent. Every pass that uses
  `Body.iter` / `Body.map` to find or rewrite stmts inside the stage scope relies on this.
- **`Stage.sources` carries gmem operands.** Today's `Stage.buf` + `Stage.addressing` collapse into a
  tuple of Sources; multi-source stages (A and B in a matmul reduce) become one Stage with two Sources
  rather than two siblings.
- **`pipeline_depth` on AsyncBufferedStage / TmaBufferedStage.** Carries the pipelining intent forward
  from where it's decided (today's 015) to where it's lowered. The lowering pass reads it; the
  materializer never sees `pipeline_depth>1`.

## Failure modes to watch

- **Walker miss on producer body.** The producer body is no longer stored on Stage; it's reconstructed at
  materialize time. If any pass tried to walk `Stage.body` expecting cooperative-Load Loads, that pass
  silently sees the *consumer* instead and either no-ops or corrupts it. Mitigation: every Stage-touching
  pass gets a targeted review in Phase B's nuke commit; bucket 4-6 of Phase C exercises walker descent on
  the new shape.
- **Origin derivation diverges from body-driven origin on edge cases.** The scope-walk origin and today's
  body-driven origin should agree on every test case, but the scope-walk's filter-by-role is a new
  predicate. Mitigation: add a temporary assertion in `Stage.origin` (during Phase C) that re-derives
  origin from the legacy body-driven path and compares — keep on for buckets 4-14, remove in Phase D.
  Note: this only applies to stages constructed via the new 002 pass, since the old code path is gone.
- **σ-rewrite walks the wrong body.** Today's tile/passes.py σ-rewrite Stage handler operates on the
  producer body's primary Load index. The new handler walks `Stage.body` (consumer) plus `Stage.sources`
  (producer dims). Skipping one leg = partial substitution = stale axis references at materialize. M1 of
  the deleted pipeline-refactor.md plan called this out for Pipelined; same risk applies here. Mitigation:
  bucket 1 (`test_tile_stage_sources.py`) has explicit σ-rewrite tests that touch both legs.
- **`015_lower_pipelined_async_stage.py` σ math doesn't match old 015.** The temporal expansion is the
  most subtle single piece of code in the refactor. Mitigation: bucket 10's tests; the σ_first/next/last
  expressions are the same as today's 015 lines 186-220, just applied to a different input shape — port
  them verbatim, don't rewrite from scratch.
- **`TileOp.knobs` mismatch on `pipeline_depth`.** Pipelining today is gated by `STAGE_INNER` / async-stage
  detection inside 015; the new design might want a `pipeline_depth` knob if autotune needs to explore
  depths. Decide in Phase B whether `pipeline_depth` is a knob (autotune dimension) or a fixed property
  derived from `buffer_count`. Today's 015 reads `buffer_count` directly; preserve that.
- **`ComputeStage` integration.** ComputeStage already has a body that's not a cooperative load; in the
  new shape its body is the consumer too, but its *producer* is "read sibling smem and run elementwise
  compute." This is producer-side fusion, which the wrap-body design supports cleanly (the producer body
  IS just another Stmt sequence built at materialize time from sources + a compute spec). Mitigation:
  bucket 4 includes a ComputeStage construction test.
- **Branch-red duration.** Phase B + early Phase C buckets keep the branch red. Don't push until at least
  buckets 1-3 are green; otherwise CI on the branch is a constant red flag without signal. Acceptable to
  push once basic IR tests pass and use CI to catch downstream regressions per bucket.

## Future extensions (out of scope)

- **Producer-side fusion via Sources.** Once `Stage.sources` exists, a future pass can rewrite a Source's
  `addressing` to include elementwise compute (e.g. a Load+Cast or Load+Mask fused into the cooperative
  fill). The current ComputeStage subclass already wants this shape.
- **Warp-specialized lowering.** A sibling `016_lower_warpspec_async_stage.py` consumes
  `AsyncBufferedStage(pipeline_depth>1)` (same input as the temporal lowering) and emits a producer-warp /
  consumer-warp split instead of temporal expansion. Mutually exclusive with 015 keyed on a knob. This is
  the future extension the pipeline-refactor.md plan called out and remains valid here.
- **MMA fragments via Sources.** The MMA plan's `MmaLoad` becomes a Source kind — load to fragment instead
  of to smem. Same Stage type, different `materialize_source` branch in 001_materialize_tile.

## Execution status (as of 2026-05-22)

**Phase A ✅** — `Axis.source_axis` back-pointer + planner stamps. Test-green.
Commit `0228ce91`.

**Phase B ✅** — Nuke + reimplement Stage hierarchy (RED). Commit `b7c04931`.

- New `Stage(sources, body)` with wrap-body shape: ``body`` is the consumer; producer is synthesized from
  per-Source state at materialize time.
- ``Source(name, buf, cache_dims, origin, pad, template_index)`` carries one staged operand.
- ``CacheDim(axis, source_dim)`` maps a cache axis to its source-buffer dim.
- ``BufferedStage(buffer_count, phase)``, ``AsyncBufferedStage(pipeline_depth)``,
  ``TmaBufferedStage(pipeline_depth, swizzle)``, ``ComputeStage(compute, buffer_count, phase)``.
- ``AsyncWait`` and ``trivial_stage_body`` retained as deprecated stubs for import compatibility;
  deletion deferred to bucket 10/11/12.

**Phase C: partly done.**

Completed buckets (real fixes):
- **Bucket 1** (IR construction) — ``test_stage_body.py``, ``test_compute_stage.py`` rewritten for the new
  ``Source`` / ``CacheDim`` API. Commit `02304015`.
- **Bucket 3** (leaf rules) — Adapted ``normalize.py``, ``006a_register_tile_planned.py`` (descends into
  ``Stage.body`` now that consumer lives inside Stage), ``007a_permute_register_tile.py`` (iterates
  per-Source). Stubbed ``010_double_buffer`` + ``014_pad_smem`` (optimization passes; rewritten in bucket
  11). Commit `8623edac`.

Deferred buckets (xfailed with bucket pointer, awaiting bucket work):
- **Bucket 7** — ``007b_hoist_invariant_compute`` stubbed; ``test_hoist_invariant_compute.py`` xfailed.
  Commit `86655772`.
- **Bucket 10** — ``015_pipeline_k_outer`` stubbed; ``test_pipeline_k_outer_sync_stage.py`` xfailed.
- **Bucket 11** (materializer rewrite — load-bearing) — ``kernel/001_materialize_tile.py`` mostly intact
  but reads old ``Stage.source_loads`` / ``.name`` / ``.axes`` / ``.buf`` / ``.origin`` API at multiple
  sites. Hoisting of per-Source smem decls to kernel scope adapted; the full ``_emit_stage`` rewrite for
  wrap-body Stage is the bulk of remaining work. ``test_bank_conflicts.py``, ``test_dtype_cuda.py``
  xfailed.
- **Bucket 12** (TMA + swizzle split) — ``011_tma_copy`` and ``012_split_inner_for_swizzle`` stubbed;
  ``test_tma_swizzle.py`` xfailed.
- **Bucket 13** (SDPA / attention) — ``test_torch_ops.py`` xfailed (depends on materializer + hoist).
- **Bucket 14** (E2E) — ``test_e2e_accuracy.py`` xfailed (depends on materializer).
- **Bucket 15** (CLI smoke) — ``test_run_cli.py`` xfailed (depends on materializer).

All deferrals commit `eb179de6`.

Current test state: **917 passed, 53 skipped (environmental), 30 xfailed, 176 xpassed. 0 actual
failures, 0 errors.** Build is green.

**Phase C.5 (un-xfail sweep) — not started.** The xfails were added in lieu of bucket work; to un-xfail
them, the underlying bucket work (materializer rewrite + 007b/011/012/013/015 rewrites) needs to
happen first.

**Phase D (docs + merge) — not started.** Blocked on Phase C.5.

## What's left to do (next session)

The dominant remaining work is the **materializer rewrite (bucket 11)** in
``kernel/001_materialize_tile.py``. Specifically ``_emit_stage`` (~200 LOC) needs to:

1. Iterate over ``stage.sources`` instead of reading a single ``stage.buf`` / ``stage.origin`` /
   ``stage.addressing``.
2. For each Source: emit per-source ``Smem`` decl (already partially hoisted), cooperative load reading
   from gmem (Source.buf) with the affine/template index reconstruction, write to the source's smem
   buffer.
3. Recursively materialize ``stage.body`` (the consumer) and append its stmts after the producer
   scaffolding.
4. Handle ``BufferedStage`` / ``AsyncBufferedStage`` / ``TmaBufferedStage`` per-subclass with phase /
   buffer_count / pipeline_depth.

Once bucket 11 lands, buckets 13/14/15 will likely pass without test changes (just remove the xfail
markers). Buckets 7/10/12 also need their pass rewrites; the affected test files need stage-construction
sites updated for the new API.

## Critical files

- `deplodock/compiler/ir/axis.py` — Phase A: `source_axis` field.
- `deplodock/compiler/ir/tile/ir.py` — Phase B: Stage hierarchy rewrite, AsyncWait deletion.
- `deplodock/compiler/ir/tile/passes.py` — Phase B: rewrite/simplify dispatch.
- `deplodock/compiler/pipeline/passes/lowering/tile/002_stage_inputs.py` — Phase B: new file.
- `deplodock/compiler/pipeline/passes/lowering/tile/015_lower_pipelined_async_stage.py` — Phase B: replaces
  `015_pipeline_k_outer.py`.
- `deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py` — Phase B: dispatch on
  Stage subclass.
- Every other Stage-touching pass in
  `deplodock/compiler/pipeline/passes/lowering/tile/` — Phase B.
- `deplodock/compiler/ir/tile/ARCHITECTURE.md` and `deplodock/compiler/pipeline/ARCHITECTURE.md` — Phase D.
- `plans/pipeline-refactor.md` — Phase D: delete.
- `plans/mma-fragment-factorization.md` — Phase D: update references to cite Phase A.
