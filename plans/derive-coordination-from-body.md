# Derive Coordination From Body — Drop Tile-IR Coordination Metadata

**Status:** In progress on `feature/partition-planner`. M1 (helper) and M2 (cross-check) landed; revised
scope below after M2 discovery.

**Original premise:** every coordination decision (cooperative reduce, atomic write, broadcast guard) is
derivable from data flow on the tile body, so the planner-emitted tags + coordination-pass-emitted markers
are pure redundancy — delete the lot.

**Revised after M2:** four of five are derivable, but `ThreadTile.cooperative_axes` is NOT. After the
partition planner emits the body, the cooperative-K stride pattern lives in the load *index expression*
(``x[..., a2*512 + a3*256 + a1]``) rather than in any loop kind — and BR=K degenerate kernels have no
stride at all (the trivial 1-iteration loop is inlined, leaving the Accum directly in the ThreadTile body).
A structural rule like "StridedLoop bound to thread axis ⇒ cooperative" misses both shapes. The planner's
tag is the *only* structural source of truth.

So `cooperative_axes` stays. The other four redundancies still go: `splitk_axes`, `Combine`,
`Write.reduce_op`, and `Cond(coop == 0)` wrappers are all derivable from `cooperative_axes` + body index
expressions. Coordination's job collapses from "infer + emit" to "emit" — and even that can be folded
into the materializer once the helper consumes `cooperative_axes` as input.

## Context

The partition planner constructs typed tile flavors (`GridTile`, `ThreadTile`, `RegisterTile`, `SerialTile`)
and tags two of them with axis-subset metadata describing partitioning decisions:

- `GridTile.splitk_axes: tuple[str, ...]` — block-axis names whose CTAs race to the same output cell (cross-CTA
  K-split). Set by the planner when it lifts a K-axis as `Role.SPLITK_BLOCK`.
- `ThreadTile.cooperative_axes: tuple[str, ...]` — thread-axis names whose threads share a reduction (intra-CTA
  K-stride). Set by the planner when it lifts a K-axis as `Role.COOPERATIVE_STRIDE`.

The downstream `001_coordination` pass (`passes/lowering/tile/001_coordination.py`) reads these tags and
materializes the corresponding leaf-level coordination:

- **splitk_axes** → rewrites each `Write` whose index doesn't reference the lifted axis to
  `reduce_op=ElementwiseImpl("add")` (lowers to `atomicAdd` at codegen). For `Write(value=add(indep, dep))`
  Writes (matmul-with-bias shape), decomposes into `atomic(dep) + Cond(axis == 0, atomic(indep))` so the
  axis-independent term lands exactly once.
- **cooperative_axes** → emits `Combine(name, op)` sibling stmts after each reduce subtree (the materializer
  lowers `Combine` to smem tree-halve / warp-shuffle). Wraps each scalar `Write` whose index doesn't reference
  the cooperative axis in `Cond(coop_axis == 0, ...)` so only one thread writes the broadcast value.

After `001_coordination` runs, **both representations are live simultaneously**: the tile still carries the
tag, AND the body now carries the lowered primitives. Every downstream pass and the materializer must agree
that the two are consistent. They never diverge today, but only because nothing else touches them.

### The redundancy in concrete terms

For a cooperative-K RMSNorm (the dump in `dump/05_lowering_tile.kernels.txt`):

```
Tile(axes=(a0:1, t:256), cooperative_axes=(t,)):     ← tag
    StridedLoop(a1 = t; < 3584; += 256):  # reduce
        acc0 <- add(acc0, v0)                         ← Accum inside a thread loop
    Combine(acc0, op=add)                             ← stmt
    v1 = divide(acc0, in0)                            ← acc0 escapes the thread loop
```

Three things saying the same fact:

1. `cooperative_axes=(t,)` on `ThreadTile`.
2. `Combine(acc0, op=add)` as a body stmt.
3. The data-flow pattern itself: "an `Accum` updated inside a loop bound to thread axis `t`, whose name is
   read after the loop."

For split-K matmul:

```
GridTile(axes=(M_b, N_b, K_s), splitk_axes=(K_s,)):  ← tag
    ThreadTile(axes=(M_t, N_t)):
        for K_o:                                      # reduce
            acc <- add(acc, multiply(a, b))
        out[M_b·BM+M_t, N_b·BN+N_t] += acc            ← Write.reduce_op=add (stmt)
        Cond(K_s == 0):
            out[..., ...] += bias[N_b·BN + N_t]       ← decomposed indep term
```

Three things saying the same fact:

1. `splitk_axes=(K_s,)` on `GridTile`.
2. `Write.reduce_op=add` + the `Cond(K_s == 0)` decomposition.
3. The data-flow pattern: "an output `Write` whose index doesn't reference an enclosing block axis."

### Why the redundancy exists

Historical: the planner needed to communicate "K is parallel" to a downstream pass that would actually emit
the corrective primitives. The tag was the channel. The channel was never deleted after coordination ran,
because there was no harm in leaving it — every other pass just ignored it.

Each tag is also derivable from the body. The body is the source of truth at codegen time (it's what the
materializer translates), so the tag carries no information beyond what's already encoded in the loop nest +
Accum + Write structure.

## Goals

1. **Two sources of truth, no more.** Body shape + `ThreadTile.cooperative_axes`. Everything else
   (atomic, broadcast guard, Combine emission point) is derived by the helper at materialize time.
2. **Delete `001_coordination`.** All its work moves into the materializer (`008_materialize_tile.py`) as
   inference at codegen time, driven by the helper.
3. **Delete redundant fields and stmts.** `GridTile.splitk_axes`, `Write.reduce_op`, the `Combine` Stmt,
   and the synthesized `Cond(coop == 0)` wrappers all go away. `ThreadTile.cooperative_axes` stays.
4. **Honest IR.** Reading a `TileOp.body` no longer requires mentally diffing "what's tagged vs. what's
   computed" for the four derivable cases. Cooperativity remains a single declarative tag on the tile.

## Non-goals

- Changing the planner's geometry decisions. Same axes, same extents, same tile flavors emitted — only the
  tag fields are removed.
- Changing CUDA codegen output. Same `atomicAdd`, same warp-shuffle / smem tree-halve, same `if (t == 0)`
  guards in the generated source. Only the IR-level encoding changes.
- Touching the autotune knobs (`SPLITK`, `BR`). They still control planner choices; the planner still emits
  the same set of variants.

## Design — the inference rules

These are the rules the materializer (or a thin pre-materialize escape-analysis helper) applies in lieu of
reading the deleted metadata.

### Rule 1 — Cooperative combine (replaces `cooperative_axes` + `Combine`)

> Given an `Accum(name=acc, op=⊕)` inside a loop nest containing any thread axis `t`, if `acc` is read at any
> point outside the innermost loop nest bounded by `t`, then a cross-`t` combine of `⊕` over `acc` must be
> emitted at the first use point outside that nest.

Implementation: scope walk collecting `{accum_name → enclosing_thread_axes}` and `{ssa_name → first_escape_point}`.
At the escape point, emit the warp-shuffle / smem tree-halve directly (same lowering `Combine` triggers today).

Edge cases:
- Multiple cooperative axes (`t1`, `t2`) over the same Accum: combine across the product. Today the planner
  only sets one cooperative axis per ThreadTile; the rule degrades gracefully if that ever changes.
- Accum is used only inside the same thread loop (no escape): no combine. Today the tag would still be set
  but `coordination` would also no-op via "nothing to write outside the reduce." Same result.
- Nested reduces (cooperative-K of a softmax-style two-reduce kernel): each Accum is independently checked
  against its own escape point. The two reduces in `RMSNorm(x²).sum` and `mean` are both single-Accum.

### Rule 2 — Atomic Write (replaces `splitk_axes` + `Write.reduce_op`)

> Given a `Write(output=out, index=...)` inside a `GridTile`, if the index expressions do not jointly cover
> every enclosing block axis (i.e. some block axis `K_s` doesn't appear in any index), then multiple CTAs are
> racing to the same address — codegen must emit `atomicAdd` instead of plain store.

Implementation: collect `free_vars(index)` per Write, compare against the set of enclosing `GridTile.axes`
names. Missing axes ⇒ atomic. Same check `_write_indexed_by` does today, just hoisted to codegen.

### Rule 3 — Bias-term decomposition (replaces the planner-emit-then-coordination-decompose dance)

> For a Write classified as atomic by Rule 2, if `Write.value` is the SSA name produced by `Assign(op=add,
> args=(a, b))` where exactly one of `a`/`b` transitively depends on an `Accum` from the lifted axis's
> reduction, decompose at codegen into:
> - one atomicAdd of the axis-dependent arg,
> - one `if (K_s == 0) atomicAdd(axis_indep_arg)` for the constant term.

Implementation: same def-DAG walker as `_compute_axis_dep_set` / `_rewrite_write_atomic` today; runs at
codegen instead of in a separate pass. The atomicity question (Rule 2) determines whether this rule fires.

### Rule 4 — Broadcast-write guard (replaces synthesized `Cond(coop == 0)` wrappers)

> Given a `Write` whose index doesn't reference an enclosing cooperative thread axis `t` (per Rule 1), codegen
> emits the store inside `if (t == 0) { ... }`.

Implementation: trivial check at the `Write` codegen site — does any enclosing cooperative axis fail to appear
in `free_vars(index)`? If so, wrap the emit in the guard.

### What's NOT inferable from the body

- **Which thread axes are cooperative vs. output-partition.** This is the one fact that *is* a planner
  decision and isn't recoverable from the body alone, because both cases produce a `ThreadTile` with axes.
  However, it doesn't need to be: Rule 1 only triggers on Accum-escape, which is a structural property of
  the body, not the axis-role labeling. If an Accum escapes through a thread axis, that axis is functionally
  cooperative — regardless of whether the planner "intended" it. If no Accum escapes, the axis is functionally
  output-partition. The role label and the structural fact are equivalent.
- **The combine operator.** Today `Combine(name=acc, op=⊕)` carries `⊕` explicitly. Rule 1 recovers `⊕` from
  the `Accum.op` of the producer. Already redundant in the current encoding — the `Combine.op` is asserted
  to match `Accum.op` at materialize time anyway (see `ir/tile/ir.py:163` "cross-check; if the strategy
  constructs a Combine with the wrong op...").

## Milestones

Single branch `feature/derive-coordination`. Milestone commits after `make test` passes per step. No separate
PRs (per project convention: [feedback_single_branch_milestones]).

### M1 — Add escape-analysis helper (no behavior change)

New file `deplodock/compiler/ir/tile/escape_analysis.py`:
- `accum_escapes_thread_axis(tile_op: TileOp) -> dict[str, set[str]]` — returns `{accum_name → set of thread
  axis names the accum escapes through}`.
- `write_atomicity(tile_op: TileOp) -> dict[Write, frozenset[str]]` — returns `{write → set of enclosing
  block axes NOT in its index}` (non-empty set ⇒ atomic).
- `write_broadcast_axes(tile_op: TileOp) -> dict[Write, frozenset[str]]` — returns `{write → set of enclosing
  cooperative thread axes NOT in its index}` (non-empty ⇒ Cond-guard needed).

Tests in `tests/compiler/ir/test_escape_analysis.py`: hand-built TileOps covering RMSNorm-cooperative,
softmax-cooperative, matmul-splitK, matmul-bias-splitK, plain pointwise, plain matmul (no coordination).
Verify the helper matches what `001_coordination` currently emits.

**No production code uses the helper yet.** Just lands the analysis.

### M2 — Materializer reads the helper alongside the existing tags

`008_materialize_tile.py`: when emitting code for a Write / Accum, consult both the existing tags AND the
helper. Assert they agree (build-time soundness check). Lets us catch divergence before deleting the tags.

Run the full test suite + `bench-kernels` smoke. No assertion failures ⇒ tags and helper are equivalent
across the existing kernel zoo. Failures ⇒ the helper has a missing edge case.

### M3 — Switch materializer to helper, ignore tags

`008_materialize_tile.py`: stop reading `splitk_axes` / `cooperative_axes` / `Write.reduce_op` / `Combine`
stmts. Use the helper exclusively. Tags still on the IR but inert.

Coordination pass `001_coordination` becomes a no-op (still in the pass list, doesn't rewrite anything). Run
full test suite + `bench-kernels`.

### M4 — Delete the coordination pass

Remove `passes/lowering/tile/001_coordination.py`. Remove from the tile pass directory's auto-pickup. Update
`pipeline/ARCHITECTURE.md` and `passes/lowering/tile/ARCHITECTURE.md` to drop the coordination step from the
pass order.

### M5 — Delete the redundant Stmt: `Combine`

Remove `Combine` from `ir/tile/ir.py`. Remove from `graph.py` registry. Remove the `Combine` references in
`_helpers.py`, `stmt/passes.py`, `stmt/normalize.py`. Add a migration note in `ir/tile/ARCHITECTURE.md`
documenting that cooperative-reduce coordination is now codegen-inferred.

### M6 — Delete the redundant field: `splitk_axes` (cooperative_axes stays)

Strip `splitk_axes` from `GridTile` dataclass. Remove planner code that populates it. Remove the
`_pretty_label` branch that renders `splitk=(...)` — `GridTile` renders as plain `grid`.

Update `tile/passes.py` axis-rename helper to drop the splitk propagation logic.

**`ThreadTile.cooperative_axes` stays.** The cooperative-K stride pattern lives in load index expressions
post-staging, and degenerate BR=K kernels have no stride loop at all, so the tag is the only structural
source of truth for cooperativity. The helper reads it as input rather than deriving it (see M2 discovery).

### M7 — Delete the redundant field: `Write.reduce_op`

Strip from `ir/stmt/leaves.py`. Remove from `stmt/passes.py` and `kernel/002_stamp_types.py` (the only
non-coordination producers). Update `kernel/007_vectorize_stores.py` to consult the escape-analysis helper
instead of `Write.reduce_op` for its "don't vectorize atomic writes" guard.

Update `cuda/001_lower_kernelop.py:65` (the only Write.reduce_op reader at CUDA-lowering time) to use the
helper.

### M8 — Final cleanup

- Re-run `make test` + `make bench-kernels`. Verify no perf regression on the existing kernel zoo.
- Update `passes/lowering/tile/ARCHITECTURE.md`, `passes/lowering/kernel/ARCHITECTURE.md`,
  `passes/lowering/cuda/ARCHITECTURE.md`, `ir/tile/ARCHITECTURE.md`, `pipeline/ARCHITECTURE.md` to reflect
  the deleted machinery.
- Update the partition-planner docstring (the long top-of-file comment that walks through the
  `SPLITK_BLOCK → BLOCK → ... → COOPERATIVE_STRIDE` order — still accurate at the planner level but no
  longer reflects what reaches the IR).

## Audit — every reader of the targeted fields/stmts

Generated from `grep -rn "splitk_axes\|cooperative_axes\|reduce_op\|Combine" deplodock/compiler/`.

### Readers of `GridTile.splitk_axes`

| File | Lines | Use | Disposition |
|---|---|---|---|
| `passes/lowering/tile/001_coordination.py` | 7, 14, 72-76, 106, 113-114 | Trigger for atomic-Write rewrite | Deleted in M4 |
| `passes/lowering/tile/000_partition_planner.py` | 306-307, 352-353 | Producer: planner sets the tag | Stripped in M6 |
| `passes/lowering/kernel/008_materialize_tile.py` | 169 | Pass-through during materialize | Replaced by helper in M3 |
| `passes/lowering/tile/_helpers.py` | 75, 89 | Pass-through during tile-rewrite helpers | Stripped in M6 |
| `ir/tile/passes.py` | 108, 112-113 | Axis-rename propagation | Stripped in M6 |
| `ir/tile/ir.py` | 517-531 | Dataclass field + pretty render + with_bodies | Stripped in M6 |

### Readers of `ThreadTile.cooperative_axes`

| File | Lines | Use | Disposition |
|---|---|---|---|
| `passes/lowering/tile/001_coordination.py` | 8, 19, 64, 95-102 | Trigger for Combine emission + Cond-guard | Deleted in M4 |
| `passes/lowering/tile/002_stage_inputs.py` | 158, 270 | `is_cooperative` flag for stage candidate walk | **See below** |
| `passes/lowering/tile/000_partition_planner.py` | 308-309, 355-356 | Producer: planner sets the tag | Stripped in M6 |
| `passes/lowering/kernel/008_materialize_tile.py` | 505 | Pass-through during materialize | Replaced by helper in M3 |
| `passes/lowering/tile/_helpers.py` | 75, 80, 86 | Pass-through during tile-rewrite helpers | Stripped in M6 |
| `ir/tile/passes.py` | 125, 127 | Axis-rename propagation | Stripped in M6 |
| `ir/tile/ir.py` | 543-562 | Dataclass field + pretty render + with_bodies | Stripped in M6 |

**`002_stage_inputs.py` is the one non-trivial reader.** It uses `is_cooperative = bool(tt.cooperative_axes)`
to gate which `_collect_candidates` branch runs. The cooperative case takes a different walk that descends
into `serial_outer` SerialTiles to find loads. M3 needs to replace this with the helper:
`is_cooperative = any(accum_escapes_thread_axis(tile_op).values())`. Same boolean, different source. Verify
in M3 that the staging output is byte-identical against a few cooperative kernels.

### Readers of `Write.reduce_op`

| File | Lines | Use | Disposition |
|---|---|---|---|
| `ir/stmt/leaves.py` | 587-680 | Dataclass field + validation + pretty render | Stripped in M7 |
| `ir/stmt/passes.py` | 107, 172 | Field propagation during stmt rewrites | Stripped in M7 |
| `passes/lowering/tile/001_coordination.py` | 221, 239, 244, 260 | Producer: coordination sets the field | Deleted in M4 |
| `passes/lowering/kernel/002_stamp_types.py` | 198 | Field pass-through during dtype stamping | Stripped in M7 |
| `passes/lowering/kernel/007_vectorize_stores.py` | 17, 134-135, 191 | "Don't vectorize atomic writes" guard | Replaced by helper in M7 |
| `passes/lowering/cuda/001_lower_kernelop.py` | 65 | Atomic-Write codegen at CUDA boundary | Replaced by helper in M7 |
| `ir/stmt/normalize.py` | 914 | Documentation mention in op-equivalence rules | Update to remove reference |

### Readers of `Combine` Stmt

| File | Lines | Use | Disposition |
|---|---|---|---|
| `ir/tile/ir.py` | 149-174, 1228 | Class definition + pretty render + export | Stripped in M5 |
| `graph.py` | 260, 307 | Stmt registry for deserialization | Stripped in M5 |
| `passes/lowering/tile/_helpers.py` | 137 | Mentioned in leaf-stmt list comment | Update comment |
| `passes/lowering/tile/001_coordination.py` | 9, 19-20, 39, 64, 89, 130, 139, 298-340 | Producer: coordination emits Combine | Deleted in M4 |
| `passes/lowering/tile/000_partition_planner.py` | 14, 45, 58, 477, 538 | Mentioned in docstrings (not code) | Update docstrings |
| `passes/lowering/kernel/008_materialize_tile.py` | 5 | Docstring mention of "Combine becomes smem tree-halve" | Update to "cooperative reduce inferred from body becomes ..." |
| `ir/stmt/passes.py` | 157, 210 | Pass-through in stmt rewriter | Stripped in M5 |

### Tests that exercise the targeted fields

- `tests/compiler/passes/test_decompose_rules.py::test_rms_norm_decomposes` — exercises a cooperative-K
  RMSNorm end-to-end. Will keep working through M3 (helper inferences the same fact); M4 deletes the pass
  but the test compiles the same kernel so coverage holds.
- Test files for `001_coordination` itself: search for any direct unit tests; convert to test
  `escape_analysis.py` instead.
- `tests/perf/` rmsnorm cases: smoke-bench in M8 to confirm no SASS-level perf regression.

## Risks

1. **Helper false negative on Accum escape.** If the analysis misses a case where an Accum escapes through a
   thread axis, the materializer won't emit the combine and the kernel produces wrong results silently. M2
   (parallel-run mode with assertion against the existing tags) is the safety net — full test suite must
   pass with the assertion live.
2. **Performance regression from `is_cooperative` reclassification in `002_stage_inputs`.** If the helper
   declares cooperativity earlier/later than the tag did, staging might pick a different set of buffers.
   M3 includes a kernel-output diff against the cooperative kernel zoo as a checkpoint.
3. **Hidden coupling.** Some pass we haven't audited may rely on the tag's presence (e.g. as a proxy for
   "this is a reduce kernel"). M2's parallel-run + full test suite is the catch-all; if any test fails
   between M2 and M3, that's the signal.
4. **Documentation churn.** Six ARCHITECTURE.md files mention these fields or the coordination pass. The
   M8 doc sweep is non-trivial; budget time for it.

## What stays

- All four tile flavors (`GridTile` / `ThreadTile` / `RegisterTile` / `SerialTile`) — the *type* of tile
  still encodes block-vs-thread-vs-register binding. We're only deleting the per-tile metadata.
- `Accum` / `Write` ops themselves — unchanged shape, just `Write.reduce_op` field removed.
- The planner's geometry decisions — `Role.SPLITK_BLOCK` and `Role.COOPERATIVE_STRIDE` stay as planner-internal
  labels driving axis-extent and tile-flavor choices. They just don't reach the IR anymore.
- All CUDA codegen output — `atomicAdd`, warp-shuffle reduction, smem tree-halve, `if (t == 0)` guards all
  emit the same source. The encoding before codegen is what changes.
