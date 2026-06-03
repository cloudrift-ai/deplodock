# Structural forks in the two-level search

**Branch:** `feature/structural-forks-two-level` (not yet created)
**Status:** planned — design only. Enables tile-lowering passes (`017_atomic_free_splitk`, the Stream-K combine in
`plans/atomic-free-streamk.md`) to create *new kernels* that get first-class per-op tuning, and to expose the
combine-strategy choice as an outer terminal the fusion MCTS compares.

## Problem

The two-level tuner (`pipeline/search/two_level.py`) rests on one claim: **op-variant forks are separable** (every
multi-option fork is an in-place `Op` rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`), while
**fusion forks change which ops exist** and are not. It draws the outer/inner boundary at a *fixed pass index*: outer =
`OUTER_PASSES` (`frontend` + `loop`), inner = `LOWERING_PASSES` (`tile → kernel → cuda`), terminal = the cursor reaching
`partition_loops`. The bet is that *every* kernel-set-changing decision happens in `OUTER_PASSES`.

`017_atomic_free_splitk` and the Stream-K combine break that bet: a **lowering** pass makes a kernel-set-changing
decision (splice a combine kernel as a `Graph` fragment). Today this is tolerated by a hack — the spliced combine lands
*inside the matmul's single-node slice*, and `inner_reward` sums every `CudaOp` in the slice (`two_level.py:205-212`),
so the combine's cost is folded into the matmul's `op_cache_key` row. Consequences:

- The combine is **not a first-class op**: no own `op_cache_key` row, no own slice, no effort gating, not deduped or
  shared across the graph.
- If the combine ever **forks** (e.g. emitted as a `LoopOp` so `partition_loops` tiles it), its forks **cross-product**
  with the matmul's under one slice patience — the exact starvation `two_level.py` was built to avoid
  (`project_mcts_exploration_limit`).
- The combine-**strategy** choice (atomic / atomic-free / Stream-K) is a kernel-set-changing fork but lives *below* the
  outer/inner boundary, so the outer MCTS can't compare strategies as terminals.

### The real culprit is `SPLITK`, not 017

The kernel set is a function of an **inner** knob. `SPLITK` is simultaneously an op-variant tile knob (the innermost
level of partition's `BR → (BM,BN) → (FM,FN) → (BK,SPLITK)` fork tree) **and** a structural trigger (`SPLITK > 1`
requires a combine kernel to exist). 017 / Stream-K are just where that latent violation materializes into nodes. This
is why "move the combine emission to a loop pass" fails: `SPLITK` does not exist until partition runs. The fix is to make
`SPLITK` exist earlier.

## Design

Three steps, landable in order. Step 1 is inert until a driver reads it, so it merges standalone.

### Step 1 — classify forks by effect (via option type)

The Op/Graph return-type split **is** the classification — it is already the axis the engine dispatches on (`apply`
branches `isinstance(option, Op)` vs `Graph` at `candidate.py:171-178`; `from_option` routes `from_op` / `from_graph` at
`candidate.py:263`). No enum, no field on `Fork` (a parallel field would just have to track the type and drift):

- **`Op` rebind** → in-place, kernel set unchanged → **op-variant** (inner / separable).
- **`Graph` splice** → structure changes → **structural** (outer / non-separable).

Compute it once at the fork-spawn site in `Pipeline.search` (`pipeline.py:496-516`), where the raw `options` list is
concrete — no thunk fired:

```python
structural = any(isinstance(o, Graph) for o in options)
search.push(forks[0], *forks[1:], best=best, structural=structural)
```

**Why the spawn site, not a `Fork` field.** A pending leaf `Fork` wraps `lambda: [graph]` vs `lambda: [op]`; a consumer
holding only the `Fork` would have to fire `expand()` to see the type — cheap for 017's leaf but **expensive for
partition leaves** (`expand()` runs `_build_split_body` + full body normalization, the very thing `LazyCandidate.score`
at `candidate.py:329` is careful never to trigger). Classifying at push time from `options` avoids it and matches the
natural decision point ("should this fork point spawn outer siblings at all"). The only thing untypable from the raw
list is a hypothetical *branch* `Fork` whose leaves are Graphs; today the sole branch-Fork emitter is the partition
planner (all `TileOp` leaves → op-variant), so `isinstance(o, Fork) → op-variant` holds. YAGNI until a structural
hierarchical fork exists.

**Correctness.** Within lowering it is exact: the only Graph-returning rule is the splice family (017, the Stream-K
combine); every other tile/kernel/cuda rule returns an `Op` rebind. Frontend `decomposition` / `compose_indexmaps` also
return Graphs — also structural, also correct, and harmless because the outer already owns every frontend/loop fork.

### Step 2 — promote `SPLITK` to a structural (pre-partition) knob

Hoist `SPLITK` out of partition's innermost `(BK, SPLITK)` fork level into a knob the **outer** search branches on,
pinned (via the existing `Knob.narrow` pin path) before partition runs. With `SPLITK = S` known pre-partition:

- The matmul stays a `LoopOp` tagged `SPLITK = S`; the rest of partition (`BM/BN/FM/FN/BK`) remains an op-variant inner
  fork tuned in the matmul's slice with `SPLITK` pinned.
- The combine-emission (017 / Stream-K) is re-expressed as a **loop-dialect** `Graph` fork gated on the `SPLITK`/strategy
  pin: "given a matmul `LoopOp` with `SPLITK = S`, produce `{matmul-LoopOp → workspace, combine-LoopOp → out}`". The
  combine is now an ordinary post-fusion `LoopOp` — sliceable, partition-tunable, deduped — and is safe as a `LoopOp`
  (the last-turn objection, cross-product under the matmul's patience, is gone because it gets its own slice).

This is the load-bearing change; it removes the `SPLITK`-is-both-inner-and-structural contradiction.

### Step 3 — redraw the two-level boundary on the effect, not the pass index

Replace the fixed `OUTER_PASSES` / `LOWERING_PASSES` slice with an effect filter:

- **Outer driver** runs the lowering passes too, but only **branches** on structural forks
  (`structural=True` from Step 1); at op-variant forks it collapses to the greedy / DB-best child (defers to inner).
  Each structural fork (a combine strategy) becomes a distinct **outer terminal**.
- **Inner terminal predicate** becomes "no structural fork pending" instead of "cursor reached `partition_loops`". Each
  spliced combine `LoopOp` is a real node in the outer terminal graph, picked up by `inner_reward._kernel_nodes`
  (`two_level.py:120-124`) + `single_node_graph` like any kernel — first-class slice, dedup, effort gating, transfer.

## Milestones (single branch, commit after each `make test` passes)

1. **S1 — classify + plumb.** One-line `structural = any(isinstance(o, Graph) for o in options)` at the spawn site;
   thread `structural=` through `Search.push` (default `False`, ignored by `GreedySearch`/`TuningSearch` for now). Inert.
   Tests: 017's fork point reads `structural=True`; a partition leaf / `020_stage_inputs` / `070_pad_smem` rebind reads
   `False`.
2. **S2 — `SPLITK` as a structural knob.** Hoist `SPLITK` to a pre-partition pin; re-express 017's combine-emission as a
   `SPLITK`-gated loop-dialect `Graph` fork producing a combine `LoopOp`. Validate accuracy parity with today's atomic /
   atomic-free paths (`tests/compiler/e2e/test_streamk_matmul.py`, `test_atomic_free_splitk`).
3. **S3 — boundary redraw.** Outer branches only on structural forks; inner terminal = no structural fork pending. The
   combine appears as its own tuned op (own `perf` / `op_effort` rows). Confirm the outer enumerates one terminal per
   combine strategy and the Σ-per-op reward ranks them. Update `pipeline/ARCHITECTURE.md` (two-level section) +
   `two_level.py` module docstring.

## Verification

- `./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup` green at each milestone.
- S1: a focused unit test on the spawn-site predicate (no GPU).
- S2/S3: `deplodock tune --code "torch.matmul(...)" -v` shows the combine kernel as its own `[tune]` op leaf with its own
  `op_cache_key`; the outer reports `> 1` terminal once atomic / atomic-free are both structural options.
- `make test` + `make lint` green.

## Relationship to `plans/atomic-free-streamk.md`

The Stream-K plan ships with a **fixed pre-tiled `TileOp` combine** + (cheap) per-`CudaOp` perf attribution — it does
NOT need this refactor to land B4. This plan is the general mechanism that makes *any* tile-pass-created kernel a
first-class tuned op and any combine strategy an outer terminal; do it for its own sake afterwards. It also cleanly
hosts a future fusion search (`two_level.py:24` already calls the outer tree "the clean insertion point for fusion search
when those forks exist").
