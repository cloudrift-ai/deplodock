# Structural forks in the two-level search

**Status:** Steps 1–3 (milestones M1–M3) **landed** 2026-06-10 on `feature/structural-forks-two-level-search`;
Step 4 (the `SPLITK` hoist, M4) remains — see the open questions noted under its milestone. Two structural-fork
emitters exist in lowering: `tile/005_split_demoted` (the demoted-matmul / SDPA-prologue split, PRs #219/#220 — now
**first-class**: outer-branched, per-kernel slices, greedy-priced) and `tile/017_atomic_free_splitk` plus the
planned Stream-K combine (`plans/atomic-free-streamk.md`) — both still on the tolerated slice-sum path until M4.
This plan makes kernel-set-changing lowering decisions first-class: each split kernel gets its own inner slice,
keep-vs-split becomes an outer-terminal comparison, and — the piece `SPLIT_CONE` deployment was blocked on
(`plans/qwen3-embedding-layer0-tune-findings.md` finding 5, follow-up 2) — greedy `compile` / `run` price and pick
a structural option unpinned.

## Problem

The two-level tuner (`pipeline/search/two_level.py`) rests on one claim: **op-variant forks are separable** (every
multi-option fork is an in-place `Op` rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`),
while **fusion forks change which ops exist** and are not. It draws the outer/inner boundary at a *fixed pass
index*: outer = `OUTER_PASSES` (`frontend` + `loop`), inner = `LOWERING_PASSES` (`tile → kernel → cuda`), terminal
= the cursor reaching `partition_loops`. The bet is that *every* kernel-set-changing decision happens in
`OUTER_PASSES`. Two lowering rules now break that bet:

- `005_split_demoted` offers `keep-fused Op` vs `OptionFork(Graph)` — a two-or-three-kernel split fragment — at the
  head of `lowering/tile`, before partition.
- `017_atomic_free_splitk` / Stream-K splice a combine kernel as a `Graph` fragment *below* partition, triggered by
  the inner `SPLITK` knob.

### What actually happens today (tolerated, not graceful)

The engine already digests the splice mechanically. The spawn site is `Run.drive` (`pipeline.py:682-700`): the raw
`options` list is concrete there, `LazyCandidate.from_option` (`candidate.py:235`) lifts each `Op`/`Graph` into a
leaf fork, and apply dispatches on the option type (`candidate.py:163` / `:196`). Inside the fused op's single-node
slice the inner MCTS explores the `Graph` branch as ordinary siblings, and `_bench_terminal` benches every `CudaOp`
in the slice — writing **each split kernel its own** `perf` / `lowering` rows under its own `op_cache_key` (an
earlier revision of this plan claimed the spliced kernels get no rows of their own; per-CudaOp attribution has been
correct for a while — the Qwen3 layer-0 tune left 339 `SPLIT_CONE: true` rows that dedup and transfer normally).
The terminal reward is the whole-slice sum, so keep-vs-split is *rewarded* correctly, and `inner_reward` records
the best whole-slice total as a bookkeeping row under the **fused** op's key (`two_level.py:247`).

What is still wrong, all measured on the Qwen3-Embedding-0.6B layer-0 tune (finding 5):

1. **Cross-product starvation.** The split's kernels share ONE slice and ONE patience: the QK^T split explores
   xna × xnb × consumer fork trees in a single `TuningSearch` — the exact starvation the two-level split was built
   to avoid (`project_mcts_exploration_limit`). The clean re-tune with splits in the space ran 1243 s vs the 859 s
   pre-split baseline.
2. **No first-class search identity.** Split kernels have DB-row identity but no own slice, effort accounting, or
   progress leaf; a structurally identical producer repeated across 28 layers re-walks inside every parent slice
   (benches are DB-cache-served, the tree walks are not).
3. **Keep-vs-split is invisible to the outer.** It is an inner sibling decision; the outer MCTS cannot compare
   "fused layer" vs "split layer" as terminals, which is the natural cost model for a kernel-set decision.
4. **No deploy path.** `policy/greedy._is_structural` (`greedy.py:56`, filter at `:134`) drops `Graph` leaves
   because `prior.mean_scores` prices ONE kernel's knob row — a multi-kernel option has no row to price. So
   unpinned `run` / `compile` keep the fused kernels (179 µs vs 138 µs pinned on layer 0), and the final greedy
   assembly in `run_two_level_tune` (`two_level.py:361`) has the same blindness — the tune can report a Σ-per-op
   best that *includes* split totals (via the bookkeeping row) which the assembled graph cannot realize. The
   `tune --bench` full-model table inherits the gap.

### The `SPLITK` culprit — 017 only, not 005

For 017 / Stream-K the kernel set is a function of an **inner** knob: `SPLITK` is simultaneously an op-variant tile
knob (the innermost level of partition's `BR → (BM,BN) → (FM,FN) → (BK,SPLITK)` fork tree) **and** a structural
trigger (`SPLITK > 1` requires a combine kernel to exist). That is why "move the combine emission to a loop pass"
fails: `SPLITK` does not exist until partition runs. The fix (Step 4) is to make it exist earlier.

`005_split_demoted` has **no such knot**: it sits at a fixed position (head of `lowering/tile`, before partition)
and its offer depends only on the fused body's shape, not on any inner knob. So the boundary redraw (Step 2) can
land before the `SPLITK` hoist and immediately serves the `SPLIT_CONE` family; 017 stays on the tolerated slice-sum
path until Step 4 brings it under the same roof.

## Design

Four steps, landable in order; each is inert or backward-compatible until the next reads it.

### Step 1 — classify forks by effect (via option type)

The Op/Graph return-type split **is** the classification — it is already the axis the engine dispatches on. No
enum, no field on `Fork` (a parallel field would just have to track the type and drift):

- **`Op` rebind** → in-place, kernel set unchanged → **op-variant** (inner / separable).
- **`Graph` splice** → structure changes → **structural** (outer / non-separable).

Compute it once at the fork-spawn site in `Run.drive` (`pipeline.py:682-700`), where the raw `options` list is
concrete — no thunk fired:

```python
structural = any(isinstance(o, Graph) for o in options)
search.push(*forks, parent=token, structural=structural)
```

**Why the spawn site, not a `Fork` field.** A pending leaf `Fork` wraps `lambda: [graph]` vs `lambda: [op]`; a
consumer holding only the `Fork` would have to fire `expand()` to see the type — cheap for a splice leaf but
**expensive for partition leaves** (`expand()` runs `_build_split_body` + full body normalization, the very thing
`LazyCandidate.score` is careful never to trigger). Classifying at push time avoids it and matches the natural
decision point. The only thing untypable from the raw list is a hypothetical *branch* `Fork` whose leaves are
Graphs; today the sole branch-Fork emitter is the partition planner (all `TileOp` leaves → op-variant), so
`isinstance(o, Fork) → op-variant` holds. YAGNI until a structural hierarchical fork exists.

**Correctness.** Within lowering it is exact: the only Graph-returning rules are the splice family (005, 017, the
Stream-K combine); every other tile/kernel/cuda rule returns an `Op` rebind. Frontend `decomposition` /
`compose_indexmaps` also return Graphs — also structural, also correct, and harmless because the outer already owns
every frontend/loop fork.

### Step 2 — redraw the two-level boundary on the effect, not the pass index

Replace the fixed `OUTER_PASSES` / `LOWERING_PASSES` boundary with an effect filter. The cursor is already
rule-indexed (`Cursor.rule_idx`), so no pass-list re-slicing is needed:

- The **outer driver** keeps driving past fusion, through the pre-partition tile rules. At a **structural** fork
  (Step 1's flag) it branches — each side is a distinct outer subtree. At an op-variant fork it would collapse to
  the greedy child; today no op-variant fork exists before partition, so in practice the outer just absorbs 005's
  offer sites.
- The **outer terminal** becomes "cursor reached `partition_loops` AND no structural fork pending". Each terminal
  is a fused graph whose kernel set is final: split producers/consumers are real `LoopOp` nodes, picked up by
  `inner_reward._kernel_nodes` like any kernel — own slice, own patience, own progress leaf, deduped by
  `op_cache_key` across layers and terminals (28 identical P@V splits = one inner search, 28 DB-served positions).
- **017 is unaffected**: its splice fires below partition, inside the inner slice runs, and stays on the tolerated
  slice-sum path (the inner must keep accepting `Graph` splices) until Step 4 hoists its decision above partition.

**Outer breadth.** Branching per offer *site* is `2^sites` (a 28-layer model has ~3 sites × 28 layers). The
decision is really per unique kernel, and the stamps are already deterministic per offer site — so memoize the
branch choice per `(rule, op_cache_key)` within a trajectory: identical sites take the same side, and the outer
tree stays linear in *unique* kernels. MCTS sampling plus DB-cached per-op rewards make each extra terminal cheap
(a terminal whose ops are all known is a pure DB read), but the memo is what keeps the tree honest.

The Σ bookkeeping row under the fused op's key (`two_level.py:247`) becomes obsolete for outer-branched splits —
each branch's Σ is computed by the outer from true per-kernel bests; keep writing it only for inner-tolerated
splices (017) until Step 4.

### Step 3 — greedy pricing for structural options (the deploy path)

`policy/greedy` stops filtering structural leaves and prices them. At a fork whose leaves include a `Graph` option:

- **Fused side**: the prior's predicted-best over the op's own enumeration — the score greedy would assign at the
  op's partition fork anyway.
- **Split side**: `Σ` over the fragment's kernels of the prior's predicted-best, obtained by a **nested greedy
  descent** over the fragment (`Pipeline.build(LOWERING_PASSES)` with `GreedySearch`, no backend, CPU-only),
  recording the chosen leaf's predicted µs at each kernel's partition fork. Memoize per `op_cache_key`.
- Pick the smaller predicted Σ; continue the main descent down that branch.

Guards and consequences:

- **Cold start keeps today's behavior.** Price structural options only when the *trained* `CatBoostPrior` is
  loaded; with only the `AnalyticPrior` (or nothing), keep option-0 — the keep-fused emission order 005 documents
  as the greedy/no-prior fallback. Σ-comparisons through the analytic model are unvalidated; don't let a cold
  compile change kernel sets. `DEPLODOCK_SPLIT_CONE=1/0` stays the manual override either way.
- **Rejected alternative: DB replay.** The Σ bookkeeping row (or per-kernel DB bests) could price the decision from
  measurements, but greedy is prior-only by design — the learned-prior work deliberately removed `_best_fork` DB
  replay. Don't reintroduce it through the back door.
- **Blocklist interplay.** `Pipeline.run`'s validate-rejection retry (`pipeline.py:485-505`) blocklists a failed
  node's tile identity and re-drives. A structural pick introduces fresh node ids; a kernel of the chosen fragment
  failing to lower must surface as a rejection that re-drives down the *other* branch (treat the structural choice
  itself as blockable). Risk item — needs an explicit test.
- This step automatically fixes the assembly inconsistency: `run_two_level_tune`'s final
  `Pipeline.build(CUDA_PASSES).run(...)` and `tune --bench`'s full-model table go through the same greedy, so the
  reported Σ and the assembled graph agree again.

### Step 4 — promote `SPLITK` to a structural (pre-partition) knob

Hoist `SPLITK` out of partition's innermost `(BK, SPLITK)` fork level into a knob the **outer** search branches on,
pinned (via the existing `Knob.narrow` pin path) before partition runs. With `SPLITK = S` known pre-partition:

- The matmul stays a `LoopOp` tagged `SPLITK = S`; the rest of partition (`BM/BN/FM/FN/BK`) remains an op-variant
  inner fork tuned in the matmul's slice with `SPLITK` pinned.
- The combine-emission (017 / Stream-K) is re-expressed as a **pre-partition** `Graph` fork gated on the
  `SPLITK`/strategy pin: "given a matmul `LoopOp` with `SPLITK = S`, produce `{matmul-LoopOp → workspace,
  combine-LoopOp → out}`". The combine becomes an ordinary post-fusion `LoopOp` — and Step 2's boundary picks the
  fork up with no further engine work: own slice, outer-terminal strategy comparison (atomic / atomic-free /
  Stream-K), dedup. The inner's slice-sum tolerance for `Graph` splices can then be retired.

## Milestones (single branch, commit after each `make test` passes)

1. **M1 — classify + plumb (Step 1).** ✅ landed. One-liner at the spawn site; `structural=` threaded through
   `Search.push` (default `False`, accepted-and-ignored by `GreedySearch`/`TuningSearch`). Tests: 005's and 017's
   fork points read `structural=True`; partition leaves read `False` (`test_structural_push.py`).
2. **M2 — boundary redraw (Step 2).** ✅ landed. Outer branches on pre-partition structural forks
   (`two_level.outer_pipeline()`) with the per-`(rule, op_cache_key)` decision memo (`Candidate.structural_memo`,
   replayed inline by `Run.drive`); outer terminal = partition cursor + no structural fork pending; inner keeps
   tolerating sub-partition splices (017). Verified live on the norm→linear 32×1024×3072 shape: 2 outer terminals
   (fused Σ 238.8 µs vs split Σ 20.3 µs), the `_xn` producer tuned as its own op leaf. The full Qwen3-Embedding
   layer-0 wall-time acceptance (recovery toward the 859 s pre-split baseline) still wants a dedicated re-tune.
3. **M3 — greedy pricing (Step 3).** ✅ landed. `GreedySearch._pick_structural` prices both sides via nested
   per-kernel descents (tile-only, CPU, memoized per `op_cache_key`), gated on the *trained* prior; cold compile
   keeps today's fused kernel sets (`test_greedy_compile_keeps_fused_kernel` green); a lowering-rejected structural
   pick retires structural picks on `Pipeline.run`'s retry and re-drives keep-fused
   (`test_greedy_structural_pick_falls_back_on_lowering_failure`). Verified live: after the tune above, unpinned
   `run --code` deploys the two-kernel split.
4. **M4 — `SPLITK` hoist (Step 4).** Remaining. Re-express 017's combine emission pre-partition; validate accuracy
   parity with today's atomic / atomic-free paths (`test_atomic_free_splitk` + new e2e coverage); confirm the outer
   enumerates one terminal per combine strategy and Σ-per-op ranks them. Retire the inner slice-sum tolerance + the
   fused-key bookkeeping row. Update `pipeline/ARCHITECTURE.md` (two-level section) + `two_level.py` module
   docstring. Open design questions found while scoping: (a) the per-op `SPLITK` pin — `Knob.narrow` reads
   process-global env pins, which can't express a per-branch pin; partition's `_plan_kernel` must learn to clip
   `splitk_choices` from `op.knobs`; (b) hoisted `SPLITK` candidates must stay consistent with partition's
   BR/BK-coupled divisor checks (else a hoisted branch can enumerate zero rows and strand the node), which wants
   the candidate derivation extracted from `_plan_kernel`; (c) the loop-dialect K-split for the atomic-free
   fragment needs the linear-residual (`matmul_add` / 015) interplay settled — restrict the pre-partition offer to
   plain-epilogue matmuls first, or move the residual add into the combine; (d) retiring 017 must not strand the
   in-flight Stream-K B4 branch (`plans/atomic-free-streamk.md`), which reuses 017's fragment shape.

## Verification

- `./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup` green at each milestone.
- M1: focused unit test on the spawn-site predicate (no GPU).
- M2: `test_tune_explores_fused_and_split_terminals` upgraded — both kinds appear as *outer* terminals; per-op
  progress denominators count split kernels.
- M3: end-to-end on the finding-5 shapes — the per-kernel table from `run --bench` matches the pinned-split table
  within noise; `eval prior --dataset db` reachability covers the new `_xn` producer shapes.
- M4: `deplodock tune --code "torch.matmul(...)" -v` shows the combine kernel as its own `[tune]` op leaf; outer
  reports > 1 terminal once atomic / atomic-free are both structural options.
- `make test` + `make lint` green.

## Relationship to other plans

- **`plans/qwen3-embedding-layer0-tune-findings.md` finding 5** — follow-up 2 (greedy structural deploy) is
  Steps 1–3 of this plan; follow-up 1 (Select-leaf fragment fold) and follow-up 3 (prior pick-reachability on the
  `_xn` shapes) are orthogonal to it.
- **`plans/atomic-free-streamk.md`** — ships with a fixed pre-tiled `TileOp` combine + per-`CudaOp` perf
  attribution and does NOT need this refactor to land B4. This plan is the general mechanism that makes *any*
  tile-pass-created kernel a first-class tuned op and any combine strategy an outer terminal.
- The outer tree remains the clean insertion point for a future fusion search (`two_level.py` docstring) — Step 2's
  effect filter is exactly the hook a multi-option fusion fork would use.
