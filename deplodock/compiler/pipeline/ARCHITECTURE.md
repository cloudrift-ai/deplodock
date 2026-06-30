# Pipeline Architecture

A pattern-based rewrite engine over a frozen pass layout, plus the autotune search and dump hooks. This file covers the
engine and the search; the pass-authoring invariants live in [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md), and what
each IR dialect looks like is in `ir/ARCHITECTURE.md`.

## Modules

The engine core is `pipeline.py` (`Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` for the frozen layout, plus `Run` —
the per-run state and engine loop) and `fork.py` (the `Fork` interface with `OptionFork` / `ThunkFork`, and the reusable
`Level` + `build_fork_tree` lazy knob-cartesian tree builder). `knob.py` owns the `Knob` descriptor system and the
`DEPLODOCK_<KNOB>` env namespace (it borrows `config.knob_var` / `config.knob_raw`; `format_tuning_knobs` renders the
real tuning knobs for `tune` output). `dump.py` and `rule_diff.py` are the dump / `-vv` presentation layers.

The autotune state lives under `search/`. The persistent store is `SearchDB` (`db.py`, SQLite). The in-memory MCTS lives
with its only reader, `TuningSearch`, in `policy/mcts.py` (greedy compiles use `policy/greedy.greedy_decide` instead, no
tree). `two_level.py` is the two-level tuner (outer structural MCTS, inner per-op reward). The ONE ranking path is
`prior/`: a `Prior` ABC with the cold `AnalyticPrior` and the learned `CatBoostPrior` composed behind `FallbackPrior`
(`load_prior`). `data/` is the harmonized read-view over the three data sources (golden configs / DB `perf` rows / prior
reservoir) — `Sample`, `Dataset`, and `ShapeKey` (the single golden↔measured join key). `golden.py` holds
`GoldenConfig` and its matmul / reduce / pointwise subclasses (the `AnalyticPrior`'s ground truth). `keys.py` defines
`op_cache_key` / `dialect_of` / `source_chain`; `slice.py` isolates one finalized kernel into a standalone graph;
`diagnostics.py` (under `prior/`) backs the `eval` reachability / calibration reports.

The passes themselves are `passes/{frontend,loop,lowering}/`. Each pass directory's rules and invariants are documented
in [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the per-pass overview table is near the end of this file.

## Engine

### Chain matcher

A `Pattern(name, op_type, constraints={})` matches one node by op type plus optional `node.op` field equality. A pattern
list matches a chain: the seed matches `pattern[0]`, its sole consumer matches `pattern[1]`, and so on. Multi-node
patterns only fire when each intermediate node has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered seed; overlaps between matches are allowed (the
rewriter exits after the first successful rewrite per iteration, so overlap is just candidate enumeration). `Match.nodes`
maps each pattern entry's name to the matched `Node`. `Match.consumed` and `Match.output` are overridable by the rewrite
function to control which nodes the splicer removes and which node's edges get rewired.

### Rule module convention

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]   # required
def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]:
    ...
```

The dispatcher binds parameters by name. Reserved names: `graph`, `match`, `root`, `out`, `ctx`. Pattern names from
`PATTERN` bind to matched `Node` objects. Anything else binds positionally to `root.inputs[i]`. Take only what you need —
`ctx` is optional. Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as rules — they're shared helpers.

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of `match.output` (defaults to `match.root_node_id`);
  fragment `InputOp` nodes reference existing graph nodes by id, non-Input nodes get fresh ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op` directly, preserving the node id, inputs list,
  output Tensor, and hints. Used by the lowering rules because `KernelOp.arg_order` / `CudaOp.arg_order` embed the
  original node id as the output buffer name, so a fresh id would break the generated kernel's buffer binding.
- **List = autotune fork.** A rule unsure which parameter to use returns the alternatives as a list, in any order — the
  engine spawns one `LazyCandidate` per option (sharing the parent's graph snapshot) and hands them ALL to a `Search`
  policy, which ranks them via a `Prior` (see below). Single-option returns (or a bare `Graph` / `Op`) are the
  deterministic case — no fork.

Raise `RuleSkipped(reason)` to decline a match — the engine logs the reason at DEBUG and moves on.

**Idempotence requirement.** Every rule MUST be idempotent on its own output. The engine re-runs the entire pipeline on
each popped candidate from pass 0; rules whose output is already in the graph must `RuleSkipped` or have a pattern that
no longer matches. Most rules satisfy this implicitly via op-type changes (`LoopOp` → `TileOp`); the rest have explicit
`raise RuleSkipped("already X")` guards.

### Forks and the one ranking path

Ranking is always the policy's job over a single `Prior` — Forks carry NO score, and nothing materializes or scores a
`TileOp` just to rank it (the per-variant `lazy_score` / `score_tile_geometry` formulas, the `Fork.score` /
`Search.score_of` plumbing, the DB-best `_best_fork` replay, and the `_priority_*` enumeration sort are all gone). The
`Prior` featurizes the row knobs directly (`knob.knob_features`). There is ONE ranking path: the hand-coded
`AnalyticPrior` cold (a real heuristic *score* over the engineered `D_*` geometry / occupancy features, not emission
order; a separate `_W_A_DYN` weight set ranks symbolic-axis masked-tile kernels, selected on the stamped
`S_ext_n_symbolic_axis`) and the learned `CatBoostPrior` once trained, composed behind `FallbackPrior`. `TuningSearch`
(`tune`) ranks the PUCT frontier; `greedy_decide` (`compile` / `run`, via `Run.resolve`) picks via `Prior.pick` —
measured -O3 reservoir evidence first (`evidence_pick`: the candidate prefix-consistent with the fastest `H_opt=3` row of
the same op), the `mean_score` argmin otherwise.

`FallbackPrior` splits its two surfaces once the learned half is fitted: `mean_score` / `mean_scores` / `pick` (deploy +
eval) are pure-learned + evidence, but `score` — the MCTS *selection* signal — tilts the learned µs by the analytic's
dimensionless ranking multiplier (`learned · analytic**W`, `W = config.analytic_tilt`, neutral 1.0), so PUCT still
explores a region the cold heuristic prices well but the data-poor learned model buries.

**Lazy hierarchical forks.** `Fork` (`fork.py`) is an interface: `knobs` (the knob delta the fork pins — the variant
identity the perf DB and prior key on, read without expanding), `is_leaf`, and `expand()` (the next level of options).
The search loop pops a Fork-pending `LazyCandidate`, invokes `expand()` to materialize the children, pushes them back,
and continues; only the subtrees the search walks into get materialized. `OptionFork` is a concrete `Op` / `Graph` leaf;
`ThunkFork` is a generic flat fork (`expand_fn(knobs)` a function of the fork's own delta, so siblings share one
function). Multi-level knob-cartesian forks reuse `build_fork_tree`: a rule supplies the per-level `Level`s + a
`materialize=` callable and gets back a lazy ROOT `_Branch` whose `expand()` builds children on demand in grouping order.

**Knob-stamp invariant.** Every emitted variant carries an explicit value for every declared knob — no realized leaf has
an absent knob. Each `Knob` declares an `off` value (its "unused / declined" sentinel), and the pipeline fills any of a
pass's declared knobs the variant left unspecified at the **pass boundary** (`Cursor.advance` → `_off_fill_pass`, via
`knob.apply_off_defaults`), covering a pass that acted, declined, was skipped, or returned no variants alike. Scoping the
fill to the just-finished pass avoids prematurely stamping a later pass's knob (which would trip that pass's idempotency
guard). The point: the learned prior NaN-fills absent feature columns, so with explicit OFF values NaN means *only*
"not-yet-decided" (a partial fork prefix during descent), distinct from "decided: unused" (an OFF value on a complete
leaf). A knob with no `off` (the `_UNSET` default — universal knobs like `BN` / `BM`, always set by their pass) is never
auto-filled. Tier discrimination is value-based throughout (`knob.is_warp` / `knob.mma_atom`). Verified by
`tests/compiler/passes/test_knob_stamp_invariant.py`.

**Validate / raise as fork pruning.** A rewrite that *returns* an op failing `Op.validate(ctx)` (e.g. a `KernelOp` whose
smem exceeds `ctx.max_dynamic_smem`) is filtered by `Candidate.try_rewrite` — correct as fork pruning (sibling branches
carry other tile shapes) but fatal in a single-path greedy compile, where it leaves the node un-lowered. `Pipeline.run`
installs a `rejections` sink on the `Run` recording each drop `(node, pass, reason)`; after the terminal settles,
`_raise_on_unlowered` raises a loud `LoweringError` naming any still-un-lowered node instead of leaking a cryptic
`non-CudaOp` `TypeError` to the backend. The sink is absent under `tune`, so the fork-pruning path stays silent. A
rewrite that *raises* mid-lowering (a deterministic pass hitting an un-representable shape) is the same dead end for an
exception: greedy `resolve` lets it propagate; under `tune`, `Run.drive` catches it per-candidate, drops that subtree,
and bumps `Run._dropped_candidates` — without this, one search-only un-lowerable fork aborted the whole tune.

### Tile lowering at the pipeline level

`lowering/tile/` lowers each fused `LoopOp` to a kernel-ready `TileOp` over the block-DAG Tile IR (`ir/tile/ir.py`):
`010_recognize` (lift `LoopOp` → `TileOp`, recognize flash / softmax carriers, normalize to `Monoid`s) → `020_schedule`
(map free axes to the grid, pick the reduce partition + output `TILE` fragment, and **atomize** — resolve the
algebra→hardware-atom binding structurally onto the schedule as each warp / cooperative option is built, so an unbindable
atom is rejected at fork construction; `_atomize.py`, see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md)) → `030_split`
(cross-CTA split-K as a graph rewrite). It **never dispatches on a named shape** — every decision is gated on the reduce axes' carrier algebra read
off the body (`MAP` / `SEMIRING` / `MONOID`; flash attention is the `MONOID` algebra on the streaming schedule, a twisted
monoid is a monoid, selected structurally), not on a matmul / pointwise / attention archetype. The full design lives in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md). Two interactions reach up to the pipeline level:

**Demoted-matmul split (`CUT`) as an outer structural fork.** `split/010_split_demoted` may un-fuse a demoted matmul (a
multiply operand reading a computed / K-folded cone that keeps the matmul off the warp tier) into an `xn`-producer +
clean-gemm-consumer `Graph` fragment — a kernel-set change. The two-level tuner owns the offer as an outer structural
fork: keep-vs-split branches the outer tree, each side's kernels are tuned in first-class per-op slices, and the
Σ-per-op terminal rewards compare the kernel sets; greedy deploys the split only via the *trained* prior's structural
pricing, never cold. The `op.knobs` `CUT` stamp is the considered-vs-declined idiom (`keys.py`): simultaneously the
rule's idempotence guard, the learned prior's training signal (absent = never offered → NaN-filled; `"0"` / `"1"` = the
decision), and the `op_cache_key` separation that keeps each decision state distinct in the search tree. The stamp is
deterministic per offer site, so identical kernels across graphs stamp identically and keep sharing perf rows.

**The reusable Fork-tree builder.** The tree-building algorithm (group params by per-level knob keys, collapse
single-key levels, skip empty-key levels, defer leaf materialization to `expand()`) lives in `fork.py` as `Level` +
`build_fork_tree`. Rules with multi-level knob-cartesian forks reuse it; one-shot flat forks stay inline as `ThunkFork`s.

### Drivers

`Pipeline.build(passes)` wraps a pass list; the result exposes the compile entry points, each driving one of the `Run`
engine loops (`drive` for exploration, `resolve` for deterministic resolution):

- `Pipeline.run(graph, *, backend=None, db=None) -> Graph` — single-shot greedy compile: a deterministic resolution
  (`Run.resolve`) with the greedy pick (`greedy_decide`), NOT a search — no frontier, no tree, no benching. The decide
  flattens each fork point to its complete leaves (`fork.flatten_leaves`) and picks the `Prior`'s `mean_scores` argmin —
  `AnalyticPrior` cold, `CatBoostPrior` once trained; option-0 (first leaf, emission order) only if the prior fails to
  load entirely. The graph is copied once per attempt and resolved in place — no per-fork copies. **Structural options
  are priced, never raw-scored**: with the trained prior loaded, `greedy_decide`'s `_pick_structural` prices each side
  (a nested `resolve` per kernel over a `lowering/tile`-only pipeline, the price being the `score` of its slice-resolve's
  partition-fork `Decision`, memoized per `op_cache_key`) and the cheaper kernel set wins, so an unpinned compile deploys
  the splits `tune` measured best. Cold, or when a side is unpriceable, the structural leaf is filtered — a cold compile
  never changes kernel sets. Retries are decide-wrappers over a deterministic re-resolve (every other choice replays
  identically — cheap non-chronological backtracking, no snapshots): a structural pick that leaves a fragment kernel
  un-lowered retires structural picks wholesale and re-resolves the keep-fused branch before falling back to tile
  blocklisting.
- `async Pipeline.tune_async(graph, *, search, backend=None, db=None)` — the (async-only) autotune sweep. Pass a
  `TuningSearch(patience=, ucb_c=)`; the async generator yields one terminal `Candidate` per fully-explored rollout, and
  `tune_async` benches each via `await _bench_terminal_async` (writes per-kernel `perf` / `lowering` / inventory rows,
  returns the aggregate `PerfStats`), then calls `search.observe(stats, status)`. With `backend=None` the bench is
  stubbed to `latency_us=1.0` and nothing is persisted, so a backend-less sweep never overwrites tuned rows.
- `Run.drive(graph) -> Iterator[(token, Candidate)]` — the exploration engine loop (`tune`). `Run` is the per-run state
  object (`pipeline` + `ctx` + `search` + `db` + `backend` + `dump` + `rejections`): `Pipeline` stays a frozen,
  shareable pass layout while every run-scoped sink lives on the Run, reached through the candidate (`cand.run.dump`,
  `cand.ctx`). `drive` seeds the root candidate, then per iteration pops a `LazyCandidate`, resolves it, runs one rule
  batch (`Run._step`, shared with `resolve`), and pushes successors under the pop's token. Selection is `TuningSearch`'s
  job (PUCT over the learned prior); the perf DB still *records* every bench as training data. Each fork push is
  classified by effect at the spawn site (where the raw option list is concrete): any `Graph`-splicing option (a
  kernel-set change) marks the push `structural=True`; an `Op` rebind is op-variant (`False`).
- `Run.resolve(graph, decide) -> (Graph, list[Decision])` — the deterministic-resolution counterpart. Both entry points
  share one rule-batch body (`Run._step`), but `resolve` is a fold, not a search: ONE live graph mutated in place (no
  sibling snapshots, no per-fork copies — the terminal IS the seeded graph), and at each undecided fork a `decide`
  callback gets a `ForkPoint` (the `Match`, the raw options as the rule emitted them, the pre-decision op, `ctx`) and
  returns the option to apply. The returned trace — one `Decision(rule_name, node_id, chosen_kind, knob_delta, score,
  n_options)` per decided fork — is the resolution's only process-state output: "did this compile take a structural
  pick", "what did the partition fork predict for this kernel" are trace queries, never accumulated policy attributes.

### The keying map: two identities

Everything the search stores or replays is keyed by one of TWO identities — when adding a cache or table, pick one;
don't invent a third:

- **Variant identity = `(context, knobs)`** — anything *predictive or replayable*. The `S_*` structural features
  (`loop/stamp` stamps a stmt/op histogram + loop extents + operand dtypes) make the merged knob dict a COMPLETE
  identity, so a prior is a pure function of it. The learned prior is exactly `score(features(ctx, knobs))`: the
  structural facts are already in the knob dict, so `knob.knob_features` turns it straight into the model feature vector
  (the `S_*` knobs pass through; tuning knobs encode by type, `MMA` expands to atom props).
- **Measurement identity = `(ctx.structural_key, op_cache_key)`** — ground truth about *materialized leaves*: `perf`
  rows (the per-variant replay cache), op inventory (`loop_op` / `tile_op` / `kernel_op` / `cuda_op`), and two-level
  dedup. The structural `child_key` on `lowering` rows is measurement linkage (it joins the inventory), NOT a replay key.

### Search persistence: on-disk inventory vs in-memory MCTS

The autotune state is split across two cooperating modules. **`SearchDB`** (`db.py`) is a SQLite store partitioned into
the four op-inventory tables (one row per op encountered along any lowering chain, keyed by `op_cache_key`), a `lowering`
edge table (one row per rewrite hop carrying the knob delta plus a best-median upsert — `best_per_op_time` walks the
chain to resolve a pre-final op's measured cost; loop→loop source hops are skipped as structural/decision hops), a
backend-partitioned `perf` table (full stats + `backend` + `status` + `knobs` + `captured`), and a `node` table — one
row per **search-tree node** (every partial branch + leaf of a per-kernel search), keyed by `digest(context_key, gpu,
op_sig, tunable-knob set)`, carrying the full feature dict the prior sees, a keep-the-minimum value-of-position latency,
a `parent_key` pointer, a `gpu` column, and depth bookkeeping (written by `record_nodes`). The `gpu` identity
(`Context.hardware_id`, the PCIe product name) is folded into the node key so a cross-hardware dataset never collides:
`context_key` (cc + opt) can't separate same-die SKUs (H100 vs H200 share cc + SM count), so without `gpu` their rows
would merge and keep-min would silently drop one card's data (the `H_total_mem` VRAM feature is what then lets the prior
model the difference). `node` and `perf` are content-keyed (parent-tree-independent) and survive a `_SCHEMA_VERSION`
bump; only the topology-keyed `lowering` table is dropped on mismatch.

**`SearchTree`** (`policy/mcts.py`) is pure-Python in-memory MCTS state, colocated with `TuningSearch` because MCTS is
the only policy that reads it. Each tree node wraps a `LazyCandidate` and carries `visits`, `best_reward` (max reward
over the subtree's measured leaves), and a `live` counter that filters out drained subtrees. Lineage is TOKEN-THREADED,
not call-order-dependent: `pop()` returns `(token, candidate)` (the token IS the `SearchNode`), the engine pushes
children with `parent=token` and observes the terminal with the same token, so the tree stays correct however the engine
interleaves pops / pushes / observes. It is rebuilt fresh each process; cached `perf` rows ensure no re-bench on warm
starts. Greedy compiles build no tree (they don't go through a `Search`).

`_bench_terminal_async` is the only path that knows about all four parts (graph, DB, tree-through-`search.observe`,
backend). It short-circuits when every `CudaOp` in the graph already has a `perf` row for the current `(context_key,
backend)`. Otherwise it does one `await backend.benchmark_async(...)`, walks `Op.source` once to record op inventory +
lowering edges + the `perf` row per kernel, and returns the aggregate `PerfStats` for the search to score.

## Tuning workflow

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the lowering pipeline with different
knob choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the lowest
measured latency.

### Two-level search: outer structural MCTS + inner separable per-op tuning

`deplodock tune` does **not** run one MCTS over the whole graph. The pipeline applies rules sequentially, so two kinds of
fork — **op-variant** forks (tile / pad / stage choices for one kernel) and **structural** forks (which kernels exist:
fusion grouping, the demoted-matmul split) — would nest and cross-product under one global patience, starving deep ops.
The two kinds have opposite structure, so `two_level.py` splits them on the fork's *effect* (the spawn-site `Op`-rebind /
`Graph`-splice classification):

- **Outer search** (`run_two_level_tune`) drives the graph-changing passes — `frontend` + `loop` plus the pre-partition
  head of `lowering/tile` (`010_split_demoted`'s keep-vs-split offer followed by the non-forking post-split re-fusion
  aliases). A **terminal** is the state where the cursor reaches `partition_loops` with every structural fork resolved.
  Each terminal is a candidate fused graph; its **reward** is `1 / Σ best-per-op time` from the inner search,
  backpropagated by the reused `TuningSearch`. Structurally identical offer sites within one trajectory take the same
  side: `Run.drive` replays the first decision read off the trajectory's own graph (`_replay_structural_decision`), so
  the outer tree stays linear in *unique* kernels instead of `2^sites`. Fusion itself is still deterministic (no rule
  emits a multi-option fusion fork), so a graph with no structural offers yields one terminal and this reduces to "tune
  each op once, sum, assemble". The global prior also drives the outer PUCT: each terminal emits one composed Σ row per
  structural decision it realized (features `{ctx, pre-decision op knobs, decision delta}`, label = the Σ of that side's
  per-kernel bests), so a warm re-tune descends the predicted-cheaper kernel set first.
- **Inner search** (`_inner_reward_async`) tunes each finalized kernel **independently** in its own single-node slice
  (`single_node_graph`, `slice.py`) with a plain `TuningSearch` over the lowering passes only (`tile → kernel → cuda`).
  The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic `InputOp`;
  the root op is shared **by reference**, so its body — and thus `op_cache_key` — is byte-for-byte the full-graph op's.
  Because the inner tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget —
  `Σ_k n_k` benches, never the product. **Leaves are deduped by `op_cache_key`**: 24 RMSNorm LoopOps across 24 layers
  collapse to one work unit, and the outer `total_us` accumulates `best * multiplicity` so the reward stays
  multiplicity-weighted. The progress denominator is the deduped count, so Qwen3-Embedding-0.6B's ~14 unique kernels
  show as 14/14 not 14/337.

**Separability + the structural handoff.** Op-variant forks are separable: every multi-option fork is an in-place `Op`
rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`. Results key structurally (`op_cache_key` =
name-invariant body+knobs digest), so a kernel tuned in its slice transfers to the assembled graph unchanged **and** is
shared across outer terminals — two fusion candidates sharing an identical op reuse its tuning (a DB hit). After the best
fusion is picked, the assembled `Graph[CudaOp]` is benched **once** for the real in-context whole-graph latency;
comparing it to the `Σ` estimate is the **separability check** — a gap exposes L2 / clock / launch coupling the isolated
benches can't see (in practice <2% for small graphs).

**Always re-run, replay from the cache.** The inner search runs for **every** op on every pass — it is never skipped on
prior effort. Replay is cheap, not gated: each benched terminal hits the per-variant `perf` cache, so an already-measured
variant is served from the DB with no GPU bench. An identical re-run (same prior) re-walks the same deterministic
trajectory → every terminal is a cache hit → zero benches and the same total. But the global learned prior keeps changing
(it refits across ops and runs), so the same patience can steer the MCTS down a *different* trajectory; re-running lets it
reach and bench the genuinely-new variants the improved prior surfaces, replaying the rest for free. (The old `op_effort`
"skip already-tuned" gate is gone — it suppressed exactly that prior-driven re-exploration.)

**Per-kernel GPU parallelism (`--gpus N` / `--devices 0,1,2`).** Because the inner search tunes each unique kernel
independently, the per-op loop fans out across GPUs. The whole tuner is async-only: `run_two_level_tune` `await`s
`_inner_reward_async` per outer terminal, which runs one coroutine per unique kernel over an `asyncio.Queue` of
`len(pool)` device-pinned `CudaBackend`s — each pops a backend, drives its op's whole inner search via
`Pipeline.tune_async`, then returns the backend. So `len(pool)` benches run at once, one per GPU. **True single-thread
asyncio**: every Python statement (lowering, DB writes, prior `add_rows` / `maybe_refit` / `checkpoint`) runs on the one
event-loop thread and yields only at the bench `await`, so the shared `db` / `prior` need no locks. Each op seeds its
`TuningSearch` by `seed + op_idx` and the reward is a commutative `Σ`, so the per-op DB bests and `total_us` are
byte-identical regardless of slot count; only the learned `prior.json` varies run-to-run (rows arrive in completion
order). The **default single-GPU** path is a one-slot pool whose coroutines acquire the lone worker in `op_idx` order —
strictly sequential, identical to the old serial loop. A backend pins its async worker to a physical GPU via the child
spawn env (`CUDA_VISIBLE_DEVICES`, plus a per-device `DEPLODOCK_GPU_LOCK` suffix), never mutating the parent
`os.environ`. Parallelism is bounded by the unique-kernel count; devices must be homogeneous.

**Search dynamics.** Each level reuses the **same** SP-MCTS (`policy/mcts.py`) — outer over structural forks, inner over
one op's forks — with max-Q normalized UCB1. **Selection** is PUCT (`_select`): `score(c) = Q(c) + c · P(c) ·
√(N_parent+1)/(1+N_c)`, where `Q = best_reward/global_best` (0 if unvisited), `reward = 1/median_us`, and `P` is the
prior's predicted reward on the same scale (the prior predicts latency `û(c)`, which `_select` converts to `1/û` and
normalizes by the same `global_best` — no softmax; `c = --ucb-c`). The prior is the SOLE signal — greedy tiebreak, the
static `TileOp.score` tiebreak, and the `+∞`-unvisited UCB rule are all gone. A confidently-slow sibling (large `û` →
small `P`) is deprioritized instead of force-benched. **Expansion** is implicit (one rule batch per pop, one child per
alternative). **Simulation** is the actual `await backend.benchmark_async(...)` on the terminal. **Backprop** walks the
popped candidate's parent chain updating `visits` and `best_reward`. **Patience** counts terminals since the last new
global best; when it exceeds `--patience N` (default 50), the level exits.

### Learned prior

ONE global `CatBoostPrior` across every kernel, GPU and nvcc setting — not per-op, not partitioned by regime. Op
structure (`S_*`) and the host/hardware regime (`H_*` — GPU compute capability + nvcc opt level, from `Context.features`)
are **features in every row**, not a cache key. Training signal is **value-of-position**: real benches exist only at
leaves, but the prior ranks partial-knob siblings at every fork level, so the label for any node is the best (min) median
latency µs over its benched descendants (`1/best_reward`) — the prior regresses on **latency**, and the `1/û` conversion
lives in the MCTS `_select` loop, not the stored data. `TuningSearch._collect_rows` walks the live tree and emits
`(knobs, label)` for every node with a benched descendant. A directly-benched **leaf** uses its `realized_knobs` — the
FULL config read off the resolved graph's op in `observe` (so knobs stamped at deterministic, non-forking lowering steps
— `FK` / `BK` / `SPLITK` / `STAGE` — are captured, not just the fork knobs). A **branch** falls back to `_node_knobs`
(its partial `fork.knobs` prefix under the op's `S_*` / `H_*` base), carrying the value-of-position label.

Alongside that reservoir feed, the same finished tree is walked once by `_collect_node_records` and persisted to the
`node` SQLite table via `record_nodes` — the keyed, deduplicated, parent-linked counterpart to the unkeyed/sampled
reservoir. The prior still *trains* from the in-memory reservoir, but the `node` store is *read back* by `eval prior
--dataset nodes` (`iter_nodes` → `diagnostics.node_report`): **per card**, it groups nodes by `parent_key` and scores the
**fork sibling-ranking** — does the prior order each fork's children (the partial configs it ranks during `_select`) by
their best-reachable latency? — the search-faithful evaluation no leaf-only view can give. The per-card grouping matters
for a cross-hardware dataset: same-die SKUs (H100/H200) share an `S_*` op signature but not their latencies, so mixing
them would corrupt both metrics — the `gpu` key keeps their rows distinct.

Why CatBoost (chosen by `scripts/prior_bakeoff.py`): the model's greedy pick must not run off to a degenerate corner. A
linear model (the former `BayesianRidgePrior`) is monotone in every knob, so its optimum is always a corner of the
candidate box — the `BR=1` blow-up (4us → 232us / invalid kernels). Any **bounded** tree ensemble is off-manifold-safe
(an un-benched extreme inherits the nearest leaf's value), and among them CatBoost also generalizes to an *untuned* op
near-perfectly (leave-one-op-out pick ratio ~1.0 vs xgb/lgbm 1.18, rf 1.31). So one global CatBoost prior is good enough
on a new op that it is **not refit within an op's own search** — it is a fixed model per run. The dataset is bounded +
batched (`base.Prior`): each tuned op's value-of-position rows stream into a reservoir-sampled dataset capped at
`MAX_ROWS` (100k, Algorithm R across runs), and the model refits (`maybe_refit`) on a dataset-size-tiered cadence
(`REFIT_SCHEDULE` — frequently while data-poor, coarsening as it grows), then checkpoints. End-of-run does a
`maybe_refit(force=True)` so even a small tune ends with a fitted model. The checkpoint is a JSON file
(`config.prior_path()`, `~/.cache/deplodock/prior.json`) holding the CatBoost `cbm` blob (base64) + the dataset; `tune`
writes it, `compile` / `run` read it.

**-O3 deployable samples.** The sweep compiles at `-Xcicc -O1` (fast, but a *ranking* signal — it ties configs that
differ at -O3, e.g. a `REDUCE` ILP fold or a warp tile's `WSPEC`). So whenever a bench lands **within `DEPLODOCK_O3_TOL`
(default 15%, `config.o3_tol`) of the best -O1 so far** — a band wider than a strict new best, so near-tied contenders all
qualify — the engine re-benches it at `-Xcicc -O3` (`_rebench_o3`) and `observe_o3` records an extra row with the same
realized knobs tagged `H_opt=3` (the deployable regime). Each config is re-benched at most once. The `H_*` feature lets
the -O1 (broad) and -O3 (near-best) rows coexist; `compile` / `run` run at -O3 (`H_opt=3`) so greedy ranks by the
deployable rows and reaches the true optimum. The `nvcc_flags` override rides the bench request to the worker, so only
winners pay the -O3 recompile and the cubin cache keys on the flags. All tune/bench timings are **CUDA-graph-captured** by
default (pure GPU time); each `perf` row records its mode in the `captured` column, and on write a captured measurement
supersedes a wall-semantics one for the same key (never the reverse), so old rows upgrade in place.

**Greedy uses the prior too — and flattens.** `greedy_decide` (the `Run.resolve` decide for `compile` / `run`) lazy-loads
the global `Prior` via `load_prior`. The lazy fork tree is an MCTS structure — it stages knob choices across levels
(`BR` → `BM/BN` → `FM/FN`) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level: a branch carries only a
*partial* tile, and `knob.knob_features` can't compute its area / occupancy until `FM/FN` are pinned, so the prior would
be blind at the `BM/BN` choice. Instead greedy **flattens** each fork point to its complete leaves
(`fork.flatten_leaves` expands branches depth-first; only knob dicts, materialization stays deferred to the chosen leaf)
and picks the lowest `Prior.mean_scores` over the full `{H_*, S_*, complete-knob-row}` vector in one batched `predict`,
invariant to the tree's level order. Cold the `AnalyticPrior` ranks (including a positive `MMA_tier` warp-preference);
only if `load_prior` returns nothing does it take option-0. Greedy benches nothing, so it can only *use* a prior, never
train one.

**Greedy validity fallback.** The prior ranks by predicted latency, which can rank a tile that fails `validate(ctx)`
(smem / thread budget) first — `tune` benches-and-skips it, but greedy benches nothing. So when a deterministic compile
leaves a node un-lowered, `Pipeline.run` blocklists that tile's `tile_identity` (its planner knobs) and **re-resolves**:
`greedy_decide(blocked=…)` drops the matching leaf and picks the next-best. Bounded by `_MAX_GREEDY_RETRIES`. When the
retry budget exhausts with the node still un-lowered (a *learned* prior can rank many over-budget tiles above the first
in-budget one), `Pipeline.run` takes one last **option-0 (emission-order) resolve** (`greedy_decide(prior=None)`): the
planner emits a budget-safe tile first, so it lowers whenever any in-budget tile exists. Only when even option-0 overflows
does `_raise_on_unlowered` fire the loud `LoweringError`.

**Driving the loop.** `deplodock tune <model_or_ir | --code EXPR>` probes a `Context`, opens the tuning database
(`DEPLODOCK_TUNE_DB` or `~/.cache/deplodock/autotune.db`), and calls `run_two_level_tune(...)`. The DB accumulates rows
across runs; re-running resumes from the cached state. On default verbosity (and a tty) a `TuneProgress` draws a live
single-line bar (completed/total tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail), threaded
as an optional `progress=` through `run_two_level_tune` (duck-typed, so the search package keeps no `commands/`
dependency); `-v` shows the per-`[tune]` INFO lines instead, `-q` is quiet. `--bench` re-benches the tuned winner at -O3
(deployable, not the -O1 ranking pass): the full model against the real torch module and each kernel via its `.torch.json`
provenance reproducer, vs eager / `torch.compile` / Deplodock.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK` /
`STR`), candidate `hints` (advisory — the rule still validates structural fit), and a help string. Rules declare them as
module-level constants and stamp values into `TileOp.knobs` dicts; the autotuner reads those back as the per-hop knob
delta in the `lowering` table. The registry (`knob.registry()`) auto-collects every `Knob` instance in every loaded rule
module — no manual registration. `knob.py` also owns the `DEPLODOCK_<KNOB>` env namespace (decode per `Knob` type;
`config.py` remains the sole owner of `os.environ`).

**Pinning knobs from the environment.** Two equivalent forms:

- **Per-knob:** `DEPLODOCK_<NAME>=<value>` (e.g. `DEPLODOCK_STAGE=d2/cp`). Read by the rule that owns the knob via
  `Knob.narrow`. The env-var key is built by `config.knob_var` and read via `config.knob_raw` / `config.int_env`.
- **Aggregate:** `DEPLODOCK_KNOBS="K1=V1,K2=V2,..."` (e.g. `DEPLODOCK_KNOBS="TILE=a:mma_m16n8k16_f16/w2x2/f2x2/k2,STAGE=d2/cp"`).
  Parsed once at `knob.py` import via `apply_knobs_env()`, which splats each entry into the corresponding
  `DEPLODOCK_<K>` var (`config.set_knob(..., overwrite=False)`). An explicit per-knob var wins over the aggregate.

Pinning replaces tuner choice (the rule emits exactly that variant instead of forking) and is **authoritative** — an env
value outside the knob's hint tuple is honored, not silently dropped (`Knob.narrow` returns `(pinned,)` regardless of
hint membership). Downstream structural gates (divisibility, threads-per-CTA budget, TMA eligibility) still apply, so a
structurally invalid pin yields an empty enumeration and the per-call-site fallback takes over. This lets a tile shape the
planner wouldn't reach on its own be explored manually.

A few pins are rejected outright (a clear `ValueError`) rather than silently degraded — they would otherwise lower to a
wrong or un-launchable kernel: a codec width must be `≥ 1` (a degenerate `b0` / `f0` / `n0` no longer parses to a
silently-dropped level); a warp `TILE` pin needs its **static** contraction K to be a multiple of the inner mma K-step
(`atom_k·bk`) since the warp K-loop has no static-K tail masking (a **symbolic** K is fine — it reaches the masked
zero-filled tier); a scalar `TILE` parallel block (`par_n·par_m`) is capped at the 1024-thread/CTA hardware limit; and a
`BOOL` knob rejects an unrecognized value instead of coercing a typo (`ture`) to `False`.

**Registered knobs** (declared across `passes/lowering/tile/*.py`; see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md)
for the per-rule mechanics):

| Knob     | Type    | Owning rule                       | What it controls                                                        |
|----------|---------|-----------------------------------|-------------------------------------------------------------------------|
| `TILE`   | STR (codec) | `lowering/tile/020_schedule` | **Unified output-fragment** codec — a contraction's output tile is *either* the **scalar** register sub-tile `n<N>[x<M>]/f<fn>[x<fm>]` (parallel thread-tile `n`/`m`, register sub-tile `f`) *or* the **warp** tensor-core mma tile `a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>` (atom + warps + register sub-tile + K-chunk), never both. The value self-discriminates: an `a:<atom>` token selects the warp `WarpTile` (`schedule.is_warp_codec`), otherwise the scalar `TilePlan`. Empty = per-cell. |
| `REDUCE` | STR (codec) | `lowering/tile/020_schedule` | Reduce-axis partition codec `g<n>[a\|k]/b<n>/r<n>` — `g` cross-CTA split-K (+ finalize letter), `b` cooperative-thread fold, `r` ILP register fold. Empty = serial (the per-thread remainder is derived, never spelled). |
| `STAGE`  | STR (codec) | `lowering/tile/020_schedule` → `lowering/kernel/010_materialize` | Operand-staging codec `d<depth>/sync\|cp\|tma[/ring][/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both fragments of the `TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport (`tma` folds in what the old `TMA` bool selected), `p<reg_depth>` the smem→register double-buffer. `stage=None` (unset / unparseable) = gmem-direct. See `lowering/kernel/ARCHITECTURE.md`. |
| `CUT`    | BINMASK | `split/010_split_demoted`         | Cut a demoted matmul's computed multiply-operand cone(s) into producer kernel(s) + the clean gemm — the structural fork above. Width-1 today; `"0"` = considered-and-declined, `"1"` = cut. Deliberately declares no `off=` (to preserve the absent-vs-declined distinction). |
| `FLASH`  | BOOL    | `loop/recognize/010_recognize_flash` | Fuse SDPA into one streaming online-softmax kernel (the `Monoid` carrier) instead of the score-materializing `010_sdpa` decomposition. Recovers causal / additive mask + GQA + RoPE-fused producers structurally from the fused body. Static or symbolic-`seq_len`. Off by default (env pin `DEPLODOCK_FLASH=1` today). |
| `ONLINE_SOFTMAX` | BOOL | `loop/recognize/020_recognize_online_softmax` | Fuse a standalone two-pass softmax (row-max + `Σ exp(x−max)`) into one streaming `(m, d)` `Monoid` pass. Off by default. |
| `S_*`    | FLOAT   | `loop/stamp/020_stamp_structural_features` | The LoopOp's structural features (stmt/op histogram + loop extents + operand dtypes). Not tunable — identity facts that make a knob dict a complete variant identity (the learned prior's feature vector). Skipped by `format_tuning_knobs`. |

The `REDUCE` codec's cross-CTA split is its `g<n>` field (GRID stage), and the **finalize** is that field's trailing
letter — `g<n>a` = in-place `atomicAdd` (one kernel, additive carriers only), `g<n>k` = deferred `__partial` workspace +
a sibling combine kernel (any carrier; the only legal arm for the twisted flash `(m, l, O)` split-KV). Pin via
`DEPLODOCK_REDUCE=g2k` (one flat knob — no per-axis `DEPLODOCK_REDUCE_<axis>`, no `DEPLODOCK_FINALIZE`). The split is
consumed by `lowering/tile/030_split` as a graph rewrite (partial + finalize); the letter round-trips through
`ReducePlan.parse`/`spell` and reads back as `ReducePlan.finalize`. The atomic finalize applies the kernel's projection
epilogue **per partition** before the `atomicAdd`, so it is only correct when that projection *distributes* over the add
(`Σ φ(xₛ) = φ(Σ xₛ)`): a constant scale like `mean`'s `×1/N` distributes and rides the atomic; a non-distributive
epilogue (`l2`'s `sqrt`, a fused bias/activation) is refused (`NotImplementedError` → pin `g<n>k`, which projects once
after the combine). The check is `030_split._projection_distributes`.

`BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2), the keywords `"all"` / `"none"`, or a decimal /
`0x`-hex int clamped to the candidate width. `format_tuning_knobs` drops `BOOL` knobs from the rendered `knobs=` line —
they're treated as pass-presence markers. `HOIST_COMPUTE` and `PAD_SMEM` are BOOL autotune forks emitted in a fixed order
(the greedy default first — inline-fuse / pad-on respectively); both honor their `DEPLODOCK_*` pin. The masked-K MMA slab
alignment pad is **not** a fork — it's stamped intrinsically on the `Source` at staging (a near-strict win greedy deploys
without a re-tune).

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is deterministic. Pick a fresh prefix when adding a rule; the
loader ignores the prefix itself — it's only for ordering readability. Per-pass authoring invariants are in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the tile passes (`split` → `enumeration` → `assembly`) and the
algebraic moveset are also documented there.

| Pass                      | What rules do                                                                                |
|---------------------------|----------------------------------------------------------------------------------------------|
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`.            |
| `loop/fusion/`            | `split_shared_indexmap` (first) fuses a fan-out pure-indexmap `LoopOp` into all its consumers in one rewrite; `merge_loop_ops` then splices adjacent single-consumer `LoopOp` pairs; `dedup_loads` drops identical `(input, index)` Loads. Folding scalar-constant broadcasts into consumers cuts Qwen3-Embedding-0.6B from 394 → 337 kernels. |
| `loop/recognize/`         | Pattern recognizers run AFTER the `loop/fusion` fixpoint settles (not interleaved). `recognize_flash` folds a softmax-then-P@V SDPA into one streaming flash `LoopOp` (the `FLASH` knob); `020_recognize_online_softmax` is the standalone-softmax sibling. |
| `loop/stamp/`             | `stamp_loop_names` (`provenance.name_for`, e.g. `k_rms_norm_3f2a1b`) + `stamp_structural_features` (the `S_*` dict). Runs last in the loop dialect — after fusion and recognition — so every kernel is named / stamped against its final body. |
| `lowering/tile/`          | `LoopOp → TileOp` over the block-DAG Tile IR: `010_recognize` → `020_schedule` (maps the grid, picks the reduce/output fragment, and **atomizes** — resolves the algebra→atom binding onto the schedule via `_atomize.py` when each option is built, rejecting an unbindable atom at fork construction) → `030_split`. Dispatch is on the carrier algebra (`MAP` / `SEMIRING` / `MONOID`), never a named shape. |
| `lowering/kernel/`        | `005_contract` constructs a `Semiring` contraction's high-level `Contraction` node **before** materialize — one node carrying a per-atom `Leaf` payload (`MmaLeaf` warp arm / `ScalarLeaf` register-tile arm). `010_materialize` is a thin `TileOp → KernelOp` tier dispatcher (scalar / `_reduce`) that **also folds in** expanding both contraction arms through the one atom-generic `_factor.factorize` over the shared `_tiling` layer (the geometry is derived on the `Contraction` node; the one `_factor.codegen` dispatches the codegen callables off the atom). Then the Kernel-IR peepholes: `030_stamp_types` (+ `040_demote_to_write_dtype`) resolve dtypes, `050_vectorize_loads` / `080_vectorize_stores` / `095_interleave_loads` pack/reorder memory ops, `110_drop_redundant_syncs`. See [`passes/lowering/kernel/ARCHITECTURE.md`](passes/lowering/kernel/ARCHITECTURE.md). |
| `lowering/cuda/`          | `lower_kernelop` renders the `KernelOp` body to a `__global__` source string (`ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph uniformly for every pass:
`NN_<pass_name>.{json,txt,dot}` (+ `NN_<pass_name>.kernels.txt` if any node has a non-empty `pretty_body()`). Slashes in
the pass name flatten to underscores. The pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)`. The uniform strategy means adding a pass automatically gets dumped — no registration.

Per compute kernel, `_dump_per_kernel` writes `<prefix>.kernels/<kname>.json` — a standalone sub-graph (kernel + its
`InputOp` / `ConstantOp` producers) loadable via `deplodock run --ir`. When op provenance is present (see
`compiler/provenance.py`), it also writes `<kname>.torch.json` + `.torch.txt`: the **original Torch ops** that kernel
implements, sliced from the pristine pre-decomposition graph by origin id (so the slice is always whole Torch ops), with
an `i/N` coverage header — runnable via `deplodock run --ir <kname>.torch.json --bench` to reproduce accuracy / latency
vs torch.

## Per-rule diff output (`rule_diff.py`)

At `compile -vv` (DEBUG) the engine emits one block per rule application: a unified diff between the matched subgraph and
the rewritten fragment, bracketed by `>>> <pass>:NNN_rulename` and `<<< <pass>:NNN_rulename` markers. The `<pass>` prefix
is the single-letter shorthand from `PASS_SHORTHAND` (`d` / `o` / `l` / `f` / `t` / `k` / `c`) — the same letters the CLI
accepts in `--passes dolft` (`commands/compile.py` imports `PASS_SHORTHAND` so the flag and the marker prefix can't
drift). Skipped rules collapse to a one-liner. The bracketing makes per-rule / per-pass slicing trivial via `awk`; ANSI
color is applied only inside the diff body so the markers stay plain ASCII. Color follows `compile --color`. Body-carrying
ops render through their own `pretty_body` (the in-flight `TileGraphOp` pretty-prints its block-DAG), so a tile-pass diff
reads as a readable block-DAG delta. The structured `.rules.json` dump is unaffected — the diff is purely presentation.

## Fragment splice (`engine._apply_replacement`)

1. Walk the fragment in topo order. `InputOp` nodes forward their id to the existing graph node (external reference);
   non-Input nodes are added with fresh ids.
2. `replace_node(match.output or match.root_node_id, new_output)` rewires all consumers (and `graph.outputs` slots) from
   the old output to the fragment's output id.
3. Merge hints from every consumed node into the new output.
4. Remove consumed nodes and run `_remove_orphans` to drop any now-dangling constants / inputs.

## Invariants

- A rule module must not reach into the engine's internals; its interface is `PATTERN` + `rewrite(graph, match)`.
- `pipeline/` imports from `ir/` but never from `backend/`. Lowering rules produce IR; executing that IR is the
  backend's job.
