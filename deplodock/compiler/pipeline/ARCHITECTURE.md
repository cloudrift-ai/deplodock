# Pipeline Architecture

Pattern-based rewrite engine + pass directories + dump hooks.

## Modules

```
pipeline/
├── pipeline.py    # Pattern, Match, Rule, Pass, Pipeline; engine + Pipeline.run / Pipeline.tune / Pipeline.search
├── knobs.py       # format_tuning_knobs: render real knobs (drop pass-marker booleans) for tune output
├── search/        # Autotune state: Candidate, Search policies, SearchDB + SearchTree
│   ├── candidate.py  # Candidate / LazyCandidate / Cursor data classes
│   ├── policy/       # Search ABC (base.py) + GreedySearch (greedy.py) / TuningSearch (mcts.py)
│   ├── db.py         # SearchDB SQLite store: op inventory + lowering edges + perf + op_effort
│   ├── keys.py       # op_cache_key / dialect_of / source_chain
│   ├── slice.py      # single_node_graph: isolate one finalized kernel into a standalone graph
│   ├── two_level.py  # two-level tuner: outer fusion MCTS + inner separable per-op reward
│   └── golden_configs.py  # GoldenConfig / MatmulGoldenConfig: autotuned knobs + cuBLAS-ratio per shape, fp32 (CUDA-core) + fp16 (WMMA) (a tuning-prior ground truth)
│ # SearchTree (in-memory MCTS state) lives in policy/mcts.py — MCTS is the only policy that reads it.
├── dump.py        # CompilerDump + on_pass dispatch
├── rule_diff.py   # Per-rule unified-diff renderer for ``compile -vv`` output
└── passes/
    ├── frontend/
    │   ├── decomposition/  # frontend ops → tensor-IR primitives
    │   └── optimization/   # IndexMap fusion before lift-to-loop
    ├── loop/
    │   ├── lifting/        # tensor ops → trivial LoopOp nodes
    │   └── fusion/         # fuse fan-out indexmaps into all consumers, then merge adjacent LoopOp pairs (splice)
    └── lowering/
        ├── tile/           # LoopOp → TileOp (tileify + scheduling rules)
        ├── kernel/         # TileOp → KernelOp (materialize scheduling)
        └── cuda/           # KernelOp → CudaOp (render source string)
```

## Engine (`engine.py`)

### Chain matcher

A `Pattern(name, op_type, constraints={})` matches one node by op type
plus optional `node.op` field equality. A pattern list matches a chain:
the seed matches `pattern[0]`, its sole consumer matches `pattern[1]`,
and so on. Multi-node patterns only fire when each intermediate node
has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered
seed; overlaps between matches are allowed (the rewriter exits after
the first successful rewrite per iteration so overlap is just
candidate enumeration).

`Match.nodes: dict[str, Node]` maps each pattern entry's name to the
matched `Node`. `Match.consumed` and `Match.output` are overridable by
the rewrite function to control which nodes the splicer removes and
which node's edges get rewired.

### Rule module convention

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]   # required
def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]:
    ...
```

The dispatcher binds parameters by name. Reserved names: `graph`,
`match`, `root`, `out`, `ctx`. Pattern names from `PATTERN` bind to
matched `Node` objects. Anything else binds positionally to
`root.inputs[i]`. Take only what you need — `ctx` is optional.

**Returning a list = autotune fork.** A rule that's unsure which
parameter to use returns the alternatives as a list. The engine applies
option 0 inline and pushes one `Candidate` per remaining option onto the
search queue (deep-copying the graph at the fork point). Single-option
returns (or bare `Graph` / `Op`) are the deterministic case — no fork.

**Lazy hierarchical forks via `Fork`.** A rule can also return a list of
`Fork(knobs=..., expand=..., score=...)` objects — each Fork carries the
knob delta it pins plus a thunk that produces the next level of options
(more Forks, concrete `Op`/`Graph` leaves, or a mix). The search loop
pops a Fork-pending `LazyCandidate`, invokes `expand()` to materialize
the children, pushes them back, and continues; cursor advance only
fires when the lineage resolves to a concrete leaf. Lets a rule expose
a hierarchy of decisions lazily — only the subtrees MCTS actually walks
into get materialized. `Fork.knobs` is read by `_best_fork` (for
DB-seeded greedy replay) without firing the thunk; `Fork.score` is the
MCTS prior the producing rule attaches.

The partition planner (`lowering/tile/010_partition_loops`) emits a
hierarchical Fork tree: `BR → (BM,BN) → (FM,FN) → (BK,SPLITK) → TileOp`
leaf. Sibling sorting uses `TileOp.lazy_score(ctx, shapes=..., params=...)`
— the static-method counterpart of `Op.score` that estimates the same
formula from cheap inputs (knob bundle + planner shape) so siblings rank
without anyone instantiating a TileOp. The branch tree's `Fork.score`
propagates max from leaves, matching MCTS's max-Q semantics.

Binding tiers the planner emits today: `Role.BLOCK` (→ `GridTile`),
`Role.THREAD` (→ `ThreadTile`), `Role.REGISTER` (→ `RegisterTile`).
`Role.WARP` (→ `WarpTile`) and `Role.ATOM` (→ `AtomTile`, the
hardware-atomic MMA cell tier) are wired through `_layer_kind_for` /
`_wrap_tower` but no rule in this pass emits either today — the MMA
fragment-factorization consumer plan (`plans/mma-fragment-factorization.md`)
will flip these tiers when its M3 ships, without revisiting the tower
mechanics. M1 of that plan landed the `AtomTile` flavor + empty
`_atom.ATOM_REGISTRY` + the `TileParams = ScalarTileParams |
WarpTileParams` sum-type split in `_enumeration` (`ScalarTileParams`
carries today's `(BN, BM, FM, FN, BK, SPLITK, BR)`; `WarpTileParams`
carries `(WN, WM, FM, FN, BK, SPLITK, ATOM_KIND)` — no `BR`, no
`BN`/`BM`). `085_warp_specialize` already emits `WarpTile(role)` (one
role axis = total CTA warps) wrapping `WarpSpecialize` directly,
bypassing the planner's tower builder — its role split is structural
(`Cond(role < n_producer_warps, …)`), not the σ-shifted extended
`ThreadTile` the pre-refactor shape used. The materializer drops a
`ThreadTile(tid_offset=n_producer_threads, …)` inside the consumer
branch so the original consumer thread axes decode against
`threadIdx.x - n_producer_threads`.

The tree-building algorithm itself (group params by per-level knob keys,
collapse single-key levels, sort siblings by max-propagated score, defer
leaf materialization to `expand` thunks) lives in `pipeline/fork_tree.py`
as the reusable `Level` + `build_fork_tree` pair — `partition_loops`
supplies the four `Level`s + `materialize=` + `score=` callables and
forwards the result. Future rules with multi-level knob-cartesian forks
should reuse the builder; one-shot flat forks (e.g.
`lowering/tile/085_warp_specialize`'s `WS={0,1}` 2-element list) stay
inline.

**FN > 1 lowering.** The partition planner always emits the per-cell
shape — one `RegisterTile(N_r)` wrapping the whole
`{Init, K-reduce, Write}` body, regardless of FN / SPLITK / BR / prologue
shape:

    M_r REGISTER:
      RegisterTile(N_r):
        Init(acc)
        K_o SERIAL_OUTER:
          K_i STAGE_INNER (reduce):
            <body>                               # M-axis Loads, prologue,
                                                 # Load b, Accum — replicated
                                                 # per cell by the Kernel-IR
                                                 # replicator
        Write(C, acc)

The Kernel-IR replicator (`lowering/kernel/010_split_register_axes`)
walks the body and per-cell duplicates only statements that transitively
depend on the register axis (dep-tracked via Expr free vars + SSA
def-use); N-invariant statements (e.g. Load `a[m, k]`, the RMSNorm
prologue chain) stay single-copy. `dedup_replicated`
(`lowering/kernel/011_dedup_replicated`) then runs as content-agnostic
defense in depth: structurally identical Loads and Assigns left over
after replication CSE-fold into one — the same effect the deleted
register-blocked GEMM builder used to get from its hand-written
N-invariant-cone partition (see `plans/obsolete-blocked-gemm-builder.md`).

Leaf Fork `expand` thunks call `_materialize(plan, params)` lazily —
`_build_split_body` + `TileOp.__post_init__` (which runs the full
12-pass `normalize_body`) only fire for the variant the search actually
resolves. In greedy `deplodock compile`, that's one variant per
LoopOp. Earlier behavior materialized every variant up front (dozens
to ~200 per matmul-class kernel) just to score them; the lazy split
cut whole-model Qwen3 0.6B compile from 10+ minutes to ~48 seconds.

For rules that want a custom lazy scorer, override
`Op.lazy_score(cls, ctx, *, knobs=None, shapes=None, params=None)` on
the producing Op class. The base implementation returns `None` (no
lazy estimate available); callers fall back to constructing the op
and calling `score(ctx)`. `TileOp.lazy_score` is the reference
implementation — it consumes `KernelShape` + `TileParams` from the
partition planner and matches `TileOp.score`'s formula
( `score_tile_geometry`) on the launch-geometry + cells + coalescing
keys. The smem-fit penalty needs the count of input buffers the kernel
will stage, which `lazy_score` derives from `KernelShape` (walking
`outer_m` / `extra_outer` / `prologue` for distinct `Load.input`); the
post-materialization path defaults to 2 (plain A+B) when no shape is
available, so the two scores can disagree by up to the smem-fit
penalty's ±2.5 on SDPA-with-mask / RoPE-fused kernels. That asymmetry
is acceptable because only the planner-time scorer drives sibling
ordering; the post-materialization score is a sanity metric.

**Idempotence requirement.** Every rule MUST be idempotent on its own
output. The engine re-runs the entire pipeline on each popped candidate
from pass 0; rules whose output is already in the graph must `RuleSkipped`
or have a pattern that no longer matches. Most rules satisfy this
implicitly via op-type changes (`LoopOp` → `TileOp`); the rest have
explicit `raise RuleSkipped("already X")` guards. Without idempotence,
re-runs would double-apply and corrupt graph state.

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of
  `match.output` (defaults to `match.root_node_id`); fragment `InputOp`
  nodes reference existing graph nodes by id, non-Input nodes get fresh
  ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op`
  directly, preserving the node id, inputs list, output Tensor, and
  hints. Used by the lowering rules because `KernelOp.arg_order` /
  `CudaOp.arg_order` embed the original node id as the output buffer
  name, so a fresh id from splicing would break the generated kernel's
  buffer binding.

Raise `RuleSkipped(reason)` to decline a match — the engine logs the
reason at DEBUG and moves on.

A rewrite that *returns* an op which fails `Op.validate(ctx)` (e.g. a
`100_materialize_tile` `KernelOp` whose smem exceeds `ctx.max_dynamic_smem`)
is filtered by `Candidate.try_rewrite` — correct as **fork pruning** (sibling
branches carry other tile shapes) but fatal in a single-path greedy compile,
where it leaves the node un-lowered. `Pipeline.run` installs a transient
`_lowering_rejections` sink that records each such drop `(node, pass, reason)`;
after the terminal settles, `_raise_on_unlowered` raises a loud `LoweringError`
naming any still-un-lowered node (its op is still a `LoopOp`/`TileOp`) instead
of letting it leak to `CudaBackend` as a cryptic `non-CudaOp` `TypeError`. The
sink is absent under `tune`, so the fork-pruning path stays silent and a
validate-dropped branch is a graceful dead end.

Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as
rules — they're shared helpers for the pass's rule modules.

### Drivers

`Pipeline.build(passes)` wraps a pass list; the resulting object exposes
three entry points:

- `Pipeline.run(graph, *, backend=None, db=None) -> Graph` — single-shot
  compile via `GreedySearch`. Stops at the first terminal candidate.
- `Pipeline.tune(graph, *, search, backend=None, db=None) -> Iterator[Candidate]` —
  autotune sweep. Pass a `TuningSearch(patience=, ucb_c=)`; the iterator
  yields one terminal `Candidate` per fully-explored rollout.
  `Pipeline.tune` benches each terminal via `_bench_terminal` (writes
  per-kernel `perf` / `lowering` / inventory rows, returns the aggregate
  `PerfStats`), then calls `search.observe(stats, status)`. With
  `backend=None` the bench is stubbed to `latency_us=1.0` and nothing
  is persisted — otherwise `Pipeline.run` (also routed through `tune`)
  would overwrite tuned `best_median_us` rows with the stub.
- `Pipeline.search(search, ctx, *, db=None) -> Iterator[Candidate]` — the
  inner engine loop both wrappers drive. Pops a `LazyCandidate`,
  resolves it, runs one rule batch, pushes successors. At fork points,
  if `db` is provided, looks up the best-known child for the parent
  op's `op_cache_key` in `lowering` and passes the matching fork via
  `search.push(..., best=...)`. Greedy uses `best` when present;
  tuning ignores it.

### Search persistence: on-disk inventory vs in-memory MCTS

The autotune state is split across two cooperating modules:

- **`SearchDB`** (`search/db.py`) — SQLite store partitioned into six
  tables: `loop_op`, `tile_op`, `kernel_op`, `cuda_op` (one row per op
  encountered along any lowering chain, keyed by `op_cache_key`), a
  `lowering` edge table (one row per rewrite hop carrying the knob
  delta the rule stamped at that hop plus a best-median upsert, so
  `GreedySearch` can replay the chain by matching forks against the
  delta at each step), and a backend-partitioned `perf` table carrying
  full stats (`latency_us_{median,min,max,mean,variance}`,
  `n_samples`, `backend`, `status`, `knobs`). Selection statistic is
  the median.
- **`SearchTree`** (`search/policy/mcts.py`) — pure-Python in-memory
  MCTS state, colocated with `TuningSearch` because MCTS is the only
  policy that reads it. Each tree node wraps a `LazyCandidate`; nodes
  carry `visits` and `best_reward` (max reward over the subtree's
  measured leaves), plus a `live` counter that filters out subtrees
  whose frontier has been fully drained. Tree built from push lineage
  (parent of a pushed node = the most recently popped node). Rebuilt
  fresh each process; cached `perf` rows in the DB ensure no re-bench
  on warm starts. `GreedySearch` has no tree.

`Pipeline._bench_terminal` is the only function that knows about all
four parts (graph, DB, tree-through-`search.observe`, backend). It
short-circuits when every `CudaOp` in the graph already has a `perf`
row for the current `(context_key, backend)` — no GPU bench, stats
reconstructed from the DB. Otherwise it does one
`backend.benchmark(...)` call, walks `Op.source` once to record op
inventory + lowering edges + the `perf` row per kernel, and returns
the aggregate `PerfStats` (summed across kernels) for the search to
score.

## Tuning workflow

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the lowering pipeline with
different knob choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the
configurations that produced the lowest measured latency.

### Two-level search: outer fusion MCTS + inner separable per-op tuning

`deplodock tune` does **not** run one MCTS over the whole graph. The pipeline applies rules sequentially, so two very
different kinds of fork — **op-variant** forks (tile / pad / stage choices for one kernel) and **fusion** forks (how ops
group into kernels) — would nest and cross-product under one global patience. That cross-product is what starved deep
ops (the bottleneck kernel exhausted patience before an SDPA P@V kernel reached its good tile). The two kinds have
opposite structure, so `search/two_level.py` splits them:

- **Outer search** (`run_two_level_tune`) drives only the graph-changing passes (`OUTER_PASSES` = `frontend` + `loop`,
  i.e. `LOOP_PASSES`). A **terminal** is the state where the cursor would advance into `lowering/tile`
  (`partition_loops`) — every op post-fusion and structurally final. Each terminal is a candidate fused graph; its
  **reward** is `1 / Σ best-per-op time` from the inner search, backpropagated by the reused `TuningSearch`. **Today
  fusion is deterministic** (no rule emits a multi-option *fusion* fork — see `autotune_no_graph_forks`), so the outer
  tree has exactly one terminal and this reduces to "tune each op once, sum, assemble". The outer tree is the
  generalization that lets fusion forks plug in later with no change to the inner search. The outer uses
  `Pipeline.search` directly (manual `observe`) since its reward comes from the inner tuning, not `_bench_terminal`.
- **Inner search** (`inner_reward`) tunes each finalized kernel **independently** in its own single-node slice
  (`single_node_graph`, `search/slice.py`) with a plain `TuningSearch` over `LOWERING_PASSES` only (`tile → kernel →
  cuda`). The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic
  `InputOp`; the root op is shared **by reference**, so its body — and thus `op_cache_key` — is byte-for-byte the
  full-graph op's. Lowering-only (never re-running `loop/fusion`) is what keeps that body untouched. Because the inner
  tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget — `Σ_k n_k` benches,
  never the product. **Leaves are deduped by `op_cache_key`** before iteration: 24 RMSNorm LoopOps across 24 layers
  collapse to one work unit, and the outer `total_us` accumulates `best * multiplicity` so the reward stays
  multiplicity-weighted (bit-for-bit the same as per-node iteration). `OpResult.multiplicity` carries the count;
  positions = `sum(r.multiplicity for r in reward.per_op)`. The progress denominator is the deduped count, so
  Qwen3-Embedding-0.6B's ~14 unique kernels show as 14/14 not 14/337.

**Separability + the structural handoff.** Op-variant forks are separable: every multi-option fork is an in-place `Op`
rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`. Results key structurally (`op_cache_key` =
name-invariant body+knobs digest), so a kernel tuned in its slice transfers to the assembled graph unchanged **and** is
shared across outer terminals — two fusion candidates sharing an identical op reuse its tuning (a DB hit), so fusion
search only pays for the ops that differ. After the best fusion is picked, the assembled `Graph[CudaOp]` (greedy replay
of the DB-best forks over the original graph) is benched **once** for the real in-context whole-graph latency; comparing
it to the `Σ` estimate is the **separability check** — a gap exposes L2 / clock / launch coupling the isolated benches
can't see (in practice <2% for the small graphs above).

**"Terminated" effort — skip already-tuned ops.** Per op, the inner search is gated on persisted effort
(`SearchDB.record_effort` / `effort_for` / `terminated`, the `op_effort` table keyed by `(context_key, op_key)`). An op
is `terminated` when its recorded effort `>= patience`; the recorded value is `∞` once the op's inner tree **exhausted**
(the search drained without a patience stop, i.e. `stop_reason is None`), else the `patience` it ran with (max-kept).
This makes re-runs idempotent (`cached` in the summary), higher patience re-deepen only under-tuned ops, and the whole
sweep resumable across sessions. The inner search records the **best whole-slice total** (`Σ` over the slice's CudaOps,
so a split-K main + combine both count) under the LoopOp key via `record_perf`; `best_per_op_time` prefers that direct
row and otherwise walks the `lowering` chain down to the `cuda` terminal.

**Driving the loop.** `deplodock tune <model_or_ir | --code EXPR>` probes a `Context`, opens the tuning database
(default `~/.cache/deplodock/autotune.db`, overridable via `DEPLODOCK_TUNE_DB`), and calls `run_two_level_tune(...)`.
On completion it prints one `done: N fused terminal(s) in Xs` line — the deployable numbers come from the optional
`--bench` step below. The DB accumulates rows across runs; re-running resumes from the cached state.

On default verbosity (and a tty) a `commands/tune_progress.TuneProgress` draws a live single-line bar — completed/total
tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail. The current latency is fixed-width and the
variable-length `pipeline.variant_label` knob string sits last, so the prefix up to the knobs stays put as the
per-variant latency changes (only a new best, which is rare, shifts the trailing part — no flicker). It is threaded as an optional `progress=` through `run_two_level_tune` → `inner_reward`
(duck-typed, so the search package keeps no dependency on `commands/`): one op leaf ticked per kernel, the tail updated
per benched variant (read off `TuningSearch.last_stats`). `-v` disables the bar (the per-`[tune]` INFO lines show
progress instead); `-q` is quiet (errors only). `--bench` re-benches the tuned winner at **-O3** (deployable, not the -O1 ranking pass) after tuning —
the assembled full model **against the real torch module** (eager / `torch.compile` / Deplodock, via the bundle
plumbed from `load_or_trace` → `commands/run.bench_full_model_real`) and each kernel's `.torch.json` provenance
reproducer (re-lowered greedily so the tuned forks are picked) vs eager / `torch.compile` / Deplodock via
`commands/run.bench_lowered_vs_torch`, printing
full-model + per-kernel tables and (when a dump dir is set) an HTML chart at `<dump-dir>/kernels.html`.

**Search dynamics.** Each level reuses the **same** SP-MCTS (`search/policy/mcts.py`) — outer over fusion forks, inner
over one op's forks — with max-Q normalized UCB1:

- **Selection** picks the child of the current node with the highest `Q_norm + ucb_c · sqrt(ln(parent_visits) / child_visits)`,
  where `Q_norm = child.best_reward / global_best_reward` and `reward = 1 / median_us` (so lower latency = higher reward).
  `child.score` is a rank-only structural prior the rule stamped via `TileOp.score` (no magnitude — only relative ordering).
- **Expansion** is implicit: `Pipeline.search` pops a node and runs one rule batch; every fork pushes one new child per
  alternative. The tree mirrors the graph's fork lineage.
- **Simulation** is the actual `backend.benchmark(...)` call on the terminal — for the inner search that is one real GPU
  run of a single-kernel slice per leaf.
- **Backprop** walks the popped candidate's `parent` chain up to the root, updating `visits` and `best_reward` so future
  UCB1 calls see the new max-Q.
- **Patience** counts terminals visited *since the last new global best*; when it exceeds `patience` (`--patience N`,
  default 50), `TuningSearch.stop_reason` is set and that level's `Pipeline.tune` / `Pipeline.search` exits. The inner
  search records `∞` effort when it instead drains its tree (no patience stop).

**Reading the result.** `_bench_terminal` writes one `perf` row per CudaOp per `(context_key, backend)` keyed on
`op_cache_key`, plus a `lowering` edge per rewrite hop carrying the knob delta the rule stamped (and the inner search
adds the whole-slice total under the LoopOp key). A subsequent `deplodock compile` / `deplodock run` (or
`make bench-kernels-tuned`) auto-resolves the same DB path (env or default) and replays the cached forks via
`GreedySearch`, which walks the parent op's `op_cache_key` in `lowering` and follows the best-known child at each step —
the same replay `run_two_level_tune` uses to assemble its final graph. No GPU bench is required on the replay path when
every kernel already has a `perf` row.

**Stub backend.** With `backend=None`, `_bench_terminal` short-circuits to `latency_us=1.0` and persists nothing — used by
test fixtures so `Pipeline.run`'s greedy replay doesn't clobber tuned rows with a stub when no GPU is available.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK`),
candidate `hints` (advisory — the rule still validates structural fit), and a short help string. Rules declare them as
module-level constants and stamp values into `TileOp.knobs` dicts; the autotuner reads those dicts back as the per-hop
knob delta in the `lowering` table. The registry (`knob.registry()`) auto-collects every `Knob` instance in every loaded
rule module — no manual registration.

**Pinning knobs from the environment.** Two equivalent forms:

- **Per-knob:** `DEPLODOCK_<NAME>=<value>` (e.g. `DEPLODOCK_BK=32`). Read by the rule that owns the knob (via
  `Knob.narrow`) or by `compiler/tuning.py`'s heuristics. The `DEPLODOCK_<NAME>` env-var key is built by
  `config.knob_var` and the value read via `config.knob_raw` / `config.int_env` — `deplodock/config.py` is the single
  owner of `os.environ` for all `DEPLODOCK_*` vars; `knob.py` keeps the `Knob` descriptor's per-type decode.
- **Aggregate:** `DEPLODOCK_KNOBS="K1=V1,K2=V2,..."` (e.g. `DEPLODOCK_KNOBS="BK=2,BM=16,BN=128,FM=8,FN=8,STAGE=111"`).
  Parsed once at `knob.py` import via `apply_knobs_env()`, which splats each entry into the corresponding
  `DEPLODOCK_<K>` env var (via `config.set_knob(..., overwrite=False)`) so all the per-knob readers pick it up
  uniformly. An explicit per-knob var wins over the aggregate (so `DEPLODOCK_BK=4 DEPLODOCK_KNOBS="BK=2,BM=16"` ends up
  with BK=4, BM=16).

Pinning replaces tuner choice: the rule sees the env value and emits exactly that variant instead of forking. Useful for
reproducing a tune-time variant from CI logs, A/B-comparing two configs, or pinning a known-good config in a Makefile
recipe.

Pinning is **authoritative** — an env value outside the knob's hint tuple is honored, not silently dropped. `Knob.narrow`
returns `(pinned,)` regardless of hint membership; downstream structural gates (divisibility, threads-per-CTA budget,
TMA eligibility, …) still apply, so a structurally invalid pin yields an empty enumeration and the per-call-site
fallback (`_enumeration._run(apply_pins=False)`) takes over. This lets a tile shape that the planner wouldn't reach
on its own — e.g. the article's BM=8, FM=26, fat 208×128 matmul tile — be explored manually, while peer kernels with
incompatible divisibility still get a sensible default.

**Registered knobs.** All knobs in `passes/lowering/tile/*.py`:

| Knob          | Type     | Owning rule                  | What it controls                                                                                  |
|---------------|----------|------------------------------|---------------------------------------------------------------------------------------------------|
| `BK`          | INT      | `010_partition_loops`        | Per-stage K-chunk size for matmul reductions; intra-CTA K-loop trip count = `K / BK`.             |
| `SPLITK`      | INT      | `010_partition_loops`        | Cross-CTA K-split factor for matmul; `1` = no split. Multiplies CTA count, requires a final combine. |
| `BN`          | INT      | `010_partition_loops`        | CTA innermost THREAD-axis width (the column tile each warp covers).                               |
| `BM`          | INT      | `010_partition_loops`        | CTA outer THREAD-axis width (matmul only — the row tile each warp covers).                        |
| `STAGE`       | BINMASK  | `020_stage_inputs`           | Bitmask over ranked candidate buffers — char `i` = stage buffer `i`. Selected buffers fold into one wrap-body Stage with per-source Source entries. |
| `FM`          | INT      | `010_partition_loops`        | Register-tile factor along the matmul M (output row) axis; per-thread cell-grid height.           |
| `FN`          | INT      | `010_partition_loops`        | Register-tile factor along the matmul N (output column) axis; per-thread cell-grid width. The planner emits one outer `RegisterTile(N_r)` around `{Init, K-reduce, Write}`; the Kernel-IR replicator + `dedup_replicated` pass produce the textbook blocked-GEMM shape (N-invariant Loads kept single-copy, N-dependent Accums replicated). |
| `BR`          | INT      | `010_partition_loops`        | Cooperative-K thread count (1 = pure serial chunked reduce); BR > 1 routes through the cooperative reduce path with cross-thread combine. |
| `WN`          | INT      | `010_partition_loops`        | CTA innermost WARP count along the matmul output N axis (warp-tier MMA tiles only).               |
| `WM`          | INT      | `010_partition_loops`        | CTA outer WARP count along the matmul output M axis (warp-tier MMA tiles only).                   |
| `ATOM_KIND`   | STR      | `010_partition_loops`        | Hardware matmul atom kind; see `passes/lowering/tile/_atom.py`. WMMA kinds (`wmma_m16n16k16_f16`, …) auto-enumerate at all sizes. The s16816 `mma_m16n8k16_f16` (ldmatrix + mma.sync, swizzled TMA slab) auto-enumerates as a tunable candidate on sm_90+ for large tiles (every extent ≥ 256, ≥ 2 warps so it stages — it has no gmem-direct fallback); the greedy/DB-less default stays WMMA (the static score can't see the swizzle's runtime win), so mma.sync is picked only via measured autotune (the fp16-square goldens). A pin forces it at any size / arch. |
| `TMA_SWIZZLE`     | BOOL     | `050_use_tma`                       | (legacy probe knob; superseded) TMA hardware-swizzle is now auto-on per-Source for mma.sync.      |
| `HOIST_COMPUTE`   | BOOL     | `030_hoist_invariant_compute`       | False (default) → inline-fuse Stage; True → ComputeStage + transports. Autotune fork.             |
| `PAD_SMEM`        | BOOL     | `070_pad_smem`                       | True → apply per-source ``+1`` smem pad to break bank conflicts; False → leave the slab dense. Autotune fork. |
| `GROUP_M`         | INT      | `025_swizzle_blocks`                | L2-friendly CTA-swizzle row-group size (Triton/CUTLASS convention). Default `8`; `1` is the global escape hatch (row-major decode). Stamped on the outer matmul GridTile's `swizzle_group_m` field; the renderer emits a Triton-canonical `blockIdx.x` remap so groups of `GROUP_M` CTAs walk down M before stepping N, sharing A's row tile in L2. Self-disabling on tiny / tall-skinny matmuls via the runtime `min(GROUP_M, num_m - first_m)` clamp. |
| `BUFFER_COUNT`    | INT      | `040_use_ring_buffers`              | Ring-buffer depth (and pipeline stages) for BUFFERED/ASYNC/TMA staged K-outer loops. `2` = classic double-buffer; `3`/`4` = CUTLASS-style multistage (pruned when the per-stage smem × N exceeds the cap). The greedy default orders the surviving variants by occupancy — front-loading the deepest depth that still keeps **2 CTA-blocks/SM** resident (`2 × depth × per-stage ≤ cap`), since past that the kernel drops to 1 block/SM and runs slower (measured 2048² fp16: 128×128 depth-3 = 115 µs vs depth-4 = 136 µs). This reorder fires **only for single-`StageBundle` kernels** (a pure GEMM, where the ring slab is the whole dynamic-smem footprint so the keeps-2 test is exact); a fused multi-bundle kernel (SDPA's QK+P@V) carries an intermediate cross-bundle workspace that dominates the materialized smem and is invisible to the ring-byte budget, so it keeps the shallow-first default (depth-2, always downstream-valid) — the autotuner still explores its deeper rings. |
| `TMA`             | BOOL     | `050_use_tma`                       | Promote BUFFERED/ASYNC bundles to TMA. `1` = force (hard-fail on ineligibility), `0` = skip the pass. Default on for Hopper+. |
| `ASYNC_COPY`      | BOOL     | `060_use_async_copy`                | Promote double-buffered (BUFFERED) bundles to cp.async (ASYNC). `0` = keep the synchronous double-buffer. Default on for sm_80+. |
| `PIPELINE_STAGES` | BOOL     | `080_pipeline_stages`               | Software-pipeline async-staged K-outer loops into prologue/main/epilogue. `0` = keep the depth-1 staged loop. |
| `WARP_SPECIALIZE` | BOOL     | `085_warp_specialize`               | Warp-specialize TMA staging: producer warps issue TMA, consumer warps wait + reduce. Autotune fork on depth-2 TMA rings. |
| `ATOMIC_FREE_SPLITK` | BOOL  | `017_atomic_free_splitk`            | Replace `SPLITK > 1`'s atomicAdd output with a workspace + sibling reduce kernel (deterministic accumulation). |

`BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2 set, char `i` = bit `i`), the keywords `"all"` / `"none"`,
or a decimal / `0x`-hex int clamped to the candidate width. `format_tuning_knobs` drops `BOOL` knobs from the rendered
`knobs=` line — they're treated as pass-presence markers, not values.

`HOIST_COMPUTE` is an autotune fork: `030_hoist_invariant_compute` emits both variants per fusable cone in a fixed
order (inline-fuse first as the greedy default — smaller smem, works on every architecture). Honors
`DEPLODOCK_HOIST_COMPUTE` for one-off pinning. `PAD_SMEM` follows the same shape in `070_pad_smem`: both polarities
fire whenever any source has a fixable conflict; the greedy run picks pad-on first. Honors `DEPLODOCK_PAD_SMEM` for
one-off pinning.

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is
deterministic. Pick a fresh prefix when adding a rule; the pass loader
ignores the prefix itself — it's only for ordering readability.

| Pass                       | What rules do                                                        |
|----------------------------|----------------------------------------------------------------------|
| `frontend/decomposition/`  | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused ops like `rms_norm`/`softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s. Each rule emits broadcast-explicit IR via `_broadcast.broadcast_to`. |
| `frontend/optimization/`   | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map — prevents trivial layout kernels from blocking fusion. |
| `loop/lifting/`            | `lift_*` rules wrap each surviving tensor primitive (elementwise / reduce / indexmap / gather) in a trivial one-op `LoopOp`. |
| `loop/fusion/`             | `split_shared_indexmap` (runs first) fuses a pure-indexmap `LoopOp` that fans out to ≥2 consumers into **all** of them in one rewrite — it inlines the producer's body into each consumer (reusing `splice_loop_ops`) and dissolves the producer via a single multi-output `Graph.splice` (`output={consumer_id: fused_id}`); a consumer the splicer can't take falls back to a private copy. Then `merge_loop_ops` splices the remaining adjacent single-consumer `LoopOp` pairs via `ir/loop/splicer.py::splice_graph`. The split is what lets the scalar-constant broadcasts torch.export folds into mask/RoPE scaffolding fold into their consumers (inlined as `float x = 0.0f;` literals) instead of surviving as standalone copy kernels (full Qwen3-Embedding-0.6B: 394 → 337 CUDA kernels). `dedup_loads` then drops identical `(input, index)` Loads within each fused body. Finally `stamp_loop_names` stamps `LoopOp.name` via `provenance.name_for` (e.g. `k_rms_norm_3f2a1b`) — runs last so the structural hash reflects the final fused body; the Tile dialect just forwards the name onto each emitted `TileOp` (and every dialect below copies it through unchanged). Shared `is_pure_indexmap` / `rename_write_output` helpers live in `_helpers.py`. |
| `lowering/tile/`           | Tile-IR structural passes — Stage formation, transport (cp.async / TMA), double-buffering, pipelining, smem padding. Order: `partition_loops` → `gate_splitk_residual` → `stage_inputs` → `hoist_staged_loads_above_mask` (lifts a masked-tile boundary `Cond(decoded < bound, ...)` from `010_partition_loops`: any K-pipeline stmt — `StageBundle` itself, plus `SerialTile` / `StridedTile` whose subtree carries one — is hoisted ABOVE the Cond so the cooperative load fires on every CTA thread (TMA's elected issuer / cp.async's full-CTA fan-out would otherwise be gated out). Un-staged gmem Loads in the hoisted body whose index references a gated var are wrapped in an inner `Cond(predicate, body=cone)` covering their forward SSA cone. Skips `==` Conds (the SPLITK invariant-compute guard) and bare Conds with no staged transport. Deterministic, no knob — split out of `020_stage_inputs` so the staging walk is uniform and the Cond-shape rewrite is focused) → `swizzle_blocks` (default-on
L2-friendly CTA swizzle for matmul-priority TileOps — stamps `GridTile.swizzle_group_m = DEPLODOCK_GROUP_M`,
default 8, so the renderer emits a Triton-canonical `blockIdx.x` remap; identifies matmul kernels via
`TileOp.knobs` rather than the axis-suffix convention because the body-normalizer renames axes by the time
the pass runs) → `unify_sibling_stages` (drops a `StageBundle` Source whose `buf` was already staged by a
prior sibling scope and reverts its consumer Loads back to gmem — keeps the fused RMSNorm + linear `x_smem`
single-allocation invariant when the matmul-side K_i, now visible as a reduce through transparent
`RegisterTile` wrappers, would otherwise re-stage `x`) → `hoist_invariant_compute` → `use_ring_buffers` →
`use_tma` → `use_async_copy` → `pad_smem` → `pipeline_stages` → `mark_unroll`. Coordination (split-K atomic-writes, cooperative-K Combine emission, broadcast-write guards) is no longer a separate pass: the materializer / Kernel-IR render derives those decisions from `ir/tile/escape_analysis.py` queries against the tile body. Cooperativity is derived from `Accum.axes ∩ ThreadTile.axes`; atomic writes from enclosing `GridTile.axes` vs `Write.index`. `015_gate_splitk_residual` reuses the same `Body.coordination.atomic_axes` signal to identify the split-K block axis without any axis-naming convention or role tag — when SPLITK > 1, it wraps a `matmul_add`-shape linear residual epilogue under `Cond(K_s == 0, ...)` so the residual is atomic-added exactly once across the K_s CTAs (rewrite + predicates live in sibling `_splitk_residual.py`, shared with `010_partition_loops`'s `force_splitk_one` enumeration-time gate). The partition planner's knob globals + per-mode candidate tuples + the pruned `(BN, BM, FM, FN, BK, SPLITK, BR)` cartesian generator + per-mode priority/score functions live in sibling `_enumeration.py` — `010_partition_loops.py` imports the `enumerate_cartesian` entry point and the `TileParams` cartesian element; tests can hit `_enumeration` directly without routing through `_plan_kernel`. `split_register_axes` / `permute_lane_accesses` used to live here but moved to `lowering/kernel/` once dtype-aware analytical passes consolidated there (see `plans/stamp-ssa-dtypes-and-reorder.md`); they still pattern-match `TileOp` because they run pre-materialize. |
| `lowering/kernel/`         | Pre-materialize dtype-aware analytical passes plus the final `TileOp → KernelOp` lowering. Order: `lower_atom_tile` (MMA-only: rewrites the `RegisterTile > AtomTile > matmul-cell-body` shape — emitted by the warp-tier matmul planner — into an MMA fragment chain, branching on `AtomSpec.instruction`: `"wmma"` → `MmaFragment` + `MmaFill` + per-K_i `MmaLoad`+`MmaSync` + final `MmaStore` (opaque `nvcuda::wmma`); `"mma_sync"` → `RegFragment` + per-K_i `LdmatrixLoad`+`MmaSyncPtx` + final `RegStore` (the s16816 `ldmatrix` + `mma.sync.aligned` register-array path — smem-staged only, no gmem-direct). The transform-walk skeleton is shared; only the leaf emitters (`_emit_fragments`/`_emit_chain`/`_emit_store`) differ. Strips the AtomTile wrapper. When the AtomTile body contains a `StageBundle` (the smem-staged MMA shape from `020_stage_inputs` on `AffineAddressing.block`-bearing operands), descends into the bundle, registers its Sources, and re-wraps the lowered Mma chain in the same bundle so the smem allocation + cooperative producer survive; ``MmaLoad.src_index`` is rebuilt per cache-axis (each ``Var * block``) and ``ldm`` resolves from the inner source dim's slab stride. Scalar TileOps skip; see `plans/mma-fragment-factorization.md` and `plans/mma-smem-staging.md`) → `split_register_axes` (replicates REGISTER-tagged bodies per-cell, with dep-tracked single-copy preservation of axis-invariant statements — for MMA kernels, replicates the Mma* chain per (M_r, N_r) cell, threading per-cell fragment SSA renames via the `Mma*.rewrite.register` handlers) → `dedup_replicated` (content-agnostic CSE: structurally identical Loads / Assigns left over after replication fold into one — the same shape the deleted blocked-GEMM builder used to produce by hand-partitioning N-invariant cones; see `plans/obsolete-blocked-gemm-builder.md`) → `place_inits` (places explicit `Init` Stmts at correct accumulator scope — descends into a `WarpTile`-wrapped `WarpSpecialize` to land the Init at the **consumer_body head**, above the consumer K loop and inside the role split; placing it higher would let the renderer's default per-loop init fire inside the loop and reset the accumulator every K chunk) → `stamp_types` (single body walk populating `Load.dtype` / `Assign.dtype` / `Write.value_dtype` / `Source.dtype` from `graph.nodes[buf].output.dtype`) → `demote_to_write_dtype` (folds f16-only chains feeding f16 Writes) → `vectorize_loads` (widens consecutive scalar Loads into LDS.128 / `__half2`) → `permute_lane_accesses` (chunks the N register tile into LDS.128-sized strips to remove bank conflicts on `FN > V`; skipped for MMA — `load_matrix_sync` handles its own swizzling) → `pack_fp16_pairs` (pairs scalar `__half` Inits/Accums into `__half2`; skipped for MMA — the C fragment IS the accumulator) → `vectorize_stores` (widens consecutive scalar Writes) → `flatten_wrap_stages` (flattens wrap-body `Stage(... body=[consumer])` into `[Stage(empty), *consumer]` so the materializer walks producer scaffolding then consumer siblings) → `materialize_tile` (purely-mechanical Tile → Kernel lowering; Smem decls read `Source.dtype` directly; its emit logic lives in sibling `_`-prefixed helper modules `_stage_expand` / `_combine` / `_tma_groups`, which the pass loader skips) → `drop_redundant_syncs` (Kernel-IR peephole collapsing back-to-back / leading `Sync`s at the tile-body level). All passes through `flatten_wrap_stages` pattern-match `TileOp`; `materialize_tile` consumes `TileOp` and produces the `KernelOp`; `drop_redundant_syncs` rewrites `KernelOp → KernelOp`. |
| `lowering/cuda/`           | `lower_kernelop` renders the `KernelOp` body to a `__global__` source string (via `ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

See `ir/ARCHITECTURE.md` for what each IR dialect looks like.

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph
uniformly for every pass: `NN_<pass_name>.{json,txt,dot}` (+
`NN_<pass_name>.kernels.txt` if any node has a non-empty
`pretty_body()`). Slashes in the pass name are flattened to
underscores, so `lowering/cuda` dumps as e.g. `06_lowering_cuda.*`. The
pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)` from the caller.

The uniform strategy means adding a new pass automatically gets dumped
— no per-pass registration needed. Rendering the kernel-level IRs
(loop/tile/kernel/cuda) lives in `format_kernels(graph)`, which calls
each op's own `pretty_body()`. Node ids accumulate `merged_` per
fusion step and a leading `lift_` from lifting; `_canonical_node_id`
collapses both for display, and the rendered body is rewritten with
the same map so `Load`/`Write` references match.

Per compute kernel, `_dump_per_kernel` also writes `<prefix>.kernels/<kname>.json` — a standalone sub-graph (kernel +
its `InputOp`/`ConstantOp` producers) loadable via `deplodock run --ir`. When op provenance is present (see
`compiler/provenance.py`), it additionally writes `<kname>.torch.json` + `<kname>.torch.txt`: the **original Torch ops**
that kernel implements, sliced from the pristine pre-decomposition graph stashed in `dump_input_graph`, with an `i/N`
coverage header per origin (full vs partial). Because the slice is taken from the original graph by origin id, it is
always made of whole Torch ops — runnable via `deplodock run --ir <kname>.torch.json --bench` to reproduce accuracy /
latency vs torch for exactly those ops.

## Per-rule diff output (`rule_diff.py`)

At `compile -vv` (DEBUG log level) the engine emits one block per rule application: a unified diff between the
matched subgraph before the rewrite and the rewritten fragment, bracketed by `>>> <pass>:NNN_rulename` and
`<<< <pass>:NNN_rulename` markers on their own lines. The `<pass>` prefix is the single-letter shorthand from
`PASS_SHORTHAND` (`d`=decomposition, `o`=optimization, `l`=lifting, `f`=fusion, `t`=tile, `k`=kernel, `c`=cuda) —
the same letters the CLI accepts in `--passes dolft`. Skipped rules (`RuleSkipped` exception) collapse to a
one-liner `--- <pass>:NNN_rulename skipped at <root>: <reason>`.

`PASS_SHORTHAND` is the single source of truth: `commands/compile.py` imports it to build its `--passes` shortcut
expander, so the CLI flag and the `-vv` marker prefix can never drift.

The bracketing makes per-rule and per-pass slicing trivial — `awk '/^>>> t:005/,/^<<< t:005/'` extracts a
single rule's diff and `awk '/^>>> t:/,/^<<< t:/'` extracts the entire tile-lowering pass. ANSI color (`+` green, `-` red, `@@` cyan) is applied only inside the diff body, so the markers
stay plain ASCII and `awk` matches reliably even on colored output. Color follows
`compile --color {auto,always,never}` (default `auto`: tty-aware, honors `NO_COLOR`); diff context and a
fallback line cap are tuned via `--diff-context N` and `--diff-max-lines N`. `RuleRenderConfig` is set once from
`commands/compile.py:handle_compile` via `rule_diff.set_config()`.

The structured `.rules.json` dump under `DEPLODOCK_DUMP_DIR` is unaffected — the diff is purely a presentation
layer for the human-readable `_rule_texts` channel.

## Fragment splice (`engine._apply_replacement`)

1. Walk the fragment in topo order. `InputOp` nodes forward their id to
   the existing graph node (external reference); non-Input nodes are
   added to the graph with fresh ids.
2. `replace_node(match.output or match.root_node_id, new_output)`
   rewires all consumers (and `graph.outputs` slots) from the old
   output to the fragment's output id.
3. Merge hints from every consumed node into the new output.
4. Remove consumed nodes and run `_remove_orphans` to drop any now-
   dangling constants/inputs.

## Invariants

- A rule module must not reach into the engine's internals; its
  interface is `PATTERN` + `rewrite(graph, match)`.
- `pipeline/` imports from `ir/` but never from `backend/`. Lowering
  rules produce IR; actually executing that IR is the backend's job.
