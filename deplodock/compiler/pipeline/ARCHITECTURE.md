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
│   ├── db.py         # SearchDB SQLite store: op inventory + lowering edges + perf
│   └── keys.py       # op_cache_key / dialect_of / source_chain
│ # SearchTree (in-memory MCTS state) lives in policy/mcts.py — MCTS is the only policy that reads it.
├── dump.py        # CompilerDump + on_pass dispatch
├── rule_diff.py   # Per-rule unified-diff renderer for ``compile -vv`` output
└── passes/
    ├── frontend/
    │   ├── decomposition/  # frontend ops → tensor-IR primitives
    │   └── optimization/   # IndexMap fusion before lift-to-loop
    ├── loop/
    │   ├── lifting/        # tensor ops → trivial LoopOp nodes
    │   └── fusion/         # merge adjacent LoopOp pairs (splice)
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

No production rule currently emits branch Forks — an early experiment
in the partition planner showed that a priority-rank-derived
`Fork.score` is a coarser MCTS prior than the proper `TileOp.score`,
so hierarchical exploration found best variants more slowly than the
flat list. The infrastructure stays because it's straightforward to
use once a richer per-level prior (or a search-space size that
amortizes the structural overhead) justifies it.

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

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the full pipeline with different knob
choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the configurations that
produced the lowest measured latency.

**Driving the loop.** `deplodock tune <model_or_ir | --code EXPR>` constructs a `TuningSearch(patience=N, ucb_c=C)`,
hands it to `Pipeline.tune(graph, search=...)`, and iterates terminals until the search's `stop_reason` fires. Each
terminal is one fully-lowered `Graph[CudaOp]` whose every kernel got `_bench_terminal`'d. The tuning database (default
`~/.cache/deplodock/autotune.db`, overridable via the `DEPLODOCK_TUNE_DB` env var) accumulates rows across runs;
calling `tune` again on the same expression resumes from the cached state instead of re-benching.

**Search dynamics.** SP-MCTS over the fork DAG with max-Q normalized UCB1 (see `search/policy/mcts.py`):

- **Selection** picks the child of the current node with the highest `Q_norm + ucb_c · sqrt(ln(parent_visits) / child_visits)`,
  where `Q_norm = child.best_reward / global_best_reward` and `reward = 1 / median_us` (so lower latency = higher reward).
  `child.score` is a rank-only structural prior the rule stamped via `TileOp.score` (no magnitude — only relative ordering).
- **Expansion** is implicit: `Pipeline.search` pops a node and runs one rule batch; every fork pushes one new child per
  alternative. The tree mirrors the graph's fork lineage.
- **Simulation** is the actual `backend.benchmark(...)` call on the terminal — one real GPU run per leaf.
- **Backprop** walks the popped candidate's `parent` chain up to the root, updating `visits` and `best_reward` so future
  UCB1 calls see the new max-Q.
- **Patience** counts terminals visited *since the last new global best*; when it exceeds `--patience N` (default 20),
  `TuningSearch.stop_reason` is set and `Pipeline.tune` exits.

**Reading the result.** `_bench_terminal` writes one `perf` row per CudaOp per `(context_key, backend)` keyed on
`op_cache_key`, plus a `lowering` edge per rewrite hop carrying the knob delta the rule stamped. A subsequent
`deplodock compile` / `deplodock run` (or `make bench-kernels-tuned`) auto-resolves the same DB path (env or default)
and replays the cached forks via `GreedySearch`, which walks the parent op's `op_cache_key` in `lowering` and follows
the best-known child at each step. No GPU bench is
required on the replay path when every kernel already has a `perf` row.

**Stub backend.** With `backend=None`, `_bench_terminal` short-circuits to `latency_us=1.0` and persists nothing — used by
test fixtures so `Pipeline.run`'s greedy replay doesn't clobber tuned rows with a stub when no GPU is available.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK`),
candidate `hints` (advisory — the rule still validates structural fit), and a short help string. Rules declare them as
module-level constants and stamp values into `TileOp.knobs` dicts; the autotuner reads those dicts back as the per-hop
knob delta in the `lowering` table. The registry (`knob.registry()`) auto-collects every `Knob` instance in every loaded
rule module — no manual registration.

**Pinning knobs from the environment.** Two equivalent forms:

- **Per-knob:** `DEPLODOCK_<NAME>=<value>` (e.g. `DEPLODOCK_BK=32`). Read directly by the rule that owns the knob.
- **Aggregate:** `DEPLODOCK_KNOBS="K1=V1,K2=V2,..."` (e.g. `DEPLODOCK_KNOBS="BK=2,BM=16,BN=128,FM=8,FN=8,STAGE=111"`).
  Parsed once at `knob.py` import via `apply_knobs_env()`, which splats each entry into the corresponding
  `DEPLODOCK_<K>` env var so all the per-knob readers pick it up uniformly. An explicit per-knob var wins over the
  aggregate (so `DEPLODOCK_BK=4 DEPLODOCK_KNOBS="BK=2,BM=16"` ends up with BK=4, BM=16).

Pinning replaces tuner choice: the rule sees the env value and emits exactly that variant instead of forking. Useful for
reproducing a tune-time variant from CI logs, A/B-comparing two configs, or pinning a known-good config in a Makefile
recipe.

**Registered knobs.** All knobs in `passes/lowering/tile/*.py`:

| Knob          | Type     | Owning rule                  | What it controls                                                                                  |
|---------------|----------|------------------------------|---------------------------------------------------------------------------------------------------|
| `BK`          | INT      | `002_chunk_matmul_k`         | Per-stage K-chunk size for matmul reductions; intra-CTA K-loop trip count = `K / BK`.             |
| `SPLITK`      | INT      | `003_split_matmul_k`         | Cross-CTA K-split factor for matmul; `1` = no split. Multiplies CTA count, requires a final combine. |
| `BN`          | INT      | `004_launch_geometry`        | CTA innermost THREAD-axis width (the column tile each warp covers).                               |
| `BM`          | INT      | `004_launch_geometry`        | CTA outer THREAD-axis width (matmul only — the row tile each warp covers).                        |
| `STAGE`       | BINMASK  | `002_stage_inputs`           | Bitmask over ranked candidate buffers — char `i` = stage buffer `i`. Selected buffers fold into one wrap-body Stage with per-source Source entries. |
| `FM`          | INT      | `008_register_tile`          | Register-tile factor for the next-outer tilable nest level (per-thread row tile).                 |
| `FN`          | INT      | `008_register_tile`          | Register-tile factor for the innermost tilable nest level (per-thread column tile).               |
| `TMA_SWIZZLE`     | BOOL     | `011_tma_copy`                       | Enable TMA hardware-swizzle modes (B128 / B64 / B32); default off.                                |
| `FUSED_PIPELINE`  | BOOL     | `007b_hoist_invariant_compute`       | False (default) → inline-fuse Stage; True → ComputeStage + transports. Autotune fork.             |
| `PAD_SMEM`        | BOOL     | `014_pad_smem`                       | True → apply per-source ``+1`` smem pad to break bank conflicts; False → leave the slab dense. Autotune fork. |

`BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2 set, char `i` = bit `i`), the keywords `"all"` / `"none"`,
or a decimal / `0x`-hex int clamped to the candidate width. `format_tuning_knobs` drops `BOOL` knobs from the rendered
`knobs=` line — they're treated as pass-presence markers, not values.

`FUSED_PIPELINE` is an autotune fork: `007b_hoist_invariant_compute` emits both variants per fusable cone in a fixed
order (inline-fuse first as the greedy default — smaller smem, works on every architecture). No env override; the
autotuner is the only mechanism for picking the hoist variant. `PAD_SMEM` follows the same shape in `014_pad_smem`:
both polarities fire whenever any source has a fixable conflict; the greedy run picks pad-on first. Honors
`DEPLODOCK_PAD_SMEM` for one-off pinning.

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is
deterministic. Pick a fresh prefix when adding a rule; the pass loader
ignores the prefix itself — it's only for ordering readability.

| Pass                       | What rules do                                                        |
|----------------------------|----------------------------------------------------------------------|
| `frontend/decomposition/`  | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused ops like `rms_norm`/`softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s. Each rule emits broadcast-explicit IR via `_broadcast.broadcast_to`. |
| `frontend/optimization/`   | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map — prevents trivial layout kernels from blocking fusion. |
| `loop/lifting/`            | `lift_*` rules wrap each surviving tensor primitive (elementwise / reduce / indexmap / gather) in a trivial one-op `LoopOp`. |
| `loop/fusion/`             | `merge_loop_ops` splices adjacent `LoopOp` pairs using `ir/loop/splicer.py::splice_graph`. |
| `lowering/tile/`           | `partition_planner` runs first and emits `Role` tags on body Loops (`REGISTER`, `STAGE_INNER`, `SERIAL_OUTER`, `PIPELINE`, `COOPERATIVE_STRIDE`) — see `ir/axis.py::Role`. `tileify` produces a `TileOp` per `LoopOp` (strips the outer free-Loop chain into `Tile.thread_axes`, lifts inner output-write free Loops). `register_tile_planned` runs *before* `stage_inputs` to replicate REGISTER-tagged bodies per-cell. `stage_inputs` emits wrap-body `Stage(sources=[...], body=<consumer>)` — the producer cooperative load is synthesized at materialize time from per-Source `(name, buf, cache_dims, origin)` data; the Stage's body is the consumer subtree (Loop(K_i, reduce, ...)). Order: `partition_planner` → `tileify` → `register_tile_planned` → `stage_inputs` → `permute_register_tile` → `hoist_invariant_compute` (**stubbed**) → `register_tile` → `double_buffer` → `tma_copy` (**stubbed**) → `split_inner_for_swizzle` (**stubbed**) → `async_copy` → `pad_smem` → `pipeline_k_outer` (**stubbed**) → `mark_unroll`. **Stage-wrap-body refactor (in progress):** the bucket 7/10/11/12 optimization passes (hoist / pipelining / buffering / TMA) are stubbed during the refactor; kernels still produce correct output through the sync wrap-body path, just without ring-buffering, async/TMA transport, or temporal pipelining. The rewrites are the remaining bucket work — see `plans/stage-wrap-body.md`. |
| `lowering/kernel/`         | `materialize_tile` consumes scheduling decisions and emits hardware primitives (`Smem`, `Sync`, `TreeHalve`, `StridedLoop`), mutating the node's op to `KernelOp` in place. |
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
