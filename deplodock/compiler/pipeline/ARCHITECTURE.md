# Pipeline Architecture

Pattern-based rewrite engine + pass directories + dump hooks.

## Modules

```
pipeline/
├── engine.py      # Pattern, Match, match_pattern, run_rule, run_pass, splice
├── search/        # Autotune driver: Candidate, Search policies, run_pipeline / run_autotune, SearchDB + SearchTree
│   ├── candidate.py  # Candidate / Cursor / TraceEntry / RuleResult data classes
│   ├── policy/       # Search protocol (base.py) + GreedySearch (greedy.py) / TuningSearch (mcts.py)
│   ├── driver.py     # _search_loop + run_pipeline / run_autotune entry points
│   ├── db.py         # SearchDB SQLite store: loop/tile/kernel/cuda op inventory + lowering edges + perf
│   ├── recorder.py   # record_terminal: bench → DB persist → optional tree bump; op-JSON helpers
│   └── keys.py       # op_cache_key / dialect_of / source_chain (shared by db/policy/recorder)
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

- `run_rule(graph, rule_path)` — load one rule file, apply to fixed point.
- `run_pass(graph, pass_dir)` — load every rule file in a directory,
  apply each to fixed point, then rescan the sequence until no rule
  makes further progress.
- `run_autotune(graph, passes, search=..., backend=None) -> Iterator[Candidate]` —
  drive the full search. Yields one terminal `Candidate` per
  fully-explored branch. Two built-in searches: `GreedySearch` (used by
  `run_pipeline`; stops at the first terminal. At every fork point it
  consults `SearchDB.lowering` keyed by the rewritten parent op's
  `op_cache_key`: a previously-measured winner gets picked over option
  0; otherwise option 0 — the rule's heuristic-first ordering — wins.
  Decisions are memoized per compile so repeated parent keys don't
  hit the DB) and `TuningSearch` (used by `deplodock tune`; runs the queue
  dry, exploring every fork). Both rank candidates by
  remaining unmeasured ops (DFS-equivalent on a fresh DB). The greedy
  stop is self-detected by the search: `pop` returns `None` when the
  previously popped candidate didn't return via `push` — i.e. when the
  engine yielded it as terminal.
  When `search` exposes a `tree: SearchTree`, `run_autotune` calls
  `record_terminal(cand.graph, db, tree, ctx.structural_key(), backend=...)`
  on each yielded terminal. Pass a `Backend` (typically `CudaBackend`)
  via `backend=` to record real per-kernel GPU-event statistics
  (median + min/max/mean/variance over the per-iter samples carried on
  `LaunchTime.samples`); omit it to record the stub `latency_us=1.0`.
  See `search/candidate.py:Candidate`, `search/candidate.py:TraceEntry`,
  `search/policy/{base,greedy,mcts}.py:{Search,GreedySearch,TuningSearch}`,
  `search/db.py:SearchDB`, `search/tree.py:SearchTree`, and
  `search/recorder.py:record_terminal`.

### Search persistence: on-disk inventory vs in-memory MCTS

The autotune state is split across two cooperating modules:

- **`SearchDB`** (`search/db.py`) — SQLite store partitioned into six
  tables: `loop_op`, `tile_op`, `kernel_op`, `cuda_op` (one row per op
  encountered along any lowering chain, keyed by `op_cache_key`), a
  `lowering` edge table (one row per rewrite hop along the chain —
  Loop→Tile, every intra-Tile autotune step, Tile→Kernel, Kernel→Cuda
  — carrying the knob delta the rule stamped at that hop and a
  best-median upsert across every dialect, so `GreedySearch` can
  replay the full chain by matching forks against the delta at each
  fork point), and a backend-partitioned `perf` table carrying full
  stats
  (`latency_us_{median,min,max,mean,variance}`, `n_samples`,
  `backend`, `status`, `knobs`). Selection statistic is the median.
- **`SearchTree`** (`search/policy/mcts.py`) — pure-Python in-memory
  MCTS state, colocated with `TuningSearch` because MCTS is the only
  policy that reads it. Tracks `seen_terminals` / `failed_terminals`
  / `visits` / `total_reward` per node, propagated upward on every
  recorded terminal. Rebuilt fresh each process: the engine re-fires
  every rule on warm starts (see `engine.py:_try_one_rule` →
  `tree.expand(...)`), which re-creates the tree topology; cached
  `perf` rows ensure no re-bench. UCB priors don't survive across
  runs. `GreedySearch` has no tree and `recorder.record_terminal`
  short-circuits the tree bump when it receives `tree=None`.

`search/recorder.py:record_terminal` is the only module that knows
about all four parts (graph, DB, tree, backend). It does one
`backend.benchmark(...)` call per terminal graph, then walks
`Op.source` once to record op inventory + lowering edges + the `perf`
row, then bumps the tree.
- `run_pipeline(graph, passes, dump=None)` — single-graph convenience
  wrapper around `run_autotune`. Returns the first terminal candidate's
  graph; for deterministic rules that's the only one. Run each named pass
  directory in order. After each pass, dispatches `dump.on_pass(name,
  graph)` so dump hooks land at the right stage without the caller
  hard-coding which dump method belongs to which pass.

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
| `lowering/tile/`           | `tileify` produces a `TileOp` per `LoopOp` (strips the outer free-Loop chain into `Tile.thread_axes` and lifts inner output-write free Loops); follow-up rules annotate scheduling decisions on the `Tile` (block/thread bindings, `Stage` nodes, `Combine`). Order: `chunk_matmul_k` → `split_matmul_k` → `cooperative_reduce` → `blockify_launch` → `chunk_reduce` → `stage_inputs` → `register_tile` → `double_buffer` → `tma_copy` → `split_inner_for_swizzle` → `async_copy` → `pad_smem` → `pipeline_k_outer` → `mark_unroll`. `chunk_reduce` chunks non-matmul reduces (softmax, SDPA-reduce, RMSNorm) whose post-blockify candidate slab would exceed `stage_inputs`'s 16 KB cap — fires only when chunking would actually unblock staging. Staging runs before register tiling so the classifier sees clean PAT × PAT threads; `register_tile` keeps Stages singleton across F² (only consumer Loads multiply); `double_buffer` promotes K-outer Stages to `BufferedStage` (cross-loop SSA reads are allowed when they resolve to scope-level invariant names — e.g. SDPA's per-output reduce reading prior softmax `acc0` / hoisted `reciprocal(acc1)` — so all three SDPA-reduce reduces double-buffer + TMA on sm_90+); `tma_copy` narrows eligible BufferedStages to `TmaBufferedStage` (cp.async.bulk.tensor transport, sm_90+); `async_copy` narrows the rest to `AsyncBufferedStage` (cp.async transport, sm_80+); `pad_smem` adds `Stage.pad` to break smem bank conflicts (skips `TmaBufferedStage` — TMA uses its own swizzling); `pipeline_k_outer` rotates the K-outer loop into prologue + steady-state + epilogue with `AsyncWait` placement. |
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
