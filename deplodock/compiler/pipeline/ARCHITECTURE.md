# Pipeline Architecture

Pattern-based rewrite engine + pass directories + dump hooks.

## Modules

```
pipeline/
├── engine.py      # Pattern, Match, match_pattern, run_rule, run_pass, run_pipeline, splice
├── dump.py        # CompilerDump + on_pass dispatch
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

`Match.nodes: dict[str, str]` maps each pattern entry's name to the
matched node id. `Match.consumed` and `Match.output` are overridable by
the rewrite function to control which nodes the splicer removes and
which node's edges get rewired.

### Rule module convention

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]   # required
def rewrite(graph: Graph, match: Match) -> Graph | None:
    ...
```

- Returning a `Graph` fragment splices it in place of `match.output`
  (defaults to `match.root_node_id`); fragment `InputOp` nodes reference
  existing graph nodes by id, non-Input nodes get fresh ids.
- Returning `None` means "no-op / already mutated in place" — used by
  the lowering rules because `KernelOp.arg_order` / `CudaOp.arg_order`
  embed the original node id as the output buffer name, so a fresh id
  from splicing would break the generated kernel's buffer binding.

Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as
rules — they're shared helpers for the pass's rule modules.

### Drivers

- `run_rule(graph, rule_path)` — load one rule file, apply to fixed point.
- `run_pass(graph, pass_dir)` — load every rule file in a directory,
  apply each to fixed point, then rescan the sequence until no rule
  makes further progress.
- `run_pipeline(graph, passes, dump=None)` — run each named pass
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
| `lowering/tile/`           | `tileify` produces a `TileOp` per `LoopOp` (strips the outer free-Loop chain into `Tile.thread_axes` and lifts inner output-write free Loops); follow-up rules annotate scheduling decisions on the `Tile` (block/thread bindings, `Stage` nodes, `Combine`). Order: `split_matmul_k` → `cooperative_reduce` → `blockify_launch` → `stage_inputs` → `register_tile` → `double_buffer` → `tma_copy` → `async_copy` → `pad_smem_banks` → `pipeline_async` → `unroll_small_loops`. Staging runs before register tiling so the classifier sees clean PAT × PAT threads; `register_tile` keeps Stages singleton across F² (only consumer Loads multiply); `double_buffer` promotes K-outer Stages to `BufferedStage`; `tma_copy` narrows eligible BufferedStages to `TmaBufferedStage` (cp.async.bulk.tensor transport, sm_90+); `async_copy` narrows the rest to `AsyncBufferedStage` (cp.async transport, sm_80+); `pad_smem_banks` adds `Stage.pad` to break smem bank conflicts (skips `TmaBufferedStage` — TMA uses its own swizzling); `pipeline_async` rotates the K-outer loop into prologue + steady-state + epilogue with `AsyncWait` placement. |
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
