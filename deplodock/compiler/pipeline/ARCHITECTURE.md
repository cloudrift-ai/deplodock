# Pipeline Architecture

Pattern-based rewrite engine + pass directories + dump hooks.

## Modules

```
pipeline/
├── engine.py      # Pattern, Match, match_pattern, run_rule, run_pass, run_pipeline, splice
├── dump.py        # CompilerDump + on_pass dispatch
└── passes/
    ├── decomposition/    # frontend ops → tensor-IR primitives
    ├── optimization/     # IndexMap fusion before lift-to-loop
    ├── fusion/           # tensor ops → LoopOp nodes + adjacent LoopOp merge
    └── lowering/
        ├── kernel/       # LoopOp → KernelOp (emit GpuKernel AST)
        └── cuda/         # KernelOp → CudaOp (render source string)
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

| Pass             | What rules do                                                        |
|------------------|----------------------------------------------------------------------|
| `decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused ops like `rms_norm`/`softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s. Each rule emits broadcast-explicit IR via `_broadcast.broadcast_to`. |
| `optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map — prevents trivial layout kernels from blocking fusion. |
| `fusion/`        | `lift_*` rules wrap each surviving tensor op in a trivial `LoopOp`; `merge_loop_ops` splices adjacent `LoopOp` pairs using `ir/loop/splicer.py::splice_graph`. |
| `lowering/kernel/` | Emit per-node `GpuKernel` AST (via `_emit.py`) and mutate the node's op to `KernelOp` in place. |
| `lowering/cuda/`   | Render the `GpuKernel` to a `__global__` source string (via `_emit.py`) and mutate the node's op to `CudaOp` in place. |

See `ir/ARCHITECTURE.md` for what each IR dialect looks like.

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(pass_name, graph)` dispatches to the dump methods
registered for that pass name:

| `pass_name`        | Dumps                                         |
|--------------------|-----------------------------------------------|
| `optimization`     | `10_tensor_ir.{json,txt,dot}`                 |
| `fusion`           | `20_fused_graph.json`, `20_loop_ir.txt`       |
| `lowering/kernel`  | `39_kernel_ir.txt`                            |
| `lowering/cuda`    | `40_kernels.cu`                               |

Other pass names (`decomposition` etc.) are no-ops. Pre-pass inputs are
dumped separately via `dump.dump_input_graph(graph)` from the caller.

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
