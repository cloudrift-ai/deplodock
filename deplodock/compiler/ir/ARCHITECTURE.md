# IR Dialects

Per-dialect op definitions. A `Graph` (`compiler/graph.py`) hosts nodes
from every dialect; the population shifts as passes run. For the
top-level layer/pass picture see `compiler/ARCHITECTURE.md`.

## Dialects at a glance

| Dialect           | When populated                  | Ops                                                                                                   |
|-------------------|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `base`            | always                          | `Op` (base), `InputOp`, `ConstantOp`                                                                  |
| `frontend/ir`     | after tracing                   | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp`, `UnsqueezeOp`, `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp` |
| `tensor/ir`       | after decomposition             | `ElementwiseOp`, `ReduceOp`, `ScanOp`, `GatherOp`, `ScatterOp`, `IndexMapOp`                          |
| `loop/ir`         | after fusion                    | `LoopOp` + body types (`Load`, `Assign`, `Accum`, `Write`, `Select`, `Loop`, `Axis`)                  |
| `tile/ir`         | after `lowering/tile`           | `TileOp` + scheduling stmts (`Tile`, wrap-body `Stage`/`BufferedStage`/`AsyncBufferedStage`/`TmaBufferedStage`/`ComputeStage` carrying `Source`/`CacheDim` per-operand layouts, `StridedLoop`) |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Invariants by stage

- **Frontend → tensor** (after `decomposition`): `LinearOp`, `MatmulOp`,
  `SdpaOp`, `MeanOp`, and the layout ops are gone. Only
  `ElementwiseOp`, `ReduceOp`, `IndexMapOp`, scan/gather/scatter, plus
  boundaries survive. (The broadcast-explicit invariant for
  `ElementwiseOp` inputs lives in `compiler/ARCHITECTURE.md`.)
- **Tensor → loop** (after `fusion`): only `LoopOp` + boundaries.
  Tensor-IR ops survive only *inside* `LoopOp.body` as `Assign.op` or
  `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`).
- **Loop → tile** (after `lowering/tile`): `LoopOp` nodes replaced by
  `TileOp` whose body is a `Tile` carrying scheduling decisions
  (`BIND_THREAD`/`BIND_BLOCK` axes, `Stage`). Cooperative-reduce
  emission, atomic-write classification, and broadcast-write guards
  are derived by ``escape_analysis`` at materialize / render time
  rather than carried as explicit Stmts.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body uses hardware primitives (`Smem`, `Sync`,
  `TreeHalve`, `StridedLoop`).
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.

`Op.source` is the rewrite-chain predecessor — the engine's
`_apply_one` stamps it on every 1:1 in-place rebind, so a fully
lowered `CudaOp` carries the full chain back to its originating
`LoopOp` (`cuda.source.source.source`) without any rule needing to
pass it explicitly. The base-class field is keyword-only and
`compare=False`, so subclass positional construction and equality
keep working unchanged. `source` is excluded from
`Graph.structural_key` and from `op_cache_key` — kernels rendered
along different lowering paths still dedup in the tuning cache.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expr.py`

Shared expression sublanguage used by every IR layer: `Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`,
`StridedLoop.start`/`step`, `Cond.cond`, etc. Imports nothing from
other IR files.

| Symbol                                                                       | Role                                                                     |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinaryExpr`, `Builtin`, `FuncCallExpr`, `TernaryExpr`, `CastExpr` | Expression nodes. Each has `eval(env) → value/ndarray`, `pretty()`, `substitute(mapping)`, `free_vars()`. |
| `_ExprOps`                                                                   | Mixin: Python operator overloading for expression building; default `NotImplementedError` for `pretty`/`substitute`/`free_vars`. |
| `Expr`                                                                       | Union type alias.                                                        |
| `PLACEHOLDER_PREFIX`, `placeholder`, `is_placeholder`                        | Convention for output-coord placeholders in coord maps.                  |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/frontend/decomposition/`; after that pass none of these
remain.

| Group         | Ops                                                                                       |
|---------------|-------------------------------------------------------------------------------------------|
| Layout-only   | `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp`, `UnsqueezeOp` — rewrite to `IndexMapOp`.  |
| Compound math | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp` — rewrite to elementwise + reduce chains.      |

## `tensor/ir.py`

Minimal IR fusion consumes. `IndexMapOp` is the unified layout-only op;
it replaces the frontend layout ops via `coord_map` expressions.

| Symbol                               | Role                                                           |
|--------------------------------------|----------------------------------------------------------------|
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`exp`/…).             |
| `ReduceOp`                           | Collapse one axis via associative binary op.                   |
| `ScanOp`                             | Cumulative variant of reduce.                                  |
| `GatherOp`, `ScatterOp`              | Data-dependent reads / writes.                                 |
| `IndexMapOp` + `IndexSource`         | Unified layout-only op over `Expr`.                            |

Op metadata (arity / commutative / reduce identity) lives on
`ElementwiseImpl` in `ir/elementwise.py` — the single source of truth
shared across elementwise, reduce, scan, and accumulator use sites.

## `loop/`

One `LoopOp` = one GPU kernel described as an SSA program over named
iteration axes. Free vs reduce is inferred from body structure — a
`Loop` is a reduce Loop iff its body contains an `Accum`.

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`). Defined in `ir/axis.py`, re-exported here. Carries optional `source_axis` (pre-split origin) and `real_extent` (pre-ceil-div bound for masked tiles — block axis covers `ceil_div(real_extent, BN·FN)`; materializer reads it to gate boundary lanes). Both excluded from equality / hashing. |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(input)[index...]`. `input` matches the producing graph node's id.           |
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseImpl`.                                                      |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expr.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Init`                       | Explicit accumulator initialization at an outer scope (matmul chunked-K).                                         |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expr` predicate.                                                                           |
| `Stmt`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt.py`.              |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Stmt.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expr substitution),
`Stmt.pretty(indent)` (rendered lines for kernel dumps; block stmts
recurse via `pretty_body`).

### `loop/normalize.py` — structural canonicalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `canonicalize_free_axis_order` — sort outer free Loops by axis name.
- `eliminate_copy_aliases` — drop `y = copy(x)` Assigns.
- `unify_sibling_reduce_axes` — rename sibling reduce Loops whose
  reduce-axis Load positions overlap on any `(source, dim)` pair so
  they share one canonical axis name (softmax's max + sum sweeps; the
  two matmul reductions in `silu(x@Wg) * (x@Wu)` that both index `x`
  at the same K slot). Union-find groups all transitively-overlapping
  Loops at one scope.
- `merge_sibling_reduce_loops` — concatenate sibling reduce Loops that
  share `axis.name` / `extent` into one Loop body. Gated on disjoint
  SSA defs across the two halves, the second body not reading any name
  the first body defines (blocks softmax-style sequential reduces
  where sum-exp reads `acc_max`), and no between-stmt def consumed by
  the second loop. Eliminates the duplicate K traversal in patterns
  like `silu(x@Wg) * (x@Wu)`; downstream `dedup_loads` then collapses
  the duplicate `x` loads, and the lowering passes stage both weight
  tensors symmetrically.
- `split_invariant_divides` — rewrite `divide(x, y)` into
  `reciprocal(y) + multiply(x, recip)` when `y` is loop-invariant
  w.r.t. some axis `x` depends on, so the rcp can hoist out of the
  inner loop and the per-iter cost drops from XU divide to FMA
  multiply.
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops.
- `rename_ssa_sequential` — cosmetic: Assign/Select names become `v0,
  v1, …` in definition order.
- `canonicalize_buffer_names` — rename `Load.input` / `Write.output` to
  `b0, b1, …` in encounter order. Off by default (buffer names bind to
  graph nodes) — opt in via `normalize_body(..., canonical_buffers=True)`.
  Used by `Body.structural_key()` for dedup queries where buffer identity
  doesn't matter.
- `sort_commutative_args` — sort `Assign.args` for commutative ops
  (`add` / `multiply` / `maximum` / `minimum`) so two bodies that
  differ only by argument order land in the same canonical form.
  Runs last so the sort key is the post-rename canonical SSA / buffer
  names.

`Body.structural_key()` re-runs `normalize_body(self, hoist=False,
canonical_buffers=True)` and joins `pretty_body`'s line list — a
`cached_property` returning the canonical text rendering. Two bodies
that differ only by SSA / axis names, commutative-arg order, or
external-buffer names produce the same key. Use it as a dict key /
set member when deduping candidate bodies in a search.

### `loop/simplify.py` — Expr simplification

Called inside `normalize_body`. Generic bottom-up Expr rewriter:
constant folding, algebraic identities, range-based comparison folding
(`(k0 > 2047 ? 2047 : k0) < 0 ? 0 : k0` → `k0`). `Context`/`Interval`
track integer ranges from axis extents.

Also used by `ir/kernel/normalize.py` for GpuKernel Expr simplification.

### `loop/splicer.py` — LoopOp merger

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to
splice adjacent `LoopOp` pairs. `Sigma` (from `ir/sigma.py`) is the
axis-substitution bookkeeping threaded through the merge.

### `loop/interpret.py` — numpy interpreter

`execute_loop_op(loop, input_arrays, out_shape) → ndarray` walks the
LoopOp body against pre-provided input arrays. Powers `LoopOp.forward`
— so post-fusion graphs run through the default `Backend.run`
topo-walk like any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` helper used by decomposition/fusion tests to construct
LoopOp bodies without spelling out every `Loop(Axis(…))` nest.

## `tile/`

Tile IR encodes scheduling decisions structurally — `Tile.axes` carry
`BIND_THREAD` / `BIND_BLOCK` bindings, `Stage` wraps consumer subtrees
that read smem-cached operands. Coordination decisions are derived from
the body at materialize / render time via ``ir/tile/escape_analysis.py``:
cooperative-reduce combine emission from ``Accum.axes ∩ ThreadTile.axes``,
atomic-write classification from enclosing ``GridTile.axes`` vs
``Write.index``, broadcast-write guards from cooperative thread axes vs
``Write.index``. No explicit coordination stmt or per-tile tag carries
this information. Compute leaves (`Load` / `Assign` / `Accum` / `Write`)
and control flow (`Loop` / `StridedLoop` / `Cond`) come from `ir/stmt.py`;
``Accum.axes`` carries the names of the loops being reduced over and is
the source of truth for cooperativity.

**Wrap-body Stage:** `Stage` is a block-structured Stmt whose `body` is
the *consumer* subtree using the staged smem. The producer (cooperative
`Load+Write` per source) is synthesized at materialize time from
`Stage.sources` — each `Source` carries `name` (smem buffer), `buf`
(gmem operand), `cache_dims`, `origin`, and optional `pad` /
`template_index`. Multi-source Stages (e.g. matmul A+B) load both
behind one sync boundary; sibling Stages exist only when scopes differ.

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `TileOp`              | Graph-op carrying a `Tile`-rooted body. One per kernel.                                                                              |
| `Tile`                | Axis-bound scope wrapper (`axes: tuple[BoundAxis, ...]` + body).                                                                     |
| `Stage`               | Wrap-body cooperative stage. ``sources: tuple[Source, ...]`` carries per-operand smem layouts; ``body`` is the consumer subtree. Materialize emits leading `Sync` + per-source cooperative `Load+Write` + trailing `Sync` + materialized consumer body. |
| `BufferedStage`       | `Stage` subtype with `buffer_count >= 2` rotating slabs selected by `phase`. Sync transport, ping-pong slabs (no leading `Sync`).    |
| `AsyncBufferedStage`  | `BufferedStage` subtype using `cp.async` per source; emits `CpAsyncCopy`+`CpAsyncCommit`. `pipeline_depth>1` marks the stage for temporal pipelining (prologue/main/epilogue), expanded by the bucket-10 pass. |
| `TmaBufferedStage`    | `BufferedStage` subtype using TMA (`cp.async.bulk.tensor` + mbarrier). `pipeline_depth>1` for pipelining; `swizzle` for shared-memory swizzle pattern. |
| `ComputeStage`        | `Stage` subtype for hoisted invariant compute. Carries a separate `compute: Body` (per-thread compute reading sibling-stage smem, writing this stage's smem). `external_reads()` returns `()` so 015 / tuning don't classify sibling-smem reads as gmem dependencies. |
| `Source`              | One gmem operand staged into one smem slab. Fields: `name`, `buf`, `cache_dims`, `origin`, `pad`, `template_index`. Each Stage carries one or more. |
| `CacheDim`            | One cache (smem) axis paired with the source-buffer dim it maps to. `Source.cache_dims` is a tuple of these.                          |
| `AffineAddressing`    | Property-derived addressing variant: `source_index[d] = origin[d] + decoded_coord(dims[i] == d)`. Fast path; no symbolic substitution. |
| `TemplateAddressing`  | Property-derived addressing variant: source index expressed verbatim with cache-axis Vars; materialize Sigma-substitutes them. Used for collapsed-reshape views. |
| `AsyncWait`           | Explicit wait carrier for pipelined async / TMA schedules. Emitted by `080_pipeline_stages` between issue / consume halves of each steady-state K_o iter and at the epilogue drain. ``keep`` is the cp.async ``wait_group`` arg; ``phase`` / ``slot`` are TMA mbarrier-test args. Sync-style (``pipeline_depth==1``) stages don't carry one — the materializer emits an implicit wait at the wrap boundary. |

## `kernel/`

### `kernel/ir.py` — fully-scheduled kernel form

Reuses `Tile` + leaf stmts from Tile IR; adds hardware primitives
materialized from scheduling decisions. `KernelOp` carries the body
directly (no separate AST class).

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `KernelOp`         | Graph-op wrapper around a `Tile`-rooted body. One per kernel.     |
| `Smem`             | `__shared__` array allocation (name + dtype + extents).           |
| `Sync`             | `__syncthreads()` barrier.                                        |
| `TreeHalve`        | Cross-thread tree reduction over a smem buffer.                   |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt.py`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Write`, `Select`, `Cond`. |

## `cuda/ir.py`

| Symbol    | Role                                                                        |
|-----------|-----------------------------------------------------------------------------|
| `CudaOp`  | Graph-op carrying `kernel_source`, `kernel_name`, `arg_order`, `grid`, `block`, `smem_bytes`, `zero_outputs`, `comment`. Produced by `pipeline/passes/lowering/cuda` (renders the `KernelOp` body to a `__global__` source string). |

## Graph as the single program form

There is no separate program type. A `Graph` is the execution plan:
node ids are buffer names, `node.output.shape` is the buffer shape,
`graph.topological_order()` is the launch order, and
`graph.inputs` / `graph.outputs` / `ConstantOp` membership gives each
buffer its role (input / output / constant / scratch).
