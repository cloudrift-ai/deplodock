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
| `kernel/ir`       | after `lowering/kernel`         | `GpuKernel`, `KernelOp`, statement/expression AST                                                     |
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
- **Loop → kernel** (after `lowering/kernel`): LoopOp nodes replaced by
  `KernelOp` carrying a `GpuKernel` AST. Same `node.id` preserved (see
  `pipeline/ARCHITECTURE.md`).
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors don't.    |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expr.py`

Shared expression sublanguage used by both loop IR (`Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`) and
kernel IR (`ArrayAccess.index`, `ForLoop.end`, …). Imports nothing from
other IR files.

| Symbol                                                             | Role                                                                     |
|--------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinOp`, `Builtin`, `FuncCall`, `Ternary`, `Cast`| Expression nodes. Each has `eval(env) → value/ndarray`.                  |
| `_ExprOps`                                                         | Mixin: Python operator overloading for expression building.              |
| `Expr`                                                             | Union type alias.                                                        |
| `PLACEHOLDER_PREFIX`, `placeholder`, `is_placeholder`              | Convention for output-coord placeholders in coord maps.                  |
| `substitute`                                                       | Tree rewrite: replace named `Var` nodes.                                 |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/decomposition/`; after that pass none of these
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
| `OpInfo`, `OP_REGISTRY`              | Elementwise arity / commutativity.                             |
| `ReduceInfo`, `REDUCE_REGISTRY`      | Reduce identity elements.                                      |

## `loop/`

One `LoopOp` = one GPU kernel described as an SSA program over named
iteration axes. Free vs reduce is inferred from body structure — a
`Loop` is a reduce Loop iff its body contains an `Accum`.

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`).                                                                      |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(source)[index...]`. `source` indexes into `node.inputs`.                    |
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseOp`.                                                        |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to `ACCUM_IDENTITY[op.fn]`.      |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Iteration block: `axis` + nested `body`.                                                                          |
| `Stmt`                       | Union: `Assign \| Accum \| Write \| Select \| Loop \| Load`.                                                      |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Stmt.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expr substitution).

### `loop/normalize.py` — structural canonicalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `canonicalize_free_axis_order` — sort outer free Loops by axis name.
- `eliminate_copy_aliases` — drop `y = copy(x)` Assigns.
- `unify_sibling_reduce_axes` — collapse sibling reduce Loops sharing
  an input dim (softmax's max + sum sweeps become one reduce axis).
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops.
- `rename_ssa_sequential` — cosmetic: Assign/Select names become `v0,
  v1, …` in definition order.

### `loop/simplify.py` — Expr simplification

Called inside `normalize_body`. Generic bottom-up Expr rewriter:
constant folding, algebraic identities, range-based comparison folding
(`(k0 > 2047 ? 2047 : k0) < 0 ? 0 : k0` → `k0`). `Context`/`Interval`
track integer ranges from axis extents.

Also used by `ir/kernel/normalize.py` for GpuKernel Expr simplification.

### `loop/splicer.py` + `loop/sigma.py` — LoopOp merger

The machinery `pipeline/passes/fusion/001_merge_loop_ops.py` calls to
splice adjacent `LoopOp` pairs. Sigma is axis-substitution bookkeeping
threaded through the merge.

### `loop/interpret.py` — numpy interpreter

`execute_loop_op(loop, input_arrays, out_shape) → ndarray` walks the
LoopOp body against pre-provided input arrays. Powers `LoopOp.forward`
— so post-fusion graphs run through the shared `interpret_graph` like
any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` helper used by decomposition/fusion tests to construct
LoopOp bodies without spelling out every `Loop(Axis(…))` nest.

## `kernel/`

### `kernel/ir.py` — C-like AST

| Group              | Symbols                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Expression (GPU)   | `ArrayAccess`, `Cast`, `FieldAccess`, `VectorLoad` + shared `Expr` types                                                                      |
| Statements         | `VarDecl`, `Assign`, `VarAssign`, `AugAssign`, `ForLoop`, `IfStmt`, `SyncThreads`, `ArrayDecl`, `PragmaUnroll`, `RawCode`                     |
| Kernel definition  | `GpuKernel` (AST + block_size + includes + launch_bounds), `GpuKernelParam`                                                                   |
| Graph-op wrapper   | `KernelOp` (carries a `GpuKernel` + grid/block/smem/zero_outputs/arg_order)                                                                   |
| Type alias         | `GpuExpr`                                                                                                                                     |
| Utilities          | `pretty_print`                                                                                                                                |

### `kernel/normalize.py`

Runs from `KernelOp.__post_init__`. Pushes integer ranges from
`VarDecl` inits (thread-index compositions), `IfStmt` conds, and
`ForLoop` bounds; delegates the core Expr rewrite to
`ir/loop/simplify.simplify_expr`.

## `cuda/ir.py`

| Symbol    | Role                                                                        |
|-----------|-----------------------------------------------------------------------------|
| `CudaOp`  | Graph-op carrying `kernel_source`, `kernel_name`, `arg_order`, grid, block, `smem_bytes`, `zero_outputs`, `comment`. Produced by `pipeline/passes/lowering/cuda`. |

## Graph as the single program form

There is no separate program type. A `Graph` is the execution plan:
node ids are buffer names, `node.output.shape` is the buffer shape,
`graph.topological_order()` is the launch order, and
`graph.inputs` / `graph.outputs` / `ConstantOp` membership gives each
buffer its role (input / output / constant / scratch).
