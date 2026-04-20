# Compiler Architecture

## Three-Layer Shape

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Frontend (backend-agnostic)                                                                               │
│                                                                                                                      │
│  torch_trace.py ─→ Graph populated with frontend ops                                                                 │
│      ir/graph.py:    Tensor, Node[T_Op], Graph, Hints                                                                │
│      ir/base.py:     Op, InputOp, ConstantOp                                                                         │
│      ir/frontend_ir.py: Torch-captured ops (LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp, TransposeOp,               │
│                      ReshapeOp, SliceOp, CatOp)                                                                      │
│      ir/tensor_ir.py:   minimal IR survives decomposition (ElementwiseOp, ReduceOp, IndexMapOp, ...)                    │
│      ir/expr.py:     Expr AST + coord_expr helpers                                                                   │
│                                                                                                                      │
│  RULE: No GPU, no CUDA, no backend imports.                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · Lowering → LoopProgram                                                                                    │
│                                                                                                                      │
│  pipeline.py: compile_graph(graph) -> LoopProgram                                                                    │
│    After fusion, ir/simplify.simplify_loop_op is applied to every LoopOp (const fold + clamp elimination).           │
│    LoopProgram (program/loop.py):                                                                                    │
│       LoopBuffer (shape, role) × N                                                                                   │
│       LoopLaunch (LoopOp + input/output buffer names) × N                                                            │
│       + graph_inputs/outputs/constants/constant_values                                                               │
│                                                                                                                      │
│  Each LoopOp (ir/loop/ir.py) is one GPU kernel as an SSA program:                                                       │
│      body   (tuple[Stmt, ...])        — Assign | Accum | Write | Select | Loop | Load (sole stored field)            │
│      axes   (tuple[Axis, ...])        — iteration space (computed from body's Loop tree)                             │
│      loads  (tuple[Load, ...])        — body-form external reads (computed from body)                                │
│      accums (tuple[Accum, ...])       — reduce accumulators (computed from body)                                     │
│  Reductions = ``Accum`` stmts inside a reduce ``Loop``; writes are inline.                                           │
│                                                                                                                      │
│  RULE: Operates on Graph + Loop IR only. No backend imports.                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 · Backends                                                                                                  │
│                                                                                                                      │
│  backend/base.py:  Backend ABC — compile(graph), run, benchmark                                                      │
│  backend/numpy/:   NumpyBackend — Graph interpreter (pre-fusion)                                                     │
│  backend/loop/:    LoopBackend  — LoopProgram interpreter (numpy)                                                    │
│  backend/cuda/:    CudaBackend  — LoopProgram → GpuProgram → nvcc                                                    │
│                                                                                                                      │
│  All three backends share:                                                                                           │
│    backend.compile(graph) → compiled                                                                                 │
│    backend.run(compiled, input_data=…) → ProgramResult (outputs=dict[name, ndarray], time_ms)                        │
│                                                                                                                      │
│  Codegen internals (CUDA):                                                                                           │
│    backend/kernel_codegen.py: GpuKernel → C source                                                                   │
│    backend/cuda/emit.py:      LoopProgram → GpuProgram                                                               │
│    backend/cuda/program.py:   CudaLaunch(GpuLaunch), nvcc, run                                                       │
│    backend/cuda/runner.py:    single-kernel compile + run harness                                                    │
│                                                                                                                      │
│  Program forms (shared across backends):                                                                             │
│      program/loop.py: LoopProgram + LoopBuffer + LoopLaunch                                                          │
│      program/gpu.py:  GpuProgram + GpuBuffer + GpuLaunch                                                             │
│  The imperative C-like kernel AST lives upstream in ir/kernel_ir.py (GpuKernel, Stmt, ArrayAccess, ...);                   │
│  the backend consumes it.                                                                                            │
│                                                                                                                      │
│  RULE: GPU specifics live here; everything above is portable.                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

See `ir/ARCHITECTURE.md` for the per-dialect breakdown of each IR level,
and `program/ARCHITECTURE.md` for the Program-form pairing.

## Canonical Data Flow

```
PyTorch module
   │  torch_trace.trace_module(...)
   ▼
Graph (Layer 1)
   │  backend.compile(graph)  — unified across numpy / loop / cuda
   │    ├─ NumpyBackend: wrap Graph; numpy walk in run
   │    ├─ LoopBackend:  compile_graph(graph) → LoopProgram; interpret in run
   │    └─ CudaBackend:  compile_graph(graph) → LoopProgram
   │                     compile_kernels(lp)  → GpuProgram
   ▼
LoopProgram (Layer 2: LoopBuffers + LoopLaunches)
   │  [CUDA path]
   │    for launch in loop_program.launches:
   │      emit.emit_kernel(launch, name, loop_program)  → GpuKernel
   │      kernel_codegen.emit_kernel(GpuKernel) → C source
   │      wrap as CudaLaunch, collect in GpuProgram
   ▼
GpuProgram (Layer 3)
   │  backend/cuda/program.generate_source(program)
   ▼
.cu  →  nvcc  →  GPU
```

## Structural LoopOp Cheat Sheet

SSA invariants are enforced by `LoopOp.__post_init__`: unique names,
defined-before-use, no forward references.

| Slot                                         | Type                                      | Used for                                                       |
|----------------------------------------------|-------------------------------------------|----------------------------------------------------------------|
| `Axis.name` / `extent` / `kind`              | `str` / `int` / `"free"                   | One iteration variable; reduce axes pair with accumulators     |
| `Load.name` / `source` / `index`             | `str` / `int` / `tuple[Expr, ...]`        | Body-form external read: ``name = load src[index...]``          |
| `Accum.name` / `value` / `op`                | `str` / `str` / `ElementwiseOp`           | Reduce fold: ``name = op(name, value)`` inside a reduce Loop   |
| `Assign.op`                                  | `ElementwiseOp`                           | Pure SSA body op; ReduceOp is NOT valid here                   |
| `Write.output` / `index` / `value`           | `int` / `tuple[Expr, ...]` / `str`        | Inline store at a position                                     |
| `Select.branches[i].value` / `select`        | `str` / `Expr`                            | Coord-predicated SSA binding (replaces old Mux)                |
| `LoopOp.body`                                | `tuple[Stmt, ...]`                        | Sole stored field; `axes` / `loads` / `accums` are computed    |
| `LoopLaunch.input_names`                     | `list[str]`                               | Per-source external buffer name (indexed by `Load.source`)     |
| `LoopLaunch.output_name`                     | `str`                                     | External buffer written by this LoopOp                         |

Analogies (use in module/class docstrings):

- **Named iteration axes** like Halide ``Var``/``RVar`` or MLIR
  ``linalg.generic`` iterator types.
- **Accumulator state** like MLIR ``scf.for`` iter_args — the
  ``Accum`` statement is the combine-and-write.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store) for `LoopOp`.

## Shape inference

Every ``Op`` subclass implements ``infer_output_shape(input_shapes) → tuple``.
The compiler never stores shapes redundantly — shapes are derived when
needed by calling ``infer_output_shape`` on the op instance.

| Op type                                               | Rule                                                                                                                                                                         |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ``ElementwiseOp``                                     | ``broadcast_shapes(*input_shapes)`` — NumPy right-aligned broadcast. Inputs MUST be broadcast-compatible; matmul decomposition inserts unsqueeze IndexMapOps to ensure this. |
| ``ReduceOp``                                          | Drop the ``axis`` dim from the input shape.                                                                                                                                  |
| ``IndexMapOp``                                        | Returns ``self.out_shape`` — the output shape is part of the op definition because it cannot be derived from the coord_map + input shape (e.g., reshape ``(12,) → (3, 4)``). |
| ``TransposeOp``                                       | Permute the input shape by ``self.axes``.                                                                                                                                    |
| ``ReshapeOp``                                         | Returns ``self.shape``.                                                                                                                                                      |
| ``LinearOp`` / ``MatmulOp`` / ``SdpaOp`` / ``MeanOp`` | High-level ops; decomposed before shape inference runs on the kernel IR.                                                                                                     |
| ``LoopOp``                                            | ``infer_output_shape()`` returns the tuple of free-axis extents.                                                                                                             |

**Invariant**: after decomposition + optimization, every ``ElementwiseOp("mul")``
in the graph has broadcast-compatible inputs. Matmul ``A(..., M, K) × B(..., K, N)``
is decomposed with unsqueeze IndexMapOps (``A → A[..., M, K, 1]``,
``B → B[..., 1, K, N]``) so the mul broadcasts to ``(..., M, K, N)`` and
the reduce over ``K`` (axis -2) yields ``(..., M, N)``. No special-case
shape inference for contractions.

## Fusion policy (rules/fusion/)

Lift-then-splice, driven by the rewriter's fixpoint loop:

1. **Lift each tensor op into a trivial one-op ``LoopOp``** — one rule
   per tensor op (``001_lift_elementwise``, ``002_lift_reduce``,
   ``003_lift_indexmap``, ``004_lift_gather``). Each produces a kernel
   whose iteration space matches its op's output shape, with axes named
   ``a0, a1, ...`` and a Write at the identity index. Body-form
   ``Load`` statements express each input's access pattern directly as
   ``Expr``s over axis Vars. Reductions contribute one ``Accum``
   statement inside a reduce ``Loop`` over the reduction axis;
   multi-source IndexMapOps (cat / concat) contribute a ``Select``
   prelude choosing among per-source Loads.
2. **Splice adjacent ``LoopOp`` pairs** (``005_merge_loop_ops`` calls
   ``_splice.py::splice_loop_ops``). Under the lift convention every
   kernel writes at an identity index, so the common case is a direct
   tree splice — replace each consumer ``Load(source=target)`` with the
   producer's body inlined. A minimal σ handles the two non-identity
   cases:
   - ``Var(p)`` in ``writer.index[k]`` → bind ``p → reader.index[k]``.
   - ``Literal(c)`` in ``writer.index[k]`` → no binding (keep-dim
     reduction slot, broadcast position).
   Anything else in the writer (``Var(p) ± c``, ``Cast``,
   multiplicative) → splice refuses and the kernels stay separate.
3. **Axis-alias detection** — when both producer and consumer have
   reduce axes that index the same external buffer at the same dim as
   bare ``Var``s with matching extents, the producer's reduce axis is
   aliased to the consumer's. This collapses ``sum(x, -1) + max(x, -1)``
   into a single kernel with two accumulators sharing one reduce sweep.
4. **Bake producer body as a tree** — walk ``producer.body``: strip each
   ``Loop`` whose axis is σ-bound or extent-1 (consumer iterates those;
   singletons are no-ops), keep surviving reduce Loops with renamed axes,
   apply σ / SSA-rename / source-remap to every stmt, drop the target
   ``Write``. The result preserves the producer's natural scope shape.
5. **Splice into consumer tree, per-stmt** — for each top-level producer
   baked stmt, compute its effective consumer-axis dependencies (direct
   Expr refs plus axis deps propagated via SSA references to earlier
   stmts). Each stmt's **splice parent** is the innermost consumer Loop
   whose axis sits in those deps. Walk consumer body once, and at each
   Loop prepend the stmts bucketed for that axis. Row-space reduce
   sweeps (deps ``{a0}``) land as siblings at ``Loop(a0)``; element-space
   producer Loads (deps ``{a0, a1}``) land inside ``Loop(a1)``. No
   flatten/renest pass — the output IR matches the schedule the emitter
   would otherwise hoist at codegen time.
6. **Fixpoint** — lift rules fire once per tensor op; the splice fires
   repeatedly on adjacent LoopOps. Patterns the splicer can't handle
   (multi-read — the consumer reads the producer at multiple distinct
   indices, e.g. softmax's sum sweep + final divide) simply stay as
   separate kernels. Copy-elimination and SSA canonicalisation are no
   longer fusion-pass rules — they live in ``ir/loop/normalize.py`` and
   run inside ``LoopOp.__post_init__`` on every constructed body.

The splicer subsumes two passes the old pipeline had as separate concerns:

- **Layout composition** (chained single-source IndexMapOps): the
  producer's ``coord_map`` survives as body Load indices; when the
  consumer's Load is an identity read, the splice trivially inlines
  without σ, so composed layouts come out correctly.
- **Absorption of identity-alias / keep-dim reductions**: keep-dim
  reduction writes use ``Literal(0)`` on the reduced dim — the splicer
  allows this without binding, and the reducer's ``Accum`` lands at the
  merged kernel's reduce scope naturally.

``optimization/002_insert_broadcast_indexmap.py`` survives because it
addresses a different concern — promoting implicit elementwise
broadcasts to explicit ``IndexMapOp``s — which the splicer does not
handle.

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_load_access(index, buf, src_shape, env) -> Expr`: emits `ArrayAccess(buffer, coord)` for a body `Load.index` pattern evaluated in the axis environment. Select statements inside the SSA body are handled separately by `_emit_select`, which lowers `SelectBranch`es into a nested `Ternary` chain.
- `_emit_body`: unified entry for one kernel body. Calls `ir.loop.plan.analyze_kernel(loop, out_shape)` to produce a `KernelPlan`, then delegates to `_emit_plan`. The same plan powers `LoopProgram.pretty_print_launch` so the dump view mirrors the loop nest codegen emits.
- `_emit_plan`: walks `plan.steps` — each step is either `Inline` (a straight-line block of `Assign` / `Select` / `Write` / `Accum` / `Load`) or `Loop` (a K-loop over a reduce axis). Loads inside a reduce Loop's body materialize inside that loop; Loads in prelude Inline steps happen once per row. Grid is 1D over flat free-axis extents; block is `(256, 1, 1)` for pure-elementwise plans, `(1, 1, 1)` for plans containing a `Loop` step. Empty bodies (copy kernels) are a pointwise subcase.
- Reductions (single `Loop` step) emit an accumulator init + `Accum` fold; max reductions use `fmaxf` instead of `AugAssign`. Softmax-style cross-iteration patterns (reduce_max → sub+exp → reduce_sum → div) and contractions (mul → reduce_sum) fall out naturally from multiple `Loop` steps recomputing per-element values as needed.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Numpy backend (`backend/numpy/backend.py`)

`NumpyBackend` derives from `Backend` (same ABC as `CudaBackend`).
Every `Op` subclass implements `forward(*inputs: np.ndarray) -> np.ndarray` — the
numpy equivalent of the tensor operation. `NumpyBackend.compile(graph)` stores the
graph; `run()` walks it in topological order, calling `forward` at each node,
reshaping outputs to match declared `node.output.shape`. No GPU required.

Covered ops: all elementwise functions, reductions, scans, gather/scatter,
transpose/reshape/unsqueeze/slice/cat, linear, matmul, SDPA, mean, plus
`IndexMapOp` and `LoopOp`. `LoopOp.forward` delegates to
`backend/loop/backend.py::execute_loop_op`, the same numpy interpreter that
`LoopBackend` uses — this lets `NumpyBackend` execute graphs *after* fusion,
so fusion rules can be validated numerically on CPU without CUDA.

## Testing

- `tests/compiler/test_ir.py`, `test_shape_inference.py`, `test_indexmap.py`, `test_backend_ir.py` — Layer 1 unit tests.
- `tests/compiler/test_kernel_op.py` — structural IR construction + invariant violations.
- `tests/compiler/test_lower.py` — `compile_graph` group-discovery + LoopOp assembly.
- `tests/compiler/test_emit.py` — recursive-descent codegen source-level assertions + on-GPU numerical checks.
- `tests/compiler/test_pipeline.py` — end-to-end on small synthetic graphs.
- `tests/compiler/test_torch_trace*.py`, `test_real_trace.py`, `test_hints.py` — tracer / hint coverage.
- `tests/compiler/test_torch_ops.py` — Op.forward() + numpy backend: per-op tests + torch cross-checks.
- `tests/compiler/rules/` — all rewrite rules (decomposition, optimization, fusion).

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
