# IR Architecture

One file per IR level. Each dialect defines the ops that are legal in the
graph at a specific point in the compilation pipeline. A single ``Graph``
(``graph.py``) hosts nodes from every dialect; as rewrite passes run, the
population of op types in the graph changes.

## Pipeline

Layer numbering matches `compiler/ARCHITECTURE.md`: **Layer 1** is the
backend-agnostic Graph IR (pre- and post-decomposition), **Layer 2** is
the Loop IR produced by fusion/lowering, **Layer 3** is the GPU IR
consumed by the backend.

```
PyTorch module
   │  torch_trace
   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 · Frontend — Graph populated with FRONTEND ops (frontend.py)                                                 │
│   LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp, TransposeOp, ReshapeOp, SliceOp, CatOp                            │
│   + InputOp / ConstantOp boundary sentinels (base.py)                                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │  rules/decomposition/*
   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 · Frontend — Graph populated with TENSOR / MINIMAL ops (tensor.py)                                           │
│   ElementwiseOp, ReduceOp, ScanOp, GatherOp, ScatterOp,                                                              │
│   IndexMapOp (subsumes Slice / Cat / Transpose / Reshape / Unsqueeze via coord_map over expr.py)                     │
│   + InputOp / ConstantOp                                                                                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │  rules/optimization/*  (broadcast insertion)
   │  rules/fusion/*        (lift each tensor op → LoopOp, then merge adjacent LoopOp pairs)
   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 · Lowering — Graph populated with LOOP IR ops (loop.py)                                                      │
│   LoopOp (one per GPU kernel) — SSA program over named Axes. Ports read external buffers via Expr index              │
│   patterns; Accum stmts carry accumulator state; body statements (Assign/Accum/Write/Select/Load) execute            │
│   in order. + InputOp / ConstantOp as buffer sources.                                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │  compile_graph → LoopProgram (program/loop.py)
   │  backend/cuda/emit.compile_kernels
   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 · Backends — GPU IR (gpu.py) — one GpuKernel per kernel                                                      │
│   Imperative C-like AST: VarDecl, Assign, ForLoop, IfStmt, ArrayAccess, Cast, VectorLoad, ...                        │
│   Wrapped into GpuProgram (program/gpu.py) for execution.                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │  backend/kernel_codegen.py
   ▼
C/C++ source → nvcc → GPU
```

## Per-file summary

### `base.py` — shared root (all layers)

Cross-cutting types used at every stage.

| Symbol        | Role                                                                             |
|---------------|----------------------------------------------------------------------------------|
| ``Op``        | Base class. Subclasses implement ``infer_output_shape`` and ``forward`` (numpy). |
| ``InputOp``   | Sentinel: graph input tensor. Value supplied by executor.                        |
| ``ConstantOp``| Sentinel: weights / RoPE tables / scalar constants.                              |
| ``_keepdim_axis``| Shape helper — shared by ``ReduceOp`` (tensor) and ``MeanOp`` (frontend).        |

**Rule:** No dependencies on other IR modules. Everyone may import from here.

### `graph.py` — container + hints (Layers 1–2)

Hosts nodes from every dialect as the rewriter progresses.

| Symbol            | Role                                                                |
|-------------------|---------------------------------------------------------------------|
| ``Tensor``        | Name + shape + dtype. One per node output.                          |
| ``Node[T_Op]``    | Wraps one ``Op``. Generic on op subtype.                            |
| ``Graph``         | DAG. Insertion, removal, topo order, (de)serialization.             |
| ``Hints``         | Advisory metadata bag (dotted keys, e.g. ``cuda.matmul.strategy``). |
| ``resolve_hints`` | Merge graph + node hints.                                           |

**Rule:** Imports ``base``; lazy-imports ``frontend``/``tensor``/``loop``
inside ``Graph.from_dict`` for op-class reconstruction.

### `simplify.py` — generic Expr / IR simplifier (all layers)

Pure, idempotent bottom-up rewrite over the shared ``Expr`` AST plus the
GPU-specific extensions. Applied at every pipeline stage that emits IR
the user might inspect: after fusion (on ``LoopOp`` Expr fields) and
after kernel emission (on ``GpuKernel`` statement trees).

| Symbol                                       | Role                                                                                  |
|----------------------------------------------|---------------------------------------------------------------------------------------|
| ``Interval`` / ``Context``                   | Integer-range context threaded through the walk (Axis extents / ForLoop bounds / IfStmt conds). |
| ``simplify_expr(e, ctx)``                    | Core Expr rewriter: const fold, algebraic identities, Ternary collapse, range-based comparison folding. |
| ``infer_range(e, ctx)``                      | Companion range analysis over integer Exprs.                                          |
| ``simplify_loop_op(op)``                     | LoopOp walker — seeds Context from axis extents.                                      |
| ``simplify_kernel(k)``                       | GpuKernel walker — pushes ranges from VarDecl (nonneg thread-index compositions) + IfStmt conds + ForLoop bounds. |

**Rule:** Imports ``expr``, ``loop_ir``, ``kernel_ir``. No dependencies
on passes / pipeline / backend — pure IR → IR.

### `expr.py` — shared expression sublanguage (all layers)

Backend-agnostic expression AST used by both the loop IR (coord maps,
Select predicates) and the GPU IR (array indices, loop bounds). Plus
the coord-expression helpers that operate on it.

| Symbol                                                                  | Role                                                                     |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| ``Var``, ``Literal``, ``BinOp``, ``Builtin``, ``FuncCall``, ``Ternary`` | Expression nodes. Each has an ``eval(env)`` method for numpy evaluation. |
| ``_ExprOps``                                                            | Mixin: Python operator overloading for expression building.              |
| ``Expr``                                                                | Type alias for the union.                                                |
| ``PLACEHOLDER_PREFIX``                                                  | Convention: ``out_coord_N`` names output coordinates.                    |
| ``placeholder`` / ``is_placeholder``                                    | Build / recognize placeholder vars.                                      |
| ``substitute``                                                          | Tree rewrite: replace named ``Var`` nodes.                               |

**Rule:** No imports from other IR files.

### `frontend.py` — Torch IR (Layer 1, pre-decomposition)

Ops captured directly from PyTorch tracing. Every one of them has a
decomposition rule that rewrites it into ``tensor`` primitives. After the
decomposition pass completes, none of these ops should remain.

| Group           | Ops                                                                                                     |
|-----------------|---------------------------------------------------------------------------------------------------------|
| Layout-only     | ``TransposeOp``, ``ReshapeOp``, ``SliceOp``, ``CatOp``, ``UnsqueezeOp`` — decomposed to ``IndexMapOp``. |
| Compound math   | ``LinearOp``, ``MatmulOp``, ``SdpaOp``, ``MeanOp`` — decomposed to elementwise + reduce chains.         |

**Rule:** Imports ``base`` only. Must not depend on ``tensor`` / ``loop``
(decomposition rewrites *into* those; the frontend is upstream).

### `tensor.py` — minimal IR (Layer 1, post-decomposition)

What survives after decomposition. This is the dialect that fusion
consumes.

| Symbol                               | Role                                                       |
|--------------------------------------|------------------------------------------------------------|
| ``ElementwiseOp``                    | Per-element scalar function (``add``/``mul``/``exp``/...). |
| ``ReduceOp``                         | Collapse one axis via associative binary op.               |
| ``ScanOp``                           | Cumulative variant of reduce.                              |
| ``GatherOp`` / ``ScatterOp``         | Data-dependent reads / writes.                             |
| ``IndexMapOp`` + ``IndexSource``     | Unified layout-only op over ``Expr``.                      |
| ``OpInfo`` / ``OP_REGISTRY``         | Elementwise arity / commutativity.                         |
| ``ReduceInfo`` / ``REDUCE_REGISTRY`` | Reduce identity elements.                                  |

**Rule:** Imports ``base`` and ``shape_utils`` (for broadcasting).
Lazy-imports ``expr`` inside ``IndexMapOp.forward`` / ``is_identity`` only.

### `loop.py` — Loop IR, structural SSA (Layer 2)

After fusion, each ``LoopOp`` is exactly one GPU kernel described as an
SSA program over a **named iteration space**. "Loop" refers to the tiled
loop-nest that codegen eventually emits — one ``LoopOp`` maps to one
``GpuKernel`` and one CUDA launch.

| Symbol                        | Role                                                                                                              |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ``Axis``                      | Named iteration variable (``name`` + ``extent``). Free vs reduce is inferred from body structure — a Loop is a reduce Loop iff its body contains an Accum (see ``LoopOp.reduce_axis_names``). |
| ``LoopOp``                    | One kernel: ``inputs`` + nested ``body`` (``axes`` / ``accums`` are computed properties).                         |
| ``Port``                      | Access pattern for one external buffer — ``index: tuple[Expr, ...]`` over axis Vars.                              |
| ``Load``                      | Body-form port read: ``name = load(src)[index...]``; introduces an SSA name.                                      |
| ``Accum``                     | Reduce accumulator: ``name = op(name, value)`` inside a reduce ``Loop``. Implicitly initialized to ``ACCUM_IDENTITY[op.fn]``; after the Loop the ``name`` is in scope with the finalized value. |
| ``Assign``                    | SSA body stmt: ``name = op(args)`` with ``op: ElementwiseOp``.                                                    |
| ``Write``                     | Write an SSA value to output ``output`` at ``index``.                                                             |
| ``Select`` + ``SelectBranch`` | Coord-predicated binding (replaces the old Mux).                                                                  |
| ``Loop``                      | Explicit iteration block: ``axis`` (free or reduce) + nested ``body``. Body runs ``axis.extent`` times.           |
| ``Stmt``                      | Union: ``Assign \| Accum \| Write \| Select \| Loop \| Load``.                                                    |

``LoopOp.body`` is a nested tree of ``Loop`` blocks (outer free Loops
for the grid iteration, inner reduce Loops for per-row sweeps). Reading
top-to-bottom matches execution order. ``flatten_body(body)`` extracts
the leaf statement sequence for consumers that want a linear view
regardless of block structure.

**Rule:** Imports ``base``, ``expr``, and ``tensor`` (``ElementwiseOp``
only — ``ReduceOp`` is NOT a valid ``Assign.op``). Reductions are
modeled as ``Accum`` statements inside a reduce ``Loop``. SSA names
defined inside a Loop body are scoped to that body — only ``Accum``
targets cross Loop boundaries.

### `loop_plan.py` — analysis: LoopOp → KernelPlan (Layer 2)

Walks a ``LoopOp``'s nested ``Loop`` tree and produces an explicit
nested-loop view as a ``KernelPlan``: ordered ``Loop`` / ``Inline``
steps with accumulators, rematerialization sets, and trailing writes.
Consumed by the CUDA emitter (``backend/cuda/emit.py``). The human
dump view uses ``ir.loop_ir.pretty_print`` directly since the IR is
already nested.

| Symbol              | Role                                                                                                |
|---------------------|-----------------------------------------------------------------------------------------------------|
| ``Accum``           | Reduction accumulator: ``var`` (e.g. ``acc0``), ``fn``, ``identity``, SSA ``src``, ``result`` name. |
| ``Loop``            | K-loop step: ``recompute`` + ``body`` (``Assign`` / ``Select``) + optional ``accum`` / ``stores_output``. |
| ``Inline``          | Straight-line block of ``Assign`` / ``Select`` statements (no loop).                                |
| ``TrailingWrite``   | Write emitted once per thread after all reduce sweeps (for non-elementwise outputs).                |
| ``KernelPlan``      | Tuple of ``Step`` + per-element port set + output thread count + trailing writes.                   |
| ``analyze_kernel``  | Entry point: walks the body's ``Loop`` tree; each reduce ``Loop`` block → an optional ``Inline`` prelude (LICM: loop-invariant assigns hoisted out of the K-loop) + one ``Loop`` step. |

**Rule:** Imports ``expr``, ``loop``. No dependency on ``program`` or
any backend — this analysis is pure structural IR.

### `gpu.py` — GPU IR, imperative C-like AST (Layer 3)

The last IR before text. One ``GpuKernel`` per kernel; the rest of the
hierarchy is the C/C++ statement + expression AST the codegen emits.

| Group              | Symbols                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| GPU-specific expr  | ``ArrayAccess``, ``Cast``, ``FieldAccess``, ``VectorLoad``                                                                                    |
| Statements         | ``VarDecl``, ``Assign``, ``VarAssign``, ``AugAssign``, ``ForLoop``, ``IfStmt``, ``SyncThreads``, ``ArrayDecl``, ``PragmaUnroll``, ``RawCode`` |
| Kernel def         | ``GpuKernel``, ``GpuKernelParam``                                                                                                             |
| Type alias         | ``GpuExpr``                                                                                                                                   |
| Utilities          | ``pretty_print``                                                                                                                              |

**Rule:** Imports ``expr`` only. No dependency on the higher dialects
(frontend / tensor / loop) — by the time we are emitting C code, the
loop-IR ops have already been translated into statements.

## Invariants by stage

- **After tracing** (Layer 1): only frontend ops + ``InputOp`` / ``ConstantOp``.
- **After decomposition** (Layer 1): no ``LinearOp``, ``MatmulOp``, ``SdpaOp``,
  ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``, ``SliceOp``,
  ``CatOp``. Only ``ElementwiseOp``, ``ReduceOp``, ``IndexMapOp`` (plus
  scan / gather / scatter for non-decomposed primitives), and boundary
  sentinels.
- **After fusion** (Layer 2): only ``LoopOp`` + ``InputOp`` + ``ConstantOp``.
  Tensor-IR ops survive only *inside* ``LoopOp.body`` as ``Assign.op``.
- **GPU IR** (Layer 3) sees only expressions and statements — no ``Op``
  subclass appears.

## Sub-IRs shared across stages

- **``expr.py``** is the common expression sublanguage. It appears inside
  ``IndexMapOp.coord_map`` (tensor), ``Port.index`` /
  ``SelectBranch.select`` (loop), and ``ArrayAccess.index`` /
  ``ForLoop.end`` / everywhere in GPU IR. Coord-expression helpers
  (``substitute``, ``placeholder``, ``is_placeholder``) are pure AST
  operations and live here too.
- **``graph.py``** is the shared container. The *contents* change per stage
  but the container doesn't.

## Program forms — the execution view

The Loop IR and GPU IR each have a matching *Program* form (in
``compiler/program/``) that bundles many kernels with their buffer
metadata and launch order. See ``compiler/program/ARCHITECTURE.md`` for
the per-level breakdown.

- ``LoopProgram`` (``program/loop.py``) wraps ``LoopOp``s as ``LoopLaunch``es
  over ``LoopBuffer``s. Built by ``compile_graph``; authoritative source
  for buffer shapes.
- ``GpuProgram`` (``program/gpu.py``) wraps ``GpuKernel``s as ``GpuLaunch``es
  (usually ``CudaLaunch``) over ``GpuBuffer``s. Produced by codegen.

Codegen is a program-to-program lowering: ``LoopProgram → GpuProgram``.

## See also

- ``compiler/ARCHITECTURE.md`` — pipeline-level view of how frontend /
  lowering / backend fit together.
- ``compiler/program/ARCHITECTURE.md`` — the LoopProgram / GpuProgram
  symmetric pairing.
- ``compiler/rules/`` — the decomposition / optimization / fusion passes
  that transform one IR stage into the next.
- ``compiler/backend/cuda/emit.py`` — LoopProgram → GpuProgram lowering.
- ``compiler/backend/kernel_codegen.py`` — GpuKernel → C source.
