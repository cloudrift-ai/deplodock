# IR Architecture

One file per IR level. Each dialect defines the ops that are legal in the
graph at a specific point in the compilation pipeline. A single ``Graph``
(``graph.py``) hosts nodes from every dialect; as rewrite passes run, the
population of op types in the graph changes.

## Pipeline

```
PyTorch module
   │  torch_trace
   ▼
┌────────────────────────────────────────────────────────────┐
│ Graph populated with FRONTEND ops (frontend.py)            │
│   LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp,         │
│   TransposeOp, ReshapeOp, SliceOp, CatOp                   │
│   + InputOp / ConstantOp boundary sentinels (base.py)      │
└────────────────────────────────────────────────────────────┘
   │  rules/decomposition/*
   ▼
┌────────────────────────────────────────────────────────────┐
│ Graph populated with TENSOR / MINIMAL ops (tensor.py)      │
│   ElementwiseOp, ReduceOp, ScanOp, GatherOp, ScatterOp,    │
│   IndexMapOp (subsumes Slice / Cat / Transpose / Reshape / │
│               Unsqueeze via coord_map over expr.py)        │
│   + InputOp / ConstantOp                                   │
└────────────────────────────────────────────────────────────┘
   │  rules/optimization/*  (IndexMap composition, broadcast insertion)
   │  rules/fusion/*        (assemble_kernels, wrap_indexmap)
   ▼
┌────────────────────────────────────────────────────────────┐
│ Graph populated with LOOP IR ops (loop.py)                 │
│   LoopOp (one per GPU kernel) — SSA program whose ports    │
│   read / write external buffers; Mux for coord-predicated  │
│   dispatch; Combine for elementwise composition.           │
│   + InputOp / ConstantOp still present as buffer sources.  │
└────────────────────────────────────────────────────────────┘
   │  compile_graph → LoopProgram (program/loop.py)
   │  backend/cuda/emit.compile_kernels
   ▼
┌────────────────────────────────────────────────────────────┐
│ GPU IR (gpu.py) — one GpuKernel per kernel                 │
│   Imperative C-like AST: VarDecl, Assign, ForLoop, IfStmt, │
│   ArrayAccess, Cast, VectorLoad, ...                       │
│   Wrapped into GpuProgram (program/gpu.py) for execution.  │
└────────────────────────────────────────────────────────────┘
   │  backend/kernel_codegen.py
   ▼
C/C++ source → nvcc → GPU
```

## Per-file summary

### `base.py` — shared root

Cross-cutting types used at every stage.

| Symbol        | Role                                                     |
|---------------|----------------------------------------------------------|
| ``Op``        | Base class. Subclasses implement ``infer_output_shape`` and ``forward`` (numpy). |
| ``InputOp``   | Sentinel: graph input tensor. Value supplied by executor. |
| ``ConstantOp``| Sentinel: weights / RoPE tables / scalar constants.      |
| ``_drop_axis``| Shape helper — shared by ``ReduceOp`` (tensor) and ``MeanOp`` (frontend). |

**Rule:** No dependencies on other IR modules. Everyone may import from here.

### `graph.py` — container + hints

Hosts nodes from every dialect as the rewriter progresses.

| Symbol            | Role                                                  |
|-------------------|-------------------------------------------------------|
| ``Tensor``        | Name + shape + dtype. One per node output.            |
| ``Node[T_Op]``    | Wraps one ``Op``. Generic on op subtype.              |
| ``Graph``         | DAG. Insertion, removal, topo order, (de)serialization. |
| ``Hints``         | Advisory metadata bag (dotted keys, e.g. ``cuda.matmul.strategy``). |
| ``resolve_hints`` | Merge graph + node hints.                             |

**Rule:** Imports ``base``; lazy-imports ``frontend``/``tensor``/``loop``
inside ``Graph.from_dict`` for op-class reconstruction.

### `expr.py` — shared expression sublanguage

Backend-agnostic expression AST used by both the loop IR (coord maps,
Mux selectors) and the GPU IR (array indices, loop bounds). Plus the
coord-expression helpers that operate on it.

| Symbol                      | Role                                        |
|-----------------------------|---------------------------------------------|
| ``Var``, ``Literal``, ``BinOp``, ``Builtin``, ``FuncCall``, ``Ternary`` | Expression nodes. Each has an ``eval(env)`` method for numpy evaluation. |
| ``_ExprOps``                | Mixin: Python operator overloading for expression building. |
| ``Expr``                    | Type alias for the union.                   |
| ``PLACEHOLDER_PREFIX``      | Convention: ``out_coord_N`` names output coordinates. |
| ``placeholder`` / ``is_placeholder`` | Build / recognize placeholder vars. |
| ``substitute``              | Tree rewrite: replace named ``Var`` nodes.  |

**Rule:** No imports from other IR files.

### `frontend.py` — Torch IR

Ops captured directly from PyTorch tracing. Every one of them has a
decomposition rule that rewrites it into ``tensor`` primitives. After the
decomposition pass completes, none of these ops should remain.

| Group          | Ops                                                   |
|----------------|-------------------------------------------------------|
| Layout-only    | ``TransposeOp``, ``ReshapeOp``, ``SliceOp``, ``CatOp``, ``UnsqueezeOp`` — decomposed to ``IndexMapOp``. |
| Compound math  | ``LinearOp``, ``MatmulOp``, ``SdpaOp``, ``MeanOp`` — decomposed to elementwise + reduce chains. |

**Rule:** Imports ``base`` only. Must not depend on ``tensor`` / ``loop``
(decomposition rewrites *into* those; the frontend is upstream).

### `tensor.py` — minimal IR (post-decomposition)

What survives after decomposition. This is the dialect that fusion
consumes.

| Symbol             | Role                                                |
|--------------------|-----------------------------------------------------|
| ``ElementwiseOp``  | Per-element scalar function (``add``/``mul``/``exp``/...). |
| ``ReduceOp``       | Collapse one axis via associative binary op.        |
| ``ScanOp``         | Cumulative variant of reduce.                       |
| ``GatherOp`` / ``ScatterOp`` | Data-dependent reads / writes.            |
| ``IndexMapOp`` + ``IndexSource`` | Unified layout-only op over ``Expr``. |
| ``OpInfo`` / ``OP_REGISTRY`` | Elementwise arity / commutativity.        |
| ``ReduceInfo`` / ``REDUCE_REGISTRY`` | Reduce identity elements.         |

**Rule:** Imports ``base`` and ``shape_utils`` (for broadcasting).
Lazy-imports ``expr`` inside ``IndexMapOp.forward`` / ``is_identity`` only.

### `loop.py` — Loop IR (structural SSA)

After fusion, each ``LoopOp`` is exactly one GPU kernel described as an
SSA program over iteration space. "Loop" refers to the tiled loop-nest
that codegen eventually emits — one ``LoopOp`` maps to one ``GpuKernel``
and one CUDA launch.

| Symbol                      | Role                                         |
|-----------------------------|----------------------------------------------|
| ``LoopOp``                  | One kernel: inputs tree + SSA body + outputs tree. |
| ``Port``                    | Leaf: external buffer access + optional ``IndexMapOp``. |
| ``Mux`` + ``MuxBranch``     | Coord-predicated dispatch (input or output side). |
| ``Combine``                 | Operadic composition of N sub-inputs + elementwise chain. |
| ``Assign``                  | SSA body statement: ``name = op(args)``.     |
| ``LoopInput`` / ``LoopOutput`` | Type unions.                              |
| ``ElementwiseChain``        | ``tuple[ElementwiseOp, ...]`` alias.         |

**Rule:** Imports ``base`` and ``tensor`` (needs ``ElementwiseOp`` /
``ReduceOp`` / ``IndexMapOp`` for ``Combine.ops``, ``Assign.op``,
``Port.indexmap``). ``expr`` is imported under ``TYPE_CHECKING`` only
(``Expr`` appears in ``MuxBranch.select`` annotation).

### `gpu.py` — GPU IR (imperative C-like AST)

The last IR before text. One ``GpuKernel`` per kernel; the rest of the
hierarchy is the C/C++ statement + expression AST the codegen emits.

| Group              | Symbols                                             |
|--------------------|-----------------------------------------------------|
| GPU-specific expr  | ``ArrayAccess``, ``Cast``, ``FieldAccess``, ``VectorLoad`` |
| Statements         | ``VarDecl``, ``Assign``, ``VarAssign``, ``AugAssign``, ``ForLoop``, ``IfStmt``, ``SyncThreads``, ``ArrayDecl``, ``PragmaUnroll``, ``RawCode`` |
| Kernel def         | ``GpuKernel``, ``GpuKernelParam``                   |
| Type alias         | ``GpuExpr``                                         |
| Utilities          | ``pretty_print``                                    |

**Rule:** Imports ``expr`` only. No dependency on the higher dialects
(frontend / tensor / loop) — by the time we are emitting C code, the
loop-IR ops have already been translated into statements.

## Invariants by stage

- **After tracing**: only frontend ops + ``InputOp`` / ``ConstantOp``.
- **After decomposition**: no ``LinearOp``, ``MatmulOp``, ``SdpaOp``,
  ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``, ``SliceOp``,
  ``CatOp``. Only ``ElementwiseOp``, ``ReduceOp``, ``IndexMapOp`` (plus
  scan / gather / scatter for non-decomposed primitives), and boundary
  sentinels.
- **After fusion**: only ``LoopOp`` + ``InputOp`` + ``ConstantOp``.
  Tensor-IR ops survive only *inside* ``LoopOp.body`` as ``Assign.op``.
- **GPU IR** sees only expressions and statements — no ``Op`` subclass
  appears.

## Sub-IRs shared across stages

- **``expr.py``** is the common expression sublanguage. It appears inside
  ``IndexMapOp.coord_map`` (tensor), ``Mux.select`` / ``MuxBranch.select``
  (loop), and ``ArrayAccess.index`` / ``ForLoop.end`` / everywhere in
  GPU IR. Coord-expression helpers (``substitute``, ``placeholder``,
  ``is_placeholder``) are pure AST operations and live here too.
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
