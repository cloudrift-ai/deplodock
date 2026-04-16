# Compiler Architecture

## Three-Layer Shape

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Frontend (backend-agnostic)                           │
│                                                                  │
│  torch_trace.py ─→ Graph IR (ir.py, ops.py)                      │
│      Tensor, Node[T_Op], Graph; primitive Op subclasses          │
│      (ElementwiseOp, ReduceOp, IndexMapOp, LinearOp, MatmulOp,   │
│       SdpaOp, MeanOp, ...).                                      │
│                                                                  │
│  RULE: No GPU, no CUDA, no backend imports.                      │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 2 · Lowering → structural KernelOps                       │
│                                                                  │
│  lower.py:   lower(graph: Graph) -> list[KernelOp]               │
│  pipeline.py: public entry (compile_graph)                        │
│                                                                  │
│  Each KernelOp is one GPU kernel described as a tiled dataflow   │
│  pipeline (see ops.py docstring for analogies):                  │
│      inputs (Port | Mux | Combine) →                             │
│        [contraction (ContractionCore)] →                         │
│        [reduce_stages (tuple[ReduceStage, ...])] →               │
│        [epilogue (Node[ElementwiseOp], ...)] →                   │
│        outputs (Port | Mux)                                      │
│                                                                  │
│  RULE: Operates on Graph + KernelOp IR only. No backend imports. │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 3 · Backend (emit + run)                                  │
│                                                                  │
│  backend/base.py:           Backend ABC (compile, run, benchmark)│
│  backend/program.py:        Buffer, Launch, Program              │
│  backend/ir/expr.py:        Shared Expr AST (Var, BinOp, ...)    │
│  backend/ir/kernel_ir.py:   Imperative AST (KernelDef, Stmt, ...)│
│  backend/ir/kernel_codegen.py: KernelDef → C source              │
│  backend/cuda/emit.py:      KernelOp → KernelDef via recursive   │
│                              descent; no classification.         │
│  backend/cuda/backend.py:   CudaBackend.compile(list[KernelOp])  │
│  backend/cuda/program.py:   generate_source, nvcc, run           │
│  backend/cuda/runner.py:    single-kernel compile + run harness  │
│                                                                  │
│  RULE: GPU specifics live here; everything above is portable.    │
└──────────────────────────────────────────────────────────────────┘
```

## Canonical Data Flow

```
PyTorch module
   │  torch_trace.trace_module(...)
   ▼
Graph (Layer 1)
   │  pipeline.compile_graph(graph)  →  lower.lower(graph)
   ▼
list[KernelOp] (Layer 2, structural IR)
   │  CudaBackend.compile(kernels, graph_inputs, graph_outputs)
   │    └─ for k in kernels:
   │         emit.emit_kernel(k, name)  → KernelDef
   │         kernel_codegen.emit_kernel(KernelDef) → C source
   │         wrap as CudaLaunch, collect in Program
   ▼
Program (Layer 3)
   │  program.generate_source(program)
   ▼
.cu  →  nvcc  →  GPU
```

## Structural KernelOp Cheat Sheet

Every slot has a single predefined op type; `__post_init__` enforces it.

| Slot                          | Type                                     | Used for                                                            |
| ----------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| `Port.buffer_id`              | `str`                                    | External buffer read/write                                          |
| `Port.indexmap`               | `IndexMapOp \| None`                     | Layout transform at the load/store                                  |
| `Mux.branches[i].input`       | `KernelInput` (recursive)                | Dispatched read (cat, RoPE halves, KV cache)                        |
| `Mux.branches[i].select`      | `Expr` (from `backend/ir/expr`)          | Output-coord predicate                                              |
| `Combine.sources`             | `tuple[KernelInput, ...]`                | N sub-inputs to an elementwise chain                                |
| `Combine.ops`                 | `tuple[Node[ElementwiseOp], ...]`        | Per-coord elementwise work                                          |
| `ContractionCore.operand`     | `KernelInput`                            | The per-K value (for matmul: `Combine((a, b), ops=(mul,))`)         |
| `ContractionCore.k_axis`      | `int`                                    | Reduction axis of the operand                                       |
| `ContractionCore.reduce`      | `Node[ReduceOp]`                         | Contraction reduce (sum / max / prod)                               |
| `ReduceStage.pre_ops`         | `tuple[Node[ElementwiseOp], ...]`        | Chain between reduces in a multi-reduce kernel                      |
| `ReduceStage.reduce`          | `Node[ReduceOp]`                         | One reduce in the chain                                             |
| `KernelOp.epilogue`           | `tuple[Node[ElementwiseOp], ...]`        | Post-body elementwise on the output coord space (bias, activation)  |

Analogies (use in module/class docstrings):

- **Dataflow / signal-flow graph** for `KernelInput` trees.
- **Hardware multiplexer** (FPGA N-to-1 mux / 1-to-N demux) for `Mux`.
- **Operad / expression tree** for `Combine`.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store) for `KernelOp`.
- **Systolic core** for `ContractionCore`.

## Lowering policy (lower.py)

Greedy walk over topo-sorted graph nodes:

1. **Matmul pair** — `ReduceOp(sum)` whose sole input is a fan-out-1 `ElementwiseOp(mul)` collapses into a single `ContractionCore` `KernelOp`.
2. **Everything else** — each compute node (ElementwiseOp / ReduceOp) becomes its own singleton `KernelOp`.
3. `InputOp` / `ConstantOp` nodes emit no kernel — they survive as external `Port.buffer_id` leaves.

Decomposition of `LinearOp` / `SdpaOp` / `MatmulOp` / `MeanOp` and layout-op (`TransposeOp`, `SliceOp`, `CatOp`, ...) handling are not yet in the lowering — they will be added incrementally as new patterns (and their tests) come online.

## Codegen policy (backend/cuda/emit.py)

Recursive descent over the `KernelOp` structure — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Per-variant walkers:

- `_emit_input_value(KernelInput, coord) -> Expr`:
  - `Port` → `ArrayAccess(buffer, coord)`.
  - `Mux` → nested `Ternary` chain over branch `select` predicates.
  - `Combine` → emit each source to a temporary, then fold the `ops` chain.
- `_emit_pointwise_body`: 1D grid over flat numel, 256 threads/block, 1 thread/coord.
- `_emit_reduce_body`: 1 block/row, serial K-loop, single-thread reduction.
- `_emit_contraction_body`: 2D grid (N, M, 1), 1 thread/(m, n), serial K-loop.
- `_apply_epilogue`: fold `KernelOp.epilogue` chain onto the body output.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Testing

- `tests/compiler/test_ir.py`, `test_shape_inference.py`, `test_indexmap.py`, `test_backend_ir.py` — Layer 1 unit tests.
- `tests/compiler/test_kernel_op.py` — structural IR construction + invariant violations.
- `tests/compiler/test_lower.py` — `lower()` group-discovery + KernelOp assembly.
- `tests/compiler/test_emit.py` — recursive-descent codegen source-level assertions + on-GPU numerical checks.
- `tests/compiler/test_pipeline.py` — end-to-end on small synthetic graphs.
- `tests/compiler/test_torch_trace*.py`, `test_real_trace.py`, `test_hints.py` — tracer / hint coverage.

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
