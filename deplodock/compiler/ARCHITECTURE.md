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
│  Each KernelOp is one GPU kernel described as an SSA program:    │
│      inputs (Port | Mux | Combine) →                             │
│        body (tuple[Assign, ...])  — SSA: name = op(args) →       │
│        outputs (Port | Mux)                                      │
│  Contraction (matmul) is detected by pattern-matching the body.  │
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

SSA invariants are enforced by `KernelOp.__post_init__`: unique names,
defined-before-use, no forward references.

| Slot                          | Type                                     | Used for                                                            |
| ----------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| `Port.buffer_id`              | `str`                                    | External buffer read/write                                          |
| `Port.indexmap`               | `IndexMapOp \| None`                     | Layout transform at the load/store                                  |
| `Mux.branches[i].input`       | `KernelInput` (recursive)                | Dispatched read (cat, RoPE halves, KV cache)                        |
| `Mux.branches[i].select`      | `Expr` (from `backend/ir/expr`)          | Output-coord predicate                                              |
| `Combine.sources`             | `tuple[KernelInput, ...]`                | N sub-inputs to an elementwise chain                                |
| `Combine.ops`                 | `tuple[ElementwiseOp, ...]`              | Per-coord elementwise work                                          |
| `Assign.name`                 | `str`                                    | SSA value name                                                      |
| `Assign.op`                   | `ElementwiseOp \| ReduceOp`              | The operation to apply                                              |
| `Assign.args`                 | `tuple[str, ...]`                        | References to input Port buffer_ids or prior Assign names           |
| `KernelOp.inputs`             | `tuple[KernelInput, ...]`                | Port, Mux, or Combine input trees                                   |
| `KernelOp.body`               | `tuple[Assign, ...]`                     | SSA body: name = op(args)                                           |
| `KernelOp.outputs`            | `tuple[KernelOutput, ...]`               | Port or Mux output targets                                          |

Analogies (use in module/class docstrings):

- **Dataflow / signal-flow graph** for `KernelInput` trees.
- **Hardware multiplexer** (FPGA N-to-1 mux / 1-to-N demux) for `Mux`.
- **Operad / expression tree** for `Combine`.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store) for `KernelOp`.

## Lowering policy (rules/fusion/assemble_kernels.py)

Grammar-based fusion via the rewriter:

1. **Contraction** (optional) — `ElementwiseOp(mul)` + `ReduceOp(sum)` pair becomes two Assigns in the kernel body.
2. **Stage** (repeating) — `pre_ops* + reduce` groups become Assigns.
3. **Epilogue** — trailing `ElementwiseOp` chain becomes Assigns.
4. `InputOp` / `ConstantOp` nodes emit no kernel — they survive as external `Port.buffer_id` leaves.

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_input_value(KernelInput, coord) -> Expr`:
  - `Port` → `ArrayAccess(buffer, coord)`.
  - `Mux` → nested `Ternary` chain over branch `select` predicates.
  - `Combine` → emit each source to a temporary, then fold the `ops` chain.
- `_detect_contraction(kernel)`: pattern-matches SSA body for a binary ElementwiseOp (both args are input Ports) followed by a ReduceOp consuming it.
- `_emit_pointwise_body`: 1D grid over flat numel, 256 threads/block, 1 thread/coord. Walks body Assigns as inline expressions.
- `_emit_reduce_body`: 1 block/row, serial K-loop per ReduceOp Assign, inline elementwise for ElementwiseOp Assigns.
- `_emit_contraction_body`: 2D grid (N, M, batch), 1 thread/(m, n), serial K-loop. Post-contraction Assigns emitted as flat elementwise.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Testing

- `tests/compiler/test_ir.py`, `test_shape_inference.py`, `test_indexmap.py`, `test_backend_ir.py` — Layer 1 unit tests.
- `tests/compiler/test_kernel_op.py` — structural IR construction + invariant violations.
- `tests/compiler/test_lower.py` — `lower()` group-discovery + KernelOp assembly.
- `tests/compiler/test_emit.py` — recursive-descent codegen source-level assertions + on-GPU numerical checks.
- `tests/compiler/test_pipeline.py` — end-to-end on small synthetic graphs.
- `tests/compiler/test_torch_trace*.py`, `test_real_trace.py`, `test_hints.py` — tracer / hint coverage.

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
