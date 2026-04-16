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
│  lower.py:   extract_kernels(graph) -> list[KernelOp]            │
│  pipeline.py: compile_graph(graph) -> CompileResult              │
│    CompileResult: kernels + graph_inputs/outputs/constants       │
│                                                                  │
│  Each KernelOp is one GPU kernel described as an SSA program:    │
│      inputs (Port | Mux | Combine) →                             │
│        body (tuple[Assign, ...])  — SSA: name = op(args) →       │
│        outputs (Port | Mux)                                      │
│  Contraction (matmul) is lowered as mul + reduce_sum Assigns.  │
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
   │  pipeline.compile_graph(graph)  →  CompileResult
   ▼
CompileResult (Layer 2: kernels + metadata)
   │  CudaBackend.compile(result.kernels, graph_inputs, graph_outputs)
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

## Shape inference

Every ``Op`` subclass implements ``infer_output_shape(input_shapes) → tuple``.
The compiler never stores shapes redundantly — shapes are derived when
needed by calling ``infer_output_shape`` on the op instance.

| Op type | Rule |
|---|---|
| ``ElementwiseOp`` | ``broadcast_shapes(*input_shapes)`` — NumPy right-aligned broadcast. Inputs MUST be broadcast-compatible; matmul decomposition inserts unsqueeze IndexMapOps to ensure this. |
| ``ReduceOp`` | Drop the ``axis`` dim from the input shape. |
| ``IndexMapOp`` | Returns ``self.out_shape`` — the output shape is part of the op definition because it cannot be derived from the coord_map + input shape (e.g., reshape ``(12,) → (3, 4)``). |
| ``TransposeOp`` | Permute the input shape by ``self.axes``. |
| ``ReshapeOp`` | Returns ``self.shape``. |
| ``LinearOp`` / ``MatmulOp`` / ``SdpaOp`` / ``MeanOp`` | High-level ops; decomposed before shape inference runs on the kernel IR. |
| ``KernelOp`` | ``infer_shapes() → dict[str, tuple]`` walks the SSA body, calling each ``Assign.op.infer_output_shape`` with the arg shapes. ``infer_output_shape()`` returns the last Assign's shape. |

**Invariant**: after decomposition + optimization, every ``ElementwiseOp("mul")``
in the graph has broadcast-compatible inputs. Matmul ``A(..., M, K) × B(..., K, N)``
is decomposed with unsqueeze IndexMapOps (``A → A[..., M, K, 1]``,
``B → B[..., 1, K, N]``) so the mul broadcasts to ``(..., M, K, N)`` and
the reduce over ``K`` (axis -2) yields ``(..., M, N)``. No special-case
shape inference for contractions.

## Lowering policy (rules/fusion/assemble_kernels.py)

Backward-cone region growing via the rewriter:

1. **Forward BFS** from seed — collect all downstream fusable primitives
   (``ElementwiseOp``, ``ReduceOp``), enforcing that reductions form a
   single linear chain (no parallel reductions).
2. **Find output** — last node in topo order with no consumers in the
   forward set.
3. **Backward cone** — from output, absorb nodes whose consumers are
   all in the region. Trims nodes with side-outputs.
4. **Body** — topologically sorted ``Assign`` statements from the region.
5. ``InputOp`` / ``ConstantOp`` / ``IndexMapOp`` nodes emit no kernel —
   they survive as external ``Port.buffer_id`` leaves (IndexMapOps at
   boundaries are absorbed into ``Port.indexmap``).

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_input_value(KernelInput, coord) -> Expr`:
  - `Port` → `ArrayAccess(buffer, coord)`.
  - `Mux` → nested `Ternary` chain over branch `select` predicates.
  - `Combine` → emit each source to a temporary, then fold the `ops` chain.
- `_emit_body`: unified dispatcher, selects `_emit_flat` (no ReduceOp) or `_emit_segments` (ReduceOp present).
- `_emit_flat`: 1D grid over flat numel, 256 threads/block, 1 thread/coord. Walks body Assigns as inline expressions. Handles copy kernels (empty body).
- `_emit_segments`: 1 block/row, segment-based codegen. Body is split into segments at ReduceOp boundaries. Each segment with per-element references gets its own K-loop; per-element values from prior segments are recomputed via transitive dependency analysis. Supports cross-iteration-space patterns (e.g. softmax: reduce_max → sub+exp → reduce_sum → div) and contractions (mul → reduce_sum). Max reduction uses `fmaxf` instead of `AugAssign`.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Numpy backend (`backend/numpy/backend.py`)

`NumpyBackend` derives from `Backend` (same ABC as `CudaBackend`).
Every `Op` subclass implements `forward(*inputs: np.ndarray) -> np.ndarray` — the
numpy equivalent of the tensor operation. `NumpyBackend.compile(graph)` stores the
graph; `run()` / `run_arrays()` walk it in topological order, calling `forward` at
each node, reshaping outputs to match declared `node.output.shape`. No GPU required.

Covered ops: all elementwise functions, reductions, scans, gather/scatter,
transpose/reshape/unsqueeze/slice/cat, linear, matmul, SDPA, mean.
`IndexMapOp` and `KernelOp` raise `NotImplementedError` (structural IR).

## Testing

- `tests/compiler/test_ir.py`, `test_shape_inference.py`, `test_indexmap.py`, `test_backend_ir.py` — Layer 1 unit tests.
- `tests/compiler/test_kernel_op.py` — structural IR construction + invariant violations.
- `tests/compiler/test_lower.py` — `lower()` group-discovery + KernelOp assembly.
- `tests/compiler/test_emit.py` — recursive-descent codegen source-level assertions + on-GPU numerical checks.
- `tests/compiler/test_pipeline.py` — end-to-end on small synthetic graphs.
- `tests/compiler/test_torch_trace*.py`, `test_real_trace.py`, `test_hints.py` — tracer / hint coverage.
- `tests/compiler/test_torch_ops.py` — Op.forward() + numpy backend: per-op tests + torch cross-checks.
- `tests/compiler/rules/` — all rewrite rules (decomposition, optimization, fusion).

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
