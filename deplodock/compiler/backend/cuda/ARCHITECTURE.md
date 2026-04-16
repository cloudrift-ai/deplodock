# CUDA Backend Architecture

The CUDA backend turns a list of structural `KernelOp`s (produced by
`deplodock.compiler.lower`) into a runnable `Program` — one `CudaLaunch`
per kernel. Source generation is a **recursive descent** over the
structural IR; there's no classification pass, no `Schedule` dataclass,
and no intermediate `LoopIR` stage.

## Module Layout

```
cuda/
├── backend.py   # CudaBackend(Backend): compile(list[KernelOp]) → Program
├── emit.py      # emit_kernel(KernelOp, name) → KernelDef (recursive descent)
├── program.py   # CudaLaunch, generate_source(), nvcc compilation + run
└── runner.py    # Single-kernel compile + run + benchmark harness
```

Shared infrastructure (backend-agnostic) lives in
`deplodock/compiler/backend/ir/`:

```
backend/ir/
├── expr.py           # Var, Literal, BinOp, Ternary, FuncCall
├── kernel_ir.py      # KernelDef, Stmt (VarDecl / Assign / ForLoop / IfStmt / ...)
└── kernel_codegen.py # KernelDef → C source
```

## Emission Pipeline

```
list[KernelOp]
    │
    │  compile_kernels(kernels, graph_inputs, graph_outputs)
    │    for each KernelOp:
    │      emit.emit_kernel(k, name) → (KernelDef, arg_order)
    │      kernel_codegen.emit_kernel(KernelDef) → C source string
    │      wrap as CudaLaunch(kernel_source, grid, block, args)
    ▼
Program
    │  generate_source(program)
    ▼
.cu  →  nvcc  →  GPU
```

## Walkers and Emitters in `emit.py`

The emitter walks `KernelOp`'s SSA `body` (a sequence of `Assign`
statements). Each `Assign` is `name = op(args)` where args reference
input `Port.buffer_id`s or prior `Assign` names.

**Input-tree walker** (`_emit_input_value`): evaluates a `KernelInput`
at a coord `Expr`, appending temporaries to a `list[Stmt]`.

| IR variant  | Emission                                                    |
| ----------- | ----------------------------------------------------------- |
| `Port`      | `ArrayAccess(buffer, coord)` or indexmap load               |
| `Mux`       | Nested `Ternary` chain over `branch.select`                 |
| `Combine`   | Temporaries for each source + `VarDecl` fold over ops chain |

**Body emitter**: unified `_emit_body` dispatches based solely on the
presence of `ReduceOp` in the SSA body:

| Pattern                    | Emitter          | Emission                                                      |
| -------------------------- | ---------------- | ------------------------------------------------------------- |
| No ReduceOp (pointwise)    | `_emit_flat`     | 1D grid; 256 threads/block; inline elementwise per Assign     |
| ReduceOp present           | `_emit_segments` | 1 block/row; segment-based K-loops with recomputation         |

Contractions (matmul = mul + sum) go through `_emit_segments` naturally:
the `mul` is an elementwise Assign inside the K-loop, and the `sum` is
the segment's ReduceOp accumulator. No special detection or classification.

## Naive Schedule Policy

Correctness-first. No shared memory, no async copies, no TMA, no
vectorization, no tiling beyond the trivial. Tuning and tiling are
follow-up commits.

- **Pointwise** (no `ReduceOp` in body):
  1D grid, `block = (256, 1, 1)`, one thread per flat output coord.
  Empty bodies (copy kernels) are a subcase.
- **Reduce / contraction** (`ReduceOp` in body):
  1D grid over the outer rows (broadcast shape minus reduce axis);
  `block = (1, 1, 1)`. The body is split into segments at ReduceOp
  boundaries. Each segment with per-element references emits its own
  K-loop. Per-element values from prior segments are recomputed inside
  the current loop (transitive dependency analysis). Supports
  cross-iteration-space patterns like softmax (reduce_max -> elementwise
  -> reduce_sum -> elementwise) and contractions (mul -> reduce_sum).
  Max reduction uses `fmaxf(acc, val)` instead of `AugAssign`.

## Buffer Roles

`CudaBackend.compile(kernels, graph_inputs=..., graph_outputs=...)`
marks buffers as `input` / `output` / `scratch` based on whether the
buffer-id appears in the caller-supplied graph-boundary sets. Any
buffer produced by one kernel and consumed by another is `scratch`.

## Constraints to Preserve

- `emit.py` must not import `backend.cuda.generators` or
  `backend.cuda.schedule` (deleted). The recursive-descent emitter is
  the only code path.
- A new backend (ROCm, SYCL, ...) reuses `backend/ir/*` wholesale;
  only the per-variant walkers in its emit.py need to be rewritten.
- The structural IR types (`KernelOp`, `Port`, `Mux`, `Combine`, ...)
  stay backend-agnostic — do NOT import from `backend/cuda/` into
  `ops.py` or `lower.py`.
