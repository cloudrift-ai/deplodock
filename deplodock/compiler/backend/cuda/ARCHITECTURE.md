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

**Body emitters**: three paths selected by `_detect_contraction` and
the presence of `ReduceOp` Assigns:

| Pattern                    | Emitter                  | Emission                                                      |
| -------------------------- | ------------------------ | ------------------------------------------------------------- |
| Contraction (mul+reduce)   | `_emit_contraction_body` | 2D grid (N, M, batch); serial K-loop; post-contraction SSA    |
| Reduce (no contraction)    | `_emit_reduce_body`      | 1 block/row; serial K-loop per ReduceOp Assign                |
| Pointwise (no ReduceOp)    | `_emit_pointwise_body`   | 1D grid; inline elementwise per Assign                        |

**Contraction detection** (`_detect_contraction`): pattern-matches the
SSA body for a binary `ElementwiseOp` whose both args are input Port
names, followed by a `ReduceOp` consuming it. Returns the index and
the two Assigns, or `None`.

## Naive Schedule Policy

Correctness-first. No shared memory, no async copies, no TMA, no
vectorization, no tiling beyond the trivial. Tuning and tiling are
follow-up commits.

- **Pointwise** (no `ReduceOp` in body):
  1D grid, `block = (256, 1, 1)`, one thread per flat output coord.
- **Reduce chain** (`ReduceOp` in body, no contraction):
  1D grid over the post-reduce shape; `block = (1, 1, 1)`; each block
  runs a serial K-loop over the reduced axis.
- **Contraction** (detected mul+reduce pattern):
  2D grid `(N, M, batch)`; `block = (1, 1, 1)`; each thread iterates K
  serially.

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
