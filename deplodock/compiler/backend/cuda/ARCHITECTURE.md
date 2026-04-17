# CUDA Backend Architecture

The CUDA backend takes a `Graph` and lowers it via `compile_graph →
LoopProgram → compile_kernels → GpuProgram` — one `CudaLaunch` per
kernel. Source generation is a **recursive descent** over the structural
IR; there's no classification pass, no `Schedule` dataclass, and no
intermediate `LoopIR` stage.

**Peer backends:**
- `backend/numpy/` — `NumpyBackend` evaluates the Graph directly (pre-fusion).
- `backend/loop/` — `LoopBackend` interprets the `LoopProgram` via numpy
  whole-tensor operations (post-fusion, pre-codegen). Used as a
  triangulation reference: disagreement with CUDA on the same
  `LoopProgram` implicates codegen; disagreement with numpy implicates
  fusion. All three backends expose the same `compile(graph)` /
  `run(compiled, input_data=…) → ProgramResult` API (`backend/base.py`).

## Module Layout

```
cuda/
├── backend.py   # CudaBackend(Backend): compile(LoopProgram) → GpuProgram
├── emit.py      # compile_kernels(LoopProgram) → GpuProgram
│                #   emit_kernel(LoopLaunch, name, LoopProgram) → GpuKernel
├── program.py   # CudaLaunch(GpuLaunch), generate_source(), nvcc + run
└── runner.py    # Single-kernel compile + run + benchmark harness
```

Shared infrastructure (backend-agnostic) lives in two places:

```
compiler/ir/gpu.py         # GpuKernel, GpuKernelParam, Stmt hierarchy, expressions
compiler/program/gpu.py    # GpuBuffer, GpuLaunch, GpuProgram (runnable program form)
backend/kernel_codegen.py  # GpuKernel → C source (up one level; not CUDA-specific)
```

## Emission Pipeline

```
LoopProgram
    │
    │  compile_kernels(loop_program)
    │    for each LoopLaunch in loop_program.launches:
    │      emit.emit_kernel(launch, name, loop_program) → (GpuKernel, arg_order)
    │      kernel_codegen.emit_kernel(GpuKernel) → C source string
    │      wrap as CudaLaunch(kernel_source, grid, block, args)
    ▼
GpuProgram
    │  generate_source(program)
    ▼
.cu  →  nvcc  →  GPU
```

## Walkers and Emitters in `emit.py`

The emitter walks each `LoopOp`'s SSA `body` — a sequence of `Assign` / `Update` / `Write` / `Select` statements. Each `Assign` is `name = op(args)` where args reference input Ports by `$N` position or prior `Assign.name`s; the `$N` → buffer name mapping lives on the `LoopLaunch`.

**Port loads** (`_emit_port_load(port, buf_name, src_shape, axis_env)`): evaluates `Port.index` in the current axis environment and emits `ArrayAccess(buffer, coord)`. There is no `indexmap` field on `Port` — absorbed identity-alias IndexMapOps have already been baked into `Port.index` during fusion; non-identity IndexMapOps are lowered to their own copy kernel by `003_wrap_indexmap`.

**Select lowering** (`_emit_select(stmt, values, axis_env)`): lowers a body `Select` into a nested `Ternary` chain over each `SelectBranch.select` predicate. (There is no `Mux` / `Combine` IR — those were replaced by `Select` / `SelectBranch`.)

**Body emitter**: `_emit_body` calls `backend.kernel_plan.analyze_kernel(loop, dollar_shapes, out_shape)` to produce a `KernelPlan`, then delegates to `_emit_plan`. The plan decomposes the body into ordered `Inline` (straight-line block) and `Loop` (K-loop over a reduce axis) steps:

| Pattern                 | Plan shape             | Emission                                                             |
| ----------------------- | ---------------------- | -------------------------------------------------------------------- |
| No ReduceOp (pointwise) | `Inline` steps only    | 1D grid over free-axis numel; `block = (256, 1, 1)`; inline Assigns  |
| ReduceOp present        | `Loop` + other steps   | 1D grid over outer rows; `block = (1, 1, 1)`; K-loops with recompute |

Contractions (matmul = mul + sum) fall out of the `Loop` path: the `mul` is an elementwise Assign inside the K-loop and the `sum` is the accumulator's `Update`. No special detection or classification.

## Shape handling

Shapes are read from the `LoopProgram`, never recomputed:

- `program.shape(name)` — per-buffer shape lookup.
- `program.dollar_shapes(launch)` — `$N → shape` map for a launch's Ports
  (the external buffer shape bound to each Port position).
- `program.output_shape(launch)` — shape of the launch's output buffer.

No fallback call to `LoopOp.infer_output_shape(...)` happens in the
codegen path.

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

`LoopProgram.from_graph` assigns each buffer a `role` based on graph-level
position (`input` / `output` / `constant` / `scratch`). `compile_kernels`
filters to buffers actually referenced by some launch (or a referenced
graph constant) when emitting `GpuBuffer`s.

## Constraints to Preserve

- `emit.py` must not import `backend.cuda.generators` or
  `backend.cuda.schedule` (deleted). The recursive-descent emitter is
  the only code path.
- A new backend (ROCm, SYCL, ...) reuses `ir/gpu.py`, `program/gpu.py`,
  and `backend/kernel_codegen.py` wholesale; only its own `emit.py` /
  `program.py` need to be rewritten.
- The loop-IR types (`LoopOp`, `Port`, `LocalBuffer`, `Assign`, `Update`,
  `Write`, `Select`, `SelectBranch`, ...) stay backend-agnostic — do NOT
  import from `backend/cuda/` into `ir/loop.py` or `program/loop.py`.
