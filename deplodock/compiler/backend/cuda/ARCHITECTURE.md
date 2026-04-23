# CUDA Backend Architecture

The CUDA backend takes a `Graph` and lowers it through two structural
passes — `passes/lowering/kernel` (LoopOp → KernelOp) and
`passes/lowering/cuda` (KernelOp → CudaOp) — producing a `Graph[CudaOp]`
that the runtime walks to compile each kernel via `cupy.RawKernel`
(NVRTC) and dispatch launches directly from Python. Source generation
is a **recursive descent** over the structural IR; there's no
classification pass, no `Schedule` dataclass, and no intermediate
`LoopIR` stage.

**Peer backends:**
- `backend/numpy/` — `NumpyBackend` evaluates the Graph directly (pre-fusion).
- `backend/loop/` — `LoopBackend` interprets the fused `Graph[LoopOp]`
  via numpy whole-tensor operations. Used as a triangulation reference:
  disagreement with CUDA implicates codegen; disagreement with numpy
  implicates fusion. All three backends expose the same `compile(graph)`
  / `run(compiled, input_data=…) → ProgramResult` API (`backend/base.py`).

## Module Layout

```
cuda/
├── backend.py   # CudaBackend(Backend): compile(graph) → Graph[CudaOp]
├── emit.py      # emit_kernel(node, name, graph) → (GpuKernel, arg_order)
│                # launch_config(node) → (grid, block)
├── program.py   # Graph[CudaOp] → cupy.RawKernel dispatch + per-kernel event timing
└── runner.py    # Single-kernel cupy dispatch (used by tuning/diagnostics scripts)
```

Shared infrastructure (backend-agnostic) lives at the IR layer:

```
compiler/ir/kernel/ir.py   # GpuKernel, GpuKernelParam, Stmt hierarchy, expressions, KernelOp graph-op
compiler/ir/kernel/emit.py # per-node LoopOp → GpuKernel codegen (emit_kernel, launch_config)
compiler/ir/cuda/ir.py     # CudaOp   (graph-op with rendered CUDA source)
ir/cuda/emit.py  # GpuKernel → C source (one level up; not CUDA-specific)
```

## Emission Pipeline

```
Graph[LoopOp]   (output of pipeline.compile_graph)
    │
    │  passes/lowering/kernel/lower_loopop.py
    │    for each LoopOp node:
    │      emit_kernel(node, name, graph) → (GpuKernel, arg_order)
    │      node.op = KernelOp(kernel, kernel_name, arg_order, grid, block, …)
    │      # KernelOp.__post_init__ runs ir.kernel.normalize.normalize_kernel
    │      # (const fold / clamp eliminate) automatically.
    ▼
Graph[KernelOp]
    │
    │  passes/lowering/cuda/lower_kernelop.py
    │    for each KernelOp node:
    │      emit_kernel_source(op.kernel) → C source string
    │      node.op = CudaOp(kernel_source, kernel_name, arg_order, grid, block, …)
    ▼
Graph[CudaOp]
    │  backend/cuda/program._compile(graph)
    │    for each unique kernel_name: cp.RawKernel(source, name, -O2 --use_fast_math)
    │    (NVRTC caches per source)
    ▼
cupy dispatch: ndarray-per-buffer allocation, kernel(grid, block, args)  →  GPU
```

## Walkers and Emitters in `emit.py`

The emitter walks each `LoopOp`'s SSA `body` — a sequence of `Load` /
`Assign` / `Accum` / `Write` / `Select` statements. Each `Assign` is
`name = op(args)` where args reference prior SSA names; external buffer
reads are explicit `Load` stmts keyed by `source` int into
`node.inputs`.

**Load emission** (`_emit_load_access(index, buf_name, src_shape, axis_env)`):
evaluates `Load.index` in the current axis environment and emits
`ArrayAccess(buffer, coord)`. There is no separate indexing abstraction —
every IndexMapOp is lifted to a one-op `LoopOp` by
`003_lift_indexmap`, and the splicer in `ir/loop/splicer.py` inlines the
coord_map into the consumer's body Loads whenever a splice is legal.

**Select lowering** (`_emit_select(stmt, values, axis_env)`): lowers a
body `Select` into a nested `Ternary` chain over each
`SelectBranch.select` predicate.

**Body emitter**: `_emit_body(node, graph)` threads a `_Ctx` through the
recursive walk. Grid is 1D over flat free-axis extents; block is
`(256, 1, 1)`.

## Shape handling

Shapes come from the graph itself, never recomputed:

- `graph.nodes[name].output.shape` — per-buffer shape lookup.
- For a LoopOp node, the effective source shapes are
  `[graph.nodes[n].output.shape for n in node.inputs]`.

No fallback call to `LoopOp.infer_output_shape(...)` happens in the
codegen path.

## Benchmark Mode Timing

`benchmark_program(graph, warmup, num_iters)` runs two levels of
timing around the dispatch loop:

- **Program total**: one global `cupy.cuda.Event` pair wraps all
  `num_iters` iterations; the delta / `num_iters` populates
  `BenchmarkResult.time_ms`.
- **Per-launch**: one event pair per launch index, re-recorded every
  iteration; `cupy.cuda.get_elapsed_time` is accumulated into an
  `acc[i]` list, averaged, and emitted as `LaunchTime(idx, kernel_name,
  time_ms)` entries in `BenchmarkResult.per_launch`.

`zero_outputs` calls (cupy `.fill(0)`) run before the launch's event
pair — they aren't counted as kernel time.

## Naive Schedule Policy

Correctness-first. No shared memory, no async copies, no vectorization.

- **Pointwise** (no `ReduceOp` in body): 1D grid, `block = (256, 1, 1)`,
  one thread per flat output coord.
- **Reduce / contraction** (`ReduceOp` in body): 1D grid over the outer
  rows (broadcast shape minus reduce axis). The body is split into
  segments at ReduceOp boundaries. Each segment with per-element
  references emits its own K-loop. Per-element values from prior
  segments are recomputed inside the current loop.

## Buffer Roles

`backend/cuda/program._buffers(graph)` classifies each node on the fly:

- in `graph.inputs` → `input`
- `isinstance(node.op, ConstantOp)` → `constant`
- in `graph.outputs` → `output`
- otherwise → `scratch`

## Constraints to Preserve

- `emit.py` must stay pure codegen — it imports from `ir/` and returns
  `GpuKernel`; it must not import from the runtime / passes layer.
- A new backend (ROCm, SYCL, ...) reuses `ir/kernel/`, `ir/cuda/`,
  and `ir/cuda/emit.py` wholesale; only its own
  `emit.py` / `program.py` / lowering pass need to be rewritten.
- The loop-IR types (`LoopOp`, `Load`, `Assign`, `Accum`, `Write`,
  `Select`, `SelectBranch`, ...) stay backend-agnostic — do NOT import
  from `backend/cuda/` into `ir/loop/ir.py`.
