# Program-form Architecture

Each IR dialect that has "kernel launches" has a matching *Program* form.
The IR describes one unit of compute (a single loop nest, a single GPU
kernel); the Program describes many of them wired together into a
runnable whole, with authoritative buffer metadata.

```
Graph (frontend → tensor → loop IR, populated in place by rewrites)
   │ pipeline.compile_graph
   ▼
LoopProgram    ← program/loop.py  +  ir/loop/ir.py
   │ backend/cuda/emit.compile_kernels
   ▼
GpuProgram     ← program/gpu.py   +  ir/kernel_ir.py
   │ backend/cuda/program.generate_source
   ▼
.cu source
```

## `loop.py` — LoopProgram

Pairs with `ir/loop/ir.py`. Produced by `LoopProgram.from_graph(graph)`
after fusion completes.

| Type                                                                                    | Role                                                              |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `LoopBuffer(name, shape, dtype, role)`                                                  | One buffer. Shape is authoritative; `size` is a derived property. |
| `LoopLaunch(loop, input_names, output_name)`                                            | One `LoopOp` invocation wired to named buffers (`Load.source` order). |
| `LoopProgram(name, buffers, launches, graph_inputs/outputs/constants, constant_values)` | The full post-fusion program.                                     |

`LoopProgram` exposes shape queries — `shape(name)`, `output_shape(launch)`,
`source_shapes(launch)` — so codegen never recomputes shapes.

**Rule:** No `aliases` field. At loop level, identity/metadata-only
IndexMaps are baked directly into the consuming `Load.index` Exprs
during fusion — there is no separate metadata-only-alias concept here
(unlike `GpuProgram`, which has `aliases` for buffer-pointer aliasing
at the device level). Each IndexMapOp is lifted to a one-op `LoopOp`
by `rules/fusion/003_lift_indexmap.py`; if its consumer is another
`LoopOp` and the splicer's σ can align the indices, the merge rule
absorbs the copy kernel into the consumer's body Loads.

## `gpu.py` — GpuProgram

Pairs with `ir/kernel_ir.py`. Produced by `backend/cuda/emit.compile_kernels`
by lowering each `LoopLaunch` to a `CudaLaunch(GpuLaunch)`.

| Type                                                                                 | Role                                                                    |
|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `GpuBuffer(name, size, dtype, role)`                                                 | One buffer. `size` is element count (resolved from `LoopBuffer.shape`). |
| `GpuLaunch(kernel_source, kernel_name, grid, block, args, smem_bytes, zero_outputs)` | One GPU kernel invocation.                                              |
| `GpuProgram(name, buffers, launches, defines, includes, aliases)`                    | The full device program.                                                |

Backend-specific subclasses (`backend/cuda/program.py::CudaLaunch`) add
fields like `tma_descriptors`.

## The symmetry

Each level's Program is:
1. **A bag of buffers** — the data plane with roles + sizes/shapes.
2. **An ordered list of launches** — the control plane, each referring to
   one IR unit (`LoopOp` or `GpuKernel`).
3. **Program-level metadata** — inputs, outputs, constants, (for GPU:
   defines, includes, aliases).

Lowering between two Program forms is the natural home for:
- **Codegen** (`LoopProgram → GpuProgram`)
- **Buffer allocation / liveness** (as a pass on `LoopProgram`)
- **Launch-order optimizations**

Each transformation stays at one program level or lowers between adjacent
levels, never reaches back into the Graph.

## See also

- `compiler/ir/ARCHITECTURE.md` — per-dialect structural IR descriptions.
- `compiler/ARCHITECTURE.md` — the three-layer pipeline view.
