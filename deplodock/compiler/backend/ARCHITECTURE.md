# Backend Architecture

Three backends with a shared interpreter. Every backend exposes the
same two-step API (`backend/base.py`):

```python
compiled = backend.compile(graph)
result   = backend.run(compiled, input_data={"x": np.ndarray(...)})
# RunResult(outputs: dict[name, ndarray], time_ms: float | None)
bench    = backend.benchmark(compiled, input_data=…, warmup=, num_iters=)
# BenchmarkResult(time_ms, min_ms, max_ms, num_launches, per_launch)
```

What differs is `compile()` — how far the backend lowers the graph:

| Backend        | `compile(graph)` does                              | `run()` goes through |
|----------------|----------------------------------------------------|----------------------|
| `NumpyBackend` | returns the graph as-is (no-op)                    | `interpret_graph`    |
| `LoopBackend`  | runs decomposition → optimization → fusion         | `interpret_graph`    |
| `CudaBackend`  | fusion + `lowering/kernel` + `lowering/cuda`       | cupy/NVRTC dispatch  |

`numpy` and `loop` backends share the same runtime path — the only
distinction is whether the graph has been fused yet. See
`backend/cuda/ARCHITECTURE.md` for the CUDA-specific dispatch.

## Shared interpreter (`interpret.py`)

`interpret_graph(graph, input_data) → dict[name, ndarray]` walks the
graph in topological order and calls `node.op.forward(*args)` at each
compute node. `InputOp` and `ConstantOp` boundaries are seeded from
`input_data`; post-compute outputs are reshaped to
`node.output.shape` when the target shape is statically known.

Every op implements `forward(*inputs: np.ndarray) → np.ndarray`,
including `LoopOp`: its `forward` delegates to
`ir/loop/interpret.execute_loop_op`, the body-level numpy interpreter.
That's why the same walker works for pre-fusion graphs (LoopOp absent)
and post-fusion graphs (LoopOp is just another `Op` subclass).

Consumed by `NumpyBackend.run` and `LoopBackend.run` verbatim.

## Backend ABC (`base.py`)

`Backend` is an abstract base class with `compile`, `run`, `benchmark`.
The default `benchmark` does wall-time iterations around `run`; the
CUDA backend overrides it to populate per-launch CUDA-event timings.

Result dataclasses:

- `RunResult(outputs, time_ms)`
- `BenchmarkResult(time_ms, min_ms?, max_ms?, num_launches, per_launch?)`
- `LaunchTime(idx, kernel_name, time_ms)` — one per kernel per bench run.

## Numpy backend (`numpy/`)

Thinnest backend. `compile` returns the graph; `run` calls
`interpret_graph`. Used for correctness testing (no GPU required) and
as the ground truth the loop and CUDA backends are triangulated
against.

## Loop backend (`loop/`)

Runs the fusion pipeline to turn the graph into `Graph[LoopOp]`, then
executes via `interpret_graph`. Since `LoopOp.forward` works, this is
the numpy backend with fusion in front.

Used as the second axis of triangulation: **loop vs numpy disagreement
implicates fusion; loop vs CUDA disagreement implicates codegen.**

## CUDA backend (`cuda/`)

See `cuda/ARCHITECTURE.md`. Runs the full lowering chain and dispatches
kernels via cupy `RawKernel` (NVRTC-compiled).

## Invariants

- `interpret_graph` must work on any graph where every op has
  `forward`. It doesn't know about dialects — it just dispatches
  through `Op.forward`.
- A new backend (ROCm, SYCL, Metal) reuses `ir/` and
  `pipeline/passes/lowering/` wholesale; only its own dispatch layer
  (equivalent of `cuda/program.py`) needs to be written.
