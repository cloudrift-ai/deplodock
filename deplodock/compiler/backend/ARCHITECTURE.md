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
| `NumpyBackend` | returns the graph as-is (no-op)                    | default `Backend.run`|
| `LoopBackend`  | runs decomposition → optimization → fusion         | default `Backend.run`|
| `CudaBackend`  | fusion + `lowering/kernel` + `lowering/cuda`       | cupy/NVRTC dispatch  |

`numpy` and `loop` backends share the same runtime path — the only
distinction is whether the graph has been fused yet. See
`backend/cuda/ARCHITECTURE.md` for the CUDA-specific dispatch.

## Torch reference (`torch_ref.py`)

Not a `Backend` — a small Graph→torch evaluator that runs a frontend-dialect graph through **real PyTorch**, the eager /
`torch.compile` baseline for `deplodock run --ir`. Each frontend / tensor op is mapped to its torch twin
(`RmsNormOp`→`F.rms_norm`, `SdpaOp`→`F.scaled_dot_product_attention`, `LinearOp`→`F.linear`, `ElementwiseOp`/`ReduceOp`→
the torch elementwise/reduce, layout ops→view/transpose/cat). `is_runnable(graph)` is `True` only when every compute op
has a mapping — layout/data-dependent ops that appear post-decomposition (`IndexMapOp` / `GatherOp` / `ScatterOp`) are
unsupported, so `run --ir` falls back to deplodock-only benchmarking for non-frontend IR. `build_callable(graph,
input_tensors)` returns a pure `fn(*tensors)` (scalar constants read inline) so `torch.compile` can trace it. Used to
turn a dumped `<kname>.torch.json` reproducer into an accuracy + latency comparison vs torch — see `../provenance.py`
and `commands/run.py:_handle_run_ir`.

## Backend ABC (`base.py`)

`Backend` is an abstract base class with `compile`, `run`, `benchmark`.

`Backend.run` provides a default implementation: walks the compiled
graph in topological order and calls `node.op.forward(*args)` at each
compute node. `InputOp` and `ConstantOp` boundaries are seeded from
`input_data`; post-compute outputs are reshaped to `node.output.shape`
when the target shape is statically known.

Every op implements `forward(*inputs: np.ndarray) → np.ndarray`,
including `LoopOp`: its `forward` delegates to
`ir/loop/interpret.execute_loop_op`, the body-level numpy interpreter.
That's why the same default `run` works for pre-fusion graphs (LoopOp
absent) and post-fusion graphs (LoopOp is just another `Op` subclass).
`NumpyBackend` and `LoopBackend` inherit it verbatim; `CudaBackend`
overrides with cupy dispatch.

The default `benchmark` does wall-time iterations around `run`; the
CUDA backend overrides it to populate per-launch CUDA-event timings.

Result dataclasses:

- `RunResult(outputs, time_ms)`
- `BenchmarkResult(time_ms, min_ms?, max_ms?, num_launches, per_launch?)`
- `LaunchTime(idx, kernel_name, time_ms)` — one per kernel per bench run.

## Numpy backend (`numpy/`)

Thinnest backend. `compile` returns the graph; `run` is inherited from
`Backend`. Used for correctness testing (no GPU required) and as the
ground truth the loop and CUDA backends are triangulated against.

## Loop backend (`loop/`)

Runs the fusion pipeline to turn the graph into `Graph[LoopOp]`, then
executes via the inherited `Backend.run`. Since `LoopOp.forward` works,
this is the numpy backend with fusion in front.

Used as the second axis of triangulation: **loop vs numpy disagreement
implicates fusion; loop vs CUDA disagreement implicates codegen.**

## CUDA backend (`cuda/`)

See `cuda/ARCHITECTURE.md`. Runs the full lowering chain and dispatches
kernels via cupy `RawKernel` (NVRTC-compiled).

## Invariants

- The default `Backend.run` must work on any graph where every op has
  `forward`. It doesn't know about dialects — it just dispatches
  through `Op.forward`.
- A new backend (ROCm, SYCL, Metal) reuses `ir/` and
  `pipeline/passes/lowering/` wholesale; only its own dispatch layer
  (equivalent of `cuda/program.py`) needs to be written.
