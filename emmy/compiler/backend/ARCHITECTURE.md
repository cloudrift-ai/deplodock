# Backend Architecture

Three backends with a shared interpreter. Every backend exposes the
same two-step API (`backend/base.py`):

```python
compiled = backend.compile(graph)
result   = backend.run(compiled, input_data={"x": np.ndarray(...)})
# RunResult(outputs: dict[name, ndarray], time_ms: float | None)
bench    = await backend.benchmark_async(compiled, warmup=, num_iters=)  # async-only; callers asyncio.run at the CLI boundary
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
`torch.compile` baseline for `emmy run --ir`. Each frontend / tensor op is mapped to its torch twin
(`RmsNormOp`→`F.rms_norm`, `LayerNormOp`→`F.layer_norm`, `SdpaOp`→`F.scaled_dot_product_attention`, `LinearOp`→`F.linear`, `ElementwiseOp`/`ReduceOp`→
the torch elementwise/reduce, layout ops→view/transpose/cat). `is_runnable(graph)` is `True` only when every compute op
has a mapping — layout/data-dependent ops that appear post-decomposition (`IndexMapOp` / `GatherOp` / `ScatterOp`) are
unsupported, so `run --ir` falls back to emmy-only benchmarking for non-frontend IR. `build_callable(graph,
input_tensors)` returns a pure `fn(*tensors)` (scalar constants read inline) so `torch.compile` can trace it. Symbolic
graphs work too: `build_callable` binds every symbolic axis name to its concrete extent read off the supplied tensors
(the CUDA launch convention) and bakes the env into the per-node callables — shape-resolving sites (`ReshapeOp` target
shape, `IndexMapOp` out-shape and coord/select exprs) eval through it, so a dynamic-trace `<kname>.torch.json`
reproducer gets the same vs-torch comparison as a static one (benched at the `Dim` hint by
`commands/run.py::bench_lowered_vs_torch`, which sizes its random inputs by hint-resolving symbolic dims). Used to
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
including `LoopOp`: its `forward` delegates to `ir/loop/runner.py`
(`execute_loop_op_cpp`), which renders the body to C++ and JIT-compiles
it in-process via cppyy / Cling. That's why the same default `run` works
for pre-fusion graphs (LoopOp absent) and post-fusion graphs (LoopOp is
just another `Op` subclass).
`NumpyBackend` and `LoopBackend` inherit it verbatim; `CudaBackend`
overrides with cupy dispatch.

The default `benchmark` does wall-time iterations around `run`; the
CUDA backend overrides it to populate per-launch CUDA-event timings.

Result dataclasses:

- `RunResult(outputs, time_ms)`
- `BenchmarkResult(time_ms, min_ms?, max_ms?, num_launches, per_launch?, captured, e2e_ms?, e2e_min_ms?)` —
  `time_ms`/`min_ms` sum per-launch solo windows; `e2e_ms`/`e2e_min_ms` (automatic for multi-launch programs
  under capture) time the whole program as replays of one all-launches CUDA graph — the only
  end-to-end-comparable number for multi-kernel programs.
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
