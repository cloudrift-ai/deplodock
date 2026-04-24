# CUDA Backend

CUDA-specific dispatch. Shared backend contract lives in
`backend/ARCHITECTURE.md`. The lowering chain that produces the
`Graph[CudaOp]` this backend consumes lives in `pipeline/passes/lowering/`
(see `pipeline/ARCHITECTURE.md`).

## Modules

```
cuda/
├── backend.py   # CudaBackend(Backend) — drives lowering + delegates dispatch
├── program.py   # Graph[CudaOp] → cupy.RawKernel dispatch + per-kernel event timing
└── runtime.py   # has_cuda_gpu() predicate (cupy-based)
```

## Compile

`CudaBackend.compile(graph)` runs
`run_pipeline(graph, [..., "lowering/kernel", "lowering/cuda"], dump=…)`,
producing a `Graph[CudaOp]` where every compute node carries a rendered
`__global__` source plus its launch geometry (grid / block / smem /
arg_order).

## Dispatch (`program.py`)

`_compile(graph) → _Compiled` walks the lowered graph:

- Classifies each node as `input` / `constant` / `output` / `scratch`
  from `graph.inputs` / `ConstantOp` membership / `graph.outputs`.
- Compiles each unique `kernel_name` via `cupy.RawKernel(source, name,
  options=("--use_fast_math",))`. Kernels are emitted with
  `extern "C" __global__` so NVRTC doesn't name-mangle them.
- Builds a static launch plan: per launch, a tuple of
  `(kernel, arg_names, grid, block, smem_bytes, zero_outputs)`.

`run_program(graph, input_data) → RunResult`:

1. Allocate a `cupy.ndarray` for every buffer. Inputs + optional
   constant overrides come from `input_data`; scalar `ConstantOp`s
   materialize as single-element arrays; scratch buffers zero-init.
   Inputs without supplied data get a deterministic pseudo-random fill
   (useful for standalone compile-and-benchmark scripts).
2. Launch each kernel in topological order; `zero_outputs` fills run
   before the launch.
3. Copy `graph.outputs` buffers back to numpy.

`benchmark_program(graph, input_data, warmup, num_iters)` adds a
warmup loop + timed loop wrapped in `cupy.cuda.Event` pairs — one
global pair for `BenchmarkResult.time_ms`, one pair per launch index
(averaged over iters) for `BenchmarkResult.per_launch`.

`run_program_debug(...)` snapshots every non-input buffer after each
launch — consumed by `--dump-dir` runs.

## GPU detection (`runtime.py`)

`has_cuda_gpu()` returns `True` iff `cupy` imports and
`cupy.cuda.runtime.getDeviceCount() > 0`. Used as a pytest skip
predicate.

## Invariants

- `CudaOp.arg_order` embeds the original node id as the output buffer
  name. The lowering rules therefore mutate node ops **in place**
  instead of splicing a fresh node — see `pipeline/ARCHITECTURE.md`
  under "Rule module convention".
- The CUDA backend imports from `ir/` and `pipeline/` but never into
  them. A ROCm/SYCL/Metal backend replaces `program.py` only.
