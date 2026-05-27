# CUDA Backend

CUDA-specific dispatch. Shared backend contract lives in
`backend/ARCHITECTURE.md`. The lowering chain that produces the
`Graph[CudaOp]` this backend consumes lives in `pipeline/passes/lowering/`
(see `pipeline/ARCHITECTURE.md`).

## Modules

```
cuda/
├── backend.py   # CudaBackend(Backend) — drives lowering + delegates dispatch
├── nvcc.py      # offline `nvcc --cubin` compile (+ content-addressed cubin cache) → RawModule load
└── program.py   # Graph[CudaOp] → kernel dispatch (via nvcc.py) + per-kernel event timing
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
- Compiles each unique `kernel_name` via `nvcc.load_function` (`nvcc.py`):
  offline `nvcc --cubin` into a content-addressed disk cache, then a cupy
  `RawModule` load. This avoids the driver PTX→SASS JIT cupy's NVRTC path
  pays on a cold compile — ~3× faster on the complex tile-search kernels that
  dominate autotune, and the compile step is GPU-free so the cubin cache can be
  warmed by a parallel pool (planned). Falls back to `cupy.RawKernel` (NVRTC)
  when `nvcc` is absent or a compile fails (`DEPLODOCK_NO_NVCC=1` forces the
  fallback). Kernels are emitted with `extern "C" __global__` so neither
  toolchain name-mangles them (and the cubin symbol loads by `kernel_name`).
  Compile vs load is split (`compile_to_cubin` / `load_function`) so the
  GPU-free compile can run off-process; the loaded `Function` is launch- and
  smem-attr-compatible with `RawKernel`.
- **Opt level** comes from `DEPLODOCK_NVCC_FLAGS` (`nvcc.effective_flags`): the
  CLI sets it — `tune` → `-Xcicc -O1`, `compile`/`run` → nvcc default -O3,
  `--nvcc-flags` overrides. -O1 dodges a cicc/LLVM front-end blowup on big
  unrolled register-tile kernels (cicc, not ptxas, is the cost: a tall-thin
  register tile unrolls into a ~5K-instruction basic block → up to 21 s → 0.1 s
  at -O1) but is **NOT runtime-optimal** (reductions/attention ~1.5–3× slower),
  so it's a tune-time *ranking* knob only. The flags are folded into both the
  cubin cache key and `Context.structural_key` (the `perf` context key), so
  -O1-tuned and -O3 measurements never collide. The bench-worker subprocess
  inherits the env, so its compiles use the same flags.
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

## Invariants

- `CudaOp.arg_order` embeds the original node id as the output buffer
  name. The lowering rules therefore mutate node ops **in place**
  instead of splicing a fresh node — see `pipeline/ARCHITECTURE.md`
  under "Rule module convention".
- The CUDA backend imports from `ir/` and `pipeline/` but never into
  them. A ROCm/SYCL/Metal backend replaces `program.py` only.
