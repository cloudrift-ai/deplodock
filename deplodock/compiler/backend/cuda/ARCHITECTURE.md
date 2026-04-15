# CUDA Backend Architecture

The CUDA backend maps `OpKernel` tags from the execution plan to CUDA C kernels,
compiles them via nvcc, and runs them on GPU. It extends the shared backend
infrastructure (`backend/program.py`, `backend/kernel_ir.py`, `backend/loop_ir.py`,
`backend/codegen.py`) with CUDA-specific features: TMA descriptors, nvcc
compilation, and GPU-tuned kernel generators.

## Module Layout

```
cuda/
├── backend.py        # CudaBackend(Backend): compile plan → Program
├── program.py        # CudaLaunch, TmaDescriptorSpec, source gen, nvcc, run
├── schedule.py       # Schedule: all kernel structure decisions as a dataclass
├── generators/       # Kernel code generation
│   ├── analysis.py       # TileAnalysis: classify KernelOp patterns (dispatch on core type)
│   ├── loop_lower.py     # lower_generic(): Schedule-driven LoopIR emission
│   ├── loop_codegen.py   # LoopIR → KernelDef (imperative C AST)
│   └── tiled.py          # Public API: generate_kernel(), lower_tiled()
├── runner.py         # Single-kernel compile + run + benchmark harness
└── tuning.py         # Per-GPU empirical tuning profiles (RTX 5090, H200, Pro 6000)
```

## Schedule + LoopIR Pipeline

Kernel generation is driven by a `Schedule` dataclass that encodes all kernel
structure decisions (grid mapping, accumulator shape, reduction loops, etc.).
A single `lower_generic()` reads the Schedule and emits LoopIR — no pattern
matching.

```
KernelOp → analyze() → TileAnalysis ──→ build_schedule() → Schedule
                                                                ↓
                CUDA source ← emit_kernel() ← KernelDef ← loop_ir_to_kernel(LoopProgram, Schedule)
                                                                                 ↑
                                                                     lower_generic() → LoopProgram
```

`analyze()` is pure assembly — it calls accessor methods on
`KernelOp` (Layer 1 IR) and packs the results into ``TileAnalysis``.
Source of truth lives on `KernelOp`:
- `body_ops()`, `phases()`, `reduce_fn_names()`, `port_indexmaps()` —
  pure structural decomposition of the core annotation.
- `input_accesses(shapes, output_shape)` — per-input AccessPattern dict.
- `contraction_info(shapes)` — `ContractionInfo | None` resolving M/N/K
  + batch_dims + GQA batch groups for ContractionCore kernels.
- `epilogue_needs_per_element(shapes, output_shape)` — bool indicating
  whether the epilogue requires a second per-element pass (RMSNorm-style).
- `tile_pattern(shapes, output_shape)` →
  `"pointwise" | "row_reduce" | "reduce_broadcast" | "contraction"`.
- `tile_dims(shapes, output_shape)` → `(rows, cols, k_dim)`.

Combined matmul→softmax kernels carry their downstream row reduces in
`core.post_stages`; `KernelOp.phases()` walks prologue + post_stages to
build the reduces / inter_reduce / epilogue alternation.

The `KernelOp.body_ops()` method returns a dedup'd
`(id, op, inputs)` list covering prologue + core + epilogue in topo
order — this is the one backend reader that still needs a flat walk
(pointwise body emit in `tiled.py` / `loop_lower.py`). All other
readers use the structured fields directly.

LoopProgram is purely structural (name, params, body, dim_strides).  Backend
metadata (block_size, tile dims, TMA config, batching, includes) lives on the
Schedule and is read by ``loop_ir_to_kernel()`` when building the KernelDef.

**Schedule** (`schedule.py`) contains: `GridSpec` (1d/2d_swizzle/2d_standard),
`AccumulatorSpec` (None/scalar/register-tile), `ReductionSpec` (loop params +
warp_reduce flag), plus tile dims, k_splits, load strategy, and batching.

**LoopIR** (`backend/loop_ir.py`) makes the loop structure explicit:
`ParallelAxis`, `LoopNest`, `Alloc`, `Load`/`Store`, `Compute`,
`Accumulate`/`WarpReduce`, `Guard`, `RegAccess`, `RawLoopOp`.

Pointwise, row-reduce, multi-reduce, and naive contraction are fully expressed
in LoopIR via `lower_generic()`. TMA and smem contraction strategies use
`SmemPipelineKLoop` (expanded to LoopIR or rendered as inline PTX by codegen).

### Online reduction for contraction + multi-reduce

When a contraction (matmul) is followed by a multi-reduce (e.g. softmax),
the lowering code emits a **generic online reduction** via
`_emit_online_contraction_reduce()`. Instead of parallelizing over N
(which requires one CTA to own the full row for reductions), N is tiled
sequentially:

1. **Loop 1**: K-loop + head reduce (e.g. max) + write raw scores to output
2. **Loop 2..n**: For each subsequent reduce, load raw scores, apply
   inter-reduce ops, compute running reduce via warp_xor
3. **Final loop**: Load raw scores, apply all inter-reduce + epilogue ops
   using final reduce values, write normalized output

This works for any multi-reduce pattern (softmax, log-sum-exp, etc.)
without hardcoded correction factors. The grid is 1D over M-tiles
(`GridSpec("1d_contraction")`).

**TODO**: When N <= tile_n, the N-tile loops have 1 iteration. A future
KernelIR optimizer pass could fuse them into a single in-register pass.

## SGEMM Strategies (generators/tiled.py)

| Strategy          | Description                           | Best for                |
|-------------------|---------------------------------------|-------------------------|
| `naive`           | Direct global loads, no shared memory | Test baseline           |
| `smem`            | Shared-memory A + global B, splitK    | Opt-in, benchmarking    |
| **`tma_db`**      | **TMA double-buffer, size-adaptive**  | **Production default**  |
| `tma_db_tf32`     | TF32 via tensor cores (wmma)          | TF32 precision ok       |
| `tma_db_fma_tf32` | Concurrent FMA + TF32 hybrid          | Mixed precision ok      |

### Strategy selection in `_select_strategy` (backend.py)

All op patterns (contraction, pointwise, reduce) are compiled through a single
`_compile_single()` entry point. For contractions, `_select_strategy()` loads
the GPU tuning profile and applies hint overrides:

- **TMA** is always used for the compiler pipeline (best end-to-end despite
  per-kernel cuBLAS gap for small M)
- **smem** is opt-in via `cuda.matmul.strategy` hint or `--strategy smem`
  in `bench_matmul.py`
- **M-aware thread_m clamping**: `tile_m = ty * thread_m <= M` (avoids wasting tile rows)
- **M-aware k_splits**: when `M < tile_m`, increases k_splits to fill the GPU
- **Epilogue fusion**: contraction + bias/activation/residual add fused in-register
  after K-loop (k_splits forced to 1 when epilogue present)
- **Batched contraction**: `TileAnalysis.batch_dims` detects >2D matmul operands
  (e.g. multi-head attention QK^T). Uses `blockIdx.z` for batch loop with pointer
  offsets (`A+batch*M*K`, `B+batch*K*N`, `C+batch*M*N`). Currently naive-only
  (TMA batched descriptors in program.py not yet supported).

### Kernel structure (TMA double-buffer)

```
Grid: CTA-swizzle (linearized blockIdx.x), blockIdx.z for k_splits
Block: (32, 8) = 256 threads
Each thread: thread_m rows x 4 cols = thread_m*4 outputs

Phase 1: CTA-swizzle grid → (bm, bn) tile coordinates
Phase 2: mbarrier init, first TMA prefetch
Phase 3: Double-buffered K-loop
  - Wait on mbarrier (current stage)
  - Prefetch next tile (thread 0 only)
  - FMA: thread_m x 4 accumulator registers
Phase 4: Epilogue (optional): bias add, activation, etc.
Phase 5: Write via W() macro (atomicAdd when k_splits > 1)
```

### Kernel structure (smem)

```
Grid: standard 2D (blockIdx.x=cols, blockIdx.y=rows), blockIdx.z for k_splits
Block: (32, 4) = 128 threads
Each thread: thread_m rows x 4 cols

Phase 1: Thread-to-output mapping (row_base, col_base)
Phase 2: K-tile loop
  - Load A tile into shared memory (padded stride for bank conflicts)
  - __syncthreads
  - FMA: A from shared, B from global (float4 fast path + scalar fallback)
  - __syncthreads
Phase 3: Write (direct or atomicAdd for k_splits > 1)
```

## Tuning Profiles (tuning.py)

Per-GPU strategy maps indexed by `max(M, N)`. Each entry: `(threshold, hints_dict)`.

### RTX 5090 (sm_120, 170 SMs)

| Size threshold | Strategy | thread_m | block_k | k_splits |
|---------------|----------|----------|---------|----------|
| 256 | tma_db | 8 | 32 | 4 |
| 512 | tma_db | 8 | 32 | 4 |
| 1024 | tma_db | 8 | 32 | 1 |
| 2048 | tma_db | 26 | 32 | 1 |
| 4096 | tma_db | 20 | 32 | 1 |
| 8192 | tma_db | 28 | 32 | 1 |
| 16384 | tma_db | 28 | 32 | 1 |

When M < tile_m, `_select_strategy` clamps thread_m to `M // ty` and may
increase k_splits to fill the GPU (up to 8).

## Performance (RTX 5090, sm_120, batch=1)

### Square matrices (TMA)

| Size     | TM   | BK   | K-splits  | Eff vs cuBLAS | TFLOPS  |
|----------|------|------|-----------|---------------|---------|
| **1024** | 8    | 32   | 1         | **101%**      | 49.0    |
| **2048** | 26   | 32   | 1         | 83%           | 57.0    |
| **4096** | 20   | 32   | 1         | **97%**       | 58.9    |
| 8192     | 28   | 32   | 1         | 97%           | 56.9    |

### Rectangular matrices (transformer block shapes, seq_len=32)

| Shape (MxNxK)  | Eff vs cuBLAS | Notes |
|----------------|---------------|-------|
| 32x2048x2048   | 18% | Q/O projection (TinyLlama). cuBLAS uses CUTLASS simt_128x32 |
| 32x5632x2048   | 31% | gate/up projection |
| 32x18944x3584  | 70% | gate/up projection (Qwen 7B) |
| 128x2048x2048  | 33% | Q/O projection at seq_len=128 |
| 128x18944x3584 | **97%** | Matches cuBLAS at seq_len=128 with large N |

For small M (32-128), cuBLAS dispatches `cutlass_80_simt_sgemm_128x32_8x5` — a
CUTLASS SIMT kernel with 128x32 tiles and 40 threads, plus a separate splitK
reduce kernel. Our TMA kernel has higher per-block setup cost (mbarrier init,
descriptor setup) that dominates when there are few blocks.

### End-to-end transformer block (TinyLlama, seq_len=32, fp32)

| Backend | Latency (us) | vs Eager |
|---------|-------------|----------|
| Eager PyTorch | 319 | 1.00x |
| torch.compile | 222 | 1.44x |
| **Deplodock** | **282** | **1.13x** |

The remaining gap vs torch.compile is primarily from attention (10 separate 4D
singleton kernels vs fused SDPA), not matmul efficiency.

## Benchmark Experiment

Reproducible via:
```bash
# All shapes, all batches, TMA vs smem
deplodock bench experiments/sgemm_cublas_vs_tma --local

# Just batch=1 TMA
deplodock bench experiments/sgemm_cublas_vs_tma --local --filter "batch=1" --filter "strategy=adaptive"

# Just batch=1 smem
deplodock bench experiments/sgemm_cublas_vs_tma --local --filter "batch=1" --filter "strategy=smem"
```

Results land in `experiments/sgemm_cublas_vs_tma/<timestamp>/report.md`.

## Buffer Aliases

Reshape/transpose ops produce buffer aliases instead of empty kernel launches.
The output buffer shares the input's device pointer — no allocation, no launch.
Aliases resolve transitively (reshape of reshape → single alias to the original).

## CudaLaunch

`CudaLaunch` extends the base `Launch` (from `backend/program.py`) with
`tma_descriptors: list[TmaDescriptorSpec]`. TMA descriptors are created at
runtime via `cuTensorMapEncodeTiled` and passed as `__grid_constant__` kernel
parameters.
