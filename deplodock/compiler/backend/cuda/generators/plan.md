# Transformer Compiler — Status & Next Steps

## What We Built

Single unified code path: `analysis.py` → `tiled.py` handles all kernel patterns.

```
FusedRegionOp → analyze() → TileAnalysis → lower_tiled() → KernelDef → emit_kernel() → CUDA
```

### Kernel Generator (`tiled.py`)
- **Pointwise**: flat parallel loop, AST-based
- **Row reduction / reduce+broadcast**: 1 block/row, warp shuffle + smem, 2-pass epilogue
- **Contraction (naive)**: CTA-swizzle grid, 8×4 coarsened threads, 64×128 tiles, bounds-safe W() macro
- **Contraction (tma_db)**: same structure + TMA double-buffered shared memory pipeline, mbarrier pipelining
- Naive and TMA share: grid, thread mapping, accumulators, FMA structure, write macro
- k_splits: K-dimension parallelism via blockIdx.z + atomicAdd

### Fusion (`fusion.py`)
- Greedy merge with convexity + codegen feasibility checks
- Multi-consumer edges allowed (enables softmax max+epilogue fusion)
- Constraints: convexity, contraction isolation (pure 2-op only), single reduce per region, 2D dimensionality, shape compatibility

### Program Pipeline (`program.py` → `backend.py`)
- TmaDescriptorSpec: cuTensorMapEncodeTiled + cudaFuncSetAttribute in generated source
- Tuning profiles: per-GPU BK, k_splits, strategy selection from `tuning.py`
- Automatic fallback: naive for K < 32, reduced k_splits when K too small

### Correctness
- 476 tests pass
- Reference tests: SiLU, reduce_sum, RMSNorm, matmul, softmax, full TinyLlama pipeline
- Qwen2.5-7B seq_len=32: 114688/114688 nonzero, no NaN

## Current Performance

### Qwen2.5-7B layer 0, seq_len=32, batch=1, fp32, RTX 5090

| Backend | Latency (us) | vs Eager |
|---------|-------------|----------|
| Eager PyTorch | 1110 | 1.00x |
| torch.compile | 1038 | 1.07x |
| **Deplodock** | **1871** | **0.59x** |

### Qwen2.5-7B layer 0, seq_len=512, batch=1, fp16, RTX 5090

| Backend | Latency (us) | vs Eager |
|---------|-------------|----------|
| Eager PyTorch | 1347 | 1.00x |
| torch.compile | 1269 | 1.06x |
| **Deplodock** | **13322** | **0.10x** |

### TinyLlama fixture, seq_len=32, fp32

| Metric | Value |
|--------|-------|
| Latency | 676 us |
| TMA matmuls | 7 |
| Total launches | 47 |
| Correctness | NaN=0, 65536/65536 nonzero |

## Bottleneck Analysis

### Why seq_len=32 is 1.7x slower than eager

Per-kernel profiling (Qwen2.5-7B, seq_len=32):

| Kernel | Time (us) | Blocks | Issue |
|--------|-----------|--------|-------|
| Q/O-proj (32×3584 @ 3584) | 134 each | 224 | SM underutilization (M=32 < tile_m=64) |
| K/V-proj (32×512 @ 3584, k_splits=4) | 36 each | 128 | OK with k_splits |
| gate/up-proj (32×18944 @ 3584) | 250 each | 1184 | Good occupancy but long K-loop |
| down-proj (32×3584 @ 18944) | 727 | 224 | Longest K-loop, poor M occupancy |
| All fused kernels | ~35 total | — | Negligible |
| **Total matmul** | **~1587** | | 90% of total time |
| **Total** | **~1740** | | |

Root cause: **TMA tiles (64×128) are too large for M=32**. Each tile covers the entire M dimension with half the threads idle. cuBLAS auto-selects smaller tiles.

### Why seq_len=512 is 10x slower than eager

Not yet profiled. Likely: attention ops (QK^T, softmax, attn@V) are unfused 3D/4D operations going through the singleton path (1-op-per-kernel, no TMA). At seq_len=512, attention is the dominant computation.

## What's Missing

### Codegen Limitations (blocks fusion + performance)

1. **2D-only indexing**: The codegen uses `row * cols + j` for all tensors. Ops with >2 non-trivial dimensions (attention heads × seq × dim) can't be fused. This forces attention into ~15 separate single-op kernels.

2. **Single reduction pass**: Only one reduce per kernel. Softmax (max + sum) requires two separate kernels. Multi-pass reduction needs codegen support for multiple accumulator sets + warp shuffles.

3. **No contraction + epilogue fusion**: Contraction regions are isolated to exactly 2 ops (mul + reduce). Matmul + bias add, matmul + activation, etc. can't fuse. The `_detect_contraction` in `analysis.py` needs to handle prologue/epilogue around the contraction core.

4. **No FP16 support**: All computation is FP32. FP16 needs `__half` types, vectorized loads, and potentially tensor core instructions.

### Fusion Limitations

5. **Contraction isolation too strict**: A matmul + residual add should be one kernel (the add is just a post-k-loop epilogue). Currently blocked because `_can_merge` requires pure 2-op contraction regions.

6. **Multi-reduce blocked**: Softmax max + sum can't share a kernel. Need multi-pass codegen first.

7. **>2D ops blocked**: Any op producing a tensor with >2 non-trivial dims is blocked from fusion. This excludes all attention ops.

### Infrastructure

8. **No cuBLAS fallback**: For small M, cuBLAS is 3-5x faster than our TMA. A runtime decision based on M could dispatch to cuBLAS.

9. **No multi-output regions**: Plan/backend only supports single-output fused regions. Multi-consumer nodes that have consumers in different regions force materialization.

10. **Tuning gap for rectangular matrices**: The tuning profile uses `max(M, N)` as size proxy. For M=32, N=3584 matrices, `max=3584` selects configs optimized for large square matrices (no k_splits). Need M-aware tuning.

## Recommended Priority Order

### High impact, moderate effort
1. **Contraction + epilogue fusion** — fuse matmul + bias/activation. Unblocks ~10 fewer kernel launches per layer. Requires extending `_detect_contraction` to handle prologue/epilogue ops.
2. **M-aware k_splits** — use `min(M, N)` or block count to decide k_splits, not max dim. Simple heuristic change in `_compile_matmul`.
3. **cuBLAS fallback** — for M < 64, emit `cublasSgemm` instead of our kernel. Biggest single perf win for small batch inference.

### High impact, high effort
4. **Multi-dimensional codegen** — support 3D/4D tensor indexing in reduction kernels. Enables attention op fusion. Major codegen rewrite.
5. **Multi-pass reduction** — codegen support for softmax (max + sum in one kernel). Requires multiple accumulator sets + warp shuffles.
6. **FP16 support** — `__half` types, `__hfma2`, FP16 TMA descriptors.

### Lower priority
7. **Flash attention pattern** — online softmax with correction (Neptune approach). Depends on #4 and #5.
8. **Horizontal fusion** — fuse independent parallel ops (like combo kernels in Inductor).
9. **General tuning infrastructure** — ILP or lookup-table based tuning profiles indexed by (pattern, M, N, K).

## Files

```
deplodock/compiler/
├── fusion.py                        # Greedy fusion with convexity + codegen checks
├── backend/cuda/
│   ├── backend.py                   # _compile_fused_region, _compile_matmul dispatch
│   ├── generators/
│   │   ├── analysis.py              # TileAnalysis: pattern classification
│   │   └── tiled.py                 # Unified generator: all patterns + strategies
│   ├── program.py                   # TmaDescriptorSpec, Launch, generate_source
│   ├── codegen.py                   # KernelDef → CUDA C source
│   ├── ir.py                        # CUDA AST (Expr, Stmt, KernelDef)
│   ├── tuning.py                    # Per-GPU strategy profiles
│   └── runner.py                    # Legacy single-kernel runner
├── ir.py                            # Graph, Node, Tensor
├── ops.py                           # Op types (FusedRegionOp, etc.)
├── plan.py                          # ExecutionPlan, plan_graph
└── hints.py                         # Hints metadata
```
