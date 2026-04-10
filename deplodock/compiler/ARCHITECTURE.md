# Compiler Architecture

## Overview

Minimal tensor IR and regex-style graph transformation engine. Represents tensor computations with a small canonical op set, then applies ordered rewrite rules to progressively transform naive expressions into optimized forms.

## Module Layout

```
compiler/
├── ops.py         # Op base class + all op types (minimal + fused)
├── ir.py          # Tensor, Node, Graph — the compute graph data structure
├── pattern.py     # Pattern AST + text parser for matching rules
├── matcher.py     # Graph pattern matching engine
├── rewriter.py    # Pass/Rule/Rewriter — rule loading and application
├── trace.py       # CompilerTrace — structured JSON trace for AI-in-the-loop
├── pipeline.py    # compile_and_run() + compile_graph() — pipeline entry points
├── benchmark.py   # Multi-size benchmark harness with cuBLAS comparison
├── torch_trace.py # PyTorch module → Graph IR conversion (optional torch dep)
├── rules/         # Ordered rule files organized by pass
│   ├── fusion/    # Fuse ops: RMSNorm, matmul, softmax, SiLU+mul
│   └── tiling/    # Tile loops for memory hierarchy
└── cuda/          # CUDA backend
    ├── ir.py      # Imperative AST (Expr, Stmt, KernelDef, VectorLoad, etc.)
    ├── codegen.py # CUDA IR → CUDA C source string
    ├── lower.py   # Graph IR → CUDA IR (MatmulConfig + strategy dispatch)
    └── runner.py  # Compile with nvcc + execute + benchmark + parse output
```

## IR Design

- **Ops** are plain dataclasses inheriting from `Op`. No enum — the class name is the type. New ops are added by subclassing.
- **Nodes** reference inputs by string ID. Fan-out is natural — multiple nodes can list the same input ID.
- **Graph** is an ordered dict of nodes with `add_node`, `remove_node`, `replace_node`, `consumers`, `topological_order`, `to_dict`, and `from_dict`.

### Op Categories

**Minimal ops** (for lowering from PyTorch):
- `InputOp` — dynamic input tensor
- `ConstantOp(name)` — fixed tensor (weights, RoPE tables, scalars)
- `ElementwiseOp(fn)` — scalar function per element (mul, add, exp, rsqrt, neg, recip, ...)
- `ReduceOp(fn, axis)` — collapse dimension (sum, max, prod)
- `ScanOp(fn, axis)`, `GatherOp(axis)`, `ScatterOp(axis, reduce_fn)` — other primitives
- `TransposeOp(axes)` — permute dimensions
- `ReshapeOp(shape)` — reshape without data copy

**Fused ops** (assembly targets):
- `MatmulOp` — matrix multiply (from `Reduce{sum}(Elementwise{mul})`)
- `FusedRMSNormOp(eps)` — rsqrt(mean(x²) + eps) * x * weight
- `FusedSoftmaxOp(axis)` — online softmax
- `FusedSiLUMulOp` — silu(gate) * up
- `FusedAttentionOp(num_heads, head_dim, scale)` — flash attention (future)
- `FusedReduceElementwiseOp(reduce_fn, elementwise_fn, axis)` — generic fused reduce

## Pattern Language

Text syntax parsed into an AST:
```
Reduce{sum, $k}(Elementwise{mul}($A, $B))
```
- Op names map to class names via `_OP_CLASS_MAP`
- `$var` captures subgraphs; same name = same node (fan-out)
- `|` separates alternatives (for commutativity)
- `_` wildcard matches any single node

## Rewrite Engine

- **Rules**: Python files exporting `PATTERN` string + `rewrite(graph, match)` function
- **Passes**: Subdirectories of `rules/` (fusion, tiling, etc.), each applied to fixed point
- **Ordering**: Passes run sequentially; within a pass, rules apply in filename order with restart-on-match

### Fusion Rules (Incremental Assembly)

Rules are numbered to control ordering. Earlier rules match longer chains to prevent shorter rules from consuming their sub-patterns.

| Rule | Pattern | Output |
|------|---------|--------|
| `000_fuse_rmsnorm` | `(x * rsqrt(sum(x*x) * inv_n + eps)) * w` | `FusedRMSNormOp` |
| `001_fuse_reduce_elementwise` | `Reduce{sum}(Elementwise{mul}(A, B))` | `MatmulOp` |
| `002_fuse_softmax` | `exp(x-max(x)) / sum(exp(x-max(x)))` | `FusedSoftmaxOp` |
| `003_fuse_silu_mul` | `gate * recip(1 + exp(-gate)) * up` | `FusedSiLUMulOp` |

**Key ordering constraint**: RMSNorm (000) must run before matmul (001) because RMSNorm contains `Reduce{sum}(Elementwise{mul}($x, $x))` which the matmul rule would otherwise consume.

**Future**: `010_fuse_attention` would match `Matmul → Scale → FusedSoftmax → Matmul` and produce `FusedAttentionOp` (flash attention).

## PyTorch Tracer

`torch_trace.py` converts a PyTorch module to our Graph IR via `torch.export`. ATen ops map to our minimal opset; parameters become `ConstantOp` nodes. PyTorch is an optional dependency.

## Data Flow

```
Graph → matcher.match_pattern(graph, pattern) → list[Match]
Match → rule.rewrite(graph, match) → Graph (modified)
Graph → Pass.apply(graph) → Graph (fixed point)
Graph → Rewriter.apply(graph) → Graph (all passes)
Graph → cuda.lower_graph(graph) → KernelDef
KernelDef → cuda.emit_kernel(kernel) → CUDA C source
source → cuda.run_kernel(...) → list[float] output
```

## CUDA Backend

Two-level IR design:
- **Graph IR** (high-level): declarative tensor ops, pattern-matched and rewritten
- **CUDA IR** (low-level): imperative AST — expressions, statements, loops, thread mapping

Key CUDA IR expression types include `VectorLoad` (float4 coalesced loads), `FieldAccess` (struct field access for float4.x/.y/.z/.w), `Ternary`, and `FuncCall`. Statement types include `ArrayDecl` (shared memory), `PragmaUnroll`, `IfStmt` (with optional else_body), and `ForLoop` with optional step.

### Lowering Strategies

Lowering is controlled by `MatmulConfig(strategy=...)`. Available strategies:

| Strategy | Description | Best for |
|----------|-------------|----------|
| `naive` | 1 thread per output element, direct global loads | Test baseline |
| **`tma_db`** | **TMA double-buffer, size-adaptive tile selection** | **Production default** |
| `tma_db_tf32` | Pure TF32 via tensor cores (wmma) | TF32 precision ok |
| `tma_db_fma_tf32` | Concurrent FMA + TF32 hybrid (both pipes) | Mixed precision ok |

### Size-Adaptive Strategy Selection

The production default is `adaptive` (in `bench_matmul.py`) which uses the `tma_db` strategy
with per-size tile parameters from `tuning.py`. The adaptive map selects the best `thread_m`
and `block_k` for each matrix size based on empirical benchmarking.

#### sm_120 (Blackwell, CUDA 13.2, cuBLAS 13.3)

cuBLAS on sm_120 uses CUTLASS `simt_sgemm_256x128_8x4` (pure FP32, not tensor cores). ncu profiling shows 73% FMA utilization vs our 68% — the gap is SASS-level instruction scheduling (warp phase alignment from ptxas's LDS distribution choices).

| Size | thread_m | BK | Block | Eff vs cuBLAS | TFLOPS |
|------|----------|-----|-------|--------------|--------|
| 256  | 6        | 256 | 32x4  | **104%** | 3.0  |
| 512  | 6        | 256 | 32x4  | **169%** | 14.5 |
| 1024 | 6        | 256 | 32x4  | 85%  | 34.6 |
| 2048 | 8        | 32  | 32x4  | 77%  | 47.6 |
| 4096 | 12       | 64  | 32x4  | 82-86%  | 51-58 |
| 8192 | 16       | 32  | 32x4  | 76%  | 51.4 |
| 16384| 12       | 64  | 32x4  | 71%  | 48.9 |

#### TMA Double-Buffer with K-Splitting (FP32-accurate, best overall)

Strategy `tma_db` uses TMA (Tensor Memory Accelerator) for zero-overhead global→shared loading with double-buffer pipelining. TMA loading overlaps with FMA computation on separate hardware.

Key features:
- **K-splitting** (`k_splits`): Splits K dimension across gridDim.z for more grid parallelism at small sizes. Split 0 writes directly; subsequent splits use atomicAdd.
- **Size-adaptive thread_m**: Larger tiles (TM=26-28, giving BM=208-224) for large matrices to increase compute density per thread. Smaller tiles (TM=8, BM=64) with more K-splits for small matrices.
- **L2 promotion**: CU_TENSOR_MAP_L2_PROMOTION_L2_256B for better cache utilization.
- **OOB fill**: CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA for correct partial tiles with non-aligned sizes.

| Size | TM | BK | K-splits | Eff vs cuBLAS | TFLOPS |
|------|-----|-----|----------|--------------|--------|
| 256  | 8  | 32 | 4 | ~95% | 4.1 |
| 512  | 8  | 32 | 4 | ~93% | 19.3 |
| **1024** | 8  | 32 | 1 | **101%** | 49.0 |
| **2048** | 26 | 32 | 1 | **106%** | 72.8 |
| **4096** | 20 | 32 | 1 | **101%** | 67.4 |
| 8192 | 28 | 32 | 1 | 96% | 60.2 |
| 16384 | 28 | 32 | 1 | 89% | 56.9 |

Consistently beats cuBLAS at 1024, 2048, and 4096. Key optimizations: M/N/K as compile-time `#define` constants (not kernel params) lets nvcc optimize loop bounds and eliminate dead branches. Combined with compile-time k_splits elimination, the kernel has zero runtime overhead for the common case.

ncu profiling shows identical occupancy (16.67%) and compute throughput (~78%) to cuBLAS at large sizes — the remaining 4-11% gap is SASS-level instruction scheduling from C code vs cuBLAS's hand-optimized SASS.

#### Per-GPU Tuning Profiles

`deplodock/compiler/cuda/tuning.py` holds the empirically-tuned `tma_db` strategy maps and dispatches by GPU name (from `nvidia-smi --query-gpu=name`). Both the RTX 5090 and the RTX PRO 6000 Blackwell report `sm_120` and identical per-SM smem, so compute capability cannot distinguish them — the meaningful differences are SM count, clocks, and how the SASS scheduler reacts to large thread tiles. The largest divergence measured is at 4096 (5090 prefers TM=20, Pro 6000 prefers TM=24, ~7% gap if you pick the wrong one).

Unknown GPUs fall back to the 5090 profile. To add a new GPU, sweep `--strategy tma_db --thread-m N` per size and append a new entry to `_PROFILES` in `tuning.py`.

### Non-Aligned Size Support

All float4 strategies support non-power-of-2 and non-rectangular matrices:
- **Float4 alignment guard**: `N % 4 == 0` check gates float4 loads; scalar fallback when N is not 4-aligned
- **Per-element write bounds**: Each output element has individual `row < M && col < N` check
- **Scalar inner loop fallback**: Edge columns use per-element B loads instead of float4

### Runner and Benchmarking

The runner generates complete `.cu` programs, compiles with nvcc (`-O3 --use_fast_math`), and supports two modes:
- `run_kernel()`: correctness testing with embedded data
- `run_benchmark()`: performance comparison with curand init and cuBLAS comparison

The `benchmark.py` module provides:
- `run_benchmark_suite()`: single strategy across all sizes
- `run_adaptive_benchmark_suite()`: per-size strategy selection via threshold map
- `MATRIX_SIZES`: standard power-of-2 sizes 256-16384
- `MATRIX_SIZES_EXTENDED`: includes non-rectangular, non-power-of-2, and odd sizes

## Structured Trace & Pipeline

`pipeline.compile_and_run()` orchestrates the full cycle and produces a `CompilerTrace` (JSON-serializable):

```json
{
  "input_graph": { "nodes": {...}, "inputs": [...], "outputs": [...] },
  "passes": [{ "pass": "fusion", "rules_applied": [...], "graph_before": {...}, "graph_after": {...} }],
  "cuda_kernel": "__global__ void fused_matmul(...) { ... }",
  "execution": { "output": [...], "correct": true, "max_error": 0.0, "kernel_time_ms": 0.042, "dimensions": {"M": 4, "N": 2, "K": 3} }
}
```

Designed for an AI-in-the-loop optimization cycle: AI reads trace → modifies rules/IR → runs pipeline → evaluates performance → iterates.

The `benchmark.py` module provides `BenchmarkSuite` for multi-size benchmarking with cuBLAS comparison. Traces are saved to `results/matmul_overnight/` with timestamps for later inspection.
