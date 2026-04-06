# Compiler Architecture

## Overview

Minimal tensor IR and regex-style graph transformation engine. Represents tensor computations with a small canonical op set, then applies ordered rewrite rules to progressively transform naive expressions into optimized forms.

## Module Layout

```
compiler/
├── ops.py        # Op base class + op types (ElementwiseOp, ReduceOp, etc.)
├── ir.py         # Tensor, Node, Graph — the compute graph data structure
├── pattern.py    # Pattern AST + text parser for matching rules
├── matcher.py    # Graph pattern matching engine
├── rewriter.py   # Pass/Rule/Rewriter — rule loading and application
├── trace.py      # CompilerTrace — structured JSON trace for AI-in-the-loop
├── pipeline.py   # compile_and_run() — end-to-end pipeline with tracing
├── benchmark.py  # Multi-size benchmark harness with cuBLAS comparison
├── rules/        # Ordered rule files organized by pass
│   ├── fusion/   # Fuse ops to avoid intermediates
│   └── tiling/   # Tile loops for memory hierarchy
└── cuda/         # CUDA backend
    ├── ir.py     # Imperative AST (Expr, Stmt, KernelDef, VectorLoad, etc.)
    ├── codegen.py # CUDA IR → CUDA C source string
    ├── lower.py  # Graph IR → CUDA IR (MatmulConfig + strategy dispatch)
    └── runner.py # Compile with nvcc + execute + benchmark + parse output
```

## IR Design

- **Ops** are plain dataclasses inheriting from `Op`. No enum — the class name is the type. New ops (e.g. `FusedReduceElementwiseOp`) are added by subclassing.
- **Nodes** reference inputs by string ID. Fan-out is natural — multiple nodes can list the same input ID.
- **Graph** is an ordered dict of nodes with `add_node`, `remove_node`, `replace_node`, `consumers`, and `topological_order`.

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
| `naive` | 1 thread per output element, direct global loads | Baseline |
| `smem_tiled` | Shared memory tiles, cooperative loading | Educational |
| `register_blocked` | Register blocking with outer product | Large tiles |
| `coarsened_f4` | float4 vectorized B loads, 4 cols/thread | Medium sizes |
| `coarsened_2r4c` | 2 rows x 4 cols per thread, float4 B | Medium-large |
| **`hybrid_smem_f4`** | **Shared mem A + float4 B + 2-row coarsening** | **512+** |
| `hybrid_1r_f4` | Shared mem A + float4 B + 1-row (more parallelism) | 512 |
| `flat_scalar` | No smem, 1 thread/element, 1D grid | 256 and below |
| `flat_f4` | No smem, float4 B, 1D grid | Small with N%4==0 |

### Size-Adaptive Strategy Selection

`hybrid_smem_f4` supports configurable rows per thread via `thread_m` and `__launch_bounds__`.

#### sm_120 (Blackwell, CUDA 13.0, cuBLAS 13.1)

cuBLAS on sm_120 uses CUTLASS `simt_sgemm_256x128_8x4` (pure FP32, not tensor cores). ncu profiling shows 65% FMA utilization vs our 54% — the gap is SASS-level instruction scheduling.

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
| 512  | 8  | 32 | 4 | ~94% | 18.8 |
| **1024** | 8  | 32 | 1 | **101%** | 48.8 |
| **2048** | 26 | 32 | 1 | **105%** | 71.8 |
| **4096** | 20 | 32 | 1 | **100.3%** | 66.5 |
| 8192 | 28 | 32 | 1 | 97% | 60.0 |
| 16384 | 28 | 32 | 1 | 89% | 56.4 |

Consistently beats cuBLAS at 1024, 2048, and 4096. ncu profiling shows identical occupancy (16.67%) and compute throughput (~78%) to cuBLAS at large sizes — the remaining 3-11% gap is SASS-level instruction scheduling from C code vs cuBLAS's hand-optimized SASS (confirmed: zero bank conflicts in cuBLAS vs 6.4M in ours, but the stall impact is <0.2% of total time).

#### FP32 Hybrid (older approach)

Beats cuBLAS at 256 and 512. At 1024+ limited to ~91% due to C-level code generation overhead vs cuBLAS's SASS-optimized LDGSTS loading.

The `run_adaptive_benchmark_suite()` function uses a threshold-based strategy map to pick the best config per size.

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
