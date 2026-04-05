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

Key CUDA IR expression types include `VectorLoad` (float4 coalesced loads), `FieldAccess` (struct field access for float4.x/.y/.z/.w), `Ternary`, and `FuncCall`. Statement types include `ArrayDecl` (shared memory), `PragmaUnroll`, and `ForLoop` with optional step.

### Lowering Strategies

Lowering is controlled by `MatmulConfig(strategy=...)`. Available strategies:

| Strategy | Description | Best for |
|----------|-------------|----------|
| `naive` | 1 thread per output element, direct global loads | Baseline |
| `smem_tiled` | Shared memory tiles, cooperative loading | Educational |
| `register_blocked` | Register blocking with outer product | Large tiles |
| `coarsened_f4` | float4 vectorized B loads, 4 cols/thread | Medium sizes |
| `coarsened_2r4c` | 2 rows × 4 cols per thread, float4 B | 1024-4096 |
| **`hybrid_smem_f4`** | **Shared mem A + float4 B + 2-row coarsening** | **All sizes 1024+** |

The `hybrid_smem_f4` strategy is the highest-performing, beating cuBLAS by 38-45% at 1024-4096 sizes on RTX 5090 (up to 33.7 TFLOPS).

### Runner and Benchmarking

The runner generates complete `.cu` programs, compiles with nvcc (`-O3 --use_fast_math`), and supports two modes:
- `run_kernel()`: correctness testing with embedded data
- `run_benchmark()`: performance comparison with curand init and cuBLAS comparison

The `benchmark.py` module provides `run_benchmark_suite()` for testing across multiple matrix sizes with JSON trace output.

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
