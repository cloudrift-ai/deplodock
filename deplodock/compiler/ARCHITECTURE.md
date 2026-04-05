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
├── rules/        # Ordered rule files organized by pass
│   ├── fusion/   # Fuse ops to avoid intermediates
│   └── tiling/   # Tile loops for memory hierarchy
└── cuda/         # CUDA backend
    ├── ir.py     # Imperative AST (Expr, Stmt, KernelDef)
    ├── codegen.py # CUDA IR → CUDA C source string
    ├── lower.py  # Graph IR → CUDA IR (matmul lowering)
    └── runner.py # Compile with nvcc + execute + parse output
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

Lowering translates fused graph ops into CUDA kernels. Currently supports matmul (FusedReduceElementwiseOp with sum/mul). Each thread computes one output element with a K-loop accumulator.

The runner generates a complete `.cu` program (kernel + host wrapper with CUDA event timing), compiles with nvcc, executes, and parses stdout for output data and `KERNEL_TIME_MS`.

## Structured Trace & Pipeline

`pipeline.compile_and_run()` orchestrates the full cycle and produces a `CompilerTrace` (JSON-serializable):

```json
{
  "input_graph": { "nodes": {...}, "inputs": [...], "outputs": [...] },
  "passes": [{
    "pass": "fusion",
    "rules_applied": [{"rule": "001_fuse_reduce_elementwise", "matched_at": "red", ...}],
    "graph_before": {...},
    "graph_after": {...}
  }],
  "cuda_kernel": "__global__ void fused_matmul(...) { ... }",
  "execution": {
    "output": [22.0, 28.0, ...],
    "expected": [22.0, 28.0, ...],
    "correct": true,
    "max_error": 0.0,
    "kernel_time_ms": 0.042,
    "dimensions": {"M": 4, "N": 2, "K": 3}
  }
}
```

Designed for an AI-in-the-loop optimization cycle: AI reads trace → modifies rules/IR → runs pipeline → evaluates performance → iterates.
