# Compiler Architecture

## Layered Design

The compiler has four layers. Each layer depends only on the layers above it. **All new code MUST follow this layering — do NOT import from a lower layer into a higher one.**

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Frontend (backend-agnostic)                           │
│                                                                 │
│  torch_trace.py ─→ Graph IR (ops.py, ir.py)                     │
│  PyTorch module        Tensor, Node, Graph                      │
│                                                                 │
│  RULE: No GPU, no CUDA, no backend imports.                     │
│        Only standard library + torch (optional).                │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: Optimization (backend-agnostic)                       │
│                                                                 │
│  rewriter.py ─→ optimized Graph                                 │
│  pattern.py, matcher.py                                         │
│  rules/fusion/*.py                                              │
│                                                                 │
│  RULE: Operates on Graph IR only. No backend imports.           │
│        Must not reference CUDA, grid, block, or kernel source.  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: Execution Plan (backend-agnostic)                     │
│                                                                 │
│  plan.py:           BufferSpec, OpKernel, ExecutionPlan         │
│  backend.py:        Backend ABC (compile, run, benchmark)       │
│                                                                 │
│  RULE: Describes WHAT to compute, not HOW.                      │
│        No kernel source, no grid/block, no GPU API calls.       │
│        OpKernel.op is a string tag — the backend resolves it.   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: Backend (CUDA, ROCm, ...)                             │
│                                                                 │
│  cuda/backend.py:   CudaBackend (implements Backend ABC)        │
│  cuda/program.py:   Buffer, Launch, Program, compile, run       │
│  cuda/generators/:  matmul.py (SGEMM), fused.py (pointwise/reduce)│
│  cuda/codegen.py:   KernelDef → CUDA C source                   │
│  cuda/ir.py:        CUDA imperative AST (Expr, Stmt, KernelDef) │
│  cuda/runner.py:    Legacy single-kernel compile + run          │
│                                                                 │
│  RULE: All GPU-specific code lives here.                        │
│        A new backend (e.g., rocm/) implements the same          │
│        Backend ABC and maps OpKernel tags to .hip templates.    │
└─────────────────────────────────────────────────────────────────┘
```

## Canonical Data Flow

**Every execution MUST follow this flow.** Do not create shortcuts that bypass layers.

```
                     Layer 1                    Layer 2
PyTorch module ──→ Graph IR ──→ Decomposition ──→ Fusion ──→ auto_fuse ──→ optimized Graph
                                  (sdpa, silu,     (matmul)    (general fusion:
                                   pow → prims)                 region discovery +
                                                                kernel generation)
                                                                        │
                     Layer 3                                            │
                     plan_graph(graph) ──→ ExecutionPlan ◄──────────────┘
                                                │
                     Layer 4                    │
                     CudaBackend.compile(plan) ──→ Program
                                                │
                     program.py                 │
                     generate_source() ──→ .cu ──→ nvcc ──→ GPU
```

### Concrete example: graph-driven (primary path)

```python
# Layer 1-2: trace + optimize
from deplodock.compiler.torch_trace import trace_module
graph = trace_module(block, (x,), kwargs={"position_embeddings": (cos, sin)})
graph = Rewriter.from_directory(rules_dir).apply(graph)  # decompose + fuse

# Layer 3: plan from graph (backend-agnostic)
from deplodock.compiler.plan import plan_graph
plan = plan_graph(graph)

# Layer 4: backend (CUDA-specific)
from deplodock.compiler.backend.cuda.backend import CudaBackend
backend = CudaBackend()
program = backend.compile(plan)
result = backend.benchmark(program)
```

### Alternative: config-driven (for testing without a model)

```python
plan = plan_block(BlockConfig(hidden_dim=2048, num_heads=32, ...))
# Then same backend.compile(plan) flow.
```

### Concrete example: single matmul

```python
# Layer 1-2: graph + optimization + hints
graph = build_matmul_graph()
graph.hints.set("cuda.matmul.strategy", "tma_db")
graph.hints.set("cuda.matmul.block_k", 32)
graph = Rewriter.from_directory(rules_dir).apply(graph)

# Layer 3-4: plan + backend (hints flow automatically)
plan = plan_graph(graph)
program = CudaBackend().compile(plan)
result = backend.run(program)
```

## What NOT to do

- **Do NOT import from `cuda/` in Layer 1-3 modules.** If you need GPU-specific behavior, add it to `cuda/backend.py`.
- **Do NOT put kernel source strings in Layer 3.** `OpKernel.op` is a tag; the backend resolves it to actual code.
- **Do NOT hardcode grid/block/smem in planners.** That's the backend's job.
- **Do NOT add kernel source as Python f-strings.** Use `generators/fused.py` for auto-generated kernels or `generators/matmul.py` for SGEMM strategies.

## Module Layout

```
compiler/
├── ops.py            # [L1] Op base class + all op types
├── ir.py             # [L1] Tensor, Node, Graph (with Hints on Node + Graph)
├── hints.py          # [L1] Hints metadata bag + resolve_hints()
├── torch_trace.py    # [L1] PyTorch → Graph IR (optional torch dep)
├── pattern.py        # [L2] Pattern AST + text parser
├── matcher.py        # [L2] Graph pattern matching engine
├── rewriter.py       # [L2] Pass/Rule/Rewriter
├── rules/            # [L2] Ordered rule files by pass
│   ├── decomposition/ #     Decompose high-level ops → primitives
│   └── fusion/       #      (empty — auto_fuse handles all fusion)
├── fusion.py         # [L2] auto_fuse: automatic fusion region discovery
├── plan.py           # [L3] BufferSpec, OpKernel, ExecutionPlan, plan_graph
├── dump.py           # [--] CompilerDump: debug artifact collector (cross-layer)
├── pipeline.py       # [L2] compile_graph (graph optimization only)
├── backend/          # [L3+L4] Backend abstraction + implementations
│   ├── __init__.py   #      Re-exports from base.py
│   ├── base.py       # [L3] Backend ABC, ProgramResult, BenchmarkResult
│   └── cuda/         # [L4] CUDA backend
│       ├── backend.py    #  CudaBackend implements Backend ABC
│       ├── program.py    #  Buffer, Launch, Program — compile + run
│       ├── ir.py         #  CUDA imperative AST (KernelDef, Expr, Stmt)
│       ├── codegen.py    #  KernelDef → CUDA C source
│       ├── generators/   #  Kernel generators
│       │   ├── matmul.py #    Hand-optimised SGEMM (naive, TMA, TF32)
│       │   └── fused.py  #    Auto pointwise + reduction from FusedRegionOp
│       ├── runner.py     #  Legacy single-kernel compile + run + benchmark
│       └── tuning.py     #  Per-GPU empirical tuning profiles
```

`*` — `pipeline.py` has CUDA imports (legacy). New code should use the Backend ABC instead.

Benchmark orchestration (multi-size sweeps, result collection, summary tables) lives in `scripts/bench_matmul.py` and `scripts/bench_block.py`, not in the compiler package. The compiler only provides execution primitives.

## Execution Plan (Layer 3)

An `ExecutionPlan` describes what to compute without how:

```python
@dataclass
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"

@dataclass
class OpKernel:
    op: str                # "rmsnorm", "triple_matmul", "attention_qk", ...
    inputs: list[str]      # buffer names
    outputs: list[str]     # buffer names
    params: dict           # op-specific dimensions and config

@dataclass
class ExecutionPlan:
    name: str
    buffers: list[BufferSpec]
    ops: list[OpKernel]
```

`OpKernel.op` is a string tag. The CUDA backend maps it to a `.cu` template. A ROCm backend would map the same tag to a `.hip` template. The planner doesn't know or care which backend runs it.

## Adding a New Backend

1. Create `rocm/` directory alongside `cuda/`
2. Implement `RocmBackend(Backend)` in `rocm/backend.py`
3. Map the same `OpKernel.op` tags to `.hip` templates

## Graph Transformation (Layer 2)

Three stages:

### 1. Decomposition (rules/decomposition/)

Decomposes high-level ops from `torch.export` into primitives:

| Rule | Decomposes | Into |
|------|-----------|------|
| `001_decompose_sdpa` | `sdpa(Q, K, V)` | QK^T → scale → softmax → @V |
| `002_decompose_silu` | `silu(x)` | `x * recip(1 + exp(-x))` |
| `003_decompose_pow` | `pow(x, 2)` | `mul(x, x)` |

### 2. Auto-fusion (fusion.py)

`auto_fuse(graph)` discovers fusion regions from intermediate tensor sizes:
- Scores each single-consumer edge by `product(intermediate_shape)`
- Greedy merge highest-score-first
- Structural ops (reshape, transpose) always merge
- Produces `FusedRegionOp` nodes

`cuda/kernel_gen.py` generates CUDA kernels from each `FusedRegionOp` by walking its primitive ops directly — no hand-written templates needed for pointwise or reduction patterns. The CUDA backend auto-generates kernels during `compile()` for any `FusedRegionOp` that doesn't have a pre-filled `kernel_source`.

## CUDA Backend Details

### SGEMM Strategies (cuda/lower.py)

| Strategy          | Description                          | Best for               |
|-------------------|--------------------------------------|------------------------|
| `naive`           | 1 thread per output element          | Test baseline          |
| **`tma_db`**      | **TMA double-buffer, size-adaptive** | **Production default** |
| `tma_db_tf32`     | TF32 via tensor cores (wmma)         | TF32 precision ok      |
| `tma_db_fma_tf32` | Concurrent FMA + TF32 hybrid         | Mixed precision ok     |

### TMA Performance (RTX 5090, sm_120)

| Size     | TM   | BK   | K-splits  | Eff vs cuBLAS | TFLOPS  |
|----------|------|------|-----------|---------------|---------|
| **1024** | 8    | 32   | 1         | **101%**      | 49.0    |
| **2048** | 26   | 32   | 1         | **106%**      | 72.8    |
| **4096** | 20   | 32   | 1         | **101%**      | 67.4    |
| 8192     | 28   | 32   | 1         | 96%           | 60.2    |

## Hints (hints.py)

Hints are advisory metadata attached to `Node` or `Graph` that influence compiler decisions without changing computation semantics. Backends may ignore unknown hints.

- **`Node.hints`**: per-node hints (e.g. this matmul uses strategy X)
- **`Graph.hints`**: graph-wide hints (e.g. all matmuls use strategy Y)
- **`resolve_hints(graph, node_id)`**: merges graph + node hints (node wins on conflict)

Keys use dotted namespaces. The `cuda.matmul.*` namespace controls matmul lowering:

```python
graph.hints.set("cuda.matmul.strategy", "tma_db")
graph.hints.set("cuda.matmul.block_k", 32)
graph.nodes["n5"].hints.set("cuda.matmul.threads_y", 8)  # per-node override
```

Flow through the pipeline:
1. **Rewriter/Fusion**: read/write hints on graph nodes directly
2. **plan_graph()**: resolves hints per node → stores in `OpKernel.params["_hints"]`
3. **CudaBackend**: reads `params["_hints"]`, sets them on the lowering graph, calls `lower_matmul(graph)` which reads hints internally

Hints survive serialization (`to_dict`/`from_dict`) and deep copy. Old graphs without hints deserialize correctly (empty defaults).

## Debug Dump Infrastructure

`dump.py` provides `CompilerDump` — an opt-in artifact collector that writes intermediate compilation results to disk. Activated via:

- **Env var**: `DEPLODOCK_DUMP_DIR=/tmp/dump`
- **CLI arg**: `--dump-dir /tmp/dump` on `compile` and `run` commands
- **Test fixture**: `dump_dir` writes to `_test_data/<test_name>/`

The dump directory is cleared on creation. Files are numbered for natural ordering:

```
00_input_graph.json        # Graph before optimization
01_pass_<name>_*.json      # Per-pass before/after graphs + rules (01-19)
20_fused_graph.json        # Graph after auto_fuse
30_execution_plan.json     # Backend-agnostic plan
40_program_summary.json    # Program metadata (buffers, launches)
40_kernel_NN_<name>.cu     # Individual kernel sources
50_full_program.cu         # Complete generated .cu file
60_result.json             # Execution outputs
60_benchmark.json          # Timing results
```

`CompilerDump` is cross-layer by design — it serializes objects from all layers but does not import GPU-specific code at module level (uses `TYPE_CHECKING` guards). Callers pass it through the pipeline and call `dump.dump_*()` at each stage.
