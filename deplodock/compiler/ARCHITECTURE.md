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
│  LAYER 3: Execution Plan + Shared Infrastructure                │
│                                                                 │
│  plan.py:            BufferSpec, OpKernel, ExecutionPlan        │
│  backend/base.py:    Backend ABC (compile, run, benchmark)      │
│  backend/program.py: Buffer, Launch, Program                    │
│  backend/kernel_ir.py:      Kernel AST (Expr, Stmt, KernelDef)        │
│  backend/loop_ir.py: LoopIR (LoopProgram, LoopOp, LoopExpr)    │
│  backend/codegen.py: KernelDef → C source                       │
│                                                                 │
│  RULE: Describes WHAT to compute, not HOW.                      │
│        program.py, ir.py, loop_ir.py, codegen.py are backend-   │
│        agnostic — shared by all backends. No CUDA/HIP-specific  │
│        API calls. OpKernel.op is a string tag — the backend     │
│        resolves it.                                              │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: Backend (CUDA, ROCm, ...)                             │
│                                                                 │
│  cuda/backend.py:   CudaBackend (implements Backend ABC)        │
│  cuda/program.py:   CudaLaunch, TmaDescriptorSpec, source gen   │
│  cuda/generators/:  analysis, loop_lower, loop_codegen, tiled   │
│  cuda/runner.py:    Single-kernel compile + run                 │
│  cuda/tuning.py:    Per-GPU empirical tuning profiles           │
│                                                                 │
│  RULE: All GPU-specific code lives here.                        │
│        CudaLaunch extends Launch with TMA descriptor metadata.  │
│        A new backend (e.g., rocm/) implements the same          │
│        Backend ABC, reuses backend/{program,ir,codegen}.py,     │
│        and provides its own generators + tuning.                │
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
- **Do NOT add kernel source as Python f-strings.** Use the LoopIR pipeline (`loop_lower.py` → `loop_codegen.py`) for new kernel patterns — all patterns go through LoopIR.

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
│   ├── program.py    # [L3] Buffer, Launch, Program — shared across backends
│   ├── ir.py         # [L3] Kernel AST (KernelDef, Expr, Stmt) — shared
│   ├── loop_ir.py    # [L3] LoopIR (LoopProgram, LoopOp, LoopExpr) — shared
│   ├── codegen.py    # [L3] KernelDef → C source — shared
│   └── cuda/         # [L4] CUDA backend
│       ├── backend.py    #  CudaBackend implements Backend ABC
│       ├── program.py    #  CudaLaunch, TmaDescriptorSpec, source gen, nvcc
│       ├── generators/   #  Kernel generators
│       │   ├── analysis.py    #  TileAnalysis: classify FusedRegionOp patterns
│       │   ├── loop_lower.py  #  TileAnalysis → LoopIR (CUDA lowering)
│       │   ├── loop_codegen.py #  LoopIR → KernelDef (CUDA codegen)
│       │   └── tiled.py       #  Public API: generate_kernel(), lower_tiled()
│       ├── runner.py     #  Single-kernel compile + run + benchmark
│       └── tuning.py     #  Per-GPU empirical tuning profiles
```

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
3. Reuse shared Layer 3 modules: `backend/program.py` (Buffer, Launch, Program), `backend/kernel_ir.py` (kernel AST), `backend/codegen.py` (C printer)
4. Optionally subclass `Launch` for backend-specific metadata (like CUDA's `CudaLaunch` adds `tma_descriptors`)
5. Write ROCm-specific generators (tile sizes, warp size 64, no TMA) and tuning profiles
6. Map the same `OpKernel.op` tags to `.hip` templates

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

Fusion constraints enforced by `_can_merge()`:
- **Convexity**: merged region must be a convex subgraph (no external nodes between)
- **Contraction + epilogue**: contraction core (mul+sum) can fuse with pointwise epilogue ops (bias add, activation, residual add from graph-level inputs)
- **Single reduce**: one reduction pass per kernel (softmax needs two regions: max+sub+exp and sum+div)
- **Dimensionality**: >2D tensors are allowed if they don't expand rank beyond inputs (enables 4D softmax, rotary embedding fusion). Broadcast rank expansion (matmul-style 2D→3D) is only allowed for standard contractions.
- **Shape compatibility**: all full-rank external inputs must have the same total size (scalars, vectors, and per-row tensors are exempt)
- **Single output**: one external output per region

The CUDA backend auto-generates kernels during `compile()` for any `FusedRegionOp` via `generators/tiled.py`. Reshape/transpose ops become buffer aliases (zero-cost pointer assignment, no kernel launch).

## CUDA Backend Details

See [`backend/cuda/ARCHITECTURE.md`](backend/cuda/ARCHITECTURE.md) for full details on:
- SGEMM strategies (TMA, smem, naive) and when each is selected
- Per-GPU tuning profiles and M-aware optimizations
- Performance benchmarks (square, rectangular, end-to-end transformer block)
- Kernel structure diagrams for TMA and smem strategies
- Reproducible benchmark experiments

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
3. **CudaBackend**: reads `params["_hints"]` in `_compile_matmul()`, merges with tuning profile defaults, passes to `lower_tiled()` as the `hints` dict

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
35_loop_ir_<name>.txt      # LoopIR pretty-print (loop nest structure)
35_loop_ir_<name>.json     # LoopIR serialized
40_program_summary.json    # Program metadata (buffers, launches)
40_kernel_NN_<name>.cu     # Individual kernel sources
50_full_program.cu         # Complete generated .cu file
60_result.json             # Execution outputs
60_benchmark.json          # Timing results
```

`CompilerDump` is cross-layer by design — it serializes objects from all layers but does not import GPU-specific code at module level (uses `TYPE_CHECKING` guards). Callers pass it through the pipeline and call `dump.dump_*()` at each stage.
