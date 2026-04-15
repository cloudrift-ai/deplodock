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
│    LoopProgram is purely structural (no backend metadata).     │
│    Load/Store use multi-dim indices (list[LoopExpr]).           │
│    Schedule flows alongside LoopProgram through codegen.       │
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
PyTorch module ──→ Graph IR ──→ Decomposition ──→ Optimization ──→ Fusion ──→ optimized Graph
   (faithful                    (sdpa, silu, pow,  (canonicalize      (greedy region
    1:1 FX                       linear, matmul,    primitive IR)      merger + structural
    capture)                     unsqueeze, mean)                      classification into
                                                                       KernelOps with
                                                                       prologue/core/epilogue)
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
├── ops.py            # [L1] Op base class + all op types (each implements infer_output_shape)
├── ir.py             # [L1] Tensor, Node, Graph (with Hints on Node + Graph)
├── hints.py          # [L1] Hints metadata bag + resolve_hints()
├── shape_utils.py    # [L1] Shared shape utilities: broadcast_shapes, propagate_shapes
├── torch_trace.py    # [L1] PyTorch → Graph IR (optional torch dep)
├── pattern.py        # [L2] Pattern AST + text parser
├── matcher.py        # [L2] Graph pattern matching engine
├── rewriter.py       # [L2] Pass/Rule/Rewriter (DEFAULT_PASS_ORDER drives ordering)
├── rules/            # [L2] Pass directories — each is loaded explicitly via DEFAULT_PASS_ORDER
│   ├── decomposition/ #     Decompose high-level ops → primitives (runs first)
│   ├── optimization/  #     Canonicalize primitive IR (runs after decomposition)
│   └── fusion/       #      Greedy fusion + structural classification (KernelOp core)
├── fusion.py         # [L2] auto_fuse: thin shim that loads rules/fusion/* (legacy callers)
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

Four stages, run in order: **decomposition → optimization → fusion → auto-fuse**.

Pass ordering is explicit, controlled by `DEFAULT_PASS_ORDER` in `rewriter.py`. Within each pass, rule files are loaded alphabetically by filename (`001_*.py`, `002_*.py`, ...). Adding a new pass requires both creating its rule directory and adding its name to `DEFAULT_PASS_ORDER`.

Each `Op` subclass implements `infer_output_shape(input_shapes)`, used by graph rewrites to re-derive shapes after upstream changes (`shape_utils.propagate_shapes`).

### 1. Decomposition (rules/decomposition/)

Decomposes high-level ops from `torch.export` into primitives:

| Rule | Decomposes | Into |
|------|-----------|------|
| `001_decompose_sdpa` | `sdpa(Q, K, V)` | QK^T → scale → softmax → @V |
| `002_decompose_silu` | `silu(x)` | `x * recip(1 + exp(-x))` |
| `003_decompose_pow` | `pow(x, 2)` | `mul(x, x)` |
| `004_decompose_linear` | `linear(x, w[, b])` | `transpose(w) → mul → sum [→ add(b)]` |
| `005_decompose_matmul` | `matmul(a, b[, bias])` | `mul → sum [→ add(bias)]` |
| `007_decompose_mean` | `mean(x, axis)` | `sum(x, axis) → div(count)` |
| `010_unsqueeze_to_indexmap` | `unsqueeze(x, dim)` | `IndexMapOp` with one source, identity coord_map skipping the inserted axis |
| `011_transpose_to_indexmap` | `transpose(x, axes=(a, b))` | `IndexMapOp` with one source, coord_map swapping placeholders for axes a and b |
| `013_slice_to_indexmap` | `slice(x, dim, start)` | `IndexMapOp` with one source, coord_map adding `start` offset on the slice axis |
| `014_cat_to_indexmap` | `cat([a, b], dim)` | `IndexMapOp` with two sources; source 0's `select` picks output positions where the dim coord is below the split |

Guideline: the tracer does a **faithful 1:1 capture** of FX ops. It never
decomposes or normalizes — even compound ops like `Linear`, `Sdpa`, `Mean`,
`Unsqueeze` become their own Torch IR nodes (`LinearOp`, `SdpaOp`, `MeanOp`,
`UnsqueezeOp`). All lowering to primitives happens here in rewrite passes,
so the tracer stays small and the decomposition lives in one place.

### 2. Optimization (rules/optimization/)

Canonicalizes the primitive IR after decomposition. Operating on
primitives instead of high-level ops means one set of rules works
regardless of how the original op was expressed (e.g. `SdpaOp` vs
hand-rolled QK^T+softmax+V).

| Rule | Pattern | Action |
|------|---------|--------|
| `001_merge_index_maps` | `IndexMap($inner)` where `$inner` is itself an `IndexMapOp` | Compose the two IndexMaps via coord-map substitution (substitute outer's placeholders into the inner's coord_map). The result reads the inner's input directly with one combined coord_map. Identity-after-composition cases become free buffer aliases via the backend's noop alias detection. |

Note: `IndexMapOp` (in `ops.py`) subsumes Slice/Cat/Transpose/Unsqueeze. Its
`sources` list is one or more `IndexSource(input_idx, coord_map, select)`,
where `coord_map[i]` is a `LoopExpr` over placeholder vars
`Var("out_coord_0")`, `Var("out_coord_1")`, ... that produces the i-th
input index. `select` (optional, used by cat) is a boolean LoopExpr picking
which output positions read this source. Multi-source IndexMaps lower to a
Ternary load chain in the standalone `_compile_indexmap` kernel.

Rule conventions:
- Return the same `Graph` object for ineligible matches — `Pass.apply`
  treats identity-preserving returns as no-ops, so the fixed-point loop
  doesn't spin on patterns that match more nodes than they can act on.
- Eligible rewrites must `g = graph.copy()` and return the new graph.

### 3. Fusion (rules/fusion/ + fusion.py)

Fusion runs as a `Rewriter` pass (last in `DEFAULT_PASS_ORDER`). Rules in `rules/fusion/*.py` execute in filename order:

Rule-based assembly is the only path. The legacy greedy `auto_fuse` and structuring rules (`000_greedy_fusion`, `001_structure_contraction`, `002_structure_reduce`) are deleted; `auto_fuse` survives only as a thin shim that loads `rules/fusion/*` so older test imports keep working.

- `015_seed_softmax` — softmax DAG (`div(exp(sub(x, max(x))), sum(exp(sub(x, max(x)))))`) → KernelOp with two ReduceStages plus a div in prologue (2D coverage; 4D still falls to multiple kernels until the matcher gains DAG-backref unification).
- `020_seed_contraction` — `Reduce{sum}(Elementwise{mul}(A, B))` → KernelOp with ContractionCore (mul + sum stay in prologue and `core` annotates them).
- `021_seed_reduce` — single Reduce → KernelOp with one ReduceStage (the reduce node is also kept in prologue).
- `040_seed_pointwise` — wraps any standalone Elementwise / IndexMap (skipping identity IndexMaps that are buffer aliases) as a singleton pointwise KernelOp.
- `050_absorb_prologue` — pulls an upstream Elementwise / IndexMap producer (fan-out=1) into a KernelOp's prologue.
- `055_merge_kernels` — merges two adjacent KernelOps when shape and reduce-axis compat allow. Handles pointwise+pointwise, pointwise+reduce, reduce+pointwise, reduce+reduce (row-compatible), contraction+pointwise, and contraction+reduce (only when row-compatible — enables matmul→softmax fusion). Rejects pointwise+contraction (deferred to `080`) and contraction+contraction. Inputs that point at the absorbed kernel's outer-graph id are rewired to its last-internal node id.
- `060_absorb_epilogue` — appends a downstream Elementwise into the parent KernelOp's prologue (rewiring the absorbed node's input that named the kernel itself).
- `_070_absorb_indexmap_into_port` — dormant; activates once the backend's load path reads `Port.indexmap`.
- `080_absorb_a_chain` — placeholder for ContractionCore a/b-chain absorption.

**Flat-prologue convention.** Every emitting rule keeps ALL of a kernel's body nodes (including reduce / contraction nodes) in `prologue` in topo order. `core` is an annotation that points at specific nodes (ReduceStage.reduce, ContractionCore.mul/.reduce) already in prologue. `epilogue` is unused. Backend codegen reads `KernelOp.region_ops`, which dedups across prologue + core + epilogue, so reader-side compat stays unchanged. The convention also keeps per-element values (e.g. `exp` flowing into both `sum(exp)` and `div(exp, sum)`) accessible at every subsequent op.

**Cross-rule plumbing.** `rules/fusion/_assembly_helpers.py` holds the shared utilities every rule should reuse instead of copying: shape-compat (`merged_external_inputs_compat`, `broadcast_compat`), reduce-axis compat (`reduces_compatible`, `is_row_reduce`), KernelOp introspection (`kernel_kind`, `kernel_reduces_with_input_shapes`, `kernel_last_node_id`, `flatten_kernel_nodes`, `kernel_has_contraction`), graph-level checks (`fan_out_of`, `is_convex_merge`), node-rewiring (`copy_node`, `rewire_node_input`), and the bookkeeping helper `rewrite_port_references` that fixes up `Port.buffer_id`, `external_shapes`, and internal-node `inputs` lists in every other KernelOp whenever an outer-graph id changes (must be called after every `replace_node`).

Fusible op types are `ElementwiseOp`, `ReduceOp`, and `IndexMapOp`. `ReshapeOp` and `TransposeOp` are NOT directly fusible (they decompose to `IndexMapOp` upstream, or remain as buffer aliases). Identity IndexMaps (`out = X` with placeholder coord_map matching the input shape) are skipped by 040 so they stay as alias nodes for the backend's noop-alias detection.

Fusion constraints enforced across rules:
- **Convexity**: every merge preserves convex-subgraph membership in the outer graph (`is_convex_merge`).
- **Contraction isolation**: a contraction kernel can only co-exist with pointwise epilogue or row-compatible reduces (matmul→softmax). Two contractions never merge.
- **Single output**: every kernel exposes exactly one external output port. Multi-output fusion is not implemented.
- **Reduce-axis compat**: row reduces inside the same kernel must share rank and trailing dim (`reduces_compatible`); contraction reduces are excluded from this check.
- **Shape compat**: the external-input set of a merge must satisfy `merged_external_inputs_compat` — all full-rank inputs broadcast-compatible to the same row*cols indexing.

The CUDA backend auto-generates kernels during `compile()` for any `KernelOp` via `generators/tiled.py`. Reshape/transpose ops become buffer aliases (zero-cost pointer assignment, no kernel launch).

`FusedRegionOp` is a legacy dataclass retained as a backend-internal struct (used by `_build_region_and_shapes` to reconstruct region data from serialized plan params). It is no longer emitted into the outer graph — fusion emits `KernelOp`.

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
3. **CudaBackend**: reads `params["_hints"]` in `_select_strategy()`, merges with tuning profile defaults, passes to `lower_tiled()` as the `hints` dict

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
