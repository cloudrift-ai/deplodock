# Compiler Architecture

## Layered Design

The compiler has four layers. Each layer depends only on the layers above it. **All new code MUST follow this layering — do NOT import from a lower layer into a higher one.**

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Frontend (backend-agnostic)                           │
│                                                                 │
│  torch_trace.py ─→ Graph IR (ops.py, ir.py)                    │
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
│  plan.py:           BufferSpec, OpKernel, ExecutionPlan          │
│  block_planner.py:  plan_block(BlockConfig) → ExecutionPlan     │
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
│  cuda/kernels/*.cu: Kernel templates with __KERNEL_NAME__       │
│  cuda/lower.py:     Graph → KernelDef (SGEMM strategies)        │
│  cuda/codegen.py:   KernelDef → CUDA C source                  │
│  cuda/ir.py:        CUDA imperative AST (Expr, Stmt, KernelDef)│
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
                     Layer 1                Layer 2
PyTorch module ──→ Graph IR ──→ Rewriter (fusion rules) ──→ optimized Graph
                                                                  │
                     Layer 3                                      │
                     plan_block(cfg) ──→ ExecutionPlan ◄──────────┘
                     plan_matmul(graph, dims) ──→ ExecutionPlan
                                                      │
                     Layer 4                          │
                     CudaBackend.compile(plan) ──→ Program
                                                      │
                     program.py                       │
                     generate_source() ──→ .cu ──→ nvcc ──→ GPU
```

### Concrete example: transformer block

```python
# Layer 3: plan (backend-agnostic)
from deplodock.compiler.block_planner import BlockConfig, plan_block
plan = plan_block(BlockConfig(hidden_dim=2048, num_heads=32, ...))

# Layer 4: backend (CUDA-specific)
from deplodock.compiler.cuda.backend import CudaBackend
backend = CudaBackend()
program = backend.compile(plan)
result = backend.benchmark(program)
```

### Concrete example: single matmul

```python
# Layer 1-2: graph + optimization
graph = build_matmul_graph()
graph = Rewriter.from_directory(rules_dir).apply(graph)

# Layer 3-4: plan + backend
from deplodock.compiler.cuda.lower import lower_matmul_to_program
program = lower_matmul_to_program(graph, MatmulConfig(strategy="naive"), dims)
result = run_program(program)
```

## What NOT to do

- **Do NOT import from `cuda/` in Layer 1-3 modules.** If you need GPU-specific behavior, add it to `cuda/backend.py`.
- **Do NOT put kernel source strings in Layer 3.** `OpKernel.op` is a tag; the backend resolves it to actual code.
- **Do NOT hardcode grid/block/smem in planners.** That's the backend's job.
- **Do NOT add new kernels as Python f-strings.** Write `.cu` template files in `cuda/kernels/` and use `load_kernel()`.

## Module Layout

```
compiler/
├── ops.py            # [L1] Op base class + all op types
├── ir.py             # [L1] Tensor, Node, Graph
├── torch_trace.py    # [L1] PyTorch → Graph IR (optional torch dep)
├── pattern.py        # [L2] Pattern AST + text parser
├── matcher.py        # [L2] Graph pattern matching engine
├── rewriter.py       # [L2] Pass/Rule/Rewriter
├── rules/            # [L2] Ordered rule files by pass
│   └── fusion/       #      RMSNorm, matmul, softmax, SiLU+mul
├── plan.py           # [L3] BufferSpec, OpKernel, ExecutionPlan
├── block_planner.py  # [L3] plan_block(BlockConfig) → ExecutionPlan
├── backend.py        # [L3] Backend ABC (compile, run, benchmark)
├── trace.py          # [L2] CompilerTrace for AI-in-the-loop
├── pipeline.py       # [L4*] compile_graph (L2) + compile_and_run (legacy CUDA)
├── benchmark.py      # [L4*] SGEMM benchmark harness (legacy CUDA)
└── cuda/             # [L4] CUDA backend
    ├── backend.py    #      CudaBackend implements Backend ABC
    ├── program.py    #      Buffer, Launch, Program — compile + run
    ├── kernels/      #      .cu template files + load_kernel()
    │   ├── __init__.py
    │   ├── rmsnorm.cu
    │   ├── activation.cu
    │   ├── rope.cu
    │   ├── attention_qk.cu
    │   ├── attention_softmax.cu
    │   ├── attention_sv.cu
    │   ├── matmul_naive.cu
    │   ├── matmul_residual_add.cu
    │   ├── matmul_triple.cu
    │   └── matmul_dual_silu_mul.cu
    ├── ir.py         #      CUDA imperative AST (KernelDef, Expr, Stmt)
    ├── codegen.py    #      KernelDef → CUDA C source
    ├── lower.py      #      Graph → KernelDef (SGEMM strategies + TMA)
    ├── runner.py     #      Legacy single-kernel compile + run + benchmark
    └── tuning.py     #      Per-GPU empirical tuning profiles
```

`*` — `pipeline.py` and `benchmark.py` have CUDA imports (legacy). New code should use the Backend ABC instead.

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

## Block Kernel Inventory

The transformer block plan has 10 ops (7 logical, attention is 3):

| # | OpKernel.op | What it fuses | .cu template |
|---|-------------|---------------|--------------|
| 1 | `rmsnorm` | pow→mean→rsqrt→mul→scale | rmsnorm.cu |
| 2 | `triple_matmul` | Q/K/V projections sharing input | matmul_triple.cu |
| 3 | `rope` | rotary embeddings for Q and K | rope.cu |
| 4a | `attention_qk` | QK^T + scale | attention_qk.cu |
| 4b | `attention_softmax` | row-wise softmax | attention_softmax.cu |
| 4c | `attention_sv` | scores @ V | attention_sv.cu |
| 5a | `matmul_residual_add` | Wo matmul + residual | matmul_residual_add.cu |
| 5b | `rmsnorm` | post-attention norm | rmsnorm.cu |
| 6 | `dual_matmul_silu_mul` | gate+up matmuls + silu(gate)*up | matmul_dual_silu_mul.cu |
| 7 | `matmul_residual_add` | Wd matmul + residual | matmul_residual_add.cu |

Attention materializes the N×N scores matrix (naive). This is the explicit seam where flash attention would replace it.

## Adding a New Kernel

1. Write the `.cu` file in `cuda/kernels/` with `__KERNEL_NAME__` placeholder
2. Add a backwards-compatible wrapper in `cuda/kernels/__init__.py`
3. Add a handler function `_compile_<op>()` in `cuda/backend.py`
4. Register it in `_OP_HANDLERS` dict in `cuda/backend.py`
5. Use it in a planner via `OpKernel(op="<name>", ...)`

## Adding a New Backend

1. Create `rocm/` directory alongside `cuda/`
2. Implement `RocmBackend(Backend)` in `rocm/backend.py`
3. Map the same `OpKernel.op` tags to `.hip` templates
4. The planner code (`plan.py`, `block_planner.py`) doesn't change

## Fusion Rules (Layer 2)

Rules are numbered to control ordering within a pass:

| Rule | Pattern | Output |
|------|---------|--------|
| `000_fuse_rmsnorm` | `(x * rsqrt(sum(x*x) * inv_n + eps)) * w` | `FusedRMSNormOp` |
| `001_fuse_reduce_elementwise` | `Reduce{sum}(Elementwise{mul}(A, B))` | `MatmulOp` |
| `002_fuse_softmax` | `exp(x-max(x)) / sum(exp(x-max(x)))` | `FusedSoftmaxOp` |
| `003_fuse_silu_mul` | `gate * recip(1 + exp(-gate)) * up` | `FusedSiLUMulOp` |

**Key constraint**: RMSNorm (000) runs before matmul (001) to prevent the matmul rule from consuming the `sum(x*x)` sub-pattern.

## CUDA Backend Details

### SGEMM Strategies (cuda/lower.py)

| Strategy | Description | Best for |
|----------|-------------|----------|
| `naive` | 1 thread per output element | Test baseline |
| **`tma_db`** | **TMA double-buffer, size-adaptive** | **Production default** |
| `tma_db_tf32` | TF32 via tensor cores (wmma) | TF32 precision ok |
| `tma_db_fma_tf32` | Concurrent FMA + TF32 hybrid | Mixed precision ok |

### TMA Performance (RTX 5090, sm_120)

| Size | TM | BK | K-splits | Eff vs cuBLAS | TFLOPS |
|------|-----|-----|----------|--------------|--------|
| **1024** | 8  | 32 | 1 | **101%** | 49.0 |
| **2048** | 26 | 32 | 1 | **106%** | 72.8 |
| **4096** | 20 | 32 | 1 | **101%** | 67.4 |
| 8192 | 28 | 32 | 1 | 96% | 60.2 |

### Kernel Templates (cuda/kernels/)

Kernels are `.cu` files with `__KERNEL_NAME__` placeholders, loaded via:
```python
from deplodock.compiler.cuda.kernels import load_kernel
source = load_kernel("rmsnorm", kernel_name="fused_rmsnorm")
```

Do NOT write kernel source as Python f-strings. Always use `.cu` template files.
